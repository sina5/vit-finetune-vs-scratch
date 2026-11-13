"""A small, shared training loop with timing, VRAM tracking and divergence detection."""

import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from vitfinetune import constants as C
from vitfinetune.errors import ConfigError, ResultsError


class OptimizerKind(Enum):
    """Supported optimizers."""

    ADAMW = "adamw"
    SGD = "sgd"


@dataclass(frozen=True)
class TrainConfig:
    """Settings for one training run.

    Raises:
        ConfigError: if a value is out of range.
    """

    epochs: int
    lr: float
    weight_decay: float
    optimizer: OptimizerKind = OptimizerKind.ADAMW
    warmup_epochs: int = 0
    grad_clip_norm: float | None = None
    label_smoothing: float = 0.0
    eval_every: int = 1
    use_amp: bool = True
    momentum: float = C.SGD_MOMENTUM

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.lr <= 0 or self.eval_every <= 0:
            raise ConfigError("epochs, lr and eval_every must be positive")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ConfigError("warmup_epochs must be in [0, epochs)")


class WarmupCosineSchedule:
    """LR multiplier: linear warmup from ~0 to 1, then cosine decay to 0.

    With zero warmup steps it is a plain cosine decay from the first step.
    """

    def __init__(self, warmup_steps: int, total_steps: int) -> None:
        if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
            raise ConfigError("need 0 <= warmup_steps < total_steps")
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        """Return the multiplier for a 0-based optimizer step."""
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


class OptimizerFactory:
    """Builds an optimizer that does not decay biases, norms, or embeddings (1-D params)."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config

    def build(self, model: nn.Module) -> torch.optim.Optimizer:
        """Return an optimizer over the trainable parameters of `model`."""
        decay, no_decay = [], []
        for param in model.parameters():
            if param.requires_grad:
                (decay if param.ndim >= 2 else no_decay).append(param)
        if not decay and not no_decay:
            raise ConfigError("model has no trainable parameters")
        groups = [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if self.config.optimizer is OptimizerKind.SGD:
            return torch.optim.SGD(groups, lr=self.config.lr, momentum=self.config.momentum, nesterov=True)
        return torch.optim.AdamW(groups, lr=self.config.lr)


@dataclass
class EpochRecord:
    """Metrics for one epoch. Test values are None on epochs without evaluation."""

    epoch: int
    train_loss: float
    train_acc: float
    lr: float
    seconds: float
    test_loss: float | None = None
    test_acc: float | None = None


@dataclass
class RunResult:
    """Everything a run reports, saved as JSON next to the checkpoint."""

    name: str
    label: str
    history: list[EpochRecord] = field(default_factory=list)
    final_test_acc: float | None = None
    best_test_acc: float | None = None
    train_seconds: float = 0.0
    peak_vram_mib: float | None = None
    trainable_params: int = 0
    total_params: int = 0
    diverged: bool = False


BatchTransform = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class ParameterCounter:
    """Counts trainable and total parameters."""

    @staticmethod
    def count(model: nn.Module) -> tuple[int, int]:
        """Return (trainable, total)."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return trainable, total


class Trainer:
    """Trains a classifier and records loss, accuracy, wall time and peak VRAM.

    Training stops early, with `diverged=True`, if the loss stops being finite.
    An optional `batch_transform` (for example MixUp) runs on each training batch
    on the device; it may return soft labels, and training accuracy then compares
    against their argmax.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: TrainConfig,
        device: torch.device,
        batch_transform: BatchTransform | None = None,
    ) -> None:
        self.model = model
        self.batch_transform = batch_transform
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self._amp = config.use_amp and device.type == "cuda"

    def fit(self, name: str, label: str) -> RunResult:
        """Run all epochs and return the result."""
        self.model.to(self.device)
        optimizer = OptimizerFactory(self.config).build(self.model)
        steps_per_epoch = len(self.train_loader)
        schedule = WarmupCosineSchedule(
            self.config.warmup_epochs * steps_per_epoch, self.config.epochs * steps_per_epoch
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        trainable, total = ParameterCounter.count(self.model)
        result = RunResult(name=name, label=label, trainable_params=trainable, total_params=total)
        self._reset_vram()
        start = time.perf_counter()
        for epoch in range(1, self.config.epochs + 1):
            record = self._run_epoch(epoch, optimizer, scheduler)
            result.history.append(record)
            self._print(name, record)
            if not math.isfinite(record.train_loss):
                result.diverged = True
                break
        result.train_seconds = time.perf_counter() - start
        result.peak_vram_mib = self._peak_vram()
        self._summarize(result)
        return result

    def _run_epoch(self, epoch: int, optimizer: torch.optim.Optimizer, scheduler) -> EpochRecord:
        start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = self._train_epoch(optimizer, scheduler)
        record = EpochRecord(epoch, train_loss, train_acc, lr, 0.0)
        is_last = epoch == self.config.epochs
        if math.isfinite(train_loss) and (epoch % self.config.eval_every == 0 or is_last):
            record.test_loss, record.test_acc = self.evaluate()
        record.seconds = time.perf_counter() - start
        return record

    def _train_epoch(self, optimizer: torch.optim.Optimizer, scheduler) -> tuple[float, float]:
        self.model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            targets = labels
            if self.batch_transform is not None:
                images, targets = self.batch_transform(images, labels)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self._amp):
                logits = self.model(images)
                loss = self.criterion(logits.float(), targets)
            hard = targets.argmax(dim=1) if targets.ndim == 2 else targets
            if not torch.isfinite(loss):
                return float("nan"), correct / max(seen, 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * labels.shape[0]
            correct += (logits.argmax(dim=1) == hard).sum().item()
            seen += labels.shape[0]
        return loss_sum / seen, correct / seen

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """Return (loss, accuracy) on the test loader."""
        self.model.eval()
        loss_sum, correct, seen = 0.0, 0, 0
        for images, labels in self.test_loader:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self._amp):
                logits = self.model(images)
            loss_sum += self.criterion(logits.float(), labels).item() * labels.shape[0]
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.shape[0]
        return loss_sum / seen, correct / seen

    def _reset_vram(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _peak_vram(self) -> float | None:
        if self.device.type != "cuda":
            return None
        return torch.cuda.max_memory_allocated(self.device) / C.BYTES_PER_MIB

    @staticmethod
    def _summarize(result: RunResult) -> None:
        accs = [r.test_acc for r in result.history if r.test_acc is not None]
        result.final_test_acc = accs[-1] if accs else None
        result.best_test_acc = max(accs) if accs else None

    @staticmethod
    def _print(name: str, record: EpochRecord) -> None:
        test = f" test_acc={record.test_acc:.4f}" if record.test_acc is not None else ""
        print(
            f"[{name}] epoch {record.epoch:3d} loss={record.train_loss:.4f} "
            f"train_acc={record.train_acc:.4f}{test} lr={record.lr:.2e} ({record.seconds:.1f}s)",
            flush=True,
        )


class ResultStore:
    """Saves and loads `RunResult` objects as JSON files in one folder."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def path_for(self, name: str) -> Path:
        """Return the JSON path for a run name."""
        return self.folder / f"{name}{C.RESULT_SUFFIX}"

    def save(self, result: RunResult) -> Path:
        """Write the result and return its path."""
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.path_for(result.name)
        path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        return path

    def load(self, name: str) -> RunResult:
        """Read one result.

        Raises:
            ResultsError: if the file does not exist.
        """
        path = self.path_for(name)
        if not path.exists():
            raise ResultsError(f"no results at {path}; run the experiment first")
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["history"] = [EpochRecord(**r) for r in raw["history"]]
        return RunResult(**raw)

    def load_available(self, names: list[str]) -> list[RunResult]:
        """Load the results that exist, in the given order, and skip the rest."""
        return [self.load(n) for n in names if self.path_for(n).exists()]

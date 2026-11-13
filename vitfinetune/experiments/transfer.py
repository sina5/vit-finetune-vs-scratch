"""One small dataset, three ways to get a ViT-B/16 classifier.

All three use the same architecture, data, augmentation, epochs and batch size.
Only the starting weights and the set of trainable parameters change.
"""

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import timm
import torch
from torch import nn

from vitfinetune import constants as C
from vitfinetune.data import Flowers102Data
from vitfinetune.errors import ConfigError
from vitfinetune.lora import LoRAInjector, ParameterFreezer
from vitfinetune.training import OptimizerKind, ResultStore, RunResult, TrainConfig, Trainer


class TransferMethod(Enum):
    """The three training strategies."""

    SCRATCH = "scratch"
    LINEAR_PROBE = "linear_probe"
    LORA = "lora"


METHOD_LABELS = {
    TransferMethod.SCRATCH: "From scratch",
    TransferMethod.LINEAR_PROBE: "Linear probe",
    TransferMethod.LORA: f"LoRA r={C.LORA_RANK}",
}
METHOD_LR = {
    TransferMethod.SCRATCH: C.SCRATCH_LR,
    TransferMethod.LINEAR_PROBE: C.PROBE_LR,
    TransferMethod.LORA: C.LORA_LR,
}


class TransferModelBuilder:
    """Builds the ViT-B/16 for each method.

    The pretrained weights download from the Hugging Face Hub on first use.
    """

    def __init__(self, num_classes: int = C.NUM_CLASSES, pretrained: bool = True) -> None:
        self.num_classes = num_classes
        self.pretrained = pretrained

    def build(self, method: TransferMethod) -> nn.Module:
        """Return the model with the right parameters set to trainable."""
        if method is TransferMethod.SCRATCH:
            return timm.create_model(C.SCRATCH_MODEL, pretrained=False, num_classes=self.num_classes)
        model = timm.create_model(C.PRETRAINED_MODEL, pretrained=self.pretrained, num_classes=self.num_classes)
        if method is TransferMethod.LORA:
            LoRAInjector(C.LORA_TARGETS, C.LORA_RANK, C.LORA_ALPHA, C.LORA_DROPOUT).inject(model)
        ParameterFreezer((C.HEAD_NAME,)).apply(model)
        return model


@dataclass(frozen=True)
class TransferSettings:
    """Run-wide settings shared by the three methods."""

    epochs: int = C.EPOCHS
    batch_size: int = C.BATCH_SIZE
    num_workers: int = C.DEFAULT_NUM_WORKERS
    eval_every: int = C.EVAL_EVERY
    subset: int | None = None
    seed: int = C.DEFAULT_SEED

    def smoke(self) -> "TransferSettings":
        """A tiny version of these settings for a quick end-to-end check."""
        return replace(self, epochs=C.SMOKE_EPOCHS, subset=C.SMOKE_SUBSET, eval_every=1)


class MethodCatalog:
    """Parses method names from the command line."""

    @staticmethod
    def select(names: list[str] | None) -> list[TransferMethod]:
        """Return the named methods in canonical order, or all of them when `names` is None.

        Raises:
            ConfigError: if a name is unknown.
        """
        if names is None:
            return list(TransferMethod)
        valid = {m.value for m in TransferMethod}
        unknown = set(names) - valid
        if unknown:
            raise ConfigError(f"unknown methods {sorted(unknown)}; choose from {sorted(valid)}")
        return [m for m in TransferMethod if m.value in names]


class TransferRunner:
    """Trains each method on Flowers-102 and saves its result JSON."""

    def __init__(
        self, settings: TransferSettings, builder: TransferModelBuilder, data_dir: Path, output_dir: Path,
        device: torch.device,
    ) -> None:
        self.settings = settings
        self.builder = builder
        self.data_dir = data_dir
        self.store = ResultStore(output_dir)
        self.device = device

    def train_config(self, method: TransferMethod) -> TrainConfig:
        """Same schedule for every method; only the learning rate differs."""
        return TrainConfig(
            epochs=self.settings.epochs, lr=METHOD_LR[method], weight_decay=C.WEIGHT_DECAY,
            optimizer=OptimizerKind.ADAMW, warmup_epochs=min(C.WARMUP_EPOCHS, self.settings.epochs - 1),
            grad_clip_norm=C.GRAD_CLIP_NORM, eval_every=self.settings.eval_every,
        )

    def run(self, methods: list[TransferMethod]) -> list[RunResult]:
        """Train the methods in order and return their results."""
        data = Flowers102Data(self.data_dir, self.settings.batch_size, self.settings.num_workers, self.settings.subset)
        results = []
        for method in methods:
            torch.manual_seed(self.settings.seed)
            model = self.builder.build(method)
            trainer = Trainer(model, data.train_loader(), data.test_loader(), self.train_config(method), self.device)
            result = trainer.fit(method.value, METHOD_LABELS[method])
            print(f"[{method.value}] saved {self.store.save(result)}", flush=True)
            results.append(result)
            del model, trainer
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return results

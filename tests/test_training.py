import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vitfinetune.errors import ConfigError, ResultsError
from vitfinetune.training import (
    EpochRecord, OptimizerFactory, OptimizerKind, ResultStore, RunResult, TrainConfig, Trainer,
    WarmupCosineSchedule,
)

CPU = torch.device("cpu")


def _loader(n: int = 32) -> DataLoader:
    features = torch.randn(n, 4)
    labels = (features[:, 0] > 0).long()
    return DataLoader(TensorDataset(features, labels), batch_size=8)


class _NanModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * float("nan")


def test_warmup_rises_linearly_to_one() -> None:
    schedule = WarmupCosineSchedule(warmup_steps=4, total_steps=10)
    assert [schedule(s) for s in range(4)] == [0.25, 0.5, 0.75, 1.0]


def test_cosine_reaches_zero_at_the_end() -> None:
    schedule = WarmupCosineSchedule(warmup_steps=0, total_steps=10)
    assert schedule(0) == 1.0
    assert math.isclose(schedule(10), 0.0, abs_tol=1e-12)


@pytest.mark.parametrize("warmup,total", [(5, 5), (-1, 5), (0, 0)])
def test_schedule_rejects_bad_step_counts(warmup: int, total: int) -> None:
    with pytest.raises(ConfigError):
        WarmupCosineSchedule(warmup, total)


def test_config_rejects_warmup_longer_than_training() -> None:
    with pytest.raises(ConfigError):
        TrainConfig(epochs=3, lr=1e-3, weight_decay=0.0, warmup_epochs=3)


def test_optimizer_skips_decay_on_one_dimensional_params() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    decay, no_decay = OptimizerFactory(TrainConfig(epochs=1, lr=1e-3, weight_decay=0.1)).build(model).param_groups
    assert (len(decay["params"]), len(no_decay["params"])) == (1, 3)
    assert no_decay["weight_decay"] == 0.0


def test_optimizer_builds_sgd_when_asked() -> None:
    config = TrainConfig(epochs=1, lr=0.1, weight_decay=0.0, optimizer=OptimizerKind.SGD)
    assert isinstance(OptimizerFactory(config).build(nn.Linear(2, 2)), torch.optim.SGD)


def test_optimizer_rejects_fully_frozen_model() -> None:
    with pytest.raises(ConfigError):
        OptimizerFactory(TrainConfig(epochs=1, lr=1e-3, weight_decay=0.0)).build(nn.Linear(2, 2).requires_grad_(False))


def test_trainer_learns_a_separable_problem() -> None:
    config = TrainConfig(epochs=15, lr=0.05, weight_decay=0.0)
    result = Trainer(nn.Linear(4, 2), _loader(), _loader(), config, CPU).fit("t", "T")
    assert result.final_test_acc is not None and result.final_test_acc > 0.9
    assert len(result.history) == 15 and not result.diverged


def test_trainer_evaluates_only_on_schedule_and_last_epoch() -> None:
    config = TrainConfig(epochs=5, lr=1e-2, weight_decay=0.0, eval_every=2)
    result = Trainer(nn.Linear(4, 2), _loader(), _loader(), config, CPU).fit("t", "T")
    assert [r.epoch for r in result.history if r.test_acc is not None] == [2, 4, 5]


def test_trainer_stops_and_flags_divergence() -> None:
    config = TrainConfig(epochs=4, lr=1e-2, weight_decay=0.0)
    result = Trainer(_NanModel(), _loader(), _loader(), config, CPU).fit("t", "T")
    assert result.diverged and len(result.history) == 1


def test_result_store_round_trips(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    original = RunResult("a", "A", [EpochRecord(1, 0.5, 0.8, 1e-3, 2.0, 0.6, 0.7)], 0.7, 0.7, 2.0, None, 10, 20)
    store.save(original)
    assert store.load("a") == original


def test_result_store_skips_missing_runs(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    store.save(RunResult("b", "B"))
    assert [r.name for r in store.load_available(["a", "b"])] == ["b"]


def test_result_store_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ResultsError):
        ResultStore(tmp_path).load("missing")


def test_trainer_accepts_soft_labels_from_a_batch_transform() -> None:
    def to_soft(images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return images, nn.functional.one_hot(labels, 2).float()

    config = TrainConfig(epochs=15, lr=0.05, weight_decay=0.0)
    result = Trainer(nn.Linear(4, 2), _loader(), _loader(), config, CPU, batch_transform=to_soft).fit("t", "T")
    assert result.final_test_acc > 0.9 and result.history[-1].train_acc > 0.9

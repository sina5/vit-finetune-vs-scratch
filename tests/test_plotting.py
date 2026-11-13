from pathlib import Path

import pytest

from vitfinetune.errors import ResultsError
from vitfinetune.plotting import TransferPlotter
from vitfinetune.training import EpochRecord, RunResult

PNG_MAGIC = b"\x89PNG"


def _result(name: str, acc: float) -> RunResult:
    history = [EpochRecord(1, 1.0, 0.5, 1e-3, 1.0, 1.0, acc / 2), EpochRecord(2, 0.8, 0.6, 1e-3, 1.0, 0.9, acc)]
    return RunResult(name, name, history, acc, acc, 60.0, 900.0, 1000, 2000)


def test_transfer_plot_writes_png(tmp_path: Path) -> None:
    path = TransferPlotter().plot([_result("scratch", 0.2), _result("lora", 0.9)], tmp_path / "t.png")
    assert path.read_bytes()[:4] == PNG_MAGIC


def test_transfer_plot_rejects_empty_results(tmp_path: Path) -> None:
    with pytest.raises(ResultsError):
        TransferPlotter().plot([], tmp_path / "x.png")

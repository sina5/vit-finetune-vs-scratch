import pytest
import torch
from torch import nn

from vitfinetune.errors import ConfigError
from vitfinetune.lora import LoRAInjector, LoRALinear, ParameterFreezer


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(8, 24)
        self.proj = nn.Linear(8, 8)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.head = nn.Linear(8, 3)


def test_new_lora_layer_matches_the_base_layer() -> None:
    base = nn.Linear(8, 6)
    x = torch.randn(4, 8)
    expected = base(x)
    assert torch.allclose(LoRALinear(base, rank=2, alpha=4.0)(x), expected)


def test_lora_freezes_base_and_trains_only_low_rank_weights() -> None:
    layer = LoRALinear(nn.Linear(8, 6), rank=2, alpha=4.0)
    assert {n for n, p in layer.named_parameters() if p.requires_grad} == {"lora_a", "lora_b"}


@pytest.mark.parametrize("rank", [0, 7])
def test_lora_rejects_out_of_range_rank(rank: int) -> None:
    with pytest.raises(ConfigError):
        LoRALinear(nn.Linear(8, 6), rank=rank, alpha=1.0)


def test_injector_wraps_only_matching_layers() -> None:
    model = _Model()
    assert LoRAInjector(("qkv",), rank=2, alpha=4.0).inject(model) == 2
    assert isinstance(model.blocks[0].qkv, LoRALinear)
    assert type(model.blocks[0].proj) is nn.Linear


def test_injector_raises_when_nothing_matches() -> None:
    with pytest.raises(ConfigError):
        LoRAInjector(("missing",), rank=2, alpha=4.0).inject(_Model())


def test_freezer_keeps_lora_and_head_trainable() -> None:
    model = _Model()
    LoRAInjector(("qkv",), rank=2, alpha=4.0).inject(model)
    ParameterFreezer(("head",)).apply(model)
    assert {n for n, p in model.named_parameters() if p.requires_grad} == {
        "blocks.0.qkv.lora_a", "blocks.0.qkv.lora_b", "blocks.1.qkv.lora_a", "blocks.1.qkv.lora_b",
        "head.weight", "head.bias",
    }

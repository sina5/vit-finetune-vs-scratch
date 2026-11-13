import pytest

from vitfinetune.errors import ConfigError
from vitfinetune.experiments.transfer import MethodCatalog, TransferMethod, TransferModelBuilder
from vitfinetune.lora import LoRALinear
from vitfinetune.training import ParameterCounter


@pytest.fixture(scope="module")
def builder() -> TransferModelBuilder:
    return TransferModelBuilder(num_classes=5, pretrained=False)  # no network in tests


def test_scratch_trains_every_parameter(builder: TransferModelBuilder) -> None:
    trainable, total = ParameterCounter.count(builder.build(TransferMethod.SCRATCH))
    assert trainable == total


def test_linear_probe_trains_only_the_head(builder: TransferModelBuilder) -> None:
    model = builder.build(TransferMethod.LINEAR_PROBE)
    assert {n for n, p in model.named_parameters() if p.requires_grad} == {"head.weight", "head.bias"}


def test_lora_wraps_every_qkv_and_trains_a_small_fraction(builder: TransferModelBuilder) -> None:
    model = builder.build(TransferMethod.LORA)
    assert all(isinstance(block.attn.qkv, LoRALinear) for block in model.blocks)
    trainable, total = ParameterCounter.count(model)
    assert 0 < trainable < 0.01 * total


def test_catalog_returns_all_methods_by_default() -> None:
    assert MethodCatalog.select(None) == list(TransferMethod)


def test_catalog_rejects_unknown_method() -> None:
    with pytest.raises(ConfigError):
        MethodCatalog.select(["full_finetune"])

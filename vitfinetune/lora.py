"""Low-rank adaptation (LoRA, Hu et al., 2021) for `nn.Linear` layers.

Written here instead of imported so the method stays visible: the frozen weight W
gets a trainable update B @ A of rank r, scaled by alpha / r. B starts at zero, so
the wrapped layer first behaves exactly like the original.
"""

import math

import torch
from torch import nn

from vitfinetune.errors import ConfigError


class LoRALinear(nn.Module):
    """A frozen `nn.Linear` plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0 or rank > min(base.in_features, base.out_features):
            raise ConfigError(f"rank must be in [1, {min(base.in_features, base.out_features)}], got {rank}")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return base(x) + scale * B(A(dropout(x)))."""
        update = self.dropout(x) @ self.lora_a.T @ self.lora_b.T
        return self.base(x) + self.scale * update


class LoRAInjector:
    """Replaces chosen `nn.Linear` layers of a model with `LoRALinear` wrappers.

    Args:
        target_suffixes: a Linear is wrapped when its dotted name ends with one of these.
        rank, alpha, dropout: passed to every `LoRALinear`.
    """

    def __init__(self, target_suffixes: tuple[str, ...], rank: int, alpha: float, dropout: float = 0.0) -> None:
        if not target_suffixes:
            raise ConfigError("target_suffixes must name at least one layer")
        self.target_suffixes = target_suffixes
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

    def inject(self, model: nn.Module) -> int:
        """Wrap every matching Linear in place and return how many were wrapped.

        Raises:
            ConfigError: if no layer matches.
        """
        targets = [
            name for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name.endswith(self.target_suffixes)
        ]
        if not targets:
            raise ConfigError(f"no nn.Linear ends with {self.target_suffixes}")
        for name in targets:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, LoRALinear(getattr(parent, child), self.rank, self.alpha, self.dropout))
        return len(targets)


class ParameterFreezer:
    """Freezes a model except for LoRA weights and named modules (for example the head)."""

    def __init__(self, trainable_prefixes: tuple[str, ...]) -> None:
        self.trainable_prefixes = trainable_prefixes

    def apply(self, model: nn.Module) -> None:
        """Set `requires_grad` on every parameter of `model`."""
        for name, param in model.named_parameters():
            is_lora = name.endswith(("lora_a", "lora_b"))
            param.requires_grad_(is_lora or name.startswith(self.trainable_prefixes))

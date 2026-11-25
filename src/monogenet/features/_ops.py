"""Numerically safe primitives shared by the feature functions.

``sqrt`` and ``atan2`` have infinite or undefined gradients at exactly zero
input, which happens for all-zero or constant images (black ultrasound
background, padding). These helpers keep values exact away from zero and give
finite (zero) gradients at it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

#: Added under square roots: a bias of at most 1e-12 in the result.
TINY_SQ = 1e-24
#: Squared-norm threshold below which atan2 inputs count as the zero vector.
TINY_ATAN = 1e-20

_HALF = (torch.float16, torch.bfloat16)


def safe_sqrt(x: torch.Tensor, tiny: float = TINY_SQ) -> torch.Tensor:
    """``sqrt(x + tiny)`` for ``x >= 0``: finite gradient (0) at ``x = 0``."""
    return torch.sqrt(x + tiny)


def safe_atan2(y: torch.Tensor, x: torch.Tensor, tiny: float = TINY_ATAN) -> torch.Tensor:
    """``atan2(y, x)`` that returns 0 with zero gradient where ``x^2 + y^2 < tiny``.

    Elsewhere it is exactly ``torch.atan2`` (values and gradients).
    """
    zero = (x * x + y * y) < tiny
    x_safe = torch.where(zero, torch.ones_like(x), x)
    y_safe = torch.where(zero, torch.zeros_like(y), y)
    return torch.where(zero, torch.zeros_like(x), torch.atan2(y_safe, x_safe))


def promote_half(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run ``fn`` in float32 when its tensor arguments are float16 or bfloat16.

    Squares of half-precision amplitudes under/overflow easily and the safe
    constants above are not representable in half precision. Results are cast
    back to the input dtype.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        dtypes = {a.dtype for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)}
        half = next((d for d in dtypes if d in _HALF), None)
        if half is None:
            return fn(*args, **kwargs)

        def up(a: Any) -> Any:
            return a.float() if isinstance(a, torch.Tensor) and a.dtype in _HALF else a

        out = fn(*(up(a) for a in args), **{k: up(v) for k, v in kwargs.items()})

        def down(t: Any) -> Any:
            return t.to(half) if isinstance(t, torch.Tensor) else t

        if isinstance(out, tuple) and hasattr(out, "_fields"):  # NamedTuple
            return type(out)(*(down(t) for t in out))
        if isinstance(out, tuple):
            return tuple(down(t) for t in out)
        return down(out)

    return wrapper

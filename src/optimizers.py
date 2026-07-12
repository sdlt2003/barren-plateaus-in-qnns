#!/usr/bin/env python3
"""Optimizer registry for the barren-plateaus experiments.

Each optimizer is described by an :class:`OptimizerSpec` that knows how to build
the underlying Qiskit optimizer, whether it needs a fidelity callback (only
QNSPSA does), and a stable display label/color for plots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from qiskit_algorithms.optimizers import COBYLA, QNSPSA

# NFT (Nakanishi-Fujii-Todo) is the Qiskit-native sequential minimizer that
# exploits the sinusoidal parameter dependence, i.e. the same idea as Rotosolve.
# It is guarded so the module still imports on older qiskit-algorithms versions.
try:  # pragma: no cover - depends on installed qiskit-algorithms version
    from qiskit_algorithms.optimizers import NFT

    _HAS_NFT = True
except ImportError:  # pragma: no cover
    NFT = None  # type: ignore[assignment]
    _HAS_NFT = False


@dataclass(frozen=True)
class OptimizerSpec:
    """Description of a selectable optimizer."""

    name: str
    label: str
    color: str
    needs_fidelity: bool
    _build: Callable[..., Any]

    def build(self, budget_evals: int, fidelity: Callable | None = None) -> Any:
        """Instantiate the underlying Qiskit optimizer.

        ``budget_evals`` is passed as ``maxiter`` so Qiskit's own iteration cap
        stays aligned with the shared evaluation budget enforced in
        ``utils.make_objective``.
        """
        if self.needs_fidelity:
            if fidelity is None:
                raise ValueError(f"Optimizer '{self.name}' requires a fidelity callback.")
            return self._build(budget_evals, fidelity)
        return self._build(budget_evals)


# Insertion order defines the canonical ordering used across CSVs/plots.
OPTIMIZER_REGISTRY: dict[str, OptimizerSpec] = {
    "cobyla": OptimizerSpec(
        name="cobyla",
        label="COBYLA",
        color="#4C78A8",
        needs_fidelity=False,
        _build=lambda budget: COBYLA(maxiter=budget),
    ),
    "qnspsa": OptimizerSpec(
        name="qnspsa",
        label="QNSPSA",
        color="#54A24B",
        needs_fidelity=True,
        _build=lambda budget, fidelity: QNSPSA(fidelity=fidelity, maxiter=budget),
    ),
}

if _HAS_NFT:
    OPTIMIZER_REGISTRY["nft"] = OptimizerSpec(
        name="nft",
        label="NFT",
        color="#E45756",
        needs_fidelity=False,
        # qiskit-algorithms' NFT defaults to maxfev=1024, which silently caps the
        # number of function evaluations regardless of ``maxiter``. Without setting
        # maxfev, NFT would stop at ~1024 evals while COBYLA/QNSPSA consume the full
        # k*p budget, making the comparison unfair. Pin maxfev to the shared budget
        # so all optimizers run under the same evaluation cap (the global
        # ``BudgetExceeded`` guard in utils.make_objective remains the hard limit).
        _build=lambda budget: NFT(maxiter=budget, maxfev=budget),
    )

# Default optimizers for simulation grids. NFT is appended when available so new
# grids compare all three methods, while old runs (cobyla/qnspsa) still parse.
DEFAULT_OPTIMIZERS: tuple[str, ...] = tuple(
    name for name in ("cobyla", "qnspsa", "nft") if name in OPTIMIZER_REGISTRY
)

# Fallback color cycle for optimizer names not present in the registry (e.g.
# layer-wise variants labeled "layerwise_nft").
_FALLBACK_COLORS = ("#B279A2", "#EECA3B", "#72B7B2", "#FF9DA6", "#9D755D", "#BAB0AC")


def available_optimizers() -> tuple[str, ...]:
    return tuple(OPTIMIZER_REGISTRY.keys())


def get_optimizer_spec(name: str) -> OptimizerSpec:
    key = name.strip().lower()
    if key not in OPTIMIZER_REGISTRY:
        known = ", ".join(available_optimizers())
        raise ValueError(f"Unknown optimizer '{name}'. Available: {known}.")
    return OPTIMIZER_REGISTRY[key]


def parse_optimizer_list(raw: str) -> list[str]:
    """Parse a comma-separated optimizer list like 'cobyla,qnspsa,nft'."""
    names = [chunk.strip().lower() for chunk in raw.split(",") if chunk.strip()]
    if not names:
        raise ValueError("Optimizer list must contain at least one optimizer name.")
    seen: list[str] = []
    for name in names:
        # Validate against the registry (raises for unknown names).
        get_optimizer_spec(name)
        if name not in seen:
            seen.append(name)
    return seen


def ordered_optimizer_names(names: Iterable[str]) -> list[str]:
    """Order optimizer names by registry order, then alphabetically for extras."""
    unique = list(dict.fromkeys(names))
    registry_order = [n for n in OPTIMIZER_REGISTRY if n in unique]
    extras = sorted(n for n in unique if n not in OPTIMIZER_REGISTRY)
    return registry_order + extras


def optimizer_label(name: str) -> str:
    spec = OPTIMIZER_REGISTRY.get(name)
    if spec is not None:
        return spec.label
    return name.upper()


def optimizer_color(name: str) -> str:
    spec = OPTIMIZER_REGISTRY.get(name)
    if spec is not None:
        return spec.color
    # Deterministic fallback color based on name hash position.
    extras = sorted(n for n in _seen_extra_names if n not in OPTIMIZER_REGISTRY)
    if name not in extras:
        _seen_extra_names.add(name)
        extras = sorted(_seen_extra_names)
    idx = extras.index(name) if name in extras else 0
    return _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]


# Tracks non-registry optimizer names seen at runtime to assign stable colors.
_seen_extra_names: set[str] = set()

#!/usr/bin/env python3
"""IBM Qiskit Runtime execution mode helpers (session, batch, job)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Literal

ExecutionMode = Literal["session", "batch", "job"]


def normalize_execution_mode(mode: str | None) -> ExecutionMode:
    if mode is None or mode == "":
        return "batch"
    normalized = mode.strip().lower()
    if normalized in {"session", "batch", "job"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"Unsupported execution mode: {mode!r}. Use session, batch, or job.")


def normalize_runtime_max_time(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped in {"", "none", "None"}:
        return None
    return stripped


@contextmanager
def runtime_execution_mode(
    backend,
    *,
    mode: ExecutionMode,
    max_time: str | None = None,
) -> Iterator[object]:
    """Open an IBM Runtime scope for Estimator(mode=...).

    - batch: grouped jobs with queue priority (recommended for grid points).
    - session: dedicated QPU access until TTL.
    - job: no container; each estimator.run uses standalone job mode.
    """
    container_kwargs: dict[str, str] = {}
    if max_time:
        container_kwargs["max_time"] = max_time

    if mode == "job":
        yield backend
        return

    if mode == "batch":
        from qiskit_ibm_runtime import Batch

        with Batch(backend=backend, **container_kwargs) as batch:
            yield batch
        return

    from qiskit_ibm_runtime import Session

    with Session(backend=backend, **container_kwargs) as session:
        yield session


def describe_runtime_mode(mode: ExecutionMode, max_time: str | None) -> str:
    ttl = max_time if max_time else "IBM plan default (not set in client)"
    if mode == "job":
        return "job mode (no session/batch container)"
    return f"{mode} max_time={ttl}"

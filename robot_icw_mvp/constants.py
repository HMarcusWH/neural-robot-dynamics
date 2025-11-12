"""Shared constants and helpers for ICW integration."""

from __future__ import annotations

from typing import Dict


BACKEND_ANALYTIC = "analytic"
BACKEND_NEURAL = "neural"
BACKEND_ABSTAIN = "abstain"
BACKEND_AUTO = "auto"

_LEGACY_ALIASES: Dict[str, str] = {
    "ground-truth": BACKEND_ANALYTIC,
    "ground_truth": BACKEND_ANALYTIC,
    "gt": BACKEND_ANALYTIC,
}


def canonicalize_backend(label: str) -> str:
    """Map legacy backend labels to the canonical set."""

    if label is None:
        raise ValueError("Backend label cannot be None")
    lowered = label.lower()
    if lowered in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[lowered]
    if lowered in {BACKEND_ANALYTIC, BACKEND_NEURAL, BACKEND_ABSTAIN, BACKEND_AUTO}:
        return lowered
    raise ValueError(f"Unknown backend label '{label}'")


def is_backend_label(label: str) -> bool:
    try:
        canonicalize_backend(label)
    except ValueError:
        return False
    return True


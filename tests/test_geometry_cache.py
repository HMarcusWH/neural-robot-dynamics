import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from robot_icw_mvp.geometry.cache import LadderGeometryCache


def test_geometry_cache_shapes_and_monotonicity():
    cache = LadderGeometryCache(rung_dims=[2, 4, 6], device=torch.device("cpu"))

    first_states = torch.zeros(3, 6)
    first_snapshot = cache.observe(first_states)
    assert first_snapshot.ddp.shape == (3, 3)
    assert torch.all(first_snapshot.ddp == 0.0)
    assert torch.all(first_snapshot.rupture == 0.0)

    second_states = torch.randn(3, 6)
    second_snapshot = cache.observe(second_states)
    assert second_snapshot.ddp.shape == (3, 3)
    assert torch.all(second_snapshot.ddp >= 0.0)
    assert torch.all(second_snapshot.ddp <= 2.0)
    assert second_snapshot.rupture.shape == (3,)
    assert second_snapshot.mean_rupture() >= 0.0

    # Re-observing the same state should yield zero rupture.
    third_snapshot = cache.observe(second_states)
    assert torch.allclose(
        third_snapshot.ddp,
        torch.zeros_like(third_snapshot.ddp),
        atol=1e-6,
    )
    expected_rupture = second_snapshot.ddp.abs().sum(dim=-1)
    assert torch.allclose(third_snapshot.rupture, expected_rupture, atol=1e-6)

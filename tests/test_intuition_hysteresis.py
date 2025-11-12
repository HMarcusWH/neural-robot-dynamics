import torch

from robot_icw_mvp.geometry.cache import GeometrySnapshot
from robot_icw_mvp.intuition.zoom_controller import IntuitionController


def _make_snapshot(rupture: float, step_norm: float) -> GeometrySnapshot:
    return GeometrySnapshot(
        states=torch.zeros(1, 4),
        prev_states=None,
        ddp=torch.zeros(1, 1),
        rupture=torch.tensor([rupture]),
        step_norm=torch.tensor([step_norm]),
        rung_dims=(4,),
    )


def test_intuition_hysteresis_and_abstention():
    controller = IntuitionController(
        hysteresis_lo=0.1,
        hysteresis_hi=0.2,
        abstain_quantile=0.5,
        residual_window=16,
        default_backend="ground-truth",
    )

    snap_lo = _make_snapshot(rupture=0.05, step_norm=0.01)
    decision_lo = controller.decide(snap_lo)
    assert decision_lo.backend == "ground-truth"
    controller.record(decision_lo.backend, snap_lo)

    snap_hi = _make_snapshot(rupture=0.25, step_norm=0.02)
    decision_hi = controller.decide(snap_hi)
    assert decision_hi.backend == "neural"
    controller.record(decision_hi.backend, snap_hi)

    snap_mid = _make_snapshot(rupture=0.15, step_norm=0.02)
    decision_mid = controller.decide(snap_mid)
    assert decision_mid.backend == "neural"
    controller.record(decision_mid.backend, snap_mid)

    snap_back = _make_snapshot(rupture=0.05, step_norm=0.01)
    decision_back = controller.decide(snap_back)
    assert decision_back.backend == "ground-truth"
    controller.record(decision_back.backend, snap_back)

    for _ in range(12):
        sample = _make_snapshot(rupture=0.12, step_norm=0.1)
        decision = controller.decide(sample)
        controller.record(decision.backend, sample)

    abstain_snapshot = _make_snapshot(rupture=0.12, step_norm=1.0)
    abstain_decision = controller.decide(abstain_snapshot)
    assert abstain_decision.abstain is True

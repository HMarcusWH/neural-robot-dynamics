import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from robot_icw_mvp.constants import canonicalize_backend, BACKEND_ANALYTIC


@pytest.mark.parametrize(
    "alias",
    ["ground-truth", "ground_truth", "analytic", "gt", "GROUND-TRUTH", "Analytic"],
)
def test_canonicalize_backend_aliases(alias):
    assert canonicalize_backend(alias) == BACKEND_ANALYTIC


def test_canonicalize_backend_invalid():
    with pytest.raises(ValueError):
        canonicalize_backend("unknown")


def test_canonicalize_backend_requires_label():
    with pytest.raises(ValueError):
        canonicalize_backend(None)

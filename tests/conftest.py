"""Shared pytest fixtures for gtrack tests."""

import importlib.util
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
_MAKER = DATA_DIR / "make_synthetic_model.py"


def _ensure_synthetic_model():
    """Generate the synthetic topological model fixtures if missing."""
    spec = importlib.util.spec_from_file_location("_make_synthetic_model", _MAKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    paths = (mod.ROT_PATH, mod.TOPO_PATH, mod.STATIC_PATH)
    if not all(p.exists() for p in paths):
        mod.main()
    return mod


@pytest.fixture(scope="session")
def synthetic_model():
    """Paths and key facts for the tiny synthetic topological model.

    See tests/data/make_synthetic_model.py for how it is constructed. Returns a
    dict with the file paths and the geometry/plate facts the tests rely on.
    """
    mod = _ensure_synthetic_model()
    return {
        "rotation_files": str(mod.ROT_PATH),
        "topology_files": str(mod.TOPO_PATH),
        "static_polygons": str(mod.STATIC_PATH),
        # A point inside plate A's rigid topological boundary (moves +30 deg).
        "plate_a_point": (10.0, 10.0),          # (lat, lon)
        # A point inside the deforming network (deforms, not a clean rigid pole).
        "network_point": (0.0, 90.0),
        # A point in a topology gap (kept, unmoved with deactivate_points=None).
        "gap_point": (0.0, -170.0),
        # The out-of-circuit point: inside plate A topologically, but the static
        # polygon labels it plate 999 which has no rigid sequence.
        "out_of_circuit_point": mod.OUT_OF_CIRCUIT_POINT,
        "out_of_circuit_plate": mod.OUT_OF_CIRCUIT_PLATE,
    }

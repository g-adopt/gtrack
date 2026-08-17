"""
Test that checkpointing and reloading a SeafloorAgeTracker produces
identical results to a continuous (non-interrupted) run.

The test evolves tracers from 410 Ma to 390 Ma in two ways:
  A) Continuous:  410 -> 390 in one go.
  B) Checkpoint:  410 -> 400, save checkpoint, reload, 400 -> 390.

The final PointClouds from A and B must be exactly equal.
"""

import numpy as np
import pytest
from pathlib import Path

from gtrack import SeafloorAgeTracker, TracerConfig

# ---------------------------------------------------------------------------
# Data paths – use the plate model shipped with the examples directory.
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "examples" / "Matthews_et_al_410_0"

ROTATION_FILES = [
    DATA_DIR / "Global_EB_250-0Ma_GK07_Matthews++.rot",
    DATA_DIR / "Global_EB_410-250Ma_GK07_Matthews++.rot",
]
TOPOLOGY_FILES = [
    DATA_DIR / "Mesozoic-Cenozoic_plate_boundaries_Matthews++.gpml",
    DATA_DIR / "Paleozoic_plate_boundaries_Matthews++.gpml",
    DATA_DIR / "TopologyBuildingBlocks_Matthews++.gpml",
]
CONTINENTAL_POLYGONS = (
    DATA_DIR / "ContPolys" / "PresentDay_ContinentalPolygons_Matthews++.shp"
)

CONFIG = TracerConfig(
    tracker_step_myr=1.0,
    tracker_point_count=10000,
    initial_spreading_rate_mm_per_yr=75.0,
    ridge_sampling_angle_deg=0.5,
    ridge_offset_angle_deg=0.01,
    collision_velocity_difference_km_per_myr=7.0,
    collision_distance_rate_km_per_myr=10.0,
)


def _data_available():
    return all(f.exists() for f in ROTATION_FILES) and all(
        f.exists() for f in TOPOLOGY_FILES
    )


@pytest.mark.skipif(not _data_available(), reason="GPlates data files not found")
def test_checkpoint_roundtrip(tmp_path):
    """Continuous run must match a run interrupted by a checkpoint reload."""

    # --- Run A: continuous 410 -> 390 ---
    tracker_a = SeafloorAgeTracker(
        rotation_files=ROTATION_FILES,
        topology_files=TOPOLOGY_FILES,
        continental_polygons=CONTINENTAL_POLYGONS,
        config=CONFIG,
    )
    tracker_a.initialize(starting_age=410)
    cloud_a = tracker_a.step_to(390)

    # --- Run B: 410 -> 400, checkpoint, reload, 400 -> 390 ---
    tracker_b = SeafloorAgeTracker(
        rotation_files=ROTATION_FILES,
        topology_files=TOPOLOGY_FILES,
        continental_polygons=CONTINENTAL_POLYGONS,
        config=CONFIG,
    )
    tracker_b.initialize(starting_age=410)
    tracker_b.step_to(400)

    checkpoint_file = str(tmp_path / "checkpoint_400Ma.npz")
    tracker_b.save_checkpoint(checkpoint_file)

    # Create a fresh tracker and reload from the checkpoint.
    tracker_b2 = SeafloorAgeTracker(
        rotation_files=ROTATION_FILES,
        topology_files=TOPOLOGY_FILES,
        continental_polygons=CONTINENTAL_POLYGONS,
        config=CONFIG,
    )
    tracker_b2.load_checkpoint(checkpoint_file)
    cloud_b = tracker_b2.step_to(390)

    # --- Compare ---
    # The checkpoint round-trips through XYZ <-> lonlat conversion, which
    # introduces sub-millimeter floating-point noise (~1e-8 m on ~6e6 m
    # coordinates, i.e. relative ~1e-12).  We use allclose with a tight
    # tolerance rather than exact equality.
    np.testing.assert_allclose(
        cloud_a.xyz, cloud_b.xyz,
        atol=1e-7, rtol=0,
        err_msg="XYZ coordinates differ between continuous and checkpoint runs",
    )
    np.testing.assert_array_equal(
        cloud_a.get_property("age"),
        cloud_b.get_property("age"),
        err_msg="Tracer ages differ between continuous and checkpoint runs",
    )

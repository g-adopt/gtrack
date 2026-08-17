"""Regression tests for sub-Myr tracker steps.

The tracker passes each positive sub-Myr span to the topological model.

These tests exercise:
  - A clean sub-Myr step over an integer-Ma range.
  - A non-integer step that also leaves a truncated final step shorter than
    the configured tracker step.
"""

import pytest
from pathlib import Path

from gtrack import SeafloorAgeTracker, TracerConfig


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


def _data_available():
    return all(f.exists() for f in ROTATION_FILES) and all(
        f.exists() for f in TOPOLOGY_FILES
    )


def _make_tracker(tracker_step_myr):
    config = TracerConfig(
        tracker_step_myr=tracker_step_myr,
        tracker_point_count=2000,
        initial_spreading_rate_mm_per_yr=75.0,
        ridge_sampling_angle_deg=1.0,
        ridge_offset_angle_deg=0.01,
        collision_velocity_difference_km_per_myr=7.0,
        collision_distance_rate_km_per_myr=10.0,
    )
    return SeafloorAgeTracker(
        rotation_files=ROTATION_FILES,
        topology_files=TOPOLOGY_FILES,
        continental_polygons=CONTINENTAL_POLYGONS,
        config=config,
    )


@pytest.mark.skipif(not _data_available(), reason="GPlates data files not found")
def test_half_myr_step():
    """A half-Myr tracker step works over an integer-Ma range."""
    tracker = _make_tracker(tracker_step_myr=0.5)
    tracker.initialize(starting_age=12)
    cloud = tracker.step_to(10)

    assert tracker._current_age == pytest.approx(10.0)
    assert len(cloud.xyz) > 0


@pytest.mark.skipif(not _data_available(), reason="GPlates data files not found")
def test_truncated_final_step():
    """A noninteger tracker step can leave a shorter final step.

    A 0.7 Myr step from 11.5 Ma to 10 Ma gives spans of 0.7, 0.7, and 0.1 Myr.
    """
    tracker = _make_tracker(tracker_step_myr=0.7)
    tracker.initialize(starting_age=11.5)
    cloud = tracker.step_to(10.0)

    assert tracker._current_age == pytest.approx(10.0)
    assert len(cloud.xyz) > 0

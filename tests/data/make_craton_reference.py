"""Regenerate the committed topological ground truth for the craton seeds.

Writes tests/data/ref_craton_seeds_50ma_topological.npz — the XYZ positions of
the ~2919 non-zero craton seeds back-rotated 0 -> 50 Ma with the topological
PointRotator engine. This is the positional regression lock used by
tests/test_cratons_realdata.py::test_craton_seed_positions_match_topological_ground_truth.

Run deliberately (only when the engine legitimately changes) from the repo root:

    ~/Workplace/python3.12/bin/python3.12 tests/data/make_craton_reference.py

Requires the examples/Cratons_M3_B data to be present.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np

from gtrack import PointCloud, PointRotator, PolygonFilter

warnings.simplefilter("ignore")

HERE = Path(__file__).resolve().parent
CRATONS = HERE.parent.parent / "examples" / "Cratons_M3_B"
ZAH = CRATONS / "gplates_files" / "Zahirovic_2022"
ROTATION_FILES = [str(ZAH / "Zahirovic2022_CombinedRotations_fixed_crossovers.rot")]
TOPOLOGY_FILES = [
    str(ZAH / "Zahirovic2022_ActiveDeformation.gpmlz"),
    str(ZAH / "Zahirovic2022_InactiveDeformation.gpmlz"),
    str(ZAH / "Zahirovic2022_PlateBoundaries.gpmlz"),
]
CRATON_POLYGONS = str(CRATONS / "gplates_files" / "fast_velocity_anomalies_4pct.shp")
H5 = CRATONS / "continental_data" / "lithospheric_thickness_mesh.h5"
OUT = HERE / "ref_craton_seeds_50ma_topological.npz"


def main() -> None:
    with h5py.File(H5, "r") as f:
        lonlat = f["lonlat"][:]
        values = f["values"][:]
    latlon = np.column_stack([lonlat[:, 1], lonlat[:, 0]])
    cloud = PointCloud.from_latlon(latlon)
    cloud.add_property("thickness", values)

    pf = PolygonFilter(polygon_files=CRATON_POLYGONS, rotation_files=ROTATION_FILES)
    mask = pf.get_containment_mask(cloud, at_age=0.0)
    seeds = cloud.subset(mask)

    rotator = PointRotator(rotation_files=ROTATION_FILES, topology_files=TOPOLOGY_FILES)
    rotated = rotator.rotate(seeds, from_age=0.0, to_age=50.0)

    np.savez(OUT, xyz=rotated.xyz)
    print(f"wrote {OUT} ({rotated.n_points} seeds)")


if __name__ == "__main__":
    main()

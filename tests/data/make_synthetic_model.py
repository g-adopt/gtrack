"""Generate the tiny synthetic topological model used by the rotation tests.

The model is intentionally minimal but exercises every branch the new
topological ``PointRotator`` must handle:

* **Plate A** (id 1): a rigid *topological closed plate boundary* covering a
  square, rotating 30 deg about the north pole by 100 Ma. Points inside move
  rigidly by plate 1.
* **Deforming network** (feature id-wise a ``gpml:TopologicalNetwork``): a
  rectangle whose left edge rides plate 1 (+30 deg) and right edge rides plate 2
  (-20 deg), so the interior genuinely deforms (stretches) — this is the piece a
  rigid engine cannot reproduce.
* **Plate 2** (id 2): in the rotation circuit (has a rigid sequence), used only
  to drive the network's right edge.
* **Plate B** (id 999): referenced ONLY by a static polygon and has **no** rigid
  sequence in the rotation file, so it is *out of circuit*. A static-polygon
  plate-id assignment tags a point with 999; the old rigid engine would drop it,
  the new topological engine advects it by whatever topology it sits in.

Running this file (re)generates the three fixture files
``synthetic.rot``, ``synthetic_topology.gpml`` and ``synthetic_static.gpml``.
The ``synthetic_model`` fixture in tests/conftest.py generates them on demand if
they are missing, so a clean checkout does not depend on them being present.
"""

from __future__ import annotations

from pathlib import Path

import pygplates

HERE = Path(__file__).resolve().parent
ROT_PATH = HERE / "synthetic.rot"
TOPO_PATH = HERE / "synthetic_topology.gpml"
STATIC_PATH = HERE / "synthetic_static.gpml"

# Geometry of plate A's topological boundary (a square) and the deforming
# network rectangle. Kept well separated so the two topologies never overlap.
PLATE_A_SQUARE = dict(lat=(-40.0, 40.0), lon=(-40.0, 40.0))
NETWORK_RECT = dict(lat=(-30.0, 30.0), lon=(60.0, 120.0))

# A point that sits geographically inside plate A but is labelled plate B (999)
# by the static polygon — the out-of-circuit regression point.
OUT_OF_CIRCUIT_POINT = (0.0, 0.0)  # (lat, lon)
OUT_OF_CIRCUIT_PLATE = 999


def _line(coords, plate_id, name):
    f = pygplates.Feature()
    f.set_geometry(
        pygplates.PolylineOnSphere(
            [pygplates.PointOnSphere(lat, lon) for lat, lon in coords]
        )
    )
    f.set_reconstruction_plate_id(plate_id)
    f.set_valid_time(600.0, -999.0)
    f.set_name(name)
    return f


def _polygon(lat_range, lon_range, plate_id, name):
    la0, la1 = lat_range
    lo0, lo1 = lon_range
    ring = [(la1, lo0), (la1, lo1), (la0, lo1), (la0, lo0)]
    f = pygplates.Feature()
    f.set_geometry(
        pygplates.PolygonOnSphere([pygplates.PointOnSphere(la, lo) for la, lo in ring])
    )
    f.set_reconstruction_plate_id(plate_id)
    f.set_valid_time(600.0, -999.0)
    f.set_name(name)
    return f


def build_rotation_file() -> None:
    lines = [
        "0    0.0  0.0 0.0  0.0 000 !anchor",
        "0  600.0  0.0 0.0  0.0 000 !anchor",
        # Plate A (1): +30 deg about the north pole by 100 Ma.
        "1    0.0  0.0 0.0  0.0 000 !A@0",
        "1  100.0 90.0 0.0 30.0 000 !A@100",
        "1  600.0 90.0 0.0 30.0 000 !A@600",
        # Plate 2: -20 deg about the north pole by 100 Ma (in circuit).
        "2    0.0  0.0 0.0   0.0 000 !2@0",
        "2  100.0 90.0 0.0 -20.0 000 !2@100",
        "2  600.0 90.0 0.0 -20.0 000 !2@600",
        # NOTE: plate 999 (B) is deliberately absent -> out of circuit.
    ]
    ROT_PATH.write_text("\n".join(lines) + "\n")


def build_topology_file() -> None:
    # Plate A boundary: four rigid line sections on plate 1.
    la0, la1 = PLATE_A_SQUARE["lat"]
    lo0, lo1 = PLATE_A_SQUARE["lon"]
    a_lines = [
        _line([(la1, lo0), (la1, lo1)], 1, "A_top"),
        _line([(la1, lo1), (la0, lo1)], 1, "A_right"),
        _line([(la0, lo1), (la0, lo0)], 1, "A_bot"),
        _line([(la0, lo0), (la1, lo0)], 1, "A_left"),
    ]
    a_sections = [pygplates.GpmlTopologicalSection.create(l) for l in a_lines]
    a_poly = pygplates.GpmlTopologicalPolygon(a_sections)
    a_feat = pygplates.Feature.create_topological_feature(
        pygplates.FeatureType.gpml_topological_closed_plate_boundary,
        a_poly,
        name="PlateA",
        valid_time=(600.0, -999.0),
    )
    a_feat.set_reconstruction_plate_id(1)

    # Deforming network: left edge on plate 1, right edge on plate 2.
    nla0, nla1 = NETWORK_RECT["lat"]
    nlo0, nlo1 = NETWORK_RECT["lon"]
    n_lines = [
        _line([(nla1, nlo0), (nla0, nlo0)], 1, "N_left"),
        _line([(nla0, nlo0), (nla0, nlo1)], 2, "N_bot"),
        _line([(nla0, nlo1), (nla1, nlo1)], 2, "N_right"),
        _line([(nla1, nlo1), (nla1, nlo0)], 1, "N_top"),
    ]
    n_sections = [pygplates.GpmlTopologicalSection.create(l) for l in n_lines]
    n_net = pygplates.GpmlTopologicalNetwork(n_sections)
    n_feat = pygplates.Feature.create_topological_network_feature(
        n_net, name="NetX", valid_time=(600.0, -999.0)
    )
    n_feat.set_reconstruction_plate_id(1)

    fc = pygplates.FeatureCollection(a_lines + n_lines + [a_feat, n_feat])
    fc.write(str(TOPO_PATH))


def build_static_polygons() -> None:
    # A small static polygon around the out-of-circuit point, tagged plate 999.
    lat, lon = OUT_OF_CIRCUIT_POINT
    b_poly = _polygon(
        (lat - 5.0, lat + 5.0), (lon - 5.0, lon + 5.0), OUT_OF_CIRCUIT_PLATE, "PlateB"
    )
    # A larger static polygon covering the rest of plate A's square, tagged 1.
    a_static = _polygon(
        PLATE_A_SQUARE["lat"], PLATE_A_SQUARE["lon"], 1, "A_static"
    )
    fc = pygplates.FeatureCollection([b_poly, a_static])
    fc.write(str(STATIC_PATH))


def main() -> None:
    build_rotation_file()
    build_topology_file()
    build_static_polygons()
    print(f"wrote {ROT_PATH.name}, {TOPO_PATH.name}, {STATIC_PATH.name}")


if __name__ == "__main__":
    main()

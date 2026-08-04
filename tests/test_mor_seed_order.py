"""Tests for the deterministic ordering of shared sub-segments.

pygplates returns ``get_shared_sub_segments()`` in an order that varies between
processes, so anything iterating it emits the same result permuted from one run
to the next. ``ordered_sub_segments`` sorts them by resolved geometry first.

The ordering tests need no plate data: they hand the sorter stub sub-segments
in deliberately shuffled orders and assert the output does not move. A stub is
the right instrument here, because the defect being guarded against is a
property of input order alone, and the real permutation cannot be provoked
inside a single process — it varies between processes, not within one.

One test does use real data, to check the assumption the whole fix rests on:
that the key is a total order on sections that actually occur. It is skipped
when the example data is absent, like its neighbours.
"""

from itertools import permutations
from pathlib import Path

import pytest

from gtrack.mor_seeds import _sub_segment_sort_key, ordered_sub_segments


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


def _data_files_exist():
    return all(f.exists() for f in ROTATION_FILES + TOPOLOGY_FILES)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class FakePoint:
    """A point that reports a fixed lat/lon."""

    def __init__(self, lat, lon):
        self._lat_lon = (lat, lon)

    def to_lat_lon(self):
        return self._lat_lon


class FakeGeometry:
    """A resolved geometry over a fixed list of points."""

    def __init__(self, points):
        self._points = points

    def get_points(self):
        return self._points


class FakeSubSegment:
    """A shared sub-segment carrying a geometry and a section-wide feature id.

    ``feature_id`` defaults to the same value for every instance on purpose:
    that is what pygplates does, since a sub-segment reports the id of the
    section it belongs to rather than its own. A sorter that reached for it
    would be sorting on a constant.
    """

    def __init__(self, label, points, feature_id="SECTION-WIDE-ID"):
        self.label = label
        self._geometry = FakeGeometry([FakePoint(lat, lon) for lat, lon in points])
        self.feature_id = feature_id

    def get_resolved_geometry(self):
        return self._geometry

    def __repr__(self):
        return f"<FakeSubSegment {self.label}>"


class FakeSection:
    """A boundary section that hands back its sub-segments in a given order."""

    def __init__(self, sub_segments):
        self._sub_segments = list(sub_segments)

    def get_shared_sub_segments(self):
        return list(self._sub_segments)


def triple_junction_section():
    """Three sub-segments terminating at one shared point.

    This is the shape that first exposed the defect: several sub-segments of
    one section meeting at a triple junction, so their last points are
    identical and only the first point separates them.
    """
    junction = (-6.248899, 30.161842)
    return [
        FakeSubSegment("a", [(-10.0, 25.0), (-8.0, 27.0), junction]),
        FakeSubSegment("b", [(2.0, 40.0), junction]),
        FakeSubSegment("c", [(-20.0, 33.0), (-15.0, 31.5), (-9.0, 30.5), junction]),
    ]


# ---------------------------------------------------------------------------
# The order is invariant under the order it arrives in
# ---------------------------------------------------------------------------

class TestOrderingIsStable:
    def test_every_input_order_gives_one_output_order(self):
        subs = triple_junction_section()
        outputs = {
            tuple(s.label for s in ordered_sub_segments(FakeSection(order)))
            for order in permutations(subs)
        }
        # All six presentations of three sub-segments must collapse to one
        # answer. Without the sort this set has six members.
        assert len(outputs) == 1

    def test_the_one_order_is_the_sorted_one(self):
        subs = triple_junction_section()
        result = ordered_sub_segments(FakeSection(reversed(subs)))
        assert [s.label for s in result] == ["c", "a", "b"]
        keys = [_sub_segment_sort_key(s) for s in result]
        assert keys == sorted(keys)

    def test_shared_last_point_does_not_collide(self):
        # All three end at the junction, so a key built on the last point
        # alone would tie them and leave the order to the input.
        keys = [_sub_segment_sort_key(s) for s in triple_junction_section()]
        assert len({k[-1] for k in keys}) == 1  # last points genuinely identical
        assert len(set(keys)) == 3              # full geometries separate them

    def test_identical_feature_ids_are_not_an_obstacle(self):
        # Every sub-segment reports the section's feature id, so the ids are
        # all equal here, as they are in the real data. Ordering must not
        # depend on them.
        subs = triple_junction_section()
        assert len({s.feature_id for s in subs}) == 1
        assert len(ordered_sub_segments(FakeSection(subs))) == 3

    def test_sorting_does_not_drop_or_duplicate(self):
        subs = triple_junction_section()
        result = ordered_sub_segments(FakeSection(subs))
        assert sorted(s.label for s in result) == ["a", "b", "c"]
        assert {id(s) for s in result} == {id(s) for s in subs}


class TestSortKey:
    def test_key_is_the_whole_geometry(self):
        sub = FakeSubSegment("k", [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])
        assert _sub_segment_sort_key(sub) == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))

    def test_same_endpoints_different_interior_still_separate(self):
        # The narrower (first, last, n_points) key would tie these two: same
        # start, same end, same length, different middle.
        a = FakeSubSegment("a", [(0.0, 0.0), (1.0, 5.0), (2.0, 2.0)])
        b = FakeSubSegment("b", [(0.0, 0.0), (1.0, -5.0), (2.0, 2.0)])
        assert _sub_segment_sort_key(a) != _sub_segment_sort_key(b)

    def test_identical_geometries_tie(self):
        # And they are meant to. A tie is only reachable by being the same
        # shape, and every loop in mor_seeds reads nothing but the geometry,
        # so the two contribute identical output either way round.
        a = FakeSubSegment("a", [(0.0, 0.0), (1.0, 1.0)])
        b = FakeSubSegment("b", [(0.0, 0.0), (1.0, 1.0)])
        assert _sub_segment_sort_key(a) == _sub_segment_sort_key(b)


# ---------------------------------------------------------------------------
# The assumption the fix rests on, checked against real sections
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _data_files_exist(), reason="GPlates data files not found")
@pytest.mark.parametrize("age", [0.0, 100.0, 230.0])
def test_any_tie_on_real_sections_is_between_identical_geometries(age):
    """Ties are permitted, but only where they cannot matter.

    A tie leaves the relative order of those sub-segments to the input, which
    is the very thing that varies between processes. That is harmless exactly
    when the tied sub-segments are geometrically identical, because the loops
    in mor_seeds read nothing else from them. This asserts the key never ties
    two sub-segments that differ.

    Matthews 2016 at 0 Ma really does contain such a tie — a section with two
    coincident two-point sub-segments — so this is a live check, not a
    hypothetical one. Multi-sub-segment MOR sections are common rather than
    exotic, so it runs at several ages.
    """
    import pygplates

    rotation_model = pygplates.RotationModel([str(f) for f in ROTATION_FILES])
    topologies = [pygplates.FeatureCollection(str(f)) for f in TOPOLOGY_FILES]

    resolved, sections = [], []
    pygplates.resolve_topologies(topologies, rotation_model, resolved, age, sections)

    mid_ocean_ridge = pygplates.FeatureType.create_gpml("MidOceanRidge")
    checked = 0
    for section in sections:
        if section.get_feature().get_feature_type() != mid_ocean_ridge:
            continue
        sub_segments = section.get_shared_sub_segments()
        if len(sub_segments) < 2:
            continue
        checked += 1
        keys = [_sub_segment_sort_key(s) for s in sub_segments]
        for key in set(keys):
            tied = [s for s, k in zip(sub_segments, keys) if k == key]
            if len(tied) < 2:
                continue
            geometries = {
                tuple(p.to_lat_lon() for p in s.get_resolved_geometry().get_points())
                for s in tied
            }
            assert len(geometries) == 1, (
                f"at {age} Ma, {len(tied)} sub-segments share a sort key but "
                f"differ in geometry — the order between them would be left to "
                f"pygplates and would vary between processes"
            )

    assert checked > 0, f"no multi-sub-segment MOR section at {age} Ma to check"

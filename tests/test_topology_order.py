"""Tests for deterministic shared sub-segment ordering.

pygplates returns ``get_shared_sub_segments()`` in an order that varies between
processes. ``gtrack.topology_order`` sorts them so the callers that iterate
them stop inheriting that variation.

The ordering tests are data-free: they hand the sorter stub sub-segments in
every permutation and assert the output does not move. Stubs are the right
instrument, because the defect is a property of input order alone and the real
permutation cannot be provoked inside one process — it varies between
processes, not within one.

The real-data tests simulate the permutation faithfully instead: they wrap each
resolved section in a shim that hands its sub-segments back shuffled, run the
consumer through the ordering helper as production does, and require the answer
to be unchanged. Without the sort those tests fail.
"""

import random
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from gtrack.topology_order import ordered_sub_segments, sub_segment_sort_key


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

# 0 Ma is not an arbitrary choice: it is the age at which Matthews 2016 has two
# coincident sub-segments in one section, bounding different plates (911 and
# 923). That is the case the sort key cannot separate, so it is the case worth
# running the consumers against.
TIE_AGE = 0.0


def _data_files_exist():
    return all(f.exists() for f in ROTATION_FILES + TOPOLOGY_FILES)


requires_data = pytest.mark.skipif(
    not _data_files_exist(), reason="GPlates data files not found"
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class FakePoint:
    def __init__(self, lat, lon):
        self._lat_lon = (lat, lon)

    def to_lat_lon(self):
        return self._lat_lon


class FakeGeometry:
    def __init__(self, points):
        self._points = points

    def get_points(self):
        return self._points


class FakeSubSegment:
    """A sub-segment with a geometry and a section-wide feature id.

    ``feature_id`` is the same for every instance by default because that is
    what pygplates does — a sub-segment reports the id of the section it
    belongs to, not its own — so a sorter reaching for it would be sorting on
    a constant.
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
    def __init__(self, sub_segments):
        self._sub_segments = list(sub_segments)

    def get_shared_sub_segments(self):
        return list(self._sub_segments)


class ShuffledSection:
    """Wraps a real section and hands its sub-segments back in a fixed shuffle.

    This is how the process-to-process permutation is reproduced inside one
    process: the sub-segments and their geometries are the genuine resolved
    ones, only the sequence is disturbed.
    """

    def __init__(self, section, seed):
        self._sub_segments = list(section.get_shared_sub_segments())
        random.Random(seed).shuffle(self._sub_segments)

    def get_shared_sub_segments(self):
        return list(self._sub_segments)


def triple_junction_section():
    """Three sub-segments terminating at one shared point.

    The shape that first exposed the defect: sub-segments of one section
    meeting at a triple junction, so their last points coincide and only the
    rest of the geometry separates them.
    """
    junction = (-6.248899, 30.161842)
    return [
        FakeSubSegment("a", [(-10.0, 25.0), (-8.0, 27.0), junction]),
        FakeSubSegment("b", [(2.0, 40.0), junction]),
        FakeSubSegment("c", [(-20.0, 33.0), (-15.0, 31.5), (-9.0, 30.5), junction]),
    ]


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------

class TestOrderingIsStable:
    def test_every_input_order_gives_one_output_order(self):
        subs = triple_junction_section()
        outputs = {
            tuple(s.label for s in ordered_sub_segments(FakeSection(order)))
            for order in permutations(subs)
        }
        # Six presentations, one answer. Without the sort, six answers.
        assert len(outputs) == 1

    def test_the_one_order_is_the_sorted_one(self):
        subs = triple_junction_section()
        result = ordered_sub_segments(FakeSection(reversed(subs)))
        assert [s.label for s in result] == ["c", "a", "b"]
        assert [sub_segment_sort_key(s) for s in result] == sorted(
            sub_segment_sort_key(s) for s in result
        )

    def test_shared_last_point_does_not_collide(self):
        keys = [sub_segment_sort_key(s) for s in triple_junction_section()]
        assert len({k[-1] for k in keys}) == 1  # last points genuinely identical
        assert len(set(keys)) == 3              # full geometries separate them

    def test_identical_feature_ids_are_not_an_obstacle(self):
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
        assert sub_segment_sort_key(sub) == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))

    def test_same_endpoints_different_interior_still_separate(self):
        # (first, last, n_points) would tie these; the full geometry does not.
        a = FakeSubSegment("a", [(0.0, 0.0), (1.0, 5.0), (2.0, 2.0)])
        b = FakeSubSegment("b", [(0.0, 0.0), (1.0, -5.0), (2.0, 2.0)])
        assert sub_segment_sort_key(a) != sub_segment_sort_key(b)

    def test_identical_geometries_tie(self):
        # Deliberately. A tie is only reachable by being the same shape, which
        # is what makes it safe wherever the caller reads only the geometry.
        a = FakeSubSegment("a", [(0.0, 0.0), (1.0, 1.0)])
        b = FakeSubSegment("b", [(0.0, 0.0), (1.0, 1.0)])
        assert sub_segment_sort_key(a) == sub_segment_sort_key(b)


# ---------------------------------------------------------------------------
# Real sections
# ---------------------------------------------------------------------------

def _resolve(age):
    import pygplates

    rotation_model = pygplates.RotationModel([str(f) for f in ROTATION_FILES])
    topologies = [pygplates.FeatureCollection(str(f)) for f in TOPOLOGY_FILES]
    resolved, sections = [], []
    pygplates.resolve_topologies(topologies, rotation_model, resolved, age, sections)
    return resolved, sections


def _coords(geometries):
    return [tuple(p.to_lat_lon() for p in g.get_points()) for g in geometries]


@requires_data
@pytest.mark.parametrize("age", [0.0, 100.0, 230.0])
def test_any_tie_on_real_sections_is_between_identical_geometries(age):
    """Ties are permitted, but only where they cannot matter.

    A tie leaves the relative order of those sub-segments to the input, which
    is the thing that varies between processes. That is safe exactly when the
    tied sub-segments are geometrically identical. Matthews 2016 at 0 Ma
    really does contain such a tie, so this is a live check.
    """
    _, sections = _resolve(age)
    checked = 0
    for section in sections:
        sub_segments = section.get_shared_sub_segments()
        if len(sub_segments) < 2:
            continue
        checked += 1
        keys = [sub_segment_sort_key(s) for s in sub_segments]
        for key in set(keys):
            tied = [s for s, k in zip(sub_segments, keys) if k == key]
            if len(tied) > 1:
                assert len(set(_coords(s.get_resolved_geometry() for s in tied))) == 1
    assert checked > 0, f"no multi-sub-segment section at {age} Ma to check"


@requires_data
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ridge_and_subduction_geometries_survive_a_shuffle(seed, monkeypatch):
    """The two boundaries.py extractors must not see the permutation.

    Both append nothing but the resolved geometry, so ordering them by that
    geometry is enough: the returned list is identical however the
    sub-segments were presented.
    """
    import gtrack.boundaries as boundaries

    topologies = [str(f) for f in TOPOLOGY_FILES]
    rotation_model = _rotation_model()

    def extract():
        return (
            _coords(
                boundaries.extract_ridge_geometries(
                    TIE_AGE, topologies, rotation_model
                )
            ),
            _coords(
                boundaries.extract_subduction_geometries(
                    TIE_AGE, topologies, rotation_model
                )
            ),
        )

    baseline = extract()

    real = boundaries.ordered_sub_segments
    monkeypatch.setattr(
        boundaries,
        "ordered_sub_segments",
        lambda section: real(ShuffledSection(section, seed)),
    )
    assert extract() == baseline


def _rotation_model():
    import pygplates

    return pygplates.RotationModel([str(f) for f in ROTATION_FILES])


@requires_data
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_initial_ages_survive_a_shuffle(seed, monkeypatch):
    """compute_initial_ages is the site that reads more than the geometry.

    It reads ``get_sharing_resolved_topologies()`` as well, and at 0 Ma the
    two tied sub-segments genuinely bound different plates — 911 and 923 — so
    the tie the sort key cannot break is live here. It still cannot reach the
    answer: the sharing topologies decide *whether* a sub-segment is appended,
    the geometry is *what* is appended, and the list is consumed as a minimum
    over distances. This pins that, bitwise.
    """
    import pygplates

    import gtrack.initial_conditions as initial_conditions
    from gtrack.mesh import create_sphere_mesh_latlon

    lats, lons = create_sphere_mesh_latlon(4000)
    points = pygplates.MultiPointOnSphere(
        [(float(a), float(o)) for a, o in zip(lats, lons)]
    )
    resolved, sections = _resolve(TIE_AGE)

    baseline = initial_conditions.compute_initial_ages(points, resolved, sections)

    real = initial_conditions.ordered_sub_segments
    monkeypatch.setattr(
        initial_conditions, "ordered_sub_segments",
        lambda section: real(ShuffledSection(section, seed)),
    )
    shuffled = initial_conditions.compute_initial_ages(points, resolved, sections)

    for before, after in zip(baseline, shuffled):
        np.testing.assert_array_equal(before, after)


@requires_data
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_mor_seeds_survive_a_shuffle(seed, monkeypatch):
    """The seeding loops are where the defect was originally found.

    generate_mor_seeds appends seeds along each sub-segment in the order the
    sub-segments arrive, and the tracker concatenates that verbatim, so a
    permutation here is what made a cold tracker walk irreproducible. All three
    loops read nothing but the resolved geometry, so ordering by geometry has
    to make the emitted seeds identical however the sub-segments are presented
    — not merely the same set, the same array.
    """
    import pygplates

    import gtrack.mor_seeds as mor_seeds

    topologies = [pygplates.FeatureCollection(str(f)) for f in TOPOLOGY_FILES]
    rotation_model = _rotation_model()

    def seeds():
        return mor_seeds.generate_mor_seeds(TIE_AGE, topologies, rotation_model)

    base_lats, base_lons = seeds()
    assert len(base_lats) > 0, "no MOR seeds generated; the test would be vacuous"

    real = mor_seeds.ordered_sub_segments
    monkeypatch.setattr(
        mor_seeds,
        "ordered_sub_segments",
        lambda section: real(ShuffledSection(section, seed)),
    )
    shuffled_lats, shuffled_lons = seeds()

    np.testing.assert_array_equal(base_lats, shuffled_lats)
    np.testing.assert_array_equal(base_lons, shuffled_lons)

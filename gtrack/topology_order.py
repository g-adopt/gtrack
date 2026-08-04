"""Deterministic ordering of resolved topology sub-segments.

pygplates returns ``shared_boundary_section.get_shared_sub_segments()`` in an
order that varies between processes: the sequence contains the same
sub-segments, permuted. Any routine that iterates it therefore produces the
same answer permuted from one run to the next, which is enough to make a
tracker walk irreproducible once the permutation feeds a downstream algorithm
that is sensitive to input order.

Sorting the sub-segments by their resolved geometry removes the variation at
source, and it is the only thing this module does. It changes what no caller
computes — the set of sub-segments is untouched, only the order they are
visited in.

The module deliberately imports nothing, so any module can use it without
acquiring a dependency on plate-boundary or seeding code.
"""


def sub_segment_sort_key(shared_sub_segment) -> tuple:
    """Return a deterministic sort key for a shared sub-segment.

    The key is the resolved geometry itself, as the full sequence of
    ``(lat, lon)`` pairs. The geometry *values* are deterministic even though
    the order pygplates hands the sub-segments back in is not, so they are the
    only thing available to sort on.

    Three narrower keys were considered and each fails:

    - The **feature id** is the *section's* id, carried by every sub-segment of
      that section alike, so sorting on it is a no-op that looks like a fix.
    - The **sharing plate ids** are not a total order. In the section that
      first exposed this (a MidOceanRidge in Muller 2022 at 230 Ma, five
      sub-segments) they are [101], [701], [505], [101, 505, 701] and [101] —
      101 appears twice.
    - **First point, last point and length** is not a total order either, and
      this one is measured rather than hypothetical: Matthews 2016 at 0 Ma has
      a MOR section whose three sub-segments include two agreeing on all five
      values — both run from (-34.54495002, -109.25705001) to
      (-34.5453, -109.2551) in two points. Validating a key against one plate
      model is not enough; that key holds on Muller 2022 and fails here.

    Worth recording about the Muller section, since it is the shape that makes
    endpoint keys tempting and wrong: its first three sub-segments all
    terminate at the same point, the triple junction at (-6.248899, 30.161842)
    between plates 101, 505 and 701. Sub-segments meeting at a junction share
    endpoints by construction, which is why no single endpoint distinguishes
    them.

    Ties on the full geometry are still possible and are harmless wherever the
    caller reads nothing but the geometry, since two sub-segments can only tie
    by being the same shape. The sort is stable, so tied entries keep their
    input order — the one residual piece of process-dependence — and each
    caller has to be able to show that ordering cannot reach its result. See
    the module docstring of each caller for that argument.

    Args:
        shared_sub_segment: a pygplates shared sub-segment.

    Returns:
        A tuple of ``(lat, lon)`` pairs, one per point of the resolved
        geometry, in geometry order.
    """
    return tuple(
        point.to_lat_lon()
        for point in shared_sub_segment.get_resolved_geometry().get_points()
    )


def ordered_sub_segments(shared_boundary_section) -> list:
    """Return a boundary section's shared sub-segments in a stable order.

    Args:
        shared_boundary_section: a resolved topological section.

    Returns:
        The section's shared sub-segments, sorted by
        :func:`sub_segment_sort_key`.
    """
    return sorted(
        shared_boundary_section.get_shared_sub_segments(), key=sub_segment_sort_key
    )

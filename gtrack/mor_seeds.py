"""
MOR seed point generation using stage rotation (GPlately approach).

This module provides functions to generate new seafloor points at
mid-ocean ridges using the stage rotation between spreading plates.
"""

import numpy as np
from typing import List, Optional, Tuple

import pygplates


def _sub_segment_sort_key(shared_sub_segment) -> tuple:
    """Return a deterministic sort key for a shared sub-segment.

    The key is the resolved geometry itself, as the full sequence of
    ``(lat, lon)`` pairs. The geometry *values* are deterministic even though
    the order pygplates hands the sub-segments back in is not, so they are the
    only thing available to sort on.

    Three narrower keys were considered and each fails:

    - The **feature id** is the *section's* id, carried by every sub-segment
      of that section alike, so sorting on it is a no-op that looks like a fix.
    - The **sharing plate ids** are not a total order. In the section that
      first exposed this (a MidOceanRidge in Muller 2022 at 230 Ma, five
      sub-segments) they are [101], [701], [505], [101, 505, 701] and [101] —
      101 appears twice.
    - **First point, last point and length** is not a total order either, which
      is measurable rather than hypothetical: Matthews 2016 at 0 Ma has a MOR
      section whose three sub-segments include two that agree on all five
      values — both run from (-34.54495002, -109.25705001) to
      (-34.5453, -109.2551) in two points.

    Ties on the full geometry are harmless, and that is what makes this key
    sufficient rather than merely longer. Two sub-segments can only tie by
    being geometrically identical, and every loop in this module reads nothing
    from a sub-segment except its resolved geometry, so identical geometries
    contribute identical output whichever way round they are visited. The sort
    is stable, so their relative order follows the input, and that residual
    freedom cannot reach the result.

    Worth recording about the Muller section, since it is the shape that makes
    this awkward: its first three sub-segments all terminate at the same point,
    the triple junction at (-6.248899, 30.161842) between plates 101, 505 and
    701. Sub-segments meeting at a junction share endpoints, which is why no
    single endpoint distinguishes them.

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

    pygplates returns ``get_shared_sub_segments()`` in an order that varies
    between processes, so any routine that iterates it emits the same result
    permuted from one run to the next. Sorting here makes every caller
    reproducible without changing what any of them compute: the set of
    sub-segments is unchanged, only the order they arrive in.

    Args:
        shared_boundary_section: a resolved topological section.

    Returns:
        The section's shared sub-segments, sorted by
        :func:`_sub_segment_sort_key`.
    """
    return sorted(
        shared_boundary_section.get_shared_sub_segments(), key=_sub_segment_sort_key
    )


def get_stage_rotation_for_reconstructed_geometry(
    spreading_feature,
    rotation_model: pygplates.RotationModel,
    spreading_time: float,
    return_left_right_plates: bool = False,
):
    """
    Find the stage rotation of a spreading feature in its reconstructed geometry frame.

    The returned stage rotation can be used to find the spreading direction
    (stage pole) relative to the reconstructed mid-ocean ridge geometry.

    Parameters
    ----------
    spreading_feature : pygplates.Feature
        A MidOceanRidge feature with left/right plate IDs.
    rotation_model : pygplates.RotationModel
        The rotation model.
    spreading_time : float
        Time at which spreading is happening (Ma).
    return_left_right_plates : bool, default=False
        If True, return (stage_rotation, left_plate_id, right_plate_id).

    Returns
    -------
    pygplates.FiniteRotation or tuple or None
        The stage rotation in the frame of the reconstructed geometry.
        If return_left_right_plates is True, returns a 3-tuple.
        Returns None if the feature doesn't have valid plate IDs.

    Notes
    -----
    This function matches GPlately's implementation in
    `ptt.separate_ridge_transform_segments.get_stage_rotation_for_reconstructed_geometry`.

    The stage rotation is computed as:
        R(0->t, A->Left) * R(t+1->t, Left->Right) * R(0->t, A->Left)^-1

    This transforms the raw stage rotation into the reference frame of the
    reconstructed geometry so that the stage pole can be directly compared
    to the ridge geometry.
    """
    # Check for left/right plate IDs (required for MidOceanRidge features)
    left_plate_id = spreading_feature.get_left_plate(None)
    right_plate_id = spreading_feature.get_right_plate(None)

    if left_plate_id is None or right_plate_id is None:
        return None

    # Get stage rotation from right plate to left plate over 1 Myr
    # This gives the relative motion between the two plates
    stage_rotation = rotation_model.get_rotation(
        spreading_time, right_plate_id, spreading_time + 1, left_plate_id
    )

    if stage_rotation.represents_identity_rotation():
        return None

    # Transform stage rotation to the reference frame of the reconstructed geometry
    # This allows the stage pole to be directly compared to the ridge geometry
    from_stage_pole_reference_frame = rotation_model.get_rotation(
        spreading_time, left_plate_id
    )
    to_stage_pole_reference_frame = from_stage_pole_reference_frame.get_inverse()

    stage_rotation = (
        from_stage_pole_reference_frame
        * stage_rotation
        * to_stage_pole_reference_frame
    )

    if return_left_right_plates:
        return stage_rotation, left_plate_id, right_plate_id

    return stage_rotation


def generate_mor_seeds(
    time: float,
    topology_features,
    rotation_model: pygplates.RotationModel,
    ridge_sampling_degrees: float = 0.5,
    spreading_offset_degrees: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate MOR seed points using stage rotation approach.

    For each mid-ocean ridge segment, creates pairs of points offset
    perpendicular to the ridge using the stage rotation pole. This
    ensures new crust is created on both sides of the spreading center.

    Parameters
    ----------
    time : float
        Current geological time (Ma).
    topology_features : pygplates.FeatureCollection or list
        Topology feature files or collection.
    rotation_model : pygplates.RotationModel
        The rotation model.
    ridge_sampling_degrees : float, default=0.5
        Resolution for tessellating ridges in degrees (~50 km at equator).
    spreading_offset_degrees : float, default=0.01
        Angle in degrees to rotate points off the ridge (~1 km).
        GPlately uses 0.01 degrees.

    Returns
    -------
    lats : np.ndarray
        Latitudes of seed points in degrees.
    lons : np.ndarray
        Longitudes of seed points in degrees.

    Notes
    -----
    The algorithm:
    1. Resolve topologies at the current time
    2. For each MidOceanRidge boundary section:
       - Get the stage rotation (relative motion between plates)
       - Extract the stage pole (spreading axis)
       - Tessellate the ridge at the sampling resolution
       - For each ridge point (excluding endpoints):
         - Rotate +spreading_offset_degrees around stage pole
         - Rotate -spreading_offset_degrees around stage pole
       - Add both rotated points (creating symmetric spreading)
    3. All seed points have age=0 (just formed at ridge)

    This matches GPlately's `_generate_mid_ocean_ridge_points` function.

    Examples
    --------
    >>> lats, lons = generate_mor_seeds(
    ...     time=100.0,
    ...     topology_features=topology_features,
    ...     rotation_model=rotation_model
    ... )
    >>> ages = np.zeros(len(lats))  # New crust has age 0
    """
    # Resolve topologies at the current time
    resolved_topologies = []
    shared_boundary_sections = []
    pygplates.resolve_topologies(
        topology_features,
        rotation_model,
        resolved_topologies,
        time,
        shared_boundary_sections,
    )

    all_lats = []
    all_lons = []

    # Process each boundary section
    for shared_boundary_section in shared_boundary_sections:
        # Only process MidOceanRidge features
        if (
            shared_boundary_section.get_feature().get_feature_type()
            != pygplates.FeatureType.create_gpml("MidOceanRidge")
        ):
            continue

        spreading_feature = shared_boundary_section.get_feature()

        # Get stage rotation for this spreading feature
        stage_rotation = get_stage_rotation_for_reconstructed_geometry(
            spreading_feature, rotation_model, time
        )

        if stage_rotation is None:
            # Skip if we can't get a valid stage rotation
            continue

        # Get the stage pole (spreading axis)
        stage_pole, _ = stage_rotation.get_euler_pole_and_angle()

        # Create rotations to offset points from ridge
        # One rotates "left", the other "right" relative to spreading direction
        rotate_one_way = pygplates.FiniteRotation(
            stage_pole, np.radians(spreading_offset_degrees)
        )
        rotate_opposite_way = rotate_one_way.get_inverse()

        # Process each sub-segment of the ridge
        for shared_sub_segment in ordered_sub_segments(shared_boundary_section):
            # Tessellate the ridge segment
            mor_points = pygplates.MultiPointOnSphere(
                shared_sub_segment.get_resolved_geometry().to_tessellated(
                    np.radians(ridge_sampling_degrees)
                )
            )

            # Get the points (skip first and last to avoid ridge endpoints)
            points = mor_points.get_points()
            if len(points) <= 2:
                continue

            # Get interior points (skip endpoints)
            interior_points = points[1:-1]

            # Batch rotate all interior points (single C++ call per rotation)
            interior_mp = pygplates.MultiPointOnSphere(interior_points)
            rotated_left = rotate_one_way * interior_mp
            rotated_right = rotate_opposite_way * interior_mp

            # Extract lat/lon from rotated points
            for p_left, p_right in zip(rotated_left.get_points(), rotated_right.get_points()):
                lat_left, lon_left = p_left.to_lat_lon()
                lat_right, lon_right = p_right.to_lat_lon()
                all_lats.extend([lat_left, lat_right])
                all_lons.extend([lon_left, lon_right])

    return np.array(all_lats), np.array(all_lons)


def generate_mor_seeds_with_plate_ids(
    time: float,
    topology_features,
    rotation_model: pygplates.RotationModel,
    ridge_sampling_degrees: float = 0.5,
    spreading_offset_degrees: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate MOR seed points with explicit plate ID assignments.

    Like generate_mor_seeds, but also returns the plate ID for each point
    based on which side of the ridge it's on (left or right plate).

    Parameters
    ----------
    time : float
        Current geological time (Ma).
    topology_features : pygplates.FeatureCollection or list
        Topology feature files or collection.
    rotation_model : pygplates.RotationModel
        The rotation model.
    ridge_sampling_degrees : float, default=0.5
        Resolution for tessellating ridges in degrees.
    spreading_offset_degrees : float, default=0.01
        Angle in degrees to rotate points off the ridge.

    Returns
    -------
    lats : np.ndarray
        Latitudes of seed points in degrees.
    lons : np.ndarray
        Longitudes of seed points in degrees.
    plate_ids : np.ndarray
        Plate IDs for each point (left plate or right plate).

    Notes
    -----
    The plate IDs are assigned based on which side of the ridge each
    point is on. Points rotated in the "left" direction get the left
    plate ID, and points rotated in the "right" direction get the right
    plate ID.

    This is useful for the first time step when we need to explicitly
    assign plate IDs rather than querying topology.
    """
    # Resolve topologies at the current time
    resolved_topologies = []
    shared_boundary_sections = []
    pygplates.resolve_topologies(
        topology_features,
        rotation_model,
        resolved_topologies,
        time,
        shared_boundary_sections,
    )

    all_lats = []
    all_lons = []
    all_plate_ids = []

    # Process each boundary section
    for shared_boundary_section in shared_boundary_sections:
        # Only process MidOceanRidge features
        if (
            shared_boundary_section.get_feature().get_feature_type()
            != pygplates.FeatureType.create_gpml("MidOceanRidge")
        ):
            continue

        spreading_feature = shared_boundary_section.get_feature()

        # Get stage rotation with plate IDs
        result = get_stage_rotation_for_reconstructed_geometry(
            spreading_feature, rotation_model, time, return_left_right_plates=True
        )

        if result is None:
            continue

        stage_rotation, left_plate_id, right_plate_id = result

        # Get the stage pole
        stage_pole, _ = stage_rotation.get_euler_pole_and_angle()

        # Create rotations to offset points
        rotate_one_way = pygplates.FiniteRotation(
            stage_pole, np.radians(spreading_offset_degrees)
        )
        rotate_opposite_way = rotate_one_way.get_inverse()

        # Process each sub-segment
        for shared_sub_segment in ordered_sub_segments(shared_boundary_section):
            mor_points = pygplates.MultiPointOnSphere(
                shared_sub_segment.get_resolved_geometry().to_tessellated(
                    np.radians(ridge_sampling_degrees)
                )
            )

            points = mor_points.get_points()
            if len(points) <= 2:
                continue

            for point in points[1:-1]:
                # Create points on each side
                point_left = rotate_one_way * point
                point_right = rotate_opposite_way * point

                lat_left, lon_left = point_left.to_lat_lon()
                lat_right, lon_right = point_right.to_lat_lon()

                # Add with respective plate IDs
                all_lats.append(lat_left)
                all_lons.append(lon_left)
                all_plate_ids.append(left_plate_id)

                all_lats.append(lat_right)
                all_lons.append(lon_right)
                all_plate_ids.append(right_plate_id)

    return np.array(all_lats), np.array(all_lons), np.array(all_plate_ids)


def get_ridge_geometries(
    time: float,
    topology_features,
    rotation_model: pygplates.RotationModel,
) -> List[pygplates.PolylineOnSphere]:
    """
    Extract mid-ocean ridge geometries at a given time.

    Parameters
    ----------
    time : float
        Geological time (Ma).
    topology_features : pygplates.FeatureCollection or list
        Topology features.
    rotation_model : pygplates.RotationModel
        Rotation model.

    Returns
    -------
    list of pygplates.PolylineOnSphere
        Ridge geometries at the specified time.
    """
    resolved_topologies = []
    shared_boundary_sections = []
    pygplates.resolve_topologies(
        topology_features,
        rotation_model,
        resolved_topologies,
        time,
        shared_boundary_sections,
    )

    ridge_geometries = []
    for shared_boundary_section in shared_boundary_sections:
        if (
            shared_boundary_section.get_feature().get_feature_type()
            == pygplates.FeatureType.create_gpml("MidOceanRidge")
        ):
            for shared_sub_segment in ordered_sub_segments(shared_boundary_section):
                ridge_geometries.append(shared_sub_segment.get_resolved_geometry())

    return ridge_geometries

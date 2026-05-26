"""
Geometric utility functions for coordinate transformations and operations on a sphere.

All functions are vectorized using numpy for performance.
"""

from pathlib import Path

import numpy as np


# Earth's radius in meters
EARTH_RADIUS = 6.3781e6


def ensure_list(x):
    """
    Ensure x is a list. Wrap single items (str, Path) in a list.

    This is useful for APIs that accept either a single file path or a list
    of file paths, allowing users to write either:
        - rotation_files="rotations.rot"
        - rotation_files=["rot1.rot", "rot2.rot"]

    Parameters
    ----------
    x : str, Path, list, or None
        Input to convert to a list.

    Returns
    -------
    list
        The input as a list. Single str/Path items are wrapped in a list.
        None returns an empty list.

    Examples
    --------
    >>> ensure_list("file.rot")
    ['file.rot']
    >>> ensure_list(Path("file.rot"))
    [PosixPath('file.rot')]
    >>> ensure_list(["file1.rot", "file2.rot"])
    ['file1.rot', 'file2.rot']
    >>> ensure_list(None)
    []
    """
    if x is None:
        return []
    if isinstance(x, (str, Path)):
        return [x]
    return list(x)


def load_features(polygons):
    """Build a single ``pygplates.FeatureCollection`` from a flexible input.

    Accepts whatever a user is likely to pass for a polygon/feature argument:
    a single filename (str/Path), a list of filenames, an already-built
    ``FeatureCollection``, a single feature, or a sequence of features.
    Filenames are read into feature collections and merged; features are added
    directly.

    This exists because pygplates only accepts a single filename, a single
    feature, or a sequence of *features* — handing it a list of filename
    strings raises "Expected an optional filename, or sequence of features, or
    a single feature". Routing every polygon argument through here gives the
    same str/list flexibility that ``ensure_list`` gives the rotation and
    topology file arguments.

    Parameters
    ----------
    polygons : str, Path, list, pygplates.FeatureCollection, pygplates.Feature, or sequence of features
        The polygon/feature input to normalise.

    Returns
    -------
    pygplates.FeatureCollection
        A single collection holding every feature from the input.
    """
    import pygplates

    if isinstance(polygons, pygplates.FeatureCollection):
        return polygons

    # A lone Feature is iterable (over its properties), so ensure_list would
    # tear it apart. Wrap it so it is added as a single feature instead.
    if isinstance(polygons, pygplates.Feature):
        polygons = [polygons]

    collection = pygplates.FeatureCollection()
    for item in ensure_list(polygons):
        if isinstance(item, (str, Path)):
            collection.add(pygplates.FeatureCollection(str(item)))
        else:
            collection.add(item)
    return collection


def LatLon2XYZ(latlon):
    """
    Convert geographic coordinates (lat, lon) to Cartesian coordinates (x, y, z).

    Parameters
    ----------
    latlon : np.ndarray
        Array of shape (N, 2) containing [latitude, longitude] pairs in degrees.

    Returns
    -------
    xyz : np.ndarray
        Array of shape (N, 3) containing [x, y, z] Cartesian coordinates in meters.

    Examples
    --------
    >>> latlons = np.array([[0, 0], [90, 0], [-90, 0]])
    >>> xyz = LatLon2XYZ(latlons)
    >>> xyz.shape
    (3, 3)
    """
    R = EARTH_RADIUS
    lat = latlon[:, 0]
    lon = latlon[:, 1]
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    x = np.zeros([len(lon), 3])  # initialize coordinate-array
    x[:, 0] = R * np.cos(lat) * np.cos(lon)  # load x-coordinates
    x[:, 1] = R * np.cos(lat) * np.sin(lon)  # load y-coordinates
    x[:, 2] = R * np.sin(lat)  # load z-coordinates
    return x


def XYZ2LatLon(xyz):
    """
    Convert Cartesian coordinates (x, y, z) to geographic coordinates (lat, lon).

    Parameters
    ----------
    xyz : np.ndarray
        Array of shape (N, 3) containing [x, y, z] Cartesian coordinates in meters.

    Returns
    -------
    lats : np.ndarray
        Array of latitudes in degrees, shape (N,).
    lons : np.ndarray
        Array of longitudes in degrees, shape (N,).

    Examples
    --------
    >>> xyz = np.array([[6.3781e6, 0, 0], [0, 0, 6.3781e6]])
    >>> lats, lons = XYZ2LatLon(xyz)
    >>> np.allclose(lats, [0, 90])
    True
    >>> np.allclose(lons, [0, 0])
    True
    """
    R = EARTH_RADIUS
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    lats = np.arcsin(z / R) * 360 / (2 * np.pi)
    lons = np.arctan2(y, x) * 360 / (2 * np.pi)
    return lats, lons


def RefineGreatCircleArcSegment(p1, p2, N):
    """
    Refine a great circle arc segment defined by two points into N segments.

    This function takes two points on a sphere and creates a refined path
    between them by interpolating along the great circle arc.

    Parameters
    ----------
    p1 : tuple or list
        First point as (latitude, longitude) in degrees.
    p2 : tuple or list
        Second point as (latitude, longitude) in degrees.
    N : int
        Number of segments to create (N+1 points total).

    Returns
    -------
    lats : np.ndarray
        Array of latitudes along the refined segment, shape (N+1,).
    lons : np.ndarray
        Array of longitudes along the refined segment, shape (N+1,).

    Notes
    -----
    Two points that define a line going through Earth's center will give
    erroneous results.

    Examples
    --------
    >>> p1 = (0, 0)  # Equator at prime meridian
    >>> p2 = (0, 90)  # Equator at 90E
    >>> lats, lons = RefineGreatCircleArcSegment(p1, p2, 2)
    >>> len(lats)
    3
    """
    R = EARTH_RADIUS

    LatLon = np.array([p1, p2])
    XYZ = LatLon2XYZ(LatLon)
    p1 = XYZ[0, :]
    p2 = XYZ[1, :]

    n = np.arange(N + 1)  # weights
    p = np.zeros([N + 1, 3])  #
    for i in range(0, N + 1):
        p[i, :] = (float(N) - n[i]) / N * p1 + (n[i] / float(N)) * p2
    x_mag = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2 + p[:, 2] ** 2)  # compute vector magnitude
    x_mag = np.transpose(np.array([x_mag, x_mag, x_mag]))
    p = (p / x_mag) * R  # project new position back to Earth surface
    lats, lons = XYZ2LatLon(p)
    return lats, lons


def Segments2Points(segments, res):
    """
    Convert great circle arc segments to a set of points with specified resolution.

    Takes line segments defined by endpoint pairs and refines them into a set of
    points spaced at approximately the specified resolution.

    Parameters
    ----------
    segments : np.ndarray
        Array of shape (N, 4) where each row is [lat1, lon1, lat2, lon2] defining
        a segment's endpoints in degrees.
    res : float
        Target resolution for point spacing in meters.

    Returns
    -------
    lats : np.ndarray
        Array of latitudes for all refined points in degrees.
    lons : np.ndarray
        Array of longitudes for all refined points in degrees.

    Examples
    --------
    >>> segments = np.array([[0, 0, 0, 10], [0, 10, 0, 20]])
    >>> lats, lons = Segments2Points(segments, 100e3)  # 100 km resolution
    >>> len(lats) > 2  # Should have more points than segments
    True
    """
    num_segments = len(segments[:, 0])
    lats, lons = np.array([]), np.array([])

    R = EARTH_RADIUS
    # Fixed: Process all segments starting from index 0 (was bug in original starting at 1)
    for i in range(0, num_segments):
        p1, p2 = segments[i, :2], segments[i, 2:]
        lat1, lon1, lat2, lon2 = np.deg2rad(p1[0]), np.deg2rad(p1[1]), np.deg2rad(p2[0]), np.deg2rad(p2[1])
        dlon = abs(lon2 - lon1)
        dist = R * np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon))

        if dist > 0:
            N = int(round(dist / res))
        else:
            N = 0

        if N > 0:
            lats_, lons_ = RefineGreatCircleArcSegment(p1, p2, N)
            lats = np.append(lats, lats_)
            lons = np.append(lons, lons_)

    return lats, lons


def normalize_to_sphere(xyz, radius=None):
    """
    Normalize XYZ coordinates to lie on a sphere of given radius.

    Parameters
    ----------
    xyz : np.ndarray
        Array of shape (N, 3) containing Cartesian coordinates.
    radius : float, optional
        Radius of the sphere in meters. If None, uses Earth's radius.

    Returns
    -------
    xyz_normalized : np.ndarray
        Array of shape (N, 3) with points projected onto the sphere.

    Examples
    --------
    >>> xyz = np.array([[1, 0, 0], [0, 1, 0]])
    >>> normalized = normalize_to_sphere(xyz, radius=1.0)
    >>> np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    True
    """
    if radius is None:
        radius = EARTH_RADIUS

    magnitudes = np.linalg.norm(xyz, axis=1, keepdims=True)
    return (xyz / magnitudes) * radius


def inverse_distance_weighted_interpolation(values, distances, epsilon=1e-10):
    """
    Compute inverse distance weighted (IDW) interpolation.

    For each query point, computes a weighted average of neighbor values
    where weights are inversely proportional to distance.

    Parameters
    ----------
    values : np.ndarray
        Array of shape (N, K) containing values from K neighbors for N points.
    distances : np.ndarray
        Array of shape (N, K) containing distances to K neighbors for N points.
    epsilon : float, default=1e-10
        Small value to detect zero distances. Points with distance < epsilon
        are treated as exact matches and their value is used directly.

    Returns
    -------
    result : np.ndarray
        Array of shape (N,) containing interpolated values.

    Notes
    -----
    For points where any neighbor has distance < epsilon (essentially zero),
    the value from that neighbor is used directly to avoid division by zero.

    The IDW formula is: result = sum(v_i / d_i) / sum(1 / d_i)

    Examples
    --------
    >>> values = np.array([[10, 20], [30, 40]])
    >>> distances = np.array([[1.0, 2.0], [1.0, 1.0]])
    >>> result = inverse_distance_weighted_interpolation(values, distances)
    >>> # First point: (10/1 + 20/2) / (1/1 + 1/2) = 20/1.5 = 13.33
    >>> # Second point: (30/1 + 40/1) / (1/1 + 1/1) = 70/2 = 35
    """
    values = np.atleast_2d(values)
    distances = np.atleast_2d(distances)

    n_points = values.shape[0]
    result = np.zeros(n_points)

    # Handle each point
    for i in range(n_points):
        dists = distances[i]
        vals = values[i]

        # Check for zero distance (exact match)
        zero_mask = dists < epsilon
        if np.any(zero_mask):
            # Use the value from the first zero-distance neighbor
            zero_idx = np.argmax(zero_mask)
            result[i] = vals[zero_idx]
        else:
            # Standard IDW: w_i = 1/d_i
            weights = 1.0 / dists
            result[i] = np.sum(vals * weights) / np.sum(weights)

    return result


def compute_mesh_spacing_km(n_points, earth_radius_km=6378.1):
    """
    Compute approximate mesh spacing for a given number of points on a sphere.

    Parameters
    ----------
    n_points : int
        Number of points on the sphere.
    earth_radius_km : float, default=6378.1
        Earth's radius in kilometers.

    Returns
    -------
    spacing_km : float
        Approximate average spacing between mesh points in kilometers.

    Notes
    -----
    Average area per point = 4 * pi * R^2 / N
    Approximate spacing = sqrt(area per point)

    Examples
    --------
    >>> spacing = compute_mesh_spacing_km(10000)
    >>> 200 < spacing < 300  # ~10k points gives ~220 km spacing
    True
    """
    avg_area = 4 * np.pi * earth_radius_km**2 / n_points
    return np.sqrt(avg_area)

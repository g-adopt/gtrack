"""
Input/output utilities for PointCloud data.

Supports multiple formats:
- NumPy (.npy, .npz): Fast binary format for Python workflows
- GPML (.gpml, .gpmlz): Native pygplates/GPlates format
- CSV/text: Human-readable format

Also provides checkpointing functionality for saving/restoring state.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np


def load_points_numpy(
    filepath: Union[str, Path],
    xyz_columns: Tuple[int, int, int] = (0, 1, 2),
    property_columns: Optional[Dict[str, int]] = None
) -> "PointCloud":
    """
    Load points from numpy file.

    Parameters
    ----------
    filepath : str or Path
        Path to .npy or .npz file.
    xyz_columns : tuple, default=(0, 1, 2)
        Column indices for x, y, z coordinates (for .npy files).
    property_columns : dict, optional
        Mapping from property name to column index (for .npy files).

    Returns
    -------
    PointCloud
        Loaded points.

    Examples
    --------
    >>> # Load from .npz with xyz and properties
    >>> cloud = load_points_numpy('points.npz')
    >>>
    >>> # Load from .npy with specific columns
    >>> cloud = load_points_numpy(
    ...     'data.npy',
    ...     xyz_columns=(0, 1, 2),
    ...     property_columns={'depth': 3, 'temperature': 4}
    ... )
    """
    from .point_rotation import PointCloud

    filepath = Path(filepath)

    if filepath.suffix == '.npz':
        data = np.load(filepath, allow_pickle=True)
        if 'xyz' in data:
            xyz = data['xyz']
            properties = {}
            plate_ids = None

            for key in data.files:
                if key == 'xyz':
                    continue
                elif key == 'plate_ids':
                    plate_ids = data[key]
                elif key.startswith('prop_'):
                    # Properties saved with prefix
                    prop_name = key[5:]
                    properties[prop_name] = data[key]
                elif key not in ('metadata',):
                    # Assume it's a property
                    properties[key] = data[key]

            cloud = PointCloud(xyz=xyz, properties=properties)
            cloud.plate_ids = plate_ids
            return cloud
        else:
            raise ValueError("NPZ file must contain 'xyz' array")
    else:
        # .npy file - raw array
        data = np.load(filepath)
        xyz = data[:, list(xyz_columns)]

        properties = {}
        if property_columns:
            for name, col in property_columns.items():
                properties[name] = data[:, col]

        return PointCloud(xyz=xyz, properties=properties)


def load_points_latlon(
    filepath: Union[str, Path],
    latlon_columns: Tuple[int, int] = (0, 1),
    property_columns: Optional[Dict[str, int]] = None,
    delimiter: str = ',',
    skip_header: int = 0
) -> "PointCloud":
    """
    Load points from lat/lon text file (CSV, etc.).

    Parameters
    ----------
    filepath : str or Path
        Path to text file.
    latlon_columns : tuple, default=(0, 1)
        Column indices for lat, lon (in degrees).
    property_columns : dict, optional
        Mapping from property name to column index.
    delimiter : str, default=','
        Column delimiter.
    skip_header : int, default=0
        Number of header lines to skip.

    Returns
    -------
    PointCloud
        Loaded points.

    Examples
    --------
    >>> cloud = load_points_latlon(
    ...     'points.csv',
    ...     latlon_columns=(0, 1),
    ...     property_columns={'depth': 2}
    ... )
    """
    from .point_rotation import PointCloud

    data = np.loadtxt(
        filepath,
        delimiter=delimiter,
        skiprows=skip_header
    )

    latlon = data[:, list(latlon_columns)]

    properties = {}
    if property_columns:
        for name, col in property_columns.items():
            properties[name] = data[:, col]

    return PointCloud.from_latlon(latlon, properties)


def load_latlon_grid_hdf5(
    filepath: Union[str, Path],
    property_name: str
) -> "PointCloud":
    """
    Load a lat/lon gridded field from an HDF5 (or NetCDF4) file.

    The file must hold ``lon`` and ``lat`` datasets, plus the values under
    either ``property_name`` or ``z``. Both grid layouts are accepted: 1-D
    ``lon`` and ``lat`` are treated as axes and expanded onto a full mesh,
    while 2-D ones are taken as already paired with the values and only
    flattened.

    Longitudes above 180 are wrapped to the negative half, since the rest of
    gtrack works in -180 to 180 and many published grids are published in
    0 to 360.

    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5/NetCDF4 file.
    property_name : str
        Name to attach the values under, and the first dataset name looked
        for inside the file. ``z`` is accepted as a fallback, which is what
        GMT-derived grids tend to use.

    Returns
    -------
    PointCloud
        One point per grid node, carrying ``property_name``.

    Raises
    ------
    ImportError
        If h5py is not installed. It is an optional dependency; install with
        ``pip install gtrack[hdf5]``.
    KeyError
        If the file holds neither ``property_name`` nor ``z``.

    Examples
    --------
    >>> cloud = load_latlon_grid_hdf5('lithospheric_thickness.h5', 'thickness')
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised by absence
        raise ImportError(
            "load_latlon_grid_hdf5 requires h5py, which is an optional "
            "dependency of gtrack. Install it with 'pip install gtrack[hdf5]'."
        ) from exc

    from .point_rotation import PointCloud

    with h5py.File(filepath, "r") as f:
        lon = f["lon"][:]
        lat = f["lat"][:]
        if property_name in f:
            values = f[property_name][:]
        elif "z" in f:
            values = f["z"][:]
        else:
            raise KeyError(
                f"File must contain '{property_name}' or 'z'. "
                f"Available datasets: {list(f.keys())}"
            )

    lon = np.where(lon > 180, lon - 360, lon)

    if lon.ndim == 1 and lat.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        latlon = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    else:
        latlon = np.column_stack([lat.ravel(), lon.ravel()])
    values = values.ravel()

    cloud = PointCloud.from_latlon(latlon)
    cloud.add_property(property_name, values)
    return cloud


def save_points_numpy(
    cloud: "PointCloud",
    filepath: Union[str, Path],
    include_properties: bool = True
) -> None:
    """
    Save points to numpy format.

    Parameters
    ----------
    cloud : PointCloud
        Points to save.
    filepath : str or Path
        Output path (.npy or .npz).
    include_properties : bool, default=True
        If True, save properties (requires .npz format).

    Examples
    --------
    >>> save_points_numpy(cloud, 'output.npz')
    """
    filepath = Path(filepath)

    if include_properties and (cloud.properties or cloud.plate_ids is not None):
        # Use .npz for properties
        if filepath.suffix != '.npz':
            filepath = filepath.with_suffix('.npz')

        save_dict = {'xyz': cloud.xyz}

        if cloud.plate_ids is not None:
            save_dict['plate_ids'] = cloud.plate_ids

        for name, prop in cloud.properties.items():
            save_dict[f'prop_{name}'] = prop

        np.savez(filepath, **save_dict)
    else:
        # Simple .npy for xyz only
        np.save(filepath, cloud.xyz)


def save_points_latlon(
    cloud: "PointCloud",
    filepath: Union[str, Path],
    delimiter: str = ',',
    header: Optional[str] = None,
    include_properties: bool = True
) -> None:
    """
    Save points to lat/lon text file.

    Parameters
    ----------
    cloud : PointCloud
        Points to save.
    filepath : str or Path
        Output path.
    delimiter : str, default=','
        Column delimiter.
    header : str, optional
        Header line to write.
    include_properties : bool, default=True
        If True, include properties as additional columns.

    Examples
    --------
    >>> save_points_latlon(cloud, 'output.csv', header='lat,lon,depth')
    """
    latlon = cloud.latlon

    if include_properties and cloud.properties:
        # Stack lat, lon, and properties
        columns = [latlon]
        for name in sorted(cloud.properties.keys()):
            columns.append(cloud.properties[name].reshape(-1, 1))
        data = np.hstack(columns)
    else:
        data = latlon

    np.savetxt(
        filepath,
        data,
        delimiter=delimiter,
        header=header if header else '',
        comments=''
    )


def save_points_gpml(
    cloud: "PointCloud",
    filepath: Union[str, Path],
    feature_type: str = "UnclassifiedFeature",
    include_plate_ids: bool = True
) -> None:
    """
    Save points to GPML format (pygplates-compatible).

    Parameters
    ----------
    cloud : PointCloud
        Points to save.
    filepath : str or Path
        Output path (.gpml or .gpmlz for compressed).
    feature_type : str, default="UnclassifiedFeature"
        GPlates feature type for points.
    include_plate_ids : bool, default=True
        If True, include reconstruction plate IDs in features.

    Examples
    --------
    >>> save_points_gpml(cloud, 'continental_points.gpml')
    >>> save_points_gpml(cloud, 'compressed.gpmlz')  # Compressed
    """
    import pygplates
    from .geometry import XYZ2LatLon

    # Convert to lat/lon
    lats, lons = XYZ2LatLon(cloud.xyz)

    # Create features
    features = []
    for i in range(cloud.n_points):
        feature = pygplates.Feature()
        feature.set_geometry(pygplates.PointOnSphere(lats[i], lons[i]))

        if include_plate_ids and cloud.plate_ids is not None:
            feature.set_reconstruction_plate_id(int(cloud.plate_ids[i]))

        features.append(feature)

    # Save to file
    feature_collection = pygplates.FeatureCollection(features)
    feature_collection.write(str(filepath))


def load_points_gpml(
    filepath: Union[str, Path]
) -> "PointCloud":
    """
    Load points from GPML file.

    Parameters
    ----------
    filepath : str or Path
        Path to .gpml or .gpmlz file.

    Returns
    -------
    PointCloud
        Loaded points with plate_ids if available.

    Examples
    --------
    >>> cloud = load_points_gpml('points.gpml')
    """
    import pygplates
    from .point_rotation import PointCloud
    from .geometry import LatLon2XYZ

    feature_collection = pygplates.FeatureCollection(str(filepath))

    latlons = []
    plate_ids = []

    for feature in feature_collection:
        geometry = feature.get_geometry()
        if geometry is not None:
            if hasattr(geometry, 'to_lat_lon'):
                lat, lon = geometry.to_lat_lon()
                latlons.append([lat, lon])

                plate_id = feature.get_reconstruction_plate_id()
                plate_ids.append(plate_id if plate_id is not None else 0)

    if len(latlons) == 0:
        raise ValueError(f"No point features found in {filepath}")

    latlon = np.array(latlons)
    xyz = LatLon2XYZ(latlon)

    cloud = PointCloud(xyz=xyz)
    cloud.plate_ids = np.array(plate_ids, dtype=int)

    return cloud


class PointCloudCheckpoint:
    """
    Checkpoint manager for PointCloud state.

    Provides save/load functionality with metadata for checkpointing
    during long-running simulations.

    Examples
    --------
    >>> checkpoint = PointCloudCheckpoint()
    >>>
    >>> # Save with metadata
    >>> checkpoint.save(cloud, 'checkpoint_50Ma.npz', geological_age=50.0)
    >>>
    >>> # Load and get metadata
    >>> cloud, metadata = checkpoint.load('checkpoint_50Ma.npz')
    >>> print(metadata['geological_age'])  # 50.0
    """

    def save(
        self,
        cloud: "PointCloud",
        filepath: Union[str, Path],
        geological_age: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save checkpoint with metadata.

        Parameters
        ----------
        cloud : PointCloud
            Points to save.
        filepath : str or Path
            Output path (.npz format).
        geological_age : float, optional
            Current geological age for reference.
        metadata : dict, optional
            Additional metadata to save.

        Examples
        --------
        >>> checkpoint.save(
        ...     cloud, 'state.npz',
        ...     geological_age=50.0,
        ...     metadata={'simulation_step': 100}
        ... )
        """
        filepath = Path(filepath)
        if filepath.suffix != '.npz':
            filepath = filepath.with_suffix('.npz')

        meta = metadata.copy() if metadata else {}
        if geological_age is not None:
            meta['geological_age'] = geological_age

        save_dict = {
            'xyz': cloud.xyz,
            'metadata': np.array([meta], dtype=object),
        }

        if cloud.plate_ids is not None:
            save_dict['plate_ids'] = cloud.plate_ids

        for name, prop in cloud.properties.items():
            save_dict[f'prop_{name}'] = prop

        np.savez(filepath, **save_dict)

    def load(
        self,
        filepath: Union[str, Path]
    ) -> Tuple["PointCloud", Dict]:
        """
        Load checkpoint.

        Parameters
        ----------
        filepath : str or Path
            Path to checkpoint file.

        Returns
        -------
        cloud : PointCloud
            Loaded point cloud.
        metadata : dict
            Associated metadata.

        Examples
        --------
        >>> cloud, metadata = checkpoint.load('state.npz')
        >>> print(f"Loaded cloud at {metadata['geological_age']} Ma")
        """
        from .point_rotation import PointCloud

        filepath = Path(filepath)
        data = np.load(filepath, allow_pickle=True)

        xyz = data['xyz']

        # Extract metadata
        if 'metadata' in data:
            metadata = data['metadata'][0] if len(data['metadata']) > 0 else {}
        else:
            metadata = {}

        # Extract plate_ids
        plate_ids = data['plate_ids'] if 'plate_ids' in data else None

        # Extract properties
        properties = {}
        for key in data.files:
            if key.startswith('prop_'):
                prop_name = key[5:]  # Remove 'prop_' prefix
                properties[prop_name] = data[key]

        cloud = PointCloud(xyz=xyz, properties=properties, plate_ids=plate_ids)

        return cloud, metadata

    def list_checkpoints(
        self,
        directory: Union[str, Path],
        pattern: str = "*.npz"
    ) -> list:
        """
        List checkpoint files in a directory.

        Parameters
        ----------
        directory : str or Path
            Directory to search.
        pattern : str, default="*.npz"
            Glob pattern for checkpoint files.

        Returns
        -------
        list
            Sorted list of checkpoint file paths.
        """
        directory = Path(directory)
        checkpoints = sorted(directory.glob(pattern))
        return [str(cp) for cp in checkpoints]

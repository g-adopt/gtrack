# Getting Started

## Installation

Install gtrack using pip:

```bash
pip install gtrack
```

For development installation:

```bash
git clone https://github.com/g-adopt/gtrack.git
cd gtrack
pip install -e ".[dev]"
```

## Required Data Files

gtrack requires plate tectonic reconstruction data files:

- **Rotation files** (`.rot`): Define plate rotations through time
- **Topology files** (`.gpml`, `.gpmlz`): Define plate boundaries and topologies
- **Continental polygons** (optional): For filtering continental regions

These files are available from:

- [GPlates Portal](https://www.gplates.org/)
- [EarthByte Data Portal](https://www.earthbyte.org/category/resources/data-models/)

## Basic Usage

### Seafloor Age Tracking

The simplest way to compute seafloor ages:

```python
from gtrack import SeafloorAgeTracker, TracerConfig

# Configure parameters
config = TracerConfig(
    time_step=1.0,                    # Myr
    default_refinement_levels=5,      # ~10,000 mesh points
    ridge_sampling_degrees=0.5,       # ~50 km ridge tessellation
)

# Compute ages
cloud = SeafloorAgeTracker.compute_ages(
    target_age=100,
    starting_age=200,
    rotation_files=['rotations.rot'],
    topology_files=['topologies.gpmlz'],
    continental_polygons='continents.gpmlz',  # optional
    config=config,
)

# Access results
xyz = cloud.xyz                    # (N, 3) Cartesian coordinates
ages = cloud.get_property('age')   # (N,) material ages in Myr
lonlat = cloud.lonlat              # (N, 2) lon/lat in degrees
```

### Point Rotation

Rotate present-day data to past geological ages:

```python
from gtrack import PointCloud, PointRotator, PolygonFilter
import numpy as np

# Create point cloud from your data
latlon = np.array([[lat1, lon1], [lat2, lon2], ...])
cloud = PointCloud.from_latlon(latlon)
cloud.add_property('depth', depth_values)

# Filter to continental points only
polygon_filter = PolygonFilter(
    polygon_files='continental_polygons.gpmlz',
    rotation_files=['rotations.rot']
)
continental = polygon_filter.filter_inside(cloud, at_age=0.0)

# Assign plate IDs and rotate
rotator = PointRotator(
    rotation_files=['rotations.rot'],
    static_polygons='static_polygons.gpmlz'
)
continental = rotator.assign_plate_ids(continental, at_age=0.0)
rotated = rotator.rotate(continental, from_age=0.0, to_age=50.0)

# Use rotated coordinates
xyz = rotated.xyz
depths = rotated.get_property('depth')
```

## Logging

Control verbosity via environment variable:

```bash
export GTRACK_LOGLEVEL=INFO    # Progress messages
export GTRACK_LOGLEVEL=DEBUG   # Detailed debug output
export GTRACK_LOGLEVEL=WARNING # Quiet (default)
```

Or programmatically:

```python
from gtrack import enable_verbose, enable_debug

enable_verbose()  # Show progress messages
enable_debug()    # Show detailed debug output
```

## Next Steps

See the [API Reference](api.md) for detailed documentation of all classes and functions.

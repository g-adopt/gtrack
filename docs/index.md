# gtrack

**GPlates-based Tracking of Lithosphere and Kinematics**

A Python package for computing lithospheric structure through geological time using plate tectonic reconstructions. It is mainly written as a utility package for [gadopt](https://github.com/g-adopt/g-adopt) for assigning lithospheric thickness, but can be used independently.

## Overview

gtrack provides two main capabilities:

1. **Seafloor Age Tracking**: Compute oceanic lithosphere ages using Lagrangian particle tracking from mid-ocean ridges
2. **Point Rotation**: Rotate user-provided points (e.g., continental lithospheric structure) through geological time

Both return `PointCloud` objects with Cartesian XYZ coordinates, designed for integration with [gadopt](https://github.com/g-adopt/g-adopt).

## Based on GPlately

gtrack's seafloor age tracking algorithm is based on [GPlately](https://github.com/GPlates/gplately)'s `SeafloorGrid` implementation. The algorithm matches GPlately's approach for:

- Initial ocean mesh generation using icosahedral tessellation
- Mid-ocean ridge seed point generation using stage rotation poles
- Velocity-based collision detection at subduction zones
- Continental collision via polygon queries

## Performance

gtrack is written with performance in mind. All computationally intensive operations are delegated to **pygplates' C++ backend**:

- **Point reconstruction**: Uses `pygplates.TopologicalModel.reconstruct_geometry()` for efficient batch advection
- **Collision detection**: Uses pygplates' built-in `DefaultDeactivatePoints` with velocity/distance thresholds
- **Point-in-polygon queries**: Uses quad tree spatial indexing for efficient bulk queries
- **Batch point creation**: Uses `pygplates.MultiPointOnSphere` to minimize Python-C++ boundary crossings

## Installation

```bash
pip install gtrack
```

For development:

```bash
pip install -e .
```

## Quick Start

### Example 1: One-Shot Seafloor Age Computation

The simplest way to compute seafloor ages at a single geological age:

```python
from gtrack import SeafloorAgeTracker, TracerConfig

# Configure tracer parameters
config = TracerConfig(
    tracker_step_myr=1.0,
    tracker_point_count=10000,
    ridge_sampling_angle_deg=0.5,
)

# Compute seafloor ages at 100 Ma, starting from 200 Ma
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

### Example 2: Stepwise Seafloor Age Computation

For simulations that need ages at multiple intermediate geological ages:

```python
from gtrack import SeafloorAgeTracker, TracerConfig

config = TracerConfig(tracker_step_myr=1.0)

tracker = SeafloorAgeTracker(
    rotation_files=['rotations.rot'],
    topology_files=['topologies.gpmlz'],
    continental_polygons='continents.gpmlz',
    config=config,
)

# Initialize tracers at 200 Ma
tracker.initialize(starting_age=200)

# Step through time toward present
for target_age in range(195, -1, -5):
    cloud = tracker.step_to(target_age)

    # Access data at each timestep
    xyz = cloud.xyz
    ages = cloud.get_property('age')

    # Use with gadopt...

# Checkpointing for long runs
tracker.save_checkpoint('checkpoint.npz')
tracker.load_checkpoint('checkpoint.npz')
```

### Example 3: Rotating Continental Data Back in Time

Rotate present-day continental lithospheric structure to past geological ages:

```python
from gtrack import PointCloud, PointRotator, PolygonFilter
import numpy as np

# Create point cloud from your data (e.g., seismic tomography grid)
latlon = np.array([[lat1, lon1], [lat2, lon2], ...])
cloud = PointCloud.from_latlon(latlon)
cloud.add_property('lithospheric_depth', depth_values)

# Filter to keep only continental points
polygon_filter = PolygonFilter(
    polygon_files='continental_polygons.gpmlz',
    rotation_files=['rotations.rot']
)
continental = polygon_filter.filter_inside(cloud, at_age=0.0)

# Assign plate IDs and rotate to 50 Ma
rotator = PointRotator(
    rotation_files=['rotations.rot'],
    static_polygons='static_polygons.gpmlz'
)
continental = rotator.assign_plate_ids(continental, at_age=0.0)
rotated = rotator.rotate(continental, from_age=0.0, to_age=50.0)

# Use rotated coordinates with gadopt
xyz = rotated.xyz
depths = rotated.get_property('lithospheric_depth')
```

## Examples

See the `examples/` directory for complete working examples:

- `seafloor_age_simple.py` - One-shot seafloor age computation
- `seafloor_age_stepwise.py` - Stepwise computation with visualization
- `continental_rotation.py` - Rotating continental structure back in time

## Attribution

The seafloor age tracking algorithm is based on GPlately, but is mainly a pygplates wrapper.

> Müller, R. D., Cannon, J., Qin, X., Watson, R. J., Gurnis, M., Williams, S., Pfaffelmoser, T., Seton, M., Russell, S. H. J. ,Zahirovic S. (2018). [GPlates: Building a virtual Earth through deep time.](https://doi.org/10.1029/2018GC007584) Geochemistry, Geophysics, Geosystems, 19, 2243-2261.

> Mather, B. R., Müller, R. D., Zahirovic, S., Cannon, J., Chin, M., Ilano, L., Wright, N. M., Alfonso, C., Williams, S., Tetley, M., Merdith, A. (2023) [Deep time spatio-temporal data analysis using pyGPlates with PlateTectonicTools and GPlately.](https://doi.org/10.1002/gdj3.185) Geoscience Data Journal, 00, 1-8.

## License

MIT License

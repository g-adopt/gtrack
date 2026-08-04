# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

gtrack is a Python package for computing lithospheric structure through geological time using plate tectonic reconstructions. It uses **pygplates** as the underlying engine and must produce results **identical to GPlately's SeafloorGrid**.

Two main capabilities:
1. **Seafloor Age Tracking**: Compute oceanic lithosphere ages via Lagrangian particle tracking
2. **Point Rotation**: Rotate user-provided continental points through geological time

Both produce **PointCloud** objects with XYZ coordinates - directly compatible with gadopt's cKDTree interpolation.

## Build and Test Commands

```bash
# Install package (editable)
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Run all tests
pytest tests/ -v

# Run single test file
pytest tests/test_core.py -v

# Run single test
pytest tests/test_core.py::test_function_name -v

# Run tests with coverage
pytest tests/ -v --cov=gtrack --cov-report=term-missing
```

## Examples

Download example data first:
```bash
cd examples
make data          # Download plate model (Matthews et al.)
make osf-data      # Download lithospheric thickness maps from OSF
make litho-data    # Interpolate lithospheric data to sphere mesh
```

Run examples:
```bash
cd examples
python seafloor_age_simple.py    # One-shot computation
python seafloor_age_stepwise.py  # Stepwise evolution
python continental_rotation.py   # Point rotation workflow (requires litho-data)
```

Generate Jupyter notebooks from examples:
```bash
cd examples
make notebooks     # Converts .py to executed .ipynb in docs/demos/
```

## Architecture

```
gtrack/
├── __init__.py              # Public API exports
├── config.py                # TracerConfig dataclass
├── hpc_integration.py       # SeafloorAgeTracker - main seafloor age API
├── point_rotation.py        # PointCloud + PointRotator classes
├── polygon_filter.py        # PolygonFilter for continental filtering
├── boundaries.py            # Ridge/subduction boundary extraction + caching
├── mor_seeds.py             # MOR seed point generation (stage pole method)
├── initial_conditions.py    # Initial ocean point ages
├── mesh.py                  # Fibonacci sphere mesh generation
├── spatial.py               # Spatial queries and caching
├── geometry.py              # Coordinate transformations (XYZ ↔ LatLon)
├── io_formats.py            # Load/save utilities + checkpointing
└── logging.py               # Logging configuration
```

### Public API (from `gtrack/__init__.py`)

**Main classes:**
- `SeafloorAgeTracker` - Primary API for seafloor age computation
- `PointCloud` - Container for points with properties (xyz, lonlat, arbitrary scalar fields)
- `PointRotator` - Rotate points through geological time
- `PolygonFilter` - Filter points by polygon containment
- `TracerConfig` - Configuration dataclass for tracker parameters

**Utility functions:**
- `create_sphere_mesh_*` - Mesh generation (Fibonacci spiral)
- `generate_mor_seeds*` - Mid-ocean ridge seed generation
- `load_points_*` / `save_points_*` - I/O utilities

## Key Design Decisions

1. **Match GPlately output exactly**: Results must be identical to GPlately's SeafloorGrid. Reference implementation: `gplately/oceans.py`

2. **Use pygplates C++ backend**: Prefer `pygplates.TopologicalModel.reconstruct_geometry()` over Python loops for reconstruction.

3. **Cartesian XYZ internally**: Matches gadopt's coordinate system. Only convert to lat/lon at pygplates interface boundaries.

4. **No parallelization in gtrack**: Parallelization is handled by gadopt. gtrack must be serial-safe.

5. **Stage pole rotation for ridges**: Points spread symmetrically using stage pole rotation, NOT geometric perpendicular to ridge.

6. **Batched plate operations**: Group points by plate ID and apply single rotation per plate (10-50x faster than per-point rotation).

## Critical Terminology

| Term | Definition |
|------|------------|
| `geological_age` / `age` | Time before present in Ma. 0 = present |
| `from_age` / `to_age` | Source and target geological ages for rotation |
| `starting_age` | Initial geological age for tracker initialization |
| `target_age` | Geological age to evolve to |

**Direction**: SeafloorAgeTracker only evolves forward in time (decreasing geological age toward 0 = present).

## Usage Patterns

### 1. One-Shot Computation (simplest)

```python
from gtrack import SeafloorAgeTracker, TracerConfig

cloud = SeafloorAgeTracker.compute_ages(
    target_age=0,           # Present day
    starting_age=200,       # Start simulation at 200 Ma
    rotation_files=rotation_files,
    topology_files=topology_files,
    continental_polygons=continental_polygons,
    config=TracerConfig(time_step=1.0, default_refinement_levels=5),
)

xyz = cloud.xyz                    # (N, 3) Cartesian coordinates
lonlat = cloud.lonlat              # (N, 2) [lon, lat] in degrees
tracer_ages = cloud.get_property('age')  # Material ages in Myr
```

### 2. Stepwise Evolution (for intermediate states)

```python
from gtrack import SeafloorAgeTracker, TracerConfig

tracker = SeafloorAgeTracker(
    rotation_files=rotation_files,
    topology_files=topology_files,
    continental_polygons=continental_polygons,
    config=TracerConfig(time_step=1.0),
)

tracker.initialize(starting_age=300)

for target_age in [295, 280, 100, 0]:
    cloud = tracker.step_to(target_age)
    # Process cloud at each intermediate age...
```

### 3. Point Rotation (continental points)

```python
from gtrack import PointCloud, PointRotator, PolygonFilter

cloud = PointCloud.from_latlon(latlon_array)
cloud.add_property('lithospheric_depth', depths)

polygon_filter = PolygonFilter(polygon_files=continental_polygons, rotation_files=rotation_files)
continental_cloud = polygon_filter.filter_inside(cloud, at_age=0.0)

# topology_files is REQUIRED (motion is topological); static_polygons is optional
# and used only for the label-only assign_plate_ids(source="static") path.
rotator = PointRotator(
    rotation_files=rotation_files,
    topology_files=topology_files,
    static_polygons=static_polygons,
)
# assign_plate_ids is optional now (labelling only) — rotate needs no plate_ids.
rotated = rotator.rotate(continental_cloud, from_age=0.0, to_age=100.0)
```

## Logging

Control verbosity via environment variable or programmatically:

```bash
export GTRACK_LOGLEVEL=INFO    # Progress messages
export GTRACK_LOGLEVEL=DEBUG   # Detailed debug output
```

```python
from gtrack import enable_verbose, enable_debug
enable_verbose()  # Show progress messages
```

## Common Gotchas

1. **Stage pole vs geometric perpendicular**: GPlately uses stage pole rotation to spread points, NOT geometric perpendicular to ridge.

2. **Implicit plate ID assignment**: Points inherit plate IDs from the resolved topology polygon they fall within, not explicit assignment at creation.

3. **Velocity-based collision**: Collision detection uses velocity difference between plates, not just proximity to boundaries.

4. **C++ backend required**: The pygplates TopologicalModel is essential for matching GPlately results.

5. **Time direction**: `starting_age` > `target_age` always. Evolution goes forward in time (toward present = 0). (Applies to `SeafloorAgeTracker`; `PointRotator.rotate` works both directions.)

6. **PointRotator is topological, not rigid**: `PointRotator.rotate` advects points with `TopologicalModel.reconstruct_geometry` (deforming networks + rigid plates), exactly like the ocean tracker. `topology_files` is **required**. Motion does NOT depend on `plate_ids`, and with the default `deactivate_points=None` no point is ever dropped (`n_out == n_in`, properties/plate_ids preserved in input order). The old rigid per-plate-id engine — which dropped out-of-circuit points and mis-rotated continental points on plate-id mismatches — has been removed.

7. **plate_ids are labels, not motion input**: `assign_plate_ids` is optional and defaults to `source="topology"` (consistent with how points move). `source="static"` is retained for back-compat. Reconstructed continental/craton positions differ from the old rigid results **by design** (median ~4°, up to ~30° over 50 Myr on Zahirovic/Merdith data); validate against the topological result, not the old output.

8. **Continental polygon reconstruction is still rigid** (`boundaries.py`): kept deliberately to stay identical to GPlately's SeafloorGrid. Out-of-circuit continent polygons are identity-reconstructed at ages before their plate appears — a documented Phase 4 limitation, pinned by `tests/test_boundaries_reconstruction.py`. Craton/crust containment runs at `at_age=0` so it is unaffected.

9. **Indicator sources need a FRESH background, not a back-rotated grid**: use `build_indicator_source` (`gtrack/sources.py`) to build a "seeds + zero background" field. It rotates only the in-region seeds topologically and lays down a fresh uniform background grid *at the target age*. Do NOT back-rotate a fixed present-day grid: it deforms with age (gaps grow at ridges/trenches), leaving the downstream nearest-neighbour interpolation (large distance threshold) with no nearby background point. The fresh background is a correctness requirement, not a rigid-engine workaround. No `plate_ids`, reattachment, or sliver inheritance are needed.

10. **Age spans arrive carrying float round-off**: callers that convert a non-dimensional model time to Ma hand `PointRotator.rotate` a nominal zero span as ~1e-14 Myr, not 0.0. `reconstruct_geometry` rejects a span of 1e-9 Myr or less as degenerate, so `_advect_topological` guards with `ZERO_SPAN_TOLERANCE_MYR` (1e-6 Myr) rather than an exact `== 0`. The threshold has to clear pygplates' own with margin: a guard of exactly 1e-9 leaves the boundary case falling through and still raising.

## Documentation Website

The documentation is built with MkDocs and deployed to https://gtrack.gadopt.org

```bash
# Serve docs locally
pip install -e ".[docs]"
mkdocs serve

# Build docs
mkdocs build
```

GitHub Actions workflow builds and deploys on push to main.

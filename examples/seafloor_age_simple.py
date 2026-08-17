# Simple One-Shot Seafloor Age Computation
# =========================================
#
# This example demonstrates the simplest way to compute seafloor
# ages using gtrack. The `compute_ages()` class method provides a
# functional interface that handles all setup internally.
#
# Use this approach when you need ages at a single target geological
# age and don't need to access intermediate states.

# +
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from gtrack import SeafloorAgeTracker, TracerConfig
# -

# ## Data File Paths
#
# Run `make data` in the examples directory to download and extract the data.

# +
data_dir = Path("./Matthews_et_al_410_0")

rotation_files = [
    data_dir / "Global_EB_250-0Ma_GK07_Matthews++.rot",
    data_dir / "Global_EB_410-250Ma_GK07_Matthews++.rot"
]

topology_files = [
    data_dir / "Mesozoic-Cenozoic_plate_boundaries_Matthews++.gpml",
    data_dir / "Paleozoic_plate_boundaries_Matthews++.gpml",
    data_dir / "TopologyBuildingBlocks_Matthews++.gpml",
]

# Continental polygons are optional - if not provided, tracers are not
# removed when entering continental regions
continental_polygons = data_dir / \
    "ContPolys/PresentDay_ContinentalPolygons_Matthews++.shp"
# -

# ## Configuration
#
# TracerConfig controls all tracer parameters. The new API uses
# pygplates' C++ backend for efficient collision detection and
# point advection, matching GPlately's approach.

# +
config = TracerConfig(
    # Time stepping
    tracker_step_myr=1.0,

    # Mesh initialization - number of points on Fibonacci sphere
    # 10000 points gives ~115 km spacing, 40000 gives ~57 km spacing
    tracker_point_count=10000,

    # Initial age calculation
    initial_spreading_rate_mm_per_yr=75.0,

    # MOR seed generation
    ridge_sampling_angle_deg=0.5,
    ridge_offset_angle_deg=0.01,

    # Collision detection (C++ backend - GPlately compatible)
    collision_velocity_difference_km_per_myr=7.0,
    collision_distance_rate_km_per_myr=10.0,
)
# -

# ## Compute Ages in One Call
#
# The `compute_ages()` class method creates a tracker, initialises
# tracers at the starting age, and evolves them to the target age - all
# in a single call. This is the simplest way to get seafloor ages.
#
# Here we compute present-day (0 Ma) seafloor ages, starting the
# simulation at 200 Ma.

# +
cloud = SeafloorAgeTracker.compute_ages(
    target_age=0,
    starting_age=200,
    rotation_files=rotation_files,
    topology_files=topology_files,
    continental_polygons=continental_polygons,
    config=config,
)
# -
# ## Access Results
#
# The result is a PointCloud with xyz coordinates and an 'age'
# property containing the material age of each tracer.

# +
# Get coordinates and ages
# (N, 2) array with [lon, lat] in degrees
xyz = cloud.xyz
lonlat = cloud.lonlat  # (N, 2) array with [lon, lat] in degrees
tracer_ages = cloud.get_property('age')   # (N,) material ages in Myr
# -

# ## Plotting the Results
#
# Plot the tracers as a scatter plot on a Mollweide projection,
# coloured by their material age.

# +
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.Mollweide())

scatter = ax.scatter(
    lonlat[:, 0], lonlat[:, 1],
    c=tracer_ages,
    s=1,
    cmap='viridis_r',
    vmin=0, vmax=200,
    transform=ccrs.PlateCarree()
)

ax.coastlines(resolution='110m', linewidth=0.5)
ax.set_global()

cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',
                    pad=0.05, aspect=40, shrink=0.8)
cbar.set_label('Seafloor Age (Myr)')

ax.set_title('Present-Day Seafloor Ages (computed from 200 Ma)')
plt.show()

# Print summary statistics
print(f"Number of tracers: {len(tracer_ages)}")
print(f"Age range: {tracer_ages.min():.1f} - {tracer_ages.max():.1f} Myr")
print(f"Mean age: {tracer_ages.mean():.1f} Myr")
# -

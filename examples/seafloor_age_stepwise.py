# Stepwise Seafloor Age Computation with gtrack
# ================================================
#
# This example demonstrates how to use gtrack's SeafloorAgeTracker
# for computing seafloor ages using Lagrangian particle tracking.
# The stepwise interface is ideal for simulations that need to
# access ages at multiple intermediate geological ages, let's say
# for a mantle convection simulation that is using sequential data-assimilation
# approach.
#
# In this example we will:
# 1. Initialise tracers at a starting age (300 Ma).
# 2. Step through geological time toward present, visualising at selected ages.
# 3. Demonstrate checkpointing for saving and restoring state.
#
# For this example, you need GPlates data files. Adjust the paths
# below to match your system.

# +
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pygplates
from shapely.geometry.polygon import LinearRing

from gtrack import SeafloorAgeTracker, TracerConfig
# -

# ## Data File Paths
#
# gtrack requires three types of GPlates data files:
# - Rotation files (.rot): Define how plates move through time
# - Topology files (.gpmlz): Define plate boundaries
# - Continental polygons (.gpmlz): Define continental regions
#
# These files are typically distributed with GPlates or with publications done
# using GPlates. For this example we use the plate model of Matthews et al. that can be downloaded
# from https://data.gadopt.org/demos/Matthews_et_al_410_0.tar.gz.
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

# Coastlines for plotting reconstructed continental outlines
coastlines_file = data_dir / "Coastlines/Coastlines_low_res_Matthews++.gpml"

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
# -

# ## Configuration
#
# TracerConfig controls the simulation parameters. The parameters are grouped
# into several categories:
#
# **Time stepping**: The `tracker_step_myr` controls the tracker step.
# advected forward in time. Smaller values give more accurate trajectories
# but increase computation time.
#
# **Mesh initialisation**: At the starting age, tracers are placed on a
# Fibonacci sphere mesh that covers the globe uniformly. The `tracker_point_count`
# controls the resolution - more points give higher resolution but require
# more memory and computation.
#
# **Initial age calculation**: For the initial ocean tracers (those not
# born at ridges), ages are estimated based on distance to the nearest
# ridge divided by half the spreading rate. The `initial_spreading_rate_mm_per_yr`
# sets this assumed spreading rate.
#
# **MOR seed generation**: New tracers are born at mid-ocean ridges (MORs)
# at each time step. The `ridge_sampling_angle_deg` controls how densely
# the ridge is sampled, and `ridge_offset_angle_deg` offsets new tracers
# slightly from the ridge axis (to avoid numerical issues at the boundary).
#
# **Collision detection**: Tracers are removed when they collide with
# subduction zones or continental margins. The `collision_velocity_difference_km_per_myr`
# detects sudden velocity changes (indicating plate boundary crossing),
# and `collision_distance_rate_km_per_myr` sets how close a tracer must be to a
# boundary to be considered for removal.

# +
config = TracerConfig(
    # Time stepping
    tracker_step_myr=1.0,

    # Mesh initialisation - number of points on Fibonacci sphere
    # 10000 points gives ~115 km spacing, 40000 gives ~57 km spacing
    tracker_point_count=10000,

    # Initial age calculation
    initial_spreading_rate_mm_per_yr=75.0,

    # MOR seed generation
    ridge_sampling_angle_deg=0.5,
    ridge_offset_angle_deg=0.01,

    # Collision detection thresholds
    collision_velocity_difference_km_per_myr=7.0,
    collision_distance_rate_km_per_myr=10.0,
)
# -

# ## Initialise the Tracker
#
# The SeafloorAgeTracker is designed for **stepwise** simulations where you
# need tracer positions at multiple intermediate geological ages. It maintains
# the tracer state internally, allowing you to repeatedly call `step_to()` to
# advance tracers forward in time (i.e., decreasing geological age toward 0 Ma).
#
# The tracker is initialised with the GPlates data files: rotation files define
# plate motions through time, topology files define plate boundaries, and
# continental polygons (optional) define regions where oceanic tracers are removed.
#
# After creating the tracker, we call `initialize()` with a `starting_age`.
# This creates an initial set of tracers on a Fibonacci sphere mesh covering the
# ocean basins. Each tracer's initial age is estimated from its distance to
# the nearest ridge, divided by half the `initial_spreading_rate_mm_per_yr`
# from the configuration. This provides a reasonable first guess for the age
# structure of oceanic lithosphere at the starting time.

# +
tracker = SeafloorAgeTracker(
    rotation_files=rotation_files,
    topology_files=topology_files,
    continental_polygons=continental_polygons,
    config=config,
)

# Initialise tracers at the starting geological age
starting_age = 300  # Ma
n_tracers = tracker.initialize(starting_age=starting_age)
print(f"Initialised {n_tracers} tracers at {starting_age} Ma")
# -

# ## Stepwise Evolution
#
# The `step_to()` method evolves tracers to a target geological age. It handles
# all the intermediate steps internally (using `tracker_step_myr`) and
# returns a PointCloud with the tracer positions and ages at the target time.
#
# We'll step through selected ages to demonstrate the evolution:
# - 295 Ma (5 Myr after start)
# - 280 Ma (20 Myr after start)
# - 100 Ma (200 Myr after start)
# - 0 Ma (present day)

# +
# Load pygplates rotation model and coastlines for plotting reconstructed
# continental outlines at past geological ages.
rotation_model = pygplates.RotationModel([str(f) for f in rotation_files])
coastlines = pygplates.FeatureCollection(str(coastlines_file))
# -

# +
# We define a helper function to plot the tracer field at each age.
# This keeps the code clean and avoids repetition as we step through
# multiple geological ages.


def plot_tracers(cloud, geological_age, rotation_model, coastlines):
    """Plot tracer positions coloured by material age with reconstructed coastlines."""
    lonlat = cloud.lonlat
    ages = cloud.get_property('age')

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Mollweide())

    scatter = ax.scatter(
        lonlat[:, 0], lonlat[:, 1],
        c=ages,
        s=1,
        cmap='viridis_r',
        vmin=0, vmax=200,
        transform=ccrs.PlateCarree()
    )

    # Reconstruct coastlines to the target geological age
    reconstructed_coastlines = []
    pygplates.reconstruct(coastlines, rotation_model,
                          reconstructed_coastlines, float(geological_age))

    # Handle dateline wrapping and convert to shapely geometries
    date_line_wrapper = pygplates.DateLineWrapper(0.0)
    coastline_geometries = []
    for polygon in reconstructed_coastlines:
        wrapped_polygons = date_line_wrapper.wrap(
            polygon.get_reconstructed_geometry())
        for poly in wrapped_polygons:
            coastline_geometries.append(
                LinearRing([(p.get_longitude(), p.get_latitude())
                            for p in poly.get_exterior_points()])
            )

    # Plot reconstructed coastlines
    ax.add_geometries(coastline_geometries, ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black',
                      linewidth=0.5, alpha=0.8)

    ax.set_global()

    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',
                        pad=0.05, aspect=40, shrink=0.8)
    cbar.set_label('Seafloor Age (Myr)')

    ax.set_title(f'Seafloor Tracers at {geological_age} Ma')
    plt.show()

    # Print statistics
    stats_dict = {
        'count': len(ages),
        'min_age': ages.min(),
        'max_age': ages.max(),
        'mean_age': ages.mean()
    }
    print(f"  Tracers: {stats_dict['count']}")
    print(
        f"  Age range: {stats_dict['min_age']:.1f} - {stats_dict['max_age']:.1f} Myr")
    print(f"  Mean age: {stats_dict['mean_age']:.1f} Myr")
# -

# ### After 5 Myr (295 Ma)


# +
cloud = tracker.step_to(295)
plot_tracers(cloud, 295, rotation_model, coastlines)
# -

# ### After 20 Myr (280 Ma)

# +
cloud = tracker.step_to(280)
plot_tracers(cloud, 280, rotation_model, coastlines)
# -

# ### At 100 Ma (200 Myr of evolution)

# +
cloud = tracker.step_to(100)
plot_tracers(cloud, 100, rotation_model, coastlines)
# -

# ### Present Day (0 Ma)

# +
cloud = tracker.step_to(0)
plot_tracers(cloud, 0, rotation_model, coastlines)
# -

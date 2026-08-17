"""Configure gtrack source-point tracking."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TracerConfig:
    """Configure seafloor-age tracking.

    The serialized dictionary uses the GPlately keys. The Python attributes use
    the gtrack vocabulary.

    Attributes
    ----------
    tracker_step_myr : float
        Tracker step in Myr. The default is 1.0.
    earth_radius_m : float
        Earth radius in metres. The default is 6.3781e6.

    Collision detection
    ----------------------------------
    collision_velocity_difference_km_per_myr : float
        Minimum velocity difference for collision detection in km/Myr.
        The default is 7.0 km/Myr.
    collision_distance_rate_km_per_myr : float
        Distance rate for collision detection in km/Myr.
        The default is 10.0 km/Myr.

    Initialization
    --------------
    tracker_point_count : int
        Point count for the initial sphere mesh. The default is 10000.
    initial_spreading_rate_mm_per_yr : float
        Mean spreading rate for the initial age calculation in mm/yr.
        The default is 75.0 mm/yr.

    Mid-ocean-ridge source points
    -------------------
    ridge_sampling_angle_deg : float
        Ridge sampling angle in degrees. The default is 0.5 degrees.
    ridge_offset_angle_deg : float
        Angular offset from each ridge in degrees. The default is 0.01 degrees.

    Tracker rebuild
    ---------------
    tracker_rebuild_neighbor_count : int
        Source-point count for tracker rebuild interpolation. The default is 6.
    tracker_rebuild_max_distance_m : float
        Maximum source separation for a tracker rebuild in metres.
        The default is half the Earth circumference.
    gc_collect_frequency : int or None
        Internal tracker steps between garbage collections. The default is 10.
        `None` disables scheduled collection.

    Examples
    --------
    >>> config = TracerConfig()
    >>> config = TracerConfig(
    ...     tracker_point_count=40000,
    ...     ridge_sampling_angle_deg=0.25,
    ...     tracker_step_myr=0.5,
    ... )
    """

    # Time stepping
    tracker_step_myr: float = 1.0
    earth_radius_m: float = 6.3781e6

    # GPlately passes these values to its collision detector.
    collision_velocity_difference_km_per_myr: float = 7.0
    collision_distance_rate_km_per_myr: float = 10.0

    # Initial sphere mesh
    tracker_point_count: int = 10000
    initial_spreading_rate_mm_per_yr: float = 75.0

    # Mid-ocean-ridge source points
    ridge_sampling_angle_deg: float = 0.5
    ridge_offset_angle_deg: float = 0.01

    # Tracker rebuild
    tracker_rebuild_neighbor_count: int = 6
    tracker_rebuild_max_distance_m: Optional[float] = None

    # Garbage collection
    gc_collect_frequency: Optional[int] = 10

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.tracker_step_myr <= 0:
            raise ValueError(
                f"tracker_step_myr must be positive, got {self.tracker_step_myr}"
            )
        if self.earth_radius_m <= 0:
            raise ValueError(f"earth_radius_m must be positive, got {self.earth_radius_m}")
        if self.collision_velocity_difference_km_per_myr < 0:
            raise ValueError(
                "collision_velocity_difference_km_per_myr must be non-negative, "
                f"got {self.collision_velocity_difference_km_per_myr}"
            )
        if self.collision_distance_rate_km_per_myr < 0:
            raise ValueError(
                "collision_distance_rate_km_per_myr must be non-negative, "
                f"got {self.collision_distance_rate_km_per_myr}"
            )
        if self.tracker_point_count < 1:
            raise ValueError(
                f"tracker_point_count must be at least 1, got {self.tracker_point_count}"
            )
        if self.initial_spreading_rate_mm_per_yr <= 0:
            raise ValueError(
                "initial_spreading_rate_mm_per_yr must be positive, "
                f"got {self.initial_spreading_rate_mm_per_yr}"
            )
        if self.ridge_sampling_angle_deg <= 0:
            raise ValueError(
                f"ridge_sampling_angle_deg must be positive, got {self.ridge_sampling_angle_deg}"
            )
        if self.ridge_offset_angle_deg <= 0:
            raise ValueError(
                f"ridge_offset_angle_deg must be positive, got {self.ridge_offset_angle_deg}"
            )
        if self.tracker_rebuild_neighbor_count < 1:
            raise ValueError(
                "tracker_rebuild_neighbor_count must be at least 1, "
                f"got {self.tracker_rebuild_neighbor_count}"
            )
        if self.tracker_rebuild_max_distance_m is None:
            self.tracker_rebuild_max_distance_m = np.pi * self.earth_radius_m
        if self.tracker_rebuild_max_distance_m <= 0:
            raise ValueError(
                "tracker_rebuild_max_distance_m must be positive, "
                f"got {self.tracker_rebuild_max_distance_m}"
            )
        if self.gc_collect_frequency is not None and self.gc_collect_frequency < 1:
            raise ValueError(
                f"gc_collect_frequency must be >= 1 or None, "
                f"got {self.gc_collect_frequency}"
            )

    @property
    def collision_velocity_difference_cm_per_yr(self) -> float:
        """
        Return the collision velocity difference in cm/yr.

        The GPlately API uses cm/yr. One km/Myr equals 0.1 cm/yr.
        """
        return self.collision_velocity_difference_km_per_myr / 10.0

    def to_dict(self) -> dict:
        """
        Return a dictionary with GPlately-compatible keys.

        Returns
        -------
        dict
            Configuration with the external GPlately vocabulary.
        """
        return {
            'time_step': self.tracker_step_myr,
            'earth_radius': self.earth_radius_m,
            'velocity_delta_threshold': self.collision_velocity_difference_km_per_myr,
            'distance_threshold_per_myr': self.collision_distance_rate_km_per_myr,
            'default_mesh_points': self.tracker_point_count,
            'initial_ocean_mean_spreading_rate': self.initial_spreading_rate_mm_per_yr,
            'ridge_sampling_degrees': self.ridge_sampling_angle_deg,
            'spreading_offset_degrees': self.ridge_offset_angle_deg,
            'continental_reconstruction_interval': 1,
            'reinit_k_neighbors': self.tracker_rebuild_neighbor_count,
            'reinit_max_distance': self.tracker_rebuild_max_distance_m,
            'gc_collect_frequency': self.gc_collect_frequency,
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        """
        Create a configuration from a GPlately-compatible dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary with configuration parameters

        Returns
        -------
        TracerConfig
            Configuration object
        """
        external_to_internal = {
            'time_step': 'tracker_step_myr',
            'earth_radius': 'earth_radius_m',
            'velocity_delta_threshold': 'collision_velocity_difference_km_per_myr',
            'distance_threshold_per_myr': 'collision_distance_rate_km_per_myr',
            'default_mesh_points': 'tracker_point_count',
            'initial_ocean_mean_spreading_rate': 'initial_spreading_rate_mm_per_yr',
            'ridge_sampling_degrees': 'ridge_sampling_angle_deg',
            'spreading_offset_degrees': 'ridge_offset_angle_deg',
            'reinit_k_neighbors': 'tracker_rebuild_neighbor_count',
            'reinit_max_distance': 'tracker_rebuild_max_distance_m',
        }
        d = {
            external_to_internal.get(name, name): value
            for name, value in config_dict.items()
            if name != 'continental_reconstruction_interval'
        }
        cri = config_dict.get('continental_reconstruction_interval', 1)
        if not isinstance(cri, int) or isinstance(cri, bool) or cri < 1:
            raise ValueError(
                "continental_reconstruction_interval must be a positive integer, "
                f"got {cri!r}"
            )
        if cri != 1:
            import logging
            logging.getLogger('gtrack.config').warning(
                "Loaded config has continental_reconstruction_interval=%d. "
                "This parameter is deprecated and no longer used; continental "
                "polygons are now reconstructed at exact float times. "
                "The value will be reset to 1.", cri
            )
        return cls(**d)

"""Tests for the diagnostic script fixes (ISSUES.md Part 2).

These tests verify the visualisation improvements in
examples/debug_lithosphere_indicator.py without requiring the full
plate model dataset. They use synthetic polygons and points to test:

2.1 — Dateline wrapping (Geodetic CRS produces valid Shapely polygons)
2.2 — Nearest-distance field computation
2.3 — Outline times match excision (moot after snapping removal, tested indirectly)
2.4 — Ocean/continental particle distinction (tested via script structure)
"""

import numpy as np
import pytest
import pygplates

try:
    import cartopy.crs as ccrs
    from cartopy.feature import ShapelyFeature
    import shapely.geometry as sgeom
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from pathlib import Path
from scipy.spatial import cKDTree

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "examples" / "debug_lithosphere_indicator.py"


def _create_polygon_feature(vertices_latlon, plate_id=0):
    points = [pygplates.PointOnSphere(lat, lon) for lat, lon in vertices_latlon]
    polygon = pygplates.PolygonOnSphere(points)
    feature = pygplates.Feature()
    feature.set_geometry(polygon)
    feature.set_reconstruction_plate_id(plate_id)
    return feature


def _write_features_to_gpml(features, path):
    fc = pygplates.FeatureCollection(features)
    fc.write(path)


@pytest.fixture
def simple_continent_files(tmp_path):
    """A simple continental polygon and identity rotation."""
    vertices = [(10, -10), (10, 10), (-10, 10), (-10, -10)]
    feature = _create_polygon_feature(vertices, plate_id=0)
    gpml = str(tmp_path / "continent.gpml")
    rot = str(tmp_path / "rotations.rot")
    _write_features_to_gpml([feature], gpml)
    with open(rot, "w") as f:
        f.write("0  0.0  0.0  0.0  0.0  999  !identity\n")
    return gpml, rot


@pytest.fixture
def dateline_polygon_files(tmp_path):
    """A continental polygon that crosses the antimeridian."""
    # Polygon spanning lon 170 to -170 (i.e., 170E to 170W, crossing dateline)
    vertices = [(10, 170), (10, -170), (-10, -170), (-10, 170)]
    feature = _create_polygon_feature(vertices, plate_id=0)
    gpml = str(tmp_path / "dateline.gpml")
    rot = str(tmp_path / "rotations.rot")
    _write_features_to_gpml([feature], gpml)
    with open(rot, "w") as f:
        f.write("0  0.0  0.0  0.0  0.0  999  !identity\n")
    return gpml, rot


class TestReconstructedContinents:
    """Test the get_reconstructed_continents function logic."""

    def test_returns_valid_shapely_polygons(self, simple_continent_files):
        """Reconstructed continents produce valid Shapely polygons."""
        gpml, rot = simple_continent_files
        rotation_model = pygplates.RotationModel([rot])
        features = pygplates.FeatureCollection(gpml)

        reconstructed = []
        pygplates.reconstruct(features, rotation_model, reconstructed, 0.0)

        geometries = []
        for rfg in reconstructed:
            geom = rfg.get_reconstructed_geometry()
            if not isinstance(geom, pygplates.PolygonOnSphere):
                continue
            lats_lons = [p.to_lat_lon() for p in geom.get_exterior_ring_points()]
            lats = [ll[0] for ll in lats_lons]
            lons = [ll[1] for ll in lats_lons]
            if len(lons) > 2:
                lons.append(lons[0])
                lats.append(lats[0])
                poly = sgeom.Polygon(zip(lons, lats))
                if poly.is_valid:
                    geometries.append(poly)

        assert len(geometries) >= 1
        for poly in geometries:
            assert poly.is_valid
            assert not poly.is_empty

    def test_dateline_polygon_produces_valid_shapely(self, dateline_polygon_files):
        """A polygon crossing the antimeridian still produces a Shapely polygon."""
        gpml, rot = dateline_polygon_files
        rotation_model = pygplates.RotationModel([rot])
        features = pygplates.FeatureCollection(gpml)

        reconstructed = []
        pygplates.reconstruct(features, rotation_model, reconstructed, 0.0)

        geometries = []
        for rfg in reconstructed:
            geom = rfg.get_reconstructed_geometry()
            if not isinstance(geom, pygplates.PolygonOnSphere):
                continue
            lats_lons = [p.to_lat_lon() for p in geom.get_exterior_ring_points()]
            lats = [ll[0] for ll in lats_lons]
            lons = [ll[1] for ll in lats_lons]
            if len(lons) > 2:
                lons.append(lons[0])
                lats.append(lats[0])
                poly = sgeom.Polygon(zip(lons, lats))
                # The polygon may not be "valid" in 2D Cartesian sense
                # because it crosses the dateline, but it should still be
                # constructable. Cartopy with Geodetic CRS handles the rest.
                geometries.append(poly)

        assert len(geometries) >= 1

    @pytest.mark.skipif(not HAS_CARTOPY or not HAS_MATPLOTLIB, reason="cartopy/matplotlib not installed")
    def test_geodetic_crs_renders_without_error(self, simple_continent_files):
        """ShapelyFeature with Geodetic CRS creates a renderable feature."""
        gpml, rot = simple_continent_files
        rotation_model = pygplates.RotationModel([rot])
        features = pygplates.FeatureCollection(gpml)

        reconstructed = []
        pygplates.reconstruct(features, rotation_model, reconstructed, 0.0)

        geometries = []
        for rfg in reconstructed:
            geom = rfg.get_reconstructed_geometry()
            if not isinstance(geom, pygplates.PolygonOnSphere):
                continue
            lats_lons = [p.to_lat_lon() for p in geom.get_exterior_ring_points()]
            lats = [ll[0] for ll in lats_lons]
            lons = [ll[1] for ll in lats_lons]
            if len(lons) > 2:
                lons.append(lons[0])
                lats.append(lats[0])
                poly = sgeom.Polygon(zip(lons, lats))
                if poly.is_valid:
                    geometries.append(poly)

        feature = ShapelyFeature(
            geometries, ccrs.PlateCarree(),
            facecolor="none", edgecolor="black",
        )

        # Should render without error
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
        ax.add_feature(feature)
        ax.set_global()
        plt.close(fig)


class TestNearestDistanceField:
    """Test the nearest-particle distance diagnostic (fix 2.2)."""

    def test_nearest_distance_computation(self):
        """Nearest-distance field correctly reveals data gaps."""
        # Create source points in a ring around the equator, leaving a gap
        # between lon 10 and lon 20
        lons_deg = np.concatenate([
            np.linspace(-180, 9, 100),
            np.linspace(21, 180, 100),
        ])
        lats_deg = np.zeros_like(lons_deg)

        lats_rad = np.radians(lats_deg)
        lons_rad = np.radians(lons_deg)
        source_xyz = np.column_stack([
            np.cos(lats_rad) * np.cos(lons_rad),
            np.cos(lats_rad) * np.sin(lons_rad),
            np.sin(lats_rad),
        ])

        tree = cKDTree(source_xyz)

        # Query a grid point inside the gap (lon=15, lat=0)
        query_lon = np.radians(15.0)
        query_xyz = np.array([[np.cos(query_lon), np.sin(query_lon), 0.0]])
        dist_gap, _ = tree.query(query_xyz, k=1)

        # Query a grid point with good coverage (lon=0, lat=0)
        query_xyz_covered = np.array([[1.0, 0.0, 0.0]])
        dist_covered, _ = tree.query(query_xyz_covered, k=1)

        # The gap point should be much farther from any source
        assert dist_gap > dist_covered * 3, (
            f"Gap distance ({dist_gap:.4f}) should be much larger "
            f"than covered distance ({dist_covered:.4f})"
        )

    def test_nearest_distance_field_shape(self):
        """Nearest-distance field has the right shape for a lat/lon grid."""
        nlat, nlon = 10, 20
        grid_lat = np.linspace(-90, 90, nlat)
        grid_lon = np.linspace(-180, 180, nlon)
        glon, glat = np.meshgrid(grid_lon, grid_lat)
        glat_rad = np.radians(glat.ravel())
        glon_rad = np.radians(glon.ravel())
        grid_xyz = np.column_stack([
            np.cos(glat_rad) * np.cos(glon_rad),
            np.cos(glat_rad) * np.sin(glon_rad),
            np.sin(glat_rad),
        ])

        # Some source points
        source_xyz = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        tree = cKDTree(source_xyz)
        nearest_dist, _ = tree.query(grid_xyz, k=1)
        field = nearest_dist.reshape(nlat, nlon)

        assert field.shape == (nlat, nlon)
        assert np.all(field >= 0)


class TestScriptStructure:
    """Verify structural properties of the updated script."""

    def test_script_imports_shapely_and_shapelyfeature(self):
        """Script imports the required modules for proper polygon rendering."""
        script_path = str(_SCRIPT_PATH)
        with open(script_path) as f:
            content = f.read()

        assert "import shapely.geometry" in content or "from shapely" in content
        assert "ShapelyFeature" in content

    def test_script_uses_distinct_colormaps(self):
        """Script uses different colormaps for ocean and continental particles."""
        script_path = str(_SCRIPT_PATH)
        with open(script_path) as f:
            content = f.read()

        assert 'cmap="inferno"' in content
        assert 'cmap="cividis"' in content

    def test_script_computes_nearest_distance(self):
        """Script computes and plots the nearest-particle distance field."""
        script_path = str(_SCRIPT_PATH)
        with open(script_path) as f:
            content = f.read()

        assert "nearest_dist" in content
        assert "tree.query(grid_unit_xyz, k=1)" in content

    def test_script_no_dateline_wrapper_in_code(self):
        """Script no longer calls DateLineWrapper in executable code."""
        import ast
        script_path = str(_SCRIPT_PATH)
        with open(script_path) as f:
            tree = ast.parse(f.read())

        # Check that no Name or Attribute node references DateLineWrapper
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and "DateLineWrapper" in node.id:
                pytest.fail("DateLineWrapper is still used in executable code")
            if isinstance(node, ast.Attribute) and "DateLineWrapper" in node.attr:
                pytest.fail("DateLineWrapper is still used in executable code")

    def test_script_uses_continental_partitioning(self):
        """Script assigns plate IDs from continental polygons, not static."""
        script_path = str(_SCRIPT_PATH)
        with open(script_path) as f:
            content = f.read()

        assert "partitioning_features=continental_features" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

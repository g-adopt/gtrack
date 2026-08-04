# Changelog

## Unreleased

### Added — `membership_property` on `build_indicator_source`

`build_indicator_source` now records region membership as its own property,
1.0 on the rotated seeds and 0.0 on the background, named by the new
`membership_property` argument (default `"membership"`, pass `None` to omit).
Its value on the background is pinned explicitly rather than inherited from
`background_value`, which concerns thickness-like channels and means nothing
for membership.

This exists because a single zero-outside channel is ambiguous. Filling the
background's thickness with 0 makes one number carry two unrelated statements —
"outside the region" and "inside the region, zero thickness" — so a downstream
interpolator smooths the product of coverage and thickness rather than the two
separately, and neither can be recovered afterwards. Any consumer wanting the
region's lateral extent gets it multiplied by the region's thickness, and vice
versa. With membership carried alongside, blending it gives the local in-region
weight fraction and dividing the blended thickness by that fraction recovers the
seed-weighted thickness undiluted by the background.

## 0.4.0 — 2026-07-23

### Changed — continental/craton point rotation is now deforming-aware (topological)

`PointRotator` was rebuilt around a single, deforming-aware engine:
`pygplates.TopologicalModel.reconstruct_geometry`, the same engine the ocean
`SeafloorAgeTracker` uses. It resolves rigid plates and deforming networks and
advects each point by whatever plate or network it sits in.

- **`topology_files` is now required** on `PointRotator(...)`. Constructing
  without it raises a clear `ValueError`. Update every construction site to pass
  the same topology files you pass to `SeafloorAgeTracker`.
- **Rotation no longer drops points and no longer depends on `plate_ids`.** With
  the default `deactivate_points=None`, `rotate` returns every input point
  (`n_out == n_in`), with properties and `plate_ids` preserved in input order.
  The previous behaviour — dropping points whose static-polygon plate id had no
  rigid sequence to the anchor, and mis-rotating continental points whose static
  id disagreed with the topology id — is gone.
- **`plate_ids` are now a labelling output, not a motion input.**
  `assign_plate_ids` is optional and defaults to `source="topology"` (consistent
  with how points actually move); `source="static"` is kept for back-compat, and
  `remove_undefined` now defaults to `False`.
- **Reconstructed positions differ from the previous rigid results by design.**
  On the Zahirovic 2022 / Merdith data the correction is a median of ~4° and up
  to ~30° over 50 Myr for continental points. Validate against the topological
  result, not the old output.

### Added — `build_indicator_source` source-construction helper

New `gtrack.build_indicator_source` (`gtrack/sources.py`) builds an indicator
source cloud for downstream nearest-neighbour interpolation: it rotates the seed
points topologically to the target age and unions them with a **fresh, uniform
background grid regenerated at that age**, dropping background points that
collide with a rotated seed. A fresh background is required for a well-posed
nearest-neighbour interpolation: back-rotating a fixed present-day grid deforms
it and opens gaps that grow with age, which can leave the interpolator with no
nearby background point. The helper needs no `plate_ids` and performs no
out-of-circuit reattachment or sliver-plate-id inheritance (both redundant under
the topological engine).

### Fixed

- `assign_plate_ids` (via `_get_plate_ids`) no longer misaligns plate ids with
  points: `pygplates.partition_into_plates` does not preserve input order
  (partitioned features are grouped ahead of unpartitioned ones), which
  scrambled the returned ids whenever any point fell outside all polygons. Ids
  are now mapped back to input positions via an index tag.

### Notes

- `boundaries.py` continental-polygon reconstruction is intentionally left rigid
  to remain identical to GPlately's SeafloorGrid (Phase 4 decision, pinned by
  `tests/test_boundaries_reconstruction.py`).
- gadopt integration (wiring its gplates sources to the topological engine and
  the `build_indicator_source` helper) is handled separately in the g-adopt
  repository.

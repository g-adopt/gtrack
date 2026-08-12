# Changelog

## 0.6.0 — 2026-08-12

### Added — an optional oceanic surface amplitude, published as a channel

`LithosphereCloudSource` now takes an optional `oceanic_amplitude` callable that
maps oceanic thickness in km to a lateral amplitude in `[0, 1]`. When it is
given, the source publishes a `surface_amplitude` channel alongside `thickness`
and `age`: the oceanic seeds carry the (clipped) amplitude the callable returns,
the continental seeds carry a flat `1.0`, and `provides` grows to name the new
channel. Left at its default of `None` nothing changes — no callable runs and no
channel is published, so existing consumers see exactly the clouds they saw
before.

The point is to let a consumer weaken thin, young seafloor without touching
continents. The lithosphere indicator drives viscosity as `1000^indicator`, so a
ridge that reads a full-strength lid prints a spurious signal in surface dynamic
topography that no smoothing or depth knob can remove; fading the merged
oceanic-plus-continental thickness removes it but then weakens thin cratonic and
marginal lithosphere in the bargain. Publishing the amplitude per point, faded on
the ocean and pinned to one on the continents, keeps the ocean-versus-continent
decision here in the source where the two populations are still separate. It
mirrors how `PolygonIndicatorSource` publishes `membership`, so a consumer only
has to read the channel and multiply.

## 0.5.0 — 2026-08-04

### Changed — mid-ocean ridge seeding is now reproducible between processes, and this moves numbers

pygplates returns `shared_boundary_section.get_shared_sub_segments()` in an
order that varies from one process to the next. Every routine that iterated it
therefore emitted the same result permuted between runs, and downstream that
permutation is not inert: it is what made a cold tracker walk irreproducible,
and it was the origin of the ~1.5e-4 spread seen in the g-adopt lithosphere
demo. Measured on Müller 2022 at 230 Ma, eight separate processes produced
three distinct seed orders.

All six call sites now visit sub-segments through the new
`gtrack.topology_order.ordered_sub_segments`, which sorts them by their full
resolved geometry. Narrower keys were tried and each fails: the feature id
belongs to the section so every sub-segment reports the same value, the sharing
plate ids are not a total order, and first-point/last-point/length is not
either — Matthews 2016 at 0 Ma has a section with two coincident two-point
sub-segments agreeing on all five.

**This is a reproducibility fix, not a physics correction, and it changes
results.** Cloud size moves by roughly 0.5% (n=16770 against a previous modal
16856). The order pygplates happened to return was itself an arbitrary member
of the permutation family; this picks a canonical member of the same family, so
any number that moves does so because it was never well defined. Anyone holding
a reference generated before this change should regenerate it, and cannot
attribute the difference against the old value.

### Added — age-source scaffolding for driving gtrack across geological time

Four additions that let a consumer walk gtrack through time without importing a
gtrack class or reimplementing gtrack's conventions:

- `AgeCloudSource`, a runtime-checkable Protocol saying only what a consumer
  needs — which properties the clouds carry (`provides`, excluding `xyz`), how
  to ask for one at an age (`at_age`), how to reject an age (`validate_age`),
  and whether the source may only be walked backwards
  (`monotonic_backward`). Because it is a Protocol, a numpy test double
  satisfies it, so a consumer becomes testable without reconstruction data.
- `PointCloud.from_data`, absorbing the cloud dispatch each caller was writing
  for itself: an existing cloud, a gridded file, a `(latlon, values)` pair, or a
  scalar broadcast onto a Fibonacci mesh. It rejects `bool`, which the code it
  was lifted from accepted — `bool` subclasses `int`, so `True` silently became
  a thickness of one everywhere.
- `load_latlon_grid_hdf5`, including the two easy-to-miss details: longitudes
  above 180 wrap to the negative half, and 1-D lat/lon are axes to be meshed
  while 2-D ones are already paired with the values. Needs the new `hdf5`
  extra; the core install stays numpy, scipy and pygplates.
- `CheckpointPolicy`, owning the checkpoint naming and resume convention.
  `best_at_or_before` means at or before in *time*, so it returns the smallest
  age still at least the target — the youngest checkpoint no younger than where
  the caller wants to be.

### Added — `LithosphereCloudSource` and `PolygonIndicatorSource`

The two source recipes, moved into gtrack beside the primitives they drive.
`LithosphereCloudSource` provides `{"thickness", "age"}` and walks forward only;
`PolygonIndicatorSource` provides `{"masked_thickness", "membership"}` and holds
no walk state, so its ages may be asked for in any order.

Two things are now inexpressible rather than merely fixed. `LithosphereCloudConfig`
nests a `TracerConfig` as a typed field instead of merging a loose kwargs dict,
so there is no second path to a tracker knob and no way for one to be silently
shadowed. And `PolygonIndicatorConfig` has no `n_points`: `background_n` and
`seed_fallback_n` are independent fields, because the single parameter they
replace was doing two unrelated jobs — sizing the background grid rebuilt at
every age, and standing in as a mesh size when the thickness input is a bare
number. Only the first is ever active on real data, and it also sets the
collision-removal radius, so a user adjusting what they believed was a seed
count was moving the region's edge by up to a few hundred kilometres.

The polygon channel is `masked_thickness`, not `thickness`. On a bounded source
the channel carries the region's depth multiplied by its membership once a
downstream blend has run, which is a different quantity from an unbounded
thickness field, and naming them apart is what lets a consumer's
required-versus-provided check reject a pairing that would read it as plain
depth.

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

### Fixed — a nominally zero age span no longer raises

`PointRotator._advect_topological` guarded against a zero-length span with an
exact `span == 0` test. Callers that derive ages by converting a
non-dimensional model time to Ma hand it float round-off instead — a measured
~1.4e-14 Myr — which fell through the guard into `reconstruct_geometry`, which
rejects a span it considers degenerate (measured: 1e-9 Myr or less) with
"Oldest time cannot be later than (or same as) youngest time."

The guard is now a tolerance, `ZERO_SPAN_TOLERANCE_MYR = 1e-6`. The value clears
pygplates' own threshold with margin rather than matching it, since a tolerance
of exactly 1e-9 leaves spans landing on the boundary still raising. 1e-6 Myr is
one year of plate motion, of order 10 cm, and still five orders of magnitude
below the shortest step gtrack supports.

### Documented — `background_n` also sets the collision-removal radius

The `background_n` docstring now records that it is not only a cost knob: it
fixes the collision-removal radius, `exclusion_factor * sqrt(4*pi /
background_n)`, so a coarser background deletes background points further out
from every seed and leaves an annulus holding seeds but no background. A
consumer sampling there sees only seeds and reads full membership, so the region
is effectively dilated by roughly that radius — 320 km at `background_n=5000`,
113 km at 40000. The dilation is a hole in the point cloud rather than a
property of the interpolation, so sharpening a downstream kernel does not reduce
it.

### Testing — CI now fetches the plate model, and the dead regression is honest

CI previously ran `pytest tests/` with no data-fetch step, so every test needing
reconstruction data was skipped there — including the craton oracle, which
therefore ran on one machine only. The workflow now downloads the Matthews model
and caches the tarball rather than the extracted tree, deliberately without
`restore-keys`: a fuzzy cache hit after a Makefile URL change would restore the
old tarball, satisfy make's target, and quietly test against the wrong plate
model.

Making the data reachable exposed `test_tracker_200_to_180_regression`, which
had been skipped rather than passing since the engine rewrite and raised
`TypeError` when actually run. It is repaired and green, which restores the only
pinned-number test on the tracker and ridge-spawning path — everything else
covers the rotation path.

Both its call and its reference had to be rebuilt. `default_refinement_levels`
became `default_mesh_points=10242`, the faithful translation of the level-5
icosahedral mesh it asked for. Note that this is deliberately *not* the count
that reproduces the old reference: scanning for one lands on 20480, which
returns the previous 190 Ma cloud size exactly and is a coincidence — at that
setting the median per-point position differs from the old reference by 2694 km.

The reference now stores four invariants per age rather than full arrays: exact
point count, mean age, max age, and the sum of absolute coordinates. Full arrays
at `rtol=1e-10` cannot survive being generated on one platform and asserted on
another, and a reference that goes red for the platform rather than for the code
carries no information. The four were each verified bit-identical across three
processes at different hash seeds, and verified non-vacuous: perturbing the mesh
size is caught by the point count, and perturbing the spreading rate by the mean
age. What they miss is a position-only change preserving count and mean age,
which the craton oracle already pins at `rtol=1e-9`.

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

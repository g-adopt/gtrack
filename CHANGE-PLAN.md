# CHANGE-PLAN: make continental/craton point rotation deforming-proof

**Status:** approved for implementation (Phase 0 complete, decision taken).
**Owner:** (assign)
**Audience:** engineer implementing the change end-to-end.

This document is self-contained. It records why we are changing the rotation
engine, the evidence behind the decision, the target design, and a phase-by-phase
task list with file anchors and acceptance criteria.

---

## 1. Context and the bug that started this

gtrack produces continental/craton/lithosphere indicator fields for a gadopt
mantle-convection run (see `examples/Cratons_M3_B/`). The present-day **Craton
Indicator** showed a hard cut slicing through the Central-America / northern
South-America craton. Reproduced standalone in `examples/Cratons_M3_B/`.

Root cause, established by investigation and the Phase 0 spike
(`examples/Cratons_M3_B/phase0_probe.py`):

gtrack has **two** reconstruction engines that disagree on deforming plates.

- **Deforming-aware (the ocean tracker):** `SeafloorAgeTracker` advects points
  with `pygplates.TopologicalModel.reconstruct_geometry`
  (`gtrack/hpc_integration.py:388`). It resolves topologies — rigid plates *and*
  deforming networks — and moves each point by whatever plate/network it sits in.
- **Rigid (the point rotator):** `PointRotator.rotate` moves points with
  `_move_points_batched` → `rotation_model.get_rotation(to_age, plate_id, from_age)`,
  one rigid pole per plate group (`gtrack/point_rotation.py:617`). Any `plate_id`
  with no rigid sequence to the anchor is caught by `_unreconstructable_plate_ids`
  (`:524`, used at `:874`) and **dropped**.

Continental/craton points go through the **rigid** rotator. Two failures result:

1. ~2,700 of 40,000 masked-grid points (mostly the "zero background" outside the
   craton polygon, plus ~95 genuine craton seeds such as the Kalahari craton on
   plate 77030) are **dropped** because their Merdith static-polygon plate id has
   no rigid sequence in the Zahirovic rotation file. The dropped zeros punch holes
   in the background; the downstream kNN interpolator (with
   `distance_threshold=np.pi`) then reaches across a hole and pulls in a distant
   craton — the visible "cut".
2. The quieter, larger error: **83.3%** of craton seeds get a *different* plate id
   from the Zahirovic topology than from the Merdith static polygons that
   `assign_plate_ids` uses. Where those ids carry different motion, surviving
   points are **rotated by the wrong plate's pole** — median 3.9°, up to ~30° over
   50 Myr.

## 2. Phase 0 evidence (the decision basis)

From `examples/Cratons_M3_B/phase0_probe.py` on the real Zahirovic 2022 / Merdith
data, present→50 Ma:

| Test | Result |
|------|--------|
| Topological advection keeps in-circuit seeds | 2824/2824 |
| Topological advection keeps out-of-circuit seeds (rigid drops these) | 95/95 |
| Full 40,000 masked grid survivors — rigid | 37,286 (2,714 holes) |
| Full 40,000 masked grid survivors — topological | 40,000 (**0 holes**) |
| Seeds whose Merdith id ≠ Zahirovic topology id | **83.3%** |
| Topological vs rigid using the **topology's** ids | median **0.000°**, 95pct 1.5° |
| Topological vs rigid using the **Merdith** ids (today's pipeline) | median 3.9°, 95pct 30° |

Interpretation: topological advection is correct (it equals rigid rotation with
self-consistent ids to <1.5°), keeps every point, and leaves no holes. The rigid
pipeline is wrong for most continental points because of the static-polygon ↔
rotation-model plate-id mismatch.

## 3. Decision (approved)

- **Adopt topological advection as the single rotation engine.** Remove the rigid
  per-plate rotation path and the out-of-circuit drop from `PointRotator`. No
  `method="rigid"` fallback.
- **Retain the fresh-zero-grid — as a gtrack helper.** *(Corrected: the original
  plan said "drop it". That was wrong.)* The Phase 0 "0 holes" number counted
  *dropped points*, not spatial uniformity. Rendering the plain full-grid
  topological path at 50 Ma shows that back-rotating a fixed present-day grid
  **deforms** it — the grid stretches open at divergent boundaries and piles up
  at trenches, opening gaps that GROW with age. For a "zero outside the region"
  indicator that deformation is meaningless and harmful: it can leave the
  downstream kNN interpolator (dt=π) with no nearby background zero. So a fresh,
  uniform background grid regenerated at the target age is a **correctness
  requirement** for a well-posed nearest-neighbour background, not a rigid-engine
  workaround. It is promoted to gtrack as `build_indicator_source`
  (`gtrack/sources.py`). `reattach_out_of_circuit` and `_inherit_sliver_plate_ids`
  are dropped as **redundant** (topological rotate already keeps all
  out-of-circuit seeds — Phase 0: 95/95 — and motion is plate-id-independent).
- **Accept that craton reconstruction positions change materially** vs the current
  simulation (median ~4°, up to ~30°). This is a correction. Validation compares
  against the topological result, not the old (buggy) output.

`mor_seeds.py:65` (`get_rotation` for stage-pole ridge spreading) is intentionally
rigid and is **out of scope** — do not touch it.

## 4. Target design

`PointRotator` becomes a thin wrapper around a `TopologicalModel`, mirroring the
tracker.

- `__init__` builds `pygplates.TopologicalModel(topology_features, rotation_model)`
  once. **`topology_files` becomes required** (today it is optional and gadopt
  does not pass it — see Phase 6).
- `rotate(cloud, from_age, to_age, *, time_step=1.0, deactivate_points=None)`
  advects all points topologically. Motion no longer depends on `plate_ids`; the
  method no longer requires them and no longer drops points.
- `assign_plate_ids` is retained for *labelling only* (an output property), not for
  motion. It gains an optional topology-based mode (Phase 2).

Reference implementation sketch (validated in the probe; see the tracker loop at
`hpc_integration.py:369-422` for the stepping pattern and the divisibility
constraint on `time_increment`):

```python
def _advect_topological(self, lats, lons, from_age, to_age,
                        time_step=1.0, deactivate_points=None):
    points = pygplates.MultiPointOnSphere(list(zip(lats, lons)))
    lo, hi = min(from_age, to_age), max(from_age, to_age)
    # reconstruct_geometry requires (hi - lo) to be an integer multiple of the
    # increment; pick n_steps and derive an exact increment.
    span_len = hi - lo
    if span_len == 0:
        return np.asarray(lats), np.asarray(lons), np.ones(len(lats), bool)
    n_steps = max(1, int(np.ceil(span_len / time_step)))
    increment = span_len / n_steps
    ts = self._topological_model.reconstruct_geometry(
        points,
        initial_time=from_age,
        oldest_time=hi,
        youngest_time=lo,
        time_increment=increment,
        deactivate_points=deactivate_points,   # None => keep everything
    )
    recon = ts.get_geometry_points(to_age, return_inactive_points=True)
    n = len(lats)
    out_lat = np.full(n, np.nan); out_lon = np.full(n, np.nan)
    active = np.zeros(n, dtype=bool)
    for i, p in enumerate(recon):
        if p is not None:
            out_lat[i], out_lon[i] = p.to_lat_lon()
            active[i] = True
    return out_lat, out_lon, active
```

Behavioural contract:
- `deactivate_points=None` ⇒ every input point is returned active; property arrays
  and `plate_ids` pass through unchanged in order.
- If a caller opts into deactivation, inactive points come back as `None`; the
  result cloud is subset by the active mask (properties + plate_ids subset in
  lockstep, as `_update_from_reconstructed` already does at
  `hpc_integration.py:430`).
- Direction: works both ways (present→older and older→present); Phase 5 adds a
  round-trip test.

---

## Phase 1 — Core topological rotation engine in `PointRotator`

**Goal:** replace the rigid engine with topological advection; no point is ever
silently dropped.

Tasks:
1. In `PointRotator.__init__` (`point_rotation.py:684`): build and store
   `self._topological_model = pygplates.TopologicalModel(self.topology_features,
   self.rotation_model)`. Make `topology_files` **required**; raise a clear
   `ValueError` if absent (the old static-polygons-only construction path is gone
   for motion, though static polygons may still be used for labelling in Phase 2).
2. Add `_advect_topological` (sketch above) as a private method.
3. Rewrite `PointRotator.rotate` (`:~805-905`):
   - Drop the `plate_ids is None` hard requirement for motion.
   - Remove the `_unreconstructable_plate_ids` block and the `cloud.subset(~bad_mask)`
     drop (`:872-886`).
   - Call `_advect_topological`; build the result `PointCloud` preserving
     `properties` and `plate_ids` (subset by active mask only if deactivation is
     enabled).
   - Add `time_step: float = 1.0` and `deactivate_points=None` params. Keep
     `from_age`/`to_age` positional for back-compat. `reassign_plate_ids` becomes a
     no-op or is removed (decide; if kept, document as label-only).
4. Delete the now-dead rigid helpers **only after** tests are migrated:
   `_move_points_batched` (`:571`), `_rotate_points_batch`, and
   `_unreconstructable_plate_ids` (`:524`) — unless Phase 5 keeps
   `_unreconstructable_plate_ids` purely as a test/diagnostic helper (recommended:
   keep it, unused in production, for the regression test that documents old
   behaviour).

**Acceptance:** rotating a cloud 0→50 Ma returns the same number of points as
input; positions match the Phase 0 topological result (median <0.001° vs
rigid-with-topology-ids on the example data).

## Phase 2 — Plate-id assignment (labelling only, topology-consistent)

**Goal:** stop the label/motion mismatch at the source and make plate ids (an
output property) consistent with the engine.

Tasks:
1. `assign_plate_ids` (`point_rotation.py:~730`): add
   `source="static" | "topology"` (default `"topology"`). The `"topology"` mode
   partitions with a `pygplates.TopologicalSnapshot` /
   `pygplates.PlatePartitioner` at `at_age` (validated in Phase 0), so assigned ids
   match what the topological engine actually moves points by. Keep `"static"` for
   back-compat.
2. Motion no longer needs plate ids, so `assign_plate_ids` is optional. Callers
   that only rotated (and used ids solely to satisfy `rotate`) can drop the call.
3. Note in the docstring that ids are now a labelling convenience, not a motion
   input.

**Acceptance:** with `source="topology"`, assigned ids equal the partitioner ids
used in Phase 0; on the example data, rigid-rotating by these ids matches
topological advection to <1.5° (95pct) — i.e. ids and motion are consistent.

## Phase 3 — Source-construction path (fresh grid RETAINED as a gtrack helper)

**Goal:** provide a well-posed source cloud for nearest-neighbour interpolation,
and drop only the genuinely-redundant rigid-era patches.

**Corrected premise.** The original Phase 3 said "remove the fresh-zero-grid
because topological advection keeps the background intact (0 holes)". That was
wrong — see the corrected decision in section 3. Rotating a fixed present-day
grid deforms it (gaps grow with age), so a fresh uniform grid regenerated at the
target age is required for a hole-free background. What IS redundant:
- `_inherit_sliver_plate_ids` — motion no longer depends on `plate_ids` at all;
- `reattach_out_of_circuit` — topological rotate already keeps all out-of-circuit
  seeds (Phase 0: 95/95), so there is nothing to reattach.

Tasks:
1. Add a serial-safe gtrack helper `build_indicator_source`
   (`gtrack/sources.py`, exported from `gtrack/__init__.py`): rotate the *seeds
   only* topologically (no reattach, no sliver inheritance), union with a FRESH
   uniform Fibonacci background generated at the target age, and drop background
   points within one grid-spacing of a rotated seed (set-union collision
   removal). Generalised over ALL properties, not just "thickness".
2. Source construction becomes **seeds-only rotate + fresh background**, replacing
   the full-masked-grid rotate. The example prototypes
   (`fill_exterior_with_zeros` / `compute_craton_filled`) are superseded by the
   helper; `reattach_out_of_circuit` / `inherit_sliver_plate_ids` stay
   example-only and are NOT used by the helper.

**Acceptance:** `build_indicator_source` yields a dense, hole-free background at
any age (max nearest-neighbour coverage distance bounded near the ideal grid
spacing and not growing with age); all seeds preserved at their rotated
positions; no reattach/sliver code in the helper.

## Phase 4 — Secondary audit: rigid polygon reconstruction (`boundaries.py`)

**Goal:** a conscious decision on the *other* rigid-reconstruction site.

`ContinentalPolygonCache` reconstructs continental polygons with rigid
`pygplates.reconstruct` (`boundaries.py:153`). Used by the tracker's
`_remove_continental_points` at every ocean-tracking age, and by `PolygonFilter`
when `at_age != 0`. Continental polygons whose plate ids mismatch the rotation
model (same failure class) get misplaced boundaries. Our craton/crust containment
runs at `at_age=0` (identity), so it is currently unaffected.

Tasks:
1. Quantify: at a few ages (e.g. 50, 100 Ma), compare rigid-reconstructed
   continental polygons vs topology-consistent reconstruction; measure boundary
   displacement.
2. Decide: make polygon reconstruction topology-consistent, or defer with a
   documented limitation. Either way, add a regression test that pins the chosen
   behaviour so it is not an accident.

**Acceptance:** a written decision + a test locking current or new behaviour.

## Phase 5 — Regression tests and fixtures

**Goal:** guarantee the drop/mismatch bug cannot return, with self-contained tests.

Existing tests to update (they encode old rigid/drop behaviour):
`tests/test_point_rotation.py`, `tests/test_regression.py`,
`tests/test_plate_id_alignment.py`, `tests/test_temporal_snapping.py`.

New fixtures/tests:
1. **Tiny synthetic topological model** in `tests/data/`: a rotation file with
   {anchor, plate A}, a small topology set including a deforming network, and
   static polygons that reference an extra plate B with **no** rigid sequence.
   One point on B. Assert: topological `rotate` **keeps** and advects it (the old
   rigid path would have dropped it — assert that too via the retained
   `_unreconstructable_plate_ids`, documenting the regression).
2. **No-silent-drop invariant:** `rotate` with `deactivate_points=None` returns
   `n_out == n_in` for arbitrary inputs.
3. **Round-trip:** rotate 0→T then T→0 returns ~original positions (tol a few
   km), both directions.
4. **Property/plate_id preservation** across `rotate`.
5. **Engine-consistency:** on the synthetic model, topological advection equals
   rigid rotation using the topology's ids (mirrors the Phase 0 decisive test).
6. **Deep-time edge case:** rotate 0→(oldest age) does not raise and keeps points
   whose plate lineage may vanish (define + pin behaviour).
7. **Slow/optional real-data test** (skip if `examples/Cratons_M3_B` data absent):
   all 2,919 craton seeds survive 0→50 Ma; positions match the Phase 0 numbers.
8. Fold the `examples/Cratons_M3_B` reproduction into a numeric end-to-end check
   (background coverage fraction == 1.0; craton seed count preserved) instead of
   eyeballed plots.

**Acceptance:** `pytest tests/ -v` green; the synthetic drop-regression test fails
against the old engine and passes against the new one.

## Phase 6 — gadopt integration and end-to-end validation

**Goal:** wire gadopt to the new API and prove the craton cut is gone.

Files: `gadopt/gplates/sources.py` (and any `PointRotator(...)` construction).

Tasks:
1. **Pass topology files to `PointRotator`.** Today (`sources.py:546` and `:788`)
   gadopt constructs `PointRotator(rotation_files=..., static_polygons=...)` with
   **no** `topology_files`. Add
   `topology_files=gplates_connector.topology_filenames`. Without this, Phase 1's
   required-topology check will raise. This is the single most important
   integration edit.
2. Simplify `LithosphereSource._load_continental` (`sources.py:556`) and
   `PolygonSource._load_region` (`:794`): drop `_inherit_sliver_plate_ids`
   (`sources.py:167`, `:807`) and the plate-id gymnastics that existed only to
   satisfy rigid rotation. `assign_plate_ids` may still be called if ids are wanted
   as output, but is no longer needed for `rotate`.
3. Confirm `PolygonSource._compute_sources` (`:810`) and
   `LithosphereSource._compute_sources` (`:609`) now produce hole-free clouds.
4. **Validation:** extend `examples/Cratons_M3_B/reproduce.py` with the downstream
   kNN-gaussian interpolation (k=300, σ=0.04, dt=π) + the quintic output, render
   the craton **indicator** at ~10 Ma (drops occur there; 0 Ma has none), and show
   legacy-vs-fixed side by side. The cut must be gone and the Kalahari/Indian
   cratons present. This is the definitive proof — the source-cloud scatter only
   proves the precondition.

**Acceptance:** craton indicator at ~10 Ma shows no cut, full craton coverage;
positions consistent with the Phase 0 topological ground truth.

## Phase 7 — Docs, changelog, cleanup

Tasks:
1. Update `CLAUDE.md` "Common Gotchas": replace the rigid-rotation framing with the
   topological-advection contract; note the plate-id-is-labelling change and the
   deliberate position shift.
2. Docstrings on `PointRotator`, `rotate`, `assign_plate_ids`; module docstring on
   `point_rotation.py` describing the single engine.
3. Changelog entry: "Continental/craton point rotation is now deforming-aware
   (topological). Rotation no longer drops out-of-circuit points and no longer
   depends on static-polygon plate ids; reconstructed positions differ from
   previous rigid results by design."
4. Remove dead rigid helpers if not retained for tests; delete example-only
   prototypes from the production narrative (keep them in `examples/`).

---

## Risks and open questions

- **`reconstruct_geometry` divisibility constraint.** `(oldest-youngest)` must be an
  integer multiple of `time_increment`. Handle by deriving the increment from an
  integer `n_steps` (sketch above). Mirror the tracker's stepping if in doubt.
- **Performance.** One `reconstruct_geometry` call per `rotate` (C++ internal
  stepping). Probe: 2,919 points, 50×1-Myr steps ≈ 0.3 s. Deep-time (0→400 Ma) is
  ~400 internal steps — validate it stays acceptable; rotating only nonzero seeds
  is the lever if not.
- **Deactivation policy.** Default `None` (keep all) is right for continental/craton
  back-rotation. Do not import the ocean tracker's collision/network-exit
  deactivation here.
- **Anchor plate.** Default anchor matched rigid to 0.000° in Phase 0. Expose
  `anchor_plate_id` only if a caller needs it.
- **API break.** `topology_files` becomes required on `PointRotator`. Every
  internal + gadopt construction site must be updated (Phase 6.1). Grep for
  `PointRotator(` before merging.

## File-change inventory (quick reference)

| File | Change |
|------|--------|
| `gtrack/point_rotation.py` | New topological engine; `__init__` builds `TopologicalModel`, `topology_files` required; rewrite `rotate`; retire rigid helpers; topology-based `assign_plate_ids` |
| `gtrack/boundaries.py` | Phase 4 decision (polygon reconstruction consistency) |
| `tests/data/` | New tiny synthetic topological model fixture |
| `tests/test_point_rotation.py`, `test_regression.py`, `test_plate_id_alignment.py`, `test_temporal_snapping.py` | Update to new behaviour; add drop-regression, round-trip, no-drop, consistency, deep-time tests |
| `gadopt/gplates/sources.py` | Pass `topology_files` to `PointRotator`; drop sliver-inheritance and rigid-era plate-id handling |
| `examples/Cratons_M3_B/reproduce.py` | Add interpolation + quintic; legacy-vs-fixed craton indicator |
| `CLAUDE.md`, changelog | Phase 7 docs |

## Definition of done

1. `PointRotator` advects topologically; no silent drops; positions match Phase 0.
2. `pytest tests/ -v` green, including the new synthetic drop-regression test.
3. gadopt sources pass topology files and build hole-free craton clouds.
4. Craton indicator at ~10 Ma shows no cut and full craton coverage.
5. Docs/changelog updated; the position change is documented as intentional.

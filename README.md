# print-doctor

Take a cleaned AI-generator mesh and finish it into a print-ready STL —
welded, repaired, dust-free, and gated. The missing back half of
[ai-mesh-cleanup](https://github.com/1915rising/ai-mesh-cleanup).

## The problem

Voxel-remesh cleanup seals every shell of a raw generator export, but
it cannot merge solids that never touch. So "cleaned" output still
carries detached floater islands — floating hair-strand shells, crumbs,
debris — and a slicer will happily print them as loose pieces rattling
around the build plate. Cleanup's own README calls this out as its
detached-floaters limitation; nothing in the pipeline removed them.

Measured on a real cleaned full-body export (Hunyuan3D, 45k tris — a
female avatar with Christmas dress):

| Stage | Verts | Tris | Shells | Boundary edges | Watertight |
|---|---|---|---|---|---|
| Cleaned GLB, as imported | 44,601 | 44,997 | 4,715* | 35,515 | No |
| After weld + repair | 22,482 | 44,997 | 8 | 9 | No |
| After dedust (= final) | 22,449 | 44,962 | **1** | **0** | **Yes** |

\* The raw import's "shells" are an artifact of split vertices (a known
glTF exporter quirk): the surface is one body stored with duplicated
verts, so an index-based count sees thousands of fragments. The weld
collapses it to the truth — 8 shells: 1 body + 7 dust islands, the
largest dust island 847 mm³ against the body's 11,621,467 mm³
(0.007%). Those 7 islands are what this tool exists to remove.

The finished STL was then re-gated independently (trimesh, below):
voxel IoU **0.9998** against the source (worst octant 0.9994) — the
finisher removed the dust and nothing else. Height drift 0.0002 mm.
Full reports in [`examples/`](examples/).

## What it does

`print_doctor.py` (Blender, headless):

1. **Audit** the import — shells, boundary / non-manifold edges,
   zero-area triangles, watertight, height.
2. **Weld + repair** — `remove_doubles` (2e-6), delete zero-area faces,
   fill the remaining boundary holes.
3. **Dedust** — union-find the mesh into islands, measure each island's
   volume by fan triangulation, delete every island below the dust
   limit (see below).
4. **Optional sizing** — `--target-height-mm` scales uniformly to an
   exact print height and grounds the mesh at z = 0.
5. **Gate + export** — the acceptance gate (watertight, exactly one
   component, height unchanged within tolerance) must pass; the STL is
   written in millimetres either way, but the exit code tells the
   truth: 0 = PASS, 1 = FAIL, 2 = could not run.

`gate_check.py` (pure Python + trimesh) re-checks the exported STL
with a *different* library than the one that produced it, so the two
implementations have to agree: watertight, one component, height — and,
with `--reference`, shape fidelity against the source (sampled fold-angle
roughness + voxel IoU, global and per-octant).

The repair combo and dust removal are extracted from the finisher of
the 1915rising miniatures pipeline, where they were proven on 30
production miniatures; the roughness/IoU gate is adapted from that
pipeline's acceptance gate. One deliberate deviation: the weld runs
**first** here, not last. On split-vertex input, filling "holes" before
welding filled 35,515 split seams with junk (triangles 44,997 → 52,754,
non-manifold edges 0 → 9,073, measured). Welded first, the same asset
needed 9 hole fills and kept its triangle count exactly.

## Usage

```bash
# Finish a cleaned GLB into a gated STL (input assumed in metres)
blender --background --python print_doctor.py -- \
    --input cleaned.glb --output print_ready.stl --report report.json

# Same, sized to an exact 150 mm print height
blender --background --python print_doctor.py -- \
    --input cleaned.glb --output figure_150mm.stl --target-height-mm 150

# Independent re-gate of the STL (self-gate)
python3 gate_check.py print_ready.stl --height-mm 1168.363

# Independent re-gate with shape fidelity vs the source GLB
python3 gate_check.py print_ready.stl --reference cleaned.glb \
    --pitch-mm 3.5
```

### print_doctor.py arguments

| Argument | Default | Meaning |
|---|---|---|
| `--input` | *(required)* | Cleaned mesh (`.glb` / `.gltf`) |
| `--output` | *(required)* | Where to write the print-ready `.stl` |
| `--report` | — | Optional path to save the full report as JSON |
| `--dust-mm3` | `0.25` | Absolute dust floor, mm³ (the miniatures pipeline's proven threshold) |
| `--dust-rel` | `0.001` | Dust ceiling as a fraction of the main island's volume |
| `--target-height-mm` | — | Scale uniformly to this exact height, grounded at z = 0 |
| `--height-tol-mm` | `0.5` | Gate: allowed height drift, mm |
| `--height-tol-pct` | `0.5` | Gate: allowed height drift, % of the main island's height (the gate uses whichever allowance is larger) |
| `--unit-scale` | `1000` | Scene-unit → mm multiplier (input in metres) |

**The dust rule:** an island is removed when its volume is below the
*larger* of the absolute floor and the relative ceiling —
`max(dust_mm3, dust_rel × main_island_volume)`. The absolute floor
keeps tiny models from eating real parts; the relative ceiling keeps
life-size models from keeping proportionally huge "dust". On the test
asset the effective limit was 11,621 mm³ (0.1% of the body), and the
largest island removed was 847 mm³ — an order of magnitude of margin.
A genuinely separate part *above* the limit (a held accessory that
never touched the body, say) is **kept**, and the gate then fails
loudly on the component count instead of silently deleting your sword.
Lower `--dust-rel` to 0 only if you want finish_mini's original
absolute-only behaviour.

**Height gate reference:** the gate compares the final height against
the *main island's* height after repair, not the raw bounding box — a
floater hovering above the head legitimately extends the box, and
removing it must not fail the gate.

### gate_check.py arguments

| Argument | Default | Meaning |
|---|---|---|
| `--reference` | — | Source mesh to compare against (`.glb` in metres, or `.stl` in mm) |
| `--pitch-mm` | `0.4` | Voxel pitch for the IoU comparison — **scale it to the model** (see below) |
| `--min-iou` | `0.98` | IoU pass threshold (reference mode only) |
| `--ref-scale` | `1000` | Reference scale into mm for `.glb` references |
| `--height-mm` | — | Expected height; enables the height check |
| `--height-tol-mm` | `0.5` | Height tolerance, mm |

**Picking a pitch:** 0.4 mm suits miniatures (~150 mm tall). The voxel
grid is dense, so pitch must grow with the model — roughly 0.3% of its
height (the 1,168 mm test figure used 3.5 mm). In testing, a full-size
figure at a 2 mm pitch OOM-killed the gate on a 6 GB machine; at
3.5 mm the full reference gate takes ~90 s, dominated by voxelisation
and roughness sampling. The self-gate (no `--reference`) takes ~1 s.

**glTF references** are converted from Y-up to the STL's Z-up
automatically (the conversion Blender's importer applies; trimesh
loads glTF raw). Without it the IoU compares a standing figure against
a lying one and reads ~0 — measured 0.0027 before the fix, 0.9998
after.

## Requirements

- **Blender 4.x** for `print_doctor.py` — developed and tested on
  **4.2.3 LTS** (Blender's bundled Python + NumPy; no extra packages).
- **Python 3.9+ with trimesh + NumPy** for `gate_check.py`.
- A cleaned `.glb` / `.gltf` — the output of ai-mesh-cleanup or an
  equivalent voxel/decimate pass, in metres.

## Limitations — read before you rely on it

- **Geometry only, like the cleanup before it.** Voxel remeshing
  already destroyed the UVs upstream; the STL carries no materials or
  colour. This tool finishes shape, not appearance.
- **Dedust deletes detached geometry.** That is the point — but it
  means a model whose design *intends* separate floating pieces will
  lose any of them that fall below the dust limit, and the gate will
  fail (by design) on any that survive above it. This is a finisher
  for single-solid prints, not a scene processor.
- **Island volumes of open shells are estimates.** Fan triangulation
  is exact for closed islands; a floater that is itself an open
  surface gets a consistent estimate, which is all dust classification
  needs — but don't read the printed mm³ of an open island as a true
  volume.
- **The height gate guards the main body, not the silhouette.**
  Repair and dedust never move the main island's surface (hole fills
  patch sub-mm gaps); the fidelity claim rests on the IoU check, so
  run the reference gate when the shape matters.
- **Tested on exactly one asset class so far:** Hunyuan3D full-body
  output after ai-mesh-cleanup (two generations of it — the test suite
  was run against a cleaned female avatar with Christmas dress,
  45k tris, plus a
  sized 150 mm variant). Other generators and pre-welded inputs are
  untested; the method only assumes a GLB of triangles in metres.
  Issues and test reports welcome.
- **Not a slicer.** The output is a clean STL; supports, orientation
  on the plate, and wall-thickness checks for your printer are still
  your slicer's job.

## Tests

```bash
BLENDER=/path/to/blender python3 tests/run_tests.py cleaned.glb
```

Runs the full chain on your cleaned GLB (defaults to
`$PRINT_DOCTOR_TEST_GLB`): internal gate, independent self-gate,
reference IoU, and a 150 mm sizing pass — 10 assertions. On the
development asset: **10/10 PASS**.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 1915rising.

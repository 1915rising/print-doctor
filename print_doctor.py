#!/usr/bin/env python3
"""
print_doctor.py -- take a cleaned AI-generator mesh (the output of
ai-mesh-cleanup or any similar voxel/decimate pass) and finish it into a
print-ready STL: weld split vertices, repair the surface, remove the
detached floater shells cleanup leaves behind, and gate the result.

Why this exists: voxel-remesh cleanup seals every shell but cannot merge
solids that never touch, so cleaned generator output still carries
detached floater islands (measured on a real cleaned full-body export:
8 shells after welding -- 1 body + 7 dust islands, the largest dust
island 0.007% of the body's volume). A slicer will happily print those
as loose crumbs rattling around the build plate. This finisher removes
them and proves the result is one watertight solid at the right height.

Pipeline (all inside Blender, headless):
  1. Import the .glb / .gltf, join objects, bake transforms.
  2. Audit the raw import (shells, boundary / non-manifold edges,
     zero-area triangles, watertight or not, height).
  3. Weld + repair: remove_doubles (2e-6) -> delete zero-area faces ->
     fill remaining boundary holes.
  4. Dedust: drop every island whose volume is below the dust threshold
     (an absolute mm^3 floor OR a fraction of the main island's volume).
  5. Optional sizing: --target-height-mm scales uniformly and grounds
     the mesh at z = 0.
  6. Audit again, run the acceptance gate (watertight, exactly one
     component, height unchanged within tolerance), export the STL
     in millimetres, and write the JSON report.

Provenance: the repair combo and the fan-triangulated dust-island
removal are extracted from finish_mini.py, the finisher of the
1915rising miniatures pipeline (2026-11-21), where they were proven on
30 production miniatures; the audit is the numpy edge analysis from
ai-mesh-cleanup. One deliberate deviation from finish_mini: the weld
(remove_doubles) runs FIRST here, not last. finish_mini's inputs were
already-welded re-voxelled meshes, so its order was harmless there;
cleaned GLBs can carry split vertices (a documented exporter quirk),
and on the test asset below, filling "holes" before welding filled
35,515 split seams with junk faces (triangles 44,997 -> 52,754,
non-manifold edges 0 -> 9,073). Welding first fills only the real
holes (9 boundary edges on the same asset). The steps are identical;
the order is adapted to the input class, and both orders were measured.

Usage:
  blender --background --python print_doctor.py -- \\
      --input cleaned.glb --output print_ready.stl \\
      [--report report.json] [--dust-mm3 0.25] [--dust-rel 0.001] \\
      [--target-height-mm 150] [--unit-scale 1000]

Arguments (all after the "--" separator):
  --input PATH          Cleaned generator mesh (.glb / .gltf). Required.
  --output PATH         Where to write the print-ready .stl. Required.
  --report PATH         Optional: also write the full report as JSON.
  --dust-mm3 F          Absolute dust floor in mm^3 (default 0.25, the
                        miniatures pipeline's proven threshold).
  --dust-rel F          Relative dust ceiling: islands smaller than this
                        fraction of the main island's volume are dust
                        (default 0.001 = 0.1%). The effective limit is
                        the LARGER of the absolute floor and the
                        relative ceiling (see README).
  --target-height-mm F  Optional: scale uniformly so the finished mesh
                        is exactly this tall, grounded at z = 0.
  --height-tol-mm F     Gate: allowed height drift, absolute (default 0.5).
  --height-tol-pct F    Gate: allowed height drift, percent of the main
                        island's height (default 0.5). The gate uses
                        whichever allowance is larger.
  --unit-scale F        Scene-unit -> millimetre multiplier for the STL
                        export and all mm figures (default 1000: input
                        in metres, the glTF/Blender standard).

Exit codes: 0 = gate PASS, 1 = gate FAIL, 2 = could not run.

Copyright (c) 2026 1915rising -- MIT License (see LICENSE).
"""

import argparse
import json
import sys
import time
from collections import defaultdict

import bpy
import bmesh
import numpy as np


# --------------------------------------------------------------------------
# Argument parsing (Blender passes script args after "--")
# --------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(
        prog="print_doctor.py",
        description="Finish a cleaned AI-generator GLB into a gated, "
                    "print-ready STL.")
    p.add_argument("--input", required=True, help="input .glb / .gltf")
    p.add_argument("--output", required=True, help="output .stl")
    p.add_argument("--report", default=None,
                   help="optional path to write the JSON report")
    p.add_argument("--dust-mm3", type=float, default=0.25,
                   help="absolute dust floor in mm^3 (default 0.25)")
    p.add_argument("--dust-rel", type=float, default=0.001,
                   help="dust ceiling as a fraction of the main island's "
                        "volume (default 0.001)")
    p.add_argument("--target-height-mm", type=float, default=None,
                   help="scale uniformly to this final height in mm")
    p.add_argument("--height-tol-mm", type=float, default=0.5,
                   help="gate height tolerance, mm (default 0.5)")
    p.add_argument("--height-tol-pct", type=float, default=0.5,
                   help="gate height tolerance, %% of main height "
                        "(default 0.5)")
    p.add_argument("--unit-scale", type=float, default=1000.0,
                   help="scene units -> mm multiplier (default 1000)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Mesh audit -- numpy edge analysis in the style of ai-mesh-cleanup,
# extended with per-island statistics (volume / z-range / face count)
# computed on the welded topology. An edge used by exactly 1 triangle is
# a boundary edge; used by >2 it is non-manifold; a mesh with neither is
# watertight. Islands are connected components over shared vertices.
# --------------------------------------------------------------------------

def _island_stats(tris, verts):
    """Per-island signed volume, z range and face count.

    Volume uses fan triangulation from each face's first vertex (the
    finish_mini method, including its 2026-11-21 fan-triangulation fix):
    sum over faces of v0 . (vi x vi+1) / 6 across the fan. Exact for
    closed islands, a consistent estimate for open ones -- which is all
    dust classification needs.
    """
    nv = len(verts)
    parent = np.arange(nv, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]],
                            tris[:, [2, 0]]])
    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb

    vol = defaultdict(float)
    zmin = defaultdict(lambda: float("inf"))
    zmax = defaultdict(lambda: float("-inf"))
    nfaces = defaultdict(int)
    roots = np.array([find(int(v)) for v in tris[:, 0]])
    for t, r in zip(tris, roots):
        vs = verts[t]
        # fan triangulation of the (already triangulated) face set:
        # loop triangles are triangles, so the fan is the face itself,
        # but keep the general fan form finish_mini uses for polygons.
        for i in range(1, len(vs) - 1):
            vol[r] += float(vs[0] @ np.cross(vs[i], vs[i + 1])) / 6.0
        for v in vs:
            if v[2] < zmin[r]:
                zmin[r] = float(v[2])
            if v[2] > zmax[r]:
                zmax[r] = float(v[2])
        nfaces[r] += 1
    islands = [{"volume": vol[r], "faces": nfaces[r],
                "zmin": zmin[r], "zmax": zmax[r]} for r in vol]
    islands.sort(key=lambda d: -abs(d["volume"]))
    return islands


def audit_mesh(me, unit_scale):
    """Health metrics + island census for a Blender mesh datablock."""
    me.calc_loop_triangles()
    nv = len(me.vertices)
    ntris = len(me.loop_triangles)
    info = {"verts": nv, "tris": ntris}
    if nv == 0 or ntris == 0:
        info.update({"unique_edges": 0, "boundary_edges": 0,
                     "nonmanifold_edges": 0, "zero_area_tris": 0,
                     "shells": 0, "watertight": False,
                     "height_mm": 0.0, "main_height_mm": 0.0,
                     "islands": []})
        return info

    tris = np.empty(ntris * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tris)
    tris = tris.reshape(-1, 3)
    verts = np.empty(nv * 3, dtype=np.float64)
    me.vertices.foreach_get("co", verts)
    verts = verts.reshape(-1, 3)

    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e.sort(axis=1)
    keys = e[:, 0] * nv + e[:, 1]
    ukeys, counts = np.unique(keys, return_counts=True)
    boundary = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())

    a = verts[tris[:, 1]] - verts[tris[:, 0]]
    b = verts[tris[:, 2]] - verts[tris[:, 0]]
    area2 = np.sqrt((np.cross(a, b) ** 2).sum(axis=1))
    zero_area = int((area2 < 1e-12).sum())

    islands = _island_stats(tris, verts)
    main = islands[0] if islands else None
    info.update({
        "unique_edges": int(len(ukeys)),
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "zero_area_tris": zero_area,
        "shells": len(islands),
        "watertight": bool(boundary == 0 and nonmanifold == 0),
        "height_mm": round(float(verts[:, 2].max() - verts[:, 2].min())
                           * unit_scale, 3),
        "main_height_mm": round((main["zmax"] - main["zmin"]) * unit_scale, 3)
        if main else 0.0,
        "islands": [{"vol_mm3": round(abs(i["volume"]) * unit_scale ** 3, 4),
                     "faces": i["faces"],
                     "zext_mm": round((i["zmax"] - i["zmin"]) * unit_scale, 3)}
                    for i in islands],
    })
    return info


def fmt(info):
    return ("{verts:,} verts | {tris:,} tris | {shells:,} shells | "
            "{boundary_edges:,} boundary edges | {nonmanifold_edges:,} "
            "non-manifold edges | {zero_area_tris:,} zero-area tris | "
            "watertight: {wt} | height {height_mm} mm").format(
                wt="YES" if info["watertight"] else "NO", **info)


# --------------------------------------------------------------------------
# Finish stages (extracted from finish_mini.py -- see header provenance)
# --------------------------------------------------------------------------

def weld_and_repair(ob):
    """remove_doubles weld -> delete zero-area faces -> fill holes.

    Returns the step counts. Weld-first ordering is deliberate (see the
    header): it is the same combo finish_mini runs, reordered for
    split-vertex input.
    """
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    before_verts = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=2e-6)
    welded = before_verts - len(bm.verts)

    bm.faces.ensure_lookup_table()
    zero = [f for f in bm.faces if f.calc_area() < 1e-12]
    if zero:
        bmesh.ops.delete(bm, geom=zero, context='FACES')

    bm.edges.ensure_lookup_table()
    bnd = [e for e in bm.edges if len(e.link_faces) == 1]
    if bnd:
        bmesh.ops.holes_fill(bm, edges=bnd, sides=0)

    bm.to_mesh(ob.data)
    bm.free()
    return {"verts_welded": welded, "zero_area_deleted": len(zero),
            "hole_edges_filled": len(bnd)}


def dedust(ob, dust_limit_scene):
    """Remove every island whose |volume| is below the dust limit.

    Union-find over bmesh vertices/edges; island volumes by the
    fan-triangulation method (finish_mini, incl. its 2026-11-21 fix).
    Returns (islands_found, islands_removed, verts_removed).
    """
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    parent = list(range(len(bm.verts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in bm.edges:
        ra, rb = find(e.verts[0].index), find(e.verts[1].index)
        if ra != rb:
            parent[ra] = rb

    island_vol = defaultdict(float)
    for f in bm.faces:
        vs = [v.co for v in f.verts]
        root = find(f.verts[0].index)
        for i in range(1, len(vs) - 1):
            island_vol[root] += vs[0].dot(vs[i].cross(vs[i + 1])) / 6.0

    drop = [v for v in bm.verts
            if abs(island_vol[find(v.index)]) < dust_limit_scene]
    removed_islands = len({find(v.index) for v in drop})
    if drop:
        bmesh.ops.delete(bm, geom=drop, context='VERTS')
    bm.to_mesh(ob.data)
    bm.free()
    return len(island_vol), removed_islands, len(drop)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    report = {"input": args.input, "output": args.output,
              "dust_mm3": args.dust_mm3, "dust_rel": args.dust_rel,
              "target_height_mm": args.target_height_mm,
              "unit_scale": args.unit_scale}
    t0 = time.time()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.input)
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        print("ERROR: no mesh objects found in input file", flush=True)
        sys.exit(2)
    report["input_objects"] = len(meshes)
    if len(meshes) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
    ob = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    report["before"] = audit_mesh(ob.data, args.unit_scale)
    print("BEFORE:  " + fmt(report["before"]), flush=True)

    # Stage 1: weld + repair.
    t = time.time()
    report["repair"] = weld_and_repair(ob)
    report["repair_secs"] = round(time.time() - t, 1)
    report["after_repair"] = audit_mesh(ob.data, args.unit_scale)
    print("REPAIRED: " + fmt(report["after_repair"]) +
          f"  ({report['repair_secs']}s, {report['repair']})", flush=True)

    # Stage 2: dedust. The limit is the LARGER of the absolute floor and
    # the relative ceiling, so a big model does not keep proportionally
    # huge "dust" and a small model does not eat real parts.
    main_vol_mm3 = (report["after_repair"]["islands"][0]["vol_mm3"]
                    if report["after_repair"]["islands"] else 0.0)
    dust_limit_mm3 = max(args.dust_mm3, args.dust_rel * main_vol_mm3)
    dust_limit_scene = dust_limit_mm3 / args.unit_scale ** 3
    report["dust_limit_mm3"] = round(dust_limit_mm3, 4)
    t = time.time()
    found, removed, verts_removed = dedust(ob, dust_limit_scene)
    report["dedust"] = {"islands_found": found, "islands_removed": removed,
                        "verts_removed": verts_removed}
    report["dedust_secs"] = round(time.time() - t, 1)
    report["after_dedust"] = audit_mesh(ob.data, args.unit_scale)
    print("DEDUSTED: " + fmt(report["after_dedust"]) +
          f"  ({report['dedust_secs']}s, removed {removed} islands)",
          flush=True)

    # Stage 3: optional sizing to a target height, grounded at z = 0.
    if args.target_height_mm:
        cur_h = report["after_dedust"]["height_mm"]
        if cur_h <= 0:
            print("ERROR: cannot size a mesh with no height", flush=True)
            sys.exit(2)
        factor = args.target_height_mm / cur_h
        from mathutils import Matrix, Vector
        ob.data.transform(Matrix.Diagonal(Vector((factor, factor, factor,
                                                  1.0))))
        zmin = min(v.co.z for v in ob.data.vertices)
        ob.data.transform(Matrix.Translation(Vector((0.0, 0.0, -zmin))))
        report["sizing"] = {"factor": round(factor, 6),
                            "grounded_zmin_mm": 0.0}
        print(f"SIZED:   uniform factor {factor:.6f}, grounded at z=0",
              flush=True)

    # Final audit + acceptance gate.
    report["final"] = audit_mesh(ob.data, args.unit_scale)
    print("FINAL:   " + fmt(report["final"]), flush=True)
    ref_height = report["after_repair"]["main_height_mm"]
    if args.target_height_mm:
        ref_height = args.target_height_mm
    tol = max(args.height_tol_mm, args.height_tol_pct / 100.0 * ref_height)
    height_drift = abs(report["final"]["height_mm"] - ref_height)
    gate = {
        "watertight": report["final"]["watertight"],
        "exactly_one_component": report["final"]["shells"] == 1,
        "height_ref_mm": round(ref_height, 3),
        "height_final_mm": report["final"]["height_mm"],
        "height_drift_mm": round(height_drift, 4),
        "height_tol_mm": round(tol, 4),
        "height_within_tol": height_drift <= tol,
    }
    gate["pass"] = all([gate["watertight"], gate["exactly_one_component"],
                        gate["height_within_tol"]])
    report["gate"] = gate
    print("GATE:    " + ("PASS" if gate["pass"] else "FAIL") +
          f"  (watertight={gate['watertight']}, "
          f"one_component={gate['exactly_one_component']}, "
          f"height drift {gate['height_drift_mm']} mm "
          f"vs tol {gate['height_tol_mm']} mm)", flush=True)

    # Export the STL in millimetres.
    import os
    if os.path.exists(args.output):
        os.remove(args.output)
    bpy.ops.object.select_all(action='DESELECT')
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.wm.stl_export(filepath=args.output,
                          export_selected_objects=True,
                          use_scene_unit=False,
                          global_scale=args.unit_scale)
    report["stl_out"] = args.output
    report["total_secs"] = round(time.time() - t0, 1)

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps(report["gate"], indent=1))
    print(f"DONE in {report['total_secs']}s -> {args.output} "
          f"(gate {'PASS' if gate['pass'] else 'FAIL'})", flush=True)
    sys.exit(0 if gate["pass"] else 1)


if __name__ == "__main__":
    main()

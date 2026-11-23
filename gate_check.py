#!/usr/bin/env python3
"""gate_check.py -- independently gate a finished STL: watertight,
exactly one component, and (with a reference) shape fidelity vs the
source mesh via roughness + voxel IoU.

This is the cross-check half of print-doctor: print_doctor.py gates
its output inside Blender (bmesh); this script re-checks the exported
STL with trimesh, a different library, so the two implementations have
to agree before a print is trusted. With --reference it also runs the
shape-fidelity comparison: sampled adjacent-face fold angles
(roughness) and voxelised intersection-over-union, global and per
octant, against the source.

The roughness and voxel-IoU method is adapted from gate_mini.py, the
acceptance gate of the 1915rising miniatures pipeline (2026-11-21),
where it gated re-voxelled miniatures against their shipped originals.
The self-gate (watertight / one component / height) is new here, and
the voxel bookkeeping was reworked for memory: gate_mini's 0.4 mm
pitch on a full-size figure produces a dense grid that OOM-killed
this script in testing (a 1.17 m figure at 2 mm pitch, killed after
97 s). Keys here are packed int64s in one shared grid space, point
clouds are dropped as soon as they are packed, and the pitch guidance
below is scaled to the model.

Picking a pitch: gate_mini's 0.4 mm suits miniatures (~150 mm). For
larger pieces keep the grid modest -- roughly 0.3% of the model's
height (a 1.17 m figure -> ~3.5 mm) keeps the dense voxel grid and the
filled-voxel sets comfortably in RAM on an ordinary machine.

Usage:
  gate_check.py <finished.stl> [--reference <source.glb|.stl>]
                [--pitch-mm 0.4] [--min-iou 0.98]
                [--ref-scale 1000] [--height-mm H] [--height-tol-mm T]

  --reference PATH   Source mesh to compare against. A .glb is assumed
                     to be in metres and is scaled by --ref-scale into
                     the STL's millimetres; an .stl reference is taken
                     as already in millimetres (ref-scale ignored).
  --min-iou F        Fidelity pass threshold on global voxel IoU
                     (default 0.98). Only applies with --reference.
  --height-mm F      Expected height; with --height-tol-mm (default
                     0.5) adds a height check on the finished STL.

Prints a JSON report. Exit codes: 0 = PASS, 1 = FAIL, 2 = usage error.

Copyright (c) 2026 1915rising -- MIT License (see LICENSE).
"""
import argparse
import json
import sys

import numpy as np
import trimesh


def load_mesh(path, scale=1.0, gltf_to_zup=False):
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = m.to_geometry() if hasattr(m, "to_geometry") \
            else m.dump(concatenate=True)
    if scale != 1.0 or gltf_to_zup:
        m = m.copy()
    if gltf_to_zup:
        # glTF is Y-up and trimesh loads it raw; Blender (and therefore
        # print_doctor's STL) is Z-up. +90 deg about X maps
        # (x, y, z) -> (x, -z, y): the conversion Blender's importer
        # applies. Without it the IoU below compares a standing figure
        # against a lying one and reads ~0 (measured: 0.0027).
        m.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    if scale != 1.0:
        m.apply_scale(scale)
    return m


def roughness(m, sample=120000, seed=7):
    """Sampled adjacent-face fold angles (from gate_mini.py)."""
    m = m.copy()
    m.merge_vertices()
    F = np.asarray(m.faces)
    edges = np.sort(np.concatenate(
        [F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]).astype(np.int64), axis=1)
    ue, einv, ec = np.unique(edges, axis=0, return_inverse=True,
                             return_counts=True)
    einv = einv.reshape(len(F), 3)
    fn = np.asarray(m.face_normals)
    ef = np.stack([einv.reshape(-1), np.repeat(np.arange(len(F)), 3)])
    order = np.argsort(ef[0], kind="stable")
    ef = ef[:, order]
    starts = np.searchsorted(ef[0], np.arange(len(ue)))
    pairs = []
    for ei in np.where(ec == 2)[0]:
        s = starts[ei]
        pairs.append((ef[1, s], ef[1, s + 1]))
    pairs = np.array(pairs)
    rng = np.random.default_rng(seed)
    if len(pairs) > sample:
        pairs = pairs[rng.choice(len(pairs), sample, replace=False)]
    dots = np.clip(np.einsum("ij,ij->i", fn[pairs[:, 0]], fn[pairs[:, 1]]),
                   -1, 1)
    angs = np.degrees(np.arccos(dots))
    return {"mean_fold_deg": round(float(angs.mean()), 2),
            "pct_gt45": round(float(np.mean(angs > 45) * 100), 2),
            "pct_gt80": round(float(np.mean(angs > 80) * 100), 2),
            "sampled_pairs": int(len(angs))}


def voxel_iou(new, old, pitch):
    """Global + per-octant voxel IoU (method from gate_mini.py).

    Both meshes' filled voxels are packed into int64 keys in ONE
    shared grid space (same origin, same cell dims), so the global
    sets and the per-octant sets are directly comparable.
    """
    def filled_grid(m):
        vg = m.voxelized(pitch)
        mat = vg.matrix
        idx = np.argwhere(mat)
        pts = (vg.transform[:3, :3] @ idx.T).T + vg.transform[:3, 3]
        return pts

    pn = filled_grid(new)
    po = filled_grid(old)
    lo = np.minimum(pn.min(axis=0), po.min(axis=0))
    hi = np.maximum(pn.max(axis=0), po.max(axis=0))
    dims = np.round((hi - lo) / pitch).astype(np.int64) + 1

    def pack(pts):
        ijk = np.round((pts - lo) / pitch).astype(np.int64)
        packed = (ijk[:, 0] * dims[1] * dims[2]
                  + ijk[:, 1] * dims[2] + ijk[:, 2])
        return ijk, packed

    ijk_n, pk_n = pack(pn)
    del pn
    ijk_o, pk_o = pack(po)
    del po

    sn, so = set(pk_n.tolist()), set(pk_o.tolist())
    iou = len(sn & so) / max(1, len(sn | so))
    del sn, so

    ctr = (old.bounds[0] + old.bounds[1]) / 2

    def oct_sets(ijk, packed):
        pts = ijk * pitch + lo
        lab = ((pts[:, 0] > ctr[0]).astype(int) * 4
               + (pts[:, 1] > ctr[1]).astype(int) * 2
               + (pts[:, 2] > ctr[2]).astype(int))
        d = {}
        for L, t in zip(lab, packed.tolist()):
            d.setdefault(int(L), set()).add(t)
        return d

    dn = oct_sets(ijk_n, pk_n)
    do = oct_sets(ijk_o, pk_o)
    oct_ious = {}
    for L in sorted(set(dn) | set(do)):
        a, b = dn.get(L, set()), do.get(L, set())
        oct_ious[L] = round(len(a & b) / max(1, len(a | b)), 4)
    return round(iou, 4), oct_ious


def main():
    p = argparse.ArgumentParser(prog="gate_check.py",
                                description="Gate a finished STL "
                                            "(trimesh cross-check).")
    p.add_argument("stl", help="the finished .stl to gate")
    p.add_argument("--reference", default=None,
                   help="source mesh (.glb in metres, or .stl in mm)")
    p.add_argument("--pitch-mm", type=float, default=0.4,
                   help="voxel pitch for the IoU comparison "
                        "(default 0.4; scale it to the model -- "
                        "see the header notes)")
    p.add_argument("--min-iou", type=float, default=0.98,
                   help="IoU pass threshold (default 0.98)")
    p.add_argument("--ref-scale", type=float, default=1000.0,
                   help="reference scale into mm for .glb references "
                        "(default 1000: metres -> mm)")
    p.add_argument("--height-mm", type=float, default=None,
                   help="expected height in mm")
    p.add_argument("--height-tol-mm", type=float, default=0.5,
                   help="height tolerance in mm (default 0.5)")
    args = p.parse_args()

    try:
        new = load_mesh(args.stl)
    except Exception as e:
        print(json.dumps({"error": f"cannot load {args.stl}: {e}"}))
        return 2

    # An STL on disk is triangle soup (3 private verts per face), so
    # topology checks must run on a vertex-merged copy -- with
    # process=False the raw load reports one "component" per triangle.
    merged = new.copy()
    merged.merge_vertices()
    bodies = merged.split(only_watertight=False)
    height_mm = float(new.bounds[1][2] - new.bounds[0][2])
    out = {"file": args.stl,
           "verts": int(len(merged.vertices)),
           "tris": int(len(new.faces)),
           "watertight": bool(merged.is_watertight),
           "components": int(len(bodies)),
           "height_mm": round(height_mm, 3)}
    checks = {"watertight": out["watertight"],
              "exactly_one_component": out["components"] == 1}

    if args.height_mm is not None:
        drift = abs(height_mm - args.height_mm)
        checks["height_within_tol"] = bool(drift <= args.height_tol_mm)
        out["height_expected_mm"] = args.height_mm
        out["height_drift_mm"] = round(drift, 4)

    if args.reference:
        try:
            is_gltf = args.reference.lower().endswith((".glb", ".gltf"))
            old = load_mesh(args.reference,
                            scale=args.ref_scale if is_gltf else 1.0,
                            gltf_to_zup=is_gltf)
        except Exception as e:
            print(json.dumps(
                {"error": f"cannot load reference {args.reference}: {e}"}))
            return 2
        out["reference"] = args.reference
        out["roughness_new"] = roughness(new)
        out["roughness_ref"] = roughness(old)
        iou, oct_ious = voxel_iou(new, old, args.pitch_mm)
        out["iou"] = iou
        out["octant_iou"] = oct_ious
        out["worst_octant_iou"] = (min(oct_ious.values())
                                   if oct_ious else None)
        checks["iou_above_min"] = bool(iou >= args.min_iou)

    out["checks"] = checks
    out["pass"] = all(checks.values())
    print(json.dumps(out, indent=1))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""End-to-end test for print-doctor on a real cleaned generator mesh.

Usage:
  BLENDER=/path/to/blender python3 tests/run_tests.py [cleaned.glb]

The input defaults to $PRINT_DOCTOR_TEST_GLB. Use a GLB that came out
of ai-mesh-cleanup (or a similar voxel/decimate pass): geometry-only,
in metres, ideally one that still carries detached floater shells --
that is the defect this tool exists to remove.

Asserts, in order:
  1. print_doctor exits 0 (its internal gate passes).
  2. The report shows shells going to exactly 1, watertight at the end,
     and a height drift within tolerance.
  3. gate_check (self-gate mode) exits 0 and agrees on verts / tris /
     watertight / components / height.
  4. gate_check with the input as --reference reports voxel IoU >= 0.98
     (the finisher may remove dust, nothing else).
  5. A --target-height-mm 150 run lands within 0.5 mm of 150 mm,
     watertight, one component (checked by gate_check).

The reference IoU run is the slow one (~2 min on a life-size figure at
the default-scaled pitch); the rest takes seconds.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BLENDER = os.environ.get("BLENDER", "blender")
PY = sys.executable

PASS = []
FAIL = []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    src = (sys.argv[1] if len(sys.argv) > 1
           else os.environ.get("PRINT_DOCTOR_TEST_GLB"))
    if not src or not os.path.exists(src):
        print("usage: run_tests.py <cleaned.glb>  (or set "
              "PRINT_DOCTOR_TEST_GLB)")
        return 2
    tmp = tempfile.mkdtemp(prefix="print-doctor-test-")
    stl = os.path.join(tmp, "finished.stl")
    rep = os.path.join(tmp, "report.json")

    r = run([BLENDER, "-b", "--factory-startup", "--python",
             os.path.join(ROOT, "print_doctor.py"), "--",
             "--input", src, "--output", stl, "--report", rep])
    check("print_doctor exit 0 (internal gate PASS)", r.returncode == 0,
          f"(exit {r.returncode})")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        return 1
    report = json.load(open(rep))
    fin = report["final"]
    check("final shells == 1", fin["shells"] == 1,
          f"(shells {report['before']['shells']} -> "
          f"{report['after_repair']['shells']} -> {fin['shells']})")
    check("final watertight", fin["watertight"])
    check("internal gate verdict PASS", report["gate"]["pass"],
          f"(drift {report['gate']['height_drift_mm']} mm)")

    g = run([PY, os.path.join(ROOT, "gate_check.py"), stl,
             "--height-mm", str(fin["height_mm"]),
             "--height-tol-mm", "0.5"])
    gate = json.loads(g.stdout)
    check("gate_check self-gate exit 0", g.returncode == 0)
    check("gate agrees on verts/tris",
          gate["verts"] == fin["verts"] and gate["tris"] == fin["tris"],
          f"(blender {fin['verts']}/{fin['tris']} vs "
          f"trimesh {gate['verts']}/{gate['tris']})")
    check("gate agrees watertight + 1 component",
          gate["watertight"] and gate["components"] == 1)

    pitch = max(0.4, round(fin["height_mm"] * 0.003, 2))
    g2 = run([PY, os.path.join(ROOT, "gate_check.py"), stl,
              "--reference", src, "--pitch-mm", str(pitch)])
    gate2 = json.loads(g2.stdout)
    check("reference IoU >= 0.98", gate2.get("iou", 0) >= 0.98,
          f"(IoU {gate2.get('iou')}, worst octant "
          f"{gate2.get('worst_octant_iou')}, pitch {pitch} mm)")

    stl150 = os.path.join(tmp, "finished_150mm.stl")
    rep150 = os.path.join(tmp, "report_150.json")
    r5 = run([BLENDER, "-b", "--factory-startup", "--python",
              os.path.join(ROOT, "print_doctor.py"), "--",
              "--input", src, "--output", stl150, "--report", rep150,
              "--target-height-mm", "150"])
    check("sized run exit 0", r5.returncode == 0)
    g5 = run([PY, os.path.join(ROOT, "gate_check.py"), stl150,
              "--height-mm", "150", "--height-tol-mm", "0.5"])
    gate5 = json.loads(g5.stdout)
    check("sized STL gates at 150 mm",
          g5.returncode == 0 and abs(gate5["height_mm"] - 150) <= 0.5,
          f"(height {gate5['height_mm']} mm)")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

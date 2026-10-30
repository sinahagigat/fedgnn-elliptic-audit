"""
Collect the per-run validation and test scores (predictions/*.npz) of every
job into one ZIP for the repository release.

    python tools/pack_predictions.py --out predictions_v7.zip RESULTS [RESULTS3 RESULTS4 ...]

Each RESULTS folder holds job folders (A1, C5_1, P3_2, ...) with a
predictions/ subfolder. Files are stored as <job>/<variant>__seed<k>.npz.
"""
import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jobs import JOBS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    found, n = set(), 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:
        for root in map(Path, args.roots):
            for job in sorted(JOBS):
                d = root / job / "predictions"
                if not d.is_dir() or job in found:
                    continue
                found.add(job)
                for f in sorted(d.glob("*.npz")):
                    z.write(f, f"{job}/{f.name}"); n += 1
    missing = sorted(set(JOBS) - found)
    print(f"{n} score files from {len(found)} jobs written to {args.out}")
    if missing:
        print(f"jobs without predictions in the given folders ({len(missing)}): {', '.join(missing[:40])}"
              + (" ..." if len(missing) > 40 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

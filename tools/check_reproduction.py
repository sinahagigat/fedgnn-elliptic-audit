"""
Recompute every number and table of the article from the released run
records and compare them with the committed analysis/output/numbers.json and
tables.json.

    python tools/check_reproduction.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "output"


def main():
    tmp = Path(tempfile.mkdtemp())
    for f in ("numbers.json", "tables.json"):
        shutil.copy2(OUT / f, tmp / f)
    print("recomputing all numbers, tables and figures (a few minutes) ...", flush=True)
    rc = subprocess.call([sys.executable, str(ROOT / "analysis" / "build_numbers.py")], cwd=str(ROOT),
                         stdout=subprocess.DEVNULL)
    if rc != 0:
        print("build_numbers.py failed; restoring the committed files")
        for f in ("numbers.json", "tables.json"):
            shutil.copy2(tmp / f, OUT / f)
        return 1
    bad = 0
    for f in ("numbers.json", "tables.json"):
        a = json.loads((tmp / f).read_text(encoding="utf-8")); b = json.loads((OUT / f).read_text(encoding="utf-8"))
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        print(f"{f}: {len(a)} entries, {len(diff)} different", *(f"\n   {k}: {a.get(k)!r} -> {b.get(k)!r}" for k in diff[:20]))
        bad += len(diff)
    print("\nRESULT:", "PASS (every number and table reproduced exactly)" if bad == 0 else "DIFFERENCES FOUND")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

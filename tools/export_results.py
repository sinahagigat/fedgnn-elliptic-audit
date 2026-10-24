"""
Build a small zip of all finished results for sending back for analysis.

    python tools/export_results.py /content/drive/MyDrive/FedGNN_v7/results

Writes <results>/export/results_slim.zip containing, per job, the run
records without the per-round history (all final metrics are kept), plus the
merge report. Score files (predictions/*.npz) are left out to keep the zip
small; they stay in Drive and can be sent separately if needed.
"""
import io
import json
import sys
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from jobs import JOBS  # noqa: E402
import merge_results  # noqa: E402


def main():
    root = Path(sys.argv[1])
    out_dir = root / "export"
    out_dir.mkdir(exist_ok=True)
    buf = io.StringIO()
    sys.argv = ["merge_results.py", str(root)]
    with redirect_stdout(buf):
        merge_results.main()
    report = buf.getvalue()
    zpath = out_dir / "results_slim.zip"
    n_runs = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("merge_report.txt", report)
        for job in JOBS:
            f = root / job / "artifacts.json"
            if not f.is_file():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            for r in d.get("runs", {}).values():
                r.pop("history", None)
                n_runs += 1
            z.writestr(f"{job}/artifacts_slim.json", json.dumps(d, ensure_ascii=False))
    print(report)
    print(f"Exported {n_runs} runs to {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

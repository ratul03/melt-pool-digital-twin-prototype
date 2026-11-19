"""Start the digital twin prototype (Sections 4.2 and 6.1) in a web browser.

Usage
    python run_prototype.py              # then open http://127.0.0.1:8766
    python run_prototype.py --port 8800

The prototype runs on this computer only. It reads the frozen models and data of this package.
Its code expects the folder layout of the original studies, so this script copies the files it
needs into a temporary folder with that layout and points the prototype at it. Saved sessions
are written to digital_twin_prototype/saved_sessions.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

PKG = Path(__file__).resolve().parent


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_runtime():
    rt = Path(tempfile.mkdtemp(prefix="digital_twin_prototype_"))
    w12 = rt / "Thesis_C_Weeks_1_2"
    for f in (PKG / "data/prepared").glob("*.csv"):
        copy(f, w12 / "data" / f.name)
    for f in (PKG / "code/benchmark/inputs_code").iterdir():
        copy(f, w12 / "inputs/code" / f.name)
    for f in (PKG / "data/raw").glob("*.csv"):
        copy(f, w12 / "inputs/data/raw" / f.name)
    for f in (PKG / "models").iterdir():
        copy(f, w12 / "models" / f.name)
    p = PKG / "data/protocols"
    copy(p / "benchmark_protocol.json", w12 / "protocol/frozen.json")
    copy(p / "benchmark_splits.json", w12 / "protocol/splits.json")
    copy(p / "calibration_splits.json", w12 / "protocol/future_calibration_splits.json")
    copy(
        PKG / "digital_twin_prototype/prototype/workflow.py",
        rt / "Thesis_C_Weeks_3_4/prototype/workflow.py",
    )
    rel = rt / "Thesis_C_Final_Results"
    copy(PKG / "results/earlier_design/results/summary.csv", rel / "results/summary.csv")
    for f in (PKG / "results/earlier_design/figures").glob("*.png"):
        copy(f, rel / "figures" / f.name)
    copy(
        PKG / "results/chapter6_main_design/main_design_runs/summary.csv",
        rt / "chapter6/results/summary.csv",
    )
    return rt, rel, rt / "chapter6"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    rt, rel, ch6 = build_runtime()
    (PKG / "digital_twin_prototype/saved_sessions").mkdir(exist_ok=True)
    env = dict(os.environ, THESIS_RELEASE_ROOT=str(rel), THESIS_CH6_ROOT=str(ch6))
    url = f"http://127.0.0.1:{a.port}"
    if not a.no_browser:
        threading.Timer(4, lambda: webbrowser.open(url)).start()
    print("Starting the prototype at", url, "(press Ctrl+C to stop)")
    try:
        subprocess.run(
            [sys.executable, "server.py", "--port", str(a.port)],
            cwd=PKG / "digital_twin_prototype/prototype",
            env=env,
        )
    except KeyboardInterrupt:
        pass
    finally:
        shutil.rmtree(rt, ignore_errors=True)


if __name__ == "__main__":
    main()

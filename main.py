"""
Main Launcher for KMUTT Sign Language Translation System.
Provides a central entrypoint to run the English Dual-Pane GUI,
or execute data collection and training scripts.

Automatically detects and switches to the project's virtual environment (.venv)
if launched from global Python.

Usage:
    python main.py               # Launches the full Desktop GUI Application
    python main.py --app         # Launches the full Desktop GUI Application
    python main.py --realtime    # Launches headless OpenCV real-time recognition
    python main.py --collect     # Launches data collector for gestures
    python main.py --train       # Trains Random Forest and PyTorch GRU models
    python main.py --check       # Runs camera & posture diagnostic check
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Auto-Venv Detection & Seamless Re-Routing
# If launched with global Python (e.g. Python 3.14 without packages),
# automatically re-route execution into the project's .venv environment.
# --------------------------------------------------------------------------
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if VENV_PYTHON.exists():
    try:
        curr_py = Path(sys.executable).resolve()
        target_py = VENV_PYTHON.resolve()
        if curr_py != target_py:
            cmd = [str(target_py)] + sys.argv
            sys.exit(subprocess.call(cmd))
    except Exception as _e:
        pass

SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PYTHON = sys.executable


def launch_gui():
    print("[Launcher] Starting TSL Assist Desktop Application...")
    from src.ui_app import SignLanguageApp
    app = SignLanguageApp()
    app.mainloop()


def launch_script(script_name, extra_args=None):
    script_path = ROOT / "src" / script_name
    cmd = [PYTHON, str(script_path)] + (extra_args or [])
    print(f"[Launcher] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[Launcher] Script '{script_name}' exited with error code {res.returncode}")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="KMUTT Sign Language Translation System Launcher")
    parser.add_argument("--app", action="store_true", help="Launch the Desktop GUI Application (Default)")
    parser.add_argument("--realtime", action="store_true", help="Launch standalone OpenCV realtime window")
    parser.add_argument("--collect", action="store_true", help="Launch interactive sign data collection")
    parser.add_argument("--train", action="store_true", help="Train baseline & PyTorch models")
    parser.add_argument("--check", action="store_true", help="Check camera, FPS, and shoulder distance")
    parser.add_argument("--words", nargs="+", default=None, help="Gesture words for collection (e.g. --words hello thanks)")

    args, unknown = parser.parse_known_args()

    if args.realtime:
        launch_script("realtime.py", unknown)
    elif args.collect:
        collect_args = []
        if args.words:
            collect_args = ["--words"] + args.words
        launch_script("collect_data.py", collect_args + unknown)
    elif args.train:
        print("\n--- 1. Training Random Forest Baseline ---")
        launch_script("train_sklearn.py")
        print("\n--- 2. Training PyTorch GRU Sequence Model ---")
        launch_script("train_torch.py")
    elif args.check:
        launch_script("check_camera.py", unknown)
    else:
        # Default: Launch GUI App
        launch_gui()


if __name__ == "__main__":
    main()

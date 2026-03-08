#!/usr/bin/env python3
"""
main.py -- Entrypoint for the Signal-in-Noise pipeline.

Usage:
    python main.py pipeline    # Run end-to-end data pipeline
    python main.py dashboard   # Launch Streamlit dashboard
    python main.py install     # Install dependencies
"""
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_script(script_name: str):
    """Run a script from the scripts/ directory."""
    path = os.path.join(PROJECT_ROOT, "scripts", script_name)
    result = subprocess.run([sys.executable, path], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[WARN] {script_name} exited with code {result.returncode}")


def cmd_install():
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    subprocess.run([sys.executable, "-m", "nltk.downloader", "vader_lexicon", "punkt_tab", "stopwords"])


def cmd_pipeline():
    print("\n" + "#" * 70)
    print("  SIGNAL-IN-NOISE: SEC 8-K EVENT DETECTION PIPELINE")
    print("#" * 70 + "\n")

    steps = [
        "01_download_filings.py",
        "02_parse_chunk.py",
        "03_extract_features.py",
        "04_build_index.py",
        "05_train_models.py",
        "06_score_decide.py",
        "07_evaluate.py",
        "08_run_agents.py",
    ]

    for script in steps:
        run_script(script)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("  Run: python main.py dashboard")
    print("=" * 70)


def cmd_dashboard():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        os.path.join(PROJECT_ROOT, "dashboard", "app.py"),
        "--server.headless", "true",
    ])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()
    if cmd == "pipeline":
        cmd_pipeline()
    elif cmd == "dashboard":
        cmd_dashboard()
    elif cmd == "install":
        cmd_install()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()

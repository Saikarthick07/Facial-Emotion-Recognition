#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v python3.10 >/dev/null 2>&1; then
  echo "[INFO] python3.10 not found. Install with: brew install python@3.10"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "[INFO] Creating virtual environment (.venv) with python3.10"
  python3.10 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[INFO] Upgrading pip tooling"
pip install --upgrade pip setuptools wheel

echo "[INFO] Installing project requirements"
pip install -r "requirements (1).txt"

echo "[INFO] Installing web app runtime dependencies"
pip install flask pillow

echo "[INFO] Installing desktop OpenCV for local development"
pip install opencv-python

echo "[INFO] Verifying key imports"
python - <<'PY'
import cv2
import flask
import PIL
print('cv2:', cv2.__version__)
print('flask:', flask.__version__)
print('Pillow:', PIL.__version__)
PY

echo "[INFO] Starting app at http://localhost:5000"
python web_app.py

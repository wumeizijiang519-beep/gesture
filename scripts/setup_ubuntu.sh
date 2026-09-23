#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /etc/os-release
if [[ "$ID" != ubuntu || "$VERSION_ID" != 22.04 || "$(uname -m)" != x86_64 ]]; then
  echo "Reference platform is Ubuntu 22.04 x86_64. See docs/REPRODUCIBILITY.md." >&2
  exit 2
fi
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv git libgl1 libglib2.0-0 libportaudio2
python3.10 -m venv .venv
PY=.venv/bin/python
"$PY" -m pip install pip==25.0.1 setuptools==75.8.0 wheel==0.45.1
if [[ -f requirements.lock ]]; then
  "$PY" -m pip install -r requirements.lock
  "$PY" -m pip install --no-deps --no-build-isolation -e .
else
  "$PY" -m pip install -e '.[vision,dev]'
fi
"$PY" -m pip check
"$PY" -m gesture download-model
"$PY" -m gesture doctor
printf '\nInstalled. Next: source .venv/bin/activate && bash scripts/verify_all.sh\n'

#!/bin/bash
# SinglePTZ-FaceTrack launcher
# Usage: ./run.sh

set -e

cd "$(dirname "$0")"

# SDK library path
export LD_LIBRARY_PATH="$(pwd)/HCNetSDKV6.1.11.5_build20251204_linux64_ZH_20260320152102/HCNetSDKV6.1.11.5_build20251204_linux64_ZH/库文件:${LD_LIBRARY_PATH:-}"

# Activate conda environment if needed
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "single_ptz_facetrack" ]; then
    echo "Activating conda environment: single_ptz_facetrack"
    eval "$(conda shell.bash hook)"
    conda activate single_ptz_facetrack
fi

echo "============================================"
echo "SinglePTZ-FaceTrack"
echo "============================================"
echo ""
echo "Web stream: http://0.0.0.0:18080/"
echo "Commands: q=quit r=reset h=home p=pause v=record"
echo ""

python src/main.py "$@"

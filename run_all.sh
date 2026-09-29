#!/usr/bin/env bash
# Lumido: runs the whole Lighthouse reconstruction pipeline, step by step.
# Usage:  ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

$PY pipeline/step1_prepare.py   # resize images, read GPS + focal length
$PY pipeline/step2_match.py     # SIFT features + verified matches
$PY pipeline/step3_sfm.py       # camera poses (Structure-from-Motion) + GPS scale
$PY pipeline/step4_depth.py     # plane-sweep depth maps
$PY pipeline/step5_fuse.py      # multi-view fusion -> outputs/dense_cloud.ply
$PY pipeline/step6_render.py    # screenshots of the result

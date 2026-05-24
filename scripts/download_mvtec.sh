#!/usr/bin/env bash
# Download a small subset of MVTec AD for the demo.
#
# Full MVTec AD is ~5GB. We only need a couple of categories for a working demo.
# The official dataset is hosted by MVTec and requires accepting their license:
#   https://www.mvtec.com/company/research/datasets/mvtec-ad
#
# This script tries to download the "bottle" category as a smoke test.
# If the URL changes, manually grab any category and unpack it under data/mvtec/.

set -euo pipefail

DATA_DIR="${DATA_DIR:-./data/mvtec}"
mkdir -p "$DATA_DIR"

BOTTLE_URL="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951595/bottle.tar.xz"

cd "$DATA_DIR"
if [ ! -d "bottle" ]; then
    echo "[download] fetching bottle category (~80MB)..."
    curl -fL -o bottle.tar.xz "$BOTTLE_URL" || {
        echo "[download] failed — please download manually from:"
        echo "  https://www.mvtec.com/company/research/datasets/mvtec-ad"
        echo "and unpack into $DATA_DIR/<category>/"
        exit 1
    }
    tar -xf bottle.tar.xz
    rm bottle.tar.xz
fi
echo "[download] done. Layout:"
ls -la "$DATA_DIR"

#!/usr/bin/env bash
# Download Spider (and optionally BIRD) into $T2S_DATA_ROOT. We do NOT redistribute either dataset;
# this script fetches them from source, so you accept their licenses by running it.
#
# Spider:  https://yale-lily.github.io/spider   (CC BY-SA 4.0)
# BIRD:    https://bird-bench.github.io/         (CC BY-NC 4.0, research use)
#
# T2S_DATA_ROOT is the folder the code reads directly (train_spider.json lives *inside* it).
# Defaults to data/spider, matching setup/README.md.
set -euo pipefail

DATA_ROOT="${T2S_DATA_ROOT:-data/spider}"
mkdir -p "$DATA_ROOT"

echo "==> Spider"
if [ ! -f "$DATA_ROOT/train_spider.json" ]; then
  # download + unzip only if the files aren't already extracted somewhere under DATA_ROOT
  if ! find "$DATA_ROOT" -name train_spider.json 2>/dev/null | grep -q .; then
    # a uv venv has no `pip`, so pick whatever installer is available
    if command -v pip >/dev/null 2>&1; then INSTALL="pip install --quiet";
    elif python -m pip --version >/dev/null 2>&1; then INSTALL="python -m pip install --quiet";
    elif command -v uv >/dev/null 2>&1; then INSTALL="uv pip install --quiet";
    else echo "need pip or uv on PATH to install gdown"; exit 1; fi
    $INSTALL gdown
    echo "    downloading (~1GB from Google Drive)"
    # recent gdown takes the file id positionally (--id was removed)
    gdown 1TqleXec_OykOYFREKKtschzY29dUcVAQ -O "$DATA_ROOT/spider.zip"
    unzip -q "$DATA_ROOT/spider.zip" -d "$DATA_ROOT"
    rm -f "$DATA_ROOT/spider.zip"
  fi
  # normalize layout: the archive nests everything under its own top folder (spider/ or
  # spider_data/), so lift the dir that actually contains train_spider.json up into DATA_ROOT.
  found="$(find "$DATA_ROOT" -name train_spider.json 2>/dev/null | head -1 || true)"
  if [ -z "$found" ]; then
    echo "ERROR: train_spider.json not found under $DATA_ROOT after download"; exit 1
  fi
  src="$(dirname "$found")"
  if [ "$src" != "$DATA_ROOT" ]; then
    echo "    normalizing layout: $src -> $DATA_ROOT"
    ( shopt -s dotglob; mv "$src"/* "$DATA_ROOT"/ )
    rmdir "$src" 2>/dev/null || true
  fi
else
  echo "    already present, skipping"
fi

echo "==> Sanity check"
python - <<'PY'
import os
from common.data import load_split
train = load_split("train")
dev = load_split("dev")
print(f"    train={len(train)} dev={len(dev)} examples")
print(f"    example db_path exists: {os.path.exists(train[0].db_path)}")
PY

echo "==> Building the FROZEN dev subset (300 examples, seed=0)"
python -m common.data build

echo "Done. common/dev_set.json is built deterministically (seed=0), so this exact split"
echo "regenerates for anyone who runs this script. Committing it is optional (nice for readers,"
echo "not required for your own runs)."
echo "(BIRD is optional and larger; see setup/README.md if you want the harder held-out slice.)"

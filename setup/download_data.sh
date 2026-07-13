#!/usr/bin/env bash
# Download Spider (and optionally BIRD) into ./data. We do NOT redistribute either dataset; this
# script fetches them from source, so you accept their licenses by running it.
#
# Spider:  https://yale-lily.github.io/spider   (CC BY-SA 4.0)
# BIRD:    https://bird-bench.github.io/         (CC BY-NC 4.0, research use)
set -euo pipefail

DATA_ROOT="${T2S_DATA_ROOT:-data}"
mkdir -p "$DATA_ROOT"

echo "==> Spider"
if [ ! -d "$DATA_ROOT/spider" ]; then
  # Spider is distributed as a Google Drive zip; gdown handles the confirm-token dance.
  # A uv venv has no `pip`, so pick whatever installer is available.
  if command -v pip >/dev/null 2>&1; then INSTALL="pip install --quiet";
  elif python -m pip --version >/dev/null 2>&1; then INSTALL="python -m pip install --quiet";
  elif command -v uv >/dev/null 2>&1; then INSTALL="uv pip install --quiet";
  else echo "need pip or uv on PATH to install gdown"; exit 1; fi
  $INSTALL gdown
  # File id from the Spider website's download link. Update if the maintainers rotate it.
  gdown --id 1TqleXec_OykOYFREKKtschzY29dUcVAQ -O "$DATA_ROOT/spider.zip"
  unzip -q "$DATA_ROOT/spider.zip" -d "$DATA_ROOT"
  # the archive extracts to spider/; normalize just in case
  [ -d "$DATA_ROOT/spider_data" ] && mv "$DATA_ROOT/spider_data" "$DATA_ROOT/spider"
  rm -f "$DATA_ROOT/spider.zip"
else
  echo "    already present, skipping"
fi

echo "==> Sanity check"
python3 - <<'PY'
import os
from common.data import load_split
os.environ.setdefault("T2S_DATA_ROOT", "data/spider")
train = load_split("train")
dev = load_split("dev")
print(f"    train={len(train)} dev={len(dev)} examples")
print(f"    example db_path exists: {os.path.exists(train[0].db_path)}")
PY

echo "==> Building the FROZEN dev subset (300 examples, seed=0)"
python3 -m common.data build

echo "Done. Remember to commit common/dev_set.json so the dev split is frozen for everyone."
echo "(BIRD is optional and larger; see setup/README.md if you want the harder held-out slice.)"

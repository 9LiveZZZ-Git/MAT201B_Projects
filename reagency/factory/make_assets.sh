#!/bin/bash
# World of Shadow Work — build all runtime data assets from the corpus.
# Run on the CUDA box (RTX 4060 Ti). Emits ../assets/{points,edges}.bin + labels + manifest + atlas.
#
#   bash make_assets.sh
#
# Prereqs:  pip install torch --index-url https://download.pytorch.org/whl/cu121
#           pip install -r requirements.txt
#           corpus already fetched (python3 fetch_corpus.py)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

echo "== stage A: CLIP embedding (pretrained, inference only) =="
python3 stage_a_embed.py "$@"

echo "== stage B: UMAP layout + kNN webs + clusters + atlas =="
python3 stage_b_layout.py

echo "== done. assets in ../assets/ =="
ls -lh ../assets 2>/dev/null || true

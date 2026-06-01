#!/usr/bin/env bash
# make_dome_bundle.sh -- package the runtime-required assets that are NOT in git into ONE
# archive to side-load onto every AlloSphere node, alongside a `git pull`. Matches the WOSW
# "copy the finished assets to the nodes, never git-push from the factory" convention.
#
# git ALREADY carries the committed core (points.bin, edges.bin, atlas_0.png, dream_atlas.png,
# the v2 text atlases + index/pos files, ~192 instrument WAVs). This bundle carries ONLY the
# three gitignored RUNTIME assets that a clone is missing:
#     assets/vessel.wswv         (~58 MB, the morphing splat vessel)
#     v2/emergence_atlas.png     (~161 MB, the "watch it think" frames; over GitHub's hard cap)
#     assets/audio/voice_*.wav   (~44 MB, the THEM ghost-voice samples; Mac-only to re-bake)
# (assets/dreams/ raw PNGs are deliberately EXCLUDED -- the runtime loads the baked atlas, not them.)
#
# Run this on the Mac AFTER baking those assets:
#     bake_vessel_replicate.py + stage_d_vessel.py   -> assets/vessel.wswv
#     v2/bake_emergence.py                            -> v2/emergence_atlas.png
#     v2/bake_voices.py                               -> assets/audio/voice_*.wav
# Then on EACH dome node (after `git pull` of the committed core):
#     tar xzf dome_assets.tgz -C /path/to/MAT201B_Projects/reagency
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"      # reagency/factory
REA="$(dirname "$HERE")"                    # reagency
cd "$REA"
OUT="${1:-dome_assets.tgz}"

FILES=()
add() { if compgen -G "$1" >/dev/null; then FILES+=( $1 ); else echo "  (skip, absent) $1"; fi; }
add "assets/vessel.wswv"
add "v2/emergence_atlas.png"
add "assets/audio/voice_*.wav"

if [ ${#FILES[@]} -eq 0 ]; then
  echo "nothing to bundle -- bake the assets first (see header)."; exit 1
fi
echo "bundling ${#FILES[@]} file(s) (paths relative to reagency/) -> $OUT"
tar czf "$OUT" "${FILES[@]}"
echo "done: $(du -h "$OUT" | cut -f1)   $REA/$OUT"
echo "verify identical across nodes:  md5 $OUT   (or md5sum on Linux)"
echo "unpack on each node:            tar xzf $OUT -C /path/to/MAT201B_Projects/reagency"

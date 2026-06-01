#!/usr/bin/env python3
# bake_dreams.py -- World of Shadow Work v2 (Phase-2, the dream slice). ALL-192 build.
#
# The Qwen run produced 192 dreams = 64 source artworks x 3 ops (melt/outpaint/shadow). Only the
# FIRST 12 sources (corpus nodes 0..11) were concatenated into the galaxy as real type=2 nodes
# (50661..50696) with true CLIP/UMAP positions. This bake places ALL 192 dreams without a GPU
# re-bake:
#   * existing 36 (sources 0..11): KEEP their true points.bin positions (nodes 50661..50696).
#   * new 156 (sources 12..63): position AT the source artwork's galaxy node (sorted-glob index ->
#     points.bin), with a small deterministic per-op offset so the three variants of one source form
#     a tight cluster instead of stacking. Synthetic node ids 60000.. (NOT galaxy nodes; used only as
#     stable keys for dream_pos.txt + emergence_index.txt, both decoupled from points.bin types).
#
# Outputs (reagency/v2/): dream_atlas.png (grid, slot=node order) + dream_pos.txt (node + world pos +
# atlas cell + op). Also REPAIRS assets/dream_provenance.json into the 192-entry runtime stub
# (node -> dream_id + source + instruction + emergence_frames), which bake_emergence.py then reads.
import os, json, struct, glob, hashlib, array, math
from PIL import Image

HERE   = os.path.dirname(os.path.abspath(__file__))        # reagency/v2
ROOT   = os.path.dirname(HERE)                              # reagency
ASSETS = os.path.join(ROOT, "assets")
DREAMS = os.path.join(ASSETS, "dreams")

CELL, COLS = 384, 14                                       # 192 dreams -> 14x14 grid @384 = 5376x5376
OP_ORDER = {"melt": 0, "outpaint": 1, "shadow": 2}         # canonical op index (matches the orig 36 layout)
OP_MAG   = {0: 0.12, 1: 0.32, 2: 0.55}                     # per-op standoff for NEW dreams (cluster, no stack)

N_REAL_SOURCES = 12                                        # sources 0..11 are real galaxy dream-nodes
REAL_BASE = 50661                                          # first real dream node
SYN_BASE  = 60000                                          # synthetic node ids for the new 156

def op_of(entry):
    t = entry.get("op", entry.get("theme", ""))            # "dream-melt" / "melt" / ...
    for k, v in OP_ORDER.items():
        if k in t:
            return v
    return 0

def unit_from_id(did):
    """Deterministic unit vector from the dream_id (stable across runs, no RNG state)."""
    h = hashlib.md5(did.encode()).digest()
    v = [(h[i] / 255.0) * 2.0 - 1.0 for i in range(3)]
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return (v[0] / n, v[1] / n, v[2] / n)

# --- corpus filename -> galaxy node index (the stage_a_embed.py:59 sorted-glob ordering) ---
imgs = sorted(glob.glob(os.path.join(ROOT, "factory", "corpus", "images", "**", "*.jpg"), recursive=True))
name2node = {os.path.basename(p): i for i, p in enumerate(imgs)}

# --- rich provenance keyed by dream_id (source / instruction / emergence_frames / theme) ---
rich = json.load(open(os.path.join(DREAMS, "provenance.json")))

# --- positions from points.bin (WSWP: 16-byte header, then N*10 float32) ---
with open(os.path.join(ASSETS, "points.bin"), "rb") as f:
    f.read(4); ver, N, stride = struct.unpack("<iii", f.read(12))
    pts = array.array("f"); pts.frombytes(f.read(N * stride * 4))
def pos(idx): return (pts[idx * stride], pts[idx * stride + 1], pts[idx * stride + 2])

# --- assign every dream a node id + world position, deterministically ---
records = []     # (did, node, x, y, z, op)
missing_src = []
for did, e in rich.items():
    src = e.get("source", "")                              # aic_NNNNN.jpg
    if src not in name2node:
        missing_src.append(did); continue
    sn = name2node[src]                                    # corpus/galaxy node 0..63
    op = op_of(e)
    if sn < N_REAL_SOURCES:                                # existing 36: true galaxy node + true pos
        node = REAL_BASE + sn * 3 + op
        x, y, z = pos(node)
    else:                                                  # new 156: source-node pos + per-op offset
        node = SYN_BASE + (sn - N_REAL_SOURCES) * 3 + op
        bx, by, bz = pos(sn)
        ux, uy, uz = unit_from_id(did); m = OP_MAG[op]
        x, y, z = bx + ux * m, by + uy * m, bz + uz * m
    records.append((did, node, x, y, z, op))

records.sort(key=lambda r: r[1])                           # atlas slot = ascending node order (36 real, then new)

# --- pack the atlas + write dream_pos.txt + the 192-entry enriched runtime stub ---
rows_n = (len(records) + COLS - 1) // COLS
atlas = Image.new("RGBA", (CELL * COLS, CELL * rows_n), (0, 0, 0, 0))
posrows = []
enriched = {}
missing_png = 0
for slot, (did, node, x, y, z, op) in enumerate(records):
    col, row = slot % COLS, slot // COLS
    png = os.path.join(DREAMS, did + ".png")
    if os.path.exists(png):
        im = Image.open(png).convert("RGBA").resize((CELL, CELL), Image.LANCZOS)
        atlas.paste(im, (col * CELL, row * CELL))
    else:
        missing_png += 1
    r = rich[did]
    enriched[str(node)] = {
        "dream_id": did, "op": r.get("theme", r.get("op", "")), "theme": r.get("theme", ""),
        "model": "qwen", "source": r.get("source", ""), "instruction": r.get("instruction", ""),
        "emergence_frames": r.get("emergence_frames", []),
    }
    posrows.append((did, node, x, y, z, col, row, op))

atlas.save(os.path.join(HERE, "dream_atlas.png"))
with open(os.path.join(HERE, "dream_pos.txt"), "w") as f:
    f.write("# dream_id node x y z col row op  (op 0=melt 1=outpaint 2=shadow; CELL=%d COLS=%d)\n" % (CELL, COLS))
    for did, node, x, y, z, col, row, op in posrows:
        f.write("%s %d %.5f %.5f %.5f %d %d %d\n" % (did, node, x, y, z, col, row, op))

json.dump(enriched, open(os.path.join(ASSETS, "dream_provenance.json"), "w"), indent=2, ensure_ascii=False)

real = sum(1 for r in records if r[1] < SYN_BASE)
print("dreams:", len(records), "(%d real galaxy nodes + %d source-placed)" % (real, len(records) - real),
      "| images pasted:", len(records) - missing_png, "| missing PNGs:", missing_png)
if missing_src: print("WARNING: %d dreams had no source in the corpus glob:" % len(missing_src), missing_src[:5])
print("atlas: %dx%d  cell=%d cols=%d  (GL max is usually 16384)" % (atlas.size[0], atlas.size[1], CELL, COLS))
print("wrote dream_atlas.png + dream_pos.txt to", HERE)
print("repaired", os.path.join(ASSETS, "dream_provenance.json"), "(192-entry node->dream_id->rich)")

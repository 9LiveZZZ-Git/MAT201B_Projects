#!/usr/bin/env python3
# bake_dreams.py -- World of Shadow Work v2 (Phase-2, the dream slice).
# The dreams are already CONCATENATED into the galaxy (type=2 nodes 50661.. with real CLIP
# positions + edges). This LOCAL bake (PIL, no GPU):
#   1) packs the 36 dream PNGs into dream_atlas.png (grid), in galaxy-node order;
#   2) writes dream_pos.txt = each dream's node + world position + atlas cell + op, so DreamLayer
#      billboards the image AT its own node (where the ML wired it into the web);
#   3) REPAIRS the broken assets/dream_provenance.json by joining the node->dream_id stub to the
#      RICH dreams/provenance.json (source + instruction + emergence_frames) so the runtime can
#      name the source artwork and play the emergence sequence.
import os, json, struct
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))         # reagency/v2
ROOT = os.path.dirname(HERE)                               # reagency
ASSETS = os.path.join(ROOT, "assets")
DREAMS = os.path.join(ASSETS, "dreams")

CELL, COLS = 384, 6                                        # 36 dreams -> 6x6 grid @384 = 2304x2304
OP = {"dream-melt": 0, "dream-outpaint": 1, "dream-shadow": 2,
      "melt": 0, "outpaint": 1, "shadow": 2, "edge-outpaint": 1, "shadow-inpaint": 2}

# --- node -> dream_id stub (the runtime-facing provenance, currently empty of source data) ---
stub = json.load(open(os.path.join(ASSETS, "dream_provenance.json")))
# --- rich provenance keyed by dream_id (source / instruction / emergence_frames) ---
rich_path = os.path.join(DREAMS, "provenance.json")
rich = json.load(open(rich_path)) if os.path.exists(rich_path) else {}

# --- positions from points.bin (WSWP: 16-byte header, then N*10 float32) ---
with open(os.path.join(ASSETS, "points.bin"), "rb") as f:
    f.read(4); ver, N, stride = struct.unpack("<iii", f.read(12))
    import array; pts = array.array("f"); pts.frombytes(f.read(N * stride * 4))
def pos(idx): return (pts[idx*stride], pts[idx*stride+1], pts[idx*stride+2])

nodes = sorted(stub.keys(), key=lambda k: int(k))         # 50661..50696, atlas order = node order
atlas = Image.new("RGBA", (CELL * COLS, CELL * ((len(nodes) + COLS - 1) // COLS)), (0, 0, 0, 0))
rows = []
enriched = {}
missing = 0
for slot, nk in enumerate(nodes):
    e = dict(stub[nk])
    did = e.get("dream_id", "")
    col, row = slot % COLS, slot // COLS
    # paste the dream image
    png = os.path.join(DREAMS, did + ".png")
    if os.path.exists(png):
        im = Image.open(png).convert("RGBA").resize((CELL, CELL), Image.LANCZOS)
        atlas.paste(im, (col * CELL, row * CELL))
    else:
        missing += 1
    # join the rich provenance (qwen ops are single-source edits)
    r = rich.get(did, {})
    e["source"]           = r.get("source", "")
    e["instruction"]      = r.get("instruction", "")
    e["emergence_frames"] = r.get("emergence_frames", [])
    e["model"]            = "qwen"
    enriched[nk] = e
    idx = int(nk); x, y, z = pos(idx)
    op = OP.get(e.get("op", e.get("theme", "")), 0)
    rows.append((did, idx, x, y, z, col, row, op))

atlas.save(os.path.join(HERE, "dream_atlas.png"))
with open(os.path.join(HERE, "dream_pos.txt"), "w") as f:
    f.write("# dream_id node x y z col row op  (op 0=melt 1=outpaint 2=shadow; CELL=%d COLS=%d)\n" % (CELL, COLS))
    for did, idx, x, y, z, col, row, op in rows:
        f.write("%s %d %.5f %.5f %.5f %d %d %d\n" % (did, idx, x, y, z, col, row, op))

# repair the runtime-facing provenance in place (so DreamLayer / future readers get real lineage)
json.dump(enriched, open(os.path.join(ASSETS, "dream_provenance.json"), "w"), indent=2, ensure_ascii=False)

print("dreams:", len(nodes), "| images pasted:", len(nodes) - missing, "| missing PNGs:", missing)
print("atlas: %dx%d  cell=%d cols=%d" % (atlas.size[0], atlas.size[1], CELL, COLS))
print("wrote dream_atlas.png + dream_pos.txt to", HERE)
print("repaired", os.path.join(ASSETS, "dream_provenance.json"), "(joined node->dream_id->rich)")

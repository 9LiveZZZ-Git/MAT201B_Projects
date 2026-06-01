#!/usr/bin/env python3
# bake_credits.py -- World of Shadow Work v2 (T8).
# Rasterize the cited THEM worker rows (them.txt) into a credit atlas the runtime CreditLayer
# billboards in a persistent head-locked corner (never fully fades -- alpha floor ~0.12).
# LOCAL only (PIL) -- no Colab / no GPU. THEM is held OUTSIDE "we": these credits persist while
# the warm US/voice dissolves. Every credit keeps Google-able nouns (role, place, wage, company,
# source) so the honesty rule (them.txt header) survives the compression.
#
# Output (reagency/v2/):
#   credits_atlas.png   -- pale ink on transparent; ONE wrapped line per 1024x64 cell, stacked.
#   credits_index.txt   -- one row per credit: "creditId startRow numLines"
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
THEM = os.path.join(HERE, "them.txt")

CW, CH = 1024, 64          # cell: 1024 wide x 64 tall
PAD, FS = 18, 26           # left pad, font size (a touch smaller than the voice -- a ledger, not a line)

def load_font():
    for p in ("/System/Library/Fonts/Supplemental/Georgia.ttf",
              "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
              "/Library/Fonts/Georgia.ttf",
              "/System/Library/Fonts/NewYork.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/SFNS.ttf"):
        if os.path.exists(p):
            try: return ImageFont.truetype(p, FS), os.path.basename(p)
            except Exception: pass
    return ImageFont.load_default(), "default"

font, fontname = load_font()
_probe = ImageDraw.Draw(Image.new("RGBA", (CW, CH)))
def width(s): return _probe.textlength(s, font=font)

def wrap(text, maxw):
    out, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if width(t) <= maxw or not cur:
            cur = t
        else:
            out.append(cur); cur = w
    if cur: out.append(cur)
    return out

# parse them.txt data rows: ROLE | PLACE | WAGE | COMPANY/CASE | SOURCE + DATE
credits = []   # (text,)
for raw in open(THEM, encoding="utf-8"):
    line = raw.strip()
    if not line or line.startswith("#") or "|" not in line:
        continue
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 5:
        continue
    role, place, wage, company = parts[0], parts[1], parts[2], parts[3]
    source = " | ".join(parts[4:])
    credits.append("%s, %s — %s (%s; %s)" % (role, place, wage, company, source))

# wrap each credit into stacked line-cells; remember startRow + numLines per credit
cells = []          # flat list of wrapped line strings
index = []          # (creditId, startRow, numLines)
for cid, text in enumerate(credits):
    wl = wrap(text, CW - 2 * PAD)
    index.append((cid, len(cells), len(wl)))
    cells.extend(wl)

N = max(1, len(cells))
atlas = Image.new("RGBA", (CW, CH * N), (0, 0, 0, 0))
draw = ImageDraw.Draw(atlas)
for row, text in enumerate(cells):
    y = row * CH + (CH - FS) // 2 - 3
    draw.text((PAD, y), text, font=font, fill=(226, 230, 238, 255))   # pale cool ink (held apart from the warm voice)

atlas.save(os.path.join(HERE, "credits_atlas.png"))
with open(os.path.join(HERE, "credits_index.txt"), "w") as f:
    f.write("# creditId startRow numLines  (THEM worker rows from them.txt; head-locked, alpha floor ~0.12)\n")
    for cid, sr, nl in index:
        f.write("%d %d %d\n" % (cid, sr, nl))

print("font:", fontname)
print("credits:", len(credits), "| line-cells:", len(cells), "| atlas: %dx%d" % (CW, CH * N))
print("wrote credits_atlas.png + credits_index.txt to", HERE)

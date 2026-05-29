#!/usr/bin/env python3
"""
fetch_openimages.py — OPTIONAL: augment the corpus with a THEMED, CAPPED subset of
Open Images V7 (CC BY 2.0) for the labor/agency themes (tools, machines, hands, masks…).

Open Images are contemporary Flickr photos, NOT museum art — this diversifies the galaxy
but shifts its archival character; the **magic** theme is intentionally left museum-sourced
(Open Images has no occult classes). Images are CC BY 2.0 — attribution required; we record
the OI ImageID + license here (the original Flickr author/URL live in Open Images' own
metadata CSV — join that for full per-image author credit if you publish).

Runs best on Colab (bandwidth + GCS). Needs:  pip install fiftyone
"""
import argparse
import csv
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))

# Our themes -> Open Images V7 class names (labor/agency only; magic stays museum).
THEME_CLASSES = {
    "labor":  ["Tool", "Hammer", "Wrench", "Saw", "Drill", "Sewing machine",
               "Tractor", "Shovel", "Axe", "Chisel", "Wheelbarrow", "Scissors"],
    "agency": ["Human hand", "Glove", "Mask", "Doll", "Mannequin", "Toy",
               "Wheel", "Lever", "Gear"],
}

FIELDS = ["file", "theme", "query", "source", "source_id", "title",
          "creator", "date", "license", "source_url", "image_url"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=200, help="max images per OI class")
    ap.add_argument("--split", default="validation", help="validation (smaller) or train")
    ap.add_argument("--themes", nargs="*", default=list(THEME_CLASSES.keys()))
    a = ap.parse_args()

    import fiftyone.zoo as foz   # imported here so the file compiles without fiftyone

    corpus = os.path.join(HERE, "corpus")
    csv_path = os.path.join(corpus, "ATTRIBUTION.csv")
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    seen = set(r["file"] for r in rows)

    for theme in a.themes:
        outdir = os.path.join(corpus, "images", theme)
        os.makedirs(outdir, exist_ok=True)
        for cls in THEME_CLASSES.get(theme, []):
            try:
                ds = foz.load_zoo_dataset(
                    "open-images-v7", split=a.split,
                    label_types=["classifications"], classes=[cls],
                    max_samples=a.per_class, shuffle=True, only_matching=True,
                    dataset_name="oi_%s_%s" % (theme, cls.replace(" ", "_")),
                )
            except Exception as e:
                print("  [OI] %s/%s load error: %s" % (theme, cls, e))
                continue
            n = 0
            for s in ds:
                oid = os.path.splitext(os.path.basename(s.filepath))[0]   # OI filename == ImageID
                fn = "oi_%s.jpg" % oid
                rel = "images/%s/%s" % (theme, fn)
                if rel in seen:
                    continue
                try:
                    shutil.copyfile(s.filepath, os.path.join(outdir, fn))
                except Exception:
                    continue
                seen.add(rel)
                rows.append({"file": rel, "theme": theme, "query": cls,
                             "source": "Open Images V7", "source_id": oid,
                             "title": "", "creator": "", "date": "",
                             "license": "CC BY 2.0",
                             "source_url": "https://storage.googleapis.com/openimages/web/index.html",
                             "image_url": ""})
                n += 1
            print("  [OI] %-8s %-16s +%d" % (theme, cls, n))
            try:
                ds.delete()    # free the FiftyOne DB entry (copied files already kept)
            except Exception:
                pass

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print("[OI] done; corpus now %d attribution rows" % len(rows))


if __name__ == "__main__":
    main()

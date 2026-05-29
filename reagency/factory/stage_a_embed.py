#!/usr/bin/env python3
"""
stage_a_embed.py — embed the corpus (images + concept words) into ONE CLIP space.

Pretrained CLIP, INFERENCE ONLY (no training). Words and images land in the same
768-d space, L2-normalized, so cosine similarity is meaningful across both. This is
the "small pretrained model doing the work" — not training it is what keeps us in
budget.

Run on the CUDA box:
    python3 stage_a_embed.py
Outputs (intermediate, consumed by stage_b):
    work/embeddings.npy   float32 [M, 768]
    work/meta.json        [M] records {type,theme,file,label,creator,source,license,source_url}
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_attrib(corpus):
    """basename(image) -> attribution row, from ATTRIBUTION.csv."""
    out = {}
    p = os.path.join(corpus, "ATTRIBUTION.csv")
    if os.path.exists(p):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                out[os.path.basename(r["file"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus"))
    ap.add_argument("--out", default=os.path.join(HERE, "work"))
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch
    import open_clip
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[stage_a] device=%s  model=%s/%s" % (dev, a.model, a.pretrained))
    model, _, preprocess = open_clip.create_model_and_transforms(a.model, pretrained=a.pretrained)
    tokenizer = open_clip.get_tokenizer(a.model)
    model = model.to(dev).eval()

    attrib = load_attrib(a.corpus)
    img_paths = sorted(glob.glob(os.path.join(a.corpus, "images", "**", "*.jpg"), recursive=True))
    print("[stage_a] %d images, embedding..." % len(img_paths))

    embs, meta = [], []
    with torch.no_grad():
        for i in range(0, len(img_paths), a.batch):
            batch = img_paths[i:i + a.batch]
            ims, keep = [], []
            for p in batch:
                try:
                    ims.append(preprocess(Image.open(p).convert("RGB")))
                    keep.append(p)
                except Exception as e:
                    print("   skip %s: %s" % (p, e))
            if not ims:
                continue
            x = torch.stack(ims).to(dev)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            embs.append(f.cpu().numpy().astype(np.float32))
            for p in keep:
                bn = os.path.basename(p)
                row = attrib.get(bn, {})
                meta.append({
                    "type": "image",
                    "theme": os.path.basename(os.path.dirname(p)),
                    "file": os.path.relpath(p, a.corpus),
                    "label": row.get("title", bn),
                    "creator": row.get("creator", ""),
                    "source": row.get("source", ""),
                    "license": row.get("license", ""),
                    "source_url": row.get("source_url", ""),
                })
            print("   images %d/%d" % (min(i + a.batch, len(img_paths)), len(img_paths)))

    # concept words -> text embeddings in the SAME space
    words = []
    wp = os.path.join(a.corpus, "words", "concepts.txt")
    if os.path.exists(wp):
        for line in open(wp):
            s = line.strip()
            if s and not s.startswith("#"):
                words.append(s)
    if words:
        print("[stage_a] %d concept words, embedding..." % len(words))
        with torch.no_grad():
            for i in range(0, len(words), 64):
                chunk = words[i:i + 64]
                tok = tokenizer(chunk).to(dev)
                f = model.encode_text(tok)
                f = f / f.norm(dim=-1, keepdim=True)
                embs.append(f.cpu().numpy().astype(np.float32))
                for w in chunk:
                    meta.append({
                        "type": "word", "theme": "concept", "file": "",
                        "label": w, "creator": "", "source": "concept vocabulary",
                        "license": "", "source_url": "",
                    })

    E = np.concatenate(embs, axis=0)
    assert E.shape[1] in (512, 768, 1024, 1280), "unexpected embed dim %d" % E.shape[1]
    np.save(os.path.join(a.out, "embeddings.npy"), E)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"))
    print("[stage_a] DONE: %d points, dim %d -> %s" % (E.shape[0], E.shape[1], a.out))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
stage_b_layout.py — turn CLIP embeddings into the runtime data assets.

Reads work/embeddings.npy + work/meta.json (from stage_a) and writes, into the
allolib app's assets/ dir, plain binaries the C++ runtime loads with NO ML:

  points.bin   "WSWP" : [magic, i32 ver=1, i32 N, i32 stride=10] then N*10 float32
                        x,y,z, r,g,b, density, cluster, type(0=image/1=word), atlas_idx(-1=none)
  edges.bin    "WSWE" : [magic, i32 ver=1, i32 E] then E*(u32 i, u32 j, f32 weight)
                        cosine-kNN graph in FULL CLIP space (the similarity "webs" AND
                        the live Conductor's adjacency). Computed on 768-d, NOT on 3D.
  labels.txt          : cluster_id \t cx \t cy \t cz \t label   (one per cluster)
  manifest.json       : counts, dim, radius, seed, audio root/mode, atlas info
  atlas_0.png         : 2048x2048, up to 64 hero thumbnails @256px (for the human trace)

Classical ML only (UMAP + kNN + clustering). No training. Fallbacks if optional
deps are missing so it always runs.
"""
import argparse
import colorsys
import json
import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def umap_3d(E, seed):
    try:
        import umap
        n = min(30, max(5, E.shape[0] - 1))
        reducer = umap.UMAP(n_components=3, n_neighbors=n, min_dist=0.0,
                            metric="cosine", random_state=seed)
        return reducer.fit_transform(E).astype(np.float32)
    except Exception as e:
        print("[stage_b] UMAP unavailable (%s); PCA fallback" % e)
        from sklearn.decomposition import PCA
        return PCA(n_components=3, random_state=seed).fit_transform(E).astype(np.float32)


def scale_coords(C, radius):
    C = C - C.mean(axis=0, keepdims=True)
    r = np.linalg.norm(C, axis=1)
    s = radius / (np.percentile(r, 95) + 1e-9)
    return (C * s).astype(np.float32)


def knn_edges(E, k):
    """cosine kNN on L2-normalized CLIP vectors. Returns (edges, density[0..1])."""
    N = E.shape[0]
    k = min(k, N - 1)
    try:
        import faiss
        index = faiss.IndexFlatIP(E.shape[1])   # inner product == cosine (normalized)
        index.add(E)
        sim, idx = index.search(E, k + 1)        # +1 = self
    except Exception as e:
        print("[stage_b] faiss unavailable (%s); sklearn fallback" % e)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(E)
        dist, idx = nn.kneighbors(E)
        sim = 1.0 - dist                         # cosine sim

    seen, edges = set(), []
    mean_sim = np.zeros(N, dtype=np.float64)
    for i in range(N):
        cnt = 0
        for c in range(1, k + 1):                # skip self at col 0
            j = int(idx[i, c]); w = float(sim[i, c])
            mean_sim[i] += w; cnt += 1
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            edges.append((key[0], key[1], max(0.0, min(1.0, w))))
        mean_sim[i] /= max(1, cnt)
    # density ~ local neighborhood tightness, normalized to [0,1]
    d = mean_sim
    lo, hi = np.percentile(d, 5), np.percentile(d, 95)
    density = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)
    return edges, density


def cluster(E, seed):
    N = E.shape[0]
    try:
        import hdbscan
        lab = hdbscan.HDBSCAN(min_cluster_size=max(8, N // 60)).fit_predict(E)
        noise = float((lab < 0).mean())
        ncl = len(set(lab[lab >= 0]))
        if noise < 0.6 and ncl >= 3:
            print("[stage_b] HDBSCAN: %d clusters, %.0f%% noise" % (ncl, 100 * noise))
            return lab.astype(np.int32)
        print("[stage_b] HDBSCAN weak (%d cl, %.0f%% noise); KMeans fallback" % (ncl, 100 * noise))
    except Exception as e:
        print("[stage_b] hdbscan unavailable (%s); KMeans fallback" % e)
    from sklearn.cluster import KMeans
    kk = int(np.clip(round(np.sqrt(N / 2)), 6, 24))
    lab = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(E)
    print("[stage_b] KMeans: %d clusters" % kk)
    return lab.astype(np.int32)


def colorize(cluster_id, density, is_word):
    """cluster -> hue (golden angle); density -> brightness; words read pale/distinct."""
    N = len(cluster_id)
    cols = np.zeros((N, 3), dtype=np.float32)
    uniq = sorted(set(int(c) for c in cluster_id))
    hue = {c: ((i * 0.61803398875) % 1.0) for i, c in enumerate(uniq)}
    for i in range(N):
        h = hue.get(int(cluster_id[i]), 0.0)
        if cluster_id[i] < 0:           # noise / background
            r = g = b = 0.18 + 0.1 * density[i]
        elif is_word[i]:
            r, g, b = colorsys.hsv_to_rgb(h, 0.18, 0.92)   # pale: words stand out
        else:
            r, g, b = colorsys.hsv_to_rgb(h, 0.68, 0.40 + 0.45 * density[i])
        cols[i] = (r, g, b)
    return cols


def build_atlas(meta, coords, density, cluster_id, corpus, assets, max_hero=64, px=256):
    """Pack up to `max_hero` representative IMAGES into one 2048x2048 atlas.
    Returns atlas_idx[N] (slot 0..63, or -1). Used by the human-trace surfacing."""
    from PIL import Image
    N = len(meta)
    atlas_idx = np.full(N, -1, dtype=np.int32)
    # one representative per cluster first (highest density), then fill by density
    img_pts = [i for i in range(N) if meta[i]["type"] == "image"]
    if not img_pts:
        return atlas_idx
    by_cluster = {}
    for i in img_pts:
        c = int(cluster_id[i])
        if c not in by_cluster or density[i] > density[by_cluster[c]]:
            by_cluster[c] = i
    chosen = list(dict.fromkeys(list(by_cluster.values()) +
                                sorted(img_pts, key=lambda i: -density[i])))[:max_hero]
    grid = 2048 // px                       # 8
    atlas = Image.new("RGBA", (2048, 2048), (0, 0, 0, 0))
    for slot, i in enumerate(chosen):
        try:
            im = Image.open(os.path.join(corpus, meta[i]["file"])).convert("RGB")
        except Exception:
            continue
        im.thumbnail((px, px))
        gx, gy = (slot % grid) * px, (slot // grid) * px
        atlas.paste(im, (gx + (px - im.width) // 2, gy + (px - im.height) // 2))
        atlas_idx[i] = slot
    atlas.save(os.path.join(assets, "atlas_0.png"))
    print("[stage_b] atlas: %d hero thumbnails -> atlas_0.png" % len(chosen))
    return atlas_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus"))
    ap.add_argument("--assets", default=os.path.abspath(os.path.join(HERE, "..", "assets")))
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-atlas", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.assets, exist_ok=True)

    E = np.load(os.path.join(a.work, "embeddings.npy"))
    meta = json.load(open(os.path.join(a.work, "meta.json")))
    N = E.shape[0]
    print("[stage_b] %d points, dim %d" % (N, E.shape[1]))

    coords = scale_coords(umap_3d(E, a.seed), a.radius)
    edges, density = knn_edges(E, a.k)
    cluster_id = cluster(E, a.seed)
    is_word = np.array([m["type"] == "word" for m in meta])
    ptype = is_word.astype(np.float32)
    colors = colorize(cluster_id, density, is_word)
    atlas_idx = (np.full(N, -1, np.int32) if a.no_atlas
                 else build_atlas(meta, coords, density, cluster_id, a.corpus, a.assets))

    # points.bin
    with open(os.path.join(a.assets, "points.bin"), "wb") as f:
        f.write(b"WSWP"); f.write(struct.pack("<iii", 1, N, 10))
        for i in range(N):
            f.write(struct.pack("<10f", coords[i, 0], coords[i, 1], coords[i, 2],
                                colors[i, 0], colors[i, 1], colors[i, 2],
                                float(density[i]), float(cluster_id[i]),
                                float(ptype[i]), float(atlas_idx[i])))
    # edges.bin
    with open(os.path.join(a.assets, "edges.bin"), "wb") as f:
        f.write(b"WSWE"); f.write(struct.pack("<ii", 1, len(edges)))
        for (i, j, w) in edges:
            f.write(struct.pack("<IIf", i, j, w))
    # labels.txt (cluster centroids + a representative label)
    with open(os.path.join(a.assets, "labels.txt"), "w") as f:
        for c in sorted(set(int(x) for x in cluster_id if x >= 0)):
            members = [i for i in range(N) if cluster_id[i] == c]
            ctr = coords[members].mean(axis=0)
            # prefer a word as the human-readable cluster label, else a title
            words = [meta[i]["label"] for i in members if meta[i]["type"] == "word"]
            label = words[0] if words else (meta[members[0]]["label"] if members else "cluster %d" % c)
            f.write("%d\t%.3f\t%.3f\t%.3f\t%s\n" % (c, ctr[0], ctr[1], ctr[2], label))
    # cluster representatives (one image per cluster) -> Stage D vessel generator inputs.
    reps = {}
    for c in sorted(set(int(x) for x in cluster_id if x >= 0)):
        best_i, best_d = -1, -1.0
        for i in range(N):
            if int(cluster_id[i]) != c or meta[i]["type"] != "image":
                continue
            if density[i] > best_d:
                best_d, best_i = float(density[i]), i
        if best_i >= 0:
            reps[str(c)] = {"file": meta[best_i]["file"],
                            "label": meta[best_i].get("label", ""),
                            "source_url": meta[best_i].get("source_url", "")}
    json.dump(reps, open(os.path.join(a.work, "cluster_reps.json"), "w"), indent=2)
    print("[stage_b] %d cluster representatives -> work/cluster_reps.json" % len(reps))

    # manifest
    json.dump({
        "title": "World of Shadow Work", "seed": a.seed,
        "n_points": N, "n_edges": len(edges), "embed_dim": int(E.shape[1]),
        "radius": a.radius, "k": a.k,
        "n_images": int((~is_word).sum()), "n_words": int(is_word.sum()),
        "n_clusters": int(len(set(int(x) for x in cluster_id if x >= 0))),
        "audio": {"root_hz": 220.0, "mode": [0, 2, 3, 5, 7, 8, 10]},  # A natural minor
        "atlas": {"file": "atlas_0.png", "px": 256, "grid": 8} if not a.no_atlas else None,
    }, open(os.path.join(a.assets, "manifest.json"), "w"), indent=2)

    print("[stage_b] DONE -> %s" % a.assets)
    print("   points.bin (%d), edges.bin (%d), labels.txt, manifest.json" % (N, len(edges)))


if __name__ == "__main__":
    main()

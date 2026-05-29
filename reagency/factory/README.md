# World of Shadow Work — offline factory

This directory is **not** part of the allolib build. It runs offline (on **Google Colab Pro+**,
A100 ~40 GB) to turn the corpus into plain data assets the runtime loads. Pretrained-inference +
classical ML only — **no training**.

**One-stop: open `wosw_colab.ipynb` in Colab, edit the CONFIG cell, and Run All** — it runs the
whole pipeline (corpus → embeddings → galaxy + webs → image-to-3D vessels → `vessel.wswv`) and
pushes the assets back to GitHub. The per-stage scripts below are the manual reference.

## Pipeline
```
corpus/ ──fetch_corpus.py──▶ images + words (CC0, attributed)
        ──stage_a_embed.py──▶ work/embeddings.npy + work/meta.json     (CLIP ViT-L/14)
        ──stage_b_layout.py─▶ ../assets/points.bin, edges.bin,         (UMAP + kNN + clusters)
                              labels.txt, manifest.json, atlas_0.png
```

## Run
```bash
# 1. corpus (already done; re-run with higher --per-query to grow it)
python3 fetch_corpus.py

# 2. deps (CUDA box)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. build assets
bash make_assets.sh
```
On the 4060 Ti the whole thing is a couple of minutes for ~1k points. Then copy/point the
allolib app at `../assets/`.

## Outputs (the runtime contract)
- `points.bin` `WSWP`: `[magic, i32 ver, i32 N, i32 stride=10]` + N×10 f32:
  `x y z  r g b  density  cluster  type(0=image,1=word)  atlas_idx(-1=none)`
- `edges.bin` `WSWE`: `[magic, i32 ver, i32 E]` + E×(`u32 i, u32 j, f32 weight`) — cosine kNN
  in full 768-d space (the webs **and** the live Conductor's adjacency).
- `labels.txt`: `cluster_id  cx cy cz  label`
- `manifest.json`: counts, seed, audio root/mode, atlas info.
- `atlas_0.png`: ≤64 hero thumbnails @256px for the human-trace surfacing.

## Notes
- No GPU? `stage_a` runs on CPU (slower); UMAP/kNN/clustering are CPU anyway.
- Optional deps degrade gracefully: faiss→sklearn, hdbscan→KMeans, umap→PCA.
- Full CLIP vectors stay here; only the small binaries cross into the runtime.

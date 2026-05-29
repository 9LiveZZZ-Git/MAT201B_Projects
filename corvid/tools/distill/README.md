# Corvid crow-splat distilled student — offline pipeline

This directory holds the **offline** pipeline that produces
`assets/crows/student.bin`: a tiny per-point generative model the corvid runtime
loads to "hallucinate" the crow splat cloud (Part B, Phase 8 of
`../../docs/REWRITE_PLAN.md`).

> **Scope boundary.** Everything here runs OFFLINE on a workstation and MAY use
> external ML frameworks (PyTorch, diffusers, a 3DGS trainer). **None of it ships
> in the corvid binary.** The runtime only consumes the exported `student.bin`
> via a hand-rolled, dependency-free loader (`viz/SplatModel.cpp`). So the
> allolib-only rule still holds for the shipped app — this is the teacher's
> kitchen, not the dinner table.

The runtime already works **without** a student: `SplatModel` falls back to a
procedural crow-image-seeded field. The student just raises fidelity.

---

## What the runtime expects (the contract — do not drift from this)

`viz/SplatModel.cpp` loads a 2-layer tanh MLP applied **per splat point**:

```
out = W2 · tanh(W1 · in + b1) + b2
in  = [ base_pos(3) , cond(8) ]            # d_in  = 11
out = [ dpos(3) , dcol(3) , dsigma(1) ]    # d_out = 7
cond = [ action[0..4]=5 , entropy , confidence , sigmoid(value) ]  # the ThoughtVector
```

`student.bin` format (little-endian), enforced by the loader:

| field | type | value |
|---|---|---|
| magic | char[4] | `"CRVS"` |
| version | int32 | `1` |
| d_in | int32 | `11` (must equal 3 + COND) |
| d_hidden | int32 | 16–256 (must be ≤ 4096) |
| d_out | int32 | `7` |
| W1 | float32[d_hidden·d_in] | row-major `[d_hidden][d_in]` |
| b1 | float32[d_hidden] | |
| W2 | float32[d_out·d_hidden] | row-major `[d_out][d_hidden]` |
| b2 | float32[d_out] | |

`export_student.py` is the single source of truth for writing this; if the
contract changes, change it in BOTH `SplatModel.cpp` and `export_student.py`.

---

## Pipeline phases

### D0 — Smoke (no ML; do this first)
Prove the runtime loader path end-to-end with a random student:
```
python3 export_student.py --random --d-hidden 64 --out ../../assets/crows/student.bin
```
Launch `corvid_m1`; `SplatModel::usingStudent()` should now be true (the cloud is
driven by the MLP instead of the procedural fallback). This validates the format
and the C++ loader before any real training exists.

### D1 — Data prep
- Curate `assets/crows/*.jpg` (already present, 245 images).
- For each kept image: background-matte the crow, estimate a coarse depth/normal
  (any monocular depth model), and sample a target **splat set** — points with
  `(pos3, color3, sigma)` — that reconstructs the crow silhouette/appearance.
- Optionally fit a per-image **3DGS** for higher-quality multi-view targets.

### D2 — Teacher
Pick one (both are external, offline):
- **Diffusion teacher**: a small conditional diffusion model over the splat-set
  representation, conditioned on the 8-d `cond` vector (so different "thoughts"
  yield different crow morphs). Multi-step sampler.
- **3DGS teacher**: optimize Gaussian splats per crow; treat the optimized sets
  as the regression target conditioned on an identity/`cond` embedding.

### D3 — Distill → student
Distill the teacher into the tiny per-point MLP above (consistency / progressive
distillation → effectively **1-step** at runtime). Train the student so that, for
a base point + `cond`, its `(dpos, dcol, dsigma)` reproduces the teacher's
per-point delta. Keep `d_hidden` small (64–128) so the hand-rolled forward stays
real-time over a few-thousand-point cloud.

### D4 — Export + verify
```
python3 export_student.py --weights student.npz --out ../../assets/crows/student.bin
python3 export_student.py --verify ../../assets/crows/student.bin   # re-reads + checks header
```
Then run `corvid_m1` and eyeball: high entropy → cloud disperses/"hallucinates";
high confidence → coalesces toward the crow form.

---

## Acceptance criteria
- `student.bin` parses in `SplatModel::loadStudent` (magic/dims OK) → `usingStudent()`.
- Forward is finite and stable for all `cond` in range; no NaNs over a long run.
- Real-time: frame rate holds at the shipped splat count (tune `d_hidden` / point
  count if not).
- Qualitatively crow-like and visibly responsive to the thought vector.

## Files
- `export_student.py` — writes/verifies `student.bin` (format authority). Also
  `--random` smoke generator (no deps beyond numpy; falls back to stdlib).
- `requirements.txt` — offline-only deps for D1–D3 (NOT needed to build/run corvid).
- `.gitignore` — keeps scripts/plan tracked; ignores datasets, checkpoints, and
  generated weights.

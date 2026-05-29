# World of Shadow Work — Implementation Plan (v2)

*An autonomous, distributed generative-art installation for the AlloSphere, on the theme of Re:Agency.*
*Working project slug: `reagency`. Title: **World of Shadow Work** (Illich's term — the
uncompensated human labor an industrial system silently extracts; here, our images and
words are the shadow work feeding the machine's "magic").*

> v2 integrates the author's five concept decisions and a six-dimension adversarial
> red-team of v1 (compliance, dome determinism, data/budget, concept coherence, audio,
> roadmap). Material changes from v1 are marked **[v2]**. Today: 2026-05-29.
>
> **v3 (2026-05-29) — author decision: build it properly, the date is flexible.** The
> proper **generative-splat vessel is now REQUIRED** (not stretch), and full scope is
> restored. The vessel is the machine's **image-to-3D hallucination of the fixated
> cluster's representative corpus photo** — the human-trace grains dissolve *into* it.
> The 4060 Ti factory run has already started. Changes marked **[v3]**.

---

## 0. Headline decisions

| Axis | Decision |
|---|---|
| **Mode** | Autonomous "machine dream." No live input. The audience witnesses. |
| **Decision-making** | **[v2]** The wander is **live, not a baked replay** — the machine traverses its own k-NN similarity graph on the primary, choosing each step at frame time. Suits the **endless wander**. |
| **Embedding space** | Multimodal CLIP — **words + images** share one space. |
| **Conserved substance** | **[v2]** Galaxy points = smoke grains = vessel body are the **same particles**, re-laid-out per shell in GLSL. One buffer, three views. |
| **Central form** | **[v3]** **Proper generative-splat vessel (REQUIRED).** Offline **image-to-3D** (LGM/DreamGaussian) of cluster-representative corpus photos → distilled **VSLB** keyframe bank → live morph/melt. The procedural GLSL morph is now only the pre-asset fallback. |
| **Human trace (thesis)** | **[v2]** On each fixation of an image, its **actual photograph surfaces for a beat, then is re-abstracted into a grain.** This visible dissolve *is* the act of ceding. |
| **Loop** | Vessel↔galaxy by **slow bleed** (shell crossfade), made causal: the fixated cluster's grains are what form the body. |
| **Corpus** | Images + words about **agency / magic / labor**, CC0/public-domain with attribution (Art Institute of Chicago + Cleveland Museum of Art APIs, + the corvid crow set). **Fetched** — see §3. |
| **Runtime constraint** | allolib + al_ext only; custom GLSL via `al::ShaderProgram` only; no raw GL, no runtime ML. |
| **Deploy** | Linux AlloSphere cluster via `al::DistributedAppWithState` + omni. Dev/preview on commodity hardware. |

---

## 1. Concept spine

The corvid failure was *three effects with no statement*. The fix is one spine every shell,
sound, and camera move serves:

> **You are inside a system walking its own learned associations, live, with no human at
> the wheel — and watching it consume human culture into its own private order of meaning.**

### 1.1 The three shells are ONE conserved substance **[v2]**

There is a single particle buffer: every particle is **one item of the corpus** — a
photograph or a word. The camera is always falling inward; depth `d∈[0,1]` re-lays-out the
*same particles* in GLSL:

- **`d≈0` — VESSEL.** The particles collapse into a body — impossible, never-settling
  geometry (procedural polar/curl displacement; optional distilled splats layered on).
- **`d≈0.5` — SMOKE / LATENT WALK.** The body's grains stream along a vortex/curl-noise
  flow (the real *Gyre* — Mark J. Stock's fluid motion), one region melting into the next,
  connections colored by local latent density (the Anadol signature).
- **`d≈1` — GALAXY.** The grains settle into their UMAP positions: a navigable multimodal
  CLIP cloud, proximity = similarity, with k-NN **similarity webs** — what the model thinks
  belongs together.

Because it is literally the same grains, "vessel dissolves into smoke into galaxy" is true,
not a crossfade between three demos. **Acceptance test:** *a viewer who looks away mid-
transition cannot tell a cut happened.*

### 1.2 Where the agency actually is (the honesty contract) **[v2]**

The red-team correctly attacked v1's equivocation. The honest split:

- **What is near what** — the layout, the webs, the clusters — is **CLIP's learned
  judgment**, baked offline. We do not direct it. This is genuinely "the model's claim
  about what our things mean."
- **The wander** is a **live decision process on the primary**: from the current node, the
  Conductor scores the k-NN neighbors (edge weight × novelty × seeded noise) and picks the
  next. It is not a deep cognition and we will not call it one — it is **graph traversal of
  a learned similarity structure, happening on stage**, which is enough to be true.
- **Hesitation is a real readout, not a baked scalar [v2]:** it is the **entropy of the
  Conductor's next-step distribution.** When several neighbors are near-equally plausible,
  entropy is high → the camera slows and wobbles, the candidate webs flicker. The machine
  visibly hesitates *because its own similarity graph is genuinely ambiguous there.*

So we claim exactly what runs: a system traversing its learned associations live, and we
render its real uncertainty. No "inside a mind" overclaim.

### 1.3 The cede, staged **[v2]**

Pure autonomy risked removing *the act of ceding*. The **human trace** restores it as a
recurring on-screen event: when the wander fixates a photograph, the **actual image
surfaces** (a readable human picture — a worker, a tarot card, a hand), holds for a beat,
then **dissolves back into an abstract grain** as the machine re-files it by its own logic.
That dissolve, every loop, is the transfer the theme is about: *human meaning handed to,
and consumed by, the machine.* This is promoted from optional flavor to **thesis** and is a
**required** feature (it is the main thing distinguishing this from decorative AI spectacle).

### 1.4 Endless wander; slow bleed
No finite "show" — the live graph-walk wanders endlessly (seedable for a reproducible
capture, but not a replay). The vessel↔galaxy loop is a **slow bleed**: as the galaxy
fixates cluster K, zooming out, the body re-forms **from cluster K's own grains** igniting
and flowing outward — the loop's causality is shown, not stapled on.

---

## 2. References → lessons (and the anti-Anadol differentiator)

| Reference | What it is | Borrow |
|---|---|---|
| **Anadol — *Machine Hallucinations*** | StyleGAN2-ADA, cuml-UMAP→3D, density-colored connections, "trace its unconscious decisions." | Latent-walk aesthetic + density-colored webs. |
| **"Gyre 35700"** | **Mark J. Stock** 2012 vortex-particle *fluid* art (not 3D diffusion). | The **smoke motion model**: curl-noise/vortex advection (cheap GLSL, no CFD). |
| **cutterkom** (K. Brunner) | R `generativeart` (seed+formula), `circle-lines` (low-alpha segment webs). | Seeded reproducibility for captures; **webs as low-alpha additive accumulation**; polar sin/cos displacement as the **default vessel** shader. |
| **SplatFlow (CVPR 2025)** | text→3DGS; **weights unreleased**. | Stretch only. Splat generators = **LGM / DreamGaussian**; format generator-agnostic. |
| **TF Projector / Nomic Atlas** | point-cloud nav; fixation→neighbors; hierarchical labels by zoom; density maps. | Autonomous **fixation** UX; depth-gated hierarchical labels; density→nebula glow. |

**Differentiator from Anadol (required, not optional):** the critical *content* — a corpus
explicitly about **agency/magic/labor**, the **visible human→machine cede** (§1.3), and a
**live** decision process — carries an Illich argument Anadol's clouds do not. If we don't
build the human trace and commit the corpus, this is just a more expensive Anadol clone;
the plan treats those as load-bearing.

---

## 3. Two worlds: offline factory vs. allolib runtime

```
  OFFLINE FACTORY (any GPU box, any libs)  ──[ plain data: *.bin *.png *.wav *.txt ]──▶  ALLOLIB RUNTIME (allolib only, no ML, no raw GL)
```

Only data crosses the boundary — same pattern corvid used to retire its libtorch/llama
exception. **Caveat the red-team raised:** corvid only ever proved the *loader + procedural
fallback*; its distilled `student.bin` was never actually trained. So this plan keeps the
**default runtime fully procedural/data-only** and treats every trained asset (splat bank)
as a clearly-isolated stretch goal, not a dependency.

**Corpus status — DONE as the M0 gate [v2]:** `factory/fetch_corpus.py` pulls CC0/public-
domain images for agency/magic/labor from the Art Institute of Chicago + Cleveland Museum of
Art APIs, with a per-image `corpus/ATTRIBUTION.csv` (title, creator, date, license, source +
image URLs). Word list at `factory/corpus/words/concepts.txt` (agency/magic/labor + Illich
lexicon). The corvid crow set (`corvid/assets/crows/`) can be added as a fourth theme. Re-run
the script with a higher `--per-query` to grow it; it is re-entrant and license-clean.

---

## 4. Runtime architecture (allolib)

### 4.1 App skeleton + the core bet

`struct WoSW : al::DistributedAppWithState<WoSWState>` with `onCreate / onAnimate (primary)
/ onDraw (all) / onSound (primary)`. Pattern from `corvid/src/main_corvid.cpp:72-76,1035,1125`.

**The bet:** heavy geometry (the one particle buffer, the web edges) loads identically on
every node at `onCreate()`; **only ~1 KB of scalars sync per frame**; all motion is GLSL
from synced uniforms. Mesh is built **once** — never per frame. (Per-frame rebuild at 100k
points was the #1 perf trap flagged by review.)

### 4.2 `WoSWState` (POD broadcast) **[v2 — corrected]**

```cpp
struct WoSWState {
  uint32_t frame;          // monotonic; renderers DROP any packet with frame <= lastFrame
  float    simTime, seed;
  float    navPos[3], navQuat[4];   // primary computes; ALL nodes apply to nav() before draw
  float    depth;          // shell crossfade 0..1
  int32_t  curNode, nextNode;       // live walk: current + chosen-next (for legibility)
  float    hesitation;     // = entropy of the next-step distribution
  float    traceNode; float traceAlpha;   // human-trace: which image is surfacing + how strongly
  // NO audio event array here (see §4.4): audio events stay primary-only, lock-free.
};
```
Fixes baked in: **(a)** frame-number versioning is *enforced* in the receive path (drop
stale/duplicate packets); **(b)** the audio `events[]` array is **removed** from the broadcast
(it had a torn-read race and renderers had no audio to consume it) — events live in a
primary-only lock-free ring read by `onSound`.

### 4.3 Determinism / dome correctness — explicit fixes **[v2]**

The red-team found these would silently break in the dome; all are now requirements:

1. **Audio only on primary:** guard `void onSound(io){ if(!isPrimary()) return; ... }` AND
   set the `renderer` role so `CAP_AUDIO_IO` isn't auto-assigned. (Single-machine dev builds
   otherwise double-trigger audio.)
2. **Apply the synced pose on every node:** in `onDraw`/`onAnimate` on renderers, set
   `nav().pos()=state().navPos; nav().quat()=state().navQuat` *before drawing*. Renderer
   `nav()` is local state and will NOT track the primary otherwise.
3. **Disable nav input on renderers:** `if(!isPrimary()) navControl().active(false);` — a
   stray click on a render node would desync omni silently.
4. **Omni set once at startup**, never toggled at runtime (toggling recompiles allolib's
   internal shaders but leaves custom ones stale).
5. **Additive-blend determinism note:** curl-noise is seeded by synced `seed`/`frame`, so all
   nodes compute the same displacement; sub-bit GPU float differences are visually irrelevant
   *except* possibly in physically overlapping projector blends — verify on real hardware in
   M7; fall back to alpha-blend for the smoke layer if flicker appears.

### 4.4 Modules **[v2 — reorganized around the conserved buffer]**

| Module | Job | Salvage / API |
|---|---|---|
| **`Conductor`** (primary) | The live mind: holds `curNode`; scores k-NN neighbors (from `edges.bin`) by weight×novelty×seeded-noise; picks `nextNode`; computes `hesitation` = entropy; advances `depth`; drives the human-trace timing; emits audio events to a lock-free ring. Pure data structures (no ML). | New. Uses the **precomputed k-NN adjacency** (not SpatialHash — see §6). |
| **`ParticleField`** (all nodes) | THE one `al::VAOMesh` of `Mesh::POINTS` (corpus). Per-shell layout in the **vertex shader** from `uDepth,uTime,uSeed`: galaxy (static UMAP pos) → smoke (curl-noise advection) → vessel (polar/curl collapse around fixated centroid). Additive point-sprites, density color, fixation brightness. | `Mesh::POINTS` + `gl_PointSize` + `g.blendAdd()`, from `SplatModel.cpp:27-58,222-250`. Reuse `vnoise()` (`:60-76`) → 3D curl. |
| **`WebRenderer`** (all nodes) | Static `Mesh::LINES` of the k-NN edges; per-vertex endpoint-id + density; **low-alpha additive accumulation** (cutterkom); shader brightens lines incident to `uCurNode`/`uNextNode`. | `Mesh::LINES` (`al_Mesh.hpp:69-72`). |
| **`HumanTrace`** (all nodes) | On fixation, draw the fixated image's **photograph** as a camera-facing quad that surfaces then dissolves into the grain, driven by `traceNode`/`traceAlpha`. | Textured quad; image from the hero-subset atlas (256px). `al::Texture`. |
| **`CameraDirector`** (primary; pose synced) | `al::Nav` along the inward wander; ease between targets; slow/wobble ∝ `hesitation`. | `al::Nav`, `al::lerp`, `al::Quat::slerp` (`al_Quat.hpp:863-913`). |
| **`AudioEngine`** (primary) | Musical multisample + pulsaret voices; spatialized. See §7. | Pulsar voice pool + atomics (`pulsar_cern_v2.cpp:168-219` struct; rendered `:725-755`, acquire at `:739`); `gam::SamplePlayer`, `al::Lbap`, `al::Reverb`. |
| **`LabelLayer`** (all nodes) | Depth-gated hierarchical cluster labels (Nomic-style) as billboards. | Pre-rendered label textures or `al::Font`. |
| **`VesselSplats`** (all nodes, **stretch**) | Optional distilled Gaussian overlay on the vessel shell. | New `VSLB` loader; render path = `SplatModel.cpp`. |

**Fixation "grow" caveat [v2]:** per-vertex `gl_PointSize` needs `GL_PROGRAM_POINT_SIZE`,
which allolib doesn't enable in its draw path (enabling it is raw GL — forbidden). So the
fixation emphasis is done via **additive brightness/halo** (and optional camera-facing
billboard quads for hero nodes), **not** per-point size. Verify point-size behavior in M0.

### 4.5 Not carried from corvid
Agent/Entity/predator-prey/Place/flocking/RavenBrain-PPO/ReflectionThread. The "mind" is the
thin live `Conductor` + the baked similarity graph.

---

## 5. Offline factory (`factory/`, Python, never in the build)

- **Stage A — CLIP (corpus ready):** `open_clip ViT-L/14 datacomp_xl_s13b_b90k` (768-d);
  `encode_image` on the corpus + `encode_text` on `concepts.txt`; **L2-normalize both**
  (verify `emb.shape[1]==768`). Runs fine on a free Colab/Kaggle T4 or CPU for ~20k items.
- **Stage B — layout/webs/clusters/density:** `UMAP(n_components=3, n_neighbors=30,
  min_dist=0.0, metric='cosine')` → `coords.bin`. `FAISS IndexFlatIP` on the **full** CLIP
  vectors, **k=10**, dedup undirected → `edges.bin` (this adjacency is what the live
  Conductor walks). `UMAP(8d)+HDBSCAN` → `clusters.bin` + `labels.txt` (centroid + level).
  `1/mean_kNN_dist` → `density.bin`.
- **Stage C — anchors only [v2]:** the walk is **live**, so we bake **no path** — only
  cluster centroids / a few hand-chosen anchors and the per-show `seed` (`manifest.json`).
- **Stage D — vessel:**
  - **DEFAULT (no ML):** none. The vessel is the procedural GLSL morph of the shared
    particles. Ships nothing.
  - **STRETCH (only if a CUDA GPU box exists):** LGM (~5s/asset) / DreamGaussian (~2min) from
    **hand-curated prompts** (`factory/prompts.txt`, NOT raw c-TF-IDF labels). Reduce to 14
    art fields; significance-prune to fixed `G`; **normalize ALL keyframes to one global
    AABB** (per-keyframe AABBs would pop on morph) and store that AABB in the header; align
    indices (greedy NN, accept that the smoky melt hides imperfections; OT only for hero
    pairs); pack **IEEE-754 binary16** (numpy `float16`; C++ unpacks via bit ops) → `vessel.vslb`.
- **Stage E — audio bank:** a small multisample instrument set (a few sustained tonal
  timbres). **Source:** Freesound CC0 / public-domain samples (logged in `audio/CREDITS.csv`)
  or self-recorded. Cap total ≤ ~32 MB. Per-sample root pitch + loop points in `audio/bank.txt`;
  show root + mode in `manifest.json`.

---

## 6. Data contracts (formats + budget)

Little-endian, 4-byte magic + int32 header + payload (corvid convention, `SplatModel.cpp:14-25`).

| File | Layout | 20k pts | 100k pts |
|---|---|---|---|
| `coords.bin` | hdr + N×3 f32 | 240 KB | 1.2 MB |
| `edges.bin` | hdr + E×(u32,u32,f32), E≈N·5; **also the Conductor's adjacency** | 1.2 MB | 6 MB |
| `clusters.bin` | hdr + N×u16 | 40 KB | 200 KB |
| `density.bin` | hdr + N×f32 | 80 KB | 400 KB |
| **subtotal** | | **~1.6 MB** | **~7.8 MB** |
| `atlas_*.png` | hero subset only; 256px for traceable images | — | — |

**[v2] Fixes:** (a) **No SpatialHash for the galaxy** — it's toroidal/uniform-grid and the
CLIP layout is neither; the live Conductor uses the **precomputed `edges.bin` adjacency**
(O(k) per step). (b) **Thumbnail atlas honesty:** GL 3.3 `MAX_ARRAY_TEXTURE_LAYERS` is 256,
so a `Texture::create2DArray` holds ≤256 layers; a hero subset of ~200–256 traceable images
in one array is the simplest safe target (more → multiple arrays). (c) **No per-frame mesh
rebuild** — documented hard rule in `ParticleField`. (d) Full CLIP vectors **never ship**
(layout/webs/density/clusters suffice). **VSLB header carries the global AABB** (6×f32).

---

## 7. Audio — musical, multisample, spatialized **[v2 — corrected]**

- **Hardware:** AlloSphere = **54-speaker** compensated layout (`al_AlloSphereSpeakerLayout.cpp`,
  *not* 60); `audioIO().channelsOut(54)`.
- **Voices:** salvaged lock-free pool (`pulsar_cern_v2.cpp:168-219`). **Polyphony budget
  8–16 voices** (conservative for 54-ch Lbap + reverb on a render node); drop-oldest if over.
- **Voice state machine [v2]:** explicit `noteOn(cluster,pitch,amp,pos) → sustain → release`
  (fixed decay or until superseded) — not fire-and-forget.
- **Sample engine:** **pooled `gam::SamplePlayer`** (whole bank in RAM, capped ≤32 MB) for
  pitched voices; `al_ext::SoundFileBuffered` only for any long textural beds (avoids audio-
  thread file I/O).
- **Quantization [v2]:** `gam::Quantizer` exists (`Gamma/Effects.h:420-477`) for tempo-synced
  snap; pitch snapped to a **fixed root+mode chosen per show** (in `manifest.json`) via
  `f=root·2^(round(semitones)/12)`. Cluster id → chord/mode; not a free-for-all.
- **Mapping:** fixation → note onset (pitch from position-in-cluster, timbre from cluster);
  **web ignition → arpeggio over the fixated node's k-NN, ordered by edge weight**; density
  → voicing thickness; `hesitation` (entropy) → tremolo/detune (calibrated in M5); depth →
  timbre morph (galaxy clear → smoke haze → vessel drone).
- **Spatialize** each voice to its source position via `al::Lbap`; `al::Reverb` wet ≤0.4.
  **Profile `onSound` CPU in M5/M7**; if >~70%, cut polyphony / reverb / drop to `Vbap`.
- **Distributed:** `onSound` primary-only (§4.3); events via primary lock-free ring (§4.2).

---

## 8. Distribution & determinism
See §4.3 for the enforced fixes (primary-only audio, pose application, nav-input disable,
omni-set-once, frame versioning, additive-blend verification). Omni warp is per-projector and
automatic for shaders using `al_ModelViewMatrix`/`al_ProjectionMatrix`. State is ~1 KB →
no MTU risk. **Add an M2.5 two-process desktop check** (run two app instances locally) to
validate state-sync + identical draw *before* needing the dome.

---

## 9. Budget (commodity hardware)
20k–100k `GL_POINTS` at 60 fps is comfortable; interleaved VBO ≈28 B/vertex (100k≈2.8 MB
GPU), built once. Assets ~2–8 MB + a ≤256-layer hero atlas + ≤32 MB audio ≪ 32 GB. No
runtime ML / external libs / raw GL.

---

## 10. Salvage map
| Need | Source |
|---|---|
| Distributed app + state pack | `corvid/src/main_corvid.cpp:72-76,1035,1125`; `al_DistributedApp.hpp:139-196` |
| Lock-free audio pool | `pulsar_cern_v2.cpp:168-219` (Voice + `std::atomic<bool> active`), rendered `:725-755` (acquire `:739`) |
| Point-sprite Gaussian shader + `.bin` loader idiom | `corvid/viz/SplatModel.cpp:27-58,141-166,222-250` |
| 3D value-noise → curl noise | `corvid/viz/SplatModel.cpp:60-76` |
| Spatialization / reverb / streamed audio | `al_Spatializer.hpp`, `al_Lbap.hpp`, `al_Reverb.hpp`, `al_ext/soundfile/al_SoundfileBuffered.hpp`, `tools/audio/spatial_sequencer.cpp` |
| Build pattern (multi-file target) | `corvid/CMakeLists.txt`, `corvid/run_demo.sh`, root `run.sh` |
*(SpatialHash intentionally NOT salvaged for the galaxy — see §6.)*

---

## 11. Project layout
```
MAT201B_Projects/reagency/
  PLAN.md  CMakeLists.txt  run_demo.sh
  src/main_reagency.cpp
  core/   Conductor.{hpp,cpp}  WoSWState.hpp
  viz/    ParticleField  WebRenderer  HumanTrace  LabelLayer  (VesselSplats stretch)  Skybox
  audio/  AudioEngine.{hpp,cpp}
  io/     AssetLoader.{hpp,cpp}
  assets/ coords.bin edges.bin clusters.bin density.bin atlas_*.png labels.txt
          audio/*.wav audio/bank.txt manifest.json  (large bits gitignored)
  factory/ fetch_corpus.py  corpus/  requirements.txt  stage_{a,b,d,e}.py  prompts.txt
```
Ships a tiny **procedural fallback** asset set so the app runs before the factory produces
real assets (corvid pattern, `SplatModel.cpp:208-214`).

---

## 12. Roadmap & calendar **[v2 — descoped + dated]**

**[v3] Deadline relaxed (author decision): build it properly; the date is flexible.** The
REQUIRED/NICE/STRETCH tiers below are no longer a triage list gated on a hard date — the
smoke shell, the **proper generative-splat vessel**, full spatial audio, labels, and the
dome are all in scope. Treat the ordering as a sensible build sequence. The splat factory
(Stage D) runs in parallel on the 4060 Ti (already started). Original date-driven framing
retained below for reference only.

**Original (date-driven) framing — superseded by v3:** ~2–3 weeks to the MAT 201B final.
v1's M0–M7 was broader than corvid's entire 6-week arc *plus* a new ML pipeline — not
shippable. So milestones are now **REQUIRED** vs **NICE-TO-HAVE**, with a hard freeze and a
pre-agreed descope order. **Gradeable artifact = a desktop video capture**, so dome-hardware
risk cannot sink the grade.

**MINIMUM COMPELLING VERSION (fully honors the thesis): M0–M2 + thin audio + human trace.**
The galaxy + live wander + the visible human→machine cede already says the whole thing.

| # | Milestone | Tier | Target |
|---|---|---|---|
| **M0** | Scaffold: project, CMake, `DistributedAppWithState` skeleton, fallback assets, navigable point cloud. **Corpus chosen + fetched (DONE).** Verify `gl_PointSize`/billboard path. | REQUIRED | days 1–2 |
| **M1** | Factory A–B → real galaxy + webs (cluster color, density). 60 fps at N. | REQUIRED | days 2–5 |
| **M2** | `Conductor` live walk + fixation + entropy-hesitation; `CameraDirector` inward wander; **HumanTrace** (the cede); **thin audio slice** (one multisample voice on fixation). | REQUIRED | days 5–9 |
| **M2.5** | Two-process desktop distributed sanity check. | REQUIRED | day 9 |
| **M3** | Smoke shell: curl-noise/vortex advection of the shared particles; depth crossfade (slow bleed). | NICE | days 9–12 |
| **M4** | Vessel shell: **procedural** GLSL morph (default). | NICE | days 11–13 |
| **M5** | Full audio: polyphony, spatialization (54-ch Lbap), reverb, mapping; CPU profile. | NICE | days 12–15 |
| **M6** | Labels, polish, seeded capture, `manifest.json`. | NICE | days 14–16 |
| **M7** | AlloSphere bring-up (omni, multi-node, 54-speaker). | NICE (dome-gated) | if hardware/time |
| **stretch** | Distilled splat vessel (Stage D + `VesselSplats`). | STRETCH | only if GPU box + time |

**Descope order under pressure:** drop M7 → splat stretch → M6 → M5(full) → M4 → M3, never
touching the REQUIRED core. **Freeze ~3 days before the final** and capture.

**External dependencies to confirm NOW:** (1) your final-presentation date; (2) whether a
CUDA GPU box exists (gates only the splat stretch — CLIP+UMAP run on Colab/CPU); (3)
AlloSphere booking (gates only M7).

---

## 13. Top risks
1. **Concept slips back to "pretty demo."** → Human trace + corpus-argument are REQUIRED; **concept gate** (§14).
2. **Time.** → REQUIRED core is small and front-loaded; everything else explicitly droppable.
3. **Splat factory is a hidden second project.** → Default vessel needs zero ML; splats are stretch only.
4. **Dome desync (audio/pose/nav/omni).** → §4.3 fixes are mandatory; M2.5 + M7 verify.
5. **`gl_PointSize` per-vertex unsupported.** → brightness/halo + billboards, verified M0.
6. **GPU box absent.** → only the splat stretch dies; CLIP+UMAP run elsewhere.
7. **Fill-rate / additive flicker in dome.** → cap N + point size; alpha-blend fallback for smoke.

---

## 14. Resolved decisions + the concept gate **[v2]**

**Resolved:** mode (autonomous), space (multimodal CLIP), decision-making (live graph walk),
vessel (procedural default / splat stretch), human trace (required), loop (slow bleed,
causal), corpus (agency/magic/labor, CC0 + attribution, fetched), title (*World of Shadow
Work*), length (endless wander).

**Concept-acceptance gate (end of M2) — falsifiable [v2]:** *Show the build to someone with
no program notes. Without the wall text, can they read (a) that the points are human-made
images/words, and (b) that a system is consuming/reordering them on its own?* If not, the
concept isn't done regardless of engine quality. (This is the test corvid never had.)

**Remaining true polish:** exact corpus size; final sample-bank source; title alternates
(*World of Radical Monopoly* / *World of the Vernacular*).

---

## 15. Revision log
- **v3 (2026-05-29):** Author decision — build it properly, date flexible. Proper
  generative-splat vessel promoted to REQUIRED: image-to-3D (LGM/DreamGaussian) of each
  fixated cluster's representative corpus photo → distilled VSLB keyframe bank → live
  morph; the human-trace grains dissolve into it. Full scope restored (smoke + vessel +
  full spatial audio + labels + dome). Adds factory Stage D + runtime VesselSplats. Galaxy
  bloom/brightness tuned for dome washout.
- **v2 (2026-05-29):** Integrated 5 author decisions + 6-dimension red-team. Conserved-particle
  architecture; live graph-walk with entropy-hesitation; human trace promoted to thesis;
  splat factory demoted to stretch (procedural default vessel); corpus fetched (AIC+CMA CC0);
  dome-determinism fixes (primary-only audio, pose application, nav-input disable, frame
  versioning, omni-once); data fixes (edges-adjacency not SpatialHash, global-AABB VSLB,
  fp16 spec, atlas layer limits, no per-frame rebuild); audio fixes (54 speakers, gam::Quantizer,
  voice budget + state machine, sample sourcing); roadmap descoped with calendar + concept gate.
- **v1:** initial plan from the 6-way research fan-out.
```

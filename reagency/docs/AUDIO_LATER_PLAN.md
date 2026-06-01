# WOSW — Audio "Later" Tier: Deep Xenakis + Glitch Axis (Build Plan)

> Source: multi-agent design workflow (`dramaturgy-spine` + GENDYN + sieves + arborescences + glitch axis), synthesized against the **real shipped engine** `reagency/audio/AudioEngine.{hpp,cpp}` (696 lines) and `reagency/src/main_reagency.cpp`. Branch: **v2**. Continuation of `docs/AUDIO_ORCHESTRATION_PLAN.md` — reuse that doc's voice/section vocabulary (FIBSEC sub-sections, ME→YOU→THEM→US, conductor-wields-techniques, anti-mud census).
>
> **This is PLAN-ONLY.** No code is applied here. Every line number below is verified against the engine as it actually ships today (Phase 1 complete; Phase 2 pending offline assets). Where the upstream design specs cited stale line numbers, they have been re-anchored to the real file.

---

## 0. Engine read — what is REALLY there today (re-anchored line numbers)

The five upstream specs each cite a different stale snapshot. The **authoritative** current layout:

| Concept | Real location | Note |
|---|---|---|
| `Ev` struct + `enum` | `.hpp:61-62` | highest kind is `EV_RHYTHM2=5`. **No `EV_LOOP`, no `EV_GLITCH` exist yet.** |
| `Voice` struct | `.hpp:121-131` | has `pm[6]/pa[6]`, `fmt[3]`, `fcoef[3][3]`, `fz_[3][2]`, `vowels[5]`, `body`, `lp`, `tslot`, `grng` |
| `degHz()` | `.cpp:126` | one-liner: `modeBuf_[idx] + 12*oct`, root_=220 |
| `nodeDegree()` | `.cpp:127` | returns int mode-step degree |
| `update()` (sim) | `.cpp:235-354` | **still keys off the `act` ARG** (`const int a = ...` L243). **No `secT_`/`conduct()` yet.** |
| Story-pop block (THEM) | `.cpp:302-315` | already pops one worker/phrase; sets `cStoryDeg_` (L311), `cCostBeat_` (L309); fires layer-5 pedal (L312) + `whisper()` (L313). `phraseIdx_` at L307. |
| `cBeat_` store (wage→roughness) | `.cpp:274` | already adds `0.04f*cCostBeat_` |
| Eno loops (1d) | `.cpp:519-520`, fired by `fireLoop()` L435 | gated by `cSparsity_` (L518). **Loops fire as `EV_NOTE`, NOT a ring kind** — so `EV_LOOP` is conceptual only; the next free enum value is **6**. |
| `trigger()` switch | `.cpp:381-394` | layer cases 0,1,3,4,5,6,7,8; `default` L393 |
| `tickVoice()` | `.cpp:397-429` | layer-6 kick (L399), layer-9 click (L401), sample path (L402-407), layer-4 whisper (L408-420), generic additive (L421-428) |
| Risset grain alloc | `.cpp:526-527` | `allocCapped(2,28)` |
| **Grain cloud (Poisson)** | `.cpp:609-616` | `allocCapped(2,28)`; sets `g.hz` from spread/quant; **never touches `g.body`** |
| **CC0 choir (US)** | `.cpp:620-631` | `tslot=-7`, `allocCapped(2,28)` |
| **Granular shred of whisper** | `.cpp:633-645` | reads `cap_` ring; supports reverse (`smpRate*-1`, L642) |
| `cap_` capture ring | `.cpp:680`, decl `.hpp:145-146` | `CAPN=1<<16`, masked `&(CAPN-1)`; holds `leadL+leadR` (whisper only) |
| Pooled mix loop | `.cpp:649-672` | up-branch L663-664 (`dryg=0.72, rv=1.25`); `cap_` shred branch L665; choir branch L666 |
| Euclidean grid (kick) | `.cpp:517` | `euStep_`, `euLen_`, `euMask_`; `fireKick()` + Eno step here |
| 2nd Euclidean (timp) | `.cpp:522` | `euStep2_`, `euLen2_` |
| Anti-mud census | `.cpp:473-474` | `nOn` count → `busyGain` headroom |
| Cut/freeze latch | `.cpp:293-296` (sim), `.cpp:471-472` (audio glide), `.cpp:683` (apply) | `cCut_`/`cFreeze_`, one-pole ~50 ms |
| Master tanh | `.cpp:689` | after pink-tilt L685-688 |
| `cActivity_` | `.cpp:240` | swarm-driven, decays `pow(0.5,dt/0.8)` |
| Visual clock | `main_reagency.cpp:94` `CYCLE=840` | act boundaries **210/360/510/600** (L102); `audio_.update(...,act)` L366 |
| `points.bin` | `assets/points.bin` present | ver=1, stride=10, `r[7]=cluster`; `ParticleField::clusterOf(i)` (`viz/ParticleField.hpp:44`) |
| Broadcast state | `core/WoSWState.hpp` | already has `glitchEnergy` (L51) + `extractionDebt` (L52) |

**The single structural fact that gates everything:** `update()` is time-stateless — it reads only the `act` arg and writes constant palettes. The Later tier requires a within-time clock. See §4 (PREREQUISITE).

---

## 1. Thesis & how "Later" completes the arc

**One paragraph.** "World of Shadow Work" is a 14-min AlloSphere ritual whose audio spine is a Fibonacci conductor + data: it owns time, pops one named worker per phrase (THEM), drops into a structural silence at the golden mean (~555 s), and shapes a crescendo-and-cliff, not a plateau. Phase 1 shipped the conductor's *grammar* (year→pitch via `cStoryDeg_`, wage→dissonance via `cCostBeat_`/`cBeat_`, era→timbre/pan), the THEM whisper spine, the US choir (`tslot=-7`), the Eno breath, and the anti-mud census. **The "Later" tier is not four new instruments — it is four CONTROLLED DEGRADATIONS of the ME→YOU→THEM→US thesis, each owned by the conductor and pinned to exactly one rhetorical move:** GENDYN = THEM made *harsh* (the worker's difference-tone roughness given a body, alternating with WHISPER so THEM is always exactly one soloist — the human voice you hear, or the machine voice extracted from them); SIEVES = the ME machinery *revealing itself as math* (Acts I/II stay diatonic; from Act III the pitch alphabet silently swaps to a lattice computed from the corpus's own galaxy-cluster sizes — the sweetness curdles); ARBORESCENCES = the moment of *fan-out / capture* (a single attended worker explodes into a Metastaseis sheaf of glissando trajectories — one becomes a statistical mass); GLITCH = the *edit, the seam of unpaid labor showing* (bitcrush/stutter/reverse-slam/freeze, all quantized to the Euclidean grid so they read as a producer's chops, never as failure). US — the reverb-washed choir + the harmonic pad — is unchanged and load-bearing: it is the comfort the degradations sit *atop*. The whole point is that **US never quite covers them**: the cliff at 555 s strips US away and leaves the degraded THEM exposed, and in V the comfort returns *built on the alien lattice* (sieve never reverts) — comfort permanently marked by roughness, exactly the thesis.

---

## 2. Section-by-section orchestration grid

Driver clock = **FIBSEC** (see §4). Weight legend: **F** = foreground (one stochastic voice max), **A** = affect (under the spine), **S** = structural (re-tunes/re-shapes the whole fabric, makes no sound of its own), **—** = absent.

| FIBSEC sub-section | t (s) | Act | GENDYN | SIEVE | ARBOR | GLITCH | Spine foreground (THEM soloist) | US counterweight |
|---|---|---|---|---|---|---|---|---|
| **EMERGE** | 0–89 | I | — (0) | — (0, diatonic) | — (0) | — (0) | none (drone + Shepard→pad) | pad floor |
| **SEDUCE** | 89–233 | I/II | **A** 0.20 (first appearances, alt. w/ whisper) | — (0, diatonic) | — (0) | **A** 0.10 (on-grid crush ticks, charming) | WHISPER (mostly) | choir enters |
| **READ** | 233–377 | II | **F** 0.40 (trades evenly w/ WHISPER) | — (0, **last diatonic section**) | A 0.15 (hairline branches) | A faint | WHISPER ⇄ GENDYN | choir lushest |
| **EXTRACT** | 377–521 | III | **A** 0.70 (processed-voice under arbor) | **S** 1.0 (machine-math ON) | **F** 1.0 (PEAK fan-out) | **A→F** 0.80 (grinds on dense grid) | **ARBOR** (the single F) | choir present but losing |
| **CLIFF + 555s SILENCE** | 521→555 | III→IV | self-extinguish into cut | S 1.0 (silence is still tuned) | full bloom → guillotined | **F** delivers the cut (reverse-slam + freeze on grid) | — (US stripped) | US cut to near-silence + reverb freeze |
| **TURN** | 555–631 | IV | **F** 0.50 (EXPOSED, naked) | S 1.0 (lattice never reverts) | 0.20 (snapped branches) | **F** 1.0 (loudest but skeletal — each glitch an isolated edit) | GENDYN (naked) | US gone |
| **INTEGRATE** | 631–843 | V | **A** 0.30 (sparse ghosts) | **S** 1.0→0 *late & incomplete* (720→800) | 0.10 (ghost branches) | 1.0→0 by mid-V (**resolves out into clarity**) | WHISPER ghosts | choir/pad return (comfort-on-lattice) |

### Anti-mud trade-off rules (BINDING)
1. **One foreground stochastic voice at a time.** GENDYN and WHISPER are mutually exclusive via `cVoiceSel_` (xor). In EXTRACT, **ARBOR is the single F** — GENDYN is demoted to affect (amp ×0.5, no solo gate) whenever a fan is active. Never two F-stochastic voices.
2. **Glitch is always grid-locked.** Off-grid glitch = "bug." It fires *only* when `((euMask_>>euStep_)&1u)` (and the 2nd Euclidean for lighter micro-gate syncopation).
3. **Shared layer-2 pool.** Arbor grains, glitch cap_-voices, the cloud, the choir, the Risset ticks, and the whisper-shred ALL draw from `allocCapped(2,28)`. No new pool. Glitch cap_-voices capped ≤6 via a census (§7.4). The choir stays ≤8.
4. **busyGain census must include the new voices.** The `nOn` count (L473) already counts all `v.on`; arbor/glitch grains are layer-2 voices so they are covered automatically. Add **layer 10** to the foreground headroom accounting (see §7.1) so a GENDYN solo + tutti still respects the `1/sqrt(n)` law.
5. **Register discipline.** GENDYN/WHISPER soloist sits mid (around `cStoryDeg_`); choir (US) is low-mid washed (`tslot=-7`); arbor fans UP from the worker tone (+spread); the existing +12 melody/pluck lift (L336) keeps the soloist clear of the bed.

---

## 3. Per-technique implementation specs

All four are **gated reads off NEW atomics, set on the SIM thread in `update()`/`conduct()` and read on the AUDIO thread in `render()`/`tickVoice()`**, using the engine's existing `std::memory_order_relaxed` discipline. No alloc/lock/IO on the audio thread. All file IO stays in `init()` (the `loadSamples`/`loadStories` precedent, L169).

### 3.1 GENDYN — true second-order Xenakis Dynamic Stochastic Synthesis (layer 10)

**Concept (resolved to the strict design).** A single waveform period = K linear breakpoint *segments*; each breakpoint `k` has a duration `d[k]` and amplitude `a[k]`, both driven by a **second-order random walk with elastic-mirror barriers** (a per-breakpoint velocity is itself perturbed, position integrates velocity, and on hitting a barrier the excess reflects and the velocity negates). The K segments are read out as a single-cycle wavetable re-walked every period — a buzzy, formant-less harm-tone that is the antithesis of the harmonic pad. This is genuine GENDYN, **not** additive partials.

**`.hpp` insertions (no `Voice` struct growth):**
- Near L93 (next to `cStoryDeg_`): `std::atomic<float> cGendyn_{0.f}, cGendynHarsh_{0.f}; std::atomic<int> cVoiceSel_{0};` (`cVoiceSel_` 0 = whisper's turn, 1 = GENDYN's turn).
- Private method decl: `void fireGendyn();` plus `static constexpr int GSEG = 6;` (K=6 — fits `pm[6]`/`pa[6]` exactly; **resolves** the "8 vs scratch-aliasing" conflict below).
- **Voice walk state aliasing (documented).** Reuse the *existing* Voice fields, since layer-10 never enters the additive or formant branches: durations `d[k]` → `pm[k]`; amplitudes `a[k]` → `pa[k]`; time-velocities `vT[k]` → `fcoef[k/3][k%3]` (9 free); amp-velocities `vA[k]` → `fz_[k/2][k%2]` (6 free) + `vowels[0..1]` if needed; segCount → `syl`; fundamental → `hz`; walk-phase → `phase`; harshness `h` → `body`; foldback accumulator → `lp`. **Add one comment** at the Voice struct documenting the layer-10 alias map, and an `assert`/comment that layer-10 paths never touch formant code.

**`.cpp` insertions:**
- **`trigger()` new `case 10:`** (insert after the `case 8` block, before `default:` at L393): seed `segCount=GSEG`; randomize initial `d[k]∈[DMIN,DMAX]`, `a[k]∈[-1,1]` via `v.grng`; zero `vT[k]`/`vA[k]`; set box widths/step bounds from `v.body=cGendynHarsh_`; `v.life=2.5+1.5*vrand()`, `v.atk=0.03`; fundamental `v.hz=e.hz`.
- **`tickVoice()` new branch** inserted **above the generic additive path (before L421)**, alongside the existing layer-6/9/4 specials: `if (v.layer==10){ ... }` — advance `v.phase` at `fund*isr`; when a period completes, run the second-order walk + elastic mirror on all K breakpoints (a few adds/mults + 4 rand draws **once per period**, not per sample); per-sample output = linear interp between `a[seg]` and `a[seg+1]`, then `tanh(drive*x)` foldback; `return s * env * v.amp;`. Self-contained.
- **`fireGendyn()`** (private): builds `Ev{kind=EV_NOTE, layer=10, hz=degHz(int(cStoryDeg_)), amp=0.10f+0.04f*cGendyn_, pan≈center}` and `schedule()`s it (reads `cStoryDeg_`/`cGendynHarsh_`/`cGendyn_` each once).
- **Mix routing** in the pooled loop's generic `else` (the L660 branch, after the choir test): add `else if (v.layer==10){ dryg=0.95f; rv=0.6f; ppSendL += 0.2f*s*cL; ppSendR += 0.2f*s*cR; }` — dry-and-forward soloist, little reverb (roughness must stay legible). NOTE: layer 10 must be handled in the up/low/else dispatch; it is neither layer≤2 nor 6/7/9, so it lands in the generic `else` — add the explicit `cL/cR` pan there.
- **Sim control** in the story-pop block (L308-313): when a worker pops AND `cVoiceSel_==1`, call `fireGendyn()` **instead of** the `whisper()` at L313; always set `cGendynHarsh_.store(st.costN)` and toggle `cVoiceSel_.store(phraseIdx_&1)`. (The SAME `st.costN` already feeds `cCostBeat_`→`cBeat_` at L274, so one wage number drives BOTH the pad roughness AND the soloist harshness — one worker, two coupled symptoms.)

**Sim-vs-audio split.** Sim sets per-phrase floats (`cGendynHarsh_`, `cVoiceSel_`, `cGendyn_`). Audio reads them **at voice-birth only**; the sample-rate walk is pure local Voice state seeded from `v.grng`. Per-sample inner loop is atomic-free.

**Pool.** `allocCapped(10, 2)` — max 2 concurrent, but the `cVoiceSel_` mux means ~0–1 active steady-state, and it *replaces* a whisper half the time, so net voice count is unchanged.

**Params.**

| Param | Value |
|---|---|
| Segments | `GSEG = 6` |
| Fundamental | `degHz(int(cStoryDeg_))` clamped `[70,440]` Hz (low/mid soloist; anti-alias) |
| Life / atk | `2.5 + 1.5*frand()` s / `0.03` s |
| Amp | `0.10 + 0.04*cGendyn_` (caps ~0.14, same order as whisper 0.15) |
| Harshness | `h = clamp01(cGendynHarsh_)^0.6` (gamma widens audible spread) |
| Time box | `DMIN=0.3, DMAX=2.0` (relative, normalized by sum/period) |
| Time step / vel cap | `STEPT = 0.02+0.10*h` / `VTMAX = 0.15+0.35*h` |
| Amp box / step / vel cap | `[-1,+1]` / `STEPA = 0.04+0.30*h` / `VAMAX = 0.08+0.50*h` |
| Foldback drive | `1.0 + 3.0*h` → `tanh(drive*x)` (soft saw at h≈0, slammed fold at h≈1) |
| `cVoiceSel_` | `phraseIdx_ & 1`; in READ favor whisper (gendyn 1-in-3); in V gendyn rare/ghostly |
| Conductor envelope `cGendyn_` (×within-section `f`) | I:0  II:0.20  III:0.70  IV:0.50  V:0.30 (design-spine values; the GENDYN spec's {0,0.05,0.22,0.30,0.10} is the *amp* sub-envelope — keep the spine weights as the orchestration scalar, multiply by 0.30 floor for the literal amp so the two reconcile) |
| Mix | `dryg=0.95, rv=0.6, ppSend 0.2`, pan center-ish |

**RT/CPU.** Per active voice: one segment-advance compare + one interp + one `tanh` per sample (cheaper than the existing 6-`sin` additive sum at L423). The walk fires once per period (~100–400 samples), negligible. Reflect-mirrors guarantee `a[k]∈[-1,1]` → no DC runaway/NaN.

---

### 3.2 SIEVES — Xenakis pitch-alphabet from galaxy cluster sizes

**Concept.** Three pitch sieves (unions of residual classes mod 12), **derived at `init()` from the real `points.bin` cluster sizes**, baked into `int8_t sieveScale_[3][12]`. When `cSieve_>0.5` (Acts III–V), `degHz()` snaps each requested degree to the nearest lit pitch-class of the active sieve; otherwise the diatonic `modeBuf_` path is byte-for-byte unchanged. The audience does not consciously hear the swap; the same gestures land on an alien lattice and the sweetness curdles. Extraction at the level of the scale itself.

**`.hpp` insertions:**
- Near the scale block (after L47): `int8_t sieveScale_[3][12] = {};` (36 bytes, init-built, read-only on audio) + `std::atomic<float> cSieve_{0.f};`.
- Method decl: `void buildSievesFromCounts(const int* clusterCounts, int nClusters);` — **AudioEngine stays decoupled from `ParticleField`** (preferred design; resolves the dependency risk). Active sieve index reuses the existing `cClusterId_` atomic: `int(cClusterId_.load(relaxed)) % 3` (cluster −1 → row 0). **No new `sieveSel_` field needed.**

**`.cpp` insertions:**
- **`init()` wiring (L169 area).** Add an `init()` overload: `init(assetDir, sampleRate, const int* clusterCounts, int nClusters)`. `main_reagency.cpp` (after `field.init()` at L131-ish, before the existing `audio_.init` at L145) computes a 3-int histogram by iterating `field.clusterOf(i)` over `[0,count())` and passes it. `buildSievesFromCounts()` runs **before `ready_.store(true)`** (L177). **No second file read** — reuse the histogram from `ParticleField`. If counts are null (procedural galaxy), fall back to a fixed triple `{30000, 8000, 6000}`.
- **`degHz()` rewrite (L126)** — the only hot edit, still `const`, branch-light:
  - If `cSieve_>0.5`: split `degree` into `oct = degree/12`, `pc = degree%12` (fix negatives); read `sieveScale_[sc][pc]`; if not lit, nearest-lit-pc search outward up to ±6 semitones; return `root_ * pow(2, (pc + 12*oct)/12)`.
  - Else: the **unchanged** diatonic path (`modeBuf_[idx] + 12*oct`).
  - **Semantic note (document at `degHz`/`nodeDegree`):** under the sieve, the integer `degree` is reinterpreted as semitones-mod-12 rather than mode-steps. That remapping IS the dramaturgy — a future editor must not "fix" it.
- **Story-pedal register fix (L310).** When `cSieve_` is live, compute the worker register in semitones so the year→register spread survives the alphabet swap: gate `deg = int((st.yearN*2-1)*12)+12` on `cSieve_` in the story-pop block (else keep the existing mode-step formula).

**Sim-vs-audio split.** `sieveScale_` is written ONCE in `init()` (sim/main thread) before `ready_` release; read-only forever on audio — a benign, race-free publish via the existing `ready_` acquire/release. `cSieve_`/`cClusterId_` are standard atomics. `degHz()` is called on BOTH threads (sim: onArrival L196/L198, igniteArp, story-pedal L312; audio: fireTimp L432, grain cloud L613, choir L624, Risset L527, loops L438) — the atomic read + immutable array is the correct discipline.

**Sieve construction (deterministic, reproducible from sizes).** For each cluster `i` of size `s`: `M1 = 3 + (s%5)`, `R1 = s%M1`, `M2 = 2 + ((s/7)%4)`, `R2 = (s/13)%M2`; `sieveScale_[i][pc] = (pc%M1==R1 || pc%M2==R2)`; if `popcount<4`, force-set `{0,7,3,10}` (anchor to tonic/fifth). **Verified output** from the real `points.bin` (N=50697, sizes `{0:35902, 1:8466, 2:6276, -1:53}`):
- Row 0 (dense "words" gamut): pcs `{1,2,3,5,7,9,11}` — 7 notes.
- Row 1: pcs `{0,2,3,6,9,10}` — 6 notes.
- Row 2: pcs `{0,2,4,6,8,10}` — **whole-tone**, a textbook machine/symmetry signature — 6 notes.

**Control / envelope.** `cSieve_` is a hard gate (not a blend), stepped in `conduct()` on `secT_`: `0` for `secT_<322` (Acts I–II); `1` for `322≤secT_<720` (spanning the 555 cut); **linear-ramp 1→0 over `720≤secT_<800`**; `0` for `≥800`. The lattice resolves back to diatonic only at the very end, **late and incomplete** — exhaustion, not triumph (mirrors the V comfort-FAIL). Per the spine grid the weight is `S 1.0` III–V; the late partial-revert is the audio twin of the monotonic `extractionDebt` latch.

**RT/CPU.** Negligible: one relaxed load + one modulo + ≤6-iteration `int8_t[12]` search before the existing `pow` (which already dominated). No per-sample sieve work, no new call sites, zero heap/lock/IO on audio, no new voices. A scale-swap retunes only NEW onsets (no zipper); the hard switch is placed at a Fibonacci boundary where the texture already changes, masking any discontinuity.

---

### 3.3 ARBORESCENCES — Metastaseis branching of the grain cloud

**Concept.** Layer-2 cloud grains are pure additive sines whose phase advances from `v.hz` each sample (L421). A per-sample glide of `v.hz` toward a stored target IS a glissando — no new DSP primitive. Store the glide target in the free `v.body` slot (touched only by layer-4 whisper; layer-2 grains never read/write it — collision-free). Emit grains in short timed FANS from a shared trunk pitch, endpoints fanning `±octHalf · (i/N)`, so a single attended worker explodes into a Metastaseis sheaf. The trunk is seeded from `cStoryDeg_`, so the tree grows out of the THEM spine, not abstract math.

**`.hpp` insertions:**
- Near L94 (next to `cChoir_`): `std::atomic<float> cArbor_{0.f};`. (Spread reuses the existing `cCloudSpread_` atomic — no second knob; resolves the param-coherence concern.)
- Audio float near L147 (beside `choirTimer_`): `float arborTimer_ = 0.f;`.
- **No Voice struct growth** — reuse `body` (target hz) + a sentinel `tslot=-9` (verified free: −7=choir, −1=default).

**`.cpp` insertions:**
- **Sim** in `conduct()`/`update()` (the CL/SP/QU store area, L283): `static const float ARB[6]={0.f,0.f,0.15f,1.0f,0.10f,0.20f}; cArbor_.store(ARB[a]*(0.5f+0.5f*f), relaxed);` — peaks III, builds within-section via the existing `f` ramp.
- **Audio — new fan emitter** inserted **immediately after the grain-cloud block (after L616, before the choir block at L620).** Runs `arborTimer_` as a second Poisson clock; when `cArbor_>0.05`, emit a BURST of N=4–8 grains in one frame sharing `trunk = degHz(int(cStoryDeg_) + small jitter)`. Each grain `i` gets a fan offset `off = octHalf*(i/(N-1)*2-1)`; **convergent** grains start at the fan endpoint and glide to trunk (`g.hz=endHz; g.body=trunk;`); **divergent** start at trunk and glide out (`g.hz=trunk; g.body=endHz;`) — alternate via a coin so III breathes between intake and output. Each grain: `layer=2`, `tslot=-9`, `K=2`, `life=0.25+0.55*frand()` (audible sweep, vs cloud's 0.02–0.18), `amp=(0.018+0.03*arb)/sqrtf(N)*gate_[G_GRAIN]` (energy-held against the longer life), `pan=0.9*(2*frand()-1)`. **Must overwrite `g.body` AFTER `g=Voice{}`** (struct default is 0.6 — else grains glide to garbage).
- **Audio — glide in `tickVoice()`** before the additive output (before L421): `if (v.layer==2 && v.tslot==-9){ v.hz += (v.body - v.hz) * (12.f*isr); }` — a **constant glide coef, NOT `v.lp`** (resolves the lp-double-use risk; `v.lp` stays the additive output one-pole).
- **Mix: NONE.** Arbor grains are layer-2, so they flow through the existing up-branch (L663-664: `dryg=0.72, rv=1.25`), the grain ping-pong send (L669), drift pan, and `gate_[G_GRAIN]` for free.

**Sim-vs-audio split.** Sim sets one float (`cArbor_`) per buffer. Audio reads it lock-free; fan geometry uses the existing audio-thread `rng_`/`frand()`.

**Params.**

| Param | Value |
|---|---|
| Branches `N` | `4 + int(4*arb + 0.5)` (4–8) |
| Half-width (octaves) | `octHalf = 0.5 + 3.5*cCloudSpread_` (SP[3]=0.90 → ~half-oct at low, ~3.5 oct at III peak) |
| Trunk degree | `int(cStoryDeg_) + (rng%5 − 2)` |
| Poisson rate | `alam = 2 + 10*arb` (~2 fans/s hairline → ~12 peak) |
| Glide coef | `12*isr` (reaches target over ~one grain life) |
| Grain life | `0.25 + 0.55*frand()` s |
| Sentinel | `tslot = -9` |
| Envelope `cArbor_` (×`0.5+0.5*f`) | I:0  II:0.15  III:1.0  IV:0.10  V:0.20 |

**RT/CPU.** One extra `add+mul+sub` per arbor grain/sample (the glide); fan burst = 4–8 grains. Hard-bounded by the shared `allocCapped(2,28)` — III sustains ~20–50 layer-2 arbor grains but **adds zero voices beyond the 28-cap** (oldest-steal). Longer life → fewer-but-bigger grains, so the `1/sqrt(N)` amp law holds energy; the layer-2 Gaussian env (L424) ramps every grain (no edge clicks even with longer life).

---

### 3.4 GLITCH AXIS — `EV_GLITCH=6`, grid-locked edit bus

**Concept.** Five RT-safe ops, all reading the existing `cap_` ring (the captured whisper lead) with ZERO allocation, all phase-locked to the Euclidean grid so they read as deliberate splices: (1) **bitcrush** = sample-and-hold + bit-quantize on the master L/R; (2) **stutter** = re-trigger loop over `cap_`; (3) **reverse-slam** = one long `cap_` grain at negative `smpRate` timed to END on the next grid hit; (4) **freeze** = latch+loop a single `cap_` window + drive `cFreeze_` (reusing the existing reverb-freeze machinery); (5) **master micro-gate** = a fast one-pole-glided amplitude gate on the final bus. Because every op chews `cap_` (the worker's recorded voice), the glitch is never abstract noise — it is the machine *editing* the human record.

**Enum / ring resolution (BINDING — resolves the `EV_LOOP=6`/`EV_GLITCH=6`/`=7` three-way conflict):** In the *shipped* engine, Eno loops fire as `EV_NOTE` (L438), so `EV_LOOP` is conceptual only and is **not** added to the enum. The next free enum value is **6** (`EV_RHYTHM2=5` is highest). Therefore: **`EV_GLITCH = 6`** appended at L62. This **diverges from the Phase-1 doc's reservation table** (which said `EV_LOOP=6, EV_GLITCH=7`) for the correct reason that `EV_LOOP` was never realized as a ring kind — note this divergence explicitly in the commit. `islot` encodes the op: **0 = emergence-resolve (RESERVED for Phase-2, never fired from the act-III/IV path); 1 = bitcrush; 2 = stutter; 3 = reverse-slam; 4 = freeze; 5 = micro-gate.** `a/b/c` carry depth/len/reps.

**Phase-2 coordination (BINDING).** Phase 2's "dream forms from glitch into clarity" gesture also fires `EV_GLITCH islot=0` with shrinking `cStutLen_`/falling `cCrush_`. **This Later axis NEVER fires `islot=0`; it owns `islot 1–5`.** Today Phase-2 glitch is graphics-only (`glitchEnergy` is visual, not plumbed to `audio_.update`), so there is no live audio double-fire — but build the islot partition NOW so Phase-2 lands cleanly. A single arbiter: if a future `cEmerge_` is mid-transition, suppress act-glitch.

**`.hpp` insertions:**
- Enum (L62): append `, EV_GLITCH = 6`.
- New sim atomics (near L95, by `cSparsity_`): `std::atomic<float> cGlitchDens_{0.f}, cCrush_{0.f}, cStutLen_{0.f}, cGateAmt_{0.f};` and `std::atomic<int> cGlitchOp_{0};`.
- New audio-owned state (near L146, by `cap_`/`granTimer_`): `float crushPh_=0.f, crushHoldL_=0.f, crushHoldR_=0.f, microGate_=1.f, microGateTgt_=1.f; int stutOrigin_=0; float stutPlay_=0.f; int freezeHold_=0; int crushBits_=16; float capRms_=0.f;` (no atomics needed for these).

**`.cpp` insertions (all line numbers REAL):**
- **Sim control** in `conduct()`/`update()` (by the `CHOIR_LAM` table, L298): per-act glitch tables (§ below) → store `cCrush_/cStutLen_/cGateAmt_/cGlitchDens_`. **`update()` does NOT push `EV_GLITCH` events** — the density clock lives on the audio thread (grid-locked, sample-accurate); avoids ring pressure.
- **Audio firing — grid-locked**, inside the kick-grid block at **L517**, right AFTER `fireKick()`: read `act` (already loaded at L452) + `cGlitchDens_`; with `p = GLITCH_DENS[a]*(0.4+0.6*act)*(whisperOn?0.5:1.f)`, pick an op and arm it. **Big ops (reverse-slam, freeze) gate on `euStep_==0` (downbeat); crush/micro-gate on any hit.** The 2nd Euclidean (L522) can host lighter micro-gate hits. Fire **directly here, bypassing `schedule()`/`pend_`** (which would smear the grid-lock by 2–156 ms) — exactly as `fireKick` already does.
- **Bitcrush + micro-gate apply (per-sample)** just BEFORE the master tanh at **L689** (after the pink-tilt L685-688): crush hold-and-step on `crushPh_`; then `L*=microGate_; R*=microGate_; microGate_ += (microGateTgt_ - microGate_)*gco(0.006f,1,sr_)`. The existing tanh then soft-clips any requantization spike.
- **Stutter / reverse-slam / freeze** play through the **existing pooled cap_-voice path + mix branch (L665).** They are `cap_` voices (`v.smp==cap_.data()`), already routed (`dryg=1.2, rv=0.8`). Stutter needs a `tslot==-9`-style STUTTER sentinel (use a distinct value, e.g. `tslot=-11`, to avoid the arbor `-9` collision — see §6) + smpPos wrap in `tickVoice()`'s sample branch (wrap `smpPos` back to `stutOrigin_` window every `stutLen_`). Reverse-slam needs no new code (negative `smpRate` already works, L642). Freeze sets `v.life` huge + loops smpPos while `freezeHold_>0` (decremented per grid step).

**Sim-vs-audio split.** Sim sets per-act target floats. Audio owns the density clock (grid-locked), the per-sample crush/gate state, and the cap_-voice arming. `cCrush_` depth biased harsher when `cCostBeat_` is high (cheap wage → most-destroyed crush, mirroring `cBeat_` roughness). `cStutLen_` falls across the section build `f` in III.

**Params (per-act tables `[0..5]` = fallback,I,II,III,IV,V):**

| Table | I | II | III | IV | V |
|---|---|---|---|---|---|
| `GLITCH_DENS` (P(op\|grid hit)) | 0 | 0.04 | 0.30 | 0.35 | 0 (→0 ramp first half of V) |
| `GLITCH_CRUSH` (0..1 destroyed) | 0 | 0.20 | 0.75 | 0.55 | 0 (resolving) |
| `GLITCH_STUT` (P(stutter\|op)) | 0 | 0 | 0.60 | 0.40 | 0 |
| `GLITCH_GATE` (micro-gate depth) | 0 | 0.10 | 0.45 | 0.35 | 0 |

- **Bitcrush map:** `bits = 12 - 8*cCrush_ - 2*cCostBeat_` (≈12 clean → ≈2 destroyed); `lv = exp2(round(bits))`; `crushRate_ = sr_*(1 - 0.92*cCrush_)` (full → ~3.5 kHz hold at max).
- **Stutter:** `stutLen = (0.18 - 0.13*f)*beat_samples` in III (tightens across build); `beat_samples = sr_/slot`; origin = `capPos_` snapshot.
- **Reverse-slam:** one cap_ voice, `smpRate=-1`, `life=(1.0-0.5*f)` s timed to next downbeat, `amp=0.10`, `pan=0` (center slam), `atk=life*0.3`.
- **Freeze:** `freezeHold_=1–2` grid steps; window ≈0.08 s looped; set `cFreeze_=1` for the held duration (shares the SINGLE existing reverb freeze glide — do NOT add a 2nd reverb freeze).
- **Micro-gate:** target re-rolled per grid hit `(frand()<GLITCH_GATE[a])?0:1`; attack `gco(0.006f)`, release `gco(0.012f)` (slower release avoids pop).
- **All fire only when `((euMask_>>euStep_)&1u)`.** Concurrent glitch cap_-voice cap **≤6** via a census like `nOn`, shared inside the layer-2 cap of 28.

**Dramaturgy gating.** Glitch ducks under the whisper (reuse the `whisperOn` detect at L464): halve density when a whisper sounds, so edits fall in the GAPS between testimonies. Weight peaks where the drone DROPS OUT (Act IV `G_DRONE→0`, ACTG L249) → glitch is the dominant texture in the bare-pulse TURN with minimal mix conflict. **Cliff shape:** III = crescendo (rising density); fire a dense reverse-slam + freeze cluster in the 2–3 s BEFORE 555, then near-silent through the cut (the cliff is glitch-built); IV = aftershock; V = resolves out.

**cap_ silence guard.** `cap_` holds the whisper lead only (L680). Track a one-pole `capRms_` of cap_ writes; fire stutter/reverse/freeze only when `capRms_>threshold` (crush/micro-gate operate on the full bus, always have material).

**RT/CPU.** Bitcrush = phase accumulator + round + compare on 2 master samples (~4 flops/sample). Micro-gate = 1 mul + one-pole. Stutter/reverse/freeze cost NOTHING new — existing pooled cap_ voices under the 28-cap. Zero new reverb/delay taps. Glitch is grid-quantized so onsets land on existing micro-timed beats (no mid-note discontinuity); crush depth glided via atomic target + one-pole (like `cCut_`/`cFreeze_`); reverse-slam uses cap_ already windowed by grain envelopes; the final tanh catches tutti+glitch transients.

---

## 4. PREREQUISITE — make the conductor time-authoritative (the gate for all four)

All four techniques key off `conductAct_ + f` derived from accumulated `dt`. Today `update()` keys off the `act` arg (L243). The Later tier needs the doc's **1a** to land first:

- **`.cpp` file scope:** `static const float FIBSEC[12]={89,144,233,322,377,466,521,555,576,631,720,843}; static const int FIBACT[12]={1,1,1,2,2,3,3,3,4,4,5,5};` (golden-mean cut at index 7 = 555).
- **`.hpp`:** `float secT_=0.f; int conductAct_=1;`.
- **`conduct(dt)`** as the first statement of `update()` (after L236): `secT_+=dt; int s=0; while(s<11 && secT_>=FIBSEC[s]) ++s; conductAct_=FIBACT[s];` and a within-section `f = clamp01((secT_-segStart)/(FIBSEC[s]-segStart))`. **Replace `const int a` (L243) with `conductAct_`.**

### AV-SYNC DECISION (BINDING — resolves the cross-design risk)
The audio FIBSEC clock (843 s, cut 555) and the visual cycle (`CYCLE=840`, act boundaries 210/360/510/600) are **genuinely different clocks**. If audio goes `secT_`-authoritative while visuals stay on `cycleClock_`, the 555 audio cliff will NOT land on the visual turn (visual IV starts at 510, not 555).

**Decision: adopt option (a) — keep `act` advisory from the visual cycle for the *macro* structure, and derive `conductAct_`/`f`/`secT_` from the SAME `cycleClock_`-driven time** by accumulating the passed `dt` and mapping the visual boundaries, OR by passing `cycleT` into `audio_.update()`. Concretely: the four techniques key off a `secT_` that tracks the visual `cycleT` so the 555 cliff and visual TURN coincide. **This means the FIBSEC table's 555 cut must be reconciled with the visual IV boundary (510/600).** Two sub-options:
- **(a1) Lowest risk (recommended for first landing):** key the four techniques off the existing `act` arg + a within-act `f` ramp (as Phase 1 already does), and place the cliff/freeze cluster at the visual III→IV edge (510 s) rather than 555. The "golden mean" becomes the visual turn. Techniques layer in without destabilizing the shipped AV lock.
- **(a2) Full FIBSEC:** move the visual act boundaries to the FIBSEC table (210→233, 510→555, etc.) so both clocks share one source. **Flag this as a SEPARATE decision** — it touches `main_reagency.cpp:99-102` and the whole visual envelope; do it only after the four techniques are listenable.

**This document specifies the techniques in FIBSEC terms (the design intent), but the FIRST implementation lands on (a1) and the FIBSEC cut is documented as the target for a later clock-reconciliation pass.** Broadcast `secT_/conductAct_/cVoiceSel_` via `WoSWState` (which already carries `glitchEnergy`/`extractionDebt`) only if a renderer must react to the audio cliff for AV lockstep.

---

## 5. Dramaturgy arc — ME→YOU→THEM→US, build → 555 cliff → haunted V

- **EMERGE (0–89, I):** DRONE + Shepard-noise→pad only. **None of the four present.** The dream still pretends to be human/musical. This silence-of-the-machine is what makes its later arrival land. *Thesis: ME, pre-extraction.*
- **SEDUCE (89–233, I/II):** MEL/BELL/DRONE + choir (US). First faint GENDYN appearances alternating with whisper; faint on-grid bitcrush ticks (charming, tape-like). Sieve OFF. *Thesis: YOU drawn in.*
- **READ (233–377, II):** WHISPER/BELL lead. GENDYN trades evenly with WHISPER (THEM as alternating human/machine soloist fully established). **Last diatonic section** — the calm before the math. Choir at its lushest. *Thesis: YOU→THEM commitment.*
- **EXTRACT (377–521, III — DENSEST):** THE GRIND. **SIEVE flips ON** (alphabet becomes machine-math; gestures sour). **ARBOR PEAKS** (cloud fans into Metastaseis sheaves; one worker → statistical mass). GENDYN is the processed-voice harshness under it. GLITCH grinds on the dense grid. **Anti-mud binds hard:** ARBOR is the single F; GENDYN drops to affect; glitch stays grid-locked. The within-section `f` ramps the swarm/beating to the cliff edge. *Thesis: EXTRACTION — THEM processed into throughput.*
- **THE CLIFF + 555 s SILENCE (III→IV):** the crescendo does NOT plateau. At the cut latch, `cCut_→0` (one-pole ~50 ms, L471/683) **strips US and the tutti** to near-silence + reverb freeze. GLITCH delivers the cliff: reverse-slam off `cap_` + a held freeze, ON the grid, reading as the edit that severs the piece. GENDYN voices self-extinguish into the cut. Everything the four techniques built collapses. **The structural payoff of the whole arc.**
- **TURN (555–631, IV — STRIPPED):** KICK/WHISPER + bare pulse. GENDYN **EXPOSED** — with US gone, the processed-voice soloist stands naked (the most affecting moment). GLITCH at its loudest but skeletal (sparse grid → each glitch an isolated, brutally legible EDIT). SIEVE stays ON. ARBOR collapsed. *Thesis: the TURN — "you did not look for them."*
- **INTEGRATE (631–843, V):** the haunted recap. US (choir/pad) tries to return (comfort-FAIL). GLITCH **resolves out** (crush depth shrinks, stutter lengthens then stops → clean tone — the Phase-2 "image forms from glitch into clarity" gesture recapped as the haunt receding). GENDYN sparse ghosts. **SIEVE stays ON, reverting only late & incompletely (720→800)** — the math never fully leaves; the resolved clarity is a clarity ON the alien lattice. ARBOR ghost branches. *Thesis: US — comfort built atop roughness, permanently marked. The never-reverting sieve is the audio twin of the monotonic `extractionDebt` latch. The arc loops at 843.*

---

## 6. Conflict resolutions (BINDING)

1. **`EV_GLITCH` value:** `EV_GLITCH = 6` (next free; `EV_LOOP` is not a ring kind in shipped code). Diverges from the Phase-1 doc's `=7` reservation for the documented reason. `islot 0` reserved for Phase-2; this axis owns `islot 1–5`.
2. **GENDYN segment count:** **K=6** (`GSEG=6`), stored entirely in `pm[6]`/`pa[6]`, with velocities in `fcoef`/`fz_`/`vowels` scratch. Resolves the "8 segments needs scratch / cap at 6" split toward the cheaper, alias-clean option. Document the layer-10 alias map at the Voice struct.
3. **Shared state names — declare ONCE each:** `cGendyn_`, `cGendynHarsh_`, `cVoiceSel_`, `sieveScale_`, `cSieve_`, `cArbor_`, `arborTimer_`, `cGlitchDens_`, `cCrush_`, `cStutLen_`, `cGateAmt_`, `cGlitchOp_`, plus the prerequisite `secT_`/`conductAct_`/`FIBSEC`/`FIBACT`. ARBOR reuses `cCloudSpread_` for spread (no second knob). SIEVE reuses `cClusterId_` for the active row (no `sieveSel_`).
4. **Sentinel collision (arbor vs stutter):** arbor uses `tslot=-9`; the glitch-stutter cap_ voice must use a DISTINCT sentinel — **`tslot=-11`** (−7=choir, −9=arbor, −1=default, all free). The stutter wrap logic in `tickVoice()` keys on `-11`; arbor glide keys on `-9`. **This corrects the glitch spec's reuse of `-9`.**
5. **Voice/pool budget when multiple stochastic voices want the foreground:** GENDYN uses `allocCapped(10,2)` (its own layer). ARBOR + GLITCH cap_-voices + cloud + choir + Risset + shred ALL share `allocCapped(2,28)`. Glitch cap_-voices additionally capped **≤6** via a layer-2 census. **Anti-mud foreground rule:** at most one F-stochastic voice — `cVoiceSel_` muxes GENDYN xor WHISPER; arbor-peak demotes GENDYN to affect (amp ×0.5, no solo gate). `busyGain` (L473-474) must add layer-10 to the foreground headroom census.
6. **AudioEngine stays decoupled from `ParticleField`:** sieves built via `buildSievesFromCounts(const int*, int)` fed a plain `int[3]` histogram from `main_reagency.cpp` — not a `ParticleField&`.
7. **AV-sync:** §4 decision (a1) first, FIBSEC reconciliation (a2) flagged as a separate later pass.

---

## 7. Sequenced task list (dependency order)

Each step lands **listenable** before the next (scope risk: GENDYN+sieve+arbor+glitch is the largest single addition — sequence so nothing destabilizes the shipped AV lock).

| # | Task | Depends on | Unlocks / lands |
|---|---|---|---|
| **0** | **PREREQUISITE — conductor time (1a):** add `FIBSEC`/`FIBACT`/`secT_`/`conduct(dt)`; derive `conductAct_`+`f`; adopt AV-sync **(a1)** (key off `act`+`f`, cliff at visual III→IV edge). Verify the existing cut latch still fires correctly. | Phase 1 (shipped) | The within-time clock every technique reads. **No new sound yet** — pure refactor; must regression-test the existing 5-act palette unchanged. |
| **1** | **SIEVES** — `buildSievesFromCounts()` in `init()` (+ histogram in `main_reagency`); rewrite `degHz()`; `cSieve_` step in `conduct()`; story-pedal semitone fix. Validate each baked row has 4–7 notes; fall back to `modeBuf_` if a row degenerates. | #0 | Cheapest, largest reach (one bit re-tunes the whole pitched fabric). Audition the III entry (~377/visual-III) specifically. **The "alien lattice" arrives.** |
| **2** | **GENDYN** — atomics; `fireGendyn()`; `trigger() case 10`; `tickVoice()` layer-10 branch; mix routing; `cVoiceSel_` mux in story-pop; `allocCapped(10,2)`. Verify gamma-mapped harshness reads (cheap-wage stories audibly grittier); verify it self-extinguishes into the cut. | #0 (#1 colors its pitch) | The harm-tone soloist (THEM made harsh). Audition the WHISPER⇄GENDYN alternation in READ. |
| **3** | **ARBORESCENCES** — `cArbor_` atomic + `arborTimer_`; fan emitter after L616; `tslot=-9` glide in `tickVoice()`. Verify `g.body` overwritten after reset; verify the 28-cap holds; verify the fan reads as branching, not random spray. | #0 (#1 colors trunk pitch) | The III fan-out / capture mass. **Crescendo into the 555 cliff.** |
| **4** | **GLITCH AXIS** — `EV_GLITCH=6`; sim atomics + audio state; per-act tables; grid-locked firing after `fireKick()` (L517); bitcrush+micro-gate before tanh (L689); stutter (`tslot=-11`) / reverse-slam / freeze through the cap_ path; `capRms_` silence guard; ≤6 census; islot partition (1–5; 0 reserved). Verify every op fires ONLY on `euStep_` hits; verify the cliff cluster + freeze coordinate with the single existing reverb freeze. | #0, #2/#3 (so it chews a live spine), Phase-2 islot partition | The edit/seam (extraction made audible). **Delivers the cliff; resolves out in V.** |
| **5** | **Mix + anti-mud integration pass** — add layer-10 to `busyGain` foreground census; verify arbor+gendyn never both foreground (mux + demote); re-balance against US (choir/pad) at the cliff; AV-sync audition of the full arc. | #1–#4 | The legible silhouette: ABSENT in EMERGE, PEAK in EXTRACT/TURN, glitch resolves out by end-V, sieve persists. |
| **6** *(optional, later)* | **AV-sync (a2)** — reconcile `main_reagency.cpp:99-102` visual boundaries to FIBSEC so the 555 cut and visual TURN share one source. Broadcast `secT_`/`conductAct_` via `WoSWState`. | #5 | True golden-mean lock between audio cliff and visual turn. |

---

## 8. RT-hazard + determinism + perf table

| Hazard / property | Where | Mitigation |
|---|---|---|
| **Alloc/lock/IO on audio** | all new state | Floats/atomics only; `sieveScale_` baked in `init()` (file IO via the `ParticleField` histogram, sim/main thread); `buildSievesFromCounts` before `ready_` release; no new pools. |
| **Click on cliff/crush ramp/gate** | cut L471, crush/gate L689 | Crush depth glided via atomic target + one-pole (~6 ms); micro-gate attack `gco(0.006)` / release `gco(0.012)`; cut_ already one-pole (L471); stutter/reverse grains use the existing voice atk env (window edges ramped). |
| **Clip on tutti + glitch** | master tanh L689 | `1/sqrt(n)` `busyGain` law (L473-474, now incl. layer-10) + bitcrush feeds the tanh (soft-clips requantization spikes) + GENDYN amp ≤0.14, arbor `amp/sqrt(N)`. |
| **Mud: two foreground stochastic voices** | GENDYN solo + ARBOR fan in III | HARD RULE: `cVoiceSel_` muxes GENDYN xor WHISPER; arbor-peak demotes GENDYN to affect (amp ×0.5, no solo gate). |
| **Layer-2 pool contention** | arbor + glitch + cloud + choir + Risset + shred all `allocCapped(2,28)` | Single 28-cap (oldest-steal); choir ≤8, glitch cap_-voices ≤6 via census; no NVOX (160) change. |
| **Sentinel collision** | arbor `-9` vs stutter | Stutter uses `tslot=-11` (distinct from −7 choir, −9 arbor, −1 default). |
| **`cap_` of silence** (stutter/freeze of dead air) | glitch ops read `cap_` (whisper-only) | `capRms_` one-pole gate; fire stutter/reverse/freeze only when `capRms_>threshold`; crush/gate use full bus. |
| **Grid-lock smear** | glitch via `schedule()`/`pend_` would add 2–156 ms | Fire glitch DIRECTLY in the L517 euPhase block (like `fireKick`); bypass `schedule()`. |
| **Glitch reads as failure not edit** | off-grid glitch | Fire ONLY on `((euMask_>>euStep_)&1u)`; big ops on `euStep_==0`. |
| **Sieve legibility** (too sparse/dense → random) | `buildSievesFromCounts` | Construction guarantees ≥4 notes (force-set `{0,7,3,10}`); validate 4–7 per row; fall back to `modeBuf_` if degenerate. Verified rows from real `points.bin`: {7,6,6} notes. |
| **`degree` semantic remap** under sieve | `degHz()`/`nodeDegree()` | Documented at both functions (the remap IS the dramaturgy); story-pedal register fixed to semitones when `cSieve_` live so year-spread survives. |
| **Freeze + IV freeze_ stacking** (over-long reverb) | `cFreeze_`/`freeze_` L472 | Share the SINGLE reverb-freeze glide; do not add a 2nd. |
| **Phase-2 `EV_GLITCH islot=0` double-fire** | shared `EV_GLITCH` handler | Later axis owns `islot 1–5`; never fires `islot=0`; arbiter suppresses act-glitch if `cEmerge_` mid-transition. Phase-2 audio glitch not live today, but partition built now. |
| **AV-sync drift** | FIBSEC (843, cut 555) vs visual (840, IV@510) | **Decision (a1):** key techniques off `act`+`f`, cliff at visual III→IV edge first; FIBSEC reconciliation (a2) is a separate later pass. |
| **Determinism (dome)** | all new state | Audio is PRIMARY-ONLY → no cross-node audio sync needed. All new state derives from accumulated `dt` + monotonic `storyIdx_`/`phraseIdx_` + fixed-seed RNG (`rng_`/`v.grng`), exactly like Phase 1. `sieveScale_` is a pure function of `points.bin` (identical every node). Broadcast `secT_`/`conductAct_` via `WoSWState` only if a renderer must react. |
| **CPU** | all four | GENDYN: walk once/period + cheap per-sample interp+tanh (< one additive voice; net voice count unchanged via mux). SIEVE: one load+modulo+≤6-search before an already-present `pow` — a few ns. ARBOR: one add+mul+sub/grain/sample, zero new voices. GLITCH: ~4–6 flops/sample on the master bus; stutter/reverse/freeze are existing pooled voices. **Total: negligible; all O(1)/sample, all inside the existing 160-voice / 28-cap budget.** |

---

## 9. Files touched (for the implementation run)

- `reagency/audio/AudioEngine.hpp` — new atomics (`cGendyn_`/`cGendynHarsh_`/`cVoiceSel_`/`cSieve_`/`cArbor_`/`cGlitchDens_`/`cCrush_`/`cStutLen_`/`cGateAmt_`/`cGlitchOp_`), `sieveScale_[3][12]`, audio state (`arborTimer_`, crush/gate/stutter/freeze floats), `secT_`/`conductAct_`, `EV_GLITCH=6`, `GSEG`, method decls (`fireGendyn`, `buildSievesFromCounts`), Voice layer-10 alias comment.
- `reagency/audio/AudioEngine.cpp` — `conduct(dt)` + `FIBSEC`/`FIBACT`; per-act tables (ARB, GLITCH_*); `degHz()` rewrite; `buildSievesFromCounts()`; `fireGendyn()`; `trigger() case 10`; `tickVoice()` layer-10 + arbor-glide + stutter-wrap branches; arbor fan emitter (after L616); grid-locked glitch firing (L517); bitcrush+micro-gate (before L689); mix routing for layer-10; story-pop `cVoiceSel_`/`cGendynHarsh_` + semitone register fix; busyGain layer-10 census.
- `reagency/src/main_reagency.cpp` — cluster histogram from `field.clusterOf(i)` → `audio_.init(...)` overload; AV-sync wiring (pass `cycleT`/keep `act` advisory per decision a1); optional `WoSWState` broadcast of `secT_`/`conductAct_`.
- `reagency/docs/AUDIO_ORCHESTRATION_PLAN.md` — flip the "Later" checkbox / cross-reference this doc.

---

### Critical files for implementation
- /Users/matstudents/allolib_playground/MAT201B_Projects/reagency/audio/AudioEngine.cpp
- /Users/matstudents/allolib_playground/MAT201B_Projects/reagency/audio/AudioEngine.hpp
- /Users/matstudents/allolib_playground/MAT201B_Projects/reagency/src/main_reagency.cpp
- /Users/matstudents/allolib_playground/MAT201B_Projects/reagency/viz/ParticleField.hpp
- /Users/matstudents/allolib_playground/MAT201B_Projects/reagency/docs/AUDIO_ORCHESTRATION_PLAN.md
# WOSW — Audio Orchestration Master Plan

> Source: multi-agent design workflow `wosw-full-orchestration` (9 agents: Research[Xenakis/Eno/glitch/our-data]
> -> Compose[form/instrumentation/voices/data-engine] -> Synthesize). Branch: **v2** (audio uncommitted).

## BUILD STATUS (updated as we go)
- [x] **Phase 1 / 1b — Story spine (THEM):** 142 labor stories loaded; one worker popped per Fibonacci
      phrase; year->register (layer-5 pedal), wage->difference-tone roughness (`cCostBeat_`), era->pan + whisper.
- [x] **Phase 1 / 1c — Sample CHOIR (US):** CC0 bank -> reverb-washed, slow-attack granular background
      (`tslot=-7`, dryg 0.22 / rv 2.8), per-act density.
- [x] **Phase 1 / 1a — Fibonacci conductor:** phrase clock + within-section BUILD (`f` ramps the swarm/beating
      across each section; `actElapsed_`/`SEGDUR`). Act still from the visual cycle for AV-sync + key-audition.
- [x] **Phase 1 / 1d — Eno coprime loops:** 8 loops (3,5,8,13,21,34,55,89) step per beat on pitched layers,
      relative to the worker's register, gated by `cSparsity_` -> never repeats over 14 min.
- [x] **Phase 1 / 1e — Anti-mud:** per-buffer voice census -> `busyGain` headroom law on the master; layer-2
      pool capped (allocCapped 28). **PHASE 1 COMPLETE.**
- [x] **Phase 2 — real ghostly voices + dream-emergence** — CODE COMPLETE (212d4a3, verified in engine):
      2a `rVoice_` router (`AudioEngine.cpp:91`) + real-speech sample on layer-4 EV_WHISPER (`:451`, synth fallback);
      2b `whisperChorus()` (`:241`) + `cChorusN_`/`cFmtShift_` conductor-driven (`:322`); 2c `cEmerge_` (`:335`) +
      per-step `EV_GLITCH=6` (`:336/:421`). REMAINING: offline-baked voice WAVs (graceful synth fallback until then).
- [ ] **Later — deeper Xenakis (GENDYN, sieves, arborescences) + glitch axis.** (Absent from engine — see `AUDIO_LATER_PLAN.md`.)

---

All grounded: 192 samples + ATTRIBUTION (193 entries), 500 words, stories.json present with year/era fields. The engine read confirms every insertion point. Here is the synthesized build plan.

---

# WOSW — The Orchestrated Build Plan

**One spine, three techniques, one emotional core.** The **Fibonacci conductor + data** is the form (it owns time and pops one named worker per phrase). **Xenakis / Eno / glitch** are the *techniques the conductor wields* — never autonomous FX knobs. The **ghostly voices (THEM)** are the emotional payload every technique serves. The current engine is "drone+FX" for one structural reason confirmed in the read: `update()` is **stateless in time** — it reads only the `act` arg and writes *constant* `ACTG/ACT_*` palettes (L216–256) with 20% random `dropTgt_` flicker (L264–273). Nothing knows where in 14 min it is. **Fix that one thing and everything else becomes composition.**

**Conflict resolutions (binding):**
- Multiple specs add `conduct()`, `secT_`, `stories_`, `cEmerge_`, `cStoryDeg_`, `cSparsity_` — **declare each ONCE** (the data-engine spec's block is canonical).
- Two specs claim `EV_LOOP=6` AND `EV_GLITCH=6`. **Resolved:** `EV_LOOP=6`, `EV_GLITCH=7`, `GENDYN` reuses `EV_NOTE layer=10` (no ring change). Bump `enum` at L62.
- The CC0 choir is **layer 2, sentinel `v.tslot==-7`**, routed to a CHOIR tier in the mix branch — distinct from the existing grain pool so it doesn't fight the cap.
- `act` arg → **advisory only**; `secT_` is authoritative (`conductAct_` derived from `FIBSEC`). Keeps DistributedApp determinism: drive off accumulated `dt` + monotonic `storyIdx_`, all sim RNG fixed-seeded already.

---

## PHASE 1 (build now) — audibly transforms it on first listen, zero new offline assets

Five moves. All sim-side floats set in `update()/conduct()` + read in existing `render()` code. **No new ring kinds except `EV_LOOP`. No alloc/lock/IO on audio.**

### 1a. The conductor (the single highest-impact change)
**`.hpp` ~L108 (sim state):**
```cpp
float secT_=0.f, phraseT_=0.f; int conductAct_=1, subIdx_=0;
size_t storyIdx_=0;
struct Story{ float yearN, costN; int era; }; std::vector<Story> stories_;
std::atomic<float> cStoryDeg_{0.f}, cCostBeat_{0.f}, cEmerge_{0.f}, cSparsity_{0.6f};
```
**`.cpp` top (file-scope):**
```cpp
static const float FIBSEC[12] = {89,144,233,322,377,466,521,555,576,631,720,843};
static const int   FIBACT[12] = { 1,  1,  1,  2,  2,  3,  3,  3,  4,  4,  5,  5};
```
**`loadStories()`** — new method called in `init()` after L141, mirrors `loadWords()`: `fgets` each line of `v2/stories.json`, `atof` the `"year"`, `strtod` first `$` in `"figure"`, `strstr(line,"contemporary")`. `yearN=(year-1770)/255`; `costN=clamp01(1 - log10(max($,0.5f))/5.4f)` (cheap labor → high cost). Corpus verified: **143 stories, 68 contemporary / 75 historical, years 1770–2025.**

**`conduct(dt)`** — inserted as the **first statement in `update()` after L213**, *replacing the constant `aGate`/`cBeat_`/`cCloudLam_` stores with ramped ones:*
```cpp
secT_ += dt; phraseT_ += dt;
int s=0; while(s<11 && secT_>=FIBSEC[s]) ++s;
conductAct_ = FIBACT[s];
float segStart = s? FIBSEC[s-1]:0.f, f = clamp01((secT_-segStart)/std::max(1.f,FIBSEC[s]-segStart));
const int a = conductAct_;            // <-- use this instead of the act arg below
```
Then **lerp** existing targets across `f` instead of slamming constants. E.g. cloud λ ramps within III: `cCloudLam_.store(CL[a] + f*(CL[std::min(a+1,5)]-CL[a]))`. The cut latch (L258) now fires on `secT_` crossing 555, not on the host flipping `act`.

| sub | t(s) | act | cCloudLam | cSparsity | cEmerge | dominant gates |
|---|---|---|---|---|---|---|
| EMERGE | 0–89 | I | 2 | 0.85→0.6 | 0→1 | DRONE only, noise→pad |
| SEDUCE | 89–233 | I | 2→3 | 0.6→0.4 | 1 | MEL,BELL,DRONE + choir |
| READ | 233–377 | II | 4→6 | 0.5 | 1 | WHISPER,BELL |
| EXTRACT | 377–521 | III | 13→40 | 0.1 | 1 | KICK,GRAIN,TIMP |
| TURN(cut@555) | 521–631 | IV | 40→2 | 0.85 | 1 | KICK,WHISPER only |
| INTEGRATE | 631–843 | V | 2 | 0.6→0.9 | 1→0→1 | thin, topcut→1 |

### 1b. Story spine → THEM (one worker per phrase)
In `conduct()`, every `phraseT_ >= phraseLen` (phraseLen = Fibonacci beats; ~6 s ⇒ ~140 phrases) pop `stories_[storyIdx_++ % N]` and:
- `cStoryDeg_.store(int((s.yearN*2-1)*modeN_))` — historical→low register, contemporary→high. Read it in the whisper push (`whisper()` pan/pitch) and as a layer-5 pedal.
- `cCostBeat_.store(s.costN)` → **add into the existing `cBeat_.store`** (L239): `cBeat_.store(ACT_BEAT[a] + 0.04f*cCostBeat_)`. Cheaper wage = rougher difference-tone beating. **The $1.32/hr stories literally sound the harshest.**
- `cEraSel` biases `pickSample`: historical → `rTrace_/rBass_` (psaltery/organ), contemporary → grain/granular. Push one `whisper("", node, pan)` per story.

### 1c. The sample-bank → reverb-washed granular BACKGROUND CHOIR (the user's explicit request)
**New atomic `.hpp`:** `std::atomic<float> cChoir_{0.f}, cChoirLam_{0.f};`
**New audio-thread emitter** inserted at **L551** (right before the granular-shred block), mirroring the grain-cloud Poisson loop:
```cpp
choirTimer_ -= isr; float clam = cChoirLam_.load(std::memory_order_relaxed);
if (clam>0.05f && choirTimer_<=0.f){ choirTimer_ += -std::log(std::max(1e-6f,frand()))/clam;
  int gi=allocCapped(2,28); if(gi>=0){ Voice& g=vox_[gi]; g=Voice{};
    Role& R = (frand()<0.6f)?rTrace_:rBass_; int sidx=pickSample(R, degHz(nodeDegree(int(rng_&2047),-1)));
    if(sidx>=0){ const Sample& smp=samples_[sidx]; g.on=true; g.layer=2; g.tslot=-7;   // CHOIR sentinel
      g.smp=smp.data.data(); g.smpLen=int(smp.data.size()); g.smpPos=0.f;
      g.smpRate=(smp.srcSR/float(sr_))*(smp.pitched? std::pow(2.f,float((int(frand()*3)-1)*12)/12.f):1.f);
      g.amp=0.05f*cChoir_.load(std::memory_order_relaxed); g.life=1.5f+2.5f*frand(); g.atk=0.3f+0.5f*frand();
      g.phase=frand(); g.pan=0.9f*(2.f*frand()-1.f); } } }
```
**New mix branch** in the pooled loop at **L584** (before the generic `else if (v.smp)`):
```cpp
if (v.smp==cap_.data()) { dryg=1.2f; rv=0.8f; }
else if (v.tslot==-7)   { dryg=0.22f; rv=2.8f; sampSend += s*1.4f; ppSendL+=0.3f*s*cL; ppSendR+=0.3f*s*cR; }  // CHOIR: washed
else if (v.smp)         { dryg=0.38f; rv=2.1f; }
```
`cChoirLam_`/`cChoir_` driven by the conductor (table above): historical sections bias `rTrace_/rBass_` (cello/psaltery/organ), contemporary bias `rPluck_/rMetal_`. The existing `gmod` granulation (L351–352) already chops these into overlapping grains — long `life` + slow `atk` = washes, not hits. **Cap ≤ 8** via a census (move 1e). This is "US — the comfortable background built atop their voice."

### 1d. Eno coprime loops (the breath)
**`.hpp`:** the 8-loop arrays exactly as the Eno spec (`NLOOP=8`, `loopLen_={3,5,8,13,21,34,55,89}`, `loopRate_`, `loopLayer_={6,7,1,0,2,3,5,8}`, `loopAmp_`, phases/steps). **`EV_LOOP=6`** in enum (L62), `fireLoop(L)` builds an `Ev{layer=loopLayer_[L]…}`→`schedule()`. **Per-sample in render at L457** (right after the existing euPhase2 block), the 8-loop stepper from the spec, **gated by `cSparsity_`**: loop L silenced when `cSparsity_ > L/NLOOP`. At peak sparsity (EMERGE/TURN/late-INTEGRATE) only loops 0–1 survive = bare phasing pulse. Product of lengths ≈ 5.6×10⁹ steps ⇒ **never repeats in 14 min**, no automation.

### 1e. Anti-mud orchestration (makes tutti legible, not a smear)
Before the pooled loop at **L568**: census `foreN` (layers 4 + sample-voice), `midN` (layer 2). **Hard-cap** by lowering the `allocCapped` cap dynamically; when `foreN >= 2`, scale those voices' amp by `1/sqrt(foreN)` (energy-preserving, keeps tutti off the `tanh` ceiling). Choir capped ≤8 via a `choirN` count in the same census. **Register discipline:** the existing `+12` on the tune (L282) already lifts pluck above a mid whisper — keep.

**PHASE 1 RT-hazard table:**
| Hazard | Where | Mitigation |
|---|---|---|
| Click on cut/gate | cut_ L411 / gate_ glide L409 | already one-pole glided (50 ms / `cGateTau_`); choir uses slow `atk` 0.3–0.8 s |
| Clip on tutti | master `tanh` L607 | `1/sqrt(n)` amp law (1e) + choir amp 0.05 |
| Mud (choir+cloud overlap) | L551/L544 both alloc layer-2 from cap 28 | choir capped ≤8 in census; `cSparsity_` thins loops; conductor never runs cloud-peak + choir-peak together |
| Reverb feedback | sampSend×1.4 (choir) into sampRev_ | `sampRev_` decay 0.62 (short); keep choir count ≤8 |
| Alloc/lock | all new state | floats/atomics only; `loadStories` runs in `init()` (sim), never audio |
| Determinism desync | conduct on dome nodes | drive off accumulated `dt` + monotonic `storyIdx_`; fixed RNG seeds (already); push `secT_/storyIdx_/conductAct_` into `SimpleSharedState` |

---

## Phase 2 — needs offline-baked assets (dream frames + real CC0 voices)

- **Real ghostly voices (THEM made literal).** Offline factory bakes PD/CC0 spoken-word (LibriVox labor texts, CC0 Freesound murmur, our TTS of the 143 `oneLiner`s) → `assets/audio/*voice*.wav`. Add **one `Role rVoice_`** + one router line at `loadSamples` L85 (`has("voice")||has("speech")||has("ghost")`, `pitched=false`). New `EV_WHISPER` variant `e.islot<0` ⇒ in `trigger()` layer-4 (L337) play a real-voice **sample** voice that *shares layer 4's duck + `cap_` capture + granulator* — so the granular shred (L552) chews REAL human speech. The thesis made audible.
- **Synth whisper → plural haunted chorus.** `whisperChorus(word,n,spread)` pushes n=2–5 detuned/staggered `EV_WHISPER` (atomics `cChorusN_`,`cChorusDet_`,`cFmtShift_`(place→throat),`cVoiceRough_`(wage→strain),`cGranSmear_`). In `tickVoice` layer-4: one multiply `v.fmt[k]*=cFmtShift_`, `bw[k]*=(1+0.5*rough)`. 143 places = 143 throats; peaks in V (all ghosts at once).
- **Dream emergence → image forming from noise.** Per-denoise-step frames set `cEmerge_` 0→1; read inversely in render (`shepGain_*=(1-cEmerge_)`, `cGranScat_*=(1-cEmerge_)`, pad floor `*=cEmerge_`) — crossfade Shepard-noise→harmonic pad. Drives the opening 89 s + V recap. Each denoise step also fires `EV_GLITCH islot=0` with shrinking `cStutLen_`/falling `cCrush_` — **the image resolves out of glitch into clarity.**

## Later — deeper Xenakis + glitch (after Phase 1/2 land cleanly)

- **GENDYN layer 10** — stochastic-breakpoint voice (`EV_NOTE layer=10`, reuse `pm[]/pa[]` as walked segments). The "harm-tone" whose harshness scales with story `figure`; alternates with WHISPER as soloist (`cVoiceSel_`). ~8 segments, negligible CPU.
- **Sieves** — `sieveMask()` from galaxy cluster sizes (`points.bin`) → `int8_t sieveScale_[3][12]`; `degHz` reads it when `cSieve_>0.5` (ACTs III–V = "machine math" alphabet, I/II stay diatonic).
- **Arborescences** — give cloud grains a glide target (reuse `v.body` slot) + fan endpoints `±spread·i/N` → Metastaseis branching, `cArbor_` peaks in III.
- **Glitch axis** (`EV_GLITCH=7`): bitcrush at L569, stutter/reverse-slam/freeze off `cap_` near L553, master micro-gate at L600 — all gated to the Euclidean grid so they read as *edits*, not noise; density follows `cActivity_`, loud only in III/IV.

**Thesis payoff, integrated:** ME (the synth machinery) → YOU (the whisper you hear) → THEM (each named worker: their `year` sets the pitch, their `wage` sets the dissonance via `cCostBeat_`, their `era` picks the timbre family) → US (the reverb-washed sample choir, the harmonic pad the dream resolves into — comfort built atop all that roughness). One 14-min arc, one structural silence at the golden mean (555 s), one shaped crescendo-and-cliff instead of a flat plateau.

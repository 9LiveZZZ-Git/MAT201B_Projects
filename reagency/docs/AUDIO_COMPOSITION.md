# World of Shadow Work — The Audio Composition

*A plain-language analysis of the sound: what it is, how it's built, the techniques behind it, and
how every one of them serves the meaning. The score is generated live — nothing is a fixed
recording — so this describes the system and its behavior, not a single take.*

---

## 1. What the music is

It's a **generative score for a 7½-minute ritual loop** — a continuous, ever-shifting wash of
drone, tuned tones, grain clouds, and ghostly human voices that never plays the same way twice. It
is not "ambient background." It is **a piece of data-driven instrumental music** whose form, pitch,
dissonance, and timbre are all *played by the artwork's own data* — the images, the worker stories,
the shape of the galaxy.

The whole thing is organized around one sentence: **ME → YOU → THEM → US.**
- **ME** — the machine's own sound (the synthesis: drones, tones, grains). It *seems* to speak, but it can't really confess.
- **YOU** — the lead voice you actually hear and follow.
- **THEM** — the real, named workers who labeled the data. Their lives are turned into *musical parameters* (below).
- **US** — the warm, reverb-washed comfort layer (a choir, a harmonic pad) — the beauty built *on top of* THEM, which never quite covers them.

Every instrument and every technique is assigned to one of those roles. Nothing is a free "effect."

---

## 2. The form — data owns time

Most generative music drifts. This one has a **conductor**: a clock that *advances with the
piece's own logic* and divides the loop into five acts, each ~90 seconds:

| Act | ~Time | Character |
|---|---|---|
| **I — Seduction** | 0:00–1:30 | Lush, beautiful, inviting. A bed forms out of noise into a warm chord; the tune and bells enter. |
| **II — Reading** | 1:30–3:00 | The machine begins *classifying*; the lead voice takes over. |
| **III — Extraction** | 3:00–4:30 | The grind. Dense, fast, dissonant — the "cost" surfaces; a crescendo. |
| **IV — The Turn** | 4:30–6:00 | Stripped to a bare pulse. The mirror moment. |
| **V — Residue** | 6:00–7:30 | Brightens again — but only *partway*. The comfort returns marked by what came before. Then it loops. |

Three structural ideas shape this arc:

- **Crescendo-and-cliff, not a plateau.** The energy *builds* across I→III into a peak, then drops sharply at the III→IV turn — rather than sitting at one level the whole time. Within each act there's also a smaller swell (the texture thickens across the act, then resets).
- **The golden-mean turn.** The most important structural event — the drop into the bare Act-IV pulse — is placed near the *golden ratio* of the arc (a classical proportion), so the shift lands where the ear expects weight.
- **The comfort-fail + seamless loop.** Act V deliberately **does not** return to the Act-I bliss — it re-brightens only partway and stays marked by the harshness underneath (the "comfort built on roughness" idea). The end of V and the start of I sit at the *same* depth, so the loop has no hard reset: a visitor arriving mid-cycle enters at the same partial state a departing one left.

The clock also **"breathes"** — it runs slightly slower when the machine "hesitates," so the ritual feels alive rather than metronomic (kept subtle so each loop still lands near 7:30).

**One worker per phrase.** The conductor pops one named worker roughly every musical phrase, and *that worker's data becomes the music* for the phrase (next section). Over a loop you hear a procession of real people, one at a time.

---

## 3. How the data becomes music

This is the heart of it — the piece is **literally performed by the corpus and the worker stories:**

- **A worker's year → pitch / register.** Historical workers sound *low*; contemporary workers sound *high*. The timeline is mapped onto the keyboard.
- **A worker's wage → dissonance.** Cheaper labor = **rougher** sound. This is done with *difference-tone beating* (two close pitches that "beat" against each other) — the lower the wage, the harsher the beat. **The $1.32/hour stories literally sound the most painful.**
- **A worker's era → timbre.** Historical workers are voiced by warmer, older-sounding instruments (psaltery/organ/cello-like samples); contemporary workers by grainier, more digital textures.
- **The shape of the galaxy → the scale itself.** The pitch *alphabet* the machine uses is computed from the **sizes of the image clusters** in the data (a Xenakis "sieve" — see §4). In the early acts the music is sweet and diatonic; from Act III the scale silently swaps to this **alien, data-derived lattice** — the same melodies, now landing on a machine's math. The sweetness *curdles* without you consciously noticing why.

So the music isn't *about* the data — it **is** the data, sounding.

---

## 4. The techniques (the lineage)

The sound design draws on a specific tradition of 20th-century and generative composition. In plain terms:

**Just intonation drone (the bed).** The sustained chord underpinning everything is tuned in *pure
whole-number ratios* (just intonation) rather than the piano's even temperament — so it has a still,
resonant, slightly otherworldly purity. It slowly moves, never quite static.

**Iannis Xenakis — four techniques, the "machine math" of the piece:**
- **Stochastic grain clouds.** Showers of tiny sonic grains scattered by *probability* (a Poisson process — random but with a controllable density). Sparse and twinkling early; dense and grinding in Extraction. This is *granular synthesis* used as weather, not melody.
- **GENDYN (Dynamic Stochastic Synthesis).** A raw, buzzy "harm-tone" voice whose very *waveform* is drawn by a random walk that bounces off elastic barriers — Xenakis's most radical idea, sound built from noise at the sample level. It's the opposite of the sweet pad: it's THEM made *harsh*, and it trades off with the human whisper so there's only ever one "soloist."
- **Arborescences (the *Metastaseis* fan).** A single tone *branches* into a sheaf of gliding lines spreading outward — one voice exploding into a statistical mass. This is the sound of *capture*: one attended worker becoming a swarm. It peaks in Extraction.
- **Sieves.** The pitch lattice described in §3 — Xenakis's formal method for building scales from number patterns, here fed by the galaxy's cluster sizes (one of the data clusters even produces a *whole-tone* scale, a classic "machine/symmetry" signature).

**Brian Eno — generative coprime loops (the breath).** Eight short melodic loops of *coprime
lengths* (3, 5, 8, 13, 21, 34, 55, 89 beats) run simultaneously. Because their lengths share no
common factor, the combined pattern **never repeats** over the whole piece — endless variation with
no automation. A "sparsity" control thins them: at the quietest moments only the two shortest
survive, leaving a bare phasing pulse.

**Shepard/Risset tones (the eternal-rising illusion).** A bed of tones engineered to sound like
they're *forever rising* (or forever accelerating) without ever getting higher — the auditory
equivalent of an Escher staircase. It carries the opening "forming from noise" and recedes as the
image resolves.

**Granular washing.** The "US" comfort choir is made by taking real sample sounds and shredding them
into long, slow, overlapping grains drowned in reverb — a soft, enveloping haze. The same granulator
also *chews the human voice* (see §5).

**The glitch axis (the edit / the seam).** A set of digital-editing gestures — bitcrush, stutter,
reverse-slam, freeze, micro-gate — all chopping up the *captured human voice*, all locked to the
rhythmic grid so they read as a producer's deliberate edits rather than as errors. The concept: **the
machine editing the human record, the seam of unpaid labor showing.** *(Note: this was dialed
heavily back in tuning — see §8 — because in practice it read as annoying dropouts and buried the
composition. What remains is a single subtle stutter at the III→IV turn.)*

**Orchestration discipline (so the tutti stays legible).** When many voices stack up, an
*energy-preserving headroom law* (each voice scaled by 1/√n) keeps the full texture off the
distortion ceiling, and the layers are kept in distinct registers so the lead voice always cuts
through. This is what keeps the dense Extraction from becoming mud.

---

## 5. The voices — THEM made audible

The most important sound is the **THEM voice**: short fragments of real worker testimony (spoken
samples), placed one worker at a time by the conductor. When a real voice plays, the granular engine
can *shred it* — re-trigger, reverse, and stutter the captured speech — so you literally hear **the
machine editing a human's words.** When no recorded voice is available it falls back to a synthesized
formant "whisper," and in the late acts the whisper can bloom into a **plural haunted chorus** (many
detuned throats at once — "143 places, 143 throats," peaking in the residue as all the ghosts return).

The lead voice is mixed **forward**, clearly above the bed, so THEM stays the focus — it's the
emotional payload the whole machine is built to deliver.

---

## 6. The mix and the space

- **Master tone.** The output is tilted toward a warm "pink" balance (a gentle low-shelf lift + high
  roll-off) and soft-clipped (a `tanh` saturator) so peaks round over rather than harshly clipping.
- **Reverb.** Two reverbs — a tighter one and a long wash — give the dry voices presence and the
  choir/pad their cathedral haze. A "freeze" can hold a reverb tail ringing.
- **Low end.** The sub-bass is its **own** voice (a deep ~35 Hz triangle + sub pedal), deliberately
  *decoupled* from the sustained pad — it has its own per-act pulse, level, and bass-note motion, so
  the bottom of the mix has rhythm and movement instead of a static hum.
- **Level.** A global **−10 dB output trim** rides on top (a workaround for a broken house gain) — a
  true −10 dB applied after the soft-clip, on every channel.
- **Real-time discipline.** Everything that touches the audio thread is allocation-free and lock-free
  — the "what to play" decisions happen on the simulation thread and are passed as simple values, so
  the sound never stutters from the engine itself.

**The AlloSphere (dome) spatialization.** On the dome the mono/stereo mix is spread across the
54-speaker sphere:
- **Placement** uses *layer-based amplitude panning* (the dome's intended method) — sounds are panned
  to the nearest real speakers, so they localize cleanly rather than smearing everywhere (an
  Ambisonics decode was tried first and made a mess on the irregular speaker rig).
- The **THEM voice is placed at its node in the galaxy** — a worker's voice sounds *from the spot in
  the image-cloud you're looking at*.
- The **bed, reverb, and master texture are decorrelated** and spread across all the speakers as a
  diffuse, enveloping field (so the ambience wraps the room rather than pointing at one spot).
- The **bass goes to the dedicated subwoofer** (low-passed, on its own channel).
- The mix stays on the **horizontal plane** — height isn't used for placement, because the dome
  doesn't localize elevation well.

---

## 7. Act-by-act listening guide

- **I — Seduction (0:00–1:30).** Out of a hush, a Shepard-noise bed *forms into* a warm just-intonation chord; the tune and bells arrive; the first worker voices appear softly; the choir enters. Sweet, diatonic, inviting.
- **II — Reading (1:30–3:00).** The lead voice leads; the texture is at its lushest; the machine is "reading" the images (you see classification words appear). Still consonant — *the last sweet section*.
- **III — Extraction (3:00–4:30).** The grind: dense grain clouds, the arborescence fans peaking, the kick and a second rhythmic line driving, the pitch lattice **silently swapping to the alien sieve**, the worker beating getting rougher. A crescendo of cost.
- **IV — The Turn (4:30–6:00).** Everything thins to a bare, hushed pulse — the comfort layer drops away and the degraded voice is left exposed. Marked by a brief glitch-stutter at the entry (instead of the old hard silence). The mirror: *you used it; did you look for them?*
- **V — Residue (6:00–7:30).** The choir and pad return — but on the *alien lattice* (the sieve never fully reverts), and only partway bright. Worker voices linger as ghosts. Comfort permanently marked by roughness. Then the loop seam carries you back to I at the same depth.

---

## 8. Current state & tuning notes (honest)

The full system — conductor, the four Xenakis techniques, the Eno loops, the voices, the choir, the
spatialization — is **implemented and runs clean**, but most of it has had only limited end-to-end
auditioning, so the *balance* is still being tuned by ear. Recent tuning decisions, in case they
matter to a listener or a future editor:

- **No dead air / no annoying drops.** The old hard structural *cut* (a momentary silence at the
  III→IV turn) was removed — it read as a mistake — and replaced by a single subtle glitch-stutter.
  The pervasive glitch-axis ducking (a rhythmic master gate) and the per-frame dream-emergence
  stutters were **disabled** because they fought the composition. A faint continuous bed always
  remains, so the piece never fully stops.
- **Voices forward** (+~5 dB over the original) so THEM sits clearly out front.
- **−10 dB overall** to compensate the broken house gain.
- **Condensed from 14 min to 7½ min** (five equal ~90 s acts), keeping the depth/shape and the
  comfort-fail — just compressed.
- **Still on the to-do list:** a proper full listen on the dome (the 54-speaker result and the
  spatial placement can only be judged in the room), and fine balancing of the per-act palettes.

**The thesis, in one line of sound:** the machine (ME) plays you a beautiful surface (YOU), built
from real workers reduced to parameters (THEM), under a comfort that never quite covers them (US) —
and at the golden mean it strips the comfort away and leaves the roughness showing.

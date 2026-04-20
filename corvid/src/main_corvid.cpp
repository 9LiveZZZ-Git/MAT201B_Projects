// Corvid M1/M2 alpha demo — v0.3
// Energy · Reproduction · Death · Place grid (Kahan EMA) · Entities · Memory rings
#include "al/app/al_App.hpp"
#include "al/graphics/al_Shapes.hpp"
#include "al/math/al_Random.hpp"
#include "al/math/al_Vec.hpp"
#include "al/scene/al_PolySynth.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"
#include "Gamma/Envelope.h"
#include "Gamma/Oscillator.h"

#include "core/Agent.hpp"
#include "core/Memory.hpp"
#include "core/MemoryRing.hpp"
#include "core/Place.hpp"
#include "core/SpatialHash.hpp"
#include "cognition/Perception.hpp"
#ifdef CORVID_USE_RAVENNET
#include "cognition/RavenBrain.hpp"
#endif
#include "environment/AcornPlant.hpp"
#include "environment/BoulderObstacle.hpp"
#include "environment/HawkPredator.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <vector>

using namespace al;
using namespace corvid;

// ---------------------------------------------------------------------------
// BeepVoice — single-cycle sine burst for birth / death / food events
// ---------------------------------------------------------------------------
class BeepVoice : public SynthVoice {
    gam::Sine<> osc;
    gam::AD<>   env{0.002f, 0.07f};
public:
    void init() override {
        createInternalTriggerParameter("freq", 440.f, 20.f, 8000.f);
    }
    void onProcess(AudioIOData& io) override {
        osc.freq(getInternalParameterValue("freq"));
        while (io()) {
            float s = osc() * env() * 0.07f;
            io.out(0) += s;
            if (io.channelsOut() > 1) io.out(1) += s;
        }
        if (env.done()) free();
    }
    void onTriggerOn() override { env.reset(); }
};

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
struct CorvidM1 : App {
    static constexpr int   N_POOL  = 256;
    static constexpr float W       = 10.f;
    static constexpr float HALF_W  = W * 0.5f;
    static constexpr float REPRO_E = 0.62f;   // min energy to reproduce
    static constexpr float REPRO_AGE = 15.f;  // min age before first repro
    static constexpr float REPRO_CD  = 5.f;   // cooldown between reproductions
    static constexpr float REPRO_R   = 1.0f;  // max range for mate detection

    // --- agent pool ---
    std::array<Agent,  N_POOL> pool{};
    std::array<Vec3f,  N_POOL> vel{};    // steering velocity
    std::vector<int>           free_list;
    int                        n_live = 0;

    // --- place grid ---
    std::array<Place, PLACE_GRID_CELLS> places;

    // --- M2: per-agent memory rings + fixed encoder ---
    std::array<MemoryRing<256>, N_POOL> mem_rings{};
    FixedEncoder encoder;
    int cognition_counter = 0;  // increments each frame; fires perception at 5 Hz

    // --- M3: RavenNet trunk + per-agent action biases ---
#ifdef CORVID_USE_RAVENNET
    RavenBrain brain;
    // action_biases[slot][6]: biases per force (align,sep,cohere,pred,food,obs)
    // validity decays linearly over 200 ms then zeroes (spec §2.3.1)
    std::array<std::array<float, 6>, N_POOL> action_biases{};
    std::array<float, N_POOL> bias_age{};  // seconds since last forward for this slot
    Parameter w_action{"ActionBias", "Forces", 2.0f, 0.f, 4.f};
#endif

    // --- entities ---
    std::vector<std::unique_ptr<Entity>> entities;
    std::vector<HawkPredator*>           hawks;   // borrowed ptrs

    // --- spatial hash ---
    SpatialHash hash;

    // --- audio ---
    PolySynth synth;

    // --- GUI + parameters ---
    ControlGUI gui;
    Parameter w_align    {"Align",      "Forces", 1.0f,  0.f, 5.f};
    Parameter w_sep      {"Separate",   "Forces", 1.5f,  0.f, 5.f};
    Parameter w_cohere   {"Cohere",     "Forces", 1.0f,  0.f, 5.f};
    Parameter w_predator {"AvoidHawk",  "Forces", 3.0f,  0.f, 8.f};
    Parameter w_food     {"SeekFood",   "Forces", 1.5f,  0.f, 5.f};
    Parameter w_obstacle {"AvoidRock",  "Forces", 4.0f,  0.f, 8.f};
    Parameter e_drain    {"EnergyDrain","Sim",    0.020f, 0.f, 0.08f};
    Parameter view_r     {"ViewRadius", "Sim",    1.5f,  0.5f, 5.f};
    Parameter max_spd    {"MaxSpeed",   "Sim",    2.5f,  0.5f, 8.f};

    // --- stats ---
    int      n_born = 0, n_dead = 0;
    float    sim_time   = 0.f;
    uint32_t next_id    = 1;
    uint32_t next_lin   = 1;   // next lineage id

    // shared tetrahedron mesh (populated in onCreate)
    Mesh tetra_m{Mesh::TRIANGLES};

    // ---------------------------------------------------------------------------
    // Helper: toroidal wrap a Vec3f into [-HALF_W, HALF_W]
    // ---------------------------------------------------------------------------
    static void wrapPos(Vec3f& p) {
        for (int i = 0; i < 3; ++i) {
            if (p[i] >  HALF_W) p[i] -= W;
            if (p[i] < -HALF_W) p[i] += W;
        }
    }

    // Shortest toroidal delta from a to b
    static Vec3f toroidalDelta(const Vec3f& a, const Vec3f& b) {
        Vec3f d = b - a;
        for (int i = 0; i < 3; ++i) {
            if (d[i] >  HALF_W) d[i] -= W;
            if (d[i] < -HALF_W) d[i] += W;
        }
        return d;
    }

    // ---------------------------------------------------------------------------
    // Helper: trigger a quick beep (rate-limited in caller)
    // ---------------------------------------------------------------------------
    void beep(float freq) {
        auto* v = synth.getVoice<BeepVoice>();
        if (!v) return;
        v->setInternalParameterValue("freq", freq);
        synth.triggerOn(v);
    }

    // ---------------------------------------------------------------------------
    // Helper: spawn an agent at pos with given lineage (0 → new lineage)
    // ---------------------------------------------------------------------------
    int spawnAgent(Vec3f pos, float energy, uint32_t lineage) {
        if (free_list.empty()) return -1;
        int slot = free_list.back(); free_list.pop_back();
        Agent& a       = pool[slot];
        a.id           = next_id++;
        a.lineage_id   = lineage ? lineage : next_lin++;
        a.generation   = 0;
        a.live         = true;
        a.birth_t      = sim_time;
        a.death_t      = -1.f;
        a.energy       = energy;
        a.last_reproduce_t = -999.f;
        a.flash_timer  = 0.35f;
        a.flash_kind   = 0;   // green birth flash
        a.nav.pos(Vec3d(pos.x, pos.y, pos.z));
        vel[slot] = Vec3f(rnd::uniformS(), rnd::uniformS(), rnd::uniformS()).normalize() * 0.8f;
        ++n_live; ++n_born;
        return slot;
    }

    // ---------------------------------------------------------------------------
    // Helper: kill an agent (writes place event, triggers sound)
    // ---------------------------------------------------------------------------
    void killAgent(int slot) {
        Agent& a = pool[slot];
        if (!a.live) return;
        a.live    = false;
        a.death_t = sim_time;
        a.flash_timer = 0.35f;
        a.flash_kind  = 1;   // red death flash
        Vec3f p = Vec3f(float(a.nav.pos().x),
                        float(a.nav.pos().y),
                        float(a.nav.pos().z));
        writePlace(places, p, MK_DEATH_WITNESSED, -0.7f, HALF_W);
        beep(200.f);
        free_list.push_back(slot);
        --n_live; ++n_dead;
    }

    // ---------------------------------------------------------------------------
    // onCreate
    // ---------------------------------------------------------------------------
    void onCreate() override {
        nav().pos(0, 1, 22);
        nav().faceToward({0, 0, 0});

        gam::sampleRate(audioIO().framesPerSecond());
        encoder.init();
#ifdef CORVID_USE_RAVENNET
        {
            RavenNetConfig cfg;
            cfg.n_agents = N_POOL;
            brain.init(cfg);
        }
#endif

        // Build tetrahedron mesh once
        addTetrahedron(tetra_m);

        // Place grid
        initPlaces(places, HALF_W);

        // Free-list: all slots are free initially
        free_list.reserve(N_POOL);
        for (int i = N_POOL - 1; i >= 0; --i) free_list.push_back(i);

        // Spawn initial agents
        for (int i = 0; i < 60; ++i) {
            Vec3f p(rnd::uniform(-HALF_W, HALF_W),
                    rnd::uniform(-HALF_W, HALF_W),
                    rnd::uniform(-HALF_W, HALF_W));
            spawnAgent(p, rnd::uniform(0.5f, 0.9f), 0);
        }

        // 20 AcornPlants scattered around
        for (int i = 0; i < 20; ++i) {
            Vec3f p(rnd::uniform(-HALF_W * 0.9f, HALF_W * 0.9f),
                    rnd::uniform(-HALF_W * 0.9f, HALF_W * 0.9f),
                    rnd::uniform(-HALF_W * 0.9f, HALF_W * 0.9f));
            entities.push_back(std::make_unique<AcornPlant>(p));
        }

        // 5 BoulderObstacles
        for (int i = 0; i < 5; ++i) {
            Vec3f p(rnd::uniform(-HALF_W * 0.7f, HALF_W * 0.7f),
                    rnd::uniform(-HALF_W * 0.7f, HALF_W * 0.7f),
                    rnd::uniform(-HALF_W * 0.7f, HALF_W * 0.7f));
            float r = rnd::uniform(0.3f, 0.55f);
            entities.push_back(std::make_unique<BoulderObstacle>(p, r));
        }

        // 3 HawkPredators
        for (int i = 0; i < 3; ++i) {
            Vec3f p(rnd::uniformS() * HALF_W,
                    rnd::uniformS() * HALF_W,
                    rnd::uniformS() * HALF_W);
            auto hawk = std::make_unique<HawkPredator>(p, W);
            hawks.push_back(hawk.get());
            entities.push_back(std::move(hawk));
        }

        // Pre-allocate voice pool
        synth.allocatePolyphony<BeepVoice>(24);

        // GUI
        gui << w_align << w_sep << w_cohere
            << w_predator << w_food << w_obstacle
            << e_drain << view_r << max_spd;
#ifdef CORVID_USE_RAVENNET
        gui << w_action;
#endif
        gui.init();
    }

    // ---------------------------------------------------------------------------
    // onAnimate — main simulation step
    // ---------------------------------------------------------------------------
    void onAnimate(double dt_d) override {
        float dt = float(dt_d);
        sim_time += dt;

#ifdef CORVID_USE_RAVENNET
        // Decay action bias validity (spec §2.3.1: zeroes after 200 ms)
        for (int i = 0; i < N_POOL; ++i) {
            if (!pool[i].live) continue;
            bias_age[i] += dt;
            if (bias_age[i] > 0.2f)
                action_biases[i] = {};
        }
#endif

        const float vr     = view_r;
        const float vr2    = vr * vr;
        const float ms     = max_spd;
        const float wa     = w_align;
        const float ws     = w_sep;
        const float wc     = w_cohere;
        const float wp     = w_predator;
        const float wf     = w_food;
        const float wo     = w_obstacle;
        const float drain  = e_drain;

        // 1. Entity ticks
        for (auto& e : entities)
            e->tick(dt, sim_time);

        // 2. Decay place grid (EMA factor per frame)
        decayPlaces(places, 0.995f);

        // 3. Build spatial hash of live agents
        hash.rebuild(vr, W);
        hash.clear();
        for (int i = 0; i < N_POOL; ++i) {
            if (!pool[i].live) continue;
            hash.insert(i, pool[i].nav.pos());
        }

        // 4. Collect live agent positions for hawks
        std::vector<std::pair<int, Vec3f>> live_for_hawks;
        live_for_hawks.reserve(n_live);
        for (int i = 0; i < N_POOL; ++i) {
            if (!pool[i].live) continue;
            auto& p = pool[i].nav.pos();
            live_for_hawks.push_back({i, Vec3f(float(p.x), float(p.y), float(p.z))});
        }

        // 5. Hawk ticks (may strike)
        for (auto* hawk : hawks) {
            int hit = hawk->tickWithAgents(dt, sim_time, live_for_hawks);
            if (hit >= 0 && pool[hit].live) {
                pool[hit].energy -= hawk->E_damage;
                Vec3f hp(float(hawk->position.x),
                         float(hawk->position.y),
                         float(hawk->position.z));
                writePlace(places, hp, MK_PREDATOR, -0.9f, HALF_W);
                beep(180.f);
                // write predator-strike memory for hit agent
                {
                    int pidx = placeIndex(hp, HALF_W);
                    ObsVec obs{};  // minimal obs — full perception fires at 5 Hz
                    Memory m;
                    m.timestamp = sim_time;
                    m.kind      = MK_PREDATOR;
                    m.place_id  = uint32_t(pidx);
                    m.salience  = memSalience(MK_PREDATOR, pool[hit].energy + hawk->E_damage, pool[hit].energy);
                    encoder.encode(obs.v, m.vec);
                    mem_rings[hit].push(m);
                }
            }
        }

        // 6. Per-agent update
        std::vector<int> to_kill;
        std::vector<int> to_repro;
        std::vector<int> neighbors;

        for (int i = 0; i < N_POOL; ++i) {
            Agent& a = pool[i];
            if (!a.live) continue;

            Vec3f pos_i(float(a.nav.pos().x),
                        float(a.nav.pos().y),
                        float(a.nav.pos().z));

            // --- boids: query neighbors ---
            hash.query(a.nav.pos(), neighbors);

            Vec3f f_align{0,0,0}, f_sep{0,0,0}, f_cohere{0,0,0};
            int  n_nb = 0;
            for (int j : neighbors) {
                if (j == i || !pool[j].live) continue;
                Vec3f pos_j(float(pool[j].nav.pos().x),
                            float(pool[j].nav.pos().y),
                            float(pool[j].nav.pos().z));
                Vec3f d = toroidalDelta(pos_i, pos_j);
                float d2 = d.magSqr();
                if (d2 <= 0.f || d2 > vr2) continue;
                ++n_nb;
                f_align  += vel[j];
                f_cohere += d;
                if (d2 < 0.25f) {  // separation zone: 0.5 radius
                    f_sep -= d.normalize() / std::max(std::sqrt(d2), 0.001f);
                }
            }
            if (n_nb > 0) {
                f_align  = (f_align  / float(n_nb)).normalize();
                f_cohere = (f_cohere / float(n_nb)).normalize();
            }

            // --- food attraction ---
            Vec3f f_food{0,0,0};
            float best_food2 = vr2 * 4.f;
            for (auto& e : entities) {
                if (e->category != PLANT || !e->alive) continue;
                Vec3f d = toroidalDelta(pos_i, e->position);
                float d2 = d.magSqr();
                if (d2 < best_food2) {
                    best_food2 = d2;
                    f_food = d.normalize();
                }
            }

            // --- predator avoidance ---
            Vec3f f_pred{0,0,0};
            for (auto* hawk : hawks) {
                Vec3f d = toroidalDelta(pos_i, hawk->position);
                float d2 = d.magSqr();
                if (d2 < hawk->detect_r * hawk->detect_r)
                    f_pred -= d.normalize() / std::max(std::sqrt(d2), 0.1f);
            }

            // --- obstacle avoidance ---
            Vec3f f_obs{0,0,0};
            for (auto& e : entities) {
                if (e->category != OBSTACLE) continue;
                Vec3f d = toroidalDelta(pos_i, e->position);
                float d2 = d.magSqr();
                float rr = e->interaction_radius() * 2.0f;
                if (d2 < rr * rr)
                    f_obs -= d.normalize() / std::max(std::sqrt(d2), 0.05f);
            }

            // --- accumulate and steer (M3: action biases blend in per spec §2.3.1) ---
#ifdef CORVID_USE_RAVENNET
            const float* ab = action_biases[i].data();
            const float  wav = float(w_action);
            Vec3f steer = f_align  * (wa + wav * ab[0])
                        + f_sep    * (ws + wav * ab[1])
                        + f_cohere * (wc + wav * ab[2])
                        + f_pred   * (wp + wav * ab[3])
                        + f_food   * (wf + wav * ab[4])
                        + f_obs    * (wo + wav * ab[5]);
#else
            Vec3f steer = f_align  * wa
                        + f_sep    * ws
                        + f_cohere * wc
                        + f_pred   * wp
                        + f_food   * wf
                        + f_obs    * wo;
#endif

            vel[i] += steer * dt;
            float spd = vel[i].mag();
            if (spd > ms) vel[i] = vel[i] * (ms / spd);
            if (spd < 0.1f && spd > 0.f) vel[i] = vel[i] * (0.1f / spd);

            pos_i += vel[i] * dt;
            wrapPos(pos_i);
            a.nav.pos(Vec3d(pos_i.x, pos_i.y, pos_i.z));
            // Orient nav toward velocity
            if (spd > 0.01f) {
                Vec3d fwd(vel[i].x, vel[i].y, vel[i].z);
                fwd.normalize();
                a.nav.faceToward(a.nav.pos() + fwd, Vec3d(0,1,0), 0.15);
            }

            // --- energy drain ---
            a.energy -= drain * dt;

            // --- entity interactions ---
            for (auto& e : entities) {
                if (!e->alive) continue;
                Vec3f d = toroidalDelta(pos_i, e->position);
                if (d.mag() < e->interaction_radius()) {
                    auto res = e->on_interact(sim_time);
                    a.energy += res.energy_delta;
                    if (res.entity_consumed) {
                        writePlace(places, pos_i, res.memory_kind, res.valence, HALF_W);
                        beep(600.f);
                        // food memory
                        Memory fm;
                        fm.timestamp = sim_time;
                        fm.kind      = MK_FOOD;
                        fm.place_id  = uint32_t(placeIndex(pos_i, HALF_W));
                        fm.salience  = memSalience(MK_FOOD, a.energy - res.energy_delta, a.energy);
                        ObsVec fobs{}; encoder.encode(fobs.v, fm.vec);
                        mem_rings[i].push(fm);
                    }
                    if (res.agent_dies) to_kill.push_back(i);
                }
            }

            // --- death check ---
            if (a.energy <= 0.f) {
                to_kill.push_back(i);
                continue;
            }
            a.energy = std::min(a.energy, 1.0f);

            // --- reproduction check ---
            float age = sim_time - a.birth_t;
            if (a.energy >= REPRO_E
             && age >= REPRO_AGE
             && sim_time - a.last_reproduce_t >= REPRO_CD
             && n_live < N_POOL - 10)
            {
                // look for a nearby mate
                for (int j : neighbors) {
                    if (j == i || !pool[j].live) continue;
                    Vec3f d = toroidalDelta(pos_i,
                        Vec3f(float(pool[j].nav.pos().x),
                              float(pool[j].nav.pos().y),
                              float(pool[j].nav.pos().z)));
                    if (d.mag() < REPRO_R) {
                        to_repro.push_back(i);
                        break;
                    }
                }
            }

            // --- flash timer ---
            if (a.flash_timer > 0.f) a.flash_timer -= dt;
        }

        // 7. Apply deaths (deduplicate)
        std::sort(to_kill.begin(), to_kill.end());
        to_kill.erase(std::unique(to_kill.begin(), to_kill.end()), to_kill.end());
        for (int s : to_kill) killAgent(s);

        // 8. Apply reproductions
        std::sort(to_repro.begin(), to_repro.end());
        to_repro.erase(std::unique(to_repro.begin(), to_repro.end()), to_repro.end());
        for (int s : to_repro) {
            if (!pool[s].live) continue;
            Agent& a = pool[s];
            a.energy *= 0.55f;
            a.last_reproduce_t = sim_time;
            Vec3f pp(float(a.nav.pos().x), float(a.nav.pos().y), float(a.nav.pos().z));
            Vec3f off(rnd::uniformS(), rnd::uniformS(), rnd::uniformS());
            off = off.normalize() * 0.3f;
            int child = spawnAgent(pp + off, 0.4f, a.lineage_id);
            if (child >= 0) {
                pool[child].parent_a   = a.id;
                pool[child].generation = a.generation + 1;
                Vec3f cp(float(pool[child].nav.pos().x),
                         float(pool[child].nav.pos().y),
                         float(pool[child].nav.pos().z));
                writePlace(places, cp, MK_BIRTH, 0.5f, HALF_W);
                beep(1200.f);
                // birth memory for parent
                Memory bm;
                bm.timestamp = sim_time;
                bm.kind      = MK_BIRTH;
                bm.place_id  = uint32_t(placeIndex(cp, HALF_W));
                bm.salience  = 0.7f;
                ObsVec bobs{}; encoder.encode(bobs.v, bm.vec);
                mem_rings[s].push(bm);
            }
        }

        // 9. Perception tick at ~5 Hz (every 12th animate call at 60 fps)
        ++cognition_counter;
        if (cognition_counter >= 12) {
            cognition_counter = 0;
            for (int i = 0; i < N_POOL; ++i) {
                Agent& a = pool[i];
                if (!a.live) continue;

                Vec3f pos_i(float(a.nav.pos().x),
                            float(a.nav.pos().y),
                            float(a.nav.pos().z));
                int pidx = placeIndex(pos_i, HALF_W);

                // Find nearest food
                bool  has_food  = false;
                Vec3f food_dir  = {};
                float food_dist = 1.f;
                float best_fd2  = (HALF_W * 2.f) * (HALF_W * 2.f);
                for (auto& e : entities) {
                    if (e->category != PLANT || !e->alive) continue;
                    Vec3f d = e->position - pos_i;
                    for (int ax = 0; ax < 3; ++ax) {
                        if (d[ax] >  HALF_W) d[ax] -= W;
                        if (d[ax] < -HALF_W) d[ax] += W;
                    }
                    float d2 = d.magSqr();
                    if (d2 < best_fd2) {
                        best_fd2 = d2;
                        float dm = std::sqrt(d2);
                        food_dir  = dm > 0.f ? d / dm : Vec3f{};
                        food_dist = dm / (W);
                        has_food  = true;
                    }
                }

                // Find nearest predator
                bool  has_pred  = false;
                Vec3f pred_dir  = {};
                float pred_dist = 1.f;
                float best_pd2  = (HALF_W * 2.f) * (HALF_W * 2.f);
                for (auto* hawk : hawks) {
                    Vec3f d = hawk->position - pos_i;
                    for (int ax = 0; ax < 3; ++ax) {
                        if (d[ax] >  HALF_W) d[ax] -= W;
                        if (d[ax] < -HALF_W) d[ax] += W;
                    }
                    float d2 = d.magSqr();
                    if (d2 < best_pd2) {
                        best_pd2 = d2;
                        float dm = std::sqrt(d2);
                        pred_dir  = dm > 0.f ? d / dm : Vec3f{};
                        pred_dist = dm / (W);
                        has_pred  = true;
                    }
                }

                PerceptInput pin{a, vel[i], places[pidx],
                                 HALF_W, float(max_spd),
                                 0, {}, {},
                                 has_food, food_dir, food_dist,
                                 has_pred, pred_dir, pred_dist};

                ObsVec obs = buildObsVec(pin);

                Memory pm;
                pm.timestamp = sim_time;
                pm.kind      = MK_NOVELTY;
                pm.place_id  = uint32_t(pidx);
                pm.salience  = 0.15f + places[pidx].novelty_score * 0.5f;
                encoder.encode(obs.v, pm.vec);
                mem_rings[i].push(pm);

                // Decay ring salience
                mem_rings[i].decaySalience(0.98f);
            }

#ifdef CORVID_USE_RAVENNET
            // Batch all live obs and run RavenNet forward
            // Collect encoded obs and slot indices
            std::vector<float>   obs_batch;
            std::vector<int64_t> slot_idx;
            std::vector<int>     live_slots;
            obs_batch.reserve(n_live * ENC_DIM);
            slot_idx.reserve(n_live);
            live_slots.reserve(n_live);

            for (int j = 0; j < N_POOL; ++j) {
                if (!pool[j].live) continue;
                // Use latest memory vec if ring has entries, else zeros
                int mi = mem_rings[j].size() > 0 ? (mem_rings[j].size() - 1) : -1;
                if (mi >= 0) {
                    const float* v = mem_rings[j].get(mi).vec;
                    obs_batch.insert(obs_batch.end(), v, v + ENC_DIM);
                } else {
                    obs_batch.insert(obs_batch.end(), ENC_DIM, 0.f);
                }
                slot_idx.push_back(int64_t(j));
                live_slots.push_back(j);
            }

            if (!live_slots.empty()) {
                int Nb = int(live_slots.size());
                std::vector<float> biases(Nb * brain.cfg.d_action);
                std::vector<float> values(Nb);
                brain.forward(obs_batch.data(), slot_idx.data(), Nb,
                              biases.data(), values.data());
                for (int k = 0; k < Nb; ++k) {
                    int s = live_slots[k];
                    for (int d = 0; d < brain.cfg.d_action; ++d)
                        action_biases[s][d] = biases[k * brain.cfg.d_action + d];
                    bias_age[s] = 0.f;
                }
            }
#endif
        }
    }

    // ---------------------------------------------------------------------------
    // onDraw
    // ---------------------------------------------------------------------------
    void onDraw(Graphics& g) override {
        g.clear(0.03f, 0.03f, 0.06f);
        g.blending(true);
        g.blendTrans();
        g.depthTesting(true);

        // --- entities ---
        for (auto& e : entities)
            e->draw(g);

        // --- place grid novelty heat map (M2 visualizer) ---
        // Layer 1: translucent solid voxels scaled by novelty (heat map fill).
        // Layer 2: thin wireframe edges for high-novelty cells.
        // Color: cyan = net positive valence (food/birth), orange = negative (death/predator).
        {
            const float cell = W / float(PLACE_GRID_N);
            Mesh cube_solid{Mesh::TRIANGLES};
            addCube(cube_solid);

            for (auto& pl : places) {
                float nov = pl.novelty_score;
                if (nov < 0.008f) continue;

                float val  = pl.avg_valence;
                float fill = nov * 0.30f;  // solid fill alpha
                float edge = nov * 0.70f;  // wireframe alpha

                float r, gg, b;
                if (val >= 0.f) {
                    // cyan
                    r = 0.05f + val * 0.3f; gg = 0.75f; b = 0.95f;
                } else {
                    // orange
                    r = 0.95f; gg = 0.45f + val * 0.3f; b = 0.05f;
                }

                // Solid translucent voxel
                g.pushMatrix();
                g.translate(pl.center);
                g.scale(cell * 0.48f);
                g.color(r, gg, b, fill);
                g.draw(cube_solid);

                // Wireframe overlay on same cell
                if (nov > 0.05f) {
                    Mesh wf{Mesh::LINES};
                    const float h = 1.f;
                    wf.vertex(-h,-h,-h); wf.vertex( h,-h,-h);
                    wf.vertex( h,-h,-h); wf.vertex( h, h,-h);
                    wf.vertex( h, h,-h); wf.vertex(-h, h,-h);
                    wf.vertex(-h, h,-h); wf.vertex(-h,-h,-h);
                    wf.vertex(-h,-h, h); wf.vertex( h,-h, h);
                    wf.vertex( h,-h, h); wf.vertex( h, h, h);
                    wf.vertex( h, h, h); wf.vertex(-h, h, h);
                    wf.vertex(-h, h, h); wf.vertex(-h,-h, h);
                    wf.vertex(-h,-h,-h); wf.vertex(-h,-h, h);
                    wf.vertex( h,-h,-h); wf.vertex( h,-h, h);
                    wf.vertex( h, h,-h); wf.vertex( h, h, h);
                    wf.vertex(-h, h,-h); wf.vertex(-h, h, h);
                    g.color(r, gg, b, edge);
                    g.draw(wf);
                }
                g.popMatrix();
            }
        }

        // --- agents ---
        for (int i = 0; i < N_POOL; ++i) {
            Agent& a = pool[i];
            // Draw briefly after death for flash, but mark !live
            if (!a.live && a.flash_timer <= 0.f) continue;

            Vec3f pos(float(a.nav.pos().x),
                      float(a.nav.pos().y),
                      float(a.nav.pos().z));

            // Color: lineage HSV hue, energy brightness (alive) or red fade (dead)
            float hue = std::fmod(float(a.lineage_id) * 137.508f, 360.f) / 360.f;
            float bright = a.live ? (a.energy * 0.85f + 0.15f) : 0.f;
            float alpha  = a.live ? 0.85f : (a.flash_timer / 0.35f) * 0.7f;

            Color col;
            if (a.live && a.flash_timer > 0.f) {
                // flash color
                if (a.flash_kind == 0) col = Color(0.4f, 1.0f, 0.4f, alpha);  // birth green
                else                   col = Color(1.0f, 0.2f, 0.1f, alpha);  // death red
            } else if (!a.live) {
                col = Color(1.0f, 0.2f, 0.1f, alpha);
            } else {
                // HSV → RGB (simple)
                float h6 = hue * 6.f;
                int   hi = int(h6) % 6;
                float f  = h6 - int(h6);
                float p  = bright * 0.2f;
                float q  = bright * (1.f - f * 0.7f);
                float t  = bright * (1.f - (1.f - f) * 0.7f);
                float r, gg, b;
                switch (hi) {
                    case 0: r=bright; gg=t;      b=p;      break;
                    case 1: r=q;      gg=bright; b=p;      break;
                    case 2: r=p;      gg=bright; b=t;      break;
                    case 3: r=p;      gg=q;      b=bright; break;
                    case 4: r=t;      gg=p;      b=bright; break;
                    default:r=bright; gg=p;      b=q;      break;
                }
                col = Color(r, gg, b, alpha);
            }
            g.color(col);

            g.pushMatrix();
            g.translate(pos);
            // Orient along nav quaternion (set in onAnimate via faceToward)
            g.rotate(a.nav.quat());
            g.scale(0.18f);
            g.draw(tetra_m);
            g.popMatrix();
        }

        // --- GUI ---
        gui.draw(g);
    }

    // ---------------------------------------------------------------------------
    // onSound
    // ---------------------------------------------------------------------------
    void onSound(AudioIOData& io) override {
        synth.render(io);
    }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    // Heap-allocate: MemoryRing array alone is ~18 MB — too large for the default stack.
    auto app = std::make_unique<CorvidM1>();
    app->configureAudio(44100, 512, 2, 0);
    app->start();
}

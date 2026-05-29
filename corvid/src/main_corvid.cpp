// Corvid M1/M2/M3 alpha demo — v0.4
// Energy · Reproduction · Death · Place grid · Memory rings · RavenNet · Analysis HUD
#include "al/app/al_App.hpp"
#include "al/app/al_DistributedApp.hpp"
#include "al/io/al_Imgui.hpp"
#include "al/graphics/al_Shapes.hpp"
#include "al/math/al_Random.hpp"
#include "al/math/al_Vec.hpp"
#include "al/scene/al_PolySynth.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"
#include "Gamma/Envelope.h"
#include "Gamma/Oscillator.h"

#include "core/Agent.hpp"
#include "core/CorvidVizState.hpp"
#include "core/Memory.hpp"
#include "core/MemoryRing.hpp"
#include "core/Place.hpp"
#include "core/SpatialHash.hpp"
#include "cognition/Perception.hpp"
#ifdef CORVID_USE_RAVENNET
#include "cognition/RavenBrain.hpp"
#include "training/PPOBuffer.hpp"
#endif
#include "cognition/Reflection.hpp"
#ifdef CORVID_USE_LLM
#include "cognition/LlmReflection.hpp"
#endif
#include "environment/AcornPlant.hpp"
#include "environment/BoulderObstacle.hpp"
#include "environment/HawkPredator.hpp"

// Part B — native generative visualization
#include "cognition/Lens.hpp"
#include "viz/Skybox.hpp"
#include "viz/SplatModel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
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
// DistributedAppWithState: on the AlloSphere the primary (simulator) runs the
// sim + RavenNet + lens and broadcasts CorvidVizState; renderer nodes draw from
// it (Phase 10). On a single desktop the one node is primary and reads back its
// own state, so behavior is identical to the standalone app.
struct CorvidM1 : public DistributedAppWithState<CorvidVizState> {
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
    std::array<std::array<float, 6>, N_POOL> action_biases{};
    std::array<float, N_POOL> bias_age{};
    Parameter w_action{"ActionBias", "Forces", 2.0f, 0.f, 4.f};

    // --- M4: PPO rollout buffer + per-agent prev-step storage ---
    PPOBuffer<N_POOL>              ppo_buf;
    std::array<float, N_POOL * ENC_DIM_CONST> prev_obs{};   // last obs per slot
    std::array<int,   N_POOL>                 prev_action{};
    std::array<float, N_POOL>                 prev_value{};
    std::array<float, N_POOL>                 prev_logprob{};
    std::array<bool,  N_POOL>                 has_prev{};
    std::array<float, N_POOL>                 pending_reward{};  // since last tick
    float ppo_reward_acc = 0.f;
    int   ppo_reward_n   = 0;

    // --- M5: place affinity + teaching window ---
    static constexpr float T_TEACH = 30.f;  // juvenile teaching window (s)
    std::array<std::array<float, PLACE_GRID_CELLS>, N_POOL> place_affinity{};
    std::array<float, N_POOL> teach_until{};  // sim_time when teaching window ends
#endif

    // --- M13-A: Tier A reflection scheduler ---
#ifdef CORVID_USE_LLM
    ReflectionThread             reflect_thread_;
    std::array<float, N_POOL>    next_reflect_{};  // sim_time when slot next fires
    int                          reflect_rr_   = 0; // round-robin cursor
    bool                         llm_ready_    = false;
#endif

    // --- Part B: generative "thinking" visuals ---
    TunedLens     lens;            // tuned-lens readout over RavenNet hidden acts
    ThoughtVector thought;         // current focused-agent thought (drives visuals)
    Skybox        skybox;          // generative thinking skybox
    SplatModel    splats;          // distilled-student crow splat cloud
    int           focus_slot = -1; // agent whose mind we visualize
    bool          show_skybox = true;
    bool          show_splats = true;
    Parameter     w_glow{"Glow", "Visuals", 1.0f, 0.f, 3.f};

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

    // --- Analysis HUD rolling history (256 samples, ~4 min at 1 Hz) ---
    static constexpr int HIST = 256;
    struct RollingBuf {
        float v[HIST] = {};
        int   head    = 0;
        void push(float x) { v[head] = x; head = (head + 1) % HIST; }
        // Returns pointer to data starting at oldest sample (head), length HIST
        const float* data() const { return v; }
        int offset() const { return head; }
    };
    RollingBuf h_population, h_avg_energy, h_births, h_deaths,
               h_novelty, h_ravenms;
#ifdef CORVID_USE_RAVENNET
    RollingBuf h_ppo_kl, h_policy_loss, h_value_loss, h_avg_reward;
#endif
    float hud_sample_acc = 0.f;    // accumulates dt until 1s sample
    int   hud_births_tick = 0, hud_deaths_tick = 0;  // since last sample

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
#ifdef CORVID_USE_RAVENNET
        pending_reward[slot]  = 0.f;
        has_prev[slot]        = false;
        ppo_buf.reset_agent(slot);
        place_affinity[slot]  = {};
        teach_until[slot]     = sim_time + T_TEACH;
#endif
        ++n_live; ++n_born;
        ++hud_births_tick;
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
#ifdef CORVID_USE_RAVENNET
        pending_reward[slot] -= 2.0f;
        ppo_buf.mark_done(slot);
        has_prev[slot] = false;
#endif
        free_list.push_back(slot);
        --n_live; ++n_dead;
        ++hud_deaths_tick;
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

#ifdef CORVID_USE_LLM
        {
            // Model path: next to exe, or in assets/models/ relative to project root
            const char* model_candidates[] = {
                "assets/models/gemma-4-E2B-it-Q4_K_M.gguf",
                "../../assets/models/gemma-4-E2B-it-Q4_K_M.gguf",
                "../../../assets/models/gemma-4-E2B-it-Q4_K_M.gguf",
            };
            std::string model_path;
            for (auto* c : model_candidates) {
                if (FILE* f = std::fopen(c, "rb")) { std::fclose(f); model_path = c; break; }
            }
            if (model_path.empty()) { // ngl=0: CPU-only, VRAM reserved for libtorch
                if (FILE* f = std::fopen("reflection.log", "a"))
                    { std::fprintf(f, "[M13-A] no E2B model found; heuristic fallback\n"); std::fclose(f); }
            } else {
                llm_ready_ = reflect_thread_.start(model_path, /*ngl=*/0);
                if (!llm_ready_) {
                    FILE* f = std::fopen("reflection.log", "a");
                    if (f) { std::fprintf(f, "[M13-A] model load failed; heuristic fallback\n"); std::fclose(f); }
                }
            }
        }
#endif
        // --- Part B: tuned lens + generative visuals ---
        // Crow images live in assets/crows/ (resolved relative to project root,
        // matching how run.sh / corvid binaries are launched).
        const char* crow_dir_candidates[] = {
            "assets/crows", "../../assets/crows", "../../../assets/crows",
            "MAT201B_Projects/corvid/assets/crows",
        };
        std::string crow_dir, crow_one;
        for (auto* c : crow_dir_candidates) {
            if (std::filesystem::exists(c)) { crow_dir = c; break; }
        }
        if (!crow_dir.empty()) {
            for (auto& e : std::filesystem::directory_iterator(crow_dir)) {
                auto ext = e.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                    crow_one = e.path().string();
                    break;
                }
            }
        }
#ifdef CORVID_USE_RAVENNET
        lens.init(brain.cfg.d_hidden);
#else
        lens.init(128);
#endif
        if (!skybox.init(crow_dir, 60.f))
            std::fprintf(stderr, "[corvid] skybox shader failed to compile\n");
        std::string student = crow_dir.empty() ? "" : (crow_dir + "/student.bin");
        if (!splats.init(crow_one, student, 4000))
            std::fprintf(stderr, "[corvid] splat shader failed to compile\n");

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
        gui << w_glow;
        // GUI/ImGui only on the primary (desktop). Renderer nodes have no 2D GUI.
        if (isPrimary()) {
            gui.init(0, 0, false);  // don't manage ImGui frame — we do it manually
            imguiInit();
        }
    }

    // ---------------------------------------------------------------------------
    // onAnimate — main simulation step
    // ---------------------------------------------------------------------------
    void onAnimate(double dt_d) override {
        float dt = float(dt_d);
        if (isPrimary()) {
        sim_time += dt;

        // --- HUD sampling (1 Hz) ---
        hud_sample_acc += dt;
        if (hud_sample_acc >= 1.f) {
            hud_sample_acc -= 1.f;
            // avg energy over live agents
            float esum = 0.f; int ec = 0;
            for (int i = 0; i < N_POOL; ++i)
                if (pool[i].live) { esum += pool[i].energy; ++ec; }
            float avg_e = ec > 0 ? esum / float(ec) : 0.f;
            // avg novelty over place grid
            float nsum = 0.f;
            for (auto& pl : places) nsum += pl.novelty_score;
            float avg_nov = nsum / float(PLACE_GRID_CELLS);

            h_population.push(float(n_live));
            h_avg_energy.push(avg_e);
            h_births.push(float(hud_births_tick));
            h_deaths.push(float(hud_deaths_tick));
            h_novelty.push(avg_nov);
#ifdef CORVID_USE_RAVENNET
            h_ravenms.push(brain.last_ms);
#endif
            hud_births_tick = 0;
            hud_deaths_tick = 0;
        }

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
#ifdef CORVID_USE_RAVENNET
                pending_reward[hit] -= 0.5f;
                // M5: aversion to this predator location
                {
                    Vec3f hp2(float(hawk->position.x),
                              float(hawk->position.y),
                              float(hawk->position.z));
                    int pidx = placeIndex(hp2, HALF_W);
                    place_affinity[hit][pidx] = place_affinity[hit][pidx] * 0.9f - 0.1f;
                }
#endif
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

#ifdef CORVID_USE_RAVENNET
            // M5: place-affinity bias — attract/repel toward high-affinity cells
            {
                Vec3f f_place{0,0,0};
                float best_pos = 0.f, best_neg = 0.f;
                int   best_pos_idx = -1, best_neg_idx = -1;
                for (int p = 0; p < PLACE_GRID_CELLS; ++p) {
                    float aff = place_affinity[i][p];
                    if (aff > best_pos) { best_pos = aff; best_pos_idx = p; }
                    if (aff < best_neg) { best_neg = aff; best_neg_idx = p; }
                }
                if (best_pos_idx >= 0) {
                    Vec3f d = toroidalDelta(pos_i, places[best_pos_idx].center);
                    float dm = d.mag();
                    if (dm > 0.1f) f_place += (d / dm) * best_pos * 0.5f;
                }
                if (best_neg_idx >= 0) {
                    Vec3f d = toroidalDelta(pos_i, places[best_neg_idx].center);
                    float dm = d.mag();
                    if (dm > 0.1f) f_place -= (d / dm) * (-best_neg) * 0.5f;
                }
                steer += f_place;
            }
#endif
            // M13-A: apply goal-stack biases (observer-window, additive on existing forces)
            {
                for (int g = 0; g < int(a.n_goals); ++g) {
                    const GoalEntry& ge = a.goal_stack[g];
                    float w = ge.weight;
                    switch (ge.kind) {
                        case GoalKind::SEEK_FOOD:
                            steer += f_food * (w * 2.f);
                            break;
                        case GoalKind::AVOID_PREDATOR:
                            steer += f_pred * (w * 2.f);
                            break;
                        case GoalKind::EXPLORE:
                            // small random drift toward unexplored territory
                            {
                                Vec3f drift(rnd::uniformS(), rnd::uniformS(), rnd::uniformS());
                                steer += drift.normalize() * (w * 0.4f);
                            }
                            break;
                        case GoalKind::REST:
                            // dampen velocity
                            steer -= vel[i] * (w * 0.5f);
                            break;
                    }
                }
            }

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
#ifdef CORVID_USE_RAVENNET
                        pending_reward[i] += 1.0f;
                        // M5: reinforce place affinity toward this food location
                        {
                            int pidx = placeIndex(pos_i, HALF_W);
                            place_affinity[i][pidx] = place_affinity[i][pidx] * 0.9f + 0.1f;
                        }
#endif
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
#ifdef CORVID_USE_RAVENNET
                // M5: inherit adapter (LoRA B row) and place affinity from parent
                brain.inherit_adapter(child, s);
                for (int p = 0; p < PLACE_GRID_CELLS; ++p)
                    place_affinity[child][p] = place_affinity[s][p] * 0.8f;
#endif
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
            { if (FILE* f = std::fopen("m4_diag.log", "a"))
              { static int _tc=0; std::fprintf(f, "cog_tick %d n_live=%d\n", ++_tc, n_live); std::fclose(f); } }

            // M4: collect transitions from previous tick into PPO buffer
            for (int j = 0; j < N_POOL; ++j) {
                if (!pool[j].live || !has_prev[j]) continue;
                pending_reward[j] += 0.005f;  // survival bonus per tick
                StepRecord rec;
                const float* po = &prev_obs[j * ENC_DIM_CONST];
                std::copy(po, po + ENC_DIM_CONST, rec.obs);
                rec.action          = prev_action[j];
                rec.reward          = pending_reward[j];
                rec.value           = prev_value[j];
                rec.logprob         = prev_logprob[j];
                rec.done            = 0.f;
                rec.adapter_idx     = int64_t(j);
                rec.teaching_scale  = (sim_time - pool[j].birth_t) < T_TEACH ? 3.f : 1.f;
                ppo_buf.push(j, rec);
                pending_reward[j] = 0.f;
            }

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
                    const float* logits = &biases[k * brain.cfg.d_action];
                    for (int d = 0; d < brain.cfg.d_action; ++d)
                        action_biases[s][d] = logits[d];
                    bias_age[s] = 0.f;

                    // M4: store prev obs/action/value/logprob for next tick
                    float* po = &prev_obs[s * ENC_DIM_CONST];
                    std::copy(obs_batch.begin() + k * ENC_DIM,
                              obs_batch.begin() + k * ENC_DIM + ENC_DIM_CONST, po);
                    int act = sample_softmax(logits, brain.cfg.d_action,
                                             rnd::uniform());
                    prev_action[s]  = act;
                    prev_value[s]   = values[k];
                    prev_logprob[s] = log_softmax_action(logits, act, brain.cfg.d_action);
                    has_prev[s]     = true;
                }

                // --- Part B: tuned-lens readout (Phase 6) ---
                // Re-run forward exposing hidden activations, train the lens to
                // match the policy head, and build the focused agent's thought.
                {
                    const int dH = brain.cfg.d_hidden, dA = brain.cfg.d_action;
                    std::vector<float> hidden(size_t(Nb) * dH);
                    brain.forward_tap(obs_batch.data(), slot_idx.data(), Nb,
                                      nullptr, nullptr, hidden.data());
                    // targets = policy-head softmax per sample
                    std::vector<float> target(size_t(Nb) * dA);
                    for (int k = 0; k < Nb; ++k) {
                        const float* lg = &biases[k * dA];
                        float mx = lg[0];
                        for (int d = 1; d < dA; ++d) mx = std::max(mx, lg[d]);
                        float sum = 0.f;
                        for (int d = 0; d < dA; ++d) { float e = std::exp(lg[d]-mx); target[k*dA+d]=e; sum+=e; }
                        for (int d = 0; d < dA; ++d) target[k*dA+d] /= (sum + 1e-20f);
                    }
                    if (dA == LENS_ACT)
                        lens.train_step(hidden.data(), target.data(), Nb);

                    // choose focus = first live slot in the batch (stable enough)
                    focus_slot = live_slots.empty() ? -1 : live_slots[0];
                    if (focus_slot >= 0) {
                        // find this slot's row in the batch
                        int fk = 0;
                        for (int k = 0; k < Nb; ++k) if (live_slots[k] == focus_slot) { fk = k; break; }
                        int dgoal = pool[focus_slot].n_goals > 0
                                  ? int(pool[focus_slot].goal_stack[0].kind)
                                  : int(GoalKind::EXPLORE);
                        if (dA == LENS_ACT)
                            thought = lens.think(&hidden[size_t(fk) * dH], values[fk], dgoal);
                    }
                }

                // M4: run PPO train step if any rollout is full
                if (ppo_buf.any_ready()) {
                    std::array<float, N_POOL> bootstrap{};
                    for (int j = 0; j < N_POOL; ++j)
                        if (pool[j].live && has_prev[j])
                            bootstrap[j] = prev_value[j];
                    auto batch = ppo_buf.compute_batch(bootstrap.data());
                    if (batch.N >= 4) {
                        brain.train_step(batch);
                        h_ppo_kl.push(brain.last_kl);
                        h_policy_loss.push(brain.last_policy_loss);
                        h_value_loss.push(brain.last_value_loss);
                        float avg_r = 0.f;
                        for (float r : batch.returns) avg_r += r;
                        float avg_ret = batch.N > 0 ? avg_r / float(batch.N) : 0.f;
                        h_avg_reward.push(avg_ret);
                        if (FILE* f = std::fopen("ppo_learning.log", "a")) {
                            static int step = 0;
                            // count juvenile samples in batch
                            int juv_n = 0;
                            for (float ts : batch.teaching_scale)
                                if (ts > 1.5f) ++juv_n;
                            std::fprintf(f, "%d %.4f %.4f %.4f %.4f %d juv=%d\n",
                                ++step, sim_time, avg_ret,
                                brain.last_kl, brain.last_policy_loss, batch.N, juv_n);
                            std::fclose(f);
                        }
                    }
                    ppo_buf.drain_ready();
                }
            }
#endif

#ifdef CORVID_USE_LLM
            // M13-A: drain reflection results into agent goal_stacks
            {
                ReflectResult results[64];
                int nr = llm_ready_
                    ? reflect_thread_.drain(results, 64)
                    : 0;
                for (int k = 0; k < nr; ++k) {
                    int s = results[k].slot;
                    if (s < 0 || s >= N_POOL || !pool[s].live) continue;
                    if (pool[s].id != results[k].agent_id) continue; // slot recycled
                    pool[s].n_goals = int8_t(results[k].n_goals);
                    for (int g = 0; g < results[k].n_goals; ++g)
                        pool[s].goal_stack[g] = results[k].goals[g];
                    std::strncpy(pool[s].last_reflection, results[k].text, 255);
                    pool[s].last_reflection[255] = '\0';
                }

                // Submit new reflection jobs (round-robin, up to REFLECT_BATCH agents)
                if (llm_ready_ || true) {  // always schedule; heuristic runs if LLM unavailable
                    std::vector<ReflectJob> jobs;
                    int checked = 0;
                    while (checked < N_POOL && int(jobs.size()) < ReflectionThread::REFLECT_BATCH) {
                        int s = (reflect_rr_ + checked) % N_POOL;
                        ++checked;
                        if (!pool[s].live) continue;
                        // cadence: aim for ~12 sim-sec per agent minimum
                        float cadence = std::max(12.f, float(n_live) / float(ReflectionThread::REFLECT_BATCH) * 10.f);
                        if (sim_time < next_reflect_[s]) continue;

                        ReflectJob job;
                        job.slot     = s;
                        job.agent_id = pool[s].id;
                        job.sim_time = sim_time;
                        job.energy   = pool[s].energy;
                        job.birth_t  = pool[s].birth_t;

                        // Build memory digest from last 3 ring entries
                        char dbuf[512] = {};
                        int written = 0;
                        for (int mi = mem_rings[s].size() - 1;
                             mi >= 0 && mi >= int(mem_rings[s].size()) - 3; --mi) {
                            const auto& m = mem_rings[s].get(mi);
                            const char* kindstr = "event";
                            if (m.kind == MK_FOOD)            kindstr = "ate food";
                            else if (m.kind == MK_PREDATOR)   kindstr = "saw hawk";
                            else if (m.kind == MK_BIRTH)      kindstr = "had child";
                            else if (m.kind == MK_DEATH_WITNESSED) kindstr = "saw death";
                            written += std::snprintf(dbuf + written,
                                sizeof(dbuf) - written,
                                "%s(t=%.0f) ", kindstr, m.timestamp);
                            if (written >= int(sizeof(dbuf)) - 1) break;
                        }
                        std::strncpy(job.digest, dbuf, sizeof(job.digest) - 1);
                        jobs.push_back(job);
                        next_reflect_[s] = sim_time + cadence;
                    }
                    reflect_rr_ = (reflect_rr_ + checked) % N_POOL;

                    if (!jobs.empty()) {
                        if (llm_ready_) {
                            reflect_thread_.submit(jobs.data(), int(jobs.size()));
                        } else {
                            // Heuristic fallback — run inline (cheap)
                            for (auto& job : jobs) {
                                auto res = heuristicReflect(job);
                                int s2 = res.slot;
                                if (s2 < 0 || s2 >= N_POOL || !pool[s2].live) continue;
                                pool[s2].n_goals = int8_t(res.n_goals);
                                for (int g = 0; g < res.n_goals; ++g)
                                    pool[s2].goal_stack[g] = res.goals[g];
                                std::strncpy(pool[s2].last_reflection, res.text, 255);
                                pool[s2].last_reflection[255] = '\0';
                            }
                        }
                    }
                }
            }
#endif
        }
            // primary publishes the frame for renderer nodes
            packVizState();
        }  // end if(isPrimary())

        // --- all nodes: drive the generative visuals from broadcast state ---
        // (primary echoes what it just wrote; replicas use the received blob)
        thought = state().thought;
        if (show_splats) splats.update(thought, state().sim_time, dt);
    }

    // Pack the live sim into the broadcast state (primary only).
    void packVizState() {
        state().tick++;
        state().sim_time  = sim_time;
        state().thought   = thought;
        int n = 0, focus_out = -1;
        for (int i = 0; i < N_POOL && n < VIZ_MAX_AGENTS; ++i) {
            Agent& a = pool[i];
            if (!a.live && a.flash_timer <= 0.f) continue;
            AgentXform& x = state().xform[n];
            auto& p = a.nav.pos();
            x.pos[0] = float(p.x); x.pos[1] = float(p.y); x.pos[2] = float(p.z);
            auto q = a.nav.quat();
            x.quat[0] = float(q.x); x.quat[1] = float(q.y);
            x.quat[2] = float(q.z); x.quat[3] = float(q.w);
            x.lineage   = a.lineage_id;
            x.energy    = a.energy;
            x.flash     = a.flash_timer;
            x.flashKind = a.flash_kind;
            x.live      = a.live ? 1 : 0;
            if (i == focus_slot) focus_out = n;
            ++n;
        }
        state().n_agents  = n;
        state().focus_idx = focus_out;
    }

    // ---------------------------------------------------------------------------
    // onDraw
    // ---------------------------------------------------------------------------
    // Deterministic agent color from broadcast fields (same on every node).
    Color agentColor(uint32_t lineage, float energy, float flash,
                     int flashKind, bool live) {
        float alpha = live ? 0.85f : (flash / 0.35f) * 0.7f;
        if (live && flash > 0.f)
            return flashKind == 0 ? Color(0.4f, 1.0f, 0.4f, alpha)   // birth
                                  : Color(1.0f, 0.2f, 0.1f, alpha);  // death
        if (!live) return Color(1.0f, 0.2f, 0.1f, alpha);
        float hue = std::fmod(float(lineage) * 137.508f, 360.f) / 360.f;
        float bright = energy * 0.85f + 0.15f;
        float h6 = hue * 6.f; int hi = int(h6) % 6; float f = h6 - int(h6);
        float p = bright*0.2f, q = bright*(1.f-f*0.7f), t = bright*(1.f-(1.f-f)*0.7f);
        float r, gg, b;
        switch (hi) {
            case 0: r=bright; gg=t;      b=p;      break;
            case 1: r=q;      gg=bright; b=p;      break;
            case 2: r=p;      gg=bright; b=t;      break;
            case 3: r=p;      gg=q;      b=bright; break;
            case 4: r=t;      gg=p;      b=bright; break;
            default:r=bright; gg=p;      b=q;      break;
        }
        return Color(r, gg, b, alpha);
    }

    void onDraw(Graphics& g) override {
        g.clear(0.0f, 0.0f, 0.0f);

        // --- Part B: generative "thinking" skybox (draw first, behind all) ---
        // Driven by broadcast thought + time so every dome node matches.
        if (show_skybox && skybox.ready())
            skybox.draw(g, nav().pos(), state().thought, state().sim_time);

        g.blending(true);
        g.blendTrans();
        g.depthTesting(true);

        // --- entities + place grid: primary-only (not broadcast) ---
        if (isPrimary())
        for (auto& e : entities)
            e->draw(g);

        // --- Part B: crow splat cloud, anchored on the focused corvid ---
        if (show_splats && splats.ready()) {
            Vec3f c(0, 0, 0);
            int fi = state().focus_idx;
            if (fi >= 0 && fi < state().n_agents) {
                const auto& x = state().xform[fi];
                c = Vec3f(x.pos[0], x.pos[1], x.pos[2]);
            }
            splats.draw(g, c, 1.6f);
        }

        // --- place grid novelty heat map (M2 visualizer; primary only) ---
        // Layer 1: translucent solid voxels scaled by novelty (heat map fill).
        // Layer 2: thin wireframe edges for high-novelty cells.
        // Color: cyan = net positive valence (food/birth), orange = negative (death/predator).
        if (isPrimary()) {
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

        // --- GUI + Analysis: both in one manual ImGui frame ---
        imguiBeginFrame();
        gui.draw(g);  // ControlGUI panel (beginPanel/endPanel — no frame management)

        ImGui::SetNextWindowPos({0.f, 370.f}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({300.f, 720.f}, ImGuiCond_Always);
        ImGui::Begin("Analysis", nullptr,
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
        drawAnalysis();
        ImGui::End();

        imguiEndFrame();
        imguiDraw();
    }

    // ---------------------------------------------------------------------------
    // drawAnalysis — called via gui.drawFunction inside the ControlGUI frame
    // ---------------------------------------------------------------------------
    void drawAnalysis() {
        float w = ImGui::GetContentRegionAvail().x;
        ImGui::TextColored({0.6f,1.f,0.6f,1.f}, "Population");
        ImGui::Text("Live: %d  Born: %d  Dead: %d  t=%.1fs", n_live, n_born, n_dead, sim_time);
        ImGui::PlotLines("##pop", h_population.data(), HIST, h_population.offset(),
                         nullptr, 0.f, float(N_POOL), {w, 40.f});

        ImGui::TextColored({1.f,0.9f,0.4f,1.f}, "Avg Energy");
        ImGui::PlotLines("##enrg", h_avg_energy.data(), HIST, h_avg_energy.offset(),
                         nullptr, 0.f, 1.f, {w, 35.f});

        ImGui::TextColored({0.4f,0.9f,1.f,1.f}, "Births/s");
        ImGui::PlotHistogram("##brt", h_births.data(), HIST, h_births.offset(),
                             nullptr, FLT_MAX, FLT_MAX, {w, 28.f});

        ImGui::TextColored({1.f,0.4f,0.4f,1.f}, "Deaths/s");
        ImGui::PlotHistogram("##dth", h_deaths.data(), HIST, h_deaths.offset(),
                             nullptr, FLT_MAX, FLT_MAX, {w, 28.f});

        ImGui::TextColored({0.8f,0.5f,1.f,1.f}, "Place Novelty");
        ImGui::PlotLines("##nov", h_novelty.data(), HIST, h_novelty.offset(),
                         nullptr, 0.f, 0.5f, {w, 35.f});

#ifdef CORVID_USE_RAVENNET
        ImGui::TextColored({1.f,0.7f,0.2f,1.f}, "RavenNet ms");
        ImGui::SameLine();
        ImGui::Text("last=%.2f", brain.last_ms);
        if (brain.last_ms > 3.f) { ImGui::SameLine(); ImGui::TextColored({1.f,0.2f,0.2f,1.f}, "OVER"); }
        ImGui::PlotLines("##rnn", h_ravenms.data(), HIST, h_ravenms.offset(),
                         nullptr, 0.f, 5.f, {w, 35.f});
#endif

#ifdef CORVID_USE_RAVENNET
        ImGui::Separator();
        ImGui::TextColored({0.4f, 1.f, 0.8f, 1.f}, "PPO Training");
        ImGui::Text("KL=%.4f  PL=%.4f  VL=%.4f",
                    brain.last_kl, brain.last_policy_loss, brain.last_value_loss);
        if (brain.last_kl > 0.02f)
            ImGui::TextColored({1.f, 0.3f, 0.3f, 1.f}, "HIGH KL!");
        ImGui::PlotLines("##kl", h_ppo_kl.data(), HIST, h_ppo_kl.offset(),
                         "KL", 0.f, 0.05f, {w, 28.f});
        ImGui::PlotLines("##ploss", h_policy_loss.data(), HIST, h_policy_loss.offset(),
                         "PolicyLoss", FLT_MAX, FLT_MAX, {w, 28.f});
        ImGui::PlotLines("##avgret", h_avg_reward.data(), HIST, h_avg_reward.offset(),
                         "AvgReturn", FLT_MAX, FLT_MAX, {w, 28.f});

        // M5 live stats
        ImGui::Separator();
        ImGui::TextColored({0.5f, 1.f, 0.5f, 1.f}, "M5 Inheritance");
        int juveniles = 0;
        float aff_sum = 0.f; int aff_n = 0;
        for (int ii = 0; ii < N_POOL; ++ii) {
            if (!pool[ii].live) continue;
            if ((sim_time - pool[ii].birth_t) < T_TEACH) ++juveniles;
            for (int p = 0; p < PLACE_GRID_CELLS; ++p)
                if (place_affinity[ii][p] != 0.f) { aff_sum += std::fabs(place_affinity[ii][p]); ++aff_n; }
        }
        ImGui::Text("Juveniles(<30s): %d  AffCells: %d", juveniles, aff_n);
        if (aff_n > 0) ImGui::Text("AvgAffMag: %.3f", aff_sum / float(aff_n));
#endif

#ifdef CORVID_USE_LLM
        ImGui::Separator();
        ImGui::TextColored({0.4f,0.8f,1.f,1.f}, "Tier A Reflection");
        if (llm_ready_) {
            ImGui::Text("batches=%d  last=%.0fms",
                reflect_thread_.batches_done, reflect_thread_.last_batch_ms);
        } else {
            ImGui::TextColored({1.f,0.7f,0.3f,1.f}, "heuristic fallback");
        }
        // Show last reflection text for first live agent with LLM output
        for (int ii = 0; ii < N_POOL; ++ii) {
            if (!pool[ii].live || pool[ii].last_reflection[0] == '\0') continue;
            ImGui::TextWrapped("a#%u: %s", pool[ii].id, pool[ii].last_reflection);
            break;
        }
#endif
        int total_mems = 0;
        for (int i = 0; i < N_POOL; ++i)
            if (pool[i].live) total_mems += mem_rings[i].size();
        ImGui::TextColored({0.7f,0.8f,1.f,1.f}, "MemRings:");
        ImGui::SameLine();
        ImGui::Text("%d total  %.1f/agent",
                    total_mems, n_live > 0 ? float(total_mems)/float(n_live) : 0.f);
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

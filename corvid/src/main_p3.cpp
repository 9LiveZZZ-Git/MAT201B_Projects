// corvid_p3 — DistributedAppWithState + Reynolds flocking + PolySynth audio.
// Alpha screen demo: heading-colored tetrahedra, collision flash, dark BG.
// Pass --model PATH to show Gemma 4 E2B reflection overlay.

#include "Gamma/Envelope.h"
#include "Gamma/Oscillator.h"
#include "al/app/al_DistributedApp.hpp"
#include "al/graphics/al_Shapes.hpp"
#include "al/io/al_Imgui.hpp"
#include "al/math/al_Random.hpp"
#include "al/scene/al_PolySynth.hpp"
#include "al/scene/al_SynthVoice.hpp"
#include "al/types/al_Color.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"
#include "core/Agent.hpp"
#include "core/SimpleSharedState.hpp"
#include "core/SpatialHash.hpp"
#include "reactive/Boids.hpp"

#ifdef CORVID_LLM
#include "cognition/LlmSmoke.hpp"
#include <atomic>
#include <mutex>
#include <thread>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string>

using namespace al;
using namespace corvid;

static std::string g_modelPath;

const int    N = 200;
const double W = 10.0;

// --- Audio voice: short sine ping on collision ---
class BeepVoice : public SynthVoice {
  gam::Sine<> osc;
  gam::AD<>   env{0.001f, 0.08f};
public:
  void init() override {
    createInternalTriggerParameter("freq", 440.f, 80.f, 4000.f);
  }
  void onProcess(AudioIOData& io) override {
    osc.freq(getInternalParameterValue("freq"));
    while (io()) {
      float s = osc() * env() * 0.15f;
      io.out(0) += s;
      io.out(1) += s;
      if (env.done()) { free(); break; }
    }
  }
  void onTriggerOn() override { env.reset(); }
};

// --- Main app ---
class CorvidP3 : public DistributedAppWithState<SimpleSharedState> {
  std::array<Agent, N> agents;
  SpatialHash          hash;
  PolySynth            synth;
  ControlGUI           gui;
  Mesh                 tetra;
  double               prevT          = -1.0;
  float                flashTimer[N]  = {};
  float                lastTrigger[N] = {};

  Parameter    turnRate   {"Turn Rate",       "Agents", 0.05f, 0.001f, 0.3f};
  Parameter    moveSpeed  {"Move Speed",      "Agents", 2.0f,  0.1f,  10.0f};
  Parameter    tNeighbor  {"Neighbor Radius", "Agents", 1.0f,  0.1f,   4.0f};
  Parameter    tSep       {"Sep Threshold",   "Agents", 0.35f, 0.02f,  2.0f};
  Parameter    alignW     {"Align Weight",    "Agents", 1.0f,  0.0f,   5.0f};
  Parameter    cohesionW  {"Cohesion Weight", "Agents", 0.8f,  0.0f,   5.0f};
  Parameter    sepW       {"Sep Weight",      "Agents", 2.5f,  0.0f,  10.0f};
  ParameterInt kNeighbors {"K Neighbors",     "Agents", 8,     1,     30};

#ifdef CORVID_LLM
  std::string           llmText   = "Waiting...";
  std::mutex            llmMutex;
  std::atomic<bool>     llmBusy{false};
  std::atomic<bool>     llmDone{false};

  void fireLlmQuery(const std::string& concept) {
    if (llmBusy || g_modelPath.empty()) return;
    llmBusy = true;
    llmDone = false;
    { std::lock_guard<std::mutex> lk(llmMutex); llmText = "Generating..."; }
    std::thread([this, concept]() {
      corvid::LlmConfig cfg;
      cfg.modelPath = g_modelPath;
      std::string result = corvid::llmInferSingle(
        cfg,
        "You are a reflection engine for a bird simulation. "
        "Emit a 30-word description of " + concept + ".",
        80);
      std::lock_guard<std::mutex> lk(llmMutex);
      llmText = result.empty() ? "[no output — check model path]" : result;
      llmBusy = false;
      llmDone = true;
    }).detach();
  }
#endif

  void initAgents() {
    const double half = W * 0.5;
    for (int i = 0; i < N; ++i) {
      agents[i].id = i;
      agents[i].nav.pos(rnd::uniformS() * half,
                        rnd::uniformS() * half,
                        rnd::uniformS() * half);
      agents[i].nav.quat(Quatd().fromEuler(
        rnd::uniform(0.0, M_2PI),
        rnd::uniform(0.0, M_2PI),
        rnd::uniform(0.0, M_2PI)));
    }
  }

  void onCreate() override {
    nav().pos(0, 0, W * 1.5);
    nav().faceToward(Vec3d(0, 0, 0));

    addTetrahedron(tetra, 0.14f);
    tetra.generateNormals();

    for (int i = 0; i < 32; ++i) synth.getVoice<BeepVoice>(true);

    gui << turnRate << moveSpeed << tNeighbor << tSep
        << kNeighbors << alignW << cohesionW << sepW;
    gui.init();
    gui.manageImGUI(false);  // we manage the ImGui frame to add extra panels

    initAgents();
    hash.rebuild((double)tNeighbor, W);
    prevT = (double)tNeighbor;

    gam::sampleRate(audioIO().framesPerSecond());

#ifdef CORVID_LLM
    if (!g_modelPath.empty()) fireLlmQuery("curiosity");
#endif
  }

  void onAnimate(double dt) override {
    const double T   = (double)tNeighbor;
    const double S   = (double)tSep;
    const int    K   = (int)kNeighbors;
    const float  fdt = float(dt);

    if (T != prevT) { hash.rebuild(T, W); prevT = T; }
    hash.clear();
    for (int i = 0; i < N; ++i) hash.insert(i, agents[i].nav.pos());

    std::vector<int>      candidates;
    std::vector<Neighbor> nbuf;
    candidates.reserve(64);
    nbuf.reserve(64);

    for (int i = 0; i < N; ++i) {
      if (flashTimer[i]  > 0.f) flashTimer[i]  -= fdt;
      if (lastTrigger[i] > 0.f) lastTrigger[i] -= fdt;

      Nav& self = agents[i].nav;
      const Vec3d& selfPos = self.pos();

      hash.query(selfPos, candidates);
      nbuf.clear();
      for (int j : candidates) {
        if (j == i) continue;
        Vec3d  delta = toroidalDelta(selfPos, agents[j].nav.pos(), W);
        double dist  = delta.mag();
        if (dist < T) nbuf.push_back({j, dist, delta});
      }

      if ((int)nbuf.size() > K) {
        std::partial_sort(nbuf.begin(), nbuf.begin() + K, nbuf.end(),
          [](const Neighbor& a, const Neighbor& b){ return a.dist < b.dist; });
        nbuf.resize(K);
      }

      if (nbuf.empty()) {
        self.moveF((double)moveSpeed);
        self.step(dt);
        Vec3d p = self.pos();
        wrapPos(p, W);
        self.pos(p);
        continue;
      }

      Vec3d sepDir = computeSeparation(nbuf, S);
      bool  hasCollision = (sepDir.mag() > 1e-4);
      if (hasCollision && lastTrigger[i] <= 0.f) {
        auto* v = synth.getVoice<BeepVoice>();
        if (v) {
          float freq = 200.f + float(i % 32) * 55.f;
          v->setInternalParameterValue("freq", freq);
          synth.triggerOn(v, 0, i);
        }
        flashTimer[i]  = 0.2f;
        lastTrigger[i] = 0.1f;
      }

      Vec3d steer = computeAlignment(nbuf, agents.data()) * (double)alignW
                  + computeCohesion(nbuf)                 * (double)cohesionW
                  + sepDir                                * (double)sepW;
      double sm = steer.mag();
      if (sm > 1e-8) self.faceToward(selfPos + steer / sm, (double)turnRate);

      self.moveF((double)moveSpeed);
      self.step(dt);
      Vec3d p = self.pos();
      wrapPos(p, W);
      self.pos(p);
    }

    state().tick++;
    state().n_agents = N;
    for (int i = 0; i < N; ++i) {
      const Vec3d& pos  = agents[i].nav.pos();
      const Quatd& quat = agents[i].nav.quat();
      state().pos[i][0]  = float(pos[0]);
      state().pos[i][1]  = float(pos[1]);
      state().pos[i][2]  = float(pos[2]);
      state().quat[i][0] = float(quat[0]);
      state().quat[i][1] = float(quat[1]);
      state().quat[i][2] = float(quat[2]);
      state().quat[i][3] = float(quat[3]);
    }
  }

  void onDraw(Graphics& g) override {
    g.clear(0.03f, 0.03f, 0.06f);
    g.lighting(false);
    g.blending(true);
    g.blendAdd();

    const int nDraw = int(state().n_agents);
    for (int i = 0; i < nDraw; ++i) {
      Vec3d pos (state().pos [i][0], state().pos [i][1], state().pos [i][2]);
      Quatd quat(state().quat[i][3], state().quat[i][0],
                 state().quat[i][1], state().quat[i][2]);

      Vec3d fwd; quat.toVectorX(fwd);
      float azimuth = float(std::atan2(fwd[0], fwd[2]));
      float hue     = (azimuth + float(M_PI)) / float(M_2PI);

      bool  flashing = (flashTimer[i] > 0.f);
      Color c = flashing
        ? Color(1.f, 1.f, 1.f, 1.0f)
        : Color(HSV(hue, 0.75f, 1.0f), 0.85f);

      g.color(c);
      g.pushMatrix();
      g.translate(pos);
      g.rotate(quat);
      g.draw(tetra);
      g.popMatrix();
    }

    g.blending(false);

    if (hasCapability(Capability::CAP_2DGUI)) {
      imguiBeginFrame();

      gui.draw(g);  // parameter sliders panel

#ifdef CORVID_LLM
      // Gemma 4 reflection panel — bottom of screen
      ImGui::SetNextWindowPos(ImVec2(10, float(height()) - 170), ImGuiCond_Always);
      ImGui::SetNextWindowSize(ImVec2(540, 160), ImGuiCond_Always);
      ImGui::Begin("Gemma 4 E2B  — Reflection Engine", nullptr,
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse);

      // Model path (truncated to filename)
      std::string fname = g_modelPath;
      auto slash = fname.find_last_of("/\\");
      if (slash != std::string::npos) fname = fname.substr(slash + 1);
      ImGui::TextDisabled("model: %s", fname.c_str());
      ImGui::Separator();

      if (llmBusy) {
        ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.2f, 1.0f), "Generating...");
      } else {
        std::lock_guard<std::mutex> lk(llmMutex);
        ImGui::TextWrapped("%s", llmText.c_str());
      }

      ImGui::Separator();
      static const char* concepts[] = {
        "curiosity","hunger","fear","joy","calm","alertness","wonder","urgency"
      };
      static int picked = 0;
      ImGui::SetNextItemWidth(110);
      ImGui::Combo("##concept", &picked, concepts, 8);
      ImGui::SameLine();
      if (ImGui::Button("Regenerate") && !llmBusy)
        fireLlmQuery(concepts[picked]);

      ImGui::End();
#endif

      imguiEndFrame();
      imguiDraw();
    }
  }

  void onSound(AudioIOData& io) override { synth.render(io); }

  bool onKeyDown(const Keyboard& k) override {
    if (k.key() == Keyboard::ESCAPE) quit();
    return true;
  }

  void onExit() override { gui.cleanup(); }
};

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--model") && i + 1 < argc)
      g_modelPath = argv[++i];

  CorvidP3 app;
  app.configureAudio(44100, 512, 2, 0);
  app.start();
}

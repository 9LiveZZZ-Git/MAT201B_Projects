#include "Reflection.hpp"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>

namespace corvid {

// Case-insensitive substring occurrence count.
static int countCI(const char* hay, const char* needle) {
    if (!hay || !*hay) return 0;
    int n = 0;
    size_t nl = std::strlen(needle);
    for (const char* p = hay; *p; ++p) {
        size_t i = 0;
        for (; i < nl && p[i]; ++i)
            if (std::tolower((unsigned char)p[i]) != std::tolower((unsigned char)needle[i])) break;
        if (i == nl) ++n;
    }
    return n;
}

static const char* kindStr(GoalKind k) {
    switch (k) {
        case GoalKind::SEEK_FOOD:      return "seek_food";
        case GoalKind::AVOID_PREDATOR: return "avoid_predator";
        case GoalKind::EXPLORE:        return "explore";
        case GoalKind::REST:           return "rest";
    }
    return "explore";
}

// Procedural reflector (replaces the LLM; spec §2.5.2 fallback, now the only path).
// Derives goals from energy, age, and the event digest produced by the sim.
ReflectResult heuristicReflect(const ReflectJob& job) {
    ReflectResult r;
    r.slot     = job.slot;
    r.agent_id = job.agent_id;
    r.from_llm = false;

    const float age  = job.sim_time - job.birth_t;
    const int   pred = countCI(job.digest, "pred") + countCI(job.digest, "hawk");
    const int   food = countCI(job.digest, "food") + countCI(job.digest, "acorn");
    const int   obst = countCI(job.digest, "obst") + countCI(job.digest, "boulder");
    (void)obst;  // obstacles inform steering elsewhere, not goal selection

    GoalEntry cand[8];
    int ng = 0;

    // Threat avoidance dominates when predators are in the recent digest.
    if (pred > 0)
        cand[ng++] = {GoalKind::AVOID_PREDATOR, std::min(1.0f, 0.6f + 0.2f * float(pred))};

    // Hunger scales with energy deficit (plus a nudge if food was just seen).
    if (job.energy < 0.5f)
        cand[ng++] = {GoalKind::SEEK_FOOD, std::min(1.0f, (1.0f - job.energy) + 0.1f * float(food))};
    else if (food > 0)
        cand[ng++] = {GoalKind::SEEK_FOOD, std::min(0.6f, 0.2f * float(food))};

    // Explore when healthy and safe.
    if (pred == 0 && job.energy > 0.55f)
        cand[ng++] = {GoalKind::EXPLORE, std::min(1.0f, job.energy * 0.7f)};

    // Rest when old, comfortable, and unthreatened.
    if (pred == 0 && age > 30.0f && job.energy > 0.5f)
        cand[ng++] = {GoalKind::REST, 0.4f};

    if (ng == 0)
        cand[ng++] = {GoalKind::EXPLORE, 0.5f};

    std::sort(cand, cand + ng,
              [](const GoalEntry& a, const GoalEntry& b) { return a.weight > b.weight; });

    int n = std::min(ng, MAX_GOALS);
    for (int i = 0; i < n; ++i) r.goals[i] = cand[i];
    r.n_goals = n;

    std::snprintf(r.text, sizeof(r.text),
                  "[proc] e=%.2f age=%.0f pred=%d food=%d => %s w=%.2f",
                  job.energy, age, pred, food,
                  kindStr(r.goals[0].kind), r.goals[0].weight);
    return r;
}

} // namespace corvid

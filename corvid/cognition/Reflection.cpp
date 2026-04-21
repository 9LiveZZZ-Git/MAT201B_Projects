#include "Reflection.hpp"
#include <cstdio>

namespace corvid {

ReflectResult heuristicReflect(const ReflectJob& job) {
    ReflectResult r;
    r.slot     = job.slot;
    r.agent_id = job.agent_id;
    r.from_llm = false;

    int ng = 0;
    if (job.energy < 0.3f) {
        r.goals[ng++] = {GoalKind::SEEK_FOOD, 1.0f - job.energy};
    }
    if (job.energy > 0.6f) {
        r.goals[ng++] = {GoalKind::EXPLORE, job.energy * 0.6f};
    }
    if (ng == 0) {
        r.goals[ng++] = {GoalKind::EXPLORE, 0.5f};
    }
    r.n_goals = ng;

    std::snprintf(r.text, sizeof(r.text),
        "[heuristic] e=%.2f => %s w=%.2f",
        job.energy,
        r.goals[0].kind == GoalKind::SEEK_FOOD ? "seek_food" : "explore",
        r.goals[0].weight);
    return r;
}

} // namespace corvid

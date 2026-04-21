#pragma once
#include <cstdint>

namespace corvid {

enum class GoalKind : uint8_t {
    SEEK_FOOD       = 0,
    AVOID_PREDATOR  = 1,
    EXPLORE         = 2,
    REST            = 3,
};

struct GoalEntry {
    GoalKind kind   = GoalKind::EXPLORE;
    float    weight = 0.f;  // [0, 1]
};

static constexpr int MAX_GOALS = 4;

} // namespace corvid

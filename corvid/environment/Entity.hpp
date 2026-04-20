#pragma once
#include "al/graphics/al_Graphics.hpp"
#include "al/math/al_Vec.hpp"
#include <string>

namespace corvid {

// Memory kind indices matching spec §2.2.1 (subset used by entities).
enum MemoryKind : int {
    MK_PREDATOR = 0, MK_FOOD = 1, MK_CACHE = 2,
    MK_CALL_HEARD = 3, MK_CONSPECIFIC = 4, MK_DEATH_WITNESSED = 5,
    MK_HAZARD = 6, MK_REFLECTION = 7, MK_BIRTH = 8,
    MK_TEACHING = 9, MK_OBSTACLE = 10, MK_WEATHER = 11, MK_NOVELTY = 12
};

// Entity category for difficulty manager (M8) and logging.
enum EntityCategory { PLANT, PREDATOR, OBSTACLE, HAZARD, TERRAIN, STRUCTURE };

// Result of an agent-entity interaction (spec §2.6.2).
struct InteractionResult {
    float  energy_delta    = 0.f;   // applied to agent
    float  valence         = 0.f;   // written to place grid
    int    memory_kind     = MK_NOVELTY;
    bool   agent_dies      = false;
    bool   entity_consumed = false; // e.g. plant eaten
};

// Base entity class (spec §2.6.2).
// CPU tick budget: 50 µs/entity (not enforced in M1; asserted in M7+).
class Entity {
public:
    std::string    name;
    EntityCategory category  = PLANT;
    al::Vec3f      position;
    bool           alive     = true;

    virtual ~Entity() = default;

    // Per-sim-tick update.  dt is wall-dt passed by the sim.
    virtual void tick(float dt, float sim_time) = 0;

    // Called when an agent centre is within interaction_radius() of the entity.
    virtual InteractionResult on_interact(float sim_time) = 0;

    // Render the entity using Allolib graphics.
    virtual void draw(al::Graphics& g) = 0;

    // Spatial test radius (used by the engine before calling on_interact).
    virtual float interaction_radius() const = 0;

    // Broad-phase AABB half-extent (for spatial hash or culling).
    virtual float broad_radius() const { return interaction_radius() * 1.5f; }
};

} // namespace corvid

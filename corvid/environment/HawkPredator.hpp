#pragma once
#include "Entity.hpp"
#include "al/graphics/al_Shapes.hpp"
#include "al/math/al_Random.hpp"
#include "al/math/al_Vec.hpp"
#include <vector>

namespace corvid {

// HawkPredator — aerial predator (spec entity set, M1).
// Chases the nearest living agent; deals E_damage on strike; agents that
// survive write MK_PREDATOR events to the place grid.
class HawkPredator : public Entity {
public:
    al::Vec3f velocity;
    float     speed      = 4.5f;
    float     E_damage   = 0.55f;
    float     detect_r   = 3.5f;
    float     W          = 10.f;   // world side length (set by engine)
    float     last_strike = -999.f;
    float     strike_cd   = 0.8f;  // seconds between strikes

    explicit HawkPredator(al::Vec3f pos, float world_W = 10.f) {
        name     = "HawkPredator";
        category = PREDATOR;
        position = pos;
        W        = world_W;
        alive    = true;
        velocity = al::Vec3f(0.5f, 0.2f, 0.3f).normalize() * speed;
    }

    // Call from the engine, passing only the positions of live agents.
    // Returns index of struck agent (-1 if no strike this tick).
    int tickWithAgents(float dt, float sim_time,
                       const std::vector<std::pair<int, al::Vec3f>>& live_agents)
    {
        // Find nearest agent within detect range
        float best = detect_r * detect_r;
        al::Vec3f target = position + velocity.normalize() * 2.f;
        for (auto& [idx, pos] : live_agents) {
            float d2 = (pos - position).magSqr();
            if (d2 < best) { best = d2; target = pos; }
        }

        // Steer toward target
        al::Vec3f desired = (target - position).normalize() * speed;
        velocity = (velocity * 0.85f + desired * 0.15f);
        if (velocity.mag() > speed) velocity = velocity.normalize() * speed;
        position += velocity * dt;

        // Toroidal wrap
        for (int i = 0; i < 3; ++i) {
            if (position[i] >  W * 0.5f) position[i] -= W;
            if (position[i] < -W * 0.5f) position[i] += W;
        }

        // Strike check
        if (sim_time - last_strike < strike_cd) return -1;
        for (auto& [idx, pos] : live_agents) {
            if ((pos - position).mag() < interaction_radius()) {
                last_strike = sim_time;
                return idx;  // engine applies E_damage and writes place event
            }
        }
        return -1;
    }

    // Thin wrapper for the base interface (unused in M1 — engine calls tickWithAgents).
    void tick(float /*dt*/, float /*sim_time*/) override {}

    InteractionResult on_interact(float /*sim_time*/) override {
        return {-E_damage, /*valence=*/-0.9f, MK_PREDATOR, false, false};
    }

    float interaction_radius() const override { return 0.4f; }

    void draw(al::Graphics& g) override {
        g.pushMatrix();
        g.translate(position);
        // Orient along velocity
        if (velocity.mag() > 0.01f) {
            al::Vec3f fwd = velocity.normalize();
            g.rotate(al::Quatf::getRotationTo({0, 0, -1}, fwd));
        }
        g.scale(0.30f);
        al::Mesh m{al::Mesh::TRIANGLES};
        al::addTetrahedron(m);
        g.color(0.95f, 0.15f, 0.1f, 0.92f);
        g.draw(m);
        g.popMatrix();
    }
};

} // namespace corvid

#pragma once
#include "Entity.hpp"
#include "al/graphics/al_Shapes.hpp"

namespace corvid {

// AcornPlant — nutritive entity (spec entity set, M1).
// Fixed world position; restores agent energy on contact; respawns after
// T_respawn seconds.  Contributes MK_FOOD events to the place grid.
class AcornPlant : public Entity {
public:
    float E_food       = 0.45f;   // energy restored per interaction
    float T_respawn    = 10.f;    // seconds until regrowth
    float respawn_at   = -1.f;    // sim_time when plant reappears; -1 = alive

    explicit AcornPlant(al::Vec3f pos) {
        name     = "AcornPlant";
        category = PLANT;
        position = pos;
        alive    = true;
    }

    void tick(float /*dt*/, float sim_time) override {
        if (!alive && respawn_at >= 0.f && sim_time >= respawn_at) {
            alive      = true;
            respawn_at = -1.f;
        }
    }

    InteractionResult on_interact(float sim_time) override {
        if (!alive) return {};
        alive      = false;
        respawn_at = sim_time + T_respawn;
        return {E_food, /*valence=*/0.8f, MK_FOOD, /*agent_dies=*/false, /*entity_consumed=*/true};
    }

    float interaction_radius() const override { return 0.35f; }

    void draw(al::Graphics& g) override {
        g.pushMatrix();
        g.translate(position);
        float s = alive ? 0.18f : 0.06f;
        g.scale(s);
        al::Mesh m{al::Mesh::TRIANGLES};
        al::addSphere(m, 1.0, 8, 6);
        if (alive)
            g.color(0.1f, 0.85f, 0.2f, 0.9f);
        else
            g.color(0.15f, 0.3f, 0.1f, 0.3f);  // withered
        g.draw(m);
        g.popMatrix();
    }
};

} // namespace corvid

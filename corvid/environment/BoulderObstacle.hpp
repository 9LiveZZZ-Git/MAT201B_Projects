#pragma once
#include "Entity.hpp"
#include "al/graphics/al_Shapes.hpp"

namespace corvid {

// BoulderObstacle — static deflector (spec entity set, M1).
// Agents approaching within interaction_radius() are repelled; writes
// MK_OBSTACLE events to the place grid.
class BoulderObstacle : public Entity {
public:
    float radius = 0.4f;  // collision/render radius

    explicit BoulderObstacle(al::Vec3f pos, float r = 0.4f) {
        name     = "BoulderObstacle";
        category = OBSTACLE;
        position = pos;
        radius   = r;
        alive    = true;  // boulders never die
    }

    void tick(float /*dt*/, float /*sim_time*/) override {}

    InteractionResult on_interact(float /*sim_time*/) override {
        return {0.f, /*valence=*/-0.1f, MK_OBSTACLE, false, false};
    }

    float interaction_radius() const override { return radius; }

    void draw(al::Graphics& g) override {
        g.pushMatrix();
        g.translate(position);
        g.scale(radius * 1.4f);
        al::Mesh m{al::Mesh::TRIANGLES};
        al::addCube(m);
        g.color(0.45f, 0.42f, 0.40f, 0.88f);
        g.draw(m);
        g.popMatrix();
    }
};

} // namespace corvid

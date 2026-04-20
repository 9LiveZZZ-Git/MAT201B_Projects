#pragma once
#include "Memory.hpp"
#include <array>
#include <cstdint>

namespace corvid {

// Fixed-capacity ring buffer with salience-weighted eviction (spec §2.4.1).
// When full, the slot with lowest salience is overwritten rather than the
// oldest slot — preserving high-salience (surprising / consequential) events.
// Spec target capacity is 2048; 256 is used for M2 to keep Agent size sane.
template <int CAP = 256>
struct MemoryRing {
    std::array<Memory, CAP> buf{};
    int      head  = 0;  // next write slot (ring)
    int      count = 0;  // number of valid entries [0, CAP]
    uint64_t next_id = 1;

    void push(Memory m) {
        m.id = next_id++;
        if (count < CAP) {
            buf[head] = m;
            head = (head + 1) % CAP;
            ++count;
        } else {
            // Evict lowest-salience slot
            int evict = 0;
            float worst = buf[0].salience;
            for (int i = 1; i < CAP; ++i) {
                if (buf[i].salience < worst) { worst = buf[i].salience; evict = i; }
            }
            buf[evict] = m;
        }
    }

    // Iterate valid entries (unordered).
    const Memory& get(int i) const { return buf[i % CAP]; }
    int size() const { return count; }

    // Decay all saliences each sim frame (call once per tick).
    void decaySalience(float factor = 0.9995f) {
        for (int i = 0; i < count; ++i)
            buf[i].salience *= factor;
    }

    // Return the most recent memory of a given kind (-1 index if not found).
    int mostRecent(int kind) const {
        int best = -1;
        float bt = -1.f;
        for (int i = 0; i < count; ++i) {
            if (buf[i].kind == kind && buf[i].timestamp > bt) {
                bt = buf[i].timestamp; best = i;
            }
        }
        return best;
    }
};

} // namespace corvid

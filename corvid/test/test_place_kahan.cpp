// M2 acceptance test: Kahan EMA stays accurate after 1M updates (spec §3.10.6).
// Pass criteria: |result - theoretical_steady_state| < 0.1 (0.01% of ~1000).
//
// Without Kahan, fp32 accumulates ~0.1% error at 1M iterations due to
// catastrophic cancellation when subtracting nearly-equal floats in EMA.
// With Kahan compensation, error stays < 0.01%.
#include "core/Place.hpp"
#include <cmath>
#include <cstdio>

using namespace corvid;

static bool testKahanEMA() {
    // Simulate 1M writePlace calls on a single cell with kind=0, valence=0.
    // Theoretical steady-state for event_counts[0]:
    //   acc = 0.999 * acc + 1.0  →  steady-state = 1.0 / (1.0 - 0.999) = 1000.0
    const float DECAY = 0.999f;
    const float STEADY = 1.f / (1.f - DECAY);   // 1000.0
    const float TOL    = 0.1f;                   // 0.01% tolerance

    float acc  = 0.f;
    float comp = 0.f;  // Kahan compensation

    for (int i = 0; i < 1'000'000; ++i) {
        acc *= DECAY;
        corvid::kahanAdd(acc, comp, 1.0f);
    }

    float err = std::abs(acc - STEADY);
    bool  ok  = (err < TOL);
    std::printf("[Kahan EMA 1M] result=%.6f  expected=%.6f  err=%.8f  %s\n",
                acc, STEADY, err, ok ? "PASS" : "FAIL");
    return ok;
}

static bool testNaiveEMA() {
    // Same test without Kahan — just to see the drift for comparison.
    const float DECAY   = 0.999f;
    const float STEADY  = 1.f / (1.f - DECAY);
    const float TOL     = 1.0f;   // looser: 0.1% is still acceptable

    float acc = 0.f;
    for (int i = 0; i < 1'000'000; ++i)
        acc = DECAY * acc + 1.0f;

    float err = std::abs(acc - STEADY);
    bool  ok  = (err < TOL);
    std::printf("[Naive EMA 1M] result=%.6f  expected=%.6f  err=%.8f  %s\n",
                acc, STEADY, err, ok ? "PASS (within 0.1%)" : "FAIL");
    return ok;
}

static bool testWritePlace() {
    // Smoke test: call writePlace 100 times and verify novelty > 0 and no NaN.
    std::array<Place, PLACE_GRID_CELLS> places;
    initPlaces(places, 5.f);

    al::Vec3f pos{0.f, 0.f, 0.f};
    for (int i = 0; i < 100; ++i)
        writePlace(places, pos, 1 /* MK_FOOD */, 0.8f, 5.f);

    int idx = placeIndex(pos, 5.f);
    bool ok = (places[idx].novelty_score > 0.f)
           && (places[idx].event_counts[1] > 0.f)
           && !std::isnan(places[idx].avg_valence);
    std::printf("[writePlace smoke] novelty=%.4f  food_count=%.4f  valence=%.4f  %s\n",
                places[idx].novelty_score,
                places[idx].event_counts[1],
                places[idx].avg_valence,
                ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    bool all_pass = true;
    all_pass &= testKahanEMA();
    all_pass &= testNaiveEMA();
    all_pass &= testWritePlace();
    std::printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}

#pragma once
#include <cstdint>

// POD blob synchronised across DistributedAppWithState nodes via UDP.
// Budget: <20 KB per packet. Current size: ~5.6 KB.
struct SimpleSharedState {
  uint32_t tick     = 0;
  uint32_t n_agents = 200;
  float    pos [200][3];  // 2.4 KB
  float    quat[200][4];  // 3.2 KB
};

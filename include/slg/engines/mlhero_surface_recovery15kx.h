#ifndef SLG_MLHERO_SURFACE_RECOVERY15KX_H
#define SLG_MLHERO_SURFACE_RECOVERY15KX_H
#include <atomic>
namespace slg { namespace mlhero15kx {
// Diagnostic counters only; recovery itself is independent of reporter/mode.
inline std::atomic<unsigned long long> counters[2][3]{};
inline void Reset(){for(auto&side:counters)for(auto&value:side)value=0ull;}
} }
#endif

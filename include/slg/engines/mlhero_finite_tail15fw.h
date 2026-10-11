#ifndef MLHERO_FINITE_TAIL15FW_H
#define MLHERO_FINITE_TAIL15FW_H
#include <algorithm>
#include <cmath>
// Diagnostic thresholds on raw contributions, not normalized film pixels.
// Independent budgets keep ordinary bright lights from exhausting color-tail logs.
inline unsigned MLHeroFiniteTailClass15fw(double r, double g, double b) {
    if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) return 0u;
    const double peak = std::max(std::fabs(r), std::max(std::fabs(g), std::fabs(b)));
    const double y = .2126*r + .7152*g + .0722*b;
    return (peak >= 64. ? 1u : 0u) |
        ((peak >= 16. && peak >= 8.*std::max(std::fabs(y), 1e-12)) ? 2u : 0u);
}
#endif

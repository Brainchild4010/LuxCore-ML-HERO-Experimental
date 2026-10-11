#ifndef MLHERO_UNIT_REFLECTANCE15FX_H
#define MLHERO_UNIT_REFLECTANCE15FX_H
#include <cmath>
// Only for actual dimensionless material reflectance, never f*cos/pdf or flux.
// Preserve invalid values for existing diagnostics rather than hiding them.
inline float MLHeroUnitReflectance15fx(const float value) {
    if (!std::isfinite(value)) return value;
    return value < 0.f ? 0.f : (value > 1.f ? 1.f : value);
}
#endif

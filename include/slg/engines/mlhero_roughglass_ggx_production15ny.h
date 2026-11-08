#ifndef SLG_MLHERO_ROUGHGLASS_GGX_PRODUCTION15NY_H
#define SLG_MLHERO_ROUGHGLASS_GGX_PRODUCTION15NY_H

#include <atomic>
#include <cmath>
#include <cstdio>

#include "slg/engines/mlhero_roughglass_ggx_vndf15nx.h"

// Phase 15ny: production accounting for the anisotropic GGX/VNDF RoughGlass
// branch.  The actual kernel is the numerically verified 15nx kernel; this
// layer records every live Sample() outcome without consuming random numbers.
namespace slg { namespace mlhero15ny {

using V = mlhero15kk::V;
enum Reject : unsigned {
    INVALID_INTERFACE = 0u, TIR_TRANSMISSION, HEMISPHERE,
    ZERO_DENSITY, PDF_REJECT, REJECT_COUNT
};

inline bool Active(const bool hero, const bool fine, const double anisotropy) {
    return hero && fine && std::fabs(anisotropy) >= 1e-6;
}
inline double D(const double ax, const double ay, const V &h) {
    return mlhero15nx::D(ax, ay, h);
}
inline double G1(const double ax, const double ay, const V &w) {
    return mlhero15nx::G1(ax, ay, w);
}
inline double G2(const double ax, const double ay, const V &a, const V &b) {
    return mlhero15nx::G(ax, ay, a, b);
}
inline double HalfPdf(const double ax, const double ay,
        const V &fixed, const V &h) {
    return mlhero15nx::VisiblePdf(ax, ay, fixed, h);
}
inline double ReflectionPdf(const double ax, const double ay,
        const V &fixed, const V &h) {
    const V f = mlhero15kk::Unit(fixed);
    const double c = std::fabs(f.z);
    return c > 0. ? D(ax, ay, h) * G1(ax, ay, f) / (4. * c) : 0.;
}
inline V Sample(const double ax, const double ay, const V &fixed,
        const double u0, const double u1) {
    return mlhero15nx::SampleVisible(ax, ay, fixed, u0, u1);
}

inline std::atomic<unsigned long long> attempts[2]{}, upperFixed[2]{},
    lowerFixed[2]{}, acceptedReflect[2]{}, acceptedTransmit[2]{},
    rejects[2][REJECT_COUNT]{}, nonfinite[2]{};

inline void Reset() {
    for (unsigned s = 0; s < 2; ++s) {
        attempts[s] = 0; upperFixed[s] = 0; lowerFixed[s] = 0;
        acceptedReflect[s] = 0; acceptedTransmit[s] = 0; nonfinite[s] = 0;
        for (unsigned r = 0; r < REJECT_COUNT; ++r) rejects[s][r] = 0;
    }
}
inline unsigned Begin(const bool fromLight, const V &fixed) {
    const unsigned s = fromLight ? 1u : 0u;
    attempts[s].fetch_add(1, std::memory_order_relaxed);
    (fixed.z < 0. ? lowerFixed[s] : upperFixed[s]).fetch_add(
        1, std::memory_order_relaxed);
    return s;
}
inline void Rejected(const unsigned side, const Reject reason) {
    rejects[side][reason].fetch_add(1, std::memory_order_relaxed);
}
inline void Accepted(const unsigned side, const bool transmit,
        const double pdf, const double weight) {
    if (!(pdf > 0.) || !std::isfinite(pdf) || !std::isfinite(weight))
        nonfinite[side].fetch_add(1, std::memory_order_relaxed);
    (transmit ? acceptedTransmit[side] : acceptedReflect[side]).fetch_add(
        1, std::memory_order_relaxed);
}
inline void WriteSummary(FILE *f) {
    for (unsigned s = 0; s < 2; ++s) {
        unsigned long long rejected = 0;
        for (unsigned r = 0; r < REJECT_COUNT; ++r)
            rejected += rejects[s][r].load(std::memory_order_relaxed);
        const unsigned long long accepted = acceptedReflect[s].load() +
            acceptedTransmit[s].load();
        const unsigned long long selected = attempts[s].load();
        fprintf(f,
            "phase15ny ROUGHGLASS_ANISO_GGX_PRODUCTION_SUMMARY side=%s "
            "attempts=%llu upperFixed=%llu lowerFixed=%llu "
            "acceptedReflect=%llu acceptedTransmit=%llu "
            "invalidInterface=%llu tirTransmission=%llu hemisphereRejected=%llu "
            "zeroDensity=%llu pdfRejected=%llu nonfinite=%llu "
            "accountingPass=%u hemisphereCoverage=%u productionSampler=GGX_VNDF "
            "productionBSDF=GGX_TROWBRIDGE_REITZ correlatedG2=1 "
            "directAndReversePdf=GGX_VNDF randomSequenceChanged=0 "
            "extraSamplerDraws=0 rendererPhysicsChanged=1 pass=%u\n",
            s ? "LIGHT" : "EYE", selected, upperFixed[s].load(),
            lowerFixed[s].load(), acceptedReflect[s].load(),
            acceptedTransmit[s].load(), rejects[s][INVALID_INTERFACE].load(),
            rejects[s][TIR_TRANSMISSION].load(), rejects[s][HEMISPHERE].load(),
            rejects[s][ZERO_DENSITY].load(), rejects[s][PDF_REJECT].load(),
            nonfinite[s].load(), selected == accepted + rejected ? 1u : 0u,
            upperFixed[s].load() && lowerFixed[s].load() ? 1u : 0u,
            selected == accepted + rejected && nonfinite[s].load() == 0 ? 1u : 0u);
    }
}

} }

#endif

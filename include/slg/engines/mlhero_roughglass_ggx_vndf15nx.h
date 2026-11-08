#ifndef SLG_MLHERO_ROUGHGLASS_GGX_VNDF15NX_H
#define SLG_MLHERO_ROUGHGLASS_GGX_VNDF15NX_H

#include <atomic>
#include <cmath>
#include <cstdio>

#include "slg/engines/mlhero_rough_optics15kk.h"

// Phase 15nx: diagnostic-only anisotropic Trowbridge-Reitz visible-normal
// kernel, following PBRT v4.  It is deliberately independent of LuxCore's
// existing Schlick anisotropy and does not alter a production sample.
namespace slg { namespace mlhero15nx {

using V = mlhero15kk::V;
constexpr double Pi = 3.14159265358979323846264338327950288;
constexpr unsigned BudgetPerThreadAndSide = 256u;

inline bool Finite(const double v) { return std::isfinite(v); }
inline bool Finite(const V &v) {
    return Finite(v.x) && Finite(v.y) && Finite(v.z);
}
inline double Clamp01(const double v) {
    return std::fmax(0., std::fmin(1., v));
}

inline double D(const double alphaX, const double alphaY, V wm) {
    wm = mlhero15kk::Unit(wm);
    if (!(alphaX > 0.) || !(alphaY > 0.) || !(wm.z > 0.) || !Finite(wm))
        return 0.;
    // Algebraically identical to PBRT's tan^2/cos^4 form, but stable at the
    // normal and without an infinity-times-zero intermediate near grazing.
    const double e = wm.x * wm.x / (alphaX * alphaX) +
        wm.y * wm.y / (alphaY * alphaY) + wm.z * wm.z;
    return e > 0. ? 1. / (Pi * alphaX * alphaY * e * e) : 0.;
}

inline double Lambda(const double alphaX, const double alphaY, V w) {
    w = mlhero15kk::Unit(w);
    if (!(alphaX > 0.) || !(alphaY > 0.) || !Finite(w)) return 0.;
    const double z2 = w.z * w.z;
    if (!(z2 > 0.)) return 0.;
    const double a2Tan2 = (alphaX * alphaX * w.x * w.x +
        alphaY * alphaY * w.y * w.y) / z2;
    return .5 * (std::sqrt(1. + a2Tan2) - 1.);
}
inline double G1(const double alphaX, const double alphaY, const V &w) {
    return 1. / (1. + Lambda(alphaX, alphaY, w));
}
inline double G(const double alphaX, const double alphaY,
        const V &wo, const V &wi) {
    return 1. / (1. + Lambda(alphaX, alphaY, wo) +
        Lambda(alphaX, alphaY, wi));
}

inline double VisiblePdf(const double alphaX, const double alphaY,
        V w, V wm) {
    w = mlhero15kk::Unit(w); wm = mlhero15kk::Unit(wm);
    const double cosW = std::fabs(w.z);
    const double projected = mlhero15kk::Dot(w, wm);
    if (!(cosW > 0.) || !(wm.z > 0.) ||
            !(projected * w.z > 0.) || !Finite(w) || !Finite(wm))
        return 0.;
    return D(alphaX, alphaY, wm) * G1(alphaX, alphaY, w) *
        std::fabs(projected) / cosW;
}

inline V SampleVisible(const double alphaX, const double alphaY,
        V w, const double u0, const double u1) {
    w = mlhero15kk::Unit(w);
    V wh = mlhero15kk::Unit(V(alphaX * w.x, alphaY * w.y, w.z));
    if (wh.z < 0.) wh = V(-wh.x, -wh.y, -wh.z);

    V t1;
    if (wh.z < .99999) {
        const double invLength = 1. / std::sqrt(wh.x * wh.x + wh.y * wh.y);
        t1 = V(-wh.y * invLength, wh.x * invLength, 0.);
    } else {
        t1 = V(1., 0., 0.);
    }
    const V t2 = mlhero15kk::Cross(wh, t1);

    const double radius = std::sqrt(Clamp01(u0));
    const double phi = 2. * Pi * Clamp01(u1);
    const double px = radius * std::cos(phi);
    const double diskY = radius * std::sin(phi);
    const double h = std::sqrt(std::fmax(0., 1. - px * px));
    const double lerp = .5 * (1. + wh.z);
    const double py = (1. - lerp) * h + lerp * diskY;
    const double pz = std::sqrt(std::fmax(0., 1. - px * px - py * py));
    const V nh = t1 * px + t2 * py + wh * pz;
    return mlhero15kk::Unit(V(alphaX * nh.x, alphaY * nh.y,
        std::fmax(1e-6, nh.z)));
}

struct Record {
    unsigned side = 0u, material = 0u;
    unsigned long long detail = 0ull;
    double alphaX = 0., alphaY = 0., u0 = 0., u1 = 0.;
    V fixed, oldWh, candidateWh;
    double oldD = 0., oldPdf = 0., oldDOverPdf = 0.;
    double candidateD = 0., candidateG1 = 0., candidatePdf = 0.;
    bool valid = false, visible = false;
};
using Reporter = void (*)(const Record &);
inline thread_local Reporter reporter = nullptr;
inline thread_local unsigned used[2]{};
inline std::atomic<unsigned long long> attempts[2]{}, valid[2]{}, invalid[2]{},
    visibilityFailures[2]{}, oldBelowHalf[2]{}, oldBelowTenth[2]{},
    oldAboveTwo[2]{}, oldAboveTen[2]{}, details[2]{}, writeFailures[2]{};
inline std::atomic<double> oldDOverPdfSum[2]{}, oldDOverPdfMax[2]{};

inline void AtomicMax(std::atomic<double> &target, const double value) {
    double current = target.load(std::memory_order_relaxed);
    while (current < value && !target.compare_exchange_weak(current, value,
            std::memory_order_relaxed, std::memory_order_relaxed)) { }
}
inline void AtomicAdd(std::atomic<double> &target, const double value) {
    double current = target.load(std::memory_order_relaxed);
    while (!target.compare_exchange_weak(current, current + value,
            std::memory_order_relaxed, std::memory_order_relaxed)) { }
}

inline void Reset() {
    for (unsigned side = 0u; side < 2u; ++side) {
        attempts[side] = 0ull; valid[side] = 0ull; invalid[side] = 0ull;
        visibilityFailures[side] = 0ull; oldBelowHalf[side] = 0ull;
        oldBelowTenth[side] = 0ull; oldAboveTwo[side] = 0ull;
        oldAboveTen[side] = 0ull; details[side] = 0ull;
        writeFailures[side] = 0ull; oldDOverPdfSum[side] = 0.;
        oldDOverPdfMax[side] = 0.;
    }
}

struct WorkerScope {
    WorkerScope(const bool on, Reporter callback) {
        used[0] = used[1] = 0u;
        reporter = on ? callback : nullptr;
    }
    ~WorkerScope() { reporter = nullptr; used[0] = used[1] = 0u; }
    WorkerScope(const WorkerScope &) = delete;
    WorkerScope &operator=(const WorkerScope &) = delete;
};

inline void Observe(const bool fromLight, const unsigned material,
        const double alphaX, const double alphaY, const double u0,
        const double u1, V fixed, V oldWh, const double oldD,
        const double oldPdf) {
    const unsigned side = fromLight ? 1u : 0u;
    if (!reporter || used[side] >= BudgetPerThreadAndSide) return;
    ++used[side];

    Record r;
    r.side = side; r.material = material; r.alphaX = alphaX;
    r.alphaY = alphaY; r.u0 = u0; r.u1 = u1;
    r.fixed = mlhero15kk::Unit(fixed); r.oldWh = mlhero15kk::Unit(oldWh);
    r.oldD = oldD; r.oldPdf = oldPdf;
    r.oldDOverPdf = oldPdf > 0. ? oldD / oldPdf : 0.;
    r.candidateWh = SampleVisible(alphaX, alphaY, r.fixed, u0, u1);
    r.candidateD = D(alphaX, alphaY, r.candidateWh);
    r.candidateG1 = G1(alphaX, alphaY, r.fixed);
    r.candidatePdf = VisiblePdf(alphaX, alphaY, r.fixed, r.candidateWh);
    r.visible = mlhero15kk::Dot(r.fixed, r.candidateWh) * r.fixed.z > 0.;
    r.valid = Finite(r.candidateWh) && r.candidateWh.z > 0. &&
        r.candidateD > 0. && Finite(r.candidateD) &&
        r.candidateG1 > 0. && r.candidateG1 <= 1. &&
        Finite(r.candidateG1) && r.candidatePdf > 0. &&
        Finite(r.candidatePdf) && r.visible;

    attempts[side].fetch_add(1ull, std::memory_order_relaxed);
    (r.valid ? valid[side] : invalid[side]).fetch_add(1ull,
        std::memory_order_relaxed);
    if (!r.visible)
        visibilityFailures[side].fetch_add(1ull, std::memory_order_relaxed);
    if (Finite(r.oldDOverPdf) && r.oldDOverPdf >= 0.) {
        AtomicAdd(oldDOverPdfSum[side], r.oldDOverPdf);
        AtomicMax(oldDOverPdfMax[side], r.oldDOverPdf);
        if (r.oldDOverPdf < .5) oldBelowHalf[side].fetch_add(1ull);
        if (r.oldDOverPdf < .1) oldBelowTenth[side].fetch_add(1ull);
        if (r.oldDOverPdf > 2.) oldAboveTwo[side].fetch_add(1ull);
        if (r.oldDOverPdf > 10.) oldAboveTen[side].fetch_add(1ull);
    }
    r.detail = details[side].fetch_add(1ull, std::memory_order_relaxed);
    if (r.detail < 16ull) reporter(r);
}

inline void WriteRecord(FILE *f, const Record &r, const unsigned thread,
        const unsigned sample) {
    fprintf(f,
        "phase15nx ROUGHGLASS_ANISO_GGX_VNDF detail=%llu side=%s thread=%u "
        "sample=%u materialID=%u alphaX=%.17g alphaY=%.17g "
        "u=(%.17g,%.17g) fixed=(%.17g,%.17g,%.17g) "
        "oldWh=(%.17g,%.17g,%.17g) oldD=%.17g oldPdf=%.17g "
        "oldDOverPdf=%.17g candidateWh=(%.17g,%.17g,%.17g) "
        "candidateD=%.17g candidateG1=%.17g candidateVisiblePdf=%.17g "
        "visible=%u valid=%u diagnosticOnly=1 transportChanged=0 "
        "samplerDimensionsChanged=0\n",
        r.detail, r.side ? "LIGHT" : "EYE", thread, sample, r.material,
        r.alphaX, r.alphaY, r.u0, r.u1, r.fixed.x, r.fixed.y, r.fixed.z,
        r.oldWh.x, r.oldWh.y, r.oldWh.z, r.oldD, r.oldPdf,
        r.oldDOverPdf, r.candidateWh.x, r.candidateWh.y, r.candidateWh.z,
        r.candidateD, r.candidateG1, r.candidatePdf,
        r.visible ? 1u : 0u, r.valid ? 1u : 0u);
}

inline void WriteSummary(FILE *f) {
    for (unsigned side = 0u; side < 2u; ++side) {
        const unsigned long long n = attempts[side].load();
        const unsigned long long ok = valid[side].load();
        const unsigned long long bad = invalid[side].load();
        fprintf(f,
            "phase15nx ROUGHGLASS_ANISO_GGX_VNDF_SUMMARY side=%s "
            "selectedDraws=%llu validCandidates=%llu invalidCandidates=%llu "
            "visibilityFailures=%llu oldDOverPdfBelowHalf=%llu "
            "oldDOverPdfBelowTenth=%llu oldDOverPdfAboveTwo=%llu "
            "oldDOverPdfAboveTen=%llu meanOldDOverPdf=%.17g "
            "maxOldDOverPdf=%.17g detailsSelected=%llu writeFailures=%llu "
            "accountingPass=%u candidatePass=%u diagnosticOnly=1 "
            "transportChanged=0 samplerDimensionsChanged=0\n",
            side ? "LIGHT" : "EYE", n, ok, bad,
            visibilityFailures[side].load(), oldBelowHalf[side].load(),
            oldBelowTenth[side].load(), oldAboveTwo[side].load(),
            oldAboveTen[side].load(),
            n ? oldDOverPdfSum[side].load() / (double)n : 0.,
            oldDOverPdfMax[side].load(),
            details[side].load() < 16ull ? details[side].load() : 16ull,
            writeFailures[side].load(), n == ok + bad ? 1u : 0u,
            n > 0ull && bad == 0ull &&
                visibilityFailures[side].load() == 0ull ? 1u : 0u);
    }
}

} }

#endif

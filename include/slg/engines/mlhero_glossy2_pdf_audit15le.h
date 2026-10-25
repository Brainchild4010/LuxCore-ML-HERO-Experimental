#ifndef SLG_MLHERO_GLOSSY2_PDF_AUDIT15LE_H
#define SLG_MLHERO_GLOSSY2_PDF_AUDIT15LE_H

#include <atomic>
#include <cmath>
#include <cstdio>

#include "slg/engines/mlhero_metal_pdf15lb.h"
#include "slg/engines/mlhero_rough_optics15kk.h"

namespace slg { namespace mlhero15le {
using V = mlhero15kk::V;
constexpr unsigned Budget = 256, BranchCount = 2, StatusCount = 4;
enum Branch : unsigned { BASE = 0, COATING = 1 };
enum Status : unsigned { DRAW_ONLY = 0, BASE_REJECT, COATING_REJECT, ACCEPT };

struct Record {
    unsigned side = 0, branch = 0, material = 0, status = 0;
    unsigned long long detail = 0;
    bool doubleSided = false;
    double rough = 0., anis = 0., u0 = 0., u1 = 0., branchU = 0.;
    double ksFilter = 0., wBase = 0., wCoating = 0., referenceWBase = 0.;
    double wBaseReverse = 0., wCoatingReverse = 0., referenceWBaseReverse = 0.;
    V fixed, sampled, h;
    double reportedHalfPdf = 0., referenceHalfPdf = 0.;
    double reportedCoatingPdf = 0., referenceCoatingPdf = 0.;
    double reportedCoatingPdfReverse = 0., referenceCoatingPdfReverse = 0.;
    double reportedMixturePdf = 0., referenceMixturePdf = 0.;
    double reportedMixturePdfReverse = 0., referenceMixturePdfReverse = 0.;
    double reportedMixtureReplay = 0.;
};

using Reporter = void (*)(const Record &);
inline thread_local Reporter reporter = nullptr;
inline thread_local unsigned used[2][BranchCount]{};
inline std::atomic<unsigned long long> attempts[2][BranchCount]{}, status[2][BranchCount][StatusCount]{},
    selectionMismatch[2][BranchCount]{}, invalidReference[2][BranchCount]{},
    weightMismatch[2][BranchCount]{}, weightReverseMismatch[2][BranchCount]{},
    halfMismatch[2][BranchCount]{}, coatingMismatch[2][BranchCount]{},
    coatingReverseMismatch[2][BranchCount]{}, mixtureMismatch[2][BranchCount]{},
    mixtureReverseMismatch[2][BranchCount]{}, replayMismatch[2][BranchCount]{},
    details[2][BranchCount]{}, writeFailures[2][BranchCount]{};

inline double Relative(double a, double b) {
    return std::fabs(a - b) /
        std::fmax(1e-300, std::fmax(std::fabs(a), std::fabs(b)));
}

inline void Reset() {
    for (unsigned s = 0; s < 2; ++s) for (unsigned b = 0; b < BranchCount; ++b) {
        attempts[s][b] = 0; selectionMismatch[s][b] = 0; invalidReference[s][b] = 0;
        weightMismatch[s][b] = 0; weightReverseMismatch[s][b] = 0;
        halfMismatch[s][b] = 0; coatingMismatch[s][b] = 0;
        coatingReverseMismatch[s][b] = 0; mixtureMismatch[s][b] = 0;
        mixtureReverseMismatch[s][b] = 0; replayMismatch[s][b] = 0;
        details[s][b] = 0; writeFailures[s][b] = 0;
        for (unsigned k = 0; k < StatusCount; ++k) status[s][b][k] = 0;
    }
}

struct WorkerScope {
    WorkerScope(bool enabled, Reporter callback) {
        for (auto &side : used) for (auto &value : side) value = 0;
        reporter = enabled ? callback : nullptr;
    }
    ~WorkerScope() {
        reporter = nullptr;
        for (auto &side : used) for (auto &value : side) value = 0;
    }
    WorkerScope(const WorkerScope &) = delete;
    WorkerScope &operator=(const WorkerScope &) = delete;
};

struct Attempt {
    bool active = false;
    Record r;

    Attempt(bool fromLight, unsigned material, bool sampledBase, bool doubleSided,
            double rough, double anis, double u0, double u1, double branchU,
            double wBase, double ksFilter, V fixed) {
        const unsigned s = fromLight ? 1u : 0u;
        const unsigned b = sampledBase ? BASE : COATING;
        if (!reporter || used[s][b] >= Budget) return;
        ++used[s][b]; active = true;
        r.side = s; r.branch = b; r.material = material; r.doubleSided = doubleSided;
        r.rough = rough; r.anis = anis; r.u0 = u0; r.u1 = u1; r.branchU = branchU;
        r.ksFilter = ksFilter; r.wBase = wBase; r.wCoating = 1. - wBase;
        r.fixed = mlhero15kk::Unit(fixed);
        const double directTail = std::pow(1. - std::fabs(r.fixed.z), 5.);
        const double directS = ksFilter + (1. - ksFilter) * directTail;
        r.referenceWBase = 1. - .5 * (1. + directS);
        attempts[s][b].fetch_add(1ull, std::memory_order_relaxed);
        if (sampledBase != (branchU < wBase))
            selectionMismatch[s][b].fetch_add(1ull, std::memory_order_relaxed);
    }

    void Reject(Status reason) { if (active) r.status = reason; }
    bool Active() const { return active; }

    void Accept(V sampled, double reportedMixture, double reportedCoating,
            double wBaseReverse) {
        if (!active) return;
        r.status = ACCEPT;
        r.sampled = mlhero15kk::Unit(sampled);
        r.h = mlhero15kk::Unit(r.fixed + r.sampled);
        r.wBaseReverse = wBaseReverse; r.wCoatingReverse = 1. - wBaseReverse;
        const double reverseTail = std::pow(1. - std::fabs(r.sampled.z), 5.);
        const double reverseS = r.ksFilter + (1. - r.ksFilter) * reverseTail;
        r.referenceWBaseReverse = 1. - .5 * (1. + reverseS);
        r.reportedMixturePdf = reportedMixture;
        r.reportedCoatingPdf = reportedCoating;

        const double dotDirect = std::fabs(mlhero15kk::Dot(r.fixed, r.h));
        const double dotReverse = std::fabs(mlhero15kk::Dot(r.sampled, r.h));
        const double jacDirect = dotDirect > 0. ? 1. / (4. * dotDirect) : 0.;
        const double jacReverse = dotReverse > 0. ? 1. / (4. * dotReverse) : 0.;
        r.reportedHalfPdf = jacDirect > 0. ? reportedCoating / jacDirect : 0.;
        r.referenceHalfPdf = mlhero15lb::HalfPdf(r.rough, r.anis,
            r.h.x, r.h.y, r.h.z);
        r.referenceCoatingPdf = r.referenceHalfPdf * jacDirect;
        r.referenceCoatingPdfReverse = r.referenceHalfPdf * jacReverse;
        r.reportedCoatingPdfReverse = r.reportedHalfPdf * jacReverse;
        const double baseDirect = std::fabs(r.sampled.z) / 3.14159265358979323846;
        const double baseReverse = std::fabs(r.fixed.z) / 3.14159265358979323846;
        r.referenceMixturePdf = r.wBase * baseDirect +
            r.wCoating * r.referenceCoatingPdf;
        r.referenceMixturePdfReverse = r.wBaseReverse * baseReverse +
            r.wCoatingReverse * r.referenceCoatingPdfReverse;
        r.reportedMixtureReplay = r.wBase * baseDirect +
            r.wCoating * r.reportedCoatingPdf;
        r.reportedMixturePdfReverse = r.wBaseReverse * baseReverse +
            r.wCoatingReverse * r.reportedCoatingPdfReverse;
    }

    ~Attempt() {
        if (!active) return;
        const unsigned s = r.side, b = r.branch;
        status[s][b][r.status].fetch_add(1ull, std::memory_order_relaxed);
        if (r.status == ACCEPT) {
            const bool valid = r.referenceHalfPdf > 0. &&
                std::isfinite(r.referenceHalfPdf) && r.referenceCoatingPdf > 0. &&
                std::isfinite(r.referenceCoatingPdf) && r.referenceMixturePdf > 0. &&
                std::isfinite(r.referenceMixturePdf) && r.referenceMixturePdfReverse > 0. &&
                std::isfinite(r.referenceMixturePdfReverse);
            if (!valid) invalidReference[s][b].fetch_add(1ull, std::memory_order_relaxed);
            else {
                if (Relative(r.wBase, r.referenceWBase) > 2e-4)
                    weightMismatch[s][b].fetch_add(1ull, std::memory_order_relaxed);
                if (Relative(r.wBaseReverse, r.referenceWBaseReverse) > 2e-4)
                    weightReverseMismatch[s][b].fetch_add(1ull, std::memory_order_relaxed);
                if (Relative(r.reportedHalfPdf, r.referenceHalfPdf) > 2e-4)
                    halfMismatch[s][b].fetch_add(1ull, std::memory_order_relaxed);
                if (Relative(r.reportedCoatingPdf, r.referenceCoatingPdf) > 2e-4)
                    coatingMismatch[s][b].fetch_add(1ull, std::memory_order_relaxed);
                if (Relative(r.reportedCoatingPdfReverse, r.referenceCoatingPdfReverse) > 2e-4)
                    coatingReverseMismatch[s][b].fetch_add(1ull, std::memory_order_relaxed);
                if (Relative(r.reportedMixturePdf, r.referenceMixturePdf) > 2e-4)
                    mixtureMismatch[s][b].fetch_add(1ull, std::memory_order_relaxed);
                if (Relative(r.reportedMixturePdfReverse, r.referenceMixturePdfReverse) > 2e-4)
                    mixtureReverseMismatch[s][b].fetch_add(1ull, std::memory_order_relaxed);
                if (Relative(r.reportedMixturePdf, r.reportedMixtureReplay) > 2e-4)
                    replayMismatch[s][b].fetch_add(1ull, std::memory_order_relaxed);
            }
        }
        r.detail = details[s][b].fetch_add(1ull, std::memory_order_relaxed);
        if (r.detail < 16ull) reporter(r);
    }

    Attempt(const Attempt &) = delete;
    Attempt &operator=(const Attempt &) = delete;
};

inline const char *BranchName(unsigned branch) { return branch == BASE ? "BASE" : "COATING"; }
inline const char *StatusName(unsigned value) {
    const char *names[StatusCount] = {"DRAW_ONLY", "BASE_REJECT", "COATING_REJECT", "ACCEPT"};
    return value < StatusCount ? names[value] : "UNKNOWN";
}

inline void WriteRecord(FILE *f, const Record &r, unsigned thread, unsigned sample) {
    fprintf(f, "phase15le GLOSSY2_PDF_SAMPLE side=%s branch=%s detail=%llu thread=%u sample=%u materialID=%u status=%s doubleSided=%u roughness=%.17g anisotropy=%.17g u=(%.17g,%.17g,%.17g) ksFilter=%.17g weights=(%.17g,%.17g) referenceWBase=%.17g reverseWeights=(%.17g,%.17g) referenceWBaseReverse=%.17g fixed=(%.17g,%.17g,%.17g) sampled=(%.17g,%.17g,%.17g) h=(%.17g,%.17g,%.17g) reportedHalfPdf=%.17g referenceHalfPdf=%.17g halfRatio=%.17g reportedCoatingPdf=%.17g referenceCoatingPdf=%.17g coatingRatio=%.17g reportedMixturePdf=%.17g referenceMixturePdf=%.17g mixtureRatio=%.17g reportedMixturePdfReverse=%.17g referenceMixturePdfReverse=%.17g reverseMixtureRatio=%.17g diagnosticOnly=1 physicsChanged=0\n",
        r.side ? "LIGHT" : "EYE", BranchName(r.branch), r.detail, thread, sample,
        r.material, StatusName(r.status), r.doubleSided ? 1u : 0u, r.rough, r.anis,
        r.u0, r.u1, r.branchU, r.ksFilter, r.wBase, r.wCoating, r.referenceWBase,
        r.wBaseReverse, r.wCoatingReverse, r.referenceWBaseReverse,
        r.fixed.x, r.fixed.y, r.fixed.z, r.sampled.x, r.sampled.y, r.sampled.z,
        r.h.x, r.h.y, r.h.z, r.reportedHalfPdf, r.referenceHalfPdf,
        r.reportedHalfPdf > 0. ? r.referenceHalfPdf / r.reportedHalfPdf : 0.,
        r.reportedCoatingPdf, r.referenceCoatingPdf,
        r.reportedCoatingPdf > 0. ? r.referenceCoatingPdf / r.reportedCoatingPdf : 0.,
        r.reportedMixturePdf, r.referenceMixturePdf,
        r.reportedMixturePdf > 0. ? r.referenceMixturePdf / r.reportedMixturePdf : 0.,
        r.reportedMixturePdfReverse, r.referenceMixturePdfReverse,
        r.reportedMixturePdfReverse > 0. ? r.referenceMixturePdfReverse / r.reportedMixturePdfReverse : 0.);
}

inline void WriteSummary(FILE *f) {
    for (unsigned s = 0; s < 2; ++s) for (unsigned b = 0; b < BranchCount; ++b)
        fprintf(f, "phase15le GLOSSY2_PDF_SUMMARY side=%s branch=%s selectedAttempts=%llu drawOnly=%llu baseRejected=%llu coatingRejected=%llu accepted=%llu selectionMismatches=%llu directWeightMismatches=%llu reverseWeightMismatches=%llu invalidReference=%llu halfPdfMismatches=%llu coatingPdfMismatches=%llu coatingReversePdfMismatches=%llu mixturePdfMismatches=%llu mixtureReversePdfMismatches=%llu reportedMixtureReplayMismatches=%llu detailsSelected=%llu writeFailures=%llu toleranceRelative=0.0002 reference=INDEPENDENT_SCHLICK_INVERSE_CDF_PLUS_COSINE_MIXTURE rejectedSamplesIncluded=1 conditionalRenormalization=NONE diagnosticOnly=1 physicsChanged=0\n",
            s ? "LIGHT" : "EYE", BranchName(b), attempts[s][b].load(),
            status[s][b][DRAW_ONLY].load(), status[s][b][BASE_REJECT].load(),
            status[s][b][COATING_REJECT].load(), status[s][b][ACCEPT].load(),
            selectionMismatch[s][b].load(), weightMismatch[s][b].load(),
            weightReverseMismatch[s][b].load(), invalidReference[s][b].load(),
            halfMismatch[s][b].load(), coatingMismatch[s][b].load(),
            coatingReverseMismatch[s][b].load(), mixtureMismatch[s][b].load(),
            mixtureReverseMismatch[s][b].load(), replayMismatch[s][b].load(),
            details[s][b].load() < 16ull ? details[s][b].load() : 16ull,
            writeFailures[s][b].load());
}

} }
#endif

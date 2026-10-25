#ifndef SLG_MLHERO_GLOSSYTRANSLUCENT_PDF_AUDIT15LJ_H
#define SLG_MLHERO_GLOSSYTRANSLUCENT_PDF_AUDIT15LJ_H

#include <atomic>
#include <cmath>
#include <cstdio>

#include "slg/engines/mlhero_metal_direction15ky.h"
#include "slg/engines/mlhero_metal_pdf15lb.h"
#include "slg/engines/mlhero_rough_optics15kk.h"

namespace slg { namespace mlhero15lj {
using V = mlhero15kk::V;
constexpr unsigned Budget = 256, FaceCount = 2, BranchCount = 2, StatusCount = 5;
enum Face : unsigned { FRONT = 0, BACK = 1 };
enum Branch : unsigned { BASE = 0, COATING = 1 };
enum Status : unsigned { DRAW_ONLY = 0, BASE_REJECT, COATING_REJECT, GEOMETRY_REJECT, ACCEPT };

struct Record {
    unsigned side = 0, face = 0, branch = 0, material = 0, status = 0;
    unsigned long long detail = 0;
    double rough = 0., anis = 0., u0 = 0., u1 = 0., branchU = 0.;
    double ksFilter = 0., wBase = 0., wCoating = 0., referenceWBase = 0.;
    double wBaseReverse = 0., wCoatingReverse = 0., referenceWBaseReverse = 0.;
    V fixed, sampled, h, orientedReferenceSampled;
    bool orientedReferenceAccepted = false;
    double reportedHalfPdf = 0., referenceHalfPdf = 0.;
    double reportedCoatingPdf = 0., referenceCoatingPdf = 0.;
    double reportedCoatingPdfReverse = 0., referenceCoatingPdfReverse = 0.;
    double reportedMixturePdf = 0., referenceMixturePdf = 0.;
    double reportedMixturePdfReverse = 0., referenceMixturePdfReverse = 0.;
    double reportedMixtureReplay = 0.;
};

using Reporter = void (*)(const Record &);
inline thread_local Reporter reporter = nullptr;
inline thread_local unsigned used[2][FaceCount][BranchCount]{};
inline std::atomic<unsigned long long>
    attempts[2][FaceCount][BranchCount]{}, status[2][FaceCount][BranchCount][StatusCount]{},
    selectionMismatch[2][FaceCount][BranchCount]{}, invalidReference[2][FaceCount][BranchCount]{},
    weightMismatch[2][FaceCount][BranchCount]{}, weightReverseMismatch[2][FaceCount][BranchCount]{},
    halfMismatch[2][FaceCount][BranchCount]{}, coatingMismatch[2][FaceCount][BranchCount]{},
    coatingReverseMismatch[2][FaceCount][BranchCount]{}, mixtureMismatch[2][FaceCount][BranchCount]{},
    mixtureReverseMismatch[2][FaceCount][BranchCount]{}, replayMismatch[2][FaceCount][BranchCount]{},
    rejectedButOrientedReferenceAccepted[2][FaceCount][BranchCount]{},
    details[2][FaceCount][BranchCount]{}, writeFailures[2][FaceCount][BranchCount]{};

inline double Relative(double a, double b) {
    return std::fabs(a - b) / std::fmax(1e-300, std::fmax(std::fabs(a), std::fabs(b)));
}

inline void Reset() {
    for (unsigned s = 0; s < 2; ++s) for (unsigned f = 0; f < FaceCount; ++f)
        for (unsigned b = 0; b < BranchCount; ++b) {
            attempts[s][f][b] = 0; selectionMismatch[s][f][b] = 0;
            invalidReference[s][f][b] = 0; weightMismatch[s][f][b] = 0;
            weightReverseMismatch[s][f][b] = 0; halfMismatch[s][f][b] = 0;
            coatingMismatch[s][f][b] = 0; coatingReverseMismatch[s][f][b] = 0;
            mixtureMismatch[s][f][b] = 0; mixtureReverseMismatch[s][f][b] = 0;
            replayMismatch[s][f][b] = 0;
            rejectedButOrientedReferenceAccepted[s][f][b] = 0;
            details[s][f][b] = 0; writeFailures[s][f][b] = 0;
            for (unsigned k = 0; k < StatusCount; ++k) status[s][f][b][k] = 0;
        }
}

struct WorkerScope {
    WorkerScope(bool enabled, Reporter callback) {
        for (auto &side : used) for (auto &face : side) for (auto &value : face) value = 0;
        reporter = enabled ? callback : nullptr;
    }
    ~WorkerScope() {
        reporter = nullptr;
        for (auto &side : used) for (auto &face : side) for (auto &value : face) value = 0;
    }
    WorkerScope(const WorkerScope &) = delete;
    WorkerScope &operator=(const WorkerScope &) = delete;
};

struct Attempt {
    bool active = false;
    Record r;

    Attempt(bool fromLight, unsigned material, bool sampledBase,
            double rough, double anis, double u0, double u1, double branchU,
            double wBase, double ksFilter, V fixed) {
        const unsigned s = fromLight ? 1u : 0u;
        const unsigned f = fixed.z >= 0. ? FRONT : BACK;
        const unsigned b = sampledBase ? BASE : COATING;
        if (!reporter || used[s][f][b] >= Budget) return;
        ++used[s][f][b]; active = true;
        r.side = s; r.face = f; r.branch = b; r.material = material;
        r.rough = rough; r.anis = anis; r.u0 = u0; r.u1 = u1; r.branchU = branchU;
        r.ksFilter = ksFilter; r.wBase = wBase; r.wCoating = 1. - wBase;
        r.fixed = mlhero15kk::Unit(fixed);
        const double tail = std::pow(1. - std::fabs(r.fixed.z), 5.);
        const double schlick = ksFilter + (1. - ksFilter) * tail;
        r.referenceWBase = 1. - .5 * (1. + schlick);

        V wh = mlhero15ky::SampleHalf(rough, anis, u0, u1);
        if (f == BACK) wh = wh * -1.;
        const double dotFH = mlhero15kk::Dot(r.fixed, wh);
        r.orientedReferenceSampled = mlhero15kk::Unit(wh * (2. * dotFH) - r.fixed);
        r.orientedReferenceAccepted = dotFH > 0. &&
            std::fabs(r.orientedReferenceSampled.z) >= 1e-6 &&
            r.fixed.z * r.orientedReferenceSampled.z > 0.;

        attempts[s][f][b].fetch_add(1ull, std::memory_order_relaxed);
        if (sampledBase != (branchU < wBase))
            selectionMismatch[s][f][b].fetch_add(1ull, std::memory_order_relaxed);
    }

    bool Active() const { return active; }
    void Reject(Status reason) { if (active) r.status = reason; }

    void Accept(V sampled, double reportedMixture, double reportedCoating,
            double wBaseReverse) {
        if (!active) return;
        r.status = ACCEPT;
        r.sampled = mlhero15kk::Unit(sampled);
        r.h = mlhero15kk::Unit(r.fixed + r.sampled);
        r.wBaseReverse = wBaseReverse; r.wCoatingReverse = 1. - wBaseReverse;
        const double reverseTail = std::pow(1. - std::fabs(r.sampled.z), 5.);
        const double reverseSchlick = r.ksFilter + (1. - r.ksFilter) * reverseTail;
        r.referenceWBaseReverse = 1. - .5 * (1. + reverseSchlick);
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
        r.referenceMixturePdf = .5 * (r.wBase * baseDirect +
            r.wCoating * r.referenceCoatingPdf);
        r.referenceMixturePdfReverse = .5 * (r.wBaseReverse * baseReverse +
            r.wCoatingReverse * r.referenceCoatingPdfReverse);
        r.reportedMixtureReplay = .5 * (r.wBase * baseDirect +
            r.wCoating * r.reportedCoatingPdf);
        r.reportedMixturePdfReverse = .5 * (r.wBaseReverse * baseReverse +
            r.wCoatingReverse * r.reportedCoatingPdfReverse);
    }

    ~Attempt() {
        if (!active) return;
        const unsigned s = r.side, f = r.face, b = r.branch;
        status[s][f][b][r.status].fetch_add(1ull, std::memory_order_relaxed);
        if ((r.status == COATING_REJECT) && r.orientedReferenceAccepted)
            rejectedButOrientedReferenceAccepted[s][f][b].fetch_add(1ull, std::memory_order_relaxed);
        if (r.status == ACCEPT) {
            const bool valid = r.referenceHalfPdf > 0. && std::isfinite(r.referenceHalfPdf) &&
                r.referenceCoatingPdf > 0. && std::isfinite(r.referenceCoatingPdf) &&
                r.referenceMixturePdf > 0. && std::isfinite(r.referenceMixturePdf) &&
                r.referenceMixturePdfReverse > 0. && std::isfinite(r.referenceMixturePdfReverse);
            if (!valid) invalidReference[s][f][b].fetch_add(1ull, std::memory_order_relaxed);
            else {
                if (Relative(r.wBase, r.referenceWBase) > 2e-4) weightMismatch[s][f][b].fetch_add(1ull, std::memory_order_relaxed);
                if (Relative(r.wBaseReverse, r.referenceWBaseReverse) > 2e-4) weightReverseMismatch[s][f][b].fetch_add(1ull, std::memory_order_relaxed);
                if (Relative(r.reportedHalfPdf, r.referenceHalfPdf) > 2e-4) halfMismatch[s][f][b].fetch_add(1ull, std::memory_order_relaxed);
                if (Relative(r.reportedCoatingPdf, r.referenceCoatingPdf) > 2e-4) coatingMismatch[s][f][b].fetch_add(1ull, std::memory_order_relaxed);
                if (Relative(r.reportedCoatingPdfReverse, r.referenceCoatingPdfReverse) > 2e-4) coatingReverseMismatch[s][f][b].fetch_add(1ull, std::memory_order_relaxed);
                if (Relative(r.reportedMixturePdf, r.referenceMixturePdf) > 2e-4) mixtureMismatch[s][f][b].fetch_add(1ull, std::memory_order_relaxed);
                if (Relative(r.reportedMixturePdfReverse, r.referenceMixturePdfReverse) > 2e-4) mixtureReverseMismatch[s][f][b].fetch_add(1ull, std::memory_order_relaxed);
                if (Relative(r.reportedMixturePdf, r.reportedMixtureReplay) > 2e-4) replayMismatch[s][f][b].fetch_add(1ull, std::memory_order_relaxed);
            }
        }
        r.detail = details[s][f][b].fetch_add(1ull, std::memory_order_relaxed);
        if (r.detail < 8ull) reporter(r);
    }

    Attempt(const Attempt &) = delete;
    Attempt &operator=(const Attempt &) = delete;
};

inline const char *FaceName(unsigned f) { return f == FRONT ? "FRONT" : "BACK"; }
inline const char *BranchName(unsigned b) { return b == BASE ? "BASE" : "COATING"; }
inline const char *StatusName(unsigned v) {
    const char *names[StatusCount] = {"DRAW_ONLY", "BASE_REJECT", "COATING_REJECT", "GEOMETRY_REJECT", "ACCEPT"};
    return v < StatusCount ? names[v] : "UNKNOWN";
}

inline void WriteRecord(FILE *f, const Record &r, unsigned thread, unsigned sample) {
    fprintf(f, "phase15lj GLOSSYTRANSLUCENT_PDF_SAMPLE side=%s face=%s branch=%s detail=%llu thread=%u sample=%u materialID=%u status=%s roughness=%.17g anisotropy=%.17g u=(%.17g,%.17g,%.17g) ksFilter=%.17g weights=(%.17g,%.17g) referenceWBase=%.17g reverseWeights=(%.17g,%.17g) referenceWBaseReverse=%.17g fixed=(%.17g,%.17g,%.17g) sampled=(%.17g,%.17g,%.17g) h=(%.17g,%.17g,%.17g) orientedReferenceSampled=(%.17g,%.17g,%.17g) orientedReferenceAccepted=%u reportedHalfPdf=%.17g referenceHalfPdf=%.17g halfRatio=%.17g reportedCoatingPdf=%.17g referenceCoatingPdf=%.17g coatingRatio=%.17g reportedMixturePdf=%.17g referenceMixturePdf=%.17g mixtureRatio=%.17g reportedMixturePdfReverse=%.17g referenceMixturePdfReverse=%.17g reverseMixtureRatio=%.17g diagnosticOnly=1 physicsChanged=0\n",
        r.side ? "LIGHT" : "EYE", FaceName(r.face), BranchName(r.branch), r.detail,
        thread, sample, r.material, StatusName(r.status), r.rough, r.anis,
        r.u0, r.u1, r.branchU, r.ksFilter, r.wBase, r.wCoating, r.referenceWBase,
        r.wBaseReverse, r.wCoatingReverse, r.referenceWBaseReverse,
        r.fixed.x, r.fixed.y, r.fixed.z, r.sampled.x, r.sampled.y, r.sampled.z,
        r.h.x, r.h.y, r.h.z, r.orientedReferenceSampled.x,
        r.orientedReferenceSampled.y, r.orientedReferenceSampled.z,
        r.orientedReferenceAccepted ? 1u : 0u, r.reportedHalfPdf, r.referenceHalfPdf,
        r.reportedHalfPdf > 0. ? r.referenceHalfPdf / r.reportedHalfPdf : 0.,
        r.reportedCoatingPdf, r.referenceCoatingPdf,
        r.reportedCoatingPdf > 0. ? r.referenceCoatingPdf / r.reportedCoatingPdf : 0.,
        r.reportedMixturePdf, r.referenceMixturePdf,
        r.reportedMixturePdf > 0. ? r.referenceMixturePdf / r.reportedMixturePdf : 0.,
        r.reportedMixturePdfReverse, r.referenceMixturePdfReverse,
        r.reportedMixturePdfReverse > 0. ? r.referenceMixturePdfReverse / r.reportedMixturePdfReverse : 0.);
}

inline void WriteSummary(FILE *f) {
    for (unsigned s = 0; s < 2; ++s) for (unsigned face = 0; face < FaceCount; ++face)
        for (unsigned b = 0; b < BranchCount; ++b)
            fprintf(f, "phase15lj GLOSSYTRANSLUCENT_PDF_SUMMARY side=%s face=%s branch=%s selectedAttempts=%llu drawOnly=%llu baseRejected=%llu coatingRejected=%llu geometryRejected=%llu accepted=%llu selectionMismatches=%llu directWeightMismatches=%llu reverseWeightMismatches=%llu invalidReference=%llu halfPdfMismatches=%llu coatingPdfMismatches=%llu coatingReversePdfMismatches=%llu mixturePdfMismatches=%llu mixtureReversePdfMismatches=%llu reportedMixtureReplayMismatches=%llu rejectedButOrientedReferenceAccepted=%llu detailsSelected=%llu writeFailures=%llu toleranceRelative=0.0002 reference=INDEPENDENT_SCHLICK_INVERSE_CDF_PLUS_HALF_REFLECTION_MIXTURE backfaceOrientationReference=FLIPPED_GENERATED_HALF rejectedSamplesIncluded=1 conditionalRenormalization=NONE diagnosticOnly=1 physicsChanged=0\n",
                s ? "LIGHT" : "EYE", FaceName(face), BranchName(b), attempts[s][face][b].load(),
                status[s][face][b][DRAW_ONLY].load(), status[s][face][b][BASE_REJECT].load(),
                status[s][face][b][COATING_REJECT].load(), status[s][face][b][GEOMETRY_REJECT].load(),
                status[s][face][b][ACCEPT].load(), selectionMismatch[s][face][b].load(),
                weightMismatch[s][face][b].load(), weightReverseMismatch[s][face][b].load(),
                invalidReference[s][face][b].load(), halfMismatch[s][face][b].load(),
                coatingMismatch[s][face][b].load(), coatingReverseMismatch[s][face][b].load(),
                mixtureMismatch[s][face][b].load(), mixtureReverseMismatch[s][face][b].load(),
                replayMismatch[s][face][b].load(),
                rejectedButOrientedReferenceAccepted[s][face][b].load(),
                details[s][face][b].load() < 8ull ? details[s][face][b].load() : 8ull,
                writeFailures[s][face][b].load());
}

} }
#endif

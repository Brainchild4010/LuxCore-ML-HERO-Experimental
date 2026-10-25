#ifndef SLG_MLHERO_ROUGHGLASS_PDF_AUDIT15LC_H
#define SLG_MLHERO_ROUGHGLASS_PDF_AUDIT15LC_H

#include <atomic>
#include <cmath>
#include <cstdio>

#include "slg/engines/mlhero_metal_pdf15lb.h"
#include "slg/engines/mlhero_rough_optics15kk.h"

namespace slg { namespace mlhero15lc {
using V = mlhero15kk::V;
constexpr unsigned Budget = 256, StatusCount = 8;
enum Status : unsigned { DRAW_ONLY = 0, INVALID_INTERFACE, TIR_TRANSMISSION,
    HEMISPHERE_REJECT, ZERO_DENSITY, PDF_REJECT, ACCEPT_REFLECT, ACCEPT_TRANSMIT };

struct Record {
    unsigned side = 0, material = 0, status = 0;
    unsigned long long detail = 0;
    bool vndf = false, transmit = false;
    double nu = 0., nv = 0., rough = 0., anis = 0.;
    double u0 = 0., u1 = 0., branchU = 0., branchProbability = 0.;
    V fixed, h, generated;
    double brdfD = 0., reportedHalfPdf = 0., referenceHalfPdf = 0.;
    double reportedDirectionPdf = 0., referenceJacobian = 0., referenceDirectionPdf = 0.;
};
using Reporter = void (*)(const Record &);
inline thread_local Reporter reporter = nullptr;
inline thread_local unsigned used[2][2]{};
inline std::atomic<unsigned long long> attempts[2][2]{}, status[2][2][StatusCount]{},
    halfMismatch[2][2]{}, directionMismatch[2][2]{}, invalidReference[2][2]{},
    details[2][2]{}, writeFailures[2][2]{};

inline double Relative(double a, double b) {
    return std::fabs(a - b) / std::fmax(1e-300, std::fmax(std::fabs(a), std::fabs(b)));
}
inline double ReferenceHalfPdf(bool vndf, double rough, double anis, V fixed, V h) {
    fixed = mlhero15kk::Unit(fixed); h = mlhero15kk::Unit(h);
    if (!vndf) return mlhero15lb::HalfPdf(rough, anis, h.x, h.y, h.z);
    if (!(rough > 0.) || !(std::fabs(fixed.z) > 0.) || !(h.z > 0.)) return 0.;
    const double den = h.x * h.x + h.y * h.y + rough * h.z * h.z;
    if (!(den > 0.)) return 0.;
    const double d = rough / (3.14159265358979323846264338327950288 * den * den);
    return d * mlhero15kk::G1(rough, fixed) *
        std::fabs(mlhero15kk::Dot(fixed, h)) / std::fabs(fixed.z);
}
inline void Reset() {
    for (unsigned s = 0; s < 2; ++s) for (unsigned m = 0; m < 2; ++m) {
        attempts[s][m] = 0; halfMismatch[s][m] = 0; directionMismatch[s][m] = 0;
        invalidReference[s][m] = 0; details[s][m] = 0; writeFailures[s][m] = 0;
        for (unsigned k = 0; k < StatusCount; ++k) status[s][m][k] = 0;
    }
}
struct WorkerScope {
    WorkerScope(bool on, Reporter callback) {
        for (unsigned s=0;s<2;++s) for(unsigned m=0;m<2;++m) used[s][m]=0;
        reporter = on ? callback : nullptr;
    }
    ~WorkerScope() {
        reporter = nullptr;
        for (unsigned s=0;s<2;++s) for(unsigned m=0;m<2;++m) used[s][m]=0;
    }
    WorkerScope(const WorkerScope &) = delete;
    WorkerScope &operator=(const WorkerScope &) = delete;
};

class Attempt {
public:
    Attempt(bool fromLight, unsigned material, bool vndf, double nu, double nv,
            double rough, double anis, double u0, double u1, double branchU,
            V fixed, V h, double brdfD, double halfPdf) {
        r.side = fromLight ? 1u : 0u; r.material = material; r.vndf = vndf;
        const unsigned mode = vndf ? 1u : 0u;
        if (!reporter || used[r.side][mode] >= Budget) return;
        ++used[r.side][mode]; active = true; r.nu = nu; r.nv = nv; r.rough = rough; r.anis = anis;
        r.u0 = u0; r.u1 = u1; r.branchU = branchU;
        r.fixed = mlhero15kk::Unit(fixed); r.h = mlhero15kk::Unit(h);
        r.brdfD = brdfD; r.reportedHalfPdf = halfPdf;
        r.referenceHalfPdf = ReferenceHalfPdf(vndf, rough, anis, r.fixed, r.h);
    }
    void SetStatus(Status s) { if (active) r.status = s; }
    void Branch(bool transmit, double probability) {
        if (active) { r.transmit = transmit; r.branchProbability = probability; }
    }
    void Accept(V generated, double reportedPdf, double cosine,
            double cosTransmitted, double eta) {
        if (!active) return;
        r.generated = mlhero15kk::Unit(generated); r.reportedDirectionPdf = reportedPdf;
        if (r.transmit) {
            const double signedT = r.fixed.z > 0. ? -cosTransmitted : cosTransmitted;
            const double length = eta * cosine + signedT;
            r.referenceJacobian = length != 0. ? cosTransmitted / (length * length) : 0.;
            r.status = ACCEPT_TRANSMIT;
        } else {
            const double dot = std::fabs(mlhero15kk::Dot(r.fixed, r.h));
            r.referenceJacobian = dot > 0. ? 1. / (4. * dot) : 0.;
            r.status = ACCEPT_REFLECT;
        }
        r.referenceDirectionPdf = r.branchProbability * r.referenceHalfPdf * r.referenceJacobian;
    }
    ~Attempt() {
        if (!active) return;
        const unsigned m = r.vndf ? 1u : 0u;
        attempts[r.side][m].fetch_add(1, std::memory_order_relaxed);
        status[r.side][m][r.status].fetch_add(1, std::memory_order_relaxed);
        const bool valid = r.referenceHalfPdf > 0. && std::isfinite(r.referenceHalfPdf);
        if (!valid) invalidReference[r.side][m].fetch_add(1, std::memory_order_relaxed);
        else if (Relative(r.referenceHalfPdf, r.reportedHalfPdf) > 2e-4)
            halfMismatch[r.side][m].fetch_add(1, std::memory_order_relaxed);
        if ((r.status == ACCEPT_REFLECT || r.status == ACCEPT_TRANSMIT) &&
                (!(r.referenceDirectionPdf > 0.) || !std::isfinite(r.referenceDirectionPdf) ||
                 Relative(r.referenceDirectionPdf, r.reportedDirectionPdf) > 2e-4))
            directionMismatch[r.side][m].fetch_add(1, std::memory_order_relaxed);
        r.detail = details[r.side][m].fetch_add(1, std::memory_order_relaxed);
        if (r.detail < 16) reporter(r);
    }
private:
    bool active = false;
    Record r;
};

inline const char *StatusName(unsigned s) {
    static const char *names[StatusCount] = {"DRAW_ONLY", "INVALID_INTERFACE",
        "TIR_TRANSMISSION", "HEMISPHERE_REJECT", "ZERO_DENSITY", "PDF_REJECT",
        "ACCEPT_REFLECT", "ACCEPT_TRANSMIT"};
    return s < StatusCount ? names[s] : "UNKNOWN";
}
inline void WriteRecord(FILE *f, const Record &r, unsigned thread, unsigned sample) {
    fprintf(f, "phase15lc ROUGH_PDF_SAMPLE side=%s sampler=%s detail=%llu thread=%u sample=%u materialID=%u status=%s event=%s nu=%.17g nv=%.17g roughness=%.17g anisotropy=%.17g u=(%.17g,%.17g,%.17g) fixed=(%.17g,%.17g,%.17g) h=(%.17g,%.17g,%.17g) generated=(%.17g,%.17g,%.17g) brdfD=%.17g reportedHalfPdf=%.17g referenceHalfPdf=%.17g halfRatio=%.17g branchProbability=%.17g reportedDirectionPdf=%.17g referenceJacobian=%.17g referenceDirectionPdf=%.17g directionRatio=%.17g diagnosticOnly=1 physicsChanged=0\n",
        r.side ? "LIGHT" : "EYE", r.vndf ? "VNDF" : "SCHLICK_NDF", r.detail,
        thread, sample, r.material, StatusName(r.status), r.transmit ? "TRANSMIT" : "REFLECT",
        r.nu, r.nv, r.rough, r.anis, r.u0, r.u1, r.branchU,
        r.fixed.x, r.fixed.y, r.fixed.z, r.h.x, r.h.y, r.h.z,
        r.generated.x, r.generated.y, r.generated.z, r.brdfD, r.reportedHalfPdf,
        r.referenceHalfPdf, r.reportedHalfPdf > 0. ? r.referenceHalfPdf / r.reportedHalfPdf : 0.,
        r.branchProbability, r.reportedDirectionPdf, r.referenceJacobian,
        r.referenceDirectionPdf, r.reportedDirectionPdf > 0. ? r.referenceDirectionPdf / r.reportedDirectionPdf : 0.);
}
inline void WriteSummary(FILE *f) {
    for (unsigned s = 0; s < 2; ++s) for (unsigned m = 0; m < 2; ++m)
        fprintf(f, "phase15lc ROUGH_PDF_SUMMARY side=%s sampler=%s selectedDraws=%llu invalidInterface=%llu tirTransmission=%llu hemisphereRejected=%llu zeroDensity=%llu pdfRejected=%llu acceptedReflect=%llu acceptedTransmit=%llu invalidReference=%llu halfPdfMismatches=%llu directionPdfMismatches=%llu detailsSelected=%llu writeFailures=%llu toleranceRelative=0.0002 rejectedSamplesIncluded=1 conditionalRenormalization=NONE diagnosticOnly=1 physicsChanged=0\n",
            s ? "LIGHT" : "EYE", m ? "VNDF" : "SCHLICK_NDF", attempts[s][m].load(),
            status[s][m][INVALID_INTERFACE].load(), status[s][m][TIR_TRANSMISSION].load(),
            status[s][m][HEMISPHERE_REJECT].load(), status[s][m][ZERO_DENSITY].load(),
            status[s][m][PDF_REJECT].load(), status[s][m][ACCEPT_REFLECT].load(),
            status[s][m][ACCEPT_TRANSMIT].load(), invalidReference[s][m].load(),
            halfMismatch[s][m].load(), directionMismatch[s][m].load(),
            details[s][m].load() < 16 ? details[s][m].load() : 16, writeFailures[s][m].load());
}
} }
#endif

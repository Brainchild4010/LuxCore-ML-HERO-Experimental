#ifndef MLHERO_EMISSION_SOURCE15KG_H
#define MLHERO_EMISSION_SOURCE15KG_H
#include <cmath>

// Diagnostic-only TLS shared by TriangleLight, its spherical PDF and BIDIR.
namespace slg { namespace mlhero15kg {
enum class Branch { Unavailable, RejectedArea, RejectedHemisphere, Profile,
    Delta, Cone, Lambert };
inline const char *BranchName(Branch b) {
    switch(b) {
        case Branch::RejectedArea: return "REJECTED_AREA";
        case Branch::RejectedHemisphere: return "REJECTED_HEMISPHERE";
        case Branch::Profile: return "EMISSION_PROFILE";
        case Branch::Delta: return "DELTA_DIRECTION";
        case Branch::Cone: return "UNIFORM_CONE";
        case Branch::Lambert: return "COSINE_HEMISPHERE";
        default: return "UNAVAILABLE";
    }
}
struct Snapshot {
    const void *light = nullptr, *emissionFunction = nullptr;
    bool valid = false, active = false, directWritten = false, emissionWritten = false;
    bool functionPdfObserved = false;
    unsigned meshIndex = 0u, triangleIndex = 0u;
    float triangleArea = 0.f, inverseTriangleArea = 0.f, meshArea = 0.f, inverseMeshArea = 0.f;
    float emittedTheta = 0.f, cosThetaMax = 0.f, cosOutLight = 0.f;
    float directPdfA = 0.f, emissionPdfW = 0.f, functionPdf = 0.f;
    float localX = 0.f, localY = 0.f, localZ = 0.f;
    float theta = 0.f, phi = 0.f, sinTheta = 0.f, u = 0.f, v = 0.f;
    float distributionPdf = 0.f;
    double denominator = 0.;
    Branch branch = Branch::Unavailable;
};
inline thread_local bool enabled = false;
inline thread_local Snapshot current;
inline void ResetThread(bool on) { current = Snapshot(); enabled = on; }
class Capture {
    bool on;
public:
    explicit Capture(const void *source) : on(enabled) {
        if (on) {
            current = Snapshot(); current.valid = true;
            current.active = true; current.light = source;
        }
    }
    bool Enabled() const { return on; }
    ~Capture() { if (on) current.active = false; }
};
inline bool Same(float a, float b) { return (std::isnan(a) && std::isnan(b)) || a == b; }
inline bool Matches(const Snapshot &s, const void *light, float direct, float emission) {
    return s.valid && light && s.light == light && s.directWritten && s.emissionWritten &&
        Same(s.directPdfA,direct) && Same(s.emissionPdfW,emission);
}
inline const char *Cause(const Snapshot &s, bool matched) {
    if (!matched) return "SOURCE_NOT_MATCHED";
    if (!std::isfinite(s.inverseTriangleArea)) return "NONFINITE_INVERSE_AREA";
    if (s.branch == Branch::Profile && s.functionPdfObserved) {
        if (s.denominator == 0.) return s.distributionPdf == 0.f ?
            "PROFILE_ZERO_OVER_ZERO" : "PROFILE_ZERO_SINE_DENOMINATOR";
        if (!std::isfinite(s.distributionPdf)) return "PROFILE_DISTRIBUTION_NONFINITE";
        if (!std::isfinite(s.functionPdf)) return "PROFILE_PDF_NONFINITE";
    }
    if (s.branch == Branch::Cone && s.cosThetaMax == 1.f) return "CONE_ZERO_SOLID_ANGLE";
    if (!std::isfinite(s.emissionPdfW)) return "NONFINITE_PDF_PRODUCT_OR_OTHER";
    return "FINITE_SOURCE_PDF";
}
} }
#endif

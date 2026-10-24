#ifndef MLHERO_DIRECTHIT_AUDIT15KF_H
#define MLHERO_DIRECTHIT_AUDIT15KF_H
#include <cmath>

namespace slg { namespace mlhero15kf {
constexpr unsigned DetailBudgetPerStage = 16u;
struct Terms {
    float directProduct, emissionProduct, directSquare, emissionSquare;
    float directTerm, emissionTerm, reconstructedWeight;
    float zeroHistoryCandidateWeight, zeroHistoryCandidateMIS;
    double directReference, emissionReference, doubleWeight, doubleMIS;
    bool inputsFinite, directInfTimesZero, emissionInfTimesZero;
    bool reconstructedMatchesActual, candidateRecoversNonfiniteMIS;
};
inline bool SameValue(float a, float b) {
    return (std::isnan(a) && std::isnan(b)) || a == b;
}
inline Terms Analyze(float directPdfA, float emissionPdfW, float lightPickPdf,
        float dVCM, float dVC, float actualWeight, float actualMIS) {
    Terms t;
    t.inputsFinite = std::isfinite(directPdfA) && std::isfinite(emissionPdfW) &&
        std::isfinite(lightPickPdf) && std::isfinite(dVCM) && std::isfinite(dVC);
    // Reconstruct separately for diagnostics. The production terms stay untouched.
    t.directProduct = directPdfA * lightPickPdf;
    t.emissionProduct = emissionPdfW * lightPickPdf;
    t.directSquare = t.directProduct * t.directProduct;
    t.emissionSquare = t.emissionProduct * t.emissionProduct;
    t.directTerm = t.directSquare * dVCM;
    t.emissionTerm = t.emissionSquare * dVC;
    t.reconstructedWeight = t.directTerm + t.emissionTerm;
    t.directInfTimesZero = std::isinf(t.directSquare) && dVCM == 0.f;
    t.emissionInfTimesZero = std::isinf(t.emissionSquare) && dVC == 0.f;
    t.zeroHistoryCandidateWeight = (dVCM == 0.f ? 0.f : t.directTerm) +
        (dVC == 0.f ? 0.f : t.emissionTerm);
    t.zeroHistoryCandidateMIS = 1.f / (t.zeroHistoryCandidateWeight + 1.f);
    const double d = double(directPdfA) * double(lightPickPdf);
    const double e = double(emissionPdfW) * double(lightPickPdf);
    t.directReference = d * d * double(dVCM);
    t.emissionReference = e * e * double(dVC);
    t.doubleWeight = t.directReference + t.emissionReference;
    t.doubleMIS = 1. / (t.doubleWeight + 1.);
    t.reconstructedMatchesActual = SameValue(t.reconstructedWeight, actualWeight);
    t.candidateRecoversNonfiniteMIS = !std::isfinite(actualMIS) &&
        std::isfinite(t.zeroHistoryCandidateMIS);
    return t;
}
inline const char *Reason(const Terms &t, float actualWeight, float actualMIS,
        bool contributionFinite) {
    if (!t.inputsFinite) return "NONFINITE_MIS_INPUT";
    if (t.directInfTimesZero && t.emissionInfTimesZero) return "BOTH_INF_TIMES_ZERO";
    if (t.directInfTimesZero) return "DIRECT_INF_TIMES_ZERO";
    if (t.emissionInfTimesZero) return "EMISSION_INF_TIMES_ZERO";
    if (std::isnan(actualWeight) || !std::isfinite(actualMIS)) return "OTHER_INVALID_MIS";
    if (!contributionFinite) return "NONFINITE_CONTRIBUTION";
    return "INFINITE_COMPETITOR";
}
} }
#endif

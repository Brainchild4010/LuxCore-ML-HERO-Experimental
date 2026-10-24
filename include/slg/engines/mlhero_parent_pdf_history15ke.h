#ifndef MLHERO_PARENT_PDF_HISTORY15KE_H
#define MLHERO_PARENT_PDF_HISTORY15KE_H

#include <cmath>

// Diagnostic-only native parent bounce observations. No path density product.
namespace slg { namespace mlhero15ke {
constexpr unsigned PathBudget = 64u;
constexpr unsigned StepBudget = 32u;
enum class Measure { Unavailable, SolidAngle, DiscreteEvent };
inline const char *MeasureName(Measure m) {
    return m == Measure::SolidAngle ? "SOLID_ANGLE" :
        (m == Measure::DiscreteEvent ? "DISCRETE_EVENT" : "UNAVAILABLE");
}
struct Record {
    unsigned depth = 0u, event = 0u, material = 0u;
    double samplePdf = 0., queriedReversePdf = 0., nativeReverseScalar = 0.;
    double cosine = 0., rrProbability = 1.;
    Measure sampleMeasure = Measure::Unavailable, reverseMeasure = Measure::Unavailable;
    bool sampled = false, delta = false, reverseObserved = false;
    bool rrApplied = false, rrKilled = false, accepted = false;
    const char *outcome = "EARLY_RETURN";
    void Sample(unsigned ev, bool isDelta, double pdf, double cos) {
        sampled = true; event = ev; delta = isDelta; samplePdf = pdf; cosine = cos;
        if (std::isfinite(pdf) && pdf > 0.)
            sampleMeasure = delta ? Measure::DiscreteEvent : Measure::SolidAngle;
    }
    void Reverse(double pdf) {
        reverseObserved = true; nativeReverseScalar = pdf;
        // Production copies the forward scalar for delta recurrence. This is
        // not an independently evaluated reverse discrete event probability.
        if (!delta) {
            queriedReversePdf = pdf;
            if (std::isfinite(pdf) && pdf >= 0.) reverseMeasure = Measure::SolidAngle;
        }
    }
    void Roulette(double q, bool killed) {
        rrApplied = true; rrProbability = q; rrKilled = killed;
        if (killed) outcome = "RR_KILLED";
    }
    void Accept() { accepted = true; outcome = "CONTINUE"; }
    bool TypedChecksPass() const {
        if (!sampled) return sampleMeasure == Measure::Unavailable &&
            reverseMeasure == Measure::Unavailable && !accepted;
        const bool sampleValid = std::isfinite(samplePdf) && samplePdf > 0.;
        const Measure expected = sampleValid ?
            (delta ? Measure::DiscreteEvent : Measure::SolidAngle) : Measure::Unavailable;
        const bool reverseValid = reverseObserved && !delta &&
            std::isfinite(queriedReversePdf) && queriedReversePdf >= 0.;
        return sampleMeasure == expected &&
            reverseMeasure == (reverseValid ? Measure::SolidAngle : Measure::Unavailable) &&
            (!accepted || (sampleValid && reverseObserved &&
                (delta || reverseValid) && std::isfinite(rrProbability) &&
                rrProbability > 0. && rrProbability <= 1. && !rrKilled));
    }
};
struct State {
    unsigned long long ticket = 0ull;
    unsigned records = 0u, acceptedBounces = 0u, lastDepth = 0u, lastMaterial = 0u;
    bool initialized = false, selected = false, contiguous = true;
    bool mixedMaterials = false, truncated = false, writeFailed = false;
    bool Advance(const Record &r) {
        const bool linked = records ? r.depth == lastDepth + 1u : r.depth == 1u;
        contiguous = contiguous && linked;
        if (records && lastMaterial != r.material) mixedMaterials = true;
        lastDepth = r.depth; lastMaterial = r.material;
        ++records;
        if (r.accepted) ++acceptedBounces;
        return linked;
    }
};
} }
#endif

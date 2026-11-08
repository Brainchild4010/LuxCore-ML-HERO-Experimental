/***************************************************************************
 * ML HERO phase 15nr: pure direction-proof classification helpers.       *
 ***************************************************************************/

#ifndef _SLG_MLHERO_MIX_DIRECTION_ANALYSIS15NR_H
#define _SLG_MLHERO_MIX_DIRECTION_ANALYSIS15NR_H

#include <algorithm>
#include <cmath>

namespace slg { namespace mlhero15nr {

enum Classification : unsigned int {
	CHILD_SAMPLE_EVALUATE = 0u,
	FRAME_ROUNDTRIP = 1u,
	BOTH = 2u,
	UNRESOLVED = 3u
};

inline Classification Classify(const bool captureMatched,
		const bool originalEvaluateFinite, const bool sampleVsOriginalPass,
		const bool originalVsRoundTripPass) {
	if (!captureMatched || !originalEvaluateFinite)
		return UNRESOLVED;
	const bool sampleMismatch = !sampleVsOriginalPass;
	const bool roundTripMismatch = !originalVsRoundTripPass;
	if (sampleMismatch)
		return roundTripMismatch ? BOTH : CHILD_SAMPLE_EVALUATE;
	return roundTripMismatch ? FRAME_ROUNDTRIP : UNRESOLVED;
}

inline const char *ClassificationName(const unsigned int classification) {
	static const char *names[4] = {
		"CHILD_SAMPLE_EVALUATE", "FRAME_ROUNDTRIP", "BOTH", "UNRESOLVED"
	};
	return classification < 4u ? names[classification] : "UNKNOWN";
}

inline double MaxAbsDifference3(const double a[3], const double b[3]) {
	return std::max(std::fabs(a[0] - b[0]),
		std::max(std::fabs(a[1] - b[1]), std::fabs(a[2] - b[2])));
}

inline unsigned int DirectionErrorBucket(const double error) {
	if (!std::isfinite(error)) return 5u;
	if (error <= 1e-8) return 0u;
	if (error <= 1e-7) return 1u;
	if (error <= 1e-6) return 2u;
	if (error <= 1e-5) return 3u;
	return 4u;
}

} }

#endif // _SLG_MLHERO_MIX_DIRECTION_ANALYSIS15NR_H

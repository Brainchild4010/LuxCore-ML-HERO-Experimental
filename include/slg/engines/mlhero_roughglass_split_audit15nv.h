/***************************************************************************
 * ML HERO phase 15nv: RoughGlass split-readiness classification.          *
 *                                                                         *
 * Pure diagnostic helpers. No sampler, transport, PDF or MIS state is     *
 * mutated here.                                                           *
 ***************************************************************************/

#ifndef SLG_MLHERO_ROUGHGLASS_SPLIT_AUDIT15NV_H
#define SLG_MLHERO_ROUGHGLASS_SPLIT_AUDIT15NV_H

#include <cmath>

namespace slg { namespace mlhero15nv {

enum LaneClass : unsigned {
	HELPER_REJECTED = 0u,
	NATURAL_TRANSMISSION = 1u,
	NATURAL_TIR_REQUIRES_REFLECTION = 2u,
	INVALID_MEASURE = 3u
};

inline LaneClass ClassifyLane(const bool helperOK, const bool naturalTir,
		const double heroEta, const double laneEta,
		const double heroMicrofacetPdf, const double laneSharedMicrofacetPdf,
		const double naturalJacobian, const double sharedJacobian,
		const double naturalDirPdf, const double sharedDirPdf,
		const double naturalDirDelta, const double sharedWhDelta) {
	if (!helperOK)
		return HELPER_REJECTED;
	const bool commonFinite = std::isfinite(heroEta) && heroEta > 0. &&
		std::isfinite(laneEta) && laneEta > 0. &&
		std::isfinite(heroMicrofacetPdf) && heroMicrofacetPdf >= 0. &&
		std::isfinite(laneSharedMicrofacetPdf) && laneSharedMicrofacetPdf >= 0. &&
		std::isfinite(sharedJacobian) && sharedJacobian >= 0. &&
		std::isfinite(sharedDirPdf) && sharedDirPdf >= 0. &&
		std::isfinite(naturalDirDelta) && naturalDirDelta >= 0. &&
		std::isfinite(sharedWhDelta) && sharedWhDelta >= 0.;
	if (!commonFinite)
		return INVALID_MEASURE;
	if (naturalTir)
		return NATURAL_TIR_REQUIRES_REFLECTION;
	if (!std::isfinite(naturalJacobian) || naturalJacobian < 0. ||
		!std::isfinite(naturalDirPdf) || !(naturalDirPdf > 0.))
		return INVALID_MEASURE;
	return NATURAL_TRANSMISSION;
}

inline const char *LaneClassName(const LaneClass c) {
	switch (c) {
		case HELPER_REJECTED: return "HELPER_REJECTED";
		case NATURAL_TRANSMISSION: return "NATURAL_TRANSMISSION";
		case NATURAL_TIR_REQUIRES_REFLECTION:
			return "NATURAL_TIR_REQUIRES_REFLECTION";
		default: return "INVALID_MEASURE";
	}
}

} }

#endif

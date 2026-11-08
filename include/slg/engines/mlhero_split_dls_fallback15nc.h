/***************************************************************************
 * ML HERO phase 15nc: split-child DLS fallback classification helpers.    *
 ***************************************************************************/

#ifndef _SLG_MLHERO_SPLIT_DLS_FALLBACK15NC_H
#define _SLG_MLHERO_SPLIT_DLS_FALLBACK15NC_H

namespace slg {
namespace mlhero15nc {

enum Reason : unsigned {
	CONTRACT = 1u << 0,
	LANE_PLAN = 1u << 1,
	EYE_THROUGHPUT = 1u << 2,
	CONNECTION_THROUGHPUT = 1u << 3,
	ENDPOINT = 1u << 4,
	ROUGH_ENDPOINT = 1u << 5,
	TRANSLUCENT_ENDPOINT = 1u << 6,
	BSDF_VALUE = 1u << 7
};

inline unsigned BuildReasonMask(const bool splitContract,
		const bool lanePlanValid, const bool eyeThroughputScalar,
		const bool connectionThroughputScalar, const bool endpointUsable,
		const bool roughEndpointOK, const bool translucentEndpointOK,
		const bool bsdfValueScalar) {
	unsigned mask = 0u;
	if (!splitContract)
		mask |= CONTRACT;
	if (!lanePlanValid)
		mask |= LANE_PLAN;
	if (!eyeThroughputScalar)
		mask |= EYE_THROUGHPUT;
	if (!connectionThroughputScalar)
		mask |= CONNECTION_THROUGHPUT;
	if (!endpointUsable)
		mask |= ENDPOINT;
	if (!roughEndpointOK)
		mask |= ROUGH_ENDPOINT;
	if (!translucentEndpointOK)
		mask |= TRANSLUCENT_ENDPOINT;
	if (!bsdfValueScalar)
		mask |= BSDF_VALUE;
	return mask;
}

inline bool DiagnosticPass(const unsigned long long fallbacks,
		const unsigned long long capturedFallbacks,
		const unsigned long long unclassifiedFallbacks) {
	return capturedFallbacks == fallbacks && unclassifiedFallbacks == 0ull;
}

} // namespace mlhero15nc
} // namespace slg

#endif // _SLG_MLHERO_SPLIT_DLS_FALLBACK15NC_H

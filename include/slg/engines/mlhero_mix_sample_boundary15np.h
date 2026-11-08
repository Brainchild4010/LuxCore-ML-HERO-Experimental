/***************************************************************************
 * ML HERO phase 15np: boundary policy for Mix wavelength reconstruction.
 ***************************************************************************/

#ifndef _SLG_MLHERO_MIX_SAMPLE_BOUNDARY15NP_H
#define _SLG_MLHERO_MIX_SAMPLE_BOUNDARY15NP_H

namespace slg { namespace mlhero15np {

struct BoundaryPolicy {
	bool applyGeometryCosGate;
	bool applySideEventGate;
};

// BSDF::Evaluate() owns geometry/interpolated-normal boundary gates.
// BSDF::Sample() does not repeat them after the material selected a direction.
// Wavelength reconstruction must follow the native operation being rebuilt.
inline BoundaryPolicy SelectBoundaryPolicy(const bool sampled) {
	const BoundaryPolicy policy = {!sampled, !sampled};
	return policy;
}

struct BypassAccounting {
	bool sampledAttemptClosure;
	bool setUnionClosure;
	bool outcomeClosure;
	bool pass;
};

inline BypassAccounting AnalyzeBypassAccounting(
		const unsigned long long sampledAttempts,
		const unsigned long long bounceAttempts,
		const unsigned long long geometryWouldReject,
		const unsigned long long sideWouldReject,
		const unsigned long long bothWouldReject,
		const unsigned long long anyWouldReject,
		const unsigned long long acceptedAfterBypass,
		const unsigned long long fallbackAfterBypass) {
	BypassAccounting result;
	result.sampledAttemptClosure = sampledAttempts == bounceAttempts;
	result.setUnionClosure = geometryWouldReject + sideWouldReject ==
		anyWouldReject + bothWouldReject;
	result.outcomeClosure = anyWouldReject == acceptedAfterBypass +
		fallbackAfterBypass;
	result.pass = result.sampledAttemptClosure && result.setUnionClosure &&
		result.outcomeClosure;
	return result;
}

} }

#endif

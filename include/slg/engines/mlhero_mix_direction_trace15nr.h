/***************************************************************************
 * ML HERO phase 15nr: diagnostic capture of the exact selected Mix child
 * direction before it is transformed back through the Mix and BSDF frames.
 ***************************************************************************/

#ifndef _SLG_MLHERO_MIX_DIRECTION_TRACE15NR_H
#define _SLG_MLHERO_MIX_DIRECTION_TRACE15NR_H

#include <algorithm>
#include <cmath>

#include "luxrays/core/color/color.h"
#include "luxrays/core/geometry/vector.h"
#include "slg/engines/mlhero_mix_direction_analysis15nr.h"

namespace slg { namespace mlhero15nr {

struct Capture {
	const void *mixMaterial;
	const void *selectedMaterial;
	unsigned int selectedChild;
	unsigned int event;
	bool fromLight;
	bool valid;
	float outerPassThroughEvent;
	float childPassThroughEvent;
	float childWeight;
	float childPdfW;
	luxrays::Vector originalFixedDir;
	luxrays::Vector originalSampledDir;
	// Phase 15ns: exact child-local directions used by the native Mix sample.
	// The selected child owns the sampled direction; the other child receives
	// the direction produced by the one native child-frame conversion.
	luxrays::Vector childFixedDir[2];
	luxrays::Vector childSampledDir[2];
	luxrays::Spectrum childSampleWeight;
	luxrays::Spectrum childSampleNumerator;

	Capture() : mixMaterial(nullptr), selectedMaterial(nullptr),
		selectedChild(~0u), event(0u), fromLight(false), valid(false),
		outerPassThroughEvent(0.f), childPassThroughEvent(0.f),
		childWeight(0.f), childPdfW(0.f), originalFixedDir(),
		originalSampledDir(), childSampleWeight(0.f),
		childSampleNumerator(0.f) { }
};

extern thread_local Capture lastCapture;

inline void Reset(const void *mixMaterial, const float outerPassThroughEvent) {
	lastCapture = Capture();
	lastCapture.mixMaterial = mixMaterial;
	lastCapture.outerPassThroughEvent = outerPassThroughEvent;
}

inline void Record(const void *mixMaterial, const void *selectedMaterial,
		const unsigned int selectedChild, const unsigned int event,
		const bool fromLight, const float outerPassThroughEvent,
		const float childPassThroughEvent, const float childWeight,
		const float childPdfW, const luxrays::Vector &originalFixedDir,
		const luxrays::Vector &originalSampledDir,
		const luxrays::Vector &fixedDirA,
		const luxrays::Vector &sampledDirA,
		const luxrays::Vector &fixedDirB,
		const luxrays::Vector &sampledDirB,
		const luxrays::Spectrum &childSampleWeight) {
	lastCapture.mixMaterial = mixMaterial;
	lastCapture.selectedMaterial = selectedMaterial;
	lastCapture.selectedChild = selectedChild;
	lastCapture.event = event;
	lastCapture.fromLight = fromLight;
	lastCapture.outerPassThroughEvent = outerPassThroughEvent;
	lastCapture.childPassThroughEvent = childPassThroughEvent;
	lastCapture.childWeight = childWeight;
	lastCapture.childPdfW = childPdfW;
	lastCapture.originalFixedDir = originalFixedDir;
	lastCapture.originalSampledDir = originalSampledDir;
	lastCapture.childFixedDir[0] = fixedDirA;
	lastCapture.childSampledDir[0] = sampledDirA;
	lastCapture.childFixedDir[1] = fixedDirB;
	lastCapture.childSampledDir[1] = sampledDirB;
	lastCapture.childSampleWeight = childSampleWeight;
	lastCapture.childSampleNumerator = childSampleWeight * childPdfW;
	lastCapture.valid = mixMaterial && selectedMaterial &&
		std::isfinite(outerPassThroughEvent) &&
		std::isfinite(childPassThroughEvent) &&
		std::isfinite(childWeight) && childWeight > 0.f &&
		std::isfinite(childPdfW) && childPdfW > 0.f &&
		!childSampleWeight.IsNaN() && !childSampleWeight.IsInf() &&
		std::isfinite(fixedDirA.x) && std::isfinite(fixedDirA.y) &&
		std::isfinite(fixedDirA.z) && std::isfinite(sampledDirA.x) &&
		std::isfinite(sampledDirA.y) && std::isfinite(sampledDirA.z) &&
		std::isfinite(fixedDirB.x) && std::isfinite(fixedDirB.y) &&
		std::isfinite(fixedDirB.z) && std::isfinite(sampledDirB.x) &&
		std::isfinite(sampledDirB.y) && std::isfinite(sampledDirB.z);
}

inline bool ReadMatching(const void *mixMaterial,
		const void *selectedMaterial, const unsigned int selectedChild,
		const float outerPassThroughEvent, Capture *result) {
	if (!result || !lastCapture.valid ||
			lastCapture.mixMaterial != mixMaterial ||
			lastCapture.selectedMaterial != selectedMaterial ||
			lastCapture.selectedChild != selectedChild ||
			lastCapture.outerPassThroughEvent != outerPassThroughEvent)
		return false;
	*result = lastCapture;
	return true;
}

inline double VectorMaxAbsDifference(const luxrays::Vector &a,
		const luxrays::Vector &b) {
	const double av[3] = { double(a.x), double(a.y), double(a.z) };
	const double bv[3] = { double(b.x), double(b.y), double(b.z) };
	return MaxAbsDifference3(av, bv);
}

} }

#endif // _SLG_MLHERO_MIX_DIRECTION_TRACE15NR_H

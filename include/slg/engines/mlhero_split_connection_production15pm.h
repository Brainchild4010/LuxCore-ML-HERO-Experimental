/***************************************************************************
 * ML HERO phase 15pm: production split-child vertex-connection sample.   *
 ***************************************************************************/

#ifndef _SLG_MLHERO_SPLIT_CONNECTION_PRODUCTION15PM_H
#define _SLG_MLHERO_SPLIT_CONNECTION_PRODUCTION15PM_H

#include <cmath>
#include <cstdint>
#include <limits>

namespace slg {
namespace mlhero15pm {

static const unsigned int LaneCount = 8u;

struct Layout {
	bool valid;
	bool compactLaneShift;
	unsigned int base;
	unsigned int maxDepth;
	unsigned int eyeStepSize;
	unsigned int requestedDimensions;
};

inline Layout MakeLayout(const unsigned int base,
		const unsigned int maxDepth, const unsigned int eyeStepSize,
		const unsigned int requestedDimensions) {
	Layout result = { false, true, base, maxDepth, eyeStepSize,
		requestedDimensions };
	if (maxDepth == 0u || eyeStepSize <= 6u)
		return result;
	// Reuse the native BIDIR eye-vertex connection dimension
	// sampleOffset + 6. No tail dimensions are added, so Sobol's fixed
	// dimension ceiling remains valid even at the configured maximum depth.
	const std::uint64_t last = static_cast<std::uint64_t>(base) +
		static_cast<std::uint64_t>(maxDepth - 1u) * eyeStepSize + 6u;
	result.valid = last < requestedDimensions;
	return result;
}

inline bool VertexDimension(const Layout &layout, const unsigned int lane,
		const unsigned int depth, unsigned int *dimension) {
	if (!layout.valid || !dimension || lane >= LaneCount || depth == 0u ||
			depth > layout.maxDepth)
		return false;
	const std::uint64_t value = static_cast<std::uint64_t>(layout.base) +
		static_cast<std::uint64_t>(depth - 1u) * layout.eyeStepSize + 6u;
	if (value >= layout.requestedDimensions)
		return false;
	*dimension = static_cast<unsigned int>(value);
	return true;
}

inline std::uint32_t ShiftHash(const unsigned int lane,
		const unsigned int depth) {
	std::uint32_t value = 0x9e3779b9u * (lane + 1u);
	value ^= 0x85ebca6bu * depth;
	value ^= value >> 16u;
	value *= 0x7feb352du;
	value ^= value >> 15u;
	value *= 0x846ca68bu;
	value ^= value >> 16u;
	return value;
}

inline float ApplyLaneShift(const float sample, const unsigned int lane,
		const unsigned int depth) {
	if (!std::isfinite(sample) || sample < 0.f || sample >= 1.f ||
			lane >= LaneCount || depth == 0u)
		return std::numeric_limits<float>::quiet_NaN();
	const double offset = (static_cast<double>(ShiftHash(lane, depth)) + .5) /
		4294967296.;
	double shifted = static_cast<double>(sample) + offset;
	shifted -= std::floor(shifted);
	float result = static_cast<float>(shifted);
	if (result >= 1.f)
		result = std::nextafter(1.f, 0.f);
	return result;
}

} // namespace mlhero15pm
} // namespace slg

#endif

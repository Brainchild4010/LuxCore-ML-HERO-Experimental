/***************************************************************************
 * ML HERO phase 15ng: production-scope split-child DLS dimensions.       *
 ***************************************************************************/

#ifndef _SLG_MLHERO_SPLIT_DLS_PRODUCTION15NG_H
#define _SLG_MLHERO_SPLIT_DLS_PRODUCTION15NG_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace slg {
namespace mlhero15ng {

static const unsigned int LaneCount = 8u;
static const unsigned int DrawsPerVertex = 5u;

// Validate the value that can actually be written to Spectrum's float
// storage.  Reconstructing a small correction afterwards with
// (stored - before) is ill-conditioned when before is much larger than the
// correction.  The exact sum of a float and a double is formed in double,
// rounded once to float, and compared at the storage value instead.
struct StorageProof {
	bool finite;
	bool correctlyRounded;
	double exactAfter;
	double absoluteError;
	double roundingBound;
};

inline double FloatRoundingBound(const float stored) {
	if (!std::isfinite(stored))
		return 0.;
	const float lower = std::nextafter(stored,
		-std::numeric_limits<float>::infinity());
	const float upper = std::nextafter(stored,
		std::numeric_limits<float>::infinity());
	double spacing = 0.;
	if (std::isfinite(lower))
		spacing = std::max(spacing,
			std::fabs(static_cast<double>(stored) - lower));
	if (std::isfinite(upper))
		spacing = std::max(spacing,
			std::fabs(static_cast<double>(upper) - stored));
	return .5 * spacing;
}

inline StorageProof ValidateStoredCorrection(const float before,
		const double exactCorrection, const float stored) {
	StorageProof proof = { false, false, 0., 0., 0. };
	proof.exactAfter = static_cast<double>(before) + exactCorrection;
	proof.finite = std::isfinite(before) && std::isfinite(exactCorrection) &&
		std::isfinite(proof.exactAfter) && std::isfinite(stored);
	if (!proof.finite)
		return proof;
	const float expectedStored = static_cast<float>(proof.exactAfter);
	proof.absoluteError = std::fabs(static_cast<double>(stored) -
		proof.exactAfter);
	proof.roundingBound = FloatRoundingBound(stored);
	const double doubleSlack = 8. * std::numeric_limits<double>::epsilon() *
		std::max(std::numeric_limits<double>::min(),
			std::fabs(proof.exactAfter));
	proof.correctlyRounded = stored == expectedStored &&
		proof.absoluteError <= proof.roundingBound + doubleSlack;
	return proof;
}

struct Layout {
	bool valid;
	bool compactLaneShift;
	unsigned int base;
	unsigned int maxDepth;
	unsigned int laneCapacity;
	unsigned int requestedDimensions;
};

inline Layout MakeLayout(const unsigned int base,
		const unsigned int maxEyePathDepth) {
	Layout result = { false, false, base, maxEyePathDepth, 0u, base };
	const std::uint64_t capacity =
		static_cast<std::uint64_t>(DrawsPerVertex) * maxEyePathDepth;
	const std::uint64_t total = static_cast<std::uint64_t>(base) +
		static_cast<std::uint64_t>(LaneCount) * capacity;
	if (maxEyePathDepth == 0u || capacity == 0ull ||
			capacity > std::numeric_limits<unsigned int>::max() ||
			total > std::numeric_limits<unsigned int>::max())
		return result;
	result.valid = true;
	result.laneCapacity = static_cast<unsigned int>(capacity);
	result.requestedDimensions = static_cast<unsigned int>(total);
	return result;
}

// High-depth low-discrepancy layout. Each depth keeps five real sampler
// dimensions. Lanes use deterministic Cranley-Patterson rotations of those
// dimensions instead of reserving eight complete copies. A fixed rotation is
// measure preserving, so every lane remains uniform and low-discrepancy while
// the physical estimator and its PDFs are unchanged.
inline Layout MakeCompactLayout(const unsigned int base,
		const unsigned int maxEyePathDepth) {
	Layout result = { false, true, base, maxEyePathDepth, 0u, base };
	const std::uint64_t capacity =
		static_cast<std::uint64_t>(DrawsPerVertex) * maxEyePathDepth;
	const std::uint64_t total = static_cast<std::uint64_t>(base) + capacity;
	if (maxEyePathDepth == 0u || capacity == 0ull ||
			capacity > std::numeric_limits<unsigned int>::max() ||
			total > std::numeric_limits<unsigned int>::max())
		return result;
	result.valid = true;
	result.laneCapacity = static_cast<unsigned int>(capacity);
	result.requestedDimensions = static_cast<unsigned int>(total);
	return result;
}

inline Layout MakeLayoutForLimit(const unsigned int base,
		const unsigned int maxEyePathDepth,
		const unsigned int maximumDimensions) {
	const Layout full = MakeLayout(base, maxEyePathDepth);
	if (full.valid && full.requestedDimensions <= maximumDimensions)
		return full;
	const Layout compact = MakeCompactLayout(base, maxEyePathDepth);
	if (compact.valid && compact.requestedDimensions <= maximumDimensions)
		return compact;
	Layout invalid = { false, true, base, maxEyePathDepth, 0u, base };
	return invalid;
}

inline bool VertexDimensions(const Layout &layout, const unsigned int lane,
		const unsigned int depth, unsigned int dimensions[DrawsPerVertex]) {
	if (!layout.valid || !dimensions || lane >= LaneCount || depth == 0u ||
			depth > layout.maxDepth)
		return false;
	const std::uint64_t first = static_cast<std::uint64_t>(layout.base) +
		(layout.compactLaneShift ? 0ull :
			static_cast<std::uint64_t>(lane) * layout.laneCapacity) +
		static_cast<std::uint64_t>(depth - 1u) * DrawsPerVertex;
	const std::uint64_t end = first + DrawsPerVertex;
	if (end > layout.requestedDimensions)
		return false;
	for (unsigned int i = 0u; i < DrawsPerVertex; ++i)
		dimensions[i] = static_cast<unsigned int>(first + i);
	return true;
}

inline bool BlocksAreDisjoint(const Layout &layout) {
	if (!layout.valid || layout.laneCapacity == 0u || layout.compactLaneShift)
		return false;
	for (unsigned int a = 0u; a < LaneCount; ++a) {
		const std::uint64_t a0 = static_cast<std::uint64_t>(layout.base) +
			static_cast<std::uint64_t>(a) * layout.laneCapacity;
		const std::uint64_t a1 = a0 + layout.laneCapacity;
		for (unsigned int b = a + 1u; b < LaneCount; ++b) {
			const std::uint64_t b0 = static_cast<std::uint64_t>(layout.base) +
				static_cast<std::uint64_t>(b) * layout.laneCapacity;
			const std::uint64_t b1 = b0 + layout.laneCapacity;
			if (!(a1 <= b0 || b1 <= a0))
				return false;
		}
	}
	return true;
}

inline std::uint32_t LaneShiftHash(const unsigned int lane,
		const unsigned int depth, const unsigned int draw) {
	std::uint32_t value = 0x9e3779b9u * (lane + 1u);
	value ^= 0x85ebca6bu * depth;
	value ^= 0xc2b2ae35u * (draw + 1u);
	value ^= value >> 16u;
	value *= 0x7feb352du;
	value ^= value >> 15u;
	value *= 0x846ca68bu;
	value ^= value >> 16u;
	return value;
}

inline float ApplyLaneShift(const float sample, const unsigned int lane,
		const unsigned int depth, const unsigned int draw) {
	if (!std::isfinite(sample) || sample < 0.f || sample >= 1.f ||
			lane >= LaneCount || depth == 0u || draw >= DrawsPerVertex)
		return std::numeric_limits<float>::quiet_NaN();
	const double offset = (static_cast<double>(LaneShiftHash(lane, depth, draw)) + .5) /
		4294967296.;
	double shifted = static_cast<double>(sample) + offset;
	shifted -= std::floor(shifted);
	float result = static_cast<float>(shifted);
	if (result >= 1.f)
		result = std::nextafter(1.f, 0.f);
	return result;
}

inline bool SummaryPass(const unsigned long long groupsConsidered,
		const unsigned long long groupsArmed,
		const unsigned long long groupFallbacks,
		const unsigned long long groupsClosed,
		const unsigned long long attempts,
		const unsigned long long samplerDraws,
		const unsigned long long nonzero,
		const unsigned long long spectral,
		const unsigned long long rgbFallback,
		const unsigned long long failures) {
	return groupsConsidered > 0ull && groupsArmed > 0ull &&
		groupsConsidered == groupsArmed + groupFallbacks &&
		groupsClosed == groupsArmed && attempts > 0ull &&
		samplerDraws == attempts * DrawsPerVertex &&
		spectral == nonzero && rgbFallback == 0ull && failures == 0ull;
}

} // namespace mlhero15ng
} // namespace slg

#endif // _SLG_MLHERO_SPLIT_DLS_PRODUCTION15NG_H

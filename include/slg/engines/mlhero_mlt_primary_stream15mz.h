/***************************************************************************
 * ML HERO phase 15mz: fixed primary-sample layout for MLT split lanes.    *
 ***************************************************************************/

#ifndef _SLG_MLHERO_MLT_PRIMARY_STREAM15MZ_H
#define _SLG_MLHERO_MLT_PRIMARY_STREAM15MZ_H

#include <cstdint>
#include <limits>

namespace slg {
namespace mlhero15mz {

static const unsigned int LaneCount = 8u;
static const unsigned int DrawsPerDepth = 4u;

struct Layout {
	bool valid;
	unsigned int parentDimensions;
	unsigned int laneCapacity;
	unsigned int requestedDimensions;
};

inline Layout MakeLayout(const unsigned int parentDimensions,
		const unsigned int maxEyePathDepth) {
	Layout result = { false, parentDimensions, 0u, parentDimensions };
	const std::uint64_t capacity = static_cast<std::uint64_t>(DrawsPerDepth) *
		(static_cast<std::uint64_t>(maxEyePathDepth) + 1ull);
	const std::uint64_t total = static_cast<std::uint64_t>(parentDimensions) +
		static_cast<std::uint64_t>(LaneCount) * capacity;
	if (capacity == 0ull || capacity > std::numeric_limits<unsigned int>::max() ||
			total > std::numeric_limits<unsigned int>::max())
		return result;
	result.valid = true;
	result.laneCapacity = static_cast<unsigned int>(capacity);
	result.requestedDimensions = static_cast<unsigned int>(total);
	return result;
}

inline bool LaneBlock(const Layout &layout, const unsigned int lane,
		unsigned int *base, unsigned int *capacity) {
	if (!layout.valid || lane >= LaneCount || !base || !capacity)
		return false;
	const std::uint64_t laneBase =
		static_cast<std::uint64_t>(layout.parentDimensions) +
		static_cast<std::uint64_t>(lane) * layout.laneCapacity;
	const std::uint64_t laneEnd = laneBase + layout.laneCapacity;
	if (laneEnd > layout.requestedDimensions)
		return false;
	*base = static_cast<unsigned int>(laneBase);
	*capacity = layout.laneCapacity;
	return true;
}

} // namespace mlhero15mz
} // namespace slg

#endif // _SLG_MLHERO_MLT_PRIMARY_STREAM15MZ_H

/***************************************************************************
 * ML HERO phase 15nh: atomic visible split-child direct-light sampling.   *
 ***************************************************************************/

#ifndef _SLG_MLHERO_VISIBLE_SPLIT_DLS15NH_H
#define _SLG_MLHERO_VISIBLE_SPLIT_DLS15NH_H

namespace slg {
namespace mlhero15nh {

inline bool SummaryPass(const bool requested,
		const unsigned long long groupsConsidered,
		const unsigned long long groupsArmed,
		const unsigned long long preArmFallbackGroups,
		const unsigned long long groupsClosed,
		const unsigned long long appliedSamples,
		const unsigned long long closeFallbackSamples,
		const unsigned long long mutationFailures,
		const unsigned long long nonfinite) {
	return requested && groupsConsidered > 0ull && groupsArmed > 0ull &&
		groupsConsidered == groupsArmed + preArmFallbackGroups &&
		groupsClosed == groupsArmed &&
		groupsClosed == appliedSamples + closeFallbackSamples &&
		appliedSamples > 0ull && mutationFailures == 0ull &&
		nonfinite == 0ull;
}

} // namespace mlhero15nh
} // namespace slg

#endif // _SLG_MLHERO_VISIBLE_SPLIT_DLS15NH_H

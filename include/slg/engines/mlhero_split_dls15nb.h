/***************************************************************************
 * ML HERO phase 15nb: private split-child DLS ownership helpers.          *
 ***************************************************************************/

#ifndef _SLG_MLHERO_SPLIT_DLS15NB_H
#define _SLG_MLHERO_SPLIT_DLS15NB_H

namespace slg {
namespace mlhero15nb {

struct LanePlan {
	unsigned begin;
	unsigned end;
	double divisor;
	bool valid;
};

inline LanePlan MakeLanePlan(const unsigned laneCount,
		const unsigned activeLane, const bool privateSplitChild) {
	LanePlan result = {0u, laneCount, static_cast<double>(laneCount),
		laneCount >= 2u && laneCount <= 8u};
	if (!result.valid)
		return result;
	if (privateSplitChild) {
		result.valid = activeLane < laneCount;
		result.begin = activeLane;
		result.end = activeLane + 1u;
		result.divisor = 1.;
	}
	return result;
}

inline bool SummaryPass(const unsigned long long attempts,
		const unsigned long long contributions,
		const unsigned long long zeroOrOccluded,
		const unsigned long long spectralContributions,
		const unsigned long long fallbackContributions,
		const unsigned long long diagnosticDraws,
		const unsigned long long failures) {
	return attempts > 0ull && contributions > 0ull &&
		contributions + zeroOrOccluded == attempts &&
		spectralContributions == contributions &&
		fallbackContributions == 0ull && diagnosticDraws == attempts * 5ull &&
		failures == 0ull;
}

} // namespace mlhero15nb
} // namespace slg

#endif // _SLG_MLHERO_SPLIT_DLS15NB_H

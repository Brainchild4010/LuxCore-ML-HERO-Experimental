/***************************************************************************
 * ML HERO phase 15pn: dual-terminated vertex-connection proof.           *
 ***************************************************************************/

#ifndef _SLG_MLHERO_DUAL_TERMINATED_CONNECTION15PN_H
#define _SLG_MLHERO_DUAL_TERMINATED_CONNECTION15PN_H

#include <algorithm>
#include <cmath>

namespace slg {
namespace mlhero15pn {

struct Proof {
	bool finite;
	bool closes;
	double relativeError;
};

// Both split subpaths already own one live wavelength and one factor N.  The
// native packet reduction contributes 1/N and dual compensation contributes
// the second 1/N, so packet * N^2 must recover the raw live-lane estimator.
inline Proof CheckMatchingLane(const double packet[3], const double rawLane[3],
		const unsigned int laneCount) {
	Proof proof = { false, false, 0. };
	if (!packet || !rawLane || laneCount < 2u || laneCount > 8u)
		return proof;
	proof.finite = true;
	const double scaleN = (double)laneCount * (double)laneCount;
	for (unsigned int channel = 0u; channel < 3u; ++channel) {
		const double reconstructed = packet[channel] * scaleN;
		if (!std::isfinite(packet[channel]) || !std::isfinite(rawLane[channel]) ||
				!std::isfinite(reconstructed)) {
			proof.finite = false;
			break;
		}
		const double scale = std::max(1e-20,
			std::max(std::fabs(rawLane[channel]), std::fabs(reconstructed)));
		proof.relativeError = std::max(proof.relativeError,
			std::fabs(rawLane[channel] - reconstructed) / scale);
	}
	proof.closes = proof.finite && proof.relativeError <= 2e-5;
	return proof;
}

inline bool IsZero(const double value[3]) {
	return value && value[0] == 0. && value[1] == 0. && value[2] == 0.;
}

} // namespace mlhero15pn
} // namespace slg

#endif // _SLG_MLHERO_DUAL_TERMINATED_CONNECTION15PN_H

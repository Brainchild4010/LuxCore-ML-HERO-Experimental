/***************************************************************************
 * ML HERO phase 15ni: private split-child vertex-connection proof.        *
 ***************************************************************************/

#ifndef _SLG_MLHERO_SPLIT_CONNECTION15NI_H
#define _SLG_MLHERO_SPLIT_CONNECTION15NI_H

#include <algorithm>
#include <cmath>

namespace slg {
namespace mlhero15ni {

// A deterministic split child already owns 1/N of the packet estimator.
// The ordinary packet ConnectVertices path divides the remaining active lane
// by N once more.  This proof compares that untouched packet result with the
// physically required single-lane split estimator.
struct NormalizationProof {
	bool finite;
	bool closes;
	double relativeError;
};

inline bool PacketCompatible(const float *eyeWaveLengths,
		const float *eyeSampleWeights, const unsigned int eyeCount,
		const float *lightWaveLengths, const float *lightSampleWeights,
		const unsigned int lightCount) {
	if (!eyeWaveLengths || !eyeSampleWeights || !lightWaveLengths ||
			!lightSampleWeights || eyeCount < 2u || eyeCount > 8u ||
			lightCount != eyeCount)
		return false;
	for (unsigned int lane = 0u; lane < eyeCount; ++lane) {
		const double eyeLambda = eyeWaveLengths[lane];
		const double lightLambda = lightWaveLengths[lane];
		const double eyeWeight = eyeSampleWeights[lane];
		const double lightWeight = lightSampleWeights[lane];
		if (!std::isfinite(eyeLambda) || !std::isfinite(lightLambda) ||
				!std::isfinite(eyeWeight) || !std::isfinite(lightWeight) ||
				std::fabs(eyeLambda - lightLambda) >= 1e-4 ||
				std::fabs(eyeWeight - lightWeight) >= 1e-6)
			return false;
	}
	return true;
}

inline NormalizationProof CheckPacketExtraDivision(
		const double packetContribution[3],
		const double splitContribution[3],
		const unsigned int packetCount) {
	NormalizationProof proof = { false, false, 0. };
	if (!packetContribution || !splitContribution ||
			packetCount < 2u || packetCount > 8u)
		return proof;
	proof.finite = true;
	for (unsigned int channel = 0u; channel < 3u; ++channel) {
		const double packet = packetContribution[channel];
		const double split = splitContribution[channel];
		const double reconstructed = packet * packetCount;
		if (!std::isfinite(packet) || !std::isfinite(split) ||
				!std::isfinite(reconstructed)) {
			proof.finite = false;
			break;
		}
		const double scale = std::max(1e-20,
			std::max(std::fabs(split), std::fabs(reconstructed)));
		proof.relativeError = std::max(proof.relativeError,
			std::fabs(split - reconstructed) / scale);
	}
	proof.closes = proof.finite && proof.relativeError <= 2e-5;
	return proof;
}

inline bool SummaryPass(const unsigned long long targetReached,
		const unsigned long long attempts,
		const unsigned long long retainedSpectral,
		const unsigned long long reachedAfterTarget,
		const unsigned long long noContribution,
		const unsigned long long rgbFallbackVisible,
		const unsigned long long spectralProofRejected,
		const unsigned long long spectral,
		const unsigned long long normalizationProofs,
		const unsigned long long failures) {
	return targetReached > 0ull && attempts > 0ull &&
		attempts == retainedSpectral + reachedAfterTarget +
			noContribution + rgbFallbackVisible + spectralProofRejected &&
		retainedSpectral == targetReached &&
		retainedSpectral == spectral && spectral > 0ull &&
		normalizationProofs == retainedSpectral &&
		spectralProofRejected == 0ull &&
		failures == 0ull;
}

} // namespace mlhero15ni
} // namespace slg

#endif // _SLG_MLHERO_SPLIT_CONNECTION15NI_H

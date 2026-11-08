/***************************************************************************
 * Copyright 1998-2020 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 *     http://www.apache.org/licenses/LICENSE-2.0                          *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an "AS IS" BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 * See the License for the specific language governing permissions and     *
 * limitations under the License.                                          *
 ***************************************************************************/

#ifndef _SLG_MLHERO_MIX_CLOSURE_DIAGNOSTIC15NQ_H
#define _SLG_MLHERO_MIX_CLOSURE_DIAGNOSTIC15NQ_H

#include <algorithm>
#include <cmath>

namespace slg { namespace mlhero15nq {

struct Analysis {
	bool finite;
	bool doubleClosureAccepted;
	unsigned int selectedChild;
	double doubleReconstructed[3];
	double nativeNumerator[3];
	double inferredSelectedNumerator[3];
	double evaluatedSelectedNumerator[3];
	double doubleClosureRelativeError;
	double selectedNumeratorRelativeError;
	double comparableMixturePdf;
	double nativePdfRelativeError;
};

inline Analysis Analyze(const double weightA, const double weightB,
		const double childA[3], const double childB[3],
		const double childDirectPdfA, const double childDirectPdfB,
		const double childReversePdfA, const double childReversePdfB,
		const double wrapperFactor, const double nativePdfW,
		const double nativeValue[3], const double passThroughEvent,
		const bool fromLight, const double absoluteTolerance = 2e-6,
		const double relativeTolerance = 2e-4) {
	Analysis result = { true, false,
		(passThroughEvent < weightA) ? 0u : 1u,
		{ 0., 0., 0. }, { 0., 0., 0. }, { 0., 0., 0. },
		{ 0., 0., 0. }, 0., 0., 0., 0. };

	const bool childABlack = childA[0] == 0. && childA[1] == 0. &&
		childA[2] == 0.;
	const bool childBBlack = childB[0] == 0. && childB[1] == 0. &&
		childB[2] == 0.;
	// MixMaterial::Sample always keeps the selected child's weighted PDF and
	// adds the other child's PDF only when that Evaluate contribution is not
	// black. Mirror that rule exactly; this is not the more general Mix::Pdf().
	const double selectedDirectPdf = result.selectedChild == 0u ?
		weightA * childDirectPdfA : weightB * childDirectPdfB;
	const double selectedReversePdf = result.selectedChild == 0u ?
		weightA * childReversePdfA : weightB * childReversePdfB;
	const double otherDirectPdf = result.selectedChild == 0u ?
		(childBBlack ? 0. : weightB * childDirectPdfB) :
		(childABlack ? 0. : weightA * childDirectPdfA);
	const double otherReversePdf = result.selectedChild == 0u ?
		(childBBlack ? 0. : weightB * childReversePdfB) :
		(childABlack ? 0. : weightA * childReversePdfA);
	result.comparableMixturePdf = fromLight ?
		selectedReversePdf + otherReversePdf :
		selectedDirectPdf + otherDirectPdf;
	const double pdfScale = std::max(1e-12,
		std::max(std::fabs(result.comparableMixturePdf), std::fabs(nativePdfW)));
	result.nativePdfRelativeError =
		std::fabs(result.comparableMixturePdf - nativePdfW) / pdfScale;

	if (!std::isfinite(weightA) || !std::isfinite(weightB) ||
			!std::isfinite(wrapperFactor) || !(wrapperFactor > 0.) ||
			!std::isfinite(nativePdfW) || !(nativePdfW > 0.) ||
			!std::isfinite(passThroughEvent) ||
			!std::isfinite(result.comparableMixturePdf) ||
			!std::isfinite(result.nativePdfRelativeError)) {
		result.finite = false;
		return result;
	}

	double closureScale = 1e-8;
	double closureMaxAbs = 0.;
	double selectedScale = 1e-8;
	double selectedMaxAbs = 0.;
	for (unsigned int channel = 0u; channel < 3u; ++channel) {
		if (!std::isfinite(childA[channel]) ||
				!std::isfinite(childB[channel]) ||
				!std::isfinite(nativeValue[channel])) {
			result.finite = false;
			return result;
		}
		result.doubleReconstructed[channel] =
			(weightA * childA[channel] + weightB * childB[channel]) *
			wrapperFactor / nativePdfW;
		result.nativeNumerator[channel] = nativeValue[channel] *
			nativePdfW / wrapperFactor;

		const double otherNumerator = result.selectedChild == 0u ?
			weightB * childB[channel] : weightA * childA[channel];
		result.inferredSelectedNumerator[channel] =
			result.nativeNumerator[channel] - otherNumerator;
		result.evaluatedSelectedNumerator[channel] =
			result.selectedChild == 0u ? weightA * childA[channel] :
				weightB * childB[channel];

		if (!std::isfinite(result.doubleReconstructed[channel]) ||
				!std::isfinite(result.nativeNumerator[channel]) ||
				!std::isfinite(result.inferredSelectedNumerator[channel]) ||
				!std::isfinite(result.evaluatedSelectedNumerator[channel])) {
			result.finite = false;
			return result;
		}

		closureScale = std::max(closureScale,
			std::max(std::fabs(result.doubleReconstructed[channel]),
				std::fabs(nativeValue[channel])));
		closureMaxAbs = std::max(closureMaxAbs,
			std::fabs(result.doubleReconstructed[channel] - nativeValue[channel]));
		selectedScale = std::max(selectedScale,
			std::max(std::fabs(result.inferredSelectedNumerator[channel]),
				std::fabs(result.evaluatedSelectedNumerator[channel])));
		selectedMaxAbs = std::max(selectedMaxAbs,
			std::fabs(result.inferredSelectedNumerator[channel] -
				result.evaluatedSelectedNumerator[channel]));
	}

	result.doubleClosureRelativeError = closureMaxAbs / closureScale;
	result.selectedNumeratorRelativeError = selectedMaxAbs / selectedScale;
	result.doubleClosureAccepted = closureMaxAbs <=
		absoluteTolerance + relativeTolerance * closureScale;
	return result;
}

inline unsigned int RelativeErrorBucket(const double error) {
	if (!std::isfinite(error)) return 4u;
	if (error <= 2e-4) return 0u;
	if (error <= 1e-3) return 1u;
	if (error <= 1e-2) return 2u;
	return 3u;
}

} }

#endif // _SLG_MLHERO_MIX_CLOSURE_DIAGNOSTIC15NQ_H

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

#ifndef _SLG_MLHERO_MIX_CLOSURE15NM_H
#define _SLG_MLHERO_MIX_CLOSURE15NM_H

#include <algorithm>
#include <cmath>

namespace slg { namespace mlhero15nm {

struct Closure {
	bool finite;
	bool accepted;
	double scale;
	double maxAbsError;
	double relativeError;
};

inline Closure AnalyzeClosure(const double reconstructed[3],
		const double nativeValue[3], const double absoluteTolerance = 2e-6,
		const double relativeTolerance = 2e-4) {
	Closure result = { true, false, 1e-8, 0., 0. };
	for (unsigned int i = 0u; i < 3u; ++i) {
		if (!std::isfinite(reconstructed[i]) ||
				!std::isfinite(nativeValue[i])) {
			result.finite = false;
			return result;
		}
		result.scale = std::max(result.scale,
			std::max(std::fabs(reconstructed[i]), std::fabs(nativeValue[i])));
		result.maxAbsError = std::max(result.maxAbsError,
			std::fabs(reconstructed[i] - nativeValue[i]));
	}
	result.relativeError = result.maxAbsError / result.scale;
	result.accepted = result.maxAbsError <=
		absoluteTolerance + relativeTolerance * result.scale;
	return result;
}

inline double EvaluateLane(const double weightA, const double valueA,
		const double weightB, const double valueB,
		const double wrapperFactor, const double nativePdfW) {
	return (weightA * valueA + weightB * valueB) *
		wrapperFactor / nativePdfW;
}

} }

#endif // _SLG_MLHERO_MIX_CLOSURE15NM_H

/***************************************************************************
 * ML HERO phase 15nd: validated Glossy2 DLS decomposition.                *
 ***************************************************************************/

#ifndef _SLG_MLHERO_GLOSSY2_DLS15ND_H
#define _SLG_MLHERO_GLOSSY2_DLS15ND_H

#include <algorithm>
#include <cmath>

namespace slg {
namespace mlhero15nd {

struct Analysis {
	bool finite;
	bool closure;
	bool baseFit;
	bool coatingScalar;
	bool pass;
	double baseFactor;
	double coatingValue;
	double closureRelativeError;
	double baseRelativeError;
	double coatingRelativeSpread;
};

inline Analysis Analyze(const double kd[3], const double base[3],
		const double coating[3], const double combined[3],
		const double nativeValue[3]) {
	Analysis result = {true, false, false, false, false, 0., 0.,
		0., 0., 0.};
	double kdDenominator = 0.;
	double baseDotKd = 0.;
	double baseMaximum = 0.;
	double coatingMaximum = 0.;
	double closureError = 0.;
	for (unsigned i = 0u; i < 3u; ++i) {
		result.finite = result.finite && std::isfinite(kd[i]) &&
			std::isfinite(base[i]) && std::isfinite(coating[i]) &&
			std::isfinite(combined[i]) && std::isfinite(nativeValue[i]);
		kdDenominator += kd[i] * kd[i];
		baseDotKd += base[i] * kd[i];
		baseMaximum = std::max(baseMaximum, std::fabs(base[i]));
		coatingMaximum = std::max(coatingMaximum, std::fabs(coating[i]));
		const double scale = std::max(1e-20,
			std::max(std::fabs(combined[i]), std::fabs(nativeValue[i])));
		closureError = std::max(closureError,
			std::fabs(combined[i] - nativeValue[i]) / scale);
	}
	if (!result.finite)
		return result;

	result.closureRelativeError = closureError;
	result.closure = closureError <= 2e-5;
	if (kdDenominator > 1e-20)
		result.baseFactor = baseDotKd / kdDenominator;
	else
		result.baseFactor = 0.;

	double baseError = 0.;
	for (unsigned i = 0u; i < 3u; ++i)
		baseError = std::max(baseError,
			std::fabs(base[i] - kd[i] * result.baseFactor));
	result.baseRelativeError = baseError / std::max(1e-9, baseMaximum);
	result.baseFit = result.baseRelativeError <= 1e-4 &&
		std::isfinite(result.baseFactor) && result.baseFactor >= 0.;

	result.coatingValue = (coating[0] + coating[1] + coating[2]) / 3.;
	const double coatingSpread = std::max(std::fabs(coating[0] - coating[1]),
		std::max(std::fabs(coating[0] - coating[2]),
			std::fabs(coating[1] - coating[2])));
	result.coatingRelativeSpread = coatingSpread /
		std::max(1e-8, coatingMaximum);
	result.coatingScalar = result.coatingRelativeSpread <= 1e-4 &&
		std::isfinite(result.coatingValue) && result.coatingValue >= 0.;
	result.pass = result.finite && result.closure && result.baseFit &&
		result.coatingScalar;
	return result;
}

} // namespace mlhero15nd
} // namespace slg

#endif // _SLG_MLHERO_GLOSSY2_DLS15ND_H

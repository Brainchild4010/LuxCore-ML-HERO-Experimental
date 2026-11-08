/***************************************************************************
 * ML HERO phase 15na: diagnostic Glossy2 coating-weight precision replay. *
 ***************************************************************************/

#ifndef _SLG_MLHERO_GLOSSY2_WEIGHT_CAPTURE15NA_H
#define _SLG_MLHERO_GLOSSY2_WEIGHT_CAPTURE15NA_H

#include <cmath>
#include <limits>

namespace slg {
namespace mlhero15na {

struct Replay {
	bool valid;
	double rawNorm, rawAbsZ, unitAbsZ;
	double ks0, ks1, ks2, ksFilter;
	double reportedBase;
	double rawNativeFloatBase;
	double unitNativeFloatBase;
	double rawDoubleBase;
	double unitDoubleBase;
	double absErrorRaw, absErrorUnit;
	double relErrorRaw, relErrorUnit;
};

inline double Relative(const double a, const double b) {
	return std::fabs(a - b) /
		std::fmax(1e-300, std::fmax(std::fabs(a), std::fabs(b)));
}

inline double BaseWeightDouble(const double ks0, const double ks1,
		const double ks2, const double absZ) {
	const double tail = std::pow(1. - absZ, 5.);
	const double s0 = ks0 + (1. - ks0) * tail;
	const double s1 = ks1 + (1. - ks1) * tail;
	const double s2 = ks2 + (1. - ks2) * tail;
	return 1. - .5 * (1. + (s0 + s1 + s2) / 3.);
}

inline float BaseWeightFloat(const float ks0, const float ks1,
		const float ks2, const float absZ) {
	// Match FresnelTexture::SchlickEvaluate(): powf(1.f - cosi, 5.f).
	const float tail = ::powf(1.f - absZ, 5.f);
	const float s0 = ks0 + (1.f - ks0) * tail;
	const float s1 = ks1 + (1.f - ks1) * tail;
	const float s2 = ks2 + (1.f - ks2) * tail;
	const float filter = (s0 + s1 + s2) * (1.f / 3.f);
	return 1.f - .5f * (1.f + filter);
}

inline Replay Analyze(const double ks0, const double ks1, const double ks2,
		const double rawX, const double rawY, const double rawZ,
		const double reportedBase) {
	Replay result = {};
	result.rawNorm = std::sqrt(rawX * rawX + rawY * rawY + rawZ * rawZ);
	result.rawAbsZ = std::fabs(rawZ);
	result.unitAbsZ = result.rawNorm > 0. ?
		result.rawAbsZ / result.rawNorm : 0.;
	result.ks0 = ks0;
	result.ks1 = ks1;
	result.ks2 = ks2;
	result.ksFilter = (ks0 + ks1 + ks2) / 3.;
	result.reportedBase = reportedBase;
	result.rawNativeFloatBase = BaseWeightFloat(static_cast<float>(ks0),
		static_cast<float>(ks1), static_cast<float>(ks2),
		static_cast<float>(result.rawAbsZ));
	result.unitNativeFloatBase = BaseWeightFloat(static_cast<float>(ks0),
		static_cast<float>(ks1), static_cast<float>(ks2),
		static_cast<float>(result.unitAbsZ));
	result.rawDoubleBase = BaseWeightDouble(ks0, ks1, ks2, result.rawAbsZ);
	result.unitDoubleBase = BaseWeightDouble(ks0, ks1, ks2, result.unitAbsZ);
	result.absErrorRaw = std::fabs(reportedBase - result.rawDoubleBase);
	result.absErrorUnit = std::fabs(reportedBase - result.unitDoubleBase);
	result.relErrorRaw = Relative(reportedBase, result.rawDoubleBase);
	result.relErrorUnit = Relative(reportedBase, result.unitDoubleBase);
	result.valid = std::isfinite(result.rawNorm) && result.rawNorm > 0. &&
		std::isfinite(result.reportedBase) &&
		std::isfinite(result.rawNativeFloatBase) &&
		std::isfinite(result.unitNativeFloatBase) &&
		std::isfinite(result.rawDoubleBase) &&
		std::isfinite(result.unitDoubleBase);
	return result;
}

} // namespace mlhero15na
} // namespace slg

#endif // _SLG_MLHERO_GLOSSY2_WEIGHT_CAPTURE15NA_H

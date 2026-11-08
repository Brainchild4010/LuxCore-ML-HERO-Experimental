/***************************************************************************
 * ML HERO phase 15nj: shared RGB-reflectance reconstruction used by the
 * wavelength-domain Glossy2 endpoint evaluator.
 ***************************************************************************/

#ifndef _SLG_MLHERO_GLOSSY2_SPECTRAL15NJ_H
#define _SLG_MLHERO_GLOSSY2_SPECTRAL15NJ_H

#include "luxrays/core/color/color.h"
#include "luxrays/core/color/spds/rgbrefl.h"
#include "slg/engines/mlhero_unit_reflectance15fx.h"

namespace slg { namespace mlhero15nj {

inline float RGBReflSample380_780(const luxrays::RGBReflSPD &spd,
		const float waveLength) {
	// The stock table ends at 720nm and treats its upper edge as outside.
	return spd.Sample(luxrays::Clamp(waveLength, 380.f, 719.999f));
}

inline float BasisCompensatedSample(const luxrays::Spectrum &rgb,
		const float waveLength) {
	const float coeffR = 0.81789457798f * rgb.c[0] +
		0.006252348423f * rgb.c[1] + 0.007401828188f * rgb.c[2];
	const float coeffG = 0.02175390534f * rgb.c[0] +
		1.00719189644f * rgb.c[1] + 0.01943249628f * rgb.c[2];
	const float coeffB = -0.000957758341f * rgb.c[0] -
		0.02197206020f * rgb.c[1] + 1.10662686825f * rgb.c[2];

	static const luxrays::RGBReflSPD red(luxrays::RGBColor(1.f, 0.f, 0.f));
	static const luxrays::RGBReflSPD green(luxrays::RGBColor(0.f, 1.f, 0.f));
	static const luxrays::RGBReflSPD blue(luxrays::RGBColor(0.f, 0.f, 1.f));
	return luxrays::Max(0.f,
		coeffR * RGBReflSample380_780(red, waveLength) +
		coeffG * RGBReflSample380_780(green, waveLength) +
		coeffB * RGBReflSample380_780(blue, waveLength));
}

inline float Reflectance(const luxrays::Spectrum &rgbInput,
		const float waveLength, const bool basisCompensation) {
	const luxrays::Spectrum rgb = rgbInput.Clamp(0.f, 1.f);
	if (rgb.c[0] == rgb.c[1] && rgb.c[1] == rgb.c[2])
		return rgb.c[0];
	if (basisCompensation)
		return MLHeroUnitReflectance15fx(
			BasisCompensatedSample(rgb, waveLength));
	const luxrays::RGBReflSPD spd(
		luxrays::RGBColor(rgb.c[0], rgb.c[1], rgb.c[2]));
	return MLHeroUnitReflectance15fx(
		RGBReflSample380_780(spd, waveLength));
}

} }

#endif

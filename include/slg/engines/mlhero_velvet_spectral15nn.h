/***************************************************************************
 * ML HERO phase 15nn: shared scalar part of the Velvet endpoint formula.
 *
 * This helper deliberately contains no wavelength reconstruction.  It keeps
 * the native Velvet phase/thickness/geometry calculation in one place so the
 * RGB implementation and the prepared spectral endpoint cannot drift apart.
 ***************************************************************************/

#ifndef _SLG_MLHERO_VELVET_SPECTRAL15NN_H
#define _SLG_MLHERO_VELVET_SPECTRAL15NN_H

#include <cmath>

namespace slg { namespace mlhero15nn {

// Material::Evaluate() returns f(light, eye) * |cos(light)|.  Clamp the
// physical BRDF before applying that transport cosine so reciprocity is kept.
inline float CosWeightedValue(const float phaseThickness,
		const float absLightCos, const float absEyeCos) {
	const float cosineProduct = absLightCos * absEyeCos;
	if (!(cosineProduct > 0.f))
		return 0.f;

	float brdf = phaseThickness / cosineProduct;
	if (brdf > 1.f)
		brdf = 1.f;
	else if (brdf < 0.f)
		brdf = 0.f;
	return brdf * absLightCos;
}

inline float PhaseFactor(const float a1, const float a2, const float a3,
		const float thickness, const float cosv,
		const float absLightCos, const float absEyeCos) {
	const float b = 3.0f * cosv;
	const float polynomial = 1.0f + a1 * cosv +
		a2 * 0.5f * (b * cosv - 1.0f) +
		a3 * 0.5f * (5.0f * cosv * cosv * cosv - b);
	const float phaseThickness = polynomial * thickness / (4.0f * M_PI);
	return CosWeightedValue(phaseThickness, absLightCos, absEyeCos);
}

} }

#endif

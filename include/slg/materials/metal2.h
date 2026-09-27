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

#ifndef _SLG_METAL2MAT_H
#define	_SLG_METAL2MAT_H

#include "slg/textures/fresnel/fresneltexture.h"
#include "slg/materials/material.h"

namespace slg {

//------------------------------------------------------------------------------
// Metal2 material
//------------------------------------------------------------------------------

class Metal2Material : public Material {
public:
	Metal2Material(TextureConstPtr frontTransp, TextureConstPtr backTransp,
			TextureConstPtr emitted, TextureConstPtr bump,
			TextureConstPtr nn, TextureConstPtr kk, TextureConstPtr u, TextureConstPtr v);
	Metal2Material(TextureConstPtr frontTransp, TextureConstPtr backTransp,
			TextureConstPtr emitted, TextureConstPtr bump,
			FresnelTextureConstPtr ft, TextureConstPtr u, TextureConstPtr v);

	virtual MaterialType GetType() const { return METAL2; }
	virtual BSDFEvent GetEventTypes() const { return GLOSSY | REFLECT; };

	virtual luxrays::Spectrum Albedo(const HitPoint &hitPoint) const;

	virtual luxrays::Spectrum Evaluate(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir, BSDFEvent *event,
		float *directPdfW = NULL, float *reversePdfW = NULL) const;
	virtual luxrays::Spectrum Sample(const HitPoint &hitPoint,
		const luxrays::Vector &localFixedDir, luxrays::Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const;
	virtual void Pdf(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const;

	virtual void AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexsreferencedTexs) const;
	virtual void UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex);

	virtual luxrays::PropertiesUPtr ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const;

	// ML-HERO helper: explicitly evaluate measured spectral Fresnel data at a
	// requested wavelength without changing the active HERO wavelength.
	// Returns false when this Metal2 material has no measured spectral n/k data.
	bool EvaluateFresnelAtWaveLength(const HitPoint &hitPoint, const float cosi,
			const float waveLength, luxrays::Spectrum *fresnel) const;
	bool EvaluateAtWaveLength(const HitPoint &hitPoint,
			const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir,
			const float waveLength, luxrays::Spectrum *value) const;

	// ML HERO phase 15n diagnostic: expose the exact measured n/k, Fresnel term
	// and shared microfacet factor used by EvaluateAtWaveLength().
	bool EvaluateAtWaveLengthDebug(const HitPoint &hitPoint,
			const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir,
			const float waveLength,
			luxrays::Spectrum *eta, luxrays::Spectrum *kk,
			luxrays::Spectrum *fresnel, float *microfacetFactor,
			float *cosWH, luxrays::Spectrum *value) const;

	// ML-HERO phase 6 helper: evaluate the exact Metal2 BSDF sample multiplier
	// for an already sampled pair of local directions at an explicit wavelength.
	// This keeps geometry, microfacet sampling and PDF shared between HERO lanes.
	bool EvaluateSampleAtWaveLength(const HitPoint &hitPoint,
			const luxrays::Vector &localFixedDir, const luxrays::Vector &localSampledDir,
			const float waveLength, luxrays::Spectrum *sampleMultiplier) const;

	FresnelTextureConstPtr GetFresnel() const { return fresnelTex; }
	TextureConstPtr GetN() const { return n; }
	TextureConstPtr GetK() const { return k; }
	TextureConstPtr GetNu() const { return nu; }
	TextureConstPtr GetNv() const { return nv; }
	
private:
	FresnelTextureConstPtr fresnelTex;
	// For compatibility with the past
	TextureConstPtr n, k;

	TextureConstPtr nu;
	TextureConstPtr nv;
};

}

#endif	/* _SLG_METAL2MAT_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

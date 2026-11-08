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

#ifndef _SLG_VELVETMAT_H
#define	_SLG_VELVETMAT_H

#include "slg/materials/material.h"

namespace slg {

//------------------------------------------------------------------------------
// Velvet material
//------------------------------------------------------------------------------

// Phase 15nn caches the native Velvet scalar lobe separately from Kd.  The
// scalar lobe is direction/phase/thickness dependent but not dispersive; only
// Kd is reconstructed at the requested HERO wavelength.
struct MLHeroVelvetSpectralEndpoint15nn {
	BSDFEvent event;
	luxrays::Spectrum kd;
	float phaseFactor;

	MLHeroVelvetSpectralEndpoint15nn() : event(NONE), kd(0.f),
		phaseFactor(0.f) { }
};

class VelvetMaterial : public Material {
public:
	VelvetMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
			TextureConstPtr emitted, TextureConstPtr bump,
			TextureConstPtr kd, TextureConstPtr p1, TextureConstPtr p2, TextureConstPtr p3,
			TextureConstPtr thickness);

	virtual MaterialType GetType() const { return VELVET; }
	virtual BSDFEvent GetEventTypes() const { return DIFFUSE | REFLECT; };

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

	// ML HERO phase 15nn: exact Velvet endpoint behind an outer MixMaterial.
	// Sampling, event, PDF, phase function, thickness and geometry remain
	// native.  Only the cached Kd texture is evaluated at one wavelength.
	bool PrepareMLHeroSpectralEndpoint15nn(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir,
		const luxrays::Vector &localEyeDir,
		MLHeroVelvetSpectralEndpoint15nn *prepared) const;
	static bool EvaluateMLHeroSpectralEndpoint15nn(
		const MLHeroVelvetSpectralEndpoint15nn &prepared,
		const float waveLength, const bool basisCompensation,
		float *value);

	virtual void AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexsreferencedTexs) const;
	virtual void UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex);

	virtual luxrays::PropertiesUPtr ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const;

	TextureConstPtr GetKd() const { return Kd; }
	TextureConstPtr GetP1() const { return P1; }
	TextureConstPtr GetP2() const { return P2; }
	TextureConstPtr GetP3() const { return P3; }
	TextureConstPtr GetThickness() const { return Thickness; }

private:
	TextureConstPtr Kd;
	TextureConstPtr P1;
	TextureConstPtr P2;
	TextureConstPtr P3;
	TextureConstPtr Thickness;
};

}

#endif	/* _SLG_VELVETMAT_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

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

#ifndef _SLG_GLOSSYMAT_H
#define	_SLG_GLOSSYMAT_H

#include "slg/materials/material.h"

namespace slg {

// ML HERO phase 15nk: texture-independent cached state for repeated
// wavelength evaluation of one already selected Glossy2 direction pair.
struct MLHeroGlossy2SpectralEndpoint15nk {
	BSDFEvent event;
	bool baseOnly;
	bool fromLight;
	bool multibounce;
	luxrays::Spectrum kd;
	luxrays::Spectrum ks;
	luxrays::Spectrum ka;
	luxrays::Vector localFixedDir;
	luxrays::Vector localSampledDir;
	float baseGeometry;
	float cosi;
	float coso;
	float depth;
	float roughness;
	float anisotropy;
	float wrapperFactor;

	MLHeroGlossy2SpectralEndpoint15nk() : event(NONE), baseOnly(false),
		fromLight(false), multibounce(false), kd(0.f), ks(0.f), ka(0.f),
		localFixedDir(), localSampledDir(), baseGeometry(0.f), cosi(0.f),
		coso(0.f), depth(0.f), roughness(0.f), anisotropy(0.f),
		wrapperFactor(1.f) { }
};

//------------------------------------------------------------------------------
// Glossy2 material
//------------------------------------------------------------------------------

class Glossy2Material : public Material {
public:
	Glossy2Material(TextureConstPtr frontTransp, TextureConstPtr backTransp,
			TextureConstPtr emitted, TextureConstPtr bump,
			TextureConstPtr kd, TextureConstPtr ks, TextureConstPtr u, TextureConstPtr v,
			TextureConstPtr ka, TextureConstPtr d, TextureConstPtr i, const bool mbounce, const bool doublesided);

	virtual MaterialType GetType() const { return GLOSSY2; }
	virtual BSDFEvent GetEventTypes() const { return GLOSSY | REFLECT; }

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

	// ML HERO phase 15ac diagnostic: split a sampled Glossy2 bounce into
	// coating and base-layer contributions, using the already sampled direction
	// and the same mixture PDF used by Sample().
	bool EvaluateMLHeroDebugSampleComponents(const HitPoint &hitPoint,
		const luxrays::Vector &localFixedDir, const luxrays::Vector &localSampledDir,
		const float pdfW,
		luxrays::Spectrum *kdValue, luxrays::Spectrum *ksValue,
		luxrays::Spectrum *schlickS, luxrays::Spectrum *absorptionValue,
		luxrays::Spectrum *baseContribution, luxrays::Spectrum *coatingContribution,
		luxrays::Spectrum *combinedResult) const;

	// ML HERO phase 15nd: exact Evaluate() numerator decomposition for DLS.
	// Unlike the sampled-bounce helper above, these values are not divided by
	// a sampling PDF.
	bool EvaluateMLHeroEvaluateComponents15nd(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir, const luxrays::Vector &localEyeDir,
		BSDFEvent *event, luxrays::Spectrum *kdValue,
		luxrays::Spectrum *baseContribution,
		luxrays::Spectrum *coatingContribution,
		luxrays::Spectrum *combinedResult) const;

	// ML HERO phase 15nj: evaluate the original Glossy2 formula at one
	// wavelength. Kd, Schlick Ks and coating absorption Ka are independently
	// reconstructed; no RGB component is fitted to a scalar surrogate.
	bool EvaluateMLHeroAtWaveLength15nj(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir,
		const luxrays::Vector &localEyeDir, const float waveLength,
		const bool basisCompensation, BSDFEvent *event,
		float *value) const;

	// ML HERO phase 15nk: cache all texture and geometric inputs once, then
	// evaluate the same selected direction at every packet wavelength.
	bool PrepareMLHeroSpectralEndpoint15nk(const HitPoint &hitPoint,
		const luxrays::Vector &localLightDir,
		const luxrays::Vector &localEyeDir,
		MLHeroGlossy2SpectralEndpoint15nk *prepared) const;
	static bool EvaluateMLHeroSpectralEndpoint15nk(
		const MLHeroGlossy2SpectralEndpoint15nk &prepared,
		const float waveLength, const bool basisCompensation,
		float *value);
	static bool EvaluateMLHeroNativeComponentsPrepared15nk(
		const MLHeroGlossy2SpectralEndpoint15nk &prepared,
		luxrays::Spectrum *baseContribution,
		luxrays::Spectrum *coatingContribution,
		luxrays::Spectrum *combinedResult);

	virtual void AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexsreferencedTexs) const;
	virtual void UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex);

	virtual luxrays::PropertiesUPtr ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const;

	TextureConstPtr GetKd() const { return Kd; }
	TextureConstPtr GetKs() const { return Ks; }
	TextureConstPtr GetNu() const { return nu; }
	TextureConstPtr GetNv() const { return nv; }
	TextureConstPtr GetKa() const { return Ka; }
	TextureConstPtr GetDepth() const { return depth; }
	TextureConstPtr GetIndex() const { return index; }
	const bool IsMultibounce () const { return multibounce; }
	const bool IsDoubleSided () const { return doublesided; }

private:
	TextureConstPtr Kd;
	TextureConstPtr Ks;
	TextureConstPtr nu;
	TextureConstPtr nv;
	TextureConstPtr Ka;
	TextureConstPtr depth;
	TextureConstPtr index;
	const bool multibounce;
	const bool doublesided;
};

}

#endif	/* _SLG_GLOSSYMAT_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

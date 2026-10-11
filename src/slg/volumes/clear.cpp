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

#include <cstddef>
#include <limits>
#include "slg/engines/mlhero_volume_diagnostics15fp.h"
#include "slg/engines/mlhero_volume_candidate15fq.h"
#include "slg/engines/mlhero_volume_reference15fr.h"
#include "slg/textures/colordepth.h"

#include "slg/volumes/clear.h"
#include "slg/bsdf/bsdf.h"
#include "slg/textures/irregulardata.h"

// ML HERO state is currently implemented as global thread-local helpers in
// glass.cpp. Keep these declarations global to match the existing definitions.
bool GetMLHeroEnabled();
float GetMLCurrentWaveLength();
u_int GetMLHeroWavelengthCount();
float GetMLHeroWaveLengthAt(const u_int lane);
void MarkMLDispersionUsed();

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// ClearVolume
//------------------------------------------------------------------------------

ClearVolume::ClearVolume(
	TextureConstRef iorTex,
	TextureConstPtr emiTex,
	TextureConstRef a
) : Volume(iorTex, emiTex), sigmaA(a) {}

Spectrum ClearVolume::SigmaA(const HitPoint &hitPoint) const {
	// ML HERO: evaluate the original IrregularData spectrum at the current
	// per-thread hero wavelength. Standard LuxCore RGB behavior stays intact
	// for all other textures and when HERO is disabled.
	if (::GetMLHeroEnabled()) {
		const float waveLength = ::GetMLCurrentWaveLength();
		if ((waveLength >= 380.f) && (waveLength <= 780.f)) {
			const IrregularDataTexture *irregular =
				dynamic_cast<const IrregularDataTexture *>(&GetSigmaA());
			if (irregular) {
				const float value = Max(0.f, irregular->GetSpectralValue(waveLength));
				::MarkMLDispersionUsed();
				return Spectrum(value);
			}
		}
	}

	return GetSigmaA().GetSpectrumValue(hitPoint).Clamp();
}

Spectrum ClearVolume::SigmaS(const HitPoint &hitPoint) const {
	return Spectrum();
}

float ClearVolume::Scatter(const Ray &ray, const float u,
		const bool scatteredStart, Spectrum *connectionThroughput,
		Spectrum *connectionEmission) const {
	// Point where to evaluate the volume
	HitPoint hitPoint;
	hitPoint.Init();
	hitPoint.fixedDir = ray.d;
	hitPoint.p = ray.o;
	hitPoint.geometryN = hitPoint.interpolatedN = hitPoint.shadeN = Normal(-ray.d);
	hitPoint.passThroughEvent = u;
	
	const float distance = ray.maxt - ray.mint;
	Spectrum transmittance(1.f);

	const Spectrum sigma = SigmaT(hitPoint);
	if (!sigma.Black()) {
		const Spectrum tau = (distance * sigma).Clamp();
		transmittance = Exp(-tau);
	}

	// 15fp: snapshot only; no extra texture evaluation or transport changes.
	const bool audit15fp = MLHeroVolumeStateRef15fp().enabled;
	Spectrum before15fp;
	if (audit15fp) before15fp = *connectionThroughput;

	// Apply volume transmittance
	*connectionThroughput *= transmittance;
	if (audit15fp) {
		const float wave15fp = ::GetMLCurrentWaveLength();
		const bool spectral15fp = ::GetMLHeroEnabled() && wave15fp >= 380.f && wave15fp <= 780.f &&
			dynamic_cast<const IrregularDataTexture *>(&GetSigmaA());
		MLHeroVolumeRecord15fp(this, GetName().c_str(), static_cast<unsigned>(GetSigmaA().GetType()),
			spectral15fp, wave15fp, distance, sigma, transmittance, before15fp, *connectionThroughput);
        // 15fq BEGIN: bounded counterfactual only. Does not change SigmaA,
        // connectionThroughput, emission or the active HERO/dispersion state.
        if (MLHeroVolumeProbe15fq *probe15fq = MLHeroVolumeCandidateBegin15fq(this, spectral15fp)) {
            probe15fq->distance = distance;
            for (unsigned c = 0; c < 3; ++c) {
                probe15fq->rgbSigma[c] = sigma.c[c];
                probe15fq->rgbT[c] = transmittance.c[c];
            }
            probe15fq->count = Min(::GetMLHeroWavelengthCount(), 8u);
            const IrregularDataTexture *direct15fq =
                dynamic_cast<const IrregularDataTexture *>(&GetSigmaA());
            auto &auditState15fq = MLHeroVolumeCandidateStateRef15fq();
            // 15fr BEGIN: paired reference-color counterfactual. No live writes.
            auto &referenceState15fr = MLHeroVolumeReferenceStateRef15fr();
            MLHeroVolumeReferenceProbe15fr *reference15fr = nullptr;
            if (direct15fq) {
                ++referenceState15fr.unsupported; // Exact direct-spectrum route stays in 15fq.
            } else if (referenceState15fr.used < 128) {
                reference15fr = &referenceState15fr.probes[referenceState15fr.used++];
                reference15fr->bin = probe15fq->bin;
                reference15fr->sample = probe15fq->sample;
                reference15fr->count = probe15fq->count;
                const ColorDepthTexture *colorDepth15fr =
                    dynamic_cast<const ColorDepthTexture *>(&GetSigmaA());
                if (colorDepth15fr) {
                    reference15fr->source = 1;
                    reference15fr->referenceDistance = -colorDepth15fr->GetD();
                    const Spectrum input15fr = colorDepth15fr->GetKt().GetSpectrumValue(hitPoint).Clamp(1e-9f, 1.f);
                    for (unsigned c = 0; c < 3; ++c) reference15fr->rgb[c] = input15fr.c[c];
                } else {
                    // The exporter discarded source color/depth: 0.1 is ONLY a
                    // diagnostic hypothesis, not inferred metadata or live policy.
                    reference15fr->source = 2;
                    reference15fr->referenceDistance = .1;
                    for (unsigned c = 0; c < 3; ++c) {
                        double refT15fr = 0.;
                        if (!MLHeroVolumeBeerCandidate15fq(sigma.c[c], .1, &refT15fr))
                            refT15fr = std::numeric_limits<double>::quiet_NaN();
                        reference15fr->rgb[c] = refT15fr;
                    }
                }
            }
            // 15fr END

            for (unsigned lane = 0; lane < probe15fq->count; ++lane) {
                const float lambda15fq = ::GetMLHeroWaveLengthAt(lane);
                probe15fq->waves[lane] = lambda15fq;
                double sigma15fq = 0., t15fq = 0.;
                bool ok15fq = std::isfinite(lambda15fq) && lambda15fq >= 380.f && lambda15fq <= 780.f;
                if (ok15fq && direct15fq) {
                    const float value15fq = direct15fq->GetSpectralValue(lambda15fq);
                    ok15fq = std::isfinite(value15fq);
                    sigma15fq = ok15fq ? Max(0.f, value15fq) : 0.;
                } else if (ok15fq) {
                    ok15fq = MLHeroVolumeSigmaCandidate15fq(probe15fq->rgbSigma, lambda15fq, &sigma15fq);
                }
                ok15fq = ok15fq && MLHeroVolumeBeerCandidate15fq(sigma15fq, distance, &t15fq);
                probe15fq->sigma[lane] = sigma15fq;
                probe15fq->t[lane] = t15fq;
                probe15fq->ok[lane] = ok15fq ? 1u : 0u;
                ++auditState15fq.laneChecks;
                if (ok15fq) { ++probe15fq->valid; ++auditState15fq.laneValid; }
                else ++auditState15fq.laneInvalid;
                // 15fr BEGIN: evaluate the second model at this same TLS lane.
                if (reference15fr) {
                    reference15fr->wave[lane] = lambda15fq;
                    double refT15fr = 0., refSigma15fr = 0., segmentT15fr = 0.;
                    bool valid15fr = MLHeroVolumeReferenceT15fr(reference15fr->rgb, lambda15fq, &refT15fr) &&
                        MLHeroVolumeReferenceSigma15fr(refT15fr, reference15fr->referenceDistance, &refSigma15fr) &&
                        MLHeroVolumeBeerCandidate15fq(refSigma15fr, distance, &segmentT15fr);
                    reference15fr->referenceT[lane] = refT15fr;
                    reference15fr->sigma[lane] = refSigma15fr;
                    reference15fr->t[lane] = segmentT15fr;
                    reference15fr->valid[lane] = valid15fr ? 1u : 0u;
                    if (valid15fr) ++referenceState15fr.valid;
                    else ++referenceState15fr.invalid;
                }
                // 15fr END

            }
        }
        // 15fq END

	}

	// Apply volume emission
	if (volumeEmissionTex)
		*connectionEmission += *connectionThroughput * distance * volumeEmissionTex->GetSpectrumValue(hitPoint).Clamp();
	
	return -1.f;
}

Spectrum ClearVolume::Albedo(const HitPoint &hitPoint) const {
	throw runtime_error("Internal error: called ClearVolume::Albedo()");
}

Spectrum ClearVolume::Evaluate(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
		float *directPdfW, float *reversePdfW) const {
	throw runtime_error("Internal error: called ClearVolume::Evaluate()");
}

Spectrum ClearVolume::Sample(const HitPoint &hitPoint,
		const Vector &localFixedDir, Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const {
	throw runtime_error("Internal error: called ClearVolume::Sample()");
}

void ClearVolume::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	throw runtime_error("Internal error: called ClearVolume::Pdf()");
}

void ClearVolume::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Volume::AddReferencedTextures(referencedTexs);

	GetSigmaA().AddReferencedTextures(referencedTexs);
}

void ClearVolume::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Volume::UpdateTextureReferences(oldTex, newTex);

	updtex(sigmaA, oldTex, newTex);
}

PropertiesUPtr ClearVolume::ToProperties() const {
	PropertiesUPtr props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.volumes." + name + ".type")("clear"));
	props->Set(Property("scene.volumes." + name + ".absorption")(GetSigmaA().GetSDLValue()));
	props->Set(Volume::ToProperties());

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

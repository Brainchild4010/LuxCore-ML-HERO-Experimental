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

#ifndef _SLG_MLHERO_MIX_SPECTRAL15NM_H
#define _SLG_MLHERO_MIX_SPECTRAL15NM_H

#include "slg/bsdf/hitpoint.h"
#include "slg/materials/glossy2.h"
#include "slg/materials/velvet.h"
#include "slg/engines/mlhero_mix_closure_diagnostic15nq.h"
#include "slg/engines/mlhero_mix_direction_trace15nr.h"

namespace slg {

class Material;

enum MLHeroMixChildRoute15nm : u_int {
	MLHERO_MIX_CHILD_INACTIVE15NM = 0u,
	MLHERO_MIX_CHILD_SCALAR15NM = 1u,
	MLHERO_MIX_CHILD_METAL215NM = 2u,
	MLHERO_MIX_CHILD_GLOSSY215NM = 3u,
	MLHERO_MIX_CHILD_VELVET15NN = 4u
};

// Phase 15no classifies every fail-closed Mix bounce without changing the
// transport decision.  Preparation and per-lane evaluation are deliberately
// separate because the old 15nm counter combined both failure domains.
enum MLHeroMixPrepareFailure15no : u_int {
	MLHERO_MIX_PREPARE_OK15NO = 0u,
	MLHERO_MIX_PREPARE_INVALID_INPUT15NO,
	MLHERO_MIX_PREPARE_SPECULAR_EVENT15NO,
	MLHERO_MIX_PREPARE_INVALID_SAMPLE15NO,
	MLHERO_MIX_PREPARE_GEOMETRY_COS15NO,
	MLHERO_MIX_PREPARE_SIDE_EVENT15NO,
	MLHERO_MIX_PREPARE_CHILD_A15NO,
	MLHERO_MIX_PREPARE_CHILD_B15NO,
	MLHERO_MIX_PREPARE_RECONSTRUCTED_BLACK15NO,
	MLHERO_MIX_PREPARE_WRAPPER_INVALID15NO,
	MLHERO_MIX_PREPARE_CLOSURE_REJECT15NO,
	MLHERO_MIX_PREPARE_FAILURE_COUNT15NO
};

enum MLHeroMixChildFailure15no : u_int {
	MLHERO_MIX_CHILD_OK15NO = 0u,
	MLHERO_MIX_CHILD_INVALID_INPUT15NO,
	MLHERO_MIX_CHILD_NESTED_MIX15NO,
	MLHERO_MIX_CHILD_NATIVE_NONFINITE15NO,
	MLHERO_MIX_CHILD_GLOSSY2_PREPARE15NO,
	MLHERO_MIX_CHILD_VELVET_PREPARE15NO,
	MLHERO_MIX_CHILD_EVENT_MISMATCH15NO,
	MLHERO_MIX_CHILD_UNSUPPORTED_CHROMATIC15NO,
	MLHERO_MIX_CHILD_FAILURE_COUNT15NO
};

enum MLHeroMixEvaluationFailure15no : u_int {
	MLHERO_MIX_EVALUATE_OK15NO = 0u,
	MLHERO_MIX_EVALUATE_INVALID_INPUT15NO,
	MLHERO_MIX_EVALUATE_MATERIAL_MISMATCH15NO,
	MLHERO_MIX_EVALUATE_METAL2_REJECT15NO,
	MLHERO_MIX_EVALUATE_METAL2_NONSCALAR15NO,
	MLHERO_MIX_EVALUATE_GLOSSY2_REJECT15NO,
	MLHERO_MIX_EVALUATE_VELVET_REJECT15NO,
	MLHERO_MIX_EVALUATE_UNSUPPORTED_ROUTE15NO,
	MLHERO_MIX_EVALUATE_CHILD_VALUE15NO,
	MLHERO_MIX_EVALUATE_RESULT15NO,
	MLHERO_MIX_EVALUATE_FAILURE_COUNT15NO
};

struct MLHeroMixDiagnostic15no {
	u_int prepareFailure;
	u_int childFailure;
	u_int childIndex;
	u_int childType;
	u_int evaluationFailure;
	u_int evaluationChildIndex;
	u_int failedLane;
	float absDotLightDirNG;
	float absDotEyeDirNG;
	float sideTestNG;
	float sideTestIS;
	float wrapperFactor;
	float closureRelativeError;
	bool sampled;
	bool geometryGateWouldReject;
	bool sideEventGateWouldReject;
	// Phase 15nq is diagnostic-only. It decomposes a rejected native Mix
	// Sample and replays the endpoint arithmetic in double precision.
	u_int selectedChild15nq;
	u_int selectedChildType15nq;
	u_int childRoute15nq[2];
	float passThroughEvent15nq;
	float childWeight15nq[2];
	float childNative15nq[2][3];
	float childDirectPdf15nq[2];
	float childReversePdf15nq[2];
	double doubleReconstructed15nq[3];
	double nativeNumerator15nq[3];
	double inferredSelectedNumerator15nq[3];
	double evaluatedSelectedNumerator15nq[3];
	double doubleClosureRelativeError15nq;
	double selectedNumeratorRelativeError15nq;
	double comparableMixturePdf15nq;
	double nativePdfRelativeError15nq;
	bool analysisFinite15nq;
	bool doubleClosureAccepted15nq;
	bool captureMatched15nr;
	bool originalEvaluateFinite15nr;
	bool sampleVsOriginalPass15nr;
	bool originalVsRoundTripPass15nr;
	bool sampleVsRoundTripPass15nr;
	u_int originalEvent15nr;
	u_int capturedEvent15nr;
	u_int roundTripEvent15nr;
	u_int glossyBranch15nr;
	float childSamplePdf15nr;
	float originalEvaluatePdf15nr;
	float glossyWBase15nr;
	float glossyRoughness15nr;
	float glossyAnisotropy15nr;
	float originalFixedDir15nr[3];
	float originalSampledDir15nr[3];
	float roundTripFixedDir15nr[3];
	float roundTripSampledDir15nr[3];
	float childSampleNumerator15nr[3];
	float originalEvaluate15nr[3];
	float roundTripEvaluate15nr[3];
	double fixedDirectionMaxError15nr;
	double sampledDirectionMaxError15nr;
	double sampledDirectionDot15nr;
	double sampleVsOriginalRelativeError15nr;
	double originalVsRoundTripRelativeError15nr;
	double sampleVsRoundTripRelativeError15nr;
	double sampleVsOriginalPdfRelativeError15nr;
	// Phase 15ns production direction anchor audit.
	bool directionAnchorRequested15ns;
	bool directionCaptureMatched15ns;
	bool directionAnchorApplied15ns;

	MLHeroMixDiagnostic15no() :
		prepareFailure(MLHERO_MIX_PREPARE_OK15NO),
		childFailure(MLHERO_MIX_CHILD_OK15NO), childIndex(~0u),
		childType(~0u), evaluationFailure(MLHERO_MIX_EVALUATE_OK15NO),
		evaluationChildIndex(~0u), failedLane(~0u),
		absDotLightDirNG(0.f), absDotEyeDirNG(0.f), sideTestNG(0.f),
		sideTestIS(0.f), wrapperFactor(1.f), closureRelativeError(0.f),
		sampled(false), geometryGateWouldReject(false),
		sideEventGateWouldReject(false), selectedChild15nq(~0u),
		selectedChildType15nq(~0u), passThroughEvent15nq(0.f),
		doubleClosureRelativeError15nq(0.),
		selectedNumeratorRelativeError15nq(0.),
		comparableMixturePdf15nq(0.), nativePdfRelativeError15nq(0.),
		analysisFinite15nq(false), doubleClosureAccepted15nq(false),
		captureMatched15nr(false), originalEvaluateFinite15nr(false),
		sampleVsOriginalPass15nr(false), originalVsRoundTripPass15nr(false),
		sampleVsRoundTripPass15nr(false), originalEvent15nr(NONE),
		capturedEvent15nr(NONE), roundTripEvent15nr(NONE),
		glossyBranch15nr(2u), childSamplePdf15nr(0.f),
		originalEvaluatePdf15nr(0.f), glossyWBase15nr(0.f),
		glossyRoughness15nr(0.f), glossyAnisotropy15nr(0.f),
		fixedDirectionMaxError15nr(0.), sampledDirectionMaxError15nr(0.),
		sampledDirectionDot15nr(0.), sampleVsOriginalRelativeError15nr(0.),
		originalVsRoundTripRelativeError15nr(0.),
		sampleVsRoundTripRelativeError15nr(0.),
		sampleVsOriginalPdfRelativeError15nr(0.),
		directionAnchorRequested15ns(false),
		directionCaptureMatched15ns(false),
		directionAnchorApplied15ns(false) {
		for (u_int i = 0u; i < 2u; ++i) {
			childRoute15nq[i] = MLHERO_MIX_CHILD_INACTIVE15NM;
			childWeight15nq[i] = 0.f;
			childDirectPdf15nq[i] = childReversePdf15nq[i] = 0.f;
			for (u_int c = 0u; c < 3u; ++c)
				childNative15nq[i][c] = 0.f;
		}
		for (u_int c = 0u; c < 3u; ++c) {
			doubleReconstructed15nq[c] = nativeNumerator15nq[c] = 0.;
			inferredSelectedNumerator15nq[c] =
				evaluatedSelectedNumerator15nq[c] = 0.;
			originalFixedDir15nr[c] = originalSampledDir15nr[c] = 0.f;
			roundTripFixedDir15nr[c] = roundTripSampledDir15nr[c] = 0.f;
			childSampleNumerator15nr[c] = originalEvaluate15nr[c] =
				roundTripEvaluate15nr[c] = 0.f;
		}
	}
};

// Cached, texture-independent state for one child of a MixMaterial.  The
// HitPoint copy is intentional: MixMaterial applies each child's own bump map
// before evaluating it, so explicit wavelength evaluation must use the same
// child frame and texture coordinates.
struct MLHeroMixChildSpectral15nm {
	const Material *material;
	HitPoint hitPoint;
	luxrays::Vector localLightDir;
	luxrays::Vector localEyeDir;
	luxrays::Spectrum nativeValue;
	BSDFEvent event;
	float weight;
	float directPdfW;
	float reversePdfW;
	u_int route;
	MLHeroGlossy2SpectralEndpoint15nk glossy2;
	MLHeroVelvetSpectralEndpoint15nn velvet;

	MLHeroMixChildSpectral15nm() : material(nullptr), localLightDir(),
		localEyeDir(), nativeValue(0.f), event(NONE), weight(0.f),
		directPdfW(0.f), reversePdfW(0.f),
		route(MLHERO_MIX_CHILD_INACTIVE15NM), glossy2(), velvet() { }
};

// Phase 15nm/15nn supports exact Metal2, Glossy2 and Velvet wavelength
// evaluation plus genuinely achromatic child contributions.
struct MLHeroMixSpectralEndpoint15nm {
	MLHeroMixChildSpectral15nm child[2];
	luxrays::Spectrum nativeReconstructed;
	BSDFEvent combinedEvent;
	float wrapperFactor;
	float nativePdfW;
	float closureRelativeError;
	bool sampled;
	bool valid;

	MLHeroMixSpectralEndpoint15nm() : nativeReconstructed(0.f),
		combinedEvent(NONE), wrapperFactor(1.f), nativePdfW(1.f),
		closureRelativeError(0.f), sampled(false), valid(false) { }
};

}

#endif // _SLG_MLHERO_MIX_SPECTRAL15NM_H

#line 2 "materialdefs_funcs_roughglass.cl"

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

//------------------------------------------------------------------------------
// RoughGlass material
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE void RoughGlassMaterial_Albedo(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
    const float3 albedo = WHITE;

	EvalStack_PushFloat3(albedo);
}

OPENCL_FORCE_INLINE void RoughGlassMaterial_GetInteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void RoughGlassMaterial_GetExteriorVolume(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void RoughGlassMaterial_GetPassThroughTransparency(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

OPENCL_FORCE_INLINE void RoughGlassMaterial_GetEmittedRadiance(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	DefaultMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
}

//------------------------------------------------------------------------------
// ML Fine RoughGlass helpers - shared by OpenCL and CUDA
//------------------------------------------------------------------------------
// ML CUDA Fine RoughGlass HERO - Test L
//
// Port of CPU Fine RoughGlass Test C:
// - isotropic GGX VNDF sampling
// - smooth Near-Delta transition
// - <= 0.0005 UI roughness: exact Glass/HERO
// - >= 0.0050 UI roughness: full RoughGlass/VNDF
// - smoothstep stochastic BSDF mixture in between
//------------------------------------------------------------------------------

OPENCL_FORCE_INLINE bool RoughGlassMLCudaUseVNDF(const float anisotropy) {
	return fabs(anisotropy) < 1e-6f;
}

OPENCL_FORCE_INLINE float RoughGlassMLCudaNearDeltaWeight(
		const float u, const float v) {
	const float uiRoughness = sqrt(fmax(0.f, u * v));
	const float deltaStart = 0.0005f;
	const float deltaEnd = 0.005f;

	if (uiRoughness <= deltaStart)
		return 1.f;
	if (uiRoughness >= deltaEnd)
		return 0.f;

	const float x = (deltaEnd - uiRoughness) /
			(deltaEnd - deltaStart);
	return x * x * (3.f - 2.f * x);
}

OPENCL_FORCE_INLINE float RoughGlassMLCudaGGXG1(
		const float alpha, const float3 v) {
	const float cosTheta = fabs(v.z);
	if (cosTheta <= 0.f)
		return 0.f;

	const float sinTheta2 = fmax(0.f, 1.f - cosTheta * cosTheta);
	if (sinTheta2 <= 0.f)
		return 1.f;

	const float tanTheta2 = sinTheta2 / (cosTheta * cosTheta);
	const float root = sqrt(1.f + alpha * alpha * tanTheta2);
	return 2.f / (1.f + root);
}

OPENCL_FORCE_INLINE float RoughGlassMLCudaVisibleNormalPdf(
		const float roughness, const float3 fixedDir,
		const float3 wh) {
	const float cosTheta = fabs(fixedDir.z);
	if (cosTheta <= DEFAULT_COS_EPSILON_STATIC)
		return 0.f;

	const float alpha = sqrt(fmax(roughness, 1e-18f));
	const float D = SchlickDistribution_D(roughness, wh, 0.f);
	const float G1 = RoughGlassMLCudaGGXG1(alpha, fixedDir);
	return D * G1 * fabs(dot(fixedDir, wh)) / cosTheta;
}

OPENCL_FORCE_INLINE void RoughGlassMLCudaSampleVisibleNormal(
		const float roughness, const float3 fixedDir,
		const float u0, const float u1,
		float3 *wh, float *d, float *pdf) {
	const float alpha = sqrt(fmax(roughness, 1e-18f));

	float3 V = (fixedDir.z >= 0.f) ? fixedDir : -fixedDir;
	const float3 Vh = normalize(MAKE_FLOAT3(
			alpha * V.x, alpha * V.y, V.z));

	const float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	const float3 T1 = (lensq > 0.f) ?
			MAKE_FLOAT3(-Vh.y, Vh.x, 0.f) / sqrt(lensq) :
			MAKE_FLOAT3(1.f, 0.f, 0.f);
	const float3 T2 = cross(Vh, T1);

	const float r = sqrt(clamp(u0, 0.f, 1.f));
	const float phi = 2.f * M_PI_F * u1;
	const float t1 = r * cos(phi);
	float t2 = r * sin(phi);
	const float s = .5f * (1.f + Vh.z);
	t2 = (1.f - s) * sqrt(fmax(0.f, 1.f - t1 * t1)) +
			s * t2;

	const float z = sqrt(fmax(0.f, 1.f - t1 * t1 - t2 * t2));
	const float3 Nh = t1 * T1 + t2 * T2 + z * Vh;

	*wh = normalize(MAKE_FLOAT3(
			alpha * Nh.x, alpha * Nh.y, fmax(0.f, Nh.z)));
	if (wh->z < 0.f)
		*wh = -*wh;

	*d = SchlickDistribution_D(roughness, *wh, 0.f);
	*pdf = RoughGlassMLCudaVisibleNormalPdf(
			roughness, fixedDir, *wh);
}

#if defined(LUXRAYS_CUDA_DEVICE)
OPENCL_FORCE_INLINE float3 RoughGlassMLCudaSampleDelta(
		__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		const float3 fixedDir, float3 *sampledDir,
		const float u0, const float passThroughEvent,
		const float mlDispersionWaveLength,
		const float3 kr, const float3 kt,
		const float nc, const float ntBase, const float cauchyB,
		float *pdfW, BSDFEvent *event
		MATERIALS_PARAM_DECL) {
	float3 transLocalSampledDir;
	const float3 trans = GlassMaterial_EvalSpecularTransmission(
			hitPoint, fixedDir, u0,
			kt, nc, ntBase, cauchyB, mlDispersionWaveLength,
			&transLocalSampledDir);

	const float localFilmThickness =
			(material->roughglass.filmThicknessTexIndex != NULL_INDEX) ?
			Texture_GetFloatValue(
					material->roughglass.filmThicknessTexIndex,
					hitPoint TEXTURES_PARAM) :
			0.f;
	const float localFilmIor =
			(localFilmThickness > 0.f &&
					material->roughglass.filmIorTexIndex != NULL_INDEX) ?
			Texture_GetFloatValue(
					material->roughglass.filmIorTexIndex,
					hitPoint TEXTURES_PARAM) :
			1.f;

	float3 reflLocalSampledDir;
	const float3 refl = GlassMaterial_EvalSpecularReflection(
			hitPoint, fixedDir,
			kr, nc, ntBase, cauchyB, mlDispersionWaveLength,
			&reflLocalSampledDir, localFilmThickness, localFilmIor);

	float threshold;
	if (!Spectrum_IsBlack(refl)) {
		if (!Spectrum_IsBlack(trans)) {
			const float reflFilter = Spectrum_Filter(refl);
			const float transFilter = Spectrum_Filter(trans);
			threshold = transFilter / (reflFilter + transFilter);
			threshold = clamp(threshold, .25f, .75f);
		} else
			threshold = 0.f;
	} else {
		if (!Spectrum_IsBlack(trans))
			threshold = 1.f;
		else {
			*pdfW = 0.f;
			*event = NONE;
			*sampledDir = BLACK;
			return BLACK;
		}
	}

	float3 result;
	if (passThroughEvent < threshold) {
		*sampledDir = transLocalSampledDir;
		*event = SPECULAR | TRANSMIT;
		*pdfW = threshold;
		result = trans;
	} else {
		*sampledDir = reflLocalSampledDir;
		*event = SPECULAR | REFLECT;
		*pdfW = 1.f - threshold;
		result = refl;
	}

	return result / *pdfW;
}


#endif

OPENCL_FORCE_INLINE void RoughGlassMaterial_Evaluate(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float3 lightDir, eyeDir;
	EvalStack_PopFloat3(eyeDir);
	EvalStack_PopFloat3(lightDir);

	const float3 ktVal = Texture_GetSpectrumValue(material->roughglass.ktTexIndex, hitPoint TEXTURES_PARAM);
	const float3 krVal = Texture_GetSpectrumValue(material->roughglass.krTexIndex, hitPoint TEXTURES_PARAM);
	const float3 kt = Spectrum_Clamp(ktVal);
	const float3 kr = Spectrum_Clamp(krVal);

	const bool isKtBlack = Spectrum_IsBlack(kt);
	const bool isKrBlack = Spectrum_IsBlack(kr);
	if (isKtBlack && isKrBlack) {
		MATERIAL_EVALUATE_RETURN_BLACK;
	}
	
	const float nc = ExtractExteriorIors(hitPoint, material->roughglass.exteriorIorTexIndex TEXTURES_PARAM);
	const float nt = ExtractInteriorIors(hitPoint, material->roughglass.interiorIorTexIndex TEXTURES_PARAM);
	const float ntc = nt / nc;

	const float nuVal = Texture_GetFloatValue(material->roughglass.nuTexIndex, hitPoint TEXTURES_PARAM);
	const float nvVal = Texture_GetFloatValue(material->roughglass.nvTexIndex, hitPoint TEXTURES_PARAM);
	const float u = clamp(nuVal, 1e-9f, 1.f);
	const float v = clamp(nvVal, 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;
	const float deltaWeight = material->roughglass.fineRoughGlass ? RoughGlassMLCudaNearDeltaWeight(u, v) : 0.f;
	const float roughWeight = 1.f - deltaWeight;

	// Exact delta Glass has no finite-solid-angle Evaluate contribution.
	if (roughWeight <= 0.f) {
		MATERIAL_EVALUATE_RETURN_BLACK;
	}

	float directPdfW;
	BSDFEvent event;
	float3 result;
	const float threshold = isKrBlack ? 1.f : (isKtBlack ? 0.f : .5f);
	if (lightDir.z * eyeDir.z < 0.f) {
		// Transmit

		const bool entering = (CosTheta(lightDir) > 0.f);
		const float eta = entering ? (nc / nt) : ntc;

		float3 wh = eta * lightDir + eyeDir;
		if (wh.z < 0.f)
			wh = -wh;

		const float lengthSquared = dot(wh, wh);
		if (!(lengthSquared > 0.f)) {
			MATERIAL_EVALUATE_RETURN_BLACK;
		}
		wh /= sqrt(lengthSquared);
		const float cosThetaI = fabs(CosTheta(eyeDir));
		const float cosThetaIH = fabs(dot(eyeDir, wh));
		const float cosThetaOH = dot(lightDir, wh);

		const float D = SchlickDistribution_D(roughness, wh, anisotropy);
		const float G = SchlickDistribution_G(roughness, lightDir, eyeDir);
		const float specPdf = (material->roughglass.fineRoughGlass && RoughGlassMLCudaUseVNDF(anisotropy)) ?
				RoughGlassMLCudaVisibleNormalPdf(
						roughness, eyeDir, wh) :
				SchlickDistribution_Pdf(roughness, wh, anisotropy);
		const float F = FresnelCauchy_Evaluate(ntc, cosThetaOH);

		directPdfW = roughWeight * threshold * specPdf *
				(fabs(cosThetaOH) * eta * eta) / lengthSquared;

		//if (reversePdfW)
		//	*reversePdfW = threshold * specPdf * cosThetaIH / lengthSquared;

		result = roughWeight * (fabs(cosThetaOH) * cosThetaIH * D *
			G / (cosThetaI * lengthSquared)) *
			kt * (1.f - F);

        event = GLOSSY | TRANSMIT;
	} else {
		// Reflect
		const float cosThetaO = fabs(CosTheta(lightDir));
		const float cosThetaI = fabs(CosTheta(eyeDir));
		if (cosThetaO == 0.f || cosThetaI == 0.f) {
			MATERIAL_EVALUATE_RETURN_BLACK;
		}
		float3 wh = lightDir + eyeDir;
		if (all(isequal(wh, BLACK))) {
			MATERIAL_EVALUATE_RETURN_BLACK;
		}
		wh = normalize(wh);
		if (wh.z < 0.f)
			wh = -wh;

		float cosThetaH = dot(eyeDir, wh);
		const float D = SchlickDistribution_D(roughness, wh, anisotropy);
		const float G = SchlickDistribution_G(roughness, lightDir, eyeDir);
		const float specPdf = (material->roughglass.fineRoughGlass && RoughGlassMLCudaUseVNDF(anisotropy)) ?
				RoughGlassMLCudaVisibleNormalPdf(
						roughness, eyeDir, wh) :
				SchlickDistribution_Pdf(roughness, wh, anisotropy);
		const float F = FresnelCauchy_Evaluate(ntc, cosThetaH);

		directPdfW = roughWeight * (1.f - threshold) * specPdf /
				(4.f * fabs(dot(lightDir, wh)));

		//if (reversePdfW)
		//	*reversePdfW = (1.f - threshold) * specPdf / (4.f * fabs(dot(lightDir, wh));

		result = roughWeight * (D * G / (4.f * cosThetaI)) * kr * F;
		
		const float localFilmThickness = (material->roughglass.filmThicknessTexIndex != NULL_INDEX) 
										 ? Texture_GetFloatValue(material->roughglass.filmThicknessTexIndex, hitPoint TEXTURES_PARAM) : 0.f;
		if (localFilmThickness > 0.f) {
			const float localFilmIor = (material->roughglass.filmIorTexIndex != NULL_INDEX) 
									   ? Texture_GetFloatValue(material->roughglass.filmIorTexIndex, hitPoint TEXTURES_PARAM) : 1.f;
			result *= CalcFilmColor(eyeDir, localFilmThickness, localFilmIor);
		}

        event = GLOSSY | REFLECT;
	}

	EvalStack_PushFloat3(result);
	EvalStack_PushBSDFEvent(event);
	EvalStack_PushFloat(directPdfW);
}

OPENCL_FORCE_INLINE void RoughGlassMaterial_Sample(__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		__global float *evalStack, uint *evalStackOffset
		MATERIALS_PARAM_DECL) {
	float u0, u1, passThroughEvent;
	EvalStack_PopFloat(passThroughEvent);
	EvalStack_PopFloat(u1);
	EvalStack_PopFloat(u0);
	float3 fixedDir;
	EvalStack_PopFloat3(fixedDir);

	if (fabs(fixedDir.z) < DEFAULT_COS_EPSILON_STATIC) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	const float3 ktVal = Texture_GetSpectrumValue(material->roughglass.ktTexIndex, hitPoint TEXTURES_PARAM);
	const float3 krVal = Texture_GetSpectrumValue(material->roughglass.krTexIndex, hitPoint TEXTURES_PARAM);
	const float3 kt = Spectrum_Clamp(ktVal);
	const float3 kr = Spectrum_Clamp(krVal);

	const bool isKtBlack = Spectrum_IsBlack(kt);
	const bool isKrBlack = Spectrum_IsBlack(kr);
	if (isKtBlack && isKrBlack) {
		MATERIAL_SAMPLE_RETURN_BLACK;
	}

	const float nuVal = Texture_GetFloatValue(material->roughglass.nuTexIndex, hitPoint TEXTURES_PARAM);
	const float nvVal = Texture_GetFloatValue(material->roughglass.nvTexIndex, hitPoint TEXTURES_PARAM);
	const float u = clamp(nuVal, 1e-9f, 1.f);
	const float v = clamp(nvVal, 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	float3 wh;
	float d, specPdf;
	SchlickDistribution_SampleH(roughness, anisotropy, u0, u1, &wh, &d, &specPdf);
	if (wh.z < 0.f)
		wh = -wh;
	const float cosThetaOH = dot(fixedDir, wh);

	const float nc = ExtractExteriorIors(hitPoint, material->roughglass.exteriorIorTexIndex TEXTURES_PARAM);
	const float nt = ExtractInteriorIors(hitPoint, material->roughglass.interiorIorTexIndex TEXTURES_PARAM);
	const float ntc = nt / nc;

	const float coso = fabs(fixedDir.z);

	// Decide to transmit or reflect
	float threshold;
	if (!isKrBlack) {
		if (!isKtBlack)
			threshold = .5f;
		else
			threshold = 0.f;
	} else {
		if (!isKtBlack)
			threshold = 1.f;
		else {
			MATERIAL_SAMPLE_RETURN_BLACK;
		}
	}

	float pdfW;
	BSDFEvent event;
	float3 sampledDir;
	float3 result;
	if (passThroughEvent < threshold) {
		// Transmit

		const bool entering = (CosTheta(fixedDir) > 0.f);
		const float eta = entering ? (nc / nt) : ntc;
		const float eta2 = eta * eta;
		const float sinThetaIH2 = eta2 * fmax(0.f, 1.f - cosThetaOH * cosThetaOH);
		if (sinThetaIH2 >= 1.f) {
			MATERIAL_SAMPLE_RETURN_BLACK;
		}
		float cosThetaIH = sqrt(1.f - sinThetaIH2);
		if (entering)
			cosThetaIH = -cosThetaIH;
		const float length = eta * cosThetaOH + cosThetaIH;
		sampledDir = length * wh - eta * fixedDir;

		const float lengthSquared = length * length;
		pdfW = specPdf * fabs(cosThetaIH) / lengthSquared;
		if (pdfW <= 0.f) {
			MATERIAL_SAMPLE_RETURN_BLACK;
		}

		const float cosi = fabs(sampledDir.z);

		const float G = SchlickDistribution_G(roughness, fixedDir, sampledDir);
		float factor = (d / specPdf) * G * fabs(cosThetaOH) / threshold;

		//if (!hitPoint.fromLight) {
			const float F = FresnelCauchy_Evaluate(ntc, cosThetaIH);
			result = (factor / coso) * kt * (1.f - F);
		//} else {
		//	const Spectrum F = FresnelCauchy_Evaluate(ntc, cosThetaOH);
		//	result = (factor / cosi) * kt * (Spectrum(1.f) - F);
		//}

		pdfW *= threshold;
		event = GLOSSY | TRANSMIT;
	} else {
		// Reflect
		pdfW = specPdf / (4.f * fabs(cosThetaOH));
		if (pdfW <= 0.f) {
			MATERIAL_SAMPLE_RETURN_BLACK;
		}

		sampledDir = 2.f * cosThetaOH * wh - fixedDir;

		const float cosi = fabs(sampledDir.z);
		if ((cosi < DEFAULT_COS_EPSILON_STATIC) || (fixedDir.z * sampledDir.z < 0.f)) {
			MATERIAL_SAMPLE_RETURN_BLACK;
		}

		const float G = SchlickDistribution_G(roughness, fixedDir, sampledDir);
		float factor = (d / specPdf) * G * fabs(cosThetaOH) / (1.f - threshold);

		const float F = FresnelCauchy_Evaluate(ntc, cosThetaOH);
		//factor /= (!hitPoint.fromLight) ? coso : cosi;
		factor /= coso;
		result = factor * F * kr;
		
		const float localFilmThickness = (material->roughglass.filmThicknessTexIndex != NULL_INDEX) 
										 ? Texture_GetFloatValue(material->roughglass.filmThicknessTexIndex, hitPoint TEXTURES_PARAM) : 0.f;
		if (localFilmThickness > 0.f) {
			const float localFilmIor = (material->roughglass.filmIorTexIndex != NULL_INDEX) 
									   ? Texture_GetFloatValue(material->roughglass.filmIorTexIndex, hitPoint TEXTURES_PARAM) : 1.f;
			result *= CalcFilmColor(fixedDir, localFilmThickness, localFilmIor);
		}

		pdfW *= (1.f - threshold);
		event = GLOSSY | REFLECT;
	}

	EvalStack_PushFloat3(result);
	EvalStack_PushFloat3(sampledDir);
	EvalStack_PushFloat(pdfW);
	EvalStack_PushBSDFEvent(event);
}


//------------------------------------------------------------------------------
// ML CUDA HERO RoughGlass - Test L direct sampler
//------------------------------------------------------------------------------
#if defined(LUXRAYS_CUDA_DEVICE)
OPENCL_FORCE_INLINE float3 RoughGlassMaterial_SampleMLCuda(
		__global const Material* restrict material,
		__global const HitPoint *hitPoint,
		const float3 fixedDir, float3 *sampledDir,
		const float u0, const float u1,
		const float passThroughEvent,
		const float mlDispersionWaveLength,
		float *pdfW, BSDFEvent *event
		MATERIALS_PARAM_DECL) {
	if (fabs(fixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return BLACK;

	const float3 kt = Spectrum_Clamp(Texture_GetSpectrumValue(
			material->roughglass.ktTexIndex, hitPoint TEXTURES_PARAM));
	const float3 kr = Spectrum_Clamp(Texture_GetSpectrumValue(
			material->roughglass.krTexIndex, hitPoint TEXTURES_PARAM));

	const bool isKtBlack = Spectrum_IsBlack(kt);
	const bool isKrBlack = Spectrum_IsBlack(kr);
	if (isKtBlack && isKrBlack)
		return BLACK;

	const float u = clamp(Texture_GetFloatValue(
			material->roughglass.nuTexIndex, hitPoint TEXTURES_PARAM), 1e-9f, 1.f);
	const float v = clamp(Texture_GetFloatValue(
			material->roughglass.nvTexIndex, hitPoint TEXTURES_PARAM), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) :
			(u2 > 0.f ? (v2 / u2 - 1.f) : 0.f);
	const float roughness = u * v;
	const float deltaWeight = material->roughglass.fineRoughGlass ? RoughGlassMLCudaNearDeltaWeight(u, v) : 0.f;
	const float roughWeight = 1.f - deltaWeight;

	const float nc = ExtractExteriorIors(
			hitPoint, material->roughglass.exteriorIorTexIndex TEXTURES_PARAM);
	const float ntBase = ExtractInteriorIors(
			hitPoint, material->roughglass.interiorIorTexIndex TEXTURES_PARAM);

	const float cauchyB = (material->roughglass.cauchyBTex != NULL_INDEX) ?
			Texture_GetFloatValue(material->roughglass.cauchyBTex,
					hitPoint TEXTURES_PARAM) : 0.f;

	const float nt = (cauchyB > 0.f) ?
			GlassMaterial_WaveLength2IOR(mlDispersionWaveLength, ntBase, cauchyB) :
			ntBase;
	const float ntc = nt / nc;
	const float coso = fabs(fixedDir.z);

	// Smooth Near-Delta mixture. u0 selects the lobe; if RoughGlass is chosen,
	// remap u0 so VNDF still receives a uniform [0,1) variate.
	if ((deltaWeight >= 1.f) || (u0 < deltaWeight)) {
		float3 deltaResult = RoughGlassMLCudaSampleDelta(
				material, hitPoint,
				fixedDir, sampledDir,
				u0, passThroughEvent,
				mlDispersionWaveLength,
				kr, kt, nc, ntBase, cauchyB,
				pdfW, event
				MATERIALS_PARAM);

		if (!Spectrum_IsBlack(deltaResult))
			*pdfW *= fmax(deltaWeight, 1e-9f);

		return deltaResult;
	}

	const float roughU0 = (roughWeight > 0.f) ?
			clamp((u0 - deltaWeight) / roughWeight, 0.f, 1.f) :
			0.f;

	float3 wh;
	float d, specPdf;
	if (material->roughglass.fineRoughGlass && RoughGlassMLCudaUseVNDF(anisotropy))
		RoughGlassMLCudaSampleVisibleNormal(
				roughness, fixedDir,
				roughU0, u1, &wh, &d, &specPdf);
	else
		SchlickDistribution_SampleH(
				roughness, anisotropy,
				roughU0, u1, &wh, &d, &specPdf);

	if (wh.z < 0.f)
		wh = -wh;

	const float cosThetaOH = dot(fixedDir, wh);

	float threshold;
	if (!isKrBlack) {
		if (!isKtBlack)
			threshold = .5f;
		else
			threshold = 0.f;
	} else {
		if (!isKtBlack)
			threshold = 1.f;
		else
			return BLACK;
	}

	float3 result;
	if (passThroughEvent < threshold) {
		const bool entering = (CosTheta(fixedDir) > 0.f);
		const float eta = entering ? (nc / nt) : ntc;
		const float eta2 = eta * eta;
		const float sinThetaIH2 = eta2 *
				fmax(0.f, 1.f - cosThetaOH * cosThetaOH);
		if (sinThetaIH2 >= 1.f)
			return BLACK;

		float cosThetaIH = sqrt(1.f - sinThetaIH2);
		if (entering)
			cosThetaIH = -cosThetaIH;

		const float length = eta * cosThetaOH + cosThetaIH;
		*sampledDir = length * wh - eta * fixedDir;

		const float lengthSquared = length * length;
		*pdfW = specPdf * fabs(cosThetaIH) / lengthSquared;
		if (*pdfW <= 0.f)
			return BLACK;

		const float cosi = fabs(sampledDir->z);
		const float G = SchlickDistribution_G(roughness, fixedDir, *sampledDir);
		const float factor = (d / specPdf) * G *
				fabs(cosThetaOH) / threshold;

		// CUDA PathOCL local HERO route is an eye-path sampler.
		// GPU HitPoint has no fromLight flag, so use the eye-path branch.
		const float F = FresnelCauchy_Evaluate(ntc, cosThetaIH);
		result = (factor / coso) * kt * (1.f - F);

		*pdfW *= threshold;
		*event = GLOSSY | TRANSMIT;
	} else {
		*pdfW = specPdf / (4.f * fabs(cosThetaOH));
		if (*pdfW <= 0.f)
			return BLACK;

		*sampledDir = 2.f * cosThetaOH * wh - fixedDir;
		const float cosi = fabs(sampledDir->z);
		if ((cosi < DEFAULT_COS_EPSILON_STATIC) ||
				(fixedDir.z * sampledDir->z < 0.f))
			return BLACK;

		const float G = SchlickDistribution_G(roughness, fixedDir, *sampledDir);
		float factor = (d / specPdf) * G *
				fabs(cosThetaOH) / (1.f - threshold);

		const float F = FresnelCauchy_Evaluate(ntc, cosThetaOH);
		// CUDA PathOCL local HERO route is an eye-path sampler.
		factor /= coso;
		result = factor * F * kr;

		const float localFilmThickness =
				(material->roughglass.filmThicknessTexIndex != NULL_INDEX) ?
				Texture_GetFloatValue(material->roughglass.filmThicknessTexIndex,
						hitPoint TEXTURES_PARAM) : 0.f;
		if (localFilmThickness > 0.f) {
			const float localFilmIor =
					(material->roughglass.filmIorTexIndex != NULL_INDEX) ?
					Texture_GetFloatValue(material->roughglass.filmIorTexIndex,
							hitPoint TEXTURES_PARAM) : 1.f;
			result *= CalcFilmColor(fixedDir, localFilmThickness, localFilmIor);
		}

		*pdfW *= (1.f - threshold);
		*event = GLOSSY | REFLECT;
	}

	// Continuous lobe was selected with roughWeight probability.
	*pdfW *= fmax(roughWeight, 1e-9f);

	return result;
}
#endif

//------------------------------------------------------------------------------
// Material specific EvalOp
//------------------------------------------------------------------------------

OPENCL_FORCE_NOT_INLINE void RoughGlassMaterial_EvalOp(
		__global const Material* restrict material,
		const MaterialEvalOpType evalType,
		__global float *evalStack,
		uint *evalStackOffset,
		__global const HitPoint *hitPoint
		MATERIALS_PARAM_DECL) {
	switch (evalType) {
		case EVAL_ALBEDO:
			RoughGlassMaterial_Albedo(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_INTERIOR_VOLUME:
			RoughGlassMaterial_GetInteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EXTERIOR_VOLUME:
			RoughGlassMaterial_GetExteriorVolume(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_EMITTED_RADIANCE:
			RoughGlassMaterial_GetEmittedRadiance(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_GET_PASS_TROUGH_TRANSPARENCY:
			RoughGlassMaterial_GetPassThroughTransparency(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_EVALUATE:
			RoughGlassMaterial_Evaluate(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		case EVAL_SAMPLE:
			RoughGlassMaterial_Sample(material, hitPoint, evalStack, evalStackOffset MATERIALS_PARAM);
			break;
		default:
			// Something wrong here
			break;
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

#include "slg/engines/mlhero_metal_pdf15lb.h"
#include "slg/engines/mlhero_schlick_audit15la.h"
#include "slg/engines/mlhero_metal_direction15ky.h"
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

#include "slg/materials/metal2.h"

#include <atomic>
#include <cstdio>

// ML HERO shared spectral packet state (implemented in glass.cpp).
bool GetMLHeroEnabled();
u_int GetMLHeroWavelengthCount();
float GetMLHeroWaveLengthAt(const u_int lane);
float GetMLHeroSampleWeightAt(const u_int lane);

namespace {
}

using namespace std;
using namespace luxrays;
using namespace slg;

// 15ky BEGIN: one numerical Schlick geometry convention for CPU HERO Metal2.
static Vector MLHeroMetalHalf15ky(const Vector&a,const Vector&b){
    if(!GetMLHeroEnabled())return Normalize(a+b);
    using namespace slg::mlhero15kk;
    auto h=Unit(Unit(V(a))+Unit(V(b)));
    if(h.z<0.)h=h*(-1.);
    return Vector(float(h.x),float(h.y),float(h.z));
}
static float MLHeroMetalCos15ky(const Vector&a,const Vector&b){
    if(!GetMLHeroEnabled())return Dot(a,Normalize(a+b));
    using namespace slg::mlhero15kk;
    // For unit reflected directions |cos(theta_h)|=|a+b|/2.
    // Symmetric and stable even when a and b nearly cancel at grazing.
    const auto sum=Unit(V(a))+Unit(V(b));
    const float c=float(.5*std::sqrt(sum.Norm2()));
    return a.z<0.f?-c:c;
}
static float MLHeroMetalD15ky(float roughness,const Vector&h,float anisotropy){
    if(!GetMLHeroEnabled())return SchlickDistribution_D(roughness,h,anisotropy);
    const double x=h.x,y=h.y,z=h.z,r=roughness,n2=x*x+y*y+z*z;
    if(!(n2>0.)||!(r>0.)||!std::isfinite(n2))return 0.f;
    const double denominator=(x*x+y*y+r*z*z)/n2;
    return float((r/denominator)/denominator*SchlickDistribution_SchlickA(h,anisotropy)*INV_PI);
}
// 15lb: native/HERO-off probabilities are preserved exactly.
static float MLHeroMetalHalfPdf15lb(float roughness,const Vector&h,float anisotropy){
    if(!GetMLHeroEnabled())return MLHeroMetalD15ky(roughness,h,anisotropy);
    return float(slg::mlhero15lb::HalfPdf(roughness,anisotropy,h.x,h.y,h.z));
}
// 15ky END

//------------------------------------------------------------------------------
// Metal2 material
//
// LuxRender Metal2 material porting.
//------------------------------------------------------------------------------

Metal2Material::Metal2Material(
	TextureConstPtr frontTransp,
	TextureConstPtr backTransp,
	TextureConstPtr emitted,
	TextureConstPtr bump,
	TextureConstPtr nn,
	TextureConstPtr kk,
	TextureConstPtr u,
	TextureConstPtr v
) :
	Material(frontTransp, backTransp, emitted, bump),
	fresnelTex(nullptr),
	n(nn),
	k(kk),
	nu(u),
	nv(v)
{
	glossiness = ComputeGlossiness(nu, nv);
}

Metal2Material::Metal2Material(
	TextureConstPtr frontTransp,
	TextureConstPtr backTransp,
	TextureConstPtr emitted,
	TextureConstPtr bump,
	FresnelTextureConstPtr ft,
	TextureConstPtr u,
	TextureConstPtr v)
	:
	Material(frontTransp, backTransp, emitted, bump),
	fresnelTex(ft),
	n(nullptr),
	k(nullptr),
	nu(u),
	nv(v)
{
	glossiness = ComputeGlossiness(nu, nv);
}

// 15ky BEGIN: cache material inputs once, evaluate the emitted direction pair.
// No second material sample, no extra texture/Fresnel query, no PDF clamp.
static Spectrum MLHeroMetalSample15ky(const Metal2Material&m,const HitPoint&hp,
        const Frame*frame,const Vector&input,Vector*out,float u0,float u1,
        float*pdf,BSDFEvent*event,float*cosine){
    const Vector fixed=frame?slg::mlhero15ky::Local(*frame,input):slg::mlhero15kp::Normalize(input);
    slg::mlhero15la::Attempt audit15la(hp.fromLight,m.GetID());
    if(fabsf(fixed.z)<DEFAULT_COS_EPSILON_STATIC)return Spectrum();
    const float u=Clamp(m.GetNu()->GetFloatValue(hp),1e-9f,1.f);
    const float v=Clamp(m.GetNv()->GetFloatValue(hp),1e-9f,1.f);
    const float u2=u*u,v2=v*v;
    const float anis=(u2<v2)?1.f-u2/v2:(u2>0.f?v2/u2-1.f:0.f);
    const float roughness=u*v;
    using namespace slg::mlhero15kk;
    const auto h=slg::mlhero15ky::SampleHalf(roughness,anis,u0,u1);
    const auto f=Unit(V(fixed));
    const auto generated=Unit(h*(2.*slg::mlhero15kk::Dot(f,h))-f);
    if(audit15la.Active())audit15la.Draw(u,v,roughness,anis,u0,u1,f,h,
        MLHeroMetalD15ky(roughness,Vector(float(h.x),float(h.y),float(h.z)),anis));
    Vector emitted;
    if(frame){
        const slg::mlhero15kl::Frame precise(*frame);const auto world=precise.ToWorld(generated);
        emitted=slg::mlhero15kp::Normalize(Vector(float(world.x),float(world.y),float(world.z)));
    }else emitted=slg::mlhero15kp::Normalize(Vector(float(generated.x),float(generated.y),float(generated.z)));
    const Vector local=frame?slg::mlhero15ky::Local(*frame,emitted):emitted;
    if(fabsf(local.z)<DEFAULT_COS_EPSILON_STATIC||fixed.z*local.z<0.f)return Spectrum();
    Vector pairHalf=MLHeroMetalHalf15ky(fixed,local);
    // Keep the native sampled Fresnel orientation: sampled h is on the upper side.
    if(pairHalf.z<0.f)pairHalf=-pairHalf;
    const float cosWH=MLHeroMetalCos15ky(fixed,local);
    const float d=MLHeroMetalD15ky(roughness,pairHalf,anis);
    const float halfPdf=MLHeroMetalHalfPdf15lb(roughness,pairHalf,anis);
    const float probability=halfPdf/(4.f*fabsf(cosWH));
    if(!(probability>0.f)||!std::isfinite(probability)){audit15la.PdfRejected();return Spectrum();}
    audit15la.Accept(V(pairHalf),probability);
    const float G=SchlickDistribution_G(roughness,fixed,local);
    Spectrum F;
    if(m.GetFresnel())F=m.GetFresnel()->Evaluate(hp,cosWH);
    else {
        const Spectrum eta=m.GetN()->GetSpectrumValue(hp).Clamp(.001f);
        const Spectrum kk=m.GetK()->GetSpectrumValue(hp).Clamp(.001f);
        F=FresnelTexture::GeneralEvaluate(eta,kk,cosWH);
    }
    F.Clamp(0.f,1.f);
    // BRDF D and generated half-density differ. Use f*cos/pdf, without
    // conditioning on hemisphere acceptance or silently changing the BRDF.
    const float factor=float((double(d)/halfPdf)*G*fabsf(cosWH)/
        (hp.fromLight?fabsf(local.z):fabsf(fixed.z)));
    *out=emitted;*pdf=probability;*event=GLOSSY|REFLECT;
    if(cosine)*cosine=fabsf(local.z);
    return factor*F;
}
Spectrum Metal2Material::SampleMLHeroWorld15ky(const HitPoint&hp,const Frame&frame,
        Vector*out,float u0,float u1,float passThroughEvent,float*pdf,
        float*cosine,BSDFEvent*event)const{
    return MLHeroMetalSample15ky(*this,hp,&frame,hp.fixedDir,out,u0,u1,pdf,event,cosine);
}
// 15ky END

bool Metal2Material::EvaluateFresnelAtWaveLength(const HitPoint &hitPoint, const float cosi,
		const float waveLength, Spectrum *fresnel) const {
	if (!fresnelTex || !fresnel)
		return false;

	Spectrum eta, kk;
	if (!fresnelTex->GetNKAtWaveLength(hitPoint, waveLength, &eta, &kk))
		return false;

	*fresnel = FresnelTexture::GeneralEvaluate(eta, kk, cosi);
	fresnel->Clamp(0.f, 1.f);
	return true;
}

bool Metal2Material::EvaluateAtWaveLength(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		const float waveLength, Spectrum *value) const {
	if (!fresnelTex || !value)
		return false;

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	const Vector wh(MLHeroMetalHalf15ky(localLightDir,localEyeDir));
	const float cosWH = MLHeroMetalCos15ky(localLightDir,localEyeDir);
	if (fabsf(cosWH) < DEFAULT_COS_EPSILON_STATIC)
		return false;

	Spectrum F;
	if (!EvaluateFresnelAtWaveLength(hitPoint, cosWH, waveLength, &F))
		return false;

	const float G = SchlickDistribution_G(roughness, localLightDir, localEyeDir);
	*value = (MLHeroMetalD15ky(roughness, wh, anisotropy) * G /
			(4.f * fabsf(localEyeDir.z))) * F;
	return true;
}

bool Metal2Material::EvaluateAtWaveLengthDebug(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		const float waveLength,
		Spectrum *eta, Spectrum *kk, Spectrum *fresnel,
		float *microfacetFactor, float *cosWHOut, Spectrum *value) const {
	if (!fresnelTex || !eta || !kk || !fresnel ||
			!microfacetFactor || !cosWHOut || !value)
		return false;

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) :
		u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	const Vector wh(MLHeroMetalHalf15ky(localLightDir,localEyeDir));
	const float cosWH = MLHeroMetalCos15ky(localLightDir,localEyeDir);
	if (fabsf(cosWH) < DEFAULT_COS_EPSILON_STATIC)
		return false;

	if (!fresnelTex->GetNKAtWaveLength(hitPoint, waveLength, eta, kk))
		return false;

	*fresnel = FresnelTexture::GeneralEvaluate(*eta, *kk, cosWH);
	fresnel->Clamp(0.f, 1.f);

	const float G = SchlickDistribution_G(roughness, localLightDir, localEyeDir);
	const float factor = MLHeroMetalD15ky(roughness, wh, anisotropy) * G /
		(4.f * fabsf(localEyeDir.z));

	*microfacetFactor = factor;
	*cosWHOut = cosWH;
	*value = factor * (*fresnel);
	return true;
}

bool Metal2Material::EvaluateSampleAtWaveLength(const HitPoint &hitPoint,
		const Vector &localFixedDir, const Vector &localSampledDir,
		const float waveLength, Spectrum *sampleMultiplier) const {
	if (!fresnelTex || !sampleMultiplier)
		return false;

	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return false;

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	const Vector wh(MLHeroMetalHalf15ky(localFixedDir,localSampledDir));
	const float cosWH = MLHeroMetalCos15ky(localFixedDir,localSampledDir);
	if (fabsf(cosWH) < DEFAULT_COS_EPSILON_STATIC)
		return false;

	const float specPdf = MLHeroMetalHalfPdf15lb(roughness, wh, anisotropy);
	if (specPdf <= 0.f)
		return false;

	const float d = MLHeroMetalD15ky(roughness, wh, anisotropy);
	const float G = SchlickDistribution_G(roughness, localFixedDir, localSampledDir);
	const float coso = fabsf(localFixedDir.z);
	const float cosi = fabsf(localSampledDir.z);
	if ((cosi < DEFAULT_COS_EPSILON_STATIC) || (localFixedDir.z * localSampledDir.z < 0.f))
		return false;

	Spectrum F;
	if (!EvaluateFresnelAtWaveLength(hitPoint, cosWH, waveLength, &F))
		return false;

	float factor = (d / specPdf) * G * fabsf(cosWH);
	if (!hitPoint.fromLight)
		factor /= coso;
	else
		factor /= cosi;

	*sampleMultiplier = factor * F;
	return true;
}

Spectrum Metal2Material::Albedo(const HitPoint &hitPoint) const {
	Spectrum F;
	if (fresnelTex)
		F = fresnelTex->Evaluate(hitPoint, 1.f);
	else {
		// For compatibility with the past
		const Spectrum etaVal = n->GetSpectrumValue(hitPoint).Clamp(.001f);
		const Spectrum kVal = k->GetSpectrumValue(hitPoint).Clamp(.001f);
		F = FresnelTexture::GeneralEvaluate(etaVal, kVal, 1.f);
	}
	F.Clamp(0.f, 1.f);
	
	return F;
}
	
Spectrum Metal2Material::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	const Vector wh(MLHeroMetalHalf15ky(localLightDir,localEyeDir));
	const float cosWH = MLHeroMetalCos15ky(localLightDir,localEyeDir);

	if (directPdfW)
		*directPdfW = MLHeroMetalHalfPdf15lb(roughness, wh, anisotropy) / (4.f * (GetMLHeroEnabled()?fabsf(cosWH):cosWH));

	if (reversePdfW)
		*reversePdfW = MLHeroMetalHalfPdf15lb(roughness, wh, anisotropy) / (4.f * (GetMLHeroEnabled()?fabsf(cosWH):cosWH));

	Spectrum F;
	if (fresnelTex)
		F = fresnelTex->Evaluate(hitPoint, cosWH);
	else {
		// For compatibility with the past
		const Spectrum etaVal = n->GetSpectrumValue(hitPoint).Clamp(.001f);
		const Spectrum kVal = k->GetSpectrumValue(hitPoint).Clamp(.001f);
		F = FresnelTexture::GeneralEvaluate(etaVal, kVal, cosWH);
	}
	F.Clamp(0.f, 1.f);

	const float G = SchlickDistribution_G(roughness, localLightDir, localEyeDir);

	*event = GLOSSY | REFLECT;
	return (MLHeroMetalD15ky(roughness, wh, anisotropy) * G / (4.f * fabsf(localEyeDir.z))) * F;
}

Spectrum Metal2Material::Sample(const HitPoint &hitPoint,
	const Vector &localFixedDir, Vector *localSampledDir,
	const float u0, const float u1, const float passThroughEvent,
	float *pdfW, BSDFEvent *event) const {
    if(GetMLHeroEnabled())return MLHeroMetalSample15ky(*this,hitPoint,nullptr,localFixedDir,
        localSampledDir,u0,u1,pdfW,event,nullptr);
	if (fabsf(localFixedDir.z) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	Vector wh;
	float d, specPdf;
	SchlickDistribution_SampleH(roughness, anisotropy, u0, u1, &wh, &d, &specPdf);
	const float cosWH = Dot(localFixedDir, wh);
	*localSampledDir = 2.f * cosWH * wh - localFixedDir;

	const float coso = fabsf(localFixedDir.z);
	const float cosi = fabsf(localSampledDir->z);
	if ((cosi < DEFAULT_COS_EPSILON_STATIC) || (localFixedDir.z * localSampledDir->z < 0.f))
		return Spectrum();

	*pdfW = specPdf / (4.f * fabsf(cosWH));
	if (*pdfW <= 0.f)
		return Spectrum();

	const float G = SchlickDistribution_G(roughness, localFixedDir, *localSampledDir);

	Spectrum F;
	if (fresnelTex)
		F = fresnelTex->Evaluate(hitPoint, cosWH);
	else {
		// For compatibility with the past
		const Spectrum etaVal = n->GetSpectrumValue(hitPoint).Clamp(.001f);
		const Spectrum kVal = k->GetSpectrumValue(hitPoint).Clamp(.001f);
		F = FresnelTexture::GeneralEvaluate(etaVal, kVal, cosWH);
	}
	F.Clamp(0.f, 1.f);

	float factor = (d / specPdf) * G * fabsf(cosWH);
	if (!hitPoint.fromLight)
		factor /= coso;
	else
		factor /= cosi;

	*event = GLOSSY | REFLECT;

	return factor * F;
}

void Metal2Material::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	const Vector wh(MLHeroMetalHalf15ky(localLightDir,localEyeDir));

	if (directPdfW)
		*directPdfW = MLHeroMetalHalfPdf15lb(roughness, wh, anisotropy) / (4.f * fabsf(MLHeroMetalCos15ky(localLightDir,localEyeDir)));

	if (reversePdfW)
		*reversePdfW = MLHeroMetalHalfPdf15lb(roughness, wh, anisotropy) / (4.f * fabsf(MLHeroMetalCos15ky(localLightDir,localEyeDir)));
}

void Metal2Material::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	if (fresnelTex)
		fresnelTex->AddReferencedTextures(referencedTexs);
	if (n)
		n->AddReferencedTextures(referencedTexs);
	if (k)
		k->AddReferencedTextures(referencedTexs);

	nu->AddReferencedTextures(referencedTexs);
	nv->AddReferencedTextures(referencedTexs);
}

void Metal2Material::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	bool updateGlossiness = false;
	if (fresnelTex == static_cast<const FresnelTexture *>(std::addressof(oldTex)))
		fresnelTex = static_cast<const FresnelTexture *>(&newTex);
	if (n == &oldTex)
		n = &newTex;
	if (k == &oldTex)
		k = &newTex;
	if (nu == &oldTex) {
		nu = &newTex;
		updateGlossiness = true;
	}
	if (nv == &oldTex) {
		nv = &newTex;
		updateGlossiness = true;
	}
	
	if (updateGlossiness)
		glossiness = ComputeGlossiness(nu, nv);
}

PropertiesUPtr Metal2Material::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("metal2"));
	if (fresnelTex)
		props->Set(Property("scene.materials." + name + ".fresnel")(fresnelTex->GetSDLValue()));
	if (n)
		props->Set(Property("scene.materials." + name + ".n")(n->GetSDLValue()));
	if (k)
		props->Set(Property("scene.materials." + name + ".k")(k->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".uroughness")(nu->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".vroughness")(nv->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

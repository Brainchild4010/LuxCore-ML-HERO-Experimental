#include "slg/engines/mlhero_roughglass_pdf_audit15lc.h"
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

#include "slg/textures/fresnel/fresneltexture.h"
#include "slg/materials/roughglass.h"
#include "slg/materials/thinfilmcoating.h"
#include "slg/engines/mlhero_metal_pdf15lb.h"
#include "slg/usings.h"
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <cstdio>
#include "slg/engines/mlhero_rough_precision15ki.h"
#include "slg/engines/mlhero_rough_optics15kk.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// ML HERO dispersion state shared with GlassMaterial (CPU only, Test A)
//------------------------------------------------------------------------------

extern thread_local float mlDispersionWaveLength;
void MarkMLDispersionUsed();
extern thread_local bool mlHeroEnabled;
// 15gf: shared observational RoughGlass consistency audit (PATH and BIDIR).
extern thread_local bool mlHeroPathRoughConsistencyEnabled15gf;
extern void MLHeroPathRoughConsistencyReport15gf(bool fromLight, bool transmit,
        float nc, float nt, float samplePdf, float evalPdf, float pdfFunction,
        float sampleWeight, float evaluateWeight, float cosFixed, float cosSampled);
extern bool MLHeroRoughGlass15gfShouldAudit();
extern void MLHeroRoughGlass15gfReport(bool fromLight, bool transmit,
        float nc, float nt, float samplePdf, float evalPdf, float pdfFunction,
        float sampleWeight, float evaluateWeight, float cosFixed, float cosSampled);


// Phase 15gi: render-config driven CPU A/B switch. Per-thread state is set
// at the start of every eye/light render sample, so switching between renders
// does not require restarting Blender. Default remains legacy 50:50.
extern thread_local bool mlHeroRoughGlassFresnelImportance;
static inline bool RoughGlassML15ghEnabled() {
    return mlHeroEnabled && mlHeroRoughGlassFresnelImportance;
}

// Probability of selecting TRANSMISSION once the microfacet wh is sampled.
// The parameter is the signed cosine of the incoming direction against wh.
// FresnelTexture::CauchyEvaluate handles entering/exiting and TIR.
static float RoughGlassML15ghTransmitProbability(const float ntc,
        const Vector &incoming, const Vector &wh, const bool isKrBlack,
        const bool isKtBlack) {
    if (isKrBlack) return 1.f;
    if (isKtBlack) return 0.f;
    return Clamp(1.f - FresnelTexture::CauchyEvaluate(ntc,
            Dot(incoming, wh)), 0.f, 1.f);
}

static float RoughGlassMLGetWaveLength() {
	if ((mlDispersionWaveLength >= 380.f) && (mlDispersionWaveLength <= 780.f))
		return mlDispersionWaveLength;

	// Fallback only for safety. CPU Path/Bidir Goldstand normally initializes
	// the wavelength before tracing the sample.
	return 550.f;
}

static float RoughGlassMLWaveLength2IOR(const float waveLength,
		const float IOR, const float B) {
	const float A = IOR - B / Sqr(587.56f / 1000.f);
	return A + B / Sqr(waveLength / 1000.f);
}

struct RoughGlassMLSellmeierCoefficients {
	float B1, B2, B3;
	float C1, C2, C3;
};

static RoughGlassMLSellmeierCoefficients RoughGlassMLGetSellmeierCoefficients(
		const GlassSellmeierPreset preset) {
	switch (preset) {
		case GLASS_SELLMEIER_FUSED_SILICA:
			return {0.6961663f, 0.4079426f, 0.8974794f,
				0.00467914826f, 0.0135120631f, 97.9340025f};
		case GLASS_SELLMEIER_SF10:
			return {1.62153902f, 0.256287842f, 1.64447552f,
				0.0122241457f, 0.0595736775f, 147.468793f};
		case GLASS_SELLMEIER_SF11:
			return {1.73759695f, 0.313747346f, 1.89878101f,
				0.013188707f, 0.0623068142f, 155.23629f};
		case GLASS_SELLMEIER_N_BK7:
		default:
			return {1.03961212f, 0.231792344f, 1.01046945f,
				0.00600069867f, 0.0200179144f, 103.560653f};
	}
}

static float RoughGlassMLWaveLength2IORSellmeier(const float waveLength,
		const GlassSellmeierPreset preset) {
	const auto c = RoughGlassMLGetSellmeierCoefficients(preset);
	const float lambda = waveLength / 1000.f;
	const float lambda2 = lambda * lambda;
	const float n2 = 1.f +
			c.B1 * lambda2 / (lambda2 - c.C1) +
			c.B2 * lambda2 / (lambda2 - c.C2) +
			c.B3 * lambda2 / (lambda2 - c.C3);
	return sqrtf(Max(1.f, n2));
}

static float RoughGlassMLGetInteriorIOR(const HitPoint &hitPoint,
		TextureConstPtr interiorIor, TextureConstPtr cauchyB,
		const GlassDispersionModel dispersionModel,
		const GlassSellmeierPreset sellmeierPreset) {
	const float nt = ExtractInteriorIors(hitPoint, interiorIor);
	if (!mlHeroEnabled)
		return nt;

	const float waveLength = RoughGlassMLGetWaveLength();
	if (dispersionModel == GLASS_DISPERSION_SELLMEIER)
		return RoughGlassMLWaveLength2IORSellmeier(waveLength, sellmeierPreset);

	const float b = cauchyB ? cauchyB->GetFloatValue(hitPoint) : 0.f;
	if (b > 0.f)
		return RoughGlassMLWaveLength2IOR(waveLength, nt, b);

	return nt;
}


//------------------------------------------------------------------------------
// ML Fine RoughGlass - CPU Test B
//
// For isotropic RoughGlass, LuxCore's Schlick NDF is mathematically the same
// isotropic GGX/Trowbridge-Reitz NDF when:
//
//     alpha^2 = roughness = u * v
//
// The stock sampler draws half-vectors from the full NDF. At grazing angles
// this can generate many hidden/back-facing microfacets and therefore many
// rejected/black samples. Test B samples the *visible* GGX normal distribution
// (VNDF) for the isotropic case, while keeping the existing RoughGlass BRDF,
// HERO IOR, Fresnel, D and G terms unchanged.
//
// Anisotropic RoughGlass intentionally falls back to the original LuxCore
// Schlick sampler in this first test.
//------------------------------------------------------------------------------

static bool RoughGlassMLUseVNDF(const float anisotropy) {
	return fabsf(anisotropy) < 1e-6f;
}

static float RoughGlassMLGGXG1(const float alpha, const Vector &v) {
    if(mlHeroEnabled)return slg::mlhero15kk::G1(double(alpha)*alpha,slg::mlhero15kk::V(v));
	const float cosTheta = fabsf(v.z);
	if (cosTheta <= 0.f)
		return 0.f;

	const float sinTheta2 = Max(0.f, 1.f - cosTheta * cosTheta);
	if (sinTheta2 <= 0.f)
		return 1.f;

	const float tanTheta2 = sinTheta2 / (cosTheta * cosTheta);
	const float root = sqrtf(1.f + alpha * alpha * tanTheta2);
	return 2.f / (1.f + root);
}

// 15kj: normalized Schlick Z without subtracting two near-unit floats.
// This changes only HERO RoughGlass; it also handles anisotropic Schlick A.
static float RoughGlassMLDistributionD15kj(const float roughness,
        const Vector &wh,const float anisotropy) {
    if(!mlHeroEnabled)return SchlickDistribution_D(roughness,wh,anisotropy);
    const double x=wh.x,y=wh.y,z=wh.z,r=roughness;
    const double lengthSquared=x*x+y*y+z*z;
    if(!(lengthSquared>0.)||!(r>0.))return 0.f;
    const double denominator=(x*x+y*y+r*z*z)/lengthSquared;
    return (r/denominator)/denominator *
        SchlickDistribution_SchlickA(wh,anisotropy)*INV_PI;
}

// For unit directions the reflection Jacobian cancels the VNDF cosine.
// Evaluate this identity directly, avoiding the quotient of two separately
// rounded, nearly-zero dot products at grazing reflection.
static float RoughGlassMLReflectionPdf15kj(const float roughness,
        const Vector &fixed,const Vector &wh) {
    const float c=fabsf(fixed.z);
    if(c<=DEFAULT_COS_EPSILON_STATIC)return 0.f;
    return RoughGlassMLDistributionD15kj(roughness,wh,0.f)*
        RoughGlassMLGGXG1(sqrtf(Max(roughness,1e-18f)),fixed)/(4.f*c);
}

static float RoughGlassMLVisibleNormalPdf(const float roughness,
		const Vector &fixedDir, const Vector &wh) {
    if(mlHeroEnabled) {
        using namespace slg::mlhero15kk;
        const V fixed=Unit(V(fixedDir)),h=Unit(V(wh));
        const double c=std::fabs(fixed.z);
        if(c<=DEFAULT_COS_EPSILON_STATIC)return 0.f;
        return RoughGlassMLDistributionD15kj(roughness,wh,0.f)*G1(roughness,fixed)*std::fabs(Dot(fixed,h))/c;
    }
	const float cosTheta = fabsf(fixedDir.z);
	if (cosTheta <= DEFAULT_COS_EPSILON_STATIC)
		return 0.f;

	const float alpha = sqrtf(Max(roughness, 1e-18f));
	const float D = RoughGlassMLDistributionD15kj(roughness, wh, 0.f);
	const float G1 = RoughGlassMLGGXG1(alpha, fixedDir);
	return D * G1 * fabsf(Dot(fixedDir, wh)) / cosTheta;
}

// 15ld: SchlickDistribution_SampleH does not draw half-vectors from the
// Schlick BRDF distribution D.  Its radial inverse CDF and anisotropic
// azimuth warp induce the density below (shared with the corrected Metal2
// sampler).  Keep D exclusively as the BRDF term and use this value only as
// the sampling density.  Rejected directions retain their probability mass.
static float RoughGlassMLSchlickHalfPdf15ld(const float roughness,
		const float anisotropy, const Vector &wh) {
	return float(slg::mlhero15lb::HalfPdf(roughness, anisotropy,
			wh.x, wh.y, wh.z));
}

static float RoughGlassMLHalfPdf15ld(const float roughness,
		const float anisotropy, const Vector &fixedDir, const Vector &wh,
		const bool vndf) {
	return vndf ? RoughGlassMLVisibleNormalPdf(roughness, fixedDir, wh) :
			RoughGlassMLSchlickHalfPdf15ld(roughness, anisotropy, wh);
}

// 15ga: isotropic Schlick D is GGX with alpha^2 = roughness. Its
// masking/shadowing must use the matching Smith term. The legacy Schlick
// G approximation can return excessive grazing-angle reflected energy.
// This is a BSDF model correction, not a clamp of the sample or radiance.
// Anisotropic Schlick D is not this isotropic GGX model: retain its G here.
static float RoughGlassMLGeometry15ga(const float roughness,
        const float anisotropy, const Vector &fixed, const Vector &sampled) {
    if (!RoughGlassMLUseVNDF(anisotropy))
        return SchlickDistribution_G(roughness, fixed, sampled);
    const float alpha = sqrtf(Max(roughness, 1e-18f));
    return RoughGlassMLGGXG1(alpha, fixed) * RoughGlassMLGGXG1(alpha, sampled);
}

static void RoughGlassMLSampleVisibleNormal(const float roughness,
		const Vector &fixedDir, const float u0, const float u1,
		Vector *wh, float *d, float *pdf) {
	const float alpha = sqrtf(Max(roughness, 1e-18f));

	// Work in the upper hemisphere. RoughGlass later keeps the half-vector
	// convention wh.z >= 0 for both entering and exiting paths.
	Vector V = (fixedDir.z >= 0.f) ? fixedDir : -fixedDir;

	// Stretch the view direction.
	Vector Vh = Normalize(Vector(alpha * V.x, alpha * V.y, V.z));

	// Build an orthonormal basis around Vh.
	const float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	const Vector T1 = (lensq > 0.f) ?
			Vector(-Vh.y, Vh.x, 0.f) / sqrtf(lensq) :
			Vector(1.f, 0.f, 0.f);
	const Vector T2 = Cross(Vh, T1);

	// Sample a disk and warp it to the visible hemisphere.
	const float r = sqrtf(Clamp(u0, 0.f, 1.f));
	const float phi = 2.f * M_PI * u1;
	float t1 = r * cosf(phi);
	float t2 = r * sinf(phi);
	const float s = .5f * (1.f + Vh.z);
	t2 = (1.f - s) * sqrtf(Max(0.f, 1.f - t1 * t1)) + s * t2;

	const float z = sqrtf(Max(0.f, 1.f - t1 * t1 - t2 * t2));
	const Vector Nh = t1 * T1 + t2 * T2 + z * Vh;

	// Unstretch.
	*wh = Normalize(Vector(alpha * Nh.x, alpha * Nh.y, Max(0.f, Nh.z)));
	if (wh->z < 0.f)
		*wh = -*wh;

	*d = RoughGlassMLDistributionD15kj(roughness, *wh, 0.f);
	*pdf = RoughGlassMLVisibleNormalPdf(roughness, fixedDir, *wh);
}


static Vector MLHeroVector15kk(slg::mlhero15kk::V a){return Vector(a.x,a.y,a.z);}
struct MLHeroRoughEvaluation15kk {
    Spectrum value;
    float directPdf=0.f,reversePdf=0.f;
    BSDFEvent event=NONE;
    slg::mlhero15kk::Pair pair;
    double d=0.,directSpec=0.,directProbability=0.,directJacobian=0.;
};
static MLHeroRoughEvaluation15kk MLHeroEvaluateRough15kk(
        bool fromLight,const slg::mlhero15kk::V &light,const slg::mlhero15kk::V &eye,float nc,float nt,
        float roughness,float anisotropy,bool vndf,const Spectrum &kr,const Spectrum &kt) {
    using namespace slg::mlhero15kk;
    MLHeroRoughEvaluation15kk e;
    e.pair=MakePair(nc,nt,V(light),V(eye));const auto &p=e.pair;
    if(!p.valid||std::fabs(p.eye.z)<DEFAULT_COS_EPSILON_STATIC||
            std::fabs(p.light.z)<DEFAULT_COS_EPSILON_STATIC)return e;
    const Vector l=MLHeroVector15kk(p.light),o=MLHeroVector15kk(p.eye),h=MLHeroVector15kk(p.h);
    const bool rb=kr.Black(),tb=kt.Black();if(rb&&tb)return e;
    const double transmitProbability=Probability(p.transmission,rb,tb,RoughGlassML15ghEnabled());
    e.directProbability=p.transmit?transmitProbability:1.-transmitProbability;
    e.d=RoughGlassMLDistributionD15kj(roughness,h,anisotropy);
    const double g=RoughGlassMLGeometry15ga(roughness,anisotropy,l,o);
    const V fixed=fromLight?p.light:p.eye,generated=fromLight?p.eye:p.light;
    const double spec=vndf ?
        e.d*G1(roughness,fixed)*std::fabs(Dot(fixed,p.h))/std::fabs(fixed.z) :
        slg::mlhero15lb::HalfPdf(roughness,anisotropy,p.h.x,p.h.y,p.h.z);
    const double reverseSpec=vndf ?
        e.d*G1(roughness,generated)*std::fabs(Dot(generated,p.h))/std::fabs(generated.z) :
        slg::mlhero15lb::HalfPdf(roughness,anisotropy,p.h.x,p.h.y,p.h.z);
    e.directSpec=spec;
    if(p.transmit){
        const double cl=std::fabs(p.cosLight),ce=std::fabs(p.cosEye);
        e.directJacobian=(fromLight?ce:cl*p.eta*p.eta)/p.lengthSquared;
        const double reverseJacobian=(fromLight?cl*p.eta*p.eta:ce)/p.lengthSquared;
        e.directPdf=e.directProbability*spec*e.directJacobian;
        e.reversePdf=e.directProbability*reverseSpec*reverseJacobian;
        double weight=cl*ce*e.d*g*p.transmission/(std::fabs(p.eye.z)*p.lengthSquared);
        if(!fromLight)weight*=p.eta*p.eta;
        e.value=weight*kt;e.event=GLOSSY|TRANSMIT;
    }else{
        e.directJacobian=1./(4.*std::fabs(p.cosLight));
        e.directPdf=e.directProbability*(vndf?e.d*G1(roughness,fixed)/(4.*std::fabs(fixed.z)):spec*e.directJacobian);
        e.reversePdf=e.directProbability*(vndf?e.d*G1(roughness,generated)/(4.*std::fabs(generated.z)):reverseSpec*e.directJacobian);
        e.value=(e.d*g*(1.-p.transmission)/(4.*std::fabs(p.eye.z)))*kr;
        e.event=GLOSSY|REFLECT;
    }
    return e;
}
static MLHeroRoughEvaluation15kk MLHeroEvaluateRough15kk(
        bool fromLight,const Vector &light,const Vector &eye,float nc,float nt,
        float roughness,float anisotropy,bool vndf,const Spectrum &kr,const Spectrum &kt) {
    return MLHeroEvaluateRough15kk(fromLight,slg::mlhero15kk::V(light),slg::mlhero15kk::V(eye),
        nc,nt,roughness,anisotropy,vndf,kr,kt);
}


//------------------------------------------------------------------------------
// Phase 15gk transmission geometry fix audit.
// Counters represent pre-rejection candidates and are process lifetime (not per-render).
static std::atomic<unsigned long long> mlHero15gkTxSuccess[2]{};
static std::atomic<unsigned long long> mlHero15gkTxSameSide[2]{};

static inline void MLHero15gkAuditTransmission(const HitPoint &hitPoint,
        const Vector &fixedDir, const Vector &sampledDir,
        const float pdfW, const Spectrum &weight) {
    if (!mlHeroEnabled || !(pdfW > 0.f) || weight.Black())
        return;
    const unsigned int side = hitPoint.fromLight ? 1u : 0u;
    const unsigned long long total =
            mlHero15gkTxSuccess[side].fetch_add(1ull, std::memory_order_relaxed) + 1ull;
    const float sideProduct = fixedDir.z * sampledDir.z;
    const bool sameSide = !(sideProduct < 0.f);
    unsigned long long bad = mlHero15gkTxSameSide[side].load(std::memory_order_relaxed);
    if (sameSide)
        bad = mlHero15gkTxSameSide[side].fetch_add(1ull, std::memory_order_relaxed) + 1ull;

    // Limited output: first cases and powers of two; regular total checkpoints.
    const bool badCheckpoint = sameSide && ((bad <= 8ull) || ((bad & (bad - 1ull)) == 0ull));
    const bool totalCheckpoint = ((total % 1000000ull) == 0ull);
    if (badCheckpoint || totalCheckpoint) {
        fprintf(stderr,
                "ML HERO phase15gk TX_SIDE_AUDIT side=%s success=%llu sameSide=%llu "
                "sameSideCurrent=%u cosFixed=%.9g cosSampled=%.9g sideProduct=%.9g "
                "samplePdf=%.9g fresnelImportance=%u sameSideRejectEnabled=1\n",
                side ? "LIGHT" : "EYE", total, bad, sameSide ? 1u : 0u,
                fixedDir.z, sampledDir.z, sideProduct, pdfW,
                mlHeroRoughGlassFresnelImportance ? 1u : 0u);
    }
}

// Phase 15gk: the former near-delta/ideal-Glass mixture has been removed.
// Fine flag now only selects visible-normal sampling for isotropic RoughGlass.
// The legacy scene property key is retained for compatibility.
//------------------------------------------------------------------------------

// RoughGlass material
//
// LuxRender RoughGlass material porting.
//------------------------------------------------------------------------------

RoughGlassMaterial::RoughGlassMaterial(TextureConstPtr frontTransp, TextureConstPtr backTransp,
		TextureConstPtr emitted, TextureConstPtr bump,
		TextureConstPtr refl, TextureConstPtr trans,
		TextureConstPtr exteriorIorFact, TextureConstPtr interiorIorFact,
		TextureConstPtr B,
		const GlassDispersionModel dispModel,
		const GlassSellmeierPreset smPreset,
		const bool enableFineRoughGlass,
		TextureConstPtr u, TextureConstPtr v,
		TextureConstPtr filmThickness, TextureConstPtr filmIor) :
			Material(frontTransp, backTransp, emitted, bump), Kr(refl), Kt(trans),
			exteriorIor(exteriorIorFact), interiorIor(interiorIorFact), cauchyB(B),
			dispersionModel(dispModel), sellmeierPreset(smPreset),
			fineRoughGlass(enableFineRoughGlass),
			nu(u), nv(v), filmThickness(filmThickness), filmIor(filmIor) {
	glossiness = ComputeGlossiness(nu, nv);
}

bool RoughGlassMaterial::HasMLHeroDispersion(const HitPoint &hitPoint) const {
    return (dispersionModel == GLASS_DISPERSION_SELLMEIER) ||
        (cauchyB && (cauchyB->GetFloatValue(hitPoint) > 0.f));
}

Spectrum RoughGlassMaterial::Evaluate(const HitPoint &hitPoint,
	const Vector &localLightDir, const Vector &localEyeDir, BSDFEvent *event,
	float *directPdfW, float *reversePdfW) const {
    if(mlHeroEnabled){return EvaluateMLHeroPrecise15kl(hitPoint,slg::mlhero15kk::V(localLightDir),slg::mlhero15kk::V(localEyeDir),event,directPdfW,reversePdfW);}

	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);

	const bool isKtBlack = kt.Black();
	const bool isKrBlack = kr.Black();
	if (isKtBlack && isKrBlack)
		return Spectrum();

	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = RoughGlassMLGetInteriorIOR(hitPoint, interiorIor, cauchyB, dispersionModel, sellmeierPreset);
	const float ntc = nt / nc;

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;


    if(mlHeroEnabled) {
        const auto e=MLHeroEvaluateRough15kk(hitPoint.fromLight,localLightDir,localEyeDir,
            nc,nt,roughness,anisotropy,fineRoughGlass && RoughGlassMLUseVNDF(anisotropy),kr,kt);
        if(directPdfW)*directPdfW=e.directPdf;
        if(reversePdfW)*reversePdfW=e.reversePdf;
        *event=e.event;
        if((e.event & REFLECT) && !e.value.Black()) {
            const float thickness=filmThickness?filmThickness->GetFloatValue(hitPoint):0.f;
            if(thickness>0.f) {
                const float film=filmIor?filmIor->GetFloatValue(hitPoint):1.f;
                return e.value*CalcFilmColor(localEyeDir,thickness,film);
            }
        }
        return e.value;
    }

	const float legacyThreshold = isKrBlack ? 1.f : (isKtBlack ? 0.f : .5f);
	if (localLightDir.z * localEyeDir.z < 0.f) {
		// Transmit

		// Keep LuxCore's original light/eye-side convention here. For transmission
		// this eta is the inverse of the Sample() eta on an eye/radiance path;
		// that is intentional and makes the reconstructed half-vector reciprocal.
		const bool entering = (CosTheta(localLightDir) > 0.f);
		const float eta = entering ? (nc / nt) : ntc;

		Vector wh = eta * localLightDir + localEyeDir;
		if (wh.z < 0.f)
			wh = -wh;

		const float lengthSquared = wh.LengthSquared();
		if (!(lengthSquared > 0.f))
			return Spectrum();
		wh /= sqrtf(lengthSquared);
        const float threshold = RoughGlassML15ghEnabled() ?
                RoughGlassML15ghTransmitProbability(ntc,
                    hitPoint.fromLight ? localLightDir : localEyeDir,
                    wh, isKrBlack, isKtBlack) : legacyThreshold;
        const float reverseThreshold = RoughGlassML15ghEnabled() ?
                RoughGlassML15ghTransmitProbability(ntc,
                    hitPoint.fromLight ? localEyeDir : localLightDir,
                    wh, isKrBlack, isKtBlack) : legacyThreshold;
		const float cosThetaI = fabsf(CosTheta(localEyeDir));
		const float cosThetaIH = AbsDot(localEyeDir, wh);
		const float cosThetaOH = Dot(localLightDir, wh);

		const float D = RoughGlassMLDistributionD15kj(roughness, wh, anisotropy);
		const float G = RoughGlassMLGeometry15ga(roughness, anisotropy, localLightDir, localEyeDir);
		const float directSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localLightDir : localEyeDir, wh) :
				RoughGlassMLDistributionD15kj(roughness, wh, anisotropy);
		const float reverseSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localEyeDir : localLightDir, wh) :
				RoughGlassMLDistributionD15kj(roughness, wh, anisotropy);
		const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaOH);

		if (directPdfW)
			*directPdfW = threshold * directSpecPdf * (hitPoint.fromLight ? fabsf(cosThetaIH) : (fabsf(cosThetaOH) * eta * eta)) / lengthSquared;

		if (reversePdfW)
			*reversePdfW = reverseThreshold * reverseSpecPdf * (hitPoint.fromLight ? (fabsf(cosThetaOH) * eta * eta) : fabsf(cosThetaIH)) / lengthSquared;

		Spectrum result = (fabsf(cosThetaOH) * cosThetaIH * D *
			G / (cosThetaI * lengthSquared)) *
			kt * (1.f - F);

		// ML HERO phase 15cv: eye/radiance transport needs the reciprocal
		// solid-angle compression factor at a refractive microfacet interface.
		// Sample() already follows this radiance convention. Evaluate() did not,
		// which made EYE_TRANSMIT disagree with Sample()/Pdf while LIGHT_TRANSMIT
		// stayed consistent. Keep normal LuxCore untouched when HERO is disabled.
		if (mlHeroEnabled && !hitPoint.fromLight)
			result *= eta * eta;

		*event = GLOSSY | TRANSMIT;

		return result;
	} else {
		// Reflect
		const float cosThetaO = fabsf(CosTheta(localLightDir));
		const float cosThetaI = fabsf(CosTheta(localEyeDir));
		if (cosThetaO == 0.f || cosThetaI == 0.f)
			return Spectrum();
		Vector wh = localLightDir + localEyeDir;
		if (wh == Vector(0.f))
			return Spectrum();
		wh = Normalize(wh);
		if (wh.z < 0.f)
			wh = -wh;
        const float threshold = RoughGlassML15ghEnabled() ?
                RoughGlassML15ghTransmitProbability(ntc,
                    hitPoint.fromLight ? localLightDir : localEyeDir,
                    wh, isKrBlack, isKtBlack) : legacyThreshold;
        const float reverseThreshold = RoughGlassML15ghEnabled() ?
                RoughGlassML15ghTransmitProbability(ntc,
                    hitPoint.fromLight ? localEyeDir : localLightDir,
                    wh, isKrBlack, isKtBlack) : legacyThreshold;

		float cosThetaH = Dot(localEyeDir, wh);
		const float D = RoughGlassMLDistributionD15kj(roughness, wh, anisotropy);
		const float G = RoughGlassMLGeometry15ga(roughness, anisotropy, localLightDir, localEyeDir);
		const float directSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localLightDir : localEyeDir, wh) :
				RoughGlassMLDistributionD15kj(roughness, wh, anisotropy);
		const float reverseSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localEyeDir : localLightDir, wh) :
				RoughGlassMLDistributionD15kj(roughness, wh, anisotropy);
		const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaH);

		if (directPdfW)
            *directPdfW = (mlHeroEnabled && fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
                (1.f-threshold)*RoughGlassMLReflectionPdf15kj(roughness,
                    hitPoint.fromLight ? localLightDir : localEyeDir,wh) :
                (1.f-threshold)*directSpecPdf/(4.f*AbsDot(localLightDir,wh));

		if (reversePdfW)
            *reversePdfW = (mlHeroEnabled && fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
                (1.f-reverseThreshold)*RoughGlassMLReflectionPdf15kj(roughness,
                    hitPoint.fromLight ? localEyeDir : localLightDir,wh) :
                (1.f-reverseThreshold)*reverseSpecPdf/(4.f*AbsDot(localLightDir,wh));

		const Spectrum result = (D * G / (4.f * cosThetaI)) * kr * F;

		*event = GLOSSY | REFLECT;
		
		const float localFilmThickness = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
		if (localFilmThickness > 0.f) {
			const float localFilmIor = filmIor ? filmIor->GetFloatValue(hitPoint) : 1.f;
			return result * CalcFilmColor(localEyeDir, localFilmThickness, localFilmIor);
		}
		return result;
	}
}

Spectrum RoughGlassMaterial::EvaluateMLHeroPrecise15kl(const HitPoint &hitPoint,
        const slg::mlhero15kk::V &localLightDir,const slg::mlhero15kk::V &localEyeDir,
        BSDFEvent *event,float *directPdfW,float *reversePdfW) const {
    if(!mlHeroEnabled){return Evaluate(hitPoint,MLHeroVector15kk(localLightDir),MLHeroVector15kk(localEyeDir),event,directPdfW,reversePdfW);}

	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);

	const bool isKtBlack = kt.Black();
	const bool isKrBlack = kr.Black();
	if (isKtBlack && isKrBlack)
		return Spectrum();

	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = RoughGlassMLGetInteriorIOR(hitPoint, interiorIor, cauchyB, dispersionModel, sellmeierPreset);
	const float ntc = nt / nc;

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;



        const auto e=MLHeroEvaluateRough15kk(hitPoint.fromLight,localLightDir,localEyeDir,
            nc,nt,roughness,anisotropy,fineRoughGlass && RoughGlassMLUseVNDF(anisotropy),kr,kt);
        if(directPdfW)*directPdfW=e.directPdf;
        if(reversePdfW)*reversePdfW=e.reversePdf;
        *event=e.event;
        if((e.event & REFLECT) && !e.value.Black()) {
            const float thickness=filmThickness?filmThickness->GetFloatValue(hitPoint):0.f;
            if(thickness>0.f) {
                const float film=filmIor?filmIor->GetFloatValue(hitPoint):1.f;
                return e.value*CalcFilmColor(MLHeroVector15kk(localEyeDir),thickness,film);
            }
        }
        return e.value;
    
}


Spectrum RoughGlassMaterial::Sample(const HitPoint &hitPoint,
		const Vector &localFixedDir, Vector *localSampledDir,
		const float u0, const float u1, const float passThroughEvent,
		float *pdfW, BSDFEvent *event) const {
    if(mlHeroEnabled) {
        slg::mlhero15kk::V sampled;
        Spectrum r=SampleMLHeroPrecise15kl(hitPoint,slg::mlhero15kk::V(localFixedDir),&sampled,
            u0,u1,passThroughEvent,pdfW,event);
        if(!r.Black())*localSampledDir=MLHeroVector15kk(sampled);return r;
    }

    if ((mlHeroEnabled?std::fabs(slg::mlhero15kk::Unit(slg::mlhero15kk::V(localFixedDir)).z):fabsf(localFixedDir.z)) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);

	const bool isKtBlack = kt.Black();
	const bool isKrBlack = kr.Black();
	if (isKtBlack && isKrBlack)
		return Spectrum();

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = RoughGlassMLGetInteriorIOR(hitPoint, interiorIor, cauchyB, dispersionModel, sellmeierPreset);
	const float ntc = nt / nc;

	// 15gk: choose a microfacet normal directly; no stochastic delta mixing.
	const float roughU0 = u0;

    const Vector samplingFixed15kk=mlHeroEnabled?
        MLHeroVector15kk(slg::mlhero15kk::Unit(slg::mlhero15kk::V(localFixedDir))):localFixedDir;
	Vector wh;
	float d, specPdf;
	if (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy))
		RoughGlassMLSampleVisibleNormal(roughness, samplingFixed15kk,
				roughU0, u1, &wh, &d, &specPdf);
	else
		SchlickDistribution_SampleH(roughness, anisotropy,
				roughU0, u1, &wh, &d, &specPdf);
	if (wh.z < 0.f)
		wh = -wh;
    // The legacy NDF sampler reports D as its PDF.  Preserve the stable BRDF
    // D, but use the actual inverse-CDF density for the sampled half-vector.
    if(mlHeroEnabled && !(fineRoughGlass && RoughGlassMLUseVNDF(anisotropy))) {
        d=RoughGlassMLDistributionD15kj(roughness,wh,anisotropy);
        specPdf=RoughGlassMLSchlickHalfPdf15ld(roughness,anisotropy,wh);
    }
	const float cosThetaOH = Dot(localFixedDir, wh);

    const auto optics15kk=slg::mlhero15kk::MakeInterface(nc,nt,
        slg::mlhero15kk::V(localFixedDir),slg::mlhero15kk::V(wh));
    if(mlHeroEnabled && !optics15kk.valid)return Spectrum();
    if(mlHeroEnabled) {
        d=RoughGlassMLDistributionD15kj(roughness,wh,anisotropy);
        specPdf=RoughGlassMLHalfPdf15ld(roughness,anisotropy,
            localFixedDir,wh,fineRoughGlass && RoughGlassMLUseVNDF(anisotropy));
    }
    const float coso=mlHeroEnabled?std::fabs(optics15kk.fixed.z):fabsf(localFixedDir.z);


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
		else
			return Spectrum();
	}

    if(mlHeroEnabled)
        threshold=slg::mlhero15kk::Probability(optics15kk.transmission,isKrBlack,isKtBlack,RoughGlassML15ghEnabled());
    else if (RoughGlassML15ghEnabled())
        threshold = RoughGlassML15ghTransmitProbability(ntc, localFixedDir,
                wh, isKrBlack, isKtBlack);

	Spectrum result;
    if(mlHeroEnabled) {
        using namespace slg::mlhero15kk;
        const bool transmit=passThroughEvent<threshold;
        if(transmit && optics15kk.tir)return Spectrum();
        const V generated=transmit?Refract(optics15kk):Reflect(optics15kk);
        *localSampledDir=MLHeroVector15kk(generated);
        const double ci=std::fabs(generated.z),co=std::fabs(optics15kk.fixed.z);
        if(ci<DEFAULT_COS_EPSILON_STATIC ||
                (transmit?(optics15kk.fixed.z*generated.z>=0.):(optics15kk.fixed.z*generated.z<0.)))return Spectrum();
        const double probability=transmit?threshold:1.f-threshold;
        if(!(probability>0.)||!(specPdf>0.f))return Spectrum();
        const Vector normalizedFixed=MLHeroVector15kk(optics15kk.fixed);
        const float g=RoughGlassMLGeometry15ga(roughness,anisotropy,normalizedFixed,*localSampledDir);
        const double factor=(double(d)/specPdf)*g*std::fabs(optics15kk.cosine)/probability;
        if(transmit) {
            const double ctSigned=optics15kk.fixed.z>0.?-optics15kk.cosTransmitted:optics15kk.cosTransmitted;
            const double length=optics15kk.eta*optics15kk.cosine+ctSigned;
            *pdfW=probability*specPdf*optics15kk.cosTransmitted/(length*length);
            result=(factor*optics15kk.transmission/(hitPoint.fromLight?ci:co))*kt;
            MLHero15gkAuditTransmission(hitPoint,localFixedDir,*localSampledDir,*pdfW,result);
            *event=GLOSSY|TRANSMIT;
        }else{
            *pdfW=(fineRoughGlass && RoughGlassMLUseVNDF(anisotropy))?
                probability*d*G1(roughness,optics15kk.fixed)/(4.*co):
                probability*specPdf/(4.*std::fabs(optics15kk.cosine));
            result=(factor*(1.-optics15kk.transmission)/(hitPoint.fromLight?ci:co))*kr;
            *event=GLOSSY|REFLECT;
            const float thickness=filmThickness?filmThickness->GetFloatValue(hitPoint):0.f;
            if(thickness>0.f) {
                const float film=filmIor?filmIor->GetFloatValue(hitPoint):1.f;
                result*=CalcFilmColor(localFixedDir,thickness,film);
            }
        }
        if(!(*pdfW>0.f)||!std::isfinite(*pdfW))return Spectrum();
    }else {
	if (passThroughEvent < threshold) {
		// Transmit

		const bool entering = (CosTheta(localFixedDir) > 0.f);
		const float eta = entering ? (nc / nt) : ntc;
		const float eta2 = eta * eta;
		const float sinThetaIH2 = eta2 * Max(0.f, 1.f - cosThetaOH * cosThetaOH);
		if (sinThetaIH2 >= 1.f)
			return Spectrum();
		float cosThetaIH = sqrtf(1.f - sinThetaIH2);
		if (entering)
			cosThetaIH = -cosThetaIH;
		const float length = eta * cosThetaOH + cosThetaIH;
		*localSampledDir = length * wh - eta * localFixedDir;

		const float lengthSquared = length * length;
		*pdfW = specPdf * fabsf(cosThetaIH) / lengthSquared;
		if (*pdfW <= 0.f)
			return Spectrum();

		const float cosi = fabsf(localSampledDir->z);

		const float G = RoughGlassMLGeometry15ga(roughness, anisotropy, localFixedDir, *localSampledDir);
		float factor = (d / specPdf) * G * fabsf(cosThetaOH) / threshold;

		if (!hitPoint.fromLight) {
			const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaIH);
			result = (factor / coso) * kt * (1.f - F);
		} else {
			const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaOH);
			result = (factor / cosi) * kt * (1.f - F);
		}

		*pdfW *= threshold;
		// 15gk: audit transmission candidates before the geometry rejection.
		MLHero15gkAuditTransmission(hitPoint, localFixedDir, *localSampledDir,
				*pdfW, result);
		// Match Evaluate()/Pdf() hemisphere classification and pbrt's
		// rough-dielectric transmission guard. Do not resample or relabel
		// as reflection: this is a null (zero-contribution) sample.
		if ((cosi < DEFAULT_COS_EPSILON_STATIC) ||
				(localFixedDir.z * localSampledDir->z >= 0.f))
			return Spectrum();
		*event = GLOSSY | TRANSMIT;
	} else {
		// Reflect
        *pdfW = (mlHeroEnabled && fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
            RoughGlassMLReflectionPdf15kj(roughness,localFixedDir,wh) :
            specPdf/(4.f*fabsf(cosThetaOH));
		if (*pdfW <= 0.f)
			return Spectrum();

		*localSampledDir = 2.f * cosThetaOH * wh - localFixedDir;

		const float cosi = fabsf(localSampledDir->z);
		if ((cosi < DEFAULT_COS_EPSILON_STATIC) || (localFixedDir.z * localSampledDir->z < 0.f))
			return Spectrum();

		const float G = RoughGlassMLGeometry15ga(roughness, anisotropy, localFixedDir, *localSampledDir);
		float factor = (d / specPdf) * G * fabsf(cosThetaOH) / (1.f - threshold);

		const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaOH);
		factor /= (!hitPoint.fromLight) ? coso : cosi;
		result = factor * F * kr;
		
		const float localFilmThickness = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
		if (localFilmThickness > 0.f) {
			const float localFilmIor = filmIor ? filmIor->GetFloatValue(hitPoint) : 1.f;
			result *= CalcFilmColor(localFixedDir, localFilmThickness, localFilmIor);
		}

		*pdfW *= (1.f - threshold);
		*event = GLOSSY | REFLECT;
	}

    }

    // 15ki: value-owned observation of a successful native sample only.
    if(slg::mlhero15ki::armed && !result.Black()) {
        auto &s=slg::mlhero15ki::current;
        s.valid=true;s.material=this;s.fromLight=hitPoint.fromLight;
        s.transmit=(*event & TRANSMIT)!=0;
        s.vndf=fineRoughGlass && RoughGlassMLUseVNDF(anisotropy);
        s.importance=RoughGlassML15ghEnabled();s.krBlack=isKrBlack;s.ktBlack=isKtBlack;
        s.nc=nc;s.nt=nt;s.roughness=roughness;s.anisotropy=anisotropy;
        s.sampleD=d;s.sampleSpecPdf=specPdf;
        s.probability=s.transmit?threshold:1.f-threshold;s.samplePdf=*pdfW;
        s.fixed=slg::mlhero15ki::Vec(localFixedDir);
        s.sampled=slg::mlhero15ki::Vec(*localSampledDir);
        s.wh=slg::mlhero15ki::Vec(wh);
    }

	// ML HERO: mark this path as dispersive only after RoughGlass has
	// successfully sampled an actual event. The existing Path/Bidir HERO
	// weighting then applies the wavelength RGB estimator exactly once.
	const float cauchyBValue = cauchyB ? cauchyB->GetFloatValue(hitPoint) : 0.f;
	if (mlHeroEnabled && ((dispersionModel == GLASS_DISPERSION_SELLMEIER) ||
			(cauchyBValue > 0.f)))
		::MarkMLDispersionUsed();

	// 15gf: Evaluate and Pdf at the *same sampled directions* and same
	// transport orientation. Sample() returns f*cos(sampled)/pdf, while
	// Evaluate() already includes the cosine factor in LuxCore's convention.
	// No correction or replayed sample is used for actual rendering.
	if (mlHeroEnabled && (mlHeroPathRoughConsistencyEnabled15gf || MLHeroRoughGlass15gfShouldAudit()) &&
			(*pdfW > 0.f) && !result.Black()) {
		const Vector &lightDir = hitPoint.fromLight ? localFixedDir : *localSampledDir;
		const Vector &eyeDir = hitPoint.fromLight ? *localSampledDir : localFixedDir;
		BSDFEvent evalEvent = NONE;
		float evalDirectPdf = 0.f, evalReversePdf = 0.f;
		const Spectrum evaluated = Evaluate(hitPoint, lightDir, eyeDir,
				&evalEvent, &evalDirectPdf, &evalReversePdf);
		float pdfDirect = 0.f, pdfReverse = 0.f;
		Pdf(hitPoint, lightDir, eyeDir, &pdfDirect, &pdfReverse);
		const float sampleWeight = result.Filter();
		const float evaluateWeight = (evalDirectPdf > 0.f) ?
				evaluated.Filter() / evalDirectPdf : 0.f;
		if (mlHeroPathRoughConsistencyEnabled15gf)
            MLHeroPathRoughConsistencyReport15gf(hitPoint.fromLight,
                (*event & TRANSMIT) != 0, nc, nt, *pdfW, evalDirectPdf,
                pdfDirect, sampleWeight, evaluateWeight,
                fabsf(localFixedDir.z), fabsf(localSampledDir->z));
        else
            MLHeroRoughGlass15gfReport(hitPoint.fromLight,
                (*event & TRANSMIT) != 0, nc, nt, *pdfW, evalDirectPdf,
                pdfDirect, sampleWeight, evaluateWeight,
                fabsf(localFixedDir.z), fabsf(localSampledDir->z));
	}
	return result;
}

Spectrum RoughGlassMaterial::SampleMLHeroPrecise15kl(const HitPoint &hitPoint,
        const slg::mlhero15kk::V &localFixedPrecise15kl,slg::mlhero15kk::V *localSampledPrecise15kl,
        float u0,float u1,float passThroughEvent,float *pdfW,BSDFEvent *event) const {
    const Vector localFixedDir=MLHeroVector15kk(localFixedPrecise15kl);
    Vector localSampledStorage;Vector *localSampledDir=&localSampledStorage;
    if(!mlHeroEnabled){
        Spectrum r=Sample(hitPoint,localFixedDir,localSampledDir,u0,u1,passThroughEvent,pdfW,event);
        if(!r.Black())*localSampledPrecise15kl=slg::mlhero15kk::V(*localSampledDir);return r;
    }

    if ((mlHeroEnabled?std::fabs(slg::mlhero15kk::Unit(localFixedPrecise15kl).z):fabsf(localFixedDir.z)) < DEFAULT_COS_EPSILON_STATIC)
		return Spectrum();

	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);

	const bool isKtBlack = kt.Black();
	const bool isKrBlack = kr.Black();
	if (isKtBlack && isKrBlack)
		return Spectrum();

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = RoughGlassMLGetInteriorIOR(hitPoint, interiorIor, cauchyB, dispersionModel, sellmeierPreset);
	const float ntc = nt / nc;

	// 15gk: choose a microfacet normal directly; no stochastic delta mixing.
	const float roughU0 = u0;

    const Vector samplingFixed15kk=mlHeroEnabled?
        MLHeroVector15kk(slg::mlhero15kk::Unit(localFixedPrecise15kl)):localFixedDir;
	Vector wh;
	float d, specPdf;
	if (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy))
		RoughGlassMLSampleVisibleNormal(roughness, samplingFixed15kk,
				roughU0, u1, &wh, &d, &specPdf);
	else
		SchlickDistribution_SampleH(roughness, anisotropy,
				roughU0, u1, &wh, &d, &specPdf);
	if (wh.z < 0.f)
		wh = -wh;
    // The legacy NDF sampler reports D as its PDF.  Preserve the stable BRDF
    // D, but use the actual inverse-CDF density for the sampled half-vector.
    if(mlHeroEnabled && !(fineRoughGlass && RoughGlassMLUseVNDF(anisotropy))) {
        d=RoughGlassMLDistributionD15kj(roughness,wh,anisotropy);
        specPdf=RoughGlassMLSchlickHalfPdf15ld(roughness,anisotropy,wh);
    }
	slg::mlhero15lc::Attempt audit15lc(hitPoint.fromLight, GetID(),
		fineRoughGlass && RoughGlassMLUseVNDF(anisotropy), u, v, roughness,
		anisotropy, roughU0, u1, passThroughEvent,
		slg::mlhero15kk::V(samplingFixed15kk), slg::mlhero15kk::V(wh), d, specPdf);
	const float cosThetaOH = Dot(localFixedDir, wh);

    const auto optics15kk=slg::mlhero15kk::MakeInterface(nc,nt,
        localFixedPrecise15kl,slg::mlhero15kk::V(wh));
    if(mlHeroEnabled && !optics15kk.valid) {
		audit15lc.SetStatus(slg::mlhero15lc::INVALID_INTERFACE);
		return Spectrum();
	}
    if(mlHeroEnabled) {
        d=RoughGlassMLDistributionD15kj(roughness,wh,anisotropy);
        specPdf=RoughGlassMLHalfPdf15ld(roughness,anisotropy,
            localFixedDir,wh,fineRoughGlass && RoughGlassMLUseVNDF(anisotropy));
    }
    const float coso=mlHeroEnabled?std::fabs(optics15kk.fixed.z):fabsf(localFixedDir.z);


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
		else
			return Spectrum();
	}

    if(mlHeroEnabled)
        threshold=slg::mlhero15kk::Probability(optics15kk.transmission,isKrBlack,isKtBlack,RoughGlassML15ghEnabled());
    else if (RoughGlassML15ghEnabled())
        threshold = RoughGlassML15ghTransmitProbability(ntc, localFixedDir,
                wh, isKrBlack, isKtBlack);

	Spectrum result;
    if(mlHeroEnabled) {
        using namespace slg::mlhero15kk;
        const bool transmit=passThroughEvent<threshold;
        const double probability=transmit?threshold:1.f-threshold;
		audit15lc.Branch(transmit, probability);
        if(transmit && optics15kk.tir) {
			audit15lc.SetStatus(slg::mlhero15lc::TIR_TRANSMISSION);
			return Spectrum();
		}
        const V generated=transmit?Refract(optics15kk):Reflect(optics15kk);
        *localSampledDir=MLHeroVector15kk(generated);
        *localSampledPrecise15kl=generated;
        const double ci=std::fabs(generated.z),co=std::fabs(optics15kk.fixed.z);
        if(ci<DEFAULT_COS_EPSILON_STATIC ||
                (transmit?(optics15kk.fixed.z*generated.z>=0.):(optics15kk.fixed.z*generated.z<0.))) {
			audit15lc.SetStatus(slg::mlhero15lc::HEMISPHERE_REJECT);
			return Spectrum();
		}
        if(!(probability>0.)||!(specPdf>0.f)) {
			audit15lc.SetStatus(slg::mlhero15lc::ZERO_DENSITY);
			return Spectrum();
		}
        const Vector normalizedFixed=MLHeroVector15kk(optics15kk.fixed);
        const float g=RoughGlassMLGeometry15ga(roughness,anisotropy,normalizedFixed,*localSampledDir);
        const double factor=(double(d)/specPdf)*g*std::fabs(optics15kk.cosine)/probability;
        if(transmit) {
            const double ctSigned=optics15kk.fixed.z>0.?-optics15kk.cosTransmitted:optics15kk.cosTransmitted;
            const double length=optics15kk.eta*optics15kk.cosine+ctSigned;
            *pdfW=probability*specPdf*optics15kk.cosTransmitted/(length*length);
            result=(factor*optics15kk.transmission/(hitPoint.fromLight?ci:co))*kt;
            MLHero15gkAuditTransmission(hitPoint,localFixedDir,*localSampledDir,*pdfW,result);
            *event=GLOSSY|TRANSMIT;
        }else{
            *pdfW=(fineRoughGlass && RoughGlassMLUseVNDF(anisotropy))?
                probability*d*G1(roughness,optics15kk.fixed)/(4.*co):
                probability*specPdf/(4.*std::fabs(optics15kk.cosine));
            result=(factor*(1.-optics15kk.transmission)/(hitPoint.fromLight?ci:co))*kr;
            *event=GLOSSY|REFLECT;
            const float thickness=filmThickness?filmThickness->GetFloatValue(hitPoint):0.f;
            if(thickness>0.f) {
                const float film=filmIor?filmIor->GetFloatValue(hitPoint):1.f;
                result*=CalcFilmColor(localFixedDir,thickness,film);
            }
        }
        if(!(*pdfW>0.f)||!std::isfinite(*pdfW)) {
			audit15lc.SetStatus(slg::mlhero15lc::PDF_REJECT);
			return Spectrum();
		}
		audit15lc.Accept(generated, *pdfW, optics15kk.cosine,
			optics15kk.cosTransmitted, optics15kk.eta);
    }else {
	if (passThroughEvent < threshold) {
		// Transmit

		const bool entering = (CosTheta(localFixedDir) > 0.f);
		const float eta = entering ? (nc / nt) : ntc;
		const float eta2 = eta * eta;
		const float sinThetaIH2 = eta2 * Max(0.f, 1.f - cosThetaOH * cosThetaOH);
		if (sinThetaIH2 >= 1.f)
			return Spectrum();
		float cosThetaIH = sqrtf(1.f - sinThetaIH2);
		if (entering)
			cosThetaIH = -cosThetaIH;
		const float length = eta * cosThetaOH + cosThetaIH;
		*localSampledDir = length * wh - eta * localFixedDir;

		const float lengthSquared = length * length;
		*pdfW = specPdf * fabsf(cosThetaIH) / lengthSquared;
		if (*pdfW <= 0.f)
			return Spectrum();

		const float cosi = fabsf(localSampledDir->z);

		const float G = RoughGlassMLGeometry15ga(roughness, anisotropy, localFixedDir, *localSampledDir);
		float factor = (d / specPdf) * G * fabsf(cosThetaOH) / threshold;

		if (!hitPoint.fromLight) {
			const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaIH);
			result = (factor / coso) * kt * (1.f - F);
		} else {
			const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaOH);
			result = (factor / cosi) * kt * (1.f - F);
		}

		*pdfW *= threshold;
		// 15gk: audit transmission candidates before the geometry rejection.
		MLHero15gkAuditTransmission(hitPoint, localFixedDir, *localSampledDir,
				*pdfW, result);
		// Match Evaluate()/Pdf() hemisphere classification and pbrt's
		// rough-dielectric transmission guard. Do not resample or relabel
		// as reflection: this is a null (zero-contribution) sample.
		if ((cosi < DEFAULT_COS_EPSILON_STATIC) ||
				(localFixedDir.z * localSampledDir->z >= 0.f))
			return Spectrum();
		*event = GLOSSY | TRANSMIT;
	} else {
		// Reflect
        *pdfW = (mlHeroEnabled && fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
            RoughGlassMLReflectionPdf15kj(roughness,localFixedDir,wh) :
            specPdf/(4.f*fabsf(cosThetaOH));
		if (*pdfW <= 0.f)
			return Spectrum();

		*localSampledDir = 2.f * cosThetaOH * wh - localFixedDir;

		const float cosi = fabsf(localSampledDir->z);
		if ((cosi < DEFAULT_COS_EPSILON_STATIC) || (localFixedDir.z * localSampledDir->z < 0.f))
			return Spectrum();

		const float G = RoughGlassMLGeometry15ga(roughness, anisotropy, localFixedDir, *localSampledDir);
		float factor = (d / specPdf) * G * fabsf(cosThetaOH) / (1.f - threshold);

		const float F = FresnelTexture::CauchyEvaluate(ntc, cosThetaOH);
		factor /= (!hitPoint.fromLight) ? coso : cosi;
		result = factor * F * kr;
		
		const float localFilmThickness = filmThickness ? filmThickness->GetFloatValue(hitPoint) : 0.f;
		if (localFilmThickness > 0.f) {
			const float localFilmIor = filmIor ? filmIor->GetFloatValue(hitPoint) : 1.f;
			result *= CalcFilmColor(localFixedDir, localFilmThickness, localFilmIor);
		}

		*pdfW *= (1.f - threshold);
		*event = GLOSSY | REFLECT;
	}

    }

    // 15ki: value-owned observation of a successful native sample only.
    if(slg::mlhero15ki::armed && !result.Black()) {
        auto &s=slg::mlhero15ki::current;
        s.valid=true;s.material=this;s.fromLight=hitPoint.fromLight;
        s.precise15kl=true;s.fixedPrecise15kl=localFixedPrecise15kl;
        s.sampledPrecise15kl=*localSampledPrecise15kl;s.whPrecise15kl=optics15kk.h;
        s.transmit=(*event & TRANSMIT)!=0;
        s.vndf=fineRoughGlass && RoughGlassMLUseVNDF(anisotropy);
        s.importance=RoughGlassML15ghEnabled();s.krBlack=isKrBlack;s.ktBlack=isKtBlack;
        s.nc=nc;s.nt=nt;s.roughness=roughness;s.anisotropy=anisotropy;
        s.sampleD=d;s.sampleSpecPdf=specPdf;
        s.probability=s.transmit?threshold:1.f-threshold;s.samplePdf=*pdfW;
        s.fixed=slg::mlhero15ki::Vec(localFixedDir);
        s.sampled=slg::mlhero15ki::Vec(*localSampledDir);
        s.wh=slg::mlhero15ki::Vec(wh);
    }

	// ML HERO: mark this path as dispersive only after RoughGlass has
	// successfully sampled an actual event. The existing Path/Bidir HERO
	// weighting then applies the wavelength RGB estimator exactly once.
	const float cauchyBValue = cauchyB ? cauchyB->GetFloatValue(hitPoint) : 0.f;
	if (mlHeroEnabled && ((dispersionModel == GLASS_DISPERSION_SELLMEIER) ||
			(cauchyBValue > 0.f)))
		::MarkMLDispersionUsed();

	// 15gf: Evaluate and Pdf at the *same sampled directions* and same
	// transport orientation. Sample() returns f*cos(sampled)/pdf, while
	// Evaluate() already includes the cosine factor in LuxCore's convention.
	// No correction or replayed sample is used for actual rendering.
	if (mlHeroEnabled && (mlHeroPathRoughConsistencyEnabled15gf || MLHeroRoughGlass15gfShouldAudit()) &&
			(*pdfW > 0.f) && !result.Black()) {
		const auto &lightDir = hitPoint.fromLight ? localFixedPrecise15kl : *localSampledPrecise15kl;
		const auto &eyeDir = hitPoint.fromLight ? *localSampledPrecise15kl : localFixedPrecise15kl;
		BSDFEvent evalEvent = NONE;
		float evalDirectPdf = 0.f, evalReversePdf = 0.f;
		const Spectrum evaluated = EvaluateMLHeroPrecise15kl(hitPoint, lightDir, eyeDir,
				&evalEvent, &evalDirectPdf, &evalReversePdf);
		float pdfDirect = 0.f, pdfReverse = 0.f;
		PdfMLHeroPrecise15kl(hitPoint, lightDir, eyeDir, &pdfDirect, &pdfReverse);
		const float sampleWeight = result.Filter();
		const float evaluateWeight = (evalDirectPdf > 0.f) ?
				evaluated.Filter() / evalDirectPdf : 0.f;
		if (mlHeroPathRoughConsistencyEnabled15gf)
            MLHeroPathRoughConsistencyReport15gf(hitPoint.fromLight,
                (*event & TRANSMIT) != 0, nc, nt, *pdfW, evalDirectPdf,
                pdfDirect, sampleWeight, evaluateWeight,
                fabsf(localFixedDir.z), fabsf(localSampledDir->z));
        else
            MLHeroRoughGlass15gfReport(hitPoint.fromLight,
                (*event & TRANSMIT) != 0, nc, nt, *pdfW, evalDirectPdf,
                pdfDirect, sampleWeight, evaluateWeight,
                fabsf(localFixedDir.z), fabsf(localSampledDir->z));
	}
	return result;

}



bool RoughGlassMaterial::EvaluateMLHeroMicrofacetOwnershipDebug(const HitPoint &hitPoint,
		const Vector &localFixedDir, const Vector &localSharedDir,
		const BSDFEvent sampledEvent, const float heroWaveLength,
		const float laneWaveLength, float *heroEta, float *laneEta,
		float *heroMicrofacetPdf, float *laneSharedMicrofacetPdf,
		float *naturalJacobian, float *sharedJacobian,
		float *naturalDirPdf, float *sharedDirPdf,
		float *naturalDirDelta, float *sharedWhDelta, bool *naturalTir) const {
    if(mlHeroEnabled)return EvaluateMLHeroMicrofacetOwnershipPrecise15kl(hitPoint,slg::mlhero15kk::V(localFixedDir),slg::mlhero15kk::V(localSharedDir),sampledEvent,heroWaveLength,laneWaveLength,heroEta,laneEta,heroMicrofacetPdf,laneSharedMicrofacetPdf,naturalJacobian,sharedJacobian,naturalDirPdf,sharedDirPdf,naturalDirDelta,sharedWhDelta,naturalTir);

	if (!(sampledEvent & TRANSMIT) || !(sampledEvent & GLOSSY) ||
			(sampledEvent & SPECULAR) ||
			(heroWaveLength < 380.f) || (heroWaveLength > 780.f) ||
			(laneWaveLength < 380.f) || (laneWaveLength > 780.f))
		return false;

	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const bool isKtBlack = kt.Black();
	const bool isKrBlack = kr.Black();
	if (isKtBlack || (isKtBlack && isKrBlack))
		return false;

	const float threshold = isKrBlack ? 1.f : .5f;
	if (!(threshold > 0.f))
		return false;

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float ntBase = ExtractInteriorIors(hitPoint, interiorIor);
	const float b = cauchyB ? cauchyB->GetFloatValue(hitPoint) : 0.f;
	auto ntAt = [&](const float wl) -> float {
		if (dispersionModel == GLASS_DISPERSION_SELLMEIER)
			return RoughGlassMLWaveLength2IORSellmeier(wl, sellmeierPreset);
		if (b > 0.f)
			return RoughGlassMLWaveLength2IOR(wl, ntBase, b);
		return ntBase;
	};

	const float ntHero = ntAt(heroWaveLength);
	const float ntLane = ntAt(laneWaveLength);
	if (!(nc > 1e-20f) || !(ntHero > 1e-20f) || !(ntLane > 1e-20f))
		return false;

    // 15kk: keep the existing ownership diagnostic on the same interface
    // mapping and branch probabilities as the native production material.
    if(mlHeroEnabled) {
        using namespace slg::mlhero15kk;
        const V fixed=Unit(V(localFixedDir)),shared=Unit(V(localSharedDir));
        const auto hero=MakePair(nc,ntHero,hitPoint.fromLight?fixed:shared,
            hitPoint.fromLight?shared:fixed);
        if(!hero.valid||!hero.transmit)return false;
        const auto lane=MakePair(nc,ntLane,hitPoint.fromLight?fixed:shared,
            hitPoint.fromLight?shared:fixed);
        const auto natural=MakeInterface(nc,ntLane,fixed,hero.h);
        const bool tir=!natural.valid||natural.tir;
        const Vector h=MLHeroVector15kk(hero.h);
        const double hd=RoughGlassMLDistributionD15kj(roughness,h,anisotropy);
        const double spec=hd*((fineRoughGlass&&RoughGlassMLUseVNDF(anisotropy))?
            G1(roughness,fixed)*std::fabs(Dot(fixed,hero.h))/std::fabs(fixed.z):1.);
        double nj=0.,np=0.;V nd;
        if(!tir) {
            nd=Refract(natural);
            const double ct=natural.fixed.z>0.?-natural.cosTransmitted:natural.cosTransmitted;
            const double length=natural.eta*natural.cosine+ct;
            nj=natural.cosTransmitted/(length*length);
            np=Probability(natural.transmission,isKrBlack,isKtBlack,RoughGlassML15ghEnabled())*spec*nj;
        }
        const auto qe=MLHeroEvaluateRough15kk(hitPoint.fromLight,
            hitPoint.fromLight?localFixedDir:localSharedDir,
            hitPoint.fromLight?localSharedDir:localFixedDir,
            nc,ntLane,roughness,anisotropy,fineRoughGlass&&RoughGlassMLUseVNDF(anisotropy),kr,kt);
        if(heroEta)*heroEta=fixed.z>0.?double(nc)/ntHero:double(ntHero)/nc;
        if(laneEta)*laneEta=fixed.z>0.?double(nc)/ntLane:double(ntLane)/nc;
        if(heroMicrofacetPdf)*heroMicrofacetPdf=spec;
        if(laneSharedMicrofacetPdf)*laneSharedMicrofacetPdf=qe.directSpec;
        if(naturalJacobian)*naturalJacobian=nj;
        if(sharedJacobian)*sharedJacobian=qe.directJacobian;
        if(naturalDirPdf)*naturalDirPdf=np;
        if(sharedDirPdf)*sharedDirPdf=qe.directPdf;
        if(naturalTir)*naturalTir=tir;
        if(naturalDirDelta)*naturalDirDelta=tir?2.:1.-std::fmax(-1.,std::fmin(1.,Dot(nd,shared)));
        if(sharedWhDelta)*sharedWhDelta=1.-std::fmax(-1.,std::fmin(1.,Dot(hero.h,lane.h)));
        return true;
    }

	const bool entering = (CosTheta(localFixedDir) > 0.f);
	const float eHero = entering ? (nc / ntHero) : (ntHero / nc);
	const float eLane = entering ? (nc / ntLane) : (ntLane / nc);
	if (heroEta) *heroEta = eHero;
	if (laneEta) *laneEta = eLane;

	Vector heroWhUn = localSharedDir + eHero * localFixedDir;
	const float heroWhLen2 = heroWhUn.LengthSquared();
	if (!(heroWhLen2 > 1e-30f))
		return false;
	Vector heroWh = heroWhUn / sqrtf(heroWhLen2);
	if (heroWh.z < 0.f)
		heroWh = -heroWh;

	const float heroSpecPdf = RoughGlassMLHalfPdf15ld(roughness, anisotropy,
			localFixedDir, heroWh,
			fineRoughGlass && RoughGlassMLUseVNDF(anisotropy));
	if (heroMicrofacetPdf) *heroMicrofacetPdf = heroSpecPdf;

	// Same RNG / same sampled microfacet normal, but lane-specific eta.
	const float cosOHNatural = Dot(localFixedDir, heroWh);
	const float laneEta2 = eLane * eLane;
	const float sinIH2 = laneEta2 * Max(0.f, 1.f - cosOHNatural * cosOHNatural);
	const bool tir = (sinIH2 >= 1.f);
	if (naturalTir) *naturalTir = tir;

	Vector laneNaturalDir(0.f);
	float naturalJac = 0.f;
	float naturalPdf = 0.f;
	if (!tir) {
		float cosIH = sqrtf(Max(0.f, 1.f - sinIH2));
		if (entering)
			cosIH = -cosIH;
		const float length = eLane * cosOHNatural + cosIH;
		laneNaturalDir = length * heroWh - eLane * localFixedDir;
		const float length2 = length * length;
		if (length2 > 1e-30f) {
			naturalJac = fabsf(cosIH) / length2;
			naturalPdf = threshold * heroSpecPdf * naturalJac;
		}
	}
	if (naturalJacobian) *naturalJacobian = naturalJac;
	if (naturalDirPdf) *naturalDirPdf = naturalPdf;

	// Alternative lane technique evaluated at the shared HERO direction. This
	// generally requires a *different* microfacet normal because eta(lambda)
	// changes the refraction mapping.
	Vector laneWhUn = localSharedDir + eLane * localFixedDir;
	const float laneWhLen2 = laneWhUn.LengthSquared();
	if (!(laneWhLen2 > 1e-30f))
		return false;
	Vector laneWh = laneWhUn / sqrtf(laneWhLen2);
	if (laneWh.z < 0.f)
		laneWh = -laneWh;

	const float laneSpecPdf = RoughGlassMLHalfPdf15ld(roughness, anisotropy,
			localFixedDir, laneWh,
			fineRoughGlass && RoughGlassMLUseVNDF(anisotropy));
	if (laneSharedMicrofacetPdf) *laneSharedMicrofacetPdf = laneSpecPdf;

	const float sharedCosIH = fabsf(Dot(localSharedDir, laneWh));
	const float sharedJac = sharedCosIH / laneWhLen2;
	const float sharedPdf = threshold * laneSpecPdf * sharedJac;
	if (sharedJacobian) *sharedJacobian = sharedJac;
	if (sharedDirPdf) *sharedDirPdf = sharedPdf;

	if (naturalDirDelta) {
		if (tir)
			*naturalDirDelta = 2.f;
		else {
			const Vector a = Normalize(laneNaturalDir);
			const Vector bdir = Normalize(localSharedDir);
			*naturalDirDelta = 1.f - Clamp(Dot(a, bdir), -1.f, 1.f);
		}
	}
	if (sharedWhDelta)
		*sharedWhDelta = 1.f - Clamp(Dot(heroWh, laneWh), -1.f, 1.f);

	return true;
}
bool RoughGlassMaterial::EvaluateMLHeroMicrofacetOwnershipPrecise15kl(const HitPoint &hitPoint,
		const slg::mlhero15kk::V &localFixedPrecise15kl, const slg::mlhero15kk::V &localSharedPrecise15kl,
		const BSDFEvent sampledEvent, const float heroWaveLength,
		const float laneWaveLength, float *heroEta, float *laneEta,
		float *heroMicrofacetPdf, float *laneSharedMicrofacetPdf,
		float *naturalJacobian, float *sharedJacobian,
		float *naturalDirPdf, float *sharedDirPdf,
		float *naturalDirDelta, float *sharedWhDelta, bool *naturalTir) const  {
    if(!mlHeroEnabled)return EvaluateMLHeroMicrofacetOwnershipDebug(hitPoint,MLHeroVector15kk(localFixedPrecise15kl),MLHeroVector15kk(localSharedPrecise15kl),sampledEvent,heroWaveLength,laneWaveLength,heroEta,laneEta,heroMicrofacetPdf,laneSharedMicrofacetPdf,naturalJacobian,sharedJacobian,naturalDirPdf,sharedDirPdf,naturalDirDelta,sharedWhDelta,naturalTir);
    const Vector localFixedDir=MLHeroVector15kk(localFixedPrecise15kl),localSharedDir=MLHeroVector15kk(localSharedPrecise15kl);

	if (!(sampledEvent & TRANSMIT) || !(sampledEvent & GLOSSY) ||
			(sampledEvent & SPECULAR) ||
			(heroWaveLength < 380.f) || (heroWaveLength > 780.f) ||
			(laneWaveLength < 380.f) || (laneWaveLength > 780.f))
		return false;

	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const bool isKtBlack = kt.Black();
	const bool isKrBlack = kr.Black();
	if (isKtBlack || (isKtBlack && isKrBlack))
		return false;

	const float threshold = isKrBlack ? 1.f : .5f;
	if (!(threshold > 0.f))
		return false;

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;

	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float ntBase = ExtractInteriorIors(hitPoint, interiorIor);
	const float b = cauchyB ? cauchyB->GetFloatValue(hitPoint) : 0.f;
	auto ntAt = [&](const float wl) -> float {
		if (dispersionModel == GLASS_DISPERSION_SELLMEIER)
			return RoughGlassMLWaveLength2IORSellmeier(wl, sellmeierPreset);
		if (b > 0.f)
			return RoughGlassMLWaveLength2IOR(wl, ntBase, b);
		return ntBase;
	};

	const float ntHero = ntAt(heroWaveLength);
	const float ntLane = ntAt(laneWaveLength);
	if (!(nc > 1e-20f) || !(ntHero > 1e-20f) || !(ntLane > 1e-20f))
		return false;

    // 15kk: keep the existing ownership diagnostic on the same interface
    // mapping and branch probabilities as the native production material.
    if(mlHeroEnabled) {
        using namespace slg::mlhero15kk;
        const V fixed=Unit(localFixedPrecise15kl),shared=Unit(localSharedPrecise15kl);
        const auto hero=MakePair(nc,ntHero,hitPoint.fromLight?fixed:shared,
            hitPoint.fromLight?shared:fixed);
        if(!hero.valid||!hero.transmit)return false;
        const auto lane=MakePair(nc,ntLane,hitPoint.fromLight?fixed:shared,
            hitPoint.fromLight?shared:fixed);
        const auto natural=MakeInterface(nc,ntLane,fixed,hero.h);
        const bool tir=!natural.valid||natural.tir;
        const Vector h=MLHeroVector15kk(hero.h);
        const double hd=RoughGlassMLDistributionD15kj(roughness,h,anisotropy);
        const double spec=hd*((fineRoughGlass&&RoughGlassMLUseVNDF(anisotropy))?
            G1(roughness,fixed)*std::fabs(Dot(fixed,hero.h))/std::fabs(fixed.z):1.);
        double nj=0.,np=0.;V nd;
        if(!tir) {
            nd=Refract(natural);
            const double ct=natural.fixed.z>0.?-natural.cosTransmitted:natural.cosTransmitted;
            const double length=natural.eta*natural.cosine+ct;
            nj=natural.cosTransmitted/(length*length);
            np=Probability(natural.transmission,isKrBlack,isKtBlack,RoughGlassML15ghEnabled())*spec*nj;
        }
        const auto qe=MLHeroEvaluateRough15kk(hitPoint.fromLight,
            hitPoint.fromLight?localFixedPrecise15kl:localSharedPrecise15kl,
            hitPoint.fromLight?localSharedPrecise15kl:localFixedPrecise15kl,
            nc,ntLane,roughness,anisotropy,fineRoughGlass&&RoughGlassMLUseVNDF(anisotropy),kr,kt);
        if(heroEta)*heroEta=fixed.z>0.?double(nc)/ntHero:double(ntHero)/nc;
        if(laneEta)*laneEta=fixed.z>0.?double(nc)/ntLane:double(ntLane)/nc;
        if(heroMicrofacetPdf)*heroMicrofacetPdf=spec;
        if(laneSharedMicrofacetPdf)*laneSharedMicrofacetPdf=qe.directSpec;
        if(naturalJacobian)*naturalJacobian=nj;
        if(sharedJacobian)*sharedJacobian=qe.directJacobian;
        if(naturalDirPdf)*naturalDirPdf=np;
        if(sharedDirPdf)*sharedDirPdf=qe.directPdf;
        if(naturalTir)*naturalTir=tir;
        if(naturalDirDelta)*naturalDirDelta=tir?2.:1.-std::fmax(-1.,std::fmin(1.,Dot(nd,shared)));
        if(sharedWhDelta)*sharedWhDelta=1.-std::fmax(-1.,std::fmin(1.,Dot(hero.h,lane.h)));
        return true;
    }

	const bool entering = (CosTheta(localFixedDir) > 0.f);
	const float eHero = entering ? (nc / ntHero) : (ntHero / nc);
	const float eLane = entering ? (nc / ntLane) : (ntLane / nc);
	if (heroEta) *heroEta = eHero;
	if (laneEta) *laneEta = eLane;

	Vector heroWhUn = localSharedDir + eHero * localFixedDir;
	const float heroWhLen2 = heroWhUn.LengthSquared();
	if (!(heroWhLen2 > 1e-30f))
		return false;
	Vector heroWh = heroWhUn / sqrtf(heroWhLen2);
	if (heroWh.z < 0.f)
		heroWh = -heroWh;

	const float heroSpecPdf = RoughGlassMLHalfPdf15ld(roughness, anisotropy,
			localFixedDir, heroWh,
			fineRoughGlass && RoughGlassMLUseVNDF(anisotropy));
	if (heroMicrofacetPdf) *heroMicrofacetPdf = heroSpecPdf;

	// Same RNG / same sampled microfacet normal, but lane-specific eta.
	const float cosOHNatural = Dot(localFixedDir, heroWh);
	const float laneEta2 = eLane * eLane;
	const float sinIH2 = laneEta2 * Max(0.f, 1.f - cosOHNatural * cosOHNatural);
	const bool tir = (sinIH2 >= 1.f);
	if (naturalTir) *naturalTir = tir;

	Vector laneNaturalDir(0.f);
	float naturalJac = 0.f;
	float naturalPdf = 0.f;
	if (!tir) {
		float cosIH = sqrtf(Max(0.f, 1.f - sinIH2));
		if (entering)
			cosIH = -cosIH;
		const float length = eLane * cosOHNatural + cosIH;
		laneNaturalDir = length * heroWh - eLane * localFixedDir;
		const float length2 = length * length;
		if (length2 > 1e-30f) {
			naturalJac = fabsf(cosIH) / length2;
			naturalPdf = threshold * heroSpecPdf * naturalJac;
		}
	}
	if (naturalJacobian) *naturalJacobian = naturalJac;
	if (naturalDirPdf) *naturalDirPdf = naturalPdf;

	// Alternative lane technique evaluated at the shared HERO direction. This
	// generally requires a *different* microfacet normal because eta(lambda)
	// changes the refraction mapping.
	Vector laneWhUn = localSharedDir + eLane * localFixedDir;
	const float laneWhLen2 = laneWhUn.LengthSquared();
	if (!(laneWhLen2 > 1e-30f))
		return false;
	Vector laneWh = laneWhUn / sqrtf(laneWhLen2);
	if (laneWh.z < 0.f)
		laneWh = -laneWh;

	const float laneSpecPdf = RoughGlassMLHalfPdf15ld(roughness, anisotropy,
			localFixedDir, laneWh,
			fineRoughGlass && RoughGlassMLUseVNDF(anisotropy));
	if (laneSharedMicrofacetPdf) *laneSharedMicrofacetPdf = laneSpecPdf;

	const float sharedCosIH = fabsf(Dot(localSharedDir, laneWh));
	const float sharedJac = sharedCosIH / laneWhLen2;
	const float sharedPdf = threshold * laneSpecPdf * sharedJac;
	if (sharedJacobian) *sharedJacobian = sharedJac;
	if (sharedDirPdf) *sharedDirPdf = sharedPdf;

	if (naturalDirDelta) {
		if (tir)
			*naturalDirDelta = 2.f;
		else {
			const Vector a = Normalize(laneNaturalDir);
			const Vector bdir = Normalize(localSharedDir);
			*naturalDirDelta = 1.f - Clamp(Dot(a, bdir), -1.f, 1.f);
		}
	}
	if (sharedWhDelta)
		*sharedWhDelta = 1.f - Clamp(Dot(heroWh, laneWh), -1.f, 1.f);

	return true;

}


void RoughGlassMaterial::Pdf(const HitPoint &hitPoint,
		const Vector &localLightDir, const Vector &localEyeDir,
		float *directPdfW, float *reversePdfW) const {
    if(mlHeroEnabled){PdfMLHeroPrecise15kl(hitPoint,slg::mlhero15kk::V(localLightDir),slg::mlhero15kk::V(localEyeDir),directPdfW,reversePdfW);return;}

	if (directPdfW)
		*directPdfW = 0.f;
	if (reversePdfW)
		*reversePdfW = 0.f;

	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);

	const bool isKtBlack = kt.Black();
	const bool isKrBlack = kr.Black();
	if (isKtBlack && isKrBlack)
		return;

	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = RoughGlassMLGetInteriorIOR(hitPoint, interiorIor, cauchyB, dispersionModel, sellmeierPreset);
	const float ntc = nt / nc;

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;


    if(mlHeroEnabled) {
        const auto e=MLHeroEvaluateRough15kk(hitPoint.fromLight,localLightDir,localEyeDir,
            nc,nt,roughness,anisotropy,fineRoughGlass && RoughGlassMLUseVNDF(anisotropy),kr,kt);
        if(directPdfW)*directPdfW=e.directPdf;
        if(reversePdfW)*reversePdfW=e.reversePdf;
        return;
    }

	const float legacyThreshold = isKrBlack ? 1.f : (isKtBlack ? 0.f : .5f);
	if (localLightDir.z * localEyeDir.z < 0.f) {
		// Transmit

		// Keep LuxCore's original light/eye-side convention here. For transmission
		// this eta is the inverse of the Sample() eta on an eye/radiance path;
		// that is intentional and makes the reconstructed half-vector reciprocal.
		const bool entering = (CosTheta(localLightDir) > 0.f);
		const float eta = entering ? (nc / nt) : ntc;

		Vector wh = eta * localLightDir + localEyeDir;
		if (wh.z < 0.f)
			wh = -wh;

		const float lengthSquared = wh.LengthSquared();
		if (!(lengthSquared > 0.f))
			return;

		wh /= sqrtf(lengthSquared);
        const float threshold = RoughGlassML15ghEnabled() ?
                RoughGlassML15ghTransmitProbability(ntc,
                    hitPoint.fromLight ? localLightDir : localEyeDir,
                    wh, isKrBlack, isKtBlack) : legacyThreshold;
        const float reverseThreshold = RoughGlassML15ghEnabled() ?
                RoughGlassML15ghTransmitProbability(ntc,
                    hitPoint.fromLight ? localEyeDir : localLightDir,
                    wh, isKrBlack, isKtBlack) : legacyThreshold;
		const float cosThetaIH = AbsDot(localEyeDir, wh);
		const float cosThetaOH = AbsDot(localLightDir, wh);

		const float directSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localLightDir : localEyeDir, wh) :
				RoughGlassMLDistributionD15kj(roughness, wh, anisotropy);
		const float reverseSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localEyeDir : localLightDir, wh) :
				RoughGlassMLDistributionD15kj(roughness, wh, anisotropy);

		if (directPdfW)
			*directPdfW = threshold * directSpecPdf *
					(hitPoint.fromLight ? cosThetaIH : (cosThetaOH * eta * eta)) /
					lengthSquared;

		if (reversePdfW)
			*reversePdfW = reverseThreshold * reverseSpecPdf *
					(hitPoint.fromLight ? (cosThetaOH * eta * eta) : cosThetaIH) /
					lengthSquared;
	} else {
		// Reflect
		const float cosThetaO = fabsf(CosTheta(localLightDir));
		const float cosThetaI = fabsf(CosTheta(localEyeDir));
		if (cosThetaO == 0.f || cosThetaI == 0.f)
			return;

		Vector wh = localLightDir + localEyeDir;
		if (wh == Vector(0.f))
			return;
		wh = Normalize(wh);
		if (wh.z < 0.f)
			wh = -wh;
        const float threshold = RoughGlassML15ghEnabled() ?
                RoughGlassML15ghTransmitProbability(ntc,
                    hitPoint.fromLight ? localLightDir : localEyeDir,
                    wh, isKrBlack, isKtBlack) : legacyThreshold;
        const float reverseThreshold = RoughGlassML15ghEnabled() ?
                RoughGlassML15ghTransmitProbability(ntc,
                    hitPoint.fromLight ? localEyeDir : localLightDir,
                    wh, isKrBlack, isKtBlack) : legacyThreshold;

		const float directSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localLightDir : localEyeDir, wh) :
				RoughGlassMLDistributionD15kj(roughness, wh, anisotropy);
		const float reverseSpecPdf = (fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
				RoughGlassMLVisibleNormalPdf(roughness,
						hitPoint.fromLight ? localEyeDir : localLightDir, wh) :
				RoughGlassMLDistributionD15kj(roughness, wh, anisotropy);

		if (directPdfW)
            *directPdfW = (mlHeroEnabled && fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
                (1.f-threshold)*RoughGlassMLReflectionPdf15kj(roughness,
                    hitPoint.fromLight ? localLightDir : localEyeDir,wh) :
                (1.f-threshold)*directSpecPdf/(4.f*AbsDot(localLightDir,wh));

		if (reversePdfW)
            *reversePdfW = (mlHeroEnabled && fineRoughGlass && RoughGlassMLUseVNDF(anisotropy)) ?
                (1.f-reverseThreshold)*RoughGlassMLReflectionPdf15kj(roughness,
                    hitPoint.fromLight ? localEyeDir : localLightDir,wh) :
                (1.f-reverseThreshold)*reverseSpecPdf/(4.f*AbsDot(localLightDir,wh));
	}
}

void RoughGlassMaterial::PdfMLHeroPrecise15kl(const HitPoint &hitPoint,
        const slg::mlhero15kk::V &localLightDir,const slg::mlhero15kk::V &localEyeDir,
        float *directPdfW,float *reversePdfW) const {
    if(!mlHeroEnabled){Pdf(hitPoint,MLHeroVector15kk(localLightDir),MLHeroVector15kk(localEyeDir),directPdfW,reversePdfW);return;}

	if (directPdfW)
		*directPdfW = 0.f;
	if (reversePdfW)
		*reversePdfW = 0.f;

	const Spectrum kt = Kt->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
	const Spectrum kr = Kr->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);

	const bool isKtBlack = kt.Black();
	const bool isKrBlack = kr.Black();
	if (isKtBlack && isKrBlack)
		return;

	const float nc = ExtractExteriorIors(hitPoint, exteriorIor);
	const float nt = RoughGlassMLGetInteriorIOR(hitPoint, interiorIor, cauchyB, dispersionModel, sellmeierPreset);
	const float ntc = nt / nc;

	const float u = Clamp(nu->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float v = Clamp(nv->GetFloatValue(hitPoint), 1e-9f, 1.f);
	const float u2 = u * u;
	const float v2 = v * v;
	const float anisotropy = (u2 < v2) ? (1.f - u2 / v2) : u2 > 0.f ? (v2 / u2 - 1.f) : 0.f;
	const float roughness = u * v;



        const auto e=MLHeroEvaluateRough15kk(hitPoint.fromLight,localLightDir,localEyeDir,
            nc,nt,roughness,anisotropy,fineRoughGlass && RoughGlassMLUseVNDF(anisotropy),kr,kt);
        if(directPdfW)*directPdfW=e.directPdf;
        if(reversePdfW)*reversePdfW=e.reversePdf;
        return;
    
}


void RoughGlassMaterial::AddReferencedTextures(std::unordered_set<const Texture *>  &referencedTexs) const {
	Material::AddReferencedTextures(referencedTexs);

	Kr->AddReferencedTextures(referencedTexs);
	Kt->AddReferencedTextures(referencedTexs);
	if (exteriorIor)
		exteriorIor->AddReferencedTextures(referencedTexs);
	if (interiorIor)
		interiorIor->AddReferencedTextures(referencedTexs);
	if (cauchyB)
		cauchyB->AddReferencedTextures(referencedTexs);
	nu->AddReferencedTextures(referencedTexs);
	nv->AddReferencedTextures(referencedTexs);
	if (filmThickness)
		filmThickness->AddReferencedTextures(referencedTexs);
	if (filmIor)
		filmIor->AddReferencedTextures(referencedTexs);
}

void RoughGlassMaterial::UpdateTextureReferences(TextureConstRef oldTex, TextureRef newTex) {
	Material::UpdateTextureReferences(oldTex, newTex);

	bool updateGlossiness = false;
	if (Kr == &oldTex)
		Kr = &newTex;
	if (Kt == &oldTex)
		Kt = &newTex;
	if (exteriorIor == &oldTex)
		exteriorIor = &newTex;
	if (interiorIor == &oldTex)
		interiorIor = &newTex;
	if (cauchyB == &oldTex)
		cauchyB = &newTex;
	if (nu == &oldTex) {
		nu = &newTex;
		updateGlossiness = true;
	}
	if (nv == &oldTex) {
		nv = &newTex;
		updateGlossiness = true;
	}
	if (filmThickness == &oldTex)
		filmThickness = &newTex;
	if (filmIor == &oldTex)
		filmIor = &newTex;

	if (updateGlossiness)
		glossiness = ComputeGlossiness(nu, nv);
}

PropertiesUPtr RoughGlassMaterial::ToProperties(const ImageMapCache &imgMapCache, const bool useRealFileName) const  {
	auto props = std::make_unique<Properties>();

	const string name = GetName();
	props->Set(Property("scene.materials." + name + ".type")("roughglass"));
	props->Set(Property("scene.materials." + name + ".kr")(Kr->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".kt")(Kt->GetSDLValue()));
	if (exteriorIor)
		props->Set(Property("scene.materials." + name + ".exteriorior")(exteriorIor->GetSDLValue()));
	if (interiorIor)
		props->Set(Property("scene.materials." + name + ".interiorior")(interiorIor->GetSDLValue()));
	if (cauchyB)
		props->Set(Property("scene.materials." + name + ".cauchyb")(cauchyB->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".dispersionmodel")(
			dispersionModel == GLASS_DISPERSION_SELLMEIER ? "sellmeier" : "cauchy"));
	if (dispersionModel == GLASS_DISPERSION_SELLMEIER) {
		string presetName = "n_bk7";
		switch (sellmeierPreset) {
			case GLASS_SELLMEIER_FUSED_SILICA: presetName = "fused_silica"; break;
			case GLASS_SELLMEIER_SF10: presetName = "sf10"; break;
			case GLASS_SELLMEIER_SF11: presetName = "sf11"; break;
			default: break;
		}
		props->Set(Property("scene.materials." + name + ".sellmeierpreset")(presetName));
	}
	props->Set(Property("scene.materials." + name + ".fineroughglass.enable")(fineRoughGlass));
	props->Set(Property("scene.materials." + name + ".uroughness")(nu->GetSDLValue()));
	props->Set(Property("scene.materials." + name + ".vroughness")(nv->GetSDLValue()));
	if (filmThickness)
		props->Set(Property("scene.materials." + name + ".filmthickness")(filmThickness->GetSDLValue()));
	if (filmIor)
		props->Set(Property("scene.materials." + name + ".filmior")(filmIor->GetSDLValue()));
	props->Set(Material::ToProperties(imgMapCache, useRealFileName));

	return props;
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

// Pure double reference for near-critical Fresnel sensitivity. No density,
// sample direction, branch choice or throughput uses these values.
static slg::mlhero15ki::CriticalAngles MLHeroCriticalAngles15kj(
        const slg::mlhero15ki::Snapshot &s,slg::mlhero15ki::Vec h) {
    slg::mlhero15ki::CriticalAngles a;
    const double x=s.fixed.x,y=s.fixed.y,z=s.fixed.z,hx=h.x,hy=h.y,hz=h.z;
    const double f2=x*x+y*y+z*z,n2=hx*hx+hy*hy+hz*hz;
    if(!(f2>0.)||!(n2>0.)||!(s.nc>0.f)||!(s.nt>0.f))return a;
    a.fixedNorm=f2;a.normalNorm=n2;
    a.cosine=(x*hx+y*hy+z*hz)/std::sqrt(f2*n2);
    const double cx=y*hz-z*hy,cy=z*hx-x*hz,cz=x*hy-y*hx;
    const double sineSquared=(cx*cx+cy*cy+cz*cz)/(f2*n2);
    const double eta=a.cosine>0.?double(s.nc)/s.nt:double(s.nt)/s.nc;
    a.discriminant=1.-eta*eta*sineSquared;
    if(a.discriminant<=0.){a.transmission=0.;return a;}
    const double ci=std::fabs(a.cosine),ct=std::sqrt(a.discriminant),ntc=1./eta;
    const double parallel=(ct-ntc*ci)/(ct+ntc*ci);
    const double perpendicular=(ci-ntc*ct)/(ci+ntc*ct);
    a.transmission=1.-.5*(parallel*parallel+perpendicular*perpendicular);
    return a;
}

// Diagnostic-only factorization at the original and world-round-tripped directions.
static slg::mlhero15ki::Factors MLHeroRoughFactors15ki(
        const slg::mlhero15ki::Snapshot &s,const Vector &sampled) {
    using namespace slg::mlhero15ki;
    Factors out;
    const Vector fixed(s.fixed.x,s.fixed.y,s.fixed.z);
    if(mlHeroEnabled) {
        const auto e=MLHeroEvaluateRough15kk(s.fromLight,s.fromLight?fixed:sampled,
            s.fromLight?sampled:fixed,s.nc,s.nt,s.roughness,s.anisotropy,s.vndf,
            Spectrum(s.krBlack?0.f:1.f),Spectrum(s.ktBlack?0.f:1.f));
        out.wh=Vec(MLHeroVector15kk(e.pair.h));out.d=e.d;out.referenceD=e.d;
        out.specPdf=e.directSpec;out.probability=e.directProbability;
        out.jacobian=e.directJacobian;out.pdf=e.directPdf;
        out.angles=MLHeroCriticalAngles15kj(s,out.wh);
        return out;
    }
    const Vector light=s.fromLight?fixed:sampled,eye=s.fromLight?sampled:fixed;
    Vector wh;
    float eta=1.f,lengthSquared=0.f;
    if(s.transmit){
        eta=light.z>0.f?s.nc/s.nt:s.nt/s.nc;
        wh=eta*light+eye;lengthSquared=wh.LengthSquared();
        if(!(lengthSquared>0.f))return out;
        if(wh.z<0.f)wh=-wh;
        wh/=sqrtf(lengthSquared);
    }else{
        wh=light+eye;if(!(wh.LengthSquared()>0.f))return out;
        wh=Normalize(wh);if(wh.z<0.f)wh=-wh;
    }
    out.wh=Vec(wh);out.angles=MLHeroCriticalAngles15kj(s,Vec(wh));
    out.d=RoughGlassMLDistributionD15kj(s.roughness,wh,s.anisotropy);
    out.referenceD=ReferenceZ(s.roughness,Vec(wh))*SchlickDistribution_SchlickA(wh,s.anisotropy)*INV_PI;
    out.specPdf=s.vndf?RoughGlassMLVisibleNormalPdf(s.roughness,fixed,wh):
        RoughGlassMLDistributionD15kj(s.roughness,wh,s.anisotropy);
    const float transmission=s.importance?
        RoughGlassML15ghTransmitProbability(s.nt/s.nc,fixed,wh,s.krBlack,s.ktBlack):
        (s.krBlack?1.f:(s.ktBlack?0.f:.5f));
    out.probability=s.transmit?transmission:1.f-transmission;
    out.jacobian=s.transmit?
        (s.fromLight?AbsDot(eye,wh):AbsDot(light,wh)*eta*eta)/lengthSquared:
        1.f/(4.f*AbsDot(light,wh));
    out.pdf=out.specPdf*out.probability*out.jacobian;
    if(mlHeroEnabled && s.vndf && !s.transmit)
        out.pdf=out.probability*RoughGlassMLReflectionPdf15kj(s.roughness,fixed,wh);

    return out;
}

slg::mlhero15ki::Comparison MLHeroCompareRoughGlass15ki(
        const HitPoint &hitPoint,const slg::mlhero15kk::V &roundTripPrecise) {
    using namespace slg::mlhero15ki;
    const Snapshot s=current;
    const Vector roundTrip=MLHeroVector15kk(roundTripPrecise);
    Comparison out;
    if(!s.valid||!s.material)return out;
    const auto *material=static_cast<const RoughGlassMaterial *>(s.material);
    const Vector fixed(s.fixed.x,s.fixed.y,s.fixed.z),sampled(s.sampled.x,s.sampled.y,s.sampled.z);
    float reverse;
    if(s.precise15kl) {
        const auto fixed=s.fixedPrecise15kl,sampled=s.sampledPrecise15kl;
        auto evaluate=[&](slg::mlhero15kk::V generated,Factors &f)->float {
            const auto l=s.fromLight?fixed:generated,e=s.fromLight?generated:fixed;
            float p,r;material->PdfMLHeroPrecise15kl(hitPoint,l,e,&p,&r);
            const auto q=MLHeroEvaluateRough15kk(s.fromLight,l,e,s.nc,s.nt,s.roughness,s.anisotropy,
                s.vndf,Spectrum(s.krBlack?0.f:1.f),Spectrum(s.ktBlack?0.f:1.f));
            f.wh=Vec(MLHeroVector15kk(q.pair.h));f.d=q.d;f.referenceD=q.d;
            f.specPdf=q.directSpec;f.probability=q.directProbability;f.jacobian=q.directJacobian;f.pdf=q.directPdf;
            f.angles=MLHeroCriticalAngles15kj(s,f.wh);return p;
        };
        out.localPdf=evaluate(sampled,out.local);
        out.roundTripPdf=evaluate(roundTripPrecise,out.roundTrip);
        out.sampledAngles=MLHeroCriticalAngles15kj(s,s.wh);return out;
    }

    material->Pdf(hitPoint,s.fromLight?fixed:sampled,s.fromLight?sampled:fixed,&out.localPdf,&reverse);
    material->Pdf(hitPoint,s.fromLight?fixed:roundTrip,s.fromLight?roundTrip:fixed,&out.roundTripPdf,&reverse);
    out.sampledAngles=MLHeroCriticalAngles15kj(s,s.wh);
    out.local=MLHeroRoughFactors15ki(s,sampled);
    out.roundTrip=MLHeroRoughFactors15ki(s,roundTrip);
    return out;
}

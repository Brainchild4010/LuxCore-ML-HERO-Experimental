#include "slg/engines/mlhero_metal_direction15ky.h"
#include "slg/engines/mlhero_surface_recovery15kx.h"
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

#include "slg/bsdf/bsdf.h"
#include "slg/scene/scene.h"
#include "slg/materials/glass.h"
#include "slg/materials/roughglass.h"
#include "slg/engines/mlhero_rough_frame15kl.h"
#include "slg/engines/mlhero_direction_trace15ko.h"
#include "slg/engines/mlhero_volume_direction15kp.h"
#include "slg/engines/mlhero_surface_trace15kv.h"
extern thread_local bool mlHeroEnabled;
#include <memory>
#include <cmath>
#include <limits>

// 15fe BEGIN
#include <cmath>
// 15fe END
#include "slg/materials/metal2.h"
// 15fd BEGIN diagnostic-only
#include "slg/materials/mattetranslucent.h"
// 15fd END diagnostic-only
#include "slg/materials/glossy2.h"
#include "slg/materials/mix.h"
#include "slg/engines/mlhero_mix_spectral15nm.h"
#include "slg/engines/mlhero_mix_closure15nm.h"
#include "slg/engines/mlhero_mix_sample_boundary15np.h"

// Explicit wavelength evaluation must preserve the renderer's sampled state.
extern thread_local float mlDispersionWaveLength;
extern thread_local bool mlDispersionUsed;
extern thread_local bool mlHeroPacketFallbackUsed;

using namespace luxrays;
using namespace slg;
using namespace std;

// 15ky BEGIN: one local-direction convention for every CPU Metal2 wrapper.
static float MLHeroMetalNG15ky(const Vector&v,const Normal&n,bool metal){
    if(!mlHeroEnabled||!metal)return Dot(v,n);
    using namespace slg::mlhero15kk;
    return float(slg::mlhero15kk::Dot(Unit(V(v)),V(n)));
}
static float MLHeroMetalAbsNG15ky(const Vector&v,const Normal&n,bool metal){
    return fabsf(MLHeroMetalNG15ky(v,n,metal));
}
static Vector MLHeroMetalLocal15ky(const Frame&f,const Vector&v,bool metal){
    return mlHeroEnabled&&metal?slg::mlhero15ky::Local(f,v):f.ToLocal(v);
}
// 15ky END

// 15kw BEGIN: read-only exceptional mesh audit; no extra texture or sampling calls.
static void MLHeroRawNormals15kw(const HitPoint&hp,slg::mlhero15kv::Trace&t){
    using slg::mlhero15kv::V;
    const ExtTriangleMesh*base=nullptr;
    const Transform*normalTransform=nullptr;
    switch(hp.mesh->GetType()){
        case TYPE_EXT_TRIANGLE:base=static_cast<const ExtTriangleMesh*>(hp.mesh.get());break;
        case TYPE_EXT_TRIANGLE_INSTANCE:{
            const auto*instance=static_cast<const ExtInstanceTriangleMesh*>(hp.mesh.get());
            base=&instance->GetExtTriangleMesh();normalTransform=&instance->GetTransformation();break;
        }
        case TYPE_EXT_TRIANGLE_MOTION:
            base=&static_cast<const ExtMotionTriangleMesh*>(hp.mesh.get())->GetExtTriangleMesh();
            normalTransform=&hp.localToWorld;break;
        default:return;
    }
    t.rawAvailable=true;t.rawNormals=base->HasNormals();t.transformApplied=normalTransform!=nullptr;
    const Transform&tr=normalTransform?*normalTransform:Transform::TRANS_IDENTITY;
    const auto matrix=tr.GetMatrix();
    for(unsigned r=0;r<4;++r)for(unsigned c=0;c<4;++c){
        t.matrix[r][c]=matrix.m[r][c];t.inverseMatrix[r][c]=tr.mInv.m[r][c];
    }
    const auto&triangle=base->GetTriangles()[hp.triangleIndex];
    for(unsigned i=0;i<3;++i){
        const unsigned vi=triangle.v[i];t.rawVertex[i]=V::From(base->GetVertices()[vi]);
        if(t.rawNormals){
            const Normal n=base->GetNormals()[vi];t.rawNormal[i]=V::From(n);
            if(!t.rawNormal[i].Finite())t.rawNonfiniteMask|=1u<<i;
            else if(t.rawNormal[i].Norm2()==0.)t.rawZeroMask|=1u<<i;
            const Normal signedN=base->GetShadeNormal(Transform::TRANS_IDENTITY,hp.triangleIndex,i);
            t.baseShade[i]=V::From(signedN);
            t.preNormalize[i]=V::From(normalTransform?tr*signedN:signedN);
        }
    }
    t.rawFaceCross=slg::mlhero15kv::Cross(t.rawVertex[1]-t.rawVertex[0],t.rawVertex[2]-t.rawVertex[0]);
    if(t.rawNormals){
        const float b1=hp.triangleBariCoord1,b2=hp.triangleBariCoord2,b0=1.f-b1-b2;
        const Normal*ns=base->GetNormals();
        const Normal blend=b0*ns[triangle.v[0]]+b1*ns[triangle.v[1]]+b2*ns[triangle.v[2]];
        t.rawBlendFloat=V::From(blend);
        t.rawBlendDouble=t.rawNormal[0]*(1.-t.b1-t.b2)+t.rawNormal[1]*t.b1+t.rawNormal[2]*t.b2;
        const Normal n=base->InterpolateTriNormal(Transform::TRANS_IDENTITY,hp.triangleIndex,b1,b2);
        t.baseInterpolated=V::From(n);
        t.interpolatedPreNormalize=V::From(normalTransform?tr*n:n);
    }
}
// 15kw END

// 15kv BEGIN: invoked only for CPU HERO BIDIR workers with Current diagnostics.
static void MLHeroSurfaceOrigin15kv(const HitPoint&hp,const Material&material,const Frame&frame,
        const slg::mlhero15kv::Snapshot&before,const slg::mlhero15kv::V&beforeDndu,
        const slg::mlhero15kv::V&beforeDndv,unsigned initKind,unsigned recoveryOutcome=0){
    using namespace slg::mlhero15kv;
    if(!reporter)return;
    Trace t;t.recoveryOutcome=recoveryOutcome;t.side=hp.fromLight?1u:0u;t.initKind=initKind;t.before=before;t.after=Snapshot::From(hp);
    t.frameX=V::From(frame.X);t.frameY=V::From(frame.Y);t.frameZ=V::From(frame.Z);
    if(!Select(t))return;
    t.materialID=material.GetID();t.materialType=(unsigned)material.GetType();t.materialName=material.GetName().c_str();
    const auto bump=material.GetBumpTexture();t.bump=bool(bump);t.bumpDistance=material.GetBumpSampleDistance();
    if(bump){t.bumpType=(unsigned)bump->GetType();t.bumpName=bump->GetName().c_str();}
    t.position=V::From(hp.p);t.fixed=V::From(hp.fixedDir);t.triangle=hp.triangleIndex;t.objectID=hp.objectID;
    t.b1=hp.triangleBariCoord1;t.b2=hp.triangleBariCoord2;
    t.beforeDndu=beforeDndu;t.beforeDndv=beforeDndv;t.afterDndu=V::From(hp.dndu);t.afterDndv=V::From(hp.dndv);
    if(hp.mesh){
        MLHeroRawNormals15kw(hp,t);
        t.meshType=(unsigned)hp.mesh->GetType();t.vertexNormals=hp.mesh->HasNormals();
        const auto&tri=hp.mesh->GetTriangles()[hp.triangleIndex];
        for(unsigned i=0;i<3;++i){
            t.vertexIndex[i]=tri.v[i];t.vertex[i]=V::From(hp.mesh->GetVertex(hp.localToWorld,tri.v[i]));
            if(t.vertexNormals)t.normal[i]=V::From(hp.mesh->GetShadeNormal(hp.localToWorld,hp.triangleIndex,i));
        }
        t.weightedNormal=t.normal[0]*(1.-t.b1-t.b2)+t.normal[1]*t.b1+t.normal[2]*t.b2;
        t.faceCross=Cross(t.vertex[1]-t.vertex[0],t.vertex[2]-t.vertex[0]);
    }
    reporter(t);
}
// 15kv END

// 15kx BEGIN: CPU HERO recovery for invalid interpolated surface normals.
// Work on a copy: a bad geometry normal or failed basis leaves the hit unchanged.
static bool MLHeroRecoverNormal15kx(HitPoint&hp){
    using slg::mlhero15kv::V;using slg::mlhero15kv::BadAxis;
    const V g=V::From(hp.geometryN);
    if(BadAxis(g)||!hp.mesh)return false;
    const double invLength=1./std::sqrt(g.Norm2());
    const Normal flat(float(g.x*invLength),float(g.y*invLength),float(g.z*invLength));
    if(BadAxis(V::From(flat)))return false;
    HitPoint recovered=hp;
    recovered.interpolatedN=flat;recovered.shadeN=flat;
    // Preserve native UV tangent construction; invalid source-normal derivatives
    // are replaced by zero because this fallback uses a locally flat normal.
    recovered.mesh->GetDifferentials(recovered.localToWorld,recovered.triangleIndex,
        flat,0,&recovered.dpdu,&recovered.dpdv,&recovered.dndu,&recovered.dndv);
    recovered.dndu=Normal(0.f,0.f,0.f);recovered.dndv=Normal(0.f,0.f,0.f);
    auto frameValid=[](const Frame&f){return !BadAxis(V::From(f.X))&&!BadAxis(V::From(f.Y))&&!BadAxis(V::From(f.Z));};
    if(!V::From(recovered.dpdu).Finite()||!V::From(recovered.dpdv).Finite()||!frameValid(recovered.GetFrame())){
        CoordinateSystem(Vector(flat),&recovered.dpdu,&recovered.dpdv);
    }
    if(!V::From(recovered.dpdu).Finite()||!V::From(recovered.dpdv).Finite()||!frameValid(recovered.GetFrame()))return false;
    hp.interpolatedN=recovered.interpolatedN;hp.shadeN=recovered.shadeN;
    hp.dpdu=recovered.dpdu;hp.dpdv=recovered.dpdv;hp.dndu=recovered.dndu;hp.dndv=recovered.dndv;
    return true;
}
static void MLHeroGuardNormal15kx(HitPoint&hp,const Material&material,unsigned initKind){
    using namespace slg::mlhero15kv;
    if(!mlHeroEnabled||!BadAxis(V::From(hp.interpolatedN)))return;
    const HitPoint original=hp;
    const bool repaired=MLHeroRecoverNormal15kx(hp);
    if(reporter){
        const unsigned side=hp.fromLight?1u:0u;
        slg::mlhero15kx::counters[side][0].fetch_add(1ull,std::memory_order_relaxed);
        slg::mlhero15kx::counters[side][repaired?1:2].fetch_add(1ull,std::memory_order_relaxed);
        // Report original evidence before recovery/material bump, with outcome.
        if(repaired)MLHeroSurfaceOrigin15kv(original,material,original.GetFrame(),Snapshot::From(original),
            V::From(original.dndu),V::From(original.dndv),initKind,repaired?1u:2u);
    }
}
// 15kx END

// Used when hitting a surface
void BSDF::Init(
		const bool fixedFromLight, const bool throughShadowTransparency,
		SceneConstRef scene, const Ray &ray, const RayHit &rayHit,
		const float passThroughEvent, const PathVolumeInfo *volInfo) {
	// Get the scene object
	sceneObject = &scene.GetObjects().GetSceneObject(rayHit.meshIndex);

	// Get the mesh
	auto& mesh = sceneObject->GetExtMesh();
	mesh.GetLocal2World(ray.time, hitPoint.localToWorld);

	hitPoint.Init(fixedFromLight, throughShadowTransparency,
			scene, rayHit.meshIndex, rayHit.triangleIndex,
			ray(rayHit.t), -ray.d,
			rayHit.b1, rayHit.b2,
			passThroughEvent);

	// Get the material
	material = &sceneObject->GetMaterial();
    MLHeroGuardNormal15kx(hitPoint,*material,0u);

	// Set interior and exterior volumes
	volInfo->SetHitPointVolumes(hitPoint,
		material->GetInteriorVolume(hitPoint, hitPoint.passThroughEvent),
		material->GetExteriorVolume(hitPoint, hitPoint.passThroughEvent),
		scene.HasDefaultWorldVolume() ?
		VolumeConstPtr(&scene.GetDefaultWorldVolume()) :
		VolumeConstPtr(nullptr)
	);

	// Check if it is a light source
	if (material->IsLightSource())
		triangleLightSource =
			&scene.GetLightSources().GetLightSourceByMeshAndTriIndex(
				rayHit.meshIndex, rayHit.triangleIndex
			);
	else
		triangleLightSource = nullptr;

    const bool surfaceAudit15kv=mlHeroEnabled&&slg::mlhero15kv::reporter;
    slg::mlhero15kv::Snapshot before15kv;slg::mlhero15kv::V dndu15kv,dndv15kv;
    if(surfaceAudit15kv){before15kv=slg::mlhero15kv::Snapshot::From(hitPoint);
        dndu15kv=slg::mlhero15kv::V::From(hitPoint.dndu);dndv15kv=slg::mlhero15kv::V::From(hitPoint.dndv);}
	// Apply bump or normal mapping (15kv observed)
	material->Bump(&hitPoint);

	// Build the local reference system
	frame = hitPoint.GetFrame();
    if(surfaceAudit15kv)MLHeroSurfaceOrigin15kv(hitPoint,*material,frame,before15kv,dndu15kv,dndv15kv,0u);
}

// Used when have a point of a surface
void BSDF::Init(
		SceneConstRef scene,
		const u_int meshIndex,
		const u_int triangleIndex,
		const Point &surfacePoint,
		const float surfacePointBary1,
		const float surfacePointBary2,
		const float time,
		const float passThroughEvent,
		const PathVolumeInfo *volInfo
) {
	// Get the scene object
	sceneObject = &scene.GetObjects().GetSceneObject(meshIndex);

	// Get the mesh
	auto& mesh = sceneObject->GetExtMesh();
	mesh.GetLocal2World(time, hitPoint.localToWorld);

	const Vector fixedDir = Vector(mesh.GetGeometryNormal(hitPoint.localToWorld, triangleIndex));
	hitPoint.Init(false, false,
			scene, meshIndex, triangleIndex,
			surfacePoint, fixedDir,
			surfacePointBary1, surfacePointBary2, passThroughEvent);

	// Get the material
	material = &sceneObject->GetMaterial();
    MLHeroGuardNormal15kx(hitPoint,*material,1u);

	// Set interior and exterior volumes
	volInfo->SetHitPointVolumes(hitPoint,
			material->GetInteriorVolume(hitPoint, hitPoint.passThroughEvent),
			material->GetExteriorVolume(hitPoint, hitPoint.passThroughEvent),
			scene.HasDefaultWorldVolume() ?
				VolumeConstPtr(&scene.GetDefaultWorldVolume()) :
				VolumeConstPtr()
	);

	// Check if it is a light source
	if (material->IsLightSource())
		triangleLightSource = &scene.GetLightSources().GetLightSourceByMeshAndTriIndex(meshIndex, triangleIndex);
	else
		triangleLightSource = nullptr;

    const bool surfaceAudit15kv=mlHeroEnabled&&slg::mlhero15kv::reporter;
    slg::mlhero15kv::Snapshot before15kv;slg::mlhero15kv::V dndu15kv,dndv15kv;
    if(surfaceAudit15kv){before15kv=slg::mlhero15kv::Snapshot::From(hitPoint);
        dndu15kv=slg::mlhero15kv::V::From(hitPoint.dndu);dndv15kv=slg::mlhero15kv::V::From(hitPoint.dndv);}
	// Apply bump or normal mapping
	material->Bump(&hitPoint);

	// Build the local reference system
	frame = hitPoint.GetFrame();
    if(surfaceAudit15kv)MLHeroSurfaceOrigin15kv(hitPoint,*material,frame,before15kv,dndu15kv,dndv15kv,1u);
}

// Used when hitting a volume scatter point
void BSDF::Init(
	const bool fixedFromLight,
	const bool throughShadowTransparency,
	SceneConstRef scene,
	const luxrays::Ray &ray,
	VolumeConstRef volume,
	const float t,
	const float passThroughEvent
) {
	hitPoint.fromLight = fixedFromLight;
	hitPoint.throughShadowTransparency = throughShadowTransparency;
	hitPoint.passThroughEvent = passThroughEvent;

	hitPoint.p = ray(t);
	hitPoint.fixedDir = mlHeroEnabled ? slg::mlhero15kp::Normalize(-ray.d) : -ray.d;

	sceneObject = nullptr;
	material = &volume;

	hitPoint.geometryN = Normal(hitPoint.fixedDir);
	hitPoint.interpolatedN = hitPoint.geometryN;
	hitPoint.shadeN = hitPoint.geometryN;

	hitPoint.intoObject = true;
	hitPoint.interiorVolume = &volume;
	hitPoint.exteriorVolume = &volume;

	triangleLightSource = nullptr;

	hitPoint.defaultUV = UV(0.f, 0.f);

	CoordinateSystem(Vector(hitPoint.shadeN), &hitPoint.dpdu, &hitPoint.dpdv);
	hitPoint.dndu = Normal();
	hitPoint.dndv = Normal();

	hitPoint.mesh = nullptr;
	hitPoint.triangleIndex = NULL_INDEX;
	hitPoint.triangleBariCoord1 = 0.f;
	hitPoint.triangleBariCoord2 = 0.f;

	hitPoint.objectID = NULL_INDEX;

	// Build the local reference system
	frame.SetFromZ(hitPoint.shadeN);
    if(mlHeroEnabled) {
        const slg::mlhero15kp::Basis unitFrame(hitPoint.shadeN);
        if(unitFrame.valid) {
            frame.X=slg::mlhero15kp::Cast<Vector>(unitFrame.x);
            frame.Y=slg::mlhero15kp::Cast<Vector>(unitFrame.y);
            frame.Z=slg::mlhero15kp::Cast<Vector>(unitFrame.z);
        }
    }
}

void BSDF::MoveHitPoint(const Point &p, const Normal &n) {
	hitPoint.p = p;
	hitPoint.geometryN = n;
	hitPoint.interpolatedN = n;
	hitPoint.shadeN = n;

	Vector x, y;
	CoordinateSystem(Vector(n), &x, &y);
	frame = Frame(x, y, n);
}

bool BSDF::IsAlbedoEndPoint(const AlbedoSpecularSetting albedoSpecularSetting,
		const float albedoSpecularGlossinessThreshold) const {
	const BSDFEvent eventTypes = GetEventTypes();
	if (!IsDelta() && !((eventTypes & GLOSSY) && (GetGlossiness() < albedoSpecularGlossinessThreshold)))
		return true;

	switch (albedoSpecularSetting) {
		case NO_REFLECT_TRANSMIT:
			return true;
		case ONLY_REFLECT:
			return !((eventTypes & REFLECT) && !(eventTypes & TRANSMIT));
		case ONLY_TRANSMIT:
			return !(!(eventTypes & REFLECT) && (eventTypes & TRANSMIT));
		case REFLECT_TRANSMIT:
			return !((eventTypes & REFLECT) || (eventTypes & TRANSMIT));
		default:
			throw runtime_error("Unknown AlbedoSpecularSetting in BSDF::IsAlbedoEndPoint(): " + ToString(albedoSpecularSetting));
	}
}

bool BSDF::IsCameraInvisible() const {
	return (sceneObject) ? sceneObject->IsCameraInvisible() : false;
}

u_int BSDF::GetObjectID() const {
	return (sceneObject) ? sceneObject->GetID() : std::numeric_limits<u_int>::max();
}

static string MaterialNULLptrName = "NULL pointer";

const string &BSDF::GetMaterialName() const {
	if (material)
		return material->GetName();
	else
		return MaterialNULLptrName;
}

Spectrum BSDF::Albedo() const {
	return material->Albedo(hitPoint);
}

Spectrum BSDF::EvaluateTotal() const {
	return material->EvaluateTotal(hitPoint);
}

bool BSDF::HasBakeMap(const BakeMapType type) const {
	return sceneObject && sceneObject->HasBakeMap(type);
}

Spectrum BSDF::GetBakeMapValue() const {
	return sceneObject->GetBakeMapValue(hitPoint.GetUV(sceneObject->GetBakeMapUVIndex()));
}

//------------------------------------------------------------------------------
// "A Microfacet-Based Shadowing Function to Solve the Bump Terminator Problem"
// by Alejandro Conty Estevez, Pascal Lecocq, and Clifford Stein
// http://www.aconty.com/pdf/bump-terminator-nvidia2019.pdf
//------------------------------------------------------------------------------

// Return alpha ^2 parameter from normal divergence
//static float ShadowTerminatorAvoidanceAlpha2(const Normal &Ni, const Normal &Ns) {
//	const float cos_d = Min(fabsf(Dot(Ni , Ns)), 1.f);
//	const float tan2_d = (1.f - cos_d * cos_d) / (cos_d * cos_d);
//
//	return Clamp(.125f * tan2_d, 0.f, 1.f);
//}

// Shadowing factor
//static float ShadowTerminatorAvoidanceFactor(const Normal &Ni, const Normal &Ns,
//		const Vector &lightDir) {
//	const float alpha2 = ShadowTerminatorAvoidanceAlpha2(Ni, Ns);
//	if (alpha2 > 0.f) {
//		const float cos_i = Max(fabsf(Dot(Ni , lightDir)), 1e-6f);
//		const float tan2_i = (1.f - cos_i * cos_i) / (cos_i * cos_i);
//
//		return 2.f / (1.f + sqrtf(1.f + alpha2 * tan2_i));
//	} else
//		return 1.f;
//}

//------------------------------------------------------------------------------
// "Taming the Shadow Terminator"
// by Matt Jen-Yuan Chiang, Yining Karl Li and Brent Burley
// https://www.yiningkarlli.com/projects/shadowterminator.html
//------------------------------------------------------------------------------

static float ShadowTerminatorAvoidanceFactor(const Normal &Ni, const Normal &Ns,
		const Vector &lightDir) {
	const float dotNsLightDir = Dot(Ns, lightDir);
	if (dotNsLightDir <= 0.f)
		return 0.f;

	const float dotNiNs = Dot(Ni, Ns);
	if (dotNiNs <= 0.f)
		return 0.f;

	const float G = Min(1.f, Dot(Ni, lightDir) / (dotNsLightDir * dotNiNs));
	if (G <= 0.f)
		return 0.f;
	
	const float G2 = G * G;
	const float G3 = G2 * G;

	return -G3 + G2 + G;
}

Spectrum BSDF::Evaluate(const Vector &generatedDir,
		BSDFEvent *event, float *directPdfW, float *reversePdfW) const {
	const Vector &eyeDir = hitPoint.fromLight ? generatedDir : hitPoint.fixedDir;
	const Vector &lightDir = hitPoint.fromLight ? hitPoint.fixedDir : generatedDir;

	const float dotLightDirNG = MLHeroMetalNG15ky(lightDir,hitPoint.geometryN,material->GetType()==METAL2);
	const float absDotLightDirNG = fabsf(dotLightDirNG);
	const float dotEyeDirNG = MLHeroMetalNG15ky(eyeDir,hitPoint.geometryN,material->GetType()==METAL2);
	const float absDotEyeDirNG = fabsf(dotEyeDirNG);

	if (!IsVolume()) {
		// These kind of tests make sense only for materials

		// Avoid glancing angles
		if ((absDotLightDirNG < DEFAULT_COS_EPSILON_STATIC) ||
				(absDotEyeDirNG < DEFAULT_COS_EPSILON_STATIC))
			return Spectrum();

		// Check geometry normal and light direction side
		const float sideTestNG = dotEyeDirNG * dotLightDirNG;
		const BSDFEvent matEvents = material->GetEventTypes();
		if (((sideTestNG > 0.f) && !(matEvents & REFLECT)) ||
				((sideTestNG < 0.f) && !(matEvents & TRANSMIT)))
			return Spectrum();

		// Check shading normal and light direction side
		const float sideTestIS = Dot(eyeDir, hitPoint.interpolatedN) * Dot(lightDir, hitPoint.interpolatedN);
		if (((sideTestIS > 0.f) && !(matEvents & REFLECT)) ||
				((sideTestIS < 0.f) && !(matEvents & TRANSMIT)))
			return Spectrum();
	}

	const Vector localLightDir = MLHeroMetalLocal15ky(frame,lightDir,material->GetType()==METAL2);
	const Vector localEyeDir = MLHeroMetalLocal15ky(frame,eyeDir,material->GetType()==METAL2);
    Spectrum result;
    if(mlHeroEnabled && material->GetType()==ROUGHGLASS) {
        const slg::mlhero15kl::Frame precise(frame);
        result=static_cast<const RoughGlassMaterial &>(*material).EvaluateMLHeroPrecise15kl(hitPoint,
            precise.ToLocal(lightDir),precise.ToLocal(eyeDir),event,directPdfW,reversePdfW);
    }else result=material->Evaluate(hitPoint,localLightDir,localEyeDir,event,directPdfW,reversePdfW);
	verify (!result.IsNaN() && !result.IsInf());
	if (result.Black())
		return result;

	if (!IsVolume()) {
		// Shadow terminator artefact avoidance
		if ((*event & REFLECT) &&
				(*event & (DIFFUSE | GLOSSY)) &&
				(hitPoint.shadeN != hitPoint.interpolatedN))
			result *= ShadowTerminatorAvoidanceFactor(hitPoint.GetLandingInterpolatedN(),
					hitPoint.GetLandingShadeN(), lightDir);

		// Adjoint BSDF (not for volumes)
		if (hitPoint.fromLight)
			result *= (absDotEyeDirNG / absDotLightDirNG);
	}

	return result;
}

bool BSDF::GetMLHeroMixMetadata15nl(MaterialType *typeA, MaterialType *typeB,
		u_int *idA, u_int *idB, float *mixFactor) const {
	if (!typeA || !typeB || !idA || !idB || !mixFactor ||
			!material || material->GetType() != MIX)
		return false;

	const MixMaterial &mix = static_cast<const MixMaterial &>(*material);
	const Material &matA = mix.GetMaterialA();
	const Material &matB = mix.GetMaterialB();
	*typeA = matA.GetType();
	*typeB = matB.GetType();
	*idA = matA.GetID();
	*idB = matB.GetID();
	*mixFactor = Clamp(mix.GetMixFactor().GetFloatValue(hitPoint), 0.f, 1.f);
	return std::isfinite(*mixFactor);
}

bool BSDF::EvaluateMLHeroMixComponents15nl(const Vector &generatedDir,
		Spectrum *weightedA, Spectrum *weightedB, Spectrum *combinedResult,
		BSDFEvent *eventA, BSDFEvent *eventB,
		float *directPdfA, float *directPdfB,
		float *reversePdfA, float *reversePdfB) const {
	if (!weightedA || !weightedB || !combinedResult || !eventA || !eventB ||
			!directPdfA || !directPdfB || !reversePdfA || !reversePdfB ||
			!material || material->GetType() != MIX)
		return false;

	*weightedA = Spectrum();
	*weightedB = Spectrum();
	*combinedResult = Spectrum();
	*eventA = NONE;
	*eventB = NONE;
	*directPdfA = *directPdfB = 0.f;
	*reversePdfA = *reversePdfB = 0.f;

	const MixMaterial &mix = static_cast<const MixMaterial &>(*material);
	const Material &matA = mix.GetMaterialA();
	const Material &matB = mix.GetMaterialB();
	const float weightB = Clamp(mix.GetMixFactor().GetFloatValue(hitPoint), 0.f, 1.f);
	const float weightA = 1.f - weightB;

	const Vector &eyeDir = hitPoint.fromLight ? generatedDir : hitPoint.fixedDir;
	const Vector &lightDir = hitPoint.fromLight ? hitPoint.fixedDir : generatedDir;
	const Vector localLightDir = frame.ToLocal(lightDir);
	const Vector localEyeDir = frame.ToLocal(eyeDir);
	const Frame mixFrame(hitPoint.GetFrame());
	const bool transmit = Sgn(localLightDir.z) != Sgn(localEyeDir.z);

	if (weightA > 0.f && ((!transmit && (matA.GetEventTypes() & REFLECT)) ||
			(transmit && (matA.GetEventTypes() & TRANSMIT)))) {
		HitPoint childHitPoint(hitPoint);
		matA.Bump(&childHitPoint);
		const Frame childFrame(childHitPoint.GetFrame());
		const Vector childLight = childFrame.ToLocal(mixFrame.ToWorld(localLightDir));
		const Vector childEye = childFrame.ToLocal(mixFrame.ToWorld(localEyeDir));
		const Spectrum childValue = matA.Evaluate(childHitPoint, childLight, childEye,
			eventA, directPdfA, reversePdfA);
		*weightedA = weightA * childValue;
	}

	if (weightB > 0.f && ((!transmit && (matB.GetEventTypes() & REFLECT)) ||
			(transmit && (matB.GetEventTypes() & TRANSMIT)))) {
		HitPoint childHitPoint(hitPoint);
		matB.Bump(&childHitPoint);
		const Frame childFrame(childHitPoint.GetFrame());
		const Vector childLight = childFrame.ToLocal(mixFrame.ToWorld(localLightDir));
		const Vector childEye = childFrame.ToLocal(mixFrame.ToWorld(localEyeDir));
		const Spectrum childValue = matB.Evaluate(childHitPoint, childLight, childEye,
			eventB, directPdfB, reversePdfB);
		*weightedB = weightB * childValue;
	}

	*combinedResult = *weightedA + *weightedB;
	return !weightedA->IsNaN() && !weightedA->IsInf() &&
		!weightedB->IsNaN() && !weightedB->IsInf() &&
		!combinedResult->IsNaN() && !combinedResult->IsInf();
}

// 15nm BEGIN: exact, fail-closed wavelength evaluation behind MixMaterial.
static bool MLHeroPrepareMixChild15nm(const Material &childMaterial,
		const float weight, const HitPoint &outerHitPoint,
		const Frame &outerFrame, const Vector &outerLocalLightDir,
		const Vector &outerLocalEyeDir, const bool transmit,
		const Vector *anchoredLocalLightDir15ns,
		const Vector *anchoredLocalEyeDir15ns,
		MLHeroMixChildSpectral15nm *child, u_int *failure15no) {
	if (failure15no)
		*failure15no = MLHERO_MIX_CHILD_OK15NO;
	if (!child || !std::isfinite(weight) || weight < 0.f) {
		if (failure15no)
			*failure15no = MLHERO_MIX_CHILD_INVALID_INPUT15NO;
		return false;
	}

	child->material = &childMaterial;
	child->weight = weight;
	child->nativeValue = Spectrum(0.f);
	child->event = NONE;
	child->directPdfW = 0.f;
	child->reversePdfW = 0.f;
	child->route = MLHERO_MIX_CHILD_INACTIVE15NM;
	if (!(weight > 0.f))
		return true;

	// Nested MIX needs a recursive endpoint representation.  Do not silently
	// flatten it as an achromatic value in this first visible fix.
	if (childMaterial.GetType() == MIX) {
		if (failure15no)
			*failure15no = MLHERO_MIX_CHILD_NESTED_MIX15NO;
		return false;
	}

	child->hitPoint = outerHitPoint;
	childMaterial.Bump(&child->hitPoint);
	const Frame childFrame(child->hitPoint.GetFrame());
	if (anchoredLocalLightDir15ns && anchoredLocalEyeDir15ns) {
		child->localLightDir = *anchoredLocalLightDir15ns;
		child->localEyeDir = *anchoredLocalEyeDir15ns;
	} else {
		child->localLightDir = childFrame.ToLocal(
			outerFrame.ToWorld(outerLocalLightDir));
		child->localEyeDir = childFrame.ToLocal(
			outerFrame.ToWorld(outerLocalEyeDir));
	}
	// Native Mix::Sample classifies its evaluated second child in the child's
	// exact local frame. Preserve that rule for an anchored sampled endpoint;
	// Evaluate-only endpoints retain Mix::Evaluate's outer-side policy.
	const bool routeTransmit15ns = anchoredLocalLightDir15ns ?
		(Sgn(child->localLightDir.z) != Sgn(child->localEyeDir.z)) : transmit;
	const BSDFEvent childEvents = childMaterial.GetEventTypes();
	if ((!routeTransmit15ns && !(childEvents & REFLECT)) ||
			(routeTransmit15ns && !(childEvents & TRANSMIT)))
		return true;
	child->nativeValue = childMaterial.Evaluate(child->hitPoint,
		child->localLightDir, child->localEyeDir, &child->event,
		&child->directPdfW, &child->reversePdfW);
	if (child->nativeValue.IsNaN() || child->nativeValue.IsInf()) {
		if (failure15no)
			*failure15no = MLHERO_MIX_CHILD_NATIVE_NONFINITE15NO;
		return false;
	}

	if (child->nativeValue.Black()) {
		child->route = MLHERO_MIX_CHILD_SCALAR15NM;
		return true;
	}

	switch (childMaterial.GetType()) {
		case METAL2:
			child->route = MLHERO_MIX_CHILD_METAL215NM;
			return true;
		case GLOSSY2: {
			const Glossy2Material &glossy2 =
				static_cast<const Glossy2Material &>(childMaterial);
			if (!glossy2.PrepareMLHeroSpectralEndpoint15nk(child->hitPoint,
					child->localLightDir, child->localEyeDir,
					&child->glossy2)) {
				if (failure15no)
					*failure15no = MLHERO_MIX_CHILD_GLOSSY2_PREPARE15NO;
				return false;
			}
			if (child->glossy2.event != child->event) {
				if (failure15no)
					*failure15no = MLHERO_MIX_CHILD_EVENT_MISMATCH15NO;
				return false;
			}
			child->route = MLHERO_MIX_CHILD_GLOSSY215NM;
			return true;
		}
		case VELVET: {
			const VelvetMaterial &velvet =
				static_cast<const VelvetMaterial &>(childMaterial);
			if (!velvet.PrepareMLHeroSpectralEndpoint15nn(child->hitPoint,
					child->localLightDir, child->localEyeDir,
					&child->velvet)) {
				if (failure15no)
					*failure15no = MLHERO_MIX_CHILD_VELVET_PREPARE15NO;
				return false;
			}
			if (child->velvet.event != child->event) {
				if (failure15no)
					*failure15no = MLHERO_MIX_CHILD_EVENT_MISMATCH15NO;
				return false;
			}
			child->route = MLHERO_MIX_CHILD_VELVET15NN;
			return true;
		}
		default: {
			// A genuinely scalar native child has no RGB chroma to reconstruct.
			// Chromatic unsupported children remain on the original RGB path.
			const float scale = Max(1.f, Max(fabsf(child->nativeValue.c[0]),
				Max(fabsf(child->nativeValue.c[1]),
					fabsf(child->nativeValue.c[2]))));
			if ((fabsf(child->nativeValue.c[0] - child->nativeValue.c[1]) >
					1e-6f * scale) ||
					(fabsf(child->nativeValue.c[0] - child->nativeValue.c[2]) >
					1e-6f * scale)) {
				if (failure15no)
					*failure15no = MLHERO_MIX_CHILD_UNSUPPORTED_CHROMATIC15NO;
				return false;
			}
			child->route = MLHERO_MIX_CHILD_SCALAR15NM;
			return true;
		}
	}
}

bool BSDF::PrepareMLHeroMixSpectralEndpoint15nm(
		const Vector &generatedDir, const BSDFEvent expectedEvent,
		const Spectrum &nativeValue, const bool sampled,
		const float nativePdfW,
		MLHeroMixSpectralEndpoint15nm *prepared,
		MLHeroMixDiagnostic15no *diagnostic15no) const {
	if (diagnostic15no)
		*diagnostic15no = MLHeroMixDiagnostic15no();
	if (diagnostic15no)
		diagnostic15no->sampled = sampled;
	if (!prepared || !material || material->GetType() != MIX) {
		if (diagnostic15no)
			diagnostic15no->prepareFailure =
				MLHERO_MIX_PREPARE_INVALID_INPUT15NO;
		return false;
	}
	if (expectedEvent & SPECULAR) {
		if (diagnostic15no)
			diagnostic15no->prepareFailure =
				MLHERO_MIX_PREPARE_SPECULAR_EVENT15NO;
		return false;
	}
	if ((sampled && (!std::isfinite(nativePdfW) || !(nativePdfW > 0.f))) ||
			nativeValue.IsNaN() || nativeValue.IsInf()) {
		if (diagnostic15no)
			diagnostic15no->prepareFailure =
				MLHERO_MIX_PREPARE_INVALID_SAMPLE15NO;
		return false;
	}
	*prepared = MLHeroMixSpectralEndpoint15nm();
	prepared->sampled = sampled;
	prepared->nativePdfW = sampled ? nativePdfW : 1.f;

	const Vector &eyeDir = hitPoint.fromLight ? generatedDir : hitPoint.fixedDir;
	const Vector &lightDir = hitPoint.fromLight ? hitPoint.fixedDir : generatedDir;
	const float absDotLightDirNG = AbsDot(lightDir, hitPoint.geometryN);
	const float absDotEyeDirNG = AbsDot(eyeDir, hitPoint.geometryN);
	if (diagnostic15no) {
		diagnostic15no->absDotLightDirNG = absDotLightDirNG;
		diagnostic15no->absDotEyeDirNG = absDotEyeDirNG;
	}
	const bool geometryGateWouldReject15np =
		absDotLightDirNG < DEFAULT_COS_EPSILON_STATIC ||
		absDotEyeDirNG < DEFAULT_COS_EPSILON_STATIC;
	if (diagnostic15no)
		diagnostic15no->geometryGateWouldReject =
			geometryGateWouldReject15np;
	const slg::mlhero15np::BoundaryPolicy boundaryPolicy15np =
		slg::mlhero15np::SelectBoundaryPolicy(sampled);
	if (boundaryPolicy15np.applyGeometryCosGate &&
			geometryGateWouldReject15np) {
		if (diagnostic15no)
			diagnostic15no->prepareFailure =
				MLHERO_MIX_PREPARE_GEOMETRY_COS15NO;
		return false;
	}

	const BSDFEvent mixEvents = material->GetEventTypes();
	const float sideTestNG = Dot(eyeDir, hitPoint.geometryN) *
		Dot(lightDir, hitPoint.geometryN);
	const float sideTestIS = Dot(eyeDir, hitPoint.interpolatedN) *
		Dot(lightDir, hitPoint.interpolatedN);
	if (diagnostic15no) {
		diagnostic15no->sideTestNG = sideTestNG;
		diagnostic15no->sideTestIS = sideTestIS;
	}
	const bool sideEventGateWouldReject15np =
		((sideTestNG > 0.f) && !(mixEvents & REFLECT)) ||
			((sideTestNG < 0.f) && !(mixEvents & TRANSMIT)) ||
			((sideTestIS > 0.f) && !(mixEvents & REFLECT)) ||
			((sideTestIS < 0.f) && !(mixEvents & TRANSMIT));
	if (diagnostic15no)
		diagnostic15no->sideEventGateWouldReject =
			sideEventGateWouldReject15np;
	if (boundaryPolicy15np.applySideEventGate &&
			sideEventGateWouldReject15np) {
		if (diagnostic15no)
			diagnostic15no->prepareFailure =
				MLHERO_MIX_PREPARE_SIDE_EVENT15NO;
		return false;
	}

	const Vector outerLocalLightDir = frame.ToLocal(lightDir);
	const Vector outerLocalEyeDir = frame.ToLocal(eyeDir);
	const Frame outerFrame(hitPoint.GetFrame());
	const bool transmit = Sgn(outerLocalLightDir.z) !=
		Sgn(outerLocalEyeDir.z);
	const MixMaterial &mix = static_cast<const MixMaterial &>(*material);
	const float weightB = Clamp(mix.GetMixFactor().GetFloatValue(hitPoint),
		0.f, 1.f);
	const float weightA = 1.f - weightB;
	// 15ns: a sampled Mix bounce already computed the exact local direction
	// pair for both children. Reuse that native state so the wavelength endpoint
	// is evaluated at the same point on sharp lobes. Evaluate-only DLS and
	// connection endpoints deliberately retain their ordinary frame path.
	const u_int selectedChild15ns =
		(hitPoint.passThroughEvent < weightA) ? 0u : 1u;
	const Material *selectedMaterial15ns = selectedChild15ns == 0u ?
		&mix.GetMaterialA() : &mix.GetMaterialB();
	slg::mlhero15nr::Capture capture15ns;
	bool captureMatched15ns = false;
	if (sampled) {
		if (diagnostic15no)
			diagnostic15no->directionAnchorRequested15ns = true;
		captureMatched15ns = slg::mlhero15nr::ReadMatching(material,
			selectedMaterial15ns, selectedChild15ns,
			hitPoint.passThroughEvent, &capture15ns) &&
			capture15ns.fromLight == hitPoint.fromLight &&
			capture15ns.event == (u_int)expectedEvent;
		if (diagnostic15no) {
			diagnostic15no->directionCaptureMatched15ns = captureMatched15ns;
			diagnostic15no->directionAnchorApplied15ns = captureMatched15ns;
		}
	}
	Vector anchoredLight15ns[2];
	Vector anchoredEye15ns[2];
	if (captureMatched15ns) {
		for (u_int i = 0u; i < 2u; ++i) {
			anchoredLight15ns[i] = hitPoint.fromLight ?
				capture15ns.childFixedDir[i] : capture15ns.childSampledDir[i];
			anchoredEye15ns[i] = hitPoint.fromLight ?
				capture15ns.childSampledDir[i] : capture15ns.childFixedDir[i];
		}
	}
	if (diagnostic15no) {
		diagnostic15no->passThroughEvent15nq = hitPoint.passThroughEvent;
		diagnostic15no->childWeight15nq[0] = weightA;
		diagnostic15no->childWeight15nq[1] = weightB;
		diagnostic15no->selectedChild15nq =
			(hitPoint.passThroughEvent < weightA) ? 0u : 1u;
		diagnostic15no->selectedChildType15nq =
			(u_int)(diagnostic15no->selectedChild15nq == 0u ?
				mix.GetMaterialA().GetType() : mix.GetMaterialB().GetType());
	}
	u_int childFailure15no = MLHERO_MIX_CHILD_OK15NO;
	if (!MLHeroPrepareMixChild15nm(mix.GetMaterialA(), weightA, hitPoint,
			outerFrame, outerLocalLightDir, outerLocalEyeDir, transmit,
			captureMatched15ns ? &anchoredLight15ns[0] : nullptr,
			captureMatched15ns ? &anchoredEye15ns[0] : nullptr,
			&prepared->child[0], &childFailure15no)) {
		if (diagnostic15no) {
			diagnostic15no->prepareFailure = MLHERO_MIX_PREPARE_CHILD_A15NO;
			diagnostic15no->childFailure = childFailure15no;
			diagnostic15no->childIndex = 0u;
			diagnostic15no->childType = (u_int)mix.GetMaterialA().GetType();
		}
		return false;
	}
	if (!MLHeroPrepareMixChild15nm(mix.GetMaterialB(), weightB, hitPoint,
			outerFrame, outerLocalLightDir, outerLocalEyeDir, transmit,
			captureMatched15ns ? &anchoredLight15ns[1] : nullptr,
			captureMatched15ns ? &anchoredEye15ns[1] : nullptr,
			&prepared->child[1], &childFailure15no)) {
		if (diagnostic15no) {
			diagnostic15no->prepareFailure = MLHERO_MIX_PREPARE_CHILD_B15NO;
			diagnostic15no->childFailure = childFailure15no;
			diagnostic15no->childIndex = 1u;
			diagnostic15no->childType = (u_int)mix.GetMaterialB().GetType();
		}
		return false;
	}
	if (diagnostic15no) {
		for (u_int i = 0u; i < 2u; ++i) {
			const MLHeroMixChildSpectral15nm &child = prepared->child[i];
			diagnostic15no->childRoute15nq[i] = child.route;
			diagnostic15no->childDirectPdf15nq[i] = child.directPdfW;
			diagnostic15no->childReversePdf15nq[i] = child.reversePdfW;
			for (u_int c = 0u; c < 3u; ++c)
				diagnostic15no->childNative15nq[i][c] = child.nativeValue.c[c];
		}
	}

	Spectrum reconstructed(0.f);
	prepared->combinedEvent = NONE;
	for (u_int i = 0u; i < 2u; ++i) {
		const MLHeroMixChildSpectral15nm &child = prepared->child[i];
		if (child.route != MLHERO_MIX_CHILD_INACTIVE15NM &&
				!child.nativeValue.Black()) {
			reconstructed += child.weight * child.nativeValue;
			prepared->combinedEvent |= child.event;
		}
	}
	if (reconstructed.Black()) {
		if (diagnostic15no)
			diagnostic15no->prepareFailure =
				MLHERO_MIX_PREPARE_RECONSTRUCTED_BLACK15NO;
		return false;
	}

	prepared->wrapperFactor = 1.f;
	if ((expectedEvent & REFLECT) &&
			(expectedEvent & (DIFFUSE | GLOSSY)) &&
			(hitPoint.shadeN != hitPoint.interpolatedN))
		prepared->wrapperFactor *= ShadowTerminatorAvoidanceFactor(
			hitPoint.GetLandingInterpolatedN(), hitPoint.GetLandingShadeN(),
			lightDir);
	if (hitPoint.fromLight)
		prepared->wrapperFactor *= absDotEyeDirNG / absDotLightDirNG;
	if (diagnostic15no)
		diagnostic15no->wrapperFactor = prepared->wrapperFactor;
	if (!std::isfinite(prepared->wrapperFactor) ||
			prepared->wrapperFactor < 0.f) {
		if (diagnostic15no)
			diagnostic15no->prepareFailure =
				MLHERO_MIX_PREPARE_WRAPPER_INVALID15NO;
		return false;
	}

	reconstructed *= prepared->wrapperFactor / prepared->nativePdfW;
	prepared->nativeReconstructed = reconstructed;
	double reconstructed15nm[3], nativeValue15nm[3];
	for (u_int channel = 0u; channel < 3u; ++channel) {
		reconstructed15nm[channel] = reconstructed.c[channel];
		nativeValue15nm[channel] = nativeValue.c[channel];
	}
	const slg::mlhero15nm::Closure closure15nm =
		slg::mlhero15nm::AnalyzeClosure(reconstructed15nm,
			nativeValue15nm);
	prepared->closureRelativeError = float(closure15nm.relativeError);
	if (diagnostic15no)
		diagnostic15no->closureRelativeError =
			prepared->closureRelativeError;
	if (diagnostic15no && sampled) {
		double childA15nq[3], childB15nq[3];
		for (u_int channel = 0u; channel < 3u; ++channel) {
			childA15nq[channel] = prepared->child[0].nativeValue.c[channel];
			childB15nq[channel] = prepared->child[1].nativeValue.c[channel];
		}
		const slg::mlhero15nq::Analysis analysis15nq =
			slg::mlhero15nq::Analyze(weightA, weightB, childA15nq,
				childB15nq, prepared->child[0].directPdfW,
				prepared->child[1].directPdfW,
				prepared->child[0].reversePdfW,
				prepared->child[1].reversePdfW,
				prepared->wrapperFactor, prepared->nativePdfW,
				nativeValue15nm, hitPoint.passThroughEvent,
				hitPoint.fromLight);
		diagnostic15no->analysisFinite15nq = analysis15nq.finite;
		diagnostic15no->doubleClosureAccepted15nq =
			analysis15nq.doubleClosureAccepted;
		diagnostic15no->selectedChild15nq = analysis15nq.selectedChild;
		diagnostic15no->doubleClosureRelativeError15nq =
			analysis15nq.doubleClosureRelativeError;
		diagnostic15no->selectedNumeratorRelativeError15nq =
			analysis15nq.selectedNumeratorRelativeError;
		diagnostic15no->comparableMixturePdf15nq =
			analysis15nq.comparableMixturePdf;
		diagnostic15no->nativePdfRelativeError15nq =
			analysis15nq.nativePdfRelativeError;
		for (u_int channel = 0u; channel < 3u; ++channel) {
			diagnostic15no->doubleReconstructed15nq[channel] =
				analysis15nq.doubleReconstructed[channel];
			diagnostic15no->nativeNumerator15nq[channel] =
				analysis15nq.nativeNumerator[channel];
			diagnostic15no->inferredSelectedNumerator15nq[channel] =
				analysis15nq.inferredSelectedNumerator[channel];
			diagnostic15no->evaluatedSelectedNumerator15nq[channel] =
				analysis15nq.evaluatedSelectedNumerator[channel];
		}

		// 15nr: only a native closure rejection pays for this extra Evaluate.
		// Re-evaluate the selected child at the exact local direction produced by
		// its Sample(), before MixMaterial transforms it through world and back.
		if (!closure15nm.finite || !closure15nm.accepted) {
			const u_int selected15nr = diagnostic15no->selectedChild15nq;
			if (selected15nr < 2u) {
				const MLHeroMixChildSpectral15nm &child15nr =
					prepared->child[selected15nr];
				slg::mlhero15nr::Capture capture15nr;
				if (slg::mlhero15nr::ReadMatching(material,
						child15nr.material, selected15nr,
						hitPoint.passThroughEvent, &capture15nr)) {
					diagnostic15no->captureMatched15nr = true;
					diagnostic15no->capturedEvent15nr = capture15nr.event;
					diagnostic15no->roundTripEvent15nr = child15nr.event;
					diagnostic15no->childSamplePdf15nr = capture15nr.childPdfW;

					const Vector originalLight15nr = hitPoint.fromLight ?
						capture15nr.originalFixedDir :
						capture15nr.originalSampledDir;
					const Vector originalEye15nr = hitPoint.fromLight ?
						capture15nr.originalSampledDir :
						capture15nr.originalFixedDir;
					BSDFEvent originalEvent15nr = NONE;
					float originalDirectPdf15nr = 0.f;
					float originalReversePdf15nr = 0.f;
					const Spectrum originalEvaluate15nr =
						child15nr.material->Evaluate(child15nr.hitPoint,
							originalLight15nr, originalEye15nr,
							&originalEvent15nr, &originalDirectPdf15nr,
							&originalReversePdf15nr);
					const float originalPdf15nr = hitPoint.fromLight ?
						originalReversePdf15nr : originalDirectPdf15nr;
					diagnostic15no->originalEvent15nr = originalEvent15nr;
					diagnostic15no->originalEvaluatePdf15nr = originalPdf15nr;
					diagnostic15no->originalEvaluateFinite15nr =
						!originalEvaluate15nr.IsNaN() &&
						!originalEvaluate15nr.IsInf() &&
						std::isfinite(originalPdf15nr);

					const Vector roundTripFixed15nr = hitPoint.fromLight ?
						child15nr.localLightDir : child15nr.localEyeDir;
					const Vector roundTripSampled15nr = hitPoint.fromLight ?
						child15nr.localEyeDir : child15nr.localLightDir;
					diagnostic15no->fixedDirectionMaxError15nr =
						slg::mlhero15nr::VectorMaxAbsDifference(
							capture15nr.originalFixedDir, roundTripFixed15nr);
					diagnostic15no->sampledDirectionMaxError15nr =
						slg::mlhero15nr::VectorMaxAbsDifference(
							capture15nr.originalSampledDir, roundTripSampled15nr);
					diagnostic15no->sampledDirectionDot15nr =
						double(Dot(capture15nr.originalSampledDir,
							roundTripSampled15nr));

					double sampleNumerator15nr[3];
					double originalEvaluateArray15nr[3];
					double roundTripEvaluate15nr[3];
					for (u_int channel = 0u; channel < 3u; ++channel) {
						sampleNumerator15nr[channel] =
							capture15nr.childSampleNumerator.c[channel];
						originalEvaluateArray15nr[channel] =
							originalEvaluate15nr.c[channel];
						roundTripEvaluate15nr[channel] =
							child15nr.nativeValue.c[channel];
						diagnostic15no->originalFixedDir15nr[channel] =
							capture15nr.originalFixedDir[channel];
						diagnostic15no->originalSampledDir15nr[channel] =
							capture15nr.originalSampledDir[channel];
						diagnostic15no->roundTripFixedDir15nr[channel] =
							roundTripFixed15nr[channel];
						diagnostic15no->roundTripSampledDir15nr[channel] =
							roundTripSampled15nr[channel];
						diagnostic15no->childSampleNumerator15nr[channel] =
							capture15nr.childSampleNumerator.c[channel];
						diagnostic15no->originalEvaluate15nr[channel] =
							originalEvaluate15nr.c[channel];
						diagnostic15no->roundTripEvaluate15nr[channel] =
							child15nr.nativeValue.c[channel];
					}
					const slg::mlhero15nm::Closure sampleOriginal15nr =
						slg::mlhero15nm::AnalyzeClosure(sampleNumerator15nr,
							originalEvaluateArray15nr);
					const slg::mlhero15nm::Closure originalRoundTrip15nr =
						slg::mlhero15nm::AnalyzeClosure(originalEvaluateArray15nr,
							roundTripEvaluate15nr);
					const slg::mlhero15nm::Closure sampleRoundTrip15nr =
						slg::mlhero15nm::AnalyzeClosure(sampleNumerator15nr,
							roundTripEvaluate15nr);
					diagnostic15no->sampleVsOriginalPass15nr =
						sampleOriginal15nr.finite && sampleOriginal15nr.accepted;
					diagnostic15no->originalVsRoundTripPass15nr =
						originalRoundTrip15nr.finite &&
						originalRoundTrip15nr.accepted;
					diagnostic15no->sampleVsRoundTripPass15nr =
						sampleRoundTrip15nr.finite &&
						sampleRoundTrip15nr.accepted;
					diagnostic15no->sampleVsOriginalRelativeError15nr =
						sampleOriginal15nr.relativeError;
					diagnostic15no->originalVsRoundTripRelativeError15nr =
						originalRoundTrip15nr.relativeError;
					diagnostic15no->sampleVsRoundTripRelativeError15nr =
						sampleRoundTrip15nr.relativeError;
					const double pdfScale15nr = Max(1e-12,
						Max(fabs(double(capture15nr.childPdfW)),
							fabs(double(originalPdf15nr))));
					diagnostic15no->sampleVsOriginalPdfRelativeError15nr =
						fabs(double(capture15nr.childPdfW) -
							double(originalPdf15nr)) / pdfScale15nr;

					if (child15nr.route == MLHERO_MIX_CHILD_GLOSSY215NM) {
						const float wCoating15nr = SchlickBSDF_CoatingWeight(
							child15nr.glossy2.ks,
							capture15nr.originalFixedDir);
						diagnostic15no->glossyWBase15nr = 1.f - wCoating15nr;
						diagnostic15no->glossyBranch15nr =
							capture15nr.childPassThroughEvent <
								diagnostic15no->glossyWBase15nr ? 0u : 1u;
						diagnostic15no->glossyRoughness15nr =
							child15nr.glossy2.roughness;
						diagnostic15no->glossyAnisotropy15nr =
							child15nr.glossy2.anisotropy;
					}
				}
			}
		}
	}
	// The absolute term protects black/near-black endpoints; the relative term
	// admits only ordinary floating-point differences between Sample/Evaluate.
	if (!closure15nm.finite || !closure15nm.accepted) {
		if (diagnostic15no)
			diagnostic15no->prepareFailure =
				MLHERO_MIX_PREPARE_CLOSURE_REJECT15NO;
		return false;
	}

	prepared->valid = true;
	return true;
}

bool BSDF::EvaluateMLHeroMixPrepared15nm(
		const MLHeroMixSpectralEndpoint15nm &prepared,
		const float waveLength, const bool glossy2BasisCompensation,
		const bool velvetBasisCompensation,
		Spectrum *value, MLHeroMixDiagnostic15no *diagnostic15no) const {
	if (diagnostic15no) {
		diagnostic15no->evaluationFailure =
			MLHERO_MIX_EVALUATE_OK15NO;
		diagnostic15no->evaluationChildIndex = ~0u;
	}
	if (!value || !material || material->GetType() != MIX ||
			!prepared.valid || !std::isfinite(waveLength)) {
		if (diagnostic15no)
			diagnostic15no->evaluationFailure =
				MLHERO_MIX_EVALUATE_INVALID_INPUT15NO;
		return false;
	}

	double spectralValue = 0.0;
	for (u_int i = 0u; i < 2u; ++i) {
		const MLHeroMixChildSpectral15nm &child = prepared.child[i];
		if (child.route == MLHERO_MIX_CHILD_INACTIVE15NM)
			continue;
		if (diagnostic15no)
			diagnostic15no->evaluationChildIndex = i;

		float childValue = 0.f;
		switch (child.route) {
			case MLHERO_MIX_CHILD_SCALAR15NM:
				childValue = (child.nativeValue.c[0] + child.nativeValue.c[1] +
					child.nativeValue.c[2]) / 3.f;
				break;
			case MLHERO_MIX_CHILD_METAL215NM: {
				Spectrum metalValue;
				if (!child.material || child.material->GetType() != METAL2) {
					if (diagnostic15no)
						diagnostic15no->evaluationFailure =
							MLHERO_MIX_EVALUATE_MATERIAL_MISMATCH15NO;
					return false;
				}
				const Metal2Material &metal2 =
					static_cast<const Metal2Material &>(*child.material);
				if (!metal2.EvaluateAtWaveLength(child.hitPoint,
						child.localLightDir, child.localEyeDir,
						waveLength, &metalValue)) {
					if (diagnostic15no)
						diagnostic15no->evaluationFailure =
							MLHERO_MIX_EVALUATE_METAL2_REJECT15NO;
					return false;
				}
				const float metalScale = Max(1.f, Max(fabsf(metalValue.c[0]),
					Max(fabsf(metalValue.c[1]), fabsf(metalValue.c[2]))));
				if (fabsf(metalValue.c[0] - metalValue.c[1]) >
						1e-6f * metalScale ||
						fabsf(metalValue.c[0] - metalValue.c[2]) >
						1e-6f * metalScale) {
					if (diagnostic15no)
						diagnostic15no->evaluationFailure =
							MLHERO_MIX_EVALUATE_METAL2_NONSCALAR15NO;
					return false;
				}
				childValue = (metalValue.c[0] + metalValue.c[1] +
					metalValue.c[2]) / 3.f;
				break;
			}
			case MLHERO_MIX_CHILD_GLOSSY215NM:
				if (!Glossy2Material::EvaluateMLHeroSpectralEndpoint15nk(
						child.glossy2, waveLength,
						glossy2BasisCompensation, &childValue)) {
					if (diagnostic15no)
						diagnostic15no->evaluationFailure =
							MLHERO_MIX_EVALUATE_GLOSSY2_REJECT15NO;
					return false;
				}
				break;
			case MLHERO_MIX_CHILD_VELVET15NN:
				if (!VelvetMaterial::EvaluateMLHeroSpectralEndpoint15nn(
						child.velvet, waveLength,
						velvetBasisCompensation, &childValue)) {
					if (diagnostic15no)
						diagnostic15no->evaluationFailure =
							MLHERO_MIX_EVALUATE_VELVET_REJECT15NO;
					return false;
				}
				break;
			default:
				if (diagnostic15no)
					diagnostic15no->evaluationFailure =
						MLHERO_MIX_EVALUATE_UNSUPPORTED_ROUTE15NO;
				return false;
		}
		if (!std::isfinite(childValue) || childValue < 0.f) {
			if (diagnostic15no)
				diagnostic15no->evaluationFailure =
					MLHERO_MIX_EVALUATE_CHILD_VALUE15NO;
			return false;
		}
		spectralValue += double(child.weight) * double(childValue);
	}

	spectralValue *= double(prepared.wrapperFactor) /
		double(prepared.nativePdfW);
	if (!std::isfinite(spectralValue) || spectralValue < 0.0 ||
			spectralValue > double(std::numeric_limits<float>::max())) {
		if (diagnostic15no)
			diagnostic15no->evaluationFailure =
				MLHERO_MIX_EVALUATE_RESULT15NO;
		return false;
	}
	*value = Spectrum(float(spectralValue));
	return true;
}
// 15nm END

Spectrum BSDF::ShadowCatcherSample(Vector *sampledDir,
		float *pdfW, float *absCosSampledDir, BSDFEvent *event) const {
	// Just continue to trace the ray
	*sampledDir = -hitPoint.fixedDir;
	*absCosSampledDir = AbsDot(*sampledDir, hitPoint.geometryN);

	*pdfW = 1.f;
	*event = SPECULAR | TRANSMIT;
	const Spectrum result(1.f);

	// Adjoint BSDF
	if (hitPoint.fromLight) {
		const float absDotFixedDirNG = AbsDot(hitPoint.fixedDir, hitPoint.geometryN);
		const float absDotSampledDirNG = AbsDot(*sampledDir, hitPoint.geometryN);
		return result * (absDotSampledDirNG / absDotFixedDirNG);
	} else
		return result;
}

Spectrum BSDF::Sample(Vector *sampledDir,
		const float u0, const float u1,
		float *pdfW, float *absCosSampledDir,
		BSDFEvent *event) const {
    Spectrum result;
    if(mlHeroEnabled && material->GetType()==GLASS) {
        // 15kr: ideal-glass deltas use unit directions in a double orthonormal basis.
        const slg::mlhero15kl::Frame precise(frame);
        const auto local=precise.ToLocal(hitPoint.fixedDir);
        const Vector fixed=slg::mlhero15kp::Normalize(Vector(local.x,local.y,local.z));
        Vector nativeGenerated;
        result=material->Sample(hitPoint,fixed,&nativeGenerated,u0,u1,hitPoint.passThroughEvent,pdfW,event);
        if(result.Black())return result;
        const Vector generated=slg::mlhero15kp::Normalize(nativeGenerated);
        const auto world=precise.ToWorld(slg::mlhero15kk::V(generated));
        *sampledDir=slg::mlhero15kp::Normalize(Vector(world.x,world.y,world.z));
        *absCosSampledDir=fabsf(CosTheta(generated));
        if(slg::mlhero15ko::armed)slg::mlhero15ko::Capture(hitPoint.fixedDir,fixed,nativeGenerated,*sampledDir,
            precise.x,precise.y,precise.z,true,false);
    }else if(mlHeroEnabled && material->GetType()==ROUGHGLASS) {
        const slg::mlhero15kl::Frame precise(frame);
        slg::mlhero15kk::V generated;
        result=static_cast<const RoughGlassMaterial &>(*material).SampleMLHeroPrecise15kl(hitPoint,
            precise.ToLocal(hitPoint.fixedDir),&generated,u0,u1,hitPoint.passThroughEvent,pdfW,event);
        if(result.Black())return result;
        const auto world=precise.ToWorld(generated);
        *sampledDir=Vector(world.x,world.y,world.z);
        *absCosSampledDir=std::fabs(generated.z);
        if(slg::mlhero15ko::armed)slg::mlhero15ko::Capture(hitPoint.fixedDir,precise.ToLocal(hitPoint.fixedDir),generated,*sampledDir,
            precise.x,precise.y,precise.z,true,IsVolume());
    }else if(mlHeroEnabled && material->GetType()==METAL2) {
        result=static_cast<const Metal2Material&>(*material).SampleMLHeroWorld15ky(hitPoint,frame,
            sampledDir,u0,u1,hitPoint.passThroughEvent,pdfW,absCosSampledDir,event);
        if(result.Black())return result;
        if(slg::mlhero15ko::armed){
            const slg::mlhero15kl::Frame precise(frame);
            const Vector fixed=slg::mlhero15ky::Local(frame,hitPoint.fixedDir);
            const Vector generated=slg::mlhero15ky::Local(frame,*sampledDir);
            slg::mlhero15ko::Capture(hitPoint.fixedDir,fixed,generated,*sampledDir,
                precise.x,precise.y,precise.z,true,false);
        }
    }else {
        const Vector fixed=MLHeroMetalLocal15ky(frame,hitPoint.fixedDir,material->GetType()==METAL2);Vector generated;
        result=material->Sample(hitPoint,fixed,&generated,u0,u1,hitPoint.passThroughEvent,pdfW,event);
        if(result.Black())return result;
        *absCosSampledDir=fabsf(CosTheta(generated));*sampledDir=frame.ToWorld(generated);
        if(mlHeroEnabled && IsVolume())*sampledDir=slg::mlhero15kp::Normalize(*sampledDir);
        if(slg::mlhero15ko::armed)slg::mlhero15ko::Capture(hitPoint.fixedDir,fixed,generated,*sampledDir,frame.X,frame.Y,frame.Z,false,IsVolume());
    }

	// Shadow terminator artefact avoidance
	if ((*event & REFLECT) &&
			(*event & (DIFFUSE | GLOSSY))
			&& (hitPoint.shadeN != hitPoint.interpolatedN)) {
		const Vector &lightDir = hitPoint.fromLight ? hitPoint.fixedDir : (*sampledDir);

		result *= ShadowTerminatorAvoidanceFactor(hitPoint.GetLandingInterpolatedN(),
				hitPoint.GetLandingShadeN(), lightDir);
	}

	// Adjoint BSDF
	if (hitPoint.fromLight) {
		const float absDotFixedDirNG = (mlHeroEnabled && material->GetType()==GLASS) ?
            AbsDot(slg::mlhero15kp::Normalize(hitPoint.fixedDir),hitPoint.geometryN) :
            MLHeroMetalAbsNG15ky(hitPoint.fixedDir,hitPoint.geometryN,material->GetType()==METAL2);
		const float absDotSampledDirNG = MLHeroMetalAbsNG15ky(*sampledDir,hitPoint.geometryN,material->GetType()==METAL2);
		result *= (absDotSampledDirNG / absDotFixedDirNG);
	}

	return result;
}

// 15fe BEGIN
bool BSDF::GetMLHeroMatteTranslucentFactors15fe(const Vector &dir,
        const BSDFEvent event, const bool sampled, Spectrum *kr, Spectrum *kt,
        float *wrapper) const {
    if (!wrapper || !GetMLHeroMatteTranslucentInputs15fd(kr, kt) ||
            !(event & DIFFUSE) || (event & (GLOSSY | SPECULAR)) ||
            bool(event & REFLECT) == bool(event & TRANSMIT)) return false;
    for (u_int c = 0u; c < 3u; ++c)
        if (!std::isfinite(kr->c[c]) || !std::isfinite(kt->c[c])) return false;
    const bool reflect = (event & REFLECT) != 0;
    const Vector &eyeDir = hitPoint.fromLight ? dir : hitPoint.fixedDir;
    const Vector &lightDir = hitPoint.fromLight ? hitPoint.fixedDir : dir;
    const Vector localLight = frame.ToLocal(lightDir), localEye = frame.ToLocal(eyeDir);
    if (reflect != (localLight.z * localEye.z > 0.f)) return false;
    float factor;
    if (sampled) {
        const Vector fixed = frame.ToLocal(hitPoint.fixedDir), generated = frame.ToLocal(dir);
        if (fabsf(fixed.z) < DEFAULT_COS_EPSILON_STATIC ||
                fabsf(generated.z) < DEFAULT_COS_EPSILON_STATIC) return false;
        const Spectrum transmission = *kt * (Spectrum(1.f) - *kr);
        const bool rBlack = kr->Black(), tBlack = transmission.Black();
        if (rBlack && tBlack) return false;
        const float threshold = !rBlack ? (!tBlack ? .5f : 1.f) : 0.f;
        // The native proposal remains authoritative, including black-lobe cases.
        if (reflect != (hitPoint.passThroughEvent < threshold)) return false;
        const float probability = reflect ? threshold : 1.f - threshold;
        if (!(probability > 0.f)) return false;
        factor = 1.f / probability;
        if (hitPoint.fromLight) factor *= fabsf(fixed.z / generated.z);
        if (reflect && hitPoint.shadeN != hitPoint.interpolatedN)
            factor *= ShadowTerminatorAvoidanceFactor(hitPoint.GetLandingInterpolatedN(),
                hitPoint.GetLandingShadeN(), lightDir);
        if (hitPoint.fromLight) {
            const float fixedNG = AbsDot(hitPoint.fixedDir, hitPoint.geometryN);
            if (!(fixedNG > 0.f)) return false;
            factor *= AbsDot(dir, hitPoint.geometryN) / fixedNG;
        }
    } else {
        const float eyeNG = AbsDot(eyeDir, hitPoint.geometryN);
        const float lightNG = AbsDot(lightDir, hitPoint.geometryN);
        if (eyeNG < DEFAULT_COS_EPSILON_STATIC || lightNG < DEFAULT_COS_EPSILON_STATIC)
            return false;
        // Both sides are supported by this material, as in BSDF::Evaluate().
        factor = INV_PI * fabsf(localLight.z);
        if (reflect && hitPoint.shadeN != hitPoint.interpolatedN)
            factor *= ShadowTerminatorAvoidanceFactor(hitPoint.GetLandingInterpolatedN(),
                hitPoint.GetLandingShadeN(), lightDir);
        if (hitPoint.fromLight) factor *= eyeNG / lightNG;
    }
    if (!std::isfinite(factor) || factor < 0.f) return false;
    *wrapper = factor;
    return true;
}
// 15fe END

// 15fd BEGIN diagnostic-only
bool BSDF::GetMLHeroMatteTranslucentInputs15fd(Spectrum *kr, Spectrum *kt) const {
    if (!kr || !kt || !material || material->GetType() != MATTETRANSLUCENT)
        return false;
    const MatteTranslucentMaterial &m = static_cast<const MatteTranslucentMaterial &>(*material);
    *kr = m.GetKr()->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
    *kt = m.GetKt()->GetSpectrumValue(hitPoint).Clamp(0.f, 1.f);
    return true;
}
// 15fd END diagnostic-only

bool BSDF::EvaluateMLHeroMetal2SampleAtWaveLength(const Vector &sampledDir,
		const float waveLength, Spectrum *sampleMultiplier) const {
	if (!sampleMultiplier || !material || (material->GetType() != METAL2))
		return false;

	const Metal2Material &metal2 = static_cast<const Metal2Material &>(*material);

	const Vector localFixedDir = MLHeroMetalLocal15ky(frame,hitPoint.fixedDir,material->GetType()==METAL2);
	const Vector localSampledDir = MLHeroMetalLocal15ky(frame,sampledDir,material->GetType()==METAL2);
	Spectrum result;
	if (!metal2.EvaluateSampleAtWaveLength(hitPoint, localFixedDir, localSampledDir,
			waveLength, &result))
		return false;

	// Match the post-processing performed by BSDF::Sample() exactly.
	if ((REFLECT & (GLOSSY | REFLECT)) &&
			(hitPoint.shadeN != hitPoint.interpolatedN)) {
		const Vector &lightDir = hitPoint.fromLight ? hitPoint.fixedDir : sampledDir;
		result *= ShadowTerminatorAvoidanceFactor(hitPoint.GetLandingInterpolatedN(),
				hitPoint.GetLandingShadeN(), lightDir);
	}

	if (hitPoint.fromLight) {
		const float absDotFixedDirNG = MLHeroMetalAbsNG15ky(hitPoint.fixedDir,hitPoint.geometryN,material->GetType()==METAL2);
		const float absDotSampledDirNG = MLHeroMetalAbsNG15ky(sampledDir,hitPoint.geometryN,material->GetType()==METAL2);
		result *= (absDotSampledDirNG / absDotFixedDirNG);
	}

	*sampleMultiplier = result;
	return true;
}

bool BSDF::EvaluateMLHeroMetal2AtWaveLength(const Vector &generatedDir,
		const float waveLength, Spectrum *value) const {
	if (!value || !material || (material->GetType() != METAL2))
		return false;

	const Vector &eyeDir = hitPoint.fromLight ? generatedDir : hitPoint.fixedDir;
	const Vector &lightDir = hitPoint.fromLight ? hitPoint.fixedDir : generatedDir;

	const float dotLightDirNG = MLHeroMetalNG15ky(lightDir,hitPoint.geometryN,material->GetType()==METAL2);
	const float absDotLightDirNG = fabsf(dotLightDirNG);
	const float dotEyeDirNG = MLHeroMetalNG15ky(eyeDir,hitPoint.geometryN,material->GetType()==METAL2);
	const float absDotEyeDirNG = fabsf(dotEyeDirNG);
	if ((absDotLightDirNG < DEFAULT_COS_EPSILON_STATIC) ||
			(absDotEyeDirNG < DEFAULT_COS_EPSILON_STATIC))
		return false;
	if (dotEyeDirNG * dotLightDirNG <= 0.f)
		return false;
	if (Dot(eyeDir, hitPoint.interpolatedN) * Dot(lightDir, hitPoint.interpolatedN) <= 0.f)
		return false;

	const Metal2Material &metal2 = static_cast<const Metal2Material &>(*material);
	const Vector localLightDir = MLHeroMetalLocal15ky(frame,lightDir,material->GetType()==METAL2);
	const Vector localEyeDir = MLHeroMetalLocal15ky(frame,eyeDir,material->GetType()==METAL2);
	Spectrum result;
	if (!metal2.EvaluateAtWaveLength(hitPoint, localLightDir, localEyeDir, waveLength, &result))
		return false;

	if ((hitPoint.shadeN != hitPoint.interpolatedN))
		result *= ShadowTerminatorAvoidanceFactor(hitPoint.GetLandingInterpolatedN(),
				hitPoint.GetLandingShadeN(), lightDir);

	if (hitPoint.fromLight)
		result *= (absDotEyeDirNG / absDotLightDirNG);

	*value = result;
	return true;
}

bool BSDF::EvaluateMLHeroMetal2DebugAtWaveLength(const Vector &generatedDir,
		const float waveLength,
		Spectrum *eta, Spectrum *kk, Spectrum *fresnel,
		float *microfacetFactor, float *wrapperFactor, float *cosWH,
		Spectrum *value) const {
	if (!eta || !kk || !fresnel || !microfacetFactor ||
			!wrapperFactor || !cosWH || !value ||
			!material || (material->GetType() != METAL2))
		return false;

	const Vector &eyeDir = hitPoint.fromLight ? generatedDir : hitPoint.fixedDir;
	const Vector &lightDir = hitPoint.fromLight ? hitPoint.fixedDir : generatedDir;

	const float dotLightDirNG = MLHeroMetalNG15ky(lightDir,hitPoint.geometryN,material->GetType()==METAL2);
	const float absDotLightDirNG = fabsf(dotLightDirNG);
	const float dotEyeDirNG = MLHeroMetalNG15ky(eyeDir,hitPoint.geometryN,material->GetType()==METAL2);
	const float absDotEyeDirNG = fabsf(dotEyeDirNG);
	if ((absDotLightDirNG < DEFAULT_COS_EPSILON_STATIC) ||
			(absDotEyeDirNG < DEFAULT_COS_EPSILON_STATIC))
		return false;
	if (dotEyeDirNG * dotLightDirNG <= 0.f)
		return false;
	if (Dot(eyeDir, hitPoint.interpolatedN) *
			Dot(lightDir, hitPoint.interpolatedN) <= 0.f)
		return false;

	const Metal2Material &metal2 = static_cast<const Metal2Material &>(*material);
	const Vector localLightDir = MLHeroMetalLocal15ky(frame,lightDir,material->GetType()==METAL2);
	const Vector localEyeDir = MLHeroMetalLocal15ky(frame,eyeDir,material->GetType()==METAL2);

	Spectrum result;
	if (!metal2.EvaluateAtWaveLengthDebug(hitPoint,
			localLightDir, localEyeDir, waveLength,
			eta, kk, fresnel, microfacetFactor, cosWH, &result))
		return false;

	float correction = 1.f;

	if ((hitPoint.shadeN != hitPoint.interpolatedN)) {
		const float shadowFactor = ShadowTerminatorAvoidanceFactor(
				hitPoint.GetLandingInterpolatedN(),
				hitPoint.GetLandingShadeN(), lightDir);
		result *= shadowFactor;
		correction *= shadowFactor;
	}

	if (hitPoint.fromLight) {
		const float lightTransportFactor = absDotEyeDirNG / absDotLightDirNG;
		result *= lightTransportFactor;
		correction *= lightTransportFactor;
	}

	*wrapperFactor = correction;
	*value = result;
	return true;
}



bool BSDF::EvaluateMLHeroGlassDebugAtWaveLength(const Vector &sampledDir,
		const BSDFEvent sampledEvent, const float samplePdfW,
		const float waveLength,
		float *exteriorIOR, float *interiorIORBase, float *interiorIORLambda,
		float *fresnelR, float *eta, float *eta2, float *transportFactor,
		float *directionDelta, Spectrum *sampleMultiplier,
		float *cosFixedOut, float *sinI2Out, float *sinT2Out, bool *tirOut,
		float *laneEventPdfWOut) const {
	if (!material || (material->GetType() != GLASS))
		return false;

	const GlassMaterial &glass = static_cast<const GlassMaterial &>(*material);
	// 15mv-PDF: Glass::Sample() uses the precise double frame and normalized
	// local directions above. Reconstruct the audit/production helper inputs
	// through that identical path; the legacy float Frame::ToLocal() can move
	// the Fresnel event mass beyond the strict 2e-5 PDF tolerance at rare deep,
	// grazing-angle vertices.
	const slg::mlhero15kl::Frame precise15mv(frame);
	const auto fixed15mv = precise15mv.ToLocal(hitPoint.fixedDir);
	const auto sampled15mv = precise15mv.ToLocal(sampledDir);
	const Vector localFixedDir = slg::mlhero15kp::Normalize(Vector(
		fixed15mv.x, fixed15mv.y, fixed15mv.z));
	const Vector localSampledDir = slg::mlhero15kp::Normalize(Vector(
		sampled15mv.x, sampled15mv.y, sampled15mv.z));
	return glass.EvaluateMLHeroDebugAtWaveLength(hitPoint,
			localFixedDir, localSampledDir, sampledEvent, samplePdfW, waveLength,
			exteriorIOR, interiorIORBase, interiorIORLambda,
			fresnelR, eta, eta2, transportFactor, directionDelta, sampleMultiplier,
			cosFixedOut, sinI2Out, sinT2Out, tirOut, laneEventPdfWOut);
}


bool BSDF::HasMLHeroGlassDispersion() const {
    return material && material->GetType() == GLASS &&
        static_cast<const GlassMaterial &>(*material).HasMLHeroDispersion(hitPoint);
}

bool BSDF::EvaluateMLHeroGlassSampleAtWaveLength(const Vector &dir,
        const BSDFEvent event, const float pdfW, const float wavelength,
        Spectrum *value, float *directionDelta) const {
    if (!value || !material || material->GetType() != GLASS || !(event & SPECULAR) ||
        !(event & (REFLECT | TRANSMIT)) || !std::isfinite(pdfW) || pdfW <= 0.f ||
        !std::isfinite(wavelength) || wavelength < 380.f || wavelength > 780.f) return false;
    float nc = 0.f, base = 0.f, nt = 0.f, fresnel = 0.f, eta = 0.f, eta2 = 0.f;
    float transport = 0.f, delta = 0.f;
    const bool valid = EvaluateMLHeroGlassDebugAtWaveLength(dir, event, pdfW, wavelength,
        &nc, &base, &nt, &fresnel, &eta, &eta2, &transport, &delta, value);
    if (directionDelta) *directionDelta = delta;
    // A delta transmission may only carry the wavelength owning this direction.
    // Reflection is common to all packet wavelengths. Callers evaluate only
    // HERO on dispersive transmission; no companion-direction approximation.
    if (!valid || !std::isfinite(delta) || delta > 2e-4f) return false;
    if (hitPoint.fromLight) {
        const float fixedNg = AbsDot(hitPoint.fixedDir, hitPoint.geometryN);
        if (fixedNg <= 1e-20f) return false;
        *value *= AbsDot(dir, hitPoint.geometryN) / fixedNg;
    }
    return !value->IsNaN() && !value->IsInf() && !value->IsNeg();
}

bool BSDF::HasMLHeroRoughGlassDispersion() const {
    return material && material->GetType() == ROUGHGLASS &&
        static_cast<const RoughGlassMaterial &>(*material).HasMLHeroDispersion(hitPoint);
}

bool BSDF::EvaluateMLHeroRoughGlassAtWaveLength(const Vector &dir,
        const BSDFEvent expectedEvent, const float wavelength, Spectrum *value) const {
    if (!value || !material || material->GetType() != ROUGHGLASS ||
        wavelength < 380.f || wavelength > 780.f || !std::isfinite(wavelength)) return false;
    struct Restore {
        float wavelength; bool used, fallback;
        Restore() : wavelength(mlDispersionWaveLength), used(mlDispersionUsed),
            fallback(mlHeroPacketFallbackUsed) {}
        ~Restore() { mlDispersionWaveLength = wavelength; mlDispersionUsed = used;
            mlHeroPacketFallbackUsed = fallback; }
    } restore;
    mlDispersionWaveLength = wavelength;
    BSDFEvent laneEvent = NONE;
    float pdf = 0.f, reversePdf = 0.f;
    *value = Evaluate(dir, &laneEvent, &pdf, &reversePdf);
    // Zero is a valid spectral contribution at a companion wavelength.
    return !value->IsNaN() && !value->IsInf() && !value->IsNeg() &&
        (value->Black() || ((laneEvent & (REFLECT | TRANSMIT)) ==
            (expectedEvent & (REFLECT | TRANSMIT))));
}

bool BSDF::EvaluateMLHeroRoughGlassMicrofacetOwnershipDebug(
		const Vector &sampledDir, const BSDFEvent sampledEvent,
		const float heroWaveLength, const float laneWaveLength,
		float *heroEta, float *laneEta,
		float *heroMicrofacetPdf, float *laneSharedMicrofacetPdf,
		float *naturalJacobian, float *sharedJacobian,
		float *naturalDirPdf, float *sharedDirPdf,
		float *naturalDirDelta, float *sharedWhDelta, bool *naturalTir,
		Vector *naturalChildDir, BSDFEvent *naturalChildEvent,
		Spectrum *naturalChildValue, float *naturalChildDirectPdf,
		float *naturalChildReversePdf, bool *naturalChildPairValid,
		bool *naturalChildPairTransmit, float *naturalChildAbsLightZ,
		float *naturalChildAbsEyeZ,
		bool *naturalChildHemisphereZero,
		Spectrum *naturalChildWrappedValue) const {
	if (!material || (material->GetType() != ROUGHGLASS))
		return false;

	const RoughGlassMaterial &roughGlass = static_cast<const RoughGlassMaterial &>(*material);
    if(mlHeroEnabled) {
        const slg::mlhero15kl::Frame precise(frame);
        slg::mlhero15kk::V localNaturalChild15nz;
		Spectrum localNaturalChildValue15ob = Spectrum();
		BSDFEvent localNaturalChildEvent15ob = NONE;
        const bool ok15nz = roughGlass.EvaluateMLHeroMicrofacetOwnershipPrecise15kl(hitPoint,
            precise.ToLocal(hitPoint.fixedDir),precise.ToLocal(sampledDir),sampledEvent,
            heroWaveLength,laneWaveLength,heroEta,laneEta,heroMicrofacetPdf,laneSharedMicrofacetPdf,
            naturalJacobian,sharedJacobian,naturalDirPdf,sharedDirPdf,naturalDirDelta,sharedWhDelta,naturalTir,
			(naturalChildDir || naturalChildWrappedValue) ?
				&localNaturalChild15nz : nullptr,
			(naturalChildEvent || naturalChildWrappedValue) ?
				&localNaturalChildEvent15ob : nullptr,
			(naturalChildValue || naturalChildWrappedValue) ?
				&localNaturalChildValue15ob : nullptr,
			naturalChildDirectPdf,naturalChildReversePdf,
			naturalChildPairValid,naturalChildPairTransmit,
			naturalChildAbsLightZ,naturalChildAbsEyeZ,
			naturalChildHemisphereZero);
		if (naturalChildValue)
			*naturalChildValue = localNaturalChildValue15ob;
		if (naturalChildEvent)
			*naturalChildEvent = localNaturalChildEvent15ob;
        if (ok15nz && naturalChildDir) {
            const slg::mlhero15kk::V world15nz = precise.ToWorld(localNaturalChild15nz);
            *naturalChildDir = Vector(world15nz.x, world15nz.y, world15nz.z);
        }
		if (naturalChildWrappedValue) {
			*naturalChildWrappedValue = localNaturalChildValue15ob;
			if (ok15nz && !naturalChildWrappedValue->Black()) {
				const slg::mlhero15kk::V world15ob = precise.ToWorld(localNaturalChild15nz);
				const Vector generated15ob(world15ob.x, world15ob.y, world15ob.z);
				if ((localNaturalChildEvent15ob & REFLECT) &&
						(localNaturalChildEvent15ob & (DIFFUSE | GLOSSY)) &&
						(hitPoint.shadeN != hitPoint.interpolatedN)) {
					const Vector &lightDir15ob = hitPoint.fromLight ?
						hitPoint.fixedDir : generated15ob;
					*naturalChildWrappedValue *= ShadowTerminatorAvoidanceFactor(
						hitPoint.GetLandingInterpolatedN(),
						hitPoint.GetLandingShadeN(), lightDir15ob);
				}
				if (hitPoint.fromLight) {
					const float fixedNG15ob = AbsDot(hitPoint.fixedDir,
						hitPoint.geometryN);
					const float generatedNG15ob = AbsDot(generated15ob,
						hitPoint.geometryN);
					if (fixedNG15ob > 0.f)
						*naturalChildWrappedValue *= generatedNG15ob / fixedNG15ob;
					else
						*naturalChildWrappedValue = Spectrum();
				}
			}
		}
        return ok15nz;
    }
	if (naturalChildDir) *naturalChildDir = Vector();
	if (naturalChildEvent) *naturalChildEvent = NONE;
	if (naturalChildValue) *naturalChildValue = Spectrum();
	if (naturalChildDirectPdf) *naturalChildDirectPdf = 0.f;
	if (naturalChildReversePdf) *naturalChildReversePdf = 0.f;
	if (naturalChildPairValid) *naturalChildPairValid = false;
	if (naturalChildPairTransmit) *naturalChildPairTransmit = false;
	if (naturalChildAbsLightZ) *naturalChildAbsLightZ = 0.f;
	if (naturalChildAbsEyeZ) *naturalChildAbsEyeZ = 0.f;
	if (naturalChildHemisphereZero) *naturalChildHemisphereZero = false;
	if (naturalChildWrappedValue) *naturalChildWrappedValue = Spectrum();
	const Vector localFixedDir = frame.ToLocal(hitPoint.fixedDir);
	const Vector localSharedDir = frame.ToLocal(sampledDir);
	return roughGlass.EvaluateMLHeroMicrofacetOwnershipDebug(hitPoint,
			localFixedDir, localSharedDir, sampledEvent,
			heroWaveLength, laneWaveLength, heroEta, laneEta,
			heroMicrofacetPdf, laneSharedMicrofacetPdf,
			naturalJacobian, sharedJacobian,
			naturalDirPdf, sharedDirPdf,
			naturalDirDelta, sharedWhDelta, naturalTir);
}

bool BSDF::EvaluateMLHeroGlossy2DebugSampleComponents(
		const Vector &sampledDir, const float pdfW,
		Spectrum *kdValue, Spectrum *ksValue,
		Spectrum *schlickS, Spectrum *absorptionValue,
		Spectrum *baseContribution, Spectrum *coatingContribution,
		float *wrapperFactor, Spectrum *combinedResult) const {
	if (!kdValue || !ksValue || !schlickS || !absorptionValue ||
			!baseContribution || !coatingContribution ||
			!wrapperFactor || !combinedResult ||
			!material || (material->GetType() != GLOSSY2))
		return false;

	const Glossy2Material &glossy2 =
		static_cast<const Glossy2Material &>(*material);

	const Vector localFixedDir = frame.ToLocal(hitPoint.fixedDir);
	const Vector localSampledDir = frame.ToLocal(sampledDir);

	Spectrum base, coating, combined;
	if (!glossy2.EvaluateMLHeroDebugSampleComponents(
			hitPoint, localFixedDir, localSampledDir, pdfW,
			kdValue, ksValue, schlickS, absorptionValue,
			&base, &coating, &combined))
		return false;

	float correction = 1.f;

	if ((hitPoint.shadeN != hitPoint.interpolatedN)) {
		const Vector &lightDir = hitPoint.fromLight ? hitPoint.fixedDir : sampledDir;
		const float shadowFactor = ShadowTerminatorAvoidanceFactor(
			hitPoint.GetLandingInterpolatedN(),
			hitPoint.GetLandingShadeN(), lightDir);
		base *= shadowFactor;
		coating *= shadowFactor;
		combined *= shadowFactor;
		correction *= shadowFactor;
	}

	if (hitPoint.fromLight) {
		const float absDotFixedDirNG =
			AbsDot(hitPoint.fixedDir, hitPoint.geometryN);
		const float absDotSampledDirNG =
			AbsDot(sampledDir, hitPoint.geometryN);
		const float adjointFactor =
			absDotSampledDirNG / absDotFixedDirNG;
		base *= adjointFactor;
		coating *= adjointFactor;
		combined *= adjointFactor;
		correction *= adjointFactor;
	}

	*baseContribution = base;
	*coatingContribution = coating;
	*combinedResult = combined;
	*wrapperFactor = correction;

	return true;
}

bool BSDF::EvaluateMLHeroGlossy2Components15nd(
		const Vector &generatedDir, const BSDFEvent expectedEvent,
		Spectrum *kdValue, Spectrum *baseContribution,
		Spectrum *coatingContribution, Spectrum *combinedResult) const {
	if (!kdValue || !baseContribution || !coatingContribution ||
			!combinedResult || !material || material->GetType() != GLOSSY2)
		return false;

	const Vector &eyeDir = hitPoint.fromLight ? generatedDir : hitPoint.fixedDir;
	const Vector &lightDir = hitPoint.fromLight ? hitPoint.fixedDir : generatedDir;
	const float absDotLightDirNG = AbsDot(lightDir, hitPoint.geometryN);
	const float absDotEyeDirNG = AbsDot(eyeDir, hitPoint.geometryN);
	if (absDotLightDirNG < DEFAULT_COS_EPSILON_STATIC ||
			absDotEyeDirNG < DEFAULT_COS_EPSILON_STATIC ||
			Dot(eyeDir, hitPoint.geometryN) *
				Dot(lightDir, hitPoint.geometryN) <= 0.f ||
			Dot(eyeDir, hitPoint.interpolatedN) *
				Dot(lightDir, hitPoint.interpolatedN) <= 0.f)
		return false;

	const Vector localLightDir = frame.ToLocal(lightDir);
	const Vector localEyeDir = frame.ToLocal(eyeDir);
	const Glossy2Material &glossy2 =
		static_cast<const Glossy2Material &>(*material);
	BSDFEvent event15nd;
	Spectrum kd, base, coating, combined;
	if (!glossy2.EvaluateMLHeroEvaluateComponents15nd(hitPoint,
			localLightDir, localEyeDir, &event15nd, &kd, &base,
			&coating, &combined) || event15nd != expectedEvent)
		return false;

	if ((event15nd & REFLECT) && (event15nd & (DIFFUSE | GLOSSY)) &&
			(hitPoint.shadeN != hitPoint.interpolatedN)) {
		const float shadowFactor = ShadowTerminatorAvoidanceFactor(
			hitPoint.GetLandingInterpolatedN(), hitPoint.GetLandingShadeN(),
			lightDir);
		base *= shadowFactor;
		coating *= shadowFactor;
		combined *= shadowFactor;
	}
	if (hitPoint.fromLight) {
		const float adjointFactor = absDotEyeDirNG / absDotLightDirNG;
		base *= adjointFactor;
		coating *= adjointFactor;
		combined *= adjointFactor;
	}

	if (kd.IsNaN() || kd.IsInf() || base.IsNaN() || base.IsInf() ||
			coating.IsNaN() || coating.IsInf() || combined.IsNaN() ||
			combined.IsInf())
		return false;
	*kdValue = kd;
	*baseContribution = base;
	*coatingContribution = coating;
	*combinedResult = combined;
	return true;
}

bool BSDF::EvaluateMLHeroGlossy2AtWaveLength15nj(
		const Vector &generatedDir, const BSDFEvent expectedEvent,
		const float waveLength, const bool basisCompensation,
		Spectrum *value) const {
	MLHeroGlossy2SpectralEndpoint15nk prepared;
	return PrepareMLHeroGlossy2SpectralEndpoint15nk(generatedDir,
			expectedEvent, &prepared) &&
		EvaluateMLHeroGlossy2Prepared15nk(prepared, waveLength,
			basisCompensation, value);
}

bool BSDF::PrepareMLHeroGlossy2SpectralEndpoint15nk(
		const Vector &generatedDir, const BSDFEvent expectedEvent,
		MLHeroGlossy2SpectralEndpoint15nk *prepared) const {
	if (!prepared || !material || material->GetType() != GLOSSY2)
		return false;

	const Vector &eyeDir = hitPoint.fromLight ? generatedDir : hitPoint.fixedDir;
	const Vector &lightDir = hitPoint.fromLight ? hitPoint.fixedDir : generatedDir;
	const float absDotLightDirNG = AbsDot(lightDir, hitPoint.geometryN);
	const float absDotEyeDirNG = AbsDot(eyeDir, hitPoint.geometryN);
	const float sideTestIS = Dot(eyeDir, hitPoint.interpolatedN) *
		Dot(lightDir, hitPoint.interpolatedN);
	if (absDotLightDirNG < DEFAULT_COS_EPSILON_STATIC ||
			absDotEyeDirNG < DEFAULT_COS_EPSILON_STATIC ||
			Dot(eyeDir, hitPoint.geometryN) * Dot(lightDir, hitPoint.geometryN) <= 0.f ||
			// Match BSDF::Evaluate(): the exactly tangent interpolated-normal
			// boundary (including signed zero) is not an opposite-side event.
			// Glossy2 can still evaluate its valid back-face diffuse base there.
			sideTestIS < 0.f)
		return false;

	const Vector localLightDir = frame.ToLocal(lightDir);
	const Vector localEyeDir = frame.ToLocal(eyeDir);
	const Glossy2Material &glossy2 =
		static_cast<const Glossy2Material &>(*material);
	if (!glossy2.PrepareMLHeroSpectralEndpoint15nk(hitPoint,
			localLightDir, localEyeDir, prepared) ||
			prepared->event != expectedEvent)
		return false;

	if ((prepared->event & REFLECT) &&
			(prepared->event & (DIFFUSE | GLOSSY)) &&
			(hitPoint.shadeN != hitPoint.interpolatedN))
		prepared->wrapperFactor *= ShadowTerminatorAvoidanceFactor(
			hitPoint.GetLandingInterpolatedN(), hitPoint.GetLandingShadeN(),
			lightDir);
	if (hitPoint.fromLight)
		prepared->wrapperFactor *= absDotEyeDirNG / absDotLightDirNG;
	return std::isfinite(prepared->wrapperFactor) &&
		prepared->wrapperFactor >= 0.f;
}

bool BSDF::EvaluateMLHeroGlossy2Prepared15nk(
		const MLHeroGlossy2SpectralEndpoint15nk &prepared,
		const float waveLength, const bool basisCompensation,
		Spectrum *value) const {
	if (!value || !material || material->GetType() != GLOSSY2)
		return false;
	float spectralValue;
	if (!Glossy2Material::EvaluateMLHeroSpectralEndpoint15nk(prepared,
			waveLength, basisCompensation, &spectralValue))
		return false;
	*value = Spectrum(spectralValue);
	return true;
}

bool BSDF::EvaluateMLHeroGlossy2NativeComponentsPrepared15nk(
		const MLHeroGlossy2SpectralEndpoint15nk &prepared,
		Spectrum *baseContribution, Spectrum *coatingContribution,
		Spectrum *combinedResult) const {
	return material && material->GetType() == GLOSSY2 &&
		Glossy2Material::EvaluateMLHeroNativeComponentsPrepared15nk(prepared,
			baseContribution, coatingContribution, combinedResult);
}

void BSDF::Pdf(const Vector &sampledDir, float *directPdfW, float *reversePdfW) const {
	const Vector &eyeDir = hitPoint.fromLight ? sampledDir : hitPoint.fixedDir;
	const Vector &lightDir = hitPoint.fromLight ? hitPoint.fixedDir : sampledDir;
    if(mlHeroEnabled && material->GetType()==ROUGHGLASS) {
        const slg::mlhero15kl::Frame precise(frame);
        static_cast<const RoughGlassMaterial &>(*material).PdfMLHeroPrecise15kl(hitPoint,
            precise.ToLocal(lightDir),precise.ToLocal(eyeDir),directPdfW,reversePdfW);return;
    }
	Vector localLightDir = MLHeroMetalLocal15ky(frame,lightDir,material->GetType()==METAL2);
	Vector localEyeDir = MLHeroMetalLocal15ky(frame,eyeDir,material->GetType()==METAL2);

	material->Pdf(hitPoint, localLightDir, localEyeDir, directPdfW, reversePdfW);
}

Spectrum BSDF::GetPassThroughTransparency(const bool backTracing) const {
	const Vector localFixedDir = frame.ToLocal(hitPoint.fixedDir);

	return material->GetPassThroughTransparency(hitPoint, localFixedDir,
			hitPoint.passThroughEvent, backTracing);
}

Spectrum BSDF::GetEmittedRadiance(float *directPdfA, float *emissionPdfW) const {
	return triangleLightSource ?
		triangleLightSource->GetRadiance(hitPoint, directPdfA, emissionPdfW) :
		Spectrum();
}

AlbedoSpecularSetting slg::String2AlbedoSpecularSetting(const string &type) {
	if (type == "NO_REFLECT_TRANSMIT")
		return NO_REFLECT_TRANSMIT;
	else if (type == "ONLY_REFLECT")
		return ONLY_REFLECT;
	else if (type == "ONLY_TRANSMIT")
		return ONLY_TRANSMIT;
	else if (type == "REFLECT_TRANSMIT")
		return REFLECT_TRANSMIT;
	else
		throw runtime_error("Unknown albedo specular setting in String2AlbedoSpecularSetting(): " + type);
}

const string slg::AlbedoSpecularSetting2String(const AlbedoSpecularSetting type) {
	switch (type) {
		case NO_REFLECT_TRANSMIT:
			return "NO_REFLECT_TRANSMIT";
		case ONLY_REFLECT:
			return "ONLY_REFLECT";
		case ONLY_TRANSMIT:
			return "ONLY_TRANSMIT";
		case REFLECT_TRANSMIT:
			return "REFLECT_TRANSMIT";
		default:
			throw runtime_error("Unknown albedo specular setting in AlbedoSpecularSetting2String(): " + ToString(type));
	}
}
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4

// 15kt BEGIN: reconstruct the same precise local basis used by the actual GLASS wrapper.
slg::mlhero15kt::Evidence BSDF::EvaluateMLHeroGlassReverse15kt(const Vector &sampled,
        const BSDFEvent event,const float nativeForwardMass) const {
    if(!mlHeroEnabled||material->GetType()!=GLASS)return {};
    const slg::mlhero15kl::Frame precise(frame);
    const auto lf=precise.ToLocal(hitPoint.fixedDir),lr=precise.ToLocal(sampled);
    const Vector fixed=slg::mlhero15kp::Normalize(Vector(lf.x,lf.y,lf.z));
    const Vector reverseFixed=slg::mlhero15kp::Normalize(Vector(lr.x,lr.y,lr.z));
    HitPoint reversed=hitPoint;reversed.fromLight=!hitPoint.fromLight;
    reversed.fixedDir=slg::mlhero15kp::Normalize(sampled);
    reversed.intoObject=Dot(reversed.fixedDir,hitPoint.geometryN)>0.f;
    return static_cast<const GlassMaterial &>(*material).EvaluateMLHeroReverseMass15kt(
        hitPoint,reversed,fixed,reverseFixed,event,nativeForwardMass);
}
// 15kt END

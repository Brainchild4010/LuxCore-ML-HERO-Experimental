#ifndef SLG_MLHERO_SURFACE_TRACE15KV_H
#define SLG_MLHERO_SURFACE_TRACE15KV_H
#include <atomic>
#include <cmath>
namespace slg { namespace mlhero15kv {
struct V {
    double x=0.,y=0.,z=0.;
    template<class T>static V From(const T&a){return {double(a.x),double(a.y),double(a.z)};}
    double Norm2()const{return x*x+y*y+z*z;}
    bool Finite()const{return std::isfinite(x)&&std::isfinite(y)&&std::isfinite(z);}
    V operator-(V b)const{return {x-b.x,y-b.y,z-b.z};}
    V operator+(V b)const{return {x+b.x,y+b.y,z+b.z};}
    V operator*(double b)const{return {x*b,y*b,z*b};}
};
inline V Cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline bool BadAxis(V a){return !a.Finite()||!(a.Norm2()>0.)||!std::isfinite(a.Norm2());}
struct Snapshot {
    V geometry,interpolated,shade,dpdu,dpdv;
    template<class H>static Snapshot From(const H&h){return {V::From(h.geometryN),V::From(h.interpolatedN),V::From(h.shadeN),V::From(h.dpdu),V::From(h.dpdv)};}
    unsigned NormalMask()const{return (BadAxis(geometry)?1u:0u)|(BadAxis(interpolated)?2u:0u)|(BadAxis(shade)?4u:0u);}
    unsigned DifferentialMask()const{return (!dpdu.Finite()?1u:0u)|(!dpdv.Finite()?2u:0u);}
};
struct Trace {
    unsigned kind=0,side=0,initKind=0,materialID=0,materialType=0,meshType=0,triangle=0;
    unsigned objectID=0,vertexIndex[3]{};unsigned long long detail=0;
    bool bump=false,vertexNormals=false;unsigned bumpType=0;
    const char*materialName="";const char*bumpName="";
    double b1=0.,b2=0.,bumpDistance=0.;
    V position,fixed,beforeDndu,beforeDndv,afterDndu,afterDndv;
    Snapshot before,after;
    V frameX,frameY,frameZ,vertex[3],normal[3],weightedNormal,faceCross;
    // 15kw: current in-memory base mesh, NOT original import-file data.
    unsigned recoveryOutcome=0; // 0 normal observation, 1 recovered, 2 rejected
    bool rawAvailable=false,rawNormals=false,transformApplied=false;
    unsigned rawNonfiniteMask=0,rawZeroMask=0;
    V rawVertex[3],rawNormal[3],baseShade[3],preNormalize[3];
    V rawBlendDouble,rawBlendFloat,baseInterpolated,interpolatedPreNormalize,rawFaceCross;
    double matrix[4][4]{},inverseMatrix[4][4]{};

};
using Reporter=void(*)(const Trace&);
inline thread_local Reporter reporter=nullptr;
struct WorkerScope {
    WorkerScope(bool on,Reporter fn){reporter=on?fn:nullptr;}
    ~WorkerScope(){reporter=nullptr;}
    WorkerScope(const WorkerScope&)=delete;WorkerScope&operator=(const WorkerScope&)=delete;
};
inline std::atomic<unsigned long long> events[2][3]{},details[2][3]{},writeFailures[2][3]{};
inline void Reset(){for(unsigned s=0;s<2;++s)for(unsigned k=0;k<3;++k){events[s][k]=0ull;details[s][k]=0ull;writeFailures[s][k]=0ull;}}
inline int Classify(const Trace&t){
    if(t.before.NormalMask())return 0;
    if(t.after.NormalMask())return 1;
    if(t.before.DifferentialMask()||t.after.DifferentialMask()||BadAxis(t.frameX)||BadAxis(t.frameY)||BadAxis(t.frameZ))return 2;
    return -1;
}
inline bool Select(Trace&t){
    const int k=Classify(t);if(k<0)return false;t.kind=unsigned(k);
    events[t.side][t.kind].fetch_add(1ull,std::memory_order_relaxed);
    t.detail=details[t.side][t.kind].fetch_add(1ull,std::memory_order_relaxed);
    return t.detail<16ull;
}
} }
#endif

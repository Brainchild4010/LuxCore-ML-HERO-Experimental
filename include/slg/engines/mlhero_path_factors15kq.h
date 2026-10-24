#ifndef SLG_MLHERO_PATH_FACTORS15KQ_H
#define SLG_MLHERO_PATH_FACTORS15KQ_H
#include <cmath>
#include <limits>
#include "slg/engines/mlhero_glass_factors15ku.h"
#include "slg/engines/mlhero_glass_reverse15kt.h"
// Diagnostic factor ledger. Deliberately NOT a complete spectral path PDF.
namespace slg { namespace mlhero15kq {
constexpr unsigned RootDetailBudget=64u,CandidateBudget=65536u,EdgeBudget=1024u,EdgeDetailBudget=32u;
struct V {double x=0.,y=0.,z=0.;template<class T>static V From(const T&a){return {double(a.x),double(a.y),double(a.z)};}};
inline double Dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
// Bounded, value-owned evidence for the actual prefixes selected at a connection.
constexpr unsigned ConnectionStepBudget=32u;
struct ConnectionStep15ks {
    slg::mlhero15kt::Evidence glassReverse15kt;
    unsigned sourceDepth=0,targetDepth=0,sourceMaterial=0,targetMaterial=0,event=0;
    bool delta=false,sourceVolume=false,targetVolume=false,reverseSeen=false,valid=false;
    bool nativeSeen=false,nativeMassMatches=false;
    double forward=0.,reverse=0.,rr=1.,distance2=0.,jacobian=0.,threshold=0.,nativeMass=0.;
};
struct State {
    slg::mlhero15ku::Factors glassFactors15ku;
    bool sourceGlass15ku=false;
    slg::mlhero15kt::Evidence glassReverse15kt;
    bool selected=false,valid=true,truncated=false,anyVolume=false,lastVolume=false;
    unsigned long long ticket=0ull;
    unsigned depth=0,edges=0,continuous=0,discrete=0,volumeEdges=0,reverseMissing=0;
    V position,rootPosition;
    double rootNative=0.,rootDirect=0.,rootPick=1.;unsigned rootType=0;
    double logContinuous=0.,logDiscrete=0.,logRR=0.,logJacobian=0.,logReverseContinuous=0.;
    bool pending=false,delta=false,reverseSeen=false,killed=false;
    double forward=0.,reverse=0.,rr=1.;
    double lastDistance2=0.,lastTargetCosine=0.,lastJacobian=0.;
    bool lastJacobianApplied=false;
    ConnectionStep15ks connectionSteps[ConnectionStepBudget];
    unsigned connectionStepCount=0,connectionStepsOmitted=0;
    unsigned sourceMaterial=0,sourceEvent=0;
    bool sourceVolume=false,nativeSeen=false,nativeMassMatches=false;
    double nativeThreshold=0.,nativeMass=0.;
    void SampleContext(unsigned material,unsigned event,bool volume,bool glass=false){
        sourceGlass15ku=glass;
        sourceMaterial=material;sourceEvent=event;sourceVolume=volume;
    }
    // Read-only parent observation. A recurrence scalar is never labelled a reverse delta mass.
    void NativeGlass(double threshold,double mass){
        nativeSeen=true;nativeThreshold=threshold;nativeMass=mass;
        nativeMassMatches=std::isfinite(threshold)&&threshold>=0.&&threshold<=1.&&
            std::isfinite(mass)&&mass>0.&&mass<=1.&&mass==forward;
    }
    void Sample(bool isDelta,double p){sourceGlass15ku=false;glassReverse15kt={};nativeSeen=false;nativeMassMatches=false;nativeThreshold=0.;nativeMass=0.;pending=false;delta=isDelta;forward=p;reverse=0.;reverseSeen=false;rr=1.;killed=false;}
    void Reverse(double p){reverse=p;reverseSeen=true;}
    void Roulette(double q,bool dead){rr*=q;killed=killed||dead;}
    void Accept(){pending=!killed;}
    // Native sample density is consumed once, only when the sampled edge arrives.
    template<class P,class N> bool Arrive(unsigned nextDepth,const P&p,const N&normal,bool volume,unsigned targetMaterial=0u){
        if(!selected)return false;
        if(edges>=EdgeBudget){truncated=true;return false;}
        const V target=V::From(p),n=V::From(normal);
        const V d{target.x-position.x,target.y-position.y,target.z-position.z};
        const double r2=Dot(d,d),nn=Dot(n,n);
        const bool linked=nextDepth==depth+1u;
        bool ok=linked&&std::isfinite(r2)&&r2>0.;
        const double cosine=ok&&nn>0.&&std::isfinite(nn)?std::fabs(Dot(n,d))/std::sqrt(nn*r2):0.;
        const double jacobian=volume?(ok?1./r2:0.):(ok?cosine/r2:0.);
        lastDistance2=r2;lastTargetCosine=cosine;lastJacobian=jacobian;lastJacobianApplied=false;
        if(depth>0u){
            ok=ok&&pending&&std::isfinite(forward)&&forward>0.&&std::isfinite(rr)&&rr>0.&&rr<=1.;
            if(ok){
                if(delta){++discrete;logDiscrete+=std::log(forward);++reverseMissing;}
                else {
                    ++continuous;logContinuous+=std::log(forward);
                    if(!(jacobian>0.)||!std::isfinite(jacobian))ok=false;
                    else {logJacobian+=std::log(jacobian);lastJacobianApplied=true;}
                    if(reverseSeen&&std::isfinite(reverse)&&reverse>=0.)
                        logReverseContinuous+=reverse>0.?std::log(reverse):-std::numeric_limits<double>::infinity();
                    else {++reverseMissing;ok=false;}
                }
                logRR+=std::log(rr);
            }
        }
        if(depth>0u){
            glassFactors15ku.Observe(sourceGlass15ku,delta,ok,forward,rr,glassReverse15kt);
            if(connectionStepCount<ConnectionStepBudget){
                auto &step=connectionSteps[connectionStepCount++];
                step.glassReverse15kt=glassReverse15kt;
                step.sourceDepth=depth;step.targetDepth=nextDepth;step.sourceMaterial=sourceMaterial;
                step.targetMaterial=targetMaterial;step.event=sourceEvent;step.delta=delta;
                step.sourceVolume=sourceVolume;step.targetVolume=volume;step.reverseSeen=reverseSeen;
                step.valid=ok;step.forward=forward;step.reverse=reverse;step.rr=rr;
                step.distance2=r2;step.jacobian=jacobian;step.nativeSeen=nativeSeen;
                step.nativeMassMatches=nativeMassMatches;step.threshold=nativeThreshold;step.nativeMass=nativeMass;
            }else ++connectionStepsOmitted;
        }
        valid=valid&&ok;anyVolume=anyVolume||volume||lastVolume;
        if(volume)++volumeEdges;
        position=target;lastVolume=volume;depth=nextDepth;++edges;pending=false;return true;
    }
};
} }
#endif

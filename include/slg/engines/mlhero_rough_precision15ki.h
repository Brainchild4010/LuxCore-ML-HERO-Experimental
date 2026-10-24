#ifndef MLHERO_ROUGH_PRECISION_15KI_H
#define MLHERO_ROUGH_PRECISION_15KI_H
#include <atomic>
#include <cmath>
#include "slg/engines/mlhero_rough_optics15kk.h"
namespace slg { namespace mlhero15ki {
struct Vec {
    float x=0.f,y=0.f,z=0.f;
    Vec()=default;
    template<class V> Vec(const V &v):x(v.x),y(v.y),z(v.z){}
};
struct Snapshot {
    bool valid=false,fromLight=false,transmit=false,vndf=false,importance=false;
    bool krBlack=false,ktBlack=false;
    bool precise15kl=false;
    mlhero15kk::V fixedPrecise15kl,sampledPrecise15kl,whPrecise15kl;
    const void *material=nullptr;
    float nc=0.f,nt=0.f,roughness=0.f,anisotropy=0.f;
    float sampleD=0.f,sampleSpecPdf=0.f,probability=0.f,samplePdf=0.f;
    Vec fixed,sampled,wh;
};
struct CriticalAngles {double fixedNorm=0.,normalNorm=0.,cosine=0.,discriminant=0.,transmission=0.;};
struct Factors {
    CriticalAngles angles;
    Vec wh;
    double d=0.,referenceD=0.,specPdf=0.,probability=0.,jacobian=0.,pdf=0.;
};
struct Comparison {float localPdf=0.f,roundTripPdf=0.f;Factors local,roundTrip;CriticalAngles sampledAngles;};
inline thread_local bool armed=false;
inline thread_local Snapshot current;
inline std::atomic<unsigned long long> selected[4][3]{};
inline void ResetCounters(){for(auto &side:selected)for(auto &n:side)n.store(0,std::memory_order_relaxed);}
inline void ResetThread(){armed=false;current=Snapshot();}
inline void Begin(bool enabled){armed=enabled;if(enabled){current.valid=false;current.precise15kl=false;}}
inline double Relative(double a,double b){
    const double den=std::fmax(1e-20,std::fmax(std::fabs(a),std::fabs(b)));
    return std::fabs(a-b)/den;
}
// Same Schlick Z for a normalized direction, expressed without 1-z*z.
// Schlick A is applied by the caller; this reference never enters transport.
inline double ReferenceZ(float roughness,Vec h){
    const double x=h.x,y=h.y,z=h.z,r=roughness;
    const double norm=x*x+y*y+z*z;
    if(!(norm>0.)||!(r>0.))return 0.;
    const double den=(x*x+y*y+r*z*z)/norm;
    return (r/den)/den;
}
inline int Select(unsigned c,double sample,double queried,unsigned long long &ticket){
    if(c>=4||!std::isfinite(sample)||!std::isfinite(queried)||sample<=0.||queried<=0.)return -1;
    const double e=Relative(sample,queried);
    const unsigned band=e>=.05?2:e>=.01?1:0;
    const unsigned long long cap=band==1?16:8;
    auto &n=selected[c][band];auto old=n.load(std::memory_order_relaxed);
    while(old<cap){if(n.compare_exchange_weak(old,old+1,std::memory_order_relaxed)){ticket=old;return band;}}
    return -1;
}
}}
#endif

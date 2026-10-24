// Precision of existing CPU HERO BIDIR MIS equations; no epsilon or clamping.
#ifndef SLG_MLHERO_MIS_PRECISION15KN_H
#define SLG_MLHERO_MIS_PRECISION15KN_H
#include <cmath>
#include "slg/engines/mlhero_mis_audit15km.h"
namespace slg { namespace mlhero15kn {
using State=mlhero15km::State;
struct Result {
    double cm,vc,vm;
    State Rounded() const {return State(float(cm),float(vc),float(vm));}
};
template<class A,class B> inline double AbsDot(const A &a,const B &b) {
    return std::fabs(double(a.x)*double(b.x)+double(a.y)*double(b.y)+double(a.z)*double(b.z));
}
inline Result Bounce(const State &a,float forward,float reverse,float cosine,
        bool specular,float vmFactor,float vcFactor) {
    const double c=cosine;
    if(specular) {
        const double factor=c*c;
        return {0.,double(a.vc)*factor,double(a.vm)*factor};
    }
    const double f=forward,r=reverse,q=c/f,factor=q*q,reverseSquare=r*r;
    return {(1./f)*(1./f),
        factor*(double(a.vc)*reverseSquare+double(a.cm)+double(vmFactor)),
        factor*(double(a.vm)*reverseSquare+double(a.cm)*double(vcFactor)+1.)};
}
inline Result Geometry(const State &a,double cosine,float distance,bool distanceApplied) {
    const double factor=1./(cosine*cosine),t=distance,t2=t*t;
    return {double(a.cm)*(distanceApplied?t2*t2:1.)*factor,
        double(a.vc)*factor,double(a.vm)*factor};
}
// Diagnostic replay only. Keep legacy float grouping, including EYE vs LIGHT geometry.
inline State LegacyBounce(const State &a,float f,float r,float c,bool specular,float vm,float vc) {
    if(specular)return {0.f,a.vc*(c*c),a.vm*(c*c)};
    const float q=c/f,factor=q*q,reverseSquare=r*r,inv=1.f/f;
    return {inv*inv,factor*(a.vc*reverseSquare+a.cm+vm),factor*(a.vm*reverseSquare+a.cm*vc+1.f)};
}
inline State LegacyGeometry(const State &a,float c,float t,bool distanceApplied,bool eye) {
    const float factor=1.f/(c*c),t2=t*t,dist=distanceApplied?t2*t2:1.f;
    const float cm=eye?a.cm*(dist*factor):(a.cm*dist)*factor;
    return {cm,a.vc*factor,a.vm*factor};
}
enum Stage {SpecularBounce,NonSpecularBounce,LightGeometry,EyeGeometry,StageCount};
inline const char *Name(unsigned s) {
    static const char *names[]={"SPECULAR_BOUNCE","NON_SPECULAR_BOUNCE","LIGHT_GEOMETRY","EYE_GEOMETRY"};
    return s<StageCount?names[s]:"UNKNOWN";
}
} }
#endif

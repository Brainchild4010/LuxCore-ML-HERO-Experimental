#ifndef SLG_MLHERO_METAL_DIRECTION15KY_H
#define SLG_MLHERO_METAL_DIRECTION15KY_H
#include "slg/engines/mlhero_rough_frame15kl.h"
#include "slg/engines/mlhero_volume_direction15kp.h"
namespace slg { namespace mlhero15ky {
// Same normalized double basis for Metal2 Sample, Evaluate, Pdf and lane helpers.
// Keep invalid input visible instead of replacing it with an arbitrary direction.
template<class F,class T> inline T Local(const F&f,const T&v){
    const mlhero15kl::Frame precise(f);const auto a=precise.ToLocal(v);
    return mlhero15kp::Normalize(T(float(a.x),float(a.y),float(a.z)));
}
// Same analytic Schlick inverse CDF, with sin(theta) computed directly.
// Avoid a discrete pole caused by float subtraction 1-cos(theta)^2.
inline mlhero15kk::V SampleHalf(double r,double anis,double u,double v){
    const double denominator=r*(1.-u)+u;
    const double cosine=std::sqrt(u/denominator),sine=std::sqrt(r*(1.-u)/denominator);
    const double p=1.-std::fabs(anis),p2=p*p;double a=v*4.,phi;
    const double pi=3.14159265358979323846264338327950288;
    auto angle=[&](double q){return .5*pi*std::sqrt(q*p2/(1.-q*(1.-p2)));};
    if(a<1.)phi=angle(a*a);
    else if(a<2.){a=2.-a;phi=pi-angle(a*a);}
    else if(a<3.){a-=2.;phi=pi+angle(a*a);}
    else{a=4.-a;phi=2.*pi-angle(a*a);}
    if(anis>0.)phi+=.5*pi;
    return {sine*std::cos(phi),sine*std::sin(phi),cosine};
}
} }
#endif

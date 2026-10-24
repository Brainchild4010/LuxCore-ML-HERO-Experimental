#ifndef SLG_MLHERO_METAL_PDF15LB_H
#define SLG_MLHERO_METAL_PDF15LB_H
#include <cmath>
namespace slg { namespace mlhero15lb {
// Density of mlhero15ky::SampleHalf with respect to half-vector solid angle.
// Differentiate its radial inverse CDF and its four-quadrant azimuth map.
// This is a sampling probability, NOT the Schlick BRDF distribution D.
// Rejected reflected directions retain their probability mass: no resampling.
inline double HalfPdf(double rough,double anis,double x,double y,double z){
    const double n2=x*x+y*y+z*z,p=1.-std::fabs(anis);
    if(!(rough>0.)||!(p>0.)||!(n2>0.)||!std::isfinite(n2)||
            !std::isfinite(rough)||!std::isfinite(anis))return 0.;
    const double c=std::fabs(z)/std::sqrt(n2);
    if(!(c>0.))return 0.;
    const double pi=3.14159265358979323846264338327950288;
    double angle=std::atan2(y,x)-(anis>0.?.5*pi:0.);
    angle=std::fmod(angle+4.*pi,2.*pi);
    const unsigned quadrant=unsigned(angle/(.5*pi));
    const double fraction=angle/(.5*pi)-quadrant;
    const double a=(quadrant&1u)?1.-fraction:fraction;
    const double azimuthDen=p*p+a*a*(1.-p*p);
    const double phiDensity=p*p/(2.*pi*azimuthDen*std::sqrt(azimuthDen));
    const double radialDen=(x*x+y*y+rough*z*z)/n2;
    return (2.*rough*c/radialDen)/radialDen*phiDensity;
}
} }
#endif

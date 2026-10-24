#ifndef MLHERO_ROUGH_OPTICS_15KK_H
#define MLHERO_ROUGH_OPTICS_15KK_H
#include <cmath>
namespace slg { namespace mlhero15kk {
struct V {
    double x=0.,y=0.,z=0.;
    V()=default;
    V(double a,double b,double c):x(a),y(b),z(c){}
    template<class T> explicit V(const T &v):x(v.x),y(v.y),z(v.z){}
    V operator+(V b)const{return {x+b.x,y+b.y,z+b.z};}
    V operator-(V b)const{return {x-b.x,y-b.y,z-b.z};}
    V operator*(double s)const{return {x*s,y*s,z*s};}
    double Norm2()const{return x*x+y*y+z*z;}
};
inline double Dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline V Cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline V Unit(V a){const double n=std::sqrt(a.Norm2());return n>0.&&std::isfinite(n)?a*(1./n):V();}
inline double G1(double roughness,V v){
    v=Unit(v);const double c=std::fabs(v.z);
    if(!(c>0.))return 0.;
    return 2.*c/(c+std::sqrt(c*c+roughness*(v.x*v.x+v.y*v.y)));
}
// nTransmitted/nIncident and the two nonnegative interface cosines.
inline double Transmission(double ratio,double ci,double ct){
    const double pden=ct+ratio*ci,sden=ci+ratio*ct;
    if(!(pden>0.)||!(sden>0.))return 0.;
    // 1-R = 4ab/(a+b)^2 avoids subtracting Fresnel values near one.
    return .5*(4.*ratio*ci*ct/(pden*pden)+4.*ratio*ci*ct/(sden*sden));
}
struct Interface {
    V fixed,h;
    double cosine=0.,eta=1.,discriminant=0.,cosTransmitted=0.,transmission=0.;
    bool valid=false,tir=true;
};
inline Interface MakeInterface(double nc,double nt,V incoming,V h){
    Interface a;a.fixed=Unit(incoming);a.h=Unit(h);
    if(!(nc>0.)||!(nt>0.)||a.fixed.Norm2()==0.||a.h.Norm2()==0.)return a;
    a.cosine=Dot(a.fixed,a.h);
    // wh stays on the upper side; reject invisible NDF normals.
    if(a.cosine*a.fixed.z<=0.)return a;
    a.valid=true;
    a.eta=a.fixed.z>0.?nc/nt:nt/nc;
    const double sine2=Cross(a.fixed,a.h).Norm2();
    a.discriminant=std::fma(-a.eta*a.eta,sine2,1.);
    a.tir=!(a.discriminant>0.);
    if(a.tir)return a;
    a.cosTransmitted=std::sqrt(a.discriminant);
    a.transmission=Transmission(1./a.eta,std::fabs(a.cosine),a.cosTransmitted);
    return a;
}
inline V Refract(const Interface &a){
    if(!a.valid||a.tir)return {};
    const double signedT=a.fixed.z>0.?-a.cosTransmitted:a.cosTransmitted;
    return Unit(a.h*(a.eta*a.cosine+signedT)-a.fixed*a.eta);
}
inline V Reflect(const Interface &a){return Unit(a.h*(2.*a.cosine)-a.fixed);}
inline double Probability(double transmission,bool krBlack,bool ktBlack,bool importance){
    return krBlack?1.:(ktBlack?0.:(importance?transmission:.5));
}
struct Pair {
    V light,eye,h;
    double eta=1.,lengthSquared=0.,cosLight=0.,cosEye=0.,transmission=0.;
    bool valid=false,transmit=false;
};
inline Pair MakePair(double nc,double nt,V light,V eye){
    Pair p;p.light=Unit(light);p.eye=Unit(eye);
    if(!(nc>0.)||!(nt>0.)||p.light.Norm2()==0.||p.eye.Norm2()==0.)return p;
    p.transmit=p.light.z*p.eye.z<0.;
    p.eta=p.transmit?(p.light.z>0.?nc/nt:nt/nc):1.;
    V raw=p.light*p.eta+p.eye;p.lengthSquared=raw.Norm2();
    if(!(p.lengthSquared>0.))return p;
    p.h=Unit(raw);if(p.h.z<0.)p.h=p.h*(-1.);
    p.cosLight=Dot(p.light,p.h);p.cosEye=Dot(p.eye,p.h);
    if(p.transmit){
        if(p.cosLight*p.cosEye>=0.||p.cosLight*p.light.z<=0.||p.cosEye*p.eye.z<=0.)return p;
        // Both directions are known: use both interface cosines rather than
        // reconstructing a tiny transmitted cosine by subtracting near one.
        p.transmission=Transmission(1./p.eta,std::fabs(p.cosLight),std::fabs(p.cosEye));
    }else{
        const auto a=MakeInterface(nc,nt,p.light,p.h);
        if(!a.valid)return p;
        p.transmission=a.transmission;
    }
    p.valid=true;return p;
}
}}
#endif

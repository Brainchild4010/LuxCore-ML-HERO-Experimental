// CPU HERO volume directions: unit axes and consistent angular inputs.
#ifndef SLG_MLHERO_VOLUME_DIRECTION15KP_H
#define SLG_MLHERO_VOLUME_DIRECTION15KP_H
#include <cmath>
#include <algorithm>
namespace slg { namespace mlhero15kp {
struct V {
    double x,y,z;
    template<class T> static V From(const T&a){return {double(a.x),double(a.y),double(a.z)};}
    V operator*(double a)const{return {x*a,y*a,z*a};}
    V operator+(V b)const{return {x+b.x,y+b.y,z+b.z};}
    V operator-(V b)const{return {x-b.x,y-b.y,z-b.z};}
};
inline double Dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline V Cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline bool Unit(V a,V &out){
    const double n2=Dot(a,a);
    if(!(n2>0.)||!std::isfinite(n2))return false;
    out=a*(1./std::sqrt(n2));return true;
}
template<class T> inline T Cast(V a){return T(float(a.x),float(a.y),float(a.z));}
template<class T> inline T Normalize(const T&a){
    V u;if(!Unit(V::From(a),u))return a; // Invalid directions remain visible to 15ko.
    return Cast<T>(u);
}
struct Basis {
    V x,y,z;bool valid=false;
    template<class T> explicit Basis(const T &axis){
        if(!Unit(V::From(axis),z))return;
        const double h=std::sqrt(z.x*z.x+z.y*z.y);
        V candidate=h<1e-5?V{1.,0.,0.}:V{-z.y/h,z.x/h,0.};
        if(!Unit(candidate-z*Dot(candidate,z),y))return;
        if(!Unit(Cross(z,y),x))return;
        y=Cross(x,z);valid=true;
    }
    V ToWorld(V a)const{return x*a.x+y*a.y+z*a.z;}
};
template<class T> inline float AngularCos(const T &a,const T &b){
    V x,y;
    if(!Unit(V::From(a),x)||!Unit(V::From(b),y))
        return float(Dot(V::From(a),V::From(b)));
    return float(std::max(-1.,std::min(1.,Dot(x,y))));
}
template<class T> inline bool SampleDirection(const T &fixed,float cosine,float phi,T &out){
    if(!std::isfinite(cosine)||!std::isfinite(phi))return false;
    const Basis basis(fixed);if(!basis.valid)return false;
    const double c=std::max(-1.,std::min(1.,double(cosine)));
    const double s=std::sqrt(std::max(0.,1.-c*c));
    V result=basis.ToWorld({s*std::cos(double(phi)),s*std::sin(double(phi)),c}),unit;
    if(!Unit(result,unit))return false;
    out=Cast<T>(unit);return true;
}
} }
#endif

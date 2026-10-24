#ifndef MLHERO_ROUGH_FRAME_15KL_H
#define MLHERO_ROUGH_FRAME_15KL_H
#include "slg/engines/mlhero_rough_optics15kk.h"
#include <atomic>
namespace slg { namespace mlhero15kl {
using V=mlhero15kk::V;
struct Frame {
    V x,y,z;
    template<class F> explicit Frame(const F &f){
        z=mlhero15kk::Unit(V(f.Z));
        x=mlhero15kk::Unit(V(f.X)-z*mlhero15kk::Dot(V(f.X),z));
        if(x.Norm2()==0.)x=mlhero15kk::Unit(mlhero15kk::Cross(V(f.Y),z));
        y=mlhero15kk::Cross(z,x);
    }
    template<class T> V ToLocal(const T &a)const{
        const V v(a);return {mlhero15kk::Dot(v,x),mlhero15kk::Dot(v,y),mlhero15kk::Dot(v,z)};
    }
    V ToWorld(V v)const{return x*v.x+y*v.y+z*v.z;}
};
// Diagnostic quotas only; the frame calculation never reads these counters.
inline std::atomic<unsigned long long> extraFrameDetails[2]{};
inline void ResetCounters(){for(auto &v:extraFrameDetails)v.store(0,std::memory_order_relaxed);}
inline bool SelectExtra(bool fromLight,double a,double b,unsigned long long &ticket){
    if(!(a>0.)||!(b>0.)||!std::isfinite(a)||!std::isfinite(b))return false;
    const double e=std::fabs(a-b)/std::fmax(a,b);
    if(e<.001||e>=.01)return false;
    auto &n=extraFrameDetails[fromLight?1:0];auto old=n.load(std::memory_order_relaxed);
    while(old<16){if(n.compare_exchange_weak(old,old+1,std::memory_order_relaxed)){ticket=old;return true;}}
    return false;
}
}}
#endif

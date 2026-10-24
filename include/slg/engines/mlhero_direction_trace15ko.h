// Read-only trace of native BSDF direction lengths. No renderer dependencies.
#ifndef SLG_MLHERO_DIRECTION_TRACE15KO_H
#define SLG_MLHERO_DIRECTION_TRACE15KO_H
#include <cmath>
#include <atomic>
namespace slg { namespace mlhero15ko {
constexpr unsigned RingSize=12u;
struct V {
    double x=0.,y=0.,z=0.;
    template<class T> static V From(const T &a){return {double(a.x),double(a.y),double(a.z)};}
    double Norm2() const{return x*x+y*y+z*z;}
};
inline double Dot(const V&a,const V&b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline bool Bad(double n2){return !std::isfinite(n2)||std::fabs(n2-1.)>2e-5;}
struct Step {
    unsigned depth=0u,material=0u,event=0u;
    bool valid=false,volume=false,preciseFrame=false;
    V input,localFixed,localSample,worldSample,ray,x,y,z;
    double pdf=0.,cosine=0.;
    unsigned Mask() const {
        const bool basis=Bad(x.Norm2())||Bad(y.Norm2())||Bad(z.Norm2())||
            std::fabs(Dot(x,y))>2e-5||std::fabs(Dot(x,z))>2e-5||std::fabs(Dot(y,z))>2e-5;
        return (Bad(input.Norm2())?1u:0u)|(Bad(localFixed.Norm2())?2u:0u)|
            (basis?4u:0u)|(Bad(localSample.Norm2())?8u:0u)|
            (Bad(worldSample.Norm2())?16u:0u)|(Bad(ray.Norm2())?32u:0u);
    }
};
struct Ring {
    Step steps[RingSize];unsigned count=0u,next=0u,lastSample=0u;bool haveSample=false;
    void Push(const Step&s,unsigned sample){
        if(!haveSample||sample!=lastSample||s.depth<=1u){count=0u;next=0u;}
        haveSample=true;lastSample=sample;steps[next]=s;next=(next+1u)%RingSize;
        if(count<RingSize)++count;
    }
    const Step &At(unsigned i) const{return steps[(next+RingSize-count+i)%RingSize];}
};
inline thread_local bool armed=false;
inline thread_local Step current;
inline thread_local Ring rings[2];
inline std::atomic<unsigned long long> events[2][2],details[2][2],writeFailures[2][2];
inline void ResetCounters(){for(unsigned s=0;s<2;++s)for(unsigned k=0;k<2;++k){
    events[s][k].store(0ull,std::memory_order_relaxed);
    details[s][k].store(0ull,std::memory_order_relaxed);
    writeFailures[s][k].store(0ull,std::memory_order_relaxed);
}}
inline void ResetThread(){armed=false;current=Step();rings[0]=Ring();rings[1]=Ring();}
inline void Begin(bool enabled){armed=enabled;if(enabled)current=Step();}
template<class A,class B,class C,class D,class E,class F,class G> inline void Capture(
        const A &input,const B &fixed,const C &sample,const D &world,
        const E &x,const F &y,const G &z,bool precise,bool volume) {
    if(!armed)return;
    current.valid=true;current.input=V::From(input);current.localFixed=V::From(fixed);
    current.localSample=V::From(sample);current.worldSample=V::From(world);
    current.x=V::From(x);current.y=V::From(y);current.z=V::From(z);
    current.preciseFrame=precise;current.volume=volume;
}
} }
#endif

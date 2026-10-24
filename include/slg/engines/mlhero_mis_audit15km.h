// CPU BIDIR MIS transition diagnosis. No renderer state is changed.
#ifndef SLG_MLHERO_MIS_AUDIT15KM_H
#define SLG_MLHERO_MIS_AUDIT15KM_H
#include <cmath>
namespace slg { namespace mlhero15km {
enum Stage { LightInit, EyeInit, LightGeometry, EyeGeometry, Bounce, StageCount };
inline const char *Name(unsigned s) {
    static const char *names[]={"LIGHT_INIT","EYE_INIT","LIGHT_GEOMETRY","EYE_GEOMETRY","BOUNCE"};
    return s<StageCount?names[s]:"UNKNOWN";
}
struct State {
    float cm,vc,vm;
    State(float a=0.f,float b=0.f,float c=0.f):cm(a),vc(b),vm(c){}
    unsigned Bad() const {
        return (!std::isfinite(cm)?1u:0u)|(!std::isfinite(vc)?2u:0u)|(!std::isfinite(vm)?4u:0u);
    }
};
struct Inputs {
    float forward=0.f,reverse=0.f,cosine=0.f,distance=0.f;
    float vmFactor=0.f,vcFactor=0.f,rr=1.f;
    bool specular=false,distanceApplied=false,lightCompetitor=false;
    unsigned sourceID=0u,sourceType=0u,event=0u;
};
struct Probe {
    unsigned beforeBad=0u,afterBad=0u,newBad=0u;
    // Double evaluation of THIS operation, not a complete double path history.
    double cm=0.,vc=0.,vm=0.,factor=0.,reverseSquare=0.,historyVC=0.,historyVM=0.;
    // Float intermediate replay reveals overflow followed by multiplication by zero.
    float floatFactor=0.f,floatReverseSquare=0.f,floatHistoryVC=0.f,floatHistoryVM=0.f;
};
inline Probe Analyze(unsigned stage,const State &a,const State &b,const Inputs &i) {
    Probe p;p.beforeBad=a.Bad();p.afterBad=b.Bad();p.newBad=p.afterBad&~p.beforeBad;
    const double f=i.forward,r=i.reverse,c=i.cosine,t=i.distance;
    if(stage==LightInit) {
        p.cm=(r/f)*(r/f);p.vc=i.lightCompetitor?(c/f)*(c/f):0.;p.vm=p.vc*i.vcFactor;
    } else if(stage==EyeInit) {
        p.cm=(1./f)*(1./f);
    } else if(stage==LightGeometry||stage==EyeGeometry) {
        p.factor=1./(c*c);const double dist=i.distanceApplied?(t*t)*(t*t):1.;
        p.cm=double(a.cm)*dist*p.factor;p.vc=double(a.vc)*p.factor;p.vm=double(a.vm)*p.factor;
        p.floatFactor=1.f/(i.cosine*i.cosine);
    } else if(i.specular) {
        p.factor=c*c;p.cm=0.;p.vc=double(a.vc)*p.factor;p.vm=double(a.vm)*p.factor;
        p.floatFactor=i.cosine*i.cosine;
    } else {
        p.factor=(c/f)*(c/f);p.reverseSquare=r*r;
        p.historyVC=double(a.vc)*p.reverseSquare+a.cm+i.vmFactor;
        p.historyVM=double(a.vm)*p.reverseSquare+double(a.cm)*i.vcFactor+1.;
        p.cm=(1./f)*(1./f);p.vc=p.factor*p.historyVC;p.vm=p.factor*p.historyVM;
        const float q=i.cosine/i.forward;p.floatFactor=q*q;
        p.floatReverseSquare=i.reverse*i.reverse;
        p.floatHistoryVC=a.vc*p.floatReverseSquare+a.cm+i.vmFactor;
        p.floatHistoryVM=a.vm*p.floatReverseSquare+a.cm*i.vcFactor+1.f;
    }
    return p;
}
} }
#endif

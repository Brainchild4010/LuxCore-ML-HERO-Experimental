// Read-only audit of actual material lane updates, including RR-killed bounces.
#ifndef SLG_MLHERO_MATERIAL_RR15FU_H
#define SLG_MLHERO_MATERIAL_RR15FU_H
#include <cmath>
#include <cstdio>
namespace slg { namespace {
struct MLHeroMaterialRRBin15fu {
    unsigned long long calls=0, active=0, killed=0, invalid=0, rgb=0, valid=0;
    unsigned long long candidateKilled=0, different=0, etaValid=0, validActive=0, etaActive=0, etaKilled=0, etaDifferent=0;
    double preMaxSum=0, betaSum=0, betaMax=0, qSum=0, etaQSum=0;
    unsigned long long routes[5]={};
};
struct MLHeroMaterialRRState15fu {
    bool enabled=false;
    MLHeroMaterialRRBin15fu bins[2][3][66];
};
static thread_local MLHeroMaterialRRState15fu mlHeroMaterialRR15fu;
static void MLHeroMaterialRRReset15fu(bool enabled) {
    mlHeroMaterialRR15fu=MLHeroMaterialRRState15fu();
    mlHeroMaterialRR15fu.enabled=enabled;
}
// Inputs are copied from the actual lane update. Current RR compensation
// is removed by the caller; previous RR and emission weights are retained.
static void MLHeroMaterialRRRecord15fu(unsigned side,unsigned material,unsigned depth,
        bool active,bool killed,double random,double cap,unsigned n,bool collapsed,
        const double pre[8][3],const double post[8][3],const unsigned routes[8],
        bool etaHistoryValid,const double eta[8]) {
    if(!mlHeroMaterialRR15fu.enabled||side>1||material>2)return;
    auto &a=mlHeroMaterialRR15fu.bins[side][material][depth<=64?depth:65];
    ++a.calls;if(active){++a.active;if(killed)++a.killed;}
    if(n<2||n>8){++a.invalid;return;}
    const unsigned count=collapsed?1:n;
    const double norm=collapsed?1./n:1.;
    bool finite=true,scalar=true,etaOK=etaHistoryValid&&side==0;
    double pm=0,beta=0,eb=0;
    for(unsigned i=0;i<count;++i){
        if(routes[i]<5)++a.routes[routes[i]];
        for(unsigned c=0;c<3;++c){
            finite=finite&&std::isfinite(pre[i][c])&&pre[i][c]>=0&&
                std::isfinite(post[i][c])&&post[i][c]>=0;
        }
        scalar=scalar&&pre[i][0]==pre[i][1]&&pre[i][0]==pre[i][2]&&
            post[i][0]==post[i][1]&&post[i][0]==post[i][2];
        const double p=pre[i][0]*norm,b=post[i][0]*norm;
        if(p>pm)pm=p;
        if(b>beta)beta=b;
        const double e=b*eta[i];
        etaOK=etaOK&&std::isfinite(eta[i])&&eta[i]>0&&std::isfinite(e)&&e>=0;
        if(e>eb)eb=e;
    }
    if(!finite){++a.invalid;return;}
    if(!scalar){++a.rgb;return;}
    ++a.valid;a.preMaxSum+=pm;a.betaSum+=beta;if(beta>a.betaMax)a.betaMax=beta;
    const double q=beta<cap?cap:(beta>1?1:beta);a.qSum+=q;
    if(active){++a.validActive;const bool k=q<random;
        a.candidateKilled+=k;a.different+=(k!=killed);}
    if(etaOK){++a.etaValid;const double eq=eb<cap?cap:(eb>1?1:eb);a.etaQSum+=eq;
        if(active){++a.etaActive;const bool k=eq<random;a.etaKilled+=k;a.etaDifferent+=(k!=killed);}}
}
static void MLHeroMaterialRRWrite15fu(FILE *f,const char *engine,unsigned thread) {
    if(!f||!mlHeroMaterialRR15fu.enabled)return;
    fprintf(f,"phase15fu MATERIAL_RR_CONFIG engine=%s thread=%u diagnosticOnly=1 rendererPhysicsChanged=0 killedBouncesIncluded=1 currentRRCompensationExcluded=1 collapseXNRemoved=1 strictScalarRequired=1 routes=Metal2,Glossy2Split,GenericSPD,ScalarShared,RGBShared lightEmissionScaleUnnormalized=1 candidateApplied=0\n",engine,thread);
    const char *names[3]={"MATTE","METAL2","GLOSSY2"};
    for(unsigned s=0;s<2;++s)for(unsigned m=0;m<3;++m)for(unsigned d=0;d<66;++d){
        const auto &a=mlHeroMaterialRR15fu.bins[s][m][d];if(!a.calls)continue;
        fprintf(f,"phase15fu MATERIAL_RR engine=%s thread=%u side=%s material=%s depthBin=%u calls=%llu active=%llu liveKilled=%llu invalid=%llu rgbHistory=%llu validScalar=%llu validActive=%llu candidateKilled=%llu different=%llu preMaxSum=%.17g betaSum=%.17g betaMax=%.17g candidateQSum=%.17g validEta=%llu etaActive=%llu etaKilled=%llu etaDifferent=%llu etaQSum=%.17g routes=%llu,%llu,%llu,%llu,%llu\n",engine,thread,s?"LIGHT":"EYE",names[m],d,a.calls,a.active,a.killed,a.invalid,a.rgb,a.valid,a.validActive,a.candidateKilled,a.different,a.preMaxSum,a.betaSum,a.betaMax,a.qSum,a.etaValid,a.etaActive,a.etaKilled,a.etaDifferent,a.etaQSum,a.routes[0],a.routes[1],a.routes[2],a.routes[3],a.routes[4]);
    }
}
} }
#endif

#ifndef SLG_MLHERO_MATERIAL_RR_EXPERIMENT15FV_H
#define SLG_MLHERO_MATERIAL_RR_EXPERIMENT15FV_H
#include <cmath>
#include <cstdio>
namespace slg { namespace {
// Actual post-material lane values, before current RR compensation.
// Eta history is camera-only. RGB/negative/nonfinite inputs retain baseline.
static bool MLHeroMaterialBeta15fv(unsigned n,bool collapsed,
        const double post[8][3],bool etaValid,const double eta[8],double *beta) {
    *beta=0.;if(n<2||n>8||!etaValid)return false;
    const unsigned count=collapsed?1:n;
    for(unsigned i=0;i<count;++i){
        for(unsigned c=0;c<3;++c)
            if(!std::isfinite(post[i][c])||post[i][c]<0.)return false;
        if(post[i][0]!=post[i][1]||post[i][0]!=post[i][2]||
                !std::isfinite(eta[i])||eta[i]<=0.)return false;
        const double b=post[i][0]*(collapsed?1./n:1.)*eta[i];
        if(!std::isfinite(b)||b<0.)return false;
        if(b>*beta)*beta=b;
    }
    return true;
}
struct MLHeroMaterialLiveBin15fv {
    unsigned long long active=0,selected=0,killed=0,baselineKilled=0,different=0;
    double qSum=0.;
};
struct MLHeroMaterialLiveState15fv {bool enabled=false;MLHeroMaterialLiveBin15fv bins[3][66];};
static thread_local MLHeroMaterialLiveState15fv mlHeroMaterialLive15fv;
static void MLHeroMaterialLiveReset15fv(bool enabled){
    mlHeroMaterialLive15fv=MLHeroMaterialLiveState15fv();mlHeroMaterialLive15fv.enabled=enabled;
}
static void MLHeroMaterialLiveRecord15fv(unsigned material,unsigned depth,
        bool selected,double baseline,double q,double random,bool killed){
    if(!mlHeroMaterialLive15fv.enabled||material>2)return;
    auto &a=mlHeroMaterialLive15fv.bins[material][depth<=64?depth:65];
    ++a.active;a.selected+=selected;a.killed+=killed;
    const bool old=baseline<random;a.baselineKilled+=old;
    a.different+=(old!=killed);a.qSum+=q;
}
static void MLHeroMaterialLiveWrite15fv(FILE *f,const char *engine,unsigned thread){
    if(!f||!mlHeroMaterialLive15fv.enabled)return;
    const char *names[3]={"MATTE","METAL2","GLOSSY2"};
    for(unsigned m=0;m<3;++m)for(unsigned d=0;d<66;++d){
        const auto &a=mlHeroMaterialLive15fv.bins[m][d];if(!a.active)continue;
        fprintf(f,"phase15fv MATERIAL_RR_LIVE engine=%s thread=%u side=EYE material=%s depthBin=%u active=%llu selectedSpectral=%llu liveKilled=%llu baselineWouldKill=%llu decisionDifferent=%llu liveQSum=%.17g sameLivePathComparison=1\n",engine,thread,names[m],d,a.active,a.selected,a.killed,a.baselineKilled,a.different,a.qSum);
    }
}
} }
#endif

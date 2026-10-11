// HERO 15ft: narrow Eye-Glass RR experiment; independent of diagnostics.
#ifndef SLG_MLHERO_RR_EXPERIMENT15FT_H
#define SLG_MLHERO_RR_EXPERIMENT15FT_H
#include <cmath>
#include <cstdio>
namespace slg { namespace {
static float MLHeroRRSelect15ft(float baseline,bool enabled,bool scope,
        bool valid,double beta,double cap,bool *selected) {
    *selected=false;
    if(!enabled||!scope||!valid||!std::isfinite(beta)||beta<0.||
            !std::isfinite(cap)||cap<0.||cap>1.)return baseline;
    const double q=beta<cap?cap:(beta>1.?1.:beta);
    *selected=true;return static_cast<float>(q);
}
struct MLHeroRRLiveBin15ft {
    unsigned long long active=0,selected=0,killed=0,survived=0;
    unsigned long long baselineKilled=0,different=0;
    double baselineQSum=0,liveQSum=0;
};
struct MLHeroRRLiveState15ft {bool enabled=false;MLHeroRRLiveBin15ft bins[2][67];};
static thread_local MLHeroRRLiveState15ft mlHeroRRLiveState15ft;
static void MLHeroRRLiveReset15ft(bool enabled){
    mlHeroRRLiveState15ft=MLHeroRRLiveState15ft();mlHeroRRLiveState15ft.enabled=enabled;
}
static void MLHeroRRLiveRecord15ft(unsigned side,unsigned depth,bool selected,
        float baseline,float live,float random,bool killed){
    if(!mlHeroRRLiveState15ft.enabled||side>1)return;
    auto&a=mlHeroRRLiveState15ft.bins[side][depth<=64?depth:65];
    ++a.active;a.selected+=selected;a.killed+=killed;a.survived+=!killed;
    const bool oldKill=baseline<random;
    a.baselineKilled+=oldKill;a.different+=(oldKill!=killed);
    a.baselineQSum+=baseline;a.liveQSum+=live;
}
static void MLHeroRRLiveWrite15ft(FILE*f,const char*engine,unsigned thread){
    if(!f||!mlHeroRRLiveState15ft.enabled)return;
    for(unsigned s=0;s<2;++s)for(unsigned d=0;d<67;++d){
        const auto&a=mlHeroRRLiveState15ft.bins[s][d];if(!a.active)continue;
        fprintf(f,"phase15ft RR_LIVE engine=%s thread=%u side=%s depthBin=%u active=%llu selectedEta=%llu liveKilled=%llu liveSurvived=%llu baselineWouldKill=%llu decisionDifferent=%llu baselineQSum=%.17g liveQSum=%.17g sameLivePathComparison=1\n",engine,thread,s?"LIGHT":"EYE",d,a.active,a.selected,a.killed,a.survived,a.baselineKilled,a.different,a.baselineQSum,a.liveQSum);
    }
}
} }
#endif

// 15fs: RR outcomes and accumulated scalar wavelength throughput, read-only.
#ifndef SLG_MLHERO_RR_CANDIDATE15FS_H
#define SLG_MLHERO_RR_CANDIDATE15FS_H
#include <cmath>
#include <cstdio>
namespace slg { namespace {
struct MLHeroRRBin15fs {
    unsigned long long calls=0, active=0, killed=0, survived=0, invalidNative=0;
    unsigned long long glass=0, valid=0, invalid=0, etaValid=0, validActive=0, etaActive=0;
    unsigned long long candidateKilled=0, etaKilled=0, different=0, etaDifferent=0;
    double nativeSum=0, candidateSum=0, etaSum=0, betaMax=0;
};
struct MLHeroRRState15fs {
    bool enabled=false;
    MLHeroRRBin15fs bins[2][2][67]; // eye/light, non-specular/specular, depth
};
static thread_local MLHeroRRState15fs mlHeroRRState15fs;
static void MLHeroRRReset15fs(bool enabled) {
    mlHeroRRState15fs=MLHeroRRState15fs();mlHeroRRState15fs.enabled=enabled;
}
// cap remains the user's MINIMUM survival. This is PBRT-inspired, not a
// literal PBRT implementation (which has different activation/cap semantics).
static double MLHeroRRProbability15fs(double beta,double cap) {
    return beta<cap?cap:(beta>1.?1.:beta);
}
static void MLHeroRRRecord15fs(unsigned side,unsigned depth,bool specular,
        bool active,double native,double random,bool killed,bool glass,
        bool valid,double beta,bool etaValid,double etaBeta,double cap) {
    if(!mlHeroRRState15fs.enabled||side>1)return;
    auto&a=mlHeroRRState15fs.bins[side][specular?1:0][depth<=64?depth:65];
    ++a.calls;if(active){++a.active;if(killed)++a.killed;else ++a.survived;}
    if(!std::isfinite(native)||native<0.||native>1.)++a.invalidNative;
    else a.nativeSum+=native;
    if(!glass)return;
    ++a.glass;
    if(!valid||!std::isfinite(beta)||beta<0.){++a.invalid;return;}
    ++a.valid;const double q=MLHeroRRProbability15fs(beta,cap);
    a.candidateSum+=q;if(beta>a.betaMax)a.betaMax=beta;
    if(active){++a.validActive;const bool kill=q<random;a.candidateKilled+=kill;a.different+=(kill!=killed);}
    if(etaValid&&std::isfinite(etaBeta)&&etaBeta>=0.){
        ++a.etaValid;const double eq=MLHeroRRProbability15fs(etaBeta,cap);a.etaSum+=eq;
        if(active){++a.etaActive;const bool kill=eq<random;a.etaKilled+=kill;a.etaDifferent+=(kill!=killed);}
    }
}
static void MLHeroRRWrite15fs(FILE*f,const char*engine,unsigned thread) {
    if(!f||!mlHeroRRState15fs.enabled)return;
    bool any=false;for(unsigned s=0;s<2;++s)for(unsigned e=0;e<2;++e)for(unsigned d=0;d<67;++d)
        any|=mlHeroRRState15fs.bins[s][e][d].calls!=0;
    if(!any)return;
    fprintf(f,"phase15fs RR_CONFIG engine=%s thread=%u diagnosticsOnly=1 candidateApplied=0 rendererPhysicsChanged=0 scope=BIDIRCPU candidate=postSampleAccumulatedScalarLaneMax glassOnly=1 currentRRCompensationExcluded=1 previousRRCompensationRetained=1 collapseXNRemoved=1 capIsMinimumSurvival=1 etaScope=EyeIdealGlassHistoryOnly lightEtaNotCopied=1 sameNativeRandom=1 extraSamplerDimensions=0 fullCounterfactualPath=0 lightEmissionScaleUnnormalized=1 threadLocalPartition=1 depth65=OVERFLOW\n",engine,thread);
    for(unsigned s=0;s<2;++s)for(unsigned e=0;e<2;++e)for(unsigned d=0;d<67;++d){
        const auto&a=mlHeroRRState15fs.bins[s][e][d];if(!a.calls)continue;
        fprintf(f,"phase15fs RR_BIN engine=%s thread=%u side=%s specular=%u depthBin=%u calls=%llu active=%llu nativeKilled=%llu nativeSurvived=%llu invalidNative=%llu glassCalls=%llu validCandidate=%llu invalidCandidate=%llu validEtaCandidate=%llu validActiveCandidate=%llu validActiveEtaCandidate=%llu nativeQSum=%.17g candidateQSum=%.17g etaQSum=%.17g candidateKilled=%llu etaCandidateKilled=%llu decisionDifferent=%llu etaDecisionDifferent=%llu betaMax=%.17g\n",engine,thread,s?"LIGHT":"EYE",e,d,a.calls,a.active,a.killed,a.survived,a.invalidNative,a.glass,a.valid,a.invalid,a.etaValid,a.validActive,a.etaActive,a.nativeSum,a.candidateSum,a.etaSum,a.candidateKilled,a.etaKilled,a.different,a.etaDifferent,a.betaMax);
    }
}
} }
#endif

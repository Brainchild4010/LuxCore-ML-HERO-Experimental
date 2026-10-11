#ifndef SLG_MLHERO_PATH_RR15FZ_H
#define SLG_MLHERO_PATH_RR15FZ_H
#include <cmath>
#include <cstdio>
// Wavelength-domain camera throughput only. No RGB/CIE reconstruction or xN.
inline bool MLHeroPathRRScalar15fz(double r,double g,double b,double eta,double *value) {
    *value=0.;
    if(!std::isfinite(r)||!std::isfinite(g)||!std::isfinite(b)||r<0.||r!=g||r!=b||
        !std::isfinite(eta)||eta<=0.)return false;
    *value=r*eta;
    return std::isfinite(*value)&&*value>=0.;
}
inline bool MLHeroPathEtaAdvance15fz(double history,double nativeEta2,double *next) {
    if(!std::isfinite(history)||history<=0.||!std::isfinite(nativeEta2)||nativeEta2<=0.)return false;
    *next=history/nativeEta2;
    return std::isfinite(*next)&&*next>0.;
}
inline float MLHeroPathRRSelect15fz(float baseline,bool valid,double beta,double cap,bool *selected) {
    *selected=false;
    if(!valid||!std::isfinite(beta)||beta<0.||!std::isfinite(cap)||cap<0.||cap>1.)return baseline;
    *selected=true;
    return static_cast<float>(beta<cap?cap:(beta>1.?1.:beta));
}
struct MLHeroPathRRBin15fz {
    unsigned long long active=0,selected=0,killed=0,different=0;
    double baselineQSum=0.,liveQSum=0.;
};
struct MLHeroPathRRStats15fz {
    MLHeroPathRRBin15fz bins[4];
    unsigned details=0;
};
static thread_local MLHeroPathRRStats15fz mlHeroPathRRStats15fz;
inline void MLHeroPathRRRecord15fz(unsigned kind,bool selected,float baseline,float live,float random,bool killed) {
    if(kind>3u)return;
    auto &v=mlHeroPathRRStats15fz.bins[kind];++v.active;v.selected+=selected;v.killed+=killed;
    v.different+=((baseline<random)!=killed);v.baselineQSum+=baseline;v.liveQSum+=live;
}
inline void MLHeroPathRRWrite15fz(FILE *f,unsigned thread,bool cpuPath,bool glass,bool material) {
    const char *names[4]={"GLASS","MATTE","METAL2","GLOSSY2"};
    fprintf(f,"phase15fz PATH_RR_CONFIG engine=PATHCPU thread=%u cpuPathScope=%u glassEnabled=%u materialEnabled=%u minimumSurvivalCap=1 currentRRExcluded=1 previousRRRetained=1 glassStartDepth=total materialStartDepth=diffusePlusGlossy postMaterialCandidate=1 etaHistory=idealGlassOnly collapseNormalization=PATH_ACTIVE_COUNT_NO_XN compensation=classic_packet_continuation_once rawPdfMIS=unchanged hybridDisabled=1 GPU_UNCHANGED=1 diagnosticsIndependentPhysics=1 legacyBounceLogsPreDeferredRR=1\n",thread,cpuPath?1u:0u,glass?1u:0u,material?1u:0u);
    for(unsigned k=0;k<4u;++k){const auto &v=mlHeroPathRRStats15fz.bins[k];
        fprintf(f,"phase15fz PATH_RR_LIVE thread=%u material=%s active=%llu selected=%llu killed=%llu decisionDifferent=%llu baselineQSum=%.17g liveQSum=%.17g\n",thread,names[k],v.active,v.selected,v.killed,v.different,v.baselineQSum,v.liveQSum);}
}
#endif

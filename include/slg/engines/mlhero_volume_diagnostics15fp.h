// HERO 15fp: read-only ClearVolume segment audit. Shared across translation units.
#ifndef SLG_MLHERO_VOLUME_DIAGNOSTICS15FP_H
#define SLG_MLHERO_VOLUME_DIAGNOSTICS15FP_H
#include <cmath>
#include <cstdio>
namespace slg {
struct MLHeroVolumeBin15fp {
    const void *volume=nullptr;
    unsigned flags=0, texture=0; bool back=false, spectral=false;
    char name[96]={};
    unsigned long long calls=0, valid=0, invalid=0, chromaticSigma=0, chromaticT=0;
    double distanceSum=0, distanceMin=0, distanceMax=0;
    unsigned long long finiteDistances=0;
    double sigmaSum[3]={}, tSum[3]={};
    float firstDistance=0, firstWave=0, firstSigma[3]={}, firstT[3]={}, firstBefore[3]={}, firstAfter[3]={};
};
struct MLHeroVolumeState15fp {
    bool enabled=false, back=false; unsigned flags=0, used=0;
    unsigned long long calls=0, overflow=0;
    MLHeroVolumeBin15fp bins[32];
};
// External-linkage inline function: its local static is one TLS object per thread,
// including callers in Scene, ClearVolume and the engine diagnostic writer.
inline MLHeroVolumeState15fp &MLHeroVolumeStateRef15fp() {
    static thread_local MLHeroVolumeState15fp state;
    return state;
}
inline void MLHeroVolumeReset15fp(bool enabled) {
    auto &s=MLHeroVolumeStateRef15fp(); s=MLHeroVolumeState15fp(); s.enabled=enabled;
}
class MLHeroVolumeRayScope15fp {
    unsigned flags; bool back;
public:
    MLHeroVolumeRayScope15fp(unsigned f,bool b) {
        auto &s=MLHeroVolumeStateRef15fp(); flags=s.flags; back=s.back;
        if(s.enabled){s.flags=f;s.back=b;}
    }
    ~MLHeroVolumeRayScope15fp() {
        auto &s=MLHeroVolumeStateRef15fp(); s.flags=flags;s.back=back;
    }
    MLHeroVolumeRayScope15fp(const MLHeroVolumeRayScope15fp &)=delete;
    MLHeroVolumeRayScope15fp &operator=(const MLHeroVolumeRayScope15fp &)=delete;
};
template<class S> inline void MLHeroVolumeRecord15fp(const void *volume,
        const char *name,unsigned texture,bool spectral,float wave,float distance,
        const S &sigma,const S &t,const S &before,const S &after) {
    auto &s=MLHeroVolumeStateRef15fp(); if(!s.enabled)return; ++s.calls;
    unsigned i=0;
    for(;i<s.used;++i){const auto &b=s.bins[i];
        if(b.volume==volume&&b.flags==s.flags&&b.back==s.back&&b.spectral==spectral)break;}
    if(i==s.used){
        if(s.used==32){++s.overflow;return;}
        ++s.used;auto &b=s.bins[i]; b.volume=volume;b.flags=s.flags;b.back=s.back;b.spectral=spectral;b.texture=texture;
        for(unsigned j=0;name&&name[j]&&j<95;++j){const unsigned char c=name[j];b.name[j]=(c<=32||c=='"'||c=='=')?'_':static_cast<char>(c);}
    }
    auto &b=s.bins[i]; ++b.calls;
    if(std::isfinite(distance)){
        if(!b.finiteDistances)b.distanceMin=b.distanceMax=distance;
        if(distance<b.distanceMin)b.distanceMin=distance;
        if(distance>b.distanceMax)b.distanceMax=distance;
        ++b.finiteDistances;b.distanceSum+=distance;
    }
    bool valid=true;
    for(unsigned c=0;c<3;++c)valid=valid&&std::isfinite(sigma.c[c])&&std::isfinite(t.c[c])&&std::isfinite(before.c[c])&&std::isfinite(after.c[c]);
    if(valid){++b.valid;for(unsigned c=0;c<3;++c){b.sigmaSum[c]+=sigma.c[c];b.tSum[c]+=t.c[c];}}
    else ++b.invalid;
    if(sigma.c[0]!=sigma.c[1]||sigma.c[1]!=sigma.c[2])++b.chromaticSigma;
    if(t.c[0]!=t.c[1]||t.c[1]!=t.c[2])++b.chromaticT;
    if(b.calls==1){b.firstDistance=distance;b.firstWave=wave;for(unsigned c=0;c<3;++c){
        b.firstSigma[c]=sigma.c[c];b.firstT[c]=t.c[c];b.firstBefore[c]=before.c[c];b.firstAfter[c]=after.c[c];}}
}
inline void MLHeroVolumeWrite15fp(FILE *f,const char *engine,unsigned thread) {
    const auto &s=MLHeroVolumeStateRef15fp();if(!f||!s.enabled)return;
    std::fprintf(f,"phase15fp VOLUME_CONFIG engine=%s thread=%u diagnosticsOnly=1 rendererPhysicsChanged=0 calls=%llu overflow=%llu binLimit=32 threadLocalPartition=1 sumThreadsOnce=1 noAdditionalTextureEvaluation=1 flags=EYE:1,LIGHT:2,GENERIC:4,CAMERA:8,SHADOW:16,INDIRECT:32 distances=rayParameter unitsMatchExistingScatter sigma=effectiveSigmaT spectralRoute=directIrregularAtCurrentHeroWave notFullPacketProof=1 sumsAreUnweightedSegmentStatistics=1\n",engine,thread,s.calls,s.overflow);
    for(unsigned i=0;i<s.used;++i){const auto &b=s.bins[i];
        std::fprintf(f,"phase15fp VOLUME_BIN engine=%s thread=%u bin=%u volume=%s flags=%u backTracing=%u textureType=%u spectralRoute=%u calls=%llu valid=%llu invalid=%llu chromaticSigma=%llu chromaticT=%llu finiteDistances=%llu distanceSum=%.17g distanceMin=%.17g distanceMax=%.17g sigmaSum=(%.17g,%.17g,%.17g) transmittanceSum=(%.17g,%.17g,%.17g)\n",engine,thread,i,b.name,b.flags,b.back?1u:0u,b.texture,b.spectral?1u:0u,b.calls,b.valid,b.invalid,b.chromaticSigma,b.chromaticT,b.finiteDistances,b.distanceSum,b.distanceMin,b.distanceMax,b.sigmaSum[0],b.sigmaSum[1],b.sigmaSum[2],b.tSum[0],b.tSum[1],b.tSum[2]);
        std::fprintf(f,"phase15fp VOLUME_FIRST engine=%s thread=%u bin=%u distance=%.9g currentWave=%.9g sigma=(%.9g,%.9g,%.9g) T=(%.9g,%.9g,%.9g) before=(%.9g,%.9g,%.9g) after=(%.9g,%.9g,%.9g) firstSegmentOnly=1\n",engine,thread,i,b.firstDistance,b.firstWave,b.firstSigma[0],b.firstSigma[1],b.firstSigma[2],b.firstT[0],b.firstT[1],b.firstT[2],b.firstBefore[0],b.firstBefore[1],b.firstBefore[2],b.firstAfter[0],b.firstAfter[1],b.firstAfter[2]);
    }
}
}
#endif

// HERO 15fq: coefficient reconstruction candidate; never consumed by transport.
#ifndef SLG_MLHERO_VOLUME_CANDIDATE15FQ_H
#define SLG_MLHERO_VOLUME_CANDIDATE15FQ_H
#include <cmath>
#include <cstdio>
#include "slg/engines/mlhero_volume_diagnostics15fp.h"
#include "luxrays/core/color/spds/data/rgbD65_MLHERO_380_780.h"
namespace slg {
// A candidate only: use the nonnegative R/G/B shapes of the existing 380..780
// table as normalized coefficient weights. Illumination amplitude and scale are
// deliberately NOT used for absorption. This is not measured material data.
inline bool MLHeroVolumeSigmaCandidate15fq(const float sigma[3],float wave,double *result) {
    if(!result||!std::isfinite(wave)||wave<380.f||wave>780.f)return false;
    for(unsigned c=0;c<3;++c)if(!std::isfinite(sigma[c])||sigma[c]<0.f)return false;
    if(sigma[0]==sigma[1]&&sigma[1]==sigma[2]){*result=sigma[0];return true;}
    const double pos=(static_cast<double>(wave)-380.)/10.;
    unsigned lo=static_cast<unsigned>(pos);if(lo>40)lo=40;
    const unsigned hi=lo<40?lo+1:40;const double frac=pos-lo;
    const float *shapes[3]={luxrays::mlhero_illumrgb2spect_red,luxrays::mlhero_illumrgb2spect_green,luxrays::mlhero_illumrgb2spect_blue};
    double weights[3],total=0.;
    for(unsigned c=0;c<3;++c){const double v=(1.-frac)*shapes[c][lo]+frac*shapes[c][hi];weights[c]=v>0.?v:0.;total+=weights[c];}
    if(!std::isfinite(total)||total<=0.)return false;
    *result=(weights[0]*sigma[0]+weights[1]*sigma[1]+weights[2]*sigma[2])/total;
    return std::isfinite(*result)&&*result>=0.;
}
inline bool MLHeroVolumeBeerCandidate15fq(double sigma,double distance,double *result) {
    if(!result||!std::isfinite(sigma)||sigma<0.||std::isnan(distance)||distance<0.)return false;
    // Handle a transparent wavelength on an unbounded segment before multiplying.
    if(sigma==0.||distance==0.){*result=1.;return true;}
    if(std::isinf(distance)){*result=0.;return true;}
    *result=std::exp(-sigma*distance);
    return std::isfinite(*result)&&*result>=0.&&*result<=1.;
}
struct MLHeroVolumeProbe15fq {
    unsigned bin=0, sample=0, count=0, valid=0;bool direct=false;
    float distance=0, rgbSigma[3]={},rgbT[3]={},waves[8]={};
    double sigma[8]={},t[8]={};unsigned ok[8]={};
};
struct MLHeroVolumeCandidateState15fq {
    unsigned samples[32]={},used=0;
    unsigned long long laneChecks=0,laneValid=0,laneInvalid=0;
    MLHeroVolumeProbe15fq probes[128];
};
inline MLHeroVolumeCandidateState15fq &MLHeroVolumeCandidateStateRef15fq(){static thread_local MLHeroVolumeCandidateState15fq s;return s;}
inline void MLHeroVolumeCandidateReset15fq(){MLHeroVolumeCandidateStateRef15fq()=MLHeroVolumeCandidateState15fq();}
// The existing 15fp recorder already established bounded bin identity/context.
inline MLHeroVolumeProbe15fq *MLHeroVolumeCandidateBegin15fq(const void *volume,bool direct){
    const auto &v=MLHeroVolumeStateRef15fp();if(!v.enabled)return nullptr;
    unsigned bin=0;for(;bin<v.used;++bin){const auto &b=v.bins[bin];if(b.volume==volume&&b.flags==v.flags&&b.back==v.back&&b.spectral==direct)break;}
    auto &s=MLHeroVolumeCandidateStateRef15fq();
    if(bin==v.used||s.samples[bin]==4||s.used==128)return nullptr;
    auto &p=s.probes[s.used++];p.bin=bin;p.sample=s.samples[bin]++;p.direct=direct;return &p;
}
inline void MLHeroVolumeCandidateWrite15fq(FILE *f,const char *engine,unsigned thread){
    if(!f||!MLHeroVolumeStateRef15fp().enabled)return;
    const auto &s=MLHeroVolumeCandidateStateRef15fq();
    std::fprintf(f,"phase15fq VOLUME_CANDIDATE_CONFIG engine=%s thread=%u diagnosticsOnly=1 candidateApplied=0 rendererPhysicsChanged=0 model=POSITIVE_NORMALIZED_RGB_COEFFICIENT_SHAPES_380_780 measuredSpectrum=0 RGBSigmaNotReflectance=1 coefficientUpperClamp=NONE genericReflectanceCompensation=NOT_USED directIrregular=EXACT_SPECTRAL_QUERY packet=TLS_WAVELENGTHS packetOwnershipValidated=0 samplesPerBin=4 boundedAdditionalSpectralReads=1 callsAreProbeLanesNotAllSegments=1 probes=%u laneChecks=%llu laneValid=%llu laneInvalid=%llu\n",engine,thread,s.used,s.laneChecks,s.laneValid,s.laneInvalid);
    for(unsigned i=0;i<s.used;++i){const auto &p=s.probes[i];
        std::fprintf(f,"phase15fq VOLUME_CANDIDATE_SEGMENT engine=%s thread=%u bin=%u sample=%u directIrregular=%u distance=%.9g rgbSigma=(%.9g,%.9g,%.9g) rgbT=(%.9g,%.9g,%.9g) lanes=%u validLanes=%u\n",engine,thread,p.bin,p.sample,p.direct?1u:0u,p.distance,p.rgbSigma[0],p.rgbSigma[1],p.rgbSigma[2],p.rgbT[0],p.rgbT[1],p.rgbT[2],p.count,p.valid);
        for(unsigned lane=0;lane<p.count;++lane)std::fprintf(f,"phase15fq VOLUME_CANDIDATE_LANE engine=%s thread=%u bin=%u sample=%u lane=%u lambda=%.9g valid=%u sigmaCandidate=%.17g transmittanceCandidate=%.17g candidateApplied=0\n",engine,thread,p.bin,p.sample,lane,p.waves[lane],p.ok[lane],p.sigma[lane],p.t[lane]);
    }
}
}
#endif

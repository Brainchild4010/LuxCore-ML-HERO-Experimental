// HERO 15fr: reference-transmittance reconstruction, diagnostic only.
#ifndef SLG_MLHERO_VOLUME_REFERENCE15FR_H
#define SLG_MLHERO_VOLUME_REFERENCE15FR_H
#include "slg/engines/mlhero_volume_candidate15fq.h"
namespace slg {
// Seven existing 380..780 spectral shapes, normalized by their white reference.
// Clipping makes this a bounded transmittance candidate. No illuminant scale.
inline bool MLHeroVolumeReferenceT15fr(const double rgb[3],float wave,double *result){
    if(!result||!std::isfinite(wave)||wave<380.f||wave>780.f)return false;
    for(unsigned c=0;c<3;++c)if(!std::isfinite(rgb[c])||rgb[c]<0.||rgb[c]>1.)return false;
    if(rgb[0]==rgb[1]&&rgb[1]==rgb[2]){*result=rgb[0];return true;}
    const double pos=(wave-380.)/10.;unsigned lo=static_cast<unsigned>(pos);if(lo>40)lo=40;
    const unsigned hi=lo<40?lo+1:40;const double frac=pos-lo;
    const auto sample=[=](const float *a){return (1.-frac)*a[lo]+frac*a[hi];};
    const double white=sample(luxrays::mlhero_illumrgb2spect_white);
    if(!std::isfinite(white)||white<=0.)return false;
    const double r=rgb[0],g=rgb[1],b=rgb[2];double v=0.;
    if(r<=g&&r<=b){v=r*white;
        if(g<=b)v+=(g-r)*sample(luxrays::mlhero_illumrgb2spect_cyan)+(b-g)*sample(luxrays::mlhero_illumrgb2spect_blue);
        else v+=(b-r)*sample(luxrays::mlhero_illumrgb2spect_cyan)+(g-b)*sample(luxrays::mlhero_illumrgb2spect_green);
    }else if(g<=r&&g<=b){v=g*white;
        if(r<=b)v+=(r-g)*sample(luxrays::mlhero_illumrgb2spect_magenta)+(b-r)*sample(luxrays::mlhero_illumrgb2spect_blue);
        else v+=(b-g)*sample(luxrays::mlhero_illumrgb2spect_magenta)+(r-b)*sample(luxrays::mlhero_illumrgb2spect_red);
    }else{v=b*white;
        if(r<=g)v+=(r-b)*sample(luxrays::mlhero_illumrgb2spect_yellow)+(g-r)*sample(luxrays::mlhero_illumrgb2spect_green);
        else v+=(g-b)*sample(luxrays::mlhero_illumrgb2spect_yellow)+(r-g)*sample(luxrays::mlhero_illumrgb2spect_red);
    }
    if(!std::isfinite(v))return false;
    v/=white;*result=v<0.?0.:(v>1.?1.:v);return true;
}
inline bool MLHeroVolumeReferenceSigma15fr(double transmission,double referenceDistance,double *sigma){
    if(!sigma||!std::isfinite(referenceDistance)||referenceDistance<=0.||!std::isfinite(transmission)||transmission<0.||transmission>1.)return false;
    const double safe=transmission<1.e-30?1.e-30:transmission;
    *sigma=-std::log(safe)/referenceDistance;
    return std::isfinite(*sigma)&&*sigma>=0.;
}
struct MLHeroVolumeReferenceProbe15fr{
    unsigned bin=0,sample=0,count=0,source=0;
    double referenceDistance=0,rgb[3]={};
    double wave[8]={},referenceT[8]={},sigma[8]={},t[8]={};unsigned valid[8]={};
};
struct MLHeroVolumeReferenceState15fr{
    unsigned used=0;unsigned long long valid=0,invalid=0,unsupported=0;
    MLHeroVolumeReferenceProbe15fr probes[128];
};
inline MLHeroVolumeReferenceState15fr &MLHeroVolumeReferenceStateRef15fr(){static thread_local MLHeroVolumeReferenceState15fr s;return s;}
inline void MLHeroVolumeReferenceReset15fr(){MLHeroVolumeReferenceStateRef15fr()=MLHeroVolumeReferenceState15fr();}
inline void MLHeroVolumeReferenceWrite15fr(FILE *f,const char *engine,unsigned thread){
    if(!f||!MLHeroVolumeStateRef15fp().enabled)return;
    const auto &s=MLHeroVolumeReferenceStateRef15fr();
    std::fprintf(f,"phase15fr VOLUME_REFERENCE_CONFIG engine=%s thread=%u diagnosticsOnly=1 candidateApplied=0 rendererPhysicsChanged=0 model=SEVEN_SHAPE_TRANSMITTANCE_OVER_WHITE_380_780 sourceCodes=1_ColorDepthTexture,2_AssumedReferenceFromRGBSigma assumedReferenceDistance=0.1 floorTransmission=1e-30 directIrregular=KEEP_15fq_EXACT_QUERY finalColorMatchValidated=0 packetOwnershipValidated=0 probes=%u laneValid=%llu laneInvalid=%llu unsupported=%llu\n",engine,thread,s.used,s.valid,s.invalid,s.unsupported);
    for(unsigned i=0;i<s.used;++i){const auto &p=s.probes[i];
        std::fprintf(f,"phase15fr VOLUME_REFERENCE_SEGMENT engine=%s thread=%u bin=%u sample=%u source=%u referenceDistance=%.17g referenceRGB=(%.17g,%.17g,%.17g) lanes=%u\n",engine,thread,p.bin,p.sample,p.source,p.referenceDistance,p.rgb[0],p.rgb[1],p.rgb[2],p.count);
        for(unsigned lane=0;lane<p.count;++lane)std::fprintf(f,"phase15fr VOLUME_REFERENCE_LANE engine=%s thread=%u bin=%u sample=%u lane=%u lambda=%.9g valid=%u referenceT=%.17g sigmaCandidate=%.17g segmentT=%.17g candidateApplied=0\n",engine,thread,p.bin,p.sample,lane,p.wave[lane],p.valid[lane],p.referenceT[lane],p.sigma[lane],p.t[lane]);
    }
}
}
#endif

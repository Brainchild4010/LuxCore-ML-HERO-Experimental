// HERO 15fh: read-only, pre-clamp estimator decomposition. No renderer state writes.
#ifndef SLG_MLHERO_DIAGNOSTICS15FH_H
#define SLG_MLHERO_DIAGNOSTICS15FH_H
#include "slg/engines/mlhero_volume_diagnostics15fp.h"
#include "slg/engines/mlhero_volume_candidate15fq.h"
#include "slg/engines/mlhero_volume_reference15fr.h"
#include "slg/engines/mlhero_rr_candidate15fs.h"
#include "slg/engines/mlhero_material_rr15fu.h"
#include "slg/engines/mlhero_material_rr_experiment15fv.h"
#include "slg/engines/mlhero_rr_experiment15ft.h"
#include <cmath>
#include <cstdio>
namespace slg { namespace {
static constexpr unsigned MLHeroRegions15fh=3, MLHeroBuckets15fh=6;
static constexpr unsigned MLHeroDepthBins15fj=67; // 0..64, overflow65, unknown66
struct MLHeroRegion15fh {
    unsigned long long eyes=0, finiteEyes=0, invalidEyes=0, screens=0, invalidScreens=0;
    unsigned long long depthEvents15fi[4]={};
    double depthRGB15fi[4][3]={};
    unsigned maxScatterDepth15fi=0;
    unsigned long long exactDepthEvents15fj[MLHeroDepthBins15fj]={};
    double exactDepthRGB15fj[MLHeroDepthBins15fj][3]={};
    unsigned long long events[MLHeroBuckets15fh][2]={};
    double sums[MLHeroBuckets15fh][3]={}, total[3]={}, residual[3]={}, screen[3]={};
};
struct MLHeroRRBin15fk {
    unsigned long long calls=0, active=0, counterDifferent=0, invalid=0, qDifferent=0;
    double qEvalSum=0, qCandidateSum=0, pdfSum=0, reversePdfSum=0;
};
struct MLHeroRRDetail15fk {
    unsigned depth=0, material=0, event=0;
    bool active=false, candidateActive=false;
    float pdf=0, reversePdf=0, qEval=0, qCandidate=0;
};
struct MLHeroMISBin15fl {
    unsigned long long calls=0, eligible=0, invalid=0, pairExceedsOne=0, referenceDifferent=0;
    unsigned long long skipped[8]={};
    double dlsSum=0, hitSum=0, rawOtherSum=0, rawDlsSum=0, pairMax=0, referenceDeltaMax=0;
    bool detail=false;unsigned material=0,event=0,light=0;
    double detailDls=0,detailHit=0,detailOther=0,detailRawDls=0,detailRawPdf=0,detailUsedPdf=0;
};
// 15fn: raw-PDF counterfactuals only; never returned to live transport.
static double MLHeroSquare15fn(double x) { return x*x; }
static double MLHeroConnectionWeight15fn(double eyeArea, double lightArea,
        double eyeReverse, double lightReverse, double eyeVCM, double eyeVC,
        double lightVCM, double lightVC, double vm) {
    return 1. / (1. + MLHeroSquare15fn(eyeArea)*(vm+lightVCM+lightVC*MLHeroSquare15fn(lightReverse))
        + MLHeroSquare15fn(lightArea)*(vm+eyeVCM+eyeVC*MLHeroSquare15fn(eyeReverse)));
}
static double MLHeroCameraWeight15fn(double cameraArea, double reverse,
        double vcm, double vc, double vm) {
    return 1. / (1. + MLHeroSquare15fn(cameraArea)*(vm+vcm+vc*MLHeroSquare15fn(reverse)));
}
struct MLHeroCandidateBin15fn {
    unsigned long long calls=0, skipped=0, valid=0, invalid=0, changed=0;
    double actualSum=0, candidateSum=0, maxDelta=0;
    unsigned otherDepth=0, material=0;
    double detailActual=0, detailCandidate=0, forward=0, reverse=0;
};
struct MLHeroState15fh {
    MLHeroCandidateBin15fn candidate15fn[3][67];
    MLHeroMISBin15fl terminalMIS[2][67];
    MLHeroRRBin15fk rr[8][67];
    MLHeroRRDetail15fk rrDetails[8][8];
    unsigned rrDetailCount[8]={};
    bool enabled=false, active=false, bad=false;
    unsigned width=0, height=0, mask=0;
    unsigned long long cycles=0, generatedEyes=0;
    double sample[MLHeroBuckets15fh][3]={};
    unsigned long long depthEvents15fi[4]={};
    double depthRGB15fi[4][3]={};
    unsigned maxScatterDepth15fi=0;
    unsigned long long exactDepthEvents15fj[MLHeroDepthBins15fj]={};
    double exactDepthRGB15fj[MLHeroDepthBins15fj][3]={};
    unsigned long long events[MLHeroBuckets15fh][2]={};
    unsigned touchedDepth15fj[MLHeroDepthBins15fj]={}, touchedCount15fj=0;
    MLHeroRegion15fh regions[MLHeroRegions15fh];
};
static thread_local MLHeroState15fh mlHeroState15fh;
static void MLHeroReset15fh(bool enabled) {
    MLHeroRRLiveReset15ft(enabled);
    MLHeroRRReset15fs(enabled);
    MLHeroMaterialRRReset15fu(enabled);
    MLHeroMaterialLiveReset15fv(enabled);
    MLHeroVolumeReset15fp(enabled);
    MLHeroVolumeCandidateReset15fq();
    MLHeroVolumeReferenceReset15fr();
    mlHeroState15fh=MLHeroState15fh(); mlHeroState15fh.enabled=enabled;
}
static void MLHeroCycle15fh() {
    if(!mlHeroState15fh.enabled)return;
    mlHeroState15fh.active=false; ++mlHeroState15fh.cycles;
}
static unsigned MLHeroMask15fh(float x,float y,unsigned w,unsigned h) {
    if(!w||!h||!std::isfinite(x)||!std::isfinite(y)||x<0||y<0||x>=w||y>=h)return 0;
    const double nx=(std::floor(x)+.5)/w, ny=1.-(std::floor(y)+.5)/h;
    unsigned mask=1u;
    if(nx>=.21&&nx<.74&&ny>=.24&&ny<.60)mask|=2u;
    if(nx>=.18&&nx<.27&&ny>=.65&&ny<.82)mask|=4u;
    return mask;
}
static void MLHeroBegin15fh(float x,float y,unsigned w,unsigned h) {
    auto&s=mlHeroState15fh; if(!s.enabled)return;
    for(unsigned d=0;d<4;++d){s.depthEvents15fi[d]=0;for(unsigned c=0;c<3;++c)s.depthRGB15fi[d][c]=0;}
    for(unsigned t=0;t<s.touchedCount15fj;++t){
        const unsigned d=s.touchedDepth15fj[t];s.exactDepthEvents15fj[d]=0;
        for(unsigned c=0;c<3;++c)s.exactDepthRGB15fj[d][c]=0;
    }
    s.touchedCount15fj=0;
    s.maxScatterDepth15fi=0;
    s.active=true;s.bad=false;s.mask=MLHeroMask15fh(x,y,w,h);s.width=w;s.height=h;++s.generatedEyes;
    for(unsigned b=0;b<MLHeroBuckets15fh;++b){
        for(unsigned c=0;c<3;++c)s.sample[b][c]=0;
        s.events[b][0]=s.events[b][1]=0;
    }
}
// route 0: already reconstructed spectral endpoint; 1: classic/RGB endpoint.
template<class Color> static void MLHeroRecord15fh(unsigned bucket,const Color&v,bool spectral,unsigned scatterDepth15fi=~0u) {
    auto&s=mlHeroState15fh;if(!s.enabled||!s.active||bucket>=MLHeroBuckets15fh)return;
    ++s.events[bucket][spectral?0:1];
    const unsigned band15fi=scatterDepth15fi==~0u?0u:(scatterDepth15fi<10u?1u:(scatterDepth15fi==10u?2u:3u));
    ++s.depthEvents15fi[band15fi];
    const unsigned exact15fj=scatterDepth15fi==~0u?66u:(scatterDepth15fi<=64u?scatterDepth15fi:65u);
    if(!s.exactDepthEvents15fj[exact15fj])s.touchedDepth15fj[s.touchedCount15fj++]=exact15fj;
    ++s.exactDepthEvents15fj[exact15fj];
    for(unsigned c=0;c<3;++c)if(std::isfinite(v.c[c]))s.exactDepthRGB15fj[exact15fj][c]+=v.c[c];
    if(scatterDepth15fi!=~0u && scatterDepth15fi>s.maxScatterDepth15fi)s.maxScatterDepth15fi=scatterDepth15fi;
    for(unsigned c=0;c<3;++c)if(std::isfinite(v.c[c]))s.depthRGB15fi[band15fi][c]+=v.c[c];
    for(unsigned c=0;c<3;++c){if(!std::isfinite(v.c[c]))s.bad=true;else s.sample[bucket][c]+=v.c[c];}
}
// Camera-light depth=0, after first surface depth=1, later depth>=2.
static unsigned MLHeroHitBucket15fh(unsigned depth) {return depth==0?2u:(depth==1?3u:4u);}
template<class Result> static void MLHeroFinish15fh(const Result&result) {
    auto&s=mlHeroState15fh;if(!s.enabled||!s.active)return;
    double total[3]={};bool ok=!s.bad;
    for(unsigned g=0;g<result.radiance.Size();++g)
        for(unsigned c=0;c<3;++c){const double v=result.radiance[g].c[c];ok=ok&&std::isfinite(v);total[c]+=v;}
    for(unsigned c=0;c<3;++c)ok=ok&&std::isfinite(total[c]);
    for(unsigned region=0;region<MLHeroRegions15fh;++region)if(s.mask&(1u<<region)) {
        auto&a=s.regions[region];++a.eyes;
        for(unsigned b=0;b<MLHeroBuckets15fh;++b)for(unsigned route=0;route<2;++route)a.events[b][route]+=s.events[b][route];
        if(!ok){++a.invalidEyes;continue;}++a.finiteEyes;
        for(unsigned t=0;t<s.touchedCount15fj;++t){
            const unsigned d=s.touchedDepth15fj[t];a.exactDepthEvents15fj[d]+=s.exactDepthEvents15fj[d];
            for(unsigned c=0;c<3;++c)a.exactDepthRGB15fj[d][c]+=s.exactDepthRGB15fj[d][c];
        }
        if(s.maxScatterDepth15fi>a.maxScatterDepth15fi)a.maxScatterDepth15fi=s.maxScatterDepth15fi;
        for(unsigned d=0;d<4;++d){a.depthEvents15fi[d]+=s.depthEvents15fi[d];for(unsigned c=0;c<3;++c)a.depthRGB15fi[d][c]+=s.depthRGB15fi[d][c];}
        for(unsigned c=0;c<3;++c){double tracked=0;for(unsigned b=0;b<MLHeroBuckets15fh;++b){a.sums[b][c]+=s.sample[b][c];tracked+=s.sample[b][c];}a.total[c]+=total[c];a.residual[c]+=total[c]-tracked;}
    }
    s.active=false;
}
template<class Result> static void MLHeroScreen15fh(const Result&result,unsigned w,unsigned h) {
    auto&s=mlHeroState15fh;if(!s.enabled)return;
    double total[3]={};bool ok=true;for(unsigned g=0;g<result.radiance.Size();++g)for(unsigned c=0;c<3;++c){double v=result.radiance[g].c[c];ok=ok&&std::isfinite(v);total[c]+=v;}
    for(unsigned c=0;c<3;++c)ok=ok&&std::isfinite(total[c]);
    const unsigned mask=MLHeroMask15fh(result.filmX,result.filmY,w,h);
    for(unsigned region=0;region<MLHeroRegions15fh;++region)if(mask&(1u<<region)) {auto&a=s.regions[region];++a.screens;if(!ok){++a.invalidScreens;continue;}for(unsigned c=0;c<3;++c)a.screen[c]+=total[c];}
}
// Read-only audit. Eval/pdf is a candidate for this direction, not a reverse
// proposal proof or a complete candidate MIS weight. Bounce input already is Sample.
template<class Color> static void MLHeroRRAudit15fk(unsigned stage,unsigned depth,
        unsigned material,unsigned event,const Color&value,float pdf,float reversePdf,
        float cap,float actualQ,bool active,bool candidateActive,bool sampled) {
    auto&s=mlHeroState15fh;if(!s.enabled||stage>=8)return;
    auto&a=s.rr[stage][depth<=64?depth:65];++a.calls;
    a.active+=active; a.counterDifferent+=(active!=candidateActive);
    bool ok=std::isfinite(actualQ)&&std::isfinite(cap)&&std::isfinite(pdf)&&pdf>0;
    float c[3]={};
    for(unsigned i=0;i<3;++i){c[i]=sampled?value.c[i]:value.c[i]/pdf;ok=ok&&std::isfinite(c[i]);}
    const float filter=(c[0]+c[1]+c[2])/3.f;
    ok=ok&&std::isfinite(filter);
    const float q=ok?(filter<cap?cap:(filter>1.f?1.f:filter)):0.f;
    if(!ok){++a.invalid;return;}
    a.qEvalSum+=actualQ;a.qCandidateSum+=q;a.pdfSum+=pdf;
    if(std::isfinite(reversePdf))a.reversePdfSum+=reversePdf;
    const bool different=std::fabs(actualQ-q)>1e-6f;
    a.qDifferent+=different;
    // Bounded examples focus on mismatches, plus one representative entry.
    unsigned&n=s.rrDetailCount[stage];
    if(n<8&&(different||active!=candidateActive||n==0)) {
        auto&d=s.rrDetails[stage][n++];d.depth=depth;d.material=material;d.event=event;
        d.active=active;d.candidateActive=candidateActive;d.pdf=pdf;d.reversePdf=reversePdf;
        d.qEval=actualQ;d.qCandidate=q;
    }
}
// Independent native DirectHit algebra after one non-delta Bounce + landing.
// Distances/area conversion cancel; rawOther is an inferred raw-family remainder,
// NOT independently evaluated weights of all BIDIR techniques.
static double MLHeroHitWeight15fl(double bsdfPdf,double directPdf,double emissionPdf,
        double cosSurface,double cosLight,double dVCM,double dVC,double reversePdf,
        double lightPickPdf) {
    const double p=bsdfPdf, dl=directPdf*lightPickPdf;
    const double hitDlsRatio=(dl/p)*(dl/p);
    const double emitterRatio=emissionPdf*lightPickPdf*cosSurface/(p*cosLight);
    const double earlier=emitterRatio*emitterRatio*(dVCM+dVC*reversePdf*reversePdf);
    return 1./(1.+hitDlsRatio+earlier);
}
static void MLHeroTerminalMIS15fl(unsigned stage,unsigned depth,unsigned material,
        unsigned event,unsigned light,unsigned skip,double actualDls,double rawPdf,
        double usedPdf,double directPdf,double emissionPdf,double cosSurface,
        double cosLight,double dVCM,double dVC,double reversePdf,double lightPickPdf) {
    auto&s=mlHeroState15fh;if(!s.enabled||stage>=2)return;
    auto&a=s.terminalMIS[stage][depth<=64?depth:65];++a.calls;
    if(skip){++a.skipped[skip<8?skip:7];return;}
    const double inputs[]={actualDls,rawPdf,usedPdf,directPdf,emissionPdf,cosSurface,cosLight,dVCM,dVC,reversePdf,lightPickPdf};
    for(const double value:inputs)if(!std::isfinite(value)){++a.invalid;return;}
    if(!(rawPdf>0&&directPdf>0&&lightPickPdf>0&&cosLight>0) ||
            !(usedPdf>=0&&emissionPdf>=0&&cosSurface>=0&&dVCM>=0&&dVC>=0&&reversePdf>=0)) {
        ++a.invalid;return;
    }
    const double hit=MLHeroHitWeight15fl(rawPdf,directPdf,emissionPdf,cosSurface,
            cosLight,dVCM,dVC,reversePdf,lightPickPdf);
    // Convert the independently reconstructed native hit denominator to the
    // DLS normalization. This avoids just swapping the current DLS terms.
    const double ratio=(directPdf*lightPickPdf/rawPdf)*(directPdf*lightPickPdf/rawPdf);
    const double rawDls=ratio*hit;
    double other=1.-hit-rawDls;
    if(other<0&&other>-1e-12)other=0;
    const double pair=actualDls+hit,delta=actualDls-rawDls;
    if(!std::isfinite(hit)||!std::isfinite(rawDls)||!std::isfinite(other)||other<0||
            !std::isfinite(actualDls)||actualDls<0||actualDls>1) {++a.invalid;return;}
    ++a.eligible;a.dlsSum+=actualDls;a.hitSum+=hit;a.rawOtherSum+=other;a.rawDlsSum+=rawDls;
    a.pairExceedsOne+=(pair>1.+1e-5);a.referenceDifferent+=(std::fabs(delta)>1e-5);
    if(pair>a.pairMax)a.pairMax=pair;
    if(std::fabs(delta)>a.referenceDeltaMax)a.referenceDeltaMax=std::fabs(delta);
    if(!a.detail||std::fabs(delta)>std::fabs(a.detailDls-a.detailRawDls)){
        a.detail=true;a.material=material;a.event=event;a.light=light;
        a.detailDls=actualDls;a.detailHit=hit;a.detailOther=other;a.detailRawDls=rawDls;
        a.detailRawPdf=rawPdf;a.detailUsedPdf=usedPdf;
    }
}
static void MLHeroCandidate15fn(unsigned stage, unsigned depth, unsigned otherDepth,
        unsigned material, double actual, double candidate, double forward, double reverse,
        bool eligible) {
    auto&s=mlHeroState15fh;if(!s.enabled || stage>=3)return;
    auto&a=s.candidate15fn[stage][depth<=64?depth:65];++a.calls;
    if(!eligible){++a.skipped;return;}
    if(!std::isfinite(actual)||!std::isfinite(candidate)||actual<0||actual>1||candidate<0||candidate>1
            ||!std::isfinite(forward)||!std::isfinite(reverse)||forward<0||reverse<0){++a.invalid;return;}
    ++a.valid;a.actualSum+=actual;a.candidateSum+=candidate;
    const double delta=std::fabs(candidate-actual);if(delta>1e-5)++a.changed;
    if(delta>=a.maxDelta){a.maxDelta=delta;a.otherDepth=otherDepth;a.material=material;
        a.detailActual=actual;a.detailCandidate=candidate;a.forward=forward;a.reverse=reverse;}
}
static void MLHeroWrite15fh(FILE*f,const char*engine,unsigned thread) {
    MLHeroRRLiveWrite15ft(f,engine,thread);
    MLHeroRRWrite15fs(f,engine,thread);
    MLHeroMaterialRRWrite15fu(f,engine,thread);
    MLHeroMaterialLiveWrite15fv(f,engine,thread);
    MLHeroVolumeWrite15fp(f,engine,thread);
    MLHeroVolumeCandidateWrite15fq(f,engine,thread);
    MLHeroVolumeReferenceWrite15fr(f,engine,thread);
    const auto&s=mlHeroState15fh;if(!f||!s.enabled)return;
    fprintf(f,"phase15fh ESTIMATOR_CONFIG engine=%s thread=%u threadLocalPartition=1 sumThreadsOnce=1 cycles=%llu generatedEyes=%llu width=%u height=%u preClamp=1 preImagePipeline=1 filterWeightApplied=0 samplerWeightApplied=0 eyeMeanDenominator=finiteEyes rawScreenSumsNotEyeEnergy=1 nonfiniteExcludedFromDiagnosticsOnly=1 regions=FULL,BALL,SOCKEL_LEFT ROI_TOP_ORIGIN=1 BALL=0.21,0.24,0.74,0.60 SOCKEL_LEFT=0.18,0.65,0.27,0.82\n",engine,thread,s.cycles,s.generatedEyes,s.width,s.height);
    fprintf(f,"phase15fk RR_CONFIG engine=%s thread=%u diagnosticsOnly=1 rendererPhysicsChanged=0 candidateApplied=0 capIsMinimumSurvival=1 candidate=directionalEvalDivForwardPDF reverseCandidateNotProven=1 sumsDenominator=validCalls noEnergyEstimate=1 detailBudgetPerStage=8 threadLocalPartition=1 stageCallsIncludeRejectedSamples=1\n",engine,thread);
    fprintf(f,"phase15fl MIS_CONFIG engine=%s thread=%u rendererPhysicsChanged=0 candidateApplied=0 pairedHitReconstructed=1 fullBidirClosure=0 rawOtherIsReferenceOnly=1 eligibility=MatteOrMatteTranslucentSurfacePlainFlatTrianglePositionIndependentLightPick noAdditionalRays=1 noAdditionalSamplerDimensions=1 tolerance=1e-5 skippedReasons=1_MISOrVisibility,2_Depth,3_LightOrMaterial,4_Strategy,5_Volume,6_LightNormals,7_Other\n",engine,thread);
    fprintf(f,"phase15fn BIDIR_RAW_CONFIG engine=%s thread=%u candidateApplied=0 rendererPhysicsChanged=0 rawHistory=1 localCounterfactualOnly=1 fullRuntimePathClosure=0 stageDepth=EyeExceptCameraLight eligibility=NonVolumeNonTerminatedVCMOff noAdditionalRays=1 noAdditionalSamplerDimensions=1\n",engine,thread);
    const char*candidateNames15fn[]={"CONNECT_VERTICES","CONNECT_CAMERA","DLS"};
    for(unsigned stage=0;stage<3;++stage)for(unsigned depth=0;depth<67;++depth){
        const auto&a=s.candidate15fn[stage][depth];if(!a.calls)continue;
        fprintf(f,"phase15fn BIDIR_RAW_BIN engine=%s thread=%u stage=%s depthBin=%u calls=%llu skipped=%llu valid=%llu invalid=%llu changed=%llu actualWeightSum=%.17g candidateWeightSum=%.17g maxDelta=%.17g\n",engine,thread,candidateNames15fn[stage],depth,a.calls,a.skipped,a.valid,a.invalid,a.changed,a.actualSum,a.candidateSum,a.maxDelta);
        if(a.valid)fprintf(f,"phase15fn BIDIR_RAW_DETAIL engine=%s thread=%u stage=%s depthBin=%u otherDepth=%u materialType=%u actual=%.17g candidate=%.17g rawForward=%.17g rawReverse=%.17g\n",engine,thread,candidateNames15fn[stage],depth,a.otherDepth,a.material,a.detailActual,a.detailCandidate,a.forward,a.reverse);
    }
    const char*misStages[]={"PATH_TERMINAL_PAIR","BIDIR_TERMINAL_PAIR"};
    for(unsigned stage=0;stage<2;++stage)for(unsigned depth=0;depth<67;++depth){const auto&a=s.terminalMIS[stage][depth];if(!a.calls)continue;
        fprintf(f,"phase15fl MIS_TERMINAL engine=%s thread=%u stage=%s depthBin=%u calls=%llu eligible=%llu invalid=%llu pairExceedsOne=%llu referenceDifferent=%llu dlsWeightSum=%.17g reconstructedHitWeightSum=%.17g rawReferenceOtherWeightSum=%.17g rawReferenceDlsWeightSum=%.17g pairMax=%.17g referenceDeltaMax=%.17g skipped=(%llu,%llu,%llu,%llu,%llu,%llu,%llu)\n",engine,thread,misStages[stage],depth,a.calls,a.eligible,a.invalid,a.pairExceedsOne,a.referenceDifferent,a.dlsSum,a.hitSum,a.rawOtherSum,a.rawDlsSum,a.pairMax,a.referenceDeltaMax,a.skipped[1],a.skipped[2],a.skipped[3],a.skipped[4],a.skipped[5],a.skipped[6],a.skipped[7]);
        if(a.detail)fprintf(f,"phase15fl MIS_TERMINAL_DETAIL engine=%s thread=%u stage=%s depthBin=%u materialType=%u event=%u lightID=%u actualDls=%.17g reconstructedHit=%.17g rawReferenceOther=%.17g rawReferenceDls=%.17g pairSum=%.17g rawForwardPdf=%.17g usedForwardPdf=%.17g fullBidirClosure=0\n",engine,thread,misStages[stage],depth,a.material,a.event,a.light,a.detailDls,a.detailHit,a.detailOther,a.detailRawDls,a.detailDls+a.detailHit,a.detailRawPdf,a.detailUsedPdf);
    }
    const char*rrNames[]={"BD_BOUNCE_EYE","BD_BOUNCE_LIGHT","BD_CONNECT_EYE","BD_CONNECT_LIGHT","BD_CONNECT_CAMERA","BD_DLS","PATH_BOUNCE","PATH_DLS"};
    for(unsigned stage=0;stage<8;++stage){
        for(unsigned depth=0;depth<67;++depth){const auto&a=s.rr[stage][depth];if(!a.calls)continue;
            fprintf(f,"phase15fk RR_BIN engine=%s thread=%u stage=%s depthBin=%u kind=%s calls=%llu active=%llu counterDifferent=%llu invalidCandidate=%llu qDifferent=%llu qActualInputSum=%.17g qCandidateSum=%.17g rawForwardPDFSum=%.17g rawReversePDFSum=%.17g\n",engine,thread,rrNames[stage],depth,depth<=64?"EXACT":(depth==65?"OVER64":"UNKNOWN"),a.calls,a.active,a.counterDifferent,a.invalid,a.qDifferent,a.qEvalSum,a.qCandidateSum,a.pdfSum,a.reversePdfSum);
        }
        for(unsigned i=0;i<s.rrDetailCount[stage];++i){const auto&d=s.rrDetails[stage][i];
            fprintf(f,"phase15fk RR_DETAIL engine=%s thread=%u stage=%s depth=%u materialType=%u event=%u actualActive=%u candidateActive=%u rawForwardPDF=%.9g rawReversePDF=%.9g qActualInput=%.9g qCandidate=%.9g\n",engine,thread,rrNames[stage],d.depth,d.material,d.event,d.active?1u:0u,d.candidateActive?1u:0u,d.pdf,d.reversePdf,d.qEval,d.qCandidate);
        }
    }
    const char*names[]={"DLS_FIRST","DLS_LATER","CAMERA_LIGHT","HIT_AFTER_FIRST","HIT_LATER","CONNECT_VERTICES"};
    for(unsigned region=0;region<MLHeroRegions15fh;++region){const auto&a=s.regions[region];
        fprintf(f,"phase15fh ESTIMATOR_REGION engine=%s thread=%u region=%u eyeSamples=%llu finiteEyes=%llu invalidEyes=%llu screenSamples=%llu invalidScreens=%llu\n",engine,thread,region,a.eyes,a.finiteEyes,a.invalidEyes,a.screens,a.invalidScreens);
        for(unsigned b=0;b<MLHeroBuckets15fh;++b)fprintf(f,"phase15fh ESTIMATOR_BUCKET engine=%s thread=%u region=%u bucket=%s spectralEvents=%llu classicEvents=%llu sumRGB=(%.17g,%.17g,%.17g)\n",engine,thread,region,names[b],a.events[b][0],a.events[b][1],a.sums[b][0],a.sums[b][1],a.sums[b][2]);
        const char*depthNames15fi[]={"UNKNOWN","LT10","EQ10","GT10"};
        for(unsigned d=0;d<4;++d)fprintf(f,"phase15fi DEPTH_BAND engine=%s thread=%u region=%u band=%s finiteSampleEvents=%llu maxScatterDepth=%u sumRGB=(%.17g,%.17g,%.17g) denominator=finiteEyes threadLocalPartition=1 rawScreenExcluded=1 thresholdFixed10=1\n",engine,thread,region,depthNames15fi[d],a.depthEvents15fi[d],a.maxScatterDepth15fi,a.depthRGB15fi[d][0],a.depthRGB15fi[d][1],a.depthRGB15fi[d][2]);
        for(unsigned d=0;d<MLHeroDepthBins15fj;++d)fprintf(f,"phase15fj DEPTH_EXACT engine=%s thread=%u region=%u scatterDepth=%u kind=%s finiteSampleEvents=%llu sumRGB=(%.17g,%.17g,%.17g) denominator=finiteEyes threadLocalPartition=1 rawScreenExcluded=1\n",engine,thread,region,d,d<=64u?"EXACT":(d==65u?"OVER64":"UNKNOWN"),a.exactDepthEvents15fj[d],a.exactDepthRGB15fj[d][0],a.exactDepthRGB15fj[d][1],a.exactDepthRGB15fj[d][2]);
        fprintf(f,"phase15fh ESTIMATOR_TOTAL engine=%s thread=%u region=%u totalEyeRGB=(%.17g,%.17g,%.17g) residualRGB=(%.17g,%.17g,%.17g) rawScreenRGB=(%.17g,%.17g,%.17g)\n",engine,thread,region,a.total[0],a.total[1],a.total[2],a.residual[0],a.residual[1],a.residual[2],a.screen[0],a.screen[1],a.screen[2]);
    }
}
} }
#endif

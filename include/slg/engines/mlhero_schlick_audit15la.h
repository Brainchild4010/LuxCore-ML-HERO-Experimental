#ifndef SLG_MLHERO_SCHLICK_AUDIT15LA_H
#define SLG_MLHERO_SCHLICK_AUDIT15LA_H
#include <atomic>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include "slg/engines/mlhero_rough_optics15kk.h"
namespace slg { namespace mlhero15la {
using V=mlhero15kk::V;
constexpr double Pi=3.14159265358979323846264338327950288;
constexpr unsigned Budget=256,Bins=16;
struct Density {bool valid=false;double generated=0.,claimedIndependent=0.,cdfU=0.,cdfV=0.;};
// Independent inverse-map derivatives, not a call to SampleHalf or renderer D/Pdf.
inline double PhiPdf(double phi,double anis){
    const double p=1.-std::fabs(anis);if(!(p>0.))return 0.;
    if(anis>0.)phi-=Pi*.5;
    phi=std::fmod(phi+4.*Pi,2.*Pi);
    const unsigned quadrant=unsigned(phi/(Pi*.5));
    const double fraction=phi/(Pi*.5)-quadrant;
    const double q=(quadrant&1u)?1.-fraction:fraction;
    const double d=p*p+q*q*(1.-p*p);
    return p*p/(2.*Pi*d*std::sqrt(d));
}
inline Density Reference(double rough,double anis,V input){
    Density d;const double n2=input.Norm2();
    if(!(rough>0.)||!std::isfinite(rough)||!std::isfinite(anis)||!(std::fabs(anis)<1.)||!(n2>0.)||!std::isfinite(n2))return d;
    const V h=input*(1./std::sqrt(n2));
    const double sine2=h.x*h.x+h.y*h.y,c=h.z;
    if(!(c>0.)||!(sine2>0.))return d; // pole/equator are measure-zero coordinate endpoints
    const double den=sine2+rough*c*c,phi=std::atan2(h.y,h.x);
    const double radialDerivative=2.*rough*c/(den*den);
    d.generated=radialDerivative*PhiPdf(phi,anis);
    const double p=1.-std::fabs(anis);
    const double w=(anis>0.?h.x:h.y)/std::sqrt(sine2);
    const double A=std::sqrt(p/(p*p+w*w*(1.-p*p)));
    d.claimedIndependent=rough/(Pi*den*den)*A;
    d.cdfU=rough*c*c/den;
    double angle=phi-(anis>0.?Pi*.5:0.);angle=std::fmod(angle+4.*Pi,2.*Pi);
    const unsigned quadrant=unsigned(angle/(Pi*.5));
    const double fraction=angle/(Pi*.5)-quadrant,q=(quadrant&1u)?1.-fraction:fraction;
    const double t=q/std::sqrt(p*p+q*q*(1.-p*p));
    d.cdfV=(quadrant+((quadrant&1u)?1.-t:t))*.25;
    d.valid=std::isfinite(d.generated)&&d.generated>0.&&std::isfinite(d.claimedIndependent)&&d.claimedIndependent>0.;
    return d;
}
inline double Relative(double a,double b){return std::fabs(a-b)/std::fmax(1e-300,std::fmax(std::fabs(a),std::fabs(b)));}
struct Record {
    unsigned side=0,material=0,status=0;unsigned long long detail=0;
    double nu=0.,nv=0.,rough=0.,anis=0.,u0=0.,u1=0.,claimedHalf=0.,reportedPdf=0.;
    V fixed,h,pair;Density draw,emitted;
};
using Reporter=void(*)(const Record&);
inline thread_local Reporter reporter=nullptr;
inline thread_local unsigned used[2]{};
inline std::atomic<unsigned long long> attempts[2]{},draws[2]{},status[2][4]{},invalidReference[2]{},densityMismatch[2]{},pdfMismatch[2]{},cdfMismatch[2]{},claimedReplayMismatch[2]{},details[2]{},writeFailures[2]{};
inline std::atomic<unsigned long long> histogram[2][2][Bins]{};
inline void Reset(){for(unsigned s=0;s<2;++s){
    attempts[s]=0;draws[s]=0;invalidReference[s]=0;densityMismatch[s]=0;pdfMismatch[s]=0;cdfMismatch[s]=0;claimedReplayMismatch[s]=0;details[s]=0;writeFailures[s]=0;
    for(auto&v:status[s])v=0;
    for(auto&axis:histogram[s])for(auto&v:axis)v=0;
}}
struct WorkerScope {
    WorkerScope(bool on,Reporter callback){used[0]=used[1]=0;reporter=on?callback:nullptr;}
    ~WorkerScope(){reporter=nullptr;used[0]=used[1]=0;}
    WorkerScope(const WorkerScope&)=delete;WorkerScope&operator=(const WorkerScope&)=delete;
};
struct Attempt {
    bool active=false;Record r;
    Attempt(bool light,unsigned material){const unsigned s=light?1u:0u;
        if(!reporter||used[s]>=Budget)return;
        ++used[s];active=true;r.side=s;r.material=material;attempts[s].fetch_add(1ull,std::memory_order_relaxed);
    }
    bool Active()const{return active;}
    void Draw(double nu,double nv,double rough,double anis,double u0,double u1,V fixed,V h,double claimed){
        if(!active)return;
        r.nu=nu;r.nv=nv;r.rough=rough;r.anis=anis;r.u0=u0;r.u1=u1;r.fixed=fixed;r.h=h;r.claimedHalf=claimed;r.status=1;
        r.draw=Reference(rough,anis,h);draws[r.side].fetch_add(1ull,std::memory_order_relaxed);
    }
    void PdfRejected(){if(active)r.status=2;}
    void Accept(V pair,double pdf){if(!active)return;r.status=3;r.pair=pair;r.reportedPdf=pdf;r.emitted=Reference(r.rough,r.anis,pair);}
    ~Attempt(){
        if(!active)return;
        const unsigned s=r.side;status[s][r.status].fetch_add(1ull,std::memory_order_relaxed);
        if(r.status){
            if(!r.draw.valid)invalidReference[s].fetch_add(1ull,std::memory_order_relaxed);
            else{
                if(Relative(r.draw.generated,r.claimedHalf)>2e-4)densityMismatch[s].fetch_add(1ull,std::memory_order_relaxed);
                if(Relative(r.draw.claimedIndependent,r.claimedHalf)>2e-4)claimedReplayMismatch[s].fetch_add(1ull,std::memory_order_relaxed);
                double vError=std::fabs(r.draw.cdfV-r.u1);vError=std::fmin(vError,1.-vError);
                if(std::fabs(r.draw.cdfU-r.u0)>1e-8||vError>1e-8)cdfMismatch[s].fetch_add(1ull,std::memory_order_relaxed);
                for(unsigned axis=0;axis<2;++axis){double u=axis?r.draw.cdfV:r.draw.cdfU;
                    unsigned bin=u<=0.?0u:u>=1.?Bins-1:unsigned(u*Bins);
                    histogram[s][axis][bin].fetch_add(1ull,std::memory_order_relaxed);
                }
            }
            if(r.status==3){
                if(!r.emitted.valid)invalidReference[s].fetch_add(1ull,std::memory_order_relaxed);
                else{
                    const auto fixed=mlhero15kk::Unit(r.fixed),h=mlhero15kk::Unit(r.pair);
                    const double jacobian=4.*std::fabs(mlhero15kk::Dot(fixed,h));
                    if(jacobian>0.&&Relative(r.emitted.generated/jacobian,r.reportedPdf)>2e-4)
                        pdfMismatch[s].fetch_add(1ull,std::memory_order_relaxed);
                }
            }
        }
        r.detail=details[s].fetch_add(1ull,std::memory_order_relaxed);
        if(r.detail<16ull)reporter(r);
    }
    Attempt(const Attempt&)=delete;Attempt&operator=(const Attempt&)=delete;
};
inline void WriteRecord(FILE*f,const Record&r,unsigned thread,unsigned sample){
    const auto fixed=mlhero15kk::Unit(r.fixed),h=mlhero15kk::Unit(r.pair);
    const double jacobian=4.*std::fabs(mlhero15kk::Dot(fixed,h));
    fprintf(f,"phase15la SCHLICK_SAMPLE side=%s detail=%llu thread=%u sample=%u materialID=%u outcome=%u nu=%.17g nv=%.17g roughness=%.17g anisotropy=%.17g u=(%.17g,%.17g) h=(%.17g,%.17g,%.17g) fixed=(%.17g,%.17g,%.17g) referenceValid=%u generatedHalfDensity=%.17g brdfHalfD=%.17g independentlyReplayedD=%.17g generatedOverBrdfD=%.17g recoveredU=(%.17g,%.17g) emittedReferenceValid=%u reportedDirectionPdf=%.17g emittedReferenceDirectionPdf=%.17g measure=SOLID_ANGLE noRenormalizationAfterRejection=1 diagnosticOnly=1 metal2PdfFix15lb=1\n",
        r.side?"LIGHT":"EYE",r.detail,thread,sample,r.material,r.status,r.nu,r.nv,r.rough,r.anis,r.u0,r.u1,
        r.h.x,r.h.y,r.h.z,r.fixed.x,r.fixed.y,r.fixed.z,r.draw.valid?1u:0u,r.draw.generated,r.claimedHalf,r.draw.claimedIndependent,
        r.draw.valid&&r.claimedHalf>0.?r.draw.generated/r.claimedHalf:0.,r.draw.cdfU,r.draw.cdfV,
        r.emitted.valid?1u:0u,r.reportedPdf,r.emitted.valid&&jacobian>0.?r.emitted.generated/jacobian:0.);
}
inline void WriteSummary(FILE*f){for(unsigned s=0;s<2;++s){
    fprintf(f,"phase15la SCHLICK_SUMMARY side=%s selectedAttempts=%llu halfDraws=%llu preDrawRejected=%llu hemisphereRejected=%llu pdfRejected=%llu geometryPdfAccepted=%llu invalidReferenceStages=%llu ndfDensityDifferences=%llu emittedPdfMismatches=%llu inverseCdfMismatches=%llu claimedDReplayMismatches=%llu detailsSelected=%llu writeFailures=%llu selection=FIRST_256_ATTEMPTS_PER_WORKER_SIDE acceptedMeaning=GEOMETRY_PDF_BEFORE_FRESNEL histogramNotIIDTest=1 physicsChanged=1 correctedMetal2Pdf=1 ndfIsNotSamplingDensity=1\n",
        s?"LIGHT":"EYE",attempts[s].load(),draws[s].load(),status[s][0].load(),status[s][1].load(),status[s][2].load(),status[s][3].load(),
        invalidReference[s].load(),densityMismatch[s].load(),pdfMismatch[s].load(),cdfMismatch[s].load(),claimedReplayMismatch[s].load(),
        details[s].load()<16ull?details[s].load():16ull,writeFailures[s].load());
    for(unsigned axis=0;axis<2;++axis){fprintf(f,"phase15la SCHLICK_CDF_HISTOGRAM side=%s axis=%u bins=16 counts=",s?"LIGHT":"EYE",axis);
        for(unsigned i=0;i<Bins;++i)fprintf(f,i?",%llu":"%llu",histogram[s][axis][i].load());fprintf(f," diagnosticOnly=1 metal2PdfFix15lb=1\n");}
}}
// Composite Simpson in c=cos(theta) and phi, independent of production sampling.
inline void WriteFixtures(FILE*f){
    const double rows[3][2]={{.0025,0.},{.09,0.},{.015,1.-.0025/.09}};
    for(unsigned row=0;row<3;++row)for(unsigned resolution:{16384u,32768u}){
        const double rough=rows[row][0],anis=rows[row][1];double radial=0.,radialD=0.,radialDCos=0.,angular=0.,angularA=0.,acceptedRadial=0.,acceptedClaim=0.;
        for(unsigned i=0;i<=resolution;++i){const double c=double(i)/resolution;
            const double weight=(i==0||i==resolution)?1.:(i&1u)?4.:2.;
            const double den=(1.-c)*(1.+c)+rough*c*c;
            radial+=weight*2.*rough*c/(den*den);radialD+=weight*rough/(den*den);radialDCos+=weight*rough*c/(den*den);
            const double lo=std::sqrt(.5),ca=lo+(1.-lo)*c,da=(1.-ca)*(1.+ca)+rough*ca*ca;
            acceptedRadial+=weight*2.*rough*ca/(da*da)*(1.-lo);
            acceptedClaim+=weight*rough/(da*da)*(1.-lo);
            const double phi=2.*Pi*i/resolution,p=1.-std::fabs(anis),w=anis>0.?std::cos(phi):std::sin(phi);
            angular+=weight*PhiPdf(phi,anis);angularA+=weight*std::sqrt(p/(p*p+w*w*(1.-p*p)));
        }
        radial/=3.*resolution;radialD/=3.*resolution;radialDCos/=3.*resolution;
        angular*=2.*Pi/(3.*resolution);angularA*=2.*Pi/(3.*resolution);
        acceptedRadial/=3.*resolution;acceptedClaim/=3.*resolution;
        fprintf(f,"phase15la SCHLICK_INTEGRAL fixture=%u resolution=%u roughness=%.17g anisotropy=%.17g generatedHalfDensityIntegral=%.17g brdfDIntegral=%.17g brdfDCosIntegral=%.17g independentQuadrature=COMPOSITE_SIMPSON_C_AND_PHI expectedGeneratedIntegral=1 diagnosticOnly=1 metal2PdfFix15lb=1\n",
            row,resolution,rough,anis,radial*angular,radialD*angularA/Pi,radialDCos*angularA/Pi);
        fprintf(f,"phase15la SCHLICK_REFLECTION_MASS fixture=%u resolution=%u fixedLocalZ=1 generatedAcceptedDirectionMass=%.17g exactAcceptanceFromRadialCDF=%.17g trueRejectedMass=%.17g legacyDAsPdfAcceptedMass=%.17g legacyDAsPdfPlusTrueRejected=%.17g rejectionIsNotResampled=1 diagnosticOnly=1 metal2PdfFix15lb=1\n",
            row,resolution,acceptedRadial*angular,1./(1.+rough),rough/(1.+rough),acceptedClaim*angularA/Pi,
            acceptedClaim*angularA/Pi+rough/(1.+rough));
    }
}
} }
#endif

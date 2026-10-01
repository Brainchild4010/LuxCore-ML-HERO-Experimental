/***************************************************************************
 * ML HERO 15do tone mapper                                                *
 ***************************************************************************/
#include <boost/lexical_cast.hpp>
#include <cmath>
#include "luxrays/kernels/kernels.h"
#include "luxrays/utils/serializationutils.h"
#include "slg/kernels/kernels.h"
#include "slg/film/film.h"
#include "slg/film/imagepipeline/plugins/tonemaps/reinhardheroexposure.h"
using namespace std; using namespace luxrays; using namespace slg;
BOOST_CLASS_EXPORT_IMPLEMENT(slg::ReinhardHeroExposureToneMap)
ReinhardHeroExposureToneMap::ReinhardHeroExposureToneMap() : exposureBias(0.f), burn(4.f), hardwareDevice(nullptr), hwAccumBuffer(nullptr), opRGBValuesReduceKernel(nullptr), opRGBValueAccumulateKernel(nullptr), applyKernel(nullptr) {}
ReinhardHeroExposureToneMap::ReinhardHeroExposureToneMap(const float ev, const float b) : exposureBias(ev), burn(b), hardwareDevice(nullptr), hwAccumBuffer(nullptr), opRGBValuesReduceKernel(nullptr), opRGBValueAccumulateKernel(nullptr), applyKernel(nullptr) {}
ReinhardHeroExposureToneMap::~ReinhardHeroExposureToneMap() { delete opRGBValuesReduceKernel; delete opRGBValueAccumulateKernel; delete applyKernel; if (hardwareDevice) hardwareDevice->FreeBuffer(&hwAccumBuffer); }
void ReinhardHeroExposureToneMap::Apply(Film &film, const u_int index) {
    RGBColor *rgbPixels = (RGBColor *)((Spectrum *)film.channel_IMAGEPIPELINEs[index]->GetPixels());
    const u_int pixelCount = film.GetWidth() * film.GetHeight();
    const bool hasPN = film.HasChannel(Film::RADIANCE_PER_PIXEL_NORMALIZED), hasSN = film.HasChannel(Film::RADIANCE_PER_SCREEN_NORMALIZED);
    float logSum = 0.f;
    for (u_int i=0;i<pixelCount;++i) if (film.HasSamples(hasPN,hasSN,i) && !rgbPixels[i].IsInf()) logSum += logf(Max(rgbPixels[i].Y(),1e-6f));
    float Ywa = pixelCount ? expf(logSum / pixelCount) : 1.f; if (!(Ywa > 0.f)) Ywa = 1.f;
    const float gain = powf(2.f, exposureBias), key=.18f, safeBurn=Max(burn,1e-6f), invB2=1.f/(safeBurn*safeBurn);
    #pragma omp parallel for
    for (int i=0;i<(int)pixelCount;++i) if (film.HasSamples(hasPN,hasSN,i)) {
        const float Y=Max(rgbPixels[i].Y(),0.f); const float L=key*gain*Y/Ywa;
        const float Ld=L*(1.f+L*invB2)/(1.f+L);
        rgbPixels[i] *= (Y > 1e-6f) ? (Ld/Y) : 0.f;
    }
}
void ReinhardHeroExposureToneMap::AddHWChannelsUsed(unordered_set<Film::FilmChannelType, hash<int> > &s) const { s.insert(Film::IMAGEPIPELINE); }
void ReinhardHeroExposureToneMap::ApplyHW(Film &film, const u_int index) {
    const u_int pixelCount=film.GetWidth()*film.GetHeight(), workSize=RoundUp((pixelCount+1)/2,64u);
    if (!applyKernel) {
        film.ctx->SetVerbose(true); hardwareDevice=film.hardwareDevice; hardwareDevice->AllocBufferRW(&hwAccumBuffer,nullptr,(workSize/64)*sizeof(float)*3,"Accumulation");
        vector<string> opts; opts.push_back("-D LUXRAYS_OPENCL_KERNEL"); opts.push_back("-D SLG_OPENCL_KERNEL"); HardwareDeviceProgram *program=nullptr;
        hardwareDevice->CompileProgram(&program,opts,luxrays::ocl::KernelSource_luxrays_types+luxrays::ocl::KernelSource_color_types+luxrays::ocl::KernelSource_color_funcs+slg::ocl::KernelSource_tonemap_reinhardhero_exposure_funcs+slg::ocl::KernelSource_tonemap_reduce_funcs,"ReinhardHeroExposureToneMap");
        hardwareDevice->GetKernel(program,&opRGBValuesReduceKernel,"OpRGBValuesReduce"); hardwareDevice->GetKernel(program,&opRGBValueAccumulateKernel,"OpRGBValueAccumulate"); hardwareDevice->GetKernel(program,&applyKernel,"ReinhardHeroExposureToneMap_Apply"); delete program;
        u_int a=0; hardwareDevice->SetKernelArg(opRGBValuesReduceKernel,a++,film.GetWidth()); hardwareDevice->SetKernelArg(opRGBValuesReduceKernel,a++,film.GetHeight()); hardwareDevice->SetKernelArg(opRGBValuesReduceKernel,a++,film.hw_IMAGEPIPELINE); hardwareDevice->SetKernelArg(opRGBValuesReduceKernel,a++,hwAccumBuffer);
        a=0; hardwareDevice->SetKernelArg(opRGBValueAccumulateKernel,a++,workSize/64); hardwareDevice->SetKernelArg(opRGBValueAccumulateKernel,a++,hwAccumBuffer);
        a=0; hardwareDevice->SetKernelArg(applyKernel,a++,film.GetWidth()); hardwareDevice->SetKernelArg(applyKernel,a++,film.GetHeight()); hardwareDevice->SetKernelArg(applyKernel,a++,film.hw_IMAGEPIPELINE); hardwareDevice->SetKernelArg(applyKernel,a++,exposureBias); hardwareDevice->SetKernelArg(applyKernel,a++,burn); hardwareDevice->SetKernelArg(applyKernel,a++,hwAccumBuffer); film.ctx->SetVerbose(false);
    }
    hardwareDevice->EnqueueKernel(opRGBValuesReduceKernel,HardwareDeviceRange(workSize),HardwareDeviceRange(64)); hardwareDevice->EnqueueKernel(opRGBValueAccumulateKernel,HardwareDeviceRange(64),HardwareDeviceRange(64)); hardwareDevice->EnqueueKernel(applyKernel,HardwareDeviceRange(RoundUp(pixelCount,256u)),HardwareDeviceRange(256));
}

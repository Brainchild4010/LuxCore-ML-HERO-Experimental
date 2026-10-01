/***************************************************************************
 * ML HERO 15do tone mapper                                                *
 ***************************************************************************/
#include <boost/lexical_cast.hpp>
#include <cmath>
#include "luxrays/kernels/kernels.h"
#include "luxrays/utils/serializationutils.h"
#include "slg/kernels/kernels.h"
#include "slg/film/film.h"
#include "slg/film/imagepipeline/plugins/tonemaps/reinhardheroclassic.h"
using namespace std; using namespace luxrays; using namespace slg;
BOOST_CLASS_EXPORT_IMPLEMENT(slg::ReinhardHeroClassicToneMap)
ReinhardHeroClassicToneMap::ReinhardHeroClassicToneMap() : key(.18f), whitePoint(4.f), hardwareDevice(nullptr), hwAccumBuffer(nullptr), opRGBValuesReduceKernel(nullptr), opRGBValueAccumulateKernel(nullptr), applyKernel(nullptr) {}
ReinhardHeroClassicToneMap::ReinhardHeroClassicToneMap(const float k, const float w) : key(k), whitePoint(w), hardwareDevice(nullptr), hwAccumBuffer(nullptr), opRGBValuesReduceKernel(nullptr), opRGBValueAccumulateKernel(nullptr), applyKernel(nullptr) {}
ReinhardHeroClassicToneMap::~ReinhardHeroClassicToneMap() { delete opRGBValuesReduceKernel; delete opRGBValueAccumulateKernel; delete applyKernel; if (hardwareDevice) hardwareDevice->FreeBuffer(&hwAccumBuffer); }
void ReinhardHeroClassicToneMap::Apply(Film &film, const u_int index) {
    RGBColor *rgbPixels = (RGBColor *)((Spectrum *)film.channel_IMAGEPIPELINEs[index]->GetPixels());
    const u_int pixelCount = film.GetWidth() * film.GetHeight();
    const bool hasPN = film.HasChannel(Film::RADIANCE_PER_PIXEL_NORMALIZED), hasSN = film.HasChannel(Film::RADIANCE_PER_SCREEN_NORMALIZED);
    float logSum = 0.f;
    for (u_int i=0;i<pixelCount;++i) if (film.HasSamples(hasPN,hasSN,i) && !rgbPixels[i].IsInf()) logSum += logf(Max(rgbPixels[i].Y(),1e-6f));
    float Ywa = pixelCount ? expf(logSum / pixelCount) : 1.f; if (!(Ywa > 0.f)) Ywa = 1.f;
    const float safeKey = Max(key, 1e-6f), safeWhite = Max(whitePoint, 1e-6f), invW2 = 1.f/(safeWhite*safeWhite);
    #pragma omp parallel for
    for (int i=0;i<(int)pixelCount;++i) if (film.HasSamples(hasPN,hasSN,i)) {
        const float Y = Max(rgbPixels[i].Y(), 0.f); const float L = safeKey * Y / Ywa;
        const float Ld = L * (1.f + L * invW2) / (1.f + L);
        rgbPixels[i] *= (Y > 1e-6f) ? (Ld / Y) : 0.f;
    }
}
void ReinhardHeroClassicToneMap::AddHWChannelsUsed(unordered_set<Film::FilmChannelType, hash<int> > &s) const { s.insert(Film::IMAGEPIPELINE); }
void ReinhardHeroClassicToneMap::ApplyHW(Film &film, const u_int index) {
    const u_int pixelCount=film.GetWidth()*film.GetHeight(), workSize=RoundUp((pixelCount+1)/2,64u);
    if (!applyKernel) {
        film.ctx->SetVerbose(true); hardwareDevice=film.hardwareDevice; hardwareDevice->AllocBufferRW(&hwAccumBuffer,nullptr,(workSize/64)*sizeof(float)*3,"Accumulation");
        vector<string> opts; opts.push_back("-D LUXRAYS_OPENCL_KERNEL"); opts.push_back("-D SLG_OPENCL_KERNEL"); HardwareDeviceProgram *program=nullptr;
        hardwareDevice->CompileProgram(&program,opts,luxrays::ocl::KernelSource_luxrays_types+luxrays::ocl::KernelSource_color_types+luxrays::ocl::KernelSource_color_funcs+slg::ocl::KernelSource_tonemap_reinhardhero_classic_funcs+slg::ocl::KernelSource_tonemap_reduce_funcs,"ReinhardHeroClassicToneMap");
        hardwareDevice->GetKernel(program,&opRGBValuesReduceKernel,"OpRGBValuesReduce"); hardwareDevice->GetKernel(program,&opRGBValueAccumulateKernel,"OpRGBValueAccumulate"); hardwareDevice->GetKernel(program,&applyKernel,"ReinhardHeroClassicToneMap_Apply"); delete program;
        u_int a=0; hardwareDevice->SetKernelArg(opRGBValuesReduceKernel,a++,film.GetWidth()); hardwareDevice->SetKernelArg(opRGBValuesReduceKernel,a++,film.GetHeight()); hardwareDevice->SetKernelArg(opRGBValuesReduceKernel,a++,film.hw_IMAGEPIPELINE); hardwareDevice->SetKernelArg(opRGBValuesReduceKernel,a++,hwAccumBuffer);
        a=0; hardwareDevice->SetKernelArg(opRGBValueAccumulateKernel,a++,workSize/64); hardwareDevice->SetKernelArg(opRGBValueAccumulateKernel,a++,hwAccumBuffer);
        a=0; hardwareDevice->SetKernelArg(applyKernel,a++,film.GetWidth()); hardwareDevice->SetKernelArg(applyKernel,a++,film.GetHeight()); hardwareDevice->SetKernelArg(applyKernel,a++,film.hw_IMAGEPIPELINE); hardwareDevice->SetKernelArg(applyKernel,a++,key); hardwareDevice->SetKernelArg(applyKernel,a++,whitePoint); hardwareDevice->SetKernelArg(applyKernel,a++,hwAccumBuffer); film.ctx->SetVerbose(false);
    }
    hardwareDevice->EnqueueKernel(opRGBValuesReduceKernel,HardwareDeviceRange(workSize),HardwareDeviceRange(64)); hardwareDevice->EnqueueKernel(opRGBValueAccumulateKernel,HardwareDeviceRange(64),HardwareDeviceRange(64)); hardwareDevice->EnqueueKernel(applyKernel,HardwareDeviceRange(RoundUp(pixelCount,256u)),HardwareDeviceRange(256));
}

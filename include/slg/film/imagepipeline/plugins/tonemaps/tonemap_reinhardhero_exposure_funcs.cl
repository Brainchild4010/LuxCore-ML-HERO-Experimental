#line 2 "tonemap_reinhardhero_exposure_funcs.cl"
__kernel void ReinhardHeroExposureToneMap_Apply(const uint filmWidth,const uint filmHeight,__global float *channel_IMAGEPIPELINE,const float exposureBias,const float burn,__global float *totalRGB) {
    const size_t gid=get_global_id(0); const uint pixelCount=filmWidth*filmHeight; if(gid>=pixelCount) return;
    if(!isinf(channel_IMAGEPIPELINE[gid*3])) {
        float Ywa=native_exp(totalRGB[0]/pixelCount); if(!(Ywa>0.f)) Ywa=1.f;
        const float gain=native_powr(2.f,exposureBias), key=.18f, safeBurn=fmax(burn,1e-6f), invB2=1.f/(safeBurn*safeBurn);
        __global float *pixel=&channel_IMAGEPIPELINE[gid*3]; float3 v=VLOAD3F(pixel); const float Y=fmax(Spectrum_Y(v),0.f); const float L=key*gain*Y/Ywa; const float Ld=L*(1.f+L*invB2)/(1.f+L); v *= (Y>1e-6f)?(Ld/Y):0.f; VSTORE3F(v,pixel);
    }
}

OPENCL_FORCE_INLINE float3 REDUCE_OP(const float3 a, const float3 b) {
    if (Spectrum_IsNanOrInf(b)) return a;
    const float y=fmax(Spectrum_Y(b),1e-6f); return a+native_log(y);
}
OPENCL_FORCE_INLINE float3 ACCUM_OP(const float3 a,const float3 b){return a+b;}

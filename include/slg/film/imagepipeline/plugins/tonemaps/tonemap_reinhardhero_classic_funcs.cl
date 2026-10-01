#line 2 "tonemap_reinhardhero_classic_funcs.cl"
__kernel void ReinhardHeroClassicToneMap_Apply(const uint filmWidth,const uint filmHeight,__global float *channel_IMAGEPIPELINE,const float key,const float whitePoint,__global float *totalRGB) {
    const size_t gid=get_global_id(0); const uint pixelCount=filmWidth*filmHeight; if(gid>=pixelCount) return;
    if(!isinf(channel_IMAGEPIPELINE[gid*3])) {
        float Ywa=native_exp(totalRGB[0]/pixelCount); if(!(Ywa>0.f)) Ywa=1.f;
        const float safeKey=fmax(key,1e-6f), safeWhite=fmax(whitePoint,1e-6f), invW2=1.f/(safeWhite*safeWhite);
        __global float *pixel=&channel_IMAGEPIPELINE[gid*3]; float3 v=VLOAD3F(pixel); const float Y=fmax(Spectrum_Y(v),0.f); const float L=safeKey*Y/Ywa; const float Ld=L*(1.f+L*invW2)/(1.f+L); v *= (Y>1e-6f)?(Ld/Y):0.f; VSTORE3F(v,pixel);
    }
}

OPENCL_FORCE_INLINE float3 REDUCE_OP(const float3 a, const float3 b) {
    if (Spectrum_IsNanOrInf(b)) return a;
    const float y=fmax(Spectrum_Y(b),1e-6f); return a+native_log(y);
}
OPENCL_FORCE_INLINE float3 ACCUM_OP(const float3 a,const float3 b){return a+b;}

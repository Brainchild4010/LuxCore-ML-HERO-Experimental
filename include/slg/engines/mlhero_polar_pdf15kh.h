#ifndef MLHERO_POLAR_PDF15KH_H
#define MLHERO_POLAR_PDF15KH_H
#include <atomic>
#include <cmath>

namespace slg { namespace mlhero15kh {
struct Coordinates {
    double theta=0., phi=0., sine=0.;
    bool valid=false, exactPole=false, roundedAxis=false;
};
inline Coordinates FromDirection(double x,double y,double z) {
    Coordinates a;
    if (!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z)) return a;
    const double transverse=std::hypot(x,y);
    const double length=std::hypot(transverse,z);
    if (!(length>0.)) return a;
    a.valid=true;
    a.exactPole=transverse==0.;
    a.roundedAxis=!a.exactPole && std::fabs(z)>=1.;
    a.theta=std::atan2(transverse,z);
    a.phi=std::atan2(y,x);
    if(a.phi<0.) a.phi+=6.283185307179586476925286766559;
    a.sine=transverse/length;
    return a;
}
inline bool IsExactPole(double x,double y,double z) {
    const auto a=FromDirection(x,y,z);
    return a.valid && a.exactPole;
}
// Instrumentation is gated by the existing per-worker diagnostic state.
// The mathematical functions above never depend on diagnostic settings.
inline std::atomic<unsigned long long> roundedPolarRecovered{0ull};
inline std::atomic<unsigned long long> exactPolePdfZero{0ull};
inline std::atomic<unsigned long long> invalidDirectionPdfZero{0ull};
inline std::atomic<unsigned long long> rejectedSampleEndpoints{0ull};
inline void ResetCounters() {
    roundedPolarRecovered.store(0ull,std::memory_order_relaxed);
    exactPolePdfZero.store(0ull,std::memory_order_relaxed);
    invalidDirectionPdfZero.store(0ull,std::memory_order_relaxed);
    rejectedSampleEndpoints.store(0ull,std::memory_order_relaxed);
}
} }
#endif

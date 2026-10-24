#ifndef SLG_MLHERO_GLASS_REVERSE15KT_H
#define SLG_MLHERO_GLASS_REVERSE15KT_H
#include <cmath>
namespace slg { namespace mlhero15kt {
enum Status : unsigned { NOT_ATTEMPTED=0,INVALID_INPUT=1,INVALID_LOBES=2,EVALUATED=3,INVALID_WAVELENGTH=4 };
// Conditional native event mass at one fixed HERO wavelength and the same surface.
// This is neither a solid-angle density nor a complete reverse path probability.
struct Evidence {
    unsigned status=NOT_ATTEMPTED;
    bool forwardMatches=false,supported=false,reverseTir=false;
    bool reverseFromLight=false,forwardIntoObject=false,reverseIntoObject=false;
    float forwardMass=0.f,reverseMass=0.f,forwardThreshold=0.f,reverseThreshold=0.f;
    float nc=0.f,nt=0.f,wavelength=-1.f,fixedCosine=0.f,reverseCosine=0.f;
    double directionResidual=-1.;
};
inline bool CloseMass(double a,double b){
    return std::isfinite(a)&&std::isfinite(b)&&std::fabs(a-b)<=2.e-6;
}
} }
#endif

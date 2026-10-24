#ifndef SLG_MLHERO_GLASS_FACTORS15KU_H
#define SLG_MLHERO_GLASS_FACTORS15KU_H
#include <cmath>
#include "slg/engines/mlhero_glass_reverse15kt.h"
namespace slg { namespace mlhero15ku {
enum Outcome : unsigned { OTHER=0,VERIFIED=1,INVALID_EDGE=2,MISSING_EVIDENCE=3,
    FORWARD_MISMATCH=4,UNSUPPORTED_REVERSE=5,INVALID_MASS=6 };
// A product of native conditional GLASS event masses ONLY. Never a strategy/path PDF.
struct Factors {
    unsigned glassEdges=0,matched=0,missing=0,otherDelta=0,zeroReverse=0;
    unsigned lastOutcome=OTHER;
    double logForwardMatched=0.,logReverseMatched=0.,logReverseOverForwardMatched=0.;
    double logForwardRRMatched=0.;
    void Observe(bool glass,bool delta,bool validEdge,double forward,double rr,
            const slg::mlhero15kt::Evidence&e){
        lastOutcome=OTHER;
        if(!glass){if(delta)++otherDelta;return;}
        ++glassEdges;
        if(e.status==slg::mlhero15kt::EVALUATED&&e.reverseMass==0.f)++zeroReverse;
        if(!delta||!validEdge)lastOutcome=INVALID_EDGE;
        else if(e.status!=slg::mlhero15kt::EVALUATED)lastOutcome=MISSING_EVIDENCE;
        else if(!e.forwardMatches||!slg::mlhero15kt::CloseMass(e.forwardMass,forward))lastOutcome=FORWARD_MISMATCH;
        else if(!e.supported)lastOutcome=UNSUPPORTED_REVERSE;
        else if(!std::isfinite(forward)||forward<=0.||forward>1.||
                !std::isfinite(e.reverseMass)||e.reverseMass<=0.f||e.reverseMass>1.f||
                !std::isfinite(rr)||rr<=0.||rr>1.)lastOutcome=INVALID_MASS;
        else {
            lastOutcome=VERIFIED;++matched;
            const double f=std::log(forward),r=std::log(double(e.reverseMass));
            logForwardMatched+=f;logReverseMatched+=r;logReverseOverForwardMatched+=r-f;
            logForwardRRMatched+=std::log(rr);return;
        }
        ++missing;
    }
    bool Complete(bool prefixValid,bool prefixTruncated)const{
        return glassEdges>0&&missing==0&&matched==glassEdges&&prefixValid&&!prefixTruncated;
    }
};
} }
#endif

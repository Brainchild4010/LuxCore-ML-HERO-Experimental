/***************************************************************************
 * ML HERO phase 15ns: accounting for the native Mix direction anchor.    *
 ***************************************************************************/

#ifndef _SLG_MLHERO_MIX_DIRECTION_ANCHOR15NS_H
#define _SLG_MLHERO_MIX_DIRECTION_ANCHOR15NS_H

namespace slg { namespace mlhero15ns {

struct Accounting {
	bool requestClosure;
	bool captureClosure;
	bool applicationClosure;
	bool outcomeClosure;
	bool rejectSubset;
	bool pass;
};

inline Accounting AnalyzeAccounting(const unsigned long long requested,
		const unsigned long long bounceAttempts,
		const unsigned long long captureMatched,
		const unsigned long long captureUnmatched,
		const unsigned long long anchorApplied,
		const unsigned long long accepted,
		const unsigned long long fallback,
		const unsigned long long closureReject) {
	Accounting result;
	result.requestClosure = requested == bounceAttempts;
	result.captureClosure = requested == captureMatched + captureUnmatched;
	result.applicationClosure = anchorApplied == captureMatched;
	result.outcomeClosure = requested == accepted + fallback;
	result.rejectSubset = closureReject <= fallback;
	result.pass = result.requestClosure && result.captureClosure &&
		result.applicationClosure && result.outcomeClosure &&
		result.rejectSubset;
	return result;
}

} }

#endif // _SLG_MLHERO_MIX_DIRECTION_ANCHOR15NS_H

/***************************************************************************
 * ML HERO phase 15no: pure accounting closure for Mix bounce diagnostics.
 ***************************************************************************/

#ifndef _SLG_MLHERO_MIX_FALLBACK15NO_H
#define _SLG_MLHERO_MIX_FALLBACK15NO_H

namespace slg { namespace mlhero15no {

struct Accounting {
	bool attemptClosure;
	bool prepareReasonClosure;
	bool evaluationReasonClosure;
	bool pass;
};

inline Accounting AnalyzeAccounting(const unsigned long long attempts,
		const unsigned long long accepted,
		const unsigned long long prepareFallbacks,
		const unsigned long long evaluationFallbacks,
		const unsigned long long prepareReasonTotal,
		const unsigned long long evaluationReasonTotal) {
	Accounting result;
	result.attemptClosure = attempts == accepted + prepareFallbacks +
		evaluationFallbacks;
	result.prepareReasonClosure = prepareFallbacks == prepareReasonTotal;
	result.evaluationReasonClosure = evaluationFallbacks ==
		evaluationReasonTotal;
	result.pass = result.attemptClosure && result.prepareReasonClosure &&
		result.evaluationReasonClosure;
	return result;
}

} }

#endif

/***************************************************************************
 * ML HERO phase 15nz: RoughGlass natural child-state audit.              *
 *                                                                         *
 * Pure classification helpers. No sampler, transport, PDF, MIS or film   *
 * state is mutated here.                                                  *
 ***************************************************************************/

#ifndef SLG_MLHERO_ROUGHGLASS_CHILD_STATE15NZ_H
#define SLG_MLHERO_ROUGHGLASS_CHILD_STATE15NZ_H

#include <cmath>

namespace slg { namespace mlhero15nz {

enum ChildClass : unsigned {
	TRANSMISSION_READY = 0u,
	TIR_REFLECTION_READY = 1u,
	HEMISPHERE_ZERO_READY = 2u,
	HELPER_REJECTED = 3u,
	INVALID_DIRECTION = 4u,
	INVALID_EVENT = 5u,
	INVALID_PDF = 6u,
	INVALID_VALUE = 7u,
	PDF_CLOSURE_MISMATCH = 8u
};

inline ChildClass Classify(const bool helperOK, const bool naturalTir,
		const bool hemisphereZero, const double directionLengthSquared,
		const bool eventTransmit, const bool eventReflect,
		const bool eventGlossy, const bool eventSpecular,
		const double directPdf, const double reversePdf,
		const bool valueFinite, const bool valueNonNegative,
		const bool valueBlack,
		const double directPdfClosureRelativeError) {
	if (!helperOK)
		return HELPER_REJECTED;
	if (!std::isfinite(directionLengthSquared) ||
			std::fabs(directionLengthSquared - 1.) > 2e-5)
		return INVALID_DIRECTION;
	if (hemisphereZero) {
		if (!eventGlossy || eventSpecular ||
				(naturalTir ? (!eventReflect || eventTransmit) :
					(!eventTransmit || eventReflect)))
			return INVALID_EVENT;
		if (!std::isfinite(directPdf) || directPdf != 0. ||
				!std::isfinite(reversePdf) || reversePdf != 0.)
			return INVALID_PDF;
		if (!valueFinite || !valueNonNegative || !valueBlack)
			return INVALID_VALUE;
		return HEMISPHERE_ZERO_READY;
	}
	if (!eventGlossy || eventSpecular ||
			(naturalTir ? (!eventReflect || eventTransmit) :
				(!eventTransmit || eventReflect)))
		return INVALID_EVENT;
	if (!std::isfinite(directPdf) || !(directPdf > 0.) ||
			!std::isfinite(reversePdf) || !(reversePdf > 0.))
		return INVALID_PDF;
	if (!valueFinite || !valueNonNegative)
		return INVALID_VALUE;
	if (!naturalTir && (!std::isfinite(directPdfClosureRelativeError) ||
			directPdfClosureRelativeError > 2e-4))
		return PDF_CLOSURE_MISMATCH;
	return naturalTir ? TIR_REFLECTION_READY : TRANSMISSION_READY;
}

inline bool IsReady(const ChildClass c) {
	return c == TRANSMISSION_READY || c == TIR_REFLECTION_READY ||
		c == HEMISPHERE_ZERO_READY;
}

inline const char *ChildClassName(const ChildClass c) {
	switch (c) {
		case TRANSMISSION_READY: return "TRANSMISSION_READY";
		case TIR_REFLECTION_READY: return "TIR_REFLECTION_READY";
		case HEMISPHERE_ZERO_READY: return "HEMISPHERE_ZERO_READY";
		case HELPER_REJECTED: return "HELPER_REJECTED";
		case INVALID_DIRECTION: return "INVALID_DIRECTION";
		case INVALID_EVENT: return "INVALID_EVENT";
		case INVALID_PDF: return "INVALID_PDF";
		case INVALID_VALUE: return "INVALID_VALUE";
		default: return "PDF_CLOSURE_MISMATCH";
	}
}

} }

#endif

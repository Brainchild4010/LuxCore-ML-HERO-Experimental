#ifndef SLG_MLHERO_GLASS_REVERSE_PRECISION15MX_H
#define SLG_MLHERO_GLASS_REVERSE_PRECISION15MX_H

#include <algorithm>
#include <cmath>

namespace slg { namespace mlhero15mx {

struct V {
	double x = 0., y = 0., z = 0.;

	V() = default;
	V(const double vx, const double vy, const double vz) : x(vx), y(vy), z(vz) { }

	template <class T> explicit V(const T &v) : x(v.x), y(v.y), z(v.z) { }

	V operator *(const double s) const { return V(x * s, y * s, z * s); }
	V operator -(const V &v) const { return V(x - v.x, y - v.y, z - v.z); }
};

inline double Dot(const V &a, const V &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline bool Unit(const V &v, V *result) {
	if (!result)
		return false;
	const double length2 = Dot(v, v);
	if (!(length2 > 0.) || !std::isfinite(length2))
		return false;
	const double inverseLength = 1. / std::sqrt(length2);
	*result = v * inverseLength;
	return std::isfinite(result->x) && std::isfinite(result->y) &&
		std::isfinite(result->z);
}

inline double Distance(const V &a, const V &b) {
	const V d = a - b;
	return std::sqrt(std::max(0., Dot(d, d)));
}

// Ideal local reflection. Both arguments point away from the surface, as in
// LuxCore's BSDF convention.
inline bool Reflect(const V &fixedInput, V *sampled) {
	V fixed;
	if (!Unit(fixedInput, &fixed) || !sampled)
		return false;
	return Unit(V(-fixed.x, -fixed.y, fixed.z), sampled);
}

// Double-precision version of GlassMaterial::EvalSpecularTransmission's
// direction equation. The tangential sine is formed directly from x/y. This
// avoids subtracting two nearly equal numbers for a grazing fixed direction.
inline bool Refract(const V &fixedInput, const double exteriorIOR,
		const double interiorIOR, V *sampled, bool *tir = nullptr) {
	if (tir)
		*tir = false;
	V fixed;
	if (!Unit(fixedInput, &fixed) || !sampled || !(exteriorIOR > 0.) ||
			!(interiorIOR > 0.) || !std::isfinite(exteriorIOR) ||
			!std::isfinite(interiorIOR))
		return false;

	const bool entering = fixed.z > 0.;
	const double eta = entering ? exteriorIOR / interiorIOR :
		interiorIOR / exteriorIOR;
	const double sinIncident2 = std::max(0., fixed.x * fixed.x + fixed.y * fixed.y);
	const double sinTransmitted2 = eta * eta * sinIncident2;
	if (!(sinTransmitted2 < 1.)) {
		if (tir)
			*tir = true;
		return false;
	}

	const double cosTransmitted = std::sqrt(std::max(0., 1. - sinTransmitted2)) *
		(entering ? -1. : 1.);
	return Unit(V(-eta * fixed.x, -eta * fixed.y, cosTransmitted), sampled);
}

struct Proof {
	bool valid = false;
	bool supported = false;
	bool forwardTIR = false;
	bool reverseTIR = false;
	double forwardResidual = -1.;
	double reverseResidual = -1.;
	double combinedResidual = -1.;
};

// Stable reciprocal-support proof:
//  1. compare the actual native Float direction with an independent Double
//     forward construction;
//  2. reverse the unrounded Double construction using the reverse HitPoint's
//     IORs and compare it with the original fixed direction.
//
// Feeding the rounded native direction back into Snell is intentionally
// avoided. At grazing incidence that operation amplifies one Float ULP in the
// transmitted direction into ~1e-4 of reconstructed normal direction.
inline Proof Prove(const V &fixedInput, const V &actualSampledInput,
		const bool transmit, const double forwardExteriorIOR,
		const double forwardInteriorIOR, const double reverseExteriorIOR,
		const double reverseInteriorIOR, const double tolerance = 1.e-4) {
	Proof p;
	V fixed, actualSampled;
	if (!Unit(fixedInput, &fixed) || !Unit(actualSampledInput, &actualSampled) ||
			!(tolerance >= 0.) || !std::isfinite(tolerance))
		return p;

	V exactForward;
	const bool forwardOK = transmit ?
		Refract(fixed, forwardExteriorIOR, forwardInteriorIOR, &exactForward,
			&p.forwardTIR) : Reflect(fixed, &exactForward);
	if (!forwardOK)
		return p;

	V exactReverse;
	const bool reverseOK = transmit ?
		Refract(exactForward, reverseExteriorIOR, reverseInteriorIOR, &exactReverse,
			&p.reverseTIR) : Reflect(exactForward, &exactReverse);
	if (!reverseOK)
		return p;

	p.forwardResidual = Distance(actualSampled, exactForward);
	p.reverseResidual = Distance(exactReverse, fixed);
	p.combinedResidual = std::max(p.forwardResidual, p.reverseResidual);
	p.valid = std::isfinite(p.forwardResidual) && std::isfinite(p.reverseResidual) &&
		std::isfinite(p.combinedResidual);
	p.supported = p.valid && p.combinedResidual <= tolerance;
	return p;
}

} }

#endif

/***************************************************************************
 * ML HERO phase 15my: diagnostic-only Metropolis primary-state digest.    *
 *                                                                         *
 * This helper deliberately knows nothing about rendering or scheduling.   *
 * Equal realized primary-sample vectors therefore produce equal digests   *
 * independent of thread, worker step, ticket and execution order.          *
 ***************************************************************************/

#ifndef _SLG_MLHERO_MLT_SAMPLER_AUDIT15MY_H
#define _SLG_MLHERO_MLT_SAMPLER_AUDIT15MY_H

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace slg {
namespace mlhero15my {

struct PrimaryStateDigest {
	std::uint64_t hash;
	std::uint32_t activeDimensions;
};

inline std::uint64_t Mix64(std::uint64_t x) {
	x ^= x >> 30;
	x *= 0xbf58476d1ce4e5b9ull;
	x ^= x >> 27;
	x *= 0x94d049bb133111ebull;
	return x ^ (x >> 31);
}

inline PrimaryStateDigest Digest(const float *values,
		const unsigned int *stamps, const std::size_t count,
		const std::uint32_t selectedStamp) {
	PrimaryStateDigest result = { 0ull, 0u };
	if (!values || !stamps || selectedStamp == 0u)
		return result;

	std::uint64_t hash = 0x15a17e5eed15a17eull;
	for (std::size_t i = 0; i < count; ++i) {
		if (stamps[i] != selectedStamp)
			continue;

		std::uint32_t bits = 0u;
		static_assert(sizeof(bits) == sizeof(values[i]),
			"15my requires 32-bit float primary samples");
		std::memcpy(&bits, &values[i], sizeof(bits));
		hash = Mix64(hash ^ Mix64(
			(static_cast<std::uint64_t>(i) << 32) | bits));
		++result.activeDimensions;
	}

	if (result.activeDimensions > 0u)
		result.hash = Mix64(hash ^
			(static_cast<std::uint64_t>(result.activeDimensions) *
				0x9e3779b97f4a7c15ull));
	return result;
}

} // namespace mlhero15my
} // namespace slg

#endif // _SLG_MLHERO_MLT_SAMPLER_AUDIT15MY_H

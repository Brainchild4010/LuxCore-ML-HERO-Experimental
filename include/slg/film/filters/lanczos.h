/***************************************************************************
 * LuxCoreRender - Lanczos reconstruction filter
 * SPDX-License-Identifier: Apache-2.0
 ***************************************************************************/
#ifndef _SLG_LANCZOS_FILTER_H
#define _SLG_LANCZOS_FILTER_H

#include <cmath>
#include "luxrays/utils/serializationutils.h"
#include "slg/film/filters/filter.h"

namespace slg {

// Separable Lanczos windowed sinc. Width is the actual kernel support radius.
// The number of lobes controls oscillations within that support.
// With radius=lobes=3 this is the canonical Lanczos-3 kernel.
class LanczosFilter : public Filter {
public:
    LanczosFilter(const float xw, const float yw, const float lobes) :
            Filter(xw, yw), lobes(lobes) { }
    virtual ~LanczosFilter() { }

    virtual FilterType GetType() const { return GetObjectType(); }
    virtual std::string GetTag() const { return GetObjectTag(); }
    virtual float Evaluate(const float x, const float y) const {
        return Lanczos1D(x, xWidth) * Lanczos1D(y, yWidth);
    }
    virtual luxrays::PropertiesUPtr ToProperties() const;

    static FilterType GetObjectType() { return FILTER_LANCZOS; }
    static std::string GetObjectTag() { return "LANCZOS"; }
    static luxrays::PropertiesUPtr ToProperties(const luxrays::Properties &cfg);
    static FilterUPtr FromProperties(const luxrays::Properties &cfg);
    static slg::ocl::Filter *FromPropertiesOCL(const luxrays::Properties &cfg);

    float lobes;
    friend class boost::serialization::access;
private:
    static luxrays::PropertiesUPtr GetDefaultProps();
    LanczosFilter() { }
    template<class Archive> void serialize(Archive &ar, const u_int version) {
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Filter);
        ar & lobes;
    }

    static float NormalizedSinc(const float x) {
        if (fabsf(x) < 1e-5f)
            return 1.f;
        const float pix = static_cast<float>(M_PI) * x;
        return sinf(pix) / pix;
    }
    float Lanczos1D(const float x, const float radius) const {
        const float absX = fabsf(x);
        if (absX >= radius)
            return 0.f;
        // t ranges [0,lobes] across the physical support radius.
        const float t = absX * (lobes / radius);
        return NormalizedSinc(t) * NormalizedSinc(t / lobes);
    }
};

}
BOOST_CLASS_VERSION(slg::LanczosFilter, 1)
BOOST_CLASS_EXPORT_KEY(slg::LanczosFilter)
#endif

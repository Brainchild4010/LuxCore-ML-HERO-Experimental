/***************************************************************************
 * LuxCoreRender - Lanczos reconstruction filter
 * SPDX-License-Identifier: Apache-2.0
 ***************************************************************************/
#include "slg/film/filters/lanczos.h"
#include <memory>
#include <stdexcept>
#include <cmath>

using namespace std;
using namespace luxrays;
using namespace slg;

BOOST_CLASS_EXPORT_IMPLEMENT(slg::LanczosFilter)

PropertiesUPtr LanczosFilter::ToProperties() const {
    auto props = std::make_unique<Properties>();
    *props << Filter::ToProperties() <<
            Property("film.filter.lanczos.lobes")(lobes);
    return props;
}

PropertiesUPtr LanczosFilter::ToProperties(const Properties &cfg) {
    PropertiesUPtr props = std::make_unique<Properties>();
    *props << cfg.Get(GetDefaultProps()->Get("film.filter.type")) <<
            cfg.Get(GetDefaultProps()->Get("film.filter.lanczos.lobes"));
    return props;
}

FilterUPtr LanczosFilter::FromProperties(const Properties &cfg) {
    const float width = cfg.Get(Property("film.filter.width")(3.f)).Get<double>();
    const float xw = cfg.Get(Property("film.filter.xwidth")(width)).Get<double>();
    const float yw = cfg.Get(Property("film.filter.ywidth")(width)).Get<double>();
    const float lobes = cfg.Get(Property("film.filter.lanczos.lobes")(3.f)).Get<double>();

    if (!std::isfinite(xw) || !std::isfinite(yw) || xw <= 0.f || yw <= 0.f)
        throw runtime_error("Lanczos filter radius must be finite and positive");
    if (!std::isfinite(lobes) || lobes < 1.f || lobes > 16.f)
        throw runtime_error("Lanczos filter lobes must be finite and between 1 and 16");
    return std::make_unique<LanczosFilter>(xw, yw, lobes);
}

slg::ocl::Filter *LanczosFilter::FromPropertiesOCL(const Properties &cfg) {
    const float width = cfg.Get(Property("film.filter.width")(3.f)).Get<double>();
    const float xw = cfg.Get(Property("film.filter.xwidth")(width)).Get<double>();
    const float yw = cfg.Get(Property("film.filter.ywidth")(width)).Get<double>();
    // Validate all parameters via the same CPU constructor path.
    FromProperties(cfg);
    slg::ocl::Filter *oclFilter = new slg::ocl::Filter();
    oclFilter->widthX = xw;
    oclFilter->widthY = yw;
    return oclFilter;
}

PropertiesUPtr LanczosFilter::GetDefaultProps() {
    auto props = std::make_unique<Properties>();
    *props << Filter::GetDefaultProps() <<
            Property("film.filter.type")(GetObjectTag()) <<
            Property("film.filter.width")(3.f) <<
            Property("film.filter.lanczos.lobes")(3.f);
    return props;
}

/***************************************************************************
 * Reinhard Hero Exposure - ML HERO experimental extension                 *
 ***************************************************************************/
#ifndef _SLG_REINHARDHEROEXPOSURE_TONEMAP_H
#define _SLG_REINHARDHEROEXPOSURE_TONEMAP_H

#include "luxrays/core/hardwaredevice.h"
#include "luxrays/utils/serializationutils.h"
#include "slg/film/imagepipeline/plugins/tonemaps/tonemap.h"

namespace slg {
class ReinhardHeroExposureToneMap : public ToneMap {
public:
    ReinhardHeroExposureToneMap();
    ReinhardHeroExposureToneMap(const float ev, const float b);
    virtual ~ReinhardHeroExposureToneMap();
    virtual ToneMapType GetType() const { return TONEMAP_REINHARD_HERO_EXPOSURE; }
    virtual ToneMap *Copy() const { return new ReinhardHeroExposureToneMap(exposureBias, burn); }
    virtual void Apply(Film &film, const u_int index);
    virtual bool CanUseHW() const { return true; }
    virtual void AddHWChannelsUsed(Film::FilmChannels &hwChannelsUsed) const;
    virtual void ApplyHW(Film &film, const u_int index);
    float exposureBias, burn;
    friend class boost::serialization::access;
private:
    template<class Archive> void serialize(Archive &ar, const u_int version) {
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ToneMap); ar & exposureBias; ar & burn;
    }
    luxrays::HardwareDevice *hardwareDevice;
    luxrays::HardwareDeviceBuffer *hwAccumBuffer;
    luxrays::HardwareDeviceKernel *opRGBValuesReduceKernel, *opRGBValueAccumulateKernel, *applyKernel;
};
}
BOOST_CLASS_VERSION(slg::ReinhardHeroExposureToneMap, 1)
BOOST_CLASS_EXPORT_KEY(slg::ReinhardHeroExposureToneMap)
#endif

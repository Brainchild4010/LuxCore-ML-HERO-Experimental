/***************************************************************************
 * Reinhard Hero tone mapping - ML HERO experimental extension             *
 ***************************************************************************/

#ifndef _SLG_REINHARDHERO_TONEMAP_H
#define _SLG_REINHARDHERO_TONEMAP_H

#include <cmath>
#include <string>

#include "luxrays/core/hardwaredevice.h"
#include "luxrays/utils/serializationutils.h"
#include "slg/film/imagepipeline/plugins/tonemaps/tonemap.h"

namespace slg {

class ReinhardHeroToneMap : public ToneMap {
public:
    ReinhardHeroToneMap();
    ReinhardHeroToneMap(const float preS, const float postS, const float b);
    virtual ~ReinhardHeroToneMap();

    virtual ToneMapType GetType() const { return TONEMAP_REINHARD_HERO; }

    virtual ToneMap *Copy() const {
        return new ReinhardHeroToneMap(preScale, postScale, burn);
    }

    virtual void Apply(Film &film, const u_int index);

    virtual bool CanUseHW() const { return true; }
    virtual void AddHWChannelsUsed(Film::FilmChannels &hwChannelsUsed) const;
    virtual void ApplyHW(Film &film, const u_int index);

    float preScale, postScale, burn;

    friend class boost::serialization::access;

private:
    template<class Archive> void serialize(Archive &ar, const u_int version) {
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ToneMap);
        ar & preScale;
        ar & postScale;
        ar & burn;
    }

    luxrays::HardwareDevice *hardwareDevice;
    luxrays::HardwareDeviceBuffer *hwAccumBuffer;
    luxrays::HardwareDeviceKernel *opRGBValuesReduceKernel;
    luxrays::HardwareDeviceKernel *opRGBValueAccumulateKernel;
    luxrays::HardwareDeviceKernel *applyKernel;
};

}

BOOST_CLASS_VERSION(slg::ReinhardHeroToneMap, 1)
BOOST_CLASS_EXPORT_KEY(slg::ReinhardHeroToneMap)

#endif

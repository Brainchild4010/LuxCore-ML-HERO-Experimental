/***************************************************************************
 * Reinhard Hero Classic - ML HERO experimental extension                  *
 ***************************************************************************/
#ifndef _SLG_REINHARDHEROCLASSIC_TONEMAP_H
#define _SLG_REINHARDHEROCLASSIC_TONEMAP_H

#include "luxrays/core/hardwaredevice.h"
#include "luxrays/utils/serializationutils.h"
#include "slg/film/imagepipeline/plugins/tonemaps/tonemap.h"

namespace slg {
class ReinhardHeroClassicToneMap : public ToneMap {
public:
    ReinhardHeroClassicToneMap();
    ReinhardHeroClassicToneMap(const float k, const float w);
    virtual ~ReinhardHeroClassicToneMap();
    virtual ToneMapType GetType() const { return TONEMAP_REINHARD_HERO_CLASSIC; }
    virtual ToneMap *Copy() const { return new ReinhardHeroClassicToneMap(key, whitePoint); }
    virtual void Apply(Film &film, const u_int index);
    virtual bool CanUseHW() const { return true; }
    virtual void AddHWChannelsUsed(Film::FilmChannels &hwChannelsUsed) const;
    virtual void ApplyHW(Film &film, const u_int index);
    float key, whitePoint;
    friend class boost::serialization::access;
private:
    template<class Archive> void serialize(Archive &ar, const u_int version) {
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ToneMap); ar & key; ar & whitePoint;
    }
    luxrays::HardwareDevice *hardwareDevice;
    luxrays::HardwareDeviceBuffer *hwAccumBuffer;
    luxrays::HardwareDeviceKernel *opRGBValuesReduceKernel, *opRGBValueAccumulateKernel, *applyKernel;
};
}
BOOST_CLASS_VERSION(slg::ReinhardHeroClassicToneMap, 1)
BOOST_CLASS_EXPORT_KEY(slg::ReinhardHeroClassicToneMap)
#endif

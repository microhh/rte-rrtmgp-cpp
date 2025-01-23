#ifndef RAYTRACER_DEFINITIONS_H
#define RAYTRACER_DEFINITIONS_H

#include "types.h"


namespace Raytracer_definitions
{
    template<typename T>
    struct Vector
    {
        T x;
        T y;
        T z;
    };

    struct Optics_scat
    {
        Float k_sca_gas;
        Float k_sca_cld;
        Float k_sca_aer;
        Float asy_cld;
        Float asy_aer;
    };

    enum class Photon_kind { Direct, Diffuse };
    enum class Photon_cloud_status { no_cld, cld };

    struct Photon
    {
        Vector<Float>position;
        Vector<Float>direction;
        Photon_kind kind;
        Photon_cloud_status cloud_status;
    };
}
#endif

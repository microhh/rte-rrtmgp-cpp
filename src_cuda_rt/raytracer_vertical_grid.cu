#include <algorithm>
#include <cmath>

#include "raytracer_vertical_grid.h"


Vertical_grid_gpu build_vertical_grid(const Array<Float,1>& z_lev, const int nz, const int kn_z)
{
    Array<Float,1> kn_z_lev({kn_z+1});
    const Float fz = Float(nz) / Float(kn_z);
    for (int k=0; k<kn_z; ++k)
        kn_z_lev({k+1}) = z_lev({static_cast<int>(k*fz)+1});
    kn_z_lev({kn_z+1}) = z_lev({nz+1});

    Float dz_min = z_lev({2}) - z_lev({1});
    for (int k=1; k<nz; ++k)
        dz_min = std::min(dz_min, z_lev({k+2}) - z_lev({k+1}));

    const Float zsize = z_lev({nz+1});
    const int lut_size = static_cast<int>(std::ceil(zsize/dz_min));
    const Float lut_dz = zsize / lut_size;

    Array<int,1> z_lut({lut_size});
    Array<int,1> kn_z_lut({lut_size});
    for (int i=0; i<lut_size; ++i)
    {
        const Float z = i*lut_dz;
        int k = 0;
        while (k < nz-1 && z >= z_lev({k+2}))
            ++k;
        z_lut({i+1}) = k;
        k = 0;
        while (k < kn_z-1 && z >= kn_z_lev({k+2}))
            ++k;
        kn_z_lut({i+1}) = k;
    }

    Array<Float,1> z_lev_rt({nz+1});
    for (int k=1; k<=nz+1; ++k)
        z_lev_rt({k}) = z_lev({k});

    return {Array_gpu<Float,1>(z_lev_rt), Array_gpu<Float,1>(kn_z_lev),
            Array_gpu<int,1>(z_lut), Array_gpu<int,1>(kn_z_lut), lut_dz, zsize};
}

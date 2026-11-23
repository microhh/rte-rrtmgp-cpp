#ifndef RAYTRACER_VERTICAL_GRID_H
#define RAYTRACER_VERTICAL_GRID_H

#include "types.h"
#include "array.h"

#ifdef __CUDACC__

struct Vertical_grid_gpu
{
    Array_gpu<Float,1> z_lev;
    Array_gpu<Float,1> kn_z_lev;
    Array_gpu<int,1> z_lut;
    Array_gpu<int,1> kn_z_lut;
    Float lut_dz;
    Float zsize;
};


Vertical_grid_gpu build_vertical_grid(const Array<Float,1>& z_lev, const int nz, const int kn_z);
#endif

#endif

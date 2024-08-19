#ifndef RAYTRACER_KERNELS_BW_H
#define RAYTRACER_KERNELS_BW_H

#include <curand_kernel.h>

#include "types.h"
#include "raytracer_definitions.h"
#include "raytracer_functions.h"


// CvH TODO THIS SHOULD GO IN A NAMESPACE, CHECK WITH MENNO.

using Raytracer_definitions::Vector;
using Raytracer_definitions::Optics_scat;
using namespace Raytracer_functions;



#ifdef RTE_USE_SP
constexpr int bw_kernel_block= 512;
constexpr int bw_kernel_grid = 1024;
#else
constexpr int bw_kernel_block = 256;
constexpr int bw_kernel_grid = 256;
#endif
constexpr Float k_null_gas_min = Float(1.e-3);

// sun has a half angle of .266 degrees
constexpr Float cos_half_angle = Float(0.9999891776066407); // cos(half_angle);
constexpr Float sun_solid_angle = Float(6.799910294339209e-05); // 2.*M_PI*(1-cos_half_angle);
constexpr Float sun_solid_angle_reciprocal = Float(14706.07635563193); 


struct Grid_knull
{
    Float k_max;
    Float k_min;
};


struct Camera
{
    Vector<Float> position;
    bool fisheye = true;

    // rotation matrix for fisheye version - we need to do implement this in a nice way at some point
    Vector<Float> mx;
    Vector<Float> my;
    Vector<Float> mz;
    Float f_zoom;

    // regular camera
    Float fov;
    Vector<Float> cam_width;
    Vector<Float> cam_height;
    Vector<Float> cam_depth;

    void setup_rotation_matrix(const Float yaw_deg, const Float pitch_deg, const Float roll_deg)
    {
        const Float yaw = yaw_deg / Float(180.) * M_PI;
        const Float pitch = pitch_deg / Float(180.) * M_PI;
        const Float roll = roll_deg / Float(180.) * M_PI;
        mx = {cos(yaw)*sin(pitch), (cos(yaw)*cos(pitch)*sin(roll)-sin(yaw)*cos(roll)), (cos(yaw)*cos(pitch)*cos(roll)+sin(yaw)*sin(roll))};
        my = {sin(yaw)*sin(pitch), (sin(yaw)*cos(pitch)*sin(roll)+cos(yaw)*cos(roll)), (sin(yaw)*cos(pitch)*cos(roll)-cos(yaw)*sin(roll))};
        mz = {-cos(pitch), sin(pitch)*sin(roll), sin(pitch)*cos(roll)};
    }

    void setup_normal_camera(const Camera camera)
    {
        if (!fisheye)
        {
            const Vector<Float> dir_tmp = {0, 0, 1};
            const Vector<Float> dir_cam = normalize(Vector<Float>({dot(camera.mx,dir_tmp), dot(camera.my,dir_tmp), dot(camera.mz,dir_tmp)*Float(-1)}));
            Vector<Float> dir_up;
            if ( (int(dir_cam.z)==1) || (int(dir_cam.z)==-1) )
                dir_up = {1, 0, 0};
            else
                dir_up = {0, 0, 1};

            cam_width = normalize(cross(dir_cam, dir_up));
            cam_height = normalize(cross(dir_cam, cam_width));
            cam_depth = dir_cam / tan(fov/Float(180)*M_PI/Float(2.));
        
            if (camera.nx > camera.ny)
                cam_height = cam_height * Float(camera.ny)/Float(camera.nx);
            else if (camera.ny > camera.nx)
                cam_width = cam_width * Float(camera.nx)/Float(camera.ny);
        }
    }

    // size of output arrays, either number of horizontal and vertical pixels, or number of zenith/azimuth angles of fisheye lens. Default to 1024
    int ny = 1024;
    int nx = 1024;
};


__global__
void ray_tracer_kernel_bw(
        const int igpt,
        const int photons_per_pixel,
        const Grid_knull* __restrict__ k_null_grid,
        Float* __restrict__ camera_count,
        Float* __restrict__ camera_shot,
        int* __restrict__ counter,
        const Float* __restrict__ k_ext, const Optics_scat* __restrict__ scat_asy,
        const Float* __restrict__ k_ext_bg, const Optics_scat* __restrict__ scat_asy_bg,
        const Float* __restrict__ z_lev_bg,
        const Float* __restrict__ r_eff,
        const Float* __restrict__ surface_albedo,
        const Float* __restrict__ land_use_map,
        const Float mu,
        const Vector<Float> grid_size,
        const Vector<Float> grid_d,
        const Vector<int> grid_cells,
        const Vector<int> kn_grid,
        const Vector<Float> sun_direction,
        const Camera camera,
        const int nbg,
        const Float* __restrict__ mie_cdf,
        const Float* __restrict__ mie_ang,
        const Float* __restrict__ mie_phase,
        const Float* __restrict__ mie_phase_ang,
        const int mie_table_size);
            
__global__
void accumulate_clouds_kernel(
    const Float* __restrict__ lwp, 
    const Float* __restrict__ iwp, 
    const Float* __restrict__ tau_cloud,
    const Vector<Float> grid_d,
    const Vector<Float> grid_size,
    const Vector<int> grid_cells,
    Float* __restrict__ liwp_cam, 
    Float* __restrict__ tauc_cam, 
    Float* __restrict__ dist_cam, 
    Float* __restrict__ zen_cam, 
    const Camera camera);

#endif

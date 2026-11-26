#ifndef RAYTRACER_CAMERA_H
#define RAYTRACER_CAMERA_H

#include <cmath>

#include "types.h"
#include "raytracer_definitions.h"
#include "raytracer_functions.h"

// Camera definition shared by the backward ray tracers (shortwave and longwave)

using Raytracer_definitions::Vector;
using namespace Raytracer_functions;

struct Camera
{
    Vector<Float> position;
    int cam_type; // (0: fisheye, 1: rectangular, 2: top-of-atmosphere radiances)

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
        mx = {cos(yaw)*cos(pitch), (cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll)), (cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll))};
        my = {sin(yaw)*cos(pitch), (sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll)), (sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll))};
        mz = {-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)};
    }

    void setup_normal_camera(const Camera camera)
    {
        if (camera.cam_type != 0)
        {
            cam_height = Vector<Float>({camera.mx.z, camera.my.z, camera.mz.z});
            cam_depth = Vector<Float>({camera.mx.x, camera.my.x, camera.mz.x});
            cam_width = Float(-1) * normalize(cross(cam_depth, cam_height));

            cam_depth = cam_depth / tan(fov/Float(180)*M_PI/Float(2.));

            if (camera.nx > camera.ny)
                cam_height = cam_height * Float(camera.ny)/Float(camera.nx);
            else if (camera.ny > camera.nx)
                cam_width = cam_width * Float(camera.nx)/Float(camera.ny);
        }
    }

    // size of output arrays, either number of horizontal and vertical pixels, or number of zenith/azimuth angles of fisheye lens. Default to 1024
    int ny = 1024;
    int nx = 1024;
    Int npix;
};

#endif

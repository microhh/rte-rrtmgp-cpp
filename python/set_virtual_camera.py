# Add/modify the "camera-settings" group to/of the ray tracer input file (rte_rrtmgp_input.nc) to set the location, direction, and viewing angles of the virtual camera of the backward ray tracer implementation. Solar zenith and azimuth angles can also be set with this script.
#
# Usage: python set_virtual_camera.py --yaw -90 --pitch 0 --roll 90 --fisheye 0 --fov 100 --px 0 --py 0 --pz 1000 --nx 512 --ny 512
#
# Two presets are available:
#     1) 'radiance' (python set_virtual_camera.py --radiance): virtual sky-view camera with 180 degrees field of view located at the surface looking upwards, e.g. to compute hemispheric radiance distributions. To do so, also use the '--broadband flag' when running the backward ray tracer.
#
#     2) 'image' (python set_virtual_camera.py --image): square (but currently only rectangular) camera with 80 degrees field of view located at 500 m altitude looking horizontally towards the east.
#

import netCDF4 as nc
import numpy as np
import argparse

camera_variables = {
    "yaw": "Horizontal direction of camera, 0: east, 90: north, 180/-180: weast, -90/270: south",
    "pitch": "Vertical direction of camera, -90: vertical upward, 0: horizontal, 90: vertically downward.",
    "roll": "Roll of camera over the direction of camera",
    "fisheye": "Whether to use fisheye lens (1) or a rectangular/square lens (0)",
    "f_zoom": "Zoom factor (if fisheye=1)",
    "fov": "Field of view (if fisheye=0)",
    "px": "Location of camera in x-direction",
    "py": "Location of camera in y-direction",
    "pz": "Location of camera in z-direction",
    "nx": "Number of camera pixels in horizontal direction (fisheye=0) or number of camera zenith angles (fisheye=1)",
    "ny": "Number of camera pixels in vertical direction (fisheye=0) or number of camera azimuth angles (fisheye=1)",
    }

parser = argparse.ArgumentParser()
parser.add_argument("--radiance", action='store_true', help="example settings for computing hemispheric radiance distributions with a sky view camera")
parser.add_argument("--image", action='store_true', help="example settings for creating visual images with a square camera looking horizontally")
parser.add_argument("--sza", type=float, help="solar zenith angle")
parser.add_argument("--azi", type=float, help="solar azimuth angle")
parser.add_argument("--name", type=str, default='rte_rrtmgp_input.nc', help="raytracer input file")

for var in camera_variables.keys():
    parser.add_argument("--{}".format(var), type=float, help=camera_variables[var])

args = vars(parser.parse_args())

# open netcdf file
ncf = nc.Dataset(args['name'],"r+")

# add group if it does not exsist yet
if not "camera-settings" in ncf.groups:
    cam = ncf.createGroup("camera-settings")
else:
    cam = ncf.groups['camera-settings']

# add variables: if not available yet
for var in camera_variables:
    try:
        cam.createVariable(var,"f8")
    except:
        pass

# example radiance settings
if args['radiance']:
    cam["yaw"][:]   = 0
    cam["pitch"][:] = -90
    cam["roll"][:]  = 0
    cam["fisheye"][:]= 1
    cam["f_zoom"][:]= 1
    cam["px"][:] = 0
    cam["py"][:] = 0
    cam["pz"][:] = 0
    cam["nx"][:] = 1024
    cam["ny"][:] = 1024

# example imagery settings
if args['image']:
    cam["yaw"][:] = 0
    cam["pitch"][:] = 0
    cam["roll"][:]  = 0
    cam["fisheye"][:]= 0
    cam["fov"][:] = 80
    cam["px"][:] = 0.
    cam["py"][:] = 0.
    cam["pz"][:] = 500.
    cam["nx"][:] = 1024
    cam["ny"][:] = 1024

for var in camera_variables:
    if not args[var] is None:
        cam[var][:] = args[var]

if not args['sza'] is None:
    ncf['mu0'][:] = np.cos(np.deg2rad(args['sza']))
if not args['azi'] is None:
    ncf['azi'][:] = np.deg2rad(args['azi'])

print("Camera settings:")
for v in camera_variables.keys():
    print("{:6}{:>8}".format(v, str(cam[v][:])))
print("{:6}{:>8}".format("sza", str(np.round(np.rad2deg(np.arccos(ncf['mu0'][0,0])),1))))
print("{:6}{:>8}".format("azi", str(np.round(np.rad2deg(ncf['azi'][0,0]),1))))

ncf.close()

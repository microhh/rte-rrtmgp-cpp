# Convert XYZ tristimulus values from backward ray tracer to RGB values

import matplotlib.pyplot as pl
import numpy as np
from PIL import Image
import netCDF4 as nc
import argparse
import re
import colour # pip install colour-science

sRGB = colour.RGB_COLOURSPACES['sRGB']

parser = argparse.ArgumentParser()
parser.add_argument('--name',default='rte_rrtmgp_output.nc', help="Raytracer output file")
parser.add_argument('--fisheye', action='store_true', help="If true: output is on a radial grid (zenith,azimuth), else: output is on a rectangular/square grid")
parser.add_argument('--save_to_file', action='store_true', help="If true: image is saved to 'image.png', otherwise it is shown")
parser.add_argument('--p_norm', default=98, type=float, help="Percentile of luminance to use for luminance normalization, defaults to 98")
parser.add_argument('--dpi',   default=300, type=int, help="dots-per-inch to determine figure size based on number of pixels")
parser.add_argument('--illuminant', default="E", help="Assumed original illuminant of XYZ values, which are transformed to D65 corresponding to sRGB. Defaults to E [1/3,1/3]. '--illuminant ?' prints all options ")
parser.add_argument('--cat', default="CAT02", help="Chromatic adaptation transform, defaults to CAT02. '--cat ?' prints all options")
args = parser.parse_args()

if args.illuminant == "?":
    print("\nIlluminants:")
    for key in list(colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'].keys()):
        print("{},".format(key), 'white point (x,y):', *colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][key])

if args.cat == "None":
    args.cat = None

if args.cat == "?":
    print("\nChromatic adaptation transformations:")
    for options in re.findall(r"'\s*([^']*?)\s*'", colour.adaptation.matrix_chromatic_adaptation_VonKries.__annotations__['transform']):
        print(options)

# Open ray traacer output file
ncf  = nc.Dataset(args.name)
nx = ncf.dimensions['x'].size
ny = ncf.dimensions['y'].size

# Load XYZ values
XYZ = ncf['XYZ'][:].reshape((3, nx*ny))

# Illuminance normalization
lum_norm = np.percentile(XYZ[1], args.p_norm)
XYZ[:] /= lum_norm

# Convert to RGB
illuminant = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][args.illuminant]
RGB = colour.XYZ_to_RGB(XYZ.T, colourspace=sRGB, illuminant=illuminant, chromatic_adaptation_transform=args.cat, apply_cctf_encoding=True).T
RGB = np.minimum(1, np.maximum(0, RGB))

if args.fisheye:
    # radial pixel coordinates
    az = np.deg2rad(np.linspace(0,360,ny+1))
    th = np.linspace(0, 0.5*np.pi,nx+1)
    r  = np.tan(th/2)

    # reshape RGB array
    RGB = RGB.reshape((3, ny, nx))
    RGB = RGB.swapaxes(0,2)

    # plot
    fig,ax = pl.subplots(figsize=(nx/args.dpi, ny/args.dpi), frameon=False, subplot_kw={'projection':'polar'})
    ax.grid(False)
    ax.pcolormesh(az[::], r, RGB,rasterized=True)

else:
    # pixel coordinates
    px = np.linspace(0,1,nx+1)
    py = np.linspace(0,1,ny+1)

    # reshape RGB array
    RGB = RGB.reshape((3, ny, nx))
    RGB = RGB.swapaxes(0,2).swapaxes(0,1)[::-1,:]
    
    # plot
    fig,ax = pl.subplots(figsize=(nx/args.dpi, ny/args.dpi), frameon=False)
    ax.grid(False)
    ax.pcolormesh(px, py, RGB, rasterized=True)

pl.subplots_adjust(left=0,right=1,top=1,bottom=0)
ax.set_yticks([])
ax.set_xticklabels([])

if args.save_to_file:
    pl.savefig("image.png", transparent=True, dpi=args.dpi)
else:
    pl.show()


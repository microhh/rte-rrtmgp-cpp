import numpy as np
import netCDF4 as nc
nz=176
nzh=177
oz = 176
ozh= 177
Nx=240
Ny=240
nx=72
ny=72
dx = 100
path = "./"
time = 21600
name = "{:07d}".format(time) 

azi = np.deg2rad(137)

qt = np.fromfile(path+"qt.%s"%name).reshape(nz,Ny,Nx)[:oz,:ny,:nx]
ql = np.fromfile(path+"ql.%s"%name).reshape(nz,Ny,Nx)[:oz,:ny,:nx]
qv = qt - ql

tlay = np.fromfile(path+"T.%s"%name).reshape(nz,Ny,Nx)[:oz,:ny,:nx]
tlev = np.zeros((nz+1,ny,nx))
tlev[1:-1] = (tlay[1:]+tlay[:-1])/2.
tlev[0] = 2*tlay[0]-tlev[1]
tlev[-1] = 2*tlay[-1]-tlev[-2]

ncstats =  nc.Dataset(path+"cabauw.default.0018000.nc")
z = ncstats.variables['z'][:oz]
zh = ncstats.variables['zh'][:ozh]
Play = ncstats.groups['thermo'].variables["phydro"][0][:oz]
Plev = ncstats.groups['thermo'].variables["phydroh"][0][:ozh]
qv = qv/(0.622-qv)

radidx = abs(ncstats.variables['time'][:] - time).argmin()

sza = ncstats.groups['radiation'].variables['sza'][radidx]
print(sza,np.rad2deg(sza))
layermass = np.abs((Plev[1:]-Plev[:-1]))/9.81
lwp = ql*layermass[:,np.newaxis,np.newaxis]*1e3

iwp = np.zeros(lwp.shape)
#pdx = np.abs(ncstats.variables['z'][:] - 8e3).argmin()
#iwp[pdx,:,:] = 5e-6*layermass[pdx] 

#rel/rei
Nc0 = 100e6
Ni0 = 1e5
fpnrw = 4./3*np.pi*Nc0*1000
fpnri = 4./3*np.pi*Ni0*1000
sig = 1.34
fac = np.exp(np.log(sig)*np.log(sig))*1e6
rel = fac*((lwp/(zh[1:]-zh[:-1])[:,np.newaxis,np.newaxis])/fpnrw)**(1./3)
rei = fac*((iwp/(zh[1:]-zh[:-1])[:,np.newaxis,np.newaxis])/fpnri)**(1./3)
rel = np.maximum(2.5, np.minimum(rel, 21.5))
rei = np.maximum(10 , np.minimum(rei, 180))
rel = np.where(lwp==0,0,rel)
rei = np.where(iwp==0,0,rei)

###write output
nc_infile  = nc.Dataset("rte_rrtmgp_input.nc", mode="w", datamodel="NETCDf8")
x = dx/2. + np.arange(nx)*dx
y = dx/2. + np.arange(ny)*dx
xh = np.arange(nx+1)*dx
yh = np.arange(nx+1)*dx
ncstats.close()
print(qv.min(),qv.max())
#dimensions:
for nc_file in [nc_infile]:
    nc_file.createDimension("lay",len(z))
    nc_file.createDimension("lev",len(zh))
    nc_file.createDimension("y",len(y))
    nc_file.createDimension("x",len(x))
    nc_file.createDimension("yh",len(yh))
    nc_file.createDimension("xh",len(xh))
    nc_file.createDimension("band_sw",14)
    nc_file.createDimension("band_lw",16)
    zvar = nc_file.createVariable("z", 'f8', ("lay",))
    zvar[:] = z
    zhvar = nc_file.createVariable("zh", 'f8', ("lev",))
    zhvar[:] = zh
    yvar = nc_file.createVariable("yh", 'f8', ("yh",))
    yvar[:] = yh
    xvar = nc_file.createVariable("xh", 'f8', ("xh",))
    xvar[:] = xh

co2 = 400e-6
o3 = 1e-10
ch4 = 1800e-9
o2 = 0.21
n2 = 0.78
n2o= 1.e-7
nc_co2 = nc_infile.createVariable('vmr_co2', "f8")
nc_co2[:] = co2
nc_ch4 = nc_infile.createVariable('vmr_ch4', "f8")
nc_ch4[:] = ch4
nc_o2 = nc_infile.createVariable('vmr_o2', "f8")
nc_o2[:] = o2
nc_o3 = nc_infile.createVariable('vmr_o3', "f8")
nc_o3[:] = o3
nc_n2 = nc_infile.createVariable('vmr_n2', "f8")
nc_n2[:] = n2
nc_n2o = nc_infile.createVariable('vmr_n2o', "f8")
nc_n2o[:] = n2o

#input fields
nc_qv = nc_infile.createVariable('vmr_h2o', "f8", ("lay","y","x"))
nc_qv[:] = qv
nc_tlay = nc_infile.createVariable('t_lay', "f8", ("lay","y","x"))
nc_tlay[:] = tlay
nc_tlev = nc_infile.createVariable('t_lev', "f8", ("lev","y","x"))
nc_tlev[:] = tlev

nc_iwp = nc_infile.createVariable('iwp', "f8", ("lay","y","x"))
nc_iwp[:] = iwp
nc_lwp = nc_infile.createVariable('lwp', "f8", ("lay","y","x"))
nc_lwp[:] = lwp
nc_rel = nc_infile.createVariable('rel', "f8", ("lay","y","x"))
nc_rel[:] = rel
nc_rei = nc_infile.createVariable('rei', "f8", ("lay","y","x"))
nc_rei[:] = rei

nc_Play = nc_infile.createVariable('p_lay', "f8", ("lay","y","x"))
nc_Play[:] = Play[:,np.newaxis,np.newaxis]
nc_Plev = nc_infile.createVariable('p_lev', "f8", ("lev","y","x"))
nc_Plev[:] = Plev[:,np.newaxis,np.newaxis]

nc_sza= nc_infile.createVariable('sza', "f8")
nc_sza[:] = sza
nc_azi = nc_infile.createVariable('azi', "f8")
nc_azi[:] = azi
nc_tsfc = nc_infile.createVariable('t_sfc', "f8", ("y","x"))
nc_tsfc[:] = tlev[0]
nc_emis = nc_infile.createVariable('emis_sfc', "f8", ("y","x","band_lw"))
nc_emis[:] = .98
nc_albdir = nc_infile.createVariable('sfc_alb_dir', "f8", ("y","x","band_sw"))
nc_albdir[:] = .07
nc_albdif = nc_infile.createVariable('sfc_alb_dif', "f8", ("y","x","band_sw"))
nc_albdif[:] = .07

nc_infile.close()


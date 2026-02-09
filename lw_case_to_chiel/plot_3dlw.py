import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc

data = nc.Dataset("rte_rrtmgp_output.nc")
# ref = nc.Dataset("ref.nc")

z = data.variables["z"][:]
dz = z[1] - z[0]

hr_1d = (data.variables["lw_flux_net"][1:len(z)+1] - data.variables["lw_flux_net"][:len(z)]) / dz
hr_3d = data.variables["rt_lw_flux_abs"][:, :, :]
# hr_1d_ref = (ref.variables["lw_flux_net"][1:len(z)+1] - ref.variables["lw_flux_net"][:len(z)]) / dz

plt.figure()
plt.plot(hr_1d.mean(axis=(1, 2)), z)
plt.plot(hr_3d.mean(axis=(1, 2)), z)
# plt.plot(hr_1d_ref.mean(axis=(1, 2)), z, 'k:')
plt.show()

tod_up = data.variables["rt_lw_flux_tod_up"][:, :].mean()
tod_dn = data.variables["rt_lw_flux_tod_dn"][:, :].mean()
sfc_up = data.variables["rt_lw_flux_sfc_up"][:, :].mean()
sfc_dn = data.variables["rt_lw_flux_sfc_dn"][:, :].mean()
hr = (data.variables["rt_lw_flux_abs"][:, :, :].mean(axis=(1, 2)) * dz).sum()

print("Energy balance")
print(tod_up, tod_dn, sfc_up, sfc_dn, hr)
print(tod_dn - tod_up, sfc_up - sfc_dn, hr)
print((tod_dn - tod_up) + (sfc_up - sfc_dn), hr)
print((tod_dn - tod_up) + (sfc_up - sfc_dn) - hr)

print("Surface balance 1d vs 3d (toa_up, toa_dn, sfc_up, sfc_dn")
print('3d', tod_up, tod_dn, sfc_up, sfc_dn)

tod_up = data.variables["lw_flux_up"][len(z), :, :].mean()
tod_dn = data.variables["lw_flux_dn"][len(z), :, :].mean()
sfc_up = data.variables["lw_flux_up"][0, :, :].mean()
sfc_dn = data.variables["lw_flux_dn"][0, :, :].mean()
print('1d', tod_up, tod_dn, sfc_up, sfc_dn)

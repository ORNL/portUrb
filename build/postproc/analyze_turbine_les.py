from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

# windfarm_0.000010_wind-19.000000_turbine_00000006.nc
z0      = 1.e-5
workdir = "/lustre/storm/nwp501/scratch/imn/turbine_les_param/z0_1e-5"
prefix  = "windfarm"
times   = range(13,25)
winds   = range(3,26,2)
files_p = {f"{wind}_{time}": f"{workdir}/{prefix}_{z0:.6f}_wind-{wind:.6f}_precursor_{time:08d}.nc" for time in times for wind in winds}
files_n = {f"{wind}_{time}": f"{workdir}/{prefix}_{z0:.6f}_wind-{wind:.6f}_noturbine_{time:08d}.nc" for time in times for wind in winds}
files_t = {f"{wind}_{time}": f"{workdir}/{prefix}_{z0:.6f}_wind-{wind:.6f}_turbine_{time:08d}.nc"   for time in times for wind in winds}
xmin = 2
xmax = 0
z = np.array(Dataset(files_n[f"{winds[0]}_{times[0]}"],"r")["z"][:])
ratios = np.array([[0. for i in range(len(z))] for j in range(len(winds))])
for iwind in range(len(winds)) :
    wind = winds[iwind]
    for itime in range(len(times)) :
        time  = times[itime]
        nc    = Dataset(files_n[f"{wind}_{time}"],"r")
        u     = np.array(nc["avg_u"][:,:,:])
        v     = np.array(nc["avg_v"][:,:,:])
        mag   = np.sqrt(u*u+v*v)
        mag_n = mag if time==times[0] else mag_n+mag

        nc    = Dataset(files_t[f"{wind}_{time}"],"r")
        u     = np.array(nc["avg_u"][:,:,:])
        v     = np.array(nc["avg_v"][:,:,:])
        mag   = np.sqrt(u*u+v*v)
        mag_t = mag if time==times[0] else mag_t+mag
    mag_n /= len(times)
    mag_t /= len(times)
    x = np.array(nc["x"][:])
    y = np.array(nc["y"][:])
    z = np.array(nc["z"][:])
    xlen = x[-1]+(x[1]-x[0])/2
    ylen = y[-1]+(y[1]-y[0])/2

    ratio_dom = np.mean(mag_t,axis=(1,2))/np.mean(mag_n,axis=(1,2))
    ratio = 1*(10000**2-xlen*ylen)/(10000**2) + ratio_dom*(xlen*ylen)/(10000**2)
    ratios[iwind,:] = ratio
    # xmin = min(np.min(ratio),xmin)
    # xmax = max(np.max(ratio),xmax)
    # plt.plot(ratio,z,label=f"Inflow={wind}m/s")
# plt.fill_between([xmin,xmax],90-63,90+63,color="lightskyblue")
# plt.legend()
# plt.xlabel(r"$\sqrt{u*u+v*v}$ no_turbine / turbine ratio")
# plt.ylabel(r"Height (m)")
# plt.grid()
# plt.show()
# plt.close()

nc = Dataset(f"ratios_{z0:.6f}.nc","w",format='NETCDF4')
nc.createDimension('z'   ,len(z    ))
nc.createDimension('wind',len(winds))
nc.createVariable('z'     ,'f4',('z'       ))[:]   = z     [:]
nc.createVariable('wind'  ,'f4',('wind'    ))[:]   = winds [:]
nc.createVariable('ratios','f4',('wind','z'))[:,:] = ratios[:,:]
nc.close()


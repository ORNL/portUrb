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

for iwind in range(len(winds)) :
    wind = winds[iwind]
    for itime in range(len(times)) :
        time  = times[itime]
        nc    = Dataset(files_t[f"{wind}_{time}"],"r")
        u     = np.array(nc["mag_trace_turb_0"][:])
        mag   = u if time==times[0] else np.concatenate((mag,u))
    counts,bins = np.histogram(mag,bins=50)
    counts = counts / np.sum(counts)
    plt.stairs(counts,bins,fill=True)
    mbins = (bins[:len(bins)-1]+bins[1:])/2
    m = np.mean(mag)
    s = np.std(mag)
    val = 1/(s*np.sqrt(2*np.pi))*np.exp(-0.5*((mbins-m)/s)**2)
    val = val / np.sum(val)
    print(np.mean(np.abs(val-counts)))
    plt.plot(mbins,val)
    plt.title(f"{winds[iwind]}")
    plt.show()
    plt.close()

z = np.array(Dataset(files_n[f"{winds[0]}_{times[0]}"],"r")["z"][:])
noturb_u = np.array([[0. for i in range(len(z))] for j in range(len(winds))])
noturb_v = np.array([[0. for i in range(len(z))] for j in range(len(winds))])
turb_u   = np.array([[0. for i in range(len(z))] for j in range(len(winds))])
turb_v   = np.array([[0. for i in range(len(z))] for j in range(len(winds))])
for iwind in range(len(winds)) :
    wind = winds[iwind]
    for itime in range(len(times)) :
        time  = times[itime]
        nc    = Dataset(files_n[f"{wind}_{time}"],"r")
        u     = np.array(nc["avg_u"][:,:,:])
        v     = np.array(nc["avg_v"][:,:,:])
        noturb_u[iwind,:] = np.mean(u,axis=(1,2)) if time==times[0] else noturb_u[iwind,:]+np.mean(u,axis=(1,2))
        noturb_v[iwind,:] = np.mean(v,axis=(1,2)) if time==times[0] else noturb_v[iwind,:]+np.mean(v,axis=(1,2))

        nc    = Dataset(files_t[f"{wind}_{time}"],"r")
        u     = np.array(nc["avg_u"][:,:,:])
        v     = np.array(nc["avg_v"][:,:,:])
        turb_u[iwind,:] = np.mean(u,axis=(1,2)) if time==times[0] else turb_u[iwind,:]+np.mean(u,axis=(1,2))
        turb_v[iwind,:] = np.mean(v,axis=(1,2)) if time==times[0] else turb_v[iwind,:]+np.mean(v,axis=(1,2))
    noturb_u[iwind,:] /= len(times)
    noturb_v[iwind,:] /= len(times)
    turb_u  [iwind,:] /= len(times)
    turb_v  [iwind,:] /= len(times)
    x = np.array(nc["x"][:])
    y = np.array(nc["y"][:])
    z = np.array(nc["z"][:])
    xlen = x[-1]+(x[1]-x[0])/2
    ylen = y[-1]+(y[1]-y[0])/2
    turb_u[iwind,:] = noturb_u[iwind,:]*(10000**2-xlen*ylen)/(10000**2) + turb_u[iwind,:]*(xlen*ylen)/(10000**2)
    turb_v[iwind,:] = noturb_v[iwind,:]*(10000**2-xlen*ylen)/(10000**2) + turb_v[iwind,:]*(xlen*ylen)/(10000**2)
# for iwind in range(len(winds)) :
#     plt.plot(noturb_u[iwind,:]/turb_u[iwind,:],z,label=f"{winds[iwind]}")
# plt.legend()
# plt.show()
# plt.close()

nc = Dataset(f"turbine_coarse_z0_{z0:.6f}.nc","w",format='NETCDF4')
nc.createDimension('z'   ,len(z    ))
nc.createDimension('wind',len(winds))
nc.createVariable('z'       ,'f4',('z'       ))[:]   = z       [:]
nc.createVariable('wind'    ,'f4',('wind'    ))[:]   = winds   [:]
nc.createVariable('noturb_u','f4',('wind','z'))[:,:] = noturb_u[:,:]
nc.createVariable('noturb_v','f4',('wind','z'))[:,:] = noturb_v[:,:]
nc.createVariable('turb_u'  ,'f4',('wind','z'))[:,:] = turb_u  [:,:]
nc.createVariable('turb_v'  ,'f4',('wind','z'))[:,:] = turb_v  [:,:]
nc.close()


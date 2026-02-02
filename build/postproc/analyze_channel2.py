from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

prefixes = ["channel_u0constuflux-_acosut-2.000000_","channel_u0constuflux-_acosut-4.000000_","channel_u0constuflux-_acosut-8.000000_"]
times = range(1,32)

fig,ax = plt.subplots(3,2,figsize=(10,10))

for prefix in prefixes :
  nc   = Dataset(f"{prefix}{0:08d}.nc","r")
  x    = np.array(nc["x"][:])
  y    = np.array(nc["y"][:])
  z    = np.array(nc["z"][:])
  nx   = len(x)
  ny   = len(y)
  nz   = len(z)
  dx   = x[1]-x[0]
  dy   = y[1]-y[0]
  dz   = z[1]-z[0]
  xlen = x[-1]+dx/2
  ylen = y[-1]+dy/2
  zlen = z[-1]+dz/2
  mn_u    = np.zeros((nz))
  mn_tke  = np.zeros((nz))
  mn_upwp = np.zeros((nz))
  arr_ustar = []
  arr_tke   = []
  arr_umag  = []
  for i in times :
    print(f"{prefix} {(i)}/{times[-1]}")
    nc         = Dataset(f"{prefix}{i:08d}.nc","r")
    u          = np.array(nc["uvel"][:,:,:])
    v          = np.array(nc["vvel"][:,:,:])
    w          = np.array(nc["wvel"][:,:,:])
    up         = u - np.mean(u,axis=(1,2))[:,None,None]
    vp         = v - np.mean(v,axis=(1,2))[:,None,None]
    wp         = w - np.mean(w,axis=(1,2))[:,None,None]
    tke        = (up*up+vp*vp+wp*wp)/2
    mn_tke    += np.mean( tke                          , axis=(1,2) )
    mn_upwp   += np.mean( up*wp                        , axis=(1,2) )
    mn_u      += np.mean( np.array(nc["avg_u"][:,:,:]) , axis=(1,2) )
    arr_ustar += [np.mean(np.array(nc["surface_flux_sfc_ustar"][:,:]))]
    arr_tke   += [np.mean(tke)]
    arr_umag  += [np.mean(np.abs(np.array(nc["avg_u"][:,:,:])))]
  mn_u    /= len(times)
  mn_tke  /= len(times)
  mn_upwp /= len(times)
  ax[0,0].plot(mn_u   ,z,label=f"{prefix}")
  ax[0,0].set_ylabel("z (m)")
  ax[0,0].set_xlabel("avg u-velocity")
  ax[0,1].plot(mn_tke ,z,label=f"{prefix}")
  ax[0,1].set_ylabel("z (m)")
  ax[0,1].set_xlabel("Res TKE")
  ax[1,0].plot(mn_upwp,z,label=f"{prefix}")
  ax[1,0].set_ylabel("z (m)")
  ax[1,0].set_xlabel("u'w'")
  ax[1,1].plot(times,arr_ustar,label=f"{prefix}")
  ax[1,1].set_xlabel("output file ID")
  ax[1,1].set_ylabel("mean sfc friction velocity")
  ax[2,0].plot(times,arr_tke  ,label=f"{prefix}")
  ax[2,0].set_xlabel("output file ID")
  ax[2,0].set_ylabel("domain mean TKE")
  ax[2,1].plot(times,arr_umag ,label=f"{prefix}")
  ax[2,1].set_xlabel("output file ID")
  ax[2,1].set_ylabel("bulk velocity")

for axloc in ax.flatten() :
  axloc.margins(x=0)
  axloc.grid(True)
  axloc.tick_params(axis='x', rotation=35)
  axloc.legend()
fig.tight_layout()
plt.show()
plt.close()

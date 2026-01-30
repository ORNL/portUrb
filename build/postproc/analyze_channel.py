from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

fig,ax = plt.subplots(3,2,figsize=(10,10))

times = range(40,161)
arr_ustar = []
arr_tke   = []
arr_umag  = []
for i in times :
  nc   = Dataset(f"channel_u0-0.100000_z0-0.007812_{i:08d}.nc","r")
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
  u    = np.array(nc["uvel"][:,:,:])
  v    = np.array(nc["vvel"][:,:,:])
  w    = np.array(nc["wvel"][:,:,:])
  up   = u - np.mean(u,axis=(1,2))[:,None,None]
  vp   = v - np.mean(v,axis=(1,2))[:,None,None]
  wp   = w - np.mean(w,axis=(1,2))[:,None,None]
  tke  = (up*up+vp*vp+wp*wp)/2
  upwp = up*wp
  u    = np.array(nc["avg_u"][:,:,:])
  ax[0,0].plot(np.mean(u   ,axis=(1,2)),z,label=f"{i}")
  ax[0,0].set_ylabel("z (m)")
  ax[0,0].set_xlabel("avg u-velocity")
  ax[0,1].plot(np.mean(tke ,axis=(1,2)),z,label=f"{i}")
  ax[0,1].set_ylabel("z (m)")
  ax[0,1].set_xlabel("Res TKE")
  ax[1,0].plot(np.mean(upwp,axis=(1,2)),z,label=f"{i}")
  ax[1,0].set_ylabel("z (m)")
  ax[1,0].set_xlabel("u'w'")
  arr_ustar += [np.mean(np.array(nc["surface_flux_sfc_ustar"][:,:]))]
  arr_tke   += [np.mean(tke)]
  arr_umag  += [np.mean(np.abs(u))]
ax[1,1].plot(times,arr_ustar)
ax[1,1].set_xlabel("output file ID")
ax[1,1].set_ylabel("mean sfc friction velocity")
ax[2,0].plot(times,arr_tke  )
ax[2,0].set_xlabel("output file ID")
ax[2,0].set_ylabel("domain mean TKE")
ax[2,1].plot(times,arr_umag )
ax[2,1].set_xlabel("output file ID")
ax[2,1].set_ylabel("bulk velocity")
for axloc in ax.flatten() :
  axloc.margins(x=0)
  axloc.grid(True)
  axloc.tick_params(axis='x', rotation=35)
fig.tight_layout()
plt.show()
plt.close()


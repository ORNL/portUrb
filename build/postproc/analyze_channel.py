from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

def spectra(T,dx = 1) :
  spd  = np.abs( np.fft.rfft(T[0,0,:]) )**2
  freq = np.fft.rfftfreq(len(T[0,0,:]))
  spd[:] = 0
  for k in range(T.shape[0]) :
    for j in range(T.shape[1]) :
      spd[:] += np.abs( np.fft.rfft(T[k,j,:]) )**2
  spd[:] /= T.shape[0]*T.shape[1]
  return freq*2*2*np.pi/(2*dx) , spd

prefix = "delta_1.0_"
times = range(25,26)

nc   = Dataset(f"{prefix}{times[-1]:08d}.nc","r")
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
mag  = np.sqrt(u**2+v**2+w**2)

kind = np.argmin(np.abs(z-0.3))
freq,spd1 = spectra(mag[kind:kind+1,:,:],dx=dx)
fig = plt.figure(figsize=(6,3))
ax = fig.gca()
ax.plot(freq,spd1,label="Wind Speed spectra")
ax.plot(freq[1:],1.2e3*freq[1:]**(-5/3),label=r"$f^{-5/3}$")
ax.vlines(2*np.pi/(2 *dx),1.e-3,1.e3,linestyle="--",color="red")
ax.vlines(2*np.pi/(4 *dx),1.e-3,1.e3,linestyle="--",color="red")
ax.vlines(2*np.pi/(8 *dx),1.e-3,1.e3,linestyle="--",color="red")
ax.vlines(2*np.pi/(16*dx),1.e-3,1.e3,linestyle="--",color="red")
ax.text(0.9*2*np.pi/(2 *dx),2.e3,"$2  \Delta x$")
ax.text(0.9*2*np.pi/(4 *dx),2.e3,"$4  \Delta x$")
ax.text(0.9*2*np.pi/(8 *dx),2.e3,"$8  \Delta x$")
ax.text(0.9*2*np.pi/(16*dx),2.e3,"$16 \Delta x$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Frequency")
ax.set_ylabel("Spectral Power")
ax.legend(loc='lower left')
ax.set_ylim(top=1.e6)
ax.margins(x=0)
plt.margins(x=0)
plt.tight_layout()
plt.show()
plt.close()

fig,ax = plt.subplots(3,2,figsize=(10,10))

arr_ustar = []
arr_tke   = []
arr_umag  = []
for i in times :
  nc   = Dataset(f"{prefix}{i:08d}.nc","r")
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


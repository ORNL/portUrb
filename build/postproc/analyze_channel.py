from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

tend = 19
for i in range(max(0,tend-2),tend+1) :
  nc = Dataset(f"channel_{i:08d}.nc","r")
  x = np.array(nc["x"][:])
  y = np.array(nc["y"][:])
  z = np.array(nc["z"][:])
  nx = len(x)
  ny = len(y)
  nz = len(z)
  dx = x[1]-x[0]
  dy = y[1]-y[0]
  dz = z[1]-z[0]
  xlen = x[-1]+dx/2
  ylen = y[-1]+dy/2
  zlen = z[-1]+dz/2
  rho = np.array(nc["density_dry"][:,:,:])
  u   = np.array(nc["uvel"       ][:,:,:])
  v   = np.array(nc["vvel"       ][:,:,:])
  w   = np.array(nc["wvel"       ][:,:,:])
  T   = np.array(nc["temperature"][:,:,:])
  K   = np.array(nc["TKE"][:,:,:])/rho
  plt.plot(np.mean(u,axis=(1,2)),z,label=f"{i}")
plt.grid()
plt.legend()
plt.show()
plt.close()


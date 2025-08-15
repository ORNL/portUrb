from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmap import Colormap
import xarray
import pandas as pd
from scipy.ndimage import rotate

workdir = "/lustre/storm/nwp501/scratch/imn/portUrb/build"


def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))


t1=1
t2=24
times = range(t1,t2+1)
for i in range(len(times)) :
  nc_disk = Dataset(f"{workdir}/turbulent__wind-11.000000_turbine_{times[i]:08d}.nc","r")
  nc_base = Dataset(f"{workdir}/turbulent__wind-11.000000_noturbine_{times[i]:08d}.nc","r")
  if i==0:
    z = np.array(nc_base["z"][:])/1000
  uloc = np.mean(np.array(nc_base["avg_u"][:,:,:]),axis=(1,2))
  vloc = np.mean(np.array(nc_base["avg_v"][:,:,:]),axis=(1,2))
  dir_base = np.atan2(vloc,uloc)/np.pi*180
  plt.plot(np.sqrt(uloc**2+vloc**2),z,label=f"{i}hr")
  uinf = np.mean(np.sqrt(np.array(nc_disk["u_samp_trace_turb_0"][:])**2+np.array(nc_disk["v_samp_trace_turb_0"][:])**2))
  print(f"Uinf_{i}: {uinf}")
plt.legend()
plt.show()


nc_base = Dataset(f"{workdir}/turbulent__wind-11.000000_noturbine_{11:08d}.nc","r")
z = np.array(nc_base["z"][:])
u = np.mean(np.array(nc_base["avg_u"][:,:,:]),axis=(1,2))
v = np.mean(np.array(nc_base["avg_v"][:,:,:]),axis=(1,2))
plt.plot(np.sqrt(u**2+v**2),z)
z0=0.1
plt.plot(13.771*np.log((z+z0)/z0)/np.log((500+z0)/z0),z)
plt.yscale('log')
plt.show()

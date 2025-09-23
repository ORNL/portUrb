from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

hs = 5

def plot_var_mean(f,vname,col,lab) :
  nc = Dataset(f,"r")
  z = np.array(nc["z"][:])
  v = np.mean(np.array(nc[vname]),axis=(1,2))
  plt.plot(v,z,color=col,label=lab)

def plot_var_std(f,vname,col,lab) :
  nc = Dataset(f,"r")
  z = np.array(nc["z"][:])
  v = np.std(np.array(nc[vname]),axis=(1,2))
  plt.plot(v,z,color=col,label=lab)

vnames = ["uvel","vvel","wvel","TKE","theta_pert","density_pert"]

for vname in vnames :
  plot_var_mean("supercell_kessler_fixed_00000001.nc"   ,vname,"black","fixed"   )
  plot_var_mean("supercell_kessler_variable_00000001.nc",vname,"red"  ,"variable")
  plt.title(f"mean_{vname}")
  plt.legend()
  plt.show()
  plt.close()

for vname in vnames :
  plot_var_std("supercell_kessler_fixed_00000001.nc"   ,vname,"black","fixed"   )
  plot_var_std("supercell_kessler_variable_00000001.nc",vname,"red"  ,"variable")
  plt.title(f"std_{vname}")
  plt.legend()
  plt.show()
  plt.close()


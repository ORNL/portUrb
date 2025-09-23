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

time = 20

for vname in vnames :
  plot_var_mean(f"ABL_neutral_fixed_{time:08d}.nc"   ,vname,"black","fixed 20")
  plot_var_mean(f"ABL_neutral_fixed_40_{time:08d}.nc",vname,"blue" ,"fixed 40")
  plot_var_mean(f"ABL_neutral_variable_{time:08d}.nc",vname,"red"  ,"variable")
  plt.title(f"mean_{vname}")
  plt.legend()
  plt.show()
  plt.close()

for vname in vnames :
  plot_var_std(f"ABL_neutral_fixed_{time:08d}.nc"   ,vname,"black","fixed 20")
  plot_var_std(f"ABL_neutral_fixed_40_{time:08d}.nc",vname,"blue" ,"fixed 40")
  plot_var_std(f"ABL_neutral_variable_{time:08d}.nc",vname,"red"  ,"variable")
  plt.title(f"std_{vname}")
  plt.legend()
  plt.show()
  plt.close()


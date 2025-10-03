from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

files = [f"city_stretched_{i:08d}.nc" for i in range(2)]

def plot_var(vname) :
  for i in range(len(files)) :
    nc = Dataset(files[i],"r")
    z   = np.array(nc["z"][:])
    var = np.array(nc[vname][:,:,:])
    plt.plot(np.mean(var,axis=(1,2)),z,label=f"{i}",linewidth=(i+1)/5.)
  plt.legend()
  plt.title(vname)
  plt.show()
  plt.close()

plot_var("uvel")
plot_var("vvel")

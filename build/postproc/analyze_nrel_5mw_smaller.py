from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmap import Colormap

times = range(13,21)
tkes  = [ "0.000000",
          "0.200000",
          "0.400000",
          "0.600000",
          "0.800000",
          "1.000000"]

prefixes = [f"turbulent_nrel_5mw_smaller_f_TKE-{tke}" for tke in tkes]

workdir = "/lustre/orion/stf006/scratch/imn/portUrb/build"

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

nc = Dataset(f"{prefixes[0]}_{times[0]:08d}.nc","r")
x = np.array(nc["x"])/1000
y = np.array(nc["y"])/1000
z = np.array(nc["z"])/1000
nx = len(x)
ny = len(y)
nz = len(z)
dx = x[1]-x[0]
dy = y[1]-y[0]
dz = z[1]-z[0]
xlen = x[-1]+dx/2
ylen = y[-1]+dy/2
zlen = z[-1]+dz/2

uavg = [ np.array(nc.variables["avg_u"][:,:,:]) for i in range(len(prefixes)) ]
for iprefix in range(len(prefixes)) :
  for itime in range(len(times)) :
    nc = Dataset(f"{prefixes[iprefix]}_{times[itime]:08d}.nc","r")
    u = np.array(nc.variables["avg_u"][:,:,:])
    v = np.array(nc.variables["avg_v"][:,:,:])
    uavg[iprefix] = np.sqrt(u*u+v*v) if itime == 0 else uavg[iprefix]+np.sqrt(u*u+v*v)
  uavg[iprefix] /= len(times)

khub = get_ind(z,90 /1000)
ihub = get_ind(x,500/1000)
in2D = get_ind(x,(500-2 *126)/1000)
i02D = get_ind(x,(500+2 *126)/1000)
i04D = get_ind(x,(500+4 *126)/1000)
i08D = get_ind(x,(500+8 *126)/1000)
i12D = get_ind(x,(500+12*126)/1000)
i14D = get_ind(x,(500+14*126)/1000)

y1 = get_ind(y,(250-63)/1000)
y2 = get_ind(y,(250+63)/1000)
z1 = get_ind(z,(90 -63)/1000)
z2 = get_ind(z,(90 +63)/1000)


titles = ["0","-2D","0D","2D","4D","8D","12D"]
count=0
for i in [0,in2D,ihub,i02D,i04D,i08D,i12D] :
  fig = plt.figure(figsize=(12,12))
  ax  = fig.gca()
  print(titles[count])
  for itke in range(len(tkes)) :
    ax.plot(uavg[itke][khub,:,i],y,label=tkes[itke])
    print(f"{tkes[itke]} , {np.mean(uavg[itke][khub,:,i])}")
  print("")
  ax.legend()
  ax.set_title(titles[count])
  ax.margins(x=0)
  fig.tight_layout()
  plt.show()
  plt.close()
  count += 1


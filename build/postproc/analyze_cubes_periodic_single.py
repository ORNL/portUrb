from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from cubes_periodic_data import *
from scipy.stats import pearsonr
import xarray

def get_ind(arr,val) :
  return np.argmin(np.abs(arr-val))

nadd = 9
times = np.array([i for i in range(4,6)])
prefixes = ["/lustre/orion/stf006/scratch/imn/portUrb/build/cubes_periodic_"]

# prefixes = ["/lustre/storm/nwp501/scratch/imn/portUrb/build/cubes_periodic_dx0.005_z0-1.000e-07_"]

errs_glob = np.array([10000. for i in range(len(prefixes))])
for k in range(len(prefixes)) :
  prefix = prefixes[k]
  fnames = [f"{prefix}{i:08}.nc" for i in times]
  print(fnames)
  nc   = xarray.open_dataset(fnames[0])
  x    = np.array(nc["x"])
  y    = np.array(nc["y"])
  z    = np.array(nc["z"])
  nz   = len(z)
  dx   = x[1]-x[0]
  dy   = y[1]-y[0]
  xlen = x[-1]+dx/2
  ylen = y[-1]+dy/2
  p0_xsamp = [get_ind(x,2*xlen/8),get_ind(x,6*xlen/8),get_ind(x,2*xlen/8),get_ind(x,6*xlen/8)]
  p0_ysamp = [get_ind(y,6*ylen/8),get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8)]
  p1_xsamp = [get_ind(x,4*ylen/8),get_ind(x,0*ylen/8),get_ind(x,4*ylen/8),get_ind(x,0*ylen/8)]
  p1_ysamp = [get_ind(y,6*ylen/8),get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8)]
  p2_xsamp = [get_ind(x,2*xlen/8),get_ind(x,6*xlen/8),get_ind(x,2*xlen/8),get_ind(x,6*xlen/8)]
  p2_ysamp = [get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8),get_ind(y,6*ylen/8)]
  p3_xsamp = [get_ind(x,4*xlen/8),get_ind(x,0*xlen/8),get_ind(x,4*xlen/8),get_ind(x,0*xlen/8)]
  p3_ysamp = [get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8),get_ind(y,6*ylen/8)]

  errs = np.array([0. for i in range(len(fnames))])
  # Compute error norms
  for i in range(len(fnames)) :
    u0  = np.array([0. for k in range(nz)])
    u1  = np.array([0. for k in range(nz)])
    u2  = np.array([0. for k in range(nz)])
    u3  = np.array([0. for k in range(nz)])
    uw1 = np.array([0. for k in range(nz)])
    uw3 = np.array([0. for k in range(nz)])
    nc = xarray.open_dataset(fnames[i])
    for j in range(len(p0_xsamp)) :
      u0  += np.array(nc["avg_u"    ][:,p0_ysamp[j],p0_xsamp[j]])
      u1  += np.array(nc["avg_u"    ][:,p1_ysamp[j],p1_xsamp[j]])
      u2  += np.array(nc["avg_u"    ][:,p2_ysamp[j],p2_xsamp[j]])
      u3  += np.array(nc["avg_u"    ][:,p3_ysamp[j],p3_xsamp[j]])
      uw1 += np.array(nc["avg_up_wp"][:,p1_ysamp[j],p1_xsamp[j]])
      uw3 += np.array(nc["avg_up_wp"][:,p3_ysamp[j],p3_xsamp[j]])
    u0  /= len(p0_xsamp)
    u1  /= len(p0_xsamp)
    u2  /= len(p0_xsamp)
    u3  /= len(p0_xsamp)
    uw1 /= len(p0_xsamp)
    uw3 /= len(p0_xsamp)
    u0i  = np.interp(p0_y   ,z/.02,u0/10)
    u1i  = np.interp(p1_y   ,z/.02,u1/10)
    u2i  = np.interp(p2_y   ,z/.02,u2/10)
    u3i  = np.interp(p3_y   ,z/.02,u3/10)
    uw1i = np.interp(uw_p1_y,z/.02,-uw1 )
    uw3i = np.interp(uw_p3_y,z/.02,-uw3 )
    p = int(2)
    u0e  = np.sum(np.abs(p0_x      -u0i )**p)/np.sum(np.abs(p0_x      )**p)
    u1e  = np.sum(np.abs(p1_x      -u1i )**p)/np.sum(np.abs(p1_x      )**p)
    u2e  = np.sum(np.abs(p2_x      -u2i )**p)/np.sum(np.abs(p2_x      )**p)
    u3e  = np.sum(np.abs(p3_x      -u3i )**p)/np.sum(np.abs(p3_x      )**p)
    uw1e = np.sum(np.abs(uw_p1_x/10-uw1i)**p)/np.sum(np.abs(uw_p1_x/10)**p)
    uw3e = np.sum(np.abs(uw_p3_x/10-uw3i)**p)/np.sum(np.abs(uw_p3_x/10)**p)
    # u0e  = 1-pearsonr(p0_x      ,u0i ).statistic
    # u1e  = 1-pearsonr(p1_x      ,u1i ).statistic
    # u2e  = 1-pearsonr(p2_x      ,u2i ).statistic
    # u3e  = 1-pearsonr(p3_x      ,u3i ).statistic
    # uw1e = 1-pearsonr(uw_p1_x/10,uw1i).statistic
    # uw3e = 1-pearsonr(uw_p3_x/10,uw3i).statistic
    errs[i] = (u0e+u1e+u2e+u3e+uw1e+uw3e)/6
  errs2 = np.array([0. for i in range(len(errs)-nadd)])
  for i in range(len(errs)-nadd) :
    errs2[i] = np.mean(errs[i:i+nadd+1])
  errs_glob[k] = np.min(errs2)
  print(f"{prefix}: {errs_glob[k]} ")
  if (errs_glob[k] == np.min(errs_glob)) :
    imin = np.argmin(errs2)
    times_min = times[imin:imin+nadd+1]
    prefix_glob = prefix

print(f"Using prefix: {prefix_glob} with times: {times_min}")
fnames = [f"{prefix_glob}{time:08}.nc" for time in times_min]

# Compute profiles to plot
u0  = np.array([0. for k in range(nz)])
u1  = np.array([0. for k in range(nz)])
u2  = np.array([0. for k in range(nz)])
u3  = np.array([0. for k in range(nz)])
uw1 = np.array([0. for k in range(nz)])
uw3 = np.array([0. for k in range(nz)])
for i in range(len(fnames)) :
  nc = xarray.open_dataset(fnames[i])
  for j in range(len(p0_xsamp)) :
    u0  += np.array(nc["avg_u"    ][:,p0_ysamp[j],p0_xsamp[j]])
    u1  += np.array(nc["avg_u"    ][:,p1_ysamp[j],p1_xsamp[j]])
    u2  += np.array(nc["avg_u"    ][:,p2_ysamp[j],p2_xsamp[j]])
    u3  += np.array(nc["avg_u"    ][:,p3_ysamp[j],p3_xsamp[j]])
    uw1 += np.array(nc["avg_up_wp"][:,p1_ysamp[j],p1_xsamp[j]])
    uw3 += np.array(nc["avg_up_wp"][:,p3_ysamp[j],p3_xsamp[j]])
u0  /= len(fnames)*len(p0_xsamp)
u1  /= len(fnames)*len(p0_xsamp)
u2  /= len(fnames)*len(p0_xsamp)
u3  /= len(fnames)*len(p0_xsamp)
uw1 /= len(fnames)*len(p0_xsamp)
uw3 /= len(fnames)*len(p0_xsamp)

fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,sharey=True,figsize=(9,7.8))
ax1.set_xlim(-2,8)
ax1.set_ylim(0,3)
ax1.grid()
ax1.scatter(p0_x*10,p0_y,label="Experiment",facecolors='none',edgecolors='black')
ax1.plot(u0,z/.02,label=r"${\overline{u}(P_0)}$",color="red")
ax1.set_xlabel(r"$u$ velocity [m/s]")
ax1.set_ylabel(r"$z/h$")
ax1.legend()
ax1.margins(0)

ax2.set_xlim(-2,8)
ax2.set_ylim(0,3)
ax2.grid()
ax2.scatter(p1_x*10,p1_y,label="Experiment",facecolors='none',edgecolors='black')
ax2.plot(u1,z/.02,label=r"${\overline{u}(P_1)}$",color="red")
ax2.set_xlabel(r"$u$ velocity [m/s]")
ax2.set_ylabel(r"$z/h$")
ax2.legend()
ax2.margins(0)

ax3.set_xlim(-2,10)
ax3.set_ylim(0,3)
ax3.grid()
ax3.scatter(uw_p1_x,uw_p1_y,label="Experiment",facecolors='none',edgecolors='black')
ax3.plot(-uw1*10,z/.02,label=r"${\overline{u^\prime w^\prime}(P_1)}\times 10$",color="red")
ax3.set_xlabel(r"$\overline{u^\prime w^\prime}$ velocity [m^2/s^2]")
ax3.set_ylabel(r"$z/h$")
ax3.legend()
ax3.margins(0)

ax4.set_xlim(-2,8)
ax4.set_ylim(0,3)
ax4.grid()
ax4.scatter(p2_x*10,p2_y,label="Experiment",facecolors='none',edgecolors='black')
ax4.plot(u2,z/.02,label=r"${\overline{u}(P_2)}$",color="red")
ax4.set_xlabel(r"$u$ velocity [m/s]")
ax4.set_ylabel(r"$z/h$")
ax4.legend()
ax4.margins(0)

ax5.set_xlim(-2,8)
ax5.set_ylim(0,3)
ax5.grid()
ax5.scatter(p3_x*10,p3_y,label="Experiment",facecolors='none',edgecolors='black')
ax5.plot(u3,z/.02,label=r"${\overline{u}(P_3)}$",color="red")
ax5.set_xlabel(r"$u$ velocity [m/s]")
ax5.set_ylabel(r"$z/h$")
ax5.legend()
ax5.margins(0)

ax6.set_xlim(-2,10)
ax6.set_ylim(0,3)
ax6.grid()
ax6.scatter(uw_p3_x,uw_p3_y,label="Experiment",facecolors='none',edgecolors='black')
ax6.plot(-uw3*10,z/.02,label=r"${\overline{u^\prime w^\prime}(P_3)}\times 10$",color="red")
ax6.set_xlabel(r"$\overline{u^\prime w^\prime}$ velocity [m^2/s^2]")
ax6.set_ylabel(r"$z/h$")
ax6.legend()
ax6.margins(0)

fig.tight_layout()
plt.savefig("cubes_periodic_validation.png",dpi=600)
plt.show()
plt.close()


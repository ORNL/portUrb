from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os.path

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

nfiles = 41
times = np.array([120*i/(nfiles-1) for i in range(nfiles)])

workdir = "/lustre/orion/stf006/scratch/imn/RSST/CELL"

nens = 8
files_ens = [ [f"{workdir}/supercell_ens{j}_{i:08}.nc" for i in range(nfiles)] for j in range(1,nens+1)]
nexp = 6
files = [[f"{workdir}/supercell_orig_rho_350_{i:08}.nc"   for i in range(nfiles)],
         [f"{workdir}/supercell_orig_theta_350_{i:08}.nc" for i in range(nfiles)],
         [f"{workdir}/supercell_rsst_350_{i:08}.nc"       for i in range(nfiles)],
         [f"{workdir}/supercell_rsst_262_{i:08}.nc"       for i in range(nfiles)],
         [f"{workdir}/supercell_rsst_174_{i:08}.nc"       for i in range(nfiles)],
         [f"{workdir}/supercell_rsst_86_{i:08}.nc"        for i in range(nfiles)],]
labels = ["ORIG-RHO_350","ORIG-THETA_350","RSS_350","RSS_262","RSS_174","RSS_86"]
colors = ["black","red","green","blue","cyan","magenta"]
z = np.array(Dataset(files[0][0],"r")["z"])
sfc_theta_min  = np.array([[0. for i in range(nfiles)] for j in range(nexp)])
cold_pool_frac = np.array([[0. for i in range(nfiles)] for j in range(nexp)])
precip_accum   = np.array([[0. for i in range(nfiles)] for j in range(nexp)])
min_w          = np.array([[0. for i in range(nfiles)] for j in range(nexp)])
max_w          = np.array([[0. for i in range(nfiles)] for j in range(nexp)])
if ( not os.path.isfile("cell_data.npz") ) :
  for j in range(nexp) :
    for i in range(nfiles) :
      nc = Dataset(files[j][i],"r")
      sfc_theta = np.array(nc["theta_pert"][0,:,:])
      w         = np.array(nc["wvel"])
      rho_d     = np.array(nc["density_dry"])
      sfc_theta_min [j,i] = np.min(sfc_theta)
      cold_pool_frac[j,i] = np.sum(np.where(sfc_theta <= -2,True,False)) / sfc_theta.size
      precip_accum  [j,i] = np.mean(np.array(nc["micro_rainnc"]) + np.array(nc["micro_snownc"]) + np.array(nc["micro_graupelnc"]))
      min_w         [j,i] = np.min(w[:get_ind(z,3500),:,:])
      max_w         [j,i] = np.max(w)
      print(j,i)
  np.savez_compressed("cell_data.npz",sfc_theta=sfc_theta, w=w, rho_d=rho_d, sfc_theta_min=sfc_theta_min,cold_pool_frac=cold_pool_frac,precip_accum=precip_accum,min_w=min_w,max_w=max_w)
else :
  cell_data = np.load("cell_data.npz")
  sfc_theta      = cell_data["sfc_theta"     ]
  w              = cell_data["w"             ]
  rho_d          = cell_data["rho_d"         ]
  sfc_theta_min  = cell_data["sfc_theta_min" ]
  cold_pool_frac = cell_data["cold_pool_frac"]
  precip_accum   = cell_data["precip_accum"  ]
  min_w          = cell_data["min_w"         ]
  max_w          = cell_data["max_w"         ]


y_q0 = [0,-9.11,-11.40]
y_q1 = [0,-7.38,-10.03]
y_q2 = [0,-4.46, -8.94]
y_q3 = [0,-3.78, -6.65]
y_q4 = [0,-1.93, -3.46]
morr_x = np.arange(0,120,1)
morr_y_q0 = -1/25920000*(y_q0[0] - 2*y_q0[1] + y_q0[2])*morr_x**4 + 1/864000*(9*y_q0[0] - 16*y_q0[1] + 7*y_q0[2])*morr_x**3 - 1/14400*(11*y_q0[0] - 16*y_q0[1] + 5*y_q0[2])*morr_x**2 + y_q0[0]
morr_y_q4 = -1/25920000*(y_q4[0] - 2*y_q4[1] + y_q4[2])*morr_x**4 + 1/864000*(9*y_q4[0] - 16*y_q4[1] + 7*y_q4[2])*morr_x**3 - 1/14400*(11*y_q4[0] - 16*y_q4[1] + 5*y_q4[2])*morr_x**2 + y_q4[0]
# morr_y_q0 = 1/864000*(3*y_q0[0] - 4*y_q0[1] + y_q0[2])*morr_x**3 - 1/14400*(7*y_q0[0] - 8*y_q0[1] + y_q0[2])*morr_x**2 + y_q0[0]
# morr_y_q4 = 1/864000*(3*y_q4[0] - 4*y_q4[1] + y_q4[2])*morr_x**3 - 1/14400*(7*y_q4[0] - 8*y_q4[1] + y_q4[2])*morr_x**2 + y_q4[0]
for j in range(nexp) :
  plt.plot(times,sfc_theta_min[j,:],label=labels[j],color=colors[j])
plt.fill_between(morr_x,morr_y_q0,morr_y_q4,color="lightskyblue")
plt.xlabel("Time (hrs)")
plt.ylabel(r"Min sfc $\theta^\prime$ (K)")
plt.grid()
plt.legend()
plt.savefig("supercell_sfc_theta_min.png",dpi=600)
plt.show()
plt.close()


y_q0 = [0,0.000,0.006]
y_q1 = [0,0.004,0.051]
y_q2 = [0,0.006,0.144]
y_q3 = [0,0.014,0.154]
y_q4 = [0,0.022,0.190]
morr_x = np.arange(0,120,1)
morr_y_q0 = 1/864000*(3*y_q0[0] - 4*y_q0[1] + y_q0[2])*morr_x**3 - 1/14400*(7*y_q0[0] - 8*y_q0[1] + y_q0[2])*morr_x**2 + y_q0[0]
morr_y_q4 = 1/864000*(3*y_q4[0] - 4*y_q4[1] + y_q4[2])*morr_x**3 - 1/14400*(7*y_q4[0] - 8*y_q4[1] + y_q4[2])*morr_x**2 + y_q4[0]
for j in range(nexp) :
  plt.plot(times,cold_pool_frac[j,:],label=labels[j],color=colors[j])
plt.fill_between(morr_x,morr_y_q0,morr_y_q4,color="lightskyblue")
plt.xlabel("Time (hrs)")
plt.ylabel(r"Sfc cold pool fraction ($\theta^\prime \leq -2K$)")
plt.grid()
plt.legend()
plt.savefig("supercell_cold_pool_frac.png",dpi=600)
plt.show()
plt.close()


y_q0 = [0,0.04,0.37]
y_q1 = [0,0.05,0.48]
y_q2 = [0,0.10,0.97]
y_q3 = [0,0.12,1.09]
y_q4 = [0,0.14,1.28]
morr_x = np.arange(0,120,1)
morr_y_q0 = 1/864000*(3*y_q0[0] - 4*y_q0[1] + y_q0[2])*morr_x**3 - 1/14400*(7*y_q0[0] - 8*y_q0[1] + y_q0[2])*morr_x**2 + y_q0[0]
morr_y_q4 = 1/864000*(3*y_q4[0] - 4*y_q4[1] + y_q4[2])*morr_x**3 - 1/14400*(7*y_q4[0] - 8*y_q4[1] + y_q4[2])*morr_x**2 + y_q4[0]
for j in range(nexp) :
  plt.plot(times,precip_accum[j,:],label=labels[j],color=colors[j])
plt.fill_between(morr_x,morr_y_q0,morr_y_q4,color="lightskyblue")
plt.xlabel("Time (hrs)")
plt.ylabel(r"Total sfc accum precip (mm)")
plt.grid()
plt.legend()
plt.savefig("supercell_precip_accum.png",dpi=600)
plt.show()
plt.close()


for j in range(nexp) :
  plt.plot(times,min_w[j,:],label=labels[j],color=colors[j])
plt.xlabel("Time (hrs)")
plt.ylabel(r"Min Vertical Velocity (m/s)")
plt.legend()
plt.grid()
plt.legend()
plt.savefig("supercell_min_w.png",dpi=600)
plt.show()
plt.close()


for j in range(nexp) :
  plt.plot(times,max_w[j,:],label=labels[j],color=colors[j])
plt.xlabel("Time (hrs)")
plt.ylabel(r"Max Vertical Velocity (m/s)")
plt.legend()
plt.grid()
plt.legend()
plt.savefig("supercell_max_w.png",dpi=600)
plt.show()
plt.close()


for i in [20,40] :
  for j in range(nexp) :
    nc = Dataset(files[j][i],"r")
    qc = (np.array(nc["graupel"])+np.array(nc["snow"])+np.array(nc["rain_water"]))# /np.array(nc["density_dry"])
    plt.plot(np.mean(qc ,axis=(1,2))*1000,z/1000,label=labels[j],linewidth=2,color=colors[j])
  plt.xlim(left=0)
  plt.ylim(0,15)
  plt.legend()
  plt.grid()
  plt.xlabel("Horiz Avg Dry Mixing Ratio (g/kg)")
  plt.ylabel("Height (km)")
  plt.savefig(f"supercell_avg_col_cond_{i}_min.png",dpi=600)
  plt.show()
  plt.close()



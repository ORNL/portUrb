from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

nfiles = 41
times = np.array([3.*i/nfiles for i in range(nfiles)])

workdir = "/lustre/orion/stf006/scratch/imn/portUrb/build"

nexp = 6
files = [[f"{workdir}/CELL_ORIG_RHO_350/supercell_orig_rho_350_{i:08}.nc"     for i in range(nfiles)],
         [f"{workdir}/CELL_ORIG_THETA_350/supercell_orig_theta_350_{i:08}.nc" for i in range(nfiles)],
         [f"{workdir}/CELL_RSS_350/supercell_rsst_350_{i:08}.nc"              for i in range(nfiles)],
         [f"{workdir}/CELL_RSS_262/supercell_rsst_262_{i:08}.nc"              for i in range(nfiles)],
         [f"{workdir}/CELL_RSS_174/supercell_rsst_174_{i:08}.nc"              for i in range(nfiles)],
         [f"{workdir}/CELL_RSS_86/supercell_rsst_86_{i:08}.nc"                for i in range(nfiles)],]
labels = ["ORIG-RHO_350","ORIG-THETA_350","RSS_350","RSS_262","RSS_174","RSS_86"]
colors = ["black","red","green","blue","cyan","magenta"]
z = np.array(Dataset(files[0][0],"r")["z"])
sfc_theta_min  = np.array([[0. for i in range(nfiles)] for j in range(nexp)])
cold_pool_frac = np.array([[0. for i in range(nfiles)] for j in range(nexp)])
precip_accum   = np.array([[0. for i in range(nfiles)] for j in range(nexp)])
min_w          = np.array([[0. for i in range(nfiles)] for j in range(nexp)])
max_w          = np.array([[0. for i in range(nfiles)] for j in range(nexp)])
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

for j in range(nexp) :
  plt.plot(times,sfc_theta_min[j,:],label=labels[j],color=colors[j])
plt.xlabel("Time (hrs)")
plt.ylabel(r"Min sfc $\theta^\prime$ (K)")
plt.grid()
plt.legend()
plt.savefig("supercell_sfc_theta_min.png",dpi=600)
plt.show()
plt.close()

for j in range(nexp) :
  plt.plot(times,cold_pool_frac[j,:],label=labels[j],color=colors[j])
plt.xlabel("Time (hrs)")
plt.ylabel(r"Sfc cold pool fraction ($\theta^\prime \leq -2K$)")
plt.grid()
plt.legend()
plt.savefig("supercell_cold_pool_frac.png",dpi=600)
plt.show()
plt.close()

for j in range(nexp) :
  plt.plot(times,precip_accum[j,:],label=labels[j],color=colors[j])
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
    qc = (np.array(nc["rain_water"])+np.array(nc["snow"])+np.array(nc["graupel"]))/np.array(nc["density_dry"])
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



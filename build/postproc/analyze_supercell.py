from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

nfiles = 21
times = np.array([2.*i/nfiles for i in range(nfiles)])

# dx200_files = [f"/lustre/storm/nwp501/scratch/imn/supercell/supercell_200m_{i:08}.nc" for i in range(nfiles)]
# dx200_z = np.array(Dataset(dx200_files[0],"r")["z"])
# dx200_sfc_theta_min  = [0. for i in range(nfiles)]
# dx200_cold_pool_frac = [0. for i in range(nfiles)]
# dx200_precip_accum   = [0. for i in range(nfiles)]
# dx200_min_w          = [0. for i in range(nfiles)]
# dx200_max_w          = [0. for i in range(nfiles)]
# for i in range(nfiles) :
#   nc = Dataset(dx200_files[i],"r")
#   sfc_theta = np.array(nc["theta_pert"][0,:,:])
#   w         = np.array(nc["wvel"])
#   rho_d     = np.array(nc["density_dry"])
#   dx200_sfc_theta_min [i] = np.min(sfc_theta)
#   dx200_cold_pool_frac[i] = np.sum(np.where(sfc_theta <= -2,True,False)) / sfc_theta.size
#   dx200_precip_accum  [i] = np.mean(np.array(nc["micro_rainnc"]) + np.array(nc["micro_snownc"]) + np.array(nc["micro_graupelnc"]))
#   dx200_min_w         [i] = np.min(w[:get_ind(dx200_z,3500),:,:])
#   dx200_max_w         [i] = np.max(w)

R_d     = 287.
cp_d    = 7.*R_d/2.
cv_d    = cp_d - R_d
gamma_d = cp_d / cv_d
kappa_d = R_d  / cp_d
R_v     = 461.6
cp_v    = 4.*R_v
cv_v    = cp_v - R_v
p0      = 1.e5
grav    = 9.81
C0      = np.pow(R_d*np.pow(p0,-kappa_d),gamma_d)


dx1000_files = [f"/lustre/orion/stf006/scratch/imn/portUrb/build/supercell_buoy-thetap_press-orig_cs-350_{i:08}.nc" for i in range(nfiles)]
dx1000_z = np.array(Dataset(dx1000_files[0],"r")["z"])
dx1000_sfc_theta_min  = [0. for i in range(nfiles)]
dx1000_cold_pool_frac = [0. for i in range(nfiles)]
dx1000_precip_accum   = [0. for i in range(nfiles)]
dx1000_min_w          = [0. for i in range(nfiles)]
dx1000_max_w          = [0. for i in range(nfiles)]
for i in range(nfiles) :
  nc = Dataset(dx1000_files[i],"r")
  rho_d     = np.array(nc["density_dry"][:,:,:])
  rho_v     = np.array(nc["water_vapor"][:,:,:])
  temp      = np.array(nc["temperature"][:,:,:])
  press     = rho_d*R_d*temp + rho_v*R_v*temp
  theta     = np.pow(press/C0,1/gamma_d)/(rho_d+rho_v)
  sfc_theta = np.array(theta[0,:,:])
  w         = np.array(nc["wvel"])
  dx1000_sfc_theta_min [i] = np.min(sfc_theta)
  dx1000_cold_pool_frac[i] = np.sum(np.where(sfc_theta <= -2,True,False)) / sfc_theta.size
  dx1000_precip_accum  [i] = np.mean(np.array(nc["micro_rainnc"]) + np.array(nc["micro_snownc"]) + np.array(nc["micro_graupelnc"]))
  dx1000_min_w         [i] = np.min(w[:get_ind(dx1000_z,3500),:,:])
  dx1000_max_w         [i] = np.max(w)

plt.plot(times,dx1000_sfc_theta_min,label="dx=1000m",color="blue")
# plt.plot(times,dx200_sfc_theta_min ,label="dx=200m" ,color="blue",linestyle="--")
plt.xlabel("Time (hrs)")
plt.ylabel(r"Min sfc $\theta^\prime$ (K)")
plt.grid()
plt.legend()
plt.savefig("supercell_sfc_theta_min.png",dpi=600)
plt.show()
plt.close()

plt.plot(times,dx1000_cold_pool_frac,label="dx=1000m",color="blue")
# plt.plot(times,dx200_cold_pool_frac ,label="dx=200m" ,color="blue",linestyle="--")
plt.xlabel("Time (hrs)")
plt.ylabel(r"Sfc cold pool fraction ($\theta^\prime \leq -2K$)")
plt.grid()
plt.legend()
plt.savefig("supercell_cold_pool_frac.png",dpi=600)
plt.show()
plt.close()

plt.plot(times,dx1000_precip_accum,label="dx=1000m",color="blue")
# plt.plot(times,dx200_precip_accum ,label="dx=200m" ,color="blue",linestyle="--")
plt.xlabel("Time (hrs)")
plt.ylabel(r"Total sfc accum precip (mm)")
plt.grid()
plt.legend()
plt.savefig("supercell_precip_accum.png",dpi=600)
plt.show()
plt.close()

plt.plot(times,dx1000_min_w,label=r"dx=1000m, Min ($z \leq 3.5$km)",color="blue")
plt.plot(times,dx1000_max_w,label=r"dx=1000m, Max ($\forall z$)"   ,color="red" )
# plt.plot(times,dx200_min_w ,label=r"dx=200m , Min ($z \leq 3.5$km)",color="blue",linestyle="--")
# plt.plot(times,dx200_max_w ,label=r"dx=200m , Max ($\forall z$)"   ,color="red" ,linestyle="--")
plt.xlabel("Time (hrs)")
plt.ylabel(r"Vertical Velocity (m/s)")
plt.legend()
plt.grid()
plt.legend()
plt.savefig("supercell_min_max_w.png",dpi=600)
plt.show()
plt.close()


# for i in [20,40,60,80,100,120] :
for i in [20] :
  nc = Dataset(dx1000_files[i],"r")
  rho_d = np.array(nc["density_dry"])
  rho_v = np.array(nc["water_vapor"])
  rho_c = np.array(nc["cloud_water"])
  rho_r = np.array(nc["rain_water" ])
  rho_i = np.array(nc["cloud_ice"  ])
  rho_s = np.array(nc["snow"       ])
  rho_g = np.array(nc["graupel"    ])
  rho = rho_d + rho_v + rho_c + rho_r + rho_i + rho_s + rho_g
  dx1000_qc  = rho_c / rho
  dx1000_qr  = rho_r / rho
  dx1000_qi  = rho_i / rho
  dx1000_qs  = rho_s / rho
  dx1000_qg  = rho_g / rho
  dx1000_tot = rho_g+rho_r+rho_s
  # nc = Dataset(dx200_files[i],"r")
  # rho_d = np.array(nc["density_dry"])
  # rho_v = np.array(nc["water_vapor"])
  # rho_c = np.array(nc["cloud_water"])
  # rho_r = np.array(nc["rain_water" ])
  # rho_i = np.array(nc["cloud_ice"  ])
  # rho_s = np.array(nc["snow"       ])
  # rho_g = np.array(nc["graupel"    ])
  # rho = rho_d + rho_v + rho_c + rho_r + rho_i + rho_s + rho_g
  # dx200_qc  = rho_c / rho
  # dx200_qr  = rho_r / rho
  # dx200_qi  = rho_i / rho
  # dx200_qs  = rho_s / rho
  # dx200_qg  = rho_g / rho
  # dx200_tot = (rho-rho_d-rho_v) / rho
  plt.plot(np.mean(dx1000_qc ,axis=(1,2))*1000,dx1000_z/1000,label="qc (dx=1000m)",linewidth=2,color="red"    )
  plt.plot(np.mean(dx1000_qr ,axis=(1,2))*1000,dx1000_z/1000,label="qr (dx=1000m)",linewidth=2,color="black"  )
  plt.plot(np.mean(dx1000_qi ,axis=(1,2))*1000,dx1000_z/1000,label="qi (dx=1000m)",linewidth=2,color="blue"   )
  plt.plot(np.mean(dx1000_qs ,axis=(1,2))*1000,dx1000_z/1000,label="qs (dx=1000m)",linewidth=2,color="cyan"   )
  plt.plot(np.mean(dx1000_qg ,axis=(1,2))*1000,dx1000_z/1000,label="qg (dx=1000m)",linewidth=2,color="magenta")
  plt.plot(np.mean(dx1000_tot,axis=(1,2))*1000,dx1000_z/1000,label="tot(dx=1000m)",linewidth=2,color="orange" )
  # plt.plot(np.mean(dx200_qc  ,axis=(1,2))*1000,dx200_z /1000,label="qc (dx=200m)" ,linewidth=2,color="red"    ,linestyle="--")
  # plt.plot(np.mean(dx200_qr  ,axis=(1,2))*1000,dx200_z /1000,label="qr (dx=200m)" ,linewidth=2,color="black"  ,linestyle="--")
  # plt.plot(np.mean(dx200_qi  ,axis=(1,2))*1000,dx200_z /1000,label="qi (dx=200m)" ,linewidth=2,color="blue"   ,linestyle="--")
  # plt.plot(np.mean(dx200_qs  ,axis=(1,2))*1000,dx200_z /1000,label="qs (dx=200m)" ,linewidth=2,color="cyan"   ,linestyle="--")
  # plt.plot(np.mean(dx200_qg  ,axis=(1,2))*1000,dx200_z /1000,label="qg (dx=200m)" ,linewidth=2,color="magenta",linestyle="--")
  # plt.plot(np.mean(dx200_tot ,axis=(1,2))*1000,dx200_z /1000,label="tot(dx=200m)" ,linewidth=2,color="orange" ,linestyle="--")
  plt.xlim(left=0)
  plt.ylim(0,15)
  plt.legend(ncols=2)
  plt.grid()
  plt.xlabel("Horiz Avg Wet Mixing Ratio (g/kg)")
  plt.ylabel("Height (km)")
  plt.savefig(f"supercell_avg_col_moist_{i}_min.png",dpi=600)
  plt.show()
  plt.close()



from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os.path

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

nfiles = 41
times = np.array([120*i/(nfiles-1) for i in range(nfiles)])

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

orig_x1 = [0.64444,
1.44444,
2.15556,
2.57778,
2.91111,
3.2,
3.42222,
3.75556,
4.06667,
4.48889,
4.95556,
5.62222,
6.28889,
6.84444,
7.4,
8,
8.37778,
8.93333,
9.6,
10.44444,
10.91111,
11.42222,
11.8,
12.11111,
12.66667,
12.97778,
13.37778,
13.91111,
14.57778,
15.26667,
]

orig_y1 = [9.70E-04,
2.91E-03,
1.07E-02,
2.33E-02,
4.65E-02,
8.33E-02,
1.17E-01,
1.55E-01,
1.75E-01,
0.20155,
0.22771,
0.25678,
0.28585,
0.31395,
0.33915,
0.36047,
0.36725,
0.36531,
0.35078,
0.31298,
0.28198,
0.24322,
0.20543,
0.17345,
0.12791,
0.10174,
0.0688,
0.03585,
0.0126,
0,
]

orig_x2 = [4.44E-01,
7.33E-01,
1.29E+00,
1.87E+00,
2.49E+00,
3.31E+00,
4.11E+00,
5.24E+00,
6.53E+00,
7.87E+00,
8.73333,
9.46667,
10.26667,
10.66667,
11.15556,
11.77778,
12.37778,
12.84444,
13.57778,
14.04444,
14.44444,
14.82222,
15.42222,
]

orig_y2 = [0.00194,
0.00388,
0.00678,
0.01453,
0.02713,
0.06686,
0.11047,
0.15504,
0.2064,
0.24903,
0.26163,
0.2626,
0.25,
0.23643,
0.21318,
0.16957,
0.12597,
0.09012,
0.03876,
0.01357,
0.00388,
0.00097,
0,
]

morr_x = np.arange(0,20,0.5)
morr_y1 = np.interp(morr_x,orig_x1,orig_y1,left=0,right=0)
morr_y2 = np.interp(morr_x,orig_x2,orig_y2,left=0,right=0)

for i in [20,40] :
  for j in range(nexp) :
    nc = Dataset(files[j][i],"r")
    qc = (np.array(nc["graupel"]))/np.array(nc["density_dry"])
    plt.plot(np.mean(qc ,axis=(1,2))*1000,z/1000,label=labels[j],linewidth=2,color=colors[j])
  if i == 40 :
    plt.fill_betweenx(morr_x,morr_y1,morr_y2,color="lightskyblue")
  plt.xlim(left=0)
  plt.ylim(0,15)
  plt.legend()
  plt.grid()
  plt.xlabel("Horiz Avg Dry Mixing Ratio (g/kg)")
  plt.ylabel("Height (km)")
  plt.savefig(f"supercell_avg_col_cond_{i}_min.png",dpi=600)
  plt.show()
  plt.close()



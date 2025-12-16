
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

nc = Dataset("bem.nc","r")
wind    = np.array(nc["wind"   ][:])
segment = np.array(nc["segment"][:])
dT_dr   = np.array(nc["dT_dr"  ][:,:])
dQ_dr   = np.array(nc["dQ_dr"  ][:,:])
phi_r   = np.array(nc["phi_r"  ][:,:])
alpha_r = np.array(nc["alpha_r"][:,:])
Cn_r    = np.array(nc["Cn_r"   ][:,:])
Ct_r    = np.array(nc["Ct_r"   ][:,:])
a_r     = np.array(nc["a_r"    ][:,:])
ap_r    = np.array(nc["ap_r"   ][:,:])
pitch   = np.array(nc["pitch"  ][:])
omega   = np.array(nc["omega"  ][:])
thrust  = np.array(nc["thrust" ][:])
torque  = np.array(nc["torque" ][:])
power   = np.array(nc["power"  ][:])
C_T     = np.array(nc["C_T"    ][:])
C_P     = np.array(nc["C_P"    ][:])
C_Q     = np.array(nc["C_Q"    ][:])

print(f"{'U_inf':^15s}  {'pitch (deg)':^15s}  {'omega (rpm)':^15s}")
for iwind in range(0,len(wind)) :
  print(f"{wind[iwind]:15.6f}  {pitch[iwind]/np.pi*180.:15.6f}  {omega[iwind]/2/np.pi*60.:15.6f}")

plt.plot(wind,power)
plt.xlabel("Inflow wind speed (U_inf) in m/s")
plt.ylabel("Power generation (W)")
plt.show()
plt.close()

for iwind in [17,18,19,20,21,22] :
  plt.plot(segment,dT_dr[iwind,:],label=f"{wind[iwind]}")
  plt.xlabel("Radial Position (m)")
plt.ylabel("dT_dr")
plt.legend()
plt.grid()
plt.savefig("dT_dr.png")
plt.show()
plt.close()

for iwind in [17,18,19,20,21,22] :
  plt.plot(segment,dQ_dr[iwind,:]/segment[:],label=f"{wind[iwind]}")
plt.xlabel("Radial Position (m)")
plt.ylabel("dQ_dr/r")
plt.legend()
plt.grid()
plt.savefig("dQ_dr.png")
plt.show()
plt.close()

for iwind in [5,9,15,22] :
  plt.plot(segment,phi_r[iwind,:]/np.pi*180,label=f"{wind[iwind]}")
plt.xlabel("Radial Position (m)")
plt.ylabel("phi (deg)")
plt.legend()
plt.grid()
plt.savefig("phi.png")
plt.show()
plt.close()

for iwind in [5,9,15,22] :
  plt.plot(segment,a_r[iwind,:],label=f"{wind[iwind]}")
plt.xlabel("Radial Position (m)")
plt.ylabel("a")
plt.legend()
plt.grid()
plt.savefig("a.png")
plt.show()
plt.close()

for iwind in [5,9,15,22] :
  plt.plot(segment,ap_r[iwind,:],label=f"{wind[iwind]}")
plt.xlabel("Radial Position (m)")
plt.ylabel("ap")
plt.legend()
plt.grid()
plt.savefig("ap.png")
plt.show()
plt.close()

for iwind in [5,9,15,22] :
  plt.plot(segment,alpha_r[iwind,:]/np.pi*180,label=f"{wind[iwind]}")
plt.xlabel("Radial Position (m)")
plt.ylabel("alpha (deg)")
plt.legend()
plt.grid()
plt.savefig("alpha.png")
plt.show()
plt.close()





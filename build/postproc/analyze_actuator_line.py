from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from cmap import Colormap

D = 126
R = 63
H = 90

for i in range(1,5) :
  nc = Dataset(f"test_{i:08d}.nc","r")
  p  = np.array(nc["power_0"][:])
  pwr = p if i==1 else np.concatenate((pwr,p))
print(np.mean(pwr[int(0.8*len(pwr)):]))
plt.plot(np.arange(0,240,240/len(pwr)),pwr)
plt.show()
plt.close()


tend_u       = np.array(nc["turbine_tend_u"][:,:,:])
tend_v       = np.array(nc["turbine_tend_v"][:,:,:])
tend_w       = np.array(nc["turbine_tend_w"][:,:,:])
uvel         = np.array(nc["uvel"          ][:,:,:])
vvel         = np.array(nc["vvel"          ][:,:,:])
wvel         = np.array(nc["wvel"          ][:,:,:])
x            = np.array(nc["x"             ][:])
y            = np.array(nc["y"             ][:])
z            = np.array(nc["z"             ][:])
rad          = np.array(nc["radial_points" ][:])
force_axial  = np.array(nc["force_axial_0" ][:,:,:])
inflow_axial = np.array(nc["inflow_axial_0"][:,:,:])
force_tang   = np.array(nc["force_tang_0"  ][:,:,:])
inflow_tang  = np.array(nc["inflow_tang_0" ][:,:,:])

i = np.argmin(np.abs(x-6*D))
k = np.argmin(np.abs(z-H))
j1 = np.argmin(np.abs(y-1*D))
j2 = np.argmin(np.abs(y-3*D))
k2 = np.argmin(np.abs(z-1.5*D))
print(j1,j2,k2)

Y,Z = np.meshgrid(y[j1:j2],z[:k2])
plt.contourf(Y,Z,tend_u[:k2,j1:j2,i-2],levels=100,cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
plt.xlabel("x-location")
plt.ylabel("y-location")
plt.gca().invert_xaxis()
plt.gca().set_aspect("equal")
plt.colorbar(orientation="horizontal")
plt.tight_layout()
plt.show()
plt.close()

plt.contourf(Y,Z,tend_v[:k2,j1:j2,i-2],levels=100,cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
plt.gca().set_aspect("equal")
plt.colorbar(orientation="horizontal")
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()
plt.close()

plt.contourf(Y,Z,tend_w[:k2,j1:j2,i-2],levels=100,cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
plt.gca().set_aspect("equal")
plt.colorbar(orientation="horizontal")
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()
plt.close()

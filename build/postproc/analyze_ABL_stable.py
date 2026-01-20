from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmap import Colormap
import xarray

workdir = "/lustre/orion/stf006/scratch/imn/portUrb/build"


def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))


R_d     = 287.
cp_d    = 1003.
R_v     = 461.
cp_v    = 1859
p0      = 1.e5
grav    = 9.81
cv_d    = cp_d-R_d
gamma_d = cp_d/cv_d
kappa_d = R_d/cp_d
cv_v    = cp_v-R_v
C0      = np.pow(R_d*np.pow(p0,-kappa_d),gamma_d);


nc = Dataset(f"{workdir}/ABL_stable_2m_00000005.nc","r")
x = np.array(nc["x"][:])/1000
y = np.array(nc["y"][:])/1000
z = np.array(nc["z"][:])/1000
nx = len(x)
ny = len(y)
nz = len(z)
dx = x[1]-x[0]
dy = y[1]-y[0]
dz = z[1]-z[0]
xlen = x[-1]+dx/2
ylen = y[-1]+dy/2
zlen = z[-1]+dz/2
uvel  = np.array(nc["uvel"       ][:,:,:])
vvel  = np.array(nc["vvel"       ][:,:,:])
wvel  = np.array(nc["wvel"       ][:,:,:])
rho   = np.array(nc["density_dry"][:,:,:])
T     = np.array(nc["temperature"][:,:,:])
theta = np.pow((rho*R_d*T)/C0,1./gamma_d)/rho
mag   = np.sqrt(uvel*uvel+vvel*vvel)


t1 = 262.75
t2 = 265.25
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,10))
X,Y = np.meshgrid(x,y)
zind = get_ind(z,.100)
print(zind, z[zind])
mn = np.min(theta[zind,:,:])
mx = np.max(theta[zind,:,:])
CS1 = ax1.contourf(X,Y,theta[zind,:,:],levels=np.arange(t1,t2,(t2-t1)/200),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax1.axis('scaled')
ax1.set_xlabel("x-location (km)")
ax1.set_ylabel("y-location (km)")
ax1.margins(x=0)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar1 = plt.colorbar(CS1,orientation="horizontal",cax=cax1)
cbar1.ax.tick_params(labelrotation=30)

mn  = np.mean(wvel[zind,:,:])
std = np.std (wvel[zind,:,:])
CS2 = ax2.contourf(X,Y,wvel[zind,:,:],levels=np.arange(-.75,.75,1.5/200),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax2.axis('scaled')
ax2.set_xlabel("x-location (km)")
ax2.set_ylabel("y-location (km)")
ax2.margins(x=0)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar2 = plt.colorbar(CS2,orientation="horizontal",cax=cax2)
cbar2.ax.tick_params(labelrotation=30)

yind = get_ind(y,ylen/2)
zind = get_ind(z,.275)
print(zind, z[zind])
X,Z = np.meshgrid(x,z[:zind+1])
CS3 = ax3.contourf(X,Z,theta[:zind+1,yind,:],levels=np.arange(t1,t2,(t2-t1)/200),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax3.axis('scaled')
ax3.set_xlabel("x-location (km)")
ax3.set_ylabel("z-location (km)")
ax3.margins(x=0)
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar3 = plt.colorbar(CS3,orientation="horizontal",cax=cax3)
cbar3.ax.tick_params(labelrotation=30)

mn  = np.mean(wvel[:,yind,:])
std = np.std (wvel[:,yind,:])
CS4 = ax4.contourf(X,Z,wvel[:zind+1,yind,:],levels=np.arange(-.75,.75,1.5/200),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax4.axis('scaled')
ax4.set_xlabel("x-location (km)")
ax4.set_ylabel("z-location (km)")
ax4.margins(x=0)
divider = make_axes_locatable(ax4)
cax4 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar4 = plt.colorbar(CS4,orientation="horizontal",cax=cax4)
cbar4.ax.tick_params(labelrotation=30)
plt.tight_layout()
plt.savefig("ABL_convective_contourf.png",dpi=600)
plt.show()
plt.close()



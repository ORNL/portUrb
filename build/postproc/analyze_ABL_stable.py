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

tval = 62

etime  = []
ustar2 = []
bflux  = []
blhgt  = []
for i in range(1,tval+1) :
  print(f"{i}/{tval}")
  nc = Dataset(f"{workdir}/ABL_stable_1m_{i:08d}.nc","r")
  x = np.array(nc["x"][:])
  y = np.array(nc["y"][:])
  z = np.array(nc["z"][:])
  nx = len(x)
  ny = len(y)
  nz = len(z)
  dx = x[1]-x[0]
  dy = y[1]-y[0]
  dz = z[1]-z[0]
  k1 = np.argmin(np.abs(z-25))
  usloc = np.mean(np.array(nc['surface_flux_sfc_ustar'    ][:,:])**2)
  rho   = np.array(nc["density_dry"][:,:,:])
  u     = np.array(nc["uvel"       ][:,:,:])
  v     = np.array(nc["vvel"       ][:,:,:])
  w     = np.array(nc["wvel"       ][:,:,:])
  T     = np.array(nc["temperature"][:,:,:])
  K     = np.array(nc["TKE"        ][:,:,:])/rho
  theta = np.pow((rho*R_d*T)/C0,1./gamma_d)/rho
  up    = u - np.mean(u,axis=(1,2))[:,np.newaxis,np.newaxis]
  vp    = v - np.mean(v,axis=(1,2))[:,np.newaxis,np.newaxis]
  wp    = w - np.mean(w,axis=(1,2))[:,np.newaxis,np.newaxis]
  upwp  = np.mean(up*wp,axis=(1,2))
  vpwp  = np.mean(vp*wp,axis=(1,2))
  dt_dz = np.gradient(theta,dz,axis=0)
  du_dz = np.gradient(u    ,dz,axis=0)
  dv_dz = np.gradient(v    ,dz,axis=0)
  dw_dx = np.gradient(w    ,dx,axis=2)
  dw_dy = np.gradient(w    ,dy,axis=1)
  N     = np.where( dt_dz >= 0 , np.sqrt(grav/theta*(dt_dz)) , 0 )
  delta = np.pow( dx*dy*dz , 1./3. )
  ell   = np.minimum( 0.76*np.sqrt(K)/np.maximum(N,1.e-10) , delta )
  km    = 0.1 * ell * np.sqrt(K)
  tau_xz = -km*(dw_dx + du_dz)
  tau_yz = -km*(dw_dy + dv_dz)
  xz     = upwp+np.mean(tau_xz,axis=(1,2))
  yz     = vpwp+np.mean(tau_yz,axis=(1,2))
  stress = np.sqrt(xz**2+yz**2)
  bl_loc = -1
  for k in range(k1,nz) :
    if (stress[k] < 0.05*usloc) :
      z1, z2 = z[k-1], z[k]
      s1, s2 = stress[k-1], stress[k]
      bl_loc = z1 + (0.05*usloc - s1) * (z2 - z1) / (s2 - s1 + 1e-30)
      break
  etime  += [nc['etime'][0]]
  ustar2 += [np.mean(np.array(nc['surface_flux_sfc_ustar'    ][:,:])**2)]
  bflux  += [np.mean(np.array(nc['surface_flux_sfc_buoy_flux'][:,:])   )]
  blhgt  += [0.95*bl_loc]
  # plt.plot(stress,z)
  # plt.show()
  # plt.close()
  
fig,ax = plt.subplots(3,1,figsize=(6,6))
ax[0].plot(etime,ustar2)
ax[0].set_xlim(0,9*3600)
ax[0].set_ylim(0,0.2   )
ax[0].grid(True)
ax[0].margins(x=0)

ax[1].plot(etime,bflux)
ax[1].set_xlim(0,9*3600)
ax[1].set_ylim(0,0.0006)
ax[1].grid(True)
ax[1].margins(x=0)

ax[2].plot(etime,blhgt)
ax[2].set_xlim(0,9*3600)
ax[2].set_ylim(0,300)
ax[2].grid(True)
ax[2].margins(x=0)
fig.tight_layout()
plt.show()
plt.close()

nc = Dataset(f"{workdir}/ABL_stable_1m_{tval:08d}.nc","r")
x = np.array(nc["x"][:])
y = np.array(nc["y"][:])
z = np.array(nc["z"][:])
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
X,Y = np.meshgrid(x/1000,y/1000)
zind = get_ind(z,100)
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
zind = get_ind(z,275)
print(zind, z[zind])
X,Z = np.meshgrid(x/1000,z[:zind+1]/1000)
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


fig,ax = plt.subplots(1,2,figsize=(6,4))
ax[0].plot(np.mean(np.sqrt(uvel**2+vvel**2+wvel**2),axis=(1,2)),z)
ax[0].set_xlim(0,10)
ax[0].set_ylim(0,300)
ax[0].grid(True)

ax[1].plot(np.mean(theta,axis=(1,2)),z)
ax[1].set_xlim(262,268)
ax[1].set_ylim(0,300)
ax[1].grid(True)

plt.show()
plt.close()


from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmap import Colormap
import xarray
import pandas as pd
from scipy.ndimage import rotate

workdir = "/lustre/orion/stf006/scratch/imn/portUrb/build"
amrwdir = "/ccs/home/imn/exawind-benchmarks/amr-wind/atmospheric_boundary_layer/convective_abl_nrel5mw/results"

def spectra(T,dx = 1) :
  spd  = np.abs( np.fft.rfft(T[0,0,:]) )**2
  freq = np.fft.rfftfreq(len(T[0,0,:]))
  spd = 0
  for k in range(T.shape[0]) :
    for j in range(T.shape[1]) :
      spd += np.abs( np.fft.rfft(T[k,j,:]) )**2
      spd += np.abs( np.fft.rfft(T[k,:,j]) )**2
  spd /= T.shape[0]*T.shape[1]*2
  return freq[1:]*2*2*np.pi/(2*dx) , spd[1:]


def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))


# nc = Dataset(f"{workdir}/nrel_5mw_convective_precursor_00000015.nc","r")
# x = np.array(nc["x"])/1000
# y = np.array(nc["y"])/1000
# z = np.array(nc["z"])/1000
# nx = len(x)
# ny = len(y)
# nz = len(z)
# dx = x[1]-x[0]
# dy = y[1]-y[0]
# dz = z[1]-z[0]
# xlen = x[-1]+dx/2
# ylen = y[-1]+dy/2
# zlen = z[-1]+dz/2
# hs   = 5
# uvel  = np.array(nc["uvel"])
# vvel  = np.array(nc["vvel"])
# mag   = np.sqrt(uvel*uvel+vvel*vvel)
# 
# 
# fig = plt.figure(figsize=(6,6))
# ax = fig.gca()
# X,Y = np.meshgrid(x,y)
# zind = get_ind(z,.090)
# CS1 = ax.contourf(X,Y,mag[zind,:,:],levels=np.arange(0,14,0.1),cmap=Colormap('coolwarm').to_mpl(),extend="both")
# ax.set_title("Convective ABL U_h T=15,000s")
# ax.axis('scaled')
# ax.set_xlabel("x-location (km)")
# ax.set_ylabel("y-location (km)")
# ax.margins(x=0)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="3%", pad=0.1)
# cbar1 = plt.colorbar(CS1,orientation="vertical",cax=cax)
# cbar1.ax.tick_params(labelrotation=0)
# plt.tight_layout()
# plt.savefig("porturb_uhoriz_contour.png")
# plt.show()
# plt.close()
# 
# t1=16
# t2=20
# times = range(t1,t2+1)
# for i in range(len(times)) :
#   nc = Dataset(f"{workdir}/nrel_5mw_convective_precursor_{times[i]:08d}.nc","r")
#   u  = np.array(nc["avg_u"][:,:,:])
#   v  = np.array(nc["avg_v"][:,:,:])
#   w  = np.array(nc["avg_w"][:,:,:])
#   up = np.array(nc["uvel"][:,:,:])
#   vp = np.array(nc["vvel"][:,:,:])
#   wp = np.array(nc["wvel"][:,:,:])
#   Tp = np.array(nc["theta_pert"][:,:,:])+np.array(nc["hy_theta_cells"][hs:hs+nz])[:,np.newaxis,np.newaxis]
#   up -= np.mean(u ,axis=(1,2))[:,np.newaxis,np.newaxis]
#   vp -= np.mean(v ,axis=(1,2))[:,np.newaxis,np.newaxis]
#   wp -= np.mean(w ,axis=(1,2))[:,np.newaxis,np.newaxis]
#   Tp -= np.mean(Tp,axis=(1,2))[:,np.newaxis,np.newaxis]
#   maghloc = np.sqrt(u*u+v*v)
#   magloc  = np.sqrt(u*u+v*v+w*w)
#   thloc   = np.array(nc["theta_pert"][:,:,:])+np.array(nc["hy_theta_cells"][hs:hs+nz])[:,np.newaxis,np.newaxis]
#   rholoc  = np.array(nc["density_dry"][:,:,:])
#   drloc   = np.atan2(v,u)/np.pi*180
#   uuloc   = up*up
#   uvloc   = up*vp
#   uwloc   = up*wp
#   vvloc   = vp*vp
#   vwloc   = vp*wp
#   wwloc   = wp*wp
#   uTloc   = up*Tp
#   vTloc   = vp*Tp
#   wTloc   = wp*Tp
#   magh = maghloc if i==0 else magh+maghloc
#   mag  = magloc  if i==0 else mag +magloc
#   th   = thloc   if i==0 else th  +thloc
#   rho  = rholoc  if i==0 else rho +rholoc
#   dr   = drloc   if i==0 else dr  +drloc
#   uu   = uuloc   if i==0 else uu  +uuloc
#   uv   = uvloc   if i==0 else uv  +uvloc
#   uw   = uwloc   if i==0 else uw  +uwloc
#   vv   = vvloc   if i==0 else vv  +vvloc
#   vw   = vwloc   if i==0 else vw  +vwloc
#   ww   = wwloc   if i==0 else ww  +wwloc
#   uT   = uTloc   if i==0 else uT  +uTloc
#   vT   = vTloc   if i==0 else vT  +vTloc
#   wT   = wTloc   if i==0 else wT  +wTloc
# magh /= len(times)
# mag  /= len(times)
# th   /= len(times)
# rho  /= len(times)
# dr   /= len(times)
# uu   /= len(times)
# uv   /= len(times)
# uw   /= len(times)
# vv   /= len(times)
# vw   /= len(times)
# ww   /= len(times)
# uT   /= len(times)
# vT   /= len(times)
# wT   /= len(times)
# 
# amrw_zi = 803.133
# amrw_ustar = .468
# 
# # Boundary layer height
# thz = np.mean(th,axis=(1,2))
# dthz = (thz[1:] - thz[:nz-1])/(dz*1000)
# k = np.argmax(np.abs(dthz))
# zi = z[k]-dz/2
# k2 = get_ind(z,zi*0.1)
# ustar = np.mean(np.sqrt(np.abs(uv)/rho)[:k2+1,:,:])
# uref = 11.4
# zref = 90
# z0   = 0.01
# kar  = 0.4
# ustar_ideal = kar*uref/np.log((zref-z0)/z0);
# print(f"Boundary layer height: z(dtheta/dz>0.1): {zi*1000}m")
# print(f"Shear velocity: int(sqrt(u'v'/rho),0,z_i/10) : {ustar} m/s")
# print(f"Log Law shear velocity: K*u_R/log((z_R-z0)/z0): {ustar_ideal} m/s")
# 
# amrw_z = pd.read_csv(f"{amrwdir}/avgprofile_5000s_Uhoriz.csv")["z"].to_numpy()
# amrw_D = pd.read_csv(f"{amrwdir}/avgprofile_5000s_Uhoriz.csv")["Uhoriz"].to_numpy()
# fig = plt.figure(figsize=(5,6))
# ax = fig.gca()
# ax.plot(np.mean(magh,axis=(1,2)),z/zi,label="portUrb")
# ax.plot(amrw_D,amrw_z/amrw_zi        ,label="AMR-Wind")
# ax.set_ylim(0,0.6)
# ax.set_xlim(0,15)
# ax.set_xlabel("Horiz Wind Speed (m/s)")
# ax.set_ylabel("z/zi")
# ax.legend()
# ax.margins(x=0)
# plt.tight_layout()
# plt.grid()
# plt.savefig("uhoriz_z.png")
# plt.show()
# plt.close()
# 
# amrw_D = pd.read_csv(f"{amrwdir}/avgprofile_5000s_T.csv")["T"].to_numpy()
# fig = plt.figure(figsize=(5,6))
# ax = fig.gca()
# ax.plot(np.mean(th,axis=(1,2)),z/zi,label="portUrb")
# ax.plot(amrw_D,amrw_z/amrw_zi      ,label="AMR-Wind")
# ax.set_ylim(0,1.5)
# ax.set_xlim(299.5,312)
# ax.set_xlabel("Pot. Temp. (K)")
# ax.set_ylabel("z/zi")
# ax.legend()
# ax.margins(x=0)
# plt.tight_layout()
# plt.grid()
# plt.savefig("T_z.png")
# plt.show()
# plt.close()
# 
# amrw_D = pd.read_csv(f"{amrwdir}/avgprofile_5000s_WindDir.csv")["WindDir"].to_numpy()
# fig = plt.figure(figsize=(5,6))
# ax = fig.gca()
# ax.plot(np.mean(270-dr,axis=(1,2)),z/zi,label="portUrb")
# ax.plot(amrw_D,amrw_z/amrw_zi          ,label="AMR-Wind")
# ax.set_ylim(0,1.4)
# ax.set_xlim(200,300)
# ax.set_xlabel("Wind Dir (deg)")
# ax.set_ylabel("z/zi")
# ax.legend()
# ax.margins(x=0)
# plt.tight_layout()
# plt.grid()
# plt.savefig("WindDir_z.png")
# plt.show()
# plt.close()
# 
# amrw_D = pd.read_csv(f"{amrwdir}/avgprofile_5000s_TI_TKE.csv")["TI_TKE"].to_numpy()
# fig = plt.figure(figsize=(5,6))
# ax = fig.gca()
# ax.plot(np.mean(np.sqrt((uu+vv+ww)/3)/mag,axis=(1,2)),z/zi,label="portUrb")
# ax.plot(amrw_D,amrw_z/amrw_zi                             ,label="AMR-Wind")
# ax.set_ylim(0,1.4)
# ax.set_xlim(0,.11)
# ax.set_xlabel("Turb Int (-)")
# ax.set_ylabel("z/zi")
# ax.legend()
# ax.margins(x=0)
# plt.legend()
# plt.tight_layout()
# plt.grid()
# plt.savefig("TI_TKE_z.png")
# plt.show()
# plt.close()
# 
# 
# amrw_uu = pd.read_csv(f"{amrwdir}/avgprofile_5000s_uu.csv")["uu"].to_numpy()
# amrw_uv = pd.read_csv(f"{amrwdir}/avgprofile_5000s_vv.csv")["vv"].to_numpy()
# amrw_uw = pd.read_csv(f"{amrwdir}/avgprofile_5000s_ww.csv")["ww"].to_numpy()
# amrw_vv = pd.read_csv(f"{amrwdir}/avgprofile_5000s_uv.csv")["uv"].to_numpy()
# amrw_vw = pd.read_csv(f"{amrwdir}/avgprofile_5000s_uw.csv")["uw"].to_numpy()
# amrw_ww = pd.read_csv(f"{amrwdir}/avgprofile_5000s_vw.csv")["vw"].to_numpy()
# 
# fig,((ax1,ax2,ax3,ax4,ax5,ax6)) = plt.subplots(1,6,figsize=(12,6))
# ax1.plot(np.mean(uu/ustar**2,axis=(1,2)),z/zi,label="portUrb")
# ax1.plot(amrw_uu/amrw_ustar**2,amrw_z/amrw_zi,label="AMR-Wind")
# ax1.set_ylim(0,2.5)
# ax1.set_xlabel(r"u'u'/$u_\star^2$ (-)")
# ax1.set_ylabel("z/zi")
# ax1.legend()
# ax1.margins(x=0)
# ax2.plot(np.mean(uv/ustar**2,axis=(1,2)),z/zi,label="portUrb")
# ax2.plot(amrw_uv/amrw_ustar**2,amrw_z/amrw_zi,label="AMR-Wind")
# ax2.set_ylim(0,2.5)
# ax2.set_xlabel(r"u'v'/$u_\star^2$ (-)")
# ax2.set_ylabel("z/zi")
# ax2.margins(x=0)
# ax3.plot(np.mean(uw/ustar**2,axis=(1,2)),z/zi,label="portUrb")
# ax3.plot(amrw_uw/amrw_ustar**2,amrw_z/amrw_zi,label="AMR-Wind")
# ax3.set_ylim(0,2.5)
# ax3.set_xlabel(r"u'w'/$u_\star^2$ (-)")
# ax3.set_ylabel("z/zi")
# ax3.margins(x=0)
# ax4.plot(np.mean(vv/ustar**2,axis=(1,2)),z/zi,label="portUrb")
# ax4.plot(amrw_vv/amrw_ustar**2,amrw_z/amrw_zi,label="AMR-Wind")
# ax4.set_ylim(0,2.5)
# ax4.set_xlabel(r"v'v'/$u_\star^2$ (-)")
# ax4.set_ylabel("z/zi")
# ax4.margins(x=0)
# ax5.plot(np.mean(vw/ustar**2,axis=(1,2)),z/zi,label="portUrb")
# ax5.plot(amrw_vw/amrw_ustar**2,amrw_z/amrw_zi,label="AMR-Wind")
# ax5.set_ylim(0,2.5)
# ax5.set_xlabel(r"v'w'/$u_\star^2$ (-)")
# ax5.set_ylabel("z/zi")
# ax5.margins(x=0)
# ax6.plot(np.mean(ww/ustar**2,axis=(1,2)),z/zi,label="portUrb")
# ax6.plot(amrw_ww/amrw_ustar**2,amrw_z/amrw_zi,label="AMR-Wind")
# ax6.set_ylim(0,2.5)
# ax6.set_xlabel(r"w'w'/$u_\star^2$ (-)")
# ax6.set_ylabel("z/zi")
# ax6.margins(x=0)
# plt.tight_layout()
# plt.grid()
# plt.savefig("velocity_correlations_z.png")
# plt.show()
# plt.close()
# 
# 
# amrw_uT = pd.read_csv(f"{amrwdir}/avgprofile_5000s_uT.csv")["uT"].to_numpy()
# amrw_vT = pd.read_csv(f"{amrwdir}/avgprofile_5000s_vT.csv")["vT"].to_numpy()
# amrw_wT = pd.read_csv(f"{amrwdir}/avgprofile_5000s_wT.csv")["wT"].to_numpy()
# 
# fig,((ax1,ax2,ax3)) = plt.subplots(1,3,figsize=(8,6))
# ax1.plot(np.mean(uT,axis=(1,2)),z/zi,label="portUrb")
# ax1.plot(amrw_uT,amrw_z/amrw_zi     ,label="AMR-Wind")
# ax1.set_ylim(0,1.4)
# ax1.set_xlabel(r"u'$\theta$' (K m / s)")
# ax1.set_ylabel("z/zi")
# ax1.legend()
# ax1.margins(x=0)
# ax2.plot(np.mean(vT,axis=(1,2)),z/zi,label="portUrb")
# ax2.plot(amrw_vT,amrw_z/amrw_zi     ,label="AMR-Wind")
# ax2.set_ylim(0,1.4)
# ax2.set_xlabel(r"v'$\theta$' (K m / s)")
# ax2.set_ylabel("z/zi")
# ax2.margins(x=0)
# ax3.plot(np.mean(wT,axis=(1,2)),z/zi,label="portUrb")
# ax3.plot(amrw_wT,amrw_z/amrw_zi     ,label="AMR-Wind")
# ax3.set_ylim(0,1.4)
# ax3.set_xlabel(r"w'$\theta$' (K m / s)")
# ax3.set_ylabel("z/zi")
# ax3.margins(x=0)
# plt.tight_layout()
# plt.grid()
# plt.savefig("temp_correlations_z.png")
# plt.show()
# plt.close()



nc = Dataset(f"{workdir}/nrel_5mw_convective_00000006.nc","r")
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
hs   = 5
uvel  = np.array(nc["avg_u"])
vvel  = np.array(nc["avg_v"])
mag   = np.sqrt(uvel*uvel+vvel*vvel)
mag_rot = rotate(mag,30,(1,2),False)

x0 = 1.51
y0 = 2.283
z0 = 0.09
D  = .126

fig = plt.figure(figsize=(10,4))
ax = fig.gca()
X,Y = np.meshgrid((x-x0)/D,(y-y0)/D)
zind = get_ind(z,z0)
CS1 = ax.contourf(X,Y,mag_rot[zind,:,:],levels=np.arange(0,14,0.1),cmap=Colormap('coolwarm').to_mpl(),extend="both")
ax.set_title("NREL5MW AD U_h T=16,000s")
ax.axis('scaled')
ax.set_xlabel("x-location (km)")
ax.set_ylabel("y-location (km)")
ax.set_xlim(-4,10)
ax.set_ylim(-2,2)
ax.margins(x=0)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)
cbar1 = plt.colorbar(CS1,orientation="vertical",cax=cax)
cbar1.ax.tick_params(labelrotation=0)
fig.tight_layout()
plt.tight_layout()
plt.show()
plt.close()

fig = plt.figure(figsize=(10,4))
ax = fig.gca()
X,Z = np.meshgrid((x-x0)/D,(z-z0)/D)
yind = get_ind(y,y0)
CS1 = ax.contourf(X,Z,mag_rot[:,yind,:],levels=np.arange(0,14,0.1),cmap=Colormap('coolwarm').to_mpl(),extend="both")
ax.set_title("NREL5MW AD U_h T=16,000s")
ax.axis('scaled')
ax.set_xlabel("x-location (km)")
ax.set_ylabel("z-location (km)")
ax.set_xlim(-4,10)
ax.set_ylim(-.71,1.5)
ax.margins(x=0)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)
cbar1 = plt.colorbar(CS1,orientation="vertical",cax=cax)
cbar1.ax.tick_params(labelrotation=0)
fig.tight_layout()
plt.tight_layout()
plt.show()
plt.close()

fig = plt.figure(figsize=(5,6))
ax = fig.gca()
xind2 = get_ind((x-x0)/D,2)
xind4 = get_ind((x-x0)/D,4)
xind6 = get_ind((x-x0)/D,6)
xind8 = get_ind((x-x0)/D,8)
ax.plot(mag_rot[zind,:,xind2]/11.4,(y-y0)/D,label="portUrb (x/D=2)")
ax.plot(mag_rot[zind,:,xind4]/11.4,(y-y0)/D,label="portUrb (x/D=4)")
ax.plot(mag_rot[zind,:,xind6]/11.4,(y-y0)/D,label="portUrb (x/D=6)")
ax.plot(mag_rot[zind,:,xind8]/11.4,(y-y0)/D,label="portUrb (x/D=8)")
ax.set_ylim(-2.1,2.1)
ax.set_xlim(0.6,1.1)
ax.set_xlabel("u/u_inf")
ax.set_ylabel("y/D")
ax.legend()
ax.margins(x=0)
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()
plt.close()

# uref = 11.4
# zref = 90
# z0   = 0.01
# uinf = uref*np.log((z*1000+z0)/z0)/np.log((zref+z0)/z0)

fig = plt.figure(figsize=(5,6))
ax = fig.gca()
ax.plot(mag_rot[:,yind,xind2]/11.4,(z-z0)/D,label="portUrb (x/D=2)")
ax.plot(mag_rot[:,yind,xind4]/11.4,(z-z0)/D,label="portUrb (x/D=4)")
ax.plot(mag_rot[:,yind,xind6]/11.4,(z-z0)/D,label="portUrb (x/D=6)")
ax.plot(mag_rot[:,yind,xind8]/11.4,(z-z0)/D,label="portUrb (x/D=8)")
ax.set_ylim(-.72,1.35)
ax.set_xlim(0.45,1.12)
ax.set_xlabel("u/u_inf")
ax.set_ylabel("z/D")
ax.legend()
ax.margins(x=0)
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()
plt.close()

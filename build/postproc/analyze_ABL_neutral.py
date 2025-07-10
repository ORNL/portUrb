from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmap import Colormap
import xarray

workdir = "/lustre/orion/stf006/scratch/imn/portUrb/build"

def spectra(T,dx = 1) :
  spd  = np.abs( np.fft.rfft(T[0,0,:]) )**2
  freq = np.fft.rfftfreq(len(T[0,0,:]))
  spd = 0
  for k in range(T.shape[0]) :
    for j in range(T.shape[1]) :
      spd += np.abs( np.fft.rfft(T[k,j,:]) )**2
      spd += np.abs( np.fft.rfft(T[k,:,j]) )**2
  spd /= T.shape[0]*T.shape[1]*2
  return freq*2*2*np.pi/(2*dx) , spd


def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))


nc = Dataset(f"{workdir}/ABL_neutral_00000020.nc","r")
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
uvel  = np.array(nc["uvel"][:,:,:])
vvel  = np.array(nc["vvel"][:,:,:])
wvel  = np.array(nc["wvel"][:,:,:])
mag   = np.sqrt(uvel*uvel+vvel*vvel+wvel*wvel)


umin  = np.min     (mag     ,axis=(1,2))
umax  = np.max     (mag     ,axis=(1,2))
umean = np.mean    (mag     ,axis=(1,2))
uq1   = np.quantile(mag,0.25,axis=(1,2))
uq3   = np.quantile(mag,0.75,axis=(1,2))
roughness = 0.1
uref = 10
href = 500
u_mo  = uref*np.log((z*1000+roughness)/roughness)/np.log((href+roughness)/roughness);
z2 = get_ind(z,0.75)
fig = plt.figure(figsize=(6,4))
ax = fig.gca()
ax.fill_betweenx(z[:z2],umin [:z2],umax [:z2],color="lightskyblue",label="[min,max]")
ax.fill_betweenx(z[:z2],uq1  [:z2],uq3  [:z2],color="deepskyblue",label="[Q1,Q3]")
ax.plot         (umean[:z2],z[:z2],color="black",label="mean")
ax.plot         (u_mo [:z2],z[:z2],color="black",linestyle="--",label=r"Log law")
ax.set_xlabel("velocity magnitude (m/s)")
ax.set_ylabel("z-location (km)")
ax.set_yscale("log")
ax.legend(loc="upper left")
ax.set_xlim(left=0)
ax.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_uvel_height.png",dpi=600)
plt.show()
plt.close()


umean = np.mean(uvel,axis=(1,2))
vmean = np.mean(vvel,axis=(1,2))
wmean = np.mean(wvel,axis=(1,2))
up = uvel - umean[:,np.newaxis,np.newaxis]
vp = vvel - vmean[:,np.newaxis,np.newaxis]
wp = wvel - wmean[:,np.newaxis,np.newaxis]
up_wp = np.abs(up*wp)
up_wp_min  = np.min (up_wp,axis=(1,2))
up_wp_max  = np.max (up_wp,axis=(1,2))
up_wp_mean = np.mean(up_wp,axis=(1,2))
up_wp_q1   = np.quantile(up_wp,0.25,axis=(1,2))
up_wp_q3   = np.quantile(up_wp,0.75,axis=(1,2))

z2 = get_ind(z,0.75)
fig = plt.figure(figsize=(6,4))
ax = fig.gca()
ax.fill_betweenx(z[:z2],up_wp_min [:z2],up_wp_max [:z2],color="lightskyblue",label="[min,max]")
ax.fill_betweenx(z[:z2],up_wp_q1  [:z2],up_wp_q3  [:z2],color="deepskyblue",label="[Q1,Q3]")
ax.plot         (up_wp_mean[:z2],z[:z2],color="black",label="mean")
ax.set_xlabel(r"velocity correlation $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
ax.legend()
ax.margins(x=0)
plt.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_up_wp_height.png",dpi=600)
plt.show()
plt.close()

rho = np.array(nc["density_dry"][:,:,:])
tke = rho*( (up*up + vp*vp + wp*wp)/2 + np.array(nc.variables["TKE"][:,:,:]) )
tke_min  = np.min (tke,axis=(1,2))
tke_max  = np.max (tke,axis=(1,2))
tke_mean = np.mean(tke,axis=(1,2))
tke_q1   = np.quantile(tke,0.25,axis=(1,2))
tke_q3   = np.quantile(tke,0.75,axis=(1,2))

z2 = get_ind(z,0.75)
fig = plt.figure(figsize=(6,4))
ax = fig.gca()
ax.fill_betweenx(z[:z2],tke_min [:z2],tke_max [:z2],color="lightskyblue",label="[min,max]")
ax.fill_betweenx(z[:z2],tke_q1  [:z2],tke_q3  [:z2],color="deepskyblue",label="[Q1,Q3]")
ax.plot         (tke_mean[:z2],z[:z2],color="black",label="mean")
ax.set_xlabel(r"TKE $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
plt.legend()
ax.margins(x=0)
plt.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_tke_height.png",dpi=600)
plt.show()
plt.close()


kind = get_ind(z,.0786)
print(z[kind])
freq,spd1 = spectra(mag[kind:kind+1,:,:],dx=dx)
fig = plt.figure(figsize=(6,3))
ax = fig.gca()
ax.plot(freq,spd1,label="Wind Speed spectra")
ax.plot(freq[1:],2e5*freq[1:]**(-5/3),label=r"$f^{-5/3}$")
ax.vlines(2*np.pi/(2 *dx),1.e-3,1.e3,linestyle="--",color="red")
ax.vlines(2*np.pi/(4 *dx),1.e-3,1.e3,linestyle="--",color="red")
ax.vlines(2*np.pi/(8 *dx),1.e-3,1.e3,linestyle="--",color="red")
ax.vlines(2*np.pi/(16*dx),1.e-3,1.e3,linestyle="--",color="red")
ax.text(0.9*2*np.pi/(2 *dx),2.e3,"$2  \Delta x$")
ax.text(0.9*2*np.pi/(4 *dx),2.e3,"$4  \Delta x$")
ax.text(0.9*2*np.pi/(8 *dx),2.e3,"$8  \Delta x$")
ax.text(0.9*2*np.pi/(16*dx),2.e3,"$16 \Delta x$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Frequency")
ax.set_ylabel("Spectral Power")
ax.legend(loc='lower left')
ax.set_ylim(top=1.e6)
ax.margins(x=0)
plt.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_spectra.png",dpi=600)
plt.show()
plt.close()


fig = plt.figure(figsize=(6,6))
ax = fig.gca()
X,Y = np.meshgrid(x,y)
print(z[get_ind(z,.0786)])
mn  = np.mean(mag[get_ind(z,.0786),:,:])
std = np.std (mag[get_ind(z,.0786),:,:])
t1 = 4
t2 = 12
CS = ax.contourf(X,Y,mag[get_ind(z,.0786),:,:],levels=np.arange(t1,t2,(t2-t1)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax.axis('scaled')
ax.set_xlabel("x-location (km)")
ax.set_ylabel("y-location (km)")
ax.margins(x=0)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="4%", pad=0.5)
plt.colorbar(CS,orientation="horizontal",cax=cax)
plt.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_contour_xy.png",dpi=600)
plt.show()
plt.close()


fig = plt.figure(figsize=(8,4))
ax = fig.gca()
z2 = get_ind(z,0.7)
yind = int(ny/2)
X,Z = np.meshgrid(x,z[:z2])
mn  = np.mean(mag[:z2,yind,:])
std = np.std (mag[:z2,yind,:])
t1 = 4
t2 = 12
CS = ax.contourf(X,Z,mag[:z2,yind,:],levels=np.arange(t1,t2,(t2-t1)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax.axis('scaled')
ax.set_xlabel("x-location (km)")
ax.set_ylabel("z-location (km)")
ax.margins(x=0)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="4%", pad=0.5)
plt.colorbar(CS,orientation="horizontal",cax=cax)
plt.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_contour_xz.png",dpi=600)
plt.show()
plt.close()


u0  = np.mean(np.array(Dataset(f"{workdir}/ABL_neutral_00000000.nc","r")["uvel"]),axis=(1,2))
u8  = np.mean(np.array(Dataset(f"{workdir}/ABL_neutral_00000016.nc","r")["uvel"]),axis=(1,2))
u10 = np.mean(np.array(Dataset(f"{workdir}/ABL_neutral_00000020.nc","r")["uvel"]),axis=(1,2))
z2 = get_ind(z,0.75)
fig = plt.figure(figsize=(4,6))
ax = fig.gca()
ax.plot(u0 [:z2],z[:z2],color="black",linestyle="--",label="t=0 hr")
ax.plot(u8 [:z2],z[:z2],color="red",label="t=8 hr")
ax.plot(u10[:z2],z[:z2],color="blue",label="t=10hr")
ax.set_xlabel("velocity (m/s)")
ax.set_ylabel("z-location (km)")
ax.legend(loc="upper left")
ax.set_xlim(left=0)
ax.margins(x=0)
plt.tight_layout()
plt.grid()
plt.savefig("ABL_neutral_uvel_height_times.png",dpi=600)
plt.show()
plt.close()


u0  = np.mean(np.array(Dataset(f"{workdir}/ABL_neutral_00000000.nc","r")["vvel"]),axis=(1,2))
u8  = np.mean(np.array(Dataset(f"{workdir}/ABL_neutral_00000016.nc","r")["vvel"]),axis=(1,2))
u10 = np.mean(np.array(Dataset(f"{workdir}/ABL_neutral_00000020.nc","r")["vvel"]),axis=(1,2))
z2 = get_ind(z,0.75)
fig = plt.figure(figsize=(4,6))
ax = fig.gca()
ax.plot(u0 [:z2],z[:z2],color="black",linestyle="--",label="t=0 hr")
ax.plot(u8 [:z2],z[:z2],color="red",label="t=8 hr")
ax.plot(u10[:z2],z[:z2],color="blue",label="t=10hr")
ax.set_xlabel("velocity (m/s)")
ax.set_ylabel("z-location (km)")
ax.legend(loc="upper right")
ax.set_xlim(left=-0.2)
ax.margins(x=0)
plt.tight_layout()
plt.grid()
plt.savefig("ABL_neutral_vvel_height_times.png",dpi=600)
plt.show()
plt.close()


nc0  = Dataset(f"{workdir}/ABL_neutral_00000000.nc","r")
nc8  = Dataset(f"{workdir}/ABL_neutral_00000016.nc","r")
nc10 = Dataset(f"{workdir}/ABL_neutral_00000020.nc","r")
hs = 5
u0  = np.mean(np.array(nc0 ["theta_pert"])+np.array(nc0 ["hy_theta_cells"])[hs:hs+nz,np.newaxis,np.newaxis],axis=(1,2))
u8  = np.mean(np.array(nc8 ["theta_pert"])+np.array(nc8 ["hy_theta_cells"])[hs:hs+nz,np.newaxis,np.newaxis],axis=(1,2))
u10 = np.mean(np.array(nc10["theta_pert"])+np.array(nc10["hy_theta_cells"])[hs:hs+nz,np.newaxis,np.newaxis],axis=(1,2))
z2 = get_ind(z,0.75)
fig = plt.figure(figsize=(4,6))
ax = fig.gca()
ax.plot(u0 [:z2],z[:z2],color="black",linestyle="--",label="t=0 hr")
ax.plot(u8 [:z2],z[:z2],color="red",label="t=8 hr")
ax.plot(u10[:z2],z[:z2],color="blue",label="t=10hr")
ax.set_xlabel("Potential Temperature (K)")
ax.set_ylabel("z-location (km)")
ax.legend(loc="upper left")
ax.set_xlim(left=299,right=313)
ax.margins(x=0)
plt.tight_layout()
plt.grid()
plt.savefig("ABL_neutral_theta_height_times.png",dpi=600)
plt.show()
plt.close()

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmap import Colormap
import re

workdir = "/lustre/orion/stf006/scratch/imn/portUrb/build"
files = [f"{workdir}/ABL_neutral_buoy-rhop_press-orig_cs-350",
         f"{workdir}/ABL_neutral_buoy-thetap_press-orig_cs-350",
         f"{workdir}/ABL_neutral_buoy-thetap_press-rsst_cs-350",
         f"{workdir}/ABL_neutral_buoy-thetap_press-rsst_cs-100",
         f"{workdir}/ABL_neutral_buoy-thetap_press-rsst_cs-50",
         f"{workdir}/ABL_neutral_buoy-thetap_press-rsst_cs-25",
         f"{workdir}/ABL_neutral_buoy-thetap_press-rsst_cs-20",
         f"{workdir}/ABL_neutral_buoy-thetap_press-rsst_cs-15",
         f"{workdir}/ABL_neutral_buoy-thetap_press-rsst_cs-12",
         f"{workdir}/ABL_neutral_buoy-thetap_press-rsst_cs-10",]
buoy   = np.array([("rhop" if "buoy-rhop" in f else "thetap") for f in files])
press  = np.array([("orig" if "press-orig" in f else "rsst") for f in files])
cs     = np.array([int(re.search(r'cs-(\d+)', f).group(1)) for f in files])
labels = np.array([f"{press[i]}-{buoy[i]}-{cs[i]}" for i in range(len(files))])
colors = ['black','#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', # Black, Red, Green, Yellow, Blue, Orange
          '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', # Purple, Cyan, Magenta, Lime, Pink
          '#469990', '#dcbeff',]                                 # Teal, Lavender
styles = ["-" for i in range(len(files))]
nexp = len(files)
times = range(13,21)

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
hs      = 5

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

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  nc      = Dataset(f"{files[j]}_00000010.nc","r")
  z       = np.array(nc["z"][:])/1000
  nz      = len(z)
  pp      = np.array(nc["density_dry"][:,:,:])*R_d*np.array(nc["temperature"][:,:,:]) - np.array(nc["hy_pressure_cells"][hs:hs+nz])[:,np.newaxis,np.newaxis]
  rhopcs2 = (np.array(nc["density_dry"][:,:,:])-np.array(nc["hy_dens_cells"][hs:hs+nz])[:,np.newaxis,np.newaxis])*cs[j]**2
  pert    = pp if j < 2 else rhopcs2
  pert    = np.mean(pert,axis=(1,2))
  pert    = pert - np.mean(pert)
  z2      = get_ind(z,1.25)
  ax.plot(pert[:z2],z[:z2],color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel("pressure perturbation (Pa)")
ax.set_ylabel("z-location (km)")
ax.legend(loc="upper right")
# ax.set_xlim(left=0)
ax.margins(x=0)
plt.grid()
plt.tight_layout()
plt.savefig("ABL_neutral_pp_rhop_allcs_5hr.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  nc      = Dataset(f"{files[j]}_00000020.nc","r")
  z       = np.array(nc["z"][:])/1000
  nz      = len(z)
  pp      = np.array(nc["density_dry"][:,:,:])*R_d*np.array(nc["temperature"][:,:,:]) - np.array(nc["hy_pressure_cells"][hs:hs+nz])[:,np.newaxis,np.newaxis]
  rhopcs2 = (np.array(nc["density_dry"][:,:,:])-np.array(nc["hy_dens_cells"][hs:hs+nz])[:,np.newaxis,np.newaxis])*cs[j]**2
  pert    = pp if j < 2 else rhopcs2
  pert    = np.mean(pert,axis=(1,2))
  pert    = pert - np.mean(pert)
  z2      = get_ind(z,1.25)
  ax.plot(pert[:z2],z[:z2],color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel("pressure perturbation (Pa)")
ax.set_ylabel("z-location (km)")
ax.legend(loc="upper right")
# ax.set_xlim(left=0)
ax.margins(x=0)
plt.grid()
plt.tight_layout()
plt.savefig("ABL_neutral_pp_rhop_allcs_10hr.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  nc   = Dataset(f"{files[j]}_00000020.nc","r")
  z    = np.array(nc["z"][:])/1000
  uvel = np.array(nc["uvel"][:,:,:])
  vvel = np.array(nc["vvel"][:,:,:])
  wvel = np.array(nc["wvel"][:,:,:])
  mag  = np.sqrt(uvel*uvel+vvel*vvel+wvel*wvel)
  umean = np.mean(mag,axis=(1,2))
  roughness = 0.1
  uref = 10
  href = 500
  u_mo  = uref*np.log((z*1000+roughness)/roughness)/np.log((href+roughness)/roughness);
  z2 = get_ind(z,0.75)
  ax.plot(umean[:z2],z[:z2],color=colors[j],label=labels[j],linestyle=styles[j])
ax.plot(u_mo [:z2],z[:z2],color="black",linestyle="--",label=r"Log law")
ax.set_xlabel("velocity magnitude (m/s)")
ax.set_ylabel("z-location (km)")
ax.set_yscale("log")
ax.legend(loc="upper left")
ax.set_xlim(left=0)
ax.margins(x=0)
plt.grid()
plt.tight_layout()
plt.savefig("ABL_neutral_uvel_height.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  for k in range(len(times)) :
    nc   = Dataset(f"{files[j]}_{times[k]:08d}.nc","r")
    z    = np.array(nc["z"][:])/1000
    uvel = np.array(nc["uvel"][:,:,:])
    vvel = np.array(nc["vvel"][:,:,:])
    wvel = np.array(nc["wvel"][:,:,:])
    up = uvel - np.mean(uvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp = vvel - np.mean(vvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp = wvel - np.mean(wvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    up_up = up*up if k==0 else up_up+up*up
  up_up /= len(times)
  up_up_mean = np.mean(up_up,axis=(1,2))
  z2 = get_ind(z,0.75)
  ax.plot(up_up_mean[:z2],z[:z2],color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel(r"u'u' $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
ax.legend()
ax.margins(x=0)
plt.grid()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_up_up_height.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  for k in range(len(times)) :
    nc   = Dataset(f"{files[j]}_{times[k]:08d}.nc","r")
    z    = np.array(nc["z"][:])/1000
    uvel = np.array(nc["uvel"][:,:,:])
    vvel = np.array(nc["vvel"][:,:,:])
    wvel = np.array(nc["wvel"][:,:,:])
    up = uvel - np.mean(uvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp = vvel - np.mean(vvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp = wvel - np.mean(wvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    up_vp = up*vp if k==0 else up_vp+up*vp
  up_vp /= len(times)
  up_vp_mean = np.mean(up_vp,axis=(1,2))
  z2 = get_ind(z,0.75)
  ax.plot(up_vp_mean[:z2],z[:z2],color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel(r"u'v' $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
ax.legend()
ax.margins(x=0)
plt.grid()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_up_vp_height.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  for k in range(len(times)) :
    nc   = Dataset(f"{files[j]}_{times[k]:08d}.nc","r")
    z    = np.array(nc["z"][:])/1000
    uvel = np.array(nc["uvel"][:,:,:])
    vvel = np.array(nc["vvel"][:,:,:])
    wvel = np.array(nc["wvel"][:,:,:])
    up = uvel - np.mean(uvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp = vvel - np.mean(vvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp = wvel - np.mean(wvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    up_wp = up*wp if k==0 else up_wp+up*wp
  up_wp /= len(times)
  up_wp_mean = np.mean(up_wp,axis=(1,2))
  z2 = get_ind(z,0.75)
  ax.plot(up_wp_mean[:z2],z[:z2],color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel(r"u'w' $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
ax.legend()
ax.margins(x=0)
plt.grid()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_up_wp_height.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  for k in range(len(times)) :
    nc   = Dataset(f"{files[j]}_{times[k]:08d}.nc","r")
    z    = np.array(nc["z"][:])/1000
    uvel = np.array(nc["uvel"][:,:,:])
    vvel = np.array(nc["vvel"][:,:,:])
    wvel = np.array(nc["wvel"][:,:,:])
    up = uvel - np.mean(uvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp = vvel - np.mean(vvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp = wvel - np.mean(wvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp_vp = vp*vp if k==0 else vp_vp+vp*vp
  vp_vp /= len(times)
  vp_vp_mean = np.mean(vp_vp,axis=(1,2))
  z2 = get_ind(z,0.75)
  ax.plot(vp_vp_mean[:z2],z[:z2],color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel(r"v'v' $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
ax.legend()
ax.margins(x=0)
plt.grid()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_vp_vp_height.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  for k in range(len(times)) :
    nc   = Dataset(f"{files[j]}_{times[k]:08d}.nc","r")
    z    = np.array(nc["z"][:])/1000
    uvel = np.array(nc["uvel"][:,:,:])
    vvel = np.array(nc["vvel"][:,:,:])
    wvel = np.array(nc["wvel"][:,:,:])
    up = uvel - np.mean(uvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp = vvel - np.mean(vvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp = wvel - np.mean(wvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp_wp = vp*wp if k==0 else vp_wp+vp*wp
  vp_wp /= len(times)
  vp_wp_mean = np.mean(vp_wp,axis=(1,2))
  z2 = get_ind(z,0.75)
  ax.plot(vp_wp_mean[:z2],z[:z2],color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel(r"v'w' $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
ax.legend()
ax.margins(x=0)
plt.grid()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_vp_wp_height.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  for k in range(len(times)) :
    nc   = Dataset(f"{files[j]}_{times[k]:08d}.nc","r")
    z    = np.array(nc["z"][:])/1000
    uvel = np.array(nc["uvel"][:,:,:])
    vvel = np.array(nc["vvel"][:,:,:])
    wvel = np.array(nc["wvel"][:,:,:])
    up = uvel - np.mean(uvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp = vvel - np.mean(vvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp = wvel - np.mean(wvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp_wp = wp*wp if k==0 else wp_wp+wp*wp
  wp_wp /= len(times)
  wp_wp_mean = np.mean(wp_wp,axis=(1,2))
  z2 = get_ind(z,0.75)
  ax.plot(wp_wp_mean[:z2],z[:z2],color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel(r"w'w' $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
ax.legend()
ax.margins(x=0)
plt.grid()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_wp_wp_height.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  for k in range(len(times)) :
    nc   = Dataset(f"{files[j]}_{times[k]:08d}.nc","r")
    z    = np.array(nc["z"][:])/1000
    rho  = np.array(nc["density_dry"][:,:,:])
    uvel = np.array(nc["uvel"][:,:,:])
    vvel = np.array(nc["vvel"][:,:,:])
    wvel = np.array(nc["wvel"][:,:,:])
    T    = np.array(nc["temperature"][:,:,:])
    pres = rho*R_d*T
    thet = np.pow(pres/C0,1/gamma_d)/rho - np.array(nc["hy_theta_cells"][hs:hs+nz])[:,np.newaxis,np.newaxis]
    up = uvel - np.mean(uvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp = vvel - np.mean(vvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp = wvel - np.mean(wvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    tp = thet - np.mean(thet,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp_tp = wp*tp if k==0 else wp_tp+wp*tp
  wp_tp /= len(times)
  wp_tp_mean = np.mean(wp_tp,axis=(1,2))
  z2 = get_ind(z,0.75)
  ax.plot(wp_tp_mean[:z2],z[:z2],color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel(r"w'$\theta$' $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
ax.legend()
ax.margins(x=0)
plt.grid()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_wp_tp_height.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  for k in range(len(times)) :
    nc   = Dataset(f"{files[j]}_{times[k]:08d}.nc","r")
    z    = np.array(nc["z"][:])/1000
    rho  = np.array(nc["density_dry"][:,:,:])
    uvel = np.array(nc["uvel"][:,:,:])
    vvel = np.array(nc["vvel"][:,:,:])
    wvel = np.array(nc["wvel"][:,:,:])
    up = uvel - np.mean(uvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp = vvel - np.mean(vvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp = wvel - np.mean(wvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    tkeloc = rho*( (up*up + vp*vp + wp*wp)/2 + np.array(nc.variables["TKE"][:,:,:]) )
    tke = tkeloc if k==0 else tke+tkeloc
  tke /= len(times)
  tke_mean = np.mean(tke,axis=(1,2))
  z2 = get_ind(z,0.75)
  ax.plot(tke_mean[:z2],z[:z2],color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel(r"TKE $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
ax.margins(x=0)
plt.legend()
plt.grid()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_tke_height.png",dpi=600)
plt.show()
plt.close()

k1 = get_ind(z,.2)
k2 = get_ind(z,.2)
fig = plt.figure(figsize=(6,4))
ax = fig.gca()
for j in range(nexp) :
  nc   = Dataset(f"{files[j]}_00000020.nc","r")
  z    = np.array(nc["z"][:])/1000
  dx   = z[1]-z[0]
  rho  = np.array(nc["density_dry"][:,:,:])
  uvel = np.array(nc["uvel"][:,:,:])
  vvel = np.array(nc["vvel"][:,:,:])
  wvel = np.array(nc["wvel"][:,:,:])
  mag  = np.sqrt(uvel*uvel+vvel*vvel+wvel*wvel)
  freq,spd1 = spectra(mag[k1:k2+1,:,:],dx=dx)
  ax.plot(freq,spd1,color=colors[j],label=labels[j],linestyle=styles[j])
ax.plot(freq[1:],5e4*freq[1:]**(-5/3),label=r"$f^{-5/3}$",linestyle=":",color="black")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Frequency")
ax.set_ylabel("Spectral Power")
ax.legend(loc='lower left')
ax.set_ylim(top=1.e6)
ax.margins(x=0)
plt.grid()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_neutral_spectra.png",dpi=600)
plt.show()
plt.close()


fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  u0  = np.mean(np.array(Dataset(f"{files[j]}_00000000.nc","r")["uvel"][:,:,:]),axis=(1,2))
  u10 = np.mean(np.array(Dataset(f"{files[j]}_00000020.nc","r")["uvel"][:,:,:]),axis=(1,2))
  z2 = get_ind(z,0.75)
  ax.plot(u10[:z2],z[:z2],color=colors[j],label=labels[j],linestyle=styles[j])
ax.plot(u0 [:z2],z[:z2],color="black",linestyle="--",label="t=0 hr")
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


fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  u0  = np.mean(np.array(Dataset(f"{files[j]}_00000000.nc","r")["vvel"][:,:,:]),axis=(1,2))
  u10 = np.mean(np.array(Dataset(f"{files[j]}_00000020.nc","r")["vvel"][:,:,:]),axis=(1,2))
  z2 = get_ind(z,0.75)
  ax.plot(u10[:z2],z[:z2],color=colors[j],label=labels[j],linestyle=styles[j])
ax.plot(u0 [:z2],z[:z2],color="black",linestyle="--",label="t=0 hr")
ax.set_xlabel("velocity (m/s)")
ax.set_ylabel("z-location (km)")
ax.legend(loc="lower left")
ax.set_xlim(left=-0.2)
ax.margins(x=0)
plt.tight_layout()
plt.grid()
plt.savefig("ABL_neutral_vvel_height_times.png",dpi=600)
plt.show()
plt.close()


fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  nc0  = Dataset(f"{files[j]}_00000000.nc","r")
  nc10 = Dataset(f"{files[j]}_00000020.nc","r")
  hs = 5
  nz = len(z)
  rho  = np.array(nc0["density_dry"][:,:,:])
  T    = np.array(nc0["temperature"][:,:,:])
  pres = rho*R_d*T
  u0   = np.mean(np.pow(pres/C0,1/gamma_d)/rho,axis=(1,2))
  rho  = np.array(nc10["density_dry"][:,:,:])
  T    = np.array(nc10["temperature"][:,:,:])
  pres = rho*R_d*T
  u10  = np.mean(np.pow(pres/C0,1/gamma_d)/rho,axis=(1,2))
  z2 = get_ind(z,0.75)
  ax.plot(u10[:z2],z[:z2],color=colors[j],label=labels[j],linestyle=styles[j])
ax.plot(u0 [:z2],z[:z2],color="black",linestyle="--",label="t=0 hr")
ax.set_xlabel("Potential Temperature (K)")
ax.set_ylabel("z-location (km)")
ax.legend(loc="lower right")
ax.set_xlim(left=299,right=313)
ax.margins(x=0)
plt.tight_layout()
plt.grid()
plt.savefig("ABL_neutral_theta_height_times.png",dpi=600)
plt.show()
plt.close()


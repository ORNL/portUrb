from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmap import Colormap
import xarray

workdir = "/lustre/storm/nwp501/scratch/imn/rsst_paper/building"
files    = [f"{workdir}/building_orig_rho_350",
            f"{workdir}/building_orig_theta_350",
            f"{workdir}/building_rss_350",
            f"{workdir}/building_rss_100",
            f"{workdir}/building_rss_50",
            f"{workdir}/building_rss_30",
            f"{workdir}/building_rss_20"]
cs     = [350,350,350,100,50,30,20]
labels = ["ORIG-RHO_350","ORIG-THETA_350","RSS_350","RSS_100","RSS_50","RSS_30","RSS_20"]
colors = ["black","red","green","blue","cyan","magenta","orange"]
styles = ["-","-","-","-","-","-","-"]
nexp = 7
times = [11,12,13,14,15,16,17,18,19,20]

def get_ind(arr,val) :
    return np.argmin(np.abs(arr-val))

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  nc   = Dataset(f"{files[j]}_00000010.nc","r")
  z    = np.array(nc["z"])/1000
  pert = np.array(nc["pressure_pert"][:,:,:]) if j < 2 else cs[j]*cs[j]*np.array(nc["density_pert"][:,:,:])
  pert = np.ma.array(data=pert,mask=np.array(nc["immersed_proportion"][:,:,:]) > 0)
  pert = np.mean(pert,axis=(1,2))
  pert = pert - np.mean(pert)
  ax.plot(pert,z,color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel("pressure perturbation (Pa)")
ax.set_ylabel("z-location (km)")
ax.legend(loc="upper left")
ax.margins(x=0)
plt.grid()
plt.tight_layout()
plt.savefig("building_pp_rhop_allcs_5hr.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  nc   = Dataset(f"{files[j]}_00000020.nc","r")
  z    = np.array(nc["z"])/1000
  pert = np.array(nc["pressure_pert"][:,:,:]) if j < 2 else cs[j]*cs[j]*np.array(nc["density_pert"][:,:,:])
  pert = np.ma.array(data=pert,mask=np.array(nc["immersed_proportion"][:,:,:]) > 0)
  pert = np.mean(pert,axis=(1,2))
  pert = pert - np.mean(pert)
  ax.plot(pert,z,color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel("pressure perturbation (Pa)")
ax.set_ylabel("z-location (km)")
ax.legend(loc="upper left")
ax.margins(x=0)
plt.grid()
plt.tight_layout()
plt.savefig("building_pp_rhop_allcs_10hr.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  nc  = Dataset(f"{files[j]}_00000020.nc","r")
  z   = np.array(nc["z"])/1000
  u   = np.array(nc["uvel"][:,:,:])
  v   = np.array(nc["vvel"][:,:,:])
  w   = np.array(nc["wvel"][:,:,:])
  mag = np.ma.array(data=np.sqrt(u*u+v*v+w*w),mask=np.array(nc["immersed_proportion"][:,:,:]) > 0)
  mag = np.mean(mag,axis=(1,2))
  mag = mag - np.mean(mag)
  ax.plot(mag,z,color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel("velocity magnitude (m/s)")
ax.set_ylabel("z-location (km)")
ax.legend(loc="upper left")
ax.margins(x=0)
plt.grid()
plt.tight_layout()
plt.savefig("building_uvel_10hr.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  for k in range(len(times)) :
    nc   = Dataset(f"{files[j]}_{times[k]:08d}.nc","r")
    z    = np.array(nc["z"])/1000
    imm  = np.array(nc["immersed_proportion"][:,:,:]) > 0
    uvel = np.ma.array(data=nc["uvel"][:,:,:],mask=imm)
    vvel = np.ma.array(data=nc["vvel"][:,:,:],mask=imm)
    wvel = np.ma.array(data=nc["wvel"][:,:,:],mask=imm)
    up = uvel - np.mean(uvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp = vvel - np.mean(vvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp = wvel - np.mean(wvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    up_up = up*up if k==0 else up_up+up*up
  up_up /= len(times)
  up_up_mean = np.mean(up_up,axis=(1,2))
  ax.plot(up_up_mean,z,color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel(r"u'u' $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
ax.legend()
ax.margins(x=0)
plt.grid()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("building_up_up_height.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  for k in range(len(times)) :
    nc   = Dataset(f"{files[j]}_{times[k]:08d}.nc","r")
    z    = np.array(nc["z"])/1000
    imm  = np.array(nc["immersed_proportion"][:,:,:]) > 0
    uvel = np.ma.array(data=nc["uvel"][:,:,:],mask=imm)
    vvel = np.ma.array(data=nc["vvel"][:,:,:],mask=imm)
    wvel = np.ma.array(data=nc["wvel"][:,:,:],mask=imm)
    up = uvel - np.mean(uvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp = vvel - np.mean(vvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp = wvel - np.mean(wvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    up_vp = up*vp if k==0 else up_vp+up*vp
  up_vp /= len(times)
  up_vp_mean = np.mean(up_vp,axis=(1,2))
  ax.plot(up_vp_mean,z,color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel(r"u'v' $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
ax.legend(loc="upper left")
ax.margins(x=0)
plt.grid()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("building_up_vp_height.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  for k in range(len(times)) :
    nc   = Dataset(f"{files[j]}_{times[k]:08d}.nc","r")
    z    = np.array(nc["z"])/1000
    imm  = np.array(nc["immersed_proportion"][:,:,:]) > 0
    uvel = np.ma.array(data=nc["uvel"][:,:,:],mask=imm)
    vvel = np.ma.array(data=nc["vvel"][:,:,:],mask=imm)
    wvel = np.ma.array(data=nc["wvel"][:,:,:],mask=imm)
    up = uvel - np.mean(uvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp = vvel - np.mean(vvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp = wvel - np.mean(wvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    up_wp = up*wp if k==0 else up_wp+up*wp
  up_wp /= len(times)
  up_wp_mean = np.mean(up_wp,axis=(1,2))
  ax.plot(up_wp_mean,z,color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel(r"u'w' $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
ax.legend()
ax.margins(x=0)
plt.grid()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("building_up_wp_height.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  for k in range(len(times)) :
    nc   = Dataset(f"{files[j]}_{times[k]:08d}.nc","r")
    z    = np.array(nc["z"])/1000
    imm  = np.array(nc["immersed_proportion"][:,:,:]) > 0
    uvel = np.ma.array(data=nc["uvel"][:,:,:],mask=imm)
    vvel = np.ma.array(data=nc["vvel"][:,:,:],mask=imm)
    wvel = np.ma.array(data=nc["wvel"][:,:,:],mask=imm)
    up = uvel - np.mean(uvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp = vvel - np.mean(vvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp = wvel - np.mean(wvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp_vp = vp*vp if k==0 else vp_vp+vp*vp
  vp_vp /= len(times)
  vp_vp_mean = np.mean(vp_vp,axis=(1,2))
  ax.plot(vp_vp_mean,z,color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel(r"v'v' $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
ax.legend()
ax.margins(x=0)
plt.grid()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("building_vp_vp_height.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  for k in range(len(times)) :
    nc   = Dataset(f"{files[j]}_{times[k]:08d}.nc","r")
    z    = np.array(nc["z"])/1000
    imm  = np.array(nc["immersed_proportion"][:,:,:]) > 0
    uvel = np.ma.array(data=nc["uvel"][:,:,:],mask=imm)
    vvel = np.ma.array(data=nc["vvel"][:,:,:],mask=imm)
    wvel = np.ma.array(data=nc["wvel"][:,:,:],mask=imm)
    up = uvel - np.mean(uvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp = vvel - np.mean(vvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp = wvel - np.mean(wvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp_wp = vp*wp if k==0 else vp_wp+vp*wp
  vp_wp /= len(times)
  vp_wp_mean = np.mean(vp_wp,axis=(1,2))
  ax.plot(vp_wp_mean,z,color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel(r"v'w' $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
ax.legend(loc="upper left")
ax.margins(x=0)
plt.grid()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("building_vp_wp_height.png",dpi=600)
plt.show()
plt.close()

fig = plt.figure(figsize=(4,6))
ax = fig.gca()
for j in range(nexp) :
  for k in range(len(times)) :
    nc   = Dataset(f"{files[j]}_{times[k]:08d}.nc","r")
    z    = np.array(nc["z"])/1000
    imm  = np.array(nc["immersed_proportion"][:,:,:]) > 0
    uvel = np.ma.array(data=nc["uvel"][:,:,:],mask=imm)
    vvel = np.ma.array(data=nc["vvel"][:,:,:],mask=imm)
    wvel = np.ma.array(data=nc["wvel"][:,:,:],mask=imm)
    up = uvel - np.mean(uvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    vp = vvel - np.mean(vvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp = wvel - np.mean(wvel,axis=(1,2))[:,np.newaxis,np.newaxis]
    wp_wp = wp*wp if k==0 else wp_wp+wp*wp
  wp_wp /= len(times)
  wp_wp_mean = np.mean(wp_wp,axis=(1,2))
  ax.plot(wp_wp_mean,z,color=colors[j],label=labels[j],linestyle=styles[j])
ax.set_xlabel(r"w'w' $(m^2/s^2)$")
ax.set_ylabel("z-location (km)")
ax.legend()
ax.margins(x=0)
plt.grid()
plt.margins(x=0)
plt.tight_layout()
plt.savefig("building_wp_wp_height.png",dpi=600)
plt.show()
plt.close()

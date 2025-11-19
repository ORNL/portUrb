from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmap import Colormap

workdir="/lustre/storm/nwp501/scratch/imn/city_stretched_2"

def get_ind(a,v) :
  return np.argmin(np.abs(a-v))

times = range(8*4+1,10*4+1,1)
files_prec = [f"{workdir}/city_stretched_precursor_{i:08d}.nc" for i in times]
files_main = [f"{workdir}/city_stretched_{i:08d}.nc" for i in times]

# Plot vertical grid spacing
nc_prec = Dataset(files_prec[-1],"r")
zi      = np.array(nc_prec["zi"][:])
fig = plt.figure(figsize=(4,6))
ax  = fig.gca()
ax.hlines(zi,0,1,color="black")
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.set_xlim(0,1)
ax.set_ylim(470.413,623.9676)
ax.set_ylabel("Vertical Interface Locations (m)")
ax.margins(x=0)
fig.tight_layout()
plt.savefig("city_ABL_vert_grid_z.png",dpi=600)
plt.show()
plt.close()


# Intantaneous horizontal velocity at z=100 and y=midpoint
nc_prec  = Dataset(files_prec[-1],"r")
nc_main  = Dataset(files_main[-1],"r")
x        = np.array(nc_prec["x"][:])
y        = np.array(nc_prec["y"][:])
z        = np.array(nc_prec["z"][:])
ind_k    = get_ind(z,100)
ind_j    = int(len(y)/2)
k2       = get_ind(z,900)
u        = np.array(nc_prec["uvel"][:,:,:])
v        = np.array(nc_prec["vvel"][:,:,:])
mag_prec = np.sqrt(u*u+v*v)
u        = np.array(nc_main["uvel"][:,:,:])
v        = np.array(nc_main["vvel"][:,:,:])
mag_main = np.sqrt(u*u+v*v)
fig = plt.figure(figsize=(8,8),constrained_layout=True)
gs = gridspec.GridSpec(2, 2, height_ratios=[2.8, 1], hspace=0.00, wspace=0.00, figure=fig)
# Top row: larger subplots
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
# Bottom row: smaller subplots
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
X,Y = np.meshgrid(x/1000,y/1000)
mn = np.minimum(np.min(mag_prec[ind_k,:,:]),np.min(mag_main[ind_k,:,:]))
mx = np.maximum(np.max(mag_prec[ind_k,:,:]),np.max(mag_main[ind_k,:,:]))
pos = ax1.get_position()
ax1.set_position([pos.x0-0.04, pos.y0+0.07, pos.width, pos.height])
CS1 = ax1.contourf(X,Y,mag_prec[ind_k,:,:],levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax1.axis('scaled')
ax1.set_xlabel("x-location (km)")
ax1.set_ylabel("y-location (km)")
ax1.margins(x=0)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("bottom", size="1.2%", pad=0.5)
cbar1 = plt.colorbar(CS1,orientation="horizontal",cax=cax1)
cbar1.ax.tick_params(labelrotation=40)
pos = ax2.get_position()
ax2.set_position([pos.x0+0.04, pos.y0+0.07, pos.width, pos.height])
CS2 = ax2.contourf(X,Y,mag_main[ind_k,:,:],levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax2.axis('scaled')
ax2.set_xlabel("x-location (km)")
ax2.set_ylabel("y-location (km)")
ax2.margins(x=0)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("bottom", size="1.2%", pad=0.5)
cbar2 = plt.colorbar(CS2,orientation="horizontal",cax=cax2)
cbar2.ax.tick_params(labelrotation=40)
mn = np.minimum(np.min(mag_prec[:k2,ind_j,:]),np.min(mag_main[:k2,ind_j,:]))
mx = np.maximum(np.max(mag_prec[:k2,ind_j,:]),np.max(mag_main[:k2,ind_j,:]))
X,Z = np.meshgrid(x/1000,z[:k2]/1000)
pos = ax3.get_position()
ax3.set_position([pos.x0-0.04, pos.y0-0.02 , pos.width, pos.height])
CS3 = ax3.contourf(X,Z,mag_prec[:k2,ind_j,:],levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax3.axis('scaled')
ax3.set_xlabel("x-location (km)")
ax3.set_ylabel("z-location (km)")
ax3.margins(x=0)
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar3 = plt.colorbar(CS3,orientation="horizontal",cax=cax3)
cbar3.ax.tick_params(labelrotation=40)
pos = ax4.get_position()
ax4.set_position([pos.x0+0.04, pos.y0-0.02 , pos.width, pos.height])
CS4 = ax4.contourf(X,Z,mag_main[:k2,ind_j,:],levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax4.axis('scaled')
ax4.set_xlabel("x-location (km)")
ax4.set_ylabel("z-location (km)")
ax4.margins(x=0)
divider = make_axes_locatable(ax4)
cax4 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar4 = plt.colorbar(CS4,orientation="horizontal",cax=cax4)
cbar4.ax.tick_params(labelrotation=40)
plt.savefig("city_ABL_wind_4panel.png",dpi=600)
plt.show()
plt.close()


# Theta at y=midpoint
nc_prec  = Dataset(files_prec[-1],"r")
nc_main  = Dataset(files_main[-1],"r")
x        = np.array(nc_prec["x"][:])
y        = np.array(nc_prec["y"][:])
z        = np.array(nc_prec["z"][:])
ind_k    = get_ind(z,100)
ind_j    = int(len(y)/2)
k2       = get_ind(z,900)
nz       = len(z)
hs       = int((nc_prec.dimensions["z_halo"].size-nz)/2)
th_prec = np.array(nc_prec["theta_pert"][:,:,:])+np.array(nc_prec["hy_theta_cells"][hs:len(z)+hs])[:,np.newaxis,np.newaxis]
th_main = np.array(nc_main["theta_pert"][:,:,:])+np.array(nc_main["hy_theta_cells"][hs:len(z)+hs])[:,np.newaxis,np.newaxis]
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(8,2.5))
mn = np.minimum(np.min(th_prec[:k2,ind_j,:]),np.min(th_main[:k2,ind_j,:]))
mx = np.maximum(np.max(th_prec[:k2,ind_j,:]),np.max(th_main[:k2,ind_j,:]))
X,Z = np.meshgrid(x/1000,z[:k2]/1000)
CS0 = ax[0].contourf(X,Z,th_prec[:k2,ind_j,:],levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax[0].axis('scaled')
ax[0].set_xlabel("x-location (km)")
ax[0].set_ylabel("z-location (km)")
ax[0].margins(x=0)
divider = make_axes_locatable(ax[0])
cax0 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar0 = plt.colorbar(CS0,orientation="horizontal",cax=cax0)
cbar0.ax.tick_params(labelrotation=40)
CS1 = ax[1].contourf(X,Z,th_main[:k2,ind_j,:],levels=np.arange(mn,mx,(mx-mn)/100),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax[1].axis('scaled')
ax[1].set_xlabel("x-location (km)")
ax[1].set_ylabel("z-location (km)")
ax[1].margins(x=0)
divider = make_axes_locatable(ax[1])
cax1 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar1 = plt.colorbar(CS1,orientation="horizontal",cax=cax1)
cbar1.ax.tick_params(labelrotation=40)
fig.tight_layout()
plt.savefig("city_ABL_theta_2panel.png",dpi=600)
plt.show()
plt.close()


# avg_u over the last two hours
fig,ax = plt.subplots(1,3,figsize=(8,5),sharey=True)
vname = "avg_u"
for i in range(len(times)) :
  nc_prec  = Dataset(files_prec[i],"r")
  nc_main  = Dataset(files_main[i],"r")
  x        = np.array(nc_prec["x"][:])
  y        = np.array(nc_prec["y"][:])
  z        = np.array(nc_prec["z"][:])
  i1       = get_ind(x,1*1269.11407/2) # Bounding box of the city
  i2       = get_ind(x,3*1269.11407/2) # Bounding box of the city
  j1       = get_ind(y,1*1674.81256/2) # Bounding box of the city
  j2       = get_ind(y,3*1674.81256/2) # Bounding box of the city
  k2       = get_ind(z,900)
  loc_prec = np.mean(np.array(nc_prec[vname][:k2,j1:j2+1,i1:i2+1]),axis=(1,2))
  loc_main = np.mean(np.array(nc_main[vname][:k2,j1:j2+1,i1:i2+1]),axis=(1,2))
  var_prec = loc_prec if i==0 else var_prec+loc_prec
  var_main = loc_main if i==0 else var_main+loc_main
var_prec /= len(times)
var_main /= len(times)
ax[0].plot(var_prec,z[:k2],label="precursor",color="black",linestyle="--")
ax[0].plot(var_main,z[:k2],label="city"     ,color="red"  )
ax[0].set_xlim(0,12)
ax[0].set_ylim(0,900)
ax[0].set_xlabel("u-velocity (m/s)")
ax[0].set_ylabel("Height (m)")
ax[0].legend()
ax[0].margins(x=0)


# avg_v over the last two hours
vname = "avg_v"
for i in range(len(times)) :
  nc_prec  = Dataset(files_prec[i],"r")
  nc_main  = Dataset(files_main[i],"r")
  x        = np.array(nc_prec["x"][:])
  y        = np.array(nc_prec["y"][:])
  z        = np.array(nc_prec["z"][:])
  i1       = get_ind(x,1*1269.11407/2) # Bounding box of the city
  i2       = get_ind(x,3*1269.11407/2) # Bounding box of the city
  j1       = get_ind(y,1*1674.81256/2) # Bounding box of the city
  j2       = get_ind(y,3*1674.81256/2) # Bounding box of the city
  k2       = get_ind(z,900)
  loc_prec = np.mean(np.array(nc_prec[vname][:k2,j1:j2+1,i1:i2+1]),axis=(1,2))
  loc_main = np.mean(np.array(nc_main[vname][:k2,j1:j2+1,i1:i2+1]),axis=(1,2))
  var_prec = loc_prec if i==0 else var_prec+loc_prec
  var_main = loc_main if i==0 else var_main+loc_main
var_prec /= len(times)
var_main /= len(times)
ax[1].plot(var_prec,z[:k2],label="precursor",color="black",linestyle="--")
ax[1].plot(var_main,z[:k2],label="city",color="red"  )
ax[1].set_xlim(-0.25,4.25)
ax[1].set_ylim(0,900)
ax[1].set_xlabel("v-velocity (m/s)")
# ax[1].set_ylabel("Height (m)")
ax[1].legend()
ax[1].margins(x=0)


# theta over the last two hours
for i in range(len(times)) :
  nc_prec  = Dataset(files_prec[i],"r")
  nc_main  = Dataset(files_main[i],"r")
  x        = np.array(nc_prec["x"][:])
  y        = np.array(nc_prec["y"][:])
  z        = np.array(nc_prec["z"][:])
  nz       = nc_prec.dimensions["z"].size
  hs       = int((nc_prec.dimensions["z_halo"].size-nz)/2)
  i1       = get_ind(x,1*1269.11407/2) # Bounding box of the city
  i2       = get_ind(x,3*1269.11407/2) # Bounding box of the city
  j1       = get_ind(y,1*1674.81256/2) # Bounding box of the city
  j2       = get_ind(y,3*1674.81256/2) # Bounding box of the city
  k2       = get_ind(z,900)
  loc_prec = np.mean(np.array(nc_prec["theta_pert"][:k2,j1:j2+1,i1:i2+1]),axis=(1,2)) + np.array(nc_prec["hy_theta_cells"][hs:k2+hs])
  loc_main = np.mean(np.array(nc_main["theta_pert"][:k2,j1:j2+1,i1:i2+1]),axis=(1,2)) + np.array(nc_main["hy_theta_cells"][hs:k2+hs])
  var_prec = loc_prec if i==0 else var_prec+loc_prec
  var_main = loc_main if i==0 else var_main+loc_main
var_prec /= len(times)
var_main /= len(times)
ax[2].plot(var_prec,z[:k2],label="precursor",color="black",linestyle="--")
ax[2].plot(var_main,z[:k2],label="city"     ,color="red"  )
ax[2].set_xlim(299.7,314.5)
ax[2].set_ylim(0,900)
ax[2].set_xlabel("Potential Temperature (K)")
# ax[2].set_ylabel("Height (m)")
ax[2].legend()
ax[2].margins(x=0)
fig.tight_layout()
plt.savefig("city_ABL_vertical.png",dpi=600)
plt.show()
plt.close()


# # w' theta'
# nc_prec    = Dataset(files_prec[-1],"r")
# nc_main    = Dataset(files_main[-1],"r")
# x          = np.array(nc_prec["x"][:])
# y          = np.array(nc_prec["y"][:])
# z          = np.array(nc_prec["z"][:])
# nz         = nc_prec.dimensions["z"].size
# hs         = int((nc_prec.dimensions["z_halo"].size-nz)/2)
# i1         = get_ind(x,1*1269.11407/2) # Bounding box of the city
# i2         = get_ind(x,3*1269.11407/2) # Bounding box of the city
# j1         = get_ind(y,1*1674.81256/2) # Bounding box of the city
# j2         = get_ind(y,3*1674.81256/2) # Bounding box of the city
# k2         = get_ind(z,900)
# theta_prec = np.array(nc_prec["theta_pert"][:k2,:,:]) + np.array(nc_prec["hy_theta_cells"][hs:k2+hs])[:,np.newaxis,np.newaxis]
# theta_main = np.array(nc_main["theta_pert"][:k2,:,:]) + np.array(nc_main["hy_theta_cells"][hs:k2+hs])[:,np.newaxis,np.newaxis]
# w_prec     = np.array(nc_prec["wvel"      ][:k2,:,:])
# w_main     = np.array(nc_main["wvel"      ][:k2,:,:])
# wp_prec    = w_prec     - np.mean(w_prec    ,axis=(1,2))[:,np.newaxis,np.newaxis]
# wp_main    = w_main     - np.mean(w_main    ,axis=(1,2))[:,np.newaxis,np.newaxis]
# tp_prec    = theta_prec - np.mean(theta_prec,axis=(1,2))[:,np.newaxis,np.newaxis]
# tp_main    = theta_main - np.mean(theta_main,axis=(1,2))[:,np.newaxis,np.newaxis]
# wptp_prec  = np.mean(wp_prec*tp_prec,axis=(1,2)) 
# wptp_main  = np.mean(wp_main*tp_main,axis=(1,2))
# fig = plt.figure(figsize=(4,6))
# ax  = fig.gca()
# ax.plot(wptp_prec,z[:k2],label="precursor",color="black",linestyle="--")
# ax.plot(wptp_main,z[:k2],label="city"     ,color="red"  )
# ax.set_xlim(-1.1,0.5)
# ax.set_ylim(0,900)
# ax.set_xlabel(r"$w^\prime \theta^\prime$")
# ax.set_ylabel("Height (m)")
# ax.legend()
# ax.margins(x=0)
# fig.tight_layout()
# plt.savefig("city_ABL_wptp_z.png",dpi=600)
# plt.show()
# plt.close()

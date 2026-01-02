from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from cmap import Colormap

D = 126
R = 63
H = 90
R_hub = 1.5
V = 8

t_end = 4

for i in range(1,t_end+1) :
  nc = Dataset(f"test_{i:08d}.nc","r")
  p  = np.array(nc["power_0"][:])
  pwr = p if i==1 else np.concatenate((pwr,p))
  t  = np.sum(np.array(nc["force_axial_0"][:,:,:]),axis=(1,2))
  thr = t if i==1 else np.concatenate((thr,t))
  if (i == t_end) :
    print( np.mean(np.array(nc["density_dry"][:,:,:])) )
    print( np.mean( p/np.mean(np.array(nc["density_dry"][:,:,:]))*1.225 ) )
    print( np.mean( p/(0.5*np.mean(np.array(nc["density_dry"][:,:,:]))*np.pi*R*R*V*V*V) ) )
    print( np.mean( t/(0.5*np.mean(np.array(nc["density_dry"][:,:,:]))*np.pi*R*R*V*V  ) ) )
plt.plot(pwr)
# plt.ylim(2e6,2.1e6)
plt.grid()
plt.show()
plt.close()

x    = np.array(nc["x"             ][:])
y    = np.array(nc["y"             ][:])
z    = np.array(nc["z"             ][:])
rad  = np.array(nc["radial_points" ][:])
dx   = x[1]-x[0]
dy   = y[1]-y[0]
dz   = z[1]-z[0]
xlen = x[-1]+dx/2
ylen = y[-1]+dy/2
zlen = z[-1]+dz/2

plt.plot(rad,np.mean(np.array(nc["inflow_tang_0"][:,:,:]),axis=(0,1)))
plt.show()
plt.close()

pub_r = np.array([0.000792915,0.02681361,0.052525142,0.074846782,0.087100587,0.095819012,0.105177587,0.111753687,0.118624402,0.124098423,0.129103243,0.134613636,0.138596396,0.147558514,0.155913216,0.173164566,0.192892866,0.20453198,0.21962282,0.240296797,0.265844654,0.290792369,0.31645298,0.342680997,0.368265226,0.393114736,0.416200193,0.43937658,0.462698456,0.487013294,0.510320621,0.534508156,0.55805554,0.580802735,0.603768163,0.628195755,0.652910688,0.678673141,0.705097569,0.730838198,0.756724317,0.782377653,0.806296034,0.83149108,0.856675214,0.881990289,0.902533326,0.916954917,0.926837252,0.935006456,0.942830123,0.950090021,0.956364232,0.962296543,0.967505047,0.973266408,0.978918653,0.983603397])

pub_ax = np.array([0.961267269,0.955458606,0.951550847,0.947605692,0.9432949,0.938925937,0.932755791,0.926837021,0.919985458,0.913264776,0.90647346,0.899889893,0.892743326,0.878880233,0.864863405,0.836993871,0.809454659,0.796212735,0.784464527,0.776000831,0.773815311,0.776472421,0.779364288,0.780390568,0.778703646,0.773871403,0.767938091,0.761221564,0.754457256,0.748661057,0.74316194,0.737374052,0.730547419,0.723785187,0.717320037,0.711976732,0.708852187,0.708613275,0.709953256,0.710815415,0.709541913,0.706674977,0.702237457,0.697710606,0.693453828,0.693883868,0.702900177,0.715095045,0.728476161,0.742652955,0.756877532,0.770717773,0.785534434,0.799860808,0.813786226,0.829012153,0.843008206,0.8518043])

nc = Dataset(f"test_{t_end:08d}.nc","r")
inflow_axial = np.array(nc["inflow_axial_0"][:,:,:])
plt.plot(rad/R,np.mean(inflow_axial,axis=(0,1))/V,label=f"portUrb")
plt.plot(pub_r,pub_ax,label=f"openFOAM")
plt.xlabel("r/R")
plt.ylabel(r"$u_{axial}/u_{\infty}$")
plt.legend()
plt.grid()
plt.show()

nc = Dataset(f"test_{t_end:08d}.nc","r")
inflow_axial = np.array(nc["force_axial_0"][:,:,:])
plt.plot(rad/R,np.mean(inflow_axial,axis=(0,1)),label=f"portUrb")
plt.xlabel("r/R")
plt.ylabel(r"axial force (N)")
plt.legend()
plt.grid()
plt.show()

nc = Dataset(f"test_{t_end:08d}.nc","r")
inflow_axial = np.array(nc["force_tang_0"][:,:,:])
plt.plot(rad/R,np.mean(inflow_axial,axis=(0,1))*rad,label=f"portUrb")
plt.xlabel("r/R")
plt.ylabel(r"Torque (N m)")
plt.legend()
plt.grid()
plt.show()

k = np.argmin(np.abs(z-H))
i = np.argmin(np.abs(x-5*D))

nc = Dataset(f"test_{t_end:08d}.nc","r")
u = np.array(nc["avg_u"][k,:,i])
plt.plot((y-ylen/2)/D,u/V,label=f"portUrb")
plt.title("1 turbine diameter downstream")
plt.xlabel("y/D")
plt.ylabel(r"$u/u_{\infty}$")
plt.xlim(-1.5,1.5)
plt.ylim(0.4,1.1)
plt.legend()
plt.grid()
plt.show()

i = np.argmin(np.abs(x-8*D))

nc = Dataset(f"test_{t_end:08d}.nc","r")
u = np.array(nc["avg_u"][k,:,i])
plt.plot((y-ylen/2)/D,u/V,label=f"portUrb")
plt.title("4 turbine diameter downstream")
plt.xlabel("y/D")
plt.ylabel(r"$u/u_{\infty}$")
plt.xlim(-1.5,1.5)
plt.ylim(0.4,1.1)
plt.legend()
plt.grid()
plt.show()


i = np.argmin(np.abs(x-6*D))
k = np.argmin(np.abs(z-H))
i1 = np.argmin(np.abs(x-3.25*D))
i2 = np.argmin(np.abs(x-9*D))
j1 = np.argmin(np.abs(y-0.75*D))
j2 = np.argmin(np.abs(y-3.25*D))

X,Y = np.meshgrid(x[i1:i2],y[j1:j2])
plt.contourf(X,Y,np.array(nc["avg_u"][k,j1:j2,i1:i2]),levels=np.arange(3,8,0.01),cmap="jet",extend="both")
plt.xlabel("x-location")
plt.ylabel("y-location")
# plt.gca().invert_xaxis()
plt.gca().set_aspect("equal")
plt.colorbar(orientation="horizontal")
plt.tight_layout()
plt.show()
plt.close()

u = np.array(nc["uvel"][k,j1:j2,i1:i2])
v = np.array(nc["vvel"][k,j1:j2,i1:i2])
vort = np.gradient(v, dx, axis=1) - np.gradient(u, dy, axis=0)

X,Y = np.meshgrid(x[i1:i2],y[j1:j2])
plt.contourf(X,Y,np.abs(vort),levels=np.arange(0,1,1/100),cmap="jet",extend="both")
plt.xlabel("x-location")
plt.ylabel("y-location")
# plt.gca().invert_xaxis()
plt.gca().set_aspect("equal")
plt.colorbar(orientation="horizontal")
plt.tight_layout()
plt.show()
plt.close()



i  = np.argmin(np.abs(x-4.00*D))
j1 = np.argmin(np.abs(y-1.25*D))
j2 = np.argmin(np.abs(y-2.75*D))
k2 = np.argmin(np.abs(z-1.50*D))
Y,Z = np.meshgrid(y[j1:j2],z[:k2])

plt.contourf(Y,Z,np.array(nc["turbine_tend_u"][:k2,j1:j2,i-4]),levels=100,cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
plt.gca().set_aspect("equal")
plt.colorbar(orientation="horizontal")
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()
plt.close()

plt.contourf(Y,Z,np.array(nc["turbine_tend_v"][:k2,j1:j2,i-4]),levels=100,cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
plt.gca().set_aspect("equal")
plt.colorbar(orientation="horizontal")
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()
plt.close()

plt.contourf(Y,Z,np.array(nc["turbine_tend_w"][:k2,j1:j2,i-4]),levels=100,cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
plt.gca().set_aspect("equal")
plt.colorbar(orientation="horizontal")
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()
plt.close()

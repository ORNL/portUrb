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
  return freq[1:]*2*2*np.pi/(2*dx) , spd[1:]


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

nc = Dataset(f"{workdir}/ABL_convective_10m_00000004.nc","r")
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
rho   = np.array(nc["density_dry"][:,:,:])
uvel  = np.array(nc["uvel"       ][:,:,:])
vvel  = np.array(nc["vvel"       ][:,:,:])
wvel  = np.array(nc["wvel"       ][:,:,:])
T     = np.array(nc["temperature"][:,:,:])
K     = np.array(nc["TKE"        ][:,:,:])/rho
theta = np.pow((rho*R_d*T)/C0,1./gamma_d)/rho
mag   = np.sqrt(uvel*uvel+vvel*vvel)


fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,10))
t1 = 310
t2 = 315
X,Y = np.meshgrid(x,y)
zind = get_ind(z,.078)
print(zind, z[zind])
mn = np.min(theta[zind,:,:])
mx = np.max(theta[zind,:,:])
CS1 = ax1.contourf(X,Y,theta[zind,:,:],levels=np.arange(mn,mx,(mx-mn)/200),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
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
CS2 = ax2.contourf(X,Y,wvel[zind,:,:],levels=np.arange(-6,6,12/200),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
ax2.axis('scaled')
ax2.set_xlabel("x-location (km)")
ax2.set_ylabel("y-location (km)")
ax2.margins(x=0)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("bottom", size="4%", pad=0.5)
cbar2 = plt.colorbar(CS2,orientation="horizontal",cax=cax2)
cbar2.ax.tick_params(labelrotation=30)

X,Z = np.meshgrid(x,z)
yind = get_ind(y,3)
CS3 = ax3.contourf(X,Z,theta[:,yind,:],levels=np.arange(t1,t2,(t2-t1)/200),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
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
CS4 = ax4.contourf(X,Z,wvel[:,yind,:],levels=np.arange(-6,6,12/200),cmap=Colormap('cmasher:fusion_r').to_mpl(),extend="both")
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



kind = get_ind(z,.0786)
print(z[kind])
freq,spd1 = spectra(mag  [kind:kind+1,:,:],dx=dx)
freq,spd2 = spectra(wvel [kind:kind+1,:,:],dx=dx)
freq,spd3 = spectra(theta[kind:kind+1,:,:],dx=dx)
spd1 = spd1/np.mean(spd1)
spd2 = spd2/np.mean(spd2)
spd3 = spd3/np.mean(spd3)
freq = freq
fig = plt.figure(figsize=(6,4))
ax = fig.gca()
ax.plot(freq,spd1,label="Horizontal Wind Speed spectra",color="black")
ax.plot(freq,spd2,label="Vertical Velocity spectra"    ,color="blue" )
ax.plot(freq,spd3,label="Potential Temperature spectra",color="lightgreen")
ax.plot(freq,1.e3*freq**(-5/3),label=r"$f^{-5/3}$"    ,color="magenta"  )
ax.vlines(2*np.pi/(2 *dx),2.e-4,1.e1,linestyle="--",color="red")
ax.vlines(2*np.pi/(4 *dx),2.e-4,1.e1,linestyle="--",color="red")
ax.vlines(2*np.pi/(8 *dx),2.e-4,1.e1,linestyle="--",color="red")
ax.vlines(2*np.pi/(16*dx),2.e-4,1.e1,linestyle="--",color="red")
ax.text(0.9*2*np.pi/(2 *dx),2.e1,r"$2  \Delta x$")
ax.text(0.9*2*np.pi/(4 *dx),2.e1,r"$4  \Delta x$")
ax.text(0.9*2*np.pi/(8 *dx),2.e1,r"$8  \Delta x$")
ax.text(0.9*2*np.pi/(16*dx),2.e1,r"$16 \Delta x$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Frequency")
ax.set_ylabel("Spectral Power")
ax.legend(loc='lower left')
# ax.set_xlim(left=0.0045)
ax.set_ylim(top=1.e3)
ax.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_convective_spectra.png",dpi=600)
plt.show()
plt.close()


obs_z      = [\
1.07642626480086,\
2.51166128453534,\
4.12630068173663,\
10.405453893075,\
17.1331180480804,\
47.8112665949049,\
75.6189451022605,\
117.778973806961,\
159.849300322928,\
202.009329027628]
obs_mag  = [\
5.27777777777778,\
5.97222222222222,\
5.81018518518519,\
6.94444444444444,\
7.15277777777778,\
7.68518518518519,\
7.96296296296296,\
8.05555555555556,\
8.1712962962963,\
8.31018518518519]
obs_std  = [\
1.84027777777778,\
1.9212962962963,\
1.89814814814815,\
1.79398148148148,\
1.75925925925926,\
1.63194444444444,\
1.57407407407407,\
1.42361111111111,\
1.38888888888889,\
1.35416666666667]
fe_mag = [\
5.12471655328798,\
6.71201814058957,\
6.96145124716553,\
7.2108843537415,\
7.27891156462585,\
7.36961451247166,\
7.43764172335601,\
7.50566893424036,\
7.55102040816327,\
7.59637188208617,\
7.64172335600907,\
7.66439909297052,\
7.68707482993197,\
7.68707482993197,\
7.70975056689342,\
7.70975056689342,\
7.73242630385488,\
7.73242630385488,\
7.75510204081633,\
7.75510204081633,\
7.80045351473923,\
7.82312925170068,\
7.84580498866213,\
7.86848072562358]
fe_z = [\
5.01054852320675,\
15.0316455696203,\
25.0527426160338,\
35.0738396624473,\
45.0949367088608,\
55.1160337552743,\
65.1371308016878,\
75.4219409282701,\
85.7067510548523,\
95.7278481012658,\
106.276371308017,\
116.561181434599,\
126.845991561181,\
137.394514767933,\
147.943037974684,\
158.755274261603,\
169.567510548523,\
180.379746835443,\
191.455696202532,\
202.53164556962,\
213.607594936709,\
225.210970464135,\
236.550632911392,\
248.154008438819]
hg_mag = [\
4.60287891617273,\
5.89331075359864,\
6.37087214225233,\
6.56392887383573,\
6.66553767993226,\
6.75698560541914,\
6.83827265029636,\
6.91955969517358,\
6.99068585944115,\
7.04149026248942,\
7.07197290431837,\
7.08213378492803,\
7.11261642675699,\
7.13293818797629,\
7.1532599491956,\
7.1735817104149,\
7.1735817104149,\
7.1735817104149,\
7.16342082980525,\
7.16342082980525,\
7.16342082980525,\
7.1735817104149,\
7.18374259102456,\
7.20406435224386,\
7.22438611346317]
hg_z = [\
5.03177966101695,\
14.8305084745763,\
25.1588983050847,\
34.6927966101695,\
45.021186440678,\
54.8199152542373,\
64.353813559322,\
74.6822033898305,\
84.4809322033898,\
94.8093220338983,\
104.608050847458,\
114.406779661017,\
125,\
134.798728813559,\
144.862288135593,\
154.925847457627,\
164.989406779661,\
175.052966101695,\
184.851694915254,\
194.915254237288,\
204.978813559322,\
214.777542372881,\
224.841101694915,\
234.904661016949,\
244.968220338983]
sf_mag = [\
5.89508742714405,\
6.95420482930891,\
7.06411323896753,\
7.29392173189009,\
7.4338051623647,\
7.55370524562864,\
7.60366361365529,\
7.70358034970858,\
7.75353871773522,\
7.76353039134055,\
7.80349708576187,\
7.82348043297252]
sf_z = [\
10.1456815816857,\
30.1768990634755,\
50.2081165452653,\
70.2393340270552,\
90.5306971904266,\
110.301768990635,\
130.332986472425,\
150.364203954214,\
170.135275754422,\
190.426638917794,\
209.93756503642,\
229.96878251821]
wrf_mag = [\
5.7504159733777,\
6.50915141430948,\
6.9783693843594,\
7.26788685524126,\
7.47753743760399,\
7.5973377703827,\
7.68718801996672,\
7.75707154742097,\
7.80698835274542,\
7.8369384359401,\
7.87687188019967,\
7.89683860232945,\
7.91680532445924,\
7.92678868552413,\
7.93677204658902,\
7.94675540765391,\
7.96672212978369,\
7.98668885191348,\
8.00665557404326,\
8.02662229617305,\
8.03660565723794,\
8.04658901830283,\
8.05657237936772,\
8.06655574043261,\
8.0765391014975,\
8.09650582362729,\
8.09650582362729,\
8.10648918469218,\
8.10648918469218,\
8.11647254575707,\
8.11647254575707,\
8.12645590682196,\
8.13643926788686,\
8.13643926788686,\
8.14642262895175,\
8.15640599001664,\
8.16638935108153,\
8.17637271214642,\
8.18635607321131,\
8.19633943427621,\
8.19633943427621,\
8.19633943427621,\
8.2063227953411,\
8.21630615640599,\
8.22628951747088,\
8.23627287853577,\
8.23627287853577,\
8.23627287853577,\
8.24625623960067,\
8.25623960066556,\
8.25623960066556,\
8.26622296173045,\
8.27620632279534,\
8.28618968386023,\
8.28618968386023]
wrf_z = [\
2.33887733887734,\
6.75675675675676,\
11.4345114345114,\
15.5925155925156,\
20.2702702702703,\
24.4282744282744,\
28.8461538461538,\
33.2640332640333,\
37.9417879417879,\
42.3596673596674,\
47.037422037422,\
50.9355509355509,\
55.6133056133056,\
60.031185031185,\
64.7089397089397,\
69.1268191268191,\
73.5446985446986,\
78.2224532224532,\
82.9002079002079,\
87.0582120582121,\
91.4760914760915,\
95.8939708939709,\
100.31185031185,\
104.989604989605,\
109.66735966736,\
113.825363825364,\
118.243243243243,\
122.661122661123,\
127.079002079002,\
131.496881496882,\
136.174636174636,\
140.592515592516,\
145.010395010395,\
149.428274428274,\
153.846153846154,\
158.523908523909,\
163.461538461538,\
167.619542619543,\
172.037422037422,\
176.455301455301,\
181.133056133056,\
185.550935550936,\
189.968814968815,\
194.64656964657,\
199.324324324324,\
204.002079002079,\
208.419958419958,\
212.577962577963,\
216.995841995842,\
221.153846153846,\
225.831600831601,\
230.769230769231,\
235.18711018711,\
239.345114345114,\
244.282744282744]
files = [f"{workdir}/ABL_convective_10m_{i:08d}.nc" for i in range(5,7)]
for i in range(len(files)) :
  nc = Dataset(files[i],"r")
  u = np.array(nc["avg_u"][:,:,:])
  v = np.array(nc["avg_v"][:,:,:])
  w = np.array(nc["avg_w"][:,:,:])
  upup = np.array(nc["avg_up_up"][:,:,:])
  vpvp = np.array(nc["avg_vp_vp"][:,:,:])
  wpwp = np.array(nc["avg_wp_wp"][:,:,:])
  magloc  = np.sqrt(u*u+v*v+w*w)
  tkeloc  = (upup+vpvp+wpwp)/2
  mag = magloc if (i==0) else mag+magloc
  tke = tkeloc if (i==0) else tke+tkeloc
mag /= len(files)
tke /= len(files)

std = np.sqrt(2./3.*tke)
z2 = get_ind(z,0.25)
avg_mag = np.mean(mag,axis=(1,2))
avg_std = np.mean(std,axis=(1,2))
fig = plt.figure(figsize=(8,6))
ax = fig.gca()
ax.errorbar(avg_mag,    z*1000,xerr=avg_std,label="portUrb dx=10m",fmt='o', capsize=5)
ax.errorbar(obs_mag,obs_z     ,xerr=obs_std,label="Obs",fmt='o', capsize=5)
ax.plot(fe_mag ,fe_z ,linestyle="--",label="FastEddy dx=10m",linewidth=2,color="black")
ax.plot(hg_mag ,hg_z ,linestyle="--",label="HIGRAD dx=10m"  ,linewidth=2,color="green")
ax.plot(sf_mag ,sf_z ,linestyle="--",label="SOWFA dx=20m"   ,linewidth=2,color="red"  )
ax.plot(wrf_mag,wrf_z,linestyle="--",label="WRF dx=4.5m"    ,linewidth=2,color="cyan" )
ax.set_xlabel("velocity (m/s)")
ax.set_ylabel("z-location (m)")
ax.legend()
ax.set_ylim(0,250)
ax.margins(x=0)
plt.tight_layout()
plt.grid()
plt.savefig("ABL_convective_mag_obs.png",dpi=600)
plt.show()
plt.close()





files = [f"{workdir}/ABL_convective_10m_{i:08d}.nc" for i in range(5,7)]
for i in range(len(files)) :
  nc = Dataset(files[i],"r")
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
  rho  = np.array(nc["density_dry"])
  upup = np.array(nc["avg_up_up"])
  vpvp = np.array(nc["avg_vp_vp"])
  wpwp = np.array(nc["avg_wp_wp"])
  tkesgsloc = np.mean( np.array(nc["avg_tke"][:,:,:]) , axis=(1,2) )
  tkeresloc = np.mean( rho*(upup+vpvp+wpwp)/2         , axis=(1,2) )
  tkesgs = tkesgsloc if (i==0) else tkesgs+tkesgsloc
  tkeres = tkeresloc if (i==0) else tkeres+tkeresloc
tkesgs /= len(files)
tkeres /= len(files)

obs_tke = [\
2.87866108786611,\
3.19246861924686,\
3.07531380753138,\
3.07531380753138,\
3.02092050209205,\
2.87866108786611,\
2.92887029288703,\
2.98744769874477,\
3.21757322175732,\
3.23430962343096]
obs_z = [\
1.56903765690377,\
2.61506276150628,\
4.18410041841004,\
10.1987447698745,\
16.9979079497908,\
47.8556485355649,\
75.5753138075314,\
117.677824267782,\
160.041841004184,\
202.405857740586]
hg_tke = [\
1.23430962343096,\
1.47698744769874,\
1.72384937238494,\
1.96652719665272,\
2.04602510460251,\
2.19246861924686,\
2.32635983263598,\
2.34728033472803,\
2.3765690376569,\
2.38075313807531,\
2.40167364016736,\
2.42677824267782,\
2.44351464435146,\
2.46861924686192,\
2.46861924686192,\
2.46443514644351,\
2.45188284518828,\
2.43514644351464,\
2.418410041841,\
2.38075313807531,\
2.34728033472803,\
2.3347280334728,\
2.31799163179916,\
2.26359832635983,\
2.22175732217573,\
2.1673640167364,\
2.12970711297071,\
2.10460251046025,\
2.10041841004184,\
2.09623430962343,\
2.08786610878661,\
2.07949790794979,\
2.0836820083682]
hg_z = [\
4.96861924686192,\
8.10669456066946,\
10.9832635983264,\
14.1213389121339,\
15.1673640167364,\
20.1359832635983,\
25.8891213389121,\
30.8577405857741,\
41.0564853556485,\
46.2866108786611,\
51.255230125523,\
56.2238493723849,\
61.1924686192469,\
66.4225941422594,\
71.3912133891213,\
76.6213389121339,\
81.8514644351464,\
86.8200836820084,\
92.0502092050209,\
101.987447698745,\
112.186192468619,\
117.416317991632,\
127.615062761506,\
142.782426778243,\
158.21129707113,\
173.378661087866,\
183.577405857741,\
193.776150627615,\
204.23640167364,\
214.173640167364,\
224.633891213389,\
234.832635983264,\
245.292887029289]
wrf_tke = [\
1.61506276150628,\
1.94142259414226,\
2.09623430962343,\
2.2510460251046,\
2.36820083682008,\
2.38912133891213,\
2.38075313807531,\
2.3347280334728,\
2.30125523012552,\
2.28451882845188,\
2.28033472803347,\
2.27615062761506,\
2.28451882845188,\
2.29707112970711,\
2.30962343096234,\
2.28870292887029,\
2.26359832635983,\
2.2928870292887,\
2.31380753138075,\
2.29707112970711,\
2.28451882845188,\
2.26778242677824,\
2.22594142259414,\
2.20502092050209,\
2.22594142259414]
wrf_z = [\
2.35355648535565,\
6.01464435146444,\
8.89121338912134,\
13.8598326359833,\
20.9205020920502,\
26.4121338912134,\
31.1192468619247,\
41.0564853556485,\
46.8096234309623,\
50.9937238493724,\
57.0083682008368,\
67.7301255230126,\
77.928870292887,\
88.1276150627615,\
98.5878661087866,\
109.048117154812,\
129.707112970711,\
144.089958158996,\
154.550209205021,\
170.76359832636,\
185.407949790795,\
195.606694560669,\
211.035564853556,\
226.202928870293,\
243.200836820084]
sf_tke = [\
2.89958158995816,\
2.95397489539749,\
2.98326359832636,\
2.85355648535565,\
2.76569037656904,\
2.74058577405858,\
2.69874476987448,\
2.66108786610879,\
2.61087866108787,\
2.56066945606695,\
2.51464435146443,\
2.52719665271966,\
2.53974895397489,\
2.53556485355648,\
2.52719665271966,\
2.54393305439331,\
2.55648535564854,\
2.52719665271966,\
2.55648535564854,\
2.57322175732218,\
2.56066945606695,\
2.54811715481172]
sf_z = [\
9.41422594142259,\
23.5355648535565,\
29.8117154811715,\
40.5334728033473,\
48.3786610878661,\
50.7322175732218,\
60.1464435146444,\
70.6066945606695,\
89.173640167364,\
99.8953974895398,\
109.048117154812,\
120.81589958159,\
131.276150627615,\
141.213389121339,\
149.84309623431,\
162.918410041841,\
169.717573221757,\
189.853556485356,\
202.928870292887,\
210.774058577406,\
232.740585774059,\
245.292887029289]
fig = plt.figure(figsize=(6,4))
ax = fig.gca()
ax.plot(tkeres+tkesgs,z*1000,color="black",label="portUrb")
ax.plot(obs_tke,obs_z,color="red",label="Obs")
ax.plot(hg_tke,hg_z,linestyle="--",color="black",label="HiGrad")
ax.plot(wrf_tke,wrf_z,linestyle="--",color="green",label="WRF")
ax.plot(sf_tke,sf_z,linestyle="--",color="red",label="SOWFA")
ax.set_xlabel("TKE (m^2/s^2)")
ax.set_ylabel("z-location (m)")
ax.legend()
ax.set_xlim(0,5)
ax.set_ylim(0,250)
ax.margins(x=0)
plt.tight_layout()
plt.savefig("ABL_convective_tke_obs.png",dpi=600)
plt.show()
plt.close()

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray

def get_ind(arr,val) :
  return np.argmin(np.abs(arr-val))

def spectra(T,dx = 1) :
  spd  = np.abs( np.fft.rfft(T[0,0,:]) )**2
  freq = np.fft.rfftfreq(len(T[0,0,:]))
  spd[:] = 0
  for k in range(T.shape[0]) :
    for j in range(T.shape[1]) :
      spd[:] += np.abs( np.fft.rfft(T[k,j,:]) )**2
      spd[:] += np.abs( np.fft.rfft(T[k,:,j]) )**2
  spd[:] /= T.shape[0]*T.shape[1]*2
  return freq*2*2*np.pi/(2*dx) , spd

def cospec_kx_from_plane(up, wp, dx, window="hann"):
  cosp = np.fft.rfft(up[0,0,:])
  freq = np.fft.rfftfreq(len(up[0,0,:]))
  cosp[:] = 0
  up -= np.mean(up,axis=(1,2))[:,np.newaxis,np.newaxis]
  wp -= np.mean(wp,axis=(1,2))[:,np.newaxis,np.newaxis]
  for k in range(up.shape[0]) :
    for j in range(up.shape[1]) :
      fu = np.fft.rfft(up[k,j,:])
      fw = np.fft.rfft(wp[k,j,:])
      cosp[:] += np.real(fu*np.conj(fw))
      fu = np.fft.rfft(up[k,:,j])
      fw = np.fft.rfft(wp[k,:,j])
      cosp[:] += np.real(fu*np.conj(fw))
  cosp[:] /= up.shape[0]*up.shape[1]*2
  return freq*2*np.pi , cosp


workdir = "/lustre/orion/stf006/scratch/imn/portUrb/build"

t1 = 6
t2 = 7
# prefixes = ["cubes_periodic_nosgs_20_","cubes_periodic_sgs0.3_20_","cubes_periodic_sgs1.0_20_","cubes_periodic_nosgs_40_","cubes_periodic_sgs1.0_40_"]
prefixes = ["cubes_periodic_"]
labels   = ["ILES old"]
has_tke  = [True,False]
colors   = ["blue","magenta","orange","purple"]
fnames   = [ [f"{workdir}/{prefix}{i:08}.nc" for i in range(t1,t2+1)] for prefix in prefixes]

nc   = Dataset(fnames[0][0],"r")
x    = np.array(nc["x"][:])
y    = np.array(nc["y"][:])
z    = np.array(nc["z"][:])
nx   = len(x)
ny   = len(y)
nz   = len(z)
dx   = x[1]-x[0]
dy   = y[1]-y[0]
dz   = z[1]-z[0]
xlen = x[-1]+dx/2
ylen = y[-1]+dy/2
zlen = z[-1]+dz/2
p0_x = [get_ind(x,2*xlen/8),get_ind(x,6*xlen/8),get_ind(x,2*xlen/8),get_ind(x,6*xlen/8)]
p0_y = [get_ind(y,6*ylen/8),get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8)]
p1_x = [get_ind(x,4*xlen/8),get_ind(x,0*xlen/8),get_ind(x,4*xlen/8),get_ind(x,0*xlen/8)]
p1_y = [get_ind(y,6*ylen/8),get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8)]
p2_x = [get_ind(x,4*xlen/8),get_ind(x,0*xlen/8),get_ind(x,4*xlen/8),get_ind(x,0*xlen/8)]
p2_y = [get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8),get_ind(y,6*ylen/8)]
p3_x = [get_ind(x,2*xlen/8),get_ind(x,6*xlen/8),get_ind(x,2*xlen/8),get_ind(x,6*xlen/8)]
p3_y = [get_ind(y,4*ylen/8),get_ind(y,2*ylen/8),get_ind(y,0*ylen/8),get_ind(y,6*ylen/8)]



# k1 = get_ind(z,.021)
# k2 = get_ind(z,.041)
# fig = plt.figure(figsize=(6,3))
# ax = fig.gca()
# for k in range(len(prefixes)) :
#   nc  = Dataset(fnames[k][-1],"r")
#   u   = np.array(nc["uvel"][:,:,:])
#   v   = np.array(nc["vvel"][:,:,:])
#   w   = np.array(nc["wvel"][:,:,:])
#   mag = np.sqrt(u**2+v**2+w**2)
#   freq,spd = spectra(mag[k1:k2+1,:,:],dx=dx)
#   ax.plot(freq,spd,label=f"{labels[k]}")
# ax.plot(freq[1:],1.2e6*freq[1:]**(-5/3),label=r"$f^{-5/3}$")
# ax.vlines(2*np.pi/(2 *dx),1.e-3,1.e3,linestyle="--",color="red")
# ax.vlines(2*np.pi/(4 *dx),1.e-3,1.e3,linestyle="--",color="red")
# ax.vlines(2*np.pi/(8 *dx),1.e-3,1.e3,linestyle="--",color="red")
# ax.vlines(2*np.pi/(16*dx),1.e-3,1.e3,linestyle="--",color="red")
# ax.text(0.9*2*np.pi/(2 *dx),2.e3,r"$2  \Delta x$")
# ax.text(0.9*2*np.pi/(4 *dx),2.e3,r"$4  \Delta x$")
# ax.text(0.9*2*np.pi/(8 *dx),2.e3,r"$8  \Delta x$")
# ax.text(0.9*2*np.pi/(16*dx),2.e3,r"$16 \Delta x$")
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel("Frequency")
# ax.set_ylabel("Spectral Power")
# ax.legend(loc='lower left')
# # ax.set_ylim(top=1.e6)
# ax.margins(x=0)
# plt.margins(x=0)
# plt.tight_layout()
# plt.show()
# plt.close()



u0    = np.zeros((len(prefixes),len(fnames[0]),4,nz))
u1    = np.zeros((len(prefixes),len(fnames[0]),4,nz))
u2    = np.zeros((len(prefixes),len(fnames[0]),4,nz))
u3    = np.zeros((len(prefixes),len(fnames[0]),4,nz))
tke0  = np.zeros((len(prefixes),len(fnames[0]),4,nz))
tke1  = np.zeros((len(prefixes),len(fnames[0]),4,nz))
tke2  = np.zeros((len(prefixes),len(fnames[0]),4,nz))
tke3  = np.zeros((len(prefixes),len(fnames[0]),4,nz))
upwp0 = np.zeros((len(prefixes),len(fnames[0]),4,nz))
upwp1 = np.zeros((len(prefixes),len(fnames[0]),4,nz))
upwp2 = np.zeros((len(prefixes),len(fnames[0]),4,nz))
upwp3 = np.zeros((len(prefixes),len(fnames[0]),4,nz))
mn_u0    = np.zeros((len(prefixes),nz))
mn_u1    = np.zeros((len(prefixes),nz))
mn_u2    = np.zeros((len(prefixes),nz))
mn_u3    = np.zeros((len(prefixes),nz))
mn_tke0  = np.zeros((len(prefixes),nz))
mn_tke1  = np.zeros((len(prefixes),nz))
mn_tke2  = np.zeros((len(prefixes),nz))
mn_tke3  = np.zeros((len(prefixes),nz))
mn_upwp0 = np.zeros((len(prefixes),nz))
mn_upwp1 = np.zeros((len(prefixes),nz))
mn_upwp2 = np.zeros((len(prefixes),nz))
mn_upwp3 = np.zeros((len(prefixes),nz))
for k in range(len(prefixes)) :
  for i in range(len(fnames[k])) :
    print(f"{k+1}/{len(prefixes)} : {i+1}/{len(fnames[k])}")
    nc = Dataset(fnames[k][i],"r")
    for j in range(len(p0_x)) :
      u0   [k,i,j,:] = np.array(nc["avg_u"    ][:,p0_y[j],p0_x[j]])
      u1   [k,i,j,:] = np.array(nc["avg_u"    ][:,p1_y[j],p1_x[j]])
      u2   [k,i,j,:] = np.array(nc["avg_u"    ][:,p2_y[j],p2_x[j]])
      u3   [k,i,j,:] = np.array(nc["avg_u"    ][:,p3_y[j],p3_x[j]])
      tke0 [k,i,j,:] = (np.array(nc["avg_up_up"][:,p0_y[j],p0_x[j]])+np.array(nc["avg_vp_vp"][:,p0_y[j],p0_x[j]])+np.array(nc["avg_wp_wp"][:,p0_y[j],p0_x[j]]))/2
      tke1 [k,i,j,:] = (np.array(nc["avg_up_up"][:,p1_y[j],p1_x[j]])+np.array(nc["avg_vp_vp"][:,p1_y[j],p1_x[j]])+np.array(nc["avg_wp_wp"][:,p1_y[j],p1_x[j]]))/2
      tke2 [k,i,j,:] = (np.array(nc["avg_up_up"][:,p2_y[j],p2_x[j]])+np.array(nc["avg_vp_vp"][:,p2_y[j],p2_x[j]])+np.array(nc["avg_wp_wp"][:,p2_y[j],p2_x[j]]))/2
      tke3 [k,i,j,:] = (np.array(nc["avg_up_up"][:,p3_y[j],p3_x[j]])+np.array(nc["avg_vp_vp"][:,p3_y[j],p3_x[j]])+np.array(nc["avg_wp_wp"][:,p3_y[j],p3_x[j]]))/2
      # if (has_tke[k]) :
      #   tke0[k,i,j,:] += np.array(nc["avg_tke"][:,p0_y[j],p0_x[j]])/np.array(nc["density_dry"][:,p0_y[j],p0_x[j]])
      #   tke1[k,i,j,:] += np.array(nc["avg_tke"][:,p1_y[j],p1_x[j]])/np.array(nc["density_dry"][:,p1_y[j],p1_x[j]])
      #   tke2[k,i,j,:] += np.array(nc["avg_tke"][:,p2_y[j],p2_x[j]])/np.array(nc["density_dry"][:,p2_y[j],p2_x[j]])
      #   tke3[k,i,j,:] += np.array(nc["avg_tke"][:,p3_y[j],p3_x[j]])/np.array(nc["density_dry"][:,p3_y[j],p3_x[j]])
      upwp0[k,i,j,:] = np.array(nc["avg_up_wp"][:,p0_y[j],p0_x[j]])
      upwp1[k,i,j,:] = np.array(nc["avg_up_wp"][:,p1_y[j],p1_x[j]])
      upwp2[k,i,j,:] = np.array(nc["avg_up_wp"][:,p2_y[j],p2_x[j]])
      upwp3[k,i,j,:] = np.array(nc["avg_up_wp"][:,p3_y[j],p3_x[j]])
  mn_u0   [k,:] = np.mean(u0   [k,:,:,:],axis=(0,1))
  mn_u1   [k,:] = np.mean(u1   [k,:,:,:],axis=(0,1))
  mn_u2   [k,:] = np.mean(u2   [k,:,:,:],axis=(0,1))
  mn_u3   [k,:] = np.mean(u3   [k,:,:,:],axis=(0,1))
  mn_tke0 [k,:] = np.mean(tke0 [k,:,:,:],axis=(0,1))
  mn_tke1 [k,:] = np.mean(tke1 [k,:,:,:],axis=(0,1))
  mn_tke2 [k,:] = np.mean(tke2 [k,:,:,:],axis=(0,1))
  mn_tke3 [k,:] = np.mean(tke3 [k,:,:,:],axis=(0,1))
  mn_upwp0[k,:] = np.mean(upwp0[k,:,:,:],axis=(0,1))
  mn_upwp1[k,:] = np.mean(upwp1[k,:,:,:],axis=(0,1))
  mn_upwp2[k,:] = np.mean(upwp2[k,:,:,:],axis=(0,1))
  mn_upwp3[k,:] = np.mean(upwp3[k,:,:,:],axis=(0,1))


u0_x         = np.array([0.393298519,0.405794912,0.421607921,0.432379455,0.44110289,0.452935145,0.464457739,0.472195153,0.484524493,0.492113868,0.505091047,0.510720604,0.524751712,0.532213421,0.538234126,0.548634884,0.555986581,0.56742497,0.580523026,0.585246692])
u0_y         = np.array([0.161457865,0.168310171,0.174895528,0.181682236,0.188750471,0.19537136,0.202096114,0.20848012,0.215487312,0.22244986,0.229197391,0.235955855,0.242676053,0.249443628,0.25618387,0.262924112,0.269882105,0.276439218,0.283030042,0.296952405])
u1_x         = np.array([-0.035840715,-0.017031614,-0.003397087,0.00666549,0.009622195,0.012084532,0.020103084,0.019362889,0.019876272,0.020154694,0.020340761,0.023349077,0.028436729,0.050495897,0.094516446,0.12666807,0.166745892,0.206417626,0.255001579,0.319918714,0.356307785,0.393825484,0.418056343,0.445655429,0.465597232,0.488848861,0.500423066,0.525361524,0.541053657,0.555125511,0.573422587])
u1_y         = np.array([0.033810542,0.040646449,0.047240005,0.054177954,0.060993817,0.067556397,0.074629187,0.081507914,0.087850921,0.094767004,0.101422515,0.108308531,0.115185437,0.121680596,0.12846366,0.131944023,0.135495451,0.138668776,0.142309491,0.148892115,0.155524848,0.169173884,0.182717234,0.196194074,0.20955885,0.223164154,0.236740303,0.250167944,0.263508121,0.277409529,0.290688662])
u2_x         = np.array([0.017199347,0.061285087,0.082729011,0.090236896,0.115225607,0.120925787,0.140223824,0.144271129,0.150984493,0.160984595,0.178895955,0.192658149,0.230003361,0.247137856,0.268814024,0.289076352,0.301162581,0.308939382,0.323028891,0.341347698,0.376256379,0.407628422,0.435136512,0.464878768,0.477688894,0.498407562,0.521105064,0.534831945,0.554252216,0.567821552])
u2_y         = np.array([0.033744943,0.040730269,0.044137745,0.047317448,0.054232619,0.060969217,0.067596484,0.074568143,0.081220921,0.087951141,0.094679539,0.101387893,0.115076106,0.121602243,0.128449993,0.135456274,0.142099029,0.145561171,0.148883915,0.155477472,0.169079131,0.182664391,0.196096588,0.209585272,0.223171443,0.236807724,0.25024812,0.263693983,0.277145313,0.290788882])
u3_x         = np.array([0.146468625,0.153162975,0.168781768,0.179955317,0.192903975,0.205898811,0.23358346,0.269721272,0.295609082,0.323968735,0.361527178,0.401433873,0.435808799,0.469314505,0.492305368,0.49974942,0.528409226,0.549381869,0.558112095,0.580374987,0.598741329])
u3_y         = np.array([0.047215406,0.0541078,0.067719482,0.081283786,0.094616674,0.108110825,0.121579465,0.135362432,0.142100852,0.14880465,0.155292521,0.168689185,0.182303599,0.195965391,0.20938301,0.223301729,0.236622772,0.250167944,0.263728605,0.277106136,0.290622153])
upup1_lda_x  = np.array([0.00454,0.00482,0.00456,0.0047,0.00451,0.00427,0.00395,0.00374,0.00362,0.00337,0.00353,0.00303,0.00376,0.00477,0.00884,0.01155,0.0157,0.01878,0.02171,0.02392,0.02128,0.01944,0.01914,0.01867,0.01803,0.01786,0.01847,0.01918,0.01884,0.01768,0.01744,0.01821])
upup1_lda_y  = np.array([0.03441,0.04069,0.04758,0.05459,0.0613,0.06752,0.07497,0.08115,0.08788,0.09482,0.10216,0.10886,0.11535,0.12222,0.12911,0.1326,0.13549,0.13885,0.14236,0.14905,0.15587,0.16947,0.1838,0.19615,0.19692,0.20951,0.22302,0.23671,0.25029,0.26383,0.2771,0.29087])
upup2_lda_x  = np.array([0.00794,0.00692,0.00605,0.00761,0.00694,0.0078,0.00736,0.00849,0.00811,0.00966,0.01026,0.01158,0.01293,0.01455,0.0154,0.01767,0.01654,0.0176,0.0187,0.01963,0.02005,0.01907,0.01953,0.01989,0.01972,0.01875,0.01826,0.01811,0.01771,0.01741])
upup2_lda_y  = np.array([0.03441,0.04115,0.04407,0.0478,0.05464,0.0612,0.06804,0.07449,0.08139,0.0883,0.09573,0.10186,0.11559,0.12247,0.12887,0.13559,0.14219,0.14549,0.1492,0.1563,0.16956,0.18311,0.19605,0.20947,0.22292,0.2366,0.24964,0.26372,0.27729,0.29097])
upup3_lda_x  = np.array([0.00764,0.00883,0.00899,0.00902,0.00854,0.00924,0.01092,0.01274,0.01498,0.0178,0.0209,0.02014,0.02057,0.01902,0.01866,0.01746,0.0184,0.01695,0.0178,0.01727,0.01701,0.01636])
upup3_lda_y  = np.array([0.03372,0.04064,0.05417,0.06776,0.08176,0.09464,0.10845,0.12179,0.12823,0.13555,0.14194,0.15549,0.16941,0.18236,0.19599,0.20889,0.22179,0.23607,0.24923,0.26322,0.27675,0.29037])
upup1_hwa_x  = np.array([0.01945,0.0195,0.01803,0.01837,0.01911,0.01759,0.01811,0.0182,0.01797,0.0172,0.01693,0.01564,0.01525,0.01543,0.01433,0.01333])
upup1_hwa_y  = np.array([0.1764,0.18627,0.20516,0.2158,0.22692,0.23857,0.24989,0.26324,0.27606,0.29047,0.3051,0.32085,0.33664,0.35399,0.37271,0.39152])
upup2_hwa_x  = np.array([0.02063,0.0204,0.02076,0.01937,0.01797,0.01809,0.01765,0.0168,0.0167,0.01645,0.01644,0.01509,0.01513,0.01475,0.01428,0.01377,0.01287])
upup2_hwa_y  = np.array([0.17652,0.18618,0.19612,0.20521,0.21581,0.22688,0.23839,0.2505,0.26326,0.27603,0.29057,0.3055,0.32078,0.33731,0.35389,0.37254,0.39137])
upup3_hwa_x  = np.array([0.02094,0.02073,0.02049,0.01962,0.01899,0.01716,0.01719,0.0168,0.01678,0.01671,0.01586,0.01481,0.01526,0.01376,0.01417,0.0133])
upup3_hwa_y  = np.array([0.17674,0.18602,0.19539,0.20493,0.21598,0.22681,0.23763,0.25027,0.27556,0.2902,0.30512,0.32079,0.33675,0.35384,0.37143,0.39086])
vpvp1_lda_x  = np.array([0.01114,0.01049,0.01039,0.01037,0.01099,0.01073,0.01034,0.00999,0.00973,0.00953,0.00921,0.01032,0.01077,0.01189,0.01163,0.01243,0.01193,0.012,0.01171,0.01163,0.01216,0.01215,0.01218,0.01141,0.01168,0.01111,0.01108,0.01029,0.01065,0.01031])
vpvp1_lda_y  = np.array([0.03354,0.04031,0.04703,0.05436,0.06135,0.06762,0.0747,0.0814,0.08808,0.09512,0.10173,0.10834,0.11515,0.12165,0.12568,0.1288,0.13556,0.14259,0.14866,0.15566,0.16949,0.18274,0.197,0.20954,0.22307,0.23679,0.24993,0.26332,0.27704,0.29057])
vpvp2_lda_x  = np.array([0.00475,0.00515,0.00574,0.00627,0.00704,0.00725,0.00846,0.00868,0.00941,0.01038,0.01128,0.01169,0.01238,0.01288,0.01323,0.01292,0.013,0.01254,0.0119,0.01187,0.01205,0.01197,0.01075,0.01068,0.01163,0.01078,0.01045,0.01028])
vpvp2_lda_y  = np.array([0.03367,0.04071,0.04725,0.05405,0.06121,0.06805,0.07438,0.08167,0.08821,0.09471,0.10123,0.10874,0.1147,0.12215,0.12934,0.13499,0.14267,0.15579,0.16947,0.18282,0.19658,0.20902,0.22302,0.23598,0.24983,0.26317,0.27721,0.29111])
vpvp3_lda_x  = np.array([0.00906,0.00965,0.01093,0.01221,0.01292,0.01291,0.01375,0.01233,0.01294,0.01244,0.01213,0.01276,0.01152,0.0112,0.01177,0.01195,0.0111,0.01047,0.01058,0.01031,0.0097,0.00996])
vpvp3_lda_y  = np.array([0.03353,0.04044,0.05395,0.06772,0.08085,0.09446,0.10816,0.12154,0.12813,0.13532,0.14126,0.15572,0.16906,0.18281,0.19576,0.20891,0.22315,0.23578,0.24943,0.26356,0.27667,0.29022])
vpvp1_hwa_x  = np.array([0.01082,0.01138,0.01162,0.0113,0.01061,0.01089,0.01095,0.01007,0.00988,0.00957,0.00928,0.00956,0.00901,0.00887,0.00888,0.00887])
vpvp1_hwa_y  = np.array([0.1765,0.18625,0.19546,0.20518,0.21601,0.22721,0.23805,0.25037,0.27594,0.29048,0.30554,0.32093,0.33685,0.354,0.37264,0.39152])
vpvp2_hwa_x  = np.array([0.01073,0.01056,0.01019,0.0101,0.0103,0.01,0.01005,0.00954,0.00971,0.01017,0.00971,0.00906,0.00905,0.00904,0.00877,0.00839,0.00836])
vpvp2_hwa_y  = np.array([0.17638,0.18558,0.19589,0.20527,0.2161,0.22704,0.23789,0.25042,0.26279,0.27633,0.29083,0.30526,0.3212,0.33728,0.35393,0.37225,0.39131])
vpvp3_hwa_x  = np.array([0.0102,0.01057,0.01077,0.01014,0.0103,0.00996,0.00967,0.00974,0.01012,0.00984,0.00945,0.00914,0.00922,0.00876,0.00866,0.00832,0.00819])
vpvp3_hwa_y  = np.array([0.17648,0.18554,0.19534,0.20459,0.21518,0.22661,0.23796,0.25029,0.26214,0.27561,0.29009,0.30512,0.3209,0.33735,0.35368,0.37235,0.39122])
wpwp1_lda_x  = np.array([0.00591,0.00645,0.00564,0.00567,0.00552,0.00501,0.00431,0.00436,0.00425,0.00376,0.004,0.00453,0.00538,0.00822,0.01123,0.01136,0.01177,0.01052,0.0097,0.00735,0.00614,0.00657,0.00655,0.00696,0.00705,0.00693,0.00654,0.00681,0.00669,0.0077,0.00724])
wpwp1_lda_y  = np.array([0.03404,0.04097,0.04719,0.05445,0.06094,0.06768,0.07493,0.0811,0.08757,0.09494,0.10157,0.108,0.11521,0.12207,0.1287,0.13243,0.13588,0.13855,0.14238,0.14854,0.15565,0.16913,0.18344,0.19646,0.20972,0.2232,0.23686,0.24975,0.26382,0.27779,0.29128])
wpwp2_lda_x  = np.array([0.00881,0.00842,0.00885,0.00869,0.0082,0.0079,0.00883,0.00907,0.0077,0.00798,0.00825,0.00904,0.00844,0.00809,0.00817,0.00772,0.00785,0.00723,0.00706,0.00714,0.0067,0.00735,0.007,0.00689,0.00668,0.0071,0.00694,0.00696,0.00683])
wpwp2_lda_y  = np.array([0.03442,0.04084,0.04386,0.04718,0.05484,0.06121,0.06769,0.07532,0.08118,0.08764,0.09534,0.1014,0.11567,0.12232,0.1285,0.13555,0.14208,0.14532,0.15608,0.16912,0.18322,0.19603,0.20882,0.22339,0.23717,0.25012,0.26406,0.27722,0.29089])
wpwp1_hwa_x  = np.array([0.0065,0.00634,0.00651,0.00665,0.00645,0.0067,0.00685,0.00692,0.00661,0.00696,0.00677,0.00688,0.00701,0.00647,0.00642,0.00655,0.00618])
wpwp1_hwa_y  = np.array([0.17704,0.18554,0.19536,0.2055,0.21592,0.22724,0.23824,0.25078,0.2628,0.27682,0.29063,0.30553,0.32127,0.33778,0.35465,0.37206,0.39135])
wpwp2_hwa_x  = np.array([0.0077,0.00741,0.00726,0.00736,0.00685,0.00703,0.00688,0.00677,0.00657,0.00646,0.00666,0.00663,0.00634,0.00627,0.00623,0.00608,0.00604])
wpwp2_hwa_y  = np.array([0.17762,0.18586,0.19571,0.20536,0.21585,0.22767,0.23799,0.25066,0.26298,0.27671,0.29051,0.30571,0.32119,0.33756,0.35501,0.37217,0.39111])
nupwp1_lda_x = np.array([0.00041,0.00064,0.00038,0.00069,0.00053,0.00052,0.00062,-0.00001,0.00019,0.00008,0.00021,0.00023,0.00068,0.00225,0.00568,0.00648,0.00849,0.00841,0.00812,0.00624,0.00446,0.00418,0.00393,0.00412,0.00382,0.00451,0.00429,0.00457,0.00383,0.00506,0.0049])
nupwp1_lda_y = np.array([0.03341,0.04118,0.04752,0.05427,0.06161,0.06824,0.07471,0.08144,0.08825,0.09498,0.10119,0.10858,0.1156,0.122,0.1287,0.13246,0.13583,0.13948,0.14232,0.14885,0.15574,0.16898,0.18288,0.19594,0.2099,0.22311,0.23678,0.25064,0.26376,0.27756,0.29051])
nupwp2_lda_x = np.array([-0.00012,0.00067,0.00071,0.0012,0.00127,0.00163,0.00243,0.0034,0.003,0.00395,0.00397,0.00478,0.00519,0.00576,0.00491,0.00559,0.00564,0.00563,0.00553,0.00522,0.00552,0.00475,0.00511,0.00466,0.00435,0.00436,0.00422,0.00413,0.00419,0.00406])
nupwp2_lda_y = np.array([0.03356,0.04074,0.04414,0.04729,0.05373,0.06106,0.06792,0.0744,0.0811,0.08808,0.09514,0.10151,0.11508,0.1218,0.12855,0.13534,0.14188,0.1448,0.14835,0.15578,0.16892,0.18256,0.19559,0.20961,0.22272,0.2368,0.25008,0.26389,0.27777,0.29081])
nupwp1_hwa_x = np.array([0.00388,0.00402,0.00421,0.00402,0.00424,0.00446,0.00399,0.0044,0.00409,0.00431,0.00403,0.00421,0.00391,0.00381,0.00398,0.00391,0.00352])
nupwp1_hwa_y = np.array([0.17668,0.18549,0.19493,0.20528,0.21573,0.2268,0.23842,0.2503,0.26355,0.27677,0.29048,0.30541,0.32071,0.33735,0.35442,0.37258,0.39087])
nupwp2_hwa_x = np.array([0.00543,0.00515,0.00495,0.00496,0.00412,0.00411,0.0038,0.00387,0.00399,0.00379,0.00389,0.00375,0.00383,0.00374,0.00342,0.00327])
nupwp2_hwa_y = np.array([0.17684,0.1861,0.19509,0.20528,0.22696,0.23811,0.25064,0.26273,0.277,0.29056,0.30561,0.32072,0.33702,0.3543,0.37266,0.39085])

obs_zmin = max(upup1_lda_y[ 0],vpvp1_lda_y[ 0],wpwp1_lda_y[ 0])
obs_zmax = min(upup1_lda_y[-1],vpvp1_lda_y[-1],wpwp1_lda_y[-1])
obs_z1   = np.arange(obs_zmin,obs_zmax,0.005)
obs_upup1 = np.interp(obs_z1,upup1_lda_y,upup1_lda_x)
obs_vpvp1 = np.interp(obs_z1,vpvp1_lda_y,vpvp1_lda_x)
obs_wpwp1 = np.interp(obs_z1,wpwp1_lda_y,wpwp1_lda_x)
obs_tke1  = (obs_upup1+obs_vpvp1+obs_wpwp1)/2
obs_zmin = max(upup2_lda_y[ 0],vpvp2_lda_y[ 0],wpwp2_lda_y[ 0])
obs_zmax = min(upup2_lda_y[-1],vpvp2_lda_y[-1],wpwp2_lda_y[-1])
obs_z2   = np.arange(obs_zmin,obs_zmax,0.005)
obs_upup2 = np.interp(obs_z2,upup2_lda_y,upup2_lda_x)
obs_vpvp2 = np.interp(obs_z2,vpvp2_lda_y,vpvp2_lda_x)
obs_wpwp2 = np.interp(obs_z2,wpwp2_lda_y,wpwp2_lda_x)
obs_tke2  = (obs_upup2+obs_vpvp2+obs_wpwp2)/2


fig,ax = plt.subplots(2,2,figsize=(8,8))
for k in range(len(prefixes)) :
  ax[0,0].plot(mn_u0[k,:]/10,z/.02,label=f"{labels[k]} P0",color=colors[k],linestyle='-')
  ax[0,1].plot(mn_u1[k,:]/10,z/.02,label=f"{labels[k]} P1",color=colors[k],linestyle='-')
  ax[1,0].plot(mn_u2[k,:]/10,z/.02,label=f"{labels[k]} P2",color=colors[k],linestyle='-')
  ax[1,1].plot(mn_u3[k,:]/10,z/.02,label=f"{labels[k]} P3",color=colors[k],linestyle='-')
ax[0,0].scatter(u0_x,u0_y/.135,label="LDA P0",facecolors='black',edgecolors='black',s=25)
ax[0,1].scatter(u1_x,u1_y/.135,label="LDA P1",facecolors='black',edgecolors='black',s=25)
ax[1,0].scatter(u2_x,u2_y/.135,label="LDA P2",facecolors='black',edgecolors='black',s=25)
ax[1,1].scatter(u3_x,u3_y/.135,label="LDA P3",facecolors='black',edgecolors='black',s=25)
for axloc in ax.flatten() :
  axloc.set_xlim(-0.2,0.8)
  axloc.set_ylim(0,3)
  axloc.grid(True)
  axloc.legend()
  axloc.margins(x=0)
  axloc.set_xlabel(r"$u/u_\infty$")
  axloc.set_ylabel(r"$z/h$")
fig.tight_layout()
plt.savefig("u.png")
plt.show()
plt.close()


fig,ax = plt.subplots(2,2,figsize=(8,8))
for k in range(len(prefixes)) :
  ax[0,0].plot(mn_tke0[k]/10/10,z/.02,label=f"{labels[k]} P0",color=colors[k],linestyle='-')
  ax[0,1].plot(mn_tke1[k]/10/10,z/.02,label=f"{labels[k]} P1",color=colors[k],linestyle='-')
  ax[1,0].plot(mn_tke2[k]/10/10,z/.02,label=f"{labels[k]} P2",color=colors[k],linestyle='-')
  ax[1,1].plot(mn_tke3[k]/10/10,z/.02,label=f"{labels[k]} P3",color=colors[k],linestyle='-')
ax[0,1].scatter(obs_tke1,obs_z1/.135,label="LDA P1",facecolors='black',edgecolors='black',s=25)
ax[1,0].scatter(obs_tke2,obs_z2/.135,label="LDA P2",facecolors='black',edgecolors='black',s=25)
for axloc in ax.flatten() :
  axloc.set_xlim(0,0.025)
  axloc.set_ylim(0,3)
  axloc.grid(True)
  axloc.legend()
  axloc.margins(x=0)
  axloc.set_xlabel(r"$TKE_{resolved}/{u_\infty^2}$")
  axloc.set_ylabel(r"$z/h$")
fig.tight_layout()
plt.savefig("tke.png")
plt.show()
plt.close()


fig,ax = plt.subplots(2,2,figsize=(8,8))
for k in range(len(prefixes)) :
  ax[0,0].plot(-mn_upwp0[k,:]/10/10,z/.02,label=f"{labels[k]} P0",color=colors[k],linestyle='-')
  ax[0,1].plot(-mn_upwp1[k,:]/10/10,z/.02,label=f"{labels[k]} P1",color=colors[k],linestyle='-')
  ax[1,0].plot(-mn_upwp2[k,:]/10/10,z/.02,label=f"{labels[k]} P2",color=colors[k],linestyle='-')
  ax[1,1].plot(-mn_upwp3[k,:]/10/10,z/.02,label=f"{labels[k]} P3",color=colors[k],linestyle='-')
ax[0,1].scatter(nupwp1_lda_x,nupwp1_lda_y/.135,label="LDA P1",facecolors='black',edgecolors='black',s=25)
ax[1,0].scatter(nupwp2_lda_x,nupwp2_lda_y/.135,label="LDA P2",facecolors='black',edgecolors='black',s=25)
ax[0,1].scatter(nupwp1_hwa_x,nupwp1_hwa_y/.135,label="HWA P1",facecolors='red'  ,edgecolors='red'  ,s=25)
ax[1,0].scatter(nupwp2_hwa_x,nupwp2_hwa_y/.135,label="HWA P2",facecolors='red'  ,edgecolors='red'  ,s=25)
for axloc in ax.flatten() :
  axloc.set_xlim(-0.002,0.01)
  axloc.set_ylim(0,3)
  axloc.grid(True)
  axloc.legend()
  axloc.margins(x=0)
  axloc.set_xlabel(r"$-\overline{u'w'}/{u_\infty^2}$")
  axloc.set_ylabel(r"$z/h$")
fig.tight_layout()
plt.savefig("upwp.png")
plt.show()
plt.close()



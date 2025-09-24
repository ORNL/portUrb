from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

save_figs = False
# results_dir = "/lustre/orion/stf006/scratch/imn/portUrb/build"
results_dir = "/lustre/storm/nwp501/scratch/imn/portUrb/build"

# Main dictionary for numpy arrays and scalars
results = {}

def is_3d(var) :
  return var.dimensions[0]=="z" and var.dimensions[1]=="y" and var.dimensions[2]=="x"

def is_sfc(var) :
  return var.dimensions[0]=="y" and var.dimensions[1]=="x"

def is_col(var) :
  return var.dimensions[0]=="z" or var.dimensions[0]=="z_halo" or var.dimensions[0]=="zi"

def process(prefix_fixed,prefix_variable,varlist,time) :
  ncf = Dataset(f"{results_dir}/{prefix_fixed}_{time:08d}.nc")
  ncv = Dataset(f"{results_dir}/{prefix_variable}_{time:08d}.nc")
  for vname in varlist :
    var = ncf.variables[vname]
    if (is_3d(var)) :
      # Plot the mean
      qf = np.quantile(np.array(ncf.variables[vname][:,:,:]),[0.,0.25,0.5,0.75,1.],axis=(1,2))
      qv = np.quantile(np.array(ncv.variables[vname][:,:,:]),[0.,0.25,0.5,0.75,1.],axis=(1,2))
      zf = np.array(ncf.variables["z"][:])
      zv = np.array(ncv.variables["z"][:])
      plt.plot(qf[0],zf,color="black",linestyle=":" ,label="fixed min")
      plt.plot(qv[0],zv,color="red"  ,linestyle=":" ,label="var   min")
      plt.plot(qf[1],zf,color="black",linestyle="--",label="fixed Q1" )
      plt.plot(qv[1],zv,color="red"  ,linestyle="--",label="var   Q1" )
      plt.plot(qf[2],zf,color="black",linestyle="-" ,label="fixed med")
      plt.plot(qv[2],zv,color="red"  ,linestyle="-" ,label="var   med")
      plt.plot(qf[3],zf,color="black",linestyle="--",label="fixed Q3" )
      plt.plot(qv[3],zv,color="red"  ,linestyle="--",label="var   Q3" )
      plt.plot(qf[4],zf,color="black",linestyle=":" ,label="fixed max")
      plt.plot(qv[4],zv,color="red"  ,linestyle=":" ,label="var   max")
      plt.legend()
      plt.title(f"{prefix_fixed}_{vname}_mean")
      if (save_figs) :
        plt.savefig(f"{prefix_fixed}_{vname}_mean.png")
      plt.show()
      plt.close()


process(prefix_fixed="ABL_neutral_fixed",
        prefix_variable="ABL_neutral_variable",
        varlist=["density_dry","temperature","uvel","vvel","wvel","TKE","theta_pert","density_pert"],
        time=1)

process(prefix_fixed="supercell_kessler_fixed",
        prefix_variable="supercell_kessler_variable",
        varlist=["density_dry","temperature","uvel","vvel","wvel","water_vapor","cloud_liquid","precip_liquid",
                 "TKE","theta_pert","density_pert","pressure_pert"],
        time=1)


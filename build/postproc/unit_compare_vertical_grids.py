from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

save_figs = False
results_dir = "/lustre/orion/stf006/scratch/imn/portUrb/build"

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
      plt.plot(np.mean(np.array(ncf.variables[vname][:,:,:]),axis=(1,2)),
               np.array(ncf.variables["z"][:]),color="black",label="fixed")
      plt.plot(np.mean(np.array(ncv.variables[vname][:,:,:]),axis=(1,2)),
               np.array(ncv.variables["z"][:]),color="red",label="variable")
      plt.legend()
      plt.title(f"{prefix_fixed}_{vname}_mean")
      if (save_figs) :
        plt.savefig(f"{prefix_fixed}_{vname}_mean.png")
      plt.show()
      plt.close()
      # Plot the standard deviation
      plt.plot(np.std(np.array(ncf.variables[vname][:,:,:]),axis=(1,2)),
               np.array(ncf.variables["z"][:]),color="black",label="fixed")
      plt.plot(np.std(np.array(ncv.variables[vname][:,:,:]),axis=(1,2)),
               np.array(ncv.variables["z"][:]),color="red",label="variable")
      plt.legend()
      plt.title(f"{prefix_fixed}_{vname}_std")
      if (save_figs) :
        plt.savefig(f"{prefix_fixed}_{vname}_std.png")
      plt.show()
      plt.close()


process(prefix_fixed="ABL_neutral_fixed",
        prefix_variable="ABL_neutral_variable",
        varlist=["uvel","vvel","wvel","TKE","theta_pert","density_pert"],
        time=1)

process(prefix_fixed="supercell_kessler_fixed",
        prefix_variable="supercell_kessler_variable",
        varlist=["uvel","vvel","wvel","water_vapor","cloud_liquid","precip_liquid",
                 "TKE","theta_pert","density_pert","pressure_pert"],
        time=1)


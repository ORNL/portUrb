from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

generate_diff_plots = False
results_dir  = "/lustre/orion/stf006/scratch/imn/portUrb/build"

# Main dictionary for numpy arrays and scalars
results = {}

def is_3d(var) :
  return var.dimensions[0]=="z" and var.dimensions[1]=="y" and var.dimensions[2]=="x"

def is_sfc(var) :
  return var.dimensions[0]=="y" and var.dimensions[1]=="x"

def is_col(var) :
  return var.dimensions[0]=="z" or var.dimensions[0]=="z_halo" or var.dimensions[0]=="zi"

def is_trace(var) :
  return var.dimensions[0]=="num_time_steps"

def process(prefix,time,exclude) :
  nc = Dataset(f"{results_dir}/{prefix}_{time:08d}.nc")
  for vname in nc.variables.keys() :
    if np.issubdtype( nc.variables[vname].dtype , np.floating ) :
      if (not vname in exclude) :
        var = nc.variables[vname]
        if (is_3d(var)) :
          data = np.array(var[:,:,:])
          results[f"{prefix}_{vname}_mean"] = np.mean(data,axis=(1,2))
          results[f"{prefix}_{vname}_std" ] = np.std (data,axis=(1,2))
        elif (is_sfc(var)) :
          data = np.array(var[:,:])
          results[f"{prefix}_{vname}_mean"] = np.mean(data)
          results[f"{prefix}_{vname}_std" ] = np.std (data)
        elif (is_col(var)) :
          data = np.array(var[:])
          results[f"{prefix}_{vname}"] = data
        elif (is_trace(var)) :
          data = np.array(var[:])
          results[f"{prefix}_{vname}_mean"] = np.mean(data)
          results[f"{prefix}_{vname}_std" ] = np.std (data)

files = ["ABL_convective",
         "ABL_neutral_fixed",
         "ABL_neutral_variable",
         "ABL_stable",
         "city_2m",
         "supercell_kessler_fixed",
         "supercell_kessler_variable",
         "supercell_morr2mom",
         "turbulent_wind-20.000000",
         "turbulent_wind-20.000000_precursor",
         "turbulent_wind-5.000000",
         "turbulent_wind-5.000000_precursor"]
for file in files :
  process(prefix=file,time=1,exclude=["x","y","z","zi"])

baseline = loaded_data = np.load(f"{results_dir}/inputs/unit_baseline.npz")

for key in baseline.keys() :
  denom = np.mean(np.abs(baseline[key]))
  numer = np.mean(np.abs(results[key]-baseline[key]))
  if (denom > 1.e-15) :
    diff = numer / denom
  else :
    diff = numer 
  if (diff > 1.e-14) :
    print(f"{key:<60}: {diff}")
    if (generate_diff_plots and len(results[key].shape) == 1) :
      plt.plot(baseline[key],color="black",label="baseline")
      plt.plot(results [key],color="red"  ,label="results" )
      plt.title(key)
      plt.legend()
      plt.savefig(f"{key}.png")
      plt.close()


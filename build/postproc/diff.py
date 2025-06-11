from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray

winds  = [ "5.000000",
           "7.000000",
           "9.000000",
           "11.000000",
           "13.000000",
           "15.000000",
           "17.000000",
           "19.000000",
           "21.000000",
           "23.000000"]

prefixes = [f"turbulent_fixed-yaw-upstream-neutral_wind-{wind}" for wind in winds]

for i in range(len(winds)) :
  nc1 = Dataset(f"{prefixes[i]}_fixed-_precursor_00000001.nc","r")
  nc2 = Dataset(f"{prefixes[i]}_floating-_precursor_00000001.nc","r")
  for vname in nc1.variables.keys() :
    v1 = np.array(nc1[vname])
    v2 = np.array(nc2[vname])
    print(f"TKE [{winds[i]}], Var[{vname:20}], MAE: {np.sum(np.abs(v2-v1))/v1.size:12.5}")
  print("\n")


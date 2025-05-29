from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray

tkes  = [ "0.000000",
          "0.142857",
          "0.285714",
          "0.428571",
          "0.571429",
          "0.714286",
          "0.857143",
          "1.000000"]

prefixes = [f"turbulent_nrel_5mw_smaller_f_TKE-{tke}_precursor" for tke in tkes]

for i in range(1,8) :
  nc1 = Dataset(f"{prefixes[0]}_00000020.nc","r")
  nc2 = Dataset(f"{prefixes[i]}_00000020.nc","r")
  for vname in nc1.variables.keys() :
    v1 = np.array(nc1[vname])
    v2 = np.array(nc2[vname])
    print(f"TKE [{tkes[i]}], Var[{vname:20}], MAE: {np.sum(np.abs(v2-v1))/v1.size:12.5}")
  print("\n")


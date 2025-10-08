import numpy as np
import sys

fh = open(sys.argv[1],'r')
ind = 0
for line in fh :
  if line.startswith('v') :
    _, x, y, z = line.strip().split()[:4]
    mn_x = float(x) if ind == 0 else np.minimum(mn_x,float(x))
    mx_x = float(x) if ind == 0 else np.maximum(mx_x,float(x))
    mn_y = float(y) if ind == 0 else np.minimum(mn_y,float(y))
    mx_y = float(y) if ind == 0 else np.maximum(mx_y,float(y))
    mn_z = float(z) if ind == 0 else np.minimum(mn_z,float(z))
    mx_z = float(z) if ind == 0 else np.maximum(mx_z,float(z))
    ind += 1
print(f"Domain extents for file [{sys.argv[1]}]:")
print(f"x [min,max,range]: [ {mn_x:10.5f} , {mx_x:10.5f} , {mx_x-mn_x:10.5f} ]",)
print(f"y [min,max,range]: [ {mn_y:10.5f} , {mx_y:10.5f} , {mx_y-mn_y:10.5f} ]",)
print(f"z [min,max,range]: [ {mn_z:10.5f} , {mx_z:10.5f} , {mx_z-mn_z:10.5f} ]",)


# Vertically interpolate cloud cover into cloud area fraction
# For more documentation see cloud_area_fraction.ipynb

import os
import xarray as xr
import numpy as np

SOURCE = 'QUBICC'        # Can be 'NARVAL', 'QUBICC'. To set paths, input grids and variable names.

# Define all paths
path = '/pf/b/b309170/bd1179_work/qubicc'
grid_path = '/pf/b/b309170/bd1179_work/qubicc/grids'
output_path = os.path.join(path, 'vcg_data', 'cl_area_frac')

# Load files (ds_zh_lr = ds_zhalf_lowres)
ds_zh_lr = xr.open_dataset(os.path.join(grid_path, 'zghalf_icon-a_capped_upsampled_R02B05_QUBICC.nc'))
ds_zh_hr = xr.open_dataset(os.path.join(grid_path, 'qubicc_l91_zghalf_ml_0015_R02B09_G.nc'))  
# Extract values
zh_lr = ds_zh_lr.zghalf.values # Should be 32 x 20971520
zh_hr = ds_zh_hr.zghalf.values  # Should be 92 x 20971520

HORIZ_FIELDS = zh_lr.shape[1]
VERT_LAYERS_LR = zh_lr.shape[0] - 1
VERT_LAYERS_HR = zh_hr.shape[0] - 1

weights = np.load(os.path.join('/pf/b/b309170/bd1179_work/qubicc/grids', 'weights_QUBICC_R02B09_cloud_area_fraction.npy'))

files = os.listdir(os.path.join(path, 'orig_data'))
for input_file in files:

    # Skip if the file is already in output_path. Careful: g2 != nc!!
    if 'int_var_' + input_file[:-2] + 'nc' in os.listdir(output_path):
        continue

    print(input_file)

    DS = xr.open_dataset(os.path.join(path, 'orig_data', input_file))
    clc = DS.clc.values
    TIME_STEPS = len(DS.time)

    # Modify the ndarray. Desired output shape: (1, 31, 4887488). (clc_out = clc, vertically interpolated)
    clc_out = np.full((TIME_STEPS, VERT_LAYERS_LR, HORIZ_FIELDS), np.nan)

    for t in range(TIME_STEPS):
        for j in range(VERT_LAYERS_LR):    
            clc_out[t][j] = np.max(weights[j]*clc[t], axis=0) #Element-wise product

    clc_new_da = xr.DataArray(clc_out, coords={'time':DS.time, 'height':DS.height[:VERT_LAYERS_LR]}, 
                              dims=['time', 'height', 'cells'], name='clc') 

    # Save it in a new file
    output_file = 'int_var_' + input_file
    clc_new_da.to_netcdf(os.path.join(output_path, output_file))
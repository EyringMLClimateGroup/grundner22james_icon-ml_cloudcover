# Vertically interpolate cloud cover into cloud area fraction
# For more documentation see cloud_area_fraction.ipynb
# Reserve 180GB, need ~102 seconds per file

import os
import xarray as xr
import numpy as np

# Reserve 180GB for NARVAL, 500GB on QUBICC
SOURCE = 'NARVAL'        # Can be 'NARVAL', 'QUBICC'. To set paths, input grids and variable names.

# Define all paths
path = '/pf/b/b309170/scratch/orig_files'
base_path = '/pf/b/b309170/my_work/NARVAL'
output_path = os.path.join(base_path, 'data_R02B05', 'cl_area_frac')
dates = os.listdir(path)[1:]

# Load files (ds_zh_lr = ds_zhalf_lowres)
ds_zh_lr = xr.open_dataset(os.path.join(base_path, 'grid_extpar/zghalf_icon-a_capped_upsampled_R02B05.nc'))
ds_zh_hr = xr.open_dataset(os.path.join(base_path, 'grid_extpar/z_ifc_R02B10_NARVAL_fg_DOM01_ML.nc'))  
# Extract values
zh_lr = ds_zh_lr.zghalf.values
zh_hr = ds_zh_hr.z_ifc.values

HORIZ_FIELDS = zh_lr.shape[1]
VERT_LAYERS_LR = zh_lr.shape[0] - 1
VERT_LAYERS_HR = zh_hr.shape[0] - 1

# Requires 90GB, actually 160GB
weights = np.load(os.path.join('/pf/b/b309170/my_work/NARVAL/grid_extpar', 'weights_NARVAL_R02B10_cloud_area_fraction.npy'))

for date in dates:    
    files = os.listdir(os.path.join(path, date))
    for input_file in files:
        
        # Skip if the file is already in output_path
        if 'int_var_' + input_file in os.listdir(output_path):
            continue
        
        print(input_file)
        
        DS = xr.open_dataset(os.path.join(path, date, input_file))
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
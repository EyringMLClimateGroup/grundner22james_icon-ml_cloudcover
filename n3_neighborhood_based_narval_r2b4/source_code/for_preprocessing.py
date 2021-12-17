import xarray as xr
import pandas as pd
import numpy as np

def load_day(day, no_NNs, path, data_source='narval', resolution_narval='R02B04'):
    '''
    We load data from path from a given day. The data is saved in a folder per variable manner.
    This function returns an array of dataframes, where each dataframe corresponds to a specific vertical layer
    
    Parameters:
        day (string): The data of which day should we load (YYYYMMDD00 for narval, YYYYMMDD for qubicc)
        no_NNs (int): How many NNs do we want to train (= one per vertical layer)
        path (string): Path to the data
        data_source (string): 'narval' or 'qubicc'
        resolution_narval: 'R02B04' or 'R02B05'. Affects which fr_lake file is loaded.
        
    Returns:
        dfs: An array of no_NNs many dataframes, each providing training data for clc on a specific vertical layer
    '''
    day = str(day) # In case it wasn't passed as a string
    
    # Create DataFrame for loaded data
    if data_source == 'narval':
        vars_3d = ['qv', 'qc', 'qi', 'temp', 'pres', 'rho', 'zg']
        # vars_2d = ['fr_lake', 'fr_seaice']
        vars_prev = ['clc_prev']
        output = ['clc']
    elif data_source == 'qubicc':
        vars_3d = ['hus', 'clw', 'cli', 'ta', 'pfull', 'rho', 'zg']
        # vars_2d = ['fr_lake', 'fr_seaice']
        vars_prev = ['cl_prev']   
        output = ['cl']
    vars_2d = ['fr_lake']
    columns = []    
    for s in vars_3d:
        columns.append(s+'_i-2')
        columns.append(s+'_i-1')
        columns.append(s+'_i')
        columns.append(s+'_i+1')
        columns.append(s+'_i+2')
    columns.extend(vars_2d)
    columns.extend(vars_prev)
    columns.extend(output)
    
    dfs = []
    for i in range(no_NNs):
        dfs.append(pd.DataFrame(columns=columns))

    ## Output
    #clc
    vars = output
    for i in range(len(vars)):
        # Filenames depend on data-source
        if data_source == 'narval':
            # clc-filename narval: int_var_clc_R02B04_NARVALI_2013123100_cloud_DOM01_0036.nc
            clc_filenames = '/int_var_'+vars[i]+'_'+resolution_narval+'*'+day+'*_cloud_DOM01_00*.nc'
        elif data_source == 'qubicc':
            # cl-filename qubicc: int_var_hc2_02_p1m_cl_ml_20041110T110000Z.nc
            clc_filenames = '/int_var_hc2_02_p1m_'+vars[i]+'_ml_'+day+'*.nc'
        DS = xr.open_mfdataset(path+vars[i]+clc_filenames, combine='by_coords')
        var_array = getattr(DS, vars[i]).values
        not_nan = ~np.isnan(var_array[0,30,:]) #The surface-nearest layer 30 shall not contain NAN-values
        timesteps = var_array.shape[0]        
        var_array_notnan = var_array[:,:,not_nan]  #var_array_notnan.shape=25x31x1131
        # For every vertical layer we have to fill the output information of this day in the corresponding dataset
        for j in range(no_NNs):
            vert_layers = var_array.shape[1]
            # We are not interested in the uppermost layers (denoted by small indices)
            ind = vert_layers - no_NNs + j
            # We do not save the initial timestep as there is no preceding information on clc
            dfs[j][vars[i]] = np.reshape(var_array_notnan[1:,ind,:],[-1]) #shape=27144
            dfs[j][vars_prev[i]] = np.reshape(var_array_notnan[:-1,ind,:],[-1])
            
    ## Time-invariant input
    #zg
    DS = xr.open_dataset(path+'zg/zg_icon-a_capped.nc')
    var_array = DS.zg.values
    var_array_notnan = var_array[:,not_nan] #var_array_notnan.shape=31x1131
    var_array_notnan = np.repeat(np.expand_dims(var_array_notnan, 0), timesteps, axis=0) #var_array_notnan.shape=25x31x1131
    for j in range(no_NNs):
        ind = vert_layers - no_NNs + j
        dfs[j]['zg_i-2'] = np.reshape(var_array_notnan[1:,ind-2,:], [-1])
        dfs[j]['zg_i-1'] = np.reshape(var_array_notnan[1:,ind-1,:], [-1])
        dfs[j]['zg_i'] = np.reshape(var_array_notnan[1:,ind,:], [-1])
        try:
            dfs[j]['zg_i+1'] = np.reshape(var_array_notnan[1:,ind+1,:], [-1])
            dfs[j]['zg_i+2'] = np.reshape(var_array_notnan[1:,ind+2,:], [-1])
        except IndexError:
            pass
        
    #fr_lake
    if data_source == 'narval':
        DS = xr.open_dataset(path+'../grid_extpar/fr_lake_'+resolution_narval+'_NARVAL_fg_DOM01.nc')
        var_array = DS.FR_LAKE.values
    elif data_source == 'qubicc':
        DS = xr.open_dataset(path+'/fr_lake/fr_lake_'+resolution_narval+'.nc')
        var_array = DS.lake.values
    var_array = np.repeat(np.expand_dims(var_array, 0), timesteps, axis=0)
    var_array = np.repeat(np.expand_dims(var_array, 1), 31, axis=1)
    var_array_notnan = var_array[:,:,not_nan]
    for j in range(no_NNs):
        ind = vert_layers - no_NNs + j
        dfs[j]['fr_lake'] = np.reshape(var_array_notnan[1:,ind,:],[-1])
        
#     ## 2D input
#     #
#     vars = ['fr_seaice']
#     for i in range(len(vars)):
#         DS = xr.open_mfdataset(path+vars[i]+'/'+vars[i]+'_R02B04*'+day+'_fg_DOM01_00*.nc', 
#                                combine='by_coords')
#         var_array = getattr(DS, vars[i]).values
#         var_array = np.repeat(np.expand_dims(var_array, 1), 31, axis=1)
#         var_array_notnan = var_array[:,:,not_nan]
#         for j in range(no_NNs):
#             ind = vert_layers - no_NNs + j
#             dfs[j][vars[i]] = np.reshape(var_array_notnan[1:,ind,:],[-1])

    ## Hourly data
    #3D input
    vars = vars_3d
    vars.remove('zg')
    for i in range(len(vars)):
        # Filenames depend on data-source
        if data_source == 'narval':
            # 3d-filename narval: int_var_qc_R02B04_NARVALII_2016072800_fg_DOM01_0021.nc
            filenames = '/int_var_'+vars[i]+'_'+resolution_narval+'*'+day+'*_fg_DOM01_00*.nc'
        elif data_source == 'qubicc':
            # 3d-filename qubicc: int_var_hc2_02_p1m_clw_ml_20041110T090000Z.nc
            filenames = '/int_var_hc2_02_p1m_'+vars[i]+'_ml_'+day+'*.nc'
        DS = xr.open_mfdataset(path+vars[i]+filenames, combine='by_coords')
        if vars[i] == 'clw':
            var_array = getattr(DS, 'qclw_phy').values
        else:
            var_array = getattr(DS, vars[i]).values
        var_array_notnan = var_array[:,:,not_nan]
        for j in range(no_NNs):
            ind = vert_layers - no_NNs + j
            dfs[j][vars[i]+'_i-2'] = np.reshape(var_array_notnan[1:,ind-2,:], [-1])
            dfs[j][vars[i]+'_i-1'] = np.reshape(var_array_notnan[1:,ind-1,:], [-1])
            dfs[j][vars[i]+'_i'] = np.reshape(var_array_notnan[1:,ind,:], [-1])
            try:
                dfs[j][vars[i]+'_i+1'] = np.reshape(var_array_notnan[1:,ind+1,:], [-1])
                dfs[j][vars[i]+'_i+2'] = np.reshape(var_array_notnan[1:,ind+2,:], [-1])
            except IndexError:
                pass
    
    return dfs
## Functions for qubicc_models_plots.ipynb and narval_r2b4_on_narval_r2b5.ipynb ##

import os
import gc
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add path with my_classes to sys.path
sys.path.insert(0, '/pf/b/b309170/workspace_icon-ml/cloud_cover_parameterization/')

from my_classes import load_data

from tensorflow.keras import backend as K
from tensorflow import nn 
from tensorflow.keras.models import load_model

ORDER_OF_VARS_NARVAL = ['qv', 'qc', 'qi', 'temp', 'pres', 'u', 'v', 'zg', 'coriolis', 'fr_land', 'clc', 'cl_area']
(TIME_STEPS, VERT_LAYERS, HORIZ_FIELDS) = (1721, 31, 4450) # For Narval data

path = '/pf/b/b309170/workspace_icon-ml/cloud_cover_parameterization'

## Functions ##

# To load the data
def get_data_path(model_type):
    return '/pf/b/b309170/my_work/icon-ml_data/cloud_cover_parameterization/%s/based_on_var_interpolated_data'%model_type

# For NARVAL, region-based model
def add_above_and_below(var_array, key):
    '''
        var_array: 3D tensor
    '''
    if key == 'pres':
        factor = 3/4
    else:
        factor = 1
    above = (np.insert(var_array, obj=0, values=1000*np.ones((TIME_STEPS, HORIZ_FIELDS)), axis=1))[:, :-1, :]
    # Replace by the entry from the same cell if the one above is nan.
    # It is a bit suboptimal that the grid cells above can be nan in NARVAL. At least decrease pressure by 3/4.
    nan_indices = np.where(np.isnan(above))
    above[nan_indices] = factor*above[nan_indices[0], nan_indices[1]+1, nan_indices[2]]
    
    # Below is the same value as the grid cell for surface-closest layer
    below = (np.append(var_array, values=var_array[:, -1:, :], axis=1))[:, 1:, :]
    return above, below

# To make predictions
def predict(model, input_data, mean, std, batch_size=2**20):
    # Put mean and std inside the function so that we don't have to load the entire input_data at once
    for i in range(1 + input_data.shape[0]//batch_size):
        if i == 0:
            a = model.predict_on_batch((input_data[i*batch_size:(i+1)*batch_size]-mean)/std)
        else:
            a = np.concatenate((a, model.predict_on_batch((input_data[i*batch_size:(i+1)*batch_size]-mean)/std)),
                               axis=0)
        K.clear_session()
        gc.collect()
        
    pred_adj = np.minimum(np.maximum(a, 0), 100) 
    return pred_adj

# To compute R2 and mean profiles
def compute_R2_and_means(pred_output, output_data, vertical_layers):
    '''
        Returns data_means, pred_means, r2
    '''
    data_means = []
    pred_means = []
    r2 = []
    
    if output_data.shape[-1] == 27:
        # For the column-based model
        # Means
        pred_means = np.mean(pred_output, axis=0, dtype=np.float64)
        data_means = np.mean(output_data, axis=0, dtype=np.float64)
        # R2
        mse = np.mean((pred_output - output_data)**2, axis=0, dtype=np.float64)
        var = np.var(output_data, axis=0)
        r2 = 1-mse/var
    else:
        if vertical_layers is None:
            # For the NARVAL cell-based and region-based models
            vertical_layers = np.arange(5, 32)
            vertical_layers = np.repeat(np.expand_dims(vertical_layers, 0), TIME_STEPS, axis=0)
            vertical_layers = np.repeat(np.expand_dims(vertical_layers, 2), HORIZ_FIELDS, axis=2)
            vertical_layers = np.reshape(vertical_layers, -1)
        for i in range(5, 32):
            indices = np.where(vertical_layers == i)
            # Means
            pred_means.append(np.mean(pred_output[indices], dtype=np.float64))
            data_means.append(np.mean(output_data[indices], dtype=np.float64))
            # R2
            mse = np.mean((pred_output[indices] - output_data[indices])**2, dtype=np.float64)
            var = np.var(output_data[indices])
            try:
                r2.append(1-mse/var)
            except ZeroDivisionError:
                print('Caught a division by 0 error')
                r2.append(-10**5)
    return data_means, pred_means, r2

# Get R2 and means for a given model
# We also have to provide the mean and std corresponding to the model
# Basically a wrapper for compute_R2_and_means
def get_R2_and_means(model_type, model, model_mean, model_std, data_source, 
                     output_type='cloud_cover', narval_data=None):
    '''
        data_source: 'narval' or 'qubicc'
        output_type: 'cloud_cover' or 'cloud_area'
        model_type: 'grid_cell_based_QUBICC_R02B05', 'region_based_one_nn_R02B05', 
                    'grid_column_based_QUBICC_R02B05', 'region_based_one_nn_with_rh_R02B05' (QUBICC models)
                    'grid_cell_based_v3' 'grid_column_based' (NARVAL models)
        model_training_source: 'narval' or 'qubicc'
        
        Returns: data_means, pred_means, r2_profile
    '''
    
    # Is this a NARVAL or a QUBICC model?
    if model_type in ['grid_cell_based_QUBICC_R02B05', 'region_based_one_nn_R02B05', 'region_based_one_nn_with_rh_R02B05', 'grid_column_based_QUBICC_R02B05']:
        model_training_source='qubicc'
    else:
        model_training_source='narval'
    
    # To load the model
    custom_objects = {}
    custom_objects['leaky_relu'] = nn.leaky_relu

    # Load model
    if model_training_source=='qubicc':
        clc_model_path = '%s/saved_models/%s_R2B5_QUBICC/%s'%(model_type, output_type, model)
        clc_model = load_model(os.path.join(path, clc_model_path), custom_objects)
    else:
        clc_model_path = '%s/saved_models/%s'%(model_type, model)
        clc_model = load_model(os.path.join(path, clc_model_path))

    # Load QUBICC data
    if data_source == 'qubicc':
        # Yields input_data, output_data and vertical_layers
        if model_type=='region_based_one_nn_with_rh_R02B05':
            # We use the region_based_one_nn_R02B05 data path, but need to adjust some input features
            data_path = get_data_path('region_based_one_nn_R02B05')
        else:
            data_path = get_data_path(model_type)
        input_data = np.load(os.path.join(data_path, 'cloud_cover_input_qubicc.npy'), mmap_mode='r')
        output_data = np.load(os.path.join(data_path, '%s_output_qubicc.npy'%output_type), mmap_mode='r')
        # We need to transpose the column-based data. It should have (no_samples, no_features).
        if input_data.shape[0] < input_data.shape[1]:
            input_data = np.transpose(input_data)
            output_data = np.transpose(output_data)
        # There is no vertical_layers file in the case of the column-based model
        if model_type=='grid_cell_based_QUBICC_R02B05' or model_type=='region_based_one_nn_R02B05':
            vertical_layers = np.load(os.path.join(data_path, 'samples_vertical_layers_qubicc.npy'), 
                                      mmap_mode='r')
        elif model_type=='grid_column_based_QUBICC_R02B05':
            vertical_layers = None 
            # We need to remove some input features for the column-based model
            remove_fields = [27, 28, 29, 30, 31, 32, 135, 136, 137]
            input_data = np.delete(input_data, remove_fields, axis=1)
        elif model_type=='region_based_one_nn_with_rh_R02B05':
            vertical_layers = np.load(os.path.join(data_path, 'samples_vertical_layers_qubicc.npy'), 
                                      mmap_mode='r')
            # Taken from preprocessing_narval
            input_variables = np.array(['qv', 'qc', 'qi', 'temp', 'pres', 'u', 'v', 'zg', 'coriolis', 'qv_below', 'qv_above',
                                       'qc_below', 'qc_above', 'qi_below', 'qi_above', 'temp_below', 'temp_above',
                                       'pres_below', 'pres_above', 'u_below', 'u_above', 'v_below', 'v_above', 
                                       'zg_below', 'zg_above','temp_sfc'])

            # We only need a subset of these input variables here: qi, qc, T, RH(p,qv,T)
            indices = [0,1,2,3,4,9,10,11,12,13,14,15,16,17,18]

            input_variables = input_variables[indices]
            input_data = input_data[:, indices]

            # Specific humidity to Relative humidity
            T0 = 273.15
            pres_ind = np.where(input_variables == 'pres')[0][0]
            temp_ind = np.where(input_variables == 'temp')[0][0]
            qv_ind = np.where(input_variables == 'qv')[0][0]

            pres_below_ind = np.where(input_variables == 'pres_below')[0][0]
            pres_above_ind = np.where(input_variables == 'pres_above')[0][0]
            temp_below_ind = np.where(input_variables == 'temp_below')[0][0]
            temp_above_ind = np.where(input_variables == 'temp_above')[0][0]
            qv_below_ind = np.where(input_variables == 'qv_below')[0][0]
            qv_above_ind = np.where(input_variables == 'qv_above')[0][0]

            r = 0.00263*input_data[:, pres_ind]*input_data[:, qv_ind]*np.exp((17.67*(input_data[:, temp_ind]-T0))/(input_data[:, temp_ind]-29.65))**(-1)
            r_below = 0.00263*input_data[:, pres_below_ind]*input_data[:, qv_below_ind]*np.exp((17.67*(input_data[:, temp_below_ind]-T0))/(input_data[:, temp_below_ind]-29.65))**(-1)
            r_above = 0.00263*input_data[:, pres_above_ind]*input_data[:, qv_above_ind]*np.exp((17.67*(input_data[:, temp_above_ind]-T0))/(input_data[:, temp_above_ind]-29.65))**(-1)

            # Now we can remove qv and pres as well
            input_variables = np.array(['qc', 'qi', 'temp', 'r', 'qc_below', 'qc_above', 'qi_below', 'qi_above', 'temp_below', 'temp_above', 'r_below', 'r_above'])
            in_and_out_variables = np.array(['qc', 'qi', 'temp', 'r', 'qc_below', 'qc_above', 'qi_below', 'qi_above', 'temp_below', 'temp_above', 'r_below', 'r_above', 'clc'])

            # Remove qv and pres from input_data as well, add relative humidity
            input_data = np.concatenate((input_data[:, 1:4], np.expand_dims(r, 1), input_data[:, 7:13], np.expand_dims(r_below, 1), np.expand_dims(r_above, 1)), axis=1)
        
    # Prepare NARVAL data
    if data_source == 'narval':
        vertical_layers = None
        # Yields input_data and output_data
        if output_type == 'cloud_cover':
            narval_data.pop('cl_area')
        elif output_type == 'cloud_area':
            narval_data.pop('clc')

        narval_data['zg'] = np.repeat(np.expand_dims(narval_data['zg'], 0), TIME_STEPS, axis=0)
        narval_data['coriolis'] = np.repeat(np.expand_dims(narval_data['coriolis'], 0), TIME_STEPS, axis=0)
        narval_data['fr_land'] = np.repeat(np.expand_dims(narval_data['fr_land'], 0), TIME_STEPS, axis=0)
#         narval_data['fr_lake'] = np.repeat(np.expand_dims(narval_data['fr_lake'], 0), TIME_STEPS, axis=0)
        
        if model_type == 'grid_cell_based_QUBICC_R02B05' or model_type == 'grid_cell_based_v3':
            narval_data.pop('fr_lake')
            narval_data.pop('rho')
            narval_data['coriolis'] = np.repeat(np.expand_dims(narval_data['coriolis'], 1), VERT_LAYERS, axis=1)
            narval_data['fr_land'] = np.repeat(np.expand_dims(narval_data['fr_land'], 1), VERT_LAYERS, axis=1)
            if model_type == 'grid_cell_based_v3':
                narval_data.pop('qc')
                narval_data.pop('u')
                narval_data.pop('v')
                narval_data.pop('coriolis')
            narval_data_reshaped = {}
            for key in narval_data.keys():
                narval_data_reshaped[key] = np.reshape(narval_data[key], -1) 
        elif model_type == 'region_based_one_nn_R02B05':
            narval_data['coriolis'] = np.repeat(np.expand_dims(narval_data['coriolis'], 1), VERT_LAYERS, axis=1)
            narval_data.pop('rho')
            narval_data.pop('fr_lake')
            narval_data.pop('fr_land')
            # Add temp_sfc
            temp_sfc = np.repeat(np.expand_dims(narval_data['temp'][:, -1, :], axis=1), VERT_LAYERS, axis=1)
            # Add above and below, and temp_sfc
            above = {}
            below = {}
            for key in ORDER_OF_VARS_NARVAL[:-4]:
                above[key], below[key] = add_above_and_below(narval_data[key], key)
            # Reshape
            narval_data_reshaped = {}
            for key in narval_data.keys():
                narval_data_reshaped[key] = np.reshape(narval_data[key], -1)
            for key in ORDER_OF_VARS_NARVAL[:-4]:
                narval_data_reshaped['%s_below'%key] = np.reshape(below[key], -1)
                narval_data_reshaped['%s_above'%key] = np.reshape(above[key], -1)
            narval_data_reshaped['temp_sfc'] = np.reshape(temp_sfc, -1)
            # The output variable has to be put to the end of the dictionary
            if output_type == 'cloud_cover':
                clc = narval_data_reshaped.pop('clc')
                narval_data_reshaped['clc'] = clc
            elif output_type == 'cloud_area':
                cl_area = narval_data_reshaped.pop('cl_area')
                narval_data_reshaped['cl_area'] = cl_area
        elif model_type == 'region_based_one_nn_with_rh_R02B05':
            # Add RH
            T0 = 273.15
            r = 0.00263*narval_data['pres']*narval_data['qv']*np.exp((17.67*(narval_data['temp']-T0))/(narval_data['temp']-29.65))**(-1)
            narval_data['rh'] = r
            # Add above and below, and temp_sfc
            above = {}
            below = {}
            # We only need zg for later book-keeping
            for key in ['qc', 'qi', 'temp', 'rh', 'zg']:
                above[key], below[key] = add_above_and_below(narval_data[key], key)
            # Reshape
            narval_data_reshaped = {}
            for key in ['qc', 'qi', 'temp', 'rh', 'zg', 'clc']: # Not optimal to fix clc here.
                if output_type == 'cloud_area':
                    raise Exception("Cloud area. Be careful here.") 
                narval_data_reshaped[key] = np.reshape(narval_data[key], -1)
            for key in ['qc', 'qi', 'temp', 'rh', 'zg'] :
                narval_data_reshaped['%s_below'%key] = np.reshape(below[key], -1)
                narval_data_reshaped['%s_above'%key] = np.reshape(above[key], -1)
            # The output variable has to be put to the end of the dictionary
            if output_type == 'cloud_cover':
                clc = narval_data_reshaped.pop('clc')
                narval_data_reshaped['clc'] = clc
            elif output_type == 'cloud_area':
                cl_area = narval_data_reshaped.pop('cl_area')
                narval_data_reshaped['cl_area'] = cl_area
        
        elif model_type == 'grid_column_based_QUBICC_R02B05' or model_type == 'grid_column_based':
            if model_type == 'grid_column_based_QUBICC_R02B05':
                narval_data.pop('fr_lake')
                narval_data.pop('rho')
            elif model_type == 'grid_column_based':
                narval_data.pop('fr_land')
            narval_data.pop('coriolis')
            narval_data.pop('u')
            narval_data.pop('v')
            # Reshape
            narval_data_reshaped = {}
            for key in narval_data.keys():
                if narval_data[key].shape[1] == VERT_LAYERS:  
                    # Removing data above 21kms
                    for i in range(4, VERT_LAYERS):
                        new_key = '{}{}{:d}'.format(key,'_',(i+17)) # Should start at 21
                        narval_data_reshaped[new_key] = np.reshape(narval_data[key][:,i,:], -1)
                else:
                    narval_data_reshaped[key] = np.reshape(narval_data[key], -1)
            # I think the order of the variables is fine here as well         
            
        df = pd.DataFrame.from_dict(narval_data_reshaped)
        if model_type in ['grid_cell_based_QUBICC_R02B05', 'region_based_one_nn_R02B05', 'region_based_one_nn_with_rh_R02B05', 'grid_cell_based_v3']:
            df = df.loc[df['zg'] < 21000] 
            if model_type == 'region_based_one_nn_with_rh_R02B05':
                df.drop(columns=['zg', 'zg_below', 'zg_above'], inplace=True)
            print(df.columns)
        
            # Drop the target (clc or cl_area) from df and put it in output_data
            output_data = np.float32(pd.DataFrame(df.iloc[:, -1]))
            df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
        elif model_type in ['grid_column_based_QUBICC_R02B05', 'grid_column_based']:
            # Drop the target (clc or cl_area) from df and put it in output_data
            output_data = np.float32(pd.DataFrame(df.iloc[:, -27:]))
            for i in range(27):
                df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
            # Remove features that were constant
            if model_type == 'grid_cell_based_QUBICC_R02B05': 
                remove_fields = [27, 28, 29, 30, 31, 32, 135, 136, 137]
            elif model_type == 'grid_column_based':
                remove_fields = [27, 162, 163, 164]
            remove_fields = -np.sort(-np.array(remove_fields)) #Careful with the order of removal
            for i in remove_fields:
                df.drop(df.columns[i], axis=1, inplace=True)
        
        input_data = np.float32(df)

    # Evaluate model on QUBICC data -> R2, mean per vertical layer
    # We might need to reduce a redundant dimension with np.squeeze to achieve pred_output.shape = output_data.shape
    pred_output = np.squeeze(predict(clc_model, input_data, model_mean, model_std))
    
    return compute_R2_and_means(pred_output, np.squeeze(output_data), vertical_layers)

# Write to file
def write_to_file(model_type_short, data_source_capitalized, data_means, pred_means, r2_profile, output_type='cloud_cover', model_training_source='qubicc'):
    '''
        model_type_short: 'Cell-based', 'Region-based', 'Column-based'
        data_source_capitalized: 'QUBICC', 'NARVAL'
        model_training_source: 'qubicc', 'narval'
    '''
    file_name='%s_models_r2_and_mean_values.txt'%model_training_source
    with open(os.path.join(path, '/', file_name), 'a') as file:
        file.write('%s model, %s, %s data: \n'%(model_type_short, output_type, data_source_capitalized))
        file.write('Data averages: \n')
        file.write(str(data_means) + '\n')
        file.write('Prediction averages: \n')
        file.write(str(pred_means) + '\n')
        file.write('R2 profile: \n')
        file.write(str(r2_profile) + '\n\n')
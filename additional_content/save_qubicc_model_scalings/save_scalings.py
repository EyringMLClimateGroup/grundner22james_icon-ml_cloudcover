import os
import numpy as np

from sklearn.preprocessing import StandardScaler

'''
    We collect all means and variances in a text file:
    We first load the preprocessed QUBICC R2B5 data, that has not yet been normalized.
    For every model, for every split (of the three-fold cross-validation split) and for every input feature 
    we save the input feature's means and variances in a file called qubicc_scalings.txt.
'''

def set_training_validation_folds(samples_total, samples_narval):
    training_folds = []
    validation_folds = []
    two_week_incr = (samples_total-samples_narval)//6

    for i in range(3):
        # Note that this is a temporal split since time was the first dimension in the original tensor
        first_incr = np.arange(samples_narval+two_week_incr*i, samples_narval+two_week_incr*(i+1))
        second_incr = np.arange(samples_narval+two_week_incr*(i+3), samples_narval+two_week_incr*(i+4))
        
        print(second_incr)

        validation_folds.append(np.append(first_incr, second_incr))
        training_folds.append(np.arange(samples_narval, samples_total))
        training_folds[i] = np.setdiff1d(training_folds[i], validation_folds[i])
        
    return training_folds, validation_folds

path = '/pf/b/b309170/my_work/icon-ml_data/cloud_cover_parameterization'

cell_path = 'grid_cell_based_QUBICC_R02B05/based_on_var_interpolated_data/cloud_cover_input_qubicc.npy'
column_path = 'grid_column_based_QUBICC_R02B05/based_on_var_interpolated_data/cloud_cover_input_qubicc.npy'
region_path = 'region_based_one_nn_R02B05/based_on_var_interpolated_data/cloud_cover_input_qubicc.npy'

# Cell
cell_data = np.load(os.path.join(path, cell_path))
        
if cell_data.shape[0] < cell_data.shape[1]:
    cell_data = np.transpose(cell_data)

(samples_total, no_of_features) = cell_data.shape
                    
training_folds, validation_folds = set_training_validation_folds(samples_total, 0)
                    
for i in range(3):                
    scaler = StandardScaler(copy=False)
    scaler.fit(cell_data[training_folds[i]])
    with open('qubicc_scalings.txt', 'a') as file:
        file.write('Cell %d: \n'%i)
        file.write(str(scaler.mean_)+'\n')
        file.write(str(scaler.var_)+'\n')
        
# Region
region_data = np.load(os.path.join(path, region_path))
        
if region_data.shape[0] < region_data.shape[1]:
    region_data = np.transpose(region_data)

(samples_total, no_of_features) = region_data.shape
                    
training_folds, validation_folds = set_training_validation_folds(samples_total, 0)
                    
for i in range(3):                
    scaler = StandardScaler(copy=False)
    scaler.fit(region_data[training_folds[i]])
    with open('qubicc_scalings.txt', 'a') as file:
        file.write('Region %d: \n'%i)
        file.write(str(scaler.mean_)+'\n')
        file.write(str(scaler.var_)+'\n')
        
# Column
column_data = np.load(os.path.join(path, column_path))
        
if column_data.shape[0] < column_data.shape[1]:
    column_data = np.transpose(column_data)

(samples_total, no_of_features) = column_data.shape
                    
training_folds, validation_folds = set_training_validation_folds(samples_total, 0)
                    
for i in range(3):                
    scaler = StandardScaler(copy=False)
    scaler.fit(column_data[training_folds[i]])
    with open('qubicc_scalings.txt', 'a') as file:
        file.write('Column %d: \n'%i)
        file.write(str(scaler.mean_)+'\n')
        file.write(str(scaler.var_)+'\n')


#!/bin/bash
#=============================================================================
#SBATCH --account=bd1179

#SBATCH --partition=prepost    # Specify partition name
#SBATCH --ntasks=12            # Specify max. number of tasks to be invoked
#SBATCH --cpus-per-task=2      # Specify number of CPUs per task
#SBATCH --time=10:00:00        # Set a limit on the total run time

#SBATCH --job-name=hcg_cl_area_frac.sh

# Bind your OpenMP threads
export OMP_NUM_THREADS=12
export KMP_AFFINITY=verbose,granularity=thread,compact,1
export KMP_STACKSIZE=64m
#=============================================================================

# Script to horizontally coarse-grain cloud area fraction
# Source grid: R02B09 (20971520 grid cells)
# Target grids: R02B04, R02B05

exp=hc2

source_path="/pf/b/b309170/bd1179_work/qubicc/vcg_data/cl_area_frac/${exp}"
source_grid="/pf/b/b309170/my_work/QUBICC/grids/icon_grid_0015_R02B09_G.nc"
target_grid_R02B04="/pf/b/b309170/my_work/QUBICC/grids/icon_grid_0013_R02B04_G.nc"
target_grid_R02B05="/pf/b/b309170/my_work/QUBICC/grids/icon_grid_0019_R02B05_G.nc"

files="`ls $source_path`"

# Convert to R02B04
target_path_R02B04="/pf/b/b309170/my_work/QUBICC/data_var_vertinterp/cl_area_fraction"

# Convert to R02B05
target_path_R02B05="/pf/b/b309170/my_work/QUBICC/data_var_vertinterp_R02B05/cl_area_fraction"

for file in $files; do
    # Read filename
    file_name=`echo $file | cut -d "." -f1`
    
    # Do not overwrite        
    if ! [[ -f ${target_path_R02B05}/${file_name}_R02B05.nc ]] # Do not overwrite
        then
            # Add resolution to the end of the filename
            cdo -f nc -P 24 remapcon,${target_grid_R02B04} -setgrid,${source_grid} ${source_path}/${file} ${target_path_R02B04}/${file_name}_R02B04.nc
            cdo -f nc -P 24 remapcon,${target_grid_R02B05} -setgrid,${source_grid} ${source_path}/${file} ${target_path_R02B05}/${file_name}_R02B05.nc
    fi
done

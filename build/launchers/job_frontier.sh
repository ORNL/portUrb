#!/bin/bash
#SBATCH -A stf006
#SBATCH -J portUrb
#SBATCH -o %x-%j.out
#SBATCH -t 3:00:00
#SBATCH -N 16
#SBATCH --partition batch

# #SBATCH --partition extended

num_tasks=`echo "$SLURM_JOB_NUM_NODES*8" | bc`
cd /lustre/orion/stf006/scratch/imn/portUrb/build
source machines/frontier/frontier_gpu_O2.env
# srun -n $num_tasks -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl ./inputs/input_abl_stable.yaml
# srun -n $num_tasks -c 1 --gpus-per-task=1 --gpu-bind=closest ./supercell ./inputs/input_supercell.yaml
srun -n $num_tasks -c 1 --gpus-per-task=1 --gpu-bind=closest ./city_stretched
# srun -n $num_tasks -c 1 --gpus-per-task=1 --gpu-bind=closest ./wind_farm2 ./inputs/input_windfarm2.yaml


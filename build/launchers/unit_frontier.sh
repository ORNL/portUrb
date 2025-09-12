#!/bin/bash
#SBATCH -A stf006
#SBATCH -J portUrb
#SBATCH -o %x-%j.out
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH --partition batch

# #SBATCH --partition extended

num_tasks=`echo "$SLURM_JOB_NUM_NODES*8" | bc`
cd /lustre/orion/stf006/scratch/imn/portUrb/build
source machines/frontier/frontier_gpu.env
srun -n $num_tasks -c 1 --gpus-per-task=1 --gpu-bind=closest ./supercell                >& supercell.out
srun -n $num_tasks -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_convective           >& abl_convective.out
srun -n $num_tasks -c 1 --gpus-per-task=1 --gpu-bind=closest ./city                     >& city.out
srun -n $num_tasks -c 1 --gpus-per-task=1 --gpu-bind=closest ./turbine_neutral_ensemble >& turbine_neutral_ensemble.out
srun -n $num_tasks -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_stable               >& abl_stable.out


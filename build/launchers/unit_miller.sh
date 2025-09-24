#!/bin/bash
#SBATCH --cluster fawbush
#SBATCH --partition ampere
#SBATCH --cluster-constraint=green
#SBATCH --exclusive
#SBATCH -A nwp501
#SBATCH -J portUrb
#SBATCH -o %x-%j.out
#SBATCH -t 24:00:00
#SBATCH -N 4

export GATOR_INITIAL_MB=39000
cd /lustre/storm/nwp501/scratch/imn/portUrb/build
source machines/miller/miller_gpu.env
num_tasks=`echo "$SLURM_NNODES*4" | bc`
date
srun -N $SLURM_NNODES -n $num_tasks -c 32 --gpus-per-task=1 --gpu-bind=closest ./abl_convective             >& abl_convective.out
srun -N $SLURM_NNODES -n $num_tasks -c 32 --gpus-per-task=1 --gpu-bind=closest ./abl_neutral_fixed          >& abl_neutral_fixed.out
srun -N $SLURM_NNODES -n $num_tasks -c 32 --gpus-per-task=1 --gpu-bind=closest ./abl_neutral_variable       >& abl_neutral_variable.out
srun -N $SLURM_NNODES -n $num_tasks -c 32 --gpus-per-task=1 --gpu-bind=closest ./abl_stable                 >& abl_stable.out
srun -N $SLURM_NNODES -n $num_tasks -c 32 --gpus-per-task=1 --gpu-bind=closest ./city                       >& city.out
srun -N $SLURM_NNODES -n $num_tasks -c 32 --gpus-per-task=1 --gpu-bind=closest ./supercell_kessler_fixed    >& supercell_kessler_fixed.out
srun -N $SLURM_NNODES -n $num_tasks -c 32 --gpus-per-task=1 --gpu-bind=closest ./supercell_kessler_variable >& supercell_kessler_variable.out
srun -N $SLURM_NNODES -n $num_tasks -c 32 --gpus-per-task=1 --gpu-bind=closest ./supercell_morr2mom         >& supercell_morr2mom.out
srun -N $SLURM_NNODES -n $num_tasks -c 32 --gpus-per-task=1 --gpu-bind=closest ./turbine_neutral_ensemble   >& turbine_neutral_ensemble.out
srun -N $SLURM_NNODES -n $num_tasks -c 32 --gpus-per-task=1 --gpu-bind=closest ./turbine_simple             >& turbine_simple.out
date


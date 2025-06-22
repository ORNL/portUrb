#!/bin/bash
#SBATCH --cluster fawbush
#SBATCH --partition ampere
#SBATCH --cluster-constraint=green
#SBATCH --exclusive
#SBATCH -A nwp501
#SBATCH -J portUrb
#SBATCH -o %x-%j.out
#SBATCH -t 24:00:00
#SBATCH -N 20

export GATOR_INITIAL_MB=39000

cd /lustre/storm/nwp501/scratch/imn/portUrb/build
source machines/miller/miller_gpu.env

num_tasks=`echo "$SLURM_NNODES*4" | bc`
srun -N $SLURM_NNODES -n $num_tasks -c 32 --gpus-per-task=1 --gpu-bind=closest ./turbine_neutral_ensemble

# turbine_convective_ensemble  turbine_neutral_ensemble     turbine_stable_ensemble

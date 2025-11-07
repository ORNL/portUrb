#!/bin/bash
#SBATCH --cluster-constraint=blue
#SBATCH --exclusive
#SBATCH -A nwp501-arch
#SBATCH -J portUrb
#SBATCH -o %x-%j.out
#SBATCH -t 6:00:00
#SBATCH -N 12

# #SBATCH --exclude=arch13

cd /lustre/storm/nwp501/scratch/imn/portUrb/build
source machines/miller/miller_arch_gpu.env

num_tasks=`echo "$SLURM_NNODES*4" | bc`
srun -N $SLURM_NNODES -n $num_tasks -c 72 --gpus-per-task=1 --gpu-bind=closest ./turbine_fitch


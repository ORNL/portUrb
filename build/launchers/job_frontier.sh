#!/bin/bash
#SBATCH -A stf006
#SBATCH -J portUrb
#SBATCH -o %x-%j.out
#SBATCH -t 7:00:00
#SBATCH -N 4
#SBATCH --partition service
# #SBATCH --partition extended


num_tasks=`echo "$SLURM_JOB_NUM_NODES*8" | bc`
cd /lustre/orion/stf006/scratch/imn/portUrb/build
source machines/frontier/frontier_gpu_O3.env

srun -N $SLURM_JOB_NUM_NODES -n $num_tasks -c 1 --gpus-per-task=1 --gpu-bind=closest ./channel 2>&1 | tee channel.out

# cat <<EOF > rsst1.yaml
# cs: 350
# buoy_theta: false
# rsst: false
# EOF
# srun -N 4 -n 32 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_stable ./rsst1.yaml 2>&1 | tee abl_stable1.out &
# 
# cat <<EOF > rsst2.yaml
# cs: 350
# buoy_theta: true
# rsst: false
# EOF
# srun -N 4 -n 32 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_stable ./rsst2.yaml 2>&1 | tee abl_stable2.out &
# 
# cat <<EOF > rsst3.yaml
# cs: 350
# buoy_theta: true
# rsst: true
# EOF
# srun -N 4 -n 32 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_stable ./rsst3.yaml 2>&1 | tee abl_stable3.out &
# 
# cat <<EOF > rsst4.yaml
# cs: 100
# buoy_theta: true
# rsst: true
# EOF
# srun -N 4 -n 32 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_stable ./rsst4.yaml 2>&1 | tee abl_stable4.out &


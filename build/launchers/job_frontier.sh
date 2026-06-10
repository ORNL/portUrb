#!/bin/bash
#SBATCH -A stf006
#SBATCH -J portUrb
#SBATCH -o %x-%j.out
#SBATCH -t 2:00:00
#SBATCH -N 4
#SBATCH --partition service
#SBATCH -q develop
# #SBATCH --partition extended


num_tasks=`echo "$SLURM_JOB_NUM_NODES*8" | bc`
cd /lustre/orion/stf006/scratch/imn/portUrb/build
source machines/frontier/frontier_gpu_O3.env

srun -n $num_tasks -c 1 --gpus-per-task=1 --gpu-bind=closest ./shallow_convection

# cat <<EOF > rsst1.yaml
# cs: 350
# buoy_theta: false
# rsst: false
# EOF
# srun -N 1 -n 8  -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_stable ./rsst1.yaml 2>&1 | tee abl_stable1.out &
# 
# cat <<EOF > rsst2.yaml
# cs: 350
# buoy_theta: true
# rsst: false
# EOF
# srun -N 1 -n 8  -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_stable ./rsst2.yaml 2>&1 | tee abl_stable2.out &
# 
# cat <<EOF > rsst3.yaml
# cs: 350
# buoy_theta: true
# rsst: true
# EOF
# srun -N 1 -n 8  -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_stable ./rsst3.yaml 2>&1 | tee abl_stable3.out &
# 
# cat <<EOF > rsst4.yaml
# cs: 100
# buoy_theta: true
# rsst: true
# EOF
# srun -N 1 -n 8  -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_stable ./rsst4.yaml 2>&1 | tee abl_stable4.out &
# 
# wait
# 
# cat <<EOF > rsst5.yaml
# cs: 50
# buoy_theta: true
# rsst: true
# EOF
# srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_stable ./rsst5.yaml 2>&1 | tee abl_stable5.out &
# 
# cat <<EOF > rsst6.yaml
# cs: 25
# buoy_theta: true
# rsst: true
# EOF
# srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_stable ./rsst6.yaml 2>&1 | tee abl_stable6.out &
# 
# cat <<EOF > rsst7.yaml
# cs: 15
# buoy_theta: true
# rsst: true
# EOF
# srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_stable ./rsst7.yaml 2>&1 | tee abl_stable7.out &
# 
# cat <<EOF > rsst8.yaml
# cs: 10
# buoy_theta: true
# rsst: true
# EOF
# srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_stable ./rsst8.yaml 2>&1 | tee abl_stable8.out &
# 
# wait


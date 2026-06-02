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

srun -n $num_tasks -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_neutral

# cat <<EOF > rsst1.yaml
# cs: 350
# buoy_theta: false
# rsst: false
# EOF
# srun -N 1 -n 8  -c 1 --gpus-per-task=1 --gpu-bind=closest ./turbine_simplest ./rsst1.yaml 2>&1 | tee turbine_simplest1.out &
# 
# cat <<EOF > rsst2.yaml
# cs: 350
# buoy_theta: true
# rsst: false
# EOF
# srun -N 1 -n 8  -c 1 --gpus-per-task=1 --gpu-bind=closest ./turbine_simplest ./rsst2.yaml 2>&1 | tee turbine_simplest2.out &
# 
# cat <<EOF > rsst3.yaml
# cs: 350
# buoy_theta: true
# rsst: true
# EOF
# srun -N 1 -n 8  -c 1 --gpus-per-task=1 --gpu-bind=closest ./turbine_simplest ./rsst3.yaml 2>&1 | tee turbine_simplest3.out &
# 
# cat <<EOF > rsst4.yaml
# cs: 250
# buoy_theta: true
# rsst: true
# EOF
# srun -N 1 -n 8  -c 1 --gpus-per-task=1 --gpu-bind=closest ./turbine_simplest ./rsst4.yaml 2>&1 | tee turbine_simplest4.out &
# 
# wait
# 
# cat <<EOF > rsst5.yaml
# cs: 150
# buoy_theta: true
# rsst: true
# EOF
# srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./turbine_simplest ./rsst5.yaml 2>&1 | tee turbine_simplest5.out &
# 
# cat <<EOF > rsst6.yaml
# cs: 100
# buoy_theta: true
# rsst: true
# EOF
# srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./turbine_simplest ./rsst6.yaml 2>&1 | tee turbine_simplest6.out &
# 
# cat <<EOF > rsst7.yaml
# cs: 80
# buoy_theta: true
# rsst: true
# EOF
# srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./turbine_simplest ./rsst7.yaml 2>&1 | tee turbine_simplest7.out &
# 
# cat <<EOF > rsst8.yaml
# cs: 60
# buoy_theta: true
# rsst: true
# EOF
# srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./turbine_simplest ./rsst8.yaml 2>&1 | tee turbine_simplest8.out &
# 
# wait
# 
# cat <<EOF > rsst9.yaml
# cs: 40
# buoy_theta: true
# rsst: true
# EOF
# srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./turbine_simplest ./rsst9.yaml 2>&1 | tee turbine_simplest9.out &
# 
# cat <<EOF > rsst10.yaml
# cs: 20
# buoy_theta: true
# rsst: true
# EOF
# srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./turbine_simplest ./rsst10.yaml 2>&1 | tee turbine_simplest10.out &
# 
# wait
# 

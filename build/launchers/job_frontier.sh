#!/bin/bash
#SBATCH -A stf006
#SBATCH -J portUrb
#SBATCH -o %x-%j.out
#SBATCH -t 6:00:00
#SBATCH -N 4
#SBATCH --partition service

# #SBATCH --partition extended

num_tasks=`echo "$SLURM_JOB_NUM_NODES*8" | bc`
cd /lustre/orion/stf006/scratch/imn/portUrb/build
source machines/frontier/frontier_gpu_O3.env

cat <<EOF > rsst1.yaml
cs: 350
buoy_theta: false
rsst: false
EOF
srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_neutral ./rsst1.yaml 2>&1 | tee abl_neutral1.out &

cat <<EOF > rsst2.yaml
cs: 350
buoy_theta: true
rsst: false
EOF
srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_neutral ./rsst2.yaml 2>&1 | tee abl_neutral2.out &

cat <<EOF > rsst3.yaml
cs: 350
buoy_theta: true
rsst: true
EOF
srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_neutral ./rsst3.yaml 2>&1 | tee abl_neutral3.out &

cat <<EOF > rsst4.yaml
cs: 100
buoy_theta: true
rsst: true
EOF
srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_neutral ./rsst4.yaml 2>&1 | tee abl_neutral4.out &

wait

cat <<EOF > rsst5.yaml
cs: 50
buoy_theta: true
rsst: true
EOF
srun -N 1 -n 4 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_neutral ./rsst5.yaml 2>&1 | tee abl_neutral5.out &

cat <<EOF > rsst6.yaml
cs: 25
buoy_theta: true
rsst: true
EOF
srun -N 1 -n 4 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_neutral ./rsst6.yaml 2>&1 | tee abl_neutral6.out &

cat <<EOF > rsst7.yaml
cs: 20
buoy_theta: true
rsst: true
EOF
srun -N 1 -n 4 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_neutral ./rsst7.yaml 2>&1 | tee abl_neutral7.out &

cat <<EOF > rsst8.yaml
cs: 15
buoy_theta: true
rsst: true
EOF
srun -N 1 -n 4 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_neutral ./rsst8.yaml 2>&1 | tee abl_neutral8.out &

cat <<EOF > rsst9.yaml
cs: 12
buoy_theta: true
rsst: true
EOF
srun -N 1 -n 4 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_neutral ./rsst9.yaml 2>&1 | tee abl_neutral9.out &

cat <<EOF > rsst10.yaml
cs: 10
buoy_theta: true
rsst: true
EOF
srun -N 1 -n 4 -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl_neutral ./rsst10.yaml 2>&1 | tee abl_neutral10.out &

wait


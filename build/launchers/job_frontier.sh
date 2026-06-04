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

# cat <<EOF > rsst3.yaml
# cs: 350
# buoy_theta: true
# rsst: true
# EOF
# srun -n $num_tasks -c 1 --gpus-per-task=1 --gpu-bind=closest ./supercell ./inputs/input_supercell.yaml ./rsst3.yaml

cat <<EOF > rsst1.yaml
cs: 350
buoy_theta: false
rsst: false
EOF
srun -N 1 -n 8  -c 1 --gpus-per-task=1 --gpu-bind=closest ./supercell ./inputs/input_supercell.yaml ./rsst1.yaml 2>&1 | tee supercell1.out &

cat <<EOF > rsst2.yaml
cs: 350
buoy_theta: true
rsst: false
EOF
srun -N 1 -n 8  -c 1 --gpus-per-task=1 --gpu-bind=closest ./supercell ./inputs/input_supercell.yaml ./rsst2.yaml 2>&1 | tee supercell2.out &

cat <<EOF > rsst3.yaml
cs: 350
buoy_theta: true
rsst: true
EOF
srun -N 1 -n 8  -c 1 --gpus-per-task=1 --gpu-bind=closest ./supercell ./inputs/input_supercell.yaml ./rsst3.yaml 2>&1 | tee supercell3.out &

cat <<EOF > rsst4.yaml
cs: 250
buoy_theta: true
rsst: true
EOF
srun -N 1 -n 8  -c 1 --gpus-per-task=1 --gpu-bind=closest ./supercell ./inputs/input_supercell.yaml ./rsst4.yaml 2>&1 | tee supercell4.out &

wait

cat <<EOF > rsst5.yaml
cs: 150
buoy_theta: true
rsst: true
EOF
srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./supercell ./inputs/input_supercell.yaml ./rsst5.yaml 2>&1 | tee supercell5.out &

cat <<EOF > rsst6.yaml
cs: 100
buoy_theta: true
rsst: true
EOF
srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./supercell ./inputs/input_supercell.yaml ./rsst6.yaml 2>&1 | tee supercell6.out &

cat <<EOF > rsst7.yaml
cs: 80
buoy_theta: true
rsst: true
EOF
srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./supercell ./inputs/input_supercell.yaml ./rsst7.yaml 2>&1 | tee supercell7.out &

cat <<EOF > rsst8.yaml
cs: 60
buoy_theta: true
rsst: true
EOF
srun -N 1 -n 8 -c 1 --gpus-per-task=1 --gpu-bind=closest ./supercell ./inputs/input_supercell.yaml ./rsst8.yaml 2>&1 | tee supercell8.out &

wait


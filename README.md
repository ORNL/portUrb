# portUrb

Welcome to portUrb: your friendly neighborhood portable urban flow model. The goal is simple, readable, extensible, portable, and fast code to quickly prototype workflows for developing surrogate models for unresolved or poorly resolved processes in turbulent atmospheric fluid dynamics with obstacles. portUrb currently handles stratified, buoyancy-driven flows, shear-driven boundary layer turbulence, moist microphysics, and sub-grid-scale turbulence.

![city_2m_q_1_smaller](https://github.com/user-attachments/assets/b27cf5cb-d117-48ae-b424-ae9d2b2dde7d)
![22mw_blades_11 4mps_vortmag_0 6_smaller](https://github.com/user-attachments/assets/a383bff5-2b70-456c-9240-77dd55359b0b)
![supercell_smaller](https://github.com/user-attachments/assets/1bbcee17-2751-4e8b-b357-0453439c3697)

## Example workflow

```bash
git clone git@github.com:ORNL/portUrb.git
cd portUrb
git submodule update --init
cd build
source machines/frontier/frontier_gpu.env
./cmakescript.sh ../experiments/examples
make -j8 supercell
# Edit ./inputs/input_supercell.yaml
num_tasks=`echo "$SLURM_JOB_NUM_NODES*8" | bc`
srun -n $num_tasks -c 1 --gpus-per-task=1 --gpu-bind=closest ./supercell ./inputs/input_supercell.yaml
```


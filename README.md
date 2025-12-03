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

## Core Classes to Understand

When working with portUrb and designing your own experiments and modules, the classes below are the important classes to understand with basic APIs and examples. For more detailed information, there is a lot of documentation inside the `portUrb/model/core/*.h` files, where this is implemented. 

### YAKL Classes

The YAKL is A Kokkos Layer library (https://github.com/mrnorman/YAKL) is the workhorse of portable programming for portUrb, and as the name implies, it is based on the Kokkos portable C++ library (https://github.com/kokkos/kokkos). Have no familiarity with Kokkos or YAKL? Fret not. There is plenty of approachable documentation to get more used to it as well as examples in this codebase to learn from more directly. YAKL launches parallel work with a `parallel_for` function or reduction functions (sum, minval, maxval, minloc, etc.).

The most important thing to understand is inside a `parallel_for`, there is **no guarantee** of the order the indices will execute. So the work needs to be truly parallel. For cases with data races, there are atomics and other possibilities to learn from in portUrb as needed.

The next most important thing to understand are the YAKL `Array` objects, which in portUrb are typically declared as `real1d`, `float2d`, `int3d`, `bool4d`, `double5d`, etc, where the first part is the type, and the second part declares how many dimensions it has, where the right-most dimension varies the fastest. You can get more information about YAKL `Array` objects here: https://github.com/mrnorman/YAKL/wiki/Multi-Dimensional-Dynamically-Allocated-Arrays

In accelerated HPC environments, there are two basic memory spaces: host and device. Device memory should only be accessed inside `parallel_for` functions or device reductions, and host memory should only be accessed outside `parallel_for` functions. By default, all types are on the device, so please keep that in mind. portUrb defines host types with `realHost2d`, `intHost1d`, etc. with the "Host" intermediate in the type.

For more information about YAKL, in general, please checkout the documentation here: https://github.com/mrnorman/YAKL/wiki

### portUrb programming norms

In portUrb, the dimensions are nearly always ordered as z, y, x (in order of indexing), and the z-dimension varies the slowest in memory while the x-dimension varies the fastest.

The `real` type is, by default double precision (64-bit) floating point. While some modules operate mostly in single precision (32-bit) for extra efficiency, I strongly recommend you do not try to change the `real` type to single precision, or nasty things are likely to happen with cancellation errors that may lead to artifacts in your solution.

### `Coupler`: `portUrb/model/core/coupler.h`

The `Coupler` class is by far the most important class to understand in portUrb since it holds nearly all of the simulation data and options and it is passed to all modules to perform modifications to the Coupler's data.
* Constructors and destructors:
   * There is only one constructor (a default constructor), which initializes an empty `Coupler` object, so you'll always be declaring one as `core::Coupler coupler;`
   * There are move and move assignment constructors, but there are no copy and copy assignment constructors. This means you cannot accidentally copy an entire coupler object, which would consume a lot of runtime if that happened. If you wish to copy a coupler object, there is a `Coupler::clone_into(Coupler & other_coupler)` function detailed later that you can use.
   * The destructor cleans everything up for you, so you don't need to explicitly deallocate or remove the things you add to a coupler to avoid memory leaks. Once you `Coupler::finalize()` or simple let the object fall out of scope, it'll clean everything up for you. 
* `void Coupler::init(...);`
   * The function signiture for this is: `void init( ParallelComm par_comm , real1d const & zint , size_t ny_glob , size_t nx_glob , real ylen , real xlen );`. There are other optional parameters, but unless you're running hi-resolution and lo-resolution simulations on the same domain and trying to match the MPI decomposition exactly (which is very uncommon), you can safely ignore the optional parameters and focus on the ones above.
       * This function sets the coupler's model grid and performs the parallel decomposition within the MPI tasks encapsulated by the `par_comm` object. You'll call this before anything else in the coupler.
       * `par_comm`: This is a `core::ParallelComm` object described in more detail later, which basically provides the coupler with an MPI communicator and convenient MPI operators to perform on that communicator. For most simulations, the communicator is simply `MPI_COMM_WORLD`.
       * `zint`: This is a device 1-D real `Array` object that contains the vertical interface heights. YAKL has variable vertical grid spacing, so this will define the vertical grid that is constant throughout the entire domain. YAKL has some convenience functions in the `Coupler` class to generate these interfaces for you, and those will be described later.
       * `ny_glob` and `nx_glob`: These are the *total* number of grid cells in the y-direction and x-direction, respectively, for your simulation. x and y represent the horizontal dimensions of the model. 
       * `ylen` and `xlen`: These are the domain sizes in meters of the y-direction and x-direction of the domain, and uniform grid spacing is used in the horizontal dimensions. 
   * There are a lot of accessor functions in the `Coupler` class to get information about the domain and current coupler state below. If you see something you don't recognize, it'll probably be covered later.
```C++
    // Get the parallel communicator convenience class for this coupler
    ParallelComm              get_parallel_comm         () const;
    // Get the x-dimension length of the domain in meters
    real                      get_xlen                  () const;
    // Get the y-dimension length of the domain in meters
    real                      get_ylen                  () const;
    // Get the z-dimension length of the domain in meters
    real                      get_zlen                  () const;
    // Get the number of MPI ranks in the communicator
    int                       get_nranks                () const;
    // Get my MPI rank ID in the communicator
    int                       get_myrank                () const;
    // Get the total global number of cells in the x-direction
    size_t                    get_nx_glob               () const;
    // Get the total global number of cells in the y-direction
    size_t                    get_ny_glob               () const;
    // Get the number of MPI processes in the x-direction
    int                       get_nproc_x               () const;
    // Get the total number of cells in the z-direction (same for all MPI processes)
    int                       get_nz                    () const;
    // Get the number of MPI processes in the y-direction
    int                       get_nproc_y               () const;
    // Get my MPI process ID in the x-direction
    int                       get_px                    () const;
    // Get my MPI process ID in the y-direction
    int                       get_py                    () const;
    // Get my beginning global index in the x-direction (inclusive)
    size_t                    get_i_beg                 () const;
    // Get my beginning global index in the y-direction (inclusive)
    size_t                    get_j_beg                 () const;
    // Get my ending global index in the x-direction (inclusive)
    size_t                    get_i_end                 () const;
    // Get my ending global index in the y-direction (inclusive)
    size_t                    get_j_end                 () const;
    // Check if the simulation is 2D (i.e., only one cell in the y-direction)
    bool                      is_sim2d                  () const;
    // Check if I am the main MPI process (rank 0)
    bool                      is_mainproc               () const;
    // Get the neighbor rank ID matrix as a const reference
    SArray<int,2,3,3> const & get_neighbor_rankid_matrix() const;
    // Get the DataManager as a const reference for read-only access to allocated variables
    DataManager       const & get_data_manager_readonly () const;
    // Get the DataManager as a non-const reference for read-write access to allocated variables
    DataManager             & get_data_manager_readwrite()      ;
    // Get the x-direction grid spacing in meters
    real                      get_dx                    () const;
    // Get the y-direction grid spacing in meters
    real                      get_dy                    () const;
    // Get the z-direction grid spacing in meters as a 1D array
    real1d                    get_dz                    () const;
    // Get the z-direction interface heights as a 1D array
    real1d                    get_zint                  () const;
    // Get the z-direction midpoint heights as a 1D array
    real1d                    get_zmid                  () const;
    // Get the number of tracers registered with the coupler
    int                       get_num_tracers           () const;
    // Get the number of cells in the x-direction on this MPI process
    int                       get_nx                    ();
    // Get the number of cells in the y-direction on this MPI process
    int                       get_ny                    () ;
```
   * You can also specify "options" according to a type, a label, and a value that can be used for coordination among the different modules that act on a `Coupler` object.
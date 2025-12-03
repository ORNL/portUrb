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

Because most atmospheric parameterizations are independent in the horizontal dimensions and highly dependent in the vertical dimensions, the domain is only decomposed over MPI in the horizontal (x,y) directions with no MPI decomposition in the vertical (z) direction. 

### `Coupler`: `portUrb/model/core/coupler.h`

The `Coupler` class is by far the most important class to understand in portUrb since it holds nearly all of the simulation data and options and it is passed to all modules to perform modifications to the Coupler's data. Only the most important aspects are covered here. For more detailed information, please look in the file it is implemented in where more documentation is available. 
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
   * You can generate various vertical grids with `real1d Coupler::generate_levels_equal( int nz , real zlen );` (equal grid spacing), `real1d Coupler::generate_levels_exp( int nz , real zlen , real dz0 );` (exponentially increasing grid spacing starting with dz0 at the surface), and `real1d Coupler::generate_levels_const_low_high( real zlen , real dz0 , real z1 , real z2 , real dz2 )` (constant dz0 grid spacing below z1, transitioning between z1 and z2 to dz2 and constant at dz2 above z2). These create the `zint` variable to input to `Coupler::init(...)`.
   * You can set various "options" in the coupler with `void Coupler::set_option<T>( std::string label , T value );` that help you coordinate different modules as they alter the coupler state. You can obtain the options that have been set at any point with `T Coupler::get_option<T>(std::string label);` if you know the option exists or `T Coupler::get_option<T>( std::string label , T value_if_does_not_exist )`. The type `T` will be things like `int` `real` `bool` or `std::string`, and they can specify things like physical constants, latitude, boundary condition labels, or other options that control module behavior for different modules. Just make sure that the types `T` match for the same label when setting and getting options, or you'll get an error. You can check if an option label exists with `bool Coupler::option_exists(std::string label)` as well. You can also delete an option with `void Coupler::delete_option(std::string label)`. 
   * There is a `void Coupler::run_module(F const &func , std::string name);` function you can use to run modules, where portUrb will automatically time them, check for invalid values in coupler variables, and trace which variables were potentially modified for you depending on compilation options you use. You can run modules without this, but this will add those other conveniences if you want them. The function `F` can be a `std::function` or a C++ `lambda`, but it must be a void function that takes only one parameter: a `Coupler` object. `run_module` will pass the coupler object you call it from into the module function.
   * Tracers get a first-class seat in the `Coupler` class because there are things that need to automatically happen to them such as advection, diffusion, adding to the total density in the dynamical core (mass loading), and remaining non-negative (using a "hole-filler" approach). You can register a tracer with the coupler with `void Coupler::add_tracer( std::string tracer_name , std::string tracer_desc = "" , bool positive = true , bool adds_mass = false , bool diffuse = true )`. All tracers must be "massy" or mass-weighted variables to be advected properly in the dynamical core, and all mass-adding tracers must be densities in units of kg/m^3 (not mixing ratios). Tracers are automatically added to file output, restart overwriting, and are automatically advected. If you declare it as adding mass, it will add to the dynamical core's density. If you declare it as positive, it will be kept from obtaining negative values. If you declare it as diffused, the SGS turbulence closure will diffuse it appropriately. You can get the properties of a tracer with `void Coupler::get_tracer_info(std::string tracer_name , std::string &tracer_desc, bool &tracer_found , bool &positive , bool &adds_mass, bool &diffuse);`. You can get a list of the tracer labels with `std::vector<std::string> Coupler::get_tracer_names()`, and you can see if a tracer of a given label exists with `bool Coupler::tracer_exists( std::string tracer_name )`. Tracers should be registered with `add_tracer` before you call the `init()` function of the dynamical core so that it knows about them when it is initialized.
   * You can write an output file with `void Coupler::write_output_file( std::string prefix );` where `prefix` is the prefix of the output filename that is appended with a zero-padded 8-digit output counter index followed by `.nc`, where the counter is incremented each time this function is called. The output is in NetCDF4 foramat using the parallel-netcdf library. 
   * The `void Coupler::overwrite_with_restart()` will overwrite current coupler data and values with the restart file name specified by the coupler option "restart_file", which is a `std::string`.
   * The coupler also has halo creation and halo exchanging routines you can use for halo exchanges that use periodic BC's in the horizontal (you can overwrite these ghost cells at the domain boundaries with boundary condition data if you wish). Please search the `model/modules` and `experiments/examples` directories for usage examples of these.
   * All allocated variables in simulations will typically be managed using the `DataManager` class described below. The `Coupler` object has two routines to get the Coupler's `DataManager` object that manages nearly all the model arrays: one routine to get it with read-write access and another to get it with read-only access. **Importantly**, you can only accepta a DataManager object as a reference because their copy and copy assignment operators have been deleted to avoid accidentally copying large amounts of simulation data and to keep multiple DataManager objects from aliasing the same memory, which might be quite hard to manage for the user. Therefore, for read-only access, you will obtain the `DataManager` with `auto &dm = coupler.get_data_manager_readonly();` or `auto &dm = coupler.get_data_manager_readwrite();`. 
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

### `DataManager`: `portUrb/model/core/DataManager.h`

The `DataManager` class manages all allocated data in the model, which is most of the model data altogether, really. Once you get a `DataManager` object with `auto &dm = coupler.get_data_manager_readonly()` or `auto &dm = coupler.get_data_manager_readwrite()`, you can manipulate the coupler's data with the following, where only the most important routines are covered. For more detail, please see the source file where there is more documentation provided or the `model/modules` directory for examples. 
* Constructors and destructors:
   * There is only one constructor (a default constructor), which initializes an empty `DataManger` object. You almost never need to declare your own because the `Coupler` has its own `DataManager` object, so this is not something you'll typically use or declare yourself. 
   * There are move and move assignment constructors, but there are no copy and copy assignment constructors. This means you cannot accidentally copy an entire `DataManager` object, which would consume a lot of runtime if that happened. If you wish to copy a `DataManager` object, there is a `DataManager::clone_into(DataManager & other_dm)` function detailed later that you can use. Again, you only need to do this yourself if you have your own object, which is probably quite rare. The `Coupler` has its own object, and when you call `Coupler::clone_into`, its `DataManager` object automatically copies into the target `Coupler` object's `DataManager` object. 
   * The destructor cleans everything up for you, so you don't need to explicitly deallocate or remove the things you add to a coupler to avoid memory leaks. Once the Coupler object falls out of scope, it calls the `DataManager`'s destructor, which cleans everything up. If you have your own `DataManager` object aside from the coupler's, again, once it falls out of scope, it'll clean things up for you. 
* You register and allocate new arrays with: `void DataManager::register_and_allocate( std::string name , std::string desc , std::vector<int> dims );`
   * This is what you use to allocate a new array and register it with the `DataManager`. The name you pass is a unique label, and you cannot have more than one entry with the same label. The description is artibrary to how you want to describe the data. The dimensionality of the array is specified by how many dimensions you pass in the `dims` argument. The type of the array is specified by the template parameter `T`. To create an array called "my_array" of type `float` with four dimensions, for instance, you would typically write `coupler.get_data_manager_readwrite().register_and_allocate<float>("my_array","my description",{num_vars,nz,ny,nx});`
   * All of the `Coupler` object's `DataManager` data is inherently in device memory, so please only access individual elements of those arrays within `parallel_for` functions or reductions. 
* You can then get that array with the `Array<T,N> DataManager::get<T,N>( std::string name );` function. For example, to retrieve the aforementioned array in a read-only capacity, you would write `coupler.get_data_manager_readonly().get<float const,4>("my_array");`. To get the array in a read-write capacity, you would write `coupler.get_data_manager_readwrite().get<float,4>("my_array");`. What this actually does under the hood is create a YAKL `Array` object that wraps the internal pointer for the memory used by that variable. The type checking guards against memory errors, and it's important to try to be "`const` correct" so that the coupler knows what you're only reading and what you're actually modifying. There are `DataManager` routines that can tell you what variables have been potentiall modified by a given module (see the source file for details), and using `readonly` and `const` when you're not writing to the variable will make that information more precise. 
* You can get the data as a level-column indexing if you want with `Array<T,2> DataManager::get_lev_col<T>( std::string name );`, where the first dimension of the data with that name is assumed to be the vertical index and the rest of the dimensions are combined together to represent the horizontal indices. This is useful for physica parameterizations that often index data as level,column and do not care about separate x and y dimensions. 
* You can get the data as a single collapsed dimension with `Array<T,1> DataManager::get_collapsed<T>( std::string name );`, where all indices are collapsed into a single dimension. This is useful for situations where you don't care about vertical or horizontal definitions and the same operation will be used no matter which direction is being accessed.


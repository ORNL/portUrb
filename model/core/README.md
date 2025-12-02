# Contains portUrb's main functionality

* `coupler.h`: This contains the `Coupler` class, the most important class in portUrb, which coordinates all of the simulation data and options. All modules in portUrb act on a `Coupler` object's interal data. The coupler's functionality is detailed in the main repository `README.md` file, so please go there for more information.
* `DataManager.h`: This contains the `DataManager` class, which is tied with the `Options` class in terms of secondary importance compared to the `Coupler` class. The `DataManager` class allows users to organize, register, unregister, and retrieve YAKL `Array` objects of many different underlying types and dimensionalities. The functionality is detailed in the main repository `README.md` file, so please go there for more information.
* `Options.h`: This contains the `Options` class, which is tied with the `DataManager` class in terms of secondary importance compared to the `Coupler` class. The `Options` class allows users to organize simulation options via many different underlying types to declare bollean options, physical constants, boundary condition labels, and more. The functionality is detailed in the main repository `README.md` file, so please go there for more information.
* `ParallelComm.h`: This contains capability for MPI communication and activity using the `Coupler` object's MPI sub-communicator such as exchanges and reductions. The functionality is detailed in the main repository `README.md` file, so please go there for more information.
* `Ensembler.h`: This is a small convenience class that allows users to implement ensembles easily within a single executable with multiple ensemble perturbation options.
```C++
// Example usage
core::Coupler coupler;
coupler.set_option<std::string>("ensemble_stdout","ensemble_" );
coupler.set_option<std::string>("out_prefix"     ,"output_"   );
core::Ensembler ensembler;
// Add grid spacing dimension
{
  // Defines how to modify the coupler for each ensemble's options and output files
  auto func_coupler = [=] (int ind, core::Coupler &coupler) {
    int nx, ny;
    if      (ind == 0) { nx = 1000; ny = 1000; }
    else if (ind == 1) { nx = 2000; ny = 2000; }
    else               { Kokkos::abort("ERROR: Too many ensemble indices declared in register_dimension"); }
    // Allow the coupler to see nx and ny within the ensemble using options
    coupler.set_option<int>("ens_nx_glob",nx);
    coupler.set_option<int>("ens_ny_glob",ny);
    // Append strings to the stdout filename and the output filename to differentiate ensemble outputs
    ensembler.append_coupler_string(coupler,"ensemble_stdout",std::string("nx_glob-")+std::to_string(nx));
    ensembler.append_coupler_string(coupler,"out_prefix"     ,std::string("nx_glob-")+std::to_string(wind));
  };
  // Defines the number of ranks each ensemble needs as a multiple of the base number of ranks for each ensemble
  // Some ensembles will have more computational requirements than others for this dimension
  auto func_nranks = [=] (int ind) -> int {
    if      (ind == 0) { return 1; }
    else if (ind == 1) { return 4; } // half the grid spacing in x and y requires 4x more tasks
    else               { Kokkos::abort("ERROR: Too many ensemble indices declared in register_dimension"); }
    return -1;  // Doesn't get here, but compilers complain if you don't put this here.
  };
  // Declare two possible indices for this ensemble dimension,
  //     and pass the nranks and coupler modification functions for this ensemble dimension
  ensembler.register_dimension( 2 , func_nranks , func_coupler );
}
// Add option dimension for latitude of the simulation
{
  auto func_coupler = [=] (int ind, core::Coupler &coupler) {
    if      (ind == 0) { coupler.set_option<real>("latitude",0. ); }
    else if (ind == 1) { coupler.set_option<real>("latitude",10.); }
    else if (ind == 2) { coupler.set_option<real>("latitude",20.); }
    else               { Kokkos::abort("ERROR: Too many ensemble indices declared in register_dimension"); }
    auto lat = coupler.get_option<real>("latitude");
    // Append strings to the stdout filename and the output filename to differentiate ensemble outputs
    ensembler.append_coupler_string(coupler,"ensemble_stdout",std::string("lat-")+std::to_string(lat));
    ensembler.append_coupler_string(coupler,"out_prefix"     ,std::string("lat-")+std::to_string(lat));
  };
  auto func_nranks = [=] (int ind) -> int { return 1; }; // Latitude doesn't affect the computational requirements
  ensembler.register_dimension( 3 , func_nranks , func_coupler );
}
// Get the parallel communicator, and declare a base of 4 MPI tasks for ensembles.
// One of the grid spacing ensembles has a multiplier of 1 and one has a multiplier of 4.
// We have a total of 2*3=6 total ensembles across the ensemble dimensions.
// Half will have 4*4=16 MPI tasks, and half will have 4*1=4 MPI tasks.
// So this simulation will need a total of 3*16 + 3*4 = 60 total MPI tasks
auto par_comm = ensembler.create_coupler_comm( coupler , 4 , MPI_COMM_WORLD );
// The simulation should only proceed for this task if this task is within the first 60 MPI tasks required.
// Otherwise, this MPI tasks should be a no-op.
if (par_comm.valid()) {
  // Set the coupler's parallel MPI communicator object to this task's ensemble sub-communicator
  coupler.set_parallel_comm( par_comm );  
  // Redirect the stdout and stderr streams to the appropriate file name, and save the original streams
  // This way, each ensemble's cout and cerr goes to its own individual file.
  auto orig_cout_buf = std::cout.rdbuf();
  auto orig_cerr_buf = std::cerr.rdbuf();
  std::ofstream ostr(coupler.get_option<std::string>("ensemble_stdout")+std::string(".out"));
  std::cout.rdbuf(ostr.rdbuf());
  std::cerr.rdbuf(ostr.rdbuf());
  // Get the options you set for the ensembles in the coupler modification functions
  auto lat     = coupler.get_option<real>("latitude");
  auto nx_glob = coupler.set_option<int >("ens_nx_glob");
  auto ny_glob = coupler.set_option<int >("ens_ny_glob");
  // Proceed with the simulation using the ensemble-specific values
  // ... [simulation code]
  // After the simulation is completed for this ensemble, return stdout and stderr buffers to original
  std::cout.rdbuf(orig_cout_buf);
  std::cerr.rdbuf(orig_cerr_buf);
}
```
* `MultipleFields.h`: This is a small convenience class that allows users to aggregate multiple YAKL `Array` objects of the same dimensionality and underlying type into an object with an extra dimension in order to index the object as if the separate `Array` objects were part of an array object with an extra slowest varying dimension. This is helpful when trying to aggregate multiple variables and apply a single action an all of them such as is often done with dynamics variables and tracers.
```C++
// Example Usage to test if some variables have any values less than zero
// Assume a core::Coupler & coupler object was passed into this code
// The line below declares a MultiField of multiple Array objects of type real const, each with 3 dimensions
core::MultiField<real const,3> fields;
auto rho_d = coupler.get_data_manager_readonly().get<real const,3>("density_dry");
auto temp  = coupler.get_data_manager_readonly().get<real const,3>("temp"       );
auto rho_v = coupler.get_data_manager_readonly().get<real const,3>("water_vapor");
fields.add_field( rho_d );
fields.add_field( temp  );
fields.add_field( rho_v );
int num_fields = fields.get_num_fields();
int nz = coupler.get_nz();
int ny = coupler.get_ny();
int nx = coupler.get_nx();
yakl::ScalarLiveOut<bool> has_neg(false);
yakl::c::parallel_for( YAKL_AUTO_LABEL() ,
                       yakl::c::SimpleBounds<4>(num_fields,nz,ny,nx) ,
                       KOKKOS_LAMBDA (int l, int k, int j, int i) {
  // Index the MultiField object as if it were a single 4-dimensional array
  if ( fields(l,k,j,i) < 0 ) has_neg = true;
});
if (has_neg.hostRead()) Kokkos::abort("Negative value in density, temperature, or water_vapor");
// This would be much faster with yakl::intrinsics::minval, but the code above is to demonstrate usage
```
* `Counter.h`: Small class to implement a counter object to keep track of time for various recurrent tasks like file I/O and informing users about the current simulation status through `stdout` messages.

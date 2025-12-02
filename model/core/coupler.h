
#pragma once

#include "main_header.h"
#include "DataManager.h"
#include "YAKL_pnetcdf.h"
#include "MultipleFields.h"
#include "Options.h"
#include "ParallelComm.h"

// The Coupler class holds everything a component or module of this model would need in order to perform its
// changes to the model state


namespace core {

  class Coupler {
  protected:

    // Holds information about a Tracer field registered with the coupler
    // Tracers are transported by the dynamical core, written to output files, and overwritten from restart files
    // Tracers must be mass-weighted quantities to be properly transported by the dynamical core
    struct Tracer {
      std::string name;        // Tracer variable name
      std::string desc;        // Tracer description
      bool        positive;    // Whether the tracer is constrained to be positive
      bool        adds_mass;   // Whether the tracer adds mass to the total density
      bool        diffuse;     // Whether the tracer is to be diffused by the SGS closure
    };

    // Holds information about an output variable registered with the coupler
    // Variables registered must have one of the dims specifications listed in Coupler::DIMS_*
    struct OutputVar {
      std::string name;       // Variable name
      int         dims;       // Variable dims specification (1 = column, 2 = surface, 3 = 3D)
      size_t      type_hash;  // Hash of the variable type for type checking during output
    };

    real        xlen;          // Domain length in the x-direction in meters
    real        ylen;          // Domain length in the y-direction in meters
    real        zlen;          // Domain length in the z-direction in meters
    int         file_counter;  // Number of files that have been written so far
    real1d      zint;          // Interface heights of z levels (variable vertical grid)
    real1d      zmid;          // Interface heights of z levels (variable vertical grid)
    real1d      dz;            // Grid spacing of vertical cells (variable vertical grid)
    Options     options;       // Organizes shared scalar options
    DataManager dm;            // Organizes shared variables
    std::vector<Tracer>    tracers;     // Organizes tracer entries for transport and diffusion
    std::vector<OutputVar> output_vars; // Organizes output variables on the standard grid dims
    // Allows modules to register their own output writing functions for variables not on the standard grid dims
    std::vector<std::function<void(core::Coupler &coupler , yakl::SimplePNetCDF &nc)>> out_write_funcs;
    // Allows modules to register their own restart functions for variables absent from output_vars
    std::vector<std::function<void(core::Coupler &coupler , yakl::SimplePNetCDF &nc)>> restart_read_funcs;
    // Keeps track of timing information so users can assess how long modules take
    std::chrono::time_point<std::chrono::high_resolution_clock> inform_timer;
    // MPI parallelization information
    ParallelComm par_comm;   // Object to hold the MPI communicator information for this coupler and
                             //   perform MPI operations such as reductions on the communicator
    size_t nx_glob;          // Total global number of cells in the x-direction (summing all MPI Processes)
    size_t ny_glob;          // Total global number of cells in the y-direction (summing all MPI Processes)
    int    nz;               // Total number of cells in the z-direction
    int    nproc_x;          // Number of parallel processes distributed over the x-dimension
    int    nproc_y;          // Number of parallel processes distributed over the y-dimension
                             //   nproc_x * nproc_y  must equal  nranks
    int    px;               // My process ID in the x-direction
    int    py;               // My process ID in the y-direction
    size_t i_beg;            // Beginning of my x-direction global index
    size_t j_beg;            // Beginning of my y-direction global index
    size_t i_end;            // End of my x-direction global index
    size_t j_end;            // End of my y-direction global index
    SArray<int,2,3,3> neigh; // List of neighboring rank IDs;  1st index: y;  2nd index: x
                             // Y: 0 = south;  1 = middle;  2 = north
                             // X: 0 = west ;  1 = center;  3 = east 

  public:

    // Entries for the dims specification of output_vars for variables on standard grid dimensions
    int static constexpr DIMS_COLUMN  = 1; // Column variable (z dimension only)
    int static constexpr DIMS_SURFACE = 2; // Surface variable (x and y dimensions only)
    int static constexpr DIMS_3D      = 3; // 3D variable (x, y, and z dimensions)

    // Default constructor to initialize values to safe defaults
    Coupler() {
      this->xlen         = -1;
      this->ylen         = -1;
      this->zlen         = -1;
      this->file_counter = 0;
      this->inform_timer = std::chrono::high_resolution_clock::now();
      this->nx_glob      = 0;
      this->ny_glob      = 0;
      this->nz           = 0;
      this->nproc_x      = 0;
      this->nproc_y      = 0;
      this->px           = 0;
      this->py           = 0;
      this->i_beg        = 0;
      this->j_beg        = 0;
      this->i_end        = 0;
      this->j_end        = 0;
    }


    // Default move constructors
    Coupler(Coupler &&)                 = default;
    Coupler &operator=(Coupler &&)      = default;
    // Don't allow copy constructors. Always pass the coupler by reference
    Coupler(Coupler const &)            = delete;
    Coupler &operator=(Coupler const &) = delete;


    // Destructor to clean up allocated memory
    ~Coupler() {
      Kokkos::fence();
      dm.finalize();
      options.finalize();
      this->tracers      = std::vector<Tracer>();
      this->xlen         = -1;
      this->ylen         = -1;
      this->zlen         = -1;
      this->file_counter = 0;
      this->nx_glob      = 0;
      this->ny_glob      = 0;
      this->nz           = 0;
      this->nproc_x      = 0;
      this->nproc_y      = 0;
      this->px           = 0;
      this->py           = 0;
      this->i_beg        = 0;
      this->j_beg        = 0;
      this->i_end        = 0;
      this->j_end        = 0;
    }


    // To replicate a coupler, you should create a new coupler and "clone_into" that coupler
    void clone_into( Coupler &coupler ) const {
      coupler.xlen               = this->xlen              ;
      coupler.ylen               = this->ylen              ;
      coupler.zlen               = this->zlen              ;
      coupler.par_comm           = this->par_comm          ;
      coupler.nx_glob            = this->nx_glob           ;
      coupler.ny_glob            = this->ny_glob           ;
      coupler.nz                 = this->nz                ;
      coupler.nproc_x            = this->nproc_x           ;
      coupler.nproc_y            = this->nproc_y           ;
      coupler.px                 = this->px                ;
      coupler.py                 = this->py                ;
      coupler.i_beg              = this->i_beg             ;
      coupler.j_beg              = this->j_beg             ;
      coupler.i_end              = this->i_end             ;
      coupler.j_end              = this->j_end             ;
      coupler.neigh              = this->neigh             ;
      coupler.file_counter       = this->file_counter      ;
      coupler.tracers            = this->tracers           ;
      coupler.output_vars        = this->output_vars       ;
      coupler.out_write_funcs    = this->out_write_funcs   ;
      coupler.restart_read_funcs = this->restart_read_funcs;
      coupler.inform_timer       = this->inform_timer      ;
      coupler.zint               = this->zint              ;
      coupler.zmid               = this->zmid              ;
      coupler.dz                 = this->dz                ;
      this->dm     .clone_into( coupler.dm      );
      this->options.clone_into( coupler.options );
    }


    // Set the parallel communicator for this coupler for MPI operations
    void set_parallel_comm(ParallelComm par_comm) { this->par_comm = par_comm; }


    // Initialize the coupler with domain and parallelization information
    // The nproc_x_in, nproc_y_in, px_in, py_in, i_beg_in, i_end_in, j_beg_in, and j_end_in parameters
    //   are optional inputs that allow the user to manually set the MPI decomposition
    //   for multi-resolution experiments where the automatic decomposition may not naturally align grids properly
    // par_comm     : ParallelComm object holding the MPI communicator information
    // zint         : 1D array of vertical interface heights (size nz+1)
    // ny_glob      : Total global number of cells in the y-direction (summing all MPI Processes)
    // nx_glob      : Total global number of cells in the x-direction (summing all MPI Processes)
    // ylen        : Domain length in the y-direction in meters
    // xlen        : Domain length in the x-direction in meters
    // nproc_x_in  : (optional) Manually set the number of MPI processes in the x-direction
    // nproc_y_in  : (optional) Manually set the number of MPI processes in the y-direction
    // px_in       : (optional) Manually set my MPI process ID in the x-direction
    // py_in       : (optional) Manually set my MPI process ID in the y-direction
    // i_beg_in    : (optional) Manually set my beginning global index in the x-direction
    // i_end_in    : (optional) Manually set my ending global index in the x-direction
    // j_beg_in    : (optional) Manually set my beginning global index in the y-direction
    // j_end_in    : (optional) Manually set my ending global index in the y-direction
    // If any of these optional parameters are non-positive, they will be automatically computed
    //   based on a balanced domain decomposition
    // Note that indices are inclusive, so the number of cells in x is (i_end - i_beg + 1)
    //   and the number of cells in y is (j_end - j_beg + 1)
    // Note that the vertical grid is assumed to be the same on all MPI processes
    //   but the horizontal grid can be different on each MPI process
    //   to allow for domain decomposition
    // Note that the vertical grid can be variable spacing
    // Note that this function must be called before any variables are registered with the coupler
    // Note that this function performs MPI communication and should be called by all MPI processes
    //   in the communicator
    void init( ParallelComm par_comm                           ,
               real1d const & zint                             ,
               size_t ny_glob         , size_t nx_glob         ,
               real   ylen            , real   xlen            ,
               int    nproc_x_in = -1 , int    nproc_y_in = -1 ,
               int    px_in      = -1 , int    py_in      = -1 ,
               int    i_beg_in   = -1 , int    i_end_in   = -1 ,
               int    j_beg_in   = -1 , int    j_end_in   = -1 ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      this->par_comm = par_comm;
      this->zint     = zint.createDeviceCopy();
      this->nx_glob  = nx_glob;
      this->ny_glob  = ny_glob;
      this->nz       = zint.size()-1;
      this->xlen     = xlen;
      this->ylen     = ylen;
      this->zlen     = zint.createHostCopy()(nz);
      // allocate and compute dz and zmid from zint
      this->dz       = real1d("dz"  ,nz);
      this->zmid     = real1d("zmid",nz);
      YAKL_SCOPE( dz   , this->dz   );
      YAKL_SCOPE( zmid , this->zmid );
      parallel_for( YAKL_AUTO_LABEL() , nz , KOKKOS_LAMBDA (int k) {
        dz  (k) =  zint(k+1) - zint(k);
        zmid(k) = (zint(k+1) + zint(k))/2;
      });

      int nranks = par_comm.get_size();
      int myrank = par_comm.get_rank_id();

      // TODO: sim2d isn't really used
      bool sim2d = ny_glob == 1;
      if (sim2d) {
        nproc_x = nranks;
        nproc_y = 1;
      } else {
        // Decompose the horizontal grid in two dimension in a manner that minimizes surface-to-volume ratio
        std::vector<real> nproc_y_choices; // Possible choices for nproc_y
        // Find choices for nproc_y that evenly divide nranks
        for (nproc_y = 1; nproc_y <= nranks; nproc_y++) {
          if (nranks % nproc_y == 0) { nproc_y_choices.push_back(nproc_y); }
        }
        // Compute the domain's physical aspect ratio and choose the nproc_y that best matches it
        real aspect_real = static_cast<double>(ny_glob)/nx_glob;
        nproc_y = nproc_y_choices.at(0);
        real aspect = static_cast<double>(nproc_y)/(nranks/nproc_y);
        real min_dist = std::abs(aspect-aspect_real);
        for (int i=1; i < nproc_y_choices.size(); i++) {
          aspect = static_cast<double>(nproc_y_choices.at(i))/(nranks/nproc_y);
          real dist = std::abs(aspect-aspect_real);
          if (dist < min_dist) {
            nproc_y = nproc_y_choices.at(i);
            min_dist = dist;
          }
        }
        // Back out the corresponding nproc_x
        nproc_x = nranks / nproc_y;
      }

      // Get my ID within each dimension's number of ranks in each dimension
      py = myrank / nproc_x;
      px = myrank % nproc_x;

      // Get my beginning and ending indices in the x- and y- directions
      // Note that i_beg, i_end, j_beg, and j_end are all inclusive indices
      double nper;
      nper = ((double) nx_glob)/nproc_x;
      i_beg = static_cast<size_t>( round( nper* px    )   );
      i_end = static_cast<size_t>( round( nper*(px+1) )-1 );
      nper = ((double) ny_glob)/nproc_y;
      j_beg = static_cast<size_t>( round( nper* py    )   );
      j_end = static_cast<size_t>( round( nper*(py+1) )-1 );

      // For multi-resolution experiments, the user might want to set these manually to ensure that
      //   grids match up properly when decomposed into ranks
      if (nproc_x_in > 0) nproc_x = nproc_x_in;
      if (nproc_y_in > 0) nproc_y = nproc_y_in;
      if (px_in      > 0) px      = px_in     ;
      if (py_in      > 0) py      = py_in     ;
      if (i_beg_in   > 0) i_beg   = i_beg_in  ;
      if (i_end_in   > 0) i_end   = i_end_in  ;
      if (j_beg_in   > 0) j_beg   = j_beg_in  ;
      if (j_end_in   > 0) j_end   = j_end_in  ;

      // Determine my number of grid cells for my MPI task
      int nx = i_end - i_beg + 1;
      int ny = j_end - j_beg + 1;
      // Determine neighboring rank IDs in a fully tensored 3x3 grid
      for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
          int pxloc = px+i-1;
          while (pxloc < 0        ) { pxloc = pxloc + nproc_x; }
          while (pxloc > nproc_x-1) { pxloc = pxloc - nproc_x; }
          int pyloc = py+j-1;
          while (pyloc < 0        ) { pyloc = pyloc + nproc_y; }
          while (pyloc > nproc_y-1) { pyloc = pyloc - nproc_y; }
          neigh(j,i) = pyloc * nproc_x + pxloc;
        }
      }

      // Register the grid dimensions with the data manager so that variables using these dimensions
      //   automatically get the correct dimension labels.
      dm.add_dimension( "x" , nx );
      dm.add_dimension( "y" , ny );
      dm.add_dimension( "z" , nz );
      // Set initial elapsed time to zero
      set_option<real>("elapsed_time",0._fp);
      // Inform the user about the domain decomposition, grid, , number of DOFs, etc.
      if (is_mainproc()) {
        std::cout << "There are a total of " << nz << " x "
                                             << ny_glob << " x "
                                             << nx_glob << " = "
                                             << nz*ny_glob*nx_glob << " DOFs" << std::endl;
        std::cout << "MPI Decomposition using " << nproc_x << " x " << nproc_y << " = " << nranks << " tasks" << std::endl;
        std::cout << "There are roughly " << nz << " x "
                                          << ny << " x "
                                          << nx << " = "
                                          << nz*ny*nx << " DOFs per task" << std::endl;
        std::cout << "The domain is " << get_xlen()/1000 << "km x "
                                      << get_ylen()/1000 << "km x "
                                      << get_zlen()/1000 << "km in the x, y, and z directions" << std::endl;
        std::cout << "The horizontal grid spacing is " << get_dx() << "m and "
                                                       << get_dy() << "m in the x and y directions" << std::endl;
        std::cout << "The vertical grid spacing is (in meters): ";
        auto dzhost = dz.createHostCopy();
        for (int k=0; k < nz; k++) { std::cout << dzhost(k); if (k < nz-1) std::cout << " , "; }
        std::cout << std::endl;
        auto zinthost = zint.createHostCopy();
        std::cout << "The vertical grid interfaces are (in meters): ";
        for (int k=0; k < nz+1; k++) { std::cout << zinthost(k); if (k < nz) std::cout << " , "; }
        std::cout << std::endl;
      }
    }

    // Get the parallel communicator convenience class for this coupler
    ParallelComm              get_parallel_comm         () const { return this->par_comm              ; }
    
    // Get the x-dimension length of the domain in meters
    real                      get_xlen                  () const { return this->xlen                  ; }

    // Get the y-dimension length of the domain in meters
    real                      get_ylen                  () const { return this->ylen                  ; }

    // Get the z-dimension length of the domain in meters
    real                      get_zlen                  () const { return this->zlen                  ; }

    // Get the number of MPI ranks in the communicator
    int                       get_nranks                () const { return this->par_comm.get_size()   ; }
    
    // Get my MPI rank ID in the communicator
    int                       get_myrank                () const { return this->par_comm.get_rank_id(); }

    // Get the total global number of cells in the x-direction
    size_t                    get_nx_glob               () const { return this->nx_glob               ; }

    // Get the total global number of cells in the y-direction
    size_t                    get_ny_glob               () const { return this->ny_glob               ; }

    // Get the number of MPI processes in the x-direction
    int                       get_nproc_x               () const { return this->nproc_x               ; }

    // Get the total number of cells in the z-direction (same for all MPI processes)
    int                       get_nz                    () const { return this->nz                    ; }

    // Get the number of MPI processes in the y-direction
    int                       get_nproc_y               () const { return this->nproc_y               ; }

    // Get my MPI process ID in the x-direction
    int                       get_px                    () const { return this->px                    ; }

    // Get my MPI process ID in the y-direction
    int                       get_py                    () const { return this->py                    ; }

    // Get my beginning global index in the x-direction (inclusive)
    size_t                    get_i_beg                 () const { return this->i_beg                 ; }

    // Get my beginning global index in the y-direction (inclusive)
    size_t                    get_j_beg                 () const { return this->j_beg                 ; }

    // Get my ending global index in the x-direction (inclusive)
    size_t                    get_i_end                 () const { return this->i_end                 ; }

    // Get my ending global index in the y-direction (inclusive)
    size_t                    get_j_end                 () const { return this->j_end                 ; }

    // Check if the simulation is 2D (i.e., only one cell in the y-direction)
    bool                      is_sim2d                  () const { return this->ny_glob == 1          ; }

    // Check if I am the main MPI process (rank 0)
    bool                      is_mainproc               () const { return this->get_myrank() == 0     ; }

    // Get the neighbor rank ID matrix as a const reference
    SArray<int,2,3,3> const & get_neighbor_rankid_matrix() const { return this->neigh                 ; }

    // Get the DataManager as a const reference for read-only access to allocated variables
    DataManager       const & get_data_manager_readonly () const { return this->dm                    ; }

    // Get the DataManager as a non-const reference for read-write access to allocated variables
    DataManager             & get_data_manager_readwrite()       { return this->dm                    ; }

    // Get the x-direction grid spacing in meters
    real                      get_dx                    () const { return get_xlen() / get_nx_glob()  ; }

    // Get the y-direction grid spacing in meters
    real                      get_dy                    () const { return get_ylen() / get_ny_glob()  ; }

    // Get the z-direction grid spacing in meters as a 1D array
    real1d                    get_dz                    () const { return this->dz                    ; }

    // Get the z-direction interface heights as a 1D array
    real1d                    get_zint                  () const { return this->zint                  ; }

    // Get the z-direction midpoint heights as a 1D array
    real1d                    get_zmid                  () const { return this->zmid                  ; }

    // Get the number of tracers registered with the coupler
    int                       get_num_tracers           () const { return tracers.size()              ; }


    // Get the number of cells in the x-direction on this MPI process
    int get_nx() const {
      if (dm.find_dimension("x") == -1) return -1;
      return dm.get_dimension_size("x");
    }


    // Get the number of cells in the y-direction on this MPI process
    int get_ny() const {
      if (dm.find_dimension("y") == -1) return -1;
      return dm.get_dimension_size("y");
    }


    // Add the option of the templated type only if it does not already exist
    template <class T>
    void add_option_if_empty( std::string key , T value ) {
      if (!option_exists(key)) options.add_option<T>(key,value);
    }


    // Add the option of the templated type unconditionally
    template <class T>
    void add_option( std::string key , T value ) {
      options.add_option<T>(key,value);
    }


    // Set the option of the templated type unconditionally (same as add_option)
    template <class T>
    void set_option( std::string key , T value ) {
      options.set_option<T>(key,value);
    }


    // Get the option of the given name (must match the templated type)
    template <class T>
    T get_option( std::string key ) const {
      return options.get_option<T>(key);
    }


    // Get the option of the given name, or return the provided default value if the option does not exist
    template <class T>
    T get_option( std::string key , T val ) const {
      if (option_exists(key)) return options.get_option<T>(key);
      return val;
    }


    // Check if an option with the given name exists
    bool option_exists( std::string key ) const {
      return options.option_exists(key);
    }


    // Delete the option with the given name (option must exist)
    void delete_option( std::string key ) {
      options.delete_option(key);
    }


    // Run a coupler module function with standardized pre- and post-function operations
    // The function must take a Coupler reference as its only argument
    // The name parameter is used for timing and tracing purposes
    // The function is run with barriers before and after if PORTURB_FUNCTION_TIMER_BARRIER is defined
    // NaN checks are performed before and after the function if PORTURB_NAN_CHECKS is defined
    // Function timing is performed if PORTURB_FUNCTION_TIMERS is defined
    // Function variable use tracing is performed if PORTURB_FUNCTION_TRACE is defined
    // The function signature is: void func( core::Coupler &coupler )
    template <class F>
    void run_module( F const &func , std::string name ) {
      #ifdef PORTURB_FUNCTION_TIMER_BARRIER
        par_comm.barrier();
      #endif
      #ifdef PORTURB_NAN_CHECKS
        if (check_for_nan()) { std::cerr << "WARNING: NaNs before [" << name << "]" << std::endl; endrun(); }
      #endif
      #ifdef PORTURB_FUNCTION_TRACE
        dm.clean_all_entries();
      #endif
      #ifdef PORTURB_FUNCTION_TIMERS
        yakl::timer_start( name.c_str() );
      #endif

      // Run the module function with the current coupler as input
      func( *this );

      #ifdef PORTURB_FUNCTION_TIMERS
        #ifdef PORTURB_FUNCTION_TIMER_BARRIER
          par_comm.barrier();
        #endif
        yakl::timer_stop ( name.c_str() );
      #endif
      #ifdef PORTURB_FUNCTION_TRACE
        auto dirty_entry_names = dm.get_dirty_entries();
        std::cout << "PortUrb Module " << name << " wrote to the following coupler entries: ";
        for (int e=0; e < dirty_entry_names.size(); e++) {
          std::cout << dirty_entry_names.at(e);
          if (e < dirty_entry_names.size()-1) std::cout << ", ";
        }
        std::cout << "\n\n";
      #endif
      #ifdef PORTURB_NAN_CHECKS
        if (check_for_nan()) { std::cerr << "WARNING: NaNs created in [" << name << "]" << std::endl; endrun(); }
      #endif
      #ifdef PORTURB_FUNCTION_TIMER_BARRIER
        par_comm.barrier();
      #endif
    }


    // Check all standard fields for NaNs or infinite values; report which fields have NaNs
    bool check_for_nan() const {
      std::vector<std::string> names; // Holds the names of the fields being checked
      auto &dm = get_data_manager_readonly(); // Get the data manager for read-only access
      MultiField<real const,3> fields; // Holds the fields being checked
      // Accrue the standard fields to be checked for NaNs or infinite values
      fields.add_field(dm.get<real const,3>("density_dry"));  names.push_back("density_dry");
      fields.add_field(dm.get<real const,3>("uvel"       ));  names.push_back("uvel"       );
      fields.add_field(dm.get<real const,3>("vvel"       ));  names.push_back("vvel"       );
      fields.add_field(dm.get<real const,3>("wvel"       ));  names.push_back("wvel"       );
      fields.add_field(dm.get<real const,3>("temp"       ));  names.push_back("temp"       );
      auto tracer_names = get_tracer_names();
      for (int tr=0; tr < tracer_names.size(); tr++) {
        fields.add_field(dm.get<real const,3>(tracer_names.at(tr)));
        names.push_back(tracer_names.at(tr));
      }
      yakl::ScalarLiveOut<bool> nan_present(false); // Whether any NaNs were found in any field
      bool1d field_has_nan("field_has_nan",fields.get_num_fields()); // Whether an individual field has NaNs or infs
      field_has_nan = false; // Initialize to no NaNs
      // Check all fields for NaNs or infinite values
      yakl::c::parallel_for( YAKL_AUTO_LABEL() , yakl::c::SimpleBounds<4>(fields.get_num_fields(),get_nz(),get_ny(),get_nx()) ,
                                                 KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (std::isnan(fields(l,k,j,i)) || !std::isfinite(fields(l,k,j,i))) {
          nan_present = true;
          field_has_nan(l) = true;
        }
      });
      auto field_has_nan_host = field_has_nan.createHostCopy(); // Copy to host for reporting
      // Report which fields have NaNs
      for (int l=0; l < field_has_nan_host.size(); l++) {
        if (field_has_nan_host(l)) {
          std::cout << names.at(l) << ": has NaN" << std::endl;
        }
      }
      return nan_present.hostRead(); // Return whether any NaNs were found
    }

    
    // Add a tracer to the coupler's list of tracers
    // If the tracer already exists, check that the attributes match and return the existing index
    // tracer_name : Name of the tracer variable
    // tracer_desc : (optional) Description of the tracer variable
    // positive    : (optional) Whether the tracer is constrained to be positive (default: true)
    // adds_mass   : (optional) Whether the tracer adds to the total mass (default: false)
    // diffuse     : (optional) Whether the tracer is diffused by diffusion operators (default: true)
    // Returns the index of the added or existing tracer
    // Note that the tracer variable is allocated in the DataManager with dimensions (z,y,x)
    //  and can be accessed with dm.get<real,3>(tracer_name)
    // Note that the tracer variable is not initialized to any particular value upon allocation
    //  and should be initialized by the user after adding the tracer
    // Note that the tracer variable is registered with the DataManager using the provided name
    //  and description and can be accessed with that name
    // The tracer variable uses the standard coupler grid dimensions of (z,y,x)
    //  which are automatically registered with the DataManager during coupler initialization
    //  so the user does not need to manually specify dimensions when adding tracers
    // Note that the tracer variable is of type real (floating point) and cannot be of any other type
    //  to ensure compatibility with coupler operations
    // Note that if a tracer with the same name but different attributes is added, an error is raised
    //  and the program terminates
    // This function must be called after coupler initialization and before using the tracer variable
    //  in any coupler operations or modules that rely on tracer information
    int add_tracer( std::string tracer_name      ,
                    std::string tracer_desc = "" ,
                    bool positive  = true        ,
                    bool adds_mass = false       ,
                    bool diffuse   = true        ) {
      int ind = get_tracer_index(tracer_name); // Check if the tracer already exists
      // If the tracer exists, check that the attributes match and return the existing index
      if (ind != -1) {
        if (tracers.at(ind).positive != positive) {
          std::cerr << "ERROR: adding tracer [" << tracer_name
                    << "] that already exists with different positivity attribute";
          endrun();
        }
        if (tracers.at(ind).adds_mass != adds_mass) {
          std::cerr << "ERROR: adding tracer [" << tracer_name
                    << "] that already exists with different add_mass attribute";
          endrun();
        }
        if (tracers.at(ind).diffuse != diffuse) {
          std::cerr << "ERROR: adding tracer [" << tracer_name
                    << "] that already exists with different diffuse attribute";
          endrun();
        }
        return ind;
      }
      // if the tracer does not exist, register and allocate it in the DataManager
      int nz   = get_nz();
      int ny   = get_ny();
      int nx   = get_nx();
      // Register and allocate the tracer variable in the DataManager with dimensions (z,y,x)
      dm.register_and_allocate<real>( tracer_name , tracer_desc , {nz,ny,nx} , {"z","y","x"} );
      // Add the tracer to the coupler's list of tracers
      tracers.push_back( { tracer_name , tracer_desc , positive , adds_mass , diffuse } );
      // Return the index of the newly added tracer
      return tracers.size()-1;
    }

    
    // Get the names of all registered tracers as a vector of strings
    std::vector<std::string> get_tracer_names() const {
      std::vector<std::string> ret;
      for (int i=0; i < tracers.size(); i++) { ret.push_back( tracers.at(i).name ); }
      return ret;
    }

    
    // Get information about a registered tracer
    // tracer_name  : Name of the tracer variable to query
    // tracer_desc  : (output) Description of the tracer variable
    // tracer_found : (output) Whether the tracer was found
    // positive     : (output) Whether the tracer is constrained to be positive
    // adds_mass    : (output) Whether the tracer adds to the total density
    // diffuse      : (output) Whether the tracer is diffused by the SGS diffusion scheme
    void get_tracer_info(std::string tracer_name , std::string &tracer_desc, bool &tracer_found ,
                         bool &positive , bool &adds_mass, bool &diffuse) const {
      for (int i=0; i < tracers.size(); i++) {
        // If the tracer exists, set the output parameters and return
        if (tracer_name == tracers.at(i).name) {
          positive    = tracers.at(i).positive ;
          tracer_desc = tracers.at(i).desc     ;
          adds_mass   = tracers.at(i).adds_mass;
          diffuse     = tracers.at(i).diffuse  ;
          tracer_found = true;
          return;
        }
      }
      // otherwise, set tracer_found to false
      tracer_found = false;
    }

    
    // Get the index of a registered tracer by name or -1 if it does not exist
    // tracer_name : Name of the tracer variable to query
    // Returns the index of the tracer or -1 if it does not exist
    // The index can be used to access the tracer in the coupler's tracer list
    int get_tracer_index( std::string tracer_name ) const {
      for (int i=0; i < tracers.size(); i++) { if (tracer_name == tracers.at(i).name) return i; }
      return -1;
    }

    
    // Check if a tracer with the given name exists
    // tracer_name : Name of the tracer variable to query
    // Returns true if the tracer exists, false otherwise
    bool tracer_exists( std::string tracer_name ) const {
      return get_tracer_index(tracer_name) != -1;
    }


    // Get the type hash code for the given templated type (not used directly by user)
    template <class T> size_t get_type_hash() const {
      return typeid(typename std::remove_cv<T>::type).hash_code();
    }


    // Register an output variable with the coupler for output to NetCDF files
    // name : Name of the variable to output
    // dims : Number of dimensions of the variable (DIMS_1D, DIMS_2D, DIMS_3D)
    // The variable must be registered with the DataManager using the given name
    //  and allocated with the correct dimensions
    // The variable type T must match the type used when registering the variable with the DataManager
    //  or an error will be raised during output operations
    // This function does not allocate the variable; it only registers it for output operations
    //  so the user must ensure that the variable is properly registered and allocated beforehand
    // This function must be called before any output operations are performed
    template <class T> void register_output_variable( std::string name , int dims ) {
      output_vars.push_back({name,dims,get_type_hash<T>()});
    }


    // Register a function to write output variables to NetCDF files
    // The function must take a Coupler reference and a yakl::SimplePNetCDF reference as its arguments
    // The function is called during output operations to write additional variables to the NetCDF file
    // The function signature is: void func( core::Coupler &coupler , yakl::SimplePNetCDF &nc )
    // The function must be registered before any output operations are performed
    // Multiple functions can be registered and they will be called in the order they were registered
    // This allows for modular output operations where different modules can write their own variables
    //  to the same NetCDF file
    // The function is responsible for writing its own variables to the NetCDF file using the provided
    //  yakl::SimplePNetCDF object
    // The function can access coupler data and variables using the provided Coupler reference
    // The SimplePNetCDF object is already opened for writing when the function is called
    void register_write_output_module( std::function<void(core::Coupler &coupler ,
                                                          yakl::SimplePNetCDF &nc)> func ) {
      out_write_funcs.push_back( func );
    };


    // Register a function to overwrite coupler variables from NetCDF files during restart operations
    // The function must take a Coupler reference and a yakl::SimplePNetCDF reference as its arguments
    // The function is called during restart read operations to overwrite coupler variables
    // The function signature is: void func( core::Coupler &coupler , yakl::SimplePNetCDF &nc )
    // The function must be registered before any restart read operations are performed
    // Multiple functions can be registered and they will be called in the order they were registered
    // This allows for modular restart operations where different modules can overwrite their own variables
    //  from the same NetCDF file
    // The function is responsible for reading its own variables from the NetCDF file using the provided
    //  yakl::SimplePNetCDF object
    // The function can access coupler data and variables using the provided Coupler reference
    // The SimplePNetCDF object is already opened for reading when the function is called
    void register_overwrite_with_restart_module( std::function<void(core::Coupler &coupler ,
                                                                    yakl::SimplePNetCDF &nc)> func ) {
      restart_read_funcs.push_back( func );
    };


    // Track the maximum wind speed encountered during the simulation
    // Returns the current maximum wind speed in m/s
    // This must be called each time step after the velocity fields have been updated
    // The maximum wind speed is stored as an option with the key "coupler_max_wind"
    //  and can be accessed with get_option<real>("coupler_max_wind")
    // If the option does not exist, it is initialized to zero
    // The maximum wind speed is computed as sqrt(u^2 + v^2 + w^2)
    //  across all grid points and all MPI processes
    // The maximum wind speed is updated only if the newly computed value exceeds the current maximum
    //  to track the overall maximum throughout the simulation
    // The function uses parallel reduction to compute the maximum wind speed across all MPI processes
    // The function requires that the velocity fields "uvel", "vvel", and "wvel" are registered
    //  and allocated in the DataManager with dimensions (z,y,x)
    real track_max_wind() {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto u = get_data_manager_readonly().get_collapsed<real const>("uvel"); // Get u-velocity
      auto v = get_data_manager_readonly().get_collapsed<real const>("vvel"); // Get v-velocity
      auto w = get_data_manager_readonly().get_collapsed<real const>("wvel"); // Get w-velocity
      auto mag = u.createDeviceObject(); // Create array to hold wind speed magnitude
      // Compute wind speed magnitude at each grid point
      parallel_for( YAKL_AUTO_LABEL() , mag.size() , KOKKOS_LAMBDA (int i) {
        mag(i) = std::sqrt( u(i)*u(i) + v(i)*v(i) + w(i)*w(i) );
      });
      // Perform parallel reduction to find maximum wind speed across all MPI processes
      auto mx = par_comm.reduce( yakl::intrinsics::maxval(mag) , MPI_MAX , 0 );
      // Update the coupler_max_wind option if the new maximum exceeds the current maximum
      set_option<real>("coupler_max_wind",std::max(mx,get_option<real>("coupler_max_wind",0.)));
      // Return the current maximum wind speed
      return get_option<real>("coupler_max_wind");
    }


    // Inform the user about the current simulation status
    // Prints elapsed simulation time, wall clock time since the last time this routine was called,
    //  maximum wind speed, and maximum vertical velocity
    // This can be called as often as desired
    // The routine uses the coupler's internal timer to track wall clock time between calls
    // The routine requires that the velocity fields "uvel", "vvel", and "wvel" are registered
    //  and allocated in the DataManager with dimensions (z,y,x)
    // The routine uses parallel reduction to compute maximum wind speed and vertical velocity
    //  across all MPI processes
    // The output is printed only by the main MPI process of the coupler's communicator
    void inform_user( ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      Kokkos::fence(); // Ensure prior device operations are complete before timing
      auto t2 = std::chrono::high_resolution_clock::now(); // Get current time
      std::chrono::duration<double> dur_step = t2 - inform_timer; // Compute duration since last call
      inform_timer = t2; // Update timer for next call
      auto u = get_data_manager_readonly().get_collapsed<real const>("uvel"); // Get u-velocity
      auto v = get_data_manager_readonly().get_collapsed<real const>("vvel"); // Get v-velocity
      auto w = get_data_manager_readonly().get_collapsed<real const>("wvel"); // Get w-velocity
      auto mag = u.createDeviceObject(); // Create array to hold wind speed magnitude
      // Compute wind speed magnitude at each grid point
      parallel_for( YAKL_AUTO_LABEL() , mag.size() , KOKKOS_LAMBDA (int i) {
        mag(i) = std::sqrt( u(i)*u(i) + v(i)*v(i) + w(i)*w(i) );
      });
      // Perform parallel reduction to find maximum wind speed and maximum vertical velocity across all MPI processes
      auto wind_mag = par_comm.reduce( yakl::intrinsics::maxval(mag ) , MPI_MAX , 0 );
      auto w_mag    = par_comm.reduce( yakl::intrinsics::maxval(yakl::intrinsics::abs(w)) , MPI_MAX , 0 );
      // Print the information only on the main MPI process for the coupler's communicator
      if (is_mainproc()) {
        std::cout << "Etime ["
                  << std::scientific << std::setw(10) << get_option<real>("elapsed_time") << " s] , Walltime [" 
                  << std::scientific << std::setw(10) << dur_step.count()                 << " s] , max wind ["
                  << std::scientific << std::setw(10) << wind_mag                         << " m/s] , max(abs(w)) ["
                  << std::scientific << std::setw(10) << w_mag                            << " m/s]" << std::endl;
      }
    }


    // Generate equally spaced vertical levels
    // nz   : Number of vertical levels
    // zlen : Total vertical length in meters
    // Returns a 1D array of size nz+1 containing the vertical interface heights in meters
    // The levels are equally spaced from 0 to zlen
    // The returned array can be used to set the coupler's vertical grid interfaces
    real1d generate_levels_equal( int nz , real zlen ) const {
      realHost1d zint("zint",nz+1);
      for (int i=0; i < nz+1; i++) { zint(i) = i*zlen/nz; }
      return zint.createDeviceCopy();
    }


    // Generate exponentially stretched vertical levels
    // nz   : Number of vertical levels
    // zlen : Total vertical length in meters
    // dz0  : Minimum vertical grid spacing in meters at the bottom of the domain
    // Returns a 1D array of size nz+1 containing the vertical interface heights in meters
    // The levels are exponentially stretched from dz0 at the bottom to larger spacings at the top
    // The returned array can be used to set the coupler's vertical grid interfaces
    // If dz0 is larger than zlen/nz, it is set to zlen/nz to ensure valid spacing
    // The stretching factor is determined iteratively to ensure the total length matches zlen
    // The stretching results in finer resolution near the bottom and coarser resolution near the top
    // This is useful for simulations where higher resolution is needed near the surface
    real1d generate_levels_exp( int nz , real zlen , real dz0 ) const {
      using yakl::intrinsics::sum;
      using yakl::componentwise::operator-; // Allows use of '-' on yakl arrays
      using yakl::componentwise::operator*; // Allows use of '*' on yakl arrays
      if (dz0 > zlen / nz) dz0 = zlen/nz; // Ensure dz0 is not too large
      realHost1d dz("dz",nz); // Array to hold vertical spacings
      real f1 = 0; // Lower bound for stretching factor
      real f2 = 5; // Upper bound for stretching factor
      // Perform iterations
      while (f2-f1 >= 1.e-13) { // Iterate until convergence
        real f = (f1+f2)/2; // Midpoint stretching factor
        dz(0) = dz0;        // Set bottom spacing
        // Compute stretched spacings
        for (int k=1; k < nz; k++) { dz(k) = dz(k-1)*f; }
        if (sum(dz) > zlen) { f2 = f; } // Adjust upper bound if total length exceeds zlen
        else                { f1 = f; } // Adjust lower bound otherwise
      }
      realHost1d zint("zint",nz+1); // Array to hold vertical interface heights on host
      zint(0) = 0; // Bottom interface at 0
      for (int k=0; k < nz; k++) { zint(k+1) = zint(k) + dz(k); } // Compute interface heights
      // Scale to ensure total length matches zlen exactly
      zint = zint * (zlen / zint(nz));
      // return a device copy of the interface heights
      return zint.createDeviceCopy();
    }


    // Generate vertical levels with variable grid spacing stretching from dz0 at the bottom to dz1 at height z1
    // zlen : Total vertical length in meters
    // dz0  : Minimum vertical grid spacing in meters at the bottom of the domain
    // z1   : Height in meters where the grid spacing transitions to dz1
    // dz1  : Vertical grid spacing in meters above height z1
    // Returns a 1D array of size nz+1 containing the vertical interface heights in meters
    // The levels transition smoothly from dz0 at the bottom to dz1 above height z1
    // The returned array can be used to set the coupler's vertical grid interfaces
    // 
    // Below is sagemath code used to derive the polynomial for the variable grid spacing
    // The resulting C++ code implements the derived polynomial spacing function
    // var('dz0,dz1')
    // N      = 5
    // coefs  = coefs_1d(N,0,'a')
    // p      = poly_1d(N,coefs,x)
    // constr = vector([ p.subs(x=0) , p.diff(x).subs(x=0) , p.subs(x=1) , p.diff(x).subs(x=1) , p.diff(x,2).subs(x=1) ])
    // p      = poly_1d(N,jacobian(constr,coefs)^-1*vector([dz0,0,dz1,0,0]),x)
    // print(p)
    real1d generate_levels_const_high( real zlen , real dz0 , real z1 , real dz1 ) const {
      using yakl::intrinsics::sum;
      using yakl::componentwise::operator-; // Allows use of '-' on yakl arrays
      using yakl::componentwise::operator*; // Allows use of '*' on yakl arrays
      int nzmax = (int) std::ceil(zlen/std::min(dz0,dz1)); // Maximum possible number of levels
      realHost1d dz("dz",nzmax); // Array to hold vertical spacings using host memory
      dz = dz0; // Initialize all spacings to dz0
      // Perform iterations
      for (int iter=0; iter < 100; iter++) { // Iterate to refine spacings
        real z = 0; // Current height in the domain
        for (int k=0; k < nzmax; k++) { // Loop over all possible levels
          real zn = (z+dz(k)/2)/z1; // Normalized height for spacing calculation
          // Compute spacing based on height
          if (zn <= 1) { dz(k) = -3*(dz0-dz1)*zn*zn*zn*zn+8*(dz0-dz1)*zn*zn*zn-6*(dz0-dz1)*zn*zn+dz0; }
          else         { dz(k) = dz1;                                                                 }
          z += dz(k); // Update current height
        }
      }
      // Find actual number of vertical levels
      real z = 0; // Reset current height
      int nz = 0; // Actual number of levels
      for (int k=0; k < nzmax; k++) { // Loop to determine actual number of levels
        z += dz(k); // Update current height
        if (z >= zlen) { nz = k+1; break; } // Check if total length exceeded. if so, set nz and break
      }
      realHost1d zint("zint",nz+1); // Array to hold vertical interface heights on host
      zint(0) = 0; // Bottom interface at 0
      for (int k=0; k < nz; k++) { zint(k+1) = zint(k) + dz(k); } // Compute interface heights
      // Scale to ensure total length matches zlen exactly
      zint = zint * (zlen / zint(nz));
      // return a device copy of the interface heights
      return zint.createDeviceCopy();
    }


    // Generate vertical levels with variable grid spacing stretching from dz0 to dz2 over the range [z1,z2]
    // zlen : Total vertical length in meters
    // dz0  : Minimum vertical grid spacing in meters at the bottom of the domain
    // z1   : Height in meters where the grid spacing transition begins
    // z2   : Height in meters where the grid spacing transition ends
    // dz2  : Vertical grid spacing in meters above height z2
    // Returns a 1D array of size nz+1 containing the vertical interface heights in meters
    // The levels transition smoothly from dz0 at the bottom to dz2 above height z2
    // The returned array can be used to set the coupler's vertical grid interfaces
    real1d generate_levels_const_low_high( real zlen , real dz0 , real z1 , real z2 , real dz2 ) const {
      using yakl::intrinsics::sum;
      using yakl::componentwise::operator-; // Allows use of '-' on yakl arrays
      using yakl::componentwise::operator*; // Allows use of '*' on yakl arrays
      int nzmax = (int) std::ceil(zlen/std::min(dz0,dz2)); // Maximum possible number of levels
      realHost1d dz("dz",nzmax); // Array to hold vertical spacings using host memory
      dz = dz0; // Initialize all spacings to dz0
      // Perform iterations
      for (int iter=0; iter < 100; iter++) { // Iterate to refine spacings
        real z = 0; // Current height in the domain
        for (int k=0; k < nzmax; k++) { // Loop over all possible levels
          real zn = (z+dz(k)/2-z1)/(z2-z1); // Normalized height for spacing calculation
          // Compute spacing based on height
          if      (zn <  0) { dz(k) = dz0;                                                                 }
          else if (zn <= 1) { dz(k) = -3*(dz0-dz2)*zn*zn*zn*zn+8*(dz0-dz2)*zn*zn*zn-6*(dz0-dz2)*zn*zn+dz0; }
          else              { dz(k) = dz2;                                                                 }
          z += dz(k); // Update current height
        }
      }
      // Find actual number of vertical levels
      real z = 0; // Reset current height
      int nz = 0; // Actual number of levels
      for (int k=0; k < nzmax; k++) { // Loop to determine actual number of levels
        z += dz(k); // Update current height
        if (z >= zlen) { nz = k+1; break; } // Check if total length exceeded. if so, set nz and break
      }
      realHost1d zint("zint",nz+1); // Array to hold vertical interface heights on host
      zint(0) = 0; // Bottom interface at 0
      for (int k=0; k < nz; k++) { zint(k+1) = zint(k) + dz(k); } // Compute interface heights
      // Scale to ensure total length matches zlen exactly
      zint = zint * (zlen / zint(nz));
      // return a device copy of the interface heights
      return zint.createDeviceCopy();
    }


    // Write output/restart file with the given prefix
    // prefix  : Prefix for the output file name
    // verbose : Whether to print status messages (default: true)
    // The output file is written in NetCDF format using yakl::SimplePNetCDF
    // The file name is constructed as prefix_XXXXXXXX.nc where XXXXXXXX is a zero-padded file counter
    // The file contains the coupler's standard fields and any registered output variables
    // The standard fields include density_dry, uvel, vvel, wvel, temperature, and tracers
    // The file also contains metadata such as elapsed time and file counter
    // The function uses parallel I/O to write the file efficiently across MPI processes
    // The function prints status messages if verbose is true and the current process is the main process
    // The function uses MPI_Info hints to optimize I/O performance for large files
    // The function increments the file counter after writing the file to ensure unique file names
    void write_output_file( std::string prefix , bool verbose = true ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      typedef unsigned char uchar; // Define uchar type for unsigned char for output variables of that type
      yakl::timer_start("coupler_output"); // Start timer for output operation
      if (verbose && is_mainproc()) std::cout << "*** Beginning outputOutput/restart file ***" << std::endl;
      auto nx          = get_nx();                         // Local number of x grid points
      auto ny          = get_ny();                         // Local number of y grid points
      auto nz          = get_nz();                         // Number of z grid points
      auto dx          = get_dx();                         // Grid spacing in x direction
      auto dy          = get_dy();                         // Grid spacing in y direction
      auto num_tracers = get_num_tracers();                // Number of registered tracers
      auto C0          = get_option<real>("C0");           // Proportionality constant for pressure computation
      auto R_d         = get_option<real>("R_d");          // Specific gas constant for dry air in J/(kg·K)
      auto gamma       = get_option<real>("gamma_d");      // Ratio of specific heats for dry air (cp/cv)
      auto etime       = get_option<real>("elapsed_time"); // Current elapsed time of the model
      int  i_beg       = get_i_beg();                      // Starting index in x direction for this MPI process (inclusive)
      int  j_beg       = get_j_beg();                      // Starting index in y direction for this MPI process (inclusive)
      auto zint        = get_zint();                       // Vertical interface heights
      //////////////////////////////////////////////////////
      // FILE OPERATIONS
      //////////////////////////////////////////////////////
      yakl::SimplePNetCDF nc(par_comm.get_mpi_comm()); // Create SimplePNetCDF object with the coupler's MPI communicator
      std::stringstream fname; // String stream to construct file name
      fname << prefix << "_" << std::setw(8) << std::setfill('0') << file_counter << ".nc";
      MPI_Info info; // Create MPI_Info object for I/O hints
      MPI_Info_create(&info); // Initialize MPI_Info object
      MPI_Info_set(info, "romio_no_indep_rw",    "true");    // Set I/O hints for performance
      MPI_Info_set(info, "nc_header_align_size", "1048576"); // Set I/O hints for performance
      MPI_Info_set(info, "nc_var_align_size",    "1048576"); // Set I/O hints for performance
      // Create the NetCDF file with the specified name to overwrite existing files and use 64-bit offsets for large files
      nc.create(fname.str() , NC_CLOBBER | NC_64BIT_DATA , info );
      //////////////////////////////////////////////////////
      // DIMENSIONS
      //////////////////////////////////////////////////////      
      nc.create_dim( "x"   , get_nx_glob() ); // Create global x dimension
      nc.create_dim( "y"   , get_ny_glob() ); // Create global y dimension
      nc.create_dim( "z"   , nz );            // Create z dimension
      nc.create_dim( "zi"  , nz+1 );          // Create z-interface dimension
      nc.create_dim( "t"   , 1 );             // Create time dimension, even though it's only one time per file
      std::vector<std::string> dimnames_column  = {"z"};         // define dimension names for column variables
      std::vector<std::string> dimnames_surface = {"y","x"};     // define dimension names for surface variables
      std::vector<std::string> dimnames_3d      = {"z","y","x"}; // define dimension names for 3D variables
      //////////////////////////////////////////////////////
      // CREATE VARIABLES
      ////////////////////////////////////////////////////// 
      nc.create_var<float>( "x"   , {"x"} );                // Create x-coordinate variable
      nc.create_var<float>( "y"   , {"y"} );                // Create y-coordinate variable
      nc.create_var<float>( "z"   , {"z"} );                // Create z-coordinate variable
      nc.create_var<float>( "zi"  , {"zi"} );               // Create z-interface coordinate variable
      nc.create_var<float>( "density_dry"  , dimnames_3d ); // Create dry density variable
      nc.create_var<float>( "uvel"         , dimnames_3d ); // Create u-velocity variable
      nc.create_var<float>( "vvel"         , dimnames_3d ); // Create v-velocity variable
      nc.create_var<float>( "wvel"         , dimnames_3d ); // Create w-velocity variable
      nc.create_var<float>( "temperature"  , dimnames_3d ); // Create temperature variable
      nc.create_var<float>( "etime"        , {"t"} );       // Create elapsed time variable
      nc.create_var<int  >( "file_counter" , {"t"} );       // Create variable to store current file counter
      auto tracer_names = get_tracer_names();
      // Create tracer variables
      for (int tr = 0; tr < num_tracers; tr++) { nc.create_var<float>( tracer_names.at(tr) , dimnames_3d ); }
      // Create user-registered output variables, each according to the specified dimensions and type
      for (int ivar = 0; ivar < output_vars.size(); ivar++) {
        auto name = output_vars.at(ivar).name;
        auto hash = output_vars.at(ivar).type_hash;
        auto dims = output_vars.at(ivar).dims;
        if        (dims == DIMS_COLUMN ) {
          if      (hash == get_type_hash<float >()) { nc.create_var<float >(name,dimnames_column ); }
          else if (hash == get_type_hash<double>()) { nc.create_var<float >(name,dimnames_column ); }
          else if (hash == get_type_hash<int   >()) { nc.create_var<int   >(name,dimnames_column ); }
          else if (hash == get_type_hash<uchar >()) { nc.create_var<uchar >(name,dimnames_column ); }
        } else if (dims == DIMS_SURFACE) {
          if      (hash == get_type_hash<float >()) { nc.create_var<float >(name,dimnames_surface); }
          else if (hash == get_type_hash<double>()) { nc.create_var<float >(name,dimnames_surface); }
          else if (hash == get_type_hash<int   >()) { nc.create_var<int   >(name,dimnames_surface); }
          else if (hash == get_type_hash<uchar >()) { nc.create_var<uchar >(name,dimnames_surface); }
        } else if (dims == DIMS_3D     ) {
          if      (hash == get_type_hash<float >()) { nc.create_var<float >(name,dimnames_3d     ); }
          else if (hash == get_type_hash<double>()) { nc.create_var<float >(name,dimnames_3d     ); }
          else if (hash == get_type_hash<int   >()) { nc.create_var<int   >(name,dimnames_3d     ); }
          else if (hash == get_type_hash<uchar >()) { nc.create_var<uchar >(name,dimnames_3d     ); }
        }
      }
      nc.enddef(); // End define mode, which means we can now write data to the file
      //////////////////////////////////////////////////////
      // WRITE DATA TO FILE
      ////////////////////////////////////////////////////// 
      // Create and write the x-coordinate data to file
      float1d xloc("xloc",nx);
      parallel_for( YAKL_AUTO_LABEL() , nx , KOKKOS_LAMBDA (int i) { xloc(i) = (i+i_beg+0.5)*dx; });
      nc.write_all( xloc , "x" , {i_beg} );
      // Create and write the y-coordinate data to file
      float1d yloc("yloc",ny);
      parallel_for( YAKL_AUTO_LABEL() , ny , KOKKOS_LAMBDA (int j) { yloc(j) = (j+j_beg+0.5)*dy; });
      nc.write_all( yloc , "y" , {j_beg} );
      nc.begin_indep_data(); // Begin independent data section for variables that are the same on all processes
      if (is_mainproc()) nc.write( zmid         , "z"            ); // Write z midpoints from main process
      if (is_mainproc()) nc.write( zint         , "zi"           ); // Write z interfaces from main process
      if (is_mainproc()) nc.write( (float)etime , "etime"        ); // Write elapsed time from main process
      if (is_mainproc()) nc.write( file_counter , "file_counter" ); // Write file counter from main process
      nc.end_indep_data(); // End independent data section so that other processes can write their own data
      auto &dm = get_data_manager_readonly(); // Get a reference to the read-only DataManager
      std::vector<MPI_Offset> start_3d      = {0,j_beg,i_beg}; // Starting indices for 3D variables
      std::vector<MPI_Offset> start_surface = {  j_beg,i_beg}; // Starting indices for surface variables
      nc.write_all(dm.get<real const,3>("density_dry").as<float>(),"density_dry",start_3d); // Write dry density
      nc.write_all(dm.get<real const,3>("uvel"       ).as<float>(),"uvel"       ,start_3d); // Write u-velocity
      nc.write_all(dm.get<real const,3>("vvel"       ).as<float>(),"vvel"       ,start_3d); // Write v-velocity
      nc.write_all(dm.get<real const,3>("wvel"       ).as<float>(),"wvel"       ,start_3d); // Write w-velocity
      nc.write_all(dm.get<real const,3>("temp"       ).as<float>(),"temperature",start_3d); // Write temperature
      // Write tracer variables to file
      for (int i=0; i < tracer_names.size(); i++) {
        nc.write_all(dm.get<real const,3>(tracer_names.at(i)).as<float>(),tracer_names.at(i),start_3d);
      }
      // Write user-registered output variables to file according to their specified dimensions and type
      for (int ivar = 0; ivar < output_vars.size(); ivar++) {
        auto name = output_vars.at(ivar).name;      // Get variable name
        auto hash = output_vars.at(ivar).type_hash; // Get variable type hash
        auto dims = output_vars.at(ivar).dims;      // Get variable dimensions
        if        (dims == DIMS_COLUMN ) { // For column variables, write 1D data independently from main process
          nc.begin_indep_data(); // Begin independent data section
          if (is_mainproc()) {
            // Write the column variable data based on its type
            if      (hash == get_type_hash<float >()) { nc.write(dm.get<float  const,1>(name),name); }
            else if (hash == get_type_hash<double>()) { nc.write(dm.get<double const,1>(name).as<float>(),name); }
            else if (hash == get_type_hash<int   >()) { nc.write(dm.get<int    const,1>(name),name); }
            else if (hash == get_type_hash<uchar >()) { nc.write(dm.get<uchar  const,1>(name),name); }
          }
          nc.end_indep_data(); // End independent data section
        } else if (dims == DIMS_SURFACE) { // For surface variables, write 2D data from each process
          if      (hash == get_type_hash<float >()) { nc.write_all(dm.get<float  const,2>(name),name,start_surface); }
          else if (hash == get_type_hash<double>()) { nc.write_all(dm.get<double const,2>(name).as<float>(),name,start_surface); }
          else if (hash == get_type_hash<int   >()) { nc.write_all(dm.get<int    const,2>(name),name,start_surface); }
          else if (hash == get_type_hash<uchar >()) { nc.write_all(dm.get<uchar  const,2>(name),name,start_surface); }
        } else if (dims == DIMS_3D     ) { // For 3D variables, write 3D data from each process
          if      (hash == get_type_hash<float >()) { nc.write_all(dm.get<float  const,3>(name),name,start_3d); }
          else if (hash == get_type_hash<double>()) { nc.write_all(dm.get<double const,3>(name).as<float>(),name,start_3d); }
          else if (hash == get_type_hash<int   >()) { nc.write_all(dm.get<int    const,3>(name),name,start_3d); }
          else if (hash == get_type_hash<uchar >()) { nc.write_all(dm.get<uchar  const,3>(name),name,start_3d); }
        }
      }
      // Execute the user-registered output functions to write any additional data to the file
      for (int i=0; i < out_write_funcs.size(); i++) { out_write_funcs.at(i)(*this,nc); }
      nc.close(); // Close the NetCDF file
      file_counter++; // Increment the file counter for the next output for a unique file name
      yakl::timer_stop("coupler_output"); // Stop the output timer
      // Print status message if verbose and on main process
      if (verbose && is_mainproc()) {
        std::cout << "*** Output/restart file written ***  -->  Etime , Output time: "
                  << std::scientific << std::setw(10) << etime            << " , " 
                  << std::scientific << std::setw(10) << timer_last("coupler_output") << std::endl;
      }
    }


    // Get the MPI data type corresponding to the template type T using the coupler's parallel communicator.
    // T : Data type for which to get the corresponding MPI data type (default: real)
    // Returns the MPI_Datatype corresponding to type T using the coupler's parallel communicator
    // This function is useful for performing MPI operations with data of type T in the coupler's parallel environment
    template<class T=real> MPI_Datatype get_mpi_data_type() const { return par_comm.get_type<T>(); }


    // Overwrite the coupler's data with values read from a restart file specified in the coupler options.
    // The restart file is read in NetCDF format using yakl::SimplePNetCDF.
    // The function reads standard coupler fields and any registered restart read functions.
    // Standard fields include density_dry, uvel, vvel, wvel, temperature, and tracers.
    // The function also reads any registered output variables automatically based on their dimensions and types.
    // The function updates the coupler's elapsed time and file counter based on the restart file.
    // The function uses parallel I/O to read the file efficiently across MPI processes.
    // The function prints a status message if the current process is the main process.
    void overwrite_with_restart() {
      typedef unsigned char uchar; // Useful for output variables of type unsigned char
      yakl::timer_start("overwrite_with_restart"); // Start timer for restart operation
      // Print status message if on main process
      if (is_mainproc())  std::cout << "*** Restarting from file: "
                                    << get_option<std::string>("restart_file") << std::endl;
      int i_beg = get_i_beg(); // Get starting index in x direction for this MPI process (inclusive)
      int j_beg = get_j_beg(); // Get starting index in y direction for this MPI process (inclusive)
      auto tracer_names = get_tracer_names();
      yakl::SimplePNetCDF nc(par_comm.get_mpi_comm()); // Create SimplePNetCDF object with the coupler's MPI communicator
      nc.open( get_option<std::string>("restart_file") , NC_NOWRITE ); // Open the restart file in read-only mode
      nc.begin_indep_data(); // Begin independent data section
      real etime;
      if (is_mainproc()) nc.read( etime        , "etime"        ); // Read elapsed time from main process
      if (is_mainproc()) nc.read( file_counter , "file_counter" ); // Read file counter from main process
      nc.end_indep_data(); // End independent data section
      par_comm.broadcast(file_counter); // Broadcast file counter to all processes
      par_comm.broadcast(etime       ); // Broadcast elapsed time to all processes
      set_option<real>("elapsed_time",etime); // Update coupler's elapsed time
      std::vector<MPI_Offset> start_3d      = {0,j_beg,i_beg}; // Starting indices for 3D variables
      std::vector<MPI_Offset> start_surface = {  j_beg,i_beg}; // Starting indices for surface variables
      std::vector<MPI_Offset> start_column  = {0            }; // Starting index for column variables
      nc.read_all(dm.get<real,3>("density_dry"),"density_dry",start_3d); // Read dry density
      nc.read_all(dm.get<real,3>("uvel"       ),"uvel"       ,start_3d); // Read u-velocity
      nc.read_all(dm.get<real,3>("vvel"       ),"vvel"       ,start_3d); // Read v-velocity
      nc.read_all(dm.get<real,3>("wvel"       ),"wvel"       ,start_3d); // Read w-velocity
      nc.read_all(dm.get<real,3>("temp"       ),"temperature",start_3d); // Read temperature
      // Read tracer variables from file
      for (int i=0; i < tracer_names.size(); i++) {
        if (nc.var_exists(tracer_names.at(i))) {
          nc.read_all(dm.get<real,3>(tracer_names.at(i)),tracer_names.at(i),start_3d);
        }
      }
      // Read user-registered output variables from file according to their specified dimensions and type
      for (int ivar = 0; ivar < output_vars.size(); ivar++) {
        auto name = output_vars.at(ivar).name;
        auto hash = output_vars.at(ivar).type_hash;
        auto dims = output_vars.at(ivar).dims;
        if        (dims == DIMS_COLUMN ) {
          if      (hash == get_type_hash<float >()) { nc.read_all(dm.get<float ,1>(name),name,start_column); }
          else if (hash == get_type_hash<double>()) { nc.read_all(dm.get<double,1>(name),name,start_column); }
          else if (hash == get_type_hash<int   >()) { nc.read_all(dm.get<int   ,1>(name),name,start_column); }
          else if (hash == get_type_hash<uchar >()) { nc.read_all(dm.get<uchar ,1>(name),name,start_column); }
        } else if (dims == DIMS_SURFACE) {
          if      (hash == get_type_hash<float >()) { nc.read_all(dm.get<float ,2>(name),name,start_surface); }
          else if (hash == get_type_hash<double>()) { nc.read_all(dm.get<double,2>(name),name,start_surface); }
          else if (hash == get_type_hash<int   >()) { nc.read_all(dm.get<int   ,2>(name),name,start_surface); }
          else if (hash == get_type_hash<uchar >()) { nc.read_all(dm.get<uchar ,2>(name),name,start_surface); }
        } else if (dims == DIMS_3D     ) {
          if      (hash == get_type_hash<float >()) { nc.read_all(dm.get<float ,3>(name),name,start_3d); }
          else if (hash == get_type_hash<double>()) { nc.read_all(dm.get<double,3>(name),name,start_3d); }
          else if (hash == get_type_hash<int   >()) { nc.read_all(dm.get<int   ,3>(name),name,start_3d); }
          else if (hash == get_type_hash<uchar >()) { nc.read_all(dm.get<uchar ,3>(name),name,start_3d); }
        }
      }
      // Execute the user-registered restart read functions to read any additional data from the file
      for (int i=0; i < restart_read_funcs.size(); i++) { restart_read_funcs.at(i)(*this,nc); }
      nc.close(); // Close the NetCDF file
      file_counter++; // Increment the file counter for the next output for a unique file name
      yakl::timer_stop("overwrite_with_restart"); // Stop the restart timer
    }


    // Create a new MultiField with halos added and exchange halo values periodically in the horizontal
    // fields_in : Input MultiField without halos
    // hs        : Halo size to add around each field
    // Returns a new MultiField with halos added and halo values exchanged in the horizontal directions
    // The input MultiField must have at least one field
    // The output MultiField will have the same number of fields as the input, each with halos added
    // The function uses parallel_for to copy the interior values and then calls halo_exchange to exchange halo values
    // This template specialization is for 3D MultiFields
    // The return type has const and volatile qualifiers removed from the underlying type in case a const or
    //  volatile MultiField is passed in
    // It is assumed that all fields have the same dimensions. If that is untrue, an error is raised, and execution aborts.
    // The vertical halos are left undefined as halo exchange is only performed in the horizontal directions.
    // Typically, the user will fill vertical halos with appropriate boundary conditions after calling this function.
    template <class T>
    MultiField<typename std::remove_cv<T>::type,3>
    create_and_exchange_halos( MultiField<T,3> const &fields_in , int hs ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      typedef typename std::remove_cv<T>::type T_NOCV; // Remove const and volatile qualifiers from T
      if (fields_in.get_num_fields() == 0) Kokkos::abort("ERROR: create_and_exchange_halos: create_halos input has zero fields");
      auto num_fields = fields_in.get_num_fields();       // Get number of fields in input MultiField
      auto nz         = fields_in.get_field(0).extent(0); // Get vertical extent of first field
      auto ny         = fields_in.get_field(0).extent(1); // Get y extent of first field
      auto nx         = fields_in.get_field(0).extent(2); // Get x extent of first field
      MultiField<T_NOCV,3> fields_out; // Create output MultiField
      for (int i=0; i < num_fields; i++) { // Loop over all fields to create output fields with halos
        auto field = fields_in.get_field(i); // Get current input field
        // Check that current field has the same dimensions as the first field
        if ( field.extent(0) != nz || field.extent(1) != ny || field.extent(2) != nx ) {
          Kokkos::abort("ERROR: create_and_exchange_halos: sizes not equal among fields");
        }
        // Allocate output field with halos added
        yakl::Array<T_NOCV,3,yakl::memDevice,yakl::styleC> ret(field.label(),nz+2*hs,ny+2*hs,nx+2*hs);
        fields_out.add_field( ret ); // Add output field to output MultiField
      }
      // Copy interior values from input MultiField to output MultiField using parallel_for
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        fields_out(l,hs+k,hs+j,hs+i) = fields_in(l,k,j,i);
      });
      // Exchange halo values in the output MultiField in the horizontal directions (vertical halos undefined!)
      halo_exchange( fields_out , hs );
      return fields_out;
    }


    // Same as above but for 2D MultiFields
    template <class T>
    MultiField<typename std::remove_cv<T>::type,2>
    create_and_exchange_halos( MultiField<T,2> const &fields_in , int hs ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      typedef typename std::remove_cv<T>::type T_NOCV;
      if (fields_in.get_num_fields() == 0) Kokkos::abort("ERROR: create_and_exchange_halos: create_halos input has zero fields");
      auto num_fields = fields_in.get_num_fields();
      auto ny         = fields_in.get_field(0).extent(0);
      auto nx         = fields_in.get_field(0).extent(1);
      MultiField<T_NOCV,2> fields_out;
      for (int i=0; i < num_fields; i++) {
        auto field = fields_in.get_field(i);
        if ( field.extent(0) != ny || field.extent(1) != nx ) {
          Kokkos::abort("ERROR: create_and_exchange_halos: sizes not equal among fields");
        }
        yakl::Array<T_NOCV,2,yakl::memDevice,yakl::styleC> ret(field.label(),ny+2*hs,nx+2*hs);
        fields_out.add_field( ret );
      }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_fields,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int j, int i) {
        fields_out(l,hs+j,hs+i) = fields_in(l,j,i);
      });
      halo_exchange( fields_out , hs );
      return fields_out;
    }


    // Exchange halo values periodically in the horizontal directions (vertical halos undefined!)
    // fields : MultiField of 3-D fields with halos to exchange
    // hs     : Halo size around each field
    // The input MultiField must have at least one field
    // It is assumed that all fields have the same dimensions. If that is untrue, an error is raised, and execution aborts.
    // The function uses parallel_for to pack and unpack halo values and the coupler's parallel communicator to perform the exchanges
    // This template specialization is for 3D MultiFields
    // The vertical halos are left undefined as halo exchange is only performed in the horizontal directions.
    // Typically, the user will fill vertical halos with appropriate boundary conditions after calling this function.
    template <class T>
    void halo_exchange( core::MultiField<T,3> & fields , int hs ) const {
      #ifdef YAKL_AUTO_PROFILE
        par_comm.barrier();
        yakl::timer_start("halo_exchange");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      if (fields.get_num_fields() == 0) Kokkos::abort("ERROR: halo_exchange: create_halos input has zero fields");
      int  npack  = fields.get_num_fields();             // Number of fields to exchange
      auto nz     = fields.get_field(0).extent(0)-2*hs;  // Number of vertical cells without halos
      auto ny     = fields.get_field(0).extent(1)-2*hs;  // Number of y cells without halos
      auto nx     = fields.get_field(0).extent(2)-2*hs;  // Number of x cells without halos
      auto &neigh = get_neighbor_rankid_matrix();        // Get neighbor rank ID matrix from coupler's parallel communicator
      // Check that all fields have the same dimensions
      for (int i=0; i < npack; i++) {
        auto field = fields.get_field(i);
        if ( field.extent(0) != nz+2*hs ||
             field.extent(1) != ny+2*hs ||
             field.extent(2) != nx+2*hs ) {
          Kokkos::abort("ERROR: halo_exchange: sizes not equal among fields");
        }
      }

      // x-direction exchanges
      {
        // Allocate send and receive buffers for west and east halos
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_W("halo_send_buf_W",npack,nz,ny,hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_E("halo_send_buf_E",npack,nz,ny,hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_W("halo_recv_buf_W",npack,nz,ny,hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_E("halo_recv_buf_E",npack,nz,ny,hs);
        // Pack halo values into send buffers using parallel_for
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,hs) ,
                                          KOKKOS_LAMBDA (int v, int k, int j, int ii) {
          halo_send_buf_W(v,k,j,ii) = fields(v,hs+k,hs+j,hs+ii);
          halo_send_buf_E(v,k,j,ii) = fields(v,hs+k,hs+j,nx+ii);
        });
        // Perform halo exchanges using the coupler's parallel communicator
        get_parallel_comm().send_receive<T,4>( { {halo_recv_buf_W,neigh(1,0),0} , {halo_recv_buf_E,neigh(1,2),1} } ,
                                               { {halo_send_buf_W,neigh(1,0),1} , {halo_send_buf_E,neigh(1,2),0} } );
        // Unpack received halo values into fields using parallel_for 
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,hs) ,
                                          KOKKOS_LAMBDA (int v, int k, int j, int ii) {
          fields(v,hs+k,hs+j,      ii) = halo_recv_buf_W(v,k,j,ii);
          fields(v,hs+k,hs+j,nx+hs+ii) = halo_recv_buf_E(v,k,j,ii);
        });
      }

      // y-direction exchanges
      {
        // Allocate send and receive buffers for south and north halos
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_S("halo_send_buf_S",npack,nz,hs,nx+2*hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_N("halo_send_buf_N",npack,nz,hs,nx+2*hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_S("halo_recv_buf_S",npack,nz,hs,nx+2*hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_N("halo_recv_buf_N",npack,nz,hs,nx+2*hs);
        // Pack halo values into send buffers using parallel_for
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,hs,nx+2*hs) ,
                                          KOKKOS_LAMBDA (int v, int k, int jj, int i) {
          halo_send_buf_S(v,k,jj,i) = fields(v,hs+k,hs+jj,i);
          halo_send_buf_N(v,k,jj,i) = fields(v,hs+k,ny+jj,i);
        });
        // 
        get_parallel_comm().send_receive<T,4>( { {halo_recv_buf_S,neigh(0,1),2} , {halo_recv_buf_N,neigh(2,1),3} } ,
                                               { {halo_send_buf_S,neigh(0,1),3} , {halo_send_buf_N,neigh(2,1),2} } );
        // Unpack received halo values into fields using parallel_for
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,hs,nx+2*hs) ,
                                          KOKKOS_LAMBDA (int v, int k, int jj, int i) {
          fields(v,hs+k,      jj,i) = halo_recv_buf_S(v,k,jj,i);
          fields(v,hs+k,ny+hs+jj,i) = halo_recv_buf_N(v,k,jj,i);
        });
      }
      // Stop profiling timer
      #ifdef YAKL_AUTO_PROFILE
        par_comm.barrier();
        yakl::timer_stop("halo_exchange");
      #endif
    }


    // Same as before except for a 4-D yakl::Array where the slowest varying index is the field index
    //  and the remaining three indices are the 3D spatial indices with halos
    template <class T>
    void halo_exchange( yakl::Array<T,4,yakl::memDevice,yakl::styleC> const & fields , int hs ) const {
      #ifdef YAKL_AUTO_PROFILE
        par_comm.barrier();
        yakl::timer_start("halo_exchange");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int  npack  = fields.extent(0);
      auto nz     = fields.extent(1)-2*hs;
      auto ny     = fields.extent(2)-2*hs;
      auto nx     = fields.extent(3)-2*hs;
      auto &neigh = get_neighbor_rankid_matrix();

      // x-direction exchanges
      {
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_W("halo_send_buf_W",npack,nz,ny,hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_E("halo_send_buf_E",npack,nz,ny,hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_W("halo_recv_buf_W",npack,nz,ny,hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_E("halo_recv_buf_E",npack,nz,ny,hs);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,hs) ,
                                          KOKKOS_LAMBDA (int v, int k, int j, int ii) {
          halo_send_buf_W(v,k,j,ii) = fields(v,hs+k,hs+j,hs+ii);
          halo_send_buf_E(v,k,j,ii) = fields(v,hs+k,hs+j,nx+ii);
        });
        get_parallel_comm().send_receive<T,4>( { {halo_recv_buf_W,neigh(1,0),0} , {halo_recv_buf_E,neigh(1,2),1} } ,
                                               { {halo_send_buf_W,neigh(1,0),1} , {halo_send_buf_E,neigh(1,2),0} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,hs) ,
                                          KOKKOS_LAMBDA (int v, int k, int j, int ii) {
          fields(v,hs+k,hs+j,      ii) = halo_recv_buf_W(v,k,j,ii);
          fields(v,hs+k,hs+j,nx+hs+ii) = halo_recv_buf_E(v,k,j,ii);
        });
      }

      // y-direction exchanges
      {
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_S("halo_send_buf_S",npack,nz,hs,nx+2*hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_N("halo_send_buf_N",npack,nz,hs,nx+2*hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_S("halo_recv_buf_S",npack,nz,hs,nx+2*hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_N("halo_recv_buf_N",npack,nz,hs,nx+2*hs);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,hs,nx+2*hs) ,
                                          KOKKOS_LAMBDA (int v, int k, int jj, int i) {
          halo_send_buf_S(v,k,jj,i) = fields(v,hs+k,hs+jj,i);
          halo_send_buf_N(v,k,jj,i) = fields(v,hs+k,ny+jj,i);
        });
        get_parallel_comm().send_receive<T,4>( { {halo_recv_buf_S,neigh(0,1),2} , {halo_recv_buf_N,neigh(2,1),3} } ,
                                               { {halo_send_buf_S,neigh(0,1),3} , {halo_send_buf_N,neigh(2,1),2} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,hs,nx+2*hs) ,
                                          KOKKOS_LAMBDA (int v, int k, int jj, int i) {
          fields(v,hs+k,      jj,i) = halo_recv_buf_S(v,k,jj,i);
          fields(v,hs+k,ny+hs+jj,i) = halo_recv_buf_N(v,k,jj,i);
        });
      }
      #ifdef YAKL_AUTO_PROFILE
        par_comm.barrier();
        yakl::timer_stop("halo_exchange");
      #endif
    }


    // Same as before except with a MultiField of 2-D fields rather than 3-D fields
    template <class T>
    void halo_exchange( core::MultiField<T,2> & fields , int hs ) const {
      #ifdef YAKL_AUTO_PROFILE
        par_comm.barrier();
        yakl::timer_start("halo_exchange");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      if (fields.get_num_fields() == 0) Kokkos::abort("ERROR: halo_exchange: create_halos input has zero fields");
      int  npack  = fields.get_num_fields();
      auto ny     = fields.get_field(0).extent(0)-2*hs;
      auto nx     = fields.get_field(0).extent(1)-2*hs;
      auto &neigh = get_neighbor_rankid_matrix();

      for (int i=0; i < npack; i++) {
        auto field = fields.get_field(i);
        if ( field.extent(0) != ny+2*hs || field.extent(1) != nx+2*hs ) {
          Kokkos::abort("ERROR: halo_exchange: sizes not equal among fields");
        }
      }

      // x-direction exchanges
      {
        yakl::Array<T,3,yakl::memDevice,yakl::styleC> halo_send_buf_W("halo_send_buf_W",npack,ny,hs);
        yakl::Array<T,3,yakl::memDevice,yakl::styleC> halo_send_buf_E("halo_send_buf_E",npack,ny,hs);
        yakl::Array<T,3,yakl::memDevice,yakl::styleC> halo_recv_buf_W("halo_recv_buf_W",npack,ny,hs);
        yakl::Array<T,3,yakl::memDevice,yakl::styleC> halo_recv_buf_E("halo_recv_buf_E",npack,ny,hs);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,ny,hs) ,
                                          KOKKOS_LAMBDA (int v, int j, int ii) {
          halo_send_buf_W(v,j,ii) = fields(v,hs+j,hs+ii);
          halo_send_buf_E(v,j,ii) = fields(v,hs+j,nx+ii);
        });
        get_parallel_comm().send_receive<T,3>( { {halo_recv_buf_W,neigh(1,0),0} , {halo_recv_buf_E,neigh(1,2),1} } ,
                                               { {halo_send_buf_W,neigh(1,0),1} , {halo_send_buf_E,neigh(1,2),0} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,ny,hs) ,
                                          KOKKOS_LAMBDA (int v, int j, int ii) {
          fields(v,hs+j,      ii) = halo_recv_buf_W(v,j,ii);
          fields(v,hs+j,nx+hs+ii) = halo_recv_buf_E(v,j,ii);
        });
      }

      // y-direction exchanges
      {
        yakl::Array<T,3,yakl::memDevice,yakl::styleC> halo_send_buf_S("halo_send_buf_S",npack,hs,nx+2*hs);
        yakl::Array<T,3,yakl::memDevice,yakl::styleC> halo_send_buf_N("halo_send_buf_N",npack,hs,nx+2*hs);
        yakl::Array<T,3,yakl::memDevice,yakl::styleC> halo_recv_buf_S("halo_recv_buf_S",npack,hs,nx+2*hs);
        yakl::Array<T,3,yakl::memDevice,yakl::styleC> halo_recv_buf_N("halo_recv_buf_N",npack,hs,nx+2*hs);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,hs,nx+2*hs) ,
                                          KOKKOS_LAMBDA (int v, int jj, int i) {
          halo_send_buf_S(v,jj,i) = fields(v,hs+jj,i);
          halo_send_buf_N(v,jj,i) = fields(v,ny+jj,i);
        });
        get_parallel_comm().send_receive<T,3>( { {halo_recv_buf_S,neigh(0,1),2} , {halo_recv_buf_N,neigh(2,1),3} } ,
                                               { {halo_send_buf_S,neigh(0,1),3} , {halo_send_buf_N,neigh(2,1),2} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,hs,nx+2*hs) ,
                                          KOKKOS_LAMBDA (int v, int jj, int i) {
          fields(v,      jj,i) = halo_recv_buf_S(v,jj,i);
          fields(v,ny+hs+jj,i) = halo_recv_buf_N(v,jj,i);
        });
      }
      #ifdef YAKL_AUTO_PROFILE
        par_comm.barrier();
        yakl::timer_stop("halo_exchange");
      #endif
    }


    // Same as halo_exchange( yakl::Array<T,4,yakl::memDevice,yakl::styleC> const & fields , int hs )
    //  except only in the x-direction
    template <class T>
    void halo_exchange_x( yakl::Array<T,4,yakl::memDevice,yakl::styleC> const & fields , int hs ) const {
      #ifdef YAKL_AUTO_PROFILE
        par_comm.barrier();
        yakl::timer_start("halo_exchange");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int  npack  = fields.extent(0);
      auto nz     = fields.extent(1)-2*hs;
      auto ny     = fields.extent(2)-2*hs;
      auto nx     = fields.extent(3)-2*hs;
      auto &neigh = get_neighbor_rankid_matrix();

      // x-direction exchanges
      {
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_W("halo_send_buf_W",npack,nz,ny,hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_E("halo_send_buf_E",npack,nz,ny,hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_W("halo_recv_buf_W",npack,nz,ny,hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_E("halo_recv_buf_E",npack,nz,ny,hs);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,hs) ,
                                          KOKKOS_LAMBDA (int v, int k, int j, int ii) {
          halo_send_buf_W(v,k,j,ii) = fields(v,hs+k,hs+j,hs+ii);
          halo_send_buf_E(v,k,j,ii) = fields(v,hs+k,hs+j,nx+ii);
        });
        get_parallel_comm().send_receive<T,4>( { {halo_recv_buf_W,neigh(1,0),0} , {halo_recv_buf_E,neigh(1,2),1} } ,
                                               { {halo_send_buf_W,neigh(1,0),1} , {halo_send_buf_E,neigh(1,2),0} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,hs) ,
                                          KOKKOS_LAMBDA (int v, int k, int j, int ii) {
          fields(v,hs+k,hs+j,      ii) = halo_recv_buf_W(v,k,j,ii);
          fields(v,hs+k,hs+j,nx+hs+ii) = halo_recv_buf_E(v,k,j,ii);
        });
      }
      #ifdef YAKL_AUTO_PROFILE
        par_comm.barrier();
        yakl::timer_stop("halo_exchange");
      #endif
    }


    // Same as halo_exchange( core::MultiField<T,3> & fields , int hs ) except only in the x-direction
    template <class T>
    void halo_exchange_x( core::MultiField<T,3> & fields , int hs ) const {
      #ifdef YAKL_AUTO_PROFILE
        par_comm.barrier();
        yakl::timer_start("halo_exchange");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      if (fields.get_num_fields() == 0) Kokkos::abort("ERROR: halo_exchange: create_halos input has zero fields");
      int  npack  = fields.get_num_fields();
      auto nz     = fields.get_field(0).extent(0)-2*hs;
      auto ny     = fields.get_field(0).extent(1)-2*hs;
      auto nx     = fields.get_field(0).extent(2)-2*hs;
      auto &neigh = get_neighbor_rankid_matrix();

      for (int i=0; i < npack; i++) {
        auto field = fields.get_field(i);
        if ( field.extent(0) != nz+2*hs ||
             field.extent(1) != ny+2*hs ||
             field.extent(2) != nx+2*hs ) {
          Kokkos::abort("ERROR: halo_exchange: sizes not equal among fields");
        }
      }

      // x-direction exchanges
      {
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_W("halo_send_buf_W",npack,nz,ny,hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_E("halo_send_buf_E",npack,nz,ny,hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_W("halo_recv_buf_W",npack,nz,ny,hs);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_E("halo_recv_buf_E",npack,nz,ny,hs);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,hs) ,
                                          KOKKOS_LAMBDA (int v, int k, int j, int ii) {
          halo_send_buf_W(v,k,j,ii) = fields(v,hs+k,hs+j,hs+ii);
          halo_send_buf_E(v,k,j,ii) = fields(v,hs+k,hs+j,nx+ii);
        });
        get_parallel_comm().send_receive<T,4>( { {halo_recv_buf_W,neigh(1,0),0} , {halo_recv_buf_E,neigh(1,2),1} } ,
                                               { {halo_send_buf_W,neigh(1,0),1} , {halo_send_buf_E,neigh(1,2),0} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,hs) ,
                                          KOKKOS_LAMBDA (int v, int k, int j, int ii) {
          fields(v,hs+k,hs+j,      ii) = halo_recv_buf_W(v,k,j,ii);
          fields(v,hs+k,hs+j,nx+hs+ii) = halo_recv_buf_E(v,k,j,ii);
        });
      }
      #ifdef YAKL_AUTO_PROFILE
        par_comm.barrier();
        yakl::timer_stop("halo_exchange");
      #endif
    }


    // Same as halo_exchange( core::MultiField<T,3> & fields , int hs ) except only in the y-direction
    template <class T>
    void halo_exchange_y( core::MultiField<T,3> & fields , int hs ) const {
      #ifdef YAKL_AUTO_PROFILE
        par_comm.barrier();
        yakl::timer_start("halo_exchange");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      if (fields.get_num_fields() == 0) Kokkos::abort("ERROR: halo_exchange: create_halos input has zero fields");
      int  npack  = fields.get_num_fields();
      auto nz     = fields.get_field(0).extent(0)-2*hs;
      auto ny     = fields.get_field(0).extent(1)-2*hs;
      auto nx     = fields.get_field(0).extent(2)-2*hs;
      auto &neigh = get_neighbor_rankid_matrix();
      Kokkos::fence();

      for (int i=0; i < npack; i++) {
        auto field = fields.get_field(i);
        if ( field.extent(0) != nz+2*hs ||
             field.extent(1) != ny+2*hs ||
             field.extent(2) != nx+2*hs ) {
          Kokkos::abort("ERROR: halo_exchange: sizes not equal among fields");
        }
      }
      // y-direction exchanges
      {
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_S("halo_send_buf_S",npack,nz,hs,nx);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_N("halo_send_buf_N",npack,nz,hs,nx);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_S("halo_recv_buf_S",npack,nz,hs,nx);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_N("halo_recv_buf_N",npack,nz,hs,nx);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,hs,nx) ,
                                          KOKKOS_LAMBDA (int v, int k, int jj, int i) {
          halo_send_buf_S(v,k,jj,i) = fields(v,hs+k,hs+jj,hs+i);
          halo_send_buf_N(v,k,jj,i) = fields(v,hs+k,ny+jj,hs+i);
        });
        get_parallel_comm().send_receive<T,4>( { {halo_recv_buf_S,neigh(0,1),2} , {halo_recv_buf_N,neigh(2,1),3} } ,
                                               { {halo_send_buf_S,neigh(0,1),3} , {halo_send_buf_N,neigh(2,1),2} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,hs,nx) ,
                                          KOKKOS_LAMBDA (int v, int k, int jj, int i) {
          fields(v,hs+k,      jj,hs+i) = halo_recv_buf_S(v,k,jj,i);
          fields(v,hs+k,ny+hs+jj,hs+i) = halo_recv_buf_N(v,k,jj,i);
        });
      }
      #ifdef YAKL_AUTO_PROFILE
        par_comm.barrier();
        yakl::timer_stop("halo_exchange");
      #endif
    }


    // Same as halo_exchange( yakl::Array<T,4,yakl::memDevice,yakl::styleC> const & fields , int hs ) except only in the y-direction
    template <class T>
    void halo_exchange_y( yakl::Array<T,4,yakl::memDevice,yakl::styleC> const & fields , int hs ) const {
      #ifdef YAKL_AUTO_PROFILE
        par_comm.barrier();
        yakl::timer_start("halo_exchange");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int  npack  = fields.extent(0);
      auto nz     = fields.extent(1)-2*hs;
      auto ny     = fields.extent(2)-2*hs;
      auto nx     = fields.extent(3)-2*hs;
      auto &neigh = get_neighbor_rankid_matrix();
      Kokkos::fence();

      // y-direction exchanges
      {
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_S("halo_send_buf_S",npack,nz,hs,nx);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_send_buf_N("halo_send_buf_N",npack,nz,hs,nx);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_S("halo_recv_buf_S",npack,nz,hs,nx);
        yakl::Array<T,4,yakl::memDevice,yakl::styleC> halo_recv_buf_N("halo_recv_buf_N",npack,nz,hs,nx);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,hs,nx) ,
                                          KOKKOS_LAMBDA (int v, int k, int jj, int i) {
          halo_send_buf_S(v,k,jj,i) = fields(v,hs+k,hs+jj,hs+i);
          halo_send_buf_N(v,k,jj,i) = fields(v,hs+k,ny+jj,hs+i);
        });
        get_parallel_comm().send_receive<T,4>( { {halo_recv_buf_S,neigh(0,1),2} , {halo_recv_buf_N,neigh(2,1),3} } ,
                                               { {halo_send_buf_S,neigh(0,1),3} , {halo_send_buf_N,neigh(2,1),2} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,hs,nx) ,
                                          KOKKOS_LAMBDA (int v, int k, int jj, int i) {
          fields(v,hs+k,      jj,hs+i) = halo_recv_buf_S(v,k,jj,i);
          fields(v,hs+k,ny+hs+jj,hs+i) = halo_recv_buf_N(v,k,jj,i);
        });
      }
      #ifdef YAKL_AUTO_PROFILE
        par_comm.barrier();
        yakl::timer_stop("halo_exchange");
      #endif
    }

  };

}




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

    struct Tracer {
      std::string name;
      std::string desc;
      bool        positive;
      bool        adds_mass;
      bool        diffuse;
    };

    struct OutputVar {
      std::string name;
      int         dims;
      size_t      type_hash;
    };

    real        xlen;          // Domain length in the x-direction in meters
    real        ylen;          // Domain length in the y-direction in meters
    real        zlen;          // Domain length in the z-direction in meters
    int         file_counter;  // Number of files that have been written so far
    real1d      zint;          // Interface heights of z levels (variable vertical grid)
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
    int static constexpr DIMS_COLUMN  = 1;
    int static constexpr DIMS_SURFACE = 2;
    int static constexpr DIMS_3D      = 3;

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
      coupler.dz                 = this->dz                ;
      this->dm     .clone_into( coupler.dm      );
      this->options.clone_into( coupler.options );
    }


    void set_parallel_comm(ParallelComm par_comm) { this->par_comm = par_comm; }


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
      this->dz       = real1d("dz",nz);
      YAKL_SCOPE(dz,this->dz);
      parallel_for( YAKL_AUTO_LABEL() , nz , KOKKOS_LAMBDA (int k) {
        dz(k) = zint(k+1) - zint(k);
      });

      int nranks = par_comm.get_size();
      int myrank = par_comm.get_rank_id();

      bool sim2d = ny_glob == 1;

      if (sim2d) {
        nproc_x = nranks;
        nproc_y = 1;
      } else {
        std::vector<real> nproc_y_choices;
        for (nproc_y = 1; nproc_y <= nranks; nproc_y++) {
          if (nranks % nproc_y == 0) { nproc_y_choices.push_back(nproc_y); }
        }
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
        nproc_x = nranks / nproc_y;
      }

      // Get my ID within each dimension's number of ranks
      py = myrank / nproc_x;
      px = myrank % nproc_x;

      // Get my beginning and ending indices in the x- and y- directions
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

      //Determine my number of grid cells
      int nx = i_end - i_beg + 1;
      int ny = j_end - j_beg + 1;
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

      dm.add_dimension( "x" , nx );
      dm.add_dimension( "y" , ny );
      dm.add_dimension( "z" , nz );
      set_option<real>("elapsed_time",0._fp);
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
        std::cout << "The vertical grid spacing is (in meters): " << dz;
        std::cout << "The vertical grid interfaces are (in meters): " << zint;
        std::cout << std::endl;
      }
    }


    ParallelComm              get_parallel_comm         () const { return this->par_comm              ; }
    real                      get_xlen                  () const { return this->xlen                  ; }
    real                      get_ylen                  () const { return this->ylen                  ; }
    real                      get_zlen                  () const { return this->zlen                  ; }
    int                       get_nranks                () const { return this->par_comm.get_size()   ; }
    int                       get_myrank                () const { return this->par_comm.get_rank_id(); }
    size_t                    get_nx_glob               () const { return this->nx_glob               ; }
    size_t                    get_ny_glob               () const { return this->ny_glob               ; }
    int                       get_nz                    () const { return this->nz                    ; }
    int                       get_nproc_x               () const { return this->nproc_x               ; }
    int                       get_nproc_y               () const { return this->nproc_y               ; }
    int                       get_px                    () const { return this->px                    ; }
    int                       get_py                    () const { return this->py                    ; }
    size_t                    get_i_beg                 () const { return this->i_beg                 ; }
    size_t                    get_j_beg                 () const { return this->j_beg                 ; }
    size_t                    get_i_end                 () const { return this->i_end                 ; }
    size_t                    get_j_end                 () const { return this->j_end                 ; }
    bool                      is_sim2d                  () const { return this->ny_glob == 1          ; }
    bool                      is_mainproc               () const { return this->get_myrank() == 0     ; }
    SArray<int,2,3,3> const & get_neighbor_rankid_matrix() const { return this->neigh                 ; }
    DataManager       const & get_data_manager_readonly () const { return this->dm                    ; }
    DataManager             & get_data_manager_readwrite()       { return this->dm                    ; }
    real                      get_dx                    () const { return get_xlen() / get_nx_glob()  ; }
    real                      get_dy                    () const { return get_ylen() / get_ny_glob()  ; }
    real1d                    get_dz                    () const { return this->dz                    ; }
    real1d                    get_zint                  () const { return this->zint                  ; }
    int                       get_num_tracers           () const { return tracers.size()              ; }


    int get_nx() const {
      if (dm.find_dimension("x") == -1) return -1;
      return dm.get_dimension_size("x");
    }


    int get_ny() const {
      if (dm.find_dimension("y") == -1) return -1;
      return dm.get_dimension_size("y");
    }


    template <class T>
    void add_option_if_empty( std::string key , T value ) {
      if (!option_exists(key)) options.add_option<T>(key,value);
    }


    template <class T>
    void add_option( std::string key , T value ) {
      options.add_option<T>(key,value);
    }


    template <class T>
    void set_option( std::string key , T value ) {
      options.set_option<T>(key,value);
    }


    template <class T>
    T get_option( std::string key ) const {
      return options.get_option<T>(key);
    }


    template <class T>
    T get_option( std::string key , T val ) const {
      if (option_exists(key)) return options.get_option<T>(key);
      return val;
    }


    bool option_exists( std::string key ) const {
      return options.option_exists(key);
    }


    void delete_option( std::string key ) {
      options.delete_option(key);
    }


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


    bool check_for_nan() const {
      std::vector<std::string> names;
      auto &dm = get_data_manager_readonly();
      MultiField<real const,3> fields;
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
      yakl::ScalarLiveOut<bool> nan_present(false);
      bool1d field_has_nan("field_has_nan",fields.get_num_fields());
      field_has_nan = false;
      yakl::c::parallel_for( YAKL_AUTO_LABEL() , yakl::c::SimpleBounds<4>(fields.get_num_fields(),get_nz(),get_ny(),get_nx()) ,
                                                 KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (std::isnan(fields(l,k,j,i)) || !std::isfinite(fields(l,k,j,i))) {
          nan_present = true;
          field_has_nan(l) = true;
        }
      });
      auto field_has_nan_host = field_has_nan.createHostCopy();
      for (int l=0; l < field_has_nan_host.size(); l++) {
        if (field_has_nan_host(l)) {
          std::cout << names.at(l) << ": has NaN" << std::endl;
        }
      }
      return nan_present.hostRead();
    }

    
    int add_tracer( std::string tracer_name      ,
                    std::string tracer_desc = "" ,
                    bool positive  = true        ,
                    bool adds_mass = false       ,
                    bool diffuse   = true        ) {
      int ind = get_tracer_index(tracer_name);
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
      int nz   = get_nz();
      int ny   = get_ny();
      int nx   = get_nx();
      dm.register_and_allocate<real>( tracer_name , tracer_desc , {nz,ny,nx} , {"z","y","x"} );
      tracers.push_back( { tracer_name , tracer_desc , positive , adds_mass , diffuse } );
      return tracers.size()-1;
    }

    
    std::vector<std::string> get_tracer_names() const {
      std::vector<std::string> ret;
      for (int i=0; i < tracers.size(); i++) { ret.push_back( tracers.at(i).name ); }
      return ret;
    }

    
    void get_tracer_info(std::string tracer_name , std::string &tracer_desc, bool &tracer_found ,
                         bool &positive , bool &adds_mass, bool &diffuse) const {
      std::vector<std::string> ret;
      for (int i=0; i < tracers.size(); i++) {
        if (tracer_name == tracers.at(i).name) {
          positive    = tracers.at(i).positive ;
          tracer_desc = tracers.at(i).desc     ;
          adds_mass   = tracers.at(i).adds_mass;
          diffuse     = tracers.at(i).diffuse  ;
          tracer_found = true;
          return;
        }
      }
      tracer_found = false;
    }

    
    int get_tracer_index( std::string tracer_name ) const {
      for (int i=0; i < tracers.size(); i++) { if (tracer_name == tracers.at(i).name) return i; }
      return -1;
    }

    
    bool tracer_exists( std::string tracer_name ) const {
      return get_tracer_index(tracer_name) != -1;
    }


    template <class T> size_t get_type_hash() const {
      return typeid(typename std::remove_cv<T>::type).hash_code();
    }


    template <class T> void register_output_variable( std::string name , int dims ) {
      output_vars.push_back({name,dims,get_type_hash<T>()});
    }


    void register_write_output_module( std::function<void(core::Coupler &coupler ,
                                                          yakl::SimplePNetCDF &nc)> func ) {
      out_write_funcs.push_back( func );
    };


    void register_overwrite_with_restart_module( std::function<void(core::Coupler &coupler ,
                                                                    yakl::SimplePNetCDF &nc)> func ) {
      restart_read_funcs.push_back( func );
    };


    real track_max_wind() {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto u = get_data_manager_readonly().get_collapsed<real const>("uvel");
      auto v = get_data_manager_readonly().get_collapsed<real const>("vvel");
      auto w = get_data_manager_readonly().get_collapsed<real const>("wvel");
      auto mag = u.createDeviceObject();
      parallel_for( YAKL_AUTO_LABEL() , mag.size() , KOKKOS_LAMBDA (int i) {
        mag(i) = std::sqrt( u(i)*u(i) + v(i)*v(i) + w(i)*w(i) );
      });
      auto mx = par_comm.reduce( yakl::intrinsics::maxval(mag) , MPI_MAX , 0 );
      set_option<real>("coupler_max_wind",std::max(mx,get_option<real>("coupler_max_wind",0.)));
      return get_option<real>("coupler_max_wind");
    }


    void inform_user( ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      Kokkos::fence();
      auto t2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> dur_step = t2 - inform_timer;
      inform_timer = t2;
      auto u = get_data_manager_readonly().get_collapsed<real const>("uvel");
      auto v = get_data_manager_readonly().get_collapsed<real const>("vvel");
      auto w = get_data_manager_readonly().get_collapsed<real const>("wvel");
      auto mag = u.createDeviceObject();
      parallel_for( YAKL_AUTO_LABEL() , mag.size() , KOKKOS_LAMBDA (int i) {
        mag(i) = std::sqrt( u(i)*u(i) + v(i)*v(i) + w(i)*w(i) );
      });
      auto wind_mag = par_comm.reduce( yakl::intrinsics::maxval(mag ) , MPI_MAX , 0 );
      auto w_mag    = par_comm.reduce( yakl::intrinsics::maxval(yakl::intrinsics::abs(w)) , MPI_MAX , 0 );
      if (is_mainproc()) {
        std::cout << "Etime ["
                  << std::scientific << std::setw(10) << get_option<real>("elapsed_time") << " s] , Walltime [" 
                  << std::scientific << std::setw(10) << dur_step.count()                 << " s] , max wind ["
                  << std::scientific << std::setw(10) << wind_mag                         << " m/s] , max(abs(w)) ["
                  << std::scientific << std::setw(10) << w_mag                            << " m/s]" << std::endl;
      }
    }


    real1d generate_levels_equal( int nz , real zlen ) const {
      realHost1d zint("zint",nz+1);
      for (int i=0; i < nz+1; i++) { zint(i) = i*zlen/nz; }
      return zint.createDeviceCopy();
    }


    void write_output_file( std::string prefix , bool verbose = true ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      typedef unsigned char uchar;
      yakl::timer_start("coupler_output");
      if (verbose && is_mainproc()) std::cout << "*** Beginning outputOutput/restart file ***" << std::endl;
      auto nx          = get_nx();
      auto ny          = get_ny();
      auto nz          = get_nz();
      auto dx          = get_dx();
      auto dy          = get_dy();
      auto num_tracers = get_num_tracers();
      auto C0          = get_option<real>("C0");
      auto R_d         = get_option<real>("R_d");
      auto gamma       = get_option<real>("gamma_d");
      auto etime       = get_option<real>("elapsed_time");
      int  i_beg       = get_i_beg();
      int  j_beg       = get_j_beg();
      auto zint        = get_zint();
      yakl::SimplePNetCDF nc(par_comm.get_mpi_comm());
      std::stringstream fname;
      fname << prefix << "_" << std::setw(8) << std::setfill('0') << file_counter << ".nc";
      MPI_Info info;
      MPI_Info_create(&info);
      MPI_Info_set(info, "romio_no_indep_rw",    "true");
      MPI_Info_set(info, "nc_header_align_size", "1048576");
      MPI_Info_set(info, "nc_var_align_size",    "1048576");
      nc.create(fname.str() , NC_CLOBBER | NC_64BIT_DATA , info );
      nc.create_dim( "x"   , get_nx_glob() );
      nc.create_dim( "y"   , get_ny_glob() );
      nc.create_dim( "z"   , nz );
      nc.create_dim( "zi"  , nz+1 );
      nc.create_dim( "t"   , 1 );
      std::vector<std::string> dimnames_column  = {"z"};
      std::vector<std::string> dimnames_surface = {"y","x"};
      std::vector<std::string> dimnames_3d      = {"z","y","x"};
      nc.create_var<float>( "x"   , {"x"} );
      nc.create_var<float>( "y"   , {"y"} );
      nc.create_var<float>( "z"   , {"z"} );
      nc.create_var<float>( "zi"  , {"zi"} );
      nc.create_var<float>( "density_dry"  , dimnames_3d );
      nc.create_var<float>( "uvel"         , dimnames_3d );
      nc.create_var<float>( "vvel"         , dimnames_3d );
      nc.create_var<float>( "wvel"         , dimnames_3d );
      nc.create_var<float>( "temperature"  , dimnames_3d );
      nc.create_var<float>( "etime"        , {"t"} );
      nc.create_var<int  >( "file_counter" , {"t"} );
      auto tracer_names = get_tracer_names();
      for (int tr = 0; tr < num_tracers; tr++) { nc.create_var<float>( tracer_names.at(tr) , dimnames_3d ); }
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
      nc.enddef();
      // x-coordinate
      float1d xloc("xloc",nx);
      parallel_for( YAKL_AUTO_LABEL() , nx , KOKKOS_LAMBDA (int i) { xloc(i) = (i+i_beg+0.5)*dx; });
      nc.write_all( xloc , "x" , {i_beg} );
      // y-coordinate
      float1d yloc("yloc",ny);
      parallel_for( YAKL_AUTO_LABEL() , ny , KOKKOS_LAMBDA (int j) { yloc(j) = (j+j_beg+0.5)*dy; });
      nc.write_all( yloc , "y" , {j_beg} );
      // z-coordinate
      float1d zloc("zloc",nz);
      parallel_for( YAKL_AUTO_LABEL() , nz , KOKKOS_LAMBDA (int k) { zloc(k) = (zint(k)+zint(k+1))/2.; });
      nc.begin_indep_data();
      if (is_mainproc()) nc.write( zloc         , "z"            );
      if (is_mainproc()) nc.write( zint         , "zi"           );
      if (is_mainproc()) nc.write( (float)etime , "etime"        );
      if (is_mainproc()) nc.write( file_counter , "file_counter" );
      nc.end_indep_data();
      auto &dm = get_data_manager_readonly();
      std::vector<MPI_Offset> start_3d      = {0,j_beg,i_beg};
      std::vector<MPI_Offset> start_surface = {  j_beg,i_beg};
      nc.write_all(dm.get<real const,3>("density_dry").as<float>(),"density_dry",start_3d);
      nc.write_all(dm.get<real const,3>("uvel"       ).as<float>(),"uvel"       ,start_3d);
      nc.write_all(dm.get<real const,3>("vvel"       ).as<float>(),"vvel"       ,start_3d);
      nc.write_all(dm.get<real const,3>("wvel"       ).as<float>(),"wvel"       ,start_3d);
      nc.write_all(dm.get<real const,3>("temp"       ).as<float>(),"temperature",start_3d);
      for (int i=0; i < tracer_names.size(); i++) {
        nc.write_all(dm.get<real const,3>(tracer_names.at(i)).as<float>(),tracer_names.at(i),start_3d);
      }
      for (int ivar = 0; ivar < output_vars.size(); ivar++) {
        auto name = output_vars.at(ivar).name;
        auto hash = output_vars.at(ivar).type_hash;
        auto dims = output_vars.at(ivar).dims;
        if        (dims == DIMS_COLUMN ) {
          nc.begin_indep_data();
          if (is_mainproc()) {
            if      (hash == get_type_hash<float >()) { nc.write(dm.get<float  const,1>(name),name); }
            else if (hash == get_type_hash<double>()) { nc.write(dm.get<double const,1>(name).as<float>(),name); }
            else if (hash == get_type_hash<int   >()) { nc.write(dm.get<int    const,1>(name),name); }
            else if (hash == get_type_hash<uchar >()) { nc.write(dm.get<uchar  const,1>(name),name); }
          }
          nc.end_indep_data();
        } else if (dims == DIMS_SURFACE) {
          if      (hash == get_type_hash<float >()) { nc.write_all(dm.get<float  const,2>(name),name,start_surface); }
          else if (hash == get_type_hash<double>()) { nc.write_all(dm.get<double const,2>(name).as<float>(),name,start_surface); }
          else if (hash == get_type_hash<int   >()) { nc.write_all(dm.get<int    const,2>(name),name,start_surface); }
          else if (hash == get_type_hash<uchar >()) { nc.write_all(dm.get<uchar  const,2>(name),name,start_surface); }
        } else if (dims == DIMS_3D     ) {
          if      (hash == get_type_hash<float >()) { nc.write_all(dm.get<float  const,3>(name),name,start_3d); }
          else if (hash == get_type_hash<double>()) { nc.write_all(dm.get<double const,3>(name).as<float>(),name,start_3d); }
          else if (hash == get_type_hash<int   >()) { nc.write_all(dm.get<int    const,3>(name),name,start_3d); }
          else if (hash == get_type_hash<uchar >()) { nc.write_all(dm.get<uchar  const,3>(name),name,start_3d); }
        }
      }
      for (int i=0; i < out_write_funcs.size(); i++) { out_write_funcs.at(i)(*this,nc); }
      nc.close();
      file_counter++;
      yakl::timer_stop("coupler_output");
      if (verbose && is_mainproc()) {
        std::cout << "*** Output/restart file written ***  -->  Etime , Output time: "
                  << std::scientific << std::setw(10) << etime            << " , " 
                  << std::scientific << std::setw(10) << timer_last("coupler_output") << std::endl;
      }
    }


    template<class T=real> MPI_Datatype get_mpi_data_type() const { return par_comm.get_type<T>(); }


    void overwrite_with_restart() {
      typedef unsigned char uchar;
      yakl::timer_start("overwrite_with_restart");
      if (is_mainproc())  std::cout << "*** Restarting from file: "
                                    << get_option<std::string>("restart_file") << std::endl;
      int i_beg = get_i_beg();
      int j_beg = get_j_beg();
      auto tracer_names = get_tracer_names();
      yakl::SimplePNetCDF nc(par_comm.get_mpi_comm());
      nc.open( get_option<std::string>("restart_file") , NC_NOWRITE );
      nc.begin_indep_data();
      real etime;
      if (is_mainproc()) nc.read( etime        , "etime"        );
      if (is_mainproc()) nc.read( file_counter , "file_counter" );
      nc.end_indep_data();
      par_comm.broadcast(file_counter);
      par_comm.broadcast(etime       );
      set_option<real>("elapsed_time",etime);
      std::vector<MPI_Offset> start_3d      = {0,j_beg,i_beg};
      std::vector<MPI_Offset> start_surface = {  j_beg,i_beg};
      std::vector<MPI_Offset> start_column  = {0            };
      nc.read_all(dm.get<real,3>("density_dry"),"density_dry",start_3d);
      nc.read_all(dm.get<real,3>("uvel"       ),"uvel"       ,start_3d);
      nc.read_all(dm.get<real,3>("vvel"       ),"vvel"       ,start_3d);
      nc.read_all(dm.get<real,3>("wvel"       ),"wvel"       ,start_3d);
      nc.read_all(dm.get<real,3>("temp"       ),"temperature",start_3d);
      for (int i=0; i < tracer_names.size(); i++) {
        if (nc.var_exists(tracer_names.at(i))) {
          nc.read_all(dm.get<real,3>(tracer_names.at(i)),tracer_names.at(i),start_3d);
        }
      }
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
      for (int i=0; i < restart_read_funcs.size(); i++) { restart_read_funcs.at(i)(*this,nc); }
      nc.close();
      file_counter++;
      yakl::timer_stop("overwrite_with_restart");
    }


    template <class T>
    MultiField<typename std::remove_cv<T>::type,3>
    create_and_exchange_halos( MultiField<T,3> const &fields_in , int hs ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      typedef typename std::remove_cv<T>::type T_NOCV;
      if (fields_in.get_num_fields() == 0) Kokkos::abort("ERROR: create_and_exchange_halos: create_halos input has zero fields");
      auto num_fields = fields_in.get_num_fields();
      auto nz         = fields_in.get_field(0).extent(0);
      auto ny         = fields_in.get_field(0).extent(1);
      auto nx         = fields_in.get_field(0).extent(2);
      MultiField<T_NOCV,3> fields_out;
      for (int i=0; i < num_fields; i++) {
        auto field = fields_in.get_field(i);
        if ( field.extent(0) != nz || field.extent(1) != ny || field.extent(2) != nx ) {
          Kokkos::abort("ERROR: create_and_exchange_halos: sizes not equal among fields");
        }
        yakl::Array<T_NOCV,3,yakl::memDevice,yakl::styleC> ret(field.label(),nz+2*hs,ny+2*hs,nx+2*hs);
        fields_out.add_field( ret );
      }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        fields_out(l,hs+k,hs+j,hs+i) = fields_in(l,k,j,i);
      });
      halo_exchange( fields_out , hs );
      return fields_out;
    }


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


    // Exchange halo values periodically in the horizontal
    template <class T>
    void halo_exchange( core::MultiField<T,3> & fields , int hs ) const {
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


    // Exchange halo values periodically in the horizontal
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


    // Exchange halo values periodically in the horizontal
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


    // Exchange halo values periodically in the horizontal
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


    // Exchange halo values periodically in the horizontal
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


    // Exchange halo values periodically in the horizontal
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


    // Exchange halo values periodically in the horizontal
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



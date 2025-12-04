
#pragma once

#include "main_header.h"
#include "coupler.h"

namespace modules {

  // This implements a turbine model based on the Fitch approach for situations where the turbine diameter
  //   is smaller than the grid spacing. The model applies forces to the flow field based on lookup tables of
  //   thrust and power coefficients as a function of wind speed and keeps track of power generation time traces.
  struct TurbineFitch {


    // This class holds information about a reference wind turbine, including lookup tables for various properties
    //   and turbine geometric properties
    struct RefTurbine {
      std::vector<real> velmag;        // Velocity magnitude at infinity (m/s)
      std::vector<real> thrust_coef;   // Thrust coefficient             (dimensionless)
      std::vector<real> power_coef;    // Power coefficient              (dimensionless)
      std::vector<real> power;         // Power generation               (MW)
      real              hub_height;    // Hub height                     (m)
      real              blade_radius;  // Blade radius                   (m)
      real1d            prop;          // Proportion of the turbine in each vertical level
      void init( core::Coupler const & coupler ) {
        YAML::Node config = YAML::LoadFile( coupler.get_option<std::string>("turbine_file") );
        if ( !config ) { endrun("ERROR: Invalid turbine input file"); }
        this->velmag       = config["velocity_magnitude"].as<std::vector<real>>();
        this->thrust_coef  = config["thrust_coef"       ].as<std::vector<real>>();
        this->power_coef   = config["power_coef"        ].as<std::vector<real>>();
        this->power        = config["power_megawatts"   ].as<std::vector<real>>();
        this->hub_height   = config["hub_height"        ].as<real>();
        this->blade_radius = config["blade_radius"      ].as<real>();
        auto nz   = coupler.get_nz();                    // Get the number of z-levels in the simulation
        auto zint = coupler.get_zint().createHostCopy(); // Get the z-interface heights as host array
        // pre-compute the proportion fo the turbine in each vertical cell
        realHost1d prop_h("prop",nz);
        int nsamp = 5; // Number of samples per cell for computing the vertical proportion
        for (int k=0; k < nz; k++) {
          prop_h(k) = 0;
          for (int kk=0; kk < nsamp; kk++) {
            real z  = zint(k)+(kk+0.5)*(zint(k+1)-zint(k))/nsamp; // Sample height within the cell
            real z0 = (z-this->hub_height)/this->blade_radius;    // Normalized height relative to hub height and blade radius
            if (std::abs(z0) < 1) prop_h(k) += std::sqrt(1-z0*z0); // 1 = y0^2+z0^2 (y0 length at this z0 location)
          }
        }
        using yakl::componentwise::operator/;  // Allow componentwise '/' for yakl::Arrays
        // Store the normalized vertical proportion array on the device in the RefTurbine
        this->prop = (prop_h/yakl::intrinsics::sum(prop_h)).createDeviceCopy();
      }
    };


    // This holds information about an individual turbine instance in the simulation (there may be multiple turbines)
    struct Turbine {
      bool               active;       // Whether this turbine affects this MPI task
      real               base_loc_x;   // x location of the tower base
      real               base_loc_y;   // y location of the tower base
      RefTurbine         ref_turbine;  // The reference turbine to use for this turbine
      std::vector<real>  power_trace;  // Time trace of power generation
      std::vector<real>  mag_trace;    // Time trace of inflow wind magnitude normal to turbine plane
    };


    // This holds a all turbines in the simulation
    struct TurbineGroup {
      std::vector<Turbine> turbines;
      void add_turbine( core::Coupler       & coupler     ,
                        real                  base_loc_x  ,
                        real                  base_loc_y  ,
                        RefTurbine    const & ref_turbine ) {
        auto dx    = coupler.get_dx();    // Grid spacing in the x-direction
        auto dy    = coupler.get_dy();    // Grid spacing in the y-direction
        auto nx    = coupler.get_nx();    // Local number of cells in the x-direction
        auto ny    = coupler.get_ny();    // Local number of cells in the y-direction
        auto nz    = coupler.get_nz();    // Number of cells in the z-direction
        auto i_beg = coupler.get_i_beg(); // Starting i-index for this MPI task
        auto j_beg = coupler.get_j_beg(); // Starting j-index for this MPI task
        // Determine the extents of this MPI task's domain
        real dom_x1 = (i_beg+0 )*dx;
        real dom_x2 = (i_beg+nx)*dx;
        real dom_y1 = (j_beg+0 )*dy;
        real dom_y2 = (j_beg+ny)*dy;
        // Determine if this turbine is active on this MPI task
        bool active = base_loc_x >= dom_x1 && base_loc_x < dom_x2 && base_loc_y >= dom_y1 && base_loc_y < dom_y2;
        // Create the turbine and add it to the list
        Turbine loc;
        loc.active      = active;
        loc.base_loc_x  = base_loc_x;
        loc.base_loc_y  = base_loc_y;
        loc.ref_turbine = ref_turbine;
        turbines.push_back(loc);
      }
    };


    TurbineGroup  turbine_group;  // All turbines in the simulation
    int           trace_size;     // Current size of the time traces


    // Initialize the turbine module by reading in turbine locations and reference turbine data
    void init( core::Coupler &coupler ) {
      RefTurbine ref_turbine;
      ref_turbine.init( coupler ); // Initialize the reference turbine data using turbine_file coupler option
      // Add turbines based on turbine_x_locs and turbine_y_locs coupler options
      if (coupler.option_exists("turbine_x_locs") && coupler.option_exists("turbine_y_locs")) {
        auto x_locs = coupler.get_option<std::vector<real>>("turbine_x_locs");
        auto y_locs = coupler.get_option<std::vector<real>>("turbine_y_locs");
        for (int iturb = 0; iturb < x_locs.size(); iturb++) {
          turbine_group.add_turbine( coupler , x_locs.at(iturb) , y_locs.at(iturb) , ref_turbine );
        }
      }
      trace_size = 0; // Initialize the trace size to zero
      // Register the output writing module to write out turbine traces at each output time
      coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        if (trace_size > 0) {
          nc.redef(); // Enter define mode to define new variables and dimensions
          nc.create_dim( "num_time_steps" , trace_size );
          // Create a variable for each turbine's power and wind magnitude traces
          for (int iturb=0; iturb < turbine_group.turbines.size(); iturb++) {
            nc.create_var<real>( std::string("power_trace_turb_")+std::to_string(iturb) , {"num_time_steps"} );
            nc.create_var<real>( std::string("mag_trace_turb_"  )+std::to_string(iturb) , {"num_time_steps"} );
          }
          nc.enddef(); // Leave define mode to write data
          nc.begin_indep_data(); // Begin independent data mode for writing from only owning MPI tasks one at a time
          for (int iturb=0; iturb < turbine_group.turbines.size(); iturb++) { // Loop through all turbines
            auto &turbine = turbine_group.turbines.at(iturb); // Grab reference to turbine for convenience
            if (turbine.active) {  // If I'm the owning task for this turbine, write out the traces
              realHost1d power_arr("power_arr",trace_size);
              realHost1d mag_arr  ("mag_arr"  ,trace_size);
              for (int i=0; i < trace_size; i++) { power_arr(i) = turbine.power_trace.at(i); }
              for (int i=0; i < trace_size; i++) { mag_arr  (i) = turbine.mag_trace  .at(i); }
              nc.write( power_arr , std::string("power_trace_turb_")+std::to_string(iturb) );
              nc.write( mag_arr   , std::string("mag_trace_turb_"  )+std::to_string(iturb) );
            }
            coupler.get_parallel_comm().barrier();
            turbine.power_trace.clear(); // Clear the traces after writing
            turbine.mag_trace  .clear();
          }
          nc.end_indep_data(); // End independent data mode for later operations on the file
        }
        trace_size = 0;  // Reset the trace size after writing
      });
    }


    // Apply thrust and power estimations from all turbines to the flow field
    void apply( core::Coupler & coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx    = coupler.get_nx   ();  // Local number of cells in the x-direction
      auto ny    = coupler.get_ny   ();  // Local number of cells in the y-direction
      auto nz    = coupler.get_nz   ();  // Number of cells in the z-direction
      auto dx    = coupler.get_dx   ();  // Grid spacing in the x-direction
      auto dy    = coupler.get_dy   ();  // Grid spacing in the y-direction
      auto dz    = coupler.get_dz   ();  // Grid spacing array in the z-direction (1-D array of size nz)
      auto i_beg = coupler.get_i_beg();  // Starting i-index for this MPI task
      auto j_beg = coupler.get_j_beg();  // Starting j-index for this MPI task
      auto &dm   = coupler.get_data_manager_readwrite();  // Get the DataManager for read/write access
      auto rho_d = dm.get<real const,3>("density_dry"  ); // Get the dry air density field
      auto uvel  = dm.get<real      ,3>("uvel"         ); // Get the u-velocity field
      auto vvel  = dm.get<real      ,3>("vvel"         ); // Get the v-velocity field
      auto tke   = dm.get<real      ,3>("TKE"          ); // Get the turbulent kinetic energy field

      // Allocate arrays to hold accumulated tendencies from all turbines, initialized to zero
      real3d tend_u  ("tend_u"  ,nz,ny,nx);
      real3d tend_v  ("tend_v"  ,nz,ny,nx);
      real3d tend_tke("tend_tke",nz,ny,nx);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        tend_u  (k,j,i) = 0;
        tend_v  (k,j,i) = 0;
        tend_tke(k,j,i) = 0;
      });

      for (int iturb = 0; iturb < turbine_group.turbines.size(); iturb++) { // Loop through all turbines
        auto &turbine = turbine_group.turbines.at(iturb); // Grab reference to turbine for convenience
        if (turbine.active) {  // If I'm the owning task for this turbine, add thrust to tendencies, compute power
          // Get reference data for later computations
          real rad             = turbine.ref_turbine.blade_radius; // Blade radius
          real base_x          = turbine.base_loc_x              ; // x location of the tower base
          real base_y          = turbine.base_loc_y              ; // y location of the tower base
          auto prop            = turbine.ref_turbine.prop        ; // Proportion of the turbine in each vertical level
          auto ref_velmag      = turbine.ref_turbine.velmag      ; // Reference velocity magnitude lookup table
          auto ref_thrust_coef = turbine.ref_turbine.thrust_coef ; // Reference thrust coefficient lookup table
          auto ref_power_coef  = turbine.ref_turbine.power_coef  ; // Reference power coefficient lookup table
          auto ref_power       = turbine.ref_turbine.power       ; // Reference power generation lookup table
          // Compute hte horizontal indices of the turbine within the local domain
          int  i               = std::min(nx-1,std::max(0,(int)std::floor(base_x/dx)-(int)i_beg));
          int  j               = std::min(ny-1,std::max(0,(int)std::floor(base_y/dy)-(int)j_beg));
          // Accumulate disk-averaged wind velocity components at the turbine plane
          yakl::ScalarLiveOut<real> u_d(0);
          yakl::ScalarLiveOut<real> v_d(0);
          parallel_for( YAKL_AUTO_LABEL() , nz , KOKKOS_LAMBDA (int k) {
            if (prop(k) > 0) {
              Kokkos::atomic_add( &(u_d()) , prop(k)*uvel(k,j,i) );
              Kokkos::atomic_add( &(v_d()) , prop(k)*vvel(k,j,i) );
            }
          });
          // Computation of disk properties
          real u0      = u_d.hostRead();     // Disk-averaged u-velocity at the turbine plane
          real v0      = v_d.hostRead();     // Disk-averaged v-velocity at the turbine plane
          real yaw     = std::atan2(v0,u0);  // Yaw angle of the inflow wind at the turbine plane
          real cos_yaw = std::cos(yaw);      // Cosine of the yaw angle
          real sin_yaw = std::sin(yaw);      // Sine of the yaw angle
          real mag0    = sqrt(u0*u0+v0*v0);  // Magnitude of the disk-averaged wind at the turbine plane
          real C_T     = std::min( 1.  , interp( ref_velmag , ref_thrust_coef , mag0 ) ); // Thrust coefficient
          real C_P     = std::min( C_T , interp( ref_velmag , ref_power_coef  , mag0 ) ); // Power coefficient
          real pwr     =                 interp( ref_velmag , ref_power       , mag0 );   // Power generation in MW
          real C_TKE   = coupler.get_option<real>("turbine_f_TKE",0.25) * (C_T - C_P);    // TKE coefficient
          // Application of disk onto tendencies
          parallel_for( YAKL_AUTO_LABEL() , nz , KOKKOS_LAMBDA (int k) {
            if (prop(k) > 0) {
              real wt = prop(k)*M_PI*rad*rad/(dx*dy*dz(k));
              // Compute tendencies implied by actuator disk thoery; Only apply TKE for disk, not blades
              real t_u    = -0.5f             *C_T  *mag0*mag0*cos_yaw*wt;
              real t_v    = -0.5f             *C_T  *mag0*mag0*sin_yaw*wt;
              real t_tke  =  0.5f*rho_d(k,j,i)*C_TKE*mag0*mag0*mag0   *wt;
              // Add to the total tendencies
              tend_u  (k,j,i) += t_u;
              tend_v  (k,j,i) += t_v;
              tend_tke(k,j,i) += t_tke;
            }
          });
          // Keep track of time traces of wind magnitude and power generation
          turbine.mag_trace  .push_back( mag0 );
          turbine.power_trace.push_back( pwr  );
        } // if (turbine.active)
      } // for (int iturb = 0; iturb < turbine_group.turbines.size(); iturb++)

      // Appoy accumulated tendencies to the flow field
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        uvel(k,j,i) += dt * tend_u  (k,j,i);
        vvel(k,j,i) += dt * tend_v  (k,j,i);
        tke (k,j,i) += dt * tend_tke(k,j,i);
      });

      trace_size++;
    }


    // Compute the disk-averaged wind velocity components at the turbine plane so that
    //   the dynamical core can perform pressure gradient forcing to specify inflow conditions
    void disk_average_wind( core::Coupler const & coupler     ,
                            RefTurbine    const & ref_turbine ,
                            real                & avg_u       ,
                            real                & avg_v       ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx      = coupler.get_nx();      // Local number of cells in the x-direction
      auto ny      = coupler.get_ny();      // Local number of cells in the y-direction
      auto nz      = coupler.get_nz();      // Number of cells in the z-direction
      auto nx_glob = coupler.get_nx_glob(); // Total global number of cells in the x-direction
      auto ny_glob = coupler.get_ny_glob(); // Total global number of cells in the y-direction
      auto &dm     = coupler.get_data_manager_readonly();  // Get the DataManager for read-only access
      auto uvel    = dm.get<real const,3>("uvel");  // Get the u-velocity field
      auto vvel    = dm.get<real const,3>("vvel");  // Get the v-velocity field
      auto prop    = ref_turbine.prop;              // Proportion of the turbine in each vertical level
      real2d udisk("udisk",ny,nx);  // Disk-averaged u-velocity at each horizontal location
      real2d vdisk("vdisk",ny,nx);  // Disk-averaged v-velocity at each horizontal location
      udisk = 0;
      vdisk = 0;
      real r_nx_ny = 1./(nx_glob*ny_glob);  // Reciprocal of total global number of horizontal cells
      // Local contribution to disk-averaged velocities
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        if (prop(k) > 0) {
          Kokkos::atomic_add( &udisk(j,i) , prop(k)*uvel(k,j,i)*r_nx_ny );
          Kokkos::atomic_add( &vdisk(j,i) , prop(k)*vvel(k,j,i)*r_nx_ny );
        }
      });
      // Reduce to get global disk-averaged velocities
      avg_u = coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(udisk),MPI_SUM);
      avg_v = coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(vdisk),MPI_SUM);
    }


    // Linear interpolation in a reference variable based on u_infinity and reference u_infinity
    real interp( std::vector<real> const &ref_umag , std::vector<real> const &ref_var , real umag ) {
      int imax = ref_umag.size()-1; // Max index for the table
      if ( umag < ref_umag.at(0) || umag > ref_umag.at(imax) ) return 0;
      int i = 0;
      while (umag > ref_umag.at(i)) { i++; }
      if (i > 0) i--;
      real fac = (ref_umag.at(i+1) - umag) / (ref_umag.at(i+1)-ref_umag.at(i));
      return fac*ref_var.at(i) + (1-fac)*ref_var.at(i+1);
    }

  };

}



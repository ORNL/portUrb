
#pragma once

#include "main_header.h"
#include "coupler.h"

namespace modules {

  // This module implements a turbine actuator disc model based on an actuator disc approach
  //   with a projected disk that actively yaws toward the wind direction and applies thrust
  //   and swirl forces to the flow field. The model also keeps time traces of power generation,
  //   yaw angle, and inflow wind speed for each turbine.
  struct TurbineActuatorDisc {


    // This class holds information about a reference wind turbine, including lookup tables for various properties
    //   and turbine geometric properties
    struct RefTurbine {
      std::vector<real> velmag;        // Velocity magnitude at infinity (m/s)
      std::vector<real> thrust_coef;   // Thrust coefficient             (dimensionless)
      std::vector<real> power_coef;    // Power coefficient              (dimensionless)
      std::vector<real> power;         // Power generation               (MW)
      std::vector<real> rotation;      // Rotation speed                 (radians / sec)
      real              hub_height;    // Hub height                     (m)
      real              blade_radius;  // Blade radius                   (m)
      real              max_yaw_speed; // Angular active yawing speed    (radians / sec)
      void init( std::string fname ) {
        YAML::Node config = YAML::LoadFile( fname );
        if ( !config ) { endrun("ERROR: Invalid turbine input file"); }
        this->velmag        = config["velocity_magnitude"].as<std::vector<real>>();
        this->thrust_coef   = config["thrust_coef"       ].as<std::vector<real>>();
        this->power_coef    = config["power_coef"        ].as<std::vector<real>>();
        this->power         = config["power_megawatts"   ].as<std::vector<real>>();
        this->rotation      = config["rotation_rpm"      ].as<std::vector<real>>(std::vector<real>());
        this->hub_height    = config["hub_height"        ].as<real>();
        this->blade_radius  = config["blade_radius"      ].as<real>();
        this->max_yaw_speed = config["max_yaw_speed"     ].as<real>(0.5)/180.*M_PI; // Convert from deg/sec to rad/sec
        for (int i=0; i < rotation.size(); i++) { rotation.at(i) *= 2*M_PI/60; }
      }
    };


    // This function computes the new yaw angle based on the current yaw angle, the wind direction,
    //   and a maximum yaw rate
    real yaw_tend( real uvel , real vvel , real dt , real yaw , real max_yaw_speed ) {
      real diff = std::atan2(vvel,uvel) - yaw;
      if (diff >  M_PI) diff -= 2*M_PI;
      if (diff < -M_PI) diff += 2*M_PI;
      real tend = diff / dt;
      if (tend > 0) { tend = std::min(  max_yaw_speed , tend ); }
      else          { tend = std::max( -max_yaw_speed , tend ); }
      return yaw+dt*tend;
    }


    // This holds information about an individual turbine in the simulation (there can be multiple turbines)
    struct Turbine {
      bool               active;         // Whether this turbine affects this MPI task
      real               base_loc_x;     // x location of the tower base
      real               base_loc_y;     // y location of the tower base
      real               yaw_angle;      // Current yaw angle (radians counter-clockwise from facing west)
      RefTurbine         ref_turbine;    // The reference turbine to use for this turbine
      core::ParallelComm par_comm;       // MPI communicator for this turbine
      std::vector<real>  power_trace;    // Time trace of power generation
      std::vector<real>  yaw_trace;      // Time trace of yaw angle
      std::vector<real>  mag_trace;      // Time trace of inflow wind magnitude normal to turbine plane
    };


    // This holds information about all turbines in the simulation
    struct TurbineGroup {
      std::vector<Turbine> turbines;  // All turbines in the simulation
      // This routine adds a turbine to the group based on its base location and reference turbine data
      // The coupler is needed in order to determine whether the turbine is active on this MPI task
      void add_turbine( core::Coupler       & coupler     ,
                        real                  base_loc_x  ,
                        real                  base_loc_y  ,
                        RefTurbine    const & ref_turbine ) {
        using yakl::c::parallel_for;
        using yakl::c::SimpleBounds;
        auto i_beg  = coupler.get_i_beg();  // Get the beginning x-direction index for this MPI task
        auto j_beg  = coupler.get_j_beg();  // Get the beginning y-direction index for this MPI task
        auto nx     = coupler.get_nx();     // Get the number of x-direction cells
        auto ny     = coupler.get_ny();     // Get the number of y-direction cells
        auto dx     = coupler.get_dx();     // Get the grid spacing in the x-direction
        auto dy     = coupler.get_dy();     // Get the grid spacing in the y-direction
        // bounds of this MPI task's domain
        real dom_x1  = (i_beg+0 )*dx;
        real dom_x2  = (i_beg+nx)*dx;
        real dom_y1  = (j_beg+0 )*dy;
        real dom_y2  = (j_beg+ny)*dy;
        // Rectangular bounds of this turbine's potential influence (3 turbine diameters in each direction from tower base)
        real turb_x1 = base_loc_x-6*ref_turbine.blade_radius;
        real turb_x2 = base_loc_x+6*ref_turbine.blade_radius;
        real turb_y1 = base_loc_y-6*ref_turbine.blade_radius;
        real turb_y2 = base_loc_y+6*ref_turbine.blade_radius;
        bool active = !( turb_x1 > dom_x2 || // Turbine's to the right
                         turb_x2 < dom_x1 || // Turbine's to the left
                         turb_y1 > dom_y2 || // Turbine's above
                         turb_y2 < dom_y1 ); // Turbine's below
        // Create a local turbine object, assign its properties, and add it to the list
        Turbine loc;
        loc.active      = active;
        loc.base_loc_x  = base_loc_x;
        loc.base_loc_y  = base_loc_y;
        loc.yaw_angle   = coupler.get_option<real>("turbine_initial_yaw",0);
        loc.ref_turbine = ref_turbine;
        // The line below calls MPI_Comm_split to create a new sub-communicator and group for MPI tasks
        //   involved with this turbine
        loc.par_comm.create( active , coupler.get_parallel_comm().get_mpi_comm() );
        turbines.push_back(loc);
      }
    };


    TurbineGroup  turbine_group;  // Holds all turbines in the simulation
    int           trace_size;     // Number of time steps recorded in the turbine traces so far
                                  // This is reset to zero after writing output each time


    // Initialize the turbine actuator disc module, adding all the specified turbines from the coupler options
    void init( core::Coupler &coupler ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx    = coupler.get_nx();    // Get the local number of x-direction cells
      auto ny    = coupler.get_ny();    // Get the local number of y-direction cells
      auto nz    = coupler.get_nz();    // Get the number of z-direction cells
      auto dx    = coupler.get_dx();    // Get the grid spacing in the x-direction
      auto dy    = coupler.get_dy();    // Get the grid spacing in the y-direction
      auto i_beg = coupler.get_i_beg(); // Get the beginning x-direction index for this MPI task
      auto j_beg = coupler.get_j_beg(); // Get the beginning y-direction index for this MPI task
      RefTurbine ref_turbine;  // Create a reference turbine and initialize it from file
      ref_turbine.init( coupler.get_option<std::string>("turbine_file") );
      // Add turbines based on the specified x and y locations from coupler options
      if (coupler.option_exists("turbine_x_locs") && coupler.option_exists("turbine_y_locs")) {
        auto x_locs = coupler.get_option<std::vector<real>>("turbine_x_locs");
        auto y_locs = coupler.get_option<std::vector<real>>("turbine_y_locs");
        for (int iturb = 0; iturb < x_locs.size(); iturb++) {
          turbine_group.add_turbine( coupler , x_locs.at(iturb) , y_locs.at(iturb) , ref_turbine );
        }
      }
      trace_size = 0;  // Initialize trace size to zero
      // Register output writing function to write turbine traces
      coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        if (trace_size > 0) {
          nc.redef();  // Enter define mode in order to add new variables and dimensions
          nc.create_dim( "num_time_steps" , trace_size );  // Create dimension for time steps
          // Add a trace variable for each turbine for power, yaw, and inflow wind speed
          for (int iturb=0; iturb < turbine_group.turbines.size(); iturb++) {
            nc.create_var<real>( std::string("power_trace_turb_")+std::to_string(iturb) , {"num_time_steps"} );
            nc.create_var<real>( std::string("yaw_trace_turb_"  )+std::to_string(iturb) , {"num_time_steps"} );
            nc.create_var<real>( std::string("mag_trace_turb_"  )+std::to_string(iturb) , {"num_time_steps"} );
          }
          nc.enddef();  // Exit define mode to write data
          nc.begin_indep_data();  // Begin independent data section for writing from only the owning MPI task
          for (int iturb=0; iturb < turbine_group.turbines.size(); iturb++) {  // Loop over all turbines
            auto &turbine = turbine_group.turbines.at(iturb); // Grab a reference to this turbine for convenience
            // Only write this data from the MPI task that contains the base location
            if (turbine.active && turbine.base_loc_x >= i_beg*dx && turbine.base_loc_x < (i_beg+nx)*dx &&
                                  turbine.base_loc_y >= j_beg*dy && turbine.base_loc_y < (j_beg+ny)*dy ) {
              realHost1d power_arr("power_arr",trace_size);  // Array to hold power trace
              realHost1d yaw_arr  ("yaw_arr"  ,trace_size);  // Array to hold yaw trace
              realHost1d mag_arr  ("mag_arr"  ,trace_size);  // Array to hold inflow wind magnitude trace
              // Load data from std:;vector to yakl::Array for writing
              for (int i=0; i < trace_size; i++) { power_arr(i) = turbine.power_trace.at(i);          }
              for (int i=0; i < trace_size; i++) { yaw_arr  (i) = turbine.yaw_trace  .at(i)/M_PI*180; }
              for (int i=0; i < trace_size; i++) { mag_arr  (i) = turbine.mag_trace  .at(i); }
              // Write the data arrays to file
              nc.write( power_arr , std::string("power_trace_turb_")+std::to_string(iturb) );
              nc.write( yaw_arr   , std::string("yaw_trace_turb_"  )+std::to_string(iturb) );
              nc.write( mag_arr   , std::string("mag_trace_turb_"  )+std::to_string(iturb) );
            }
            coupler.get_parallel_comm().barrier();
            // Clear the trace data after writing
            turbine.power_trace.clear();
            turbine.yaw_trace  .clear();
            turbine.mag_trace  .clear();
          }
          nc.end_indep_data();  // End independent data section
        }
        trace_size = 0;  // Reset trace size to zero after writing
      });
    }


    // Apply the turbine actuator disc forces and yaw updates for all turbines, accumulating tendencies from
    //   thrust and torque forces. Keep traces of the power, yaw angle, and inflow wind speed normal to the turbine plane.
    // Injects a portion of the unused thrust energy back into the flow as SGS/unresolved TKE.
    void apply( core::Coupler & coupler , float dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx    = coupler.get_nx   ();  // Get the local number of x-direction cells
      auto ny    = coupler.get_ny   ();  // Get the local number of y-direction cells
      auto nz    = coupler.get_nz   ();  // Get the number of z-direction cells
      auto dx    = coupler.get_dx   ();  // Get the grid spacing in the x-direction
      auto dy    = coupler.get_dy   ();  // Get the grid spacing in the y-direction
      auto dz    = coupler.get_dz   ();  // Get the grid spacing in the z-direction (1-D array of size nz)
      auto zint  = coupler.get_zint ();  // Get the z-interface heights (1-D array of size nz+1)
      auto zmid  = coupler.get_zmid ();  // Get the z-midpoint heights (1-D array of size nz)
      auto i_beg = coupler.get_i_beg();  // Get the beginning x-direction index for this MPI task
      auto j_beg = coupler.get_j_beg();  // Get the beginning y-direction index for this MPI task
      auto &dm   = coupler.get_data_manager_readwrite();  // Get the data manager for read/write access
      auto rho_d = dm.get<real const,3>("density_dry"  );  // Dry air density
      auto uvel  = dm.get<real      ,3>("uvel"         );  // u-velocity
      auto vvel  = dm.get<real      ,3>("vvel"         );  // v-velocity
      auto wvel  = dm.get<real      ,3>("wvel"         );  // w-velocity
      auto tke   = dm.get<real      ,3>("TKE"          );  // TKE

      // Allocate tendency arrays for u, v, w, and TKE and initialize to zero
      float3d tend_u  ("tend_u"  ,nz,ny,nx);
      float3d tend_v  ("tend_v"  ,nz,ny,nx);
      float3d tend_w  ("tend_w"  ,nz,ny,nx);
      float3d tend_tke("tend_tke",nz,ny,nx);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        tend_u  (k,j,i) = 0;
        tend_v  (k,j,i) = 0;
        tend_w  (k,j,i) = 0;
        tend_tke(k,j,i) = 0;
      });

      for (int iturb = 0; iturb < turbine_group.turbines.size(); iturb++) {  // Loop over all turbines
        auto &turbine = turbine_group.turbines.at(iturb);  // Grab a reference to this turbine for convenience
        if (turbine.active) {  // If this MPI task is potentially involved with the turbine
          // Pre-compute rotation matrix terms
          float cos_yaw = std::cos(turbine.yaw_angle);
          float sin_yaw = std::sin(turbine.yaw_angle);
          // Get reference data for later computations
          float rad             = turbine.ref_turbine.blade_radius ;  // Blade radius
          float hub_height      = turbine.ref_turbine.hub_height   ;  // Hub height
          float base_x          = turbine.base_loc_x               ;  // Tower base x-location
          float base_y          = turbine.base_loc_y               ;  // Tower base y-location
          auto  ref_velmag      = turbine.ref_turbine.velmag       ;  // Reference velocity magnitudes
          auto  ref_thrust_coef = turbine.ref_turbine.thrust_coef  ;  // Reference thrust coefficients
          auto  ref_power_coef  = turbine.ref_turbine.power_coef   ;  // Reference power coefficients
          auto  ref_power       = turbine.ref_turbine.power        ;  // Reference power generation values
          auto  ref_rotation    = turbine.ref_turbine.rotation     ;  // Reference rotation rates

          // Zero out disk weights for projection and sampling
          // Compute average winds in a 3-D tet around the turbine hub to compute upstream direction
          float3d disk_weight_angle("disk_weight_angle",nz,ny,nx);  // Weights for computing wind angle about the hub
                                                                    // The angle is used later for torque calculation
          float3d disk_weight_proj ("disk_weight_proj" ,nz,ny,nx);  // Weights for thrust projection disk
          float3d disk_weight_samp ("disk_weight_samp" ,nz,ny,nx);  // Weights for inflow sampling disk
          float3d uvel_3d          ("uvel_3d"          ,nz,ny,nx);  // For determining upstream direction
          float3d vvel_3d          ("vvel_3d"          ,nz,ny,nx);  // For determining upstream direction
          // Local MPI task's contribution to upstream wind direction
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
            // Initialize weights to zero
            disk_weight_angle(k,j,i) = 0;
            disk_weight_proj (k,j,i) = 0;
            disk_weight_samp (k,j,i) = 0;
            // Compute midpoint location for this cell
            float x = (i_beg+i+0.5f)*dx;
            float y = (j_beg+j+0.5f)*dy;
            float z = zmid(k);
            // Store the u and v velocities within the rotor disk for upstream direction calculation
            if ( z >= hub_height-rad && z <= hub_height+rad &&
                 y >= base_y    -rad && y <= base_y    +rad &&
                 x >= base_x    -rad && x <= base_x    +rad ) {
              uvel_3d(k,j,i) = uvel(k,j,i);
              vvel_3d(k,j,i) = vvel(k,j,i);
            } else {
              uvel_3d(k,j,i) = 0;
              vvel_3d(k,j,i) = 0;
            }
          });
          // Aggregate the global upstream wind velocities at the turbine from all MPI tasks
          yakl::SArray<float,1,2> weights_tot;
          weights_tot(0) = yakl::intrinsics::sum(uvel_3d);
          weights_tot(1) = yakl::intrinsics::sum(vvel_3d);
          weights_tot = turbine.par_comm.all_reduce( weights_tot , MPI_SUM , "windmill_Allreduce1" );
          // Determine the upstream wind direction for the turbine
          float up_uvel = weights_tot(0);
          float up_vvel = weights_tot(1);
          float up_dir  = coupler.get_option<real>("turbine_upstream_dir",std::atan2( up_vvel , up_uvel ));
          // Compute upstream offset at two turbine diameters upstream based on wind direction at the turbine
          float up_x_offset = -4*rad*std::cos(up_dir);
          float up_y_offset = -4*rad*std::sin(up_dir);
          {
            //////// PROJECT SAMPLING AND THRUST PROJECTION DISKS
            // Reference space is centered about the origin with the turbine disk facing toward the west
            float decay = 2*dx/rad; // Length of decay of thrust after the end of the blade radius (relative)
            float xr    = std::max(5.,5*dx);  // Thickness of the disk in the x-direction in reference space
            int num_x   = std::ceil(20/dx*xr             *2); // # cells to sample over in x-direction
            int num_y   = std::ceil(20/dx*rad*(1+decay/2)*2); // # cells to sample over in y-direction
            int num_z   = std::ceil(20/dx*rad*(1+decay/2)*2); // # cells to sample over in z-direction
            // This function computes the thrust shaping function in reference space
            auto thrust_shape = KOKKOS_LAMBDA (float x, float x2, float x3, float a) -> float {
              if (x < x2) return std::pow(-1.0*((x*x)-2*x*x2)/(x2*x2),a);
              if (x < x3) return -1.0*(2*(x*x*x)-3*(x*x)*x2-3*x2*(x3*x3)+(x3*x3*x3)-3*((x*x)-2*x*x2)*x3)/
                                      ((x2*x2*x2)-3*(x2*x2)*x3+3*x2*(x3*x3)-(x3*x3*x3));
              return 0;
            };
            // This function defines the 1-D projection shaping function in the direction normal to the disk
            auto proj_shape_1d = KOKKOS_LAMBDA ( float x , float xr ) -> float {
              float term = 1-(x/xr)*(x/xr);
              return term <= 0 ? 0 : term*term;
            };
            // Compute the local MPI task's contribution to the disk projection and sampling weights
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_z,num_y,num_x) , KOKKOS_LAMBDA (int k, int j, int i) {
              // Initial point in the y-z plane facing the negative x direction
              float x = -xr              + (2*xr             *i)/(num_x-1); // Compute reference x location
              float y = -rad*(1+decay/2) + (2*rad*(1+decay/2)*j)/(num_y-1); // Compute reference y location
              float z = -rad*(1+decay/2) + (2*rad*(1+decay/2)*k)/(num_z-1); // Compute reference z location
              float rloc = std::sqrt(y*y+z*z);  // radius of the point about the origin / hub center
              if (rloc <= rad*(1+decay/2)) {  // if within the disk + decay region
                // Compute the 3-D shaping function for this point in reference space
                float shp = thrust_shape(rloc/rad,1-decay/2,1+decay/2,0.5)*proj_shape_1d(x,xr);
                // Rotate about z-axis for yaw angle, and translate to base location
                float xp = base_x     + cos_yaw*x - sin_yaw*y;
                float yp = base_y     + sin_yaw*x + cos_yaw*y;
                float zp = hub_height + z;
                // If it's in this task's domain, atomically add to the cell's total shape function sum for projection
                //     Also, add the average angle for torque application later
                int ti = static_cast<int>(std::floor(xp/dx))-i_beg; // Compute global i-index this sampling point falls into
                int tj = static_cast<int>(std::floor(yp/dy))-j_beg; // Compute global j-index this sampling point falls into
                // Locate the vertical index (k) this sampling point falls into
                int tk = 0;
                for (int kk=0; kk < nz; kk++) {
                  if (zp >= zint(kk) && zp < zint(kk+1)) {
                    tk = kk;
                    break;
                  }
                }
                // If the sampling point is in this MPI task's domain, accumulate its contribution in the correct cell
                // This must be done with atomics because multiple threads may write to the same cell at the same time
                if ( ti >= 0 && ti < nx && tj >= 0 && tj < ny && tk >= 0 && tk < nz) {
                  Kokkos::atomic_add( &disk_weight_angle(tk,tj,ti) , shp*std::atan2(z,-y) );
                  Kokkos::atomic_add( &disk_weight_proj (tk,tj,ti) , shp );
                }
                // Now do the same thing for the upwind sampling disk for computing inflow velocity
                xp += up_x_offset;
                yp += up_y_offset;
                ti = static_cast<int>(std::floor(xp/dx))-i_beg;
                tj = static_cast<int>(std::floor(yp/dy))-j_beg;
                // tk is the same because only the horizontal location is translated
                if ( ti >= 0 && ti < nx && tj >= 0 && tj < ny && tk >= 0 && tk < nz) {
                  Kokkos::atomic_add( &disk_weight_samp(tk,tj,ti) , shp );
                }
              }
            });
          }
          // Reduce projection sums for normalization
          yakl::SArray<float,1,2> weights_tot2;
          weights_tot2(0) = yakl::intrinsics::sum(disk_weight_proj);
          weights_tot2(1) = yakl::intrinsics::sum(disk_weight_samp);
          weights_tot2 = turbine.par_comm.all_reduce( weights_tot2 , MPI_SUM , "windmill_Allreduce1" );
          float disk_proj_tot = weights_tot2(0);
          float disk_samp_tot = weights_tot2(1);
          // Compute average angle in each cell by dividing by prjection weights
          // Normalize thrust and sampling disk weights so that they sum to one
          // Aggregate weighted wind velocities in upstream sampling region to compute inflow
          float3d samp_u("samp_u",nz,ny,nx);  // u-velocity sampled at the upstream disk
          float3d samp_v("samp_v",nz,ny,nx);  // v-velocity sampled at the upstream disk
          // Compute local MPI task's contribution to normalized weights and sampled velocities
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
            if (disk_weight_proj(k,j,i) > 1.e-10) {
              disk_weight_angle(k,j,i) /= disk_weight_proj(k,j,i);
              disk_weight_proj (k,j,i) /= disk_proj_tot;
            }
            if (disk_weight_samp(k,j,i) > 0) {
              disk_weight_samp(k,j,i) /= disk_samp_tot;
              samp_u          (k,j,i)  = disk_weight_samp(k,j,i)*uvel(k,j,i);
              samp_v          (k,j,i)  = disk_weight_samp(k,j,i)*vvel(k,j,i);
            } else {
              samp_u          (k,j,i) = 0;
              samp_v          (k,j,i) = 0;
            }
          });
          // Reduce weighted sums of sampled upstream inflow u and v velocities
          SArray<float,1,2> sums;
          sums(0) = yakl::intrinsics::sum( samp_u );
          sums(1) = yakl::intrinsics::sum( samp_v );
          sums = turbine.par_comm.all_reduce( sums , MPI_SUM , "windmill_Allreduce2" );
          // Compute horizontal wind magnitude normal to the disk
          float mag0 = std::max( 0.f , sums(0)*cos_yaw + sums(1)*sin_yaw );
          // Computation of disk properties
          float C_T       = std::min( 1.f , interp( ref_velmag , ref_thrust_coef , mag0 ) );  // Thrust coefficient
          float C_P       = std::min( C_T , interp( ref_velmag , ref_power_coef  , mag0 ) );  // Power coefficient
          float pwr       =                 interp( ref_velmag , ref_power       , mag0 );    // Power generation (MW)
          float rot_speed = ref_rotation.size() > 0 ? interp( ref_velmag , ref_rotation , mag0 ) : 0; // Rotation rate (rad/s)
                rot_speed = coupler.get_option<real>("turbine_rot_fixed",rot_speed);          // Optionally override with fixed rotation rate
          float C_Q       = rot_speed == 0 ? 0 : C_P * mag0 / (rot_speed * rad);              // Torque coefficient
          float C_TKE     = coupler.get_option<real>("turbine_f_TKE",0.25) * (C_T - C_P);     // TKE injection coefficient
          // Application of disk onto u, v, w, and TKE tendencies
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
            if (disk_weight_proj(k,j,i) > 0) {
              // If normal wind is negative, don't do anything.
              if (uvel(k,j,i)*cos_yaw + vvel(k,j,i)*sin_yaw > 0) {
                float wt = disk_weight_proj(k,j,i)*M_PI*rad*rad/(dx*dy*dz(k));
                float az = disk_weight_angle(k,j,i);  // Angle for swirl calculation
                // Compute tendencies implied by actuator disk thoery; Only apply TKE for disk, not blades
                float t_u    = -0.5f             *C_T  *mag0*mag0*cos_yaw*wt;
                float t_v    = -0.5f             *C_T  *mag0*mag0*sin_yaw*wt;
                float t_tke  =  0.5f*rho_d(k,j,i)*C_TKE*mag0*mag0*mag0   *wt;
                // Compute tendencies for swirl
                float t_w    =  0.5f             *C_Q  *mag0*mag0        *std::cos(az)*wt;
                      t_u   += -0.5f             *C_Q  *mag0*mag0*sin_yaw*std::sin(az)*wt;
                      t_v   +=  0.5f             *C_Q  *mag0*mag0*cos_yaw*std::sin(az)*wt;
                // Accumulate to total tendencies
                tend_u  (k,j,i) += t_u;
                tend_v  (k,j,i) += t_v;
                tend_w  (k,j,i) += t_w;
                tend_tke(k,j,i) += t_tke;
              }
            }
          });
          // Record traces and update yaw angle
          turbine.yaw_trace  .push_back( turbine.yaw_angle );
          turbine.mag_trace  .push_back( mag0              );
          turbine.power_trace.push_back( pwr               );
          if (! coupler.get_option<bool>("turbine_fixed_yaw",false)) {
            turbine.yaw_angle = yaw_tend(up_uvel,up_vvel,dt,turbine.yaw_angle,turbine.ref_turbine.max_yaw_speed);
          }
        } // if (turbine.active)
      } // for (int iturb = 0; iturb < turbine_group.turbines.size(); iturb++)

      // Apply accumulated tendencies to the flow field
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        uvel(k,j,i) += dt * tend_u  (k,j,i);
        vvel(k,j,i) += dt * tend_v  (k,j,i);
        wvel(k,j,i) += dt * tend_w  (k,j,i);
        tke (k,j,i) += dt * tend_tke(k,j,i);
      });

      trace_size++;
    }


    // Compute the average wind velocity through the turbine disk in order to force
    //   the time-averaged inflow velocity to a specified value with pressure-gradient forcing
    void disk_average_wind( core::Coupler const & coupler     ,
                            RefTurbine    const & ref_turbine ,
                            real                & avg_u       ,
                            real                & avg_v       ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto  nx         = coupler.get_nx  ();     // Get the local number of x-direction cells
      auto  ny         = coupler.get_ny  ();     // Get the local number of y-direction cells
      auto  nz         = coupler.get_nz  ();     // Get the number of z-direction cells
      auto  dx         = coupler.get_dx  ();     // Get the grid spacing in the x-direction
      auto  nx_glob    = coupler.get_nx_glob();  // Get the global number of x-direction cells
      auto  ny_glob    = coupler.get_ny_glob();  // Get the global number of y-direction cells
      auto  zint       = coupler.get_zint().createHostCopy(); // Get the z-interface heights as host array
      auto  &dm        = coupler.get_data_manager_readonly(); // Get the data manager for read-only access
      auto  uvel       = dm.get<real const,3>("uvel"); // u-velocity
      auto  vvel       = dm.get<real const,3>("vvel"); // v-velocity
      auto  rad        = ref_turbine.blade_radius;     // Blade radius
      auto  hub_height = ref_turbine.hub_height  ;     // Hub height
      auto  decay      = 2*dx/rad; // Length of decay of thrust after the end of the blade radius (relative)
      int   num_z      = std::ceil(20/dx*rad*(1+decay/2)*2); // # cells to sample over in z-direction
      // This function computes the thrust shaping function in reference space
      auto thrust_shape = [&] (float x, float x2, float x3, float a) -> float {
        if (x < x2) return std::pow(-1.0*((x*x)-2*x*x2)/(x2*x2),a);
        if (x < x3) return -1.0*(2*(x*x*x)-3*(x*x)*x2-3*x2*(x3*x3)+(x3*x3*x3)-3*((x*x)-2*x*x2)*x3)/
                                ((x2*x2*x2)-3*(x2*x2)*x3+3*x2*(x3*x3)-(x3*x3*x3));
        return 0;
      };
      // Compute the vertical shape function for the disk averaging
      realHost1d shp_host("shp",nz);
      shp_host = 0;
      for (int k = 0; k < num_z; k++) { // Loop over sampling points in z-direction
        float z = -rad*(1+decay/2) + (2*rad*(1+decay/2)*k)/(num_z-1); // Compute reference z location
        float rloc = std::abs(z);
        if (rloc <= rad*(1+decay/2)) {  // if within the disk + decay region
          float shp_loc = thrust_shape(rloc/rad,1-decay/2,1+decay/2,0.5); // Compute the 1-D shaping function
          // Translate to hub height and accumulate to the correct vertical cell
          float zp = hub_height + z;
          int tk = 0;
          for (int kk=0; kk < nz; kk++) {
            if (zp >= zint(kk) && zp < zint(kk+1)) {
              tk = kk;
              break;
            }
          }
          // Accumulate to the correct cell
          if ( tk >= 0 && tk < nz) shp_host(tk) += shp_loc;
        }
      }
      using yakl::componentwise::operator/;  // Allow componentwise '/' for yakl::Arrays
      auto shp = (shp_host / yakl::intrinsics::sum(shp_host)).createDeviceCopy(); // Normalize and copy to device
      real2d udisk("udisk",ny,nx);  // Hold disk-averaged u-velocity
      real2d vdisk("vdisk",ny,nx);  // Hold disk-averaged v-velocity
      udisk = 0;
      vdisk = 0;
      real r_nx_ny = 1./(nx_glob*ny_glob);  // Reciprocal of total number of horizontal cells
      // Compute local MPI task's contribution to disk-averaged velocities
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        if (shp(k) > 0) {
          Kokkos::atomic_add( &udisk(j,i) , shp(k)*uvel(k,j,i)*r_nx_ny );
          Kokkos::atomic_add( &vdisk(j,i) , shp(k)*vvel(k,j,i)*r_nx_ny );
        }
      });
      // Reduce to get global disk-averaged velocities
      avg_u = coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(udisk),MPI_SUM);
      avg_v = coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(vdisk),MPI_SUM);
    }


    // Linear interpolation in a reference variable based on u_infinity and reference u_infinity
    float interp( std::vector<real> const &ref_umag , std::vector<real> const &ref_var , real umag ) {
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



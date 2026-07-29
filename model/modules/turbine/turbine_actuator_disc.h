
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
      void init( std::string fname );
    };


    // This function computes the new yaw angle based on the current yaw angle, the wind direction,
    //   and a maximum yaw rate
    real yaw_tend( real uvel , real vvel , real dt , real yaw , real max_yaw_speed );


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
                        RefTurbine    const & ref_turbine );
    };


    TurbineGroup  turbine_group;  // Holds all turbines in the simulation
    int           trace_size;     // Number of time steps recorded in the turbine traces so far
                                  // This is reset to zero after writing output each time


    // Initialize the turbine actuator disc module, adding all the specified turbines from the coupler options
    void init( core::Coupler &coupler );


    // Apply the turbine actuator disc forces and yaw updates for all turbines, accumulating tendencies from
    //   thrust and torque forces. Keep traces of the power, yaw angle, and inflow wind speed normal to the turbine plane.
    // Injects a portion of the unused thrust energy back into the flow as SGS/unresolved TKE.
    void apply( core::Coupler & coupler , float dt );


    // Compute the average wind velocity through the turbine disk in order to force
    //   the time-averaged inflow velocity to a specified value with pressure-gradient forcing
    void disk_average_wind( core::Coupler const & coupler     ,
                            RefTurbine    const & ref_turbine ,
                            real                & avg_u       ,
                            real                & avg_v       );


    // Linear interpolation in a reference variable based on u_infinity and reference u_infinity
    float interp( std::vector<real> const &ref_umag , std::vector<real> const &ref_var , real umag );

  };

}



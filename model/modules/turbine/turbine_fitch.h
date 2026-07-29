
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
      void init( core::Coupler const & coupler );
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
                        RefTurbine    const & ref_turbine );
    };


    TurbineGroup  turbine_group;  // All turbines in the simulation
    int           trace_size;     // Current size of the time traces


    // Initialize the turbine module by reading in turbine locations and reference turbine data
    void init( core::Coupler &coupler );


    // Apply thrust and power estimations from all turbines to the flow field
    void apply( core::Coupler & coupler , real dt );


    // Compute the disk-averaged wind velocity components at the turbine plane so that
    //   the dynamical core can perform pressure gradient forcing to specify inflow conditions
    void disk_average_wind( core::Coupler const & coupler     ,
                            RefTurbine    const & ref_turbine ,
                            real                & avg_u       ,
                            real                & avg_v       );


    // Linear interpolation in a reference variable based on u_infinity and reference u_infinity
    real interp( std::vector<real> const &ref_umag , std::vector<real> const &ref_var , real umag );

  };

}



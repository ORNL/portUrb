
#include "coupler.h"
#include "dynamics_rk_simpler.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "turbine_actuator_line.h"
#include "edge_sponge.h"
#include "column_nudging.h"

// Research on Aerodynamic Characteristics of Three Offshore Wind Turbines Based on Large Eddy Simulation and Actuator Line Model
// Analyzing scaling effects on offshore wind turbines using CFD
// https://www.mdpi.com/2077-1312/12/8/1341
// Compare C_T and C_P over a range of TSR
// Compare wakes at x/D=3,5,4
// NREL 5MW
// U_inf = 11.4
// omega_rpm = 12.1 and variable
// coupler.set_option<bool       >( "turbine_immerse_material" , false        );
// coupler.set_option<real       >( "turbine_pitch_fixed"      , 0.           );
// coupler.set_option<real       >( "turbine_eps_fixed"        , 3.5          );
// coupler.set_option<real       >( "dycore_max_wind"          , 30           );
// coupler.set_option<real       >( "dycore_cs"                , 100          );

// Accuracy of State-of-the-Art Actuator-Line Modeling for Wind Turbine Wakes
// Compare spanwise angle of attack, normal force coefficient, and tangential force coefficient
// NREL 5MW
// U_inf = 8
// omega_rpm = 9.156
// coupler.set_option<bool       >( "turbine_immerse_material" , false        );
// coupler.set_option<real       >( "turbine_pitch_fixed"      , 0.           );
// coupler.set_option<real       >( "turbine_eps_fixed"        , 3.9375       );

// A Comparison of Actuator Disk and Actuator Line Wind Turbine Models and Best Practices for Their Use
// To compare spanwise angle of attack, spanwise axial velocity, wake shape at x/D=1,4, mean hub velocity contours, vorticity contours, and power production, use epsilon=4.2m
// NREL 5MW
// U_inf = 8
// omega_rpm = 9.1552
// coupler.set_option<bool       >( "turbine_immerse_material" , false        );
// coupler.set_option<real       >( "turbine_pitch_fixed"      , 0.           );
// coupler.set_option<real       >( "turbine_eps_fixed"        , 4.2          );

// Study on Actuator Line Modeling of Two NREL 5-MW Wind Turbine Wakes
// This shows periodic time variability in power production and thrust
// Not clear yet what rotations rates are used at different inflow speeds.
// Not clear yet what epsilon values are used
// NREL 5MW
// U_inf = variable
// omega_rpm = variable
// coupler.set_option<bool       >( "turbine_immerse_material" , false        );

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler;

    real U_inf     = 11.4;
    real tsr       = 8;
    real omega_rpm = tsr*U_inf/63./(2.*M_PI)*60.;

    real dx = 1;
    coupler.set_option<bool>("turbine_orig_C_T",true);

    std::string turbine_file = "./inputs/NREL_5MW_126_RWT.yaml";
    YAML::Node config = YAML::LoadFile( turbine_file );
    if ( !config ) { endrun("ERROR: Invalid turbine input file"); }
    real D = config["blade_radius"].as<real>()*2;

    real        sim_time     = 1000.1;
    real        xlen         = D*10;
    real        ylen         = D*4;
    real        zlen         = D*4;
    int         nx_glob      = std::ceil(xlen/dx);    xlen = nx_glob * dx;
    int         ny_glob      = std::ceil(ylen/dx);    ylen = ny_glob * dx;
    int         nz           = std::ceil(zlen/dx);    zlen = nz      * dx;
    real        dtphys_in    = 0;
    std::string init_data    = "constant";
    real        out_freq     = 30.; // /omega_rpm*10;
    real        inform_freq  = 1.0;
    std::string out_prefix   = std::string("fu_2024_tsr_")+std::to_string((int)tsr);
    bool        is_restart   = false;
    std::string restart_file = "";
    real        latitude     = 0;
    real        roughness    = 0;
    int         dyn_cycle    = 1;

    // Things the coupler might need to know about
    coupler.set_option<real>       ( "cfl"                      , 0.6          );
    coupler.set_option<std::string>( "out_prefix"               , out_prefix   );
    coupler.set_option<std::string>( "init_data"                , init_data    );
    coupler.set_option<real       >( "out_freq"                 , out_freq     );
    coupler.set_option<bool       >( "is_restart"               , is_restart   );
    coupler.set_option<std::string>( "restart_file"             , restart_file );
    coupler.set_option<real       >( "latitude"                 , latitude     );
    coupler.set_option<real       >( "roughness"                , roughness    );
    coupler.set_option<real       >( "constant_uvel"            , U_inf        );
    coupler.set_option<real       >( "constant_vvel"            , 0            );
    coupler.set_option<real       >( "constant_temp"            , 300          );
    coupler.set_option<real       >( "constant_press"           , 105386.4     );
    coupler.set_option<std::string>( "turbine_file"             , turbine_file );
    coupler.set_option<real       >( "turbine_inflow_mag"       , U_inf        );
    coupler.set_option<real       >( "turbine_gen_eff"          , 0.944        );
    coupler.set_option<real       >( "turbine_max_power"        , 5e6          );
    coupler.set_option<real       >( "turbine_tip_decay_beg"    , 0.97         );
    coupler.set_option<real       >( "turbine_min_eps"          , dx           );
    coupler.set_option<real       >( "turbine_omega_rad_sec"    , omega_rpm*2.*M_PI/60. );
    coupler.set_option<bool       >( "turbine_immerse_material" , false        );
    coupler.set_option<real       >( "turbine_pitch_fixed"      , 0.           );
    coupler.set_option<real       >( "turbine_eps_fixed"        , 3.5          );
    coupler.set_option<real       >( "dycore_max_wind"          , 30           );
    coupler.set_option<bool       >( "dycore_buoyancy_theta"    , true         );
    coupler.set_option<real       >( "dycore_cs"                , 100          );

    // Set the turbine
    coupler.set_option<std::vector<real>>("turbine_x_locs"      ,{4*D   });
    coupler.set_option<std::vector<real>>("turbine_y_locs"      ,{ylen/2});
    coupler.set_option<std::vector<bool>>("turbine_apply_thrust",{true  });

    coupler.init( core::ParallelComm(MPI_COMM_WORLD) ,
                  coupler.generate_levels_equal(nz,zlen) ,
                  ny_glob , nx_glob , ylen , xlen );

    // They dynamical core "dycore" integrates the Euler equations and performans transport of tracers
    modules::Dynamics_Euler_Stratified_WenoFV  dycore;
    modules::Time_Averager                     time_averager;
    modules::LES_Closure                       les_closure;
    modules::TurbineActuatorLine               windmills;
    modules::EdgeSponge                        edge_sponge1;
    modules::EdgeSponge                        edge_sponge2;
    modules::ColumnNudger                      col_nudge;

    // No microphysics specified, so create a water_vapor tracer required by the dycore
    coupler.add_tracer("water_vapor","water_vapor",true,true ,true);
    coupler.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

    // Run the initialization modules
    custom_modules::sc_init   ( coupler );
    coupler.set_option<std::string>("bc_x1","open"    );
    coupler.set_option<std::string>("bc_x2","open"    );
    coupler.set_option<std::string>("bc_y1","periodic");
    coupler.set_option<std::string>("bc_y2","periodic");
    col_nudge.set_column      ( coupler , {"density_dry","temp"} );
    les_closure  .init        ( coupler );
    windmills    .init        ( coupler );
    dycore       .init        ( coupler ); // Important that dycore inits after windmills for base immersed boundaries
    time_averager.init        ( coupler );
    edge_sponge1 .set_column  ( coupler , {"density_dry","uvel","vvel","wvel","temp"} );
    edge_sponge2 .set_column  ( coupler , {"density_dry","temp"} );
    custom_modules::sc_perturb( coupler );

    // Get elapsed time (zero), and create counters for output and informing the user in stdout
    real etime = coupler.get_option<real>("elapsed_time");
    core::Counter output_counter( out_freq    , etime );
    core::Counter inform_counter( inform_freq , etime );

    // if restart, overwrite with restart data, and set the counters appropriately. Otherwise, write initial output
    if (is_restart) {
      coupler.overwrite_with_restart();
      etime = coupler.get_option<real>("elapsed_time");
      output_counter = core::Counter( out_freq    , etime-((int)(etime/out_freq   ))*out_freq    );
      inform_counter = core::Counter( inform_freq , etime-((int)(etime/inform_freq))*inform_freq );
    } else {
      coupler.write_output_file( out_prefix );
    }

    // Begin main simulation loop over time steps
    real dt = dtphys_in;
    Kokkos::fence();
    auto tm = std::chrono::high_resolution_clock::now();
    while (etime < sim_time) {
      // If dt <= 0, then set it to the dynamical core's max stable time step
      if (dtphys_in <= 0.) { dt = dycore.compute_time_step(coupler)*dyn_cycle; }
      // If we're about to go past the final time, then limit to time step to exactly hit the final time
      if (etime + dt > sim_time) { dt = sim_time - etime; }

      // Run modules
      {
        using core::Coupler;
        coupler.run_module( [&] (Coupler &c) { edge_sponge1.apply       (c,0.1,0,0,0); } , "edge_sponge1"  );
        coupler.run_module( [&] (Coupler &c) { edge_sponge2.apply       (c,0,0.1,0,0); } , "edge_sponge2"  );
        coupler.run_module( [&] (Coupler &c) { col_nudge.nudge_to_column(c,dt,dt*2);   } , "col_nudge"     );
        coupler.run_module( [&] (Coupler &c) { dycore.time_step         (c,dt);        } , "dycore"        );
        coupler.run_module( [&] (Coupler &c) { windmills.apply          (c,dt);        } , "windmills"     );
        coupler.run_module( [&] (Coupler &c) { les_closure.apply        (c,dt);        } , "les_closure"   );
        coupler.run_module( [&] (Coupler &c) { time_averager.accumulate (c,dt);        } , "time_averager" );
      }

      // Update time step
      etime += dt; // Advance elapsed time
      coupler.set_option<real>("elapsed_time",etime);
      if (inform_freq >= 0. && inform_counter.update_and_check(dt)) {
        coupler.inform_user();
        inform_counter.reset();
      }
      if (out_freq    >= 0. && output_counter.update_and_check(dt)) {
        coupler.write_output_file( out_prefix , true );
        time_averager.reset(coupler);
        output_counter.reset();
      }
    } // End main simulation loop

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}


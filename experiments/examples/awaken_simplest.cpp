
#include "coupler.h"
#include "dynamics_rk_simpler.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "windmill_actuators_yaw.h"
#include "edge_sponge.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler;

    real        sim_time     = 3600*4;
    int         nx_glob      = 475;
    int         ny_glob      = 250;
    int         nz           = 30;
    real        xlen         = 19000;
    real        ylen         = 10000;
    real        zlen         = 600;
    real        dtphys_in    = 0;
    std::string init_data    = "constant";
    real        out_freq     = 100;
    real        inform_freq  = 10;
    std::string out_prefix   = "awaken_simplest";
    bool        is_restart   = false;
    std::string restart_file = "";
    real        latitude     = 0;
    real        roughness    = 0;
    int         dyn_cycle    = 10;

    // Things the coupler might need to know about
    coupler.set_option<std::string>( "out_prefix"     , out_prefix   );
    coupler.set_option<std::string>( "init_data"      , init_data    );
    coupler.set_option<real       >( "out_freq"       , out_freq     );
    coupler.set_option<bool       >( "is_restart"     , is_restart   );
    coupler.set_option<std::string>( "restart_file"   , restart_file );
    coupler.set_option<real       >( "latitude"       , latitude     );
    coupler.set_option<real       >( "roughness"      , roughness    );
    coupler.set_option<real       >( "constant_uvel"  , 10*std::sin(M_PI/4.) );
    coupler.set_option<real       >( "constant_vvel"  , 10*std::sin(M_PI/4.) );
    coupler.set_option<real       >( "constant_temp"  , 300                 );
    coupler.set_option<real       >( "constant_press" , 1.e5                );
    coupler.set_option<std::string>( "turbine_file"             , "./inputs/NREL_2.8MW_127_RWT_new.yaml" );
    coupler.set_option<bool       >( "turbine_do_blades"        , false  );
    coupler.set_option<real       >( "turbine_initial_yaw"      , M_PI/4 );
    coupler.set_option<bool       >( "turbine_fixed_yaw"        , false  );
    coupler.set_option<bool       >( "turbine_floating_motions" , false  );

    std::ifstream in( "./inputs/KingPlains_xy_loc.txt" );
    std::vector<real> turb_xlocs, turb_ylocs;
    std::vector<bool> turb_thrust;
    real x, y, zhub;
    while (in >> x >> y >> zhub) { turb_xlocs.push_back(x);  turb_ylocs.push_back(y);  turb_thrust.push_back(true); }
    coupler.set_option<std::vector<real>>("turbine_x_locs"      ,turb_xlocs );
    coupler.set_option<std::vector<real>>("turbine_y_locs"      ,turb_ylocs );
    coupler.set_option<std::vector<bool>>("turbine_apply_thrust",turb_thrust);

    coupler.init( core::ParallelComm(MPI_COMM_WORLD) ,
                  coupler.generate_levels_equal(nz,zlen) ,
                  ny_glob , nx_glob , ylen , xlen );

    // They dynamical core "dycore" integrates the Euler equations and performans transport of tracers
    modules::Dynamics_Euler_Stratified_WenoFV  dycore;
    modules::Time_Averager                     time_averager;
    modules::LES_Closure                       les_closure;
    modules::WindmillActuators                 windmills;
    modules::EdgeSponge                        edge_sponge;

    // No microphysics specified, so create a water_vapor tracer required by the dycore
    coupler.add_tracer("water_vapor","water_vapor",true,true ,true);
    coupler.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

    // Run the initialization modules
    custom_modules::sc_init   ( coupler );
    les_closure  .init        ( coupler );
    dycore       .init        ( coupler );
    windmills    .init        ( coupler );
    time_averager.init        ( coupler );
    edge_sponge  .set_column  ( coupler );
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
        coupler.run_module( [&] (Coupler &c) { edge_sponge.apply       (c,0.02,0.02,0.02,0.02); } , "edge_sponge" );
        coupler.run_module( [&] (Coupler &c) { dycore.time_step        (c,dt); } , "dycore"        );
        coupler.run_module( [&] (Coupler &c) { windmills.apply         (c,dt); } , "windmills"     );
        coupler.run_module( [&] (Coupler &c) { les_closure.apply       (c,dt); } , "les_closure"   );
        coupler.run_module( [&] (Coupler &c) { time_averager.accumulate(c,dt); } , "time_averager" );
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


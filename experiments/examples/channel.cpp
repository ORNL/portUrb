
#include "coupler.h"
#include "dynamics_rk_simpler.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "surface_flux.h"
#include "uniform_pg_wind_forcing.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler;

    real xlen  = 12;
    real ylen  = 6;
    real zlen  = 2;
    real npnts = 128;        // USER PARAMETER 1

    real u0    = 0.1;        // USER PARAMETER 2
    real dx    = zlen/npnts;
    real z0    = dx/2;       // USER PARAMETER 3

    real        sim_time     = xlen/u0*40+0.01;
    int         nx_glob      = std::round(xlen/dx);
    int         ny_glob      = std::round(ylen/dx);
    int         nz           = std::round(zlen/dx);
    real        dtphys_in    = 0;
    std::string init_data    = "channel";
    real        out_freq     = xlen/u0*0.5;
    real        inform_freq  = xlen/u0*0.05;
    std::string out_prefix   = std::string("channel_u0-")+std::to_string(u0)+std::string("_z0-")+std::to_string(z0);
    bool        is_restart   = false;
    std::string restart_file = "";
    real        latitude     = 0;
    real        roughness    = 0.001;
    int         dyn_cycle    = 3;

    // Things the coupler might need to know about
    coupler.set_option<std::string>( "out_prefix"                         , out_prefix   );
    coupler.set_option<std::string>( "init_data"                          , init_data    );
    coupler.set_option<real       >( "out_freq"                           , out_freq     );
    coupler.set_option<bool       >( "is_restart"                         , is_restart   );
    coupler.set_option<std::string>( "restart_file"                       , restart_file );
    coupler.set_option<real       >( "latitude"                           , latitude     );
    coupler.set_option<real       >( "roughness"                          , roughness    );
    coupler.set_option<real       >( "constant_uvel"                      , u0           );
    coupler.set_option<real       >( "constant_vvel"                      , 0            );
    coupler.set_option<real       >( "constant_temp"                      , 300          );
    coupler.set_option<real       >( "constant_press"                     , 1.e5         );
    coupler.set_option<real       >( "dycore_max_wind"                    , u0*1.4       );
    coupler.set_option<bool       >( "dycore_buoyancy_theta"              , true         );
    coupler.set_option<real       >( "dycore_cs"                          , u0*1.4*2     );
    coupler.set_option<bool       >( "dycore_use_weno"                    , false        );
    coupler.set_option<bool       >( "surface_flux_force_theta"           , false        );
    coupler.set_option<bool       >( "surface_flux_stability_corrections" , false        );
    coupler.set_option<real       >( "surface_flux_kinematic_viscosity"   , 1.5e-5       );
    coupler.set_option<bool       >( "surface_flux_predict_z0h"           , false        );

    coupler.init( core::ParallelComm(MPI_COMM_WORLD) ,
                  coupler.generate_levels_equal(nz,zlen) ,
                  ny_glob , nx_glob , ylen , xlen );

    // They dynamical core "dycore" integrates the Euler equations and performans transport of tracers
    modules::Dynamics_Euler_Stratified_WenoFV  dycore;
    modules::SurfaceFlux                       sfc_flux;
    modules::Time_Averager                     time_averager;
    modules::LES_Closure                       les_closure;

    // No microphysics specified, so create a water_vapor tracer required by the dycore
    coupler.add_tracer("water_vapor","water_vapor",true,true ,true);
    coupler.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

    // Run the initialization modules
    custom_modules::sc_init   ( coupler );
    les_closure  .init        ( coupler );
    dycore       .init        ( coupler );
    sfc_flux     .init        ( coupler );
    time_averager.init        ( coupler );
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
        using modules::uniform_pg_wind_forcing_height;
        real hr = zlen/2;
        real ur = u0;
        real vr = 0;
        real tr = dt*100;
        coupler.run_module( [&] (Coupler &c) { uniform_pg_wind_forcing_height(c,dt,hr,ur,vr,tr); } , "pg_forcing" );
        coupler.run_module( [&] (Coupler &c) { dycore.time_step        (c,dt); } , "dycore"         );
        coupler.run_module( [&] (Coupler &c) { sfc_flux.apply          (c,dt); } , "surface_fluxes" );
        coupler.run_module( [&] (Coupler &c) { les_closure.apply       (c,dt); } , "les_closure"    );
        coupler.run_module( [&] (Coupler &c) { time_averager.accumulate(c,dt); } , "time_averager"  );
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



#include "coupler.h"
#include "dynamics_rk_rsst.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "surface_flux.h"
#include "geostrophic_wind_forcing.h"
#include "sponge_layer.h"
#include "surface_cooling.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    real        sim_time    = 3600*9+1;
    int         nx_glob     = 100;
    int         ny_glob     = 100;
    int         nz          = 150;
    real        xlen        = 200;
    real        ylen        = 200;
    real        zlen        = 300;
    real        dtphys_in   = 0;    // Use dycore time step
    int         dyn_cycle   = 1;
    real        out_freq    = 1800;
    real        inform_freq = 10;
    std::string out_prefix  = "ABL_stable_rss_20";
    bool        is_restart  = false;
    real        u_g         = 8;
    real        v_g         = 0;
    real        lat_g       = 73.0;
    real        scr         = 0.25/3600.;  // sfc cooling rate in K / hr

    core::Coupler coupler;
    coupler.set_option<std::string>( "out_prefix"     , out_prefix       );
    coupler.set_option<std::string>( "init_data"      , "ABL_stable"     );
    coupler.set_option<real       >( "out_freq"       , out_freq         );
    coupler.set_option<bool       >( "is_restart"     , is_restart       );
    coupler.set_option<std::string>( "restart_file"   , ""               );
    coupler.set_option<real       >( "latitude"       , 0.               );
    coupler.set_option<real       >( "roughness"      , 0.05             );
    coupler.set_option<real       >( "cfl"            , 0.6              );
    coupler.set_option<bool       >( "enable_gravity" , true             );
    coupler.set_option<real       >( "sfc_cool_rate"  , scr              );
    coupler.set_option<real       >( "dycore_max_wind"       , 10        );
    coupler.set_option<bool       >( "dycore_buoyancy_theta" , true      );
    coupler.set_option<real       >( "dycore_cs"             , 20        );

    coupler.distribute_mpi_and_allocate_coupled_state( core::ParallelComm(MPI_COMM_WORLD) , nz, ny_glob, nx_glob);

    coupler.set_grid( xlen , ylen , zlen );

    modules::Dynamics_Euler_Stratified_WenoFV     dycore;
    custom_modules::Time_Averager                 time_averager;
    modules::LES_Closure                          les_closure;

    // No microphysics specified, so create a water_vapor tracer required by the dycore
    coupler.add_tracer("water_vapor","water_vapor",true,true ,true);
    coupler.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

    custom_modules::sc_init   ( coupler );
    les_closure  .init        ( coupler );
    dycore       .init        ( coupler );
    time_averager.init        ( coupler );
    custom_modules::sc_perturb( coupler );

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
        coupler.track_max_wind();
        auto run_scr       = [&] (Coupler &c) { custom_modules::surface_cooling  (c,dt);               };
        auto run_geo       = [&] (Coupler &c) { modules::geostrophic_wind_forcing(c,dt,lat_g,u_g,v_g); };
        auto run_dycore    = [&] (Coupler &c) { dycore.time_step                 (c,dt);               };
        auto run_sponge    = [&] (Coupler &c) { modules::sponge_layer            (c,dt,dt*100,0.1);    };
        auto run_surf_flux = [&] (Coupler &c) { modules::apply_surface_fluxes    (c,dt);               };
        auto run_les       = [&] (Coupler &c) { les_closure.apply                (c,dt);               };
        auto run_tavg      = [&] (Coupler &c) { time_averager.accumulate         (c,dt);               };
        coupler.run_module( run_scr       , "sfc_cooling"         );
        coupler.run_module( run_geo       , "geostrophic_forcing" );
        coupler.run_module( run_dycore    , "dycore"              );
        coupler.run_module( run_sponge    , "sponge"              );
        coupler.run_module( run_surf_flux , "surface_fluxes"      );
        coupler.run_module( run_les       , "les_closure"         );
        coupler.run_module( run_tavg      , "time_averager"       );
      }

      // Update time step
      etime += dt; // Advance elapsed time
      coupler.set_option<real>("elapsed_time",etime);
      if (inform_freq >= 0. && inform_counter.update_and_check(dt)) {
        if (coupler.is_mainproc()) std::cout << "MaxWind [" << coupler.get_option<real>("coupler_max_wind") << "] , ";
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


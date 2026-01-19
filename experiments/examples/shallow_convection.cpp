
#include "coupler.h"
#include "dynamics_rk_simpler.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "surface_flux.h"
#include "geostrophic_wind_forcing.h"
#include "sponge_layer.h"
#include "microphysics_kessler.h"
#include "column_nudging.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    real        sim_time    = 3600*24+1;
    real        out_freq    = 1800;
    std::string out_prefix  = "shallow_convection";
    real        inform_freq = 10;
    real        xlen        = 50000;
    real        ylen        = 50000;
    real        zlen        = 3000;
    real        dx          = 50;
    real        dtphys_in   = 0.;
    real        dyn_cycle   = 1;
    bool        is_restart  = false;
    real        u_g         = 10;
    real        v_g         = 0 ;
    real        lat_g       = 43.289340204;
    int         nx_glob     = (int) std::ceil(xlen/dx);
    int         ny_glob     = (int) std::ceil(ylen/dx);
    int         nz          = (int) std::ceil(zlen/dx);

    core::Coupler coupler;
    coupler.set_option<std::string>( "out_prefix"                , out_prefix  );
    coupler.set_option<std::string>( "init_data"                 , "shallow_convection" );
    coupler.set_option<real       >( "out_freq"                  , out_freq    );
    coupler.set_option<bool       >( "is_restart"                , is_restart  );
    coupler.set_option<std::string>( "restart_file"              , ""          );
    coupler.set_option<real       >( "latitude"                  , 0.          );
    coupler.set_option<real       >( "cfl"                       , 0.6         );
    coupler.set_option<bool       >( "enable_gravity"            , true        );
    coupler.set_option<int        >( "micro_morr_ihail"          , 0           );
    coupler.set_option<real       >( "dycore_max_wind"           , 15          );
    coupler.set_option<bool       >( "dycore_buoyancy_theta"     , false       );
    coupler.set_option<real       >( "dycore_cs"                 , 350         );
    coupler.set_option<real       >( "roughness"                 , 2.5e-5      );
    coupler.set_option<bool       >( "kessler_no_rain"           , true        );

    coupler.init( core::ParallelComm(MPI_COMM_WORLD) ,
                  coupler.generate_levels_equal(nz,zlen) ,
                  ny_glob , nx_glob , ylen , xlen );

    modules::Dynamics_Euler_Stratified_WenoFV  dycore;
    modules::SurfaceFlux                       sfc_flux;
    modules::Time_Averager                     time_averager;
    modules::LES_Closure                       les_closure;
    modules::Microphysics_Kessler              micro;
    modules::ColumnNudger                      col_nudge;

    micro        .init        ( coupler );
    custom_modules::sc_init   ( coupler );
    les_closure  .init        ( coupler );
    dycore       .init        ( coupler );
    sfc_flux     .init        ( coupler );
    time_averager.init        ( coupler );
    col_nudge.set_column      ( coupler , {"water_vapor","temp"} );
    custom_modules::sc_perturb( coupler );

    coupler.set_option<std::string>("bc_x1","periodic");
    coupler.set_option<std::string>("bc_x2","periodic");
    coupler.set_option<std::string>("bc_y1","periodic");
    coupler.set_option<std::string>("bc_y2","periodic");
    coupler.set_option<std::string>("bc_z1","wall_free_slip");
    coupler.set_option<std::string>("bc_z2","wall_free_slip");

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
        // coupler.run_module( [&] (Coupler &c) { modules::geostrophic_wind_forcing(c,dt,lat_g,u_g,v_g); } , "geo"  );
        coupler.run_module( [&] (Coupler &c) { col_nudge.nudge_to_column_strict (c,dt,1000); } , "col_nudge"     );
        coupler.run_module( [&] (Coupler &c) { dycore.time_step                 (c,dt);      } , "dycore"        );
        coupler.run_module( [&] (Coupler &c) { sfc_flux.apply                   (c,dt);      } , "surface_flux"  );
        coupler.run_module( [&] (Coupler &c) { les_closure.apply                (c,dt);      } , "les_closure"   );
        coupler.run_module( [&] (Coupler &c) { micro.time_step                  (c,dt);      } , "microphysics"  );
        coupler.run_module( [&] (Coupler &c) { time_averager.accumulate         (c,dt);      } , "time_averager" );
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


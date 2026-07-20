
#include "coupler.h"
#include "dynamics_rk_simpler.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "surface_flux.h"
#include "geostrophic_wind_forcing.h"
#include "sponge_layer.h"
#include "overwrite_interpolate.h"
#include "column_nudging.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");
    real dx         = 10;
    real umax       = 15;
    real cfl        = 0.6;
    real cs         = umax*2;
    bool buoy_theta = true;
    bool rsst       = true;
    // real cs         = 10;
    // bool buoy_theta = true;
    // bool rsst       = true;

    real        sim_time    = 3600*10+1;
    real        xlen        = 1000;
    real        ylen        = 1000;
    real        zlen        = 1000;
    int         nx_glob     = (int) std::round(xlen/dx);
    int         ny_glob     = (int) std::round(ylen/dx);
    int         nz          = (int) std::round(zlen/dx);
    real        dtphys_in   = 0;    // Use dycore time step
    int         dyn_cycle   = 2;
    real        out_freq    = 1800;
    real        inform_freq = 10;
    std::string out_prefix  = std::string("ABL_neutral-dx_") + std::to_string((int)dx);
    bool        is_restart  = false;
    real        u_g         = 10;
    real        v_g         = 0 ;
    real        lat_g       = 43.289340204;

    core::Coupler parent;
    parent.set_option<std::string>( "out_prefix"                         , out_prefix    );
    parent.set_option<std::string>( "init_data"                          , "ABL_neutral" );
    parent.set_option<real       >( "out_freq"                           , out_freq      );
    parent.set_option<bool       >( "is_restart"                         , is_restart    );
    parent.set_option<std::string>( "restart_file"                       , ""            );
    parent.set_option<real       >( "latitude"                           , 0.            );
    parent.set_option<real       >( "roughness"                          , 0.1           );
    parent.set_option<real       >( "cfl"                                , cfl           );
    parent.set_option<bool       >( "enable_gravity"                     , true          );
    parent.set_option<real       >( "dycore_max_wind"                    , umax          );
    parent.set_option<bool       >( "dycore_rsst"                        , rsst          );
    parent.set_option<bool       >( "dycore_buoyancy_theta"              , buoy_theta    );
    parent.set_option<real       >( "dycore_cs"                          , cs            );
    parent.set_option<bool       >( "dycore_use_weno"                    , false         );
    parent.set_option<bool       >( "dycore_use_weno_immersed"           , true          );
    parent.set_option<bool       >( "surface_flux_force_theta"           , false         );
    parent.set_option<bool       >( "surface_flux_stability_corrections" , false         );
    parent.set_option<real       >( "surface_flux_kinematic_viscosity"   , 1.5e-5        );
    parent.set_option<bool       >( "surface_flux_predict_z0h"           , false         );
    parent.set_option<bool       >( "surface_flux_prescribe_wpthetap"    , false         );

    parent.init( core::ParallelComm(MPI_COMM_WORLD) ,
                  parent.generate_levels_equal(nz,zlen) ,
                  ny_glob , nx_glob , ylen , xlen );

    modules::Dynamics_Euler_Stratified_WenoFV     dycore;
    modules::SurfaceFlux                          sfc_flux;
    modules::Time_Averager                        time_averager;
    modules::LES_Closure                          les_closure;
    modules::ColumnNudger                         col_nudge;

    // No microphysics specified, so create a water_vapor tracer required by the dycore
    parent.add_tracer("water_vapor","water_vapor",true,true ,true);
    parent.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

    custom_modules::sc_init   ( parent );
    les_closure  .init        ( parent );
    dycore       .init        ( parent );
    sfc_flux     .init        ( parent );
    time_averager.init        ( parent );
    col_nudge    .set_column  ( parent );
    custom_modules::sc_perturb( parent );
    // modules::overwrite_interpolate( coupler , "ABL_neutral-dx_5_00000010.nc" , {"uvel","vvel","wvel","TKE"} );

    real etime = parent.get_option<real>("elapsed_time");
    core::Counter output_counter( out_freq    , etime );
    core::Counter inform_counter( inform_freq , etime );

    // if restart, overwrite with restart data, and set the counters appropriately. Otherwise, write initial output
    if (is_restart) {
      parent.overwrite_with_restart();
      etime = parent.get_option<real>("elapsed_time");
      output_counter = core::Counter( out_freq    , etime-((int)(etime/out_freq   ))*out_freq    );
      inform_counter = core::Counter( inform_freq , etime-((int)(etime/inform_freq))*inform_freq );
    } else {
      parent.write_output_file( out_prefix );
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
        parent.track_max_wind();
        parent.run_module( [&] (core::Coupler &c) { modules::geostrophic_wind_forcing_indiv(c,dt,lat_g,u_g,v_g); } , "geostrophic_forcing" );
        parent.run_module( [&] (core::Coupler &c) { col_nudge.nudge_to_column              (c,dt,1800);          } , "column_nudging"      );
        parent.run_module( [&] (core::Coupler &c) { dycore.time_step                       (c,dt);               } , "dycore"              );
        parent.run_module( [&] (core::Coupler &c) { modules::sponge_layer_w                (c,dt,1000,0.05);     } , "sponge"              );
        parent.run_module( [&] (core::Coupler &c) { sfc_flux.apply                         (c,dt);               } , "surface_fluxes"      );
        parent.run_module( [&] (core::Coupler &c) { les_closure.apply                      (c,dt);               } , "les_closure"         );
        parent.run_module( [&] (core::Coupler &c) { time_averager.accumulate               (c,dt);               } , "time_averager"       );
      }

      // Update time step
      etime += dt; // Advance elapsed time
      parent.set_option<real>("elapsed_time",etime);
      if (inform_freq >= 0. && inform_counter.update_and_check(dt)) {
        if (parent.is_mainproc()) std::cout << "MaxWind [" << parent.get_option<real>("coupler_max_wind") << "] , ";
        parent.inform_user();
        inform_counter.reset();
      }
      if (out_freq    >= 0. && output_counter.update_and_check(dt)) {
        parent.write_output_file( out_prefix , true );
        time_averager.reset(parent);
        output_counter.reset();
      }
    } // End main simulation loop

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}


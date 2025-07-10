
#include "coupler.h"
#include "dynamics_rk_rsst.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "surface_flux.h"
#include "geostrophic_wind_forcing.h"
#include "sponge_layer.h"
#include "microphysics_morr.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    YAML::Node config = YAML::LoadFile( std::string(argv[1]) );
    if ( !config ) { endrun("ERROR: Invalid supercell input file"); }
    auto sim_time      = config["sim_time"    ].as<real       >(7201);
    auto xlen          = config["xlen"        ].as<real       >(200000);
    auto ylen          = config["ylen"        ].as<real       >(200000);
    auto zlen          = config["zlen"        ].as<real       >(20000);
    auto nx_glob       = config["nx_glob"     ].as<int        >(400);
    auto ny_glob       = config["ny_glob"     ].as<int        >(400);
    auto nz            = config["nz"          ].as<int        >(40);
    auto out_prefix    = config["out_prefix"  ].as<std::string>("supercell");
    auto dtphys_in     = config["dt_phys"     ].as<real       >(0);
    auto dyn_cycle     = config["dyn_cycle"   ].as<int        >(10);
    auto out_freq      = config["out_freq"    ].as<real       >(900);
    auto inform_freq   = config["inform_freq" ].as<real       >(10);
    auto is_restart    = config["is_restart"  ].as<bool       >(false);
    auto restart_file  = config["restart_file"].as<std::string>("");
    auto cfl           = config["cfl"         ].as<real       >(0.6);
    auto cs            = config["cs"          ].as<real       >(350);
    auto buoy_theta    = config["buoy_theta"  ].as<bool       >(false);

    core::Coupler coupler;
    coupler.set_option<std::string>( "out_prefix"                , out_prefix  );
    coupler.set_option<std::string>( "init_data"                 , "supercell" );
    coupler.set_option<real       >( "out_freq"                  , out_freq    );
    coupler.set_option<bool       >( "is_restart"                , is_restart  );
    coupler.set_option<std::string>( "restart_file"              , restart_file);
    coupler.set_option<real       >( "latitude"                  , 0.          );
    coupler.set_option<real       >( "cfl"                       , cfl         );
    coupler.set_option<bool       >( "enable_gravity"            , true        );
    coupler.set_option<int        >( "micro_morr_ihail"          , 1           );
    coupler.set_option<real       >( "dycore_max_wind"           , 90          );
    coupler.set_option<bool       >( "dycore_buoyancy_theta"     , buoy_theta  );
    coupler.set_option<real       >( "dycore_cs"                 , cs          );

    coupler.distribute_mpi_and_allocate_coupled_state( core::ParallelComm(MPI_COMM_WORLD) , nz, ny_glob, nx_glob);

    coupler.set_grid( xlen , ylen , zlen );

    modules::Dynamics_Euler_Stratified_WenoFV  dycore;
    custom_modules::Time_Averager              time_averager;
    modules::LES_Closure                       les_closure;
    modules::Microphysics_Morrison             micro;

    micro        .init        ( coupler );
    custom_modules::sc_init   ( coupler );
    les_closure  .init        ( coupler );
    dycore       .init        ( coupler );
    time_averager.init        ( coupler );
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
        // coupler.run_module( [&] (Coupler &c) { modules::sponge_layer   (c,dt,dt,0.02); } , "sponge"         );
        coupler.run_module( [&] (Coupler &c) { dycore.time_step        (c,dt);         } , "dycore"         );
        coupler.run_module( [&] (Coupler &c) { les_closure.apply       (c,dt);         } , "les_closure"    );
        coupler.run_module( [&] (Coupler &c) { micro.time_step         (c,dt);         } , "microphysics"   );
        coupler.run_module( [&] (Coupler &c) { time_averager.accumulate(c,dt);         } , "time_averager"  );
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


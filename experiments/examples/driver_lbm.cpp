
#include "coupler.h"
#include "dynamics_lbm.h"
#include "dynamics_rk_simpler.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "surface_flux.h"
#include "geostrophic_wind_forcing.h"
#include "sponge_layer.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    real        sim_time    = 1000;
    int         nx_glob     = 200;
    int         ny_glob     = 200;
    int         nz          = 200;
    real        xlen        = 1000;
    real        ylen        = 1000;
    real        zlen        = 1000;
    real        dtphys_in   = 0;    // Use dycore time step
    real        out_freq    = 100.;
    real        inform_freq = 10.;
    std::string out_prefix  = "lbm";

    core::Coupler coupler;
    coupler.set_option<std::string>( "out_prefix"      , out_prefix    );
    coupler.set_option<real       >( "out_freq"        , out_freq      );
    coupler.set_option<real       >( "dycore_max_wind" , 20            );
    coupler.set_option<int        >( "dycore_nq"       , 19            );
    coupler.set_option<int        >( "dycore_ord"      , 2             );
    coupler.set_option<real       >( "cfl"             , 0.6           );
    coupler.set_option<std::string>( "init_data"       , "LBM"         );
    coupler.set_option<bool       >( "enable_gravity"  , false         );

    coupler.distribute_mpi_and_allocate_coupled_state( core::ParallelComm(MPI_COMM_WORLD) , nz, ny_glob, nx_glob);

    coupler.set_grid( xlen , ylen , zlen );

    modules::Dynamics_Euler_LBM                dycore_lbm;
    modules::Dynamics_Euler_Stratified_WenoFV  dycore_fv;
    auto &dycore = dycore_fv;

    // No microphysics specified, so create a water_vapor tracer required by the dycore
    coupler.add_tracer("water_vapor","water_vapor",true,true ,true);
    coupler.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

    custom_modules::sc_init( coupler );
    dycore.init            ( coupler );

    custom_modules::sc_perturb( coupler );

    real etime = coupler.get_option<real>("elapsed_time");
    core::Counter output_counter( out_freq    , etime );
    core::Counter inform_counter( inform_freq , etime );

    coupler.write_output_file( out_prefix );

    real dt = dtphys_in;
    while (etime < sim_time) {
      if (dtphys_in <= 0.) { dt = dycore.compute_time_step(coupler); }
      if (etime + dt > sim_time) { dt = sim_time - etime; }

      using core::Coupler;
      coupler.run_module( [&] (Coupler &c) { dycore.time_step(c,dt); } , "dycore" );

      etime += dt; // Advance elapsed time
      coupler.set_option<real>("elapsed_time",etime);
      if (inform_freq >= 0. && inform_counter.update_and_check(dt)) {
        coupler.inform_user();
        inform_counter.reset();
      }
      if (out_freq    >= 0. && output_counter.update_and_check(dt)) {
        coupler.write_output_file( out_prefix , true );
        output_counter.reset();
      }
    } // End main simulation loop

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}


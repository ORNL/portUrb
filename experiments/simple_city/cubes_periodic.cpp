
#include "coupler.h"
#include "dynamics_rk.h"
#include "time_averager.h"
#include "sc_init.h"
#include "les_closure.h"
#include "surface_flux.h"
#include "uniform_pg_wind_forcing.h"
#include "Ensembler.h"
#include <sstream>

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler;
    real constexpr h = 0.02;
    real constexpr dx = 0.001;

    real        sim_time     = 5;
    real        xlen         = 4*h;
    real        ylen         = 4*h;
    real        zlen         = 10*h;
    int         nx_glob      = xlen/dx;
    int         ny_glob      = ylen/dx;
    int         nz           = zlen/dx;
    real        dtphys_in    = 0;
    std::string init_data    = "cubes_periodic";
    real        out_freq     = 1.e-1;
    real        inform_freq  = 1.e-3;
    std::string out_prefix   = "cubes_periodic";
    bool        is_restart   = false;
    std::string restart_file = "";
    real        latitude     = 0;
    int         dyn_cycle    = 1;

    // Things the coupler might need to know about
    coupler.set_option<std::string>( "out_prefix"     , out_prefix   );
    coupler.set_option<std::string>( "ensemble_stdout","ensemble_"   );
    coupler.set_option<std::string>( "init_data"      , init_data    );
    coupler.set_option<real       >( "out_freq"       , out_freq     );
    coupler.set_option<bool       >( "is_restart"     , is_restart   );
    coupler.set_option<std::string>( "restart_file"   , restart_file );
    coupler.set_option<real       >( "latitude"       , latitude     );

    core::Ensembler ensembler;

    // Add wind dimension
    {
      auto func_nranks  = [=] (int ind) { return 1; };
      auto func_coupler = [=] (int ind, core::Coupler &coupler) {
        real roughness;
        if (ind == 0) roughness = 1.0e-9;
        if (ind == 1) roughness = 1.0e-8;
        if (ind == 2) roughness = 1.0e-7;
        if (ind == 3) roughness = 1.0e-6;
        if (ind == 4) roughness = 1.0e-5;
        if (ind == 5) roughness = 1.0e-4;
        if (ind == 6) roughness = 1.0e-3;
        coupler.set_option<real>("roughness",roughness);
        std::stringstream tag;
        tag << "z0-" << std::scientific << std::setprecision(3) << roughness;
        ensembler.append_coupler_string(coupler,"ensemble_stdout",tag.str());
        ensembler.append_coupler_string(coupler,"out_prefix"     ,tag.str());
      };
      ensembler.register_dimension( 7 , func_nranks , func_coupler );
    }
    auto par_comm = ensembler.create_coupler_comm( coupler , 16 , MPI_COMM_WORLD );
    auto ostr = std::ofstream(coupler.get_option<std::string>("ensemble_stdout")+std::string(".out"));
    auto orig_cout_buf = std::cout.rdbuf();
    auto orig_cerr_buf = std::cerr.rdbuf();
    std::cout.rdbuf(ostr.rdbuf());
    std::cerr.rdbuf(ostr.rdbuf());

    if (par_comm.valid()) {
      yakl::timer_start("main");

      // Coupler state is: (1) dry density;  (2) u-velocity;  (3) v-velocity;  (4) w-velocity;  (5) temperature
      //                   (6+) tracer masses (*not* mixing ratios!); and Option elapsed_time init to zero
      coupler.distribute_mpi_and_allocate_coupled_state( par_comm , nz, ny_glob, nx_glob);

      // Just tells the coupler how big the domain is in each dimensions
      coupler.set_grid( xlen , ylen , zlen );

      // They dynamical core "dycore" integrates the Euler equations and performans transport of tracers
      modules::Dynamics_Euler_Stratified_WenoFV  dycore;
      custom_modules::Time_Averager              time_averager;
      modules::LES_Closure                       les_closure;

      // No microphysics specified, so create a water_vapor tracer required by the dycore
      coupler.add_tracer("water_vapor","water_vapor",true,true ,true);
      coupler.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

      // Run the initialization modules
      custom_modules::sc_init ( coupler );
      les_closure  .init      ( coupler );
      dycore       .init      ( coupler ); // Dycore should initialize its own state here
      time_averager.init      ( coupler );

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
        coupler.write_output_file( coupler.get_option<std::string>("out_prefix") );
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
          real u = 10;
          real v = 0;
          // coupler.run_module( [&] (Coupler &c) { uniform_pg_wind_forcing_height(c,dt,9.5*h,u,v); } , "pg_forcing"     );
          {
            auto dm_r = coupler.get_data_manager_readwrite().get<real,3>("density_dry");
            auto dm_u = coupler.get_data_manager_readwrite().get<real,3>("uvel");
            auto nx = coupler.get_nx();
            auto ny = coupler.get_ny();
            auto nz = coupler.get_nz();
            yakl::c::parallel_for( YAKL_AUTO_LABEL() , yakl::c::SimpleBounds<3>(nz,ny,nx) ,
                                                       KOKKOS_LAMBDA (int k, int j, int i) {
              dm_u(k,j,i) += dt*3.875/dm_r(k,j,i);
            });
          }
          coupler.run_module( [&] (Coupler &c) { dycore.time_step              (c,dt);           } , "dycore"         );
          coupler.run_module( [&] (Coupler &c) { les_closure.apply             (c,dt);           } , "les_closure"    );
          coupler.run_module( [&] (Coupler &c) { modules::apply_surface_fluxes (c,dt);           } , "surface_fluxes" );
          coupler.run_module( [&] (Coupler &c) { time_averager.accumulate      (c,dt);           } , "time_averager"  );
        }

        // Update time step
        etime += dt; // Advance elapsed time
        coupler.set_option<real>("elapsed_time",etime);
        if (inform_freq >= 0. && inform_counter.update_and_check(dt)) {
          coupler.inform_user();
          inform_counter.reset();
        }
        if (out_freq    >= 0. && output_counter.update_and_check(dt)) {
          coupler.write_output_file( coupler.get_option<std::string>("out_prefix") , true );
          time_averager.reset(coupler);
          output_counter.reset();
        }
      } // End main simulation loop

      yakl::timer_stop("main");
    }
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}


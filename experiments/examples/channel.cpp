
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

    real        sim_time     = 8*M_PI*10;
    real        dx           = 0.02;
    real        xlen         = 8*M_PI;
    real        ylen         = 3*M_PI;
    real        zlen         = 2;
    int         nx_glob      = xlen/dx;
    int         ny_glob      = ylen/dx;
    int         nz           = zlen/dx;
    real        dtphys_in    = 0;
    std::string init_data    = "constant";
    real        out_freq     = 8*M_PI;
    real        inform_freq  = 0.1;
    std::string out_prefix   = "channel";
    bool        is_restart   = false;
    std::string restart_file = "";
    real        latitude     = 0;
    real        roughness    = 0.001;
    int         dyn_cycle    = 1;

    // Things the coupler might need to know about
    coupler.set_option<std::string>( "out_prefix"     , out_prefix   );
    coupler.set_option<std::string>( "init_data"      , init_data    );
    coupler.set_option<real       >( "out_freq"       , out_freq     );
    coupler.set_option<bool       >( "is_restart"     , is_restart   );
    coupler.set_option<std::string>( "restart_file"   , restart_file );
    coupler.set_option<real       >( "latitude"       , latitude     );
    coupler.set_option<real       >( "roughness"      , roughness    );
    coupler.set_option<real       >( "constant_uvel"  , 1            );
    coupler.set_option<real       >( "constant_vvel"  , 0            );
    coupler.set_option<real       >( "constant_temp"  , 300          );
    coupler.set_option<real       >( "constant_press" , 1.e5         );
    coupler.set_option<real       >( "dycore_max_wind"      , 2      );
    coupler.set_option<bool       >( "dycore_buoyancy_theta", true   );
    coupler.set_option<real       >( "dycore_cs"            , 4      );
    coupler.set_option<bool       >( "dycore_use_weno"      , false  );
    coupler.set_option<bool       >( "enable_gravity"       , false  );
    coupler.set_option<bool       >("surface_flux_force_theta"          ,false );
    coupler.set_option<bool       >("surface_flux_stability_corrections",false );
    coupler.set_option<real       >("surface_flux_kinematic_viscosity"  ,1.5e-5);
    coupler.set_option<bool       >("surface_flux_predict_z0h"          ,false );

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
    coupler.set_option<std::string>("bc_x1","periodic");
    coupler.set_option<std::string>("bc_x2","periodic");
    coupler.set_option<std::string>("bc_y1","periodic");
    coupler.set_option<std::string>("bc_y2","periodic");
    coupler.set_option<std::string>("bc_z1","wall_free_slip");
    coupler.set_option<std::string>("bc_z2","wall_free_slip");

    les_closure  .init        ( coupler );
    dycore       .init        ( coupler );
    sfc_flux     .init        ( coupler );
    time_averager.init        ( coupler );

    {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto i_beg     = coupler.get_i_beg();
      auto j_beg     = coupler.get_j_beg();
      auto nx_glob   = coupler.get_nx_glob();
      auto ny_glob   = coupler.get_ny_glob();
      auto nz        = coupler.get_nz();
      auto ny        = coupler.get_ny();
      auto nx        = coupler.get_nx();
      auto u         = coupler.get_data_manager_readwrite().get<real,3>("uvel");
      auto imm       = coupler.get_data_manager_readwrite().get<real,3>("immersed_proportion_halos");
      auto imm_rough = coupler.get_data_manager_readwrite().get<real,3>("immersed_roughness_halos" );
      imm_rough = roughness;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        yakl::Random rng(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + i_beg+i);
        u(k,j,i) += rng.genFP<real>(-0.2,0.2);
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny+2,nx+2) , KOKKOS_LAMBDA (int j, int i) {
        imm(1+nz,j,i) = 1;
      });
    }

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
        real hr = 1;
        real ur = 1;
        real vr = 0;
        real tr = dt*100;
        coupler.run_module( [&] (Coupler &c) { uniform_pg_wind_forcing_height(c,dt,hr,ur,vr,tr); } , "pg_forcing"     );
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


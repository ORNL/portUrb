#include "coupler.h"
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

    real dx         = 20;
    real umax       = 15;
    real cfl        = 0.6;
    real cs         = umax*2;
    bool buoy_theta = true;
    bool rsst       = true;

    real        sim_time    = 3600*10+1;
    real        xlen        = 1000;
    real        ylen        = 1000;
    real        zlen        = 1000;
    int         nx_glob     = (int) std::round(xlen/dx);
    int         ny_glob     = (int) std::round(ylen/dx);
    int         nz          = (int) std::round(zlen/dx);
    real        dtphys_in   = 0;    // Use each level's dycore time step
    int         dyn_cycle   = 2;
    real        out_freq    = 1800;
    real        inform_freq = 10;
    std::string output_prefix = std::string("ABL_neutral_nested-dx_") + std::to_string((int)dx);
    std::string level1_prefix = output_prefix + "_level1";
    std::string level2_prefix = output_prefix + "_level2";
    bool        is_restart  = false;
    std::string level1_restart_file = "";
    std::string level2_restart_file = "";
    real        u_g         = 10;
    real        v_g         = 0;
    real        lat_g       = 43.289340204;

    core::Coupler level1;
    level1.set_option<std::string>( "out_prefix"                         , level1_prefix );
    level1.set_option<std::string>( "init_data"                          , "ABL_neutral" );
    level1.set_option<real       >( "out_freq"                           , out_freq       );
    level1.set_option<bool       >( "is_restart"                         , is_restart     );
    level1.set_option<std::string>( "restart_file"                       , level1_restart_file );
    level1.set_option<real       >( "latitude"                           , 0.             );
    level1.set_option<real       >( "roughness"                          , 0.1            );
    level1.set_option<real       >( "cfl"                                , cfl            );
    level1.set_option<bool       >( "enable_gravity"                     , true           );
    level1.set_option<real       >( "dycore_max_wind"                    , umax           );
    level1.set_option<bool       >( "dycore_rsst"                        , rsst           );
    level1.set_option<bool       >( "dycore_buoyancy_theta"              , buoy_theta     );
    level1.set_option<real       >( "dycore_cs"                          , cs             );
    level1.set_option<bool       >( "dycore_use_weno"                    , false          );
    level1.set_option<bool       >( "dycore_use_weno_immersed"           , true           );
    level1.set_option<bool       >( "surface_flux_force_theta"           , false          );
    level1.set_option<bool       >( "surface_flux_stability_corrections" , false          );
    level1.set_option<real       >( "surface_flux_kinematic_viscosity"   , 1.5e-5         );
    level1.set_option<bool       >( "surface_flux_predict_z0h"           , false          );
    level1.set_option<bool       >( "surface_flux_prescribe_wpthetap"    , false          );
    level1.set_option<int        >( "dycore_max_cycles"                  , dyn_cycle+1    );

    level1.init( core::ParallelComm(MPI_COMM_WORLD) ,
                 level1.generate_levels_equal(nz,zlen) ,
                 ny_glob , nx_glob , ylen , xlen );

    // The child covers the centered half of x and y, and the bottom half of z, at twice the parent resolution.
    int const i_child_beg = nx_glob/4;
    int const j_child_beg = ny_glob/4;
    int const k_child_beg = 0;
    int const i_child_end = i_child_beg + nx_glob/2 - 1;
    int const j_child_end = j_child_beg + ny_glob/2 - 1;
    int const k_child_end = k_child_beg + nz/2 - 1;
    core::Coupler level2 = level1.create_child_coupler( i_child_beg , i_child_end ,
                                                        j_child_beg , j_child_end ,
                                                        k_child_beg , k_child_end , 2 , 2 , 2 );
    level2.set_option<std::string>("out_prefix"  ,level2_prefix       );
    level2.set_option<std::string>("restart_file",level2_restart_file);
    level2.set_option<int        >("dycore_max_cycles",dyn_cycle+1      );

    modules::Dynamics_Euler_Stratified_WenoFV dycore;
    modules::SurfaceFlux                      sfc_flux;
    modules::Time_Averager                    time_averager;
    modules::LES_Closure                      les_closure;

    // Initialize the same prognostic variables independently on both grids.
    level1.add_tracer("water_vapor","water_vapor",true,true,true);
    level2.add_tracer("water_vapor","water_vapor",true,true,true);
    level1.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;
    level2.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;
    custom_modules::sc_init(level1);
    custom_modules::sc_init(level2);

    // All child boundaries except its physical lower wall are supplied by the parent nest.
    level2.set_option<std::string>("bc_x1","nested");
    level2.set_option<std::string>("bc_x2","nested");
    level2.set_option<std::string>("bc_y1","nested");
    level2.set_option<std::string>("bc_y2","nested");
    level2.set_option<std::string>("bc_z1","wall_free_slip");
    level2.set_option<std::string>("bc_z2","nested");

    // Boundary strings must be installed before these init calls allocate nested boundary storage.
    les_closure  .init(level1);
    les_closure  .init(level2);
    dycore       .init(level1);
    dycore       .init(level2);
    sfc_flux     .init(level1);
    sfc_flux     .init(level2);
    time_averager.init(level1);
    time_averager.init(level2);
    custom_modules::sc_perturb(level1);
    custom_modules::sc_perturb(level2);

    real parent_etime = level1.get_option<real>("elapsed_time");
    real child_etime  = level2.get_option<real>("elapsed_time");
    core::Counter output_counter(out_freq,parent_etime);
    core::Counter inform_counter(inform_freq,parent_etime);

    // Restart and output each grid through its own file sequence.
    if (is_restart) {
      level1.overwrite_with_restart();
      level2.overwrite_with_restart();
      parent_etime = level1.get_option<real>("elapsed_time");
      child_etime  = level2.get_option<real>("elapsed_time");
      if (std::abs(parent_etime-child_etime) > 1.e-10) {
        Kokkos::abort("Parent and child restart elapsed times do not match");
      }
      output_counter = core::Counter(out_freq,parent_etime-((int)(parent_etime/out_freq))*out_freq);
      inform_counter = core::Counter(inform_freq,parent_etime-((int)(parent_etime/inform_freq))*inform_freq);
    } else {
      level1.write_output_file(level1_prefix);
      level2.write_output_file(level2_prefix);
    }

    real dt = dtphys_in;
    Kokkos::fence();
    while (parent_etime < sim_time) {
      // The fine grid controls a single physics step shared by both levels. Passing the child to the parent dycore
      // also forces matching internal cycle sizes and captures one child halo set at every cycle and RK stage.
      if (dtphys_in <= 0.) {
        dt = std::min(dycore.compute_time_step(level1),dycore.compute_time_step(level2))*dyn_cycle;
      }
      if (parent_etime + dt > sim_time) { dt = sim_time-parent_etime; }

      // Advance the parent first so it can populate the stage-resolved child boundary storage.
      level1.track_max_wind();
      level1.run_module( [&] (core::Coupler &c) { modules::geostrophic_wind_forcing_indiv(c,dt,lat_g,u_g,v_g); },
                         "geostrophic_forcing" );
      level1.run_module( [&] (core::Coupler &c) { dycore.time_step(c,dt,&level2);                           }, "dycore"         );
      level1.run_module( [&] (core::Coupler &c) { modules::sponge_layer_w(c,dt,1000,0.05);                 }, "sponge"         );
      level1.run_module( [&] (core::Coupler &c) { sfc_flux.apply(c,dt);                                    }, "surface_fluxes" );
      level1.run_module( [&] (core::Coupler &c) { les_closure.apply(c,dt,&level2);                         }, "les_closure"    );
      level1.run_module( [&] (core::Coupler &c) { time_averager.accumulate(c,dt);                          }, "time_averager"  );
      parent_etime += dt;
      level1.set_option<real>("elapsed_time",parent_etime);

      // Replay the matching parent cycle and RK-stage halos while advancing the child over the identical step.
      level2.set_option<real>("elapsed_time",child_etime);
      level2.track_max_wind();
      level2.run_module( [&] (core::Coupler &c) { modules::geostrophic_wind_forcing_indiv(c,dt,lat_g,u_g,v_g); },
                         "geostrophic_forcing" );
      level2.run_module( [&] (core::Coupler &c) { dycore.time_step(c,dt);                                 }, "dycore"         );
      level2.run_module( [&] (core::Coupler &c) { sfc_flux.apply(c,dt);                                   }, "surface_fluxes" );
      level2.run_module( [&] (core::Coupler &c) { les_closure.apply(c,dt);                                }, "les_closure"    );
      level2.run_module( [&] (core::Coupler &c) { time_averager.accumulate(c,dt);                         }, "time_averager"  );
      child_etime += dt;
      level2.set_option<real>("elapsed_time",child_etime);

      // Feed the fine-grid solution back to the covered parent cells; refluxing is intentionally omitted.
      dycore.overwrite_parent_volume(level1,level2);

      if (inform_freq >= 0. && inform_counter.update_and_check(dt)) {
        if (level1.is_mainproc()) {
          std::cout << "Parent MaxWind [" << level1.get_option<real>("coupler_max_wind") << "] , ";
        }
        level1.inform_user();
        if (level2.is_mainproc()) {
          std::cout << "Child  MaxWind [" << level2.get_option<real>("coupler_max_wind") << "] , ";
        }
        level2.inform_user();
        inform_counter.reset();
      }
      if (out_freq >= 0. && output_counter.update_and_check(dt)) {
        level1.write_output_file(level1_prefix,true);
        level2.write_output_file(level2_prefix,true);
        time_averager.reset(level1);
        time_averager.reset(level2);
        output_counter.reset();
      }
    }

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}

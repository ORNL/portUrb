
#include "coupler.h"
#include "dynamics_rk.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "windmill_actuators_yaw.h"
#include "surface_flux.h"
#include "surface_heat_flux.h"
#include "precursor_sponge.h"
#include "uniform_pg_wind_forcing.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler_main;
    core::Coupler coupler_prec;

    real dx = 10;

    std::string turbine_file = "./inputs/NREL_5MW_126_RWT.yaml";
    YAML::Node config = YAML::LoadFile( turbine_file );
    if ( !config ) { endrun("ERROR: Invalid turbine input file"); }
    real D     = config["blade_radius"].as<real>()*2;
    real hub_z = config["hub_height"  ].as<real>();

    real        sim_time          = 20001;
    real        xlen              = 5120;
    real        ylen              = 5120;
    real        zlen              = 1920;
    int         nx_glob           = std::ceil(xlen/dx);    xlen = nx_glob * dx;
    int         ny_glob           = std::ceil(ylen/dx);    ylen = ny_glob * dx;
    int         nz                = std::ceil(zlen/dx);    zlen = nz      * dx;
    real        dtphys_in         = 0;  // Determined by dycore CFL restriction
    std::string init_data         = "nrel_5mw_convective";
    real        out_freq          = 1000;
    real        inform_freq       = 10;
    std::string out_prefix        = "nrel_5mw_convective";
    std::string out_prefix_prec   = out_prefix+std::string("_precursor");
    bool        is_restart        = false;
    std::string restart_file      = "";
    std::string restart_file_prec = "";
    real        latitude          = 40;
    real        roughness         = 0.01;
    int         dyn_cycle         = 10;
    real        vort_freq         = -1;
    real        hub_u             = 9.8726896031426;
    real        hub_v             = 5.7;

    // Things the coupler_main might need to know about
    coupler_main.set_option<std::string>( "out_prefix"               , out_prefix        );
    coupler_main.set_option<std::string>( "init_data"                , init_data         );
    coupler_main.set_option<real       >( "out_freq"                 , out_freq          );
    coupler_main.set_option<bool       >( "is_restart"               , is_restart        );
    coupler_main.set_option<std::string>( "restart_file"             , restart_file      );
    coupler_main.set_option<std::string>( "restart_file_precursor"   , restart_file_prec );
    coupler_main.set_option<real       >( "latitude"                 , latitude          );
    coupler_main.set_option<real       >( "roughness"                , roughness         );
    coupler_main.set_option<std::string>( "turbine_file"             , turbine_file      );
    coupler_main.set_option<bool       >( "turbine_do_blades"        , false             );
    coupler_main.set_option<real       >( "turbine_initial_yaw"      , 30./180.*M_PI     );
    coupler_main.set_option<bool       >( "turbine_fixed_yaw"        , true              );
    coupler_main.set_option<bool       >( "turbine_floating_motions" , false             );
    coupler_main.set_option<bool       >( "turbine_immerse_material" , false             );
    coupler_main.set_option<real       >( "hub_height_uvel"          , hub_u             );
    coupler_main.set_option<real       >( "hub_height_vvel"          , hub_v             );
    coupler_main.set_option<real       >( "sfc_heat_flux"            , 0.005             );
    coupler_main.set_option<real       >( "kinematic_viscosity"      , 0                 );
    coupler_main.set_option<real       >( "dycore_max_wind"          , 40                );

    if (coupler_main.is_mainproc()) {
      std::cout << "Prefix:    " << out_prefix << std::endl;
      std::cout << "Domain:    " << xlen/D << " x " << ylen/D << " x " << zlen/D << std::endl;
      std::cout << "Time:      " << sim_time/3600 << std::endl;
      std::cout << "Wind:      " << std::sqrt(hub_u*hub_u+hub_v*hub_v) << std::endl;
      std::cout << "Tower:     " << coupler_main.get_option<bool>("turbine_immerse_material") << std::endl;
      std::cout << "Blades:    " << coupler_main.get_option<bool>("turbine_do_blades"       ) << std::endl;
      std::cout << "Yaw Fixed: " << coupler_main.get_option<bool>("turbine_fixed_yaw"       ) << std::endl;
      std::cout << "Yaw Angle: " << coupler_main.get_option<bool>("turbine_initial_yaw"     ) << std::endl;
    }

    // Set the turbine
    coupler_main.set_option<std::vector<real>>("turbine_x_locs"      ,{1800});
    coupler_main.set_option<std::vector<real>>("turbine_y_locs"      ,{1800});
    coupler_main.set_option<std::vector<bool>>("turbine_apply_thrust",{true});

    // Coupler state is: (1) dry density;  (2) u-velocity;  (3) v-velocity;  (4) w-velocity;  (5) temperature
    //                   (6+) tracer masses (*not* mixing ratios!); and Option elapsed_time init to zero
    coupler_main.distribute_mpi_and_allocate_coupled_state( core::ParallelComm(MPI_COMM_WORLD) , nz, ny_glob, nx_glob);

    // Just tells the coupler_main how big the domain is in each dimensions
    coupler_main.set_grid( xlen , ylen , zlen );

    // They dynamical core "dycore" integrates the Euler equations and performans transport of tracers
    modules::Dynamics_Euler_Stratified_WenoFV  dycore;
    custom_modules::Time_Averager              time_averager;
    modules::LES_Closure                       les_closure;
    modules::WindmillActuators                 windmills;

    // No microphysics specified, so create a water_vapor tracer required by the dycore
    coupler_main.add_tracer("water_vapor","water_vapor",true,true ,true);
    coupler_main.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

    // These are set for init just for periodic case
    coupler_main.set_option<real>("turbine_hub_height",hub_z);  // Height of hub / center of windmills
    // Run the initialization modules
    custom_modules::sc_init   ( coupler_main );
    les_closure  .init        ( coupler_main );
    dycore       .init        ( coupler_main ); // Important that dycore inits after windmills for base immersed boundaries
    custom_modules::sc_perturb( coupler_main );

    /////////////////////////////////////////////////////////////////////////
    // Everything previous to this is now replicated in coupler_precursor
    // From here out, the will be treated separately
    coupler_main.clone_into(coupler_prec);
    /////////////////////////////////////////////////////////////////////////

    coupler_main.set_option<std::string>("bc_x1","periodic");
    coupler_main.set_option<std::string>("bc_x2","periodic");
    coupler_main.set_option<std::string>("bc_y1","periodic");
    coupler_main.set_option<std::string>("bc_y2","periodic");
    coupler_main.set_option<std::string>("bc_z1","wall_free_slip");
    coupler_main.set_option<std::string>("bc_z2","wall_free_slip");

    coupler_prec.set_option<std::string>("bc_x1","periodic");
    coupler_prec.set_option<std::string>("bc_x2","periodic");
    coupler_prec.set_option<std::string>("bc_y1","periodic");
    coupler_prec.set_option<std::string>("bc_y2","periodic");
    coupler_prec.set_option<std::string>("bc_z1","wall_free_slip");
    coupler_prec.set_option<std::string>("bc_z2","wall_free_slip");

    windmills    .init( coupler_main );
    time_averager.init( coupler_main );
    time_averager.init( coupler_prec );

    windmills.turbine_group.turbines[0].u_samp_inertial = coupler_prec.get_option<real>("hub_height_uvel");
    windmills.turbine_group.turbines[0].v_samp_inertial = coupler_prec.get_option<real>("hub_height_vvel");

    // Get elapsed time (zero), and create counters for output and informing the user in stdout
    real etime = coupler_main.get_option<real>("elapsed_time");
    core::Counter output_counter( out_freq    , etime );
    core::Counter inform_counter( inform_freq , etime );
    core::Counter vort_counter  ( vort_freq   , etime );

    // if restart, overwrite with restart data, and set the counters appropriately. Otherwise, write initial output
    if (is_restart) {
      coupler_main.overwrite_with_restart();
      etime = coupler_main.get_option<real>("elapsed_time");
      output_counter = core::Counter( out_freq    , etime-((int)(etime/out_freq   ))*out_freq    );
      inform_counter = core::Counter( inform_freq , etime-((int)(etime/inform_freq))*inform_freq );
    } else {
      coupler_main.write_output_file( out_prefix );
    }

    if (restart_file_prec != "" && restart_file_prec != "null") {
      coupler_prec.set_option<std::string>("restart_file",restart_file_prec);
      coupler_prec.overwrite_with_restart();
      auto &dm_prec = coupler_prec.get_data_manager_readonly();
      auto &dm_main = coupler_main     .get_data_manager_readwrite();
      dm_prec.get<real const,3>("density_dry").deep_copy_to(dm_main.get<real,3>("density_dry"));
      dm_prec.get<real const,3>("uvel"       ).deep_copy_to(dm_main.get<real,3>("uvel"       ));
      dm_prec.get<real const,3>("vvel"       ).deep_copy_to(dm_main.get<real,3>("vvel"       ));
      dm_prec.get<real const,3>("wvel"       ).deep_copy_to(dm_main.get<real,3>("wvel"       ));
      dm_prec.get<real const,3>("temp"       ).deep_copy_to(dm_main.get<real,3>("temp"       ));
      dm_prec.get<real const,3>("TKE"        ).deep_copy_to(dm_main.get<real,3>("TKE"        ));
    } else {
      if (out_freq >= 0) coupler_prec.write_output_file( out_prefix_prec );
    }

    // Begin main simulation loop over time steps
    real dt = dtphys_in;
    Kokkos::fence();
    auto tm = std::chrono::high_resolution_clock::now();
    while (etime < sim_time) {
      // If dt <= 0, then set it to the dynamical core's max stable time step
      if (dtphys_in <= 0.) { dt = dycore.compute_time_step(coupler_main)*dyn_cycle; }
      // If we're about to go past the final time, then limit to time step to exactly hit the final time
      if (etime + dt > sim_time) { dt = sim_time - etime; }

      // Run modules
      real pgu, pgv;
      {
        using core::Coupler;
        using modules::uniform_pg_wind_forcing_height;
        real h = coupler_prec.get_option<real>("turbine_hub_height");
        real u = coupler_prec.get_option<real>("hub_height_uvel");
        real v = coupler_prec.get_option<real>("hub_height_vvel");
        real tau = dt*100;
        auto run_pg_frc   = [&] (Coupler &c) { std::tie(pgu,pgv) = uniform_pg_wind_forcing_height(c,dt,h,u,v,tau); };
        auto run_dycore   = [&] (Coupler &c) { dycore.time_step                                  (c,dt);           };
        auto run_sfc_flx  = [&] (Coupler &c) { modules::apply_surface_fluxes                     (c,dt);           };
        auto run_heat_flx = [&] (Coupler &c) { custom_modules::surface_heat_flux                 (c,dt);           };
        auto run_les      = [&] (Coupler &c) { les_closure.apply                                 (c,dt);           };
        auto run_tavg     = [&] (Coupler &c) { time_averager.accumulate                          (c,dt);           };
        coupler_prec.run_module( run_pg_frc   , "pg_forcing"     );
        coupler_prec.run_module( run_dycore   , "dycore"         );
        coupler_prec.run_module( run_sfc_flx  , "surface_fluxes" );
        coupler_prec.run_module( run_heat_flx , "heat_fluxes"    );
        coupler_prec.run_module( run_les      , "les_closure"    );
        coupler_prec.run_module( run_tavg     , "time_averager"  );
      }
      {
        using core::Coupler;
        using modules::uniform_pg_wind_forcing_specified;
        // custom_modules::precursor_sponge( coupler_main , coupler_prec ,
        //                                   {"uvel","vvel","wvel","temp","TKE"} ,
        //                                   (int) (0.1*nx_glob) , 0 ,
        //                                   (int) (0.1*ny_glob) , 0 );
        auto run_pg_frc   = [&] (Coupler &c) { uniform_pg_wind_forcing_specified(c,dt,pgu,pgv); };
        auto run_dycore   = [&] (Coupler &c) { dycore.time_step                 (c,dt);         };
        auto run_sfc_flx  = [&] (Coupler &c) { modules::apply_surface_fluxes    (c,dt);         };
        auto run_heat_flx = [&] (Coupler &c) { custom_modules::surface_heat_flux(c,dt);         };
        auto run_turbine  = [&] (Coupler &c) { windmills.apply                  (c,dt);         };
        auto run_les      = [&] (Coupler &c) { les_closure.apply                (c,dt);         };
        auto run_tavg     = [&] (Coupler &c) { time_averager.accumulate         (c,dt);         };
        coupler_main.run_module( run_pg_frc   , "pg_forcing"    );
        coupler_main.run_module( run_dycore   , "dycore"        );
        coupler_main.run_module( run_sfc_flx  , "surface_fluxes");
        coupler_main.run_module( run_heat_flx , "heat_fluxes"   );
        coupler_main.run_module( run_turbine  , "windmills"     );
        coupler_main.run_module( run_les      , "les_closure"   );
        coupler_main.run_module( run_tavg     , "time_averager" );
      }

      // Update time step
      etime += dt; // Advance elapsed time
      coupler_main.set_option<real>("elapsed_time",etime);
      coupler_prec.set_option<real>("elapsed_time",etime);
      if (inform_freq >= 0. && inform_counter.update_and_check(dt)) {
        coupler_main.inform_user();
        inform_counter.reset();
      }
      if (out_freq    >= 0. && output_counter.update_and_check(dt)) {
        coupler_main.write_output_file( out_prefix , true );
        coupler_prec.write_output_file( out_prefix_prec , true );
        time_averager.reset(coupler_main);
        time_averager.reset(coupler_prec);
        output_counter.reset();
      }
    } // End main simulation loop

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}


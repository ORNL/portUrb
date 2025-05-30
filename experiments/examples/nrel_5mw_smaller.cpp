
#include "coupler.h"
#include "dynamics_rk_simpler.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "windmill_actuators_yaw.h"
#include "surface_flux.h"
#include "surface_heat_flux.h"
#include "precursor_sponge.h"
#include "sponge_layer.h"
#include "uniform_pg_wind_forcing.h"
#include "Ensembler.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    bool run_main = true ;

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler_main;
    core::Coupler coupler_prec;

    coupler_main.set_option<std::string>("ensemble_stdout","ensemble_nrel_5mw_smaller" );
    coupler_main.set_option<std::string>("out_prefix"     ,"turbulent_nrel_5mw_smaller");

    // This holds all of the model's variables, dimension sizes, and options
    core::Ensembler ensembler;

    // Add wind dimension
    {
      auto func_nranks  = [=] (int ind) { return 1; };
      auto func_coupler = [=] (int ind, core::Coupler &coupler) {
        real f_TKE = ind/10.;
        coupler.set_option<real>( "turbine_f_TKE" , f_TKE );
        ensembler.append_coupler_string(coupler,"ensemble_stdout",std::string("f_TKE-")+std::to_string(f_TKE));
        ensembler.append_coupler_string(coupler,"out_prefix"     ,std::string("f_TKE-")+std::to_string(f_TKE));
      };
      ensembler.register_dimension( 11 , func_nranks , func_coupler );
    }

    auto par_comm = ensembler.create_coupler_comm( coupler_main , 4 , MPI_COMM_WORLD );
    coupler_main.set_parallel_comm( par_comm );
    // // auto par_comm = ensembler.create_coupler_comm( coupler_main , 12 , MPI_COMM_WORLD );

    auto ostr = std::ofstream(coupler_main.get_option<std::string>("ensemble_stdout")+std::string(".out"));
    auto orig_cout_buf = std::cout.rdbuf();
    auto orig_cerr_buf = std::cerr.rdbuf();
    std::cout.rdbuf(ostr.rdbuf());
    std::cerr.rdbuf(ostr.rdbuf());

    real dx = 10;

    std::string turbine_file = "./inputs/NREL_5MW_126_RWT_amrwind.yaml";
    YAML::Node config = YAML::LoadFile( turbine_file );
    if ( !config ) { endrun("ERROR: Invalid turbine input file"); }
    real D     = config["blade_radius"].as<real>()*2;
    real hub_z = config["hub_height"  ].as<real>();

    real        sim_time          = 3600*5;
    real        xlen              = 2200;
    real        ylen              = 800;
    real        zlen              = 600;
    int         nx_glob           = std::ceil(xlen/dx);    xlen = nx_glob * dx;
    int         ny_glob           = std::ceil(ylen/dx);    ylen = ny_glob * dx;
    int         nz                = std::ceil(zlen/dx);    zlen = nz      * dx;
    real        dtphys_in         = 0;  // Determined by dycore CFL restriction
    std::string init_data         = "ABL_neutral";
    real        out_freq          = 900;
    real        inform_freq       = 10;
    std::string out_prefix_prec   = coupler_main.get_option<std::string>("out_prefix")+std::string("_precursor");
    bool        is_restart        = false;
    std::string restart_file      = "";
    std::string restart_file_prec = "";
    real        latitude          = 0;
    real        roughness         = 0.01;
    int         dyn_cycle         = 1;
    real        vort_freq         = -1;
    real        hub_u             = 11.4;
    real        hub_v             = 0;

    // Things the coupler_main might need to know about
    coupler_main.set_option<std::string>( "init_data"                , init_data         );
    coupler_main.set_option<real       >( "out_freq"                 , out_freq          );
    coupler_main.set_option<bool       >( "is_restart"               , is_restart        );
    coupler_main.set_option<std::string>( "restart_file"             , restart_file      );
    coupler_main.set_option<std::string>( "restart_file_precursor"   , restart_file_prec );
    coupler_main.set_option<real       >( "latitude"                 , latitude          );
    coupler_main.set_option<real       >( "roughness"                , roughness         );
    coupler_main.set_option<std::string>( "turbine_file"             , turbine_file      );
    coupler_main.set_option<bool       >( "turbine_do_blades"        , false             );
    coupler_main.set_option<real       >( "turbine_initial_yaw"      , 0.                );
    coupler_main.set_option<bool       >( "turbine_fixed_yaw"        , true              );
    coupler_main.set_option<bool       >( "turbine_floating_motions" , false             );
    coupler_main.set_option<bool       >( "turbine_immerse_material" , false             );
    coupler_main.set_option<real       >( "hub_height_uvel"          , hub_u             );
    coupler_main.set_option<real       >( "hub_height_vvel"          , hub_v             );
    coupler_main.set_option<real       >( "kinematic_viscosity"      , 0                 );
    coupler_main.set_option<real       >( "dycore_max_wind"          , 40                );
    coupler_main.set_option<real       >( "cfl"                      , 0.7               );
    coupler_main.set_option<bool       >( "turbine_orig_C_T"         , true              );

    if (coupler_main.is_mainproc()) {
      std::cout << "Prefix:    " << coupler_main.get_option<std::string>("out_prefix") << std::endl;
      std::cout << "Domain:    " << xlen/D << " x " << ylen/D << " x " << zlen/D << std::endl;
      std::cout << "Time:      " << sim_time/3600 << std::endl;
      std::cout << "Wind:      " << std::sqrt(hub_u*hub_u+hub_v*hub_v) << std::endl;
      std::cout << "Tower:     " << coupler_main.get_option<bool>("turbine_immerse_material") << std::endl;
      std::cout << "Blades:    " << coupler_main.get_option<bool>("turbine_do_blades"       ) << std::endl;
      std::cout << "Yaw Fixed: " << coupler_main.get_option<bool>("turbine_fixed_yaw"       ) << std::endl;
      std::cout << "Yaw Angle: " << coupler_main.get_option<real>("turbine_initial_yaw"     ) << std::endl;
    }

    // Set the turbine
    coupler_main.set_option<std::vector<real>>("turbine_x_locs"      ,{500});
    coupler_main.set_option<std::vector<real>>("turbine_y_locs"      ,{ylen/2});
    coupler_main.set_option<std::vector<bool>>("turbine_apply_thrust",{true});

    // Coupler state is: (1) dry density;  (2) u-velocity;  (3) v-velocity;  (4) w-velocity;  (5) temperature
    //                   (6+) tracer masses (*not* mixing ratios!); and Option elapsed_time init to zero
    coupler_main.distribute_mpi_and_allocate_coupled_state( par_comm , nz, ny_glob, nx_glob);

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

    coupler_main.set_option<std::string>("bc_x1","open");
    coupler_main.set_option<std::string>("bc_x2","open");
    coupler_main.set_option<std::string>("bc_y1","open");
    coupler_main.set_option<std::string>("bc_y2","open");
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
      if (run_main) coupler_main.write_output_file( coupler_main.get_option<std::string>("out_prefix") );
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
    real pgu_sum=0;
    real pgv_sum=0;
    int  n_pg=0;
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
        using modules::uniform_pg_wind_forcing_specified;
        real h = coupler_prec.get_option<real>("turbine_hub_height");
        real u = coupler_prec.get_option<real>("hub_height_uvel");
        real v = coupler_prec.get_option<real>("hub_height_vvel");
        if (etime < 15000) {
          coupler_prec.run_module( [&] (Coupler &c) { std::tie(pgu,pgv) = uniform_pg_wind_forcing_height(c,dt,h,u,v,100); } , "pg_forcing" );
          if (etime >= 14000) {
            pgu_sum += pgu;
            pgv_sum += pgv;
            n_pg++;
          }
        } else {
          pgu = pgu_sum/n_pg;
          pgv = pgv_sum/n_pg;
          coupler_prec.run_module( [&] (Coupler &c) { uniform_pg_wind_forcing_specified(c,dt,pgu,pgv); } , "pg_forcing" );
        }
        // coupler_prec.run_module( [&] (Coupler &c) { modules::sponge_layer            (c,dt,dt*100,0.1);} , "top_sponge"     );
        coupler_prec.run_module( [&] (Coupler &c) { dycore.time_step                 (c,dt);           } , "dycore"         );
        coupler_prec.run_module( [&] (Coupler &c) { modules::apply_surface_fluxes    (c,dt);           } , "surface_fluxes" );
        coupler_prec.run_module( [&] (Coupler &c) { les_closure.apply                (c,dt);           } , "les_closure"    );
        coupler_prec.run_module( [&] (Coupler &c) { time_averager.accumulate         (c,dt);           } , "time_averager"  );
      }
      if (run_main) {
        using core::Coupler;
        using modules::uniform_pg_wind_forcing_specified;
        custom_modules::precursor_sponge( coupler_main , coupler_prec , {"uvel","vvel","wvel","temp"} ,
                                          (int) (0.1*nx_glob) , 0 , 0 , 0 );
        coupler_main.run_module( [&] (Coupler &c) { uniform_pg_wind_forcing_specified(c,dt,pgu,pgv);   } , "pg_forcing"     );
        // coupler_main.run_module( [&] (Coupler &c) { modules::sponge_layer            (c,dt,dt*100,0.1);} , "top_sponge"     );
        coupler_main.run_module( [&] (Coupler &c) { dycore.time_step                 (c,dt);           } , "dycore"         );
        coupler_main.run_module( [&] (Coupler &c) { modules::apply_surface_fluxes    (c,dt);           } , "surface_fluxes" );
        coupler_main.run_module( [&] (Coupler &c) { windmills.apply                  (c,dt);           } , "windmills"      );
        coupler_main.run_module( [&] (Coupler &c) { les_closure.apply                (c,dt);           } , "les_closure"    );
        coupler_main.run_module( [&] (Coupler &c) { time_averager.accumulate         (c,dt);           } , "time_averager"  );
      }

      // Update time step
      etime += dt; // Advance elapsed time
      coupler_main.set_option<real>("elapsed_time",etime);
      coupler_prec.set_option<real>("elapsed_time",etime);
      if (inform_freq >= 0. && inform_counter.update_and_check(dt)) {
        if (run_main) { coupler_main.inform_user(); }
        else          { coupler_prec.inform_user(); }
        inform_counter.reset();
      }
      if (out_freq    >= 0. && output_counter.update_and_check(dt)) {
        if (run_main) coupler_main.write_output_file( coupler_main.get_option<std::string>("out_prefix") , true );
        coupler_prec.write_output_file( out_prefix_prec , true );
        if (run_main) time_averager.reset(coupler_main);
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



#include "coupler.h"
#include "dynamics_rk_simpler.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "windmill_actuators_yaw.h"
#include "surface_flux.h"
#include "precursor_sponge.h"
#include "uniform_pg_wind_forcing.h"
#include "Ensembler.h"
#include "column_nudging.h"
#include "fluctuation_scaling.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    std::string turbine_file      = "./inputs/NREL_5MW_126_RWT_amrwind.yaml";
    YAML::Node  node              = YAML::LoadFile(turbine_file);
    real        turbine_hubz      = node["hub_height"  ].as<real>();
    real        turbine_rad       = node["blade_radius"].as<real>();
    real        D                 = turbine_rad*2;

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler_main;
    core::Coupler coupler_prec;

    coupler_main.set_option<std::string>("ensemble_stdout","ensemble_3x10" );
    coupler_main.set_option<std::string>("out_prefix"     ,"turbulent_3x10");
    coupler_main.set_option<real       >("hub_height_wind_mag",10);
    real z0 = 0.0001;
    real wind = coupler_main.get_option<real>("hub_height_wind_mag");
    for (int iter=0; iter < 100; iter++) {
      real ustar = 0.4*wind/std::log(turbine_hubz/z0);
      z0 = 0.018*ustar*ustar/9.81;
    }
    coupler_main.set_option<real>("roughness",z0);

    // This holds all of the model's variables, dimension sizes, and options
    core::Ensembler ensembler;

    // Add floating dimension
    {
      auto func_nranks  = [=] (int ind) { return 1; };
      auto func_coupler = [=] (int ind, core::Coupler &coupler) {
        if (ind == 0) {
          coupler.set_option<bool>( "turbine_floating_motions" , false );
          ensembler.append_coupler_string(coupler,"ensemble_stdout",std::string("fixed-"));
          ensembler.append_coupler_string(coupler,"out_prefix"     ,std::string("fixed-"));
        } else {
          coupler.set_option<bool>( "turbine_floating_motions" , true );
          ensembler.append_coupler_string(coupler,"ensemble_stdout",std::string("floating-"));
          ensembler.append_coupler_string(coupler,"out_prefix"     ,std::string("floating-"));
        }
      };
      ensembler.register_dimension( 2 , func_nranks , func_coupler );
    }

    auto par_comm = ensembler.create_coupler_comm( coupler_main , 64 , MPI_COMM_WORLD );
    coupler_main.set_parallel_comm( par_comm );

    auto orig_cout_buf = std::cout.rdbuf();
    auto orig_cerr_buf = std::cerr.rdbuf();
    std::ofstream ostr(coupler_main.get_option<std::string>("ensemble_stdout")+std::string(".out"));
    std::cout.rdbuf(ostr.rdbuf());
    std::cerr.rdbuf(ostr.rdbuf());

    real dx = 10;

    if (par_comm.valid()) {
      yakl::timer_start("main");

      real        sim_time          = 3600*8+1;
      std::string init_data         = "ABL_neutral2";
      real        out_freq          = 1800;
      real        inform_freq       = 10;
      int         nrows             = 3;
      int         ncols             = 10;
      int         pad_x1            = 1;
      int         pad_x2            = 3;
      int         pad_y1            = 1;
      int         pad_y2            = 1;
      real        spacing           = turbine_rad*2*10;  // 10 diameters
      real        xlen              = (ncols+pad_x1+pad_x2-1)*spacing;
      real        ylen              = (nrows+pad_y1+pad_y2-1)*spacing;
      real        zlen              = 800;
      int         nx_glob           = (int) std::round(xlen/dx);
      int         ny_glob           = (int) std::round(ylen/dx);
      int         nz                = (int) std::round(zlen/dx);
      real        dtphys_in         = 0.;  // Dycore determined time step size
      int         dyn_cycle         = 4;
      std::string out_prefix        = coupler_main.get_option<std::string>("out_prefix");
      std::string out_prefix_prec   = out_prefix+std::string("_precursor");
      std::string restart_file      = "";
      std::string restart_file_prec = "";
      bool        run_main          = true;
      coupler_main.set_option<std::string>( "init_data"                , init_data         );
      coupler_main.set_option<real       >( "out_freq"                 , out_freq          );
      coupler_main.set_option<std::string>( "restart_file"             , restart_file      );
      coupler_main.set_option<std::string>( "restart_file_precursor"   , restart_file_prec );
      coupler_main.set_option<real       >( "latitude"                 , 0.                );
      coupler_main.set_option<real       >( "dycore_max_wind"          , 50                );
      coupler_main.set_option<real       >( "cfl"                      , 0.7               );
      // Turbine parameterization options
      coupler_main.set_option<std::string>( "turbine_file"             , turbine_file      );
      coupler_main.set_option<bool       >( "turbine_do_blades"        , false             );
      coupler_main.set_option<real       >( "turbine_initial_yaw"      , 0                 );
      coupler_main.set_option<bool       >( "turbine_fixed_yaw"        , true              );
      coupler_main.set_option<real       >( "turbine_upstream_dir"     , 0                 );
      coupler_main.set_option<bool       >( "turbine_immerse_material" , false             );
      coupler_main.set_option<bool       >( "turbine_orig_C_T"         , true              );
      coupler_main.set_option<real       >( "turbine_f_TKE"            , 0.25              );
      // Place the turbines
      std::vector<real> turbine_x_locs;
      std::vector<real> turbine_y_locs;
      std::vector<bool> turbine_thrust;
      for (int j=0; j < nrows; j++) {
        for (int i=0; i < ncols; i++) {
          turbine_x_locs.push_back( (pad_x1+i)*spacing );
          turbine_y_locs.push_back( (pad_y1+j)*spacing );
          turbine_thrust.push_back( true               );
        }
      }
      coupler_main.set_option<std::vector<real>>("turbine_x_locs"      ,turbine_x_locs);
      coupler_main.set_option<std::vector<real>>("turbine_y_locs"      ,turbine_y_locs);
      coupler_main.set_option<std::vector<bool>>("turbine_apply_thrust",turbine_thrust);

      // Coupler state is: (1) dry density;  (2) u-velocity;  (3) v-velocity;  (4) w-velocity;  (5) temperature
      //                   (6+) tracer masses (*not* mixing ratios!); and Option elapsed_time init to zero
      coupler_main.distribute_mpi_and_allocate_coupled_state( par_comm , nz, ny_glob, nx_glob);

      // Just tells the coupler how big the domain is in each dimensions
      coupler_main.set_grid( xlen , ylen , zlen );

      // No microphysics specified, so create a water_vapor tracer required by the dycore
      coupler_main.add_tracer("water_vapor","water_vapor",true,true ,true);
      coupler_main.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

      // Classes
      modules::LES_Closure                       les_closure;
      modules::Dynamics_Euler_Stratified_WenoFV  dycore;
      modules::Time_Averager                     time_averager;
      modules::WindmillActuators                 windmills;
      modules::ColumnNudger                      col_nudge_prec;
      modules::ColumnNudger                      col_nudge_main;

      // Run the initialization modules on coupler_main
      custom_modules::sc_init( coupler_main );
      les_closure  .init     ( coupler_main );
      dycore       .init     ( coupler_main );

      /////////////////////////////////////////////////////////////////////////
      // Everything previous to this is now replicated in coupler_precursor
      // From here out, the will be treated separately
      coupler_main.clone_into(coupler_prec);
      /////////////////////////////////////////////////////////////////////////

      custom_modules::sc_perturb( coupler_main );
      custom_modules::sc_perturb( coupler_prec );

      coupler_prec.set_option<std::string>("bc_x1","periodic");
      coupler_prec.set_option<std::string>("bc_x2","periodic");
      coupler_prec.set_option<std::string>("bc_y1","periodic");
      coupler_prec.set_option<std::string>("bc_y2","periodic");
      coupler_prec.set_option<std::string>("bc_z1","wall_free_slip");
      coupler_prec.set_option<std::string>("bc_z2","wall_free_slip");

      coupler_main.set_option<std::string>("bc_x1","precursor");
      coupler_main.set_option<std::string>("bc_x2","open");
      coupler_main.set_option<std::string>("bc_y1","periodic");
      coupler_main.set_option<std::string>("bc_y2","periodic");
      coupler_main.set_option<std::string>("bc_z1","wall_free_slip");
      coupler_main.set_option<std::string>("bc_z2","wall_free_slip");

      windmills    .init( coupler_main );
      time_averager.init( coupler_main );
      time_averager.init( coupler_prec );

      // Get elapsed time (zero), and create counters for output and informing the user in stdout
      real etime = coupler_main.get_option<real>("elapsed_time");
      core::Counter output_counter( out_freq    , etime );
      core::Counter inform_counter( inform_freq , etime );

      // if restart, overwrite with restart data, and set the counters appropriately. Otherwise, write initial output
      if (restart_file != "" && restart_file != "null") {
        coupler_main.overwrite_with_restart();
        etime = coupler_main.get_option<real>("elapsed_time");
        output_counter = core::Counter( out_freq    , etime-((int)(etime/out_freq   ))*out_freq    );
        inform_counter = core::Counter( inform_freq , etime-((int)(etime/inform_freq))*inform_freq );
      } else {
        if (out_freq >= 0 && run_main) coupler_main.write_output_file( out_prefix );
      }

      // if restart, overwrite with restart data, and set the counters appropriately. Otherwise, write initial output
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
      while (etime < sim_time) {
        // If dt <= 0, then set it to the dynamical core's max stable time step
        if (dtphys_in <= 0.) { dt = dycore.compute_time_step(coupler_main)*dyn_cycle; }
        // If we're about to go past the final time, then limit to time step to exactly hit the final time
        if (etime + dt > sim_time) { dt = sim_time - etime; }

        // Run modules
        using core::Coupler;
        using modules::uniform_pg_wind_forcing_height;
        using modules::uniform_pg_wind_forcing_specified;
        using modules::fluctuation_scaling;

        real pgu, pgv;
        {
          real h = turbine_hubz;
          real u = coupler_prec.get_option<real>("hub_height_wind_mag");
          real v = 0;

          coupler_prec.run_module( [&] (Coupler &c) { std::tie(pgu,pgv) = uniform_pg_wind_forcing_height(c,dt,h,u,v,10); } , "pg_forcing" );
          coupler_prec.run_module( [&] (Coupler &c) { dycore.time_step             (c,dt); } , "dycore"         );
          coupler_prec.run_module( [&] (Coupler &c) { modules::apply_surface_fluxes(c,dt); } , "surface_fluxes" );
          coupler_prec.run_module( [&] (Coupler &c) { les_closure.apply            (c,dt); } , "les_closure"    );
          coupler_prec.run_module( [&] (Coupler &c) { time_averager.accumulate     (c,dt); } , "time_averager"  );
        }

        col_nudge_prec.set_column( coupler_prec , {"density_dry","temp"} );
        col_nudge_main.column = col_nudge_prec.column;
        col_nudge_main.names  = col_nudge_prec.names;

        if (run_main) {
          // auto &dm_prec = coupler_prec.get_data_manager_readwrite();
          // auto u0 = dm_prec.get<real const,3>("uvel").createDeviceCopy();
          // auto v0 = dm_prec.get<real const,3>("vvel").createDeviceCopy();
          // auto w0 = dm_prec.get<real const,3>("wvel").createDeviceCopy();
          // coupler_prec.run_module( [&] (Coupler &c) { fluctuation_scaling(c,dt,0.25,dt,{"uvel","vvel","wvel"}); } , "fluct_scaling"  );
          modules::precursor_sponge( coupler_main , coupler_prec , {"density_dry","uvel","vvel","wvel","temp"} ,
                                     nx_glob/20 , 0 , 0 , 0 );
          modules::precursor_sponge( coupler_main , coupler_prec , {"density_dry","temp"} ,
                                     0 , nx_glob/20 , 0 , 0 );
          // u0.deep_copy_to(dm_prec.get<real,3>("uvel"));
          // v0.deep_copy_to(dm_prec.get<real,3>("vvel"));
          // w0.deep_copy_to(dm_prec.get<real,3>("wvel"));
          coupler_main.run_module( [&] (Coupler &c) { uniform_pg_wind_forcing_specified(c,dt,pgu,pgv); } , "pg_forcing" );
          coupler_main.run_module( [&] (Coupler &c) { col_nudge_main.nudge_to_column(c,dt,dt*100); } , "col_nudge"  );
          coupler_main.run_module( [&] (Coupler &c) { dycore.time_step              (c,dt); } , "dycore"            );
          coupler_main.run_module( [&] (Coupler &c) { modules::apply_surface_fluxes (c,dt); } , "surface_fluxes"    );
          coupler_main.run_module( [&] (Coupler &c) { windmills.apply               (c,dt); } , "windmillactuators" );
          coupler_main.run_module( [&] (Coupler &c) { les_closure.apply             (c,dt); } , "les_closure"       );
          coupler_main.run_module( [&] (Coupler &c) { time_averager.accumulate      (c,dt); } , "time_averager"     );
        }

        // Update time step
        etime += dt; // Advance elapsed time
        coupler_main.set_option<real>("elapsed_time",etime);
        coupler_prec.set_option<real>("elapsed_time",etime);
        if (inform_freq >= 0. && inform_counter.update_and_check(dt)) {
          if (run_main) { std::cout << "MAIN: "; coupler_main.inform_user(); }
          std::cout << "PREC: "; coupler_prec.inform_user();
          inform_counter.reset();
        }
        if (out_freq    >= 0. && output_counter.update_and_check(dt)) {
          if (run_main) coupler_main.write_output_file( out_prefix , true );
          coupler_prec.write_output_file( out_prefix_prec , true );
          time_averager.reset(coupler_main);
          time_averager.reset(coupler_prec);
          output_counter.reset();
        }
      } // End main simulation loop
      yakl::timer_stop("main");
    } // if (par_comm.valid()) 

    std::cout.rdbuf(orig_cout_buf);
    std::cerr.rdbuf(orig_cerr_buf);
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}


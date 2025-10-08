
#include "coupler.h"
#include "dynamics_rk_simpler.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "windmill_actuators_yaw.h"
#include "surface_flux.h"
#include "uniform_pg_wind_forcing.h"
#include "Ensembler.h"
#include "column_nudging.h"
#include "fluctuation_scaling.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    /////////////////////////////////////////////////////////
    // IMPORTANT PARAMETERS
    /////////////////////////////////////////////////////////
    real constexpr z0    = 1.e-5;
    real constexpr dx    = 10;

    std::string turbine_file      = "./inputs/NREL_5MW_126_RWT_amrwind.yaml";
    YAML::Node  node              = YAML::LoadFile(turbine_file);
    real        turbine_hubz      = node["hub_height"  ].as<real>();
    real        turbine_rad       = node["blade_radius"].as<real>();
    real        D                 = turbine_rad*2;

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler_prec;

    coupler_prec.set_option<std::string>("ensemble_stdout",std::string("ensemble_")+std::to_string(z0));
    coupler_prec.set_option<std::string>("out_prefix"     ,std::string("windfarm_")+std::to_string(z0));

    // This holds all of the model's variables, dimension sizes, and options
    core::Ensembler ensembler;

    // Add wind dimension
    {
      auto func_nranks  = [=] (int ind) { return 1; };
      auto func_coupler = [=] (int ind, core::Coupler &coupler) {
        real wind = ind+2;
        coupler.set_option<real>("hub_height_wind_mag",wind);
        ensembler.append_coupler_string(coupler,"ensemble_stdout",std::string("wind-")+std::to_string(wind));
        ensembler.append_coupler_string(coupler,"out_prefix"     ,std::string("wind-")+std::to_string(wind));
      };
      ensembler.register_dimension( 25 , func_nranks , func_coupler );
    }

    auto par_comm = ensembler.create_coupler_comm( coupler_prec , 4 , MPI_COMM_WORLD );
    coupler_prec.set_parallel_comm( par_comm );

    auto orig_cout_buf = std::cout.rdbuf();
    auto orig_cerr_buf = std::cerr.rdbuf();
    std::ofstream ostr(coupler_prec.get_option<std::string>("ensemble_stdout")+std::string(".out"));
    std::cout.rdbuf(ostr.rdbuf());
    std::cerr.rdbuf(ostr.rdbuf());

    if (par_comm.valid()) {
      yakl::timer_start("main");

      if (coupler_prec.is_mainproc()) std::cout << "Ensemble memeber using an initial hub wind speed of ["
                                                << coupler_prec.get_option<real>("hub_height_wind_mag")
                                                << "] m/s" << std::endl;
      real        sim_time          = 3600*24+1;
      real        xlen              = 3000;
      real        ylen              = 3000;
      real        zlen              = 1000;
      int         nx_glob           = (int) std::round(xlen/dx);
      int         ny_glob           = (int) std::round(ylen/dx);
      int         nz                = (int) std::round(zlen/dx);
      real        dtphys_in         = 0.;  // Dycore determined time step size
      int         dyn_cycle         = 1;
      std::string init_data         = "ABL_neutral2";
      real        out_freq          = 3600;
      real        inform_freq       = 10;
      std::string out_prefix        = coupler_prec.get_option<std::string>("out_prefix");
      real        hub_wind          = coupler_prec.get_option<real>("hub_height_wind_mag");
      real        hub_dir           = M_PI/4.;
      coupler_prec.set_option<std::string      >( "init_data"                , init_data    );
      coupler_prec.set_option<real             >( "out_freq"                 , out_freq     );
      coupler_prec.set_option<real             >( "latitude"                 , 0.           );
      coupler_prec.set_option<std::string      >( "turbine_file"             , turbine_file );
      coupler_prec.set_option<bool             >( "turbine_floating_motions" , false        );
      coupler_prec.set_option<bool             >( "turbine_do_blades"        , false        );
      coupler_prec.set_option<real             >( "turbine_initial_yaw"      , M_PI/4.      );
      coupler_prec.set_option<bool             >( "turbine_fixed_yaw"        , false        );
      coupler_prec.set_option<bool             >( "turbine_immerse_material" , false        );
      coupler_prec.set_option<bool             >( "turbine_orig_C_T"         , true         );
      coupler_prec.set_option<real             >( "turbine_f_TKE"            , 0.25         );
      coupler_prec.set_option<real             >( "roughness"                , z0           );
      coupler_prec.set_option<real             >( "cfl"                      , 0.6          );
      coupler_prec.set_option<real             >( "dycore_max_wind"          , 50           );
      coupler_prec.set_option<bool             >( "dycore_buoyancy_theta"    , true         );
      coupler_prec.set_option<real             >( "dycore_cs"                , 60           );
      coupler_prec.set_option<std::vector<real>>( "turbine_x_locs"           , {4*D}        );
      coupler_prec.set_option<std::vector<real>>( "turbine_y_locs"           , {4*D}        );
      coupler_prec.set_option<std::vector<bool>>( "turbine_apply_thrust"     , {true}       );
      coupler_prec.set_option<real             >( "hub_height_wind_dir"      , hub_dir      );

      if (coupler_prec.is_mainproc()) std::cout << "z0:   " << z0       << "\n"
                                                << "uhub: " << hub_wind << std::endl;

      coupler_prec.init( par_comm ,
                         coupler_prec.generate_levels_equal(nz,zlen) ,
                         ny_glob , nx_glob , ylen , xlen );

      // No microphysics specified, so create a water_vapor tracer required by the dycore
      coupler_prec.add_tracer("water_vapor","water_vapor",true,true ,true);
      coupler_prec.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

      // Classes that can work on multiple couplers without issue (no internal state)
      modules::LES_Closure                       les_closure;
      modules::Dynamics_Euler_Stratified_WenoFV  dycore;
      modules::Time_Averager                     time_averager;
      modules::WindmillActuators                 windmills;

      // Run the initialization modules on coupler_prec
      custom_modules::sc_init( coupler_prec );

      core::Coupler coupler_noturb;
      core::Coupler coupler_turb;

      /////////////////////////////////////////////////////////////////////////
      // Everything previous to this is now replicated in coupler_precursor
      // From here out, the will be treated separately
      coupler_prec.clone_into(coupler_noturb);
      coupler_prec.clone_into(coupler_turb);
      /////////////////////////////////////////////////////////////////////////

      // Boundaries must be declared before dycore.init(...)
      // Set precursor boundaries
      coupler_prec.set_option<bool>("dycore_is_precursor",true);
      coupler_prec.set_option<std::string>("bc_x1","periodic");
      coupler_prec.set_option<std::string>("bc_x2","periodic");
      coupler_prec.set_option<std::string>("bc_y1","periodic");
      coupler_prec.set_option<std::string>("bc_y2","periodic");
      coupler_prec.set_option<std::string>("bc_z1","wall_free_slip");
      coupler_prec.set_option<std::string>("bc_z2","wall_free_slip");

      // Set no-turbine boundaries
      coupler_noturb.set_option<std::string>("bc_x1","precursor");
      coupler_noturb.set_option<std::string>("bc_x2","precursor");
      coupler_noturb.set_option<std::string>("bc_y1","precursor");
      coupler_noturb.set_option<std::string>("bc_y2","precursor");
      coupler_noturb.set_option<std::string>("bc_z1","wall_free_slip");
      coupler_noturb.set_option<std::string>("bc_z2","wall_free_slip");

      // Set turbine boundaries
      coupler_turb.set_option<std::string>("bc_x1","precursor");
      coupler_turb.set_option<std::string>("bc_x2","precursor");
      coupler_turb.set_option<std::string>("bc_y1","precursor");
      coupler_turb.set_option<std::string>("bc_y2","precursor");
      coupler_turb.set_option<std::string>("bc_z1","wall_free_slip");
      coupler_turb.set_option<std::string>("bc_z2","wall_free_slip");

      // Initialize the modules (init les_closure before dycore so that SGS TKE is registered as a tracer)
      les_closure  .init        ( coupler_prec );
      dycore       .init        ( coupler_prec );
      time_averager.init        ( coupler_prec );
      custom_modules::sc_perturb( coupler_prec );

      les_closure  .init        ( coupler_noturb );
      dycore       .init        ( coupler_noturb );
      time_averager.init        ( coupler_noturb );
      custom_modules::sc_perturb( coupler_noturb );

      les_closure  .init        ( coupler_turb );
      dycore       .init        ( coupler_turb );
      time_averager.init        ( coupler_turb );
      custom_modules::sc_perturb( coupler_turb );
      windmills    .init        ( coupler_turb );

      // Get elapsed time (zero), and create counters for output and informing the user in stdout
      real etime = coupler_prec.get_option<real>("elapsed_time");
      core::Counter output_counter( out_freq    , etime );
      core::Counter inform_counter( inform_freq , etime );

      if (out_freq >= 0) coupler_prec  .write_output_file( out_prefix+std::string("_precursor") );
      if (out_freq >= 0) coupler_noturb.write_output_file( out_prefix+std::string("_noturbine") );
      if (out_freq >= 0) coupler_turb  .write_output_file( out_prefix+std::string("_turbine")   );

      // Begin main simulation loop over time steps
      real dt = dtphys_in;
      while (etime < sim_time) {
        // If dt <= 0, then set it to the dynamical core's max stable time step
        if (dtphys_in <= 0.) { dt = dycore.compute_time_step(coupler_prec)*dyn_cycle; }
        // If we're about to go past the final time, then limit to time step to exactly hit the final time
        if (etime + dt > sim_time) { dt = sim_time - etime; }

        real h = turbine_hubz;
        real u = hub_wind*std::cos(hub_dir);
        real v = hub_wind*std::sin(hub_dir);
        real pgu, pgv;

        // Run modules
        using core::Coupler;
        using modules::uniform_pg_wind_forcing_height;
        using modules::uniform_pg_wind_forcing_specified;

        real2d col;
        {
          coupler_prec.run_module( [&] (Coupler &c) { std::tie(pgu,pgv) = uniform_pg_wind_forcing_height(c,dt,h,u,v,dt*100); } , "pg_forcing" );
          coupler_prec.run_module( [&] (Coupler &c) { dycore.time_step             (c,dt); } , "dycore"         );
          coupler_prec.run_module( [&] (Coupler &c) { modules::apply_surface_fluxes(c,dt); } , "surface_fluxes" );
          coupler_prec.run_module( [&] (Coupler &c) { les_closure.apply            (c,dt); } , "les_closure"    );
          coupler_prec.run_module( [&] (Coupler &c) { time_averager.accumulate     (c,dt); } , "time_averager"  );
        }

        // We're going to force the forced simulations to maintain the precursor average col density and temperature
        modules::ColumnNudger col_nudge_prec;
        col_nudge_prec.set_column( coupler_prec , {"density_dry","temp"} );

        // Force domain avg col density and temperature, and copy in precursor ghost cells
        modules::ColumnNudger col_nudge_noturb;
        col_nudge_noturb.column = col_nudge_prec.column;
        col_nudge_noturb.names  = col_nudge_prec.names;
        dycore.copy_precursor_ghost_cells( coupler_prec , coupler_noturb );

        {
          coupler_noturb.run_module( [&] (Coupler &c) { uniform_pg_wind_forcing_specified(c,dt,pgu,pgv); } , "pg_forcing" );
          coupler_noturb.run_module( [&] (Coupler &c) { col_nudge_noturb.nudge_to_column(c,dt,dt*100); } , "col_nudge");
          coupler_noturb.run_module( [&] (Coupler &c) { dycore.time_step              (c,dt); } , "dycore"            );
          coupler_noturb.run_module( [&] (Coupler &c) { modules::apply_surface_fluxes (c,dt); } , "surface_fluxes"    );
          coupler_noturb.run_module( [&] (Coupler &c) { les_closure.apply             (c,dt); } , "les_closure"       );
          coupler_noturb.run_module( [&] (Coupler &c) { time_averager.accumulate      (c,dt); } , "time_averager"     );
        }

        // Force domain avg col density and temperature, and copy in precursor ghost cells
        modules::ColumnNudger col_nudge_turb;
        col_nudge_turb.column = col_nudge_prec.column;
        col_nudge_turb.names  = col_nudge_prec.names;
        dycore.copy_precursor_ghost_cells( coupler_prec , coupler_turb );

        {
          coupler_turb.run_module( [&] (Coupler &c) { uniform_pg_wind_forcing_specified(c,dt,pgu,pgv); } , "pg_forcing" );
          coupler_turb.run_module( [&] (Coupler &c) { col_nudge_turb.nudge_to_column(c,dt,dt*100); } , "col_nudge"  );
          coupler_turb.run_module( [&] (Coupler &c) { dycore.time_step              (c,dt); } , "dycore"            );
          coupler_turb.run_module( [&] (Coupler &c) { modules::apply_surface_fluxes (c,dt); } , "surface_fluxes"    );
          coupler_turb.run_module( [&] (Coupler &c) { windmills.apply               (c,dt); } , "windmillactuators" );
          coupler_turb.run_module( [&] (Coupler &c) { les_closure.apply             (c,dt); } , "les_closure"       );
          coupler_turb.run_module( [&] (Coupler &c) { time_averager.accumulate      (c,dt); } , "time_averager"     );
        }

        // Update time step
        etime += dt; // Advance elapsed time
        coupler_noturb.set_option<real>("elapsed_time",etime);
        coupler_prec  .set_option<real>("elapsed_time",etime);
        coupler_turb  .set_option<real>("elapsed_time",etime);
        if (inform_freq >= 0. && inform_counter.update_and_check(dt)) {
          if (coupler_prec  .is_mainproc()) std::cout << "PREC: ";
          coupler_prec.inform_user();
          if (coupler_noturb.is_mainproc()) std::cout << "NOTURB: ";
          coupler_noturb.inform_user();
          if (coupler_turb  .is_mainproc()) std::cout << "TURB: ";
          coupler_turb.inform_user();
          inform_counter.reset();
        }
        if (out_freq >= 0. && output_counter.update_and_check(dt)) {
          coupler_prec  .write_output_file( out_prefix+std::string("_precursor") , true );
          coupler_noturb.write_output_file( out_prefix+std::string("_noturbine") , true );
          coupler_turb  .write_output_file( out_prefix+std::string("_turbine")   , true );
          time_averager.reset(coupler_prec  );
          time_averager.reset(coupler_noturb);
          time_averager.reset(coupler_turb  );
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


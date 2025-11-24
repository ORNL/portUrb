
#include "coupler.h"
#include "dynamics_rk_simpler.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "turbine_fitch.h"
#include "surface_flux.h"
#include "uniform_pg_wind_forcing.h"
#include "precursor_sponge.h"
#include "Ensembler.h"
#include "column_nudging.h"
#include "fluctuation_scaling.h"
#include "sponge_layer.h"
#include "YAKL_netcdf.h"

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
    bool        is_restart        = false;
    int         restart_index     = 0;

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler;

    coupler.set_option<std::string>("ensemble_stdout",std::string("ensemble_z0_")+std::to_string(z0));
    coupler.set_option<std::string>("out_prefix"     ,std::string("windfarm_fitch_z0_")+std::to_string(z0));

    // This holds all of the model's variables, dimension sizes, and options
    core::Ensembler ensembler;

    // Add wind dimension
    {
      auto func_nranks  = [=] (int ind) { return 1; };
      auto func_coupler = [=] (int ind, core::Coupler &coupler) {
        real wind = 3+ind*2;
        coupler.set_option<real>("hub_height_wind_mag",wind);
        ensembler.append_coupler_string(coupler,"ensemble_stdout",std::string("wind-")+std::to_string(wind));
        ensembler.append_coupler_string(coupler,"out_prefix"     ,std::string("wind-")+std::to_string(wind));
      };
      ensembler.register_dimension( 12 , func_nranks , func_coupler );
    }

    auto par_comm = ensembler.create_coupler_comm( coupler , 2 , MPI_COMM_WORLD );
    coupler.set_parallel_comm( par_comm );

    auto orig_cout_buf = std::cout.rdbuf();
    auto orig_cerr_buf = std::cerr.rdbuf();
    std::ofstream ostr(coupler.get_option<std::string>("ensemble_stdout")+std::string(".out"));
    std::cout.rdbuf(ostr.rdbuf());
    std::cerr.rdbuf(ostr.rdbuf());

    if (par_comm.valid()) {
      yakl::timer_start("main");

      if (coupler.is_mainproc()) std::cout << "Ensemble memeber using an initial hub wind speed of ["
                                           << coupler.get_option<real>("hub_height_wind_mag")
                                           << "] m/s" << std::endl;
      real        sim_time          = 3600*24+1;
      real        xlen              = 510000;
      real        ylen              = 510000;
      real        zlen              = 5*D;
      int         nx_glob           = 51;
      int         ny_glob           = 51;
      int         nz                = (int) std::round(zlen/dx);
      real        dtphys_in         = 0.;  // Dycore determined time step size
      int         dyn_cycle         = 1;
      std::string init_data         = "constant";
      real        out_freq          = 3600;
      real        inform_freq       = 10;
      std::string out_prefix        = coupler.get_option<std::string>("out_prefix");
      real        hub_wind          = coupler.get_option<real>("hub_height_wind_mag");
      real        hub_dir           = 0;
      coupler.set_option<std::string      >( "init_data"                , init_data    );
      coupler.set_option<real             >( "out_freq"                 , out_freq     );
      coupler.set_option<real             >( "latitude"                 , 0.           );
      coupler.set_option<real             >( "constant_uvel"            , 10           );
      coupler.set_option<real             >( "constant_vvel"            , 0            );
      coupler.set_option<real             >( "constant_temp"            , 300          );
      coupler.set_option<real             >( "constant_press"           , 1.e5         );
      coupler.set_option<std::string      >( "turbine_file"             , turbine_file );
      coupler.set_option<bool             >( "turbine_floating_motions" , false        );
      coupler.set_option<bool             >( "turbine_do_blades"        , false        );
      coupler.set_option<real             >( "turbine_initial_yaw"      , 0            );
      coupler.set_option<bool             >( "turbine_fixed_yaw"        , false        );
      coupler.set_option<bool             >( "turbine_immerse_material" , false        );
      coupler.set_option<bool             >( "turbine_orig_C_T"         , true         );
      coupler.set_option<real             >( "turbine_f_TKE"            , 0.25         );
      coupler.set_option<real             >( "roughness"                , z0           );
      coupler.set_option<real             >( "cfl"                      , 0.6          );
      coupler.set_option<real             >( "dycore_max_wind"          , 50           );
      coupler.set_option<bool             >( "dycore_buoyancy_theta"    , true         );
      coupler.set_option<real             >( "dycore_cs"                , 60           );
      coupler.set_option<std::vector<real>>( "turbine_x_locs"           , {6*D}        );
      coupler.set_option<std::vector<real>>( "turbine_y_locs"           , {ylen/2}     );
      coupler.set_option<std::vector<bool>>( "turbine_apply_thrust"     , {true}       );
      coupler.set_option<real             >( "hub_height_wind_dir"      , hub_dir      );

      if (coupler.is_mainproc()) std::cout << "z0:   " << z0       << "\n"
                                           << "uhub: " << hub_wind << std::endl;

      coupler.init( par_comm ,
                    coupler.generate_levels_equal(nz,zlen) ,
                    ny_glob , nx_glob , ylen , xlen );

      // No microphysics specified, so create a water_vapor tracer required by the dycore
      coupler.add_tracer("water_vapor","water_vapor",true,true ,true);
      coupler.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

      // Classes that can work on multiple couplers without issue (no internal state)
      modules::LES_Closure                       les_closure;
      modules::Dynamics_Euler_Stratified_WenoFV  dycore;
      modules::Time_Averager                     time_averager;
      modules::TurbineFitch                      windmills;

      // Run the initialization modules on coupler
      custom_modules::sc_init( coupler );

      auto ghost_col = compute_average_ghost_column( core::Coupler & coupler );

      {
        using yakl::c::parallel_for;
        using yakl::c::SimpleBounds;
        yakl::SimpleNetCDF nc;
        nc.open("/lustre/storm/nwp501/scratch/imn/turbine_coarse_z0_0.000010.nc",yakl::NETCDF_MODE_READ);
        float1d winds;
        float2d noturb_u;
        nc.read( noturb_u , "noturb_u" );
        int  iwind = (int) std::round((coupler.get_option<real>("hub_height_wind_mag")-3)/2);
        auto uvel  = coupler.get_data_manager_readwrite().get<real,3>("uvel");
        auto vvel  = coupler.get_data_manager_readwrite().get<real,3>("vvel");
        auto wvel  = coupler.get_data_manager_readwrite().get<real,3>("wvel");
        auto nx    = coupler.get_nx();
        auto ny    = coupler.get_ny();
        auto nz    = coupler.get_nz();
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
          uvel(k,j,i) = noturb_u(iwind,k);
          vvel(k,j,i) = 0;
          wvel(k,j,i) = 0;
        });
      }

      // Boundaries must be declared before dycore.init(...)
      // Set precursor boundaries
      coupler.set_option<bool>("dycore_is_precursor",true);
      coupler.set_option<std::string>("bc_x1","periodic");
      coupler.set_option<std::string>("bc_x2","periodic");
      coupler.set_option<std::string>("bc_y1","periodic");
      coupler.set_option<std::string>("bc_y2","periodic");
      coupler.set_option<std::string>("bc_z1","wall_free_slip");
      coupler.set_option<std::string>("bc_z2","wall_free_slip");

      // Initialize the modules (init les_closure before dycore so that SGS TKE is registered as a tracer)
      les_closure  .init( coupler );
      dycore       .init( coupler );
      time_averager.init( coupler );
      windmills    .init( coupler );

      // Get elapsed time (zero), and create counters for output and informing the user in stdout
      real etime = coupler.get_option<real>("elapsed_time");
      core::Counter output_counter( out_freq    , etime );
      core::Counter inform_counter( inform_freq , etime );

      // if restart, overwrite with restart data, and set the counters appropriately. Otherwise, write initial output
      if (is_restart) {
        std::stringstream fname_prec;
        fname_prec << out_prefix << "_precursor_" << std::setw(8) << std::setfill('0') << restart_index << ".nc";
        coupler.set_option<std::string>( "restart_file" , fname_prec  .str() );
        coupler.overwrite_with_restart();
        etime          = coupler.get_option<real>("elapsed_time");
        output_counter = core::Counter( out_freq    , etime-((int)(etime/out_freq   ))*out_freq    );
        inform_counter = core::Counter( inform_freq , etime-((int)(etime/inform_freq))*inform_freq );
      } else {
        if (out_freq >= 0) coupler.write_output_file( out_prefix+std::string("_precursor") );
      }

      // Begin main simulation loop over time steps
      real dt = dtphys_in;
      while (etime < sim_time) {
        // If dt <= 0, then set it to the dynamical core's max stable time step
        if (dtphys_in <= 0.) { dt = dycore.compute_time_step(coupler)*dyn_cycle; }
        // If we're about to go past the final time, then limit to time step to exactly hit the final time
        if (etime + dt > sim_time) { dt = sim_time - etime; }

        real u = hub_wind*std::cos(hub_dir);
        real v = hub_wind*std::sin(hub_dir);

        // Run modules
        using core::Coupler;
        using modules::uniform_pg_wind_forcing_given;
        using modules::uniform_pg_wind_forcing_specified;
        using modules::precursor_sponge;

        real2d col;
        {
          real u_in,v_in;
          windmills.disk_average_wind( coupler , windmills.turbine_group.turbines.at(0).ref_turbine , u_in , v_in );
          coupler.run_module( [&] (Coupler &c) { uniform_pg_wind_forcing_given(c,dt,u_in,v_in,u,v,300); } , "pg_forcing" );
          coupler.run_module( [&] (Coupler &c) { dycore.time_step             (c,dt); } , "dycore"         );
          coupler.run_module( [&] (Coupler &c) { modules::sponge_layer        (c,dt,300,0.1); } , "sponge" );
          coupler.run_module( [&] (Coupler &c) { modules::apply_surface_fluxes(c,dt); } , "surface_fluxes" );
          coupler.run_module( [&] (Coupler &c) { windmills.apply              (c,dt); } , "windmills"      );
          coupler.run_module( [&] (Coupler &c) { les_closure.apply            (c,dt); } , "les_closure"    );
          coupler.run_module( [&] (Coupler &c) { time_averager.accumulate     (c,dt); } , "time_averager"  );
        }

        // Update time step
        etime += dt; // Advance elapsed time
        coupler.set_option<real>("elapsed_time",etime);
        if (inform_freq >= 0. && inform_counter.update_and_check(dt)) {
          if (coupler.is_mainproc()) std::cout << "PREC: ";
          coupler.inform_user();
          inform_counter.reset();
        }
        if (out_freq >= 0. && output_counter.update_and_check(dt)) {
          coupler.write_output_file( out_prefix+std::string("_precursor") , true );
          time_averager.reset(coupler);
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


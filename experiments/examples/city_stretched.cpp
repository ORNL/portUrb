
#include "coupler.h"
#include "dynamics_rk_simpler.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "surface_flux.h"
#include "sponge_layer.h"
#include "uniform_pg_wind_forcing.h"
#include "TriMesh.h"

/*
In blender, delete the initial objects.
Import opensteetmap, buildings only, as separate objects.
Then rotate to align with your grid and delete what you want.
Export to obj with all options turn off except for triangulate faces turned on, Y Forward, Z Up.
We only want triangle faces for simplicity.
This code will handle the rest.
*/

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    real pad_x1 = 20;
    real pad_x2 = 20;
    real pad_y1 = 20;
    real pad_y2 = 20;
    real dx     = 2;
    real dy     = 2;
    real dz     = 2;

    modules::TriMesh mesh;
    mesh.load_file("/ccs/home/imn/nyc2.obj");
    mesh.zero_domain_lo();

    real        sim_time    = 7201;
    real        xlen        = std::ceil((mesh.domain_hi.x+pad_x1+pad_x2)/dx)*dx;
    real        ylen        = std::ceil((mesh.domain_hi.y+pad_y1+pad_y2)/dy)*dy;
    real        zlen        = 1800;
    int         nx_glob     = xlen/dx;
    int         ny_glob     = ylen/dy;
    int         nz          = zlen/dz;
    real        dtphys_in   = 0;    // Use dycore time step
    int         dyn_cycle   = 4;
    real        out_freq    = 900;
    real        inform_freq = 10;
    std::string out_prefix  = "city_2m";
    bool        is_restart  = false;

    mesh.add_offset(pad_x1,pad_y1);

    core::Coupler coupler;
    coupler.set_option<std::string>( "out_prefix"         , out_prefix       );
    coupler.set_option<std::string>( "init_data"          , "city_stretched" );
    coupler.set_option<real       >( "out_freq"           , out_freq         );
    coupler.set_option<bool       >( "is_restart"         , is_restart       );
    coupler.set_option<std::string>( "restart_file"       , ""               );
    coupler.set_option<real       >( "latitude"           , 0.               );
    coupler.set_option<real       >( "roughness"          , 5e-2             );
    coupler.set_option<real       >( "building_roughness" , 5.e-2            );
    coupler.set_option<real       >( "cfl"                , 0.6              );
    coupler.set_option<bool       >( "enable_gravity"     , true             );
    coupler.set_option<bool       >( "weno_all"           , true             );
    coupler.set_option<real       >( "dycore_max_wind"    , 25               );
    coupler.set_option<real       >( "dycore_cs"          , 40               );

    coupler.init( core::ParallelComm(MPI_COMM_WORLD) ,
                  coupler.generate_levels_const_low_high(zlen,2,480,600,10) ,
                  ny_glob , nx_glob , ylen , xlen );

    int nfaces = mesh.faces.extent(0);
    coupler.get_data_manager_readwrite().register_and_allocate<float>("mesh_faces","",{nfaces,3,3});
    mesh.faces.deep_copy_to( coupler.get_data_manager_readwrite().get<float,3>("mesh_faces") );
    Kokkos::fence();
    if (coupler.is_mainproc()) std::cout << mesh;

    modules::Dynamics_Euler_Stratified_WenoFV  dycore;
    modules::Time_Averager                     time_averager;
    modules::LES_Closure                       les_closure;

    // No microphysics specified, so create a water_vapor tracer required by the dycore
    coupler.add_tracer("water_vapor","water_vapor",true,true ,true);
    coupler.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

    custom_modules::sc_init   ( coupler );
    les_closure  .init        ( coupler );
    dycore       .init        ( coupler );
    time_averager.init        ( coupler );
    custom_modules::sc_perturb( coupler );

    coupler.set_option<std::string>("bc_x1","periodic");
    coupler.set_option<std::string>("bc_x2","periodic");
    coupler.set_option<std::string>("bc_y1","periodic");
    coupler.set_option<std::string>("bc_y2","periodic");

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
      coupler.write_output_file( out_prefix , true );
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
        using modules::uniform_pg_wind_forcing_height;
        real umag = 10;
        real hr = 600;
        real ur = umag*std::cos(29./180.*M_PI);
        real vr = umag*std::sin(29./180.*M_PI);
        real tr = dt*100;
        coupler.run_module( [&] (Coupler &c) { uniform_pg_wind_forcing_height(c,dt,hr,ur,vr,tr); } , "pg_forcing"     );
        coupler.run_module( [&] (Coupler &c) { dycore.time_step              (c,dt);             } , "dycore"         );
        coupler.run_module( [&] (Coupler &c) { modules::sponge_layer         (c,dt,dt,0.05);     } , "sponge"         );
        coupler.run_module( [&] (Coupler &c) { modules::apply_surface_fluxes (c,dt);             } , "surface_fluxes" );
        coupler.run_module( [&] (Coupler &c) { les_closure.apply             (c,dt);             } , "les_closure"    );
        coupler.run_module( [&] (Coupler &c) { time_averager.accumulate      (c,dt);             } , "time_averager"  );
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



#include "coupler.h"
#include "dynamics_rk_simpler.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "surface_flux.h"
#include "uniform_pg_wind_forcing.h"
#include "TriMesh.h"
#include "tank_tracer_injection.h"

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

    real scale = 1./1250.;
    real dx = 0.3*scale;
    real u0 = 0.51;

    modules::TriMesh mesh;
    mesh.load_file("/ccs/home/imn/building_terrain_120deg.obj");
    mesh.zero_domain_lo();
    mesh.apply_scaling(scale,scale,scale);

    real        xlen        = std::ceil((mesh.domain_hi.x  )/dx)*dx;
    real        ylen        = std::ceil((mesh.domain_hi.y  )/dx)*dx;
    real        zlen        = std::ceil((mesh.domain_hi.z*5)/dx)*dx;
    real        sim_time    = xlen/u0*10;
    int         nx_glob     = xlen/dx;
    int         ny_glob     = ylen/dx;
    int         nz          = zlen/dx;
    real        dtphys_in   = 0;    // Use dycore time step
    int         dyn_cycle   = 4;
    real        out_freq    = xlen/u0/2;
    real        inform_freq = xlen/u0/20;
    std::string out_prefix  = "city_2m";
    bool        is_restart  = false;

    core::Coupler coupler;
    coupler.set_option<std::string>( "out_prefix"                         , out_prefix  );
    coupler.set_option<std::string>( "init_data"                          , "tank_set"  );
    coupler.set_option<real       >( "out_freq"                           , out_freq    );
    coupler.set_option<bool       >( "is_restart"                         , is_restart  );
    coupler.set_option<std::string>( "restart_file"                       , ""          );
    coupler.set_option<real       >( "latitude"                           , 0.          );
    coupler.set_option<real       >( "roughness"                          , dx/20.      );
    coupler.set_option<real       >( "init_density"                       , 1           );
    coupler.set_option<real       >( "init_temperature"                   , 300         );
    coupler.set_option<real       >( "init_uvel"                          , u0          );
    coupler.set_option<real       >( "init_vvel"                          , 0           );
    coupler.set_option<real       >( "cfl"                                , 0.6         );
    coupler.set_option<real       >( "dycore_max_wind"                    , 1.5         );
    coupler.set_option<real       >( "dycore_cs"                          , 4.5         );
    coupler.set_option<bool       >( "dycore_use_weno"                    , false       );
    coupler.set_option<bool       >( "dycore_use_weno_immersed"           , true        );
    coupler.set_option<bool       >( "dycore_buoyancy_theta"              , false       );
    coupler.set_option<bool       >( "dycore_immersed_hypervis"           , false       );
    coupler.set_option<real       >( "kinematic_viscosity"                , 2.e-6       );
    coupler.set_option<real       >( "les_closure_delta_multiplier"       , 0.3         );
    coupler.set_option<bool       >( "surface_flux_force_theta"           , false       );
    coupler.set_option<bool       >( "surface_flux_stability_corrections" , false       );

    coupler.init( core::ParallelComm(MPI_COMM_WORLD) ,
                  coupler.generate_levels_const_low_high(zlen,dx,11.2*scale,16*scale,dx*4) ,
                  ny_glob , nx_glob , ylen , xlen );

    int nfaces = mesh.faces.extent(0);
    coupler.get_data_manager_readwrite().register_and_allocate<float>("mesh_faces","",{nfaces,3,3});
    mesh.faces.deep_copy_to( coupler.get_data_manager_readwrite().get<float,3>("mesh_faces") );
    Kokkos::fence();
    if (coupler.is_mainproc()) std::cout << mesh;

    modules::Dynamics_Euler_Stratified_WenoFV  dycore;
    modules::SurfaceFlux                       sfc_flux;
    modules::Time_Averager                     time_averager;
    modules::LES_Closure                       les_closure;

    // No microphysics specified, so create a water_vapor tracer required by the dycore
    coupler.add_tracer("water_vapor","water_vapor",true,true ,true);
    coupler.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;
    coupler.add_tracer("tank_tracer","tank_tracer",true,false,true);
    coupler.get_data_manager_readwrite().get<real,3>("tank_tracer") = 0;

    custom_modules::sc_init   ( coupler );
    les_closure  .init        ( coupler );
    dycore       .init        ( coupler );
    sfc_flux     .init        ( coupler );
    time_averager.init        ( coupler );
    custom_modules::sc_perturb( coupler );

    real etime = coupler.get_option<real>("elapsed_time");
    core::Counter output_counter( out_freq    , etime );
    core::Counter inform_counter( inform_freq , etime );

    // if restart, overwrite with restart data, and set the counters appropriately. Otherwise, write initial output
    if (is_restart) {
      coupler.overwrite_with_restart();
      etime = coupler.get_option<real>("elapsed_time");
      output_counter = core::Counter( out_freq    , etime-((int)(etime/out_freq   ))*out_freq    );
      inform_counter = core::Counter( inform_freq , etime-((int)(etime/inform_freq))*inform_freq );
    } else if (out_freq >= 0) {
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
        using custom_modules::tank_tracer_injection;
        real x1   = 120*scale;
        real x2   = 140*scale;
        real y1   = 95 *scale;
        real y2   = 105*scale;
        real z1   = 1  *scale;
        real z2   = 2  *scale;
        real conc = 0.1;
        real wvel = 0.828932;
        coupler.run_module( [&] (Coupler &c) { tank_tracer_injection(c,dt,x1,x2,y1,y2,z1,z2,conc,wvel,"tank_tracer"); } , "tracer_inj" );
        real hr = 11.2*scale*2;
        real ur = u0;
        real vr = 0;
        real tr = xlen/u0;
        coupler.run_module( [&] (Coupler &c) { uniform_pg_wind_forcing_height(c,dt,hr,ur,vr,tr); } , "pg_forcing"     );
        coupler.run_module( [&] (Coupler &c) { dycore.time_step              (c,dt);             } , "dycore"         );
        // coupler.run_module( [&] (Coupler &c) { sfc_flux.apply                (c,dt);             } , "surface_fluxes" );
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


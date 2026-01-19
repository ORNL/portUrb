
#include "coupler.h"
#include "dynamics_rk_simpler.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "surface_flux.h"
#include "sponge_layer.h"
#include "column_nudging.h"
#include "precursor_sponge.h"
#include "geostrophic_wind_forcing.h"
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

    real dx = 4;
    real dy = 4;

    modules::TriMesh mesh;
    mesh.load_file("/ccs/home/imn/nyc2.obj");
    mesh.zero_domain_lo();

    real pad_x1 = mesh.domain_hi.x/2;
    real pad_x2 = mesh.domain_hi.x/2;
    real pad_y1 = mesh.domain_hi.y/2;
    real pad_y2 = mesh.domain_hi.y/2;

    real        sim_time          = 3600*10+1;
    real        xlen              = std::ceil((mesh.domain_hi.x+pad_x1+pad_x2)/dx)*dx;
    real        ylen              = std::ceil((mesh.domain_hi.y+pad_y1+pad_y2)/dy)*dy;
    real        zlen              = 1800;
    int         nx_glob           = xlen/dx;
    int         ny_glob           = ylen/dy;
    int         dyn_cycle         = 4;
    real        out_freq          = 900;
    real        inform_freq       = 10;
    std::string out_prefix_main   = "city_stretched";
    std::string out_prefix_prec   = out_prefix_main+std::string("_precursor");
    bool        is_restart        = false;
    std::string restart_file_main = "";
    std::string restart_file_prec = "";

    mesh.add_offset(pad_x1,pad_y1);
    DEBUG_PRINT_MAIN_VAL( mesh.domain_lo.z );

    core::Coupler coupler_main;
    coupler_main.set_option<std::string>( "init_data"          , "city_stretched"  );
    coupler_main.set_option<std::string>( "restart_file"       , restart_file_main );
    coupler_main.set_option<real       >( "latitude"           , 0                 );
    coupler_main.set_option<real       >( "roughness"          , 5.e-2             );
    coupler_main.set_option<real       >( "building_roughness" , 5.e-2             );
    coupler_main.set_option<real       >( "cfl"                , 0.6               );
    coupler_main.set_option<bool       >( "enable_gravity"     , true              );
    coupler_main.set_option<real       >( "dycore_max_wind"    , 25                );
    coupler_main.set_option<real       >( "dycore_cs"          , 60                );
    coupler_main.set_option<real       >( "geostrophic_u"      , 10.               );
    coupler_main.set_option<real       >( "geostrophic_v"      , 0.                );

    coupler_main.init( core::ParallelComm(MPI_COMM_WORLD) ,
                       coupler_main.generate_levels_const_low_high(zlen,4,480,600,10) ,
                       ny_glob , nx_glob , ylen , xlen );

    int nfaces = mesh.faces.extent(0);
    coupler_main.get_data_manager_readwrite().register_and_allocate<float>("mesh_faces","",{nfaces,3,3});
    mesh.faces.deep_copy_to( coupler_main.get_data_manager_readwrite().get<float,3>("mesh_faces") );
    Kokkos::fence();
    if (coupler_main.is_mainproc()) std::cout << mesh;

    modules::Dynamics_Euler_Stratified_WenoFV  dycore;
    modules::SurfaceFlux                       sfc_flux;
    modules::Time_Averager                     time_averager;
    modules::LES_Closure                       les_closure;
    modules::ColumnNudger                      col_nudge_prec;
    modules::ColumnNudger                      col_nudge_main;

    // No microphysics specified, so create a water_vapor tracer required by the dycore
    coupler_main.add_tracer("water_vapor","water_vapor",true,true ,true);
    coupler_main.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

    core::Coupler coupler_prec;
    coupler_main.clone_into(coupler_prec);
    coupler_prec.set_option<std::string>( "restart_file" , restart_file_prec );
    coupler_prec.set_option<std::string>( "init_data"    , "ABL_neutral"     );

    // Initialize the data for precursor and main simulations
    custom_modules::sc_init( coupler_prec );
    custom_modules::sc_init( coupler_main );

    // Set the boundary conditions on the precursor simulation
    coupler_prec.set_option<bool>("dycore_is_precursor",true);
    coupler_prec.set_option<std::string>("bc_x1","periodic");
    coupler_prec.set_option<std::string>("bc_x2","periodic");
    coupler_prec.set_option<std::string>("bc_y1","periodic");
    coupler_prec.set_option<std::string>("bc_y2","periodic");
    coupler_prec.set_option<std::string>("bc_z1","wall_free_slip");
    coupler_prec.set_option<std::string>("bc_z2","wall_free_slip");

    // Set the boundary conditions on the main simulation
    coupler_main.set_option<std::string>("bc_x1","precursor");
    coupler_main.set_option<std::string>("bc_x2","precursor");
    coupler_main.set_option<std::string>("bc_y1","precursor");
    coupler_main.set_option<std::string>("bc_y2","precursor");
    coupler_main.set_option<std::string>("bc_z1","wall_free_slip");
    coupler_main.set_option<std::string>("bc_z2","wall_free_slip");

    // Perform initializations on precursor simulation
    les_closure  .init        ( coupler_prec );
    dycore       .init        ( coupler_prec );
    sfc_flux     .init        ( coupler_prec );
    time_averager.init        ( coupler_prec );
    custom_modules::sc_perturb( coupler_prec );

    // Perform initializations on main simulation
    les_closure  .init        ( coupler_main );
    dycore       .init        ( coupler_main );
    sfc_flux     .init        ( coupler_main );
    time_averager.init        ( coupler_main );
    custom_modules::sc_perturb( coupler_main );

    real etime = coupler_main.get_option<real>("elapsed_time");
    core::Counter output_counter( out_freq    , etime );
    core::Counter inform_counter( inform_freq , etime );

    // if restart, overwrite with restart data, and set the counters appropriately. Otherwise, write initial output
    if (is_restart) {
      coupler_prec.overwrite_with_restart();
      coupler_main.overwrite_with_restart();
      etime = coupler_main.get_option<real>("elapsed_time");
      output_counter = core::Counter( out_freq    , etime-((int)(etime/out_freq   ))*out_freq    );
      inform_counter = core::Counter( inform_freq , etime-((int)(etime/inform_freq))*inform_freq );
    } else {
      coupler_prec.write_output_file( out_prefix_prec , true );
      coupler_main.write_output_file( out_prefix_main , true );
    }

    Kokkos::fence();
    auto tm = std::chrono::high_resolution_clock::now();
    while (etime < sim_time) {
      real dt = dycore.compute_time_step(coupler_main)*dyn_cycle;
      // If we're about to go past the final time, then limit to time step to exactly hit the final time
      if (etime + dt > sim_time) { dt = sim_time - etime; }

      using core::Coupler;
      using modules::geostrophic_wind_forcing;
      using modules::geostrophic_wind_forcing_specified;
      using modules::sponge_layer;
      using modules::precursor_sponge;
      real u_g   = 10;
      real v_g   = 0 ;
      real lat_g = 40.75;
      real2d col;

      // Run the precursor modules
      coupler_prec.run_module( [&] (Coupler &c) { col = geostrophic_wind_forcing(c,dt,lat_g,u_g,v_g); } , "pg_forcing" );
      coupler_prec.run_module( [&] (Coupler &c) { les_closure.apply        (c,dt);               } , "les_closure"    );
      coupler_prec.run_module( [&] (Coupler &c) { sponge_layer             (c,dt,300,0.05);      } , "sponge"         );
      coupler_prec.run_module( [&] (Coupler &c) { sfc_flux.apply           (c,dt);               } , "surface_fluxes" );
      coupler_prec.run_module( [&] (Coupler &c) { dycore.time_step         (c,dt);               } , "dycore"         );
      coupler_prec.run_module( [&] (Coupler &c) { time_averager.accumulate (c,dt);               } , "time_averager"  );

      // // Copy the precursor column to the main column nudger for nudging to the precursor column state
      col_nudge_prec.set_column( coupler_prec , {"density_dry","temp"} );
      col_nudge_main.column = col_nudge_prec.column;
      col_nudge_main.names  = col_nudge_prec.names;
      // Copy the precursor ghost cells to the main ghost cells
      dycore.copy_precursor_ghost_cells( coupler_prec , coupler_main );

      // Run the main modules using precursor ghost cells / open boundaries
      precursor_sponge( coupler_main , coupler_prec , {"density_dry","temp"} , 0 , nx_glob/10 , 0 , ny_glob/10 );
      coupler_main.run_module( [&] (Coupler &c) { col_nudge_main.nudge_to_column(c,dt,dt*100);   } , "col_nudge"  );
      coupler_main.run_module( [&] (Coupler &c) { geostrophic_wind_forcing_specified(c,dt,lat_g,u_g,v_g,col); } , "pg_forcing" );
      coupler_main.run_module( [&] (Coupler &c) { les_closure.apply        (c,dt);               } , "les_closure"    );
      coupler_main.run_module( [&] (Coupler &c) { sponge_layer             (c,dt,300,0.05);      } , "sponge"         );
      coupler_main.run_module( [&] (Coupler &c) { sfc_flux.apply           (c,dt);               } , "surface_fluxes" );
      coupler_main.run_module( [&] (Coupler &c) { dycore.time_step         (c,dt);               } , "dycore"         );
      coupler_main.run_module( [&] (Coupler &c) { time_averager.accumulate (c,dt);               } , "time_averager"  );

      // Update time step
      etime += dt; // Advance elapsed time
      coupler_prec.set_option<real>("elapsed_time",etime+dt);
      coupler_main.set_option<real>("elapsed_time",etime+dt);
      if (inform_freq >= 0. && inform_counter.update_and_check(dt)) {
        coupler_prec.inform_user();
        coupler_main.inform_user();
        inform_counter.reset();
      }
      if (out_freq    >= 0. && output_counter.update_and_check(dt)) {
        coupler_prec.write_output_file( out_prefix_prec , true );
        coupler_main.write_output_file( out_prefix_main , true );
        time_averager.reset(coupler_prec);
        time_averager.reset(coupler_main);
        output_counter.reset();
      }
    } // End main simulation loop

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}


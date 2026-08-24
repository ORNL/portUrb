#include PORTURB_DYCORE_HEADER
#include "sc_init.h"

int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  Kokkos::initialize();
  yakl::init();
  {
    int constexpr n = 6;
    real constexpr spacing = 20;
    core::Coupler coupler;
    coupler.set_option<std::string>("init_data","constant");
    coupler.set_option<real>("constant_uvel",0);
    coupler.set_option<real>("constant_vvel",0);
    coupler.set_option<real>("constant_temp",300);
    coupler.set_option<real>("constant_press",1.e5);
    coupler.set_option<bool>("enable_gravity",false);
    coupler.set_option<real>("cfl",0.6);
    coupler.set_option<real>("dycore_cs",20);
    coupler.set_option<real>("dycore_max_wind",1);
    coupler.set_option<std::string>("dycore_time_stepper","ssprk3");
    coupler.init(core::ParallelComm(MPI_COMM_WORLD),coupler.generate_levels_equal(n,n*spacing),
                 n,n,n*spacing,n*spacing);
    custom_modules::sc_init(coupler);
    coupler.set_option<std::string>("bc_x1","periodic");
    coupler.set_option<std::string>("bc_x2","periodic");
    coupler.set_option<std::string>("bc_y1","periodic");
    coupler.set_option<std::string>("bc_y2","periodic");
    coupler.set_option<std::string>("bc_z1","wall_free_slip");
    coupler.set_option<std::string>("bc_z2","wall_free_slip");

    if (coupler.get_num_tracers() != 0) endrun("ERROR: zero-tracer test unexpectedly registered a tracer");
    if (coupler.get_data_manager_readonly().entry_exists("water_vapor")) {
      endrun("ERROR: zero-tracer test unexpectedly allocated water_vapor");
    }

    modules::Dynamics_Euler_Stratified dycore;
    dycore.init(coupler);
    dycore.time_step(coupler,0.01);
    auto &dm = coupler.get_data_manager_readonly();
    if (dm.get<bool const,1>("tracer_adds_mass").extent(0) != 1) {
      endrun("ERROR: zero-tracer metadata did not use one safe internal storage slot");
    }
    dm.validate_all(true,__FILE__,__LINE__,coupler.get_myrank());
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}

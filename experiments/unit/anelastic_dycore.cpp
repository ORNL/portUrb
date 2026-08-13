#include "coupler.h"
#include "dynamics_edge_centered_anelastic.h"
#include "sc_init.h"

namespace {

template <class Array>
real max_abs(core::Coupler const & coupler, Array const & field) {
  auto flat = field.collapse();
  real1d work("anelastic_test_abs",flat.size());
  yakl::parallel_for(YAKL_AUTO_LABEL(), flat.size(), KOKKOS_LAMBDA (int i) { work(i) = std::abs(flat(i)); });
  return coupler.get_parallel_comm().all_reduce(yakl::intrinsics::maxval(work),MPI_MAX);
}

void require(core::Coupler const & coupler, bool condition, std::string const & message) {
  int const valid = coupler.get_parallel_comm().all_reduce(condition ? 1 : 0,MPI_MIN);
  if (valid == 0) endrun(message.c_str());
}

void check_pressure_face_derivative_order(core::Coupler & coupler) {
  int constexpr order = modules::Dynamics_Euler_Stratified::ord;
  auto derivative_error = [] (real spacing) {
    SArray<real,order> cell_averages;
    for (int ii = 0; ii < order; ii++) {
      int const cell = ii-order/2;
      cell_averages(ii) = (std::cos(cell*spacing)-std::cos((cell+1)*spacing))/spacing;
    }
    return std::abs(TransformMatrices::edge_der(cell_averages)/spacing-1);
  };
  real const coarse_error = derivative_error(0.4_fp);
  real const fine_error = derivative_error(0.2_fp);
  real const observed_order = std::log2(coarse_error/fine_error);
  coupler.set_option<real>("dycore_anelastic_pressure_face_derivative_order",observed_order);
  require(coupler,observed_order > order-0.5_fp,"high-order pressure face derivative failed its convergence check");
}

void run_case(std::string const & name, int flow, bool with_immersed, int n = 8, real grid_spacing = 1,
              int cube_width = 2, int cube_k_beg = 2, bool run_invariance_checks = true,
              bool use_hydrostatic_profile = true) {
  int const nx = n;
  int const ny = n;
  int const nz = n;
  real constexpr dt = 0.1;
  real constexpr q0 = 0.01;
  real const xlen = nx*grid_spacing;
  real const ylen = ny*grid_spacing;
  real const zlen = nz*grid_spacing;
  real const mean_u = n == 32 && flow == 1 && with_immersed ? 10._fp : 1._fp;
  real const velocity_perturbation = n == 32 && flow == 1 && with_immersed ? 0.5_fp : 0;

  core::Coupler coupler;
  coupler.set_option<std::string>("init_data",use_hydrostatic_profile ? "ABL_neutral" : "constant");
  coupler.set_option<real>("geostrophic_u",0);
  coupler.set_option<real>("geostrophic_v",0);
  coupler.set_option<bool>("enable_gravity",use_hydrostatic_profile);
  coupler.set_option<real>("dycore_max_wind",n == 32 ? 20._fp : 2._fp);
  coupler.set_option<real>("dycore_cs",20);
  coupler.set_option<real>("cfl",0.6);
  coupler.set_option<std::string>("dycore_time_stepper","ssprk3");
  coupler.set_option<bool>("dycore_anelastic_projection_diagnostics",true);
  coupler.set_option<bool>("dycore_anelastic_check_linearity",flow == 2 && run_invariance_checks);
  coupler.set_option<bool>("dycore_anelastic_check_cg_compatibility",n == 32);
  coupler.set_option<bool>("dycore_anelastic_use_jacobi_preconditioner",true);
  coupler.set_option<real>("dycore_anelastic_gmres_rel_tol",1.e-4);
  if (flow == 2 || with_immersed) {
    coupler.set_option<int>("dycore_anelastic_gmres_restart",100);
    coupler.set_option<int>("dycore_anelastic_gmres_max_iters",400);
  }
  coupler.init(core::ParallelComm(MPI_COMM_WORLD),coupler.generate_levels_equal(nz,zlen),ny,nx,ylen,xlen);
  coupler.add_tracer("water_vapor","water_vapor",true,false,true);
  custom_modules::sc_init(coupler);
  coupler.set_option<std::string>("bc_x1","periodic");
  coupler.set_option<std::string>("bc_x2","periodic");
  coupler.set_option<std::string>("bc_y1","periodic");
  coupler.set_option<std::string>("bc_y2","periodic");
  coupler.set_option<std::string>("bc_z1","wall_free_slip");
  coupler.set_option<std::string>("bc_z2","wall_free_slip");

  auto &dm = coupler.get_data_manager_readwrite();
  auto rho = dm.get<real const,3>("density_dry");
  auto u   = dm.get<real,3>("uvel");
  auto v   = dm.get<real,3>("vvel");
  auto w   = dm.get<real,3>("wvel");
  auto q   = dm.get<real,3>("water_vapor");
  auto imm = dm.get<real,3>("immersed_proportion");
  auto const dx = coupler.get_dx();
  auto const dy = coupler.get_dy();
  auto const dz = coupler.get_dz();
  auto const i_beg = coupler.get_i_beg();
  auto const j_beg = coupler.get_j_beg();
  auto const nx_local = coupler.get_nx();
  auto const ny_local = coupler.get_ny();
  int const cube_i_beg = (nx-cube_width)/2;
  int const cube_j_beg = (ny-cube_width)/2;
  yakl::parallel_for(YAKL_AUTO_LABEL(), yakl::SimpleBounds<3>(nz,ny_local,nx_local),
                     KOKKOS_LAMBDA (int k, int j, int i) {
    int const i_glob = i_beg+i;
    int const j_glob = j_beg+j;
    real const x = (i_beg+i+0.5_fp)*dx;
    real const y = (j_beg+j+0.5_fp)*dy;
    real const z = (k+0.5_fp)*dz(k);
    yakl::Random rng(0,3*(k*ny*nx + j_glob*nx + i_glob));
    u(k,j,i) = (flow == 1 ? mean_u : (flow == 2 ? std::sin(2*M_PI*x/xlen) : 0)) +
                 velocity_perturbation*rng.gen_uniform<real>(-1,1);
    v(k,j,i) = (flow == 2 ? 0.7_fp*std::sin(2*M_PI*y/ylen) : 0) +
                 velocity_perturbation*rng.gen_uniform<real>(-1,1);
    w(k,j,i) = (flow == 2 ? 0.4_fp*std::sin(2*M_PI*x/xlen)*std::sin(M_PI*z/zlen) : 0) +
                 velocity_perturbation*rng.gen_uniform<real>(-1,1);
    q(k,j,i) = q0*rho(k,j,i);
    imm(k,j,i) = with_immersed && k >= cube_k_beg && k < cube_k_beg+cube_width &&
                 j_glob >= cube_j_beg && j_glob < cube_j_beg+cube_width &&
                 i_glob >= cube_i_beg && i_glob < cube_i_beg+cube_width ? 1 : 0;
  });

  modules::Dynamics_Euler_Stratified dycore;
  if (n == 32) check_pressure_face_derivative_order(coupler);
  dycore.init(coupler);
  real4d state("anelastic_test_state",dycore.num_state,nz,ny_local,nx_local);
  real4d tracers("anelastic_test_tracers",1,nz,ny_local,nx_local);
  real4d state_tend("anelastic_test_state_tend",dycore.num_state,nz,ny_local,nx_local);
  real4d tracer_tend("anelastic_test_tracer_tend",1,nz,ny_local,nx_local);
  dycore.convert_coupler_to_dynamics(coupler,state,tracers);
  dycore.enforce_immersed_boundaries(coupler,state,tracers);
  dycore.compute_tendencies(coupler,state,state_tend,tracers,tracer_tend,dt,0,0);

  auto density_tend = state_tend.slice<3>(dycore.idR,yakl::COLON,yakl::COLON,yakl::COLON);
  real4d density_tend_4d("anelastic_test_density_tend",1,nz,ny_local,nx_local);
  density_tend.deep_copy_to(density_tend_4d.slice<3>(0,yakl::COLON,yakl::COLON,yakl::COLON));
  require(coupler,max_abs(coupler,density_tend_4d) == 0,name + ": persistent density changed");

  real4d fluid_tracer_tend("anelastic_test_fluid_tracer_tend",1,nz,ny_local,nx_local);
  yakl::parallel_for(YAKL_AUTO_LABEL(), yakl::SimpleBounds<3>(nz,ny_local,nx_local),
                     KOKKOS_LAMBDA (int k, int j, int i) {
    fluid_tracer_tend(0,k,j,i) = imm(k,j,i) == 0 ? tracer_tend(0,k,j,i) : 0;
  });
  real const tracer_error = max_abs(coupler,fluid_tracer_tend);
  require(coupler,tracer_error < 2.e-6,name + ": constant tracer mixing ratio was not preserved; max tendency = " +
                                      std::to_string(tracer_error));
  real const residual = coupler.get_option<real>("dycore_anelastic_last_linear_solver_rel_res");
  int const initial_solver_iters = coupler.get_option<int>("dycore_anelastic_last_linear_solver_iters");
  require(coupler,std::isfinite(residual),name + ": linear solver residual is invalid");
  if (flow == 2 || with_immersed) require(coupler,residual <= 1.1e-4,name + ": linear solver residual is too large");
  if constexpr (yakl::kokkos_debug) {
    real const boundary_flux = coupler.get_option<real>("dycore_anelastic_last_boundary_normal_flux_max");
    require(coupler,boundary_flux == 0,name + ": solid/immersed normal flux is nonzero");
    real const immersed_residual = coupler.get_option<real>("dycore_anelastic_last_immersed_residual_max");
    require(coupler,immersed_residual == 0,name + ": immersed cells contributed to the linear solver residual");
    real const pressure_mean = coupler.get_option<real>("dycore_anelastic_last_pressure_mean");
    require(coupler,std::abs(pressure_mean) < 1.e-12,name + ": pressure mean was not removed");
  }

  if (flow < 2 && !with_immersed) {
    real const state_drift = max_abs(coupler,state_tend);
    require(coupler,state_drift < 5.e-4,name + ": rest/uniform flow drifted; max tendency = " +
                                              std::to_string(state_drift));
  } else {
    if constexpr (yakl::kokkos_debug) {
      real const pre  = coupler.get_option<real>("dycore_anelastic_last_pre_div_l2");
      real const post = coupler.get_option<real>("dycore_anelastic_last_post_div_l2");
      real const rho_change = coupler.get_option<real>("dycore_anelastic_last_temporary_density_change");
      require(coupler,pre > 0 && post < 0.1_fp*pre,name + ": projection did not reduce physical mass-flux divergence");
      require(coupler,rho_change > 0,name + ": temporary advective density did not change");
    }
    if (n == 32) {
      require(coupler,coupler.get_option<std::string>("dycore_anelastic_last_linear_solver") == "CG",
              name + ": compatible operator did not select CG");
      require(coupler,coupler.get_option<std::string>("dycore_anelastic_last_preconditioner") == "Jacobi",
              name + ": projection did not use the Jacobi preconditioner");
      real checkerboard = 0;
      if constexpr (yakl::kokkos_debug) {
        real const cg_symmetry = coupler.get_option<real>("dycore_anelastic_last_cg_symmetry_error");
        bool const cg_positive = coupler.get_option<bool>("dycore_anelastic_last_cg_positive_probes");
        checkerboard = coupler.get_option<real>("dycore_anelastic_last_pressure_checkerboard_correlation");
        require(coupler,cg_symmetry < 2.e-6 && cg_positive,name + ": operator failed the CG compatibility probes");
        require(coupler,checkerboard < 0.1_fp,name + ": solved pressure contains a strong checkerboard mode");
      }
      real const face_derivative_order =
          coupler.get_option<real>("dycore_anelastic_pressure_face_derivative_order");
      if (coupler.is_mainproc()) {
        std::cout << name << ": observed pressure face-derivative order = " << face_derivative_order << std::endl;
      }

      std::string const output_prefix = "anelastic_projection_pressure_32cube";
      coupler.write_output_file(output_prefix,false);
      int constexpr stability_flow_throughs = 10;
      real const flow_through_time = xlen/mean_u;
      real const maximum_subcycle_dt = dycore.compute_time_step(coupler);
      real max_velocity_seen = 0;
      real max_checkerboard_seen = checkerboard;
      for (int step = 0; step < stability_flow_throughs; step++) {
        dycore.time_step(coupler,flow_through_time);
        real const elapsed_time = coupler.get_option<real>("elapsed_time")+flow_through_time;
        coupler.set_option<real>("elapsed_time",elapsed_time);
        require(coupler,!coupler.check_for_nan(),name + ": non-finite model state during stability integration");

        auto const &dm_after_step = coupler.get_data_manager_readonly();
        real const max_u = max_abs(coupler,dm_after_step.get<real const,3>("uvel"));
        real const max_v = max_abs(coupler,dm_after_step.get<real const,3>("vvel"));
        real const max_w = max_abs(coupler,dm_after_step.get<real const,3>("wvel"));
        real const max_pressure = max_abs(coupler,dm_after_step.get<real const,3>("anelastic_pressure_pert"));
        real const max_velocity = std::max({max_u,max_v,max_w});
        real const step_residual = coupler.get_option<real>("dycore_anelastic_last_linear_solver_rel_res");
        real step_checkerboard = 0;
        if constexpr (yakl::kokkos_debug) {
          step_checkerboard = coupler.get_option<real>("dycore_anelastic_last_pressure_checkerboard_correlation");
        }
        max_velocity_seen = std::max(max_velocity_seen,max_velocity);
        max_checkerboard_seen = std::max(max_checkerboard_seen,step_checkerboard);
        require(coupler,std::isfinite(max_pressure) && std::isfinite(step_checkerboard) &&
                        max_velocity < coupler.get_option<real>("dycore_max_wind"),
                name + ": fields became unbounded during stability integration");
        require(coupler,step_residual <= 1.1e-4,name + ": projection failed during stability integration");
        require(coupler,coupler.get_option<std::string>("dycore_anelastic_last_linear_solver") == "CG",
                name + ": CG was not retained during stability integration");
        if (coupler.is_mainproc()) {
          std::cout << name << ": flow-through " << step+1 << "/" << stability_flow_throughs
                    << ", elapsed time = " << elapsed_time << " s, maximum subcycle dt = " << maximum_subcycle_dt
                    << ", max velocity = " << max_velocity
                    << ", max pressure = " << max_pressure << ", pressure checkerboard = " << step_checkerboard
                    << ", residual = " << step_residual << std::endl;
        }
      }
      coupler.write_output_file(output_prefix,false);
      coupler.set_option<real>("dycore_anelastic_stability_max_velocity",max_velocity_seen);
      coupler.set_option<real>("dycore_anelastic_stability_max_pressure_checkerboard",max_checkerboard_seen);
    }
    if (run_invariance_checks) {
      if constexpr (yakl::kokkos_debug) {
        real const linearity = coupler.get_option<real>("dycore_anelastic_last_linearity_error");
        require(coupler,linearity < 2.e-5,name + ": matrix operator failed linearity");
      }

      auto state_tend_reference = state_tend.createDeviceCopy();
      auto tracer_tend_reference = tracer_tend.createDeviceCopy();
      coupler.set_option<real>("dycore_cs",900);
      dycore.compute_tendencies(coupler,state,state_tend,tracers,tracer_tend,dt,0,0);
      int const warm_solver_iters = coupler.get_option<int>("dycore_anelastic_last_linear_solver_iters");
      auto state_tend_flat = state_tend.collapse();
      auto state_ref_flat = state_tend_reference.collapse();
      auto tracer_tend_flat = tracer_tend.collapse();
      auto tracer_ref_flat = tracer_tend_reference.collapse();
      yakl::parallel_for(YAKL_AUTO_LABEL(), state_tend_flat.size(), KOKKOS_LAMBDA (int i) {
        state_tend_flat(i) -= state_ref_flat(i);
      });
      yakl::parallel_for(YAKL_AUTO_LABEL(), tracer_tend_flat.size(), KOKKOS_LAMBDA (int i) {
        tracer_tend_flat(i) -= tracer_ref_flat(i);
      });
      real const state_difference = max_abs(coupler,state_tend);
      real const tracer_difference = max_abs(coupler,tracer_tend);
      require(coupler,state_difference < 1.e-4_fp && tracer_difference < 1.e-4_fp,
              name + ": dycore_cs changed the anelastic solution; state/tracer differences = " +
              std::to_string(state_difference) + " / " + std::to_string(tracer_difference));
      require(coupler,warm_solver_iters <= initial_solver_iters,
              name + ": rolling pressure guess increased the iteration count");
      coupler.set_option<real>("dycore_cs",20);
      real const dt1 = dycore.compute_time_step(coupler);
      coupler.set_option<real>("dycore_cs",900);
      real const dt2 = dycore.compute_time_step(coupler);
      require(coupler,dt1 == dt2,name + ": dycore_cs changed the anelastic timestep");
      coupler.write_output_file("anelastic_dycore_output_test",false);
    }
  }

  if (coupler.is_mainproc()) {
    std::cout << name << ": PASS; " << coupler.get_option<std::string>("dycore_anelastic_last_linear_solver")
              << " iters = " << coupler.get_option<int>("dycore_anelastic_last_linear_solver_iters")
              << ", residual = " << coupler.get_option<real>("dycore_anelastic_last_linear_solver_rel_res")
              << ", tracer tendency max = " << tracer_error;
    if (flow == 2 || with_immersed) {
      if constexpr (yakl::kokkos_debug) {
        std::cout << ", pre/post divergence = "
                  << coupler.get_option<real>("dycore_anelastic_last_pre_div_l2") << " / "
                  << coupler.get_option<real>("dycore_anelastic_last_post_div_l2")
                  << ", temporary density change = "
                  << coupler.get_option<real>("dycore_anelastic_last_temporary_density_change");
        if (run_invariance_checks) {
          std::cout << ", linearity error = "
                    << coupler.get_option<real>("dycore_anelastic_last_linearity_error");
        }
      }
    }
    std::cout << std::endl;
  }
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  Kokkos::initialize();
  yakl::init();
  {
    run_case("anelastic_hydrostatic_rest",0,false);
    run_case("anelastic_uniform_periodic",1,false);
    run_case("anelastic_divergent_immersed",2,true);
    run_case("anelastic_large_uniform_cube",1,true,32,100,8,0,false,false);
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}

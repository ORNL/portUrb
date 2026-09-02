#include "coupler.h"
#include "dynamics_edge_centered_anelastic.h"
#include "sc_init.h"

namespace {

real constexpr benchmark_max_wind = 20;
real constexpr benchmark_cfl = 0.6;

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

void check_output_metadata(core::Coupler const & coupler, std::string const & filename) {
  auto const nx = coupler.get_nx();
  auto const ny = coupler.get_ny();
  auto const nz = coupler.get_nz();
  auto const hs = modules::Dynamics_Euler_Stratified::hs;
  core::FileIO nc(coupler.get_parallel_comm().get_mpi_comm(),core::FileIO::default_backend());
  nc.open(filename);

  std::string units;
  nc.readVariableAttribute(units,"x","units");
  require(coupler,units == "m","x coordinate units metadata is incorrect");
  nc.readVariableAttribute(units,"uvel","units");
  require(coupler,units == "m/s","velocity units metadata is incorrect");
  nc.readVariableAttribute(units,"density_dry_deviation","units");
  require(coupler,units == "kg/m^3","dry-density deviation units metadata is incorrect");
  nc.readVariableAttribute(units,"temperature_mean_column","units");
  require(coupler,units == "K","temperature mean-column units metadata is incorrect");
  nc.readVariableAttribute(units,"water_vapor","units");
  require(coupler,units == "kg/m^3","tracer units metadata is incorrect");
  nc.readVariableAttribute(units,"C0","units");
  require(coupler,units == "Pa (kg m^-3 K)^-gamma_d","C0 units metadata is incorrect");

  std::int32_t attribute_integer = 0;
  std::vector<std::uint64_t> attribute_vector;
  nc.readVariableAttribute(attribute_integer,"C0","test_integer");
  nc.readVariableAttribute(attribute_vector,"C0","test_vector");
  require(coupler,attribute_integer == -7,"integer variant attribute did not round trip");
  require(coupler,attribute_vector == std::vector<std::uint64_t>({2,3,5}),
          "vector variant attribute did not round trip");

  int dycore_hs = -1;
  nc.readGlobalAttribute(dycore_hs,"dycore_hs");
  require(coupler,dycore_hs == hs,"dycore_hs global attribute is incorrect");

  double C0 = 0;
  nc.begin_indep_data();
  if (coupler.is_mainproc()) nc.read(C0,"C0");
  nc.end_indep_data();
  coupler.get_parallel_comm().broadcast(C0);
  require(coupler,C0 == coupler.get_option<real>("C0"),"scalar double output value is incorrect");

  real1d x("metadata_x",nx);
  nc.read_all(x,"x",{static_cast<MPI_Offset>(coupler.get_i_beg())});
  float3d density("metadata_density",nz,ny,nx);
  nc.read_all(density,"density_dry_deviation",{0,static_cast<MPI_Offset>(coupler.get_j_beg()),
                                                 static_cast<MPI_Offset>(coupler.get_i_beg())});
  real1d density_mean("metadata_density_mean",nz);
  nc.read_all(density_mean,"density_dry_mean_column",{0});
  float3d pressure("metadata_pressure",nz,ny,nx);
  nc.read_all(pressure,"anelastic_pressure_pert",{0,static_cast<MPI_Offset>(coupler.get_j_beg()),
                                                  static_cast<MPI_Offset>(coupler.get_i_beg())});
  real1d z_halo("metadata_z_halo",nz+2*hs);
  nc.read_all(z_halo,"z_halo",{0});
  nc.close();

  auto const zmid = coupler.get_zmid();
  auto const dz   = coupler.get_dz();
  real1d z_halo_error("metadata_z_halo_error",nz+2*hs);
  yakl::parallel_for(YAKL_AUTO_LABEL(),nz+2*hs,KOKKOS_LAMBDA (int k) {
    real expected;
    if      (k < hs   ) expected = zmid(0    )-(hs-k)*dz(0   );
    else if (k >= hs+nz) expected = zmid(nz-1)+(k-hs-nz+1)*dz(nz-1);
    else                  expected = zmid(k-hs);
    z_halo_error(k) = std::abs(z_halo(k)-expected);
  });
  require(coupler,max_abs(coupler,z_halo_error) == 0,"z_halo coordinate does not replicate edge grid spacing");
}

void check_declared_derived_variables(core::Coupler const & coupler, std::string const & filename) {
  core::FileIO file(coupler.get_parallel_comm().get_mpi_comm(),core::FileIO::default_backend());
  file.open(filename);
  std::string derived;
  file.readGlobalAttribute(derived,"declare_derived_variables");
  file.close();
  require(coupler,derived.find("density_dry:float32[z,y,x] = density_dry_mean_column[:,None,None] + "
                               "density_dry_deviation; ") != std::string::npos,
          "derived dry-density reconstruction is missing");
  require(coupler,derived.find("temperature:float32[z,y,x] = temperature_mean_column[:,None,None] + "
                               "temperature_deviation; ") != std::string::npos,
          "derived temperature reconstruction is missing");
  require(coupler,derived.find("density:float32[z,y,x] = density_dry; ") != std::string::npos,
          "non-mass-adding water vapor was included in derived total density");
  require(coupler,derived.find("pressure:float32[z,y,x] = temperature * (density_dry * ") != std::string::npos,
          "derived pressure does not use the dycore equation of state");
  require(coupler,derived.find(" + water_vapor * ") != std::string::npos,
          "derived pressure omits water-vapor gas pressure");
  require(coupler,derived.find("theta:float32[z,y,x] = (pressure / ") != std::string::npos,
          "derived potential temperature is missing");
  require(coupler,derived.find("density_pert:float32[z,y,x] = density - hy_dens_cells[") != std::string::npos,
          "derived density perturbation does not use the interior hydrostatic profile");
  require(coupler,derived.find("pressure_pert:float32[z,y,x] = pressure - hy_pressure_cells[") != std::string::npos,
          "derived pressure perturbation is missing");
  require(coupler,derived.find("theta_pert:float32[z,y,x] = theta - hy_theta_cells[") != std::string::npos,
          "derived potential-temperature perturbation is missing");
}

void check_unmanaged_output_rejected(core::Coupler const & coupler) {
  core::FileIO file(coupler.get_parallel_comm().get_mpi_comm(),core::FileIO::default_backend());
  std::string const filename = "anelastic_dycore_ownership_test."+
                               std::string(core::FileIO::default_backend() == "adios2" ? "bp" : "nc");
  file.create(filename);
  file.create_dim("ownership_test",1);
  file.create_var<double>("ownership_test",{"ownership_test"});
  file.enddef();
  double1d owned("ownership_test",1);
  double1d unmanaged(owned.data(),1);
  bool rejected = false;
  try {
    file.write_all(unmanaged,"ownership_test",{0});
  } catch (std::runtime_error const &) {
    rejected = true;
  }
  require(coupler,rejected,"unmanaged direct FileIO view was not rejected");
  file.close();
}

void check_native_compression(core::Coupler const & coupler, std::string const & filename) {
#ifdef PORTURB_HAS_ADIOS2
  core::FileIO file(coupler.get_parallel_comm().get_mpi_comm(),core::FileIO::default_backend());
  file.open(filename);
  require(coupler,file.variable_has_operations("density_dry_deviation"),
          "large BP5 density-deviation variable does not contain a native ADIOS2 compression operation");

  auto const nx = coupler.get_nx();
  auto const ny = coupler.get_ny();
  auto const nz = coupler.get_nz();
  float3d restored("compressed_density",nz,ny,nx);
  file.read_all(restored,"density_dry_deviation",{0,static_cast<MPI_Offset>(coupler.get_j_beg()),
                                                   static_cast<MPI_Offset>(coupler.get_i_beg())});
  real1d density_mean("compressed_density_mean",nz);
  file.read_all(density_mean,"density_dry_mean_column",{0});
  file.close();

  auto density = coupler.get_data_manager_readonly().get<real const,3>("density_dry");
  yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(nz,ny,nx),KOKKOS_LAMBDA (int k, int j, int i) {
    restored(k,j,i) -= static_cast<float>(density(k,j,i)-density_mean(k));
  });
  require(coupler,max_abs(coupler,restored) == 0,"compressed BP5 density deviation did not round trip exactly");
#else
  (void) coupler;
  (void) filename;
#endif
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

real run_case(std::string const & name, int flow, bool with_immersed, int n = 8, real grid_spacing = 1,
              int cube_width = 2, int cube_k_beg = 2, bool run_invariance_checks = true,
              bool use_hydrostatic_profile = true, std::string const & preconditioner = "Jacobi",
              int schwarz_tile = 16, int schwarz_degree = 16, real sound_speed = 0,
              bool check_compression = false, bool stretched_vertical = false) {
  int const nx = n;
  int const ny = n;
  int const nz = n;
  real constexpr correctness_dt = 0.1;
  real constexpr q0 = 0.01;
  bool const benchmark_case = n >= 32;
  real const xlen = nx*grid_spacing;
  real const ylen = ny*grid_spacing;
  real const zlen = nz*grid_spacing;
  real const mean_u = benchmark_case && flow == 1 && with_immersed ? 10._fp : 1._fp;
  real const velocity_perturbation = benchmark_case && flow == 1 && with_immersed ? 0.5_fp : 0;

  core::Coupler coupler;
  coupler.set_option<std::string>("init_data",use_hydrostatic_profile ? "ABL_neutral" : "constant");
  coupler.set_option<real>("geostrophic_u",0);
  coupler.set_option<real>("geostrophic_v",0);
  coupler.set_option<bool>("enable_gravity",use_hydrostatic_profile);
  coupler.set_option<real>("dycore_max_wind",benchmark_case ? benchmark_max_wind : 2._fp);
  coupler.set_option<real>("dycore_cs",sound_speed > 0 ? sound_speed : 20);
  coupler.set_option<real>("cfl",benchmark_cfl);
  coupler.set_option<std::string>("dycore_time_stepper","ssprk3");
  coupler.set_option<bool>("dycore_anelastic_projection_diagnostics",true);
  coupler.set_option<bool>("dycore_anelastic_check_linearity",flow == 2 && run_invariance_checks);
  coupler.set_option<bool>("dycore_anelastic_check_cg_compatibility",benchmark_case);
  coupler.set_option<bool>("dycore_anelastic_screening",sound_speed > 0);
  coupler.set_option<std::string>("dycore_anelastic_preconditioner",preconditioner);
  coupler.set_option<bool>("dycore_anelastic_time_linear_solver",false);
  if (preconditioner == "Schwarz") {
    coupler.set_option<int>("dycore_anelastic_schwarz_tile_nx",schwarz_tile);
    coupler.set_option<int>("dycore_anelastic_schwarz_tile_ny",schwarz_tile);
    coupler.set_option<int>("dycore_anelastic_schwarz_overlap",2);
    coupler.set_option<int>("dycore_anelastic_schwarz_chebyshev_degree",schwarz_degree);
  }
  if (preconditioner == "GeometricMultigrid") {
    coupler.set_option<bool>("dycore_anelastic_use_cg",true);
    coupler.set_option<int>("dycore_anelastic_geometric_multigrid_coarse_cells",n%2 == 0 ? 64 : 2200);
    coupler.set_option<int>("dycore_anelastic_geometric_multigrid_min_cells_per_rank",64);
    coupler.set_option<int>("dycore_anelastic_geometric_multigrid_coarsening_factor_x",2);
    coupler.set_option<int>("dycore_anelastic_geometric_multigrid_coarsening_factor_y",3);
    coupler.set_option<real>("dycore_anelastic_geometric_multigrid_coarsening_factor_z",1.5_fp);
  }
  coupler.set_option<real>("dycore_anelastic_gmres_rel_tol",1.e-4);
  if (flow == 2 || with_immersed) {
    coupler.set_option<int>("dycore_anelastic_gmres_restart",100);
    coupler.set_option<int>("dycore_anelastic_gmres_max_iters",benchmark_case ? 800 : 400);
  }
  auto const zint = stretched_vertical ? coupler.generate_levels_exp(nz,zlen,0.7_fp*grid_spacing) :
                                         coupler.generate_levels_equal(nz,zlen);
  coupler.init(core::ParallelComm(MPI_COMM_WORLD),zint,ny,nx,ylen,xlen);
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
  auto const zmid = coupler.get_zmid();
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
    real const z = zmid(k);
    yakl::Random rng(0,3*(k*ny*nx + j_glob*nx + i_glob));
    u(k,j,i) = (flow == 1 ? mean_u : (flow == 2 ? std::sin(2*M_PI*x/xlen) : 0)) +
                 velocity_perturbation*rng.gen_uniform<real>(-1,1);
    v(k,j,i) = (flow == 2 ? 0.7_fp*std::sin(2*M_PI*y/ylen) : 0) +
                 velocity_perturbation*rng.gen_uniform<real>(-1,1);
    w(k,j,i) = (flow == 2 ? 0.4_fp*std::sin(2*M_PI*x/xlen)*std::sin(M_PI*z/zlen) : 0) +
                 velocity_perturbation*rng.gen_uniform<real>(-1,1);
    q(k,j,i) = q0*rho(k,j,i);
    bool immersed = with_immersed && k >= cube_k_beg && k < cube_k_beg+cube_width &&
                    j_glob >= cube_j_beg && j_glob < cube_j_beg+cube_width &&
                    i_glob >= cube_i_beg && i_glob < cube_i_beg+cube_width;
    imm(k,j,i) = immersed ? 1 : 0;
  });

  modules::Dynamics_Euler_Stratified dycore;
  real const dt = benchmark_case ? dycore.compute_time_step(coupler) : correctness_dt;
  if (benchmark_case) check_pressure_face_derivative_order(coupler);
  dycore.init(coupler);
  real4d state("anelastic_test_state",dycore.num_state,nz,ny_local,nx_local);
  real4d tracers("anelastic_test_tracers",1,nz,ny_local,nx_local);
  real4d state_tend("anelastic_test_state_tend",dycore.num_state,nz,ny_local,nx_local);
  real4d tracer_tend("anelastic_test_tracer_tend",1,nz,ny_local,nx_local);
  dycore.convert_coupler_to_dynamics(coupler,state,tracers);
  dycore.enforce_immersed_boundaries(coupler,state,tracers);
  dycore.compute_tendencies(coupler,state,state_tend,tracers,tracer_tend,dt,0,0);
  real benchmark_seconds_mean = 0;
  real benchmark_iterations_mean = 0;
  if (benchmark_case) {
    int constexpr benchmark_repetitions = 5;
    coupler.set_option<bool>("dycore_anelastic_time_linear_solver",true);
    for (int repetition = 0; repetition < benchmark_repetitions; repetition++) {
      dm.get<real,3>("anelastic_pressure_pert") = 0;
      dycore.compute_tendencies(coupler,state,state_tend,tracers,tracer_tend,dt,0,0);
      benchmark_seconds_mean += coupler.get_option<real>("dycore_anelastic_last_linear_solver_seconds");
      benchmark_iterations_mean += static_cast<real>(
          coupler.get_option<int>("dycore_anelastic_last_linear_solver_iters"));
    }
    benchmark_seconds_mean /= benchmark_repetitions;
    benchmark_iterations_mean /= benchmark_repetitions;
  }

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
  if (preconditioner == "GeometricMultigrid") {
    require(coupler,coupler.get_option<std::string>("dycore_anelastic_last_linear_solver") == "CG",
            name + ": geometric multigrid case did not use CG");
    require(coupler,coupler.get_option<std::string>("dycore_anelastic_last_preconditioner") ==
                    "GeometricMultigrid",
            name + ": geometric multigrid case did not use the requested preconditioner");
    require(coupler,coupler.get_option<int>("dycore_anelastic_geometric_multigrid_levels") >= 2,
            name + ": geometric multigrid hierarchy has fewer than two levels");
    require(coupler,coupler.get_option<int>("dycore_anelastic_geometric_multigrid_coarse_cells") > 0,
            name + ": geometric multigrid coarse level is empty");
    require(coupler,coupler.get_option<int>("dycore_anelastic_geometric_multigrid_coarse_nz") > 0,
            name + ": geometric multigrid coarse vertical extent is empty");
    if (stretched_vertical) {
      require(coupler,coupler.get_option<int>("dycore_anelastic_geometric_multigrid_coarse_nz") ==
                      static_cast<int>(std::round(n/1.5_fp)),
              name + ": geometric multigrid did not round the non-integral vertical coarsening extent");
    }
    require(coupler,coupler.get_option<int>("dycore_anelastic_geometric_multigrid_coarse_ranks") == 1,
            name + ": geometric multigrid did not agglomerate onto one coarse task");
    require(coupler,coupler.get_option<std::string>("dycore_anelastic_geometric_multigrid_interpolation") ==
                    "Quadratic",
            name + ": geometric multigrid did not use quadratic interpolation");
    require(coupler,coupler.get_option<int>(
                        "dycore_anelastic_geometric_multigrid_coarsening_factor_x") == 2 &&
                    coupler.get_option<int>(
                        "dycore_anelastic_geometric_multigrid_coarsening_factor_y") == 3 &&
                    coupler.get_option<real>(
                        "dycore_anelastic_geometric_multigrid_coarsening_factor_z") == 1.5_fp,
            name + ": geometric multigrid did not retain its directional coarsening factors");
    require(coupler,coupler.get_option<std::string>(
                        "dycore_anelastic_geometric_multigrid_coarse_smoother") == "Jacobi",
            name + ": geometric multigrid did not select the Jacobi coarse smoother");
  }
  if (flow == 2 || with_immersed) require(coupler,residual <= 1.1e-4,name + ": linear solver residual is too large");
  real const screening_coefficient =
      coupler.get_option<real>("dycore_anelastic_last_screening_inverse_length_squared");
  real const acoustic_length = dt*sound_speed;
  real const expected_screening_coefficient = sound_speed > 0 ? 1/(acoustic_length*acoustic_length) : 0;
  require(coupler,screening_coefficient == expected_screening_coefficient,
          name + ": finite-sound-speed screening coefficient does not equal 1/(dt^2*c_s^2)");
  if constexpr (yakl::kokkos_debug) {
    real const boundary_flux = coupler.get_option<real>("dycore_anelastic_last_boundary_normal_flux_max");
    require(coupler,boundary_flux == 0,name + ": solid/immersed normal flux is nonzero");
    real const immersed_residual = coupler.get_option<real>("dycore_anelastic_last_immersed_residual_max");
    require(coupler,immersed_residual == 0,name + ": immersed cells contributed to the linear solver residual");
    real const pressure_mean = coupler.get_option<real>("dycore_anelastic_last_pressure_mean");
    std::cout << "Pressure mean:" << std::scientific << std::abs(pressure_mean);
    if (sound_speed == 0) {
      require(coupler,std::abs(pressure_mean) < 3.e-6,name + ": pressure mean was not removed.");
    }
  }

  if (flow < 2 && !with_immersed) {
    real const state_drift = max_abs(coupler,state_tend);
    require(coupler,state_drift < 5.e-4,name + ": rest/uniform flow drifted; max tendency = " +
                                              std::to_string(state_drift));
  } else {
    if constexpr (yakl::kokkos_debug) {
      real const pre  = coupler.get_option<real>("dycore_anelastic_last_pre_div_l2");
      real const post = coupler.get_option<real>("dycore_anelastic_last_post_div_l2");
      real const constraint = coupler.get_option<real>("dycore_anelastic_last_screened_constraint_l2");
      real const rho_change = coupler.get_option<real>("dycore_anelastic_last_temporary_density_change");
      require(coupler,pre > 0,name + ": provisional mass-flux divergence is zero");
      if (sound_speed > 0) {
        require(coupler,constraint < 2.e-4_fp*pre,name + ": screened divergence constraint was not satisfied");
      } else {
        require(coupler,post < 0.1_fp*pre,name + ": projection did not reduce physical mass-flux divergence");
      }
      require(coupler,rho_change > 0,name + ": temporary advective density did not change");
    }
    if (benchmark_case) {
      require(coupler,coupler.get_option<std::string>("dycore_anelastic_last_linear_solver") == "CG",
              name + ": compatible operator did not select CG");
      require(coupler,coupler.get_option<std::string>("dycore_anelastic_last_preconditioner") == preconditioner,
              name + ": projection did not use the requested preconditioner");
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
        std::cout << name << ": preconditioner = " << preconditioner
                  << ", sound speed = " << (sound_speed > 0 ? std::to_string(sound_speed) : "screening off")
                  << (preconditioner == "Schwarz" ? ", tile = "+std::to_string(schwarz_tile)+
                                                       ", degree = "+std::to_string(schwarz_degree) : "")
                  << ", mean CG iterations = " << benchmark_iterations_mean
                  << ", mean solve seconds = " << benchmark_seconds_mean
                  << ", observed pressure face-derivative order = " << face_derivative_order << std::endl;
      }

      int constexpr stability_flow_throughs = 0;
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
      coupler.register_output_variable<real>("C0",core::Coupler::DIMS_SCALAR,
                                             {{"units",std::string("duplicate ignored")},
                                              {"test_integer",std::int32_t(-7)},
                                              {"test_vector",std::vector<std::uint64_t>({2,3,5})}});
      std::string const output_file = "anelastic_dycore_output_test_00000000."+
                                      std::string(core::FileIO::default_backend() == "adios2" ? "bp" : "nc");
      coupler.write_output_file("anelastic_dycore_output_test",false);
      check_output_metadata(coupler,output_file);
      check_declared_derived_variables(coupler,output_file);
      check_unmanaged_output_rejected(coupler);
      real const C0_before_restart = coupler.get_option<real>("C0");
      coupler.set_option<real>("C0",-1);
      coupler.set_option<std::string>("restart_file",output_file);
      coupler.overwrite_with_restart();
      require(coupler,coupler.get_option<real>("C0") == C0_before_restart,
              "scalar physical constant was not restored from restart");
    }
  }

  if (check_compression && core::FileIO::default_backend() == "adios2") {
    std::string const output_prefix = "anelastic_dycore_restart_test";
    coupler.set_option<std::string>("adios2_compression_compressor","lz4hc");
    coupler.set_option<int>("adios2_compression_clevel",7);
    coupler.write_output_file(output_prefix,false);
    check_native_compression(coupler,output_prefix+"_00000000.bp");
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
  return benchmark_iterations_mean;
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  Kokkos::initialize();
  yakl::init();
  {
    bool const geometric_only = argc > 1 && std::string(argv[1]) == "--geometric-only";
    int const schwarz_tile = argc > 1 && !geometric_only ? std::stoi(argv[1]) : 16;
    int const schwarz_degree = argc > 2 && !geometric_only ? std::stoi(argv[2]) : 16;
    if (geometric_only) {
      run_case("anelastic_geometric_multigrid_pure_abl",2,false,24,1,2,0,false,true,
               "GeometricMultigrid",16,16,0);
      run_case("anelastic_geometric_multigrid_pure_abl_screened_odd_grid",2,false,25,1,2,0,false,true,
               "GeometricMultigrid",16,16,350,false,true);
    } else {
      run_case("anelastic_hydrostatic_rest",0,false);
      run_case("anelastic_uniform_periodic",1,false);
      run_case("anelastic_divergent_immersed",2,true);
      real constexpr benchmark_grid_spacing = 100;
      real constexpr benchmark_dt = benchmark_cfl*benchmark_grid_spacing/benchmark_max_wind;
      std::array<real,7> const acoustic_lengths_in_cells = {12,16,24,32,48,64,0};
      std::array<real,7> none_iterations;
      std::array<real,7> schwarz_iterations;
      for (int il = 0; il < acoustic_lengths_in_cells.size(); il++) {
        real const acoustic_length_in_cells = acoustic_lengths_in_cells[il];
        real const acoustic_length = acoustic_length_in_cells*benchmark_grid_spacing;
        real const sound_speed = acoustic_length > 0 ? acoustic_length/benchmark_dt : 0;
        std::string const suffix = acoustic_length > 0 ?
            "csdt"+std::to_string(static_cast<int>(acoustic_length_in_cells))+"delta" : "off";
        none_iterations[il] = run_case("anelastic_large_uniform_cube_none_"+suffix,1,true,128,
                                       benchmark_grid_spacing,32,0,false,false,
                                       "none",schwarz_tile,schwarz_degree,sound_speed,il == 0);
        schwarz_iterations[il] = run_case("anelastic_large_uniform_cube_schwarz_"+suffix,1,true,128,
                                          benchmark_grid_spacing,32,0,false,false,
                                          "Schwarz",schwarz_tile,schwarz_degree,sound_speed);
      }
      for (int il = 1; il < acoustic_lengths_in_cells.size(); il++) {
        if (none_iterations[il] < none_iterations[il-1]) {
          endrun("ERROR: unpreconditioned screened iteration count did not increase monotonically with c_s*dt");
        }
        if (schwarz_iterations[il] < schwarz_iterations[il-1]) {
          endrun("ERROR: Schwarz-preconditioned screened iteration count did not increase monotonically with c_s*dt");
        }
      }
      if (none_iterations.back() <= none_iterations[acoustic_lengths_in_cells.size()-2] ||
          schwarz_iterations.back() <= schwarz_iterations[acoustic_lengths_in_cells.size()-2]) {
        endrun("ERROR: disabling screening did not increase both 128^3 iteration counts");
      }
      if (core::ParallelComm(MPI_COMM_WORLD).get_rank_id() == 0) {
        std::cout << "128^3 finite-sound-speed iteration study (c_s*dt/delta, c_s, Mach_20, none, Schwarz):"
                  << std::endl;
        for (int il = 0; il < acoustic_lengths_in_cells.size(); il++) {
          real const acoustic_length = acoustic_lengths_in_cells[il]*benchmark_grid_spacing;
          real const sound_speed = acoustic_length > 0 ? acoustic_length/benchmark_dt : 0;
          std::cout << "  ";
          if (sound_speed > 0) {
            std::cout << acoustic_lengths_in_cells[il] << ", " << sound_speed << ", "
                      << benchmark_max_wind/sound_speed;
          } else {
            std::cout << "off, off, off";
          }
          std::cout << ", " << none_iterations[il] << ", " << schwarz_iterations[il] << std::endl;
        }
      }
    }
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}

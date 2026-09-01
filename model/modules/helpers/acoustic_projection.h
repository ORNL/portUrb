#pragma once

#include "main_header.h"
#include "coupler.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "GMRES.h"
#include "ConjGrad.h"
#include "ConnectivityGalerkinMultigrid.h"
#include "GeometricMultigrid.h"
#include <memory>
#include <sstream>

namespace modules {

  struct AcousticProjectionConfig {
    bool diagnostics                    = false;
    bool check_linearity                = false;
    bool check_cg_compatibility         = true;
    bool use_conjugate_gradient         = true;
    bool time_linear_solver             = false;
    bool screening                      = false;
    real sound_speed                    = 350;
    real momentum_hyperviscosity        = 0.1;
    real pressure_hyperviscosity        = 0;
    std::string preconditioner          = "Jacobi";
    int gmres_restart                   = 30;
    int linear_solver_max_iterations    = 200;
    real linear_solver_relative_tolerance = 1.e-6;
    real linear_solver_absolute_tolerance = 0;
    bool linear_solver_verbose          = false;
    bool gmres_reorthogonalize           = true;
    int schwarz_tile_nx                 = 8;
    int schwarz_tile_ny                 = 8;
    int schwarz_overlap                 = 2;
    int schwarz_chebyshev_degree        = 8;
    real schwarz_chebyshev_lambda_min   = 0.02;
    real schwarz_chebyshev_lambda_max   = 2;
    std::shared_ptr<ConnectivityGalerkinMultigrid<float>> multigrid;
    int multigrid_vcycles               = 1;
    int multigrid_pre_smooth            = 1;
    int multigrid_post_smooth           = 1;
    int multigrid_aggregate_size        = 8;
    int multigrid_max_levels            = 24;
    int multigrid_coarse_max_dofs       = 256;
    int multigrid_coarse_smooth         = 16;
    real multigrid_jacobi_weight        = 2._fp/3._fp;
    std::shared_ptr<GeometricMultigrid<float>> geometric_multigrid;
    int geometric_multigrid_vcycles            = 1;
    int geometric_multigrid_pre_smooth         = 2;
    int geometric_multigrid_post_smooth        = 2;
    int geometric_multigrid_coarse_smooth      = 24;
    int geometric_multigrid_max_levels         = 20;
    int geometric_multigrid_coarse_cells       = 32768;
    int geometric_multigrid_min_cells_per_rank = 131072;
    real geometric_multigrid_jacobi_weight     = 2._fp/3._fp;
    std::shared_ptr<GeometricMultigrid<float>> tensor_line_multigrid;
    int tensor_line_multigrid_vcycles            = 1;
    int tensor_line_multigrid_pre_smooth         = 2;
    int tensor_line_multigrid_post_smooth        = 2;
    int tensor_line_multigrid_coarse_smooth      = 24;
    int tensor_line_multigrid_max_levels         = 20;
    int tensor_line_multigrid_coarse_nx          = 50;
    int tensor_line_multigrid_coarse_ny          = 50;
    int tensor_line_multigrid_min_cells_per_rank = 131072;
    real tensor_line_multigrid_jacobi_weight     = 2._fp/3._fp;
  };

  namespace detail {

  // Implementation details for the public function-based acoustic projection API.
  template <int ord>
  struct AcousticProjectionImpl {
    using FLOC = float;
    using ProjectionScalar = float;
    using Projection3d = yakl::Array<ProjectionScalar ***>;
    using Projection4d = yakl::Array<ProjectionScalar ****>;

    int static constexpr hs   = ord/2;
    int static constexpr idPP = 0;
    int static constexpr idRU = 1;
    int static constexpr idRV = 2;
    int static constexpr idRW = 3;

    template <class FP, int ORD>
    KOKKOS_INLINE_FUNCTION static void modify_stencil_immersed_der0( SArray<FP,   ORD>       & stencil,
                                                                     SArray<bool, ORD> const & immersed) {
      static_assert(ORD >= 2, "Stencil must contain at least two points");
      static_assert(ORD%2 == 0, "Edge-centered stencil order must be even");
      constexpr int hs = ORD / 2;
      // If both cells adjacent to the edge are immersed, there is no
      // immediately available in-domain value from which to extend.
      if (immersed(hs - 1) && immersed(hs))   return;
      // Extend the last in-domain value to the right.
      for (int i2 = hs; i2 < ORD; i2++) {
        if (immersed(i2)) {
          FP const boundary_value = stencil(i2-1);
          for (int i3 = i2; i3 < ORD; i3++) { stencil(i3) = boundary_value; }
          break;
        }
      }
      // Extend the last in-domain value to the left.
      for (int i2 = hs - 1; i2 >= 0; i2--) {
        if (immersed(i2)) {
          FP const boundary_value = stencil(i2+1);
          for (int i3 = i2; i3 >= 0; i3--) { stencil(i3) = boundary_value; }
          break;
        }
      }
    }

    // Apply homogeneous, candidate-dependent boundary extensions for a projection matvec.
    // Precursor data are deliberately excluded because fixed data and velocity-sign branches would make A*x affine/nonlinear.
    template <class Scalar>
    static void projection_boundary_conditions(core::Coupler const & coupler,
                                                yakl::Array<Scalar ****> const & fields) {
      using yakl::SimpleBounds;
      int constexpr idRU = 1;
      int constexpr idRV = 2;
      int constexpr idRW = 3;
      auto const nx      = coupler.get_nx();
      auto const ny      = coupler.get_ny();
      auto const nz      = coupler.get_nz();
      auto const px      = coupler.get_px();
      auto const py      = coupler.get_py();
      auto const nproc_x = coupler.get_nproc_x();
      auto const nproc_y = coupler.get_nproc_y();
      auto const bc_x1   = coupler.get_option<std::string>("bc_x1");
      auto const bc_x2   = coupler.get_option<std::string>("bc_x2");
      auto const bc_y1   = coupler.get_option<std::string>("bc_y1");
      auto const bc_y2   = coupler.get_option<std::string>("bc_y2");
      auto const bc_z1   = coupler.get_option<std::string>("bc_z1");
      auto const bc_z2   = coupler.get_option<std::string>("bc_z2");

      if (px == 0 && bc_x1 != "periodic") {
        bool const wall = bc_x1 == "wall_free_slip";
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<4>(4,nz,ny,hs),
                           KOKKOS_LAMBDA (int l, int k, int j, int ii) {
          fields(l,hs+k,hs+j,hs-1-ii) = wall && l == idRU ? 0 : fields(l,hs+k,hs+j,hs);
        });
      }
      if (px == nproc_x-1 && bc_x2 != "periodic") {
        bool const wall = bc_x2 == "wall_free_slip";
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<4>(4,nz,ny,hs),
                           KOKKOS_LAMBDA (int l, int k, int j, int ii) {
          fields(l,hs+k,hs+j,hs+nx+ii) = wall && l == idRU ? 0 : fields(l,hs+k,hs+j,hs+nx-1);
        });
      }
      if (py == 0 && bc_y1 != "periodic") {
        bool const wall = bc_y1 == "wall_free_slip";
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<4>(4,nz,hs,nx),
                           KOKKOS_LAMBDA (int l, int k, int jj, int i) {
          fields(l,hs+k,hs-1-jj,hs+i) = wall && l == idRV ? 0 : fields(l,hs+k,hs,hs+i);
        });
      }
      if (py == nproc_y-1 && bc_y2 != "periodic") {
        bool const wall = bc_y2 == "wall_free_slip";
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<4>(4,nz,hs,nx),
                           KOKKOS_LAMBDA (int l, int k, int jj, int i) {
          fields(l,hs+k,hs+ny+jj,hs+i) = wall && l == idRV ? 0 : fields(l,hs+k,hs+ny-1,hs+i);
        });
      }
      if (bc_z1 == "periodic") {
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<4>(4,hs,ny,nx),
                           KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          fields(l,kk,hs+j,hs+i) = fields(l,nz+kk,hs+j,hs+i);
        });
      } else {
        bool const wall = bc_z1 == "wall_free_slip";
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<4>(4,hs,ny,nx),
                           KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          fields(l,hs-1-kk,hs+j,hs+i) = wall && l == idRW ? 0 : fields(l,hs,hs+j,hs+i);
        });
      }
      if (bc_z2 == "periodic") {
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<4>(4,hs,ny,nx),
                           KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          fields(l,hs+nz+kk,hs+j,hs+i) = fields(l,hs+kk,hs+j,hs+i);
        });
      } else {
        bool const wall = bc_z2 == "wall_free_slip";
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<4>(4,hs,ny,nx),
                           KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          fields(l,hs+nz+kk,hs+j,hs+i) = wall && l == idRW ? 0 : fields(l,hs+nz-1,hs+j,hs+i);
        });
      }
    }

    static void apply(core::Coupler &coupler, float4d const &momentum_in,
                      float4d const &momentum_out, real3d const &pressure_io, real dt,
                      AcousticProjectionConfig const &config) {
      using yakl::SimpleBounds;
      auto const nx          = coupler.get_nx();
      auto const ny          = coupler.get_ny();
      auto const nz          = coupler.get_nz();
      auto const nx_glob     = coupler.get_nx_glob();
      auto const ny_glob     = coupler.get_ny_glob();
      auto const i_beg       = coupler.get_i_beg();
      auto const j_beg       = coupler.get_j_beg();
      auto const dx          = coupler.get_dx();
      auto const dy          = coupler.get_dy();
      auto const dz          = coupler.get_dz();
      bool const diagnostics = config.diagnostics;
      auto const imm_th      = coupler.get_option<real>("immersed_threshold",0.5);
      auto       &dm         = coupler.get_data_manager_readwrite();
      auto const immersed    = dm.get<real const,3>("dycore_immersed_proportion_halos");
      auto const rho_h       = dm.get<real const,1>("hy_dens_cells");
      auto const rho_h_edge  = dm.get<real const,1>("hy_dens_edges");
      auto const metjac_edge = dm.get<real const,1>("dycore_metjac_edges");
      auto const fluid_mask = dm.get<int const,3>("dycore_anelastic_fluid_mask");
      auto const inv_diagonal_dtless =
          dm.get<float const,3>("dycore_anelastic_projection_inv_diagonal_dtless");
      int const fluid_count = coupler.get_option<int>("dycore_anelastic_fluid_count");
      ProjectionScalar const dt_proj = static_cast<ProjectionScalar>(dt);
      ProjectionScalar const dx_proj = static_cast<ProjectionScalar>(dx);
      ProjectionScalar const dy_proj = static_cast<ProjectionScalar>(dy);
      ProjectionScalar const r_dx = ProjectionScalar(1)/dx_proj;
      ProjectionScalar const r_dy = ProjectionScalar(1)/dy_proj;
      bool const screening_enabled = config.screening;
      real const sound_speed = config.sound_speed;
      if (screening_enabled && (!std::isfinite(sound_speed) || sound_speed <= 0)) {
        endrun("ERROR: dycore_cs must be finite and positive when anelastic screening is enabled");
      }
      real const screening_length = dt*sound_speed;
      if (screening_enabled && (!std::isfinite(screening_length) || screening_length <= 0)) {
        endrun("ERROR: dt*dycore_cs must be finite and positive when anelastic screening is enabled");
      }
      real const screening_inv_length_squared_real = screening_enabled ?
          1/(screening_length*screening_length) : 0;
      ProjectionScalar const screening_inv_length_squared =
          static_cast<ProjectionScalar>(screening_inv_length_squared_real);
      coupler.set_option<real>("dycore_anelastic_last_screening_inverse_length_squared",
                               screening_inv_length_squared_real);
      bool const wall_x1 = coupler.get_option<std::string>("bc_x1") == "wall_free_slip";
      bool const wall_x2 = coupler.get_option<std::string>("bc_x2") == "wall_free_slip";
      bool const wall_y1 = coupler.get_option<std::string>("bc_y1") == "wall_free_slip";
      bool const wall_y2 = coupler.get_option<std::string>("bc_y2") == "wall_free_slip";
      bool const wall_z1 = coupler.get_option<std::string>("bc_z1") == "wall_free_slip";
      bool const wall_z2 = coupler.get_option<std::string>("bc_z2") == "wall_free_slip";
      auto const px      = coupler.get_px();
      auto const py      = coupler.get_py();
      auto const nproc_x = coupler.get_nproc_x();
      auto const nproc_y = coupler.get_nproc_y();

      int constexpr idPP = 0;
      int constexpr idRU = 1;
      int constexpr idRV = 2;
      int constexpr idRW = 3;
      using Projection3d = yakl::Array<ProjectionScalar ***>;
      using Projection4d = yakl::Array<ProjectionScalar ****>;
      Projection3d pressure          ("anelastic_projection_pressure"          ,nz,ny,nx);
      Projection3d pressure_rhs      ("anelastic_projection_rhs"               ,nz,ny,nx);
      Projection3d pressure_projected("anelastic_projection_pressure_projected",nz,ny,nx);
      Projection3d projection_work   ("anelastic_projection_work"              ,nz,ny,nx);
      Projection4d momentum_rhs      ("anelastic_projection_momentum_rhs"      ,4,nz,ny,nx);
      Projection4d momentum_work     ("anelastic_projection_momentum_work"     ,4,nz,ny,nx);
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        bool const is_fluid = fluid_mask(k,j,i) == 1;
        pressure(k,j,i) = is_fluid ? static_cast<ProjectionScalar>(pressure_io(k,j,i)) : ProjectionScalar(0);
        pressure_rhs(k,j,i) = 0;
        momentum_rhs(idPP,k,j,i) = 0;
        momentum_rhs(idRU,k,j,i) = is_fluid ? momentum_in(0,k,j,i) : 0;
        momentum_rhs(idRV,k,j,i) = is_fluid ? momentum_in(1,k,j,i) : 0;
        momentum_rhs(idRW,k,j,i) = is_fluid ? momentum_in(2,k,j,i) : 0;
      });

      // Restrict pressure to fluid cells. The unscreened Poisson operator additionally requires a zero-mean pressure
      // representative, while screening removes that constant nullspace. Applying the appropriate linear restriction to
      // every matrix input and output preserves the linearity required by Krylov solvers.
      auto project_pressure = [&] (Projection3d const & input, Projection3d const & output) {
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          projection_work(k,j,i) = fluid_mask(k,j,i) == 1 ? input(k,j,i) : 0;
        });
        if (screening_enabled) {
          yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
            output(k,j,i) = projection_work(k,j,i);
          });
          return ProjectionScalar(0);
        }
        ProjectionScalar const sum =
            coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(projection_work),MPI_SUM);
        ProjectionScalar const mean = sum/static_cast<ProjectionScalar>(fluid_count);
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          output(k,j,i) = fluid_mask(k,j,i) == 1 ? input(k,j,i)-mean : 0;
        });
        return mean;
      };
      // The previous projection is a rolling initial guess, not a prognostic RK variable. Reapply the pressure-space
      // constraints to remove roundoff drift from previous solves. Immersed geometry is fixed after initialization.
      project_pressure(pressure,pressure);

      ProjectionScalar const projection_beta =
          static_cast<ProjectionScalar>(config.momentum_hyperviscosity);
      ProjectionScalar momentum_hvcoef = projection_beta/std::pow(ProjectionScalar(2),ord);
      if ((ord/2)%2 == 0) momentum_hvcoef *= -1;
      ProjectionScalar const pressure_beta =
          static_cast<ProjectionScalar>(config.pressure_hyperviscosity);
      bool const pressure_hv_enabled = pressure_beta != 0;
      ProjectionScalar pressure_hvcoef = pressure_beta/std::pow(ProjectionScalar(2),ord);
      if ((ord/2)%2 == 1) pressure_hvcoef *= -1;
      Projection4d candidate("anelastic_projection_halos",4,nz+2*hs,ny+2*hs,nx+2*hs);
      Projection3d ru_x("anelastic_ru_x",nz,ny,nx+1);
      Projection3d rv_y("anelastic_rv_y",nz,ny+1,nx);
      Projection3d rw_z("anelastic_rw_z",nz+1,ny,nx);
      Projection3d pp_x("anelastic_pp_x",nz,ny,nx+1);
      Projection3d pp_y("anelastic_pp_y",nz,ny+1,nx);
      Projection3d pp_z("anelastic_pp_z",nz+1,ny,nx);
      Projection3d hv_x("anelastic_projection_hv_x",nz,ny,nx+1);
      Projection3d hv_y("anelastic_projection_hv_y",nz,ny+1,nx);
      Projection3d hv_z("anelastic_projection_hv_z",nz+1,ny,nx);

      // Add unit-free directional hyperviscosity to the fixed provisional momentum. Each face metric cancels the
      // matching inverse metric in the divergence below. Only the normal component samples zero immersed momentum;
      // transverse components are absent from each directional stencil.
      candidate = 0;
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        candidate(idRU,hs+k,hs+j,hs+i) = momentum_rhs(idRU,k,j,i);
        candidate(idRV,hs+k,hs+j,hs+i) = momentum_rhs(idRV,k,j,i);
        candidate(idRW,hs+k,hs+j,hs+i) = momentum_rhs(idRW,k,j,i);
      });
      if (ord > 1) coupler.halo_exchange_x(candidate,hs);
      if (ord > 1) coupler.halo_exchange_y(candidate,hs);
      projection_boundary_conditions(coupler,candidate);
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx+1), KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<ProjectionScalar,ord> s;
        for (int ii = 0; ii < ord; ii++) s(ii) = candidate(idRU,hs+k,hs+j,i+ii);
        hv_x(k,j,i) = momentum_hvcoef*dx_proj*TransformMatrices::edge_hvder(s);
        if (immersed(hs+k,hs+j,hs+i-1) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) hv_x(k,j,i) = 0;
        if ((px == 0 && i == 0 && wall_x1) || (px == nproc_x-1 && i == nx && wall_x2)) hv_x(k,j,i) = 0;
      });
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny+1,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<ProjectionScalar,ord> s;
        for (int jj = 0; jj < ord; jj++) s(jj) = candidate(idRV,hs+k,j+jj,hs+i);
        hv_y(k,j,i) = momentum_hvcoef*dy_proj*TransformMatrices::edge_hvder(s);
        if (immersed(hs+k,hs+j-1,hs+i) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) hv_y(k,j,i) = 0;
        if ((py == 0 && j == 0 && wall_y1) || (py == nproc_y-1 && j == ny && wall_y2)) hv_y(k,j,i) = 0;
      });
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz+1,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<ProjectionScalar,ord> s;
        for (int kk = 0; kk < ord; kk++) s(kk) = candidate(idRW,k+kk,hs+j,hs+i);
        ProjectionScalar const dzloc = ProjectionScalar(0.5)*(
            static_cast<ProjectionScalar>(dz(std::max(0,k-1))) +
            static_cast<ProjectionScalar>(dz(std::min(nz-1,k))));
        hv_z(k,j,i) = momentum_hvcoef*dzloc*TransformMatrices::edge_hvder(s);
        if (immersed(hs+k-1,hs+j,hs+i) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) hv_z(k,j,i) = 0;
        if ((k == 0 && wall_z1) || (k == nz && wall_z2)) hv_z(k,j,i) = 0;
      });
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        if (fluid_mask(k,j,i) == 1) {
          momentum_rhs(idRU,k,j,i) += (hv_x(k,j,i+1)-hv_x(k,j,i))*r_dx;
          momentum_rhs(idRV,k,j,i) += (hv_y(k,j+1,i)-hv_y(k,j,i))*r_dy;
          momentum_rhs(idRW,k,j,i) +=
              (hv_z(k+1,j,i)-hv_z(k,j,i))/static_cast<ProjectionScalar>(dz(k));
        }
      });

      // Convert cell-centered momentum to normal cell-edge mass fluxes. Only the normal component samples zero immersed
      // momentum, and every face touching an immersed cell is set to zero before it enters the divergence operator.
      auto compute_mass_fluxes = [&] (Projection4d const & momentum) {
        candidate = 0;
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          ProjectionScalar const r_rho_h = ProjectionScalar(1)/static_cast<ProjectionScalar>(rho_h(hs+k));
          candidate(idRU,hs+k,hs+j,hs+i) = momentum(idRU,k,j,i)*r_rho_h;
          candidate(idRV,hs+k,hs+j,hs+i) = momentum(idRV,k,j,i)*r_rho_h;
          candidate(idRW,hs+k,hs+j,hs+i) = momentum(idRW,k,j,i)*r_rho_h;
        });
        if (ord > 1) coupler.halo_exchange_x(candidate,hs);
        if (ord > 1) coupler.halo_exchange_y(candidate,hs);
        projection_boundary_conditions(coupler,candidate);
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx+1), KOKKOS_LAMBDA (int k, int j, int i) {
          SArray<ProjectionScalar,ord> s;
          for (int ii = 0; ii < ord; ii++) s(ii) = candidate(idRU,hs+k,hs+j,i+ii);
          ru_x(k,j,i) = static_cast<ProjectionScalar>(rho_h(hs+k))*TransformMatrices::edge_val(s);
          if (immersed(hs+k,hs+j,hs+i-1) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) ru_x(k,j,i) = 0;
          if ((px == 0 && i == 0 && wall_x1) || (px == nproc_x-1 && i == nx && wall_x2)) ru_x(k,j,i) = 0;
        });
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny+1,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          SArray<ProjectionScalar,ord> s;
          for (int jj = 0; jj < ord; jj++) s(jj) = candidate(idRV,hs+k,j+jj,hs+i);
          rv_y(k,j,i) = static_cast<ProjectionScalar>(rho_h(hs+k))*TransformMatrices::edge_val(s);
          if (immersed(hs+k,hs+j-1,hs+i) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) rv_y(k,j,i) = 0;
          if ((py == 0 && j == 0 && wall_y1) || (py == nproc_y-1 && j == ny && wall_y2)) rv_y(k,j,i) = 0;
        });
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz+1,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          SArray<ProjectionScalar,ord> s;
          for (int kk = 0; kk < ord; kk++) s(kk) = candidate(idRW,k+kk,hs+j,hs+i);
          SArray<ProjectionScalar,ord> s_metric = s;
          for (int kk = 0; kk < ord; kk++) {
            s_metric(kk) *= static_cast<ProjectionScalar>(dz(std::max(0,std::min(nz-1,k-hs+kk))));
          }
          rw_z(k,j,i) = static_cast<ProjectionScalar>(rho_h_edge(k))*TransformMatrices::edge_val(s_metric)/
                        static_cast<ProjectionScalar>(metjac_edge(k));
          if (immersed(hs+k-1,hs+j,hs+i) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) rw_z(k,j,i) = 0;
          if ((k == 0 && wall_z1) || (k == nz && wall_z2)) rw_z(k,j,i) = 0;
        });
      };

      // Apply a high-order pressure derivative directly to normal face mass fluxes. The derivative is reconstructed at
      // the half node from cell averages and retains a nonzero response to odd-even pressure modes. Cell-centered
      // momentum is updated separately with the existing high-order pressure gradient below.
      auto compute_pressure_corrected_mass_fluxes = [&] (Projection3d const & pp, bool add_rhs_flux) {
        if (add_rhs_flux) compute_mass_fluxes(momentum_rhs);
        candidate = 0;
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          candidate(idPP,hs+k,hs+j,hs+i) = pp(k,j,i);
        });
        if (ord > 1) coupler.halo_exchange_x(candidate,hs);
        if (ord > 1) coupler.halo_exchange_y(candidate,hs);
        projection_boundary_conditions(coupler,candidate);
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx+1), KOKKOS_LAMBDA (int k, int j, int i) {
          SArray<ProjectionScalar,ord> s;
          SArray<bool,ord> imm;
          for (int ii = 0; ii < ord; ii++) {
            s(ii)   = candidate(idPP,hs+k,hs+j,i+ii);
            imm(ii) = immersed(hs+k,hs+j,i+ii) > imm_th;
          }
          modify_stencil_immersed_der0(s,imm);
          ProjectionScalar const pressure_flux = -dt_proj*TransformMatrices::edge_der(s)*r_dx;
          ru_x(k,j,i) = (add_rhs_flux ? ru_x(k,j,i) : 0)+pressure_flux;
          if (immersed(hs+k,hs+j,hs+i-1) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) ru_x(k,j,i) = 0;
          if ((px == 0 && i == 0 && wall_x1) || (px == nproc_x-1 && i == nx && wall_x2)) ru_x(k,j,i) = 0;
        });
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny+1,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          SArray<ProjectionScalar,ord> s;
          SArray<bool,ord> imm;
          for (int jj = 0; jj < ord; jj++) {
            s(jj)   = candidate(idPP,hs+k,j+jj,hs+i);
            imm(jj) = immersed(hs+k,j+jj,hs+i) > imm_th;
          }
          modify_stencil_immersed_der0(s,imm);
          ProjectionScalar const pressure_flux = -dt_proj*TransformMatrices::edge_der(s)*r_dy;
          rv_y(k,j,i) = (add_rhs_flux ? rv_y(k,j,i) : 0)+pressure_flux;
          if (immersed(hs+k,hs+j-1,hs+i) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) rv_y(k,j,i) = 0;
          if ((py == 0 && j == 0 && wall_y1) || (py == nproc_y-1 && j == ny && wall_y2)) rv_y(k,j,i) = 0;
        });
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz+1,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          SArray<ProjectionScalar,ord> s;
          SArray<bool,ord> imm;
          for (int kk = 0; kk < ord; kk++) {
            s(kk)   = candidate(idPP,k+kk,hs+j,hs+i);
            imm(kk) = immersed(k+kk,hs+j,hs+i) > imm_th;
          }
          modify_stencil_immersed_der0(s,imm);
          ProjectionScalar const pressure_flux = -dt_proj*TransformMatrices::edge_der(s)/
                                                  static_cast<ProjectionScalar>(metjac_edge(k));
          rw_z(k,j,i) = (add_rhs_flux ? rw_z(k,j,i) : 0)+pressure_flux;
          if (immersed(hs+k-1,hs+j,hs+i) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) rw_z(k,j,i) = 0;
          if ((k == 0 && wall_z1) || (k == nz && wall_z2)) rw_z(k,j,i) = 0;
        });
      };

      // Form the momentum response to pressure. Pressure uses a zero-normal-derivative extension at immersed cells before
      // both edge interpolation and hyperviscosity; the resulting momentum is then set to zero in immersed cells. The
      // hyperviscosity has the same timestep and inverse-length-squared scaling as the pressure-divergence operator.
      auto compute_momentum_from_pressure = [&] (Projection3d const & pp, Projection4d const & momentum, bool add_rhs) {
        candidate = 0;
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          candidate(idPP,hs+k,hs+j,hs+i) = pp(k,j,i);
        });
        if (ord > 1) coupler.halo_exchange_x(candidate,hs);
        if (ord > 1) coupler.halo_exchange_y(candidate,hs);
        projection_boundary_conditions(coupler,candidate);
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx+1), KOKKOS_LAMBDA (int k, int j, int i) {
          SArray<ProjectionScalar,ord> s;
          SArray<bool,ord> imm;
          for (int ii = 0; ii < ord; ii++) {
            s(ii)   = candidate(idPP,hs+k,hs+j,i+ii);
            imm(ii) = immersed(hs+k,hs+j,i+ii) > imm_th;
          }
          modify_stencil_immersed_der0(s,imm);
          if (add_rhs) pp_x(k,j,i) = TransformMatrices::edge_val(s);
          if (pressure_hv_enabled) {
            hv_x(k,j,i) = pressure_hvcoef*dt_proj*r_dx*TransformMatrices::edge_hvder(s);
            if (immersed(hs+k,hs+j,hs+i-1) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) hv_x(k,j,i) = 0;
            if ((px == 0 && i == 0 && wall_x1) || (px == nproc_x-1 && i == nx && wall_x2)) hv_x(k,j,i) = 0;
          }
        });
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny+1,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          SArray<ProjectionScalar,ord> s;
          SArray<bool,ord> imm;
          for (int jj = 0; jj < ord; jj++) {
            s(jj)   = candidate(idPP,hs+k,j+jj,hs+i);
            imm(jj) = immersed(hs+k,j+jj,hs+i) > imm_th;
          }
          modify_stencil_immersed_der0(s,imm);
          if (add_rhs) pp_y(k,j,i) = TransformMatrices::edge_val(s);
          if (pressure_hv_enabled) {
            hv_y(k,j,i) = pressure_hvcoef*dt_proj*r_dy*TransformMatrices::edge_hvder(s);
            if (immersed(hs+k,hs+j-1,hs+i) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) hv_y(k,j,i) = 0;
          }
        });
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz+1,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          SArray<ProjectionScalar,ord> s;
          SArray<bool,ord> imm;
          for (int kk = 0; kk < ord; kk++) {
            s(kk)   = candidate(idPP,k+kk,hs+j,hs+i);
            imm(kk) = immersed(k+kk,hs+j,hs+i) > imm_th;
          }
          modify_stencil_immersed_der0(s,imm);
          if (add_rhs) {
            SArray<ProjectionScalar,ord> s_metric = s;
            for (int kk = 0; kk < ord; kk++) {
              s_metric(kk) *= static_cast<ProjectionScalar>(dz(std::max(0,std::min(nz-1,k-hs+kk))));
            }
            pp_z(k,j,i) = TransformMatrices::edge_val(s_metric)/static_cast<ProjectionScalar>(metjac_edge(k));
          }
          if (pressure_hv_enabled) {
            ProjectionScalar const dzloc = ProjectionScalar(0.5)*(
                static_cast<ProjectionScalar>(dz(std::max(0,k-1))) +
                static_cast<ProjectionScalar>(dz(std::min(nz-1,k))));
            hv_z(k,j,i) = pressure_hvcoef*dt_proj/dzloc*TransformMatrices::edge_hvder(s);
            if (immersed(hs+k-1,hs+j,hs+i) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) hv_z(k,j,i) = 0;
            if ((k == 0 && wall_z1) || (k == nz && wall_z2)) hv_z(k,j,i) = 0;
          }
        });
        if (add_rhs) {
          yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
            if (fluid_mask(k,j,i) == 1) {
              momentum(idPP,k,j,i) = 0;
              momentum(idRU,k,j,i) = momentum_rhs(idRU,k,j,i)-dt_proj*(pp_x(k,j,i+1)-pp_x(k,j,i))*r_dx;
              momentum(idRV,k,j,i) = momentum_rhs(idRV,k,j,i)-dt_proj*(pp_y(k,j+1,i)-pp_y(k,j,i))*r_dy;
              momentum(idRW,k,j,i) = momentum_rhs(idRW,k,j,i)-dt_proj*(pp_z(k+1,j,i)-pp_z(k,j,i))/
                                     static_cast<ProjectionScalar>(dz(k));
            } else {
              momentum(idPP,k,j,i) = 0;
              momentum(idRU,k,j,i) = 0;
              momentum(idRV,k,j,i) = 0;
              momentum(idRW,k,j,i) = 0;
            }
          });
        }
      };

      compute_mass_fluxes(momentum_rhs);
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        ProjectionScalar const divergence = (ru_x(k,j,i+1)-ru_x(k,j,i))*r_dx +
                                            (rv_y(k,j+1,i)-rv_y(k,j,i))*r_dy +
                                            (rw_z(k+1,j,i)-rw_z(k,j,i))/
                                            static_cast<ProjectionScalar>(dz(k));
        pressure_rhs(k,j,i) = fluid_mask(k,j,i) == 1 ? -divergence : 0;
      });
      ProjectionScalar const pressure_rhs_mean = project_pressure(pressure_rhs,pressure_rhs);
      if constexpr (yakl::kokkos_debug) {
        coupler.set_option<real>("dycore_anelastic_last_pressure_rhs_mean",pressure_rhs_mean);
        if (diagnostics && coupler.is_mainproc()) {
          std::cout << "Anelastic pressure RHS fluid mean before projection: " << pressure_rhs_mean << std::endl;
        }
      }

      auto compute_Ax = [&] (yakl::Array<ProjectionScalar *> const & x_in,
                             yakl::Array<ProjectionScalar *> const & Ax_out, MPI_Comm comm) {
        auto pp = x_in.reshape(nz,ny,nx);
        auto Ax = Ax_out.reshape(nz,ny,nx);
        project_pressure(pp,pressure_projected);
        // The cell response supplies the pressure-HV stencil, while continuity uses the compact face correction.
        if (pressure_hv_enabled) compute_momentum_from_pressure(pressure_projected,momentum_work,false);
        compute_pressure_corrected_mass_fluxes(pressure_projected,false);
        yakl::parallel_for(YAKL_AUTO_LABEL(),SimpleBounds<3>(nz,ny,nx),KOKKOS_LAMBDA (int k, int j, int i) {
          ProjectionScalar value = 0;
          if (fluid_mask(k,j,i) == 1) {
            ProjectionScalar const pressure_momentum_divergence =
                (ru_x(k,j,i+1)-ru_x(k,j,i))*r_dx +
                (rv_y(k,j+1,i)-rv_y(k,j,i))*r_dy +
                (rw_z(k+1,j,i)-rw_z(k,j,i))/static_cast<ProjectionScalar>(dz(k));
            ProjectionScalar const pressure_hv = pressure_hv_enabled ?
                (hv_x(k,j,i+1)-hv_x(k,j,i))*r_dx +
                (hv_y(k,j+1,i)-hv_y(k,j,i))*r_dy +
                (hv_z(k+1,j,i)-hv_z(k,j,i))/static_cast<ProjectionScalar>(dz(k)) : 0;
            value = pressure_momentum_divergence+pressure_hv+
                    dt_proj*screening_inv_length_squared*pressure_projected(k,j,i);
          }
          Ax(k,j,i) = value;
        });
        project_pressure(Ax,Ax);
        (void) comm;
      };
      auto compute_Ax_and_local_dot = [&] (yakl::Array<ProjectionScalar *> const & x_in,
                                           yakl::Array<ProjectionScalar *> const & Ax_out, MPI_Comm comm) {
        compute_Ax(x_in,Ax_out,comm);
        return YaklConjGrad<ProjectionScalar>::local_dot(x_in,Ax_out);
      };

      // The cached diagonal is for the unscreened unit-timestep operator. Add the timestep-dependent screening term
      // before dividing by dt. Projecting the result back into the mean-zero fluid pressure space makes this
      // P*D^{-1}*P preconditioner symmetric, as required by preconditioned CG.
      auto jacobi_preconditioner = [&] (yakl::Array<ProjectionScalar *> const & r_in,
                                        yakl::Array<ProjectionScalar *> const & z_out, MPI_Comm comm) {
        auto r = r_in.reshape(nz,ny,nx);
        auto z = z_out.reshape(nz,ny,nx);
        ProjectionScalar const r_dt = ProjectionScalar(1)/dt_proj;
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          ProjectionScalar const inv_diagonal_spatial =
              static_cast<ProjectionScalar>(inv_diagonal_dtless(k,j,i));
          ProjectionScalar const inv_diagonal = inv_diagonal_spatial/
              (ProjectionScalar(1)+screening_inv_length_squared*inv_diagonal_spatial);
          z(k,j,i) = fluid_mask(k,j,i) == 1 ? r(k,j,i)*inv_diagonal*r_dt : 0;
        });
        project_pressure(z,z);
        (void) comm;
      };

      // Classical additive Schwarz with full-column, globally anchored overlapping tiles. The local operator is the
      // second-order pressure-flux Laplacian with the fine-grid immersed connectivity and physical no-flux walls.
      // Fixed Chebyshev-Richardson steps approximate each local inverse and retain a fixed linear SPD map.
      auto schwarz_preconditioner = [&] (yakl::Array<ProjectionScalar *> const & r_in,
                                         yakl::Array<ProjectionScalar *> const & z_out, MPI_Comm comm) {
        auto r = r_in.reshape(nz,ny,nx);
        auto z = z_out.reshape(nz,ny,nx);
        int const tile_nx = config.schwarz_tile_nx;
        int const tile_ny = config.schwarz_tile_ny;
        int const overlap = config.schwarz_overlap;
        int const degree = config.schwarz_chebyshev_degree;
        int const schwarz_hs = coupler.get_option<int>("dycore_anelastic_schwarz_halo");
        int const num_tiles = coupler.get_option<int>("dycore_anelastic_schwarz_num_local_tiles");
        ProjectionScalar const lambda_min = static_cast<ProjectionScalar>(
            config.schwarz_chebyshev_lambda_min);
        ProjectionScalar const lambda_max = static_cast<ProjectionScalar>(
            config.schwarz_chebyshev_lambda_max);
        if (!(lambda_min > 0 && lambda_min < lambda_max && lambda_max >= 2)) {
          endrun("ERROR: invalid anelastic Schwarz Chebyshev spectral bounds");
        }
        int const max_tile_x = tile_nx+2*overlap;
        int const max_tile_y = tile_ny+2*overlap;
        auto const tiles = dm.get<int const,2>("dycore_anelastic_schwarz_tiles");
        bool const periodic_x = coupler.get_option<std::string>("bc_x1") == "periodic";
        bool const periodic_y = coupler.get_option<std::string>("bc_y1") == "periodic";
        bool const periodic_z = coupler.get_option<std::string>("bc_z1") == "periodic";
        int const i_beg_int = static_cast<int>(i_beg);
        int const j_beg_int = static_cast<int>(j_beg);
        int const nx_glob_int = static_cast<int>(nx_glob);
        int const ny_glob_int = static_cast<int>(ny_glob);

        auto r_halos = dm.get<ProjectionScalar,4>("dycore_anelastic_schwarz_residual_halos");
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          r_halos(0,schwarz_hs+k,schwarz_hs+j,schwarz_hs+i) = r(k,j,i);
        });
        if (coupler.get_nranks() == 1) {
          yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny+2*schwarz_hs,nx+2*schwarz_hs),
                             KOKKOS_LAMBDA (int k, int jh, int ih) {
            int const j = jh-schwarz_hs;
            int const i = ih-schwarz_hs;
            if (j >= 0 && j < ny && i >= 0 && i < nx) return;
            bool const valid = (periodic_y || (j >= 0 && j < ny)) &&
                               (periodic_x || (i >= 0 && i < nx));
            int const jj = (j%ny+ny)%ny;
            int const ii = (i%nx+nx)%nx;
            r_halos(0,schwarz_hs+k,jh,ih) = valid ? r(k,jj,ii) : 0;
          });
        } else {
          coupler.halo_exchange(r_halos,schwarz_hs);
        }

        auto local_rhs = dm.get<ProjectionScalar,4>("dycore_anelastic_schwarz_local_rhs");
        auto local_x   = dm.get<ProjectionScalar,4>("dycore_anelastic_schwarz_local_x_0");
        auto local_Ax  = dm.get<ProjectionScalar,4>("dycore_anelastic_schwarz_local_x_1");
        auto const coefficients = dm.get<float const,5>("dycore_anelastic_schwarz_coefficients");
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<4>(num_tiles,nz,max_tile_y,max_tile_x),
                           KOKKOS_LAMBDA (int tile, int k, int jj, int ii) {
          int const tx = tiles(tile,2);
          int const ty = tiles(tile,3);
          local_rhs(tile,k,jj,ii) = 0;
          local_x  (tile,k,jj,ii) = 0;
          local_Ax (tile,k,jj,ii) = 0;
          if (ii >= tx || jj >= ty || coefficients(1,tile,k,jj,ii) <= 0) return;
          int gi = tiles(tile,0)+ii;
          int gj = tiles(tile,1)+jj;
          gi = periodic_x ? (gi%nx_glob_int+nx_glob_int)%nx_glob_int : gi;
          gj = periodic_y ? (gj%ny_glob_int+ny_glob_int)%ny_glob_int : gj;
          int di = gi-i_beg_int;
          int dj = gj-j_beg_int;
          if (periodic_x && di < -schwarz_hs) di += nx_glob_int;
          if (periodic_x && di >= nx+schwarz_hs) di -= nx_glob_int;
          if (periodic_y && dj < -schwarz_hs) dj += ny_glob_int;
          if (periodic_y && dj >= ny+schwarz_hs) dj -= ny_glob_int;
          int const ih = schwarz_hs+di;
          int const jh = schwarz_hs+dj;
          local_rhs(tile,k,jj,ii) = r_halos(0,schwarz_hs+k,jh,ih);
        });

        ProjectionScalar const theta = ProjectionScalar(0.5)*(lambda_max+lambda_min);
        ProjectionScalar const delta = ProjectionScalar(0.5)*(lambda_max-lambda_min);
        for (int step = 0; step < degree; step++) {
          ProjectionScalar const angle = ProjectionScalar(M_PI)*(ProjectionScalar(2*step+1))/
                                         ProjectionScalar(2*degree);
          ProjectionScalar const omega = ProjectionScalar(1)/(theta-delta*std::cos(angle));
          yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<4>(num_tiles,nz,max_tile_y,max_tile_x),
                             KOKKOS_LAMBDA (int tile, int k, int jj, int ii) {
            int const tx = tiles(tile,2);
            int const ty = tiles(tile,3);
            ProjectionScalar const inv_diagonal_spatial = coefficients(1,tile,k,jj,ii);
            if (ii >= tx || jj >= ty || inv_diagonal_spatial <= 0) {
              local_Ax(tile,k,jj,ii) = 0;
              return;
            }
            ProjectionScalar const inv_diagonal = inv_diagonal_spatial/
                (ProjectionScalar(1)+screening_inv_length_squared*inv_diagonal_spatial);
            ProjectionScalar const center = local_x(tile,k,jj,ii);
            ProjectionScalar Ax_value =
                (static_cast<ProjectionScalar>(coefficients(0,tile,k,jj,ii))+
                 screening_inv_length_squared)*center;
            if (ii > 0)    Ax_value -= static_cast<ProjectionScalar>(coefficients(2,tile,k,jj,ii))*
                                          local_x(tile,k,jj,ii-1);
            if (ii+1 < tx) Ax_value -= static_cast<ProjectionScalar>(coefficients(3,tile,k,jj,ii))*
                                          local_x(tile,k,jj,ii+1);
            if (jj > 0)    Ax_value -= static_cast<ProjectionScalar>(coefficients(4,tile,k,jj,ii))*
                                          local_x(tile,k,jj-1,ii);
            if (jj+1 < ty) Ax_value -= static_cast<ProjectionScalar>(coefficients(5,tile,k,jj,ii))*
                                          local_x(tile,k,jj+1,ii);
            int const km = k > 0 ? k-1 : nz-1;
            int const kp = k+1 < nz ? k+1 : 0;
            if (k > 0 || periodic_z) {
              Ax_value -= static_cast<ProjectionScalar>(coefficients(6,tile,k,jj,ii))*local_x(tile,km,jj,ii);
            }
            if (k+1 < nz || periodic_z) {
              Ax_value -= static_cast<ProjectionScalar>(coefficients(7,tile,k,jj,ii))*local_x(tile,kp,jj,ii);
            }
            local_Ax(tile,k,jj,ii) = center+omega*(local_rhs(tile,k,jj,ii)-Ax_value)*inv_diagonal;
          });
          std::swap(local_x,local_Ax);
        }

        ProjectionScalar const r_dt = ProjectionScalar(1)/dt_proj;
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          ProjectionScalar correction = 0;
          int const gi = i_beg_int+i;
          int const gj = j_beg_int+j;
          for (int tile = 0; tile < num_tiles; tile++) {
            int const i0 = tiles(tile,0);
            int const j0 = tiles(tile,1);
            int ii_found = -1;
            int jj_found = -1;
            for (int shift = -1; shift <= 1; shift++) {
              int const ii = gi+shift*nx_glob_int-i0;
              int const jj = gj+shift*ny_glob_int-j0;
              if (ii >= 0 && ii < tiles(tile,2)) ii_found = ii;
              if (jj >= 0 && jj < tiles(tile,3)) jj_found = jj;
            }
            if (ii_found >= 0 && jj_found >= 0) correction += local_x(tile,k,jj_found,ii_found);
          }
          z(k,j,i) = fluid_mask(k,j,i) == 1 ? correction*r_dt : 0;
        });
        project_pressure(z,z);
        (void) comm;
      };

      auto multigrid_preconditioner = [&] (yakl::Array<ProjectionScalar *> const & r_in,
                                            yakl::Array<ProjectionScalar *> const & z_out, MPI_Comm comm) {
        if (!config.multigrid || !config.multigrid->initialized()) {
          endrun("ERROR: anelastic multigrid preconditioner was not initialized");
        }
        config.multigrid->apply(r_in,z_out,screening_inv_length_squared,dt_proj);
        project_pressure(z_out.reshape(nz,ny,nx),z_out.reshape(nz,ny,nx));
        (void) comm;
      };

      auto geometric_multigrid_preconditioner = [&] (yakl::Array<ProjectionScalar *> const & r_in,
                                                       yakl::Array<ProjectionScalar *> const & z_out,
                                                       MPI_Comm comm) {
        if (!config.geometric_multigrid || !config.geometric_multigrid->initialized()) {
          endrun("ERROR: anelastic geometric multigrid preconditioner was not initialized");
        }
        config.geometric_multigrid->apply(r_in,z_out,screening_inv_length_squared,dt_proj);
        project_pressure(z_out.reshape(nz,ny,nx),z_out.reshape(nz,ny,nx));
        (void) comm;
      };

      auto tensor_line_multigrid_preconditioner = [&] (yakl::Array<ProjectionScalar *> const & r_in,
                                                        yakl::Array<ProjectionScalar *> const & z_out,
                                                        MPI_Comm comm) {
        if (!config.tensor_line_multigrid || !config.tensor_line_multigrid->initialized()) {
          endrun("ERROR: anelastic tensor-line multigrid preconditioner was not initialized");
        }
        config.tensor_line_multigrid->apply(r_in,z_out,screening_inv_length_squared,dt_proj);
        project_pressure(z_out.reshape(nz,ny,nx),z_out.reshape(nz,ny,nx));
        (void) comm;
      };

      YaklRestartedGMRES<ProjectionScalar> gmres;
      typename YaklRestartedGMRES<ProjectionScalar>::Options opts;
      opts.restart = config.gmres_restart;
      opts.max_iters = config.linear_solver_max_iterations;
      opts.rel_tol = static_cast<ProjectionScalar>(
          config.linear_solver_relative_tolerance);
      opts.abs_tol = static_cast<ProjectionScalar>(
          config.linear_solver_absolute_tolerance);
      opts.verbose = config.linear_solver_verbose;
      opts.reorthogonalize = config.gmres_reorthogonalize;
      MPI_Comm const comm = coupler.get_parallel_comm().get_mpi_comm();

      bool const cg_check = config.check_cg_compatibility;
      bool use_cg = config.use_conjugate_gradient;
      if constexpr (yakl::kokkos_debug) {
        bool const cg_checked = coupler.get_option<bool>("dycore_anelastic_cg_compatibility_checked",false);
        if (cg_check && !cg_checked) {
        ProjectionScalar cg_symmetry_error = std::numeric_limits<ProjectionScalar>::infinity();
        bool cg_positive = false;
        Projection3d x       ("anelastic_cg_check_x"       ,nz,ny,nx);
        Projection3d y       ("anelastic_cg_check_y"       ,nz,ny,nx);
        Projection3d checker ("anelastic_cg_check_checker" ,nz,ny,nx);
        Projection3d Ax_check("anelastic_cg_check_Ax"      ,nz,ny,nx);
        Projection3d Ay_check("anelastic_cg_check_Ay"      ,nz,ny,nx);
        Projection3d Achecker("anelastic_cg_check_Achecker",nz,ny,nx);
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          int const i_glob = static_cast<int>(i_beg)+i;
          int const j_glob = static_cast<int>(j_beg)+j;
          x(k,j,i) = std::sin(ProjectionScalar(2*M_PI)*(i_glob+ProjectionScalar(0.5))/nx_glob) +
                       ProjectionScalar(0.3)*std::cos(ProjectionScalar(2*M_PI)*
                       (j_glob+ProjectionScalar(0.5))/ny_glob);
          y(k,j,i) = std::cos(ProjectionScalar(0.019)*(1+i_glob+nx_glob*(j_glob+ny_glob*k)));
          checker(k,j,i) = (i_glob+j_glob+k)%2 == 0 ? 1 : -1;
        });
        project_pressure(x,x);
        project_pressure(y,y);
        project_pressure(checker,checker);
        auto dot_fields = [&] (Projection3d const & a, Projection3d const & b) {
          yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
            projection_work(k,j,i) = a(k,j,i)*b(k,j,i);
          });
          return coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(projection_work),MPI_SUM);
        };
        auto pressure_hv_quadratic = [&] (Projection3d const & mode) {
          if (!pressure_hv_enabled) return ProjectionScalar(0);
          yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
            ProjectionScalar const pressure_hv = (hv_x(k,j,i+1)-hv_x(k,j,i))*r_dx +
                                                 (hv_y(k,j+1,i)-hv_y(k,j,i))*r_dy +
                                                 (hv_z(k+1,j,i)-hv_z(k,j,i))/
                                                 static_cast<ProjectionScalar>(dz(k));
            projection_work(k,j,i) = mode(k,j,i)*pressure_hv;
          });
          return coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(projection_work),MPI_SUM);
        };
        compute_Ax(x.collapse(),Ax_check.collapse(),comm);
        ProjectionScalar const xAx = dot_fields(x,Ax_check);
        ProjectionScalar const xHx = pressure_hv_quadratic(x);
        ProjectionScalar const x2 = dot_fields(x,x);
        compute_Ax(y.collapse(),Ay_check.collapse(),comm);
        ProjectionScalar const xAy = dot_fields(x,Ay_check);
        ProjectionScalar const yAx = dot_fields(y,Ax_check);
        ProjectionScalar const yAy = dot_fields(y,Ay_check);
        compute_Ax(checker.collapse(),Achecker.collapse(),comm);
        ProjectionScalar const checker_A_checker = dot_fields(checker,Achecker);
        ProjectionScalar const checker_H_checker = pressure_hv_quadratic(checker);
        ProjectionScalar const checker2 = dot_fields(checker,checker);
        ProjectionScalar const symmetry_scale =
            std::max(std::sqrt(std::abs(xAx*yAy)),std::numeric_limits<ProjectionScalar>::min());
        cg_symmetry_error = std::abs(xAy-yAx)/symmetry_scale;
        cg_positive = xAx > 0 && yAy > 0 && checker_A_checker > 0;
        ProjectionScalar const symmetry_tolerance =
            std::max(ProjectionScalar(1.e-10),ProjectionScalar(10)*std::numeric_limits<ProjectionScalar>::epsilon());
        use_cg = cg_symmetry_error <= symmetry_tolerance && cg_positive;
        coupler.set_option<bool>("dycore_anelastic_use_cg",use_cg);
        coupler.set_option<bool>("dycore_anelastic_cg_compatibility_checked",true);
        coupler.set_option<real>("dycore_anelastic_last_cg_symmetry_error",static_cast<real>(cg_symmetry_error));
        coupler.set_option<bool>("dycore_anelastic_last_cg_positive_probes",cg_positive);
        coupler.set_option<real>("dycore_anelastic_last_smooth_mode_response",static_cast<real>(xAx/x2));
        coupler.set_option<real>("dycore_anelastic_last_smooth_mode_hv_response",static_cast<real>(xHx/x2));
        coupler.set_option<real>("dycore_anelastic_last_checker_mode_response",
                                 static_cast<real>(checker_A_checker/checker2));
        coupler.set_option<real>("dycore_anelastic_last_checker_mode_hv_response",
                                 static_cast<real>(checker_H_checker/checker2));
        if (coupler.is_mainproc()) {
          std::cout << "Anelastic CG compatibility: symmetry error = " << cg_symmetry_error
                    << ", positive probes = " << cg_positive
                    << ", smooth total/HV response = " << xAx/x2 << " / " << xHx/x2
                    << ", checker total/HV response = " << checker_A_checker/checker2 << " / "
                    << checker_H_checker/checker2 << std::endl;
        }
        }
      }

      bool const linearity_check = config.check_linearity;
      if constexpr (yakl::kokkos_debug) {
        if (linearity_check) {
        auto x = pressure.collapse().createDeviceObject();
        auto y = pressure.collapse().createDeviceObject();
        auto z = pressure.collapse().createDeviceObject();
        auto Ax = pressure.collapse().createDeviceObject();
        auto Ay = pressure.collapse().createDeviceObject();
        auto Az = pressure.collapse().createDeviceObject();
        auto work = pressure.collapse().createDeviceObject();
        int const n = x.size();
        yakl::parallel_for(YAKL_AUTO_LABEL(), n, KOKKOS_LAMBDA (int i) {
          x(i) = std::sin(ProjectionScalar(0.013)*(i+1));
          y(i) = std::cos(ProjectionScalar(0.017)*(i+1));
          z(i) = ProjectionScalar(0.37)*x(i)-ProjectionScalar(0.21)*y(i);
        });
        compute_Ax(x,Ax,comm);
        compute_Ax(y,Ay,comm);
        compute_Ax(z,Az,comm);
        yakl::parallel_for(YAKL_AUTO_LABEL(), n, KOKKOS_LAMBDA (int i) {
          ProjectionScalar const diff =
              Az(i)-(ProjectionScalar(0.37)*Ax(i)-ProjectionScalar(0.21)*Ay(i));
          work(i) = diff*diff;
          z(i) = Az(i)*Az(i);
        });
        ProjectionScalar const err =
            std::sqrt(coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(work),MPI_SUM));
        ProjectionScalar const den =
            std::sqrt(coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(z),MPI_SUM));
        ProjectionScalar const rel = den > 0 ? err/den : err;
        coupler.set_option<real>("dycore_anelastic_last_linearity_error",static_cast<real>(rel));
        if (coupler.is_mainproc()) std::cout << "Anelastic projection linearity relative error: " << rel << std::endl;
        if (rel > ProjectionScalar(1.e3)*std::numeric_limits<ProjectionScalar>::epsilon()) {
          endrun("ERROR: anelastic projection operator is nonlinear");
        }
        }
      }

      auto Ax = pressure.collapse().createDeviceObject();
      Projection3d norm_work;
      ProjectionScalar pre_div_l2 = 0;
      if constexpr (yakl::kokkos_debug) {
        norm_work = Projection3d("anelastic_projection_norm",nz,ny,nx);
        if (diagnostics) {
          yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
            norm_work(k,j,i) = pressure_rhs(k,j,i)*pressure_rhs(k,j,i);
          });
          pre_div_l2 = std::sqrt(coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(norm_work),MPI_SUM));
        }
      }

      int solver_iters = 0;
      bool solver_converged = false;
      bool const time_linear_solver = config.time_linear_solver;
      double solver_start_time = 0;
      if (time_linear_solver) {
        Kokkos::fence();
        coupler.get_parallel_comm().barrier();
        solver_start_time = MPI_Wtime();
      }
      std::string const &preconditioner = config.preconditioner;
      if (use_cg) {
        YaklConjGrad<ProjectionScalar> cg;
        typename YaklConjGrad<ProjectionScalar>::Workspace cg_workspace{
          dm.get_collapsed<ProjectionScalar>("dycore_anelastic_cg_r"),
          dm.get_collapsed<ProjectionScalar>("dycore_anelastic_cg_z"),
          dm.get_collapsed<ProjectionScalar>("dycore_anelastic_cg_p"),
          dm.get_collapsed<ProjectionScalar>("dycore_anelastic_cg_Ap"),
          dm.get_collapsed<ProjectionScalar>("dycore_anelastic_cg_s")
        };
        typename YaklConjGrad<ProjectionScalar>::Options cg_opts;
        cg_opts.max_iters = opts.max_iters;
        // Float CG recurrence norms can undershoot the independently recomputed residual slightly. Solve 10% tighter
        // internally while retaining the user-requested tolerance for the authoritative full-operator check below.
        cg_opts.rel_tol   = ProjectionScalar(0.9)*opts.rel_tol;
        cg_opts.abs_tol   = opts.abs_tol;
        cg_opts.verbose   = opts.verbose;
        typename YaklConjGrad<ProjectionScalar>::Result cg_result;
        if (preconditioner == "TensorLineMultigrid") {
          cg_result = cg.solve(pressure.collapse(),pressure_rhs.collapse(),compute_Ax,cg_workspace,cg_opts,comm,
                               tensor_line_multigrid_preconditioner,compute_Ax_and_local_dot);
        } else if (preconditioner == "GeometricMultigrid") {
          cg_result = cg.solve(pressure.collapse(),pressure_rhs.collapse(),compute_Ax,cg_workspace,cg_opts,comm,
                               geometric_multigrid_preconditioner,compute_Ax_and_local_dot);
        } else if (preconditioner == "Multigrid") {
          cg_result = cg.solve(pressure.collapse(),pressure_rhs.collapse(),compute_Ax,cg_workspace,cg_opts,comm,
                               multigrid_preconditioner,compute_Ax_and_local_dot);
        } else if (preconditioner == "Schwarz") {
          cg_result = cg.solve(pressure.collapse(),pressure_rhs.collapse(),compute_Ax,cg_workspace,cg_opts,comm,
                               schwarz_preconditioner,compute_Ax_and_local_dot);
        } else if (preconditioner == "Jacobi") {
          cg_result = cg.solve(pressure.collapse(),pressure_rhs.collapse(),compute_Ax,cg_workspace,cg_opts,comm,
                               jacobi_preconditioner,compute_Ax_and_local_dot);
        } else {
          cg_result = cg.solve(pressure.collapse(),pressure_rhs.collapse(),compute_Ax,cg_workspace,cg_opts,comm,
                               nullptr,compute_Ax_and_local_dot);
        }
        solver_iters = cg_result.iters;
        solver_converged = cg_result.converged;
      } else {
        typename YaklRestartedGMRES<ProjectionScalar>::Result gmres_result;
        if (preconditioner == "TensorLineMultigrid") {
          gmres_result = gmres.solve(pressure.collapse(),pressure_rhs.collapse(),compute_Ax,opts,comm,nullptr,
                                     tensor_line_multigrid_preconditioner);
        } else if (preconditioner == "GeometricMultigrid") {
          gmres_result = gmres.solve(pressure.collapse(),pressure_rhs.collapse(),compute_Ax,opts,comm,nullptr,
                                     geometric_multigrid_preconditioner);
        } else if (preconditioner == "Multigrid") {
          gmres_result = gmres.solve(pressure.collapse(),pressure_rhs.collapse(),compute_Ax,opts,comm,nullptr,
                                     multigrid_preconditioner);
        } else if (preconditioner == "Schwarz") {
          gmres_result = gmres.solve(pressure.collapse(),pressure_rhs.collapse(),compute_Ax,opts,comm,nullptr,
                                     schwarz_preconditioner);
        } else if (preconditioner == "Jacobi") {
          gmres_result = gmres.solve(pressure.collapse(),pressure_rhs.collapse(),compute_Ax,opts,comm,nullptr,
                                     jacobi_preconditioner);
        } else {
          gmres_result = gmres.solve(pressure.collapse(),pressure_rhs.collapse(),compute_Ax,opts,comm);
        }
        solver_iters = gmres_result.iters;
        solver_converged = gmres_result.converged;
      }
      real solver_elapsed = 0;
      if (time_linear_solver) {
        Kokkos::fence();
        coupler.get_parallel_comm().barrier();
        solver_elapsed = static_cast<real>(MPI_Wtime()-solver_start_time);
        solver_elapsed = coupler.get_parallel_comm().all_reduce(solver_elapsed,MPI_MAX);
        coupler.set_option<real>("dycore_anelastic_last_linear_solver_seconds",solver_elapsed);
      }
      coupler.set_option<std::string>("dycore_anelastic_last_linear_solver",use_cg ? "CG" : "GMRES");
      coupler.set_option<std::string>("dycore_anelastic_last_preconditioner",preconditioner);
      // Select a deterministic mean-zero representative for the unscreened operator, or only mask immersed cells when
      // screening makes pressure unique.
      project_pressure(pressure,pressure);
      if constexpr (yakl::kokkos_debug) {
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          norm_work(k,j,i) = fluid_mask(k,j,i) == 1 ? pressure(k,j,i) : 0;
        });
        ProjectionScalar const final_pressure_sum =
            coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(norm_work),MPI_SUM);
        coupler.set_option<real>("dycore_anelastic_last_pressure_mean",
                                 static_cast<real>(final_pressure_sum/static_cast<ProjectionScalar>(fluid_count)));
      }
      if constexpr (yakl::kokkos_debug) {
        if (cg_check) {
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          norm_work(k,j,i) = fluid_mask(k,j,i) == 1 ? pressure(k,j,i)*pressure(k,j,i) : 0;
        });
        ProjectionScalar const pressure_norm_sq =
            coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(norm_work),MPI_SUM);
        ProjectionScalar max_checker_correlation = 0;
        for (int mode = 1; mode < 8; mode++) {
          yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
            int parity = 0;
            if ((mode & 1) != 0) parity += static_cast<int>(i_beg)+i;
            if ((mode & 2) != 0) parity += static_cast<int>(j_beg)+j;
            if ((mode & 4) != 0) parity += k;
            ProjectionScalar const checker_value = parity%2 == 0 ? 1 : -1;
            bool const is_fluid = fluid_mask(k,j,i) == 1;
            projection_work(k,j,i) = is_fluid ? checker_value : 0;
            norm_work(k,j,i) = is_fluid ? pressure(k,j,i)*checker_value : 0;
          });
          ProjectionScalar const checker_sum =
              coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(projection_work),MPI_SUM);
          ProjectionScalar const pressure_checker_dot =
              coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(norm_work),MPI_SUM);
          ProjectionScalar const checker_norm_sq = static_cast<ProjectionScalar>(fluid_count)-
              checker_sum*checker_sum/static_cast<ProjectionScalar>(fluid_count);
          ProjectionScalar const denominator = std::sqrt(pressure_norm_sq*checker_norm_sq);
          ProjectionScalar const correlation = denominator > 0 ? std::abs(pressure_checker_dot)/denominator : 0;
          max_checker_correlation = std::max(max_checker_correlation,correlation);
        }
        coupler.set_option<real>("dycore_anelastic_last_pressure_checkerboard_correlation",
                                 static_cast<real>(max_checker_correlation));
        if (coupler.is_mainproc()) {
          std::cout << "Anelastic solved-pressure maximum checkerboard correlation = "
                    << max_checker_correlation << std::endl;
        }
        }
      }
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        pressure_io(k,j,i) = static_cast<real>(pressure(k,j,i));
      });
      // Reapply the complete operator, including pressure hyperviscosity, for the authoritative final residual.
      compute_Ax(pressure.collapse(),Ax,comm);
      auto residual = projection_work.collapse();
      auto bflat = pressure_rhs.collapse();
      auto mask_flat = fluid_mask.collapse();
      if constexpr (yakl::kokkos_debug) {
        auto norm_work_flat = norm_work.collapse();
        yakl::parallel_for(YAKL_AUTO_LABEL(), residual.size(), KOKKOS_LAMBDA (int i) {
          ProjectionScalar const r = bflat(i)-Ax(i);
          bool const is_fluid = mask_flat(i) == 1;
          residual(i) = is_fluid ? r*r : 0;
          Ax(i) = is_fluid ? bflat(i)*bflat(i) : 0;
          norm_work_flat(i) = is_fluid ? 0 : std::abs(r);
        });
        ProjectionScalar const immersed_residual_max =
            coupler.get_parallel_comm().all_reduce(yakl::intrinsics::maxval(norm_work),MPI_MAX);
        coupler.set_option<real>("dycore_anelastic_last_immersed_residual_max",
                                 static_cast<real>(immersed_residual_max));
        if (immersed_residual_max != 0) endrun("ERROR: immersed cells contributed to the anelastic solver residual");
      } else {
        yakl::parallel_for(YAKL_AUTO_LABEL(), residual.size(), KOKKOS_LAMBDA (int i) {
          ProjectionScalar const r = bflat(i)-Ax(i);
          bool const is_fluid = mask_flat(i) == 1;
          residual(i) = is_fluid ? r*r : 0;
          Ax(i) = is_fluid ? bflat(i)*bflat(i) : 0;
        });
      }
      ProjectionScalar const true_abs =
          std::sqrt(coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(residual),MPI_SUM));
      ProjectionScalar const bnorm =
          std::sqrt(coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(Ax),MPI_SUM));
      ProjectionScalar const true_rel = bnorm > 0 ? true_abs/bnorm : true_abs;
      coupler.set_option<int>("dycore_anelastic_last_linear_solver_iters",solver_iters);
      coupler.set_option<real>("dycore_anelastic_last_linear_solver_abs_res",static_cast<real>(true_abs));
      coupler.set_option<real>("dycore_anelastic_last_linear_solver_rel_res",static_cast<real>(true_rel));
      // Preserve the established option names for callers that have not yet switched to the solver-neutral diagnostics.
      coupler.set_option<int>("dycore_anelastic_last_gmres_iters",solver_iters);
      coupler.set_option<real>("dycore_anelastic_last_gmres_abs_res",static_cast<real>(true_abs));
      coupler.set_option<real>("dycore_anelastic_last_gmres_rel_res",static_cast<real>(true_rel));
      ProjectionScalar const threshold = std::max(opts.abs_tol,opts.rel_tol*bnorm);
      if (!solver_converged || !std::isfinite(true_abs) || true_abs > threshold) {
        std::ostringstream err;
        err << "ERROR: anelastic " << (use_cg ? "CG" : "GMRES") << " failed after " << solver_iters
            << " iterations; true relative residual = " << true_rel;
        endrun(err.str().c_str());
      }
      if (use_cg) {
        int const solve_count = coupler.get_option<int>("dycore_anelastic_cg_solve_count")+1;
        real const iterations = static_cast<real>(solver_iters);
        coupler.set_option<int>("dycore_anelastic_cg_solve_count",solve_count);
        coupler.set_option<real>("dycore_anelastic_cg_iteration_sum",
                                 coupler.get_option<real>("dycore_anelastic_cg_iteration_sum")+iterations);
        coupler.set_option<real>("dycore_anelastic_cg_iteration_sum_squares",
                                 coupler.get_option<real>("dycore_anelastic_cg_iteration_sum_squares")+
                                 iterations*iterations);
        coupler.set_option<int>("dycore_anelastic_cg_iteration_min",
                                std::min(coupler.get_option<int>("dycore_anelastic_cg_iteration_min"),solver_iters));
        coupler.set_option<int>("dycore_anelastic_cg_iteration_max",
                                std::max(coupler.get_option<int>("dycore_anelastic_cg_iteration_max"),solver_iters));
        if (time_linear_solver) {
          coupler.set_option<real>("dycore_anelastic_cg_seconds_sum",
                                   coupler.get_option<real>("dycore_anelastic_cg_seconds_sum")+solver_elapsed);
          coupler.set_option<real>("dycore_anelastic_cg_seconds_min",
                                   std::min(coupler.get_option<real>("dycore_anelastic_cg_seconds_min"),solver_elapsed));
          coupler.set_option<real>("dycore_anelastic_cg_seconds_max",
                                   std::max(coupler.get_option<real>("dycore_anelastic_cg_seconds_max"),solver_elapsed));
        }
      }

      compute_momentum_from_pressure(pressure,momentum_work,true);
      compute_pressure_corrected_mass_fluxes(pressure,true);
      if constexpr (yakl::kokkos_debug) {
        if (diagnostics) {
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          ProjectionScalar const divergence = fluid_mask(k,j,i) == 1 ?
              (ru_x(k,j,i+1)-ru_x(k,j,i))*r_dx +
              (rv_y(k,j+1,i)-rv_y(k,j,i))*r_dy +
              (rw_z(k+1,j,i)-rw_z(k,j,i))/static_cast<ProjectionScalar>(dz(k)) : 0;
          norm_work(k,j,i) = divergence*divergence;
        });
        ProjectionScalar const post_div_l2 =
            std::sqrt(coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(norm_work),MPI_SUM));
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          ProjectionScalar const divergence = fluid_mask(k,j,i) == 1 ?
              (ru_x(k,j,i+1)-ru_x(k,j,i))*r_dx +
              (rv_y(k,j+1,i)-rv_y(k,j,i))*r_dy +
              (rw_z(k+1,j,i)-rw_z(k,j,i))/static_cast<ProjectionScalar>(dz(k)) : 0;
          ProjectionScalar const constraint = divergence+
              dt_proj*screening_inv_length_squared*pressure(k,j,i);
          norm_work(k,j,i) = constraint*constraint;
        });
        ProjectionScalar const screened_constraint_l2 =
            std::sqrt(coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(norm_work),MPI_SUM));
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          ProjectionScalar immersed_flux = 0;
          if (immersed(hs+k,hs+j,hs+i) > imm_th || immersed(hs+k,hs+j,hs+i+1) > imm_th) {
            immersed_flux = std::max(immersed_flux,std::abs(ru_x(k,j,i+1)));
          }
          if (immersed(hs+k,hs+j,hs+i-1) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) {
            immersed_flux = std::max(immersed_flux,std::abs(ru_x(k,j,i)));
          }
          if (immersed(hs+k,hs+j,hs+i) > imm_th || immersed(hs+k,hs+j+1,hs+i) > imm_th) {
            immersed_flux = std::max(immersed_flux,std::abs(rv_y(k,j+1,i)));
          }
          if (immersed(hs+k,hs+j-1,hs+i) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) {
            immersed_flux = std::max(immersed_flux,std::abs(rv_y(k,j,i)));
          }
          if (immersed(hs+k,hs+j,hs+i) > imm_th || immersed(hs+k+1,hs+j,hs+i) > imm_th) {
            immersed_flux = std::max(immersed_flux,std::abs(rw_z(k+1,j,i)));
          }
          if (immersed(hs+k-1,hs+j,hs+i) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) {
            immersed_flux = std::max(immersed_flux,std::abs(rw_z(k,j,i)));
          }
          norm_work(k,j,i) = immersed_flux;
          if ((k == 0 && wall_z1) || (k == nz-1 && wall_z2)) {
            norm_work(k,j,i) = std::max(norm_work(k,j,i),std::abs(k == 0 ? rw_z(0,j,i) : rw_z(nz,j,i)));
          }
        });
        ProjectionScalar const boundary_flux_max =
            coupler.get_parallel_comm().all_reduce(yakl::intrinsics::maxval(norm_work),MPI_MAX);
        coupler.set_option<real>("dycore_anelastic_last_pre_div_l2",static_cast<real>(pre_div_l2));
        coupler.set_option<real>("dycore_anelastic_last_post_div_l2",static_cast<real>(post_div_l2));
        coupler.set_option<real>("dycore_anelastic_last_screened_constraint_l2",
                                 static_cast<real>(screened_constraint_l2));
        coupler.set_option<real>("dycore_anelastic_last_boundary_normal_flux_max",
                                 static_cast<real>(boundary_flux_max));
        if (coupler.is_mainproc()) {
          std::cout << "Anelastic projection: pre/post physical mass-flux divergence L2 = "
                    << pre_div_l2 << " / " << post_div_l2
                    << ", screened constraint L2 = " << screened_constraint_l2
                    << ", " << (use_cg ? "CG" : "GMRES") << " iterations = " << solver_iters
                    << ", true relative residual = " << true_rel << std::endl;
        }
        }
      }

      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        momentum_out(0,k,j,i) = momentum_work(idRU,k,j,i);
        momentum_out(1,k,j,i) = momentum_work(idRV,k,j,i);
        momentum_out(2,k,j,i) = momentum_work(idRW,k,j,i);
      });
    }


    static void init(core::Coupler &coupler, AcousticProjectionConfig const &config) {
      using yakl::SimpleBounds;
      auto const nx      = coupler.get_nx();
      auto const ny      = coupler.get_ny();
      auto const nz      = coupler.get_nz();
      auto const nx_glob = coupler.get_nx_glob();
      auto const ny_glob = coupler.get_ny_glob();
      auto const dx      = coupler.get_dx();
      auto const dy      = coupler.get_dy();
      auto const dz      = coupler.get_dz();
      auto const px      = coupler.get_px();
      auto const py      = coupler.get_py();
      auto const nproc_x = coupler.get_nproc_x();
      auto const nproc_y = coupler.get_nproc_y();
      auto &dm           = coupler.get_data_manager_readwrite();
      auto const metjac_edges = dm.get<real const,1>("dycore_metjac_edges");

      // Immersed geometry is fixed after initialization. Cache the projection mask and its global fluid-cell count so
      // every Runge-Kutta stage avoids rebuilding the mask and performing an extra global reduction.
      dm.register_and_allocate<int>("dycore_anelastic_fluid_mask",{nz,ny,nx});
      auto fluid_mask = dm.get<int,3>("dycore_anelastic_fluid_mask");
      auto immersed_halos = dm.get<real const,3>("dycore_immersed_proportion_halos");
      real const immersed_threshold = coupler.get_option<real>("immersed_threshold",0.5);
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        fluid_mask(k,j,i) = immersed_halos(hs+k,hs+j,hs+i) <= immersed_threshold ? 1 : 0;
      });
      int const fluid_count = coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(fluid_mask),MPI_SUM);
      if (fluid_count == 0) endrun("ERROR: anelastic projection has no fluid cells");
      coupler.set_option<int>("dycore_anelastic_fluid_count",fluid_count);
      coupler.set_option<int>("dycore_anelastic_cg_solve_count",0);
      coupler.set_option<real>("dycore_anelastic_cg_iteration_sum",0);
      coupler.set_option<real>("dycore_anelastic_cg_iteration_sum_squares",0);
      coupler.set_option<int>("dycore_anelastic_cg_iteration_min",std::numeric_limits<int>::max());
      coupler.set_option<int>("dycore_anelastic_cg_iteration_max",0);
      coupler.set_option<real>("dycore_anelastic_cg_seconds_sum",0);
      coupler.set_option<real>("dycore_anelastic_cg_seconds_min",std::numeric_limits<real>::max());
      coupler.set_option<real>("dycore_anelastic_cg_seconds_max",0);
      coupler.set_option<bool>("dycore_anelastic_cg_compatibility_checked",false);
      coupler.set_option<bool>("dycore_anelastic_use_cg",config.use_conjugate_gradient);

      // When enabled, screening uses the acoustic propagation length dycore_cs*dt for each projection solve.
      coupler.set_option<bool>("dycore_anelastic_screening",config.screening);

      // Reuse the Krylov vectors across every Runge-Kutta-stage pressure solve. CG receives collapsed views and performs
      // no runtime allocation.
      dm.register_and_allocate<float>("dycore_anelastic_cg_r" ,{nz,ny,nx});
      dm.register_and_allocate<float>("dycore_anelastic_cg_z" ,{nz,ny,nx});
      dm.register_and_allocate<float>("dycore_anelastic_cg_p" ,{nz,ny,nx});
      dm.register_and_allocate<float>("dycore_anelastic_cg_Ap",{nz,ny,nx});
      dm.register_and_allocate<float>("dycore_anelastic_cg_s" ,{nz,ny,nx});
      auto cg_r  = dm.get<float,3>("dycore_anelastic_cg_r");
      auto cg_z  = dm.get<float,3>("dycore_anelastic_cg_z");
      auto cg_p  = dm.get<float,3>("dycore_anelastic_cg_p");
      auto cg_Ap = dm.get<float,3>("dycore_anelastic_cg_Ap");
      auto cg_s  = dm.get<float,3>("dycore_anelastic_cg_s");
      yakl::parallel_for(YAKL_AUTO_LABEL(),SimpleBounds<3>(nz,ny,nx),KOKKOS_LAMBDA (int k, int j, int i) {
        cg_r (k,j,i) = 0;
        cg_z (k,j,i) = 0;
        cg_p (k,j,i) = 0;
        cg_Ap(k,j,i) = 0;
        cg_s (k,j,i) = 0;
      });

      // Cache the inverse diagonal of the fixed-geometry, unscreened unit-timestep local pressure operator. The mean-zero
      // P*A*P projection is deliberately omitted from this Jacobi approximation because its dense rank-one contribution
      // would destroy locality. Runtime application adds screening and divides by dt.
      dm.register_and_allocate<float>("dycore_anelastic_projection_inv_diagonal_dtless",{nz,ny,nx});
      auto inv_diagonal = dm.get<float,3>("dycore_anelastic_projection_inv_diagonal_dtless");
      float const pressure_beta = static_cast<float>(config.pressure_hyperviscosity);
      float pressure_hvcoef = pressure_beta/std::pow(2.f,ord);
      if ((ord/2)%2 == 1) pressure_hvcoef *= -1;
      bool const pressure_hv_enabled = pressure_beta != 0;
      bool const periodic_x = coupler.get_option<std::string>("bc_x1") == "periodic";
      bool const periodic_y = coupler.get_option<std::string>("bc_y1") == "periodic";
      bool const periodic_z = coupler.get_option<std::string>("bc_z1") == "periodic";
      bool const wall_x1 = coupler.get_option<std::string>("bc_x1") == "wall_free_slip";
      bool const wall_x2 = coupler.get_option<std::string>("bc_x2") == "wall_free_slip";
      bool const wall_y1 = coupler.get_option<std::string>("bc_y1") == "wall_free_slip";
      bool const wall_y2 = coupler.get_option<std::string>("bc_y2") == "wall_free_slip";
      bool const wall_z1 = coupler.get_option<std::string>("bc_z1") == "wall_free_slip";
      bool const wall_z2 = coupler.get_option<std::string>("bc_z2") == "wall_free_slip";
      int const i_beg_int = static_cast<int>(coupler.get_i_beg());
      int const j_beg_int = static_cast<int>(coupler.get_j_beg());
      int const nx_glob_int = static_cast<int>(nx_glob);
      int const ny_glob_int = static_cast<int>(ny_glob);
      float const r_dx = 1.f/static_cast<float>(dx);
      float const r_dy = 1.f/static_cast<float>(dy);
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        if (fluid_mask(k,j,i) == 0) {
          inv_diagonal(k,j,i) = 0;
          return;
        }

        auto map_index = [] (int index, int extent, bool periodic) {
          if (periodic) return (index%extent+extent)%extent;
          return std::max(0,std::min(extent-1,index));
        };
        auto x_face_response = [&] (int face, bool hv) {
          SArray<float,ord> s;
          SArray<bool,ord> imm;
          for (int ii = 0; ii < ord; ii++) {
            int const i_glob_stencil = map_index(i_beg_int+face+ii-hs,nx_glob_int,periodic_x);
            s(ii) = i_glob_stencil == i_beg_int+i ? 1.f : 0.f;
            imm(ii) = immersed_halos(hs+k,hs+j,face+ii) > immersed_threshold;
          }
          modify_stencil_immersed_der0(s,imm);
          bool const closed = immersed_halos(hs+k,hs+j,hs+face-1) > immersed_threshold ||
                              immersed_halos(hs+k,hs+j,hs+face  ) > immersed_threshold ||
                              (px == 0         && face == 0  && wall_x1) ||
                              (px == nproc_x-1 && face == nx && wall_x2);
          if (closed) return 0.f;
          return hv ? TransformMatrices::edge_hvder(s) : TransformMatrices::edge_der(s);
        };
        auto y_face_response = [&] (int face, bool hv) {
          SArray<float,ord> s;
          SArray<bool,ord> imm;
          for (int jj = 0; jj < ord; jj++) {
            int const j_glob_stencil = map_index(j_beg_int+face+jj-hs,ny_glob_int,periodic_y);
            s(jj) = j_glob_stencil == j_beg_int+j ? 1.f : 0.f;
            imm(jj) = immersed_halos(hs+k,face+jj,hs+i) > immersed_threshold;
          }
          modify_stencil_immersed_der0(s,imm);
          bool const closed = immersed_halos(hs+k,hs+face-1,hs+i) > immersed_threshold ||
                              immersed_halos(hs+k,hs+face  ,hs+i) > immersed_threshold ||
                              (py == 0         && face == 0  && wall_y1) ||
                              (py == nproc_y-1 && face == ny && wall_y2);
          if (closed) return 0.f;
          return hv ? TransformMatrices::edge_hvder(s) : TransformMatrices::edge_der(s);
        };
        auto z_face_response = [&] (int face, bool hv) {
          SArray<float,ord> s;
          SArray<bool,ord> imm;
          for (int kk = 0; kk < ord; kk++) {
            int const k_stencil = map_index(face+kk-hs,nz,periodic_z);
            s(kk) = k_stencil == k ? 1.f : 0.f;
            imm(kk) = immersed_halos(face+kk,hs+j,hs+i) > immersed_threshold;
          }
          modify_stencil_immersed_der0(s,imm);
          bool const closed = immersed_halos(hs+face-1,hs+j,hs+i) > immersed_threshold ||
                              immersed_halos(hs+face  ,hs+j,hs+i) > immersed_threshold ||
                              (face == 0  && wall_z1) || (face == nz && wall_z2);
          if (closed) return 0.f;
          return hv ? TransformMatrices::edge_hvder(s) : TransformMatrices::edge_der(s);
        };

        float const x_der_l = x_face_response(i  ,false);
        float const x_der_r = x_face_response(i+1,false);
        float const y_der_l = y_face_response(j  ,false);
        float const y_der_r = y_face_response(j+1,false);
        float const z_der_l = z_face_response(k  ,false);
        float const z_der_r = z_face_response(k+1,false);
        float const dz_cell = static_cast<float>(dz(k));
        float diagonal = (x_der_l-x_der_r)*r_dx*r_dx + (y_der_l-y_der_r)*r_dy*r_dy +
                         (z_der_l/static_cast<float>(metjac_edges(k))-
                          z_der_r/static_cast<float>(metjac_edges(k+1)))/dz_cell;
        if (pressure_hv_enabled) {
          float const x_hv_l = x_face_response(i  ,true);
          float const x_hv_r = x_face_response(i+1,true);
          float const y_hv_l = y_face_response(j  ,true);
          float const y_hv_r = y_face_response(j+1,true);
          float const z_hv_l = z_face_response(k  ,true);
          float const z_hv_r = z_face_response(k+1,true);
          float const dz_l = 0.5f*(static_cast<float>(dz(std::max(0,k-1)))+static_cast<float>(dz(k)));
          float const dz_r = 0.5f*(static_cast<float>(dz(k))+static_cast<float>(dz(std::min(nz-1,k+1))));
          diagonal += pressure_hvcoef*((x_hv_r-x_hv_l)*r_dx*r_dx + (y_hv_r-y_hv_l)*r_dy*r_dy +
                                      (z_hv_r/dz_r-z_hv_l/dz_l)/dz_cell);
        }
        inv_diagonal(k,j,i) = std::isfinite(diagonal) && diagonal > std::numeric_limits<float>::min() ?
                              1.f/diagonal : 0;
      });
      yakl::Array<int ***> invalid_diagonal("anelastic_invalid_jacobi_diagonal",nz,ny,nx);
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        invalid_diagonal(k,j,i) = fluid_mask(k,j,i) == 1 && !(inv_diagonal(k,j,i) > 0) ? 1 : 0;
      });
      int const invalid_diagonal_count =
          coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(invalid_diagonal),MPI_SUM);
      if (invalid_diagonal_count > 0) endrun("ERROR: anelastic projection has a nonpositive Jacobi diagonal");
      coupler.set_option<bool>("dycore_anelastic_use_jacobi_preconditioner",
                               config.preconditioner == "Jacobi");

      // Set up globally anchored horizontal Schwarz tiles. Each rank stores only the tiles whose overlapped support
      // intersects its owned cells. At application time those tiles are evaluated redundantly from a wide MPI halo;
      // this avoids both atomics and an asymmetric correction exchange at rank boundaries.
      std::string const &preconditioner = config.preconditioner;
      if (preconditioner != "none" && preconditioner != "Jacobi" && preconditioner != "Schwarz" &&
          preconditioner != "Multigrid" && preconditioner != "GeometricMultigrid" &&
          preconditioner != "TensorLineMultigrid") {
        endrun("ERROR: acoustic projection preconditioner must be none, Jacobi, Schwarz, Multigrid, "
               "GeometricMultigrid, or TensorLineMultigrid");
      }
      coupler.set_option<std::string>("dycore_anelastic_preconditioner",preconditioner);
      if (preconditioner == "TensorLineMultigrid") {
        if (!config.tensor_line_multigrid) {
          endrun("ERROR: anelastic tensor-line multigrid preconditioner has no persistent solver object");
        }
        typename GeometricMultigrid<float>::Options options;
        options.vcycles = config.tensor_line_multigrid_vcycles;
        options.pre_smooth = config.tensor_line_multigrid_pre_smooth;
        options.post_smooth = config.tensor_line_multigrid_post_smooth;
        options.coarse_smooth = config.tensor_line_multigrid_coarse_smooth;
        options.max_levels = config.tensor_line_multigrid_max_levels;
        options.coarse_cells = 1;
        options.min_cells_per_rank = config.tensor_line_multigrid_min_cells_per_rank;
        options.jacobi_weight = static_cast<float>(config.tensor_line_multigrid_jacobi_weight);
        options.vertical_line_smoother = true;
        options.horizontal_only = true;
        options.require_single_coarse_rank = true;
        options.coarse_nx = config.tensor_line_multigrid_coarse_nx;
        options.coarse_ny = config.tensor_line_multigrid_coarse_ny;
        options.metadata_prefix = "dycore_anelastic_tensor_line_multigrid";
        config.tensor_line_multigrid->initialize(coupler,options);
      } else if (preconditioner == "GeometricMultigrid") {
        if (!config.geometric_multigrid) {
          endrun("ERROR: anelastic geometric multigrid preconditioner has no persistent solver object");
        }
        typename GeometricMultigrid<float>::Options options;
        options.vcycles = config.geometric_multigrid_vcycles;
        options.pre_smooth = config.geometric_multigrid_pre_smooth;
        options.post_smooth = config.geometric_multigrid_post_smooth;
        options.coarse_smooth = config.geometric_multigrid_coarse_smooth;
        options.max_levels = config.geometric_multigrid_max_levels;
        options.coarse_cells = config.geometric_multigrid_coarse_cells;
        options.min_cells_per_rank = config.geometric_multigrid_min_cells_per_rank;
        options.jacobi_weight = static_cast<float>(config.geometric_multigrid_jacobi_weight);
        config.geometric_multigrid->initialize(coupler,options);
      } else if (preconditioner == "Multigrid") {
        if (!config.multigrid) endrun("ERROR: anelastic multigrid preconditioner has no persistent solver object");
        typename ConnectivityGalerkinMultigrid<float>::Options options;
        options.vcycles = config.multigrid_vcycles;
        options.pre_smooth = config.multigrid_pre_smooth;
        options.post_smooth = config.multigrid_post_smooth;
        options.aggregate_size = config.multigrid_aggregate_size;
        options.max_levels = config.multigrid_max_levels;
        options.coarse_max_dofs = config.multigrid_coarse_max_dofs;
        options.coarse_smooth = config.multigrid_coarse_smooth;
        options.jacobi_weight = static_cast<float>(config.multigrid_jacobi_weight);
        config.multigrid->initialize(coupler,options);
      } else if (preconditioner == "Schwarz") {
        int const tile_nx = config.schwarz_tile_nx;
        int const tile_ny = config.schwarz_tile_ny;
        int const overlap = config.schwarz_overlap;
        int const degree = config.schwarz_chebyshev_degree;
        if (tile_nx <= 0 || tile_ny <= 0 || overlap < 0 || degree <= 0) {
          endrun("ERROR: invalid anelastic Schwarz tile, overlap, or Chebyshev degree");
        }
        int const schwarz_hs = std::max(tile_nx+overlap,tile_ny+overlap);
        if (schwarz_hs > nx || schwarz_hs > ny) {
          endrun("ERROR: anelastic Schwarz residual halo exceeds a local horizontal domain extent");
        }
        if (tile_nx+2*overlap >= nx_glob_int || tile_ny+2*overlap >= ny_glob_int) {
          endrun("ERROR: anelastic Schwarz tiles must be smaller than the global periodic domain");
        }

        auto intersects_owned = [] (int begin, int extent, int owned_begin, int owned_end, int global_extent,
                                    bool periodic) {
          for (int offset = 0; offset < extent; offset++) {
            int index = begin+offset;
            if (periodic) index = (index%global_extent+global_extent)%global_extent;
            if (index >= owned_begin && index <= owned_end) return true;
          }
          return false;
        };
        int const num_tiles_x = (nx_glob_int+tile_nx-1)/tile_nx;
        int const num_tiles_y = (ny_glob_int+tile_ny-1)/tile_ny;
        int num_local_tiles = 0;
        for (int tj = 0; tj < num_tiles_y; tj++) {
          int const interior_j0 = tj*tile_ny;
          int j0 = interior_j0-overlap;
          int j1 = std::min(interior_j0+tile_ny,ny_glob_int)+overlap;
          if (!periodic_y) {
            j0 = std::max(j0,0);
            j1 = std::min(j1,ny_glob_int);
          }
          bool const intersects_y = intersects_owned(j0,j1-j0,j_beg_int,j_beg_int+ny-1,ny_glob_int,periodic_y);
          if (!intersects_y) continue;
          for (int ti = 0; ti < num_tiles_x; ti++) {
            int const interior_i0 = ti*tile_nx;
            int i0 = interior_i0-overlap;
            int i1 = std::min(interior_i0+tile_nx,nx_glob_int)+overlap;
            if (!periodic_x) {
              i0 = std::max(i0,0);
              i1 = std::min(i1,nx_glob_int);
            }
            bool const intersects_x = intersects_owned(i0,i1-i0,i_beg_int,i_beg_int+nx-1,nx_glob_int,periodic_x);
            if (!intersects_x) continue;
            num_local_tiles++;
          }
        }
        if (num_local_tiles == 0) endrun("ERROR: anelastic Schwarz setup found no local tiles");
        dm.register_and_allocate<int>("dycore_anelastic_schwarz_tiles",{num_local_tiles,4});
        intHost2d tiles_host("dycore_anelastic_schwarz_tiles_host",num_local_tiles,4);
        int tile = 0;
        for (int tj = 0; tj < num_tiles_y; tj++) {
          int const interior_j0 = tj*tile_ny;
          int j0 = interior_j0-overlap;
          int j1 = std::min(interior_j0+tile_ny,ny_glob_int)+overlap;
          if (!periodic_y) {
            j0 = std::max(j0,0);
            j1 = std::min(j1,ny_glob_int);
          }
          bool const intersects_y = intersects_owned(j0,j1-j0,j_beg_int,j_beg_int+ny-1,ny_glob_int,periodic_y);
          if (!intersects_y) continue;
          for (int ti = 0; ti < num_tiles_x; ti++) {
            int const interior_i0 = ti*tile_nx;
            int i0 = interior_i0-overlap;
            int i1 = std::min(interior_i0+tile_nx,nx_glob_int)+overlap;
            if (!periodic_x) {
              i0 = std::max(i0,0);
              i1 = std::min(i1,nx_glob_int);
            }
            bool const intersects_x = intersects_owned(i0,i1-i0,i_beg_int,i_beg_int+nx-1,nx_glob_int,periodic_x);
            if (!intersects_x) continue;
            tiles_host(tile,0) = i0;
            tiles_host(tile,1) = j0;
            tiles_host(tile,2) = i1-i0;
            tiles_host(tile,3) = j1-j0;
            tile++;
          }
        }
        tiles_host.deep_copy_to(dm.get<int,2>("dycore_anelastic_schwarz_tiles"));
        dm.register_and_allocate<int>("dycore_anelastic_schwarz_mask_halos",
                                      {nz+2*schwarz_hs,ny+2*schwarz_hs,nx+2*schwarz_hs});
        auto mask_halos = dm.get<int,3>("dycore_anelastic_schwarz_mask_halos");
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
          mask_halos(schwarz_hs+k,schwarz_hs+j,schwarz_hs+i) = fluid_mask(k,j,i);
        });
        core::MultiField<int,3> mask_fields;
        mask_fields.add_field(mask_halos);
        coupler.halo_exchange(mask_fields,schwarz_hs);
        int const max_tile_x = tile_nx+2*overlap;
        int const max_tile_y = tile_ny+2*overlap;
        dm.register_and_allocate<float>("dycore_anelastic_schwarz_residual_halos",
                                        {1,nz+2*schwarz_hs,ny+2*schwarz_hs,nx+2*schwarz_hs});
        dm.register_and_allocate<float>("dycore_anelastic_schwarz_local_rhs",
                                        {num_local_tiles,nz,max_tile_y,max_tile_x});
        dm.register_and_allocate<float>("dycore_anelastic_schwarz_local_x_0",
                                        {num_local_tiles,nz,max_tile_y,max_tile_x});
        dm.register_and_allocate<float>("dycore_anelastic_schwarz_local_x_1",
                                        {num_local_tiles,nz,max_tile_y,max_tile_x});
        // Cached rectangular stencil: diagonal, inverse diagonal, and xm/xp/ym/yp/zm/zp neighbor coefficients.
        dm.register_and_allocate<float>("dycore_anelastic_schwarz_coefficients",
                                        {8,num_local_tiles,nz,max_tile_y,max_tile_x});
        auto coefficients = dm.get<float,5>("dycore_anelastic_schwarz_coefficients");
        coefficients = 0;
        auto tiles = dm.get<int const,2>("dycore_anelastic_schwarz_tiles");
        float const cx = r_dx*r_dx;
        float const cy = r_dy*r_dy;
        yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<4>(num_local_tiles,nz,max_tile_y,max_tile_x),
                           KOKKOS_LAMBDA (int tile, int k, int jj, int ii) {
          int const tx = tiles(tile,2);
          int const ty = tiles(tile,3);
          if (ii >= tx || jj >= ty) return;
          int const gi_unwrapped = tiles(tile,0)+ii;
          int const gj_unwrapped = tiles(tile,1)+jj;
          int gi = periodic_x ? (gi_unwrapped%nx_glob_int+nx_glob_int)%nx_glob_int : gi_unwrapped;
          int gj = periodic_y ? (gj_unwrapped%ny_glob_int+ny_glob_int)%ny_glob_int : gj_unwrapped;
          int di = gi-i_beg_int;
          int dj = gj-j_beg_int;
          if (periodic_x && di < -schwarz_hs) di += nx_glob_int;
          if (periodic_x && di >= nx+schwarz_hs) di -= nx_glob_int;
          if (periodic_y && dj < -schwarz_hs) dj += ny_glob_int;
          if (periodic_y && dj >= ny+schwarz_hs) dj -= ny_glob_int;
          int const ih = schwarz_hs+di;
          int const jh = schwarz_hs+dj;
          if (mask_halos(schwarz_hs+k,jh,ih) == 0) return;

          float diagonal = 0;
          auto cache_horizontal = [&] (int field, int ni, int nj, int ngi, int ngj, float coefficient,
                                       bool physical_outside) {
            if (physical_outside) return;
            if (ni < 0 || ni >= tx || nj < 0 || nj >= ty) {
              diagonal += coefficient;
              return;
            }
            int ngi_wrapped = periodic_x ? (ngi%nx_glob_int+nx_glob_int)%nx_glob_int : ngi;
            int ngj_wrapped = periodic_y ? (ngj%ny_glob_int+ny_glob_int)%ny_glob_int : ngj;
            int ndi = ngi_wrapped-i_beg_int;
            int ndj = ngj_wrapped-j_beg_int;
            if (periodic_x && ndi < -schwarz_hs) ndi += nx_glob_int;
            if (periodic_x && ndi >= nx+schwarz_hs) ndi -= nx_glob_int;
            if (periodic_y && ndj < -schwarz_hs) ndj += ny_glob_int;
            if (periodic_y && ndj >= ny+schwarz_hs) ndj -= ny_glob_int;
            if (mask_halos(schwarz_hs+k,schwarz_hs+ndj,schwarz_hs+ndi) == 0) return;
            diagonal += coefficient;
            coefficients(field,tile,k,jj,ii) = coefficient;
          };
          cache_horizontal(2,ii-1,jj,gi_unwrapped-1,gj_unwrapped,cx,!periodic_x && gi_unwrapped-1 < 0);
          cache_horizontal(3,ii+1,jj,gi_unwrapped+1,gj_unwrapped,cx,
                           !periodic_x && gi_unwrapped+1 >= nx_glob_int);
          cache_horizontal(4,ii,jj-1,gi_unwrapped,gj_unwrapped-1,cy,!periodic_y && gj_unwrapped-1 < 0);
          cache_horizontal(5,ii,jj+1,gi_unwrapped,gj_unwrapped+1,cy,
                           !periodic_y && gj_unwrapped+1 >= ny_glob_int);
          int const km = k > 0 ? k-1 : nz-1;
          int const kp = k+1 < nz ? k+1 : 0;
          if ((k > 0 || periodic_z) && mask_halos(schwarz_hs+km,jh,ih) == 1) {
            float const coefficient = 1.f/(static_cast<float>(dz(k))*static_cast<float>(metjac_edges(k)));
            diagonal += coefficient;
            coefficients(6,tile,k,jj,ii) = coefficient;
          }
          if ((k+1 < nz || periodic_z) && mask_halos(schwarz_hs+kp,jh,ih) == 1) {
            float const coefficient = 1.f/(static_cast<float>(dz(k))*static_cast<float>(metjac_edges(k+1)));
            diagonal += coefficient;
            coefficients(7,tile,k,jj,ii) = coefficient;
          }
          coefficients(0,tile,k,jj,ii) = diagonal;
          coefficients(1,tile,k,jj,ii) = diagonal > 0 ? 1.f/diagonal : 0;
        });
        coupler.set_option<int>("dycore_anelastic_schwarz_num_local_tiles",num_local_tiles);
        coupler.set_option<int>("dycore_anelastic_schwarz_halo",schwarz_hs);
      }

    }
  };

  } // namespace detail

  template <class FP, int ORD>
  KOKKOS_INLINE_FUNCTION void modify_stencil_immersed_der0(SArray<FP,ORD> &stencil,
                                                            SArray<bool,ORD> const &immersed) {
    detail::AcousticProjectionImpl<ORD>::template modify_stencil_immersed_der0<FP,ORD>(stencil,immersed);
  }

  template <int ord>
  void acoustic_projection(core::Coupler &coupler, float4d const &momentum_in,
                           float4d const &momentum_out, real3d const &pressure, real dt,
                           AcousticProjectionConfig const &config) {
    auto const nz = coupler.get_nz();
    auto const ny = coupler.get_ny();
    auto const nx = coupler.get_nx();
    bool const momentum_shape_valid =
        momentum_in.extent(0) == 3 && momentum_out.extent(0) == 3 &&
        momentum_in.extent(1) == nz && momentum_out.extent(1) == nz &&
        momentum_in.extent(2) == ny && momentum_out.extent(2) == ny &&
        momentum_in.extent(3) == nx && momentum_out.extent(3) == nx;
    bool const pressure_shape_valid =
        pressure.extent(0) == nz && pressure.extent(1) == ny && pressure.extent(2) == nx;
    if (!momentum_shape_valid || !pressure_shape_valid) {
      endrun("ERROR: acoustic projection input/output dimensions do not match the Coupler grid");
    }
    detail::AcousticProjectionImpl<ord>::apply(coupler,momentum_in,momentum_out,pressure,dt,config);
  }

  template <int ord>
  void initialize_acoustic_projection(core::Coupler &coupler, AcousticProjectionConfig const &config) {
    detail::AcousticProjectionImpl<ord>::init(coupler,config);
  }

}

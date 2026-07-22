
#pragma once

#include "main_header.h"
#include "coupler.h"

namespace modules {

  // This class implements a 1-equation SGS TKE-based SGS closure model
  struct LES_Closure {
    int static constexpr hs        = 1;  // Halo size used by the LES closure
    int static constexpr num_state = 5;  // Number of state variables (rho, rho*u, rho*v, rho*w, rho*theta)
    int static constexpr idR = 0;  // ID for density in state array
    int static constexpr idU = 1;  // ID for rho*u in state array
    int static constexpr idV = 2;  // ID for rho*v in state array
    int static constexpr idW = 3;  // ID for rho*w in state array
    int static constexpr idT = 4;  // ID for rho*theta in state array


    // Tensor-product quadratic interpolation on parent cell centers.
    KOKKOS_INLINE_FUNCTION static real nested_triquadratic( real4d const &parent_fields ,
                                                            int l , real z , real y , real x ) {
      int const nx = parent_fields.extent(3);
      int const ny = parent_fields.extent(2);
      int const nz = parent_fields.extent(1);
      int const ic = std::max(1,std::min(nx-2,static_cast<int>(std::floor(x+0.5))));
      int const jc = std::max(1,std::min(ny-2,static_cast<int>(std::floor(y+0.5))));
      int const kc = std::max(1,std::min(nz-2,static_cast<int>(std::floor(z+0.5))));
      real const tx = x-ic;
      real const ty = y-jc;
      real const tz = z-kc;
      real wx[3] = {0.5*tx*(tx-1),1-tx*tx,0.5*tx*(tx+1)};
      real wy[3] = {0.5*ty*(ty-1),1-ty*ty,0.5*ty*(ty+1)};
      real wz[3] = {0.5*tz*(tz-1),1-tz*tz,0.5*tz*(tz+1)};
      real value = 0;
      for (int kk=0; kk < 3; kk++) {
        for (int jj=0; jj < 3; jj++) {
          for (int ii=0; ii < 3; ii++) {
            value += wz[kk]*wy[jj]*wx[ii]*parent_fields(l,kc+kk-1,jc+jj-1,ic+ii-1);
          }
        }
      }
      return value;
    }



    // Project the parent LES working fields to the child boundaries at the same point that precursor fields are
    // captured. Parent and child use the same time step, so only one spatially interpolated halo set is needed.
    void interpolate_nested_bcs( core::Coupler                  & parent ,
                                 core::Coupler                  & child  ,
                                 core::MultiField<real,3> const & fields ) const {
      using yakl::SimpleBounds;
      int const nx_parent   = parent.get_nx();
      int const ny_parent   = parent.get_ny();
      int const nz_parent   = parent.get_nz();
      int const nx_glob     = parent.get_nx_glob();
      int const ny_glob     = parent.get_ny_glob();
      int const i_beg       = parent.get_i_beg();
      int const j_beg       = parent.get_j_beg();
      int const nx          = child.get_nx();
      int const ny          = child.get_ny();
      int const nz          = child.get_nz();
      int const child_ibeg  = child.get_i_beg();
      int const child_jbeg  = child.get_j_beg();
      int const ri          = child.get_option<int>("coupler_refine_x");
      int const rj          = child.get_option<int>("coupler_refine_y");
      int const rk          = child.get_option<int>("coupler_refine_z");
      int const pi          = child.get_option<size_t>("coupler_parent_i_beg");
      int const pj          = child.get_option<size_t>("coupler_parent_j_beg");
      int const pk          = child.get_option<size_t>("coupler_parent_k_beg");
      auto &parent_dm       = parent.get_data_manager_readonly();
      auto parent_hy_theta  = parent_dm.get<real const,1>("les_hy_theta_cells");
      int const num_tracers = fields.size()-num_state-1;
      int const num_fields  = num_state+num_tracers+1;
      real4d parent_fields("les_nested_parent_fields",num_fields,nz_parent,ny_glob,nx_glob);
      parent_fields = 0;
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz_parent,ny_parent,nx_parent) ,
                                              KOKKOS_LAMBDA (int l, int k, int j, int i) {
        real value = fields(l,hs+k,hs+j,hs+i);
        // The LES working state stores perturbation theta. Restore the parent's full theta before interpolation;
        // the child's hydrostatic profile is removed at the destination below.
        if (l == idT) { value += parent_hy_theta(hs+k); }
        parent_fields(l,k,j_beg+j,i_beg+i) = value;
      });
      parent.get_parallel_comm().all_reduce(parent_fields,MPI_SUM).deep_copy_to(parent_fields);
      auto &child_dm = child.get_data_manager_readwrite();
      auto hy_theta  = child_dm.get<real const,1>("les_hy_theta_cells");
      auto value_at = KOKKOS_LAMBDA (int l, int k_child, real z, real y, real x) -> real {
        real value = nested_triquadratic(parent_fields,l,z,y,x);
        if (l == idT) { value -= hy_theta(k_child); }
        // TKE must remain nonnegative. The quadratic interpolant is not monotone and can undershoot
        // zero near a boundary even when every parent value is nonnegative. Negative halo TKE is
        // subsequently passed to sqrt() and pow(K,1.5) in apply(), producing NaNs.
        if (l == num_fields-1) { value = std::max(0._fp,value); }
        return value;
      };

      if (child.get_px() == 0 && child.get_option<std::string>("bc_x1") == "nested") {
        auto bc = child_dm.get<real,4>("les_nested_x_1");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,ny+2*hs,hs) ,
                                                KOKKOS_LAMBDA (int l, int k, int j, int ii) {
          real x = pi + (child_ibeg-1-ii+0.5_fp)/ri - 0.5_fp;
          real y = pj + (child_jbeg+j-hs+0.5_fp)/rj - 0.5_fp;
          real z = pk + (k-hs+0.5_fp)/rk - 0.5_fp;
          bc(l,k,j,ii) = value_at(l,k,z,y,x);
        });
      }
      if (child.get_px() == child.get_nproc_x()-1 && child.get_option<std::string>("bc_x2") == "nested") {
        auto bc = child_dm.get<real,4>("les_nested_x_2");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,ny+2*hs,hs) ,
                                                KOKKOS_LAMBDA (int l, int k, int j, int ii) {
          real x = pi + (child_ibeg+nx+ii+0.5_fp)/ri - 0.5_fp;
          real y = pj + (child_jbeg+j-hs+0.5_fp)/rj - 0.5_fp;
          real z = pk + (k-hs+0.5_fp)/rk - 0.5_fp;
          bc(l,k,j,ii) = value_at(l,k,z,y,x);
        });
      }
      if (child.get_py() == 0 && child.get_option<std::string>("bc_y1") == "nested") {
        auto bc = child_dm.get<real,4>("les_nested_y_1");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,hs,nx+2*hs) ,
                                                KOKKOS_LAMBDA (int l, int k, int jj, int i) {
          real x = pi + (child_ibeg+i-hs+0.5_fp)/ri - 0.5_fp;
          real y = pj + (child_jbeg-1-jj+0.5_fp)/rj - 0.5_fp;
          real z = pk + (k-hs+0.5_fp)/rk - 0.5_fp;
          bc(l,k,jj,i) = value_at(l,k,z,y,x);
        });
      }
      if (child.get_py() == child.get_nproc_y()-1 && child.get_option<std::string>("bc_y2") == "nested") {
        auto bc = child_dm.get<real,4>("les_nested_y_2");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,hs,nx+2*hs) ,
                                                KOKKOS_LAMBDA (int l, int k, int jj, int i) {
          real x = pi + (child_ibeg+i-hs+0.5_fp)/ri - 0.5_fp;
          real y = pj + (child_jbeg+ny+jj+0.5_fp)/rj - 0.5_fp;
          real z = pk + (k-hs+0.5_fp)/rk - 0.5_fp;
          bc(l,k,jj,i) = value_at(l,k,z,y,x);
        });
      }
      if (child.get_option<std::string>("bc_z2") == "nested") {
        auto bc = child_dm.get<real,4>("les_nested_z_2");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,hs,ny+2*hs,nx+2*hs) ,
                                                KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          real x = pi + (child_ibeg+i-hs+0.5_fp)/ri - 0.5_fp;
          real y = pj + (child_jbeg+j-hs+0.5_fp)/rj - 0.5_fp;
          real z = pk + (nz+kk+0.5_fp)/rk - 0.5_fp;
          bc(l,kk,j,i) = value_at(l,hs+nz+kk,z,y,x);
        });
      }
    }


    // Compute total mass of r and t fields for verification purposes
    // If mult_r is true, then t field is mass-weighted (i.e., rho*theta)
    // coupler : Coupler object
    // state   : 4D state array (num_state,nz+2*hs,ny+2*hs,nx+2*hs)
    // mult_r  : whether to multiply t field by r field
    // return   : tuple of (total mass of r field, total mass of t field)
    std::tuple<real,real> compute_mass( core::Coupler & coupler , real4d const & state , bool mult_r ) const {
      using yakl::SimpleBounds;
      auto nx = coupler.get_nx(); // Local number of cells in the x-direction
      auto ny = coupler.get_ny(); // Local number of cells in the y-direction
      auto nz = coupler.get_nz(); // Number of cells in the z-direction
      real3d r("r",nz,ny,nx);     // Temporary array to hold density field
      real3d t("t",nz,ny,nx);     // Temporary array to hold theta field
      // Accumulate local mass into r and t arrays
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j , int i) {
        r(k,j,i) = state(idR,hs+k,hs+j,hs+i);
        t(k,j,i) = state(idT,hs+k,hs+j,hs+i);
        if (mult_r) t(k,j,i) *= r(k,j,i);
      });
      // Aggregate total mass across all processes
      real rmass = coupler.get_parallel_comm().all_reduce( yakl::intrinsics::sum(r) , MPI_SUM );
      real tmass = coupler.get_parallel_comm().all_reduce( yakl::intrinsics::sum(t) , MPI_SUM );
      return std::make_tuple(rmass,tmass); // Return the total masses
    }


    // Initialize LES closure module within the coupler
    // coupler : Coupler object
    // Sets up necessary variables and initial conditions for LES closure
    // Registers TKE tracer and initializes LES hydrostatic profiles
    // Assumes coupler has been initialized with grid and state variables
    // Be sure to call this init before the dynamics module's init so that it knows about TKE tracer
    // This also compute hydrostatic profiles based on initial coupler state so that operations are performed
    //   on perturbation potential temperature rather than full potential temperature
    void init( core::Coupler &coupler ) const {
      using yakl::SimpleBounds;
      auto nx      = coupler.get_nx  ();     // Local number of cells in the x-direction
      auto ny      = coupler.get_ny  ();     // Local number of cells in the y-direction
      auto nz      = coupler.get_nz  ();     // Number of cells in the z-direction
      auto dz      = coupler.get_dz  ();     // Grid spacing array in the z-direction (1-D array of size nz)
      auto nx_glob = coupler.get_nx_glob();  // Global number of cells in the x-direction
      auto ny_glob = coupler.get_ny_glob();  // Global number of cells in the y-direction
      auto gamma   = coupler.get_option<real>("gamma_d");  // Ratio of specific heats
      auto C0      = coupler.get_option<real>("C0"     );  // p = C0 * (rho * theta)^gamma
      auto grav    = coupler.get_option<real>("grav"   );  // Gravitational acceleration
      auto &dm     = coupler.get_data_manager_readwrite(); // DataManager for reading/writing variables
      // Register the TKE tracer with the coupler (positive, doesn't add mass, isn't diffused)
      // Diffusion on TKE is applied in this class separately from other tracers
      coupler.add_tracer( "TKE" , "mass-weighted TKE" , true , false , false );
      dm.get<real,3>("TKE") = 1.e-6;  // Initialize TKE to a small value everywhere to seed production
      // Allocate state, tracers, and TKE arrays, and convert coupler data to LES closure format
      // After conversion, the 
      real4d state , tracers;
      real3d tke;
      // After conversion to state, tracers, and TKE, all variables except density have density divided out
      //   e.g., velocity, potential temperature, and dry mixing ratios
      convert_coupler_to_dynamics( coupler , state , tracers , tke );
      auto num_tracers = tracers.extent(0);
      auto px          = coupler.get_px();
      auto py          = coupler.get_py();
      auto nproc_x     = coupler.get_nproc_x();
      auto nproc_y     = coupler.get_nproc_y();
      if (coupler.get_option<bool>("coupler_is_precursor",false) ||
          coupler.get_option<std::string>("bc_x1") == "precursor" ||
          coupler.get_option<std::string>("bc_x2") == "precursor" ||
          coupler.get_option<std::string>("bc_y1") == "precursor" ||
          coupler.get_option<std::string>("bc_y2") == "precursor" ||
          coupler.get_option<std::string>("bc_z2") == "precursor") {
        int num_fields = num_state + num_tracers + 1;
        if (px == 0) {
          dm.register_and_allocate<real>("les_ghost_x1",{num_fields,nz+2*hs,ny+2*hs,hs});
        }
        if (px == nproc_x-1) {
          dm.register_and_allocate<real>("les_ghost_x2",{num_fields,nz+2*hs,ny+2*hs,hs});
        }
        if (py == 0) {
          dm.register_and_allocate<real>("les_ghost_y1",{num_fields,nz+2*hs,hs,nx+2*hs});
        }
        if (py == nproc_y-1) {
          dm.register_and_allocate<real>("les_ghost_y2",{num_fields,nz+2*hs,hs,nx+2*hs});
        }
        dm.register_and_allocate<real>("les_ghost_z2",{num_fields,hs,ny+2*hs,nx+2*hs});
      }
      int const num_fields = num_state+num_tracers+1;
      if (coupler.get_option<std::string>("bc_x1") == "nested" && px == 0) {
        dm.register_and_allocate<real>("les_nested_x_1",{num_fields,nz+2*hs,ny+2*hs,hs});
      }
      if (coupler.get_option<std::string>("bc_x2") == "nested" && px == nproc_x-1) {
        dm.register_and_allocate<real>("les_nested_x_2",{num_fields,nz+2*hs,ny+2*hs,hs});
      }
      if (coupler.get_option<std::string>("bc_y1") == "nested" && py == 0) {
        dm.register_and_allocate<real>("les_nested_y_1",{num_fields,nz+2*hs,hs,nx+2*hs});
      }
      if (coupler.get_option<std::string>("bc_y2") == "nested" && py == nproc_y-1) {
        dm.register_and_allocate<real>("les_nested_y_2",{num_fields,nz+2*hs,hs,nx+2*hs});
      }
      if (coupler.get_option<std::string>("bc_z2") == "nested") {
        dm.register_and_allocate<real>("les_nested_z_2",{num_fields,hs,ny+2*hs,nx+2*hs});
      }
      // Initialize LES hydrostatic profiles for density and potential temperature using column averages
      dm.register_and_allocate<real>("les_hy_dens_cells" ,{nz+2*hs});
      dm.register_and_allocate<real>("les_hy_theta_cells",{nz+2*hs});
      auto r = dm.get<real,1>("les_hy_dens_cells" );    r = 0;
      auto t = dm.get<real,1>("les_hy_theta_cells");    t = 0;
      // Accumulate local contributions to column sums
      yakl::parallel_for( YAKL_AUTO_LABEL() , nz+2*hs , KOKKOS_LAMBDA (int k) {
        for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
            r(k) += state(idR,k,hs+j,hs+i);
            t(k) += state(idT,k,hs+j,hs+i);
          }
        }
      });
      // Aggregate global sums and compute averages over all MPI processes
      coupler.get_parallel_comm().all_reduce( r , MPI_SUM ).deep_copy_to(r);
      coupler.get_parallel_comm().all_reduce( t , MPI_SUM ).deep_copy_to(t);
      real r_nx_ny = 1./(nx_glob*ny_glob);  // Pre-compute reciprocal of global horizontal cell count
      // Compute the column averages
      yakl::parallel_for( YAKL_AUTO_LABEL() , nz+2*hs , KOKKOS_LAMBDA (int k) {
        r(k) *= r_nx_ny;
        t(k) *= r_nx_ny;
      });
      // Fill in ghost cells with hydrostatic profile extrapolation
      yakl::parallel_for( YAKL_AUTO_LABEL() , hs , KOKKOS_LAMBDA (int kk) {
        {
          int  k0       = hs;
          int  k        = k0-1-kk;
          real rho0     = r(k0);
          real theta0   = t(k0);
          real rho0_gm1 = std::pow(rho0  ,gamma-1);
          real theta0_g = std::pow(theta0,gamma  );
          r(k) = std::pow( rho0_gm1 + grav*(gamma-1)*dz(0)*(kk+1)/(gamma*C0*theta0_g) , 1./(gamma-1) );
          t(k) = theta0;
        }
        {
          int  k0       = hs+nz-1;
          int  k        = k0+1+kk;
          real rho0     = r(k0);
          real theta0   = t(k0);
          real rho0_gm1 = std::pow(rho0  ,gamma-1);
          real theta0_g = std::pow(theta0,gamma  );
          r(k) = std::pow( rho0_gm1 - grav*(gamma-1)*dz(nz-1)*(kk+1)/(gamma*C0*theta0_g) , 1./(gamma-1) );
          t(k) = theta0;
        }
      });
    }



    // Apply the 1-equation TKE-based LES closure to the state and tracers over one time step
    // coupler : Coupler object containing the data and options
    // dtphys  : Physical time step to advance the LES closure
    // Applies the LES closure to update the state and tracers in the coupler over the time step dtphys
    // This includes computing fluxes, updating TKE, and applying necessary boundary conditions
    void apply( core::Coupler &coupler , real dtphys , core::Coupler *nested_child = nullptr ) const {
      using yakl::SimpleBounds;
      auto nx             = coupler.get_nx();  // Local number of cells in the x-direction
      auto ny             = coupler.get_ny();  // Local number of cells in the y-direction
      auto nz             = coupler.get_nz();  // Number of cells in the z-direction
      auto dx             = coupler.get_dx();  // Grid spacing in the x-direction
      auto dy             = coupler.get_dy();  // Grid spacing in the y-direction
      auto dz             = coupler.get_dz();  // Grid spacing array in the z-direction (1-D array of size nz)
      auto enable_gravity = coupler.get_option<bool>("enable_gravity" , true ); // Whether gravity is enabled
      auto grav           = coupler.get_option<real>("grav");                   // Gravitational acceleration
      auto nu             = coupler.get_option<real>("kinematic_viscosity",0);  // Kinematic viscosity (m^2/s)
      auto dns            = coupler.get_option<bool>("dns",false);              // Whether to run in DNS mode (no LES closure)
      auto delta_mult     = coupler.get_option<real>("les_closure_delta_multiplier",0.3); // LES filter-width multiplier
      auto &dm            = coupler.get_data_manager_readwrite();               // DataManager for reading/writing variables
      auto immersed       = dm.get<real const,3>("immersed_proportion_halos");  // Immersed boundary proportion array with halos
      real constexpr Pr = 0.7;  // Prandtl number for SGS diffusivity
      real constexpr imm_th = 0.6;

      // Allocate and convert coupler data to LES closure format. The resulting state, tracers, and TKE
      //   all have density divided out except for the density field itself
      //   e.g., velocity, potential temperature, dry mixing ratios, and non-mass-weighted TKE
      //   TKE is not included in the tracers array since it is handled separately
      real4d state , tracers;
      real3d tke;
      convert_coupler_to_dynamics( coupler , state , tracers , tke );
      // auto mass1 = compute_mass( coupler , state , true );
      auto num_tracers = tracers.extent(0);  // Number of tracer fields for LES (TKE is not included here)
      auto hy_t = dm.get<real const,1>("les_hy_theta_cells");  // Get LES hydrostatic potential temperature profile
      // Convert potential temperature to perturbation potential temperature by removing hydrostatic profile
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        state(idT,hs+k,hs+j,hs+i) -= hy_t(hs+k);
      });
      // Aggregate the state, tracers, and TKE arrays into a single MultipleFields object for halo exchange
      core::MultiField<real,3> fields;
      for (int l=0; l < num_state  ; l++) { fields.add_field( state  .slice<3>(l,0,0,0) ); }
      for (int l=0; l < num_tracers; l++) { fields.add_field( tracers.slice<3>(l,0,0,0) ); }
      fields.add_field( tke );
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_start("les_halo_exchange");
      #endif
      coupler.halo_exchange( fields , hs );  // Do a horizontal halo exchange on all fields assuming periodic boundaries
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_stop("les_halo_exchange");
      #endif
      // Apply horizontal and vertical boundary conditions to halos
      halo_bcs( coupler , state , tracers , tke );

      // Match the precursor workflow by capturing parent working halos before applying the LES tendencies. Spatial
      // projection is triquadratic because the child boundary does not generally coincide with parent cell centers.
      if (nested_child != nullptr) {
        interpolate_nested_bcs(coupler,*nested_child,fields);
      }

      // Allocate arrays to hold fluxes of all variables in all three directions
      real3d flux_ru_x     ("flux_ru_x"                 ,nz  ,ny  ,nx+1);
      real3d flux_rv_x     ("flux_rv_x"                 ,nz  ,ny  ,nx+1);
      real3d flux_rw_x     ("flux_rw_x"                 ,nz  ,ny  ,nx+1);
      real3d flux_rt_x     ("flux_rt_x"                 ,nz  ,ny  ,nx+1);
      real3d flux_tke_x    ("flux_tke_x"                ,nz  ,ny  ,nx+1);
      real4d flux_tracers_x("flux_tracers_x",num_tracers,nz  ,ny  ,nx+1);
      real3d flux_ru_y     ("flux_ru_y"                 ,nz  ,ny+1,nx  );
      real3d flux_rv_y     ("flux_rv_y"                 ,nz  ,ny+1,nx  );
      real3d flux_rw_y     ("flux_rw_y"                 ,nz  ,ny+1,nx  );
      real3d flux_rt_y     ("flux_rt_y"                 ,nz  ,ny+1,nx  );
      real3d flux_tke_y    ("flux_tke_y"                ,nz  ,ny+1,nx  );
      real4d flux_tracers_y("flux_tracers_y",num_tracers,nz  ,ny+1,nx  );
      real3d flux_ru_z     ("flux_ru_z"                 ,nz+1,ny  ,nx  );
      real3d flux_rv_z     ("flux_rv_z"                 ,nz+1,ny  ,nx  );
      real3d flux_rw_z     ("flux_rw_z"                 ,nz+1,ny  ,nx  );
      real3d flux_rt_z     ("flux_rt_z"                 ,nz+1,ny  ,nx  );
      real3d flux_tke_z    ("flux_tke_z"                ,nz+1,ny  ,nx  );
      real4d flux_tracers_z("flux_tracers_z",num_tracers,nz+1,ny  ,nx  );
      // Allocate array to hold the TKE source terms
      real3d tke_source    ("tke_source"                ,nz  ,ny  ,nx  );

      real visc_max_x = 0.1*dx*dx/dtphys; // To cap the viscosity for stability
      real visc_max_y = 0.1*dy*dy/dtphys; // To cap the viscosity for stability

      yakl::ScalarLiveOut<bool> max_triggered(false); // Whether the viscosity capping was triggered anywhere

      // Buoyancy source
      // TKE dissipation
      // Shear production

      // Compute SGS fluxes in all three directions, looping over cell faces
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx+1) , KOKKOS_LAMBDA (int k, int j, int i) {
        int im1 = immersed(hs+k  ,hs+j  ,hs+i-1) > imm_th ? i : i-1;
        int jm1 = immersed(hs+k  ,hs+j-1,hs+i  ) > imm_th ? j : j-1;
        int km1 = immersed(hs+k-1,hs+j  ,hs+i  ) > imm_th ? k : k-1;
        int jp1 = immersed(hs+k  ,hs+j+1,hs+i  ) > imm_th ? j : j+1;
        int kp1 = immersed(hs+k+1,hs+j  ,hs+i  ) > imm_th ? k : k+1;
        // Derivatives valid at interface i-1/2
        real dz2 = dz(k) + dz(std::max(0,k-1))/2 + dz(std::min(nz-1,k+1))/2;
        real du_dz = 0.5 * ( (state(idU,hs+kp1,hs+j,hs+i-1)-state(idU,hs+km1,hs+j,hs+i-1))/(dz2 ) +
                             (state(idU,hs+kp1,hs+j,hs+i  )-state(idU,hs+km1,hs+j,hs+i  ))/(dz2 ) );
        real dw_dz = 0.5 * ( (state(idW,hs+kp1,hs+j,hs+im1)-state(idW,hs+km1,hs+j,hs+im1))/(dz2 ) +
                             (state(idW,hs+kp1,hs+j,hs+i  )-state(idW,hs+km1,hs+j,hs+i  ))/(dz2 ) );
        real dt_dz = 0.5 * ( (state(idT,hs+kp1,hs+j,hs+im1)-state(idT,hs+km1,hs+j,hs+im1))/(dz2 ) +
                             (state(idT,hs+kp1,hs+j,hs+i  )-state(idT,hs+km1,hs+j,hs+i  ))/(dz2 ) );
        real du_dy = 0.5 * ( (state(idU,hs+k,hs+jp1,hs+i-1)-state(idU,hs+k,hs+jp1,hs+i-1))/(2*dy) +
                             (state(idU,hs+k,hs+jp1,hs+i  )-state(idU,hs+k,hs+jp1,hs+i  ))/(2*dy) );
        real dv_dy = 0.5 * ( (state(idV,hs+k,hs+jp1,hs+im1)-state(idV,hs+k,hs+jp1,hs+im1))/(2*dy) +
                             (state(idV,hs+k,hs+jp1,hs+i  )-state(idV,hs+k,hs+jp1,hs+i  ))/(2*dy) );
        real du_dx = (state(idU,hs+k,hs+j,hs+i) - state(idU,hs+k,hs+j,hs+i-1))/dx;
        real dv_dx = (state(idV,hs+k,hs+j,hs+i) - state(idV,hs+k,hs+j,hs+im1))/dx;
        real dw_dx = (state(idW,hs+k,hs+j,hs+i) - state(idW,hs+k,hs+j,hs+im1))/dx;
        real dt_dx = (state(idT,hs+k,hs+j,hs+i) - state(idT,hs+k,hs+j,hs+im1))/dx;
        real dK_dx = (tke      (hs+k,hs+j,hs+i) - tke      (hs+k,hs+j,hs+im1))/dx;
        real dth_dz = (hy_t(hs+k+1)-hy_t(hs+k-1))/(dz2);
        // Quantities at interface i-1/2
        // Compute density, TKE, reference temperature, Brunt-Vaisala frequency, grid spacing, mixing length,
        //   eddy viscosity, turbulent Prandtl number, total viscosity, and total thermal viscosity
        real rho         = 0.5 * ( state(idR,hs+k,hs+j,hs+im1) + state(idR,hs+k,hs+j,hs+i) );
        real K           = dns ? 0 : 0.5 * ( tke      (hs+k,hs+j,hs+im1) + tke      (hs+k,hs+j,hs+i) );
        real tref        = hy_t(hs+k);
        real N           = dt_dz+dth_dz >= 0 ? std::sqrt(grav/tref*(dt_dz+dth_dz)) : 0;
        real delta       = std::pow( dx*dy*dz(k) , 1./3. ) * delta_mult;
        real ell         = std::min( 0.76*std::sqrt(K)/std::max(N,1.e-10) , delta );
        real km          = 0.1 * ell * std::sqrt(K);
        real Pr_t        = 0.85;
        real visc_tot    = dns ? nu : std::min( km+nu         , visc_max_x );
        real visc_tot_th = dns ? nu : std::min( km/Pr_t+nu/Pr , visc_max_x );
        // Limit the viscosity and set the max_triggered flag if the limit is reached
        if (visc_tot == visc_max_x || visc_tot_th == visc_max_x) max_triggered = true;
        // Compute the SGS fluxes in the x-direction
        flux_ru_x (k,j,i) = -rho*visc_tot   *(du_dx + du_dx - 2./3.*(du_dx+dv_dy+dw_dz))-2./3.*rho*K;
        flux_rv_x (k,j,i) = -rho*visc_tot   *(dv_dx + du_dy                            );
        flux_rw_x (k,j,i) = -rho*visc_tot   *(dw_dx + du_dz                            );
        flux_rt_x (k,j,i) = -rho*visc_tot_th*(dt_dx                                    );
        flux_tke_x(k,j,i) = -rho*visc_tot   *(dK_dx                                    );
        for (int tr=0; tr < num_tracers; tr++) {
          dt_dx = (tracers(tr,hs+k,hs+j,hs+i) - tracers(tr,hs+k,hs+j,hs+im1))/dx; // Tracer gradient
          flux_tracers_x(tr,k,j,i) = -rho*visc_tot_th*dt_dx;
        }
      });
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny+1,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        int im1 = immersed(hs+k,hs+j,hs+i-1) > imm_th ? i : i-1;
        int jm1 = immersed(hs+k,hs+j-1,hs+i) > imm_th ? j : j-1;
        int km1 = immersed(hs+k-1,hs+j,hs+i) > imm_th ? k : k-1;
        int ip1 = immersed(hs+k,hs+j,hs+i+1) > imm_th ? i : i+1;
        int kp1 = immersed(hs+k+1,hs+j,hs+i) > imm_th ? k : k+1;
        // Derivatives valid at interface j-1/2
        real dz2 = dz(k) + dz(std::max(0,k-1))/2 + dz(std::min(nz-1,k+1))/2;
        real dv_dz = 0.5 * ( (state(idV,hs+kp1,hs+j-1,hs+i)-state(idV,hs+km1,hs+j-1,hs+i))/(dz2 ) +
                             (state(idV,hs+kp1,hs+j  ,hs+i)-state(idV,hs+km1,hs+j  ,hs+i))/(dz2 ) );
        real dw_dz = 0.5 * ( (state(idW,hs+kp1,hs+jm1,hs+i)-state(idW,hs+km1,hs+jm1,hs+i))/(dz2 ) +
                             (state(idW,hs+kp1,hs+j  ,hs+i)-state(idW,hs+km1,hs+j  ,hs+i))/(dz2 ) );
        real dt_dz = 0.5 * ( (state(idT,hs+kp1,hs+jm1,hs+i)-state(idT,hs+km1,hs+jm1,hs+i))/(dz2 ) +
                             (state(idT,hs+kp1,hs+j  ,hs+i)-state(idT,hs+km1,hs+j  ,hs+i))/(dz2 ) );
        real du_dx = 0.5 * ( (state(idU,hs+k,hs+jm1,hs+ip1)-state(idU,hs+k,hs+jm1,hs+im1))/(2*dx) +
                             (state(idU,hs+k,hs+j  ,hs+ip1)-state(idU,hs+k,hs+j  ,hs+im1))/(2*dx) );
        real dv_dx = 0.5 * ( (state(idV,hs+k,hs+j-1,hs+ip1)-state(idV,hs+k,hs+j-1,hs+im1))/(2*dx) +
                             (state(idV,hs+k,hs+j  ,hs+ip1)-state(idV,hs+k,hs+j  ,hs+im1))/(2*dx) );
        real du_dy = (state(idU,hs+k,hs+j,hs+i) - state(idU,hs+k,hs+jm1,hs+i))/dy;
        real dv_dy = (state(idV,hs+k,hs+j,hs+i) - state(idV,hs+k,hs+j-1,hs+i))/dy;
        real dw_dy = (state(idW,hs+k,hs+j,hs+i) - state(idW,hs+k,hs+jm1,hs+i))/dy;
        real dt_dy = (state(idT,hs+k,hs+j,hs+i) - state(idT,hs+k,hs+jm1,hs+i))/dy;
        real dK_dy = (tke      (hs+k,hs+j,hs+i) - tke      (hs+k,hs+jm1,hs+i))/dy;
        real dth_dz = (hy_t(hs+k+1)-hy_t(hs+k-1))/(dz2);
        // Quantities at interface j-1/2
        // Compute density, TKE, reference temperature, Brunt-Vaisala frequency, grid spacing, mixing length,
        //   eddy viscosity, turbulent Prandtl number, total viscosity, and total thermal viscosity
        real rho         = 0.5 * ( state(idR,hs+k,hs+jm1,hs+i) + state(idR,hs+k,hs+j,hs+i) );
        real K           = dns ? 0 : 0.5 * ( tke      (hs+k,hs+jm1,hs+i) + tke      (hs+k,hs+j,hs+i) );
        real tref        = hy_t(hs+k);
        real N           = dt_dz+dth_dz >= 0 ? std::sqrt(grav/tref*(dt_dz+dth_dz)) : 0;
        real delta       = std::pow( dx*dy*dz(k) , 1./3. ) * delta_mult;
        real ell         = std::min( 0.76*std::sqrt(K)/std::max(N,1.e-10) , delta );
        real km          = 0.1 * ell * std::sqrt(K);
        real Pr_t        = 0.85;
        real visc_tot    = dns ? nu : std::min( km+nu         , visc_max_y );
        real visc_tot_th = dns ? nu : std::min( km/Pr_t+nu/Pr , visc_max_y );
        // Limit the viscosity and set the max_triggered flag if the limit is reached
        if (visc_tot == visc_max_y || visc_tot_th == visc_max_y) max_triggered = true;
        // Compute the SGS fluxes in the y-direction
        flux_ru_y (k,j,i) = -rho*visc_tot   *(du_dy + dv_dx                            );
        flux_rv_y (k,j,i) = -rho*visc_tot   *(dv_dy + dv_dy - 2./3.*(du_dx+dv_dy+dw_dz))-2./3.*rho*K;
        flux_rw_y (k,j,i) = -rho*visc_tot   *(dw_dy + dv_dz                            );
        flux_rt_y (k,j,i) = -rho*visc_tot_th*(dt_dy                                    );
        flux_tke_y(k,j,i) = -rho*visc_tot   *(dK_dy                                    );
        for (int tr=0; tr < num_tracers; tr++) {
          dt_dy = (tracers(tr,hs+k,hs+j,hs+i) - tracers(tr,hs+k,hs+jm1,hs+i))/dy; // Tracer gradient
          flux_tracers_y(tr,k,j,i) = -rho*visc_tot_th*dt_dy;
        }
      });
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        int im1 = immersed(hs+k,hs+j,hs+i-1) > imm_th ? i : i-1;
        int jm1 = immersed(hs+k,hs+j-1,hs+i) > imm_th ? j : j-1;
        int km1 = immersed(hs+k-1,hs+j,hs+i) > imm_th ? k : k-1;
        int ip1 = immersed(hs+k,hs+j,hs+i+1) > imm_th ? i : i+1;
        int jp1 = immersed(hs+k,hs+j+1,hs+i) > imm_th ? j : j+1;
        // Derivatives valid at interface k-1/2
        real dzloc = dz(std::max(0,k-1))/2 + dz(std::min(nz-1,k))/2;
        real du_dx = 0.5 * ( (state(idU,hs+km1,hs+j,hs+ip1) - state(idU,hs+km1,hs+j,hs+im1))/(2*dx) +
                             (state(idU,hs+k  ,hs+j,hs+ip1) - state(idU,hs+k  ,hs+j,hs+im1))/(2*dx) );
        real dw_dx = 0.5 * ( (state(idW,hs+k-1,hs+j,hs+ip1) - state(idW,hs+k-1,hs+j,hs+im1))/(2*dx) +
                             (state(idW,hs+k  ,hs+j,hs+ip1) - state(idW,hs+k  ,hs+j,hs+im1))/(2*dx) );
        real dv_dy = 0.5 * ( (state(idV,hs+km1,hs+jp1,hs+i) - state(idV,hs+km1,hs+jm1,hs+i))/(2*dy) +
                             (state(idV,hs+k  ,hs+jp1,hs+i) - state(idV,hs+k  ,hs+jm1,hs+i))/(2*dy) );
        real dw_dy = 0.5 * ( (state(idW,hs+k-1,hs+jp1,hs+i) - state(idW,hs+k-1,hs+jm1,hs+i))/(2*dy) +
                             (state(idW,hs+k  ,hs+jp1,hs+i) - state(idW,hs+k  ,hs+jm1,hs+i))/(2*dy) );
        real du_dz = (state(idU,hs+k,hs+j,hs+i) - state(idU,hs+km1,hs+j,hs+i))/dzloc;
        real dv_dz = (state(idV,hs+k,hs+j,hs+i) - state(idV,hs+km1,hs+j,hs+i))/dzloc;
        real dw_dz = (state(idW,hs+k,hs+j,hs+i) - state(idW,hs+k-1,hs+j,hs+i))/dzloc;
        real dt_dz = (state(idT,hs+k,hs+j,hs+i) - state(idT,hs+km1,hs+j,hs+i))/dzloc;
        real dK_dz = (tke      (hs+k,hs+j,hs+i) - tke      (hs+km1,hs+j,hs+i))/dzloc;
        real dth_dz = (hy_t(hs+k)-hy_t(hs+k-1))/dzloc;
        // Quantities at interface k-1/2
        // Compute density, TKE, reference temperature, Brunt-Vaisala frequency, grid spacing, mixing length,
        //   eddy viscosity, turbulent Prandtl number, total viscosity, and total thermal viscosity
        real rho         = 0.5 * ( state(idR,hs+km1,hs+j,hs+i) + state(idR,hs+k,hs+j,hs+i) );
        real K           = dns ? 0 : 0.5 * ( tke      (hs+km1,hs+j,hs+i) + tke      (hs+k,hs+j,hs+i) );
        real tref        = 0.5 * ( hy_t(hs+km1) + hy_t(hs+k) );
        real N           = dt_dz+dth_dz >= 0 ? std::sqrt(grav/tref*(dt_dz+dth_dz)) : 0;
        real delta       = std::pow( dx*dy*dzloc , 1./3. ) * delta_mult;
        real ell         = std::min( 0.76*std::sqrt(K)/std::max(N,1.e-10) , delta );
        real km          = 0.1 * ell * std::sqrt(K);
        real Pr_t        = 0.85;
        real visc_max_z  = 0.1*dzloc*dzloc/dtphys;
        real visc_tot    = dns ? nu : std::min( km+nu         , visc_max_z );
        real visc_tot_th = dns ? nu : std::min( km/Pr_t+nu/Pr , visc_max_z );
        // Limit the viscosity and set the max_triggered flag if the limit is reached
        if (visc_tot == visc_max_z || visc_tot_th == visc_max_z) max_triggered = true;
        // Compute the SGS fluxes in the z-direction
        flux_ru_z (k,j,i) = -rho*visc_tot   *(du_dz + dw_dx                            );
        flux_rv_z (k,j,i) = -rho*visc_tot   *(dv_dz + dw_dy                            );
        flux_rw_z (k,j,i) = -rho*visc_tot   *(dw_dz + dw_dz - 2./3.*(du_dx+dv_dy+dw_dz))-2./3.*rho*K;
        flux_rt_z (k,j,i) = -rho*visc_tot_th*(dt_dz                                    );
        flux_tke_z(k,j,i) = -rho*visc_tot   *(dK_dz                                    );
        for (int tr=0; tr < num_tracers; tr++) {
          dt_dz = (tracers(tr,hs+k,hs+j,hs+i) - tracers(tr,hs+km1,hs+j,hs+i))/dzloc; // Tracer gradient
          flux_tracers_z(tr,k,j,i) = -rho*visc_tot_th*dt_dz;
        }
      });

      // Warn the user if the viscosity capping was triggered anywhere. If this happens, there is likely a problem.
      if (max_triggered.hostRead()) std::cout << "WARNING: les_closure max triggered" << std::endl;

      // Compute TKE source terms in each cell
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        // Compute the vertical grid spacing, density, TKE, temperature, temperature gradient,
        //   hydrostatic temperature gradient Brunt-Vaisala frequency, grid spacing, mixing length, eddy viscosity,
        //   and turbulent Prandtl number
        real dz2    = dz(k) + dz(std::max(0,k-1))/2 + dz(std::min(nz-1,k+1))/2;
        real rho    = state(idR,hs+k,hs+j,hs+i);
        real K      = tke      (hs+k,hs+j,hs+i);
        real t      = state(idT,hs+k,hs+j,hs+i) + hy_t(hs+k);
        real dt_dz  = ( state(idT,hs+k+1,hs+j,hs+i) - state(idT,hs+k-1,hs+j,hs+i) ) / (dz2);
        real dth_dz = (hy_t(hs+k+1)-hy_t(hs+k-1))/(dz2);
        real N      = dt_dz+dth_dz >= 0 ? std::sqrt(grav/t*(dt_dz+dth_dz)) : 0;
        real delta  = std::pow( dx*dy*dz(k) , 1./3. ) * delta_mult;
        real ell    = std::min( 0.76*std::sqrt(K)/std::max(N,1.e-10) , delta );
        real km     = 0.1 * ell * std::sqrt(K);
        real Pr_t   = 0.85;
        // Compute tke cell-averaged source
        tke_source(k,j,i) = 0; // Initialize to zero for accumulation
        if (immersed(hs+k,hs+j,hs+i) <= imm_th) {
          // Buoyancy source
          if (enable_gravity) {
            tke_source(k,j,i) += -(grav*rho*km)/(t*Pr_t)*(dt_dz+dth_dz);
          }
          // TKE dissipation
          real c_eps = 0.19 + 0.74*ell/delta;
          tke_source(k,j,i) -= c_eps*rho*std::pow(K,1.5)/std::max(ell,1.e-10);
          // Shear production
          // Compute indices that do not reach into immersed boundaries
          int im1 = immersed(hs+k,hs+j,hs+i-1) > imm_th ? i : i-1;
          int ip1 = immersed(hs+k,hs+j,hs+i+1) > imm_th ? i : i+1;
          int jm1 = immersed(hs+k,hs+j-1,hs+i) > imm_th ? j : j-1;
          int jp1 = immersed(hs+k,hs+j+1,hs+i) > imm_th ? j : j+1;
          int km1 = immersed(hs+k-1,hs+j,hs+i) > imm_th ? k : k-1;
          int kp1 = immersed(hs+k+1,hs+j,hs+i) > imm_th ? k : k+1;
          // Compute derivatives
          real du_dx = ( state(idU,hs+k,hs+j,hs+i+1) - state(idU,hs+k,hs+j,hs+i-1) ) / (2*dx);
          real dv_dx = ( state(idV,hs+k,hs+j,hs+ip1) - state(idV,hs+k,hs+j,hs+im1) ) / (2*dx);
          real dw_dx = ( state(idW,hs+k,hs+j,hs+ip1) - state(idW,hs+k,hs+j,hs+im1) ) / (2*dx);
          real du_dy = ( state(idU,hs+k,hs+jp1,hs+i) - state(idU,hs+k,hs+jm1,hs+i) ) / (2*dy);
          real dv_dy = ( state(idV,hs+k,hs+j+1,hs+i) - state(idV,hs+k,hs+j-1,hs+i) ) / (2*dy);
          real dw_dy = ( state(idW,hs+k,hs+jp1,hs+i) - state(idW,hs+k,hs+jm1,hs+i) ) / (2*dy);
          real du_dz = ( state(idU,hs+kp1,hs+j,hs+i) - state(idU,hs+km1,hs+j,hs+i) ) / (dz2 );
          real dv_dz = ( state(idV,hs+kp1,hs+j,hs+i) - state(idV,hs+km1,hs+j,hs+i) ) / (dz2 );
          real dw_dz = ( state(idW,hs+k+1,hs+j,hs+i) - state(idW,hs+k-1,hs+j,hs+i) ) / (dz2 );
          // Compute shear production term
          real S11 = du_dx;
          real S22 = dv_dy;
          real S33 = dw_dz;
          real S12 = 0.5 * (du_dy + dv_dx);
          real S13 = 0.5 * (du_dz + dw_dx);
          real S23 = 0.5 * (dv_dz + dw_dy);
          real S2  = 2*(S11*S11 + S22*S22 + S33*S33) + 4*(S12*S12 + S13*S13 + S23*S23);
          real Skk = du_dx + dv_dy + dw_dz;
          tke_source(k,j,i) += rho * km * (S2 - 2./3.*Skk*Skk) - 2./3.*rho*K*Skk;
        }
      });

      // Compute total tendencies, multiply state, tracers, and TKE by density, and update state, tracers, and TKE arrays
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real tend_ru  = -(flux_ru_x (k,j,i+1) - flux_ru_x (k,j,i)) / dx -
                         (flux_ru_y (k,j+1,i) - flux_ru_y (k,j,i)) / dy -
                         (flux_ru_z (k+1,j,i) - flux_ru_z (k,j,i)) / dz(k);
        real tend_rv  = -(flux_rv_x (k,j,i+1) - flux_rv_x (k,j,i)) / dx -
                         (flux_rv_y (k,j+1,i) - flux_rv_y (k,j,i)) / dy -
                         (flux_rv_z (k+1,j,i) - flux_rv_z (k,j,i)) / dz(k);
        real tend_rw  = -(flux_rw_x (k,j,i+1) - flux_rw_x (k,j,i)) / dx -
                         (flux_rw_y (k,j+1,i) - flux_rw_y (k,j,i)) / dy -
                         (flux_rw_z (k+1,j,i) - flux_rw_z (k,j,i)) / dz(k);
        real tend_rt  = -(flux_rt_x (k,j,i+1) - flux_rt_x (k,j,i)) / dx -
                         (flux_rt_y (k,j+1,i) - flux_rt_y (k,j,i)) / dy -
                         (flux_rt_z (k+1,j,i) - flux_rt_z (k,j,i)) / dz(k);
        real tend_tke = -(flux_tke_x(k,j,i+1) - flux_tke_x(k,j,i)) / dx -
                         (flux_tke_y(k,j+1,i) - flux_tke_y(k,j,i)) / dy -
                         (flux_tke_z(k+1,j,i) - flux_tke_z(k,j,i)) / dz(k) + tke_source(k,j,i);

        state(idU,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
        state(idU,hs+k,hs+j,hs+i) += dtphys * tend_ru ;

        state(idV,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
        state(idV,hs+k,hs+j,hs+i) += dtphys * tend_rv ;

        state(idW,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
        state(idW,hs+k,hs+j,hs+i) += dtphys * tend_rw ;

        state(idT,hs+k,hs+j,hs+i) += hy_t(hs+k);
        state(idT,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
        state(idT,hs+k,hs+j,hs+i) += dtphys * tend_rt ;

        tke      (hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
        tke      (hs+k,hs+j,hs+i) += dtphys * tend_tke;
        tke      (hs+k,hs+j,hs+i) = std::max( 0. , tke(hs+k,hs+j,hs+i) );

        for (int tr=0; tr < num_tracers; tr++) {
          real tend_tracer = -(flux_tracers_x(tr,k,j,i+1) - flux_tracers_x(tr,k,j,i)) / dx -
                              (flux_tracers_y(tr,k,j+1,i) - flux_tracers_y(tr,k,j,i)) / dy -
                              (flux_tracers_z(tr,k+1,j,i) - flux_tracers_z(tr,k,j,i)) / dz(k);
          tracers(tr,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
          tracers(tr,hs+k,hs+j,hs+i) += dtphys * tend_tracer;
        }
      });

      // auto mass2 = compute_mass( coupler , state , false );
      // if (coupler.is_mainproc()) std::cout << "Mass change: "
      //                                      << (std::get<0>(mass2)-std::get<0>(mass1))/std::get<0>(mass1) << " , "
      //                                      << (std::get<1>(mass2)-std::get<1>(mass1))/std::get<1>(mass1) << std::endl;
      // Convert back to coupler's data structures from state, tracers, and tke arrays
      convert_dynamics_to_coupler( coupler , state , tracers , tke );
    }



    // Convert coupler's data to state and tracers arrays
    // The resulting arrays have halos, and all quantities except density are specific quantities (density divided out)
    //   e.g., velocities, potential temperature, dry mixing ratios, and non-mass-weighted TKE
    //   TKE is not included in the tracers array since it is handled separately
    // coupler : The coupler object containing the data
    // state   : Output state array (with halos and density divided out of momenta and potential temperature)
    // tracers : Output tracers array (with halos and density divided out)
    // tke     : Output TKE array (with halos and density divided out)
    void convert_coupler_to_dynamics( core::Coupler const &coupler ,
                                      real4d              &state   ,
                                      real4d              &tracers ,
                                      real3d              &tke     ) const {
      using yakl::SimpleBounds;
      auto nx           = coupler.get_nx();  // Number of local cells in x-direction (without halos)
      auto ny           = coupler.get_ny();  // Number of local cells in y-direction (without halos)
      auto nz           = coupler.get_nz();  // Number of cells in z-direction (without halos)
      auto R_d          = coupler.get_option<real>("R_d"    ); // Gas constant for dry air
      auto R_v          = coupler.get_option<real>("R_v"    ); // Gas constant for water vapor
      auto gamma        = coupler.get_option<real>("gamma_d"); // Ratio of specific heats for dry air
      auto C0           = coupler.get_option<real>("C0"     ); // p = C0 * theta^gamma
      auto &dm          = coupler.get_data_manager_readonly(); // Get read-only data manager
      auto tracer_names = coupler.get_tracer_names();          // Get list of tracer names
      auto dm_rho_d     = dm.get<real const,3>("density_dry"); // Get dry density from data manager
      auto dm_uvel      = dm.get<real const,3>("uvel"       ); // Get u-velocity from data manager
      auto dm_vvel      = dm.get<real const,3>("vvel"       ); // Get v-velocity from data manager
      auto dm_wvel      = dm.get<real const,3>("wvel"       ); // Get w-velocity from data manager
      auto dm_temp      = dm.get<real const,3>("temperature"); // Get temperature from data manager
      auto dm_tke       = dm.get<real const,3>("TKE"        ); // Get TKE from data manager
      // Accrue all tracers that are to be diffused by the SGS scheme
      core::MultiField<real const,3> dm_tracers;
      for (int tr=0; tr < tracer_names.size(); tr++) {
        std::string tracer_desc;
        bool        tracer_found, positive, adds_mass, diffuse;
        coupler.get_tracer_info( tracer_names[tr] , tracer_desc, tracer_found , positive , adds_mass , diffuse );
        if (diffuse) dm_tracers.add_field( dm.get<real const,3>(tracer_names[tr]) );
      }
      auto num_tracers = dm_tracers.size(); // Number of tracers to be diffused
      state   = real4d("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs); // Allocate state array with halos
      tracers = real4d("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs); // Allocate tracers array with halos
      tke     = real3d("tke"                ,nz+2*hs,ny+2*hs,nx+2*hs); // Allocate TKE array with halos
      // Compute state, tracers, and TKE arrays from coupler's data
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real rho_d = dm_rho_d(k,j,i);
        state(idR,hs+k,hs+j,hs+i) = rho_d;          // Density
        state(idU,hs+k,hs+j,hs+i) = dm_uvel(k,j,i); // u-velocity
        state(idV,hs+k,hs+j,hs+i) = dm_vvel(k,j,i); // v-velocity
        state(idW,hs+k,hs+j,hs+i) = dm_wvel(k,j,i); // w-velocity
        state(idT,hs+k,hs+j,hs+i) = pow( rho_d*R_d*dm_temp(k,j,i)/C0 , 1. / gamma ) / rho_d; // potential temperature
        tke      (hs+k,hs+j,hs+i) = std::max( 0. , dm_tke (k,j,i) / rho_d ); // non-mass-weighted TKE
        // Convert tracers to specific quantities
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i) = dm_tracers(tr,k,j,i)/rho_d; }
      });
    }



    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    // This assumes momenta, mass-weigted potential temperature, and mass-weighted TKE in the state and tracers arrays
    // coupler : The coupler object to write the data to
    // state   : Input state array (with halos and momenta, mass-weighted potential temperature, and mass-weighted TKE)
    // tracers : Input tracers array (with halos and mass-weighted tracers)
    // tke     : Input TKE array (with halos and mass-weighted TKE)
    // Note: TKE is passed separately from tracers since it is not included in the tracers array
    void convert_dynamics_to_coupler( core::Coupler &coupler ,
                                      realConst4d    state   ,
                                      realConst4d    tracers ,
                                      realConst3d    tke     ) const {
      using yakl::SimpleBounds;
      auto nx           = coupler.get_nx();  // Number of local cells in x-direction (without halos)
      auto ny           = coupler.get_ny();  // Number of local cells in y-direction (without halos)
      auto nz           = coupler.get_nz();  // Number of cells in z-direction (without halos)
      auto R_d          = coupler.get_option<real>("R_d"    );  // Gas constant for dry air
      auto R_v          = coupler.get_option<real>("R_v"    );  // Gas constant for water vapor
      auto gamma        = coupler.get_option<real>("gamma_d");  // Ratio of specific heats for dry air
      auto C0           = coupler.get_option<real>("C0"     );  // p = C0 * theta^gamma
      auto &dm          = coupler.get_data_manager_readwrite(); // Get read-write data manager
      auto tracer_names = coupler.get_tracer_names();           // Get list of tracer names
      auto dm_rho_d     = dm.get<real,3>("density_dry");     // Get dry density from data manager
      auto dm_uvel      = dm.get<real,3>("uvel"       );     // Get u-velocity from data manager
      auto dm_vvel      = dm.get<real,3>("vvel"       );     // Get v-velocity from data manager
      auto dm_wvel      = dm.get<real,3>("wvel"       );     // Get w-velocity from data manager
      auto dm_temp      = dm.get<real,3>("temperature");     // Get temperature from data manager
      auto dm_tke       = dm.get<real,3>("TKE"        );     // Get TKE from data manager
      // Accrue all tracers that are to be diffused by the SGS scheme
      core::MultiField<real,3> dm_tracers;
      for (int tr=0; tr < tracer_names.size(); tr++) {
        std::string tracer_desc;
        bool        tracer_found, positive, adds_mass, diffuse;
        coupler.get_tracer_info( tracer_names[tr] , tracer_desc, tracer_found , positive , adds_mass , diffuse );
        if (diffuse) dm_tracers.add_field( dm.get<real,3>(tracer_names[tr]) );
      }
      auto num_tracers = dm_tracers.size(); // Number of tracers to be diffused
      // Compute coupler's data from state, tracers, and TKE arrays
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real rho_d = state(idR,hs+k,hs+j,hs+i);
        dm_rho_d(k,j,i) = rho_d;
        dm_uvel (k,j,i) = state(idU,hs+k,hs+j,hs+i) / rho_d;
        dm_vvel (k,j,i) = state(idV,hs+k,hs+j,hs+i) / rho_d;
        dm_wvel (k,j,i) = state(idW,hs+k,hs+j,hs+i) / rho_d;
        dm_temp (k,j,i) = C0 * pow( state(idT,hs+k,hs+j,hs+i) , gamma ) / ( rho_d * R_d );
        dm_tke  (k,j,i) = tke(hs+k,hs+j,hs+i);
        for (int tr=0; tr < num_tracers; tr++) { dm_tracers(tr,k,j,i) = tracers(tr,hs+k,hs+j,hs+i); }
      });
    }



    // Apply halo boundary conditions to state, tracers, and TKE arrays in all directions
    // Recall that the halo exchange has already performed periodic BCs in the horizontal direction
    //   and that the vertical halos are undefined before this routine is called
    // coupler : The coupler object containing the data
    // state   : Input/output state array (with halos)
    // tracers : Input/output tracers array (with halos)
    // tke     : Input/output TKE array (with halos)
    void halo_bcs( core::Coupler       & coupler ,
                   real4d        const & state   ,
                   real4d        const & tracers ,
                   real3d        const & tke     ) const {
      using yakl::SimpleBounds;
      auto nx             = coupler.get_nx();      // Number of local cells in x-direction (without halos)
      auto ny             = coupler.get_ny();      // Number of local cells in y-direction (without halos)
      auto nz             = coupler.get_nz();      // Number of cells in z-direction (without halos)
      auto dz             = coupler.get_dz();      // Get vertical grid spacing array (1-D array of size nz)
      auto num_tracers    = tracers.extent(0);     // Number of tracers in the tracers array
      auto px             = coupler.get_px();      // Get processor x-coordinate in the processor grid
      auto py             = coupler.get_py();      // Get processor y-coordinate in the processor grid
      auto nproc_x        = coupler.get_nproc_x(); // Get number of processors in x-direction
      auto nproc_y        = coupler.get_nproc_y(); // Get number of processors in y-direction
      auto &dm            = coupler.get_data_manager_readonly();  // Get read-only data manager
      auto grav           = coupler.get_option<real>("grav");     // Gravitational acceleration
      auto gamma          = coupler.get_option<real>("gamma_d");  // Ratio of specific heats for dry air
      auto C0             = coupler.get_option<real>("C0");       // p = C0 * theta^gamma
      auto enable_gravity = coupler.get_option<bool>("enable_gravity",true); // Enable gravity flag
      auto hy_t           = dm.get<real const,1>("les_hy_theta_cells");      // Hydrostatic theta profile
      if (!enable_gravity) grav = 0;  // Disable gravity effects if the flag is false
      core::MultiField<real,3> fields;
      for (int l=0; l < num_state  ; l++) { fields.add_field( state  .slice<3>(l,0,0,0) ); }
      for (int l=0; l < num_tracers; l++) { fields.add_field( tracers.slice<3>(l,0,0,0) ); }
      fields.add_field( tke );
      int num_fields = fields.size();

      // If my MPI task is on the west x-direction boundary and the BC is open, copy values from the first interior cell
      //   for a zero-gradient BC
      if (coupler.get_option<std::string>("bc_x1") == "open" && px == 0                      ) {
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+2*hs,ny+2*hs,hs) , KOKKOS_LAMBDA (int k, int j, int ii) {
          for (int l=0; l < num_state  ; l++) state  (l,k,j,ii) = state  (l,k,j,hs+0);
          for (int l=0; l < num_tracers; l++) tracers(l,k,j,ii) = tracers(l,k,j,hs+0);
          tke(k,j,ii) = tke(k,j,hs+0);
        });
      }
      if (coupler.get_option<std::string>("bc_x1") == "precursor" && px == 0) {
        auto prec_x1 = dm.get<real const,4>("les_ghost_x1");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,ny+2*hs,hs) ,
                                                KOKKOS_LAMBDA (int l, int k, int j, int ii) {
          auto u = fields(idU,k,j,hs);
          fields(l,k,j,hs-1-ii) = u > 0 ? prec_x1(l,k,j,ii) : fields(l,k,j,hs);
        });
      }
      if (coupler.get_option<std::string>("bc_x1") == "nested" && px == 0) {
        auto nested = dm.get<real const,4>("les_nested_x_1");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,ny+2*hs,hs) ,
                                                KOKKOS_LAMBDA (int l, int k, int j, int ii) {
          fields(l,k,j,hs-1-ii) = nested(l,k,j,ii);
        });
      }

      // If my MPI task is on the east x-direction boundary and the BC is open, copy values from the last interior cell
      //   for a zero-gradient BC
      if (coupler.get_option<std::string>("bc_x2") == "open" && px == nproc_x-1) {
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+2*hs,ny+2*hs,hs) , KOKKOS_LAMBDA (int k, int j, int ii) {
          for (int l=0; l < num_state  ; l++) state  (l,k,j,hs+nx+ii) = state  (l,k,j,hs+nx-1);
          for (int l=0; l < num_tracers; l++) tracers(l,k,j,hs+nx+ii) = tracers(l,k,j,hs+nx-1);
          tke(k,j,hs+nx+ii) = tke(k,j,hs+nx-1);
        });
      }
      if (coupler.get_option<std::string>("bc_x2") == "precursor" && px == nproc_x-1) {
        auto prec_x2 = dm.get<real const,4>("les_ghost_x2");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,ny+2*hs,hs) ,
                                                KOKKOS_LAMBDA (int l, int k, int j, int ii) {
          auto u = fields(idU,k,j,hs+nx-1);
          fields(l,k,j,hs+nx+ii) = u > 0 ? fields(l,k,j,hs+nx-1) : prec_x2(l,k,j,ii);
        });
      }
      if (coupler.get_option<std::string>("bc_x2") == "nested" && px == nproc_x-1) {
        auto nested = dm.get<real const,4>("les_nested_x_2");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,ny+2*hs,hs) ,
                                                KOKKOS_LAMBDA (int l, int k, int j, int ii) {
          fields(l,k,j,hs+nx+ii) = nested(l,k,j,ii);
        });
      }

      // If my MPI task is on the south y-direction boundary and the BC is open, copy values from the first interior cell
      //   for a zero-gradient BC
      if (coupler.get_option<std::string>("bc_y1") == "open" && py == 0                      ) {
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+2*hs,hs,nx+2*hs) , KOKKOS_LAMBDA (int k, int jj, int i) {
          for (int l=0; l < num_state  ; l++) state  (l,k,jj,i) = state  (l,k,hs+0,i);
          for (int l=0; l < num_tracers; l++) tracers(l,k,jj,i) = tracers(l,k,hs+0,i);
          tke(k,jj,i) = tke(k,hs+0,i);
        });
      }
      if (coupler.get_option<std::string>("bc_y1") == "wall_free_slip" && py == 0                      ) {
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+2*hs,hs,nx+2*hs) , KOKKOS_LAMBDA (int k, int jj, int i) {
          for (int l=0; l < num_state  ; l++) state  (l,k,jj,i) = l==idV ? 0 : state  (l,k,hs+0,i);
          for (int l=0; l < num_tracers; l++) tracers(l,k,jj,i) = tracers(l,k,hs+0,i);
          tke(k,jj,i) = tke(k,hs+0,i);
        });
      }
      if (coupler.get_option<std::string>("bc_y1") == "precursor" && py == 0) {
        auto prec_y1 = dm.get<real const,4>("les_ghost_y1");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,hs,nx+2*hs) ,
                                                KOKKOS_LAMBDA (int l, int k, int jj, int i) {
          auto v = fields(idV,k,hs,i);
          fields(l,k,hs-1-jj,i) = v > 0 ? prec_y1(l,k,jj,i) : fields(l,k,hs,i);
        });
      }
      if (coupler.get_option<std::string>("bc_y1") == "nested" && py == 0) {
        auto nested = dm.get<real const,4>("les_nested_y_1");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,hs,nx+2*hs) ,
                                                KOKKOS_LAMBDA (int l, int k, int jj, int i) {
          fields(l,k,hs-1-jj,i) = nested(l,k,jj,i);
        });
      }

      // If my MPI task is on the north y-direction boundary and the BC is open, copy values from the last interior cell
      //   for a zero-gradient BC
      if (coupler.get_option<std::string>("bc_y2") == "open" && py == nproc_y-1) {
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+2*hs,hs,nx+2*hs) , KOKKOS_LAMBDA (int k, int jj, int i) {
          for (int l=0; l < num_state  ; l++) state  (l,k,hs+ny+jj,i) = state  (l,k,hs+ny-1,i);
          for (int l=0; l < num_tracers; l++) tracers(l,k,hs+ny+jj,i) = tracers(l,k,hs+ny-1,i);
          tke(k,hs+ny+jj,i) = tke(k,hs+ny-1,i);
        });
      }
      if (coupler.get_option<std::string>("bc_y2") == "wall_free_slip" && py == nproc_y-1) {
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+2*hs,hs,nx+2*hs) , KOKKOS_LAMBDA (int k, int jj, int i) {
          for (int l=0; l < num_state  ; l++) state  (l,k,hs+ny+jj,i) = l==idV ? 0 : state  (l,k,hs+ny-1,i);
          for (int l=0; l < num_tracers; l++) tracers(l,k,hs+ny+jj,i) = tracers(l,k,hs+ny-1,i);
          tke(k,hs+ny+jj,i) = tke(k,hs+ny-1,i);
        });
      }
      if (coupler.get_option<std::string>("bc_y2") == "precursor" && py == nproc_y-1) {
        auto prec_y2 = dm.get<real const,4>("les_ghost_y2");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,hs,nx+2*hs) ,
                                                KOKKOS_LAMBDA (int l, int k, int jj, int i) {
          auto v = fields(idV,k,hs+ny-1,i);
          fields(l,k,hs+ny+jj,i) = v > 0 ? fields(l,k,hs+ny-1,i) : prec_y2(l,k,jj,i);
        });
      }
      if (coupler.get_option<std::string>("bc_y2") == "nested" && py == nproc_y-1) {
        auto nested = dm.get<real const,4>("les_nested_y_2");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,hs,nx+2*hs) ,
                                                KOKKOS_LAMBDA (int l, int k, int jj, int i) {
          fields(l,k,hs+ny+jj,i) = nested(l,k,jj,i);
        });
      }

      // Apply vertical wall conditions at the bottom boundary if desired (zero gradient for all except w-velocity = 0)
      if (coupler.get_option<std::string>("bc_z1") == "wall_free_slip") {
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) ,
                                                KOKKOS_LAMBDA (int kk, int j, int i) {
          state(idU,kk,j,i) = state(idU,hs+0,j,i);
          state(idV,kk,j,i) = state(idV,hs+0,j,i);
          state(idW,kk,j,i) = 0;
          state(idT,kk,j,i) = state(idT,hs+0,j,i);
          tke  (    kk,j,i) = tke  (    hs+0,j,i);
          for (int l=0; l < num_tracers; l++) { tracers(l,kk,j,i) = tracers(l,hs+0,j,i); }
          // Extrapolate density using hydrostatic balance
          {
            int  k0       = hs;
            int  k        = k0-1-kk;
            real rho0     = state(idR,k0,j,i);
            real theta0   = state(idT,k0,j,i)+hy_t(k0);
            real rho0_gm1 = std::pow(rho0  ,gamma-1);
            real theta0_g = std::pow(theta0,gamma  );
            state(idR,k,j,i) = std::pow( rho0_gm1 + grav*(gamma-1)*dz(0)*(kk+1)/(gamma*C0*theta0_g) ,
                                         1./(gamma-1) );
          }
        });
      }

      // Apply periodic vertical conditions at the bottom boundary if desired
      if (coupler.get_option<std::string>("bc_z1") == "periodic") {
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) ,
                                                KOKKOS_LAMBDA (int kk, int j, int i) {
          state(idR,kk,j,i) = state(idR,nz+kk,j,i);
          state(idU,kk,j,i) = state(idU,nz+kk,j,i);
          state(idV,kk,j,i) = state(idV,nz+kk,j,i);
          state(idW,kk,j,i) = state(idW,nz+kk,j,i);
          state(idT,kk,j,i) = state(idT,nz+kk,j,i);
          tke  (    kk,j,i) = tke  (    nz+kk,j,i);
          for (int l=0; l < num_tracers; l++) { tracers(l,kk,j,i) = tracers(l,nz+kk,j,i); }
        });
      }

      // Apply vertical wall conditions at the top boundary if desired (zero gradient for all except w-velocity = 0)
      if (coupler.get_option<std::string>("bc_z2") == "wall_free_slip") {
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) ,
                                                KOKKOS_LAMBDA (int kk, int j, int i) {
          state(idU,hs+nz+kk,j,i) = state(idU,hs+nz-1,j,i);
          state(idV,hs+nz+kk,j,i) = state(idV,hs+nz-1,j,i);
          state(idW,hs+nz+kk,j,i) = 0;
          state(idT,hs+nz+kk,j,i) = state(idT,hs+nz-1,j,i);
          tke  (    hs+nz+kk,j,i) = tke  (    hs+nz-1,j,i);
          for (int l=0; l < num_tracers; l++) { tracers(l,hs+nz+kk,j,i) = tracers(l,hs+nz-1,j,i); }
          // Extrapolate density using hydrostatic balance
          {
            int  k0       = hs+nz-1;
            int  k        = k0+1+kk;
            real rho0     = state(idR,k0,j,i);
            real theta0   = state(idT,k0,j,i)+hy_t(k0);
            real rho0_gm1 = std::pow(rho0  ,gamma-1);
            real theta0_g = std::pow(theta0,gamma  );
            state(idR,k,j,i) = std::pow( rho0_gm1 - grav*(gamma-1)*dz(nz-1)*(kk+1)/(gamma*C0*theta0_g) ,
                                         1./(gamma-1) );
          }
        });
      }

      // Apply precursor values to inflow at the top boundary and zero-gradient values to outflow.
      if (coupler.get_option<std::string>("bc_z2") == "precursor") {
        auto prec_z2 = dm.get<real const,4>("les_ghost_z2");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,hs,ny+2*hs,nx+2*hs) ,
                                                KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          auto w = fields(idW,hs+nz-1,j,i);
          fields(l,hs+nz+kk,j,i) = w < 0 ? prec_z2(l,kk,j,i) : fields(l,hs+nz-1,j,i);
        });
      }
      if (coupler.get_option<std::string>("bc_z2") == "nested") {
        auto nested = dm.get<real const,4>("les_nested_z_2");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,hs,ny+2*hs,nx+2*hs) ,
                                                KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          fields(l,hs+nz+kk,j,i) = nested(l,kk,j,i);
        });
      }

      // Apply periodic vertical conditions at the top boundary if desired
      if (coupler.get_option<std::string>("bc_z2") == "periodic") {
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) ,
                                                KOKKOS_LAMBDA (int kk, int j, int i) {
          state(idR,hs+nz+kk,j,i) = state(idR,hs+kk,j,i);
          state(idU,hs+nz+kk,j,i) = state(idU,hs+kk,j,i);
          state(idV,hs+nz+kk,j,i) = state(idV,hs+kk,j,i);
          state(idW,hs+nz+kk,j,i) = state(idW,hs+kk,j,i);
          state(idT,hs+nz+kk,j,i) = state(idT,hs+kk,j,i);
          tke  (    hs+nz+kk,j,i) = tke  (    hs+kk,j,i);
          for (int l=0; l < num_tracers; l++) { tracers(l,hs+nz+kk,j,i) = tracers(l,hs+kk,j,i); }
        });
      }

      // Store precursor boundary data for copying to a concurrently forced simulation.
      if (coupler.get_option<bool>("coupler_is_precursor",false)) {
        auto &dm_write = coupler.get_data_manager_readwrite();
        if (px == 0) {
          auto ghost_x1 = dm_write.get<real,4>("les_ghost_x1");
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,ny+2*hs,hs) ,
                                                  KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            ghost_x1(l,k,j,ii) = fields(l,k,j,hs-1-ii);
          });
        }
        if (px == nproc_x-1) {
          auto ghost_x2 = dm_write.get<real,4>("les_ghost_x2");
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,ny+2*hs,hs) ,
                                                  KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            ghost_x2(l,k,j,ii) = fields(l,k,j,hs+nx+ii);
          });
        }
        if (py == 0) {
          auto ghost_y1 = dm_write.get<real,4>("les_ghost_y1");
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,hs,nx+2*hs) ,
                                                  KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            ghost_y1(l,k,jj,i) = fields(l,k,hs-1-jj,i);
          });
        }
        if (py == nproc_y-1) {
          auto ghost_y2 = dm_write.get<real,4>("les_ghost_y2");
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz+2*hs,hs,nx+2*hs) ,
                                                  KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            ghost_y2(l,k,jj,i) = fields(l,k,hs+ny+jj,i);
          });
        }
        auto ghost_z2 = dm_write.get<real,4>("les_ghost_z2");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,hs,ny+2*hs,nx+2*hs) ,
                                                KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          ghost_z2(l,kk,j,i) = fields(l,hs+nz+kk,j,i);
        });
      }
    }



    // Copy LES precursor ghost cells from a precursor coupler to a concurrently forced coupler.
    void copy_precursor_ghost_cells( core::Coupler const & coupler_prec , core::Coupler & coupler_main ) const {
      int  px       = coupler_main.get_px();
      int  py       = coupler_main.get_py();
      int  nproc_x  = coupler_main.get_nproc_x();
      int  nproc_y  = coupler_main.get_nproc_y();
      auto &dm_prec = coupler_prec.get_data_manager_readonly();
      auto &dm_main = coupler_main.get_data_manager_readwrite();
      if (px == 0) {
        dm_prec.get<real const,4>("les_ghost_x1").deep_copy_to(dm_main.get<real,4>("les_ghost_x1"));
      }
      if (px == nproc_x-1) {
        dm_prec.get<real const,4>("les_ghost_x2").deep_copy_to(dm_main.get<real,4>("les_ghost_x2"));
      }
      if (py == 0) {
        dm_prec.get<real const,4>("les_ghost_y1").deep_copy_to(dm_main.get<real,4>("les_ghost_y1"));
      }
      if (py == nproc_y-1) {
        dm_prec.get<real const,4>("les_ghost_y2").deep_copy_to(dm_main.get<real,4>("les_ghost_y2"));
      }
      if (coupler_main.get_option<std::string>("bc_z2") == "precursor") {
        dm_prec.get<real const,4>("les_ghost_z2").deep_copy_to(dm_main.get<real,4>("les_ghost_z2"));
      }
    }

  };

}

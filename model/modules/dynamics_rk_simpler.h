
#pragma once

#include "main_header.h"
#include "coupler.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "WenoLimiter.h"
#include <random>
#include <sstream>

namespace modules {

  // This clas simplements an A-grid (collocated) cell-centered Finite-Volume method with an upwind Godunov Riemanns
  // solver at cell edges, high-order-accurate reconstruction, Weighted Essentially Non-Oscillatory (WENO) limiting,
  // and Strong Stability Preserving Runge-Kutta time stepping.
  // The dycore prognoses full density, u-, v-, and w-momenta, and mass-weighted potential temperature
  // Since the coupler state is dry density, u-, v-, and w-velocity, and temperature, we need to convert to and from
  // the coupler state.
  // This dynamical core supports immersed boundaries (fully immersed only. Partially immersed are ignored). Immersed
  // boundaries will have no-slip wall BC's, and surface fluxes are applied in a separate module to model friction
  // based on a prescribed roughness length with Monin-Obukhov thoery.
  // You'll notice the dimensions are nz,ny,nx.

  struct Dynamics_Euler_Stratified_WenoFV {
    // Order of accuracy (numerical convergence for smooth flows) for the dynamical core
    #ifndef PORTURB_ORD
      size_t static constexpr ord = 8;
    #else
      size_t static constexpr ord = PORTURB_ORD;
    #endif
    int static constexpr hs  = ord/2; // Number of halo cells ("hs" == "halo size")
    int static constexpr num_state = 5;   // Number of state variables
    // IDs for the variables in the state vector
    int  static constexpr idR = 0;  // Density
    int  static constexpr idU = 1;  // u-momentum
    int  static constexpr idV = 2;  // v-momentum
    int  static constexpr idW = 3;  // w-momentum
    int  static constexpr idT = 4;  // Density * potential temperature



    real compute_time_step( core::Coupler const &coupler ) const {
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      real maxwave = 350 + coupler.get_option<real>( "dycore_max_wind" , 100 );
      real cfl = coupler.get_option<real>("cfl",0.70);
      return cfl * std::min( std::min( dx , dy ) , dz ) / maxwave;
    }
    // real compute_time_step( core::Coupler const &coupler ) const {
    //   using yakl::c::parallel_for;
    //   using yakl::c::SimpleBounds;
    //   auto nx = coupler.get_nx();
    //   auto ny = coupler.get_ny();
    //   auto nz = coupler.get_nz();
    //   auto dx = coupler.get_dx();
    //   auto dy = coupler.get_dy();
    //   auto dz = coupler.get_dz();
    //   auto R_d = coupler.get_option<real>("R_d");
    //   auto gamma = coupler.get_option<real>("gamma_d");
    //   auto &dm = coupler.get_data_manager_readonly();
    //   auto rho_d = dm.get<real const,3>("density_dry");
    //   auto uvel  = dm.get<real const,3>("uvel"       );
    //   auto vvel  = dm.get<real const,3>("vvel"       );
    //   auto wvel  = dm.get<real const,3>("wvel"       );
    //   auto temp  = dm.get<real const,3>("temp"       );
    //   real3d dt3d("dt3d",nz,ny,nx);
    //   real cfl = coupler.get_option<real>("cfl",0.15);
    //   parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
    //     real r = rho_d(k,j,i);
    //     real u = uvel (k,j,i);
    //     real v = vvel (k,j,i);
    //     real w = wvel (k,j,i);
    //     real T = temp (k,j,i);
    //     real p = r*R_d*T;
    //     real cs = std::sqrt(gamma*p/r);
    //     real dtx = cfl*dx/(std::abs(u)+cs);
    //     real dty = cfl*dy/(std::abs(v)+cs);
    //     real dtz = cfl*dz/(std::abs(w)+cs);
    //     dt3d(k,j,i) = std::min(std::min(dtx,dty),dtz);
    //   });
    //   real maxwave = yakl::intrinsics::minval(dt3d);
    //   return coupler.get_parallel_comm().all_reduce( maxwave , MPI_MIN );
    // }



    // Perform a time step
    void time_step(core::Coupler &coupler, real dt_phys) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers             = coupler.get_num_tracers();
      auto nx                      = coupler.get_nx();
      auto ny                      = coupler.get_ny();
      auto nz                      = coupler.get_nz();
      real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
      convert_coupler_to_dynamics( coupler , state , tracers );
      real dt_dyn = compute_time_step( coupler );
      int ncycles = (int) std::ceil( dt_phys / dt_dyn );
      dt_dyn = dt_phys / ncycles;
      for (int icycle = 0; icycle < ncycles; icycle++) { time_step_rk_3_3(coupler,state,tracers,dt_dyn); }
      convert_dynamics_to_coupler( coupler , state , tracers );
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step");
      #endif
    }



    void time_step_rk_3_3( core::Coupler & coupler ,
                           real4d const  & state   ,
                           real4d const  & tracers ,
                           real            dt_dyn  ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step_rk_3_3");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto &dm         = coupler.get_data_manager_readonly();
      auto tracer_positive = dm.get<bool const,1>("tracer_positive");
      // SSPRK3 requires temporary arrays to hold intermediate state and tracers arrays
      real4d state_tmp   ("state_tmp"   ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d tracers_tmp ("tracers_tmp" ,num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
      // To hold tendencies
      real4d state_tend  ("state_tend"  ,num_state  ,nz     ,ny     ,nx     );
      real4d tracers_tend("tracers_tend",num_tracers,nz     ,ny     ,nx     );

      enforce_immersed_boundaries( coupler , state , tracers );

      //////////////
      // Stage 1
      //////////////
      compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,hs+k,hs+j,hs+i) = state  (l,hs+k,hs+j,hs+i) + dt_dyn * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,hs+k,hs+j,hs+i) = tracers(l,hs+k,hs+j,hs+i) + dt_dyn * tracers_tend(l,k,j,i);
        }
      });

      enforce_immersed_boundaries( coupler , state_tmp , tracers_tmp );

      //////////////
      // Stage 2
      //////////////
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/4.);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,hs+k,hs+j,hs+i) = (3._fp/4._fp) * state      (l,hs+k,hs+j,hs+i) + 
                                          (1._fp/4._fp) * state_tmp  (l,hs+k,hs+j,hs+i) +
                                          (1._fp/4._fp) * dt_dyn * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,hs+k,hs+j,hs+i) = (3._fp/4._fp) * tracers    (l,hs+k,hs+j,hs+i) + 
                                          (1._fp/4._fp) * tracers_tmp(l,hs+k,hs+j,hs+i) +
                                          (1._fp/4._fp) * dt_dyn * tracers_tend(l,k,j,i);
        }
      });

      enforce_immersed_boundaries( coupler , state_tmp , tracers_tmp );

      //////////////
      // Stage 3
      //////////////
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn*2./3.);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state  (l,hs+k,hs+j,hs+i) = (1._fp/3._fp) * state      (l,hs+k,hs+j,hs+i) +
                                      (2._fp/3._fp) * state_tmp  (l,hs+k,hs+j,hs+i) +
                                      (2._fp/3._fp) * dt_dyn * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers(l,hs+k,hs+j,hs+i) = (1._fp/3._fp) * tracers    (l,hs+k,hs+j,hs+i) +
                                      (2._fp/3._fp) * tracers_tmp(l,hs+k,hs+j,hs+i) +
                                      (2._fp/3._fp) * dt_dyn * tracers_tend(l,k,j,i);
          // Ensure positive tracers stay positive
          if (tracer_positive(l))  tracers(l,hs+k,hs+j,hs+i) = std::max( 0._fp , tracers(l,hs+k,hs+j,hs+i) );
        }
      });

      enforce_immersed_boundaries( coupler , state , tracers );

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step_rk_3_3");
      #endif
    }


    void enforce_immersed_boundaries( core::Coupler       & coupler ,
                                      real4d        const & state   ,
                                      real4d        const & tracers ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("enforce_immersed_boundaries");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers     = coupler.get_num_tracers();
      auto nx              = coupler.get_nx();
      auto ny              = coupler.get_ny();
      auto nz              = coupler.get_nz();
      auto immersed_power  = coupler.get_option<real>("immersed_power",5);
      auto &dm             = coupler.get_data_manager_readonly();
      auto hy_dens_cells   = dm.get<float const,1>("hy_dens_cells" ); // Hydrostatic density
      auto hy_theta_cells  = dm.get<float const,1>("hy_theta_cells"); // Hydrostatic potential temperature
      auto immersed_prop   = dm.get<real  const,3>("dycore_immersed_proportion_halos"); // Immersed Proportion
      auto tracer_positive = dm.get<bool const,1>("tracer_positive");

      if (! dm.entry_exists("dycore_immersed_tau")) {
        coupler.get_data_manager_readwrite().register_and_allocate<int>("dycore_immersed_tau","",{nz,ny,nx});
        auto immersed_tau = coupler.get_data_manager_readwrite().get<int,3>("dycore_immersed_tau");
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
          bool not_immersed_1 = false;
          bool not_immersed_2 = false;
          bool not_immersed_3 = false;
          int rad = 1;
          for (int kk=-rad; kk <= rad; kk++) {
            for (int jj=-rad; jj <= rad; jj++) {
              for (int ii=-rad; ii <= rad; ii++) {
                if ( immersed_prop(hs+k+kk,hs+j+jj,hs+i+ii) == 0 ) not_immersed_1 = true;
              }
            }
          }
          rad = 2;
          for (int kk=-rad; kk <= rad; kk++) {
            for (int jj=-rad; jj <= rad; jj++) {
              for (int ii=-rad; ii <= rad; ii++) {
                if ( immersed_prop(hs+k+kk,hs+j+jj,hs+i+ii) == 0 ) not_immersed_2 = true;
              }
            }
          }
          rad = 3;
          for (int kk=-rad; kk <= rad; kk++) {
            for (int jj=-rad; jj <= rad; jj++) {
              for (int ii=-rad; ii <= rad; ii++) {
                if ( immersed_prop(hs+k+kk,hs+j+jj,hs+i+ii) == 0 ) not_immersed_3 = true;
              }
            }
          }
          if      ( not_immersed_1 ) { immersed_tau(k,j,i) = 8; }
          else if ( not_immersed_2 ) { immersed_tau(k,j,i) = 4; }
          else if ( not_immersed_3 ) { immersed_tau(k,j,i) = 2; }
          else                       { immersed_tau(k,j,i) = 1; }
        });
      }

      auto immersed_tau = dm.get<int const,3>("dycore_immersed_tau");

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real mult = std::pow( immersed_prop(hs+k,hs+j,hs+i) , immersed_power ) / immersed_tau(k,j,i);
        // TODO: Find a way to calculate drag in here
        // Density
        {
          auto &var = state(idR,hs+k,hs+j,hs+i);
          real  target = hy_dens_cells(hs+k);
          var = var + (target - var)*mult;
        }
        // u-momentum
        {
          auto &var = state(idU,hs+k,hs+j,hs+i);
          real  target = 0;
          var = var + (target - var)*mult;
        }
        // v-momentum
        {
          auto &var = state(idV,hs+k,hs+j,hs+i);
          real  target = 0;
          var = var + (target - var)*mult;
        }
        // w-momentum
        {
          auto &var = state(idW,hs+k,hs+j,hs+i);
          real  target = 0;
          var = var + (target - var)*mult;
        }
        // density*theta
        {
          auto &var = state(idT,hs+k,hs+j,hs+i);
          real  target = hy_dens_cells(hs+k)*hy_theta_cells(hs+k);
          var = var + (target - var)*mult;
        }
        // Tracers
        for (int tr=0; tr < num_tracers; tr++) {
          auto &var = tracers(tr,hs+k,hs+j,hs+i);
          real  target = 0;
          var = var + (target - var)*mult;
          if (tracer_positive(tr))  var = std::max( 0._fp , var );
        }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("enforce_immersed_boundaries");
      #endif
    }



    // Once you encounter an immersed boundary, set zero derivative boundary conditions
    template <class FP, size_t ORD>
    KOKKOS_INLINE_FUNCTION static void modify_stencil_immersed_der0( SArray<FP  ,1,ORD>       & stencil  ,
                                                          SArray<bool,1,ORD> const & immersed ) {
      int constexpr hs = (ORD-1)/2;
      // Don't modify the stencils of immersed cells
      if (! immersed(hs)) {
        // Move out from the center of the stencil. once you encounter a boundary, enforce zero derivative,
        //     which is essentially replication of the last in-domain value
        for (int i2=hs+1; i2<ORD; i2++) {
          if (immersed(i2)) { for (int i3=i2; i3<ORD; i3++) { stencil(i3) = stencil(i2-1); }; break; }
        }
        for (int i2=hs-1; i2>=0 ; i2--) {
          if (immersed(i2)) { for (int i3=i2; i3>=0 ; i3--) { stencil(i3) = stencil(i2+1); }; break; }
        }
      }
    }



    real static KOKKOS_INLINE_FUNCTION interp_val( SArray<float,1,ord> const &s) {
      if      constexpr (ord == 2 ) { return 0.5*(s(0)+s(1)); }
      else if constexpr (ord == 4 ) { return -1.f/12.f*s(0) + 7.f/12.f*s(1) + 7.f/12.f*s(2) - 1.f/12.f*s(3); }
      else if constexpr (ord == 6 ) { return 1.f/60.f*s(0) - 2.f/15.f*s(1) + 37.f/60.f*s(2) + 37.f/60.f*s(3) - 2.f/15.f*s(4) + 1.f/60.f*s(5); }
      else if constexpr (ord == 8 ) { return -1.f/280.f*s(0) + 29.f/840.f*s(1) - 139.f/840.f*s(2) + 533.f/840.f*s(3) + 533.f/840.f*s(4) - 139.f/840.f*s(5) + 29.f/840.f*s(6) - 1.f/280.f*s(7); }
      else if constexpr (ord == 10) { return 1.f/1260.f*s(0) - 23.f/2520.f*s(1) + 127.f/2520.f*s(2) - 473.f/2520.f*s(3) + 1627.f/2520.f*s(4) + 1627.f/2520.f*s(5) - 473.f/2520.f*s(6) + 127.f/2520.f*s(7) - 23.f/2520.f*s(8) + 1.f/1260.f*s(9); }
    }



    real static KOKKOS_INLINE_FUNCTION interp_der( SArray<float,1,ord> const &s) {
      if      constexpr (ord == 2 ) { return (s(1)-s(0)); }
      else if constexpr (ord == 4 ) { return -s(0) + 3*s(1) - 3*s(2) + s(3) ; }
      else if constexpr (ord == 6 ) { return -s(0) + 5*s(1) - 10*s(2) + 10*s(3) - 5*s(4) + s(5) ; }
      else if constexpr (ord == 8 ) { return -s(0) + 7*s(1) - 21*s(2) + 35*s(3) - 35*s(4) + 21*s(5) - 7*s(6) + s(7) ; }
      else if constexpr (ord == 10) { return -s(0) + 9*s(1) - 36*s(2) + 84*s(3) - 126*s(4) + 126*s(5) - 84*s(6) + 36*s(7) - 9*s(8) + s(9) ; }
    }



    int static constexpr idP = 5;

    void compute_tendencies( core::Coupler       & coupler      ,
                             real4d        const & state        ,
                             real4d        const & state_tend   ,
                             real4d        const & tracers      ,
                             real4d        const & tracers_tend ,
                             real                  dt           ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("compute_tendencies");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto  nx                = coupler.get_nx();    // Proces-local number of cells
      auto  ny                = coupler.get_ny();    // Proces-local number of cells
      auto  nz                = coupler.get_nz();    // Total vertical cells
      auto  dx                = coupler.get_dx();    // grid spacing
      auto  dy                = coupler.get_dy();    // grid spacing
      auto  dz                = coupler.get_dz();    // grid spacing
      auto  sim2d             = coupler.is_sim2d();  // Is this a 2-D simulation?
      auto  enable_gravity    = coupler.get_option<bool>("enable_gravity",true);
      auto  C0                = coupler.get_option<real>("C0"     );  // pressure = C0*pow(rho*theta,gamma)
      auto  grav              = coupler.get_option<real>("grav"   );  // Gravity
      auto  gamma             = coupler.get_option<real>("gamma_d");  // cp_dry / cv_dry (about 1.4)
      auto  latitude          = coupler.get_option<real>("latitude",0); // For coriolis
      auto  num_tracers       = coupler.get_num_tracers();            // Number of tracers
      auto  &dm               = coupler.get_data_manager_readonly();  // Grab read-only data manager
      auto  tracer_positive   = dm.get<bool const,1>("tracer_positive"          ); // Is a tracer positive-definite?
      auto  immersed_prop     = dm.get<real const,3>("dycore_immersed_proportion_halos"); // Immersed Proportion
      auto  any_immersed      = dm.get<bool const,3>("dycore_any_immersed10"    ); // Are any immersed in 3-D halo?
      auto  hy_dens_cells     = dm.get<float const,1>("hy_dens_cells"            ); // Hydrostatic density
      auto  hy_theta_cells    = dm.get<float const,1>("hy_theta_cells"           ); // Hydrostatic potential temperature
      auto  hy_dens_edges     = dm.get<float const,1>("hy_dens_edges"            ); // Hydrostatic density
      auto  hy_theta_edges    = dm.get<float const,1>("hy_theta_edges"           ); // Hydrostatic potential temperature
      auto  hy_pressure_edges = dm.get<float const,1>("hy_pressure_edges"        ); // Hydrostatic potential temperature
      auto  hy_pressure_cells = dm.get<float const,1>("hy_pressure_cells"        ); // Hydrostatic pressure
      auto  weno_all          = coupler.get_option<bool>("weno_all",true);
      // Compute matrices to convert polynomial coefficients to 2 GLL points and stencil values to 2 GLL points
      // These matrices will be in column-row format. That performed better than row-column format in performance tests
      real r_dx = 1./dx; // reciprocal of grid spacing
      real r_dy = 1./dy; // reciprocal of grid spacing
      real r_dz = 1./dz; // reciprocal of grid spacing
      real fcor = 2*7.2921e-5*std::sin(latitude/180*M_PI);      // For coriolis: 2*Omega*sin(latitude)

      float4d fields_loc("fields_loc",num_state+num_tracers+1,nz+2*hs,ny+2*hs,nx+2*hs);

      // Compute pressure
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        fields_loc(idP,hs+k,hs+j,hs+i) = C0*std::pow(state(idT,hs+k,hs+j,hs+i),gamma) - hy_pressure_cells(hs+k);
        real r_r = 1._fp / state(idR,hs+k,hs+j,hs+i);
        fields_loc(idR,hs+k,hs+j,hs+i) = state(idR,hs+k,hs+j,hs+i);
        for (int l=1; l < num_state  ; l++) { fields_loc(            l,hs+k,hs+j,hs+i) = state  (l,hs+k,hs+j,hs+i)*r_r; }
        for (int l=0; l < num_tracers; l++) { fields_loc(num_state+1+l,hs+k,hs+j,hs+i) = tracers(l,hs+k,hs+j,hs+i)*r_r; }
        fields_loc(idR,hs+k,hs+j,hs+i) -= hy_dens_cells (hs+k);
        fields_loc(idT,hs+k,hs+j,hs+i) -= hy_theta_cells(hs+k);
      });

      // Perform periodic halo exchange in the horizontal, and implement vertical no-slip solid wall boundary conditions
      #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
      yakl::timer_start("dycore_halo_exchange_x");
      #endif
      if (ord > 1) coupler.halo_exchange_x( fields_loc , hs );
      #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
      yakl::timer_stop("dycore_halo_exchange_x");
      yakl::timer_start("dycore_halo_exchange_y");
      #endif
      if (ord > 1) coupler.halo_exchange_y( fields_loc , hs );
      #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
      yakl::timer_stop("dycore_halo_exchange_y");
      #endif
      halo_boundary_conditions( coupler , fields_loc );

      float hv_beta = 0.01;

      float hv_coef;
      if      constexpr (ord == 2 ) { hv_coef =  std::pow(2.,-2. )*dx/dt*hv_beta; }
      else if constexpr (ord == 4 ) { hv_coef = -std::pow(2.,-4. )*dx/dt*hv_beta; }
      else if constexpr (ord == 6 ) { hv_coef =  std::pow(2.,-6. )*dx/dt*hv_beta; }
      else if constexpr (ord == 8 ) { hv_coef = -std::pow(2.,-8. )*dx/dt*hv_beta; }
      else if constexpr (ord == 10) { hv_coef =  std::pow(2.,-10.)*dx/dt*hv_beta; }

      float4d flux_x("flux_x",num_state+num_tracers,nz,ny,nx+1);
      float4d flux_y("flux_y",num_state+num_tracers,nz,ny+1,nx);
      float4d flux_z("flux_z",num_state+num_tracers,nz+1,ny,nx);

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx+1) , KOKKOS_LAMBDA (int k, int j, int i) {
        float r, ru;
        {
          SArray<bool,1,ord> immersed;
          for (int ii = 0; ii < ord; ii++) { immersed(ii) = immersed_prop(hs+k,hs+j,i+ii) > 0; }
          SArray<float,1,ord> s;
          // Density & derivative
          for (int ii = 0; ii < ord; ii++) { s(ii) = fields_loc(idR,hs+k,hs+j,i+ii); }
                r  = interp_val(s)+hy_dens_cells(hs+k);
          float dr = interp_der(s);
          // u-velocity & rho*u derivative
          for (int ii = 0; ii < ord; ii++) { s(ii) = fields_loc(idU,hs+k,hs+j,i+ii); }
          float u  = interp_val(s);
          float du = interp_der(s);
          // v-velocity and rho*v derivative
          for (int ii = 0; ii < ord; ii++) { s(ii) = fields_loc(idV,hs+k,hs+j,i+ii); }
          modify_stencil_immersed_der0( s , immersed );
          float v  = interp_val(s);
          float dv = interp_der(s);
          // w-velocity and rho*w derivative
          for (int ii = 0; ii < ord; ii++) { s(ii) = fields_loc(idW,hs+k,hs+j,i+ii); }
          modify_stencil_immersed_der0( s , immersed );
          float w  = interp_val(s);
          float dw = interp_der(s);
          // pressure perturbation
          for (int ii = 0; ii < ord; ii++) { s(ii) = fields_loc(idP,hs+k,hs+j,i+ii); }
          modify_stencil_immersed_der0( s , immersed );
          float p = interp_val(s);
          // Assemble flux vector for rho, rho*u, rho*v, and rho*w
          flux_x(idR,k,j,i) = r*u     - hv_coef*dr  ;
          flux_x(idU,k,j,i) = r*u*u+p - hv_coef*du*r;
          flux_x(idV,k,j,i) = r*u*v   - hv_coef*dv*r;
          flux_x(idW,k,j,i) = r*u*w   - hv_coef*dw*r;
          ru = r*u;
        }
        SArray<float,1,ord> s;
        // theta & rho*theta derivative
        for (int ii = 0; ii < ord; ii++) { s(ii) = fields_loc(idT,hs+k,hs+j,i+ii); }
        float t  = interp_val(s)+hy_theta_cells(hs+k);
        float dt = interp_der(s);
        flux_x(idT,k,j,i) = ru*t - hv_coef*dt*r;
        // tracers & rho*tracer derivatives
        #pragma nounroll
        for (int tr=0; tr < num_tracers; tr++) {
          for (int ii = 0; ii < ord; ii++) { s(ii) = fields_loc(num_state+1+tr,hs+k,hs+j,i+ii); }
          float t  = interp_val(s);
          float dt = interp_der(s);
          flux_x(num_state+tr,k,j,i) = ru*t - hv_coef*dt*r;
        }
      });

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny+1,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        float r, rv;
        {
          SArray<bool,1,ord> immersed;
          for (int jj = 0; jj < ord; jj++) { immersed(jj) = immersed_prop(hs+k,j+jj,hs+i) > 0; }
          SArray<float,1,ord> s;
          // Density & derivative
          for (int jj = 0; jj < ord; jj++) { s(jj) = fields_loc(idR,hs+k,j+jj,hs+i); }
                r  = interp_val(s)+hy_dens_cells(hs+k);
          float dr = interp_der(s);
          // u-velocity & rho*u derivative
          for (int jj = 0; jj < ord; jj++) { s(jj) = fields_loc(idU,hs+k,j+jj,hs+i); }
          modify_stencil_immersed_der0( s , immersed );
          float u  = interp_val(s);
          float du = interp_der(s);
          // v-velocity and rho*v derivative
          for (int jj = 0; jj < ord; jj++) { s(jj) = fields_loc(idV,hs+k,j+jj,hs+i); }
          float v  = interp_val(s);
          float dv = interp_der(s);
          // w-velocity and rho*w derivative
          for (int jj = 0; jj < ord; jj++) { s(jj) = fields_loc(idW,hs+k,j+jj,hs+i); }
          modify_stencil_immersed_der0( s , immersed );
          float w  = interp_val(s);
          float dw = interp_der(s);
          // pressure perturbation
          for (int jj = 0; jj < ord; jj++) { s(jj) = fields_loc(idP,hs+k,j+jj,hs+i); }
          modify_stencil_immersed_der0( s , immersed );
          float p = interp_val(s);
          // Assemble flux vector for rho, rho*u, rho*v, and rho*w
          flux_y(idR,k,j,i) = r*v     - hv_coef*dr ;
          flux_y(idU,k,j,i) = r*v*u   - hv_coef*du*r;
          flux_y(idV,k,j,i) = r*v*v+p - hv_coef*dv*r;
          flux_y(idW,k,j,i) = r*v*w   - hv_coef*dw*r;
          rv = r*v;
        }
        SArray<float,1,ord> s;
        // theta & rho*theta derivative
        for (int jj = 0; jj < ord; jj++) { s(jj) = fields_loc(idT,hs+k,j+jj,hs+i); }
        float t  = interp_val(s)+hy_theta_cells(hs+k);
        float dt = interp_der(s);
        flux_y(idT,k,j,i) = rv*t - hv_coef*dt*r;
        // tracers & rho*tracer derivatives
        #pragma nounroll
        for (int tr=0; tr < num_tracers; tr++) {
          for (int jj = 0; jj < ord; jj++) { s(jj) = fields_loc(num_state+1+tr,hs+k,j+jj,hs+i); }
          float t  = interp_val(s);
          float dt = interp_der(s);
          flux_y(num_state+tr,k,j,i) = rv*t - hv_coef*dt*r;
        }
      });

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        float r, rw;
        {
          SArray<bool,1,ord> immersed;
          for (int kk = 0; kk < ord; kk++) { immersed(kk) = immersed_prop(k+kk,hs+j,hs+i) > 0; }
          SArray<float,1,ord> s;
          // Density & derivative
          for (int kk = 0; kk < ord; kk++) { s(kk) = fields_loc(idR,k+kk,hs+j,hs+i); }
                r  = interp_val(s)+hy_dens_edges(k);
          float dr = interp_der(s);
          // u-velocity & rho*u derivative
          for (int kk = 0; kk < ord; kk++) { s(kk) = fields_loc(idU,k+kk,hs+j,hs+i); }
          modify_stencil_immersed_der0( s , immersed );
          float u  = interp_val(s);
          float du = interp_der(s);
          // v-velocity and rho*v derivative
          for (int kk = 0; kk < ord; kk++) { s(kk) = fields_loc(idV,k+kk,hs+j,hs+i); }
          modify_stencil_immersed_der0( s , immersed );
          float v  = interp_val(s);
          float dv = interp_der(s);
          // w-velocity and rho*w derivative
          for (int kk = 0; kk < ord; kk++) { s(kk) = fields_loc(idW,k+kk,hs+j,hs+i); }
          float w  = interp_val(s);
          float dw = interp_der(s);
          // pressure perturbation
          for (int kk = 0; kk < ord; kk++) { s(kk) = fields_loc(idP,k+kk,hs+j,hs+i); }
          modify_stencil_immersed_der0( s , immersed );
          float p = interp_val(s);
          if (k == 0 ) { w = 0; dw = 0; }
          if (k == nz) { w = 0; dw = 0; }
          // Assemble flux vector for rho, rho*u, rho*v, and rho*w
          flux_z(idR,k,j,i) = r*w     - hv_coef*dr ;
          flux_z(idU,k,j,i) = r*w*u   - hv_coef*du*r;
          flux_z(idV,k,j,i) = r*w*v   - hv_coef*dv*r;
          flux_z(idW,k,j,i) = r*w*w+p - hv_coef*dw*r;
          rw = r*w;
        }
        SArray<float,1,ord> s;
        // theta & rho*theta derivative
        for (int kk = 0; kk < ord; kk++) { s(kk) = fields_loc(idT,k+kk,hs+j,hs+i); }
        float t  = interp_val(s)+hy_theta_edges(k);
        float dt = interp_der(s);
        flux_z(idT,k,j,i) = rw*t - hv_coef*dt*r;
        // tracers & rho*tracer derivatives
        #pragma nounroll
        for (int tr=0; tr < num_tracers; tr++) {
          for (int kk = 0; kk < ord; kk++) { s(kk) = fields_loc(num_state+1+tr,k+kk,hs+j,hs+i); }
          float t  = interp_val(s);
          float dt = interp_der(s);
          flux_z(num_state+tr,k,j,i) = rw*t - hv_coef*dt*r;
        }
      });

      // Compute tendencies as the flux divergence + gravity source term + coriolis
      int mx = std::max(num_state,num_tracers);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(mx,nz,ny,nx) , KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tend(l,k,j,i) = -( flux_x(l,k,j,i+1) - flux_x(l,k,j,i) ) * r_dx
                                -( flux_y(l,k,j+1,i) - flux_y(l,k,j,i) ) * r_dy
                                -( flux_z(l,k+1,j,i) - flux_z(l,k,j,i) ) * r_dz;
          if (l == idV && sim2d) state_tend(l,k,j,i) = 0;
          if (l == idW && enable_gravity) {
            state_tend(l,k,j,i) += -grav*(state(idR,hs+k,hs+j,hs+i) - hy_dens_cells(hs+k));
          }
          if (latitude != 0 && !sim2d && l == idU) state_tend(l,k,j,i) += fcor*state(idV,hs+k,hs+j,hs+i);
          if (latitude != 0 && !sim2d && l == idV) state_tend(l,k,j,i) -= fcor*state(idU,hs+k,hs+j,hs+i);
        }
        if (l < num_tracers) {
          tracers_tend(l,k,j,i) = -( flux_x(num_state+l,k,j,i+1) - flux_x(num_state+l,k,j,i) ) * r_dx
                                  -( flux_y(num_state+l,k,j+1,i) - flux_y(num_state+l,k,j,i) ) * r_dy 
                                  -( flux_z(num_state+l,k+1,j,i) - flux_z(num_state+l,k,j,i) ) * r_dz;
        }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("compute_tendencies");
      #endif
    }



    void halo_boundary_conditions( core::Coupler const & coupler ,
                                   float4d       const & fields  ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("halo_boundary_conditions");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx              = coupler.get_nx();
      auto ny              = coupler.get_ny();
      auto nz              = coupler.get_nz();
      auto num_tracers     = coupler.get_num_tracers();
      auto &dm             = coupler.get_data_manager_readonly();
      auto hy_dens_cells   = dm.get<float const,1>("hy_dens_cells" );
      auto hy_theta_cells  = dm.get<float const,1>("hy_theta_cells");

      if (coupler.get_option<std::string>("bc_x1") == "periodic") { // Already handled in halo_exchange
      } else if (coupler.get_option<std::string>("bc_x1") == "open") {
        if (coupler.get_px() == 0) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,hs) ,
                                            KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            fields(l,hs+k,hs+j,      ii) = fields(l,hs+k,hs+j,hs+0   );
          });
        }
      } else {
        std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_x1 can only be periodic or open";
        Kokkos::abort("");
      }

      if (coupler.get_option<std::string>("bc_x2") == "periodic") { // Already handled in halo_exchange
      } else if (coupler.get_option<std::string>("bc_x2") == "open") {
        if (coupler.get_px() == coupler.get_nproc_x()-1) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,hs) ,
                                            KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            fields(l,hs+k,hs+j,hs+nx+ii) = fields(l,hs+k,hs+j,hs+nx-1);
          });
        }
      } else {
        std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_x2 can only be periodic or open";
        Kokkos::abort("");
      }

      if (coupler.get_option<std::string>("bc_y1") == "periodic") { // Already handled in halo_exchange
      } else if (coupler.get_option<std::string>("bc_y1") == "open") {
        if (coupler.get_py() == 0) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                            KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            fields(l,hs+k,      jj,hs+i) = fields(l,hs+k,hs+0   ,hs+i);
          });
        }
      } else {
        std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_y1 can only be periodic or open";
        Kokkos::abort("");
      }

      if (coupler.get_option<std::string>("bc_y2") == "periodic") { // Already handled in halo_exchange
      } else if (coupler.get_option<std::string>("bc_y2") == "open") {
        if (coupler.get_py() == coupler.get_nproc_y()-1) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                            KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            fields(l,hs+k,hs+ny+jj,hs+i) = fields(l,hs+k,hs+ny-1,hs+i);
          });
        }
      } else {
        std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_y2 can only be periodic or open";
        Kokkos::abort("");
      }

      if (coupler.get_option<std::string>("bc_z1") == "wall_free_slip") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,hs,ny,nx) ,
                                          KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          if (l == idW) {
            fields(l,kk,hs+j,hs+i) = 0;
          } else {
            fields(l,kk,hs+j,hs+i) = fields(l,hs+0,hs+j,hs+i);
          }
        });
      } else if (coupler.get_option<std::string>("bc_z1") == "periodic") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,hs,ny,nx) ,
                                          KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          fields(l,kk,hs+j,hs+i) = fields(l,nz+kk,hs+j,hs+i);
        });
      } else {
        std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_z1 can only be periodic or wall_free_slip";
        Kokkos::abort("");
      }

      if (coupler.get_option<std::string>("bc_z2") == "wall_free_slip") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,hs,ny,nx) ,
                                          KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          if (l == idW) {
            fields(l,hs+nz+kk,hs+j,hs+i) = 0;
          } else {
            fields(l,hs+nz+kk,hs+j,hs+i) = fields(l,hs+nz-1,hs+j,hs+i);
          }
        });
      } else if (coupler.get_option<std::string>("bc_z2") == "periodic") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,hs,ny,nx) ,
                                          KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          fields(l,hs+nz+kk,hs+j,hs+i) = fields(l,hs+kk,hs+j,hs+i);
        });
      } else {
        std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_z2 can only be periodic or wall_free_slip";
        Kokkos::abort("");
      }

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("halo_boundary_conditions");
      #endif
    }



    // Initialize the class data as well as the state and tracers arrays and convert them back into the coupler state
    void init(core::Coupler &coupler) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("init");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx             = coupler.get_nx();
      auto ny             = coupler.get_ny();
      auto nz             = coupler.get_nz();
      auto dz             = coupler.get_dz();
      auto nx_glob        = coupler.get_nx_glob();
      auto ny_glob        = coupler.get_ny_glob();
      auto num_tracers    = coupler.get_num_tracers();
      auto gamma          = coupler.get_option<real>("gamma_d");
      auto C0             = coupler.get_option<real>("C0"     );
      auto grav           = coupler.get_option<real>("grav"   );
      auto enable_gravity = coupler.get_option<bool>("enable_gravity",true);

      coupler.set_option<int>("dycore_hs",hs);

      num_tracers = coupler.get_num_tracers();
      bool1d tracer_adds_mass("tracer_adds_mass",num_tracers);
      bool1d tracer_positive ("tracer_positive" ,num_tracers);
      auto tracer_adds_mass_host = tracer_adds_mass.createHostCopy();
      auto tracer_positive_host  = tracer_positive .createHostCopy();
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) {
        std::string tracer_desc;
        bool        tracer_found, positive, adds_mass, diffuse;
        coupler.get_tracer_info( tracer_names.at(tr) , tracer_desc, tracer_found , positive , adds_mass , diffuse );
        tracer_positive_host (tr) = positive;
        tracer_adds_mass_host(tr) = adds_mass;
      }
      tracer_positive_host .deep_copy_to(tracer_positive );
      tracer_adds_mass_host.deep_copy_to(tracer_adds_mass);
      auto &dm = coupler.get_data_manager_readwrite();
      dm.register_and_allocate<bool>("tracer_adds_mass","",{num_tracers});
      auto dm_tracer_adds_mass = dm.get<bool,1>("tracer_adds_mass");
      tracer_adds_mass.deep_copy_to(dm_tracer_adds_mass);
      dm.register_and_allocate<bool>("tracer_positive","",{num_tracers});
      auto dm_tracer_positive = dm.get<bool,1>("tracer_positive");
      tracer_positive.deep_copy_to(dm_tracer_positive);

      real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);  state   = 0;
      real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);  tracers = 0;
      convert_coupler_to_dynamics( coupler , state , tracers );
      dm.register_and_allocate<float>("hy_dens_cells"    ,"",{nz+2*hs});
      dm.register_and_allocate<float>("hy_theta_cells"   ,"",{nz+2*hs});
      dm.register_and_allocate<float>("hy_pressure_cells","",{nz+2*hs});
      auto r = dm.get<float,1>("hy_dens_cells"    );    r = 0;
      auto t = dm.get<float,1>("hy_theta_cells"   );    t = 0;
      auto p = dm.get<float,1>("hy_pressure_cells");    p = 0;
      parallel_for( YAKL_AUTO_LABEL() , nz+2*hs , KOKKOS_LAMBDA (int k) {
        for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
            r(k) += state(idR,k,hs+j,hs+i);
            t(k) += state(idT,k,hs+j,hs+i) / state(idR,k,hs+j,hs+i);
            p(k) += C0 * std::pow( state(idT,k,hs+j,hs+i) , gamma );
          }
        }
      });
      coupler.get_parallel_comm().all_reduce( r , MPI_SUM ).deep_copy_to(r);
      coupler.get_parallel_comm().all_reduce( t , MPI_SUM ).deep_copy_to(t);
      coupler.get_parallel_comm().all_reduce( p , MPI_SUM ).deep_copy_to(p);
      real r_nx_ny = 1./(nx_glob*ny_glob);
      parallel_for( YAKL_AUTO_LABEL() , nz+2*hs , KOKKOS_LAMBDA (int k) {
        r(k) *= r_nx_ny;
        t(k) *= r_nx_ny;
        p(k) *= r_nx_ny;
      });
      parallel_for( YAKL_AUTO_LABEL() , hs , KOKKOS_LAMBDA (int kk) {
        {
          int  k0       = hs;
          int  k        = k0-1-kk;
          real rho0     = r(k0);
          real theta0   = t(k0);
          real rho0_gm1 = std::pow(rho0  ,gamma-1);
          real theta0_g = std::pow(theta0,gamma  );
          r(k) = std::pow( rho0_gm1 + grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) , 1._fp/(gamma-1) );
          t(k) = theta0;
          p(k) = C0*std::pow(r(k)*theta0,gamma);
        }
        {
          int  k0       = hs+nz-1;
          int  k        = k0+1+kk;
          real rho0     = r(k0);
          real theta0   = t(k0);
          real rho0_gm1 = std::pow(rho0  ,gamma-1);
          real theta0_g = std::pow(theta0,gamma  );
          r(k) = std::pow( rho0_gm1 - grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) , 1._fp/(gamma-1) );
          t(k) = theta0;
          p(k) = C0*std::pow(r(k)*theta0,gamma);
        }
      });

      auto create_immersed_proportion_halos = [] (core::Coupler &coupler) {
        using yakl::c::parallel_for;
        using yakl::c::SimpleBounds;;
        auto nz     = coupler.get_nz  ();
        auto ny     = coupler.get_ny  ();
        auto nx     = coupler.get_nx  ();
        auto &dm    = coupler.get_data_manager_readwrite();
        if (!dm.entry_exists("dycore_immersed_proportion_halos")) {
          auto immersed_prop = dm.get<real const,3>("immersed_proportion").createDeviceCopy<real>();
          if (dm.entry_exists("windmill_proj_weight")) {
            auto proj = dm.get<real const,3>("windmill_proj_weight");
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
              immersed_prop(k,j,i) += proj(k,j,i);
            });
          }
          core::MultiField<real,3> fields;
          fields.add_field( immersed_prop  );
          auto fields_halos = coupler.create_and_exchange_halos( fields , hs );
          dm.register_and_allocate<real>("dycore_immersed_proportion_halos","",{nz+2*hs,ny+2*hs,nx+2*hs},
                                         {"z_halod","y_halod","x_halod"});
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) , KOKKOS_LAMBDA (int kk, int j, int i) {
            fields_halos(0,      kk,j,i) = 1;
            fields_halos(0,hs+nz+kk,j,i) = 1;
          });
          fields_halos.get_field(0).deep_copy_to( dm.get<real,3>("dycore_immersed_proportion_halos") );

          {
            int hsnew = 2;
            dm.register_and_allocate<bool>("dycore_any_immersed2","",{nz,ny,nx},{"z","y","x"});
            auto any_immersed = dm.get<bool,3>("dycore_any_immersed2");
            auto fields_halos_larger = coupler.create_and_exchange_halos( fields , hsnew );
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hsnew,ny+2*hsnew,nx+2*hsnew) ,
                                              KOKKOS_LAMBDA (int kk, int j, int i) {
              fields_halos_larger(0,         kk,j,i) = 0;
              fields_halos_larger(0,hsnew+nz+kk,j,i) = 0;
            });
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
              any_immersed(k,j,i) = false;
              for (int kk=0; kk < hsnew*2+1; kk++) {
                for (int jj=0; jj < hsnew*2+1; jj++) {
                  for (int ii=0; ii < hsnew*2+1; ii++) {
                    if (fields_halos_larger(0,k+kk,j+jj,i+ii) > 0) any_immersed(k,j,i) = true;
                  }
                }
              }
            });
          }
          {
            int hsnew = 4;
            dm.register_and_allocate<bool>("dycore_any_immersed4","",{nz,ny,nx},{"z","y","x"});
            auto any_immersed = dm.get<bool,3>("dycore_any_immersed4");
            auto fields_halos_larger = coupler.create_and_exchange_halos( fields , hsnew );
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hsnew,ny+2*hsnew,nx+2*hsnew) ,
                                              KOKKOS_LAMBDA (int kk, int j, int i) {
              fields_halos_larger(0,         kk,j,i) = 0;
              fields_halos_larger(0,hsnew+nz+kk,j,i) = 0;
            });
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
              any_immersed(k,j,i) = false;
              for (int kk=0; kk < hsnew*2+1; kk++) {
                for (int jj=0; jj < hsnew*2+1; jj++) {
                  for (int ii=0; ii < hsnew*2+1; ii++) {
                    if (fields_halos_larger(0,k+kk,j+jj,i+ii) > 0) any_immersed(k,j,i) = true;
                  }
                }
              }
            });
          }
          {
            int hsnew = 6;
            dm.register_and_allocate<bool>("dycore_any_immersed6","",{nz,ny,nx},{"z","y","x"});
            auto any_immersed = dm.get<bool,3>("dycore_any_immersed6");
            auto fields_halos_larger = coupler.create_and_exchange_halos( fields , hsnew );
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hsnew,ny+2*hsnew,nx+2*hsnew) ,
                                              KOKKOS_LAMBDA (int kk, int j, int i) {
              fields_halos_larger(0,         kk,j,i) = 0;
              fields_halos_larger(0,hsnew+nz+kk,j,i) = 0;
            });
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
              any_immersed(k,j,i) = false;
              for (int kk=0; kk < hsnew*2+1; kk++) {
                for (int jj=0; jj < hsnew*2+1; jj++) {
                  for (int ii=0; ii < hsnew*2+1; ii++) {
                    if (fields_halos_larger(0,k+kk,j+jj,i+ii) > 0) any_immersed(k,j,i) = true;
                  }
                }
              }
            });
          }
          {
            int hsnew = 8;
            dm.register_and_allocate<bool>("dycore_any_immersed8","",{nz,ny,nx},{"z","y","x"});
            auto any_immersed = dm.get<bool,3>("dycore_any_immersed8");
            auto fields_halos_larger = coupler.create_and_exchange_halos( fields , hsnew );
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hsnew,ny+2*hsnew,nx+2*hsnew) ,
                                              KOKKOS_LAMBDA (int kk, int j, int i) {
              fields_halos_larger(0,         kk,j,i) = 0;
              fields_halos_larger(0,hsnew+nz+kk,j,i) = 0;
            });
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
              any_immersed(k,j,i) = false;
              for (int kk=0; kk < hsnew*2+1; kk++) {
                for (int jj=0; jj < hsnew*2+1; jj++) {
                  for (int ii=0; ii < hsnew*2+1; ii++) {
                    if (fields_halos_larger(0,k+kk,j+jj,i+ii) > 0) any_immersed(k,j,i) = true;
                  }
                }
              }
            });
          }
          {
            int hsnew = 10;
            dm.register_and_allocate<bool>("dycore_any_immersed10","",{nz,ny,nx},{"z","y","x"});
            auto any_immersed = dm.get<bool,3>("dycore_any_immersed10");
            auto fields_halos_larger = coupler.create_and_exchange_halos( fields , hsnew );
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hsnew,ny+2*hsnew,nx+2*hsnew) ,
                                              KOKKOS_LAMBDA (int kk, int j, int i) {
              fields_halos_larger(0,         kk,j,i) = 0;
              fields_halos_larger(0,hsnew+nz+kk,j,i) = 0;
            });
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
              any_immersed(k,j,i) = false;
              for (int kk=0; kk < hsnew*2+1; kk++) {
                for (int jj=0; jj < hsnew*2+1; jj++) {
                  for (int ii=0; ii < hsnew*2+1; ii++) {
                    if (fields_halos_larger(0,k+kk,j+jj,i+ii) > 0) any_immersed(k,j,i) = true;
                  }
                }
              }
            });
          }
        }
      };

      auto compute_hydrostasis_edges = [] (core::Coupler &coupler) {
        using yakl::c::parallel_for;
        using yakl::c::SimpleBounds;;
        auto nz   = coupler.get_nz  ();
        auto ny   = coupler.get_ny  ();
        auto nx   = coupler.get_nx  ();
        auto &dm  = coupler.get_data_manager_readwrite();
        if (! dm.entry_exists("hy_dens_edges"    )) dm.register_and_allocate<float>("hy_dens_edges"    ,"",{nz+1});
        if (! dm.entry_exists("hy_theta_edges"   )) dm.register_and_allocate<float>("hy_theta_edges"   ,"",{nz+1});
        if (! dm.entry_exists("hy_pressure_edges")) dm.register_and_allocate<float>("hy_pressure_edges","",{nz+1});
        auto hy_dens_cells     = dm.get<float const,1>("hy_dens_cells"    );
        auto hy_theta_cells    = dm.get<float const,1>("hy_theta_cells"   );
        auto hy_pressure_cells = dm.get<float const,1>("hy_pressure_cells");
        auto hy_dens_edges     = dm.get<float      ,1>("hy_dens_edges"    );
        auto hy_theta_edges    = dm.get<float      ,1>("hy_theta_edges"   );
        auto hy_pressure_edges = dm.get<float      ,1>("hy_pressure_edges");
        if (ord < 5) {
          parallel_for( YAKL_AUTO_LABEL() , nz+1 , KOKKOS_LAMBDA (int k) {
            hy_dens_edges    (k) = std::exp( 0.5_fp*std::log(hy_dens_cells(hs+k-1)) +
                                             0.5_fp*std::log(hy_dens_cells(hs+k  )) );
            hy_theta_edges   (k) =           0.5_fp*hy_theta_cells(hs+k-1) +
                                             0.5_fp*hy_theta_cells(hs+k  ) ;
            hy_pressure_edges(k) = std::exp( 0.5_fp*std::log(hy_pressure_cells(hs+k-1)) +
                                             0.5_fp*std::log(hy_pressure_cells(hs+k  )) );
          });
        } else {
          parallel_for( YAKL_AUTO_LABEL() , nz+1 , KOKKOS_LAMBDA (int k) {
            hy_dens_edges    (k) = std::exp( -1./12.*std::log(hy_dens_cells(hs+k-2)) +
                                              7./12.*std::log(hy_dens_cells(hs+k-1)) +
                                              7./12.*std::log(hy_dens_cells(hs+k  )) +
                                             -1./12.*std::log(hy_dens_cells(hs+k+1)) );
            hy_theta_edges   (k) =           -1./12.*hy_theta_cells(hs+k-2) +
                                              7./12.*hy_theta_cells(hs+k-1) +
                                              7./12.*hy_theta_cells(hs+k  ) +
                                             -1./12.*hy_theta_cells(hs+k+1);
            hy_pressure_edges(k) = std::exp( -1./12.*std::log(hy_pressure_cells(hs+k-2)) +
                                              7./12.*std::log(hy_pressure_cells(hs+k-1)) +
                                              7./12.*std::log(hy_pressure_cells(hs+k  )) +
                                             -1./12.*std::log(hy_pressure_cells(hs+k+1)) );
          });
        }
      };

      create_immersed_proportion_halos( coupler );
      compute_hydrostasis_edges       ( coupler );

      // These are needed for a proper restart
      coupler.register_output_variable<real>( "immersed_proportion" , core::Coupler::DIMS_3D      );
      coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        auto i_beg = coupler.get_i_beg();
        auto j_beg = coupler.get_j_beg();
        auto nz    = coupler.get_nz();
        auto ny    = coupler.get_ny();
        auto nx    = coupler.get_nx();
        nc.redef();
        nc.create_dim( "z_halo" , coupler.get_nz()+2*hs );
        nc.create_var<float>( "hy_dens_cells"     , {"z_halo"});
        nc.create_var<float>( "hy_theta_cells"    , {"z_halo"});
        nc.create_var<float>( "hy_pressure_cells" , {"z_halo"});
        nc.create_var<real>( "theta_pert"        , {"z","y","x"});
        nc.create_var<real>( "pressure_pert"     , {"z","y","x"});
        nc.create_var<real>( "density_pert"      , {"z","y","x"});
        nc.enddef();
        nc.begin_indep_data();
        auto &dm = coupler.get_data_manager_readonly();
        if (coupler.is_mainproc()) nc.write( dm.get<float const,1>("hy_dens_cells"    ) , "hy_dens_cells"     );
        if (coupler.is_mainproc()) nc.write( dm.get<float const,1>("hy_theta_cells"   ) , "hy_theta_cells"    );
        if (coupler.is_mainproc()) nc.write( dm.get<float const,1>("hy_pressure_cells") , "hy_pressure_cells" );
        nc.end_indep_data();
        real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
        real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
        convert_coupler_to_dynamics( coupler , state , tracers );
        std::vector<MPI_Offset> start_3d = {0,(MPI_Offset)j_beg,(MPI_Offset)i_beg};
        real3d data("data",nz,ny,nx);
        auto hy_dens_cells = dm.get<float const,1>("hy_dens_cells");
        yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
          data(k,j,i) = state(idR,hs+k,hs+j,hs+i) - hy_dens_cells(hs+k);
        });
        nc.write_all(data,"density_pert",start_3d);
        auto hy_theta_cells = dm.get<float const,1>("hy_theta_cells");
        yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
          data(k,j,i) = state(idT,hs+k,hs+j,hs+i) / state(idR,hs+k,hs+j,hs+i) - hy_theta_cells(hs+k);
        });
        nc.write_all(data,"theta_pert",start_3d);
        auto hy_pressure_cells = dm.get<float const,1>("hy_pressure_cells");
        yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
          data(k,j,i) = C0 * std::pow( state(idT,hs+k,hs+j,hs+i) , gamma ) - hy_pressure_cells(hs+k);
        });
        nc.write_all(data,"pressure_pert",start_3d);
      } );
      coupler.register_overwrite_with_restart_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        auto &dm = coupler.get_data_manager_readwrite();
        nc.read_all(dm.get<float,1>("hy_dens_cells"    ),"hy_dens_cells"    ,{0});
        nc.read_all(dm.get<float,1>("hy_theta_cells"   ),"hy_theta_cells"   ,{0});
        nc.read_all(dm.get<float,1>("hy_pressure_cells"),"hy_pressure_cells",{0});
        create_immersed_proportion_halos( coupler );
        compute_hydrostasis_edges       ( coupler );
      } );
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("init");
      #endif
    }



    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    void convert_dynamics_to_coupler( core::Coupler &coupler ,
                                      realConst4d    state   ,
                                      realConst4d    tracers ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("convert_dynamics_to_coupler");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto  nx          = coupler.get_nx();
      auto  ny          = coupler.get_ny();
      auto  nz          = coupler.get_nz();
      auto  R_d         = coupler.get_option<real>("R_d"    );
      auto  R_v         = coupler.get_option<real>("R_v"    );
      auto  gamma       = coupler.get_option<real>("gamma_d");
      auto  C0          = coupler.get_option<real>("C0"     );
      auto  idWV        = coupler.get_option<int >("idWV"   );
      auto  num_tracers = coupler.get_num_tracers();
      auto  &dm = coupler.get_data_manager_readwrite();
      auto  dm_rho_d = dm.get<real,3>("density_dry");
      auto  dm_uvel  = dm.get<real,3>("uvel"       );
      auto  dm_vvel  = dm.get<real,3>("vvel"       );
      auto  dm_wvel  = dm.get<real,3>("wvel"       );
      auto  dm_temp  = dm.get<real,3>("temp"       );
      auto  tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
      core::MultiField<real,3> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real,3>(tracer_names.at(tr)) ); }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real rho   = state(idR,hs+k,hs+j,hs+i);
        real u     = state(idU,hs+k,hs+j,hs+i) / rho;
        real v     = state(idV,hs+k,hs+j,hs+i) / rho;
        real w     = state(idW,hs+k,hs+j,hs+i) / rho;
        real theta = state(idT,hs+k,hs+j,hs+i) / rho;
        real press = C0 * pow( rho*theta , gamma );
        real rho_v = tracers(idWV,hs+k,hs+j,hs+i);
        real rho_d = rho;
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho_d -= tracers(tr,hs+k,hs+j,hs+i); }
        real temp = press / ( rho_d * R_d + rho_v * R_v );
        dm_rho_d(k,j,i) = rho_d;
        dm_uvel (k,j,i) = u;
        dm_vvel (k,j,i) = v;
        dm_wvel (k,j,i) = w;
        dm_temp (k,j,i) = temp;
        for (int tr=0; tr < num_tracers; tr++) { dm_tracers(tr,k,j,i) = tracers(tr,hs+k,hs+j,hs+i); }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("convert_dynamics_to_coupler");
      #endif
    }



    // Convert coupler's data to state and tracers arrays
    void convert_coupler_to_dynamics( core::Coupler const &coupler ,
                                      real4d              &state   ,
                                      real4d              &tracers ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("convert_coupler_to_dynamics");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto  nx          = coupler.get_nx();
      auto  ny          = coupler.get_ny();
      auto  nz          = coupler.get_nz();
      auto  R_d         = coupler.get_option<real>("R_d"    );
      auto  R_v         = coupler.get_option<real>("R_v"    );
      auto  gamma       = coupler.get_option<real>("gamma_d");
      auto  C0          = coupler.get_option<real>("C0"     );
      auto  idWV        = coupler.get_option<int >("idWV"   );
      auto  num_tracers = coupler.get_num_tracers();
      auto  &dm = coupler.get_data_manager_readonly();
      auto  dm_rho_d = dm.get<real const,3>("density_dry");
      auto  dm_uvel  = dm.get<real const,3>("uvel"       );
      auto  dm_vvel  = dm.get<real const,3>("vvel"       );
      auto  dm_wvel  = dm.get<real const,3>("wvel"       );
      auto  dm_temp  = dm.get<real const,3>("temp"       );
      auto  tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
      core::MultiField<real const,3> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real const,3>(tracer_names.at(tr)) ); }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real rho_d = dm_rho_d(k,j,i);
        real u     = dm_uvel (k,j,i);
        real v     = dm_vvel (k,j,i);
        real w     = dm_wvel (k,j,i);
        real temp  = dm_temp (k,j,i);
        real rho_v = dm_tracers(idWV,k,j,i);
        real press = rho_d * R_d * temp + rho_v * R_v * temp;
        real rho = rho_d;
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho += dm_tracers(tr,k,j,i); }
        real theta = pow( press/C0 , 1._fp / gamma ) / rho;
        state(idR,hs+k,hs+j,hs+i) = rho;
        state(idU,hs+k,hs+j,hs+i) = rho * u;
        state(idV,hs+k,hs+j,hs+i) = rho * v;
        state(idW,hs+k,hs+j,hs+i) = rho * w;
        state(idT,hs+k,hs+j,hs+i) = rho * theta;
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i) = dm_tracers(tr,k,j,i); }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("convert_coupler_to_dynamics");
      #endif
    }


  };

}




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
      size_t static constexpr ord = 9;
    #else
      size_t static constexpr ord = PORTURB_ORD;
    #endif
    int static constexpr hs  = (ord-1)/2+1; // Number of halo cells ("hs" == "halo size")
    int static constexpr num_state = 5;   // Number of state variables
    // IDs for the variables in the state vector
    int  static constexpr idR = 0;  // Density
    int  static constexpr idU = 1;  // u-momentum
    int  static constexpr idV = 2;  // v-momentum
    int  static constexpr idW = 3;  // w-momentum
    int  static constexpr idT = 4;  // Density * potential temperature


    template <size_t N>
    KOKKOS_INLINE_FUNCTION static void normalize( SArray<real,1,N> &s ) {
      real mn = s(0);
      real mx = s(0);
      for (int i=1; i < N; i++) {
        mn = std::min( mn , s(i) );
        mx = std::max( mx , s(i) );
      }
      real scale = 1;
      if (mx-mn > 1.e-10) scale = mx-mn;
      for (int i=0; i < N; i++) { s(i) = (s(i) - mn) / scale; }
    }



    real compute_time_step( core::Coupler const &coupler ) const {
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      real maxwave = 350 + coupler.get_option<real>( "dycore_max_wind" , 100 );
      real cfl = coupler.get_option<real>("cfl",0.60);
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
      // // SSPRK33
      // SArray<real,2,3,3> ssprk33_A;
      // SArray<real,1,3> ssprk33_b;
      // ssprk33_A = 0;
      // ssprk33_A(1,0) = 1;
      // ssprk33_A(2,0) = 1./4.;
      // ssprk33_A(2,1) = 1./4.;
      // ssprk33_b(0)   = 1./6.;
      // ssprk33_b(1)   = 1./6.;
      // ssprk33_b(2)   = 2./3.;
      for (int icycle = 0; icycle < ncycles; icycle++) {
        // time_step_generic_rk(coupler,A,b,state,tracers,dt_dyn);
        time_step_rk_3_3(coupler,state,tracers,dt_dyn);
      }
      convert_dynamics_to_coupler( coupler , state , tracers );
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step");
      #endif
    }



    template <size_t N>
    void time_step_generic_rk( core::Coupler            & coupler ,
                               SArray<real,2,N,N> const & A       ,
                               SArray<real,1,N>   const & b       ,
                               real4d             const & state   ,
                               real4d             const & tracers ,
                               real                       dt      ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      real5d state_rk_tend  ("state_rk_tend"  ,N,num_state  ,nz,ny,nx);
      real5d tracers_rk_tend("tracers_rk_tend",N,num_tracers,nz,ny,nx);
      real4d state_tmp  ("state_tmp"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d tracers_tmp("tracers_tmp",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
      enforce_immersed_boundaries( coupler , state , tracers );
      for (int is = 0; is < N; is++) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
          for (int l=0; l < num_state  ; l++) {
            state_tmp  (l,hs+k,hs+j,hs+i) = state  (l,hs+k,hs+j,hs+i);
            for (int l2=0; l2 <= is-1; l2++) { state_tmp  (l,hs+k,hs+j,hs+i) += dt*A(is,l2)*state_rk_tend  (l2,l,k,j,i); }
          }
          for (int l=0; l < num_tracers; l++) {
            tracers_tmp(l,hs+k,hs+j,hs+i) = tracers(l,hs+k,hs+j,hs+i);
            for (int l2=0; l2 <= is-1; l2++) { tracers_tmp(l,hs+k,hs+j,hs+i) += dt*A(is,l2)*tracers_rk_tend(l2,l,k,j,i); }
          }
        });
        auto state_tend   = state_rk_tend  .slice<4>(is,0,0,0,0);
        auto tracers_tend = tracers_rk_tend.slice<4>(is,0,0,0,0);
        enforce_immersed_boundaries( coupler , state_tmp , tracers_tmp );
        compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend);
      }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        for (int is = 0; is < N; is++) {
          for (int l=0; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i) += dt*b(is)*state_rk_tend  (is,l,k,j,i); }
          for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i) += dt*b(is)*tracers_rk_tend(is,l,k,j,i); }
        }
      });
      enforce_immersed_boundaries( coupler , state , tracers );
    }



    void time_step_sspab3( core::Coupler & coupler ,
                           real4d const  & state   ,
                           real4d const  & tracers ,
                           real            dt_dyn  ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step_sspab3");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto &dm         = coupler.get_data_manager_readwrite();
      auto tracer_positive = dm.get<bool const,1>("tracer_positive");

      if ( ! dm.entry_exists("state_tend_register") ) {
        dm.register_and_allocate<real>("state_register"  ,"",{3,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs});
        dm.register_and_allocate<real>("tracers_register","",{3,num_tracers,nz+2*hs,ny+2*hs,nx+2*hs});
        dm.register_and_allocate<real>("state_tend_register"  ,"",{3,num_state  ,nz,ny,nx});
        dm.register_and_allocate<real>("tracers_tend_register","",{3,num_tracers,nz,ny,nx});
        dm.get<real,5>("state_tend_register"  ) = 0;
        dm.get<real,5>("tracers_tend_register") = 0;
        auto state_register   = dm.get<real,5>("state_register"  );
        auto tracers_register = dm.get<real,5>("tracers_register");
        state  .deep_copy_to(state_register  .slice<4>(0,0,0,0,0));
        state  .deep_copy_to(state_register  .slice<4>(1,0,0,0,0));
        state  .deep_copy_to(state_register  .slice<4>(2,0,0,0,0));
        tracers.deep_copy_to(tracers_register.slice<4>(0,0,0,0,0));
        tracers.deep_copy_to(tracers_register.slice<4>(1,0,0,0,0));
        tracers.deep_copy_to(tracers_register.slice<4>(2,0,0,0,0));
        coupler.set_option<int>("ab3_current_index",2);
      }
      auto state_register        = dm.get<real,5>("state_register"       );
      auto tracers_register      = dm.get<real,5>("tracers_register"     );
      auto state_tend_register   = dm.get<real,5>("state_tend_register"  );
      auto tracers_tend_register = dm.get<real,5>("tracers_tend_register");
      auto ind_n   = coupler.get_option<int>("ab3_current_index");
      auto ind_nm1 = ind_n-1;   if (ind_nm1 < 0) ind_nm1 += 3;
      auto ind_nm2 = ind_n-2;   if (ind_nm2 < 0) ind_nm2 += 3;

      enforce_immersed_boundaries( coupler , state , tracers );

      auto state_tend_nm2   = state_tend_register  .slice<4>(ind_nm2,0,0,0,0);
      auto tracers_tend_nm2 = tracers_tend_register.slice<4>(ind_nm2,0,0,0,0);
      auto state_tend_nm1   = state_tend_register  .slice<4>(ind_nm1,0,0,0,0);
      auto tracers_tend_nm1 = tracers_tend_register.slice<4>(ind_nm1,0,0,0,0);
      auto state_tend_n     = state_tend_register  .slice<4>(ind_n  ,0,0,0,0);
      auto tracers_tend_n   = tracers_tend_register.slice<4>(ind_n  ,0,0,0,0);

      auto state_nm2   = state_register  .slice<4>(ind_nm2,0,0,0,0);
      auto tracers_nm2 = tracers_register.slice<4>(ind_nm2,0,0,0,0);
      auto state_nm1   = state_register  .slice<4>(ind_nm1,0,0,0,0);
      auto tracers_nm1 = tracers_register.slice<4>(ind_nm1,0,0,0,0);
      auto state_n     = state_register  .slice<4>(ind_n  ,0,0,0,0);
      auto tracers_n   = tracers_register.slice<4>(ind_n  ,0,0,0,0);
      state  .deep_copy_to(state_n  );
      tracers.deep_copy_to(tracers_n);

      compute_tendencies(coupler,state,state_tend_n,tracers,tracers_tend_n);
      
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state  (l,hs+k,hs+j,hs+i) =  1.908535476882378 * state_n  (l,hs+k,hs+j,hs+i)      +
                                      -1.334951446162515 * state_nm1(l,hs+k,hs+j,hs+i)      +
                                       0.426415969280137 * state_nm2(l,hs+k,hs+j,hs+i)      +
                                       1.502575553858997 * dt_dyn * state_tend_n  (l,k,j,i) +
                                      -1.654746338401493 * dt_dyn * state_tend_nm1(l,k,j,i) +
                                       0.670051276940255 * dt_dyn * state_tend_nm2(l,k,j,i);
        } else {
          l -= num_state;
          tracers(l,hs+k,hs+j,hs+i) =  1.908535476882378 * tracers_n  (l,hs+k,hs+j,hs+i)      +
                                      -1.334951446162515 * tracers_nm1(l,hs+k,hs+j,hs+i)      +
                                       0.426415969280137 * tracers_nm2(l,hs+k,hs+j,hs+i)      +
                                       1.502575553858997 * dt_dyn * tracers_tend_n  (l,k,j,i) +
                                      -1.654746338401493 * dt_dyn * tracers_tend_nm1(l,k,j,i) +
                                       0.670051276940255 * dt_dyn * tracers_tend_nm2(l,k,j,i);
          if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i) = std::max( 0._fp , tracers(l,hs+k,hs+j,hs+i) );
        }
      });

      ind_n++;   if (ind_n > 2) ind_n -= 3;
      coupler.set_option<int>("ab3_current_index",ind_n);

      enforce_immersed_boundaries( coupler , state , tracers );
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step_sspab3");
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
      compute_tendencies(coupler,state,state_tend,tracers,tracers_tend);
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
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend);
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
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend);
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


    int static constexpr idP = 5;

    void compute_tendencies( core::Coupler       & coupler      ,
                             real4d        const & state        ,
                             real4d        const & state_tend   ,
                             real4d        const & tracers      ,
                             real4d        const & tracers_tend ) const {
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
      auto  px                = coupler.get_px();    // grid spacing
      auto  py                = coupler.get_py();    // grid spacing
      auto  nproc_x           = coupler.get_nproc_x();    // grid spacing
      auto  nproc_y           = coupler.get_nproc_y();    // grid spacing
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

      SArray<float,2,ord,2> s2e;
      SArray<float,2,ord,ord> s2c;
      SArray<float,2,ord,2> c2e;
      TransformMatrices::sten_to_coefs(s2c);
      TransformMatrices::coefs_to_gll_lower(c2e);
      using yakl::intrinsics::matmul_cr;
      s2e = matmul_cr( c2e , s2c );

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
      halo_boundary_conditions( coupler , fields_loc , hs );

      typedef limiter::WenoLimiter<float,ord> Limiter;

      // Create arrays to hold cell interface interpolations
      float5d limits_x("limits_x",2,num_state+num_tracers+1,nz,ny,nx+1);
      float5d limits_y("limits_y",2,num_state+num_tracers+1,nz,ny+1,nx);
      float5d limits_z("limits_z",2,num_state+num_tracers+1,nz+1,ny,nx);

      bool wall_x1 = coupler.get_option<std::string>("bc_x1") == "wall_free_slip" && px == 0;;
      bool wall_x2 = coupler.get_option<std::string>("bc_x2") == "wall_free_slip" && px == nproc_x-1;
      bool wall_y1 = coupler.get_option<std::string>("bc_y1") == "wall_free_slip" && py == 0;
      bool wall_y2 = coupler.get_option<std::string>("bc_y2") == "wall_free_slip" && py == nproc_y-1;
      bool wall_z1 = coupler.get_option<std::string>("bc_z1") == "wall_free_slip";
      bool wall_z2 = coupler.get_option<std::string>("bc_z2") == "wall_free_slip";

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,nx+1) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        SArray<float,1,ord> stencil;
        SArray<bool ,1,ord> immersed;
        float dum;
        for (int ii=0; ii<ord; ii++) {
          immersed(ii) = immersed_prop(hs+k,hs+j,i+ii) > 0;
          stencil (ii) = fields_loc (l,hs+k,hs+j,i+ii);
        }
        if (l == idV || l == idW || l == idP) modify_stencil_immersed_der0( stencil , immersed );
        Limiter::compute_limited_edges( stencil , dum , limits_x(0,l,k,j,i) ,
                                        { !any_immersed(k,j,i) , immersed(hs-1) , immersed(hs+1) } );
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,nx+1) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        SArray<float,1,ord> stencil;
        SArray<bool ,1,ord> immersed;
        float dum;
        for (int ii=0; ii<ord; ii++) {
          immersed(ii) = immersed_prop(hs+k,hs+j,i+ii+1) > 0;
          stencil (ii) = fields_loc (l,hs+k,hs+j,i+ii+1);
        }
        if (l == idV || l == idW || l == idP) modify_stencil_immersed_der0( stencil , immersed );
        Limiter::compute_limited_edges( stencil , limits_x(1,l,k,j,i) , dum ,
                                        { !any_immersed(k,j,i) , immersed(hs-1) , immersed(hs+1) } );
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny+1,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        SArray<float,1,ord> stencil;
        SArray<bool ,1,ord> immersed;
        float dum;
        for (int jj=0; jj<ord; jj++) {
          immersed(jj) = immersed_prop(hs+k,j+jj,hs+i) > 0;
          stencil (jj) = fields_loc (l,hs+k,j+jj,hs+i);
        }
        if (l == idU || l == idW || l == idP) modify_stencil_immersed_der0( stencil , immersed );
        Limiter::compute_limited_edges( stencil , dum , limits_y(0,l,k,j,i) ,
                                        { !any_immersed(k,j,i) , immersed(hs-1) , immersed(hs+1) } );
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny+1,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        SArray<float,1,ord> stencil;
        SArray<bool ,1,ord> immersed;
        float dum;
        for (int jj=0; jj<ord; jj++) {
          immersed(jj) = immersed_prop(hs+k,j+jj+1,hs+i) > 0;
          stencil (jj) = fields_loc (l,hs+k,j+jj+1,hs+i);
        }
        if (l == idU || l == idW || l == idP) modify_stencil_immersed_der0( stencil , immersed );
        Limiter::compute_limited_edges( stencil , limits_y(1,l,k,j,i) , dum ,
                                        { !any_immersed(k,j,i) , immersed(hs-1) , immersed(hs+1) } );
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz+1,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        SArray<float,1,ord> stencil;
        SArray<bool ,1,ord> immersed;
        float dum;
        for (int kk=0; kk<ord; kk++) {
          immersed(kk) = immersed_prop(k+kk,hs+j,hs+i) > 0;
          stencil (kk) = fields_loc (l,k+kk,hs+j,hs+i);
        }
        if (l == idU || l == idV || l == idP) modify_stencil_immersed_der0( stencil , immersed );
        Limiter::compute_limited_edges( stencil , dum , limits_z(0,l,k,j,i) ,
                                        { !any_immersed(k,j,i) , immersed(hs-1) , immersed(hs+1) } );
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz+1,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        SArray<float,1,ord> stencil;
        SArray<bool ,1,ord> immersed;
        float dum;
        for (int kk=0; kk<ord; kk++) {
          immersed(kk) = immersed_prop(k+kk+1,hs+j,hs+i) > 0;
          stencil (kk) = fields_loc (l,k+kk+1,hs+j,hs+i);
        }
        if (l == idU || l == idV || l == idP) modify_stencil_immersed_der0( stencil , immersed );
        Limiter::compute_limited_edges( stencil , limits_z(1,l,k,j,i) , dum ,
                                        { !any_immersed(k,j,i) , immersed(hs-1) , immersed(hs+1) } );
      });

      // To save on space, slice the limits arrays to store single-valued interface fluxes
      using yakl::COLON;
      auto flux_x = limits_x.slice<4>(0,COLON,COLON,COLON,COLON);
      auto flux_y = limits_y.slice<4>(0,COLON,COLON,COLON,COLON);
      auto flux_z = limits_z.slice<4>(0,COLON,COLON,COLON,COLON);

      // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
      // Speed of sound and its reciprocal. Using a constant speed of sound for upwinding
      float constexpr cs   = 350.;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny+1,nx+1) , KOKKOS_LAMBDA (int k, int j, int i) {
        if (j < ny && k < nz) {
          if (wall_x1 && i == 0 ) limits_x(0,idU,k,j,i) = 0;
          if (wall_x1 && i == 0 ) limits_x(1,idU,k,j,i) = 0;
          if (wall_x2 && i == nx) limits_x(0,idU,k,j,i) = 0;
          if (wall_x2 && i == nx) limits_x(1,idU,k,j,i) = 0;
          limits_x(0,idR,k,j,i) += hy_dens_cells    (hs+k);
          limits_x(1,idR,k,j,i) += hy_dens_cells    (hs+k);
          limits_x(0,idT,k,j,i) += hy_theta_cells   (hs+k);
          limits_x(1,idT,k,j,i) += hy_theta_cells   (hs+k);
          // Acoustically upwind mass flux and pressure, linear constant Jacobian diagonalization
          float r_L  = limits_x(0,idR,k,j,i)    ;   float r_R  = limits_x(1,idR,k,j,i)    ;
          float ru_L = limits_x(0,idU,k,j,i)*r_L;   float ru_R = limits_x(1,idU,k,j,i)*r_R;
          float p_L  = limits_x(0,idP,k,j,i)    ;   float p_R  = limits_x(1,idP,k,j,i)    ;
          float p_upw  = 0.5_fp*(p_L  + p_R  - cs*(ru_R-ru_L)   );
          float ru_upw = 0.5_fp*(ru_L + ru_R -    (p_R -p_L )/cs);
          // Advectively upwind everything else
          int ind = ru_upw > 0 ? 0 : 1;
          flux_x(idR,k,j,i) = ru_upw;
          flux_x(idU,k,j,i) = ru_upw*limits_x(ind,idU,k,j,i) + p_upw;
          flux_x(idV,k,j,i) = ru_upw*limits_x(ind,idV,k,j,i);
          flux_x(idW,k,j,i) = ru_upw*limits_x(ind,idW,k,j,i);
          flux_x(idT,k,j,i) = ru_upw*limits_x(ind,idT,k,j,i);
          for (int tr=0; tr < num_tracers; tr++) {
            flux_x(num_state+tr,k,j,i) = ru_upw*limits_x(ind,num_state+1+tr,k,j,i);
          }
        }
        if (i < nx && k < nz && !sim2d) {
          if (wall_y1 && j == 0 ) limits_y(0,idV,k,j,i) = 0;
          if (wall_y1 && j == 0 ) limits_y(1,idV,k,j,i) = 0;
          if (wall_y2 && j == ny) limits_y(0,idV,k,j,i) = 0;
          if (wall_y2 && j == ny) limits_y(1,idV,k,j,i) = 0;
          limits_y(0,idR,k,j,i) += hy_dens_cells    (hs+k);
          limits_y(1,idR,k,j,i) += hy_dens_cells    (hs+k);
          limits_y(0,idT,k,j,i) += hy_theta_cells   (hs+k);
          limits_y(1,idT,k,j,i) += hy_theta_cells   (hs+k);
          // Acoustically upwind mass flux and pressure, linear constant Jacobian diagonalization
          float r_L  = limits_y(0,idR,k,j,i)    ;   float r_R  = limits_y(1,idR,k,j,i)    ;
          float rv_L = limits_y(0,idV,k,j,i)*r_L;   float rv_R = limits_y(1,idV,k,j,i)*r_R;
          float p_L  = limits_y(0,idP,k,j,i)    ;   float p_R  = limits_y(1,idP,k,j,i)    ;
          float p_upw  = 0.5_fp*(p_L  + p_R  - cs*(rv_R-rv_L)   );
          float rv_upw = 0.5_fp*(rv_L + rv_R -    (p_R -p_L )/cs);
          // Advectively upwind everything else
          int ind = rv_upw > 0 ? 0 : 1;
          flux_y(idR,k,j,i) = rv_upw;
          flux_y(idU,k,j,i) = rv_upw*limits_y(ind,idU,k,j,i);
          flux_y(idV,k,j,i) = rv_upw*limits_y(ind,idV,k,j,i) + p_upw;
          flux_y(idW,k,j,i) = rv_upw*limits_y(ind,idW,k,j,i);
          flux_y(idT,k,j,i) = rv_upw*limits_y(ind,idT,k,j,i);
          for (int tr=0; tr < num_tracers; tr++) {
            flux_y(num_state+tr,k,j,i) = rv_upw*limits_y(ind,num_state+1+tr,k,j,i);
          }
        }
        if (i < nx && j < ny) {
          if (wall_z1 && k == 0 ) limits_z(0,idW,k,j,i) = 0;
          if (wall_z1 && k == 0 ) limits_z(1,idW,k,j,i) = 0;
          if (wall_z2 && k == nz) limits_z(0,idW,k,j,i) = 0;
          if (wall_z2 && k == nz) limits_z(1,idW,k,j,i) = 0;
          limits_z(0,idR,k,j,i) += hy_dens_edges    (k);
          limits_z(1,idR,k,j,i) += hy_dens_edges    (k);
          limits_z(0,idT,k,j,i) += hy_theta_edges   (k);
          limits_z(1,idT,k,j,i) += hy_theta_edges   (k);
          // Acoustically upwind mass flux and pressure, linear constant Jacobian diagonalization
          float r_L  = limits_z(0,idR,k,j,i)    ;   float r_R  = limits_z(1,idR,k,j,i)    ;
          float rw_L = limits_z(0,idW,k,j,i)*r_L;   float rw_R = limits_z(1,idW,k,j,i)*r_R;
          float p_L  = limits_z(0,idP,k,j,i)    ;   float p_R  = limits_z(1,idP,k,j,i)    ;
          float p_upw  = 0.5_fp*(p_L  + p_R  - cs*(rw_R-rw_L)   );
          float rw_upw = 0.5_fp*(rw_L + rw_R -    (p_R -p_L )/cs);
          // Advectively upwind everything else
          int ind = rw_upw > 0 ? 0 : 1;
          flux_z(idR,k,j,i) = rw_upw;
          flux_z(idU,k,j,i) = rw_upw*limits_z(ind,idU,k,j,i);
          flux_z(idV,k,j,i) = rw_upw*limits_z(ind,idV,k,j,i);
          flux_z(idW,k,j,i) = rw_upw*limits_z(ind,idW,k,j,i) + p_upw;
          flux_z(idT,k,j,i) = rw_upw*limits_z(ind,idT,k,j,i);
          for (int tr=0; tr < num_tracers; tr++) {
            flux_z(num_state+tr,k,j,i) = rw_upw*limits_z(ind,num_state+1+tr,k,j,i);
          }
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
                                   float4d       const & fields  ,
                                   int                   hs      ) const {
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



    void edge_exchange( core::Coupler const & coupler  ,
                        float5d       const & limits_x ,
                        float5d       const & limits_y ,
                        float5d       const & limits_z ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("edge_exchange");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx             = coupler.get_nx();
      auto ny             = coupler.get_ny();
      auto nz             = coupler.get_nz();
      auto num_tracers    = coupler.get_num_tracers();
      auto &neigh         = coupler.get_neighbor_rankid_matrix();
      auto &dm            = coupler.get_data_manager_readonly();
      auto hy_dens_edges  = dm.get<float const,1>("hy_dens_edges" );
      auto hy_theta_edges = dm.get<float const,1>("hy_theta_edges");
      int npack = num_state + num_tracers + 1;

      // x-exchange
      #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
      yakl::timer_start("dycore_edge_exchange_x");
      #endif
      {
        float3d edge_send_buf_W("edge_send_buf_W",npack,nz,ny);
        float3d edge_send_buf_E("edge_send_buf_E",npack,nz,ny);
        float3d edge_recv_buf_W("edge_recv_buf_W",npack,nz,ny);
        float3d edge_recv_buf_E("edge_recv_buf_E",npack,nz,ny);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,ny) , KOKKOS_LAMBDA (int v, int k, int j) {
          edge_send_buf_W(v,k,j) = limits_x(1,v,k,j,0 );
          edge_send_buf_E(v,k,j) = limits_x(0,v,k,j,nx);
        });
        coupler.get_parallel_comm().send_receive<float,3>( { {edge_recv_buf_W,neigh(1,0),4} , {edge_recv_buf_E,neigh(1,2),5} } ,
                                                           { {edge_send_buf_W,neigh(1,0),5} , {edge_send_buf_E,neigh(1,2),4} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,ny) , KOKKOS_LAMBDA (int v, int k, int j) {
          limits_x(0,v,k,j,0 ) = edge_recv_buf_W(v,k,j);
          limits_x(1,v,k,j,nx) = edge_recv_buf_E(v,k,j);
        });
      }
      #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
      yakl::timer_stop("dycore_edge_exchange_x");
      yakl::timer_start("dycore_edge_exchange_y");
      #endif

      // y-direction exchange
      {
        float3d edge_send_buf_S("edge_send_buf_S",npack,nz,nx);
        float3d edge_send_buf_N("edge_send_buf_N",npack,nz,nx);
        float3d edge_recv_buf_S("edge_recv_buf_S",npack,nz,nx);
        float3d edge_recv_buf_N("edge_recv_buf_N",npack,nz,nx);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,nx) , KOKKOS_LAMBDA (int v, int k, int i) {
            edge_send_buf_S(v,k,i) = limits_y(1,v,k,0 ,i);
            edge_send_buf_N(v,k,i) = limits_y(0,v,k,ny,i);
        });
        coupler.get_parallel_comm().send_receive<float,3>( { {edge_recv_buf_S,neigh(0,1),6} , {edge_recv_buf_N,neigh(2,1),7} } ,
                                                           { {edge_send_buf_S,neigh(0,1),7} , {edge_send_buf_N,neigh(2,1),6} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,nx) , KOKKOS_LAMBDA (int v, int k, int i) {
          limits_y(0,v,k,0 ,i) = edge_recv_buf_S(v,k,i);
          limits_y(1,v,k,ny,i) = edge_recv_buf_N(v,k,i);
        });
      }
      #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
      yakl::timer_stop("dycore_edge_exchange_y");
      #endif

      if (coupler.get_option<std::string>("bc_x1") == "open" && coupler.get_px() == 0                      ) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_state+num_tracers+1,nz,ny) ,
                                          KOKKOS_LAMBDA (int l, int k, int j) {
          limits_x(0,l,k,j,0 ) = limits_x(1,l,k,j,0 );
        });
      }

      if (coupler.get_option<std::string>("bc_x2") == "open" && coupler.get_px() == coupler.get_nproc_x()-1) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_state+num_tracers+1,nz,ny) ,
                                          KOKKOS_LAMBDA (int l, int k, int j) {
          limits_x(1,l,k,j,nx) = limits_x(0,l,k,j,nx);
        });
      }

      if (coupler.get_option<std::string>("bc_y1") == "open" && coupler.get_py() == 0                      ) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_state+num_tracers+1,nz,nx) ,
                                          KOKKOS_LAMBDA (int l, int k, int i) {
          limits_y(0,l,k,0 ,i) = limits_y(1,l,k,0 ,i);
        });
      }

      if (coupler.get_option<std::string>("bc_y2") == "open" && coupler.get_py() == coupler.get_nproc_y()-1) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_state+num_tracers+1,nz,nx) ,
                                          KOKKOS_LAMBDA (int l, int k, int i) {
          limits_y(1,l,k,ny,i) = limits_y(0,l,k,ny,i);
        });
      }

      if (coupler.get_option<std::string>("bc_z1") == "wall_free_slip") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,ny,nx) , KOKKOS_LAMBDA (int l, int j, int i) {
          if (l == idW) {
            limits_z(0,l,0 ,j,i) = 0;
            limits_z(1,l,0 ,j,i) = 0;
          } else {
            limits_z(0,l,0 ,j,i) = limits_z(1,l,0 ,j,i);
          }
        });
      }

      if (coupler.get_option<std::string>("bc_z1") == "periodic") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,ny,nx) , KOKKOS_LAMBDA (int l, int j, int i) {
          limits_z(0,l,0 ,j,i) = limits_z(0,l,nz,j,i);
        });
      }

      if (coupler.get_option<std::string>("bc_z2") == "wall_free_slip") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,ny,nx) , KOKKOS_LAMBDA (int l, int j, int i) {
          if (l == idW || l == idT) {
            limits_z(0,l,nz,j,i) = 0;
            limits_z(1,l,nz,j,i) = 0;
          } else {
            limits_z(1,l,nz,j,i) = limits_z(0,l,nz,j,i);
          }
        });
      }

      if (coupler.get_option<std::string>("bc_z2") == "open") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,ny,nx) , KOKKOS_LAMBDA (int l, int j, int i) {
          limits_z(1,l,nz,j,i) = limits_z(0,l,nz,j,i);
        });
      }

      if (coupler.get_option<std::string>("bc_z2") == "periodic") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,ny,nx) , KOKKOS_LAMBDA (int l, int j, int i) {
          limits_z(1,l,nz,j,i) = limits_z(1,l,0 ,j,i);
        });
      }

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("edge_exchange");
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



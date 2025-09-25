
#pragma once

#include "main_header.h"
#include "coupler.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "WenoLimiter.h"
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
    int static constexpr hs  = (ord+1)/2; // Number of halo cells ("hs" == "halo size")
    int static constexpr num_state = 4;   // Number of state variables
    // IDs for the variables in the state vector
    int  static constexpr idR = 0;  // Density
    int  static constexpr idU = 1;  // u-momentum
    int  static constexpr idV = 2;  // v-momentum
    int  static constexpr idW = 3;  // w-momentum

    typedef float FLOC;



    real compute_mass( core::Coupler & coupler , real4d const & state ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      real3d r("r",nz,ny,nx);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j , int i) {
        r(k,j,i) = state(idR,k,j,i);
      });
      return coupler.get_parallel_comm().all_reduce( yakl::intrinsics::sum(r) , MPI_SUM );
    }



    real compute_time_step( core::Coupler const &coupler ) const {
      using yakl::intrinsics::minval;
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      auto cs = coupler.get_option<real>( "dycore_cs" , 350 );
      real maxwave = cs + coupler.get_option<real>( "dycore_max_wind" , 100 );
      real cfl = coupler.get_option<real>("cfl",0.70);
      return cfl * std::min( std::min( dx , dy ) , minval(dz) ) / maxwave;
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
    //   real csconst = coupler.get_option<real>( "dycore_cs" , -1 );
    //   parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
    //     real r = rho_d(k,j,i);
    //     real u = uvel (k,j,i);
    //     real v = vvel (k,j,i);
    //     real w = wvel (k,j,i);
    //     real T = temp (k,j,i);
    //     real p = r*R_d*T;
    //     real cs = csconst < 0 ? std::sqrt(gamma*p/r) : csconst;
    //     real dtx = cfl*dx   /(std::abs(u)+cs);
    //     real dty = cfl*dy   /(std::abs(v)+cs);
    //     real dtz = cfl*dz(k)/(std::abs(w)+cs);
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
      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      real4d state  ;
      real4d tracers;
      convert_coupler_to_dynamics( coupler , state , tracers );
      real dt_dyn = compute_time_step( coupler );
      int ncycles = (int) std::ceil( dt_phys / dt_dyn );
      dt_dyn = dt_phys / ncycles;
      // auto mass1 = compute_mass( coupler , state );
      for (int icycle = 0; icycle < ncycles; icycle++) { time_step_ssprk3(coupler,state,tracers,dt_dyn); }
      // auto mass2 = compute_mass( coupler , state );
      // if (coupler.is_mainproc()) std::cout << "Mass change: "
      //                                      << (std::get<0>(mass2)-std::get<0>(mass1))/std::get<0>(mass1)
      //                                      << std::endl;
      convert_dynamics_to_coupler( coupler , state , tracers );
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step");
      #endif
    }



    // Max CFL: 0.72
    void time_step_rk3( core::Coupler & coupler ,
                        real4d const  & state   ,
                        real4d const  & tracers ,
                        real            dt_dyn  ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step_rk_3_3");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers     = tracers.extent(0);
      auto nx              = coupler.get_nx();
      auto ny              = coupler.get_ny();
      auto nz              = coupler.get_nz();
      auto &dm             = coupler.get_data_manager_readonly();
      auto tracer_positive = dm.get<bool const,1>("dycore_tracer_positive");
      // SSPRK3 requires temporary arrays to hold intermediate state and tracers arrays
      real4d state_tmp   ("state_tmp"   ,num_state  ,nz,ny,nx);
      real4d tracers_tmp ("tracers_tmp" ,num_tracers,nz,ny,nx);
      // To hold tendencies
      real4d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx);
      real4d tracers_tend("tracers_tend",num_tracers,nz,ny,nx);

      enforce_immersed_boundaries( coupler , state , tracers );

      // Stage 1
      coupler.set_option<bool>("dycore_use_weno",false);
      compute_tendencies(coupler,state    ,state_tend,tracers    ,tracers_tend,dt_dyn/3);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = state  (l,k,j,i) + dt_dyn/3 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = tracers(l,k,j,i) + dt_dyn/3 * tracers_tend(l,k,j,i);
        }
      });

      // Stage 3
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/2);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = state  (l,k,j,i) + dt_dyn/2 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = tracers(l,k,j,i) + dt_dyn/2 * tracers_tend(l,k,j,i);
        }
      });

      // Stage 3
      coupler.set_option<bool>("dycore_use_weno",true);
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/1);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state      (l,k,j,i) = state  (l,k,j,i) + dt_dyn/1 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers    (l,k,j,i) = tracers(l,k,j,i) + dt_dyn/1 * tracers_tend(l,k,j,i);
          if (tracer_positive(l))  tracers(l,k,j,i) = std::max( 0._fp , tracers(l,k,j,i) );
        }
      });

      enforce_immersed_boundaries( coupler , state , tracers );

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step_rk_3_3");
      #endif
    }



    // Max CFL: 0.99
    void time_step_rk4( core::Coupler & coupler ,
                        real4d const  & state   ,
                        real4d const  & tracers ,
                        real            dt_dyn  ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step_rk_3_3");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers     = tracers.extent(0);
      auto nx              = coupler.get_nx();
      auto ny              = coupler.get_ny();
      auto nz              = coupler.get_nz();
      auto &dm             = coupler.get_data_manager_readonly();
      auto tracer_positive = dm.get<bool const,1>("dycore_tracer_positive");
      // SSPRK3 requires temporary arrays to hold intermediate state and tracers arrays
      real4d state_tmp   ("state_tmp"   ,num_state  ,nz,ny,nx);
      real4d tracers_tmp ("tracers_tmp" ,num_tracers,nz,ny,nx);
      // To hold tendencies
      real4d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx);
      real4d tracers_tend("tracers_tend",num_tracers,nz,ny,nx);

      enforce_immersed_boundaries( coupler , state , tracers );

      // Stage 1
      coupler.set_option<bool>("dycore_use_weno",false);
      compute_tendencies(coupler,state    ,state_tend,tracers    ,tracers_tend,dt_dyn/4);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = state  (l,k,j,i) + dt_dyn/4 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = tracers(l,k,j,i) + dt_dyn/4 * tracers_tend(l,k,j,i);
        }
      });

      // Stage 2
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/3);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = state  (l,k,j,i) + dt_dyn/3 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = tracers(l,k,j,i) + dt_dyn/3 * tracers_tend(l,k,j,i);
        }
      });

      // Stage 3
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/2);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = state  (l,k,j,i) + dt_dyn/2 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = tracers(l,k,j,i) + dt_dyn/2 * tracers_tend(l,k,j,i);
        }
      });

      // Stage 4
      coupler.set_option<bool>("dycore_use_weno",true);
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/1);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state      (l,k,j,i) = state  (l,k,j,i) + dt_dyn/1 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers    (l,k,j,i) = tracers(l,k,j,i) + dt_dyn/1 * tracers_tend(l,k,j,i);
          if (tracer_positive(l))  tracers(l,k,j,i) = std::max( 0._fp , tracers(l,k,j,i) );
        }
      });

      enforce_immersed_boundaries( coupler , state , tracers );

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step_rk_3_3");
      #endif
    }



    // Max CFL: 0.72
    void time_step_ssprk3( core::Coupler & coupler ,
                           real4d const  & state   ,
                           real4d const  & tracers ,
                           real            dt_dyn  ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step_rk_3_3");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers     = tracers.extent(0);
      auto nx              = coupler.get_nx();
      auto ny              = coupler.get_ny();
      auto nz              = coupler.get_nz();
      auto &dm             = coupler.get_data_manager_readonly();
      auto tracer_positive = dm.get<bool const,1>("dycore_tracer_positive");
      // SSPRK3 requires temporary arrays to hold intermediate state and tracers arrays
      real4d state_tmp   ("state_tmp"   ,num_state  ,nz,ny,nx);
      real4d tracers_tmp ("tracers_tmp" ,num_tracers,nz,ny,nx);
      // To hold tendencies
      real4d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx);
      real4d tracers_tend("tracers_tend",num_tracers,nz,ny,nx);

      enforce_immersed_boundaries( coupler , state , tracers );

      //////////////
      // Stage 1
      //////////////
      compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = state  (l,k,j,i) + dt_dyn * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = tracers(l,k,j,i) + dt_dyn * tracers_tend(l,k,j,i);
        }
      });

      //////////////
      // Stage 2
      //////////////
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/4.);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = (3._fp/4._fp) * state      (l,k,j,i) + 
                                 (1._fp/4._fp) * state_tmp  (l,k,j,i) +
                                 (1._fp/4._fp) * dt_dyn * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = (3._fp/4._fp) * tracers    (l,k,j,i) + 
                                 (1._fp/4._fp) * tracers_tmp(l,k,j,i) +
                                 (1._fp/4._fp) * dt_dyn * tracers_tend(l,k,j,i);
        }
      });

      //////////////
      // Stage 3
      //////////////
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn*2./3.);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state  (l,k,j,i) = (1._fp/3._fp) * state      (l,k,j,i) +
                             (2._fp/3._fp) * state_tmp  (l,k,j,i) +
                             (2._fp/3._fp) * dt_dyn * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers(l,k,j,i) = (1._fp/3._fp) * tracers    (l,k,j,i) +
                             (2._fp/3._fp) * tracers_tmp(l,k,j,i) +
                             (2._fp/3._fp) * dt_dyn * tracers_tend(l,k,j,i);
          // Ensure positive tracers stay positive
          if (tracer_positive(l))  tracers(l,k,j,i) = std::max( 0._fp , tracers(l,k,j,i) );
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
      auto num_tracers       = tracers.extent(0);
      auto nx                = coupler.get_nx();
      auto ny                = coupler.get_ny();
      auto nz                = coupler.get_nz();
      auto immersed_power    = coupler.get_option<real>("immersed_power",5);
      auto &dm               = coupler.get_data_manager_readonly();
      auto hy_dens_cells     = dm.get<real const,1>("hy_dens_cells" ); // Hydrostatic density
      auto hy_theta_cells    = dm.get<real const,1>("hy_theta_cells"); // Hydrostatic potential temperature
      auto immersed_prop     = dm.get<real const,3>("dycore_immersed_proportion_halos"); // Immersed Proportion
      auto tracer_positive   = dm.get<bool const,1>("dycore_tracer_positive");

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real mult = std::pow( immersed_prop(hs+k,hs+j,hs+i) , immersed_power );
        // TODO: Find a way to calculate drag in here
        // Density
        {
          auto &var = state(idR,k,j,i);
          real  target = hy_dens_cells(hs+k);
          var = var + (target - var)*mult;
        }
        // u-momentum
        {
          auto &var = state(idU,k,j,i);
          real  target = 0;
          var = var + (target - var)*mult;
        }
        // v-momentum
        {
          auto &var = state(idV,k,j,i);
          real  target = 0;
          var = var + (target - var)*mult;
        }
        // w-momentum
        {
          auto &var = state(idW,k,j,i);
          real  target = 0;
          var = var + (target - var)*mult;
        }
        // Normal tracers
        for (int tr=0; tr < num_tracers-1; tr++) {
          auto &var = tracers(tr,k,j,i);
          real  target = 0;
          var = var + (target - var)*mult;
          if (tracer_positive(tr))  var = std::max( 0._fp , var );
        }
        // Density*Theta
        {
          auto &var = tracers(num_tracers-1,k,j,i);
          real  target = hy_dens_cells(hs+k)*hy_theta_cells(hs+k);
          var = var + (target - var)*mult;
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



    real static KOKKOS_INLINE_FUNCTION hypervis( SArray<FLOC,1,ord> const &s) {
      if      constexpr (ord == 3 ) { return  ( 1.00000000f*s(0)-2.00000000f*s(1)+1.00000000f*s(2) )/4; }
      else if constexpr (ord == 5 ) { return -( 1.00000000f*s(0)-4.00000000f*s(1)+6.00000000f*s(2)-4.00000000f*s(3)+1.00000000f*s(4) )/16; }
      else if constexpr (ord == 7 ) { return  ( 1.00000000f*s(0)-6.00000000f*s(1)+15.0000000f*s(2)-20.0000000f*s(3)+15.0000000f*s(4)-6.00000000f*s(5)+1.00000000f*s(6) )/64; }
      else if constexpr (ord == 9 ) { return -( 1.00000000f*s(0)-8.00000000f*s(1)+28.0000000f*s(2)-56.0000000f*s(3)+70.0000000f*s(4)-56.0000000f*s(5)+28.0000000f*s(6)-8.00000000f*s(7)+1.00000000f*s(8) )/256; }
      else if constexpr (ord == 11) { return  ( 1.00000000f*s(0)-10.0000000f*s(1)+1.00000000f*s(10)+45.0000000f*s(2)-120.000000f*s(3)+210.000000f*s(4)-252.000000f*s(5)+210.000000f*s(6)-120.000000f*s(7)+45.0000000f*s(8)-10.0000000f*s(9) )/1024; }
    }



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
      auto nx             = coupler.get_nx();    // Proces-local number of cells
      auto ny             = coupler.get_ny();    // Proces-local number of cells
      auto nz             = coupler.get_nz();    // Total vertical cells
      auto dx             = coupler.get_dx();    // grid spacing
      auto dy             = coupler.get_dy();    // grid spacing
      auto dz             = coupler.get_dz();    // grid spacing
      auto num_tracers    = tracers.extent(0);   // Number of tracers
      auto enable_gravity = coupler.get_option<bool>("enable_gravity",true);
      auto grav           = coupler.get_option<real>("grav"   );    // Gravity
      auto latitude       = coupler.get_option<real>("latitude",0); // For coriolis
      auto &dm            = coupler.get_data_manager_readonly();    // Grab read-only data manager
      auto immersed_prop  = dm.get<real const,3>("dycore_immersed_proportion_halos"); // Immersed Proportion
      auto any_immersed2  = dm.get<bool const,3>("dycore_any_immersed2" ); // Are any immersed in 3-D halo?
      auto any_immersed4  = dm.get<bool const,3>("dycore_any_immersed4" ); // Are any immersed in 3-D halo?
      auto any_immersed6  = dm.get<bool const,3>("dycore_any_immersed6" ); // Are any immersed in 3-D halo?
      auto any_immersed8  = dm.get<bool const,3>("dycore_any_immersed8" ); // Are any immersed in 3-D halo?
      auto any_immersed10 = dm.get<bool const,3>("dycore_any_immersed10"); // Are any immersed in 3-D halo?
      auto hy_dens_cells  = dm.get<real const,1>("hy_dens_cells"        ); // Hydrostatic density
      auto hy_theta_cells = dm.get<real const,1>("hy_theta_cells"       ); // Hydrostatic potential temperature
      auto hy_theta_edges = dm.get<real const,1>("hy_theta_edges"       ); // Hydrostatic potential temperature
      auto metjac_edges   = dm.get<real const,2>("dycore_metjac_edges"  ); // Vertical metric jacobian at edges
      // Compute matrices to convert polynomial coefficients to 2 GLL points and stencil values to 2 GLL points
      // These matrices will be in column-row format. That performed better than row-column format in performance tests
      real r_dx = 1./dx; // reciprocal of grid spacing
      real r_dy = 1./dy; // reciprocal of grid spacing
      real fcor = 2*7.2921e-5*std::sin(latitude/180*M_PI);      // For coriolis: 2*Omega*sin(latitude)

      int constexpr hsm1 = hs-1;

      SArray<FLOC,1,ord> wt;
      if      constexpr (ord == 3 ) {
        wt(0) =  0.333333333f;
        wt(1) = +0.833333333f;
        wt(2) = -0.166666667f;
      } else if constexpr (ord == 5 ) {
        wt(0) = -0.0500000000f;
        wt(1) = +0.450000000f;
        wt(2) = +0.783333333f;
        wt(3) = -0.216666667f;
        wt(4) = +0.0333333333f;
      } else if constexpr (ord == 7 ) {
        wt(0 ) =  0.00952380952f;
        wt(1 ) = -0.0904761905f;
        wt(2 ) = +0.509523810f;
        wt(3 ) = +0.759523810f;
        wt(4 ) = -0.240476190f;
        wt(5 ) = +0.0595238095f;
        wt(6 ) = -0.00714285714f;
      } else if constexpr (ord == 9 ) {
        wt(0 ) = -0.00198412698f;
        wt(1 ) = +0.0218253968f;
        wt(2 ) = -0.121031746f;
        wt(3 ) = +0.545634921f;
        wt(4 ) = +0.745634921f;
        wt(5 ) = -0.254365079f;
        wt(6 ) = +0.0789682540f;
        wt(7 ) = -0.0162698413f;
        wt(8 ) = +0.00158730159f;
      } else if constexpr (ord == 11) {
        wt(0 ) =  0.000432900433f;
        wt(1 ) = -0.00551948052f;
        wt(2 ) = +0.0341630592f;
        wt(3 ) = -0.144408369f;
        wt(4 ) = +0.569877345f;
        wt(5 ) = +0.736544012f;
        wt(6 ) = -0.263455988f;
        wt(7 ) = +0.0936868687f;
        wt(8 ) = -0.0253607504f;
        wt(9 ) = +0.00440115440f;
        wt(10) = -0.000360750361f;
      }

      int num_fields = num_state + num_tracers;

      yakl::Array<FLOC,4> fields_loc("fields_loc",num_fields,nz+2*hs,ny+2*hs,nx+2*hs);

      // Compute pressure
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real r_r = 1._fp / state(idR,k,j,i);
        fields_loc(idR,hs+k,hs+j,hs+i) = state(idR,k,j,i);
        fields_loc(idU,hs+k,hs+j,hs+i) = state(idU,k,j,i) * r_r;
        fields_loc(idV,hs+k,hs+j,hs+i) = state(idV,k,j,i) * r_r;
        fields_loc(idW,hs+k,hs+j,hs+i) = state(idW,k,j,i) * r_r;
        for (int l=0; l < num_tracers; l++) { fields_loc(num_state+l,hs+k,hs+j,hs+i) = tracers(l,k,j,i)*r_r; }
        fields_loc(idR         ,hs+k,hs+j,hs+i) -= hy_dens_cells (hs+k);
        fields_loc(num_fields-1,hs+k,hs+j,hs+i) -= hy_theta_cells(hs+k);
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

      yakl::Array<FLOC,4> flux_x("flux_x",num_fields,nz,ny,nx+1);
      yakl::Array<FLOC,4> flux_y("flux_y",num_fields,nz,ny+1,nx);
      yakl::Array<FLOC,4> flux_z("flux_z",num_fields,nz+1,ny,nx);

      yakl::Array<FLOC,3> p_x("p_x",nz,ny,nx+1);
      yakl::Array<FLOC,3> p_y("p_y",nz,ny+1,nx);
      yakl::Array<FLOC,3> p_z("p_z",nz+1,ny,nx);

      yakl::Array<FLOC,3> ru_x("ru_x",nz,ny,nx+1);
      yakl::Array<FLOC,3> rv_y("rv_y",nz,ny+1,nx);
      yakl::Array<FLOC,3> rw_z("rw_z",nz+1,ny,nx);

      auto wall_z1 = coupler.get_option<std::string>("bc_z1") == "wall_free_slip";
      auto wall_z2 = coupler.get_option<std::string>("bc_z2") == "wall_free_slip";
      typedef limiter::WenoLimiter<FLOC,ord> Limiter;
      auto use_weno = coupler.get_option<bool>("dycore_use_weno",true);

      FLOC cs = coupler.get_option<real>("dycore_cs",350);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx+1) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool ,1,ord> immersed;
        SArray<FLOC,1,ord> s;
        for (int ii = 0; ii < ord; ii++) { immersed(ii) = immersed_prop (hs+k,hs+j,i+ii) > 0; }
        for (int ii = 0; ii < ord; ii++) { s       (ii) = cs*cs*fields_loc(idR,hs+k,hs+j,i+ii); }
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_L, dummy;
        Limiter::compute_limited_edges( s , dummy , p_L , { false , false , false } );
        for (int ii = 0; ii < ord; ii++) { s       (ii) = (fields_loc(idR,hs+k,hs+j,i+ii)+hy_dens_cells(hs+k))*
                                                          fields_loc (idU,hs+k,hs+j,i+ii); }
        FLOC ru_L = 0;
        for (int ii=0; ii < ord; ii++) { ru_L += wt(ord-1-ii)*s(ii); }
        for (int ii = 0; ii < ord; ii++) { immersed(ii) = immersed_prop (hs+k,hs+j,i+ii+1) > 0; }
        for (int ii = 0; ii < ord; ii++) { s       (ii) = cs*cs*fields_loc(idR,hs+k,hs+j,i+ii+1); }
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_R;
        Limiter::compute_limited_edges( s , p_R , dummy , { false , false , false } );
        for (int ii = 0; ii < ord; ii++) { s       (ii) = (fields_loc(idR,hs+k,hs+j,i+ii+1)+hy_dens_cells(hs+k))*
                                                          fields_loc (idU,hs+k,hs+j,i+ii+1); }
        FLOC ru_R = 0;
        for (int ii=0; ii < ord; ii++) { ru_R += wt(ii)*s(ii); }
        p_x (k,j,i) = 0.5f*(p_L  + p_R  - cs*(ru_R-ru_L)   );
        ru_x(k,j,i) = 0.5f*(ru_L + ru_R -    (p_R -p_L )/cs);
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny+1,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool ,1,ord> immersed;
        SArray<FLOC,1,ord> s;
        for (int jj = 0; jj < ord; jj++) { immersed(jj) = immersed_prop (hs+k,j+jj,hs+i) > 0; }
        for (int jj = 0; jj < ord; jj++) { s       (jj) = cs*cs*fields_loc(idR,hs+k,j+jj,hs+i); }
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_L, dummy;
        Limiter::compute_limited_edges( s , dummy , p_L , { false , false , false } );
        for (int jj = 0; jj < ord; jj++) { s       (jj) = (fields_loc(idR,hs+k,j+jj,hs+i)+hy_dens_cells(hs+k))*
                                                          fields_loc (idV,hs+k,j+jj,hs+i); }
        FLOC rv_L = 0;
        for (int jj=0; jj < ord; jj++) { rv_L += wt(ord-1-jj)*s(jj); }
        for (int jj = 0; jj < ord; jj++) { immersed(jj) = immersed_prop (hs+k,j+jj+1,hs+i) > 0; }
        for (int jj = 0; jj < ord; jj++) { s       (jj) = cs*cs*fields_loc(idR,hs+k,j+jj+1,hs+i); }
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_R;
        Limiter::compute_limited_edges( s , p_R , dummy , { false , false , false } );
        for (int jj = 0; jj < ord; jj++) { s       (jj) = (fields_loc(idR,hs+k,j+jj+1,hs+i)+hy_dens_cells(hs+k))*
                                                          fields_loc (idV,hs+k,j+jj+1,hs+i); }
        FLOC rv_R = 0;
        for (int jj=0; jj < ord; jj++) { rv_R += wt(jj)*s(jj); }
        p_y (k,j,i) = 0.5f*(p_L  + p_R  - cs*(rv_R-rv_L)   );
        rv_y(k,j,i) = 0.5f*(rv_L + rv_R -    (p_R -p_L )/cs);
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool ,1,ord> immersed;
        SArray<FLOC,1,ord> s;
        for (int kk = 0; kk < ord; kk++) { immersed(kk) = immersed_prop (k+kk,hs+j,hs+i) > 0; }
        for (int kk = 0; kk < ord; kk++) { s       (kk) = cs*cs*fields_loc(idR,k+kk,hs+j,hs+i); }
        for (int kk = 0; kk < ord; kk++) { s       (kk) *= dz(std::max(0,std::min(nz-1,k-hsm1-1+kk)))/dz(std::max(0,k-1)); }
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_L, dummy;
        Limiter::compute_limited_edges( s , dummy , p_L , { false , false , false } );
        p_L /= metjac_edges(1+k-1,1);
        for (int kk = 0; kk < ord; kk++) { s       (kk) = (fields_loc(idR,k+kk,hs+j,hs+i)+hy_dens_cells(k+kk))*
                                                          fields_loc (idW,k+kk,hs+j,hs+i); }
        for (int kk = 0; kk < ord; kk++) { s       (kk) *= dz(std::max(0,std::min(nz-1,k-hsm1-1+kk)))/dz(std::max(0,k-1)); }
        FLOC rw_L = 0;
        for (int kk=0; kk < ord; kk++) { rw_L += wt(ord-1-kk)*s(kk); }
        rw_L /= metjac_edges(1+k-1,1);
        if (wall_z1 && k == 0 ) rw_L = 0;
        if (wall_z2 && k == nz) rw_L = 0;
        for (int kk = 0; kk < ord; kk++) { immersed(kk) = immersed_prop (k+kk+1,hs+j,hs+i) > 0; }
        for (int kk = 0; kk < ord; kk++) { s       (kk) = cs*cs*fields_loc(idR,k+kk+1,hs+j,hs+i); }
        for (int kk = 0; kk < ord; kk++) { s       (kk) *= dz(std::max(0,std::min(nz-1,k-hsm1+kk)))/dz(std::min(nz-1,k)); }
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_R;
        Limiter::compute_limited_edges( s , p_R , dummy , { false , false , false } );
        p_R /= metjac_edges(1+k,0);
        for (int kk = 0; kk < ord; kk++) { s       (kk) = (fields_loc(idR,k+kk+1,hs+j,hs+i)+hy_dens_cells(k+kk+1))*
                                                          fields_loc (idW,k+kk+1,hs+j,hs+i); }
        for (int kk = 0; kk < ord; kk++) { s       (kk) *= dz(std::max(0,std::min(nz-1,k-hsm1+kk)))/dz(std::min(nz-1,k)); }
        FLOC rw_R = 0;
        for (int kk=0; kk < ord; kk++) { rw_R += wt(kk)*s(kk); }
        rw_R /= metjac_edges(1+k,0);
        if (wall_z1 && k == 0 ) rw_R = 0;
        if (wall_z2 && k == nz) rw_R = 0;
        p_z (k,j,i) = 0.5f*(p_L  + p_R  - cs*(rw_R-rw_L)   );
        rw_z(k,j,i) = 0.5f*(rw_L + rw_R -    (p_R -p_L )/cs);
        if (wall_z1 && k == 0 ) rw_z(k,j,i) = 0;
        if (wall_z2 && k == nz) rw_z(k,j,i) = 0;
      });

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx+1) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool ,1,ord> immersed;
        FLOC ru = ru_x(k,j,i);
        int ind = ru > 0 ? 0 : 1;
        for (int ii = 0; ii < ord; ii++) { immersed(ii) = immersed_prop(hs+k,hs+j,i+ii+ind) > 0; }
        for (int l=0; l < num_fields; l++) {
          if (l != idR) {
            SArray<FLOC,1,ord> s;
            for (int ii = 0; ii < ord; ii++) { s(ii) = fields_loc(l,hs+k,hs+j,i+ii+ind); }
            if (l == idV || l == idW) modify_stencil_immersed_der0( s , immersed );
            FLOC val;
            if (!use_weno) {
              val = 0;
              for (int ii=0; ii < ord; ii++) { val += wt(ru>0?ord-1-ii:ii)*s(ii); }
            } else {
              FLOC val_L, val_R;
              Limiter::compute_limited_edges( s , val_L , val_R , { true , immersed(hsm1-1) , immersed(hsm1+1) } );
              val = ru > 0 ? val_R : val_L;
            }
            if (l == num_fields-1) val += hy_theta_cells(hs+k);
            flux_x(l,k,j,i) = ru*val;
          }
        }
        flux_x(idR,k,j,i)  = ru;
        flux_x(idU,k,j,i) += p_x(k,j,i);
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny+1,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool ,1,ord> immersed;
        FLOC rv = rv_y(k,j,i);
        int ind = rv > 0 ? 0 : 1;
        for (int jj = 0; jj < ord; jj++) { immersed(jj) = immersed_prop(hs+k,j+jj+ind,hs+i) > 0; }
        for (int l=0; l < num_fields; l++) {
          if (l != idR) {
            SArray<FLOC,1,ord> s;
            for (int jj = 0; jj < ord; jj++) { s(jj) = fields_loc(l,hs+k,j+jj+ind,hs+i); }
            if (l == idU || l == idW) modify_stencil_immersed_der0( s , immersed );
            FLOC val;
            if (!use_weno) {
              val = 0;
              for (int jj=0; jj < ord; jj++) { val += wt(rv>0?ord-1-jj:jj)*s(jj); }
            } else {
              FLOC val_L, val_R;
              Limiter::compute_limited_edges( s , val_L , val_R , { true , immersed(hsm1-1) , immersed(hsm1+1) } );
              val = rv > 0 ? val_R : val_L;
            }
            if (l == num_fields-1) val += hy_theta_cells(hs+k);
            flux_y(l,k,j,i) = rv*val;
          }
        }
        flux_y(idR,k,j,i)  = rv;
        flux_y(idV,k,j,i) += p_y(k,j,i);
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool ,1,ord> immersed;
        FLOC rw = rw_z(k,j,i);
        int ind = rw > 0 ? 0 : 1;
        for (int kk = 0; kk < ord; kk++) { immersed(kk) = immersed_prop(k+kk+ind,hs+j,hs+i) > 0; }
        for (int l=1; l < num_fields; l++) {
          if (l != idR) {
            SArray<FLOC,1,ord> s;
            for (int kk = 0; kk < ord; kk++) { s(kk) = fields_loc(l,k+kk+ind,hs+j,hs+i); }
            if (l == idU || l == idV) modify_stencil_immersed_der0( s , immersed );
            for (int kk = 0; kk < ord; kk++) { s(kk) *= dz(std::max(0,std::min(nz-1,k-hs+ind+kk)))/
                                                        dz(std::max(0,std::min(nz-1,k-1 +ind   ))); }
            FLOC val;
            if (!use_weno) {
              val = 0;
              for (int kk=0; kk < ord; kk++) { val += wt(rw>0?ord-1-kk:kk)*s(kk); }
            } else {
              FLOC val_L, val_R;
              Limiter::compute_limited_edges( s , val_L , val_R , { true , immersed(hsm1-1) , immersed(hsm1+1) } );
              val = rw > 0 ? val_R : val_L;
            }
            val /= rw > 0 ? metjac_edges(1+k-1,1) : metjac_edges(1+k,0);
            if (l == num_fields-1) val += hy_theta_edges(k);
            flux_z(l,k,j,i) = rw*val;
          }
        }
        flux_z(idR,k,j,i)  = rw;
        flux_z(idW,k,j,i) += p_z(k,j,i);
      });

      // Compute tendencies as the flux divergence + gravity source term + coriolis
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tend(l,k,j,i) = -( flux_x(l,k,j,i+1) - flux_x(l,k,j,i) ) * r_dx
                                -( flux_y(l,k,j+1,i) - flux_y(l,k,j,i) ) * r_dy
                                -( flux_z(l,k+1,j,i) - flux_z(l,k,j,i) ) / dz(k);
          if (l == idW && enable_gravity) {
            state_tend(l,k,j,i) += grav*state(idR,k,j,i)*fields_loc(num_fields-1,hs+k,hs+j,hs+i)/
                                   hy_theta_cells(hs+k);
          }
          if (latitude != 0 && l == idU) state_tend(l,k,j,i) += fcor*state(idV,k,j,i);
          if (latitude != 0 && l == idV) state_tend(l,k,j,i) -= fcor*state(idU,k,j,i);
        } else {
          tracers_tend(l-num_state,k,j,i) = -( flux_x(l,k,j,i+1) - flux_x(l,k,j,i) ) * r_dx
                                            -( flux_y(l,k,j+1,i) - flux_y(l,k,j,i) ) * r_dy 
                                            -( flux_z(l,k+1,j,i) - flux_z(l,k,j,i) ) / dz(k);
        }
      });

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        FLOC hv_beta = 0;
        if      (any_immersed2 (k,j,i)) { hv_beta = 1.f/4.f/2.f;  }
        else if (any_immersed4 (k,j,i)) { hv_beta = 1.f/4.f/4.f;  }
        else if (any_immersed6 (k,j,i)) { hv_beta = 1.f/4.f/8.f;  }
        else if (any_immersed8 (k,j,i)) { hv_beta = 1.f/4.f/16.f; }
        else if (any_immersed10(k,j,i)) { hv_beta = 1.f/4.f/32.f; }
        if (hv_beta > 0) {
          SArray<bool ,1,ord> immersed;
          for (int ii = 0; ii < ord; ii++) { immersed(ii) = immersed_prop(hs+k,hs+j,1+i+ii) > 0; }
          for (int l=0; l < num_fields; l++) {
            SArray<FLOC,1,ord> s;
            for (int ii = 0; ii < ord; ii++) { s(ii) = fields_loc(l,hs+k,hs+j,1+i+ii); }
            if (l==idV || l==idW) modify_stencil_immersed_der0( s , immersed );
            if      (l == idR)      { state_tend  (l          ,k,j,i) +=                  hv_beta*hypervis(s)/dt; }
            else if (l < num_state) { state_tend  (l          ,k,j,i) += state(idR,k,j,i)*hv_beta*hypervis(s)/dt; }
            else                    { tracers_tend(l-num_state,k,j,i) += state(idR,k,j,i)*hv_beta*hypervis(s)/dt; }
          }
          for (int jj = 0; jj < ord; jj++) { immersed(jj) = immersed_prop(hs+k,1+j+jj,hs+i) > 0; }
          for (int l=0; l < num_fields; l++) {
            SArray<FLOC,1,ord> s;
            for (int jj = 0; jj < ord; jj++) { s(jj) = fields_loc(l,hs+k,1+j+jj,hs+i); }
            if (l==idU || l==idW) modify_stencil_immersed_der0( s , immersed );
            if      (l == idR)      { state_tend  (l          ,k,j,i) +=                  hv_beta*hypervis(s)/dt; }
            else if (l < num_state) { state_tend  (l          ,k,j,i) += state(idR,k,j,i)*hv_beta*hypervis(s)/dt; }
            else                    { tracers_tend(l-num_state,k,j,i) += state(idR,k,j,i)*hv_beta*hypervis(s)/dt; }
          }
          for (int kk = 0; kk < ord; kk++) { immersed(kk) = immersed_prop(1+k+kk,hs+j,hs+i) > 0; }
          for (int l=0; l < num_fields; l++) {
            SArray<FLOC,1,ord> s;
            for (int kk = 0; kk < ord; kk++) { s(kk) = fields_loc(l,1+k+kk,hs+j,hs+i); }
            if (l==idU || l==idV) modify_stencil_immersed_der0( s , immersed );
            if      (l == idR)      { state_tend  (l          ,k,j,i) +=                  hv_beta*hypervis(s)/dt; }
            else if (l < num_state) { state_tend  (l          ,k,j,i) += state(idR,k,j,i)*hv_beta*hypervis(s)/dt; }
            else                    { tracers_tend(l-num_state,k,j,i) += state(idR,k,j,i)*hv_beta*hypervis(s)/dt; }
          }
        }
      });

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("compute_tendencies");
      #endif
    }



    void halo_boundary_conditions( core::Coupler const & coupler ,
                                   yakl::Array<FLOC,4> const & fields  ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("halo_boundary_conditions");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto &dm         = coupler.get_data_manager_readonly();
      auto num_fields  = fields.extent(0);

      if (coupler.get_option<std::string>("bc_x1") == "periodic") { // Already handled in halo_exchange
      } else if (coupler.get_option<std::string>("bc_x1") == "open" ||
                 coupler.get_option<std::string>("bc_x1") == "precursor" ) {
        if (coupler.get_px() == 0) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz,ny,hs) ,
                                            KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            fields(l,hs+k,hs+j,hs-1-ii) = fields(l,hs+k,hs+j,hs+0);
          });
        }
      } else {
        std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_x1 can only be periodic or open";
        Kokkos::abort("");
      }

      if (coupler.get_option<std::string>("bc_x2") == "periodic") { // Already handled in halo_exchange
      } else if (coupler.get_option<std::string>("bc_x2") == "open" ||
                 coupler.get_option<std::string>("bc_x2") == "precursor" ) {
        if (coupler.get_px() == coupler.get_nproc_x()-1) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz,ny,hs) ,
                                            KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            fields(l,hs+k,hs+j,hs+nx+ii) = fields(l,hs+k,hs+j,hs+nx-1);
          });
        }
      } else {
        std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_x2 can only be periodic or open";
        Kokkos::abort("");
      }

      if (coupler.get_option<std::string>("bc_y1") == "periodic") { // Already handled in halo_exchange
      } else if (coupler.get_option<std::string>("bc_y1") == "open" ||
                 coupler.get_option<std::string>("bc_y1") == "precursor" ) {
        if (coupler.get_py() == 0) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz,hs,nx) ,
                                            KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            fields(l,hs+k,hs-1-jj,hs+i) = fields(l,hs+k,hs+0,hs+i);
          });
        }
      } else {
        std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_y1 can only be periodic or open";
        Kokkos::abort("");
      }

      if (coupler.get_option<std::string>("bc_y2") == "periodic") { // Already handled in halo_exchange
      } else if (coupler.get_option<std::string>("bc_y2") == "open" ||
                 coupler.get_option<std::string>("bc_y2") == "precursor" ) {
        if (coupler.get_py() == coupler.get_nproc_y()-1) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,nz,hs,nx) ,
                                            KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            fields(l,hs+k,hs+ny+jj,hs+i) = fields(l,hs+k,hs+ny-1,hs+i);
          });
        }
      } else {
        std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_y2 can only be periodic or open";
        Kokkos::abort("");
      }

      if (coupler.get_option<std::string>("bc_z1") == "wall_free_slip") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,hs,ny,nx) ,
                                          KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          if (l == idW) {
            fields(l,kk,hs+j,hs+i) = 0;
          } else {
            fields(l,hs-1-kk,hs+j,hs+i) = fields(l,hs+0,hs+j,hs+i);
          }
        });
      } else if (coupler.get_option<std::string>("bc_z1") == "periodic") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,hs,ny,nx) ,
                                          KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          fields(l,kk,hs+j,hs+i) = fields(l,nz+kk,hs+j,hs+i);
        });
      } else {
        std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_z1 can only be periodic or wall_free_slip";
        Kokkos::abort("");
      }

      if (coupler.get_option<std::string>("bc_z2") == "wall_free_slip") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,hs,ny,nx) ,
                                          KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          if (l == idW) {
            fields(l,hs+nz+kk,hs+j,hs+i) = 0;
          } else {
            fields(l,hs+nz+kk,hs+j,hs+i) = fields(l,hs+nz-1,hs+j,hs+i);
          }
        });
      } else if (coupler.get_option<std::string>("bc_z2") == "periodic") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_fields,hs,ny,nx) ,
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
      auto tracer_names   = coupler.get_tracer_names();
      auto &dm            = coupler.get_data_manager_readwrite();

      // Compute the metric jacobian (dz/dzeta) where zeta is the k interface index
      //
      // # Sagemath code
      // def coefs_1d(N,N0,lab) :
      //     return vector([ var(lab+'%s'%i) for i in range(N0,N0+N) ])
      // def poly_1d(N,coefs,x) :
      //     return sum( vector([ coefs[i]*x^i for i in range(N) ]) )
      // N      = 6
      // coefs  = coefs_1d(N,0,'a')
      // p      = poly_1d(N,coefs,x)
      // constr = vector([ p.subs(x=i-N/2+1) for i in range(N) ])
      // p      = poly_1d(N,jacobian(constr,coefs)^-1*coefs_1d(N,0,'s'),x)
      // print( vector([ i-N/2+1 for i in range(N) ]) )
      // print( 60*p.diff(x).subs(x=0) )
      // print( 60*p.diff(x).subs(x=1) )
      //
      dm.register_and_allocate<real>("dycore_metjac_edges","",{nz+2,2});
      auto metjac_edges = dm.get<real,2>("dycore_metjac_edges");
      parallel_for( YAKL_AUTO_LABEL() , nz+2 , KOKKOS_LAMBDA (int k_in) {
        int k = k_in-1;
        SArray<real,1,6> s;
        s(0) = -dz(std::max(0,k-1))-dz(std::max(0,k-2));
        for (int kk=1; kk < 6; kk++) { s(kk) = s(kk-1) + dz(std::max(0,std::min(nz-1,k-3+kk))); }
        for (int kk=0; kk < 6; kk++) { s(kk) /= dz(std::max(0,std::min(nz-1,k))); }
        metjac_edges(k+1,0) = ( 3*s(0)-30*s(1)-20*s(2)+60*s(3)-15*s(4)+2*s(5))/60.;
        metjac_edges(k+1,1) = (-2*s(0)+15*s(1)-60*s(2)+20*s(3)+30*s(4)-3*s(5))/60.;
      });

      coupler.set_option<int>("dycore_hs",hs);

      bool1d tracer_adds_mass("tracer_adds_mass",num_tracers);
      bool1d tracer_positive ("tracer_positive" ,num_tracers);
      auto tracer_adds_mass_host = tracer_adds_mass.createHostCopy();
      auto tracer_positive_host  = tracer_positive .createHostCopy();
      for (int tr=0; tr < num_tracers; tr++) {
        std::string tracer_desc;
        bool        tracer_found, positive, adds_mass, diffuse;
        coupler.get_tracer_info( tracer_names.at(tr) , tracer_desc, tracer_found , positive , adds_mass , diffuse );
        tracer_positive_host (tr) = positive;
        tracer_adds_mass_host(tr) = adds_mass;
      }
      tracer_positive_host .deep_copy_to(tracer_positive );
      tracer_adds_mass_host.deep_copy_to(tracer_adds_mass);
      dm.register_and_allocate<bool>("tracer_adds_mass","",{num_tracers});
      auto dm_tracer_adds_mass = dm.get<bool,1>("tracer_adds_mass");
      tracer_adds_mass.deep_copy_to(dm_tracer_adds_mass);
      dm.register_and_allocate<bool>("tracer_positive","",{num_tracers});
      auto dm_tracer_positive = dm.get<bool,1>("tracer_positive");
      tracer_positive.deep_copy_to(dm_tracer_positive);

      real4d state  ;
      real4d tracers;
      convert_coupler_to_dynamics( coupler , state , tracers );
      dm.register_and_allocate<real>("hy_dens_cells"    ,"",{nz+2*hs});
      dm.register_and_allocate<real>("hy_theta_cells"   ,"",{nz+2*hs});
      auto r = dm.get<real,1>("hy_dens_cells"    );    r = 0;
      auto t = dm.get<real,1>("hy_theta_cells"   );    t = 0;
      parallel_for( YAKL_AUTO_LABEL() , nz , KOKKOS_LAMBDA (int k) {
        for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
            r(hs+k) += state  (idR        ,k,j,i);
            t(hs+k) += tracers(num_tracers,k,j,i) / state(idR,k,j,i);
          }
        }
      });
      coupler.get_parallel_comm().all_reduce( r , MPI_SUM ).deep_copy_to(r);
      coupler.get_parallel_comm().all_reduce( t , MPI_SUM ).deep_copy_to(t);
      real r_nx_ny = 1./(nx_glob*ny_glob);
      parallel_for( YAKL_AUTO_LABEL() , nz , KOKKOS_LAMBDA (int k) {
        r(hs+k) *= r_nx_ny;
        t(hs+k) *= r_nx_ny;
      });
      parallel_for( YAKL_AUTO_LABEL() , hs , KOKKOS_LAMBDA (int kk) {
        {
          int  k0       = hs;
          int  k        = k0-1-kk;
          real rho0     = r(k0);
          real theta0   = t(k0);
          real rho0_gm1 = std::pow(rho0  ,gamma-1);
          real theta0_g = std::pow(theta0,gamma  );
          r(k) = std::pow( rho0_gm1 + grav*(gamma-1)*dz(0)*(kk+1)/(gamma*C0*theta0_g) , 1._fp/(gamma-1) );
          t(k) = theta0-(t(k0+1)-t(k0))*(kk+1);
        }
        {
          int  k0       = hs+nz-1;
          int  k        = k0+1+kk;
          real rho0     = r(k0);
          real theta0   = t(k0);
          real rho0_gm1 = std::pow(rho0  ,gamma-1);
          real theta0_g = std::pow(theta0,gamma  );
          r(k) = std::pow( rho0_gm1 - grav*(gamma-1)*dz(nz-1)*(kk+1)/(gamma*C0*theta0_g) , 1._fp/(gamma-1) );
          t(k) = theta0+(t(k0)-t(k0-1))*(kk+1);
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
          // if (dm.entry_exists("windmill_proj_weight")) {
          //   auto proj = dm.get<real const,3>("windmill_proj_weight");
          //   parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
          //     immersed_prop(k,j,i) += proj(k,j,i);
          //   });
          // }
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
        if (! dm.entry_exists("hy_dens_edges"    )) dm.register_and_allocate<real>("hy_dens_edges"    ,"",{nz+1});
        if (! dm.entry_exists("hy_theta_edges"   )) dm.register_and_allocate<real>("hy_theta_edges"   ,"",{nz+1});
        auto hy_dens_cells     = dm.get<real const,1>("hy_dens_cells"    );
        auto hy_theta_cells    = dm.get<real const,1>("hy_theta_cells"   );
        auto hy_dens_edges     = dm.get<real      ,1>("hy_dens_edges"    );
        auto hy_theta_edges    = dm.get<real      ,1>("hy_theta_edges"   );
        if (ord < 5) {
          parallel_for( YAKL_AUTO_LABEL() , nz+1 , KOKKOS_LAMBDA (int k) {
            hy_dens_edges    (k) = std::exp( 0.5_fp*std::log(hy_dens_cells(hs+k-1)) +
                                             0.5_fp*std::log(hy_dens_cells(hs+k  )) );
            hy_theta_edges   (k) =           0.5_fp*hy_theta_cells(hs+k-1) +
                                             0.5_fp*hy_theta_cells(hs+k  ) ;
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
        nc.create_var<real>( "hy_dens_cells"     , {"z_halo"});
        nc.create_var<real>( "hy_theta_cells"    , {"z_halo"});
        nc.create_var<real>( "theta_pert"        , {"z","y","x"});
        nc.create_var<real>( "density_pert"      , {"z","y","x"});
        nc.enddef();
        nc.begin_indep_data();
        auto &dm = coupler.get_data_manager_readonly();
        if (coupler.is_mainproc()) nc.write( dm.get<real const,1>("hy_dens_cells"    ) , "hy_dens_cells"     );
        if (coupler.is_mainproc()) nc.write( dm.get<real const,1>("hy_theta_cells"   ) , "hy_theta_cells"    );
        nc.end_indep_data();
        real4d state  ;
        real4d tracers;
        convert_coupler_to_dynamics( coupler , state , tracers );
        auto num_tracers = tracers.extent(0);
        std::vector<MPI_Offset> start_3d = {0,(MPI_Offset)j_beg,(MPI_Offset)i_beg};
        real3d data("data",nz,ny,nx);
        auto hy_dens_cells = dm.get<real const,1>("hy_dens_cells");
        yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
          data(k,j,i) = state(idR,k,j,i) - hy_dens_cells(hs+k);
        });
        nc.write_all(data.as<float>(),"density_pert",start_3d);
        auto hy_theta_cells = dm.get<real const,1>("hy_theta_cells");
        yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
          data(k,j,i) = tracers(num_tracers-1,k,j,i) / state(idR,k,j,i) - hy_theta_cells(hs+k);
        });
        nc.write_all(data.as<float>(),"theta_pert",start_3d);
      } );
      coupler.register_overwrite_with_restart_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        auto &dm = coupler.get_data_manager_readwrite();
        nc.read_all(dm.get<real,1>("hy_dens_cells"    ),"hy_dens_cells"    ,{0});
        nc.read_all(dm.get<real,1>("hy_theta_cells"   ),"hy_theta_cells"   ,{0});
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
      auto  nx               = coupler.get_nx();
      auto  ny               = coupler.get_ny();
      auto  nz               = coupler.get_nz();
      auto  R_d              = coupler.get_option<real>("R_d"    );
      auto  R_v              = coupler.get_option<real>("R_v"    );
      auto  gamma            = coupler.get_option<real>("gamma_d");
      auto  C0               = coupler.get_option<real>("C0"     );
      auto  idWV             = coupler.get_option<int >("idWV"   );
      auto  num_tracers      = coupler.get_num_tracers();
      auto  &dm              = coupler.get_data_manager_readwrite();
      auto  dm_rho_d         = dm.get<real      ,3>("density_dry"     );
      auto  dm_uvel          = dm.get<real      ,3>("uvel"            );
      auto  dm_vvel          = dm.get<real      ,3>("vvel"            );
      auto  dm_wvel          = dm.get<real      ,3>("wvel"            );
      auto  dm_temp          = dm.get<real      ,3>("temp"            );
      auto  tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");

      core::MultiField<real,3> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real,3>(tracer_names.at(tr)) ); }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real rho   = state  (idR        ,k,j,i);
        real u     = state  (idU        ,k,j,i) / rho;
        real v     = state  (idV        ,k,j,i) / rho;
        real w     = state  (idW        ,k,j,i) / rho;
        real theta = tracers(num_tracers,k,j,i) / rho;  // no -1 needed here for num_tracers
        real press = C0 * pow( rho*theta , gamma );
        real rho_v = tracers(idWV,k,j,i);
        real rho_d = rho;
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho_d -= tracers(tr,k,j,i); }
        real temp = press / ( rho_d * R_d + rho_v * R_v );
        dm_rho_d(k,j,i) = rho_d;
        dm_uvel (k,j,i) = u;
        dm_vvel (k,j,i) = v;
        dm_wvel (k,j,i) = w;
        dm_temp (k,j,i) = temp;
        for (int tr=0; tr < num_tracers; tr++) { dm_tracers(tr,k,j,i) = tracers(tr,k,j,i); }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("convert_dynamics_to_coupler");
      #endif
    }



    // Convert coupler's data to state and tracers arrays
    void convert_coupler_to_dynamics( core::Coupler & coupler ,
                                      real4d        & state   ,
                                      real4d        & tracers ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("convert_coupler_to_dynamics");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx               = coupler.get_nx();
      auto ny               = coupler.get_ny();
      auto nz               = coupler.get_nz();
      auto R_d              = coupler.get_option<real>("R_d"    );
      auto R_v              = coupler.get_option<real>("R_v"    );
      auto gamma            = coupler.get_option<real>("gamma_d");
      auto C0               = coupler.get_option<real>("C0"     );
      auto idWV             = coupler.get_option<int >("idWV"   );
      auto num_tracers      = coupler.get_num_tracers();
      auto &dm              = coupler.get_data_manager_readwrite();
      auto tracer_names     = coupler.get_tracer_names();
      auto dm_rho_d         = dm.get<real const,3>("density_dry"     );
      auto dm_uvel          = dm.get<real const,3>("uvel"            );
      auto dm_vvel          = dm.get<real const,3>("vvel"            );
      auto dm_wvel          = dm.get<real const,3>("wvel"            );
      auto dm_temp          = dm.get<real const,3>("temp"            );
      auto tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
      auto tracer_positive  = dm.get<bool const,1>("tracer_positive" );

      state   = real4d("state"  ,num_state    ,nz,ny,nx);
      tracers = real4d("tracers",num_tracers+1,nz,ny,nx);

      if (! dm.entry_exists("dycore_tracer_positive")) {
        dm.register_and_allocate<bool>("dycore_tracer_positive","",{num_tracers+1});
      }
      auto dc_tracer_positive = dm.get<bool,1>("dycore_tracer_positive");
      core::MultiField<real const,3> dm_tracers;
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
        state(idR,k,j,i) = rho;
        state(idU,k,j,i) = rho * u;
        state(idV,k,j,i) = rho * v;
        state(idW,k,j,i) = rho * w;
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,k,j,i) = dm_tracers(tr,k,j,i); }
        tracers(num_tracers,k,j,i) = rho * theta;
        if (k==0 && j==0 && i==0) {
          for (int tr=0; tr < num_tracers; tr++)  dc_tracer_positive(tr) = tracer_positive(tr);
          dc_tracer_positive(num_tracers) = true;
        }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("convert_coupler_to_dynamics");
      #endif
    }


  };

}



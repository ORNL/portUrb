
#pragma once

#include "main_header.h"
#include "coupler.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "WenoLimiter.h"
#include <sstream>

namespace modules {

  // This class simplements an A-grid (collocated) cell-centered Finite-Volume method with an upwind Riemann
  // solver at cell edges, high-order-accurate reconstruction, Weighted Essentially Non-Oscillatory (WENO) limiting,
  // and Strong Stability Preserving Runge-Kutta time stepping.
  // The dycore prognoses full density, u-, v-, and w-momenta, and mass-weighted virtual potential temperature
  // This dynamical core supports immersed boundaries, including partially immersed cells. Immersed
  // boundaries will have no-slip wall BC's, and surface fluxes are applied in a separate module to model friction
  // based on a prescribed roughness length with Monin-Obukhov thoery.

  struct Dynamics_Euler_Stratified_WenoFV {
    // Order of accuracy (numerical convergence rate for smooth flows) for the dynamical core
    #ifndef PORTURB_ORD
      size_t static constexpr ord = 9;
    #else
      size_t static constexpr ord = PORTURB_ORD;
    #endif
    int static constexpr hs  = (ord+1)/2; // Number of halo cells ("hs" == "halo size")
    int static constexpr num_state = 5;   // Number of state variables
    // IDs for the variables in the state vector
    int  static constexpr idR = 0;  // Density
    int  static constexpr idU = 1;  // u-momentum
    int  static constexpr idV = 2;  // v-momentum
    int  static constexpr idW = 3;  // w-momentum
    int  static constexpr idT = 4;  // Density * potential temperature

    typedef float FLOC; // Use single precision locally



    // Compute total mass of dry air and total mass of virtual potential temperature in the domain
    //  for verification purposes
    // coupler : Coupler instance
    // state   : State array from the dynamical core
    // Returns a tuple of summed dry air mass and virtual potential temperature mass
    std::tuple<real,real> compute_mass( core::Coupler & coupler , real4d const & state ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      auto dz = coupler.get_dz(); // 1D array of vertical cell grid spacing
      real3d r("r",nz,ny,nx); // Array for local mass
      real3d t("t",nz,ny,nx); // Array for local virtual potential temperature mass
      // Accumulate local mass
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j , int i) {
        r(k,j,i) = state(idR,k,j,i)*dz(k);
        t(k,j,i) = state(idT,k,j,i)*dz(k);
      });
      // Reduce the global mass across all MPI ranks
      real rmass = coupler.get_parallel_comm().all_reduce( yakl::intrinsics::sum(r) , MPI_SUM );
      real tmass = coupler.get_parallel_comm().all_reduce( yakl::intrinsics::sum(t) , MPI_SUM );
      return std::make_tuple(rmass,tmass);
    }



    // Compute the time step based on CFL condition using a global minimum over the domain with static wave speed
    real compute_time_step( core::Coupler const &coupler ) const {
      using yakl::intrinsics::minval;
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      auto cs = coupler.get_option<real>( "dycore_cs" , 350 ); // Speed of sound in m/s
      real maxwave = cs + coupler.get_option<real>( "dycore_max_wind" , 100 ); // Max wave speed in m/s (cs+wind)
      real cfl = coupler.get_option<real>("cfl",0.60);         // CFL number
      // Return the maximum stable time step based on the minimum cell size in the domain, max wave speed, and CFL number
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
    // coupler : Coupler instance
    // dt_phys : Desired physical time step to advance the solution (may be sub-cycled internally for stability)
    // Advances the solution in the coupler's data manager state and tracer arrays by dt_phys
    // Uses sub-cycling with stable dynamical core time steps as needed
    // Allocates storage for ghost cell exchanges between precursor and forced simulations if needed at the
    //  appropriate MPI ranks
    void time_step(core::Coupler &coupler, real dt_phys) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers(); // Total number of tracers
      auto nx          = coupler.get_nx(); // Number of cells in x-direction (excluding halos)
      auto ny          = coupler.get_ny(); // Number of cells in y-direction (excluding halos)
      auto nz          = coupler.get_nz(); // Number of cells in z-direction (excluding halos)
      auto px          = coupler.get_px(); // MPI decomposition ID in x-direction
      auto py          = coupler.get_py(); // MPI decomposition ID in y-direction
      auto bc_x1       = coupler.get_option<std::string>("bc_x1"); // West x-boundary condition
      auto bc_x2       = coupler.get_option<std::string>("bc_x2"); // East x-boundary condition
      auto bc_y1       = coupler.get_option<std::string>("bc_y1"); // South y-boundary condition
      auto bc_y2       = coupler.get_option<std::string>("bc_y2"); // North y-boundary condition
      auto bc_z1       = coupler.get_option<std::string>("bc_z1"); // Bottom z-boundary condition
      auto bc_z2       = coupler.get_option<std::string>("bc_z2"); // Top z-boundary condition
      auto &dm         = coupler.get_data_manager_readwrite(); // Get data manager for read/write access
      auto npx         = coupler.get_nproc_x(); // MPI Decomposition size in x-direction
      auto npy         = coupler.get_nproc_y(); // MPI Decomposition size in y-direction
      real4d state  ("state"  ,num_state  ,nz,ny,nx); // State array for the dynamical core
      real4d tracers("tracers",num_tracers,nz,ny,nx); // Tracer array for the dynamical core
      convert_coupler_to_dynamics( coupler , state , tracers ); // Convert coupler data to dynamical core format
      real dt_dyn = compute_time_step( coupler );        // Compute maximum stable dynamical core time step
      int ncycles = (int) std::ceil( dt_phys / dt_dyn ); // Determine number of sub-cycles needed for stability
      dt_dyn = dt_phys / ncycles;                        // Make sure individual sub-step time steps are equal

      // If the current coupler object is a precursor for another simulation, or the current coupler is using
      //  precursor BC's, then allocate ghost cell storage for exchanging data between the precursor and forced
      //  simulation. This storage is only needed on the MPI ranks at the domain boundaries where precursor BC's
      //  are applied.
      // We need storage for all variables, all Runge-Kutta stages, all sub-cycles, and the halo size in
      //  the horizontal directions.
      // The array that is halo exchanged has 5 state variables (num_state), all tracers (num_tracers),
      //  and a pressure variable, so num_state+num_tracers+1.
      if ( coupler.get_option<bool>("dycore_is_precursor",false)   ||
           coupler.get_option<std::string>("bc_x1") == "precursor" ||
           coupler.get_option<std::string>("bc_x2") == "precursor" ||
           coupler.get_option<std::string>("bc_y1") == "precursor" ||
           coupler.get_option<std::string>("bc_y2") == "precursor" ) {
        auto nstage = coupler.get_option<int>("dycore_num_stages"); // Number of Runge-Kutta stages
        // Make sure that dycore_ghost_[xy][12] arrays are only accessed in the correct MPI ranks
        if (px == 0) { // If we're at the west edge process of the domain
          if (dm.entry_exists("dycore_ghost_x1")) { // Make sure existing array has correct size
            if (dm.get<FLOC const,6>("dycore_ghost_x1").extent(0) != ncycles) {
              dm.unregister_and_deallocate( "dycore_ghost_x1" );
            }
          }
          if (! dm.entry_exists("dycore_ghost_x1")) { // if not existing, register and allocate
            dm.register_and_allocate<FLOC>("dycore_ghost_x1","",{ncycles,nstage,num_state+num_tracers+1,nz,ny,hs});
          }
        }
        if (px == npx-1) { // If we're at the east edge process of the domain
          if (dm.entry_exists("dycore_ghost_x2")) { // Make sure existing array has correct size
            if (dm.get<FLOC const,6>("dycore_ghost_x2").extent(0) != ncycles) {
              dm.unregister_and_deallocate( "dycore_ghost_x2" );
            }
          }
          if (! dm.entry_exists("dycore_ghost_x2")) { // if not existing, register and allocate
            dm.register_and_allocate<FLOC>("dycore_ghost_x2","",{ncycles,nstage,num_state+num_tracers+1,nz,ny,hs});
          }
        }
        if (py == 0) { // If we're at the south edge process of the domain
          if (dm.entry_exists("dycore_ghost_y1")) { // Make sure existing array has correct size
            if (dm.get<FLOC const,6>("dycore_ghost_y1").extent(0) != ncycles) {
              dm.unregister_and_deallocate( "dycore_ghost_y1" );
            }
          }
          if (! dm.entry_exists("dycore_ghost_y1")) { // if not existing, register and allocate
            dm.register_and_allocate<FLOC>("dycore_ghost_y1","",{ncycles,nstage,num_state+num_tracers+1,nz,hs,nx});
          }
        }
        if (py == npy-1) { // If we're at the north edge process of the domain
          if (dm.entry_exists("dycore_ghost_y2")) { // Make sure existing array has correct size
            if (dm.get<FLOC const,6>("dycore_ghost_y2").extent(0) != ncycles) {
              dm.unregister_and_deallocate( "dycore_ghost_y2" );
            }
          }
          if (! dm.entry_exists("dycore_ghost_y2")) { // if not existing, register and allocate
            dm.register_and_allocate<FLOC>("dycore_ghost_y2","",{ncycles,nstage,num_state+num_tracers+1,nz,hs,nx});
          }
        }
      }

      // auto mass1 = compute_mass( coupler , state );
      // Get the desired time stepper from the coupler options and perform the sub-cycled time stepping
      // Must pass the icycle number to the time stepper for proper ghost cell exchanges with precursor simulations
      auto time_stepper = coupler.get_option<std::string>("dycore_time_stepper","ssprk3");
      for (int icycle = 0; icycle < ncycles; icycle++) {
        if      (time_stepper == "linrk3") { time_step_rk3   (coupler,state,tracers,dt_dyn,icycle); }
        else if (time_stepper == "linrk4") { time_step_rk4   (coupler,state,tracers,dt_dyn,icycle); }
        else if (time_stepper == "ssprk3") { time_step_ssprk3(coupler,state,tracers,dt_dyn,icycle); }
      }
      // auto mass2 = compute_mass( coupler , state );
      // if (coupler.is_mainproc()) std::cout << "Mass change: "
      //                                      << (std::get<0>(mass2)-std::get<0>(mass1))/std::get<0>(mass1) << " , "
      //                                      << (std::get<1>(mass2)-std::get<1>(mass1))/std::get<1>(mass1) << std::endl;
      // Convert the dynamical core state and tracer arrays back to the coupler format
      convert_dynamics_to_coupler( coupler , state , tracers );
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step");
      #endif
    }



    // Max CFL: 0.72
    // This CFL is smaller than normal because dimensions are split within each RK stage
    // This is the linearly third-order, non-linearly second-order quasi-Runge-Kutta method used by WRF
    // coupler : Coupler instance
    // state   : State array from the dynamical core
    // tracers : Tracer array from the dynamical core
    // dt_dyn  : Dynamical core time step to use for this sub-step
    // icycle  : Current sub-cycle index (from 0 to ncycles-1)
    // Advances the solution in state and tracers by dt_dyn using the linRK3 method
    // The icycle number is used for proper ghost cell exchanges between precursor and forced simulations
    void time_step_rk3( core::Coupler & coupler ,
                        real4d const  & state   ,
                        real4d const  & tracers ,
                        real            dt_dyn  ,
                        int             icycle  ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step_rk_3_3");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers(); // Total number of tracers
      auto nx          = coupler.get_nx();          // Number of cells in x-direction (excluding halos)
      auto ny          = coupler.get_ny();          // Number of cells in y-direction (excluding halos)
      auto nz          = coupler.get_nz();          // Number of cells in z-direction (excluding halos)
      auto &dm         = coupler.get_data_manager_readonly(); // Get data manager for read-only access
      auto tracer_positive = dm.get<bool const,1>("tracer_positive"); // Whether each tracer is positive definite
      // RK3 requires temporary arrays to hold intermediate state and tracers arrays
      real4d state_tmp   ("state_tmp"   ,num_state  ,nz,ny,nx);
      real4d tracers_tmp ("tracers_tmp" ,num_tracers,nz,ny,nx);
      // To hold tendencies (time derivatives of state and tracers)
      real4d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx);
      real4d tracers_tend("tracers_tend",num_tracers,nz,ny,nx);

      // Set immersed boundaries in state and tracers to hydrostasis at rest
      enforce_immersed_boundaries( coupler , state , tracers );

      // Stage 1
      // Compute time derivatives of the state and tracers using a time steyp of dt/3
      compute_tendencies(coupler,state    ,state_tend,tracers    ,tracers_tend,dt_dyn/3,0,icycle);
      // Apply tendencies for the first stage for state and tracers
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = state  (l,k,j,i) + dt_dyn/3 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = tracers(l,k,j,i) + dt_dyn/3 * tracers_tend(l,k,j,i);
        }
      });

      // Stage 2
      // Compute time derivatives of the state and tracers using a time step of dt/2
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/2,1,icycle);
      // Apply tendencies for the second stage for state and tracers
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
      // Compute time derivatives of the state and tracers using a time step of dt/1
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/1,2,icycle);
      // Apply tendencies for the third stage for state and tracers
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state      (l,k,j,i) = state  (l,k,j,i) + dt_dyn/1 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers    (l,k,j,i) = tracers(l,k,j,i) + dt_dyn/1 * tracers_tend(l,k,j,i);
          // Correct tracer values to be positive definite if needed
          if (tracer_positive(l))  tracers(l,k,j,i) = std::max( 0._fp , tracers(l,k,j,i) );
        }
      });

      // Set immersed boundaries in state and tracers to hydrostasis at rest
      enforce_immersed_boundaries( coupler , state , tracers );

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step_rk_3_3");
      #endif
    }



    // Max CFL: 0.99
    // This CFL is smaller than normal because dimensions are split within each RK stage
    // This is the linearly fourth-order, non-linearly second-order quasi-Runge-Kutta method used by WRF
    // coupler : Coupler instance
    // state   : State array from the dynamical core
    // tracers : Tracer array from the dynamical core
    // dt_dyn  : Dynamical core time step to use for this sub-step
    // icycle  : Current sub-cycle index (from 0 to ncycles-1)
    // Advances the solution in state and tracers by dt_dyn using the linRK3 method
    // The icycle number is used for proper ghost cell exchanges between precursor and forced simulations
    void time_step_rk4( core::Coupler & coupler ,
                        real4d const  & state   ,
                        real4d const  & tracers ,
                        real            dt_dyn  ,
                        int             icycle  ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step_rk_3_3");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers();           // Total number of tracers
      auto nx          = coupler.get_nx();                    // Number of cells in x-direction (excluding halos)
      auto ny          = coupler.get_ny();                    // Number of cells in y-direction (excluding halos)
      auto nz          = coupler.get_nz();                    // Number of cells in z-direction (excluding halos)
      auto &dm         = coupler.get_data_manager_readonly(); // Get data manager for read-only access
      auto tracer_positive = dm.get<bool const,1>("tracer_positive"); // Whether each tracer is positive definite
      // SSPRK3 requires temporary arrays to hold intermediate state and tracers arrays
      real4d state_tmp   ("state_tmp"   ,num_state  ,nz,ny,nx);
      real4d tracers_tmp ("tracers_tmp" ,num_tracers,nz,ny,nx);
      // To hold tendencies (time derivatives of state and tracers)
      real4d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx);
      real4d tracers_tend("tracers_tend",num_tracers,nz,ny,nx);

      // Set immersed boundaries in state and tracers to hydrostasis at rest
      enforce_immersed_boundaries( coupler , state , tracers );

      // Stage 1
      // Compute time derivatives of the state and tracers using a time steyp of dt/4
      compute_tendencies(coupler,state    ,state_tend,tracers    ,tracers_tend,dt_dyn/4,0,icycle);
      // Apply tendencies for the first stage for state and tracers
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
      // Compute time derivatives of the state and tracers using a time step of dt/3
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/3,1,icycle);
      // Apply tendencies for the second stage for state and tracers
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
      // Compute time derivatives of the state and tracers using a time step of dt/2
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/2,2,icycle);
      // Apply tendencies for the third stage for state and tracers
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
      // Compute time derivatives of the state and tracers using a time step of dt/1
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/1,3,icycle);
      // Apply tendencies for the fourth stage for state and tracers
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state      (l,k,j,i) = state  (l,k,j,i) + dt_dyn/1 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers    (l,k,j,i) = tracers(l,k,j,i) + dt_dyn/1 * tracers_tend(l,k,j,i);
          // Correct tracer values to be positive definite if needed
          if (tracer_positive(l))  tracers(l,k,j,i) = std::max( 0._fp , tracers(l,k,j,i) );
        }
      });

      // Set immersed boundaries in state and tracers to hydrostasis at rest
      enforce_immersed_boundaries( coupler , state , tracers );

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step_rk_3_3");
      #endif
    }



    // Max CFL: 0.72
    // This CFL is smaller than normal because dimensions are split within each RK stage
    // This is the optimal three-stage third-order Strong Stability Preserving Runge-Kutta method
    // coupler : Coupler instance
    // state   : State array from the dynamical core
    // tracers : Tracer array from the dynamical core
    // dt_dyn  : Dynamical core time step to use for this sub-step
    // icycle  : Current sub-cycle index (from 0 to ncycles-1)
    // Advances the solution in state and tracers by dt_dyn using the linRK3 method
    // The icycle number is used for proper ghost cell exchanges between precursor and forced simulations
    void time_step_ssprk3( core::Coupler & coupler ,
                           real4d const  & state   ,
                           real4d const  & tracers ,
                           real            dt_dyn  ,
                           int             icycle  ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step_rk_3_3");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers();            // Total number of tracers
      auto nx          = coupler.get_nx();                     // Number of cells in x-direction (excluding halos)
      auto ny          = coupler.get_ny();                     // Number of cells in y-direction (excluding halos)
      auto nz          = coupler.get_nz();                     // Number of cells in z-direction (excluding halos)
      auto &dm         = coupler.get_data_manager_readonly();  // Get data manager for read-only access
      auto tracer_positive = dm.get<bool const,1>("tracer_positive"); // Whether each tracer is positive definite
      // SSPRK3 requires temporary arrays to hold intermediate state and tracers arrays
      real4d state_tmp   ("state_tmp"   ,num_state  ,nz,ny,nx);
      real4d tracers_tmp ("tracers_tmp" ,num_tracers,nz,ny,nx);
      // To hold tendencies (time derivatives of state and tracers)
      real4d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx);
      real4d tracers_tend("tracers_tend",num_tracers,nz,ny,nx);

      // Set immersed boundaries in state and tracers to hydrostasis at rest
      enforce_immersed_boundaries( coupler , state , tracers );

      // Stage 1
      // Compute time derivatives of the state and tracers using a time steyp of dt
      compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn,0,icycle);
      // Apply tendencies for the first stage for state and tracers
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = state  (l,k,j,i) + dt_dyn * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = tracers(l,k,j,i) + dt_dyn * tracers_tend(l,k,j,i);
        }
      });

      // Stage 2
      // Compute time derivatives of the state and tracers using a time step of dt/4
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/4.,1,icycle);
      // Apply tendencies for the second stage for state and tracers
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

      // Stage 3
      // Compute time derivatives of the state and tracers using a time step of dt*2/3
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn*2./3.,2,icycle);
      // Apply tendencies for the third stage for state and tracers
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

      // Set immersed boundaries in state and tracers to hydrostasis at rest
      enforce_immersed_boundaries( coupler , state , tracers );

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step_rk_3_3");
      #endif
    }



    // Enforce immersed boundary conditions by relaxing variables toward hydrostasis at rest
    // coupler : Coupler instance
    // state   : State array from the dynamical core
    // tracers : Tracer array from the dynamical core
    void enforce_immersed_boundaries( core::Coupler       & coupler ,
                                      real4d        const & state   ,
                                      real4d        const & tracers ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("enforce_immersed_boundaries");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers     = coupler.get_num_tracers();                    // Total number of tracers
      auto nx              = coupler.get_nx();                             // Number of cells in x-direction (excluding halos)
      auto ny              = coupler.get_ny();                             // Number of cells in y-direction (excluding halos)
      auto nz              = coupler.get_nz();                             // Number of cells in z-direction (excluding halos)
      auto immersed_power  = coupler.get_option<real>("immersed_power",5); // Power for immersed boundary relaxation
      auto &dm             = coupler.get_data_manager_readonly();          // Get data manager for read-only access
      auto hy_dens_cells   = dm.get<real const,1>("hy_dens_cells" );       // Hydrostatic density
      auto hy_theta_cells  = dm.get<real const,1>("hy_theta_cells");       // Hydrostatic potential temperature
      auto immersed_prop   = dm.get<real const,3>("dycore_immersed_proportion_halos"); // Immersed Proportion (with halos)
      auto tracer_positive = dm.get<bool const,1>("tracer_positive");      // Whether each tracer is positive definite

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real mult = std::pow( immersed_prop(hs+k,hs+j,hs+i) , immersed_power ); // Pre-compute multiplier
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
        // density*theta
        {
          auto &var = state(idT,k,j,i);
          real  target = hy_dens_cells(hs+k)*hy_theta_cells(hs+k);
          var = var + (target - var)*mult;
        }
        // Tracers
        for (int tr=0; tr < num_tracers; tr++) {
          auto &var = tracers(tr,k,j,i);
          real  target = 0;
          var = var + (target - var)*mult;
          if (tracer_positive(tr))  var = std::max( 0._fp , var ); // Keep positive tracers positive
        }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("enforce_immersed_boundaries");
      #endif
    }



    // Once you encounter an immersed boundary, set zero derivative boundary conditions from there out in that direction
    // stencil  : Stencil array to modify
    // immersed : Boolean array indicating which points are immersed
    template <class FP, size_t ORD>
    KOKKOS_INLINE_FUNCTION static void modify_stencil_immersed_der0( SArray<FP  ,1,ORD>       & stencil  ,
                                                                     SArray<bool,1,ORD> const & immersed ) {
      int constexpr hs = (ORD-1)/2; // Halo size
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


    // Compute hyperviscosity derivatives from a stencil of values
    // s   : Stencil of values to compute hyperviscosity from
    // ord : Order of hyperviscosity (must be odd: 3,5,...)
    real static KOKKOS_INLINE_FUNCTION hypervis( SArray<FLOC,1,ord> const &s) {
      if      constexpr (ord == 3 ) { return  ( 1.00000000f*s(0)-2.00000000f*s(1)+1.00000000f*s(2) )/4; }
      else if constexpr (ord == 5 ) { return -( 1.00000000f*s(0)-4.00000000f*s(1)+6.00000000f*s(2)-4.00000000f*s(3)+1.00000000f*s(4) )/16; }
      else if constexpr (ord == 7 ) { return  ( 1.00000000f*s(0)-6.00000000f*s(1)+15.0000000f*s(2)-20.0000000f*s(3)+15.0000000f*s(4)-6.00000000f*s(5)+1.00000000f*s(6) )/64; }
      else if constexpr (ord == 9 ) { return -( 1.00000000f*s(0)-8.00000000f*s(1)+28.0000000f*s(2)-56.0000000f*s(3)+70.0000000f*s(4)-56.0000000f*s(5)+28.0000000f*s(6)-8.00000000f*s(7)+1.00000000f*s(8) )/256; }
      else if constexpr (ord == 11) { return  ( 1.00000000f*s(0)-10.0000000f*s(1)+1.00000000f*s(10)+45.0000000f*s(2)-120.000000f*s(3)+210.000000f*s(4)-252.000000f*s(5)+210.000000f*s(6)-120.000000f*s(7)+45.0000000f*s(8)-10.0000000f*s(9) )/1024; }
    }




    int static constexpr idP = 5; // Index of pressure in total array of num_state+1+num_tracers in compute_tendencies

    // Compute the tendencies (time derivatives) of the state and tracer variables
    // coupler      : Coupler instance
    // state        : State array from the dynamical core
    // state_tend   : Output array for time derivatives of the state
    // tracers      : Tracer array from the dynamical core
    // tracers_tend : Output array for time derivatives of the tracers
    // dt           : Time step to use for this tendency calculation
    // istage       : Current RK stage index
    // icycle       : Current sub-cycle index (from 0 to ncycles-1)
    // This function fills in state_tend and tracers_tend based on the current state and tracers
    // The istage and icycle numbers are used for proper ghost cell exchanges between precursor and forced simulations
    // The dt value is provided in case any time-dependent terms are needed (e.g., for time filtering)
    void compute_tendencies( core::Coupler       & coupler      ,
                             real4d        const & state        ,
                             real4d        const & state_tend   ,
                             real4d        const & tracers      ,
                             real4d        const & tracers_tend ,
                             real                  dt           ,
                             int                   istage       ,
                             int                   icycle       ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("compute_tendencies");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx                = coupler.get_nx();           // Number of cells in x-direction (excluding halos)
      auto ny                = coupler.get_ny();           // Number of cells in y-direction (excluding halos)
      auto nz                = coupler.get_nz();           // Number of cells in z-direction (excluding halos)
      auto dx                = coupler.get_dx();           // Grid spacing in x-direction
      auto dy                = coupler.get_dy();           // Grid spacing in y-direction
      auto dz                = coupler.get_dz();           // Grid spacing in z-direction
      auto num_tracers       = coupler.get_num_tracers();  // Total number of tracers
      auto enable_gravity    = coupler.get_option<bool>("enable_gravity",true); // Whether to enable gravity
      auto C0                = coupler.get_option<real>("C0"     );    // pressure = C0*pow(rho*theta,gamma)
      auto grav              = coupler.get_option<real>("grav"   );    // Gravity
      auto gamma             = coupler.get_option<real>("gamma_d");    // cp_dry / cv_dry (about 1.4)
      auto latitude          = coupler.get_option<real>("latitude",0); // For coriolis
      auto &dm               = coupler.get_data_manager_readonly();    // Grab read-only data manager
      auto immersed_prop     = dm.get<real const,3>("dycore_immersed_proportion_halos"); // Immersed Proportion
      auto any_immersed2     = dm.get<bool const,3>("dycore_any_immersed2" ); // Are any immersed in 3-D halo within 2 cells?
      auto any_immersed4     = dm.get<bool const,3>("dycore_any_immersed4" ); // Are any immersed in 3-D halo within 4 cells?
      auto any_immersed6     = dm.get<bool const,3>("dycore_any_immersed6" ); // Are any immersed in 3-D halo within 6 cells?
      auto any_immersed8     = dm.get<bool const,3>("dycore_any_immersed8" ); // Are any immersed in 3-D halo within 8 cells?
      auto any_immersed10    = dm.get<bool const,3>("dycore_any_immersed10"); // Are any immersed in 3-D halo within 10 cells?
      auto hy_dens_cells     = dm.get<real const,1>("hy_dens_cells"        ); // Hydrostatic density in cells with halos
      auto hy_theta_cells    = dm.get<real const,1>("hy_theta_cells"       ); // Hydrostatic potential temperature in cells with halos
      auto hy_theta_edges    = dm.get<real const,1>("hy_theta_edges"       ); // Hydrostatic potential temperature at edges (no halos)
      auto hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells"    ); // Hydrostatic pressure in cells with halos
      auto metjac_edges      = dm.get<real const,2>("dycore_metjac_edges"  ); // Vertical metric jacobian at edges
      // Compute matrices to convert polynomial coefficients to 2 GLL points and stencil values to 2 GLL points
      // These matrices will be in column-row format. That performed better than row-column format in performance tests
      real r_dx = 1./dx; // reciprocal of grid spacing
      real r_dy = 1./dy; // reciprocal of grid spacing
      real fcor = 2*7.2921e-5*std::sin(latitude/180*M_PI);  // For coriolis: 2*Omega*sin(latitude)

      FLOC cs = coupler.get_option<real>("dycore_cs",350);  // Speed of sound

      int constexpr hsm1 = hs-1; // Halo size minus one

      // High-order edge interpolation weights for cell edges when not using WENO
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

      // The main working array that holds all prognostic variables plus pressure
      yakl::Array<FLOC,4> fields_loc("fields_loc",num_state+num_tracers+1,nz+2*hs,ny+2*hs,nx+2*hs);
      bool rsst = coupler.get_option<real>("dycore_cs",350) != 350; // Whether reduced speed of sound technique is being used

      // Load state and tracers into working array, dividing by density to get specific quantities, computing pressure,
      //  and subtracting hydrostatic values from density, potential temperature, and pressure
      // If Reduced Speed of Sound Technique (RSST) is being used, set pressure using cs^2 * (rho - rho_hydrostatic)
      //  Otherwise, use true pressure from equation of state
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        // Perturbation pressure if RSST is not used
        if (!rsst) fields_loc(idP,hs+k,hs+j,hs+i) = C0*std::pow(state(idT,k,j,i),gamma) - hy_pressure_cells(hs+k);
        real r_r = 1._fp / state(idR,k,j,i); // Reciprocal of density
        fields_loc(idR,hs+k,hs+j,hs+i) = state(idR,k,j,i);
        // Load in state and tracers as specific quantities
        for (int l=1; l < num_state  ; l++) { fields_loc(            l,hs+k,hs+j,hs+i) = state  (l,k,j,i)*r_r; }
        for (int l=0; l < num_tracers; l++) { fields_loc(num_state+1+l,hs+k,hs+j,hs+i) = tracers(l,k,j,i)*r_r; }
        // Remove hydrostasis from density and potential temperature
        fields_loc(idR,hs+k,hs+j,hs+i) -= hy_dens_cells (hs+k);
        fields_loc(idT,hs+k,hs+j,hs+i) -= hy_theta_cells(hs+k);
        // Perturbation pressure if RSST is used
        if (rsst) { fields_loc(idP,hs+k,hs+j,hs+i) = cs*cs*fields_loc(idR,hs+k,hs+j,hs+i); }
      });

      // Perform periodic halo exchange in the horizontal, and implement vertical no-slip solid wall boundary conditions
      #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
      yakl::timer_start("dycore_halo_exchange_x");
      #endif
      if (ord > 1) coupler.halo_exchange_x( fields_loc , hs ); // Halo exchange in x-direction
      #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
      yakl::timer_stop("dycore_halo_exchange_x");
      yakl::timer_start("dycore_halo_exchange_y");
      #endif
      if (ord > 1) coupler.halo_exchange_y( fields_loc , hs ); // Halo exchange in y-direction
      #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
      yakl::timer_stop("dycore_halo_exchange_y");
      #endif
      // Set all boundary conditions. istage and icycle are needed for proper halo exchanges between
      //  precursor and forced simulations
      halo_boundary_conditions( coupler , fields_loc , istage , icycle );

      // Storage for cell-edge fluxes in each direction
      yakl::Array<FLOC,4> flux_x("flux_x",num_state+num_tracers,nz,ny,nx+1);
      yakl::Array<FLOC,4> flux_y("flux_y",num_state+num_tracers,nz,ny+1,nx);
      yakl::Array<FLOC,4> flux_z("flux_z",num_state+num_tracers,nz+1,ny,nx);

      // Storage for cell-edge pressure in each direction
      yakl::Array<FLOC,3> p_x("p_x",nz,ny,nx+1);
      yakl::Array<FLOC,3> p_y("p_y",nz,ny+1,nx);
      yakl::Array<FLOC,3> p_z("p_z",nz+1,ny,nx);

      // Storage for cell-edge momentum in each direction
      yakl::Array<FLOC,3> ru_x("ru_x",nz,ny,nx+1);
      yakl::Array<FLOC,3> rv_y("rv_y",nz,ny+1,nx);
      yakl::Array<FLOC,3> rw_z("rw_z",nz+1,ny,nx);

      // Determine if the bottom and top boundaries are solid walls
      auto wall_z1 = coupler.get_option<std::string>("bc_z1") == "wall_free_slip";
      auto wall_z2 = coupler.get_option<std::string>("bc_z2") == "wall_free_slip";
      typedef limiter::WenoLimiter<FLOC,ord> Limiter; // Declare the WENO limiter
      auto use_weno = coupler.get_option<bool>("dycore_use_weno",true); // Whether to use WENO limiter
      auto imm_weno = coupler.get_option<bool>("dycore_use_weno_immersed",false); // Whether to use WENO limiter

      /////////////////////////////////////////////////////////////////////////////////////////
      // COMPUTE UPWIND CELL_EDGE PRESSURE AND MOMENTUM (ACOUSTIC UPWINDING)
      /////////////////////////////////////////////////////////////////////////////////////////

      // Reconstruct upwind cell-edge pressure and momentum in x-direction
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx+1) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool,1,ord> immersed; // Whether a stencil cell is immersed
        SArray<FLOC,1,ord> s;        // Stencil values
        
        // Load the stencils for cell immersion and pressure with the cell to the left of the edge as the center cell
        for (int ii = 0; ii < ord; ii++) { immersed(ii) = immersed_prop (hs+k,hs+j,i+ii) > 0; }
        for (int ii = 0; ii < ord; ii++) { s       (ii) = fields_loc(idP,hs+k,hs+j,i+ii); }
        bool do_map = immersed(hsm1-1) || immersed(hsm1+1);
        // Upon encountering an immersed boundary, set zero derivative boundary conditions from there out in that direction
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_L, dummy;  // To hold left pressure and dummy right pressure
        if (use_weno || (imm_weno && any_immersed6(k,j,i))) {
          Limiter::compute_limited_edges( s , dummy , p_L , { do_map , false , false } );
        } else {
          p_L = 0;
          for (int ii=0; ii < ord; ii++) { p_L += wt(ord-1-ii)*s(ii); }
        }

        // Load the stencil for momentum with the cell to the left of the edge as the center cell
        for (int ii = 0; ii < ord; ii++) { s(ii) = (fields_loc(idR,hs+k,hs+j,i+ii)+hy_dens_cells(hs+k))*
                                                    fields_loc(idU,hs+k,hs+j,i+ii); }
        // Non-WENO reconstruction of momentum at this edge from the left side
        FLOC ru_L = 0;
        // if (use_weno) {
        //   Limiter::compute_limited_edges( s , dummy , ru_L , { do_map , immersed(hsm1-1) , immersed(hsm1+1) } );
        // } else {
          for (int ii=0; ii < ord; ii++) { ru_L += wt(ord-1-ii)*s(ii); }
        // }

        // Load the stencils for cell immersion and pressure with the cell to the right of the edge as the center cell
        for (int ii = 0; ii < ord; ii++) { immersed(ii) = immersed_prop (hs+k,hs+j,i+ii+1) > 0; }
        for (int ii = 0; ii < ord; ii++) { s       (ii) = fields_loc(idP,hs+k,hs+j,i+ii+1); }
        do_map = immersed(hsm1-1) || immersed(hsm1+1);
        // Upon encountering an immersed boundary, set zero derivative boundary conditions from there out in that direction
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_R; // To hold right pressure
        if (use_weno || (imm_weno && any_immersed6(k,j,i))) {
          Limiter::compute_limited_edges( s , p_R , dummy , { do_map , false , false } );
        } else {
          p_R = 0;
          for (int ii=0; ii < ord; ii++) { p_R += wt(ii)*s(ii); }
        }

        // Load the stencil for momentum with the cell to the right of the edge as the center cell
        for (int ii = 0; ii < ord; ii++) { s(ii) = (fields_loc(idR,hs+k,hs+j,i+ii+1)+hy_dens_cells(hs+k))*
                                                    fields_loc(idU,hs+k,hs+j,i+ii+1); }
        // Non-WENO reconstruction of momentum at this edge from the right side
        FLOC ru_R = 0;
        // if (use_weno) {
        //   Limiter::compute_limited_edges( s , ru_R , dummy , { do_map , immersed(hsm1-1) , immersed(hsm1+1) } );
        // } else {
          for (int ii=0; ii < ord; ii++) { ru_R += wt(ii)*s(ii); }
        // }
        // Compute the upwind state of pressure and momentum at this edge
        p_x (k,j,i) = 0.5f*(p_L  + p_R  - cs*(ru_R-ru_L)   );
        ru_x(k,j,i) = 0.5f*(ru_L + ru_R -    (p_R -p_L )/cs);
      });

      // Reconstruct upwind cell-edge pressure and momentum in y-direction
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny+1,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool ,1,ord> immersed; // Whether a stencil cell is immersed
        SArray<FLOC,1,ord> s;         // Stencil values

        // Load the stencils for cell immersion and pressure with the cell left of the edge as the center cell
        for (int jj = 0; jj < ord; jj++) { immersed(jj) = immersed_prop (hs+k,j+jj,hs+i) > 0; }
        for (int jj = 0; jj < ord; jj++) { s       (jj) = fields_loc(idP,hs+k,j+jj,hs+i); }
        bool do_map = immersed(hsm1-1) || immersed(hsm1+1);
        // Upon encountering an immersed boundary, set zero derivative boundary conditions from there out in that direction
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_L, dummy; // To hold left pressure and dummy right pressure
        if (use_weno || (imm_weno && any_immersed6(k,j,i))) {
          Limiter::compute_limited_edges( s , dummy , p_L , { do_map , false , false } );
        } else {
          p_L = 0;
          for (int jj=0; jj < ord; jj++) { p_L += wt(ord-1-jj)*s(jj); }
        }

        // Load the stencil for momentum with the cell left of the edge as the center cell
        for (int jj = 0; jj < ord; jj++) { s(jj) = (fields_loc(idR,hs+k,j+jj,hs+i)+hy_dens_cells(hs+k))*
                                                    fields_loc(idV,hs+k,j+jj,hs+i); }
        // Non-WENO reconstruction of momentum at this edge from the left side
        FLOC rv_L = 0;
        // if (use_weno) {
        //   Limiter::compute_limited_edges( s , dummy , rv_L , { do_map , immersed(hsm1-1) , immersed(hsm1+1) } );
        // } else {
          for (int jj=0; jj < ord; jj++) { rv_L += wt(ord-1-jj)*s(jj); }
        // }

        // Load the stencils for cell immersion and pressure with the cell right of the edge as the center cell
        for (int jj = 0; jj < ord; jj++) { immersed(jj) = immersed_prop (hs+k,j+jj+1,hs+i) > 0; }
        for (int jj = 0; jj < ord; jj++) { s       (jj) = fields_loc(idP,hs+k,j+jj+1,hs+i); }
        do_map = immersed(hsm1-1) || immersed(hsm1+1);
        // Upon encountering an immersed boundary, set zero derivative boundary conditions from there out in that direction
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_R; // To hold right pressure
        if (use_weno || (imm_weno && any_immersed6(k,j,i))) {
          Limiter::compute_limited_edges( s , p_R , dummy , { do_map , false , false } );
        } else {
          p_R = 0;
          for (int jj=0; jj < ord; jj++) { p_R += wt(jj)*s(jj); }
        }

        // Load the stencil for momentum with the cell right of the edge as the center cell
        for (int jj = 0; jj < ord; jj++) { s(jj) = (fields_loc(idR,hs+k,j+jj+1,hs+i)+hy_dens_cells(hs+k))*
                                                    fields_loc(idV,hs+k,j+jj+1,hs+i); }
        // Non-WENO reconstruction of momentum at this edge from the right side
        FLOC rv_R = 0;
        // if (use_weno) {
        //   Limiter::compute_limited_edges( s , rv_R , dummy , { do_map , immersed(hsm1-1) , immersed(hsm1+1) } );
        // } else {
          for (int jj=0; jj < ord; jj++) { rv_R += wt(jj)*s(jj); }
        // }
        // Compute the upwind state of pressure and momentum at this edge
        p_y (k,j,i) = 0.5f*(p_L  + p_R  - cs*(rv_R-rv_L)   );
        rv_y(k,j,i) = 0.5f*(rv_L + rv_R -    (p_R -p_L )/cs);
      });

      // Reconstruct upwind cell-edge pressure and momentum in z-direction
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool ,1,ord> immersed; // Whether a stencil cell is immersed
        SArray<FLOC,1,ord> s;         // Stencil values

        // Load the stencils for cell immersion and pressure with the cell left of the edge as the center cell
        for (int kk = 0; kk < ord; kk++) { immersed(kk) = immersed_prop (k+kk,hs+j,hs+i) > 0; }
        for (int kk = 0; kk < ord; kk++) { s       (kk) = fields_loc(idP,k+kk,hs+j,hs+i); }
        for (int kk = 0; kk < ord; kk++) { s       (kk) *= dz(std::max(0,std::min(nz-1,k-hsm1-1+kk)))/dz(std::max(0,k-1)); }
        bool do_map = immersed(hsm1-1) || immersed(hsm1+1);
        // Upon encountering an immersed boundary, set zero derivative boundary conditions from there out in that direction
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_L, dummy; // To hold left pressure and dummy right pressure
        if (use_weno || (imm_weno && any_immersed6(k,j,i))) {
          Limiter::compute_limited_edges( s , dummy , p_L , { do_map , false , false } );
        } else {
          p_L = 0;
          for (int kk=0; kk < ord; kk++) { p_L += wt(ord-1-kk)*s(kk); }
        }
        p_L /= metjac_edges(1+k-1,1);

        // Load the stencil for momentum with the cell left of the edge as the center cell
        for (int kk = 0; kk < ord; kk++) { s(kk) = (fields_loc(idR,k+kk,hs+j,hs+i)+hy_dens_cells(k+kk))*
                                                    fields_loc(idW,k+kk,hs+j,hs+i); }
        // Multiply by normalized grid spacing to transform into zeta space
        for (int kk = 0; kk < ord; kk++) { s(kk) *= dz(std::max(0,std::min(nz-1,k-hsm1-1+kk)))/dz(std::max(0,k-1)); }
        // Non-WENO reconstruction of momentum at this edge from the left side
        FLOC rw_L = 0;
        // if (use_weno) {
        //   Limiter::compute_limited_edges( s , dummy , rw_L , { do_map , immersed(hsm1-1) , immersed(hsm1+1) } );
        // } else {
          for (int kk=0; kk < ord; kk++) { rw_L += wt(ord-1-kk)*s(kk); }
        // }
        rw_L /= metjac_edges(1+k-1,1);  // Divide by metric jacobian at this edge to transform to physical space
        if (wall_z1 && k == 0 ) rw_L = 0; // Impose wall boundary condition
        if (wall_z2 && k == nz) rw_L = 0; // Impose wall boundary condition

        // Load the stencils for cell immersion and pressure with the cell right of the edge as the center cell
        for (int kk = 0; kk < ord; kk++) { immersed(kk) = immersed_prop (k+kk+1,hs+j,hs+i) > 0; }
        for (int kk = 0; kk < ord; kk++) { s       (kk) = fields_loc(idP,k+kk+1,hs+j,hs+i); }
        // Multiply by normalized grid spacing to transform into zeta space
        for (int kk = 0; kk < ord; kk++) { s       (kk) *= dz(std::max(0,std::min(nz-1,k-hsm1+kk)))/dz(std::min(nz-1,k)); }
        do_map = immersed(hsm1-1) || immersed(hsm1+1);
        // Upon encountering an immersed boundary, set zero derivative boundary conditions from there out in that direction
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_R; // To hold right pressure
        if (use_weno || (imm_weno && any_immersed6(k,j,i))) {
          Limiter::compute_limited_edges( s , p_R , dummy , { do_map , false , false } );
        } else {
          p_R = 0;
          for (int kk=0; kk < ord; kk++) { p_R += wt(kk)*s(kk); }
        }
        p_R /= metjac_edges(1+k,0); // Divide by metric jacobian at this edge to transform to physical space

        // Load the stencil for momentum with the cell right of the edge as the center cell
        for (int kk = 0; kk < ord; kk++) { s(kk) = (fields_loc(idR,k+kk+1,hs+j,hs+i)+hy_dens_cells(k+kk+1))*
                                                    fields_loc(idW,k+kk+1,hs+j,hs+i); }
        // Multiply by normalized grid spacing to transform into zeta space
        for (int kk = 0; kk < ord; kk++) { s(kk) *= dz(std::max(0,std::min(nz-1,k-hsm1+kk)))/dz(std::min(nz-1,k)); }
        // Non-WENO reconstruction of momentum at this edge from the right side
        FLOC rw_R = 0;
        // if (use_weno) {
        //   Limiter::compute_limited_edges( s , rw_R , dummy , { do_map , immersed(hsm1-1) , immersed(hsm1+1) } );
        // } else {
          for (int kk=0; kk < ord; kk++) { rw_R += wt(kk)*s(kk); }
        // }
        rw_R /= metjac_edges(1+k,0); // Divide by metric jacobian at this edge to transform to physical space
        if (wall_z1 && k == 0 ) rw_R = 0; // Impose wall boundary condition
        if (wall_z2 && k == nz) rw_R = 0; // Impose wall boundary condition
        // Compute the upwind state of pressure and momentum at this edge
        p_z (k,j,i) = 0.5f*(p_L  + p_R  - cs*(rw_R-rw_L)   );
        rw_z(k,j,i) = 0.5f*(rw_L + rw_R -    (p_R -p_L )/cs);
        if (wall_z1 && k == 0 ) rw_z(k,j,i) = 0; // Impose wall boundary condition
        if (wall_z2 && k == nz) rw_z(k,j,i) = 0; // Impose wall boundary condition
      });

      //////////////////////////////////////////////////////////////////////////////////////////////
      // COMPUTE UPWIND ADVECTED QUANTITIES, AND COMPUTE TOTAL UPWIND FLUXES (ADVECTIVE UPWINDING)
      //////////////////////////////////////////////////////////////////////////////////////////////

      // Pressure will not be included in the advected fields, so accure a MultiField without pressure
      core::MultiField<FLOC,3> advect_fields;
      advect_fields.add_field( fields_loc.slice<3>(idR,0,0,0) ); // Think of these 0 indices as Fortran's (:,:,:)
      advect_fields.add_field( fields_loc.slice<3>(idU,0,0,0) );
      advect_fields.add_field( fields_loc.slice<3>(idV,0,0,0) );
      advect_fields.add_field( fields_loc.slice<3>(idW,0,0,0) );
      advect_fields.add_field( fields_loc.slice<3>(idT,0,0,0) );
      for (int tr=0; tr < num_tracers; tr++) { advect_fields.add_field( fields_loc.slice<3>(num_state+1+tr,0,0,0) ); }
      int num_fields = advect_fields.get_num_fields(); // This will be num_state+num_tracers

      // Reconstruct cell-edge advectively upwind advected quantities and compute total fluxes in x-direction
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx+1) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool ,1,ord> immersed; // Whether a stencil cell is immersed
        FLOC ru = ru_x(k,j,i);        // Acoustically upwinded momentum in x-direction
        int ind = ru > 0 ? 0 : 1;     // Determine index offset based on flow direction
        // Load the cell immersersion stencil based on upwind offset
        for (int ii = 0; ii < ord; ii++) { immersed(ii) = immersed_prop(hs+k,hs+j,i+ii+ind) > 0; }
        bool do_map = immersed(hsm1-1) || immersed(hsm1+1);
        for (int l=1; l < num_fields; l++) { // Loop over all advected fields except density
          // Gather the stencil values based on upwind offset
          SArray<FLOC,1,ord> s;
          for (int ii = 0; ii < ord; ii++) { s(ii) = advect_fields(l,hs+k,hs+j,i+ii+ind); }
          // For transverse velocities, modify stencil for immersed boundary zero-derivative condition (free-slip)
          if (l == idV || l == idW) modify_stencil_immersed_der0( s , immersed );
          FLOC val; // Reconstructed advected quantity at the edge
          if (use_weno || (imm_weno && any_immersed6(k,j,i))) {
            FLOC val_L, val_R;
            Limiter::compute_limited_edges( s , val_L , val_R , { do_map , immersed(hsm1-1) , immersed(hsm1+1) } );
            val = ru > 0 ? val_R : val_L;  // Choose value based on flow direction
          } else {
            val = 0;
            for (int ii=0; ii < ord; ii++) { val += wt(ru>0?ord-1-ii:ii)*s(ii); }
          }
          if (l == idT) val += hy_theta_cells(hs+k); // Add hydrostatic potential temperature back in
          flux_x(l,k,j,i) = ru*val;      // Compute total flux vector for advected fields
        }
        flux_x(idR,k,j,i)  = ru;         // Mass flux
        flux_x(idU,k,j,i) += p_x(k,j,i); // Momentum flux includes pressure
      });

      // Reconstruct cell-edge advectively upwind advected quantities and compute total fluxes in y-direction
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny+1,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool ,1,ord> immersed; // Whether a stencil cell is immersed
        FLOC rv = rv_y(k,j,i);        // Acoustically upwinded momentum in y-direction
        int ind = rv > 0 ? 0 : 1;     // Determine index offset based on flow direction
        // Load the cell immersion stencil based on upwind offset
        for (int jj = 0; jj < ord; jj++) { immersed(jj) = immersed_prop(hs+k,j+jj+ind,hs+i) > 0; }
        bool do_map = immersed(hsm1-1) || immersed(hsm1+1);
        for (int l=1; l < num_fields; l++) { // Loop over all advected fields except density
          // Gather the stencil values based on upwind offset
          SArray<FLOC,1,ord> s;
          for (int jj = 0; jj < ord; jj++) { s(jj) = advect_fields(l,hs+k,j+jj+ind,hs+i); }
          // For transverse velocities, modify stencil for immersed boundary zero-derivative condition (free-slip)
          if (l == idU || l == idW) modify_stencil_immersed_der0( s , immersed );
          FLOC val; // Reconstructed advected quantity at the edge
          if (use_weno || (imm_weno && any_immersed6(k,j,i))) {
            FLOC val_L, val_R;
            Limiter::compute_limited_edges( s , val_L , val_R , { do_map , immersed(hsm1-1) , immersed(hsm1+1) } );
            val = rv > 0 ? val_R : val_L; // Choose value based on flow direction
          } else {
            val = 0;
            for (int jj=0; jj < ord; jj++) { val += wt(rv>0?ord-1-jj:jj)*s(jj); }
          }
          if (l == idT) val += hy_theta_cells(hs+k); // Add hydrostatic potential temperature back in
          flux_y(l,k,j,i) = rv*val;       // Compute total flux vector for advected fields
        }
        flux_y(idR,k,j,i)  = rv;          // Mass flux
        flux_y(idV,k,j,i) += p_y(k,j,i);  // Momentum flux includes pressure
      });

      // Reconstruct cell-edge advectively upwind advected quantities and compute total fluxes in z-direction
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool ,1,ord> immersed; // Whether a stencil cell is immersed
        FLOC rw = rw_z(k,j,i);        // Acoustically upwinded momentum in z-direction
        int ind = rw > 0 ? 0 : 1;     // Determine index offset based on flow direction
        // Load the cell immersion stencil based on upwind offset
        for (int kk = 0; kk < ord; kk++) { immersed(kk) = immersed_prop(k+kk+ind,hs+j,hs+i) > 0; }
        bool do_map = immersed(hsm1-1) || immersed(hsm1+1);
        for (int l=1; l < num_fields; l++) { // Loop over all advected fields except density
          // Gather the stencil values based on upwind offset
          SArray<FLOC,1,ord> s;
          for (int kk = 0; kk < ord; kk++) { s(kk) = advect_fields(l,k+kk+ind,hs+j,hs+i); }
          // For transverse velocities, modify stencil for immersed boundary zero-derivative condition (free-slip)
          if (l == idU || l == idV) modify_stencil_immersed_der0( s , immersed );
          // Multiply by normalized grid spacing to transform into zeta space
          for (int kk = 0; kk < ord; kk++) { s(kk) *= dz(std::max(0,std::min(nz-1,k-hs+ind+kk)))/
                                                      dz(std::max(0,std::min(nz-1,k-1 +ind   ))); }
          FLOC val; // Reconstructed advected quantity at the edge
          if (use_weno || (imm_weno && any_immersed6(k,j,i))) {
            FLOC val_L, val_R;
            Limiter::compute_limited_edges( s , val_L , val_R , { do_map , immersed(hsm1-1) , immersed(hsm1+1) } );
            val = rw > 0 ? val_R : val_L; // Choose value based on flow direction
          } else {
            val = 0;
            for (int kk=0; kk < ord; kk++) { val += wt(rw>0?ord-1-kk:kk)*s(kk); }
          }
          // Divide by metric jacobian at this edge to transform to physical space
          val /= rw > 0 ? metjac_edges(1+k-1,1) : metjac_edges(1+k,0);
          if (l == idT)  val += hy_theta_edges(k); // Add hydrostatic potential temperature back in
          flux_z(l,k,j,i) = rw*val;       // Compute total flux vector for advected fields
        }
        flux_z(idR,k,j,i)  = rw;          // Mass flux
        flux_z(idW,k,j,i) += p_z(k,j,i);  // Momentum flux includes pressure
      });

      //////////////////////////////////////////////////////////////////////////////////////////////
      // COMPUTE TENDENCIES FROM FLUX DIVERGENCES AND SOURCE TERMS
      //////////////////////////////////////////////////////////////////////////////////////////////

      // Use g*rho*theta'/theta0 for buoyancy if desired or if RSST is being used
      auto buoy_theta = coupler.get_option<bool>("dycore_buoyancy_theta",false) || rsst;
      int mx = std::max(num_state,num_tracers);
      // Compute tendencies as the flux divergence + gravity source term + coriolis
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(mx,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          // Compute tendencies as the flux divergence
          state_tend(l,k,j,i) = -( flux_x(l,k,j,i+1) - flux_x(l,k,j,i) ) * r_dx
                                -( flux_y(l,k,j+1,i) - flux_y(l,k,j,i) ) * r_dy
                                -( flux_z(l,k+1,j,i) - flux_z(l,k,j,i) ) / dz(k);
          // Add gravity term to vertical momentum
          if (l == idW && enable_gravity) {
            if (buoy_theta) { // theta-based buoyancy
              real thetap = fields_loc(idT,hs+k,hs+j,hs+i);
              real rho    = state(idR,k,j,i);
              state_tend(l,k,j,i) += grav*rho*thetap/hy_theta_cells(hs+k);
            } else {          // density-based buoyancy
              state_tend(l,k,j,i) += -grav*fields_loc(idR,hs+k,hs+j,hs+i);
            }
          }
          // Add Coriolis terms to horizontal momenta
          if (latitude != 0 && l == idU) state_tend(l,k,j,i) += fcor*state(idV,k,j,i);
          if (latitude != 0 && l == idV) state_tend(l,k,j,i) -= fcor*state(idU,k,j,i);
        }
        if (l < num_tracers) {
          // Compute tendencies as the flux divergence
          tracers_tend(l,k,j,i) = -( flux_x(num_state+l,k,j,i+1) - flux_x(num_state+l,k,j,i) ) * r_dx
                                  -( flux_y(num_state+l,k,j+1,i) - flux_y(num_state+l,k,j,i) ) * r_dy 
                                  -( flux_z(num_state+l,k+1,j,i) - flux_z(num_state+l,k,j,i) ) / dz(k);
        }
      });

      //////////////////////////////////////////////////////////////////////////////////////////////
      // ADD HYPERVISCOSITY NEAR IMMERSED BOUNDARIES TO TENDENCIES
      //////////////////////////////////////////////////////////////////////////////////////////////

      if (coupler.get_option<bool>("dycore_immersed_hypervis",true)) {
        // Same as advected fields, the viscous fields do not include pressure
        core::MultiField<FLOC,3> fields_visc;
        for (int l=0; l < num_state  ; l++) { fields_visc.add_field(fields_loc.slice<3>(            l,0,0,0)); }
        for (int l=0; l < num_tracers; l++) { fields_visc.add_field(fields_loc.slice<3>(num_state+1+l,0,0,0)); }

        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
          FLOC hv_beta = 0;  // This is a multiplier to the maximum stable hyperviscosity coefficient
          // any_immersed* are pre-computed arrays that indicate if there is any immersed boundary within
          //  a certain number of cells of the current cell in any direction (including diagonals)
          // The goal is to apply stronger hyperviscosity the closer we are to an immersed boundary
          if      (any_immersed2 (k,j,i)) { hv_beta = 0.1f/2.f;  }
          else if (any_immersed4 (k,j,i)) { hv_beta = 0.1f/4.f;  }
          else if (any_immersed6 (k,j,i)) { hv_beta = 0.1f/8.f;  }
          else if (any_immersed8 (k,j,i)) { hv_beta = 0.1f/16.f; }
          else if (any_immersed10(k,j,i)) { hv_beta = 0.1f/32.f; }
          if (hv_beta > 0) {  // Only apply hyperviscosity if near an immersed boundary
            SArray<bool ,1,ord> immersed; // Whether a stencil cell is immersed
            // Gather cell immersion stencil in x-direction
            for (int ii = 0; ii < ord; ii++) { immersed(ii) = immersed_prop(hs+k,hs+j,1+i+ii) > 0; }
            for (int l=idU; l <= idW; l++) { // Loop over all advected fields
              SArray<FLOC,1,ord> s; // Stencil values
              // Gather stencil values in x-direction
              for (int ii = 0; ii < ord; ii++) { s(ii) = fields_visc(l,hs+k,hs+j,1+i+ii); }
              // For transverse velocities, modify stencil for immersed boundary zero-derivative condition
              //  to avoid mixing zero velocities without consideration of roughness length
              if (l==idV || l==idW) modify_stencil_immersed_der0( s , immersed );
              for (int ii = 0; ii < ord; ii++) { s(ii) *= fields_visc(idR,hs+k,hs+j,1+i+ii)+hy_dens_cells(hs+k); }
              // Apply hyperviscosity contribution to tendencies
              state_tend(l,k,j,i) += hv_beta*hypervis(s)/dt;
            }
            // Gather cell immersion stencil in y-direction
            for (int jj = 0; jj < ord; jj++) { immersed(jj) = immersed_prop(hs+k,1+j+jj,hs+i) > 0; }
            for (int l=idU; l <= idW; l++) { // Loop over all advected fields
              SArray<FLOC,1,ord> s; // Stencil values
              // Gather stencil values in y-direction
              for (int jj = 0; jj < ord; jj++) { s(jj) = fields_visc(l,hs+k,1+j+jj,hs+i); }
              // For transverse velocities, modify stencil for immersed boundary zero-derivative condition
              //  to avoid mixing zero velocities without consideration of roughness length
              if (l==idU || l==idW) modify_stencil_immersed_der0( s , immersed );
              for (int jj = 0; jj < ord; jj++) { s(jj) *= fields_visc(idR,hs+k,1+j+jj,hs+i)+hy_dens_cells(hs+k); }
              // Apply hyperviscosity contribution to tendencies
              state_tend(l,k,j,i) += hv_beta*hypervis(s)/dt;
            }
            // Gather cell immersion stencil in z-direction
            for (int kk = 0; kk < ord; kk++) { immersed(kk) = immersed_prop(1+k+kk,hs+j,hs+i) > 0; }
            for (int l=idU; l <= idW; l++) { // Loop over all advected fields
              SArray<FLOC,1,ord> s; // Stencil values
              // Gather stencil values in z-direction
              for (int kk = 0; kk < ord; kk++) { s(kk) = fields_visc(l,1+k+kk,hs+j,hs+i); }
              // For transverse velocities, modify stencil for immersed boundary zero-derivative condition
              //  to avoid mixing zero velocities without consideration of roughness length
              if (l==idU || l==idV) modify_stencil_immersed_der0( s , immersed );
              for (int kk = 0; kk < ord; kk++) { s(kk) *= fields_visc(idR,1+k+kk,hs+j,hs+i)+hy_dens_cells(1+k+kk); }
              // Apply hyperviscosity contribution to tendencies
              state_tend(l,k,j,i) += hv_beta*hypervis(s)/dt;
            }
          }
        });
      }

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("compute_tendencies");
      #endif
    }



    // Apply halo boundary conditions to the fields
    // Precursor BCs assume that the ghost cell data has already been copied into this coupler object
    //  before this function is called
    // coupler : reference to the coupler object
    // fields  : array of fields with halos
    // istage  : current RK stage
    // icycle  : current cycle number (for precursor data lookup)
    void halo_boundary_conditions( core::Coupler & coupler            ,
                                   yakl::Array<FLOC,4> const & fields ,
                                   int istage                         ,
                                   int icycle                         ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("halo_boundary_conditions");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx           = coupler.get_nx(); // Local number of cells in x-direction (not including halos)
      auto ny           = coupler.get_ny(); // Local number of cells in y-direction (not including halos)
      auto nz           = coupler.get_nz(); // Local number of cells in z-direction (not including halos)
      auto px           = coupler.get_px(); // MPI rank in x-direction
      auto py           = coupler.get_py(); // MPI rank in y-direction
      auto bc_x1        = coupler.get_option<std::string>("bc_x1"); // Boundary condition in west   x direction
      auto bc_x2        = coupler.get_option<std::string>("bc_x2"); // Boundary condition in east   x direction
      auto bc_y1        = coupler.get_option<std::string>("bc_y1"); // Boundary condition in south  y direction
      auto bc_y2        = coupler.get_option<std::string>("bc_y2"); // Boundary condition in north  y direction
      auto bc_z1        = coupler.get_option<std::string>("bc_z1"); // Boundary condition in bottom z direction
      auto bc_z2        = coupler.get_option<std::string>("bc_z2"); // Boundary condition in top    z direction
      auto nproc_x      = coupler.get_nproc_x();               // Number of MPI ranks in x-direction
      auto nproc_y      = coupler.get_nproc_y();               // Number of MPI ranks in y-direction
      auto num_tracers  = coupler.get_num_tracers();           // Number of tracer fields
      auto &dm          = coupler.get_data_manager_readonly(); // Get data manager as read-only

      // The halo exchange called before this has already handled periodic BCs
      // If this is a precursor-forced simulation, the ghost cells must have been copied to this coupler object
      //  before this function is called, so here we just need to copy them into the halo cells for inflow boundaries

      if (px == 0) { // If my rank is on the west edge of the domain
        if (bc_x1 == "periodic") { // Already handled in halo_exchange
        } else if (bc_x1 == "open" ) {
          // Simple zero-gradient extrapolation for open boundary
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,hs) ,
                                            KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            fields(l,hs+k,hs+j,hs-1-ii) = fields(l,hs+k,hs+j,hs+0);
          });
        } else if (bc_x1 == "precursor" ) {
          // For inflow boundaries, use precursor data in ghost cells except for pressure field
          // For outflow boundaries, use zero-gradient extrapolation
          auto prec_x1 = dm.get<FLOC const,6>("dycore_ghost_x1");
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,hs) ,
                                            KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            if (l!=idP) {
              auto u = fields(idU,hs+k,hs+j,hs);
              fields(l,hs+k,hs+j,hs-1-ii) = u > 0 ? prec_x1(icycle,istage,l,k,j,ii) : fields(l,hs+k,hs+j,hs+0);
            } else {
              fields(l,hs+k,hs+j,hs-1-ii) = fields(l,hs+k,hs+j,hs+0);
            }
          });
        } else {
          std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_x1 can only be periodic or open";
          Kokkos::abort("");
        }
      }

      if (px == nproc_x-1) { // If my rank is on the east edge of the domain
        if (bc_x2 == "periodic") { // Already handled in halo_exchange
        } else if (bc_x2 == "open" ) {
          // Simple zero-gradient extrapolation for open boundary
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,hs) ,
                                            KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            fields(l,hs+k,hs+j,hs+nx+ii) = fields(l,hs+k,hs+j,hs+nx-1);
          });
        } else if (bc_x2 == "precursor" ) {
          // For inflow boundaries, use precursor data in ghost cells except for pressure field
          // For outflow boundaries, use zero-gradient extrapolation
          auto prec_x2 = dm.get<FLOC const,6>("dycore_ghost_x2");
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,hs) ,
                                            KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            if (l!=idP) {
              auto u = fields(idU,hs+k,hs+j,hs+nx-1);
              fields(l,hs+k,hs+j,hs+nx+ii) = u > 0 ? fields(l,hs+k,hs+j,hs+nx-1) : prec_x2(icycle,istage,l,k,j,ii);
            } else {
              fields(l,hs+k,hs+j,hs+nx+ii) = fields(l,hs+k,hs+j,hs+nx-1);
            }
          });
        } else {
          std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_x2 can only be periodic or open";
          Kokkos::abort("");
        }
      }

      if (py == 0) { // If my rank is on the south edge of the domain
        if (bc_y1 == "periodic") { // Already handled in halo_exchange
        } else if (bc_y1 == "open" ) {
          // Simple zero-gradient extrapolation for open boundary
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                            KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            fields(l,hs+k,hs-1-jj,hs+i) = fields(l,hs+k,hs+0,hs+i);
          });
        } else if (bc_y1 == "precursor" ) {
          // For inflow boundaries, use precursor data in ghost cells except for pressure field
          // For outflow boundaries, use zero-gradient extrapolation
          auto prec_y1 = dm.get<FLOC const,6>("dycore_ghost_y1");
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                            KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            if (l!=idP) {
              auto v = fields(idV,hs+k,hs,hs+i);
              fields(l,hs+k,hs-1-jj,hs+i) = v > 0 ? prec_y1(icycle,istage,l,k,jj,i) : fields(l,hs+k,hs+0,hs+i);
            } else {
              fields(l,hs+k,hs-1-jj,hs+i) = fields(l,hs+k,hs+0,hs+i);
            }
          });
        } else {
          std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_y1 can only be periodic or open";
          Kokkos::abort("");
        }
      }

      if (py == nproc_y-1) { // If my rank is on the north edge of the domain
        if (bc_y2 == "periodic") { // Already handled in halo_exchange
        } else if (bc_y2 == "open" ) {
          // Simple zero-gradient extrapolation for open boundary
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                            KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            fields(l,hs+k,hs+ny+jj,hs+i) = fields(l,hs+k,hs+ny-1,hs+i);
          });
        } else if (bc_y2 == "precursor" ) {
          // For inflow boundaries, use precursor data in ghost cells except for pressure field
          // For outflow boundaries, use zero-gradient extrapolation
          auto prec_y2 = dm.get<FLOC const,6>("dycore_ghost_y2");
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                            KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            if (l!=idP) {
              auto v = fields(idV,hs+k,hs+ny-1,hs+i);
              fields(l,hs+k,hs+ny+jj,hs+i) = v > 0 ? fields(l,hs+k,hs+ny-1,hs+i) : prec_y2(icycle,istage,l,k,jj,i);
            } else {
              fields(l,hs+k,hs+ny+jj,hs+i) = fields(l,hs+k,hs+ny-1,hs+i);
            }
          });
        } else {
          std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_y2 can only be periodic or open";
          Kokkos::abort("");
        }
      }

      if (bc_z1 == "wall_free_slip") {
        // Free-slip wall boundary condition at bottom boundary
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,hs,ny,nx) ,
                                          KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          if (l == idW) {
            fields(l,kk,hs+j,hs+i) = 0;
          } else {
            fields(l,hs-1-kk,hs+j,hs+i) = fields(l,hs+0,hs+j,hs+i);
          }
        });
      } else if (bc_z1 == "periodic") {
        // Periodic boundary condition at bottom boundary
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,hs,ny,nx) ,
                                          KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          fields(l,kk,hs+j,hs+i) = fields(l,nz+kk,hs+j,hs+i);
        });
      } else {
        std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_z1 can only be periodic or wall_free_slip";
        Kokkos::abort("");
      }

      if (bc_z2 == "wall_free_slip") {
        // Free-slip wall boundary condition at top boundary
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,hs,ny,nx) ,
                                          KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          if (l == idW) {
            fields(l,hs+nz+kk,hs+j,hs+i) = 0;
          } else {
            fields(l,hs+nz+kk,hs+j,hs+i) = fields(l,hs+nz-1,hs+j,hs+i);
          }
        });
      } else if (bc_z2 == "periodic") {
        // Periodic boundary condition at top boundary
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,hs,ny,nx) ,
                                          KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          fields(l,hs+nz+kk,hs+j,hs+i) = fields(l,hs+kk,hs+j,hs+i);
        });
      } else {
        std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_z2 can only be periodic or wall_free_slip";
        Kokkos::abort("");
      }

      // If this is a precursor simualtion forcing another coupler object, then store the ghost cells
      //  into the coupler object for use by the other coupler object
      if (coupler.get_option<bool>("dycore_is_precursor",false)) {
        if (px == 0) {
          auto ghost_x1 = coupler.get_data_manager_readwrite().get<FLOC,6>("dycore_ghost_x1");
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,hs) ,
                                            KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            ghost_x1(icycle,istage,l,k,j,ii) = fields(l,hs+k,hs+j,hs-1-ii);
          });
        }
        if (px == nproc_x-1) {
          auto ghost_x2 = coupler.get_data_manager_readwrite().get<FLOC,6>("dycore_ghost_x2");
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,hs) ,
                                            KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            ghost_x2(icycle,istage,l,k,j,ii) = fields(l,hs+k,hs+j,hs+nx+ii);
          });
        }
        if (py == 0) {
          auto ghost_y1 = coupler.get_data_manager_readwrite().get<FLOC,6>("dycore_ghost_y1");
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                            KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            ghost_y1(icycle,istage,l,k,jj,i) = fields(l,hs+k,hs-1-jj,hs+i);
          });
        }
        if (py == nproc_y-1) {
          auto ghost_y2 = coupler.get_data_manager_readwrite().get<FLOC,6>("dycore_ghost_y2");
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                            KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            ghost_y2(icycle,istage,l,k,jj,i) = fields(l,hs+k,hs+ny+jj,hs+i);
          });
        }
      }

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("halo_boundary_conditions");
      #endif
    }



    // This computes the average column of the fields for ghost cell filling in idealized non-tubulent forcing simulations
    // coupler : reference to the coupler object
    // returns : average column of fields_loc from compute_tendencies
    real2d compute_average_ghost_column( core::Coupler & coupler ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx_glob           = coupler.get_nx_glob();  // Global number of cells in x-direction
      auto ny_glob           = coupler.get_ny_glob();  // Global number of cells in y-direction
      auto nx                = coupler.get_nx();       // Local number of cells in x-direction (not including halos)
      auto ny                = coupler.get_ny();       // Local number of cells in y-direction (not including halos)
      auto nz                = coupler.get_nz();       // Number of cells in z-direction (not including halos)
      auto C0                = coupler.get_option<real>("C0"     );  // pressure = C0*pow(rho*theta,gamma)
      auto gamma             = coupler.get_option<real>("gamma_d");  // cp_dry / cv_dry (about 1.4)
      auto cs                = coupler.get_option<real>("dycore_cs",350); // Speed of sound
      auto num_tracers       = coupler.get_num_tracers();   // Number of tracer fields
      // Hydrostatic pressure, density, and potential temperature over cells with halos
      auto hy_pressure_cells = coupler.get_data_manager_readonly().get<real const,1>("hy_pressure_cells");
      auto hy_dens_cells     = coupler.get_data_manager_readonly().get<real const,1>("hy_dens_cells");
      auto hy_theta_cells    = coupler.get_data_manager_readonly().get<real const,1>("hy_theta_cells");
      real4d state  ("state"  ,num_state  ,nz,ny,nx); // State variables
      real4d tracers("tracers",num_tracers,nz,ny,nx); // Tracer variables
      convert_coupler_to_dynamics( coupler , state , tracers ); // Convert coupler data to dynamics format
      real4d fields_loc("fields_loc",num_state+num_tracers+1,nz+2*hs,ny+2*hs,nx+2*hs); // Local fields with halos
      bool rsst = coupler.get_option<real>("dycore_cs",350) != 350; // Whether RSST is being used
      // Replicate the working array computation from compute_tendencies to get fields_loc populated
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        // Compute pressure perturbation if not using RSST
        if (!rsst) fields_loc(idP,hs+k,hs+j,hs+i) = C0*std::pow(state(idT,k,j,i),gamma) - hy_pressure_cells(hs+k);
        real r_r = 1._fp / state(idR,k,j,i); // Reciprocal of density
        fields_loc(idR,hs+k,hs+j,hs+i) = state(idR,k,j,i);
        // Store velocity, potential temperature, and tracers as specific quantities
        for (int l=1; l < num_state  ; l++) { fields_loc(            l,hs+k,hs+j,hs+i) = state  (l,k,j,i)*r_r; }
        for (int l=0; l < num_tracers; l++) { fields_loc(num_state+1+l,hs+k,hs+j,hs+i) = tracers(l,k,j,i)*r_r; }
        // Subtract hydrostatic contributions from density and potential temperature
        fields_loc(idR,hs+k,hs+j,hs+i) -= hy_dens_cells (hs+k);
        fields_loc(idT,hs+k,hs+j,hs+i) -= hy_theta_cells(hs+k);
        // If using RSST, compute perturbed pressure from perturbation density
        if (rsst) { fields_loc(idP,hs+k,hs+j,hs+i) = cs*cs*fields_loc(idR,hs+k,hs+j,hs+i); }
      });
      real2d ghost_col("ghost_col",num_state+num_tracers+1,nz); // Average column to return
      real r_nx_ny = 1./(nx_glob*ny_glob); // Reciprocal of global horizontal cell count
      // Compute average column for fields_loc
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(num_state+num_tracers+1,nz) , KOKKOS_LAMBDA (int l, int k) {
        ghost_col(l,k) = 0;
        for (int j=0; j < ny; j++) {
          for (int i=0; i < nx; i++) {
            ghost_col(l,k) += fields_loc(l,hs+k,hs+j,hs+i)*r_nx_ny;
          }
        }
      });
      // Sum across all MPI ranks to get global average column
      coupler.get_parallel_comm().all_reduce( ghost_col , MPI_SUM , "" ).deep_copy_to(ghost_col);
      Kokkos::fence();
      return ghost_col;
    }



    // For simulations forced by a concurrent turbulent precursor, copy the ghost cell data from the precursor coupler to the main coupler
    // coupler_prec : reference to the precursor coupler object
    // coupler_main : reference to the main coupler object
    void copy_precursor_ghost_cells( core::Coupler const & coupler_prec , core::Coupler & coupler_main ) {
      int  px          = coupler_main.get_px();       // MPI rank in x-direction
      int  py          = coupler_main.get_py();       // MPI rank in y-direction
      int  npx         = coupler_main.get_nproc_x();  // Number of MPI ranks in x-direction
      int  npy         = coupler_main.get_nproc_y();  // Number of MPI ranks in y-direction
      int  nstage      = coupler_main.get_option<int>("dycore_num_stages"); // Number of RK stages
      int  num_tracers = coupler_main.get_num_tracers(); // Number of tracer fields
      int  nx          = coupler_main.get_nx();       // Number of cells in x-direction
      int  ny          = coupler_main.get_ny();       // Number of cells in y-direction
      int  nz          = coupler_main.get_nz();       // Number of cells in z-direction
      auto &dm_prec    = coupler_prec.get_data_manager_readonly (); // Get precursor data manager as read-only
      auto &dm_main    = coupler_main.get_data_manager_readwrite(); // Get main data manager as read-write
      
      // Allocate ghost cell storage in main coupler if not already allocated
      if (px == 0) { // If my rank is on the west edge of the domain
        int ncycles = dm_prec.get<FLOC const,6>("dycore_ghost_x1").extent(0); // Number of cycles stored
        if (dm_main.entry_exists("dycore_ghost_x1")) { // If entry already exists, check size
          if (dm_main.get<FLOC const,6>("dycore_ghost_x1").extent(0) != ncycles) {
            dm_main.unregister_and_deallocate( "dycore_ghost_x1" );
          }
        }
        if (! dm_main.entry_exists("dycore_ghost_x1")) { // If entry does not exist, register and allocate it
          dm_main.register_and_allocate<FLOC>("dycore_ghost_x1","",{ncycles,nstage,num_state+num_tracers+1,nz,ny,hs});
        }
      }
      if (px == npx-1) { // If my rank is on the east edge of the domain
        int ncycles = dm_prec.get<FLOC const,6>("dycore_ghost_x2").extent(0);
        if (dm_main.entry_exists("dycore_ghost_x2")) { // If entry already exists, check size
          if (dm_main.get<FLOC const,6>("dycore_ghost_x2").extent(0) != ncycles) {
            dm_main.unregister_and_deallocate( "dycore_ghost_x2" );
          }
        }
        if (! dm_main.entry_exists("dycore_ghost_x2")) { // If entry does not exist, register and allocate it
          dm_main.register_and_allocate<FLOC>("dycore_ghost_x2","",{ncycles,nstage,num_state+num_tracers+1,nz,ny,hs});
        }
      }
      if (py == 0) { // If my rank is on the south edge of the domain
        int ncycles = dm_prec.get<FLOC const,6>("dycore_ghost_y1").extent(0);
        if (dm_main.entry_exists("dycore_ghost_y1")) { // If entry already exists, check size
          if (dm_main.get<FLOC const,6>("dycore_ghost_y1").extent(0) != ncycles) {
            dm_main.unregister_and_deallocate( "dycore_ghost_y1" );
          }
        }
        if (! dm_main.entry_exists("dycore_ghost_y1")) { // If entry does not exist, register and allocate it
          dm_main.register_and_allocate<FLOC>("dycore_ghost_y1","",{ncycles,nstage,num_state+num_tracers+1,nz,hs,nx});
        }
      }
      if (py == npy-1) { // If my rank is on the north edge of the domain
        int ncycles = dm_prec.get<FLOC const,6>("dycore_ghost_y2").extent(0);
        if (dm_main.entry_exists("dycore_ghost_y2")) { // If entry already exists, check size
          if (dm_main.get<FLOC const,6>("dycore_ghost_y2").extent(0) != ncycles) {
            dm_main.unregister_and_deallocate( "dycore_ghost_y2" );
          }
        }
        if (! dm_main.entry_exists("dycore_ghost_y2")) { // If entry does not exist, register and allocate it
          dm_main.register_and_allocate<FLOC>("dycore_ghost_y2","",{ncycles,nstage,num_state+num_tracers+1,nz,hs,nx});
        }
      }

      // Copy ghost cell data from precursor coupler to main coupler
      if (px == 0    ) dm_prec.get<FLOC const,6>("dycore_ghost_x1").deep_copy_to(dm_main.get<FLOC,6>("dycore_ghost_x1"));
      if (px == npx-1) dm_prec.get<FLOC const,6>("dycore_ghost_x2").deep_copy_to(dm_main.get<FLOC,6>("dycore_ghost_x2"));
      if (py == 0    ) dm_prec.get<FLOC const,6>("dycore_ghost_y1").deep_copy_to(dm_main.get<FLOC,6>("dycore_ghost_y1"));
      if (py == npy-1) dm_prec.get<FLOC const,6>("dycore_ghost_y2").deep_copy_to(dm_main.get<FLOC,6>("dycore_ghost_y2"));
    }



    // Initialize the class data as well as the state and tracers arrays and convert them back into the coupler state
    // coupler : reference to the coupler object
    // Make sure that all tracers are registered in the coupler before calling this function
    // This should be called after initializing the model data but before perturbing the initial conditions for
    //  things like thermals or initial potential temperature perturbations to initiate turbulence
    //  so that the hydrostatic profiles are accurately computed
    void init(core::Coupler &coupler) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("init");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx             = coupler.get_nx();       // Local number of cells in x-direction (not including halos)
      auto ny             = coupler.get_ny();       // Local number of cells in y-direction (not including halos)
      auto nz             = coupler.get_nz();       // Local number of cells in z-direction (not including halos)
      auto dz             = coupler.get_dz();       // Cell thicknesses in z-direction (1-D array of length nz)
      auto px             = coupler.get_px();       // MPI rank in x-direction
      auto py             = coupler.get_py();       // MPI rank in y-direction
      auto nproc_x        = coupler.get_nproc_x();  // Number of MPI ranks in x-direction
      auto nproc_y        = coupler.get_nproc_y();  // Number of MPI ranks in y-direction
      auto nx_glob        = coupler.get_nx_glob();  // Global number of cells in x-direction
      auto ny_glob        = coupler.get_ny_glob();  // Global number of cells in y-direction
      auto num_tracers    = coupler.get_num_tracers();  // Number of tracer fields
      auto gamma          = coupler.get_option<real>("gamma_d"); // cp_dry / cv_dry (about 1.4)
      auto C0             = coupler.get_option<real>("C0"     ); // pressure = C0*pow(rho*theta,gamma)
      auto grav           = coupler.get_option<real>("grav"   ); // Gravitational acceleration
      auto enable_gravity = coupler.get_option<bool>("enable_gravity",true); // Whether gravity is enabled
      auto &dm            = coupler.get_data_manager_readwrite(); // Get data manager as read-write
      auto tracer_names   = coupler.get_tracer_names();           // Get tracer names from coupler (std::vector<std::string>)
      // Get the time stepping scheme to set num_stages
      auto time_stepper   = coupler.get_option<std::string>("dycore_time_stepper","ssprk3");

      // Set the number of stages based on the time stepping scheme
      if      (time_stepper == "ssprk3") { coupler.set_option("dycore_num_stages",3);    }
      else if (time_stepper == "linrk3") { coupler.set_option("dycore_num_stages",3);    }
      else if (time_stepper == "linrk4") { coupler.set_option("dycore_num_stages",4);    }
      else                               { Kokkos::abort("Invalid dycore_time_stepper"); }

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

      coupler.set_option<int>("dycore_hs",hs); // Let other modules know the dycore halo size

      // Accumulate arrays that determine whethe each tracer adds mass and whether each tracer is positive definite
      // Do this on the host at first since it involves std::string operations
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
      // Copy to device, register in coupler data manager, and store in data manager memory
      tracer_positive_host .deep_copy_to(tracer_positive );
      tracer_adds_mass_host.deep_copy_to(tracer_adds_mass);
      dm.register_and_allocate<bool>("tracer_adds_mass","",{num_tracers});
      auto dm_tracer_adds_mass = dm.get<bool,1>("tracer_adds_mass");
      tracer_adds_mass.deep_copy_to(dm_tracer_adds_mass);
      dm.register_and_allocate<bool>("tracer_positive","",{num_tracers});
      auto dm_tracer_positive = dm.get<bool,1>("tracer_positive");
      tracer_positive.deep_copy_to(dm_tracer_positive);

      // Allocate state and tracer arrays, and convert coupler data to dynamics format for
      //  computing the initial hydrostatic profiles of density, potential temperature, and pressure
      real4d state  ("state"  ,num_state  ,nz,ny,nx);  state   = 0;
      real4d tracers("tracers",num_tracers,nz,ny,nx);  tracers = 0;
      convert_coupler_to_dynamics( coupler , state , tracers );
      // Compute the average column of density, potential temperature, and pressure for use
      //  in initializing the hydrostatic profiles including halo cells
      // The computation being here is why init should be called after initializing initial data
      //  but before applying perturbations to the flow
      dm.register_and_allocate<real>("hy_dens_cells"    ,"",{nz+2*hs});
      dm.register_and_allocate<real>("hy_theta_cells"   ,"",{nz+2*hs});
      dm.register_and_allocate<real>("hy_pressure_cells","",{nz+2*hs});
      auto r = dm.get<real,1>("hy_dens_cells"    );    r = 0;
      auto t = dm.get<real,1>("hy_theta_cells"   );    t = 0;
      auto p = dm.get<real,1>("hy_pressure_cells");    p = 0;
      // Local accumulations
      parallel_for( YAKL_AUTO_LABEL() , nz , KOKKOS_LAMBDA (int k) {
        for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
            r(hs+k) += state(idR,k,j,i);
            t(hs+k) += state(idT,k,j,i) / state(idR,k,j,i);
            p(hs+k) += C0 * std::pow( state(idT,k,j,i) , gamma );
          }
        }
      });
      // Global aggregations of sums
      coupler.get_parallel_comm().all_reduce( r , MPI_SUM ).deep_copy_to(r);
      coupler.get_parallel_comm().all_reduce( t , MPI_SUM ).deep_copy_to(t);
      coupler.get_parallel_comm().all_reduce( p , MPI_SUM ).deep_copy_to(p);
      // Computation of averages
      real r_nx_ny = 1./(nx_glob*ny_glob);
      parallel_for( YAKL_AUTO_LABEL() , nz , KOKKOS_LAMBDA (int k) {
        r(hs+k) *= r_nx_ny;
        t(hs+k) *= r_nx_ny;
        p(hs+k) *= r_nx_ny;
      });
      // Filling in the halo values using hydrostatic balance
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
          p(k) = C0*std::pow(r(k)*theta0,gamma);
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
          p(k) = C0*std::pow(r(k)*theta0,gamma);
        }
      });

      // This is a lambda function to create immersed proportion halos and any_immersed arrays for use
      //  by the dynamics module
      auto create_immersed_proportion_halos = [] (core::Coupler &coupler) {
        using yakl::c::parallel_for;
        using yakl::c::SimpleBounds;;
        auto nz     = coupler.get_nz  (); // Number of cells in z-direction (not including halos)
        auto ny     = coupler.get_ny  (); // Number of cells in y-direction (not including halos)
        auto nx     = coupler.get_nx  (); // Number of cells in x-direction (not including halos)
        auto &dm    = coupler.get_data_manager_readwrite(); // Get data manager as read-write
        auto wall_B = coupler.get_option<std::string>("bc_z1") == "wall_free_slip";
        auto wall_T = coupler.get_option<std::string>("bc_z1") == "wall_free_slip";
        if (!dm.entry_exists("dycore_immersed_proportion_halos")) {
          // Get the immersed_proportion field from the coupler data manager that is initialized before
          //  calling this module's init function
          auto immersed_prop = dm.get<real const,3>("immersed_proportion").createDeviceCopy<real>();
          // Create MultiField of just one field to hold the immersed proportion for halo creation and exchange
          core::MultiField<real,3> fields;
          fields.add_field( immersed_prop  );
          // Create and exchange halos (vertical is not set after calling this function)
          auto fields_halos = coupler.create_and_exchange_halos( fields , hs );
          // Create and populate dycore_immersed_proportion_halos in the coupler data manager
          dm.register_and_allocate<real>("dycore_immersed_proportion_halos","",{nz+2*hs,ny+2*hs,nx+2*hs},
                                         {"z_halod","y_halod","x_halod"});
          // Fill in the dycore's top and bottom halos with 1's to indicate fully immersed
          // This does not affect the immersed_proportion DataManager array since we are assigning to a different array
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) , KOKKOS_LAMBDA (int kk, int j, int i) {
            fields_halos(0,      kk,j,i) = wall_B ? 1 : 0;
            fields_halos(0,hs+nz+kk,j,i) = wall_T ? 1 : 0;
          });
          // Copy the field with halos into the coupler data manager array
          fields_halos.get_field(0).deep_copy_to( dm.get<real,3>("dycore_immersed_proportion_halos") );

          // The code sections below determine whether there is any immersed portion within varying halo sizes
          //  and store the results in separate arrays in the coupler data manager for use by the dynamics module
          // For each of these, when determining if there are immersed cells nearby, the top and bottom solid
          //  wall boundaries are not considered immersed.
          // These are only used to determine if / how hyperviscosity should be added near immersed boundaries.
          {
            int hsnew = 2;
            dm.register_and_allocate<bool>("dycore_any_immersed2","",{nz,ny,nx},{"z","y","x"});
            auto any_immersed = dm.get<bool,3>("dycore_any_immersed2");
            auto fields_halos_larger = coupler.create_and_exchange_halos( fields , hsnew );
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hsnew,ny+2*hsnew,nx+2*hsnew) ,
                                              KOKKOS_LAMBDA (int kk, int j, int i) {
              fields_halos_larger(0,         kk,j,i) = wall_B ? 1 : 0;
              fields_halos_larger(0,hsnew+nz+kk,j,i) = wall_T ? 1 : 0;
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
              fields_halos_larger(0,         kk,j,i) = wall_B ? 1 : 0;
              fields_halos_larger(0,hsnew+nz+kk,j,i) = wall_T ? 1 : 0;
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
              fields_halos_larger(0,         kk,j,i) = wall_B ? 1 : 0;
              fields_halos_larger(0,hsnew+nz+kk,j,i) = wall_T ? 1 : 0;
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
              fields_halos_larger(0,         kk,j,i) = wall_B ? 1 : 0;
              fields_halos_larger(0,hsnew+nz+kk,j,i) = wall_T ? 1 : 0;
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
              fields_halos_larger(0,         kk,j,i) = wall_B ? 1 : 0;
              fields_halos_larger(0,hsnew+nz+kk,j,i) = wall_T ? 1 : 0;
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

      // This lambda function is to interpolate hydrostatic profiles from cell centers to edges
      //  (linear for theta, and log-linear for rho and pressure)
      auto compute_hydrostasis_edges = [] (core::Coupler &coupler) {
        using yakl::c::parallel_for;
        using yakl::c::SimpleBounds;;
        auto nz   = coupler.get_nz  (); // Number of cells in z-direction (not including halos)
        auto ny   = coupler.get_ny  (); // Number of cells in y-direction (not including halos)
        auto nx   = coupler.get_nx  (); // Number of cells in x-direction (not including halos)
        auto &dm  = coupler.get_data_manager_readwrite(); // Get data manager as read-write
        // Register edge hydrostatic values if they do not already exist
        if (! dm.entry_exists("hy_dens_edges"    )) dm.register_and_allocate<real>("hy_dens_edges"    ,"",{nz+1});
        if (! dm.entry_exists("hy_theta_edges"   )) dm.register_and_allocate<real>("hy_theta_edges"   ,"",{nz+1});
        if (! dm.entry_exists("hy_pressure_edges")) dm.register_and_allocate<real>("hy_pressure_edges","",{nz+1});
        // Obtain the cells (with halows) and edges hydrostatic values
        auto hy_dens_cells     = dm.get<real const,1>("hy_dens_cells"    );
        auto hy_theta_cells    = dm.get<real const,1>("hy_theta_cells"   );
        auto hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells");
        auto hy_dens_edges     = dm.get<real      ,1>("hy_dens_edges"    );
        auto hy_theta_edges    = dm.get<real      ,1>("hy_theta_edges"   );
        auto hy_pressure_edges = dm.get<real      ,1>("hy_pressure_edges");
        // Interpolate from cell centers to edges
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

      // Call the two lambda functions created above to set up immersed proportion halos,
      //  any_immersed arrays, and compute hydrostatic edge values
      create_immersed_proportion_halos( coupler );
      compute_hydrostasis_edges       ( coupler );

      // Register immersed_proportion as an output and restart variable
      coupler.register_output_variable<real>( "immersed_proportion" , core::Coupler::DIMS_3D      );

      // Create an output module to be called during coupler.write_output() to write hydrostatic profiles
      //   and write perturbations of potential temperature, pressure, and density to file
      // coupler : reference to the coupler object
      // nc      : reference to the SimplePNetCDF object for writing output (open and not in define mode)
      coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        auto i_beg = coupler.get_i_beg(); // Get local starting indices in x and y directions
        auto j_beg = coupler.get_j_beg(); // Get local starting indices in x and y directions
        auto nz    = coupler.get_nz();    // Get local number of cells in z-direction (not including halos)
        auto ny    = coupler.get_ny();    // Get local number of cells in y-direction (not including halos)
        auto nx    = coupler.get_nx();    // Get local number of cells in x-direction (not including halos)
        nc.redef();  // re-enter define mode to add new dimensions and variables
        nc.create_dim( "z_halo" , coupler.get_nz()+2*hs );         // Vertical dimension with halos
        nc.create_var<real>( "hy_dens_cells"     , {"z_halo"});    // Define hydrostatic density variable
        nc.create_var<real>( "hy_theta_cells"    , {"z_halo"});    // Define hydrostatic potential temperature variable
        nc.create_var<real>( "hy_pressure_cells" , {"z_halo"});    // Define hydrostatic pressure variable
        // nc.create_var<real>( "theta_pert"        , {"z","y","x"}); // Define potential temperature perturbation variable
        // nc.create_var<real>( "pressure_pert"     , {"z","y","x"}); // Define pressure perturbation variable
        // nc.create_var<real>( "density_pert"      , {"z","y","x"}); // Define density perturbation variable
        nc.enddef(); // Exit define mode to write data
        nc.begin_indep_data(); // Enter independent data mode to write 1-D arrays from main task only
        auto &dm = coupler.get_data_manager_readonly(); // Get data manager as read-only
        // Write hydrostatic profiles from main task only
        if (coupler.is_mainproc()) nc.write( dm.get<real const,1>("hy_dens_cells"    ) , "hy_dens_cells"     );
        if (coupler.is_mainproc()) nc.write( dm.get<real const,1>("hy_theta_cells"   ) , "hy_theta_cells"    );
        if (coupler.is_mainproc()) nc.write( dm.get<real const,1>("hy_pressure_cells") , "hy_pressure_cells" );
        nc.end_indep_data(); // Exit independent data mode to write 3-D perturbation arrays
        // // Allocate state and tracer arrays, and convert coupler data to dynamics format to compute perturbations
        // real4d state  ("state"  ,num_state  ,nz,ny,nx);
        // real4d tracers("tracers",num_tracers,nz,ny,nx);
        // convert_coupler_to_dynamics( coupler , state , tracers );
        // // Define the offset for writing the 3-D perturbation arrays for this MPI rank
        // std::vector<MPI_Offset> start_3d = {0,(MPI_Offset)j_beg,(MPI_Offset)i_beg};
        // real3d data("data",nz,ny,nx); // Holds local 3-D perturbation data before writing
        // auto hy_dens_cells = dm.get<real const,1>("hy_dens_cells");
        // // Compute and write perturbation density
        // yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        //   data(k,j,i) = state(idR,k,j,i) - hy_dens_cells(hs+k);
        // });
        // nc.write_all(data,"density_pert",start_3d);
        // // Compute and write perturbation potential temperature
        // auto hy_theta_cells = dm.get<real const,1>("hy_theta_cells");
        // yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        //   data(k,j,i) = state(idT,k,j,i) / state(idR,k,j,i) - hy_theta_cells(hs+k);
        // });
        // nc.write_all(data,"theta_pert",start_3d);
        // // Compute and write perturbation pressure
        // auto hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells");
        // yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        //   data(k,j,i) = C0 * std::pow( state(idT,k,j,i) , gamma ) - hy_pressure_cells(hs+k);
        // });
        // nc.write_all(data,"pressure_pert",start_3d);
      } );

      // Register a restart module to read in hydrostatic profiles from file
      // coupler : reference to the coupler object
      // nc      : reference to the SimplePNetCDF object for reading restart data (opened)
      coupler.register_overwrite_with_restart_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        auto &dm = coupler.get_data_manager_readwrite();
        nc.read_all(dm.get<real,1>("hy_dens_cells"    ),"hy_dens_cells"    ,{0});
        nc.read_all(dm.get<real,1>("hy_theta_cells"   ),"hy_theta_cells"   ,{0});
        nc.read_all(dm.get<real,1>("hy_pressure_cells"),"hy_pressure_cells",{0});
        create_immersed_proportion_halos( coupler );
        compute_hydrostasis_edges       ( coupler );
      } );
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("init");
      #endif
    }



    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    // coupler : reference to the coupler object
    // state   : dynamics state array
    // tracers : dynamics tracers array
    void convert_dynamics_to_coupler( core::Coupler &coupler ,
                                      realConst4d    state   ,
                                      realConst4d    tracers ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("convert_dynamics_to_coupler");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto  nx          = coupler.get_nx();  // Number of cells in x-direction (not including halos)
      auto  ny          = coupler.get_ny();  // Number of cells in y-direction (not including halos)
      auto  nz          = coupler.get_nz();  // Number of cells in z-direction (not including halos)
      auto  R_d         = coupler.get_option<real>("R_d"    ); // Gas constant for dry air
      auto  R_v         = coupler.get_option<real>("R_v"    ); // Gas constant for water vapor
      auto  gamma       = coupler.get_option<real>("gamma_d"); // Ratio of specific heats for dry air
      auto  C0          = coupler.get_option<real>("C0"     ); // p = C0 * (rho*theta)^gamma
      auto  idWV        = coupler.get_option<int >("idWV"   ); // Tracer index for water vapor
      auto  num_tracers = coupler.get_num_tracers(); // Number of tracers
      auto  &dm = coupler.get_data_manager_readwrite(); // Get data manager as read-write
      auto  dm_rho_d = dm.get<real,3>("density_dry"); // Get coupler dry density array
      auto  dm_uvel  = dm.get<real,3>("uvel"       ); // Get coupler u-velocity array
      auto  dm_vvel  = dm.get<real,3>("vvel"       ); // Get coupler v-velocity array
      auto  dm_wvel  = dm.get<real,3>("wvel"       ); // Get coupler w-velocity array
      auto  dm_temp  = dm.get<real,3>("temp"       ); // Get coupler temperature array
      // Get array that determines whether each tracer adds to the mass of the air mixture
      auto  tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
      // Accrue the tracer fields from the coupler data manager
      core::MultiField<real,3> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real,3>(tracer_names.at(tr)) ); }
      // Loop over all grid cells to compute dry density, velocities, temperature, and store in coupler arrays
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real rho   = state(idR,k,j,i);        // Total density
        real u     = state(idU,k,j,i) / rho;  // u-velocity
        real v     = state(idV,k,j,i) / rho;  // v-velocity
        real w     = state(idW,k,j,i) / rho;  // w-velocity
        real theta = state(idT,k,j,i) / rho;  // Potential temperature
        real press = C0 * pow( rho*theta , gamma ); // Full pressure
        real rho_v = tracers(idWV,k,j,i);     // Water vapor density
        real rho_d = rho;                     // Dry air density starting value
        // Subtract mass-adding tracers from total density to get dry air density
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho_d -= tracers(tr,k,j,i); }
        // Use equation of state to compute temperature from pressure, dry density, and water vapor density
        real temp = press / ( rho_d * R_d + rho_v * R_v );
        dm_rho_d(k,j,i) = rho_d;  // Store dry air density in coupler array
        dm_uvel (k,j,i) = u;      // Store u-velocity in coupler array
        dm_vvel (k,j,i) = v;      // Store v-velocity in coupler array
        dm_wvel (k,j,i) = w;      // Store w-velocity in coupler array
        dm_temp (k,j,i) = temp;   // Store temperature in coupler array
        // Store tracer densities in coupler arrays
        for (int tr=0; tr < num_tracers; tr++) { dm_tracers(tr,k,j,i) = tracers(tr,k,j,i); }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("convert_dynamics_to_coupler");
      #endif
    }



    // Convert coupler's data to dynamics format of state and tracers arrays
    // coupler : reference to the coupler object
    // state   : dynamics state array
    // tracers : dynamics tracers array
    void convert_coupler_to_dynamics( core::Coupler const &coupler ,
                                      real4d              &state   ,
                                      real4d              &tracers ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("convert_coupler_to_dynamics");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto  nx          = coupler.get_nx(); // Number of cells in x-direction (not including halos)
      auto  ny          = coupler.get_ny(); // Number of cells in y-direction (not including halos)
      auto  nz          = coupler.get_nz(); // Number of cells in z-direction (not including halos)
      auto  R_d         = coupler.get_option<real>("R_d"    ); // Gas constant for dry air
      auto  R_v         = coupler.get_option<real>("R_v"    ); // Gas constant for water vapor
      auto  gamma       = coupler.get_option<real>("gamma_d"); // Ratio of specific heats for dry air
      auto  C0          = coupler.get_option<real>("C0"     ); // p = C0 * (rho*theta)^gamma
      auto  idWV        = coupler.get_option<int >("idWV"   ); // Tracer index for water vapor
      auto  num_tracers = coupler.get_num_tracers(); // Number of tracers
      auto  &dm = coupler.get_data_manager_readonly(); // Get data manager as read-only
      auto  dm_rho_d = dm.get<real const,3>("density_dry"); // Get coupler dry density array
      auto  dm_uvel  = dm.get<real const,3>("uvel"       ); // Get coupler u-velocity array
      auto  dm_vvel  = dm.get<real const,3>("vvel"       ); // Get coupler v-velocity array
      auto  dm_wvel  = dm.get<real const,3>("wvel"       ); // Get coupler w-velocity array
      auto  dm_temp  = dm.get<real const,3>("temp"       ); // Get coupler temperature array
      // Get array that determines whether each tracer adds to the mass of the air mixture
      auto  tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
      // Accrue the tracer fields from the coupler data manager
      core::MultiField<real const,3> dm_tracers;
      auto tracer_names = coupler.get_tracer_names(); // Get the tracer names
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real const,3>(tracer_names.at(tr)) ); }
      // Loop over all grid cells to compute dynamics state and tracers arrays from coupler data
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real rho_d = dm_rho_d(k,j,i); // Dry air density
        real u     = dm_uvel (k,j,i); // u-velocity
        real v     = dm_vvel (k,j,i); // v-velocity
        real w     = dm_wvel (k,j,i); // w-velocity
        real temp  = dm_temp (k,j,i); // Temperature
        real rho_v = dm_tracers(idWV,k,j,i); // Water vapor density
        real press = rho_d * R_d * temp + rho_v * R_v * temp; // Full pressure
        real rho = rho_d;              // Total density starting value
        // Add mass-adding tracers to dry density to get total density
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho += dm_tracers(tr,k,j,i); }
        // Compute potential temperature from pressure and total density
        real theta = pow( press/C0 , 1._fp / gamma ) / rho;
        state(idR,k,j,i) = rho;         // Store total density in dynamics state array
        state(idU,k,j,i) = rho * u;     // Store momentum in dynamics state array
        state(idV,k,j,i) = rho * v;     // Store momentum in dynamics state array
        state(idW,k,j,i) = rho * w;     // Store momentum in dynamics state array
        state(idT,k,j,i) = rho * theta; // Store total potential temperature in dynamics state array
        // Store tracer densities in dynamics tracers array
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,k,j,i) = dm_tracers(tr,k,j,i); }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("convert_coupler_to_dynamics");
      #endif
    }


  };

}


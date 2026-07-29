
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

  struct EulerCellCentered {
    // Order of accuracy (numerical convergence rate for smooth flows) for the dynamical core
    #ifndef PORTURB_ORD
      int static constexpr ord = 9;
    #else
      int static constexpr ord = PORTURB_ORD;
    #endif
    static_assert(ord == 3 || ord == 5 || ord == 7 || ord == 9,
                  "dynamics_rk_simpler requires ord to be 3, 5, 7, or 9");
    int static constexpr hs  = (ord+1)/2; // Number of halo cells ("hs" == "halo size")
    int static constexpr num_state = 5;   // Number of state variables
    // IDs for the variables in the state vector
    int  static constexpr idR = 0;  // Density
    int  static constexpr idU = 1;  // u-momentum
    int  static constexpr idV = 2;  // v-momentum
    int  static constexpr idW = 3;  // w-momentum
    int  static constexpr idT = 4;  // Density * potential temperature

    typedef float FLOC; // Use single precision locally


    // Increase precursor ghost-cell storage when the current sub-cycle exceeds its capacity
    void ensure_dycore_max_cycles(core::Coupler &coupler, int icycle) const;



    // Compute total mass of density and total mass of virtual potential temperature in the domain
    //  for verification purposes
    // coupler : Coupler instance
    // state   : State array from the dynamical core
    // Returns a tuple of summed density mass and virtual potential temperature mass
    std::tuple<real,real> compute_mass( core::Coupler & coupler , real4d const & state ) const;



    // Compute the time step based on CFL condition using a global minimum over the domain with static wave speed
    real compute_time_step( core::Coupler const &coupler ) const;
    // real compute_time_step( core::Coupler const &coupler ) const {
    //   using yakl::SimpleBounds;
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
    //   auto temp  = dm.get<real const,3>("temperature");
    //   real3d dt3d("dt3d",nz,ny,nx);
    //   real cfl = coupler.get_option<real>("cfl",0.15);
    //   real csconst = coupler.get_option<real>( "dycore_cs" , -1 );
    //   yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
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
    void time_step(core::Coupler &coupler, real dt_phys) const;



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
                        int             icycle  ) const;



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
                        int             icycle  ) const;



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
                           int             icycle  ) const;



    // Enforce immersed boundary conditions by relaxing variables toward hydrostasis at rest
    // coupler : Coupler instance
    // state   : State array from the dynamical core
    // tracers : Tracer array from the dynamical core
    void enforce_immersed_boundaries( core::Coupler       & coupler ,
                                      real4d        const & state   ,
                                      real4d        const & tracers ) const;



    // Once you encounter an immersed boundary, set zero derivative boundary conditions from there out in that direction
    // stencil  : Stencil array to modify
    // immersed : Boolean array indicating which points are immersed
    template <class FP, int ORD>
    KOKKOS_INLINE_FUNCTION static void modify_stencil_immersed_der0( SArray<FP  ,ORD>       & stencil  ,
                                                                     SArray<bool,ORD> const & immersed ) {
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
                             int                   icycle       ) const;



    // Apply halo boundary conditions to the fields
    // Precursor BCs assume that the ghost cell data has already been copied into this coupler object
    //  before this function is called
    // coupler : reference to the coupler object
    // fields  : array of fields with halos
    // istage  : current RK stage
    // icycle  : current cycle number (for precursor data lookup)
    void halo_boundary_conditions( core::Coupler & coupler               ,
                                   yakl::Array<FLOC ****> const & fields ,
                                   int istage                            ,
                                   int icycle                            ) const;



    // This computes the average column of the fields for ghost cell filling in idealized non-tubulent forcing simulations
    // coupler : reference to the coupler object
    // returns : average column of fields_loc from compute_tendencies
    real2d compute_average_ghost_column( core::Coupler & coupler );



    // For simulations forced by a concurrent turbulent precursor, copy the ghost cell data from the precursor coupler to the main coupler
    // coupler_prec : reference to the precursor coupler object
    // coupler_main : reference to the main coupler object
    void copy_precursor_ghost_cells( core::Coupler & coupler_prec , core::Coupler & coupler_main );



    // For simulations forced by a concurrent turbulent precursor, copy the ghost cell data from the precursor coupler to the main coupler
    // coupler_prec : reference to the precursor coupler object
    // coupler_main : reference to the main coupler object
    void copy_column_to_precursor_ghost_cells( core::Coupler & coupler , real2d const & col );



    // Refresh the dycore immersed-proportion field from the coupler and populate all halo cells.
    void create_immersed_proportion_halos(core::Coupler &coupler) const;



    // Initialize the class data as well as the state and tracers arrays and convert them back into the coupler state
    // coupler : reference to the coupler object
    // Make sure that all tracers are registered in the coupler before calling this function
    // This should be called after initializing the model data but before perturbing the initial conditions for
    //  things like thermals or initial potential temperature perturbations to initiate turbulence
    //  so that the hydrostatic profiles are accurately computed
    void init(core::Coupler &coupler) const;



    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    // coupler : reference to the coupler object
    // state   : dynamics state array
    // tracers : dynamics tracers array
    void convert_dynamics_to_coupler( core::Coupler &coupler ,
                                      realConst4d    state   ,
                                      realConst4d    tracers ) const;



    // Convert coupler's data to dynamics format of state and tracers arrays
    // coupler : reference to the coupler object
    // state   : dynamics state array
    // tracers : dynamics tracers array
    void convert_coupler_to_dynamics( core::Coupler const &coupler ,
                                      real4d              &state   ,
                                      real4d              &tracers ) const;


  };

}


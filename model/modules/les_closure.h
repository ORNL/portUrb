
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


    // Compute total mass of r and t fields for verification purposes
    // If mult_r is true, then t field is mass-weighted (i.e., rho*theta)
    // coupler : Coupler object
    // state   : 4D state array (num_state,nz+2*hs,ny+2*hs,nx+2*hs)
    // mult_r  : whether to multiply t field by r field
    // return   : tuple of (total mass of r field, total mass of t field)
    std::tuple<real,real> compute_mass( core::Coupler & coupler , real4d const & state , bool mult_r ) const;


    // Initialize LES closure module within the coupler
    // coupler : Coupler object
    // Sets up necessary variables and initial conditions for LES closure
    // Registers TKE tracer and initializes LES hydrostatic profiles
    // Assumes coupler has been initialized with grid and state variables
    // Be sure to call this init before the dynamics module's init so that it knows about TKE tracer
    // This also compute hydrostatic profiles based on initial coupler state so that operations are performed
    //   on perturbation potential temperature rather than full potential temperature
    void init( core::Coupler &coupler ) const;



    // Apply the 1-equation TKE-based LES closure to the state and tracers over one time step
    // coupler : Coupler object containing the data and options
    // dtphys  : Physical time step to advance the LES closure
    // Applies the LES closure to update the state and tracers in the coupler over the time step dtphys
    // This includes computing fluxes, updating TKE, and applying necessary boundary conditions
    void apply( core::Coupler &coupler , real dtphys ) const;



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
                                      real3d              &tke     ) const;



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
                                      realConst3d    tke     ) const;



    // Apply halo boundary conditions to state, tracers, and TKE arrays in all directions
    // Recall that the halo exchange has already performed periodic BCs in the horizontal direction
    //   and that the vertical halos are undefined before this routine is called
    // coupler : The coupler object containing the data
    // state   : Input/output state array (with halos)
    // tracers : Input/output tracers array (with halos)
    // tke     : Input/output TKE array (with halos)
    void halo_bcs( core::Coupler const & coupler ,
                   real4d        const & state   ,
                   real4d        const & tracers ,
                   real3d        const & tke     ) const;

  };

}


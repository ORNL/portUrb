
#pragma once

#include "coupler.h"

namespace modules {


  struct SurfaceFlux {

    int  static constexpr idR = 0;
    int  static constexpr idU = 1;
    int  static constexpr idV = 2;
    int  static constexpr idW = 3;
    int  static constexpr idT = 4;
    int  static constexpr num_state = 5;
    int  static constexpr hs = 1;
    real static constexpr neut_thresh = 0.05;



    void init( core::Coupler & coupler );
    


    // Applies surface fluxes of momenta and temperature from the model surface as well as
    //   immersed boundaries using Monin-Obukhov similarity theory
    // coupler : Coupler object containing the data manager and options
    // dt      : Timestep size in seconds
    void apply( core::Coupler &coupler   ,
                real dt                  );



    KOKKOS_INLINE_FUNCTION static void stability_correction( real vk , real mag , real z0 , real th , real th0 ,
                                                             real grav , real Czil , real nu , real dzloc ,
                                                             bool use_z0h , bool presc_wpthp , real sfc_wpthp ,
                                                             real & ustar , real & thstar ) {
      using std::sqrt;
      using std::log;
      using std::atan;
      if (presc_wpthp) thstar = -sfc_wpthp/std::max(ustar,1e-10);
      int  max_iter = 20;
      real beta_m   = 5;
      real beta_h   = 5;
      real gamma_m  = 16;
      real gamma_h  = 16;
      real zmin     = -5;
      real zmax     =  5;
      real tol      = 1e-6;
      for (int iter = 0; iter < max_iter; iter++) {
        real ustar_prev  = ustar ;
        real thstar_prev = thstar;
        real z0h   = use_z0h ? z0*std::exp(-vk*Czil*std::sqrt(ustar*z0/nu)) : z0;
        real wpthp = -ustar*thstar;
        wpthp      = std::copysign( std::max( std::abs(wpthp) , 1.e-6 ) , wpthp );
        real L     = -ustar*ustar*ustar*th/(vk*grav*wpthp);
        real zeta  = std::max(zmin,std::min(zmax,(dzloc/2+z0)/L));
        real psi_m_1, psi_h_1;
        if (std::abs(zeta) < neut_thresh) {
          psi_m_1 = 0;
          psi_h_1 = 0;
        } else if (zeta >= 0) {
          psi_m_1 = -beta_m*zeta;
          psi_h_1 = -beta_h*zeta;
        } else {
          real xm = sqrt(sqrt(1-gamma_m*zeta));
          real xh = sqrt(sqrt(1-gamma_h*zeta));
          psi_m_1 = 2*log((1+xm)/2) + log((1+xm*xm)/2) - 2*atan(xm) + M_PI/2;
          psi_h_1 = 2*log((1+xh*xh)/2);
        }
        zeta = std::max(zmin,std::min(zmax,z0/L));
        real psi_m_2;
        if (std::abs(zeta) < neut_thresh) {
          psi_m_2 = 0;
        } else if (zeta >= 0) {
          psi_m_2 = -beta_m*zeta;
        } else {
          real xm = sqrt(sqrt(1-gamma_m*zeta));
          psi_m_2 = 2*log((1+xm)/2) + log((1+xm*xm)/2) - 2*atan(xm) + M_PI/2;
        }
        zeta = std::max(zmin,std::min(zmax,z0h/L));
        real psi_h_2;
        if (std::abs(zeta) < neut_thresh) {
          psi_h_2 = 0;
        } else if (zeta >= 0) {
          psi_h_2 = -beta_h*zeta;
        } else {
          real xh = sqrt(sqrt(1-gamma_h*zeta));
          psi_h_2 = 2*log((1+xh*xh)/2);
        }
        ustar  = vk*mag     /std::max(1.e-3,log((dzloc/2+z0 )/z0 ) - psi_m_1 + psi_m_2);
        thstar = vk*(th-th0)/std::max(1.e-3,log((dzloc/2+z0h)/z0h) - psi_h_1 + psi_h_2);
        if (presc_wpthp) thstar = -sfc_wpthp/std::max(ustar,1e-10);
        if (std::abs(ustar-ustar_prev) <= tol && std::abs(thstar-thstar_prev) <= tol) break;
      }
    }



    void change_surface_theta( core::Coupler & coupler , real dt , real rate );



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


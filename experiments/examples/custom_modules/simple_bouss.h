
#include "main_header.h"

namespace custom_modules {

  // In the context of constant pressure, theta == temp except for acoustics, which we'll ignore
  inline void simple_bouss( core::Coupler & coupler , real dt ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto dz     = coupler.get_dz();
    auto nz     = coupler.get_nz();
    auto ny     = coupler.get_ny();
    auto nx     = coupler.get_nx();
    auto grav   = coupler.get_option<real>("grav");
    auto wvel   = coupler.get_data_manager_readwrite().get<real      ,3>("wvel");
    auto TKE    = coupler.get_data_manager_readwrite().get<real      ,3>("TKE");
    auto T      = coupler.get_data_manager_readonly ().get<real const,3>("temp");
    auto rho    = coupler.get_data_manager_readonly ().get<real const,3>("density_dry" );
    auto T0     = coupler.get_data_manager_readonly ().get<real const,1>("temp_0_bouss");
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      int  kp1 = std::min(nz-1,k+1);
      int  km1 = std::max(0   ,k-1);
      real dt_dz = ( T(kp1,j,i) - T(km1,j,i) ) / (2*dz);
      real t     = T(k,j,i);
      real K     = TKE(k,j,i)/rho(k,j,i);  // TKE is mass-weighted, so divide by density
      real N     = dt_dz >= 0 ? std::sqrt(grav/t*dt_dz) : 0;
      real ell   = std::min( 0.76_fp*std::sqrt(K)/(N+1.e-20_fp) , dz );
      real km    = 0.1_fp * ell * std::sqrt(K);
      real Pr_t  = dz / (1+2*ell);
      TKE (k,j,i) += -dt*(grav*rho(k,j,i)*km)/(t*Pr_t)*dt_dz;
      wvel(k,j,i) += dt*grav*(T(k,j,i)-T0(k))/T0(k);
    });
  }

}


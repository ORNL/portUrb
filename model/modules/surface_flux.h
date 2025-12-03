
#pragma once

#include "coupler.h"

namespace modules {

  // Applies surface fluxes of momenta and temperature from the model surface as well as
  //   immersed boundaries using Monin-Obukhov similarity theory
  // coupler : Coupler object containing the data manager and options
  // dt      : Timestep size in seconds
  inline void apply_surface_fluxes( core::Coupler &coupler , real dt ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    using yakl::c::Bounds;
    auto nx        = coupler.get_nx();  // Get local number of grid points in x-direction
    auto ny        = coupler.get_ny();  // Get local number of grid points in y-direction
    auto nz        = coupler.get_nz();  // Get number of grid points in z-direction
    auto dx        = coupler.get_dx();  // Get grid spacing in x-direction
    auto dy        = coupler.get_dy();  // Get grid spacing in y-direction
    auto dz        = coupler.get_dz();  // Get grid spacing in z-direction
    auto p0        = coupler.get_option<real>("p0");    // Reference pressure in Pa
    auto R_d       = coupler.get_option<real>("R_d");   // Gas constant for dry air in J/(kg·K)
    auto cp_d      = coupler.get_option<real>("cp_d");  // Specific heat at constant pressure for dry air in J/(kg·K)
    auto &dm       = coupler.get_data_manager_readwrite(); // Get reference to the data manager (read/write)
    auto dm_r      = dm.get<real const,3>("density_dry");  // Get dry air density array
    auto dm_u      = dm.get<real      ,3>("uvel");         // Get u-velocity array
    auto dm_v      = dm.get<real      ,3>("vvel");         // Get v-velocity array
    auto dm_w      = dm.get<real      ,3>("wvel");         // Get w-velocity array
    auto dm_T      = dm.get<real      ,3>("temp");         // Get temperature array
    auto imm_prop  = dm.get<real const,3>("immersed_proportion_halos"); // Get immersed boundary proportion array
    auto imm_rough = dm.get<real const,3>("immersed_roughness_halos" ); // Get immersed boundary roughness array
    auto imm_temp  = dm.get<real const,3>("immersed_temp_halos"      ); // Get immersed boundary temperature array
    int  hs        = 1; // Halo size for immersed boundary arrays

    // Allocate arrays to hold surface flux tendencies
    real3d tend_u("tend_u",nz,ny,nx);
    real3d tend_v("tend_v",nz,ny,nx);
    real3d tend_w("tend_w",nz,ny,nx);
    real3d tend_T("tend_T",nz,ny,nx);

    real vk = 0.40;   // von karman constant

    // Compute surface flux tendencies using Monin-Obukhov similarity theory
    // This applies surface friction to neighboring cells if they are the surface or if they are
    //   immersed. 
    parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      real r = dm_r(k,j,i);  // Dry air density at this grid point
      real u = dm_u(k,j,i);  // u-velocity at this grid point
      real v = dm_v(k,j,i);  // v-velocity at this grid point
      real w = dm_w(k,j,i);  // w-velocity at this grid point
      real T = dm_T(k,j,i);  // Temperature at this grid point
      // Initialize tendencies to zero prior to accumulation
      tend_u(k,j,i) = 0;
      tend_v(k,j,i) = 0;
      tend_w(k,j,i) = 0;
      tend_T(k,j,i) = 0;
      int indk, indj, indi;  // These indices will index into neighboring cells

      // West neighbor
      indk = hs+k;  indj = hs+j;  indi = hs+i-1;
      if (imm_prop(indk,indj,indi) > 0) {
        // Compute roughness length, log of height ratio, drag coefficient, immersed temperature,
        //   and adjacent transverse velocity magnitude
        real z0   = imm_rough(indk,indj,indi);
        real lgx  = std::log((dx/2+z0)/z0);
        real c_dx = vk*vk/(lgx*lgx);
        real T0   = imm_temp(indk,indj,indi);
        real mag  = std::sqrt(v*v+w*w);
        // Add momenta tendencies (friction)
        tend_v(k,j,i) += -c_dx*(v-0 )*mag/dx;
        tend_w(k,j,i) += -c_dx*(w-0 )*mag/dx;
        // If a temperature is speified for the immersed boundary, add temperature tendency
        if (T0 != 0) {
          real z0T  = 0.1*z0;
          real lgxT = std::log((dx/2+z0T)/z0T);
          real c_dxT = vk*vk/(lgx*lgxT);
          tend_T(k,j,i) += -c_dxT*(T-T0)*mag/dx;
        }
      }

      // East neighbor
      indk = hs+k;  indj = hs+j;  indi = hs+i+1;
      if (imm_prop(indk,indj,indi) > 0) {
        // Compute roughness length, log of height ratio, drag coefficient, immersed temperature,
        //   and adjacent transverse velocity magnitude
        real z0   = imm_rough(indk,indj,indi);
        real lgx  = std::log((dx/2+z0)/z0);
        real c_dx = vk*vk/(lgx*lgx);
        real T0   = imm_temp(indk,indj,indi);
        real mag  = std::sqrt(v*v+w*w);
        // Add momenta tendencies (friction)
        tend_v(k,j,i) += -c_dx*(v-0 )*mag/dx;
        tend_w(k,j,i) += -c_dx*(w-0 )*mag/dx;
        // If a temperature is speified for the immersed boundary, add temperature tendency
        if (T0 != 0) {
          real z0T  = 0.1*z0;
          real lgxT = std::log((dx/2+z0T)/z0T);
          real c_dxT = vk*vk/(lgx*lgxT);
          tend_T(k,j,i) += -c_dxT*(T-T0)*mag/dx;
        }
      }

      // South neighbor
      indk = hs+k;  indj = hs+j-1;  indi = hs+i;
      if (imm_prop(indk,indj,indi) > 0) {
        // Compute roughness length, log of height ratio, drag coefficient, immersed temperature,
        //   and adjacent transverse velocity magnitude
        real z0   = imm_rough(indk,indj,indi);
        real lgy  = std::log((dy/2+z0)/z0);
        real c_dy = vk*vk/(lgy*lgy);
        real T0   = imm_temp(indk,indj,indi);
        real mag  = std::sqrt(u*u+w*w);
        // Add momenta tendencies (friction)
        tend_u(k,j,i) += -c_dy*(u-0 )*mag/dy;
        tend_w(k,j,i) += -c_dy*(w-0 )*mag/dy;
        // If a temperature is speified for the immersed boundary, add temperature tendency
        if (T0 != 0) {
          real z0T  = 0.1*z0;
          real lgyT = std::log((dy/2+z0T)/z0T);
          real c_dyT = vk*vk/(lgy*lgyT);
          tend_T(k,j,i) += -c_dyT*(T-T0)*mag/dy;
        }
      }

      // North neighbor
      indk = hs+k;  indj = hs+j+1;  indi = hs+i;
      if (imm_prop(indk,indj,indi) > 0) {
        // Compute roughness length, log of height ratio, drag coefficient, immersed temperature,
        //   and adjacent transverse velocity magnitude
        real z0   = imm_rough(indk,indj,indi);
        real lgy  = std::log((dy/2+z0)/z0);
        real c_dy = vk*vk/(lgy*lgy);
        real T0   = imm_temp(indk,indj,indi);
        real mag  = std::sqrt(u*u+w*w);
        // Add momenta tendencies (friction)
        tend_u(k,j,i) += -c_dy*(u-0 )*mag/dy;
        tend_w(k,j,i) += -c_dy*(w-0 )*mag/dy;
        // If a temperature is speified for the immersed boundary, add temperature tendency
        if (T0 != 0) {
          real z0T  = 0.1*z0;
          real lgyT = std::log((dy/2+z0T)/z0T);
          real c_dyT = vk*vk/(lgy*lgyT);
          tend_T(k,j,i) += -c_dyT*(T-T0)*mag/dy;
        }
      }

      // Bottom neighbor
      indk = hs+k-1;  indj = hs+j;  indi = hs+i;
      if (imm_prop(indk,indj,indi) > 0) {
        // Compute roughness length, log of height ratio, drag coefficient, immersed temperature,
        //   and adjacent transverse velocity magnitude
        real z0   = imm_rough(indk,indj,indi);
        real lgz  = std::log((dz(k)/2+z0)/z0);
        real c_dz = vk*vk/(lgz*lgz);
        real T0   = imm_temp(indk,indj,indi);
        real mag  = std::sqrt(u*u+v*v);
        // Add momenta tendencies (friction)
        tend_u(k,j,i) += -c_dz*(u-0 )*mag/dz(k);
        tend_v(k,j,i) += -c_dz*(v-0 )*mag/dz(k);
        // If a temperature is speified for the immersed boundary, add temperature tendency
        if (T0 != 0) {
          real z0T  = 0.1*z0;
          real lgzT = std::log((dz(k)/2+z0T)/z0T);
          real c_dzT = vk*vk/(lgz*lgzT);
          tend_T(k,j,i) += -c_dzT*(T-T0)*mag/dz(k);
        }
      }
      
      // Top neighbor
      indk = hs+k+1;  indj = hs+j;  indi = hs+i;
      if (imm_prop(indk,indj,indi) > 0) {
        // Compute roughness length, log of height ratio, drag coefficient, immersed temperature,
        //   and adjacent transverse velocity magnitude
        real z0   = imm_rough(indk,indj,indi);
        real lgz  = std::log((dz(k)/2+z0)/z0);
        real c_dz = vk*vk/(lgz*lgz);
        real T0   = imm_temp(indk,indj,indi);
        real mag  = std::sqrt(u*u+v*v);
        // Add momenta tendencies (friction)
        tend_u(k,j,i) += -c_dz*(u-0 )*mag/dz(k);
        tend_v(k,j,i) += -c_dz*(v-0 )*mag/dz(k);
        // If a temperature is speified for the immersed boundary, add temperature tendency
        if (T0 != 0) {
          real z0T  = 0.1*z0;
          real lgzT = std::log((dz(k)/2+z0T)/z0T);
          real c_dzT = vk*vk/(lgz*lgzT);
          tend_T(k,j,i) += -c_dzT*(T-T0)*mag/dz(k);
        }
      }
    });

    // Apply the accumulated tendencies to the state variables
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      dm_u(k,j,i) += dt*tend_u(k,j,i);
      dm_v(k,j,i) += dt*tend_v(k,j,i);
      dm_w(k,j,i) += dt*tend_w(k,j,i);
      dm_T(k,j,i) += dt*tend_T(k,j,i);
    });

  };

}


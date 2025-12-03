#pragma once

#include "coupler.h"

namespace modules {
  
  // Applies a surface heat flux to the surface layer of the temperature field
  // coupler : core::Coupler object that holds shared variables and options
  // dt      : Timestep size (s)
  inline void surface_heat_flux( core::Coupler &coupler , real dt ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto shf = coupler.get_option<real>("sfc_heat_flux");  // Get the surface heat flux (K m / s)
    auto dm_temp = coupler.get_data_manager_readwrite().get<real,3>("temp"); // Get the temperature field (K)
    auto nx = coupler.get_nx();  // Get the local number of grid cells in the x-direction
    auto ny = coupler.get_ny();  // Get the local number of grid cells in the y-direction
    auto dz = coupler.get_dz();  // Get the vertical grid spacing (m) (1-D array of size nz)
    // Apply the surface heat flux to the surface model layer
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , KOKKOS_LAMBDA (int j, int i) {
      dm_temp(0,j,i) += dt*shf/dz(0);
    });
  }

}



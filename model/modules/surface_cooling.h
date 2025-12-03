#pragma once

#include "coupler.h"

namespace modules {
  
  // Applies surface cooling to the surface temperature and immersed boundary temperature fields
  // coupler : Reference to the core::Coupler object managing shared data and options
  // dt      : Timestep size in seconds
  inline void surface_cooling( core::Coupler &coupler , real dt ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    int nx = coupler.get_nx();  // Get the number of grid points in the x-direction
    int ny = coupler.get_ny();  // Get the number of grid points in the y-direction
    real rate = coupler.get_option<real>( "sfc_cool_rate" );
    // Get the various surface temperature fields and apply cooling
    auto sfc_temp       = coupler.get_data_manager_readwrite().get<real,2>("surface_temp"       );
    auto sfc_temp_halos = coupler.get_data_manager_readwrite().get<real,2>("surface_temp_halos" );
    auto sfc_imm_temp   = coupler.get_data_manager_readwrite().get<real,3>("immersed_temp_halos").slice<2>(0,0,0);
    int hs = (sfc_temp_halos.extent(0)-ny)/2;  // Compute halo size for haloed arrays
    // Apply cooling in parallel over all surface grid points
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny+2*hs,nx+2*hs) , KOKKOS_LAMBDA (int j, int i) {
      if (j < ny && i < nx) sfc_temp(j,i) -= dt*rate;
      sfc_temp_halos(j,i) -= dt*rate;
      sfc_imm_temp  (j,i) -= dt*rate;
    });
  }

}



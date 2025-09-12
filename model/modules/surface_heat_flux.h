#pragma once

#include "coupler.h"

namespace modules {
  
  inline void surface_heat_flux( core::Coupler &coupler , real dt ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto shf = coupler.get_option<real>("sfc_heat_flux");
    auto dm_temp = coupler.get_data_manager_readwrite().get<real,3>("temp");
    auto nx = coupler.get_nx();
    auto ny = coupler.get_ny();
    auto dz = coupler.get_dz();
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , KOKKOS_LAMBDA (int j, int i) {
      dm_temp(0,j,i) += dt*shf/dz;
    });
  }

}



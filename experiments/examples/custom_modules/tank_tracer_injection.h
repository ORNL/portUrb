
#pragma once

#include "coupler.h"

namespace custom_modules {
  
  inline void tank_tracer_injection( core::Coupler & coupler     ,
                                     real            dt          ,
                                     real            x1          ,
                                     real            x2          ,
                                     real            y1          ,
                                     real            y2          ,
                                     real            z1          ,
                                     real            z2          ,
                                     real            conc        ,
                                     real            wvel        ,
                                     std::string     tracer_name ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    using yakl::componentwise::operator/;  // Allows use of '/' on yakl::Array objects
    using yakl::componentwise::operator-;  // Allows use of '-' on yakl::Array objects
    auto nx      = coupler.get_nx();      // Get local number of cells in x-direction
    auto ny      = coupler.get_ny();      // Get local number of cells in y-direction
    auto nz      = coupler.get_nz();      // Get local number of cells in z-direction
    auto dx      = coupler.get_dx();      // Get grid spacing in x-direction
    auto dy      = coupler.get_dy();      // Get grid spacing in y-direction
    auto dz      = coupler.get_dz();      // Get grid spacing in z-direction
    auto zmid    = coupler.get_zmid();    // Get vertical grid mid points
    auto i_beg   = coupler.get_i_beg();   // Get global starting index in x-direction
    auto j_beg   = coupler.get_j_beg();   // Get global starting index in y-direction
    auto &dm     = coupler.get_data_manager_readwrite(); // Get DataManager for read/write access
    auto dm_trac = dm.get<real,3>(tracer_name);
    auto dm_w    = dm.get<real,3>("wvel");
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      real x = (i_beg+i+0.5)*dx;
      real y = (j_beg+j+0.5)*dy;
      real z = zmid(k);
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2 && z >= z1 && z <= z2) {
        dm_trac(k,j,i) = conc;
        dm_w   (k,j,i) = wvel;
      }
    });
  }
}



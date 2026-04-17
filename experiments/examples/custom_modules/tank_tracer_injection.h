
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
    using yakl::parallel_for;
    using yakl::SimpleBounds;
    using yakl::componentwise::operator/;  // Allows use of '/' on yakl::Array objects
    using yakl::componentwise::operator-;  // Allows use of '-' on yakl::Array objects
    auto nx       = coupler.get_nx();      // Get local number of cells in x-direction
    auto nx_glob  = coupler.get_nx_glob(); // Get global number of cells in x-direction
    auto ny       = coupler.get_ny();      // Get local number of cells in y-direction
    auto nz       = coupler.get_nz();      // Get local number of cells in z-direction
    auto dx       = coupler.get_dx();      // Get grid spacing in x-direction
    auto dy       = coupler.get_dy();      // Get grid spacing in y-direction
    auto dz       = coupler.get_dz();      // Get grid spacing in z-direction
    auto zmid     = coupler.get_zmid();    // Get vertical grid mid points
    auto i_beg    = coupler.get_i_beg();   // Get global starting index in x-direction
    auto j_beg    = coupler.get_j_beg();   // Get global starting index in y-direction
    auto &dm      = coupler.get_data_manager_readwrite(); // Get DataManager for read/write access
    auto dm_trac  = dm.get<real      ,3>(tracer_name);
    auto dm_rho_d = dm.get<real const,3>("density_dry");
    auto dm_w     = dm.get<real      ,3>("wvel");
    real x0       = (x1+x2)/2;
    real y0       = (y1+y2)/2;
    real sigma    = (x2-x1)/4;
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      real x = (i_beg+i+0.5)*dx;
      real y = (j_beg+j+0.5)*dy;
      real z = zmid(k);
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2 && z >= z1 && z <= z2) {
        dm_trac(k,j,i) = conc*dm_rho_d(k,j,i);
        dm_w   (k,j,i) = wvel*std::exp(-((x-x0)*(x-x0)+(y-y0)*(y-y0))/(2*sigma*sigma));
      }
      int start = std::round(0.95*nx_glob);
      if ((i_beg+i) >= start) dm_trac(k,j,i) *= (nx_glob-1-(i_beg+i)) / (nx_glob-1-start);
    });
  }
}



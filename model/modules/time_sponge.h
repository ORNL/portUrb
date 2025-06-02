
#pragma once

#include "coupler.h"

namespace modules {

  inline void time_sponge( core::Coupler &coupler          ,
                           real dt                         ,
                           real average_scale              ,
                           real forcing_scale              ,
                           std::vector<std::string> vnames ,
                           int cells_x1                    ,
                           int cells_x2                    ,
                           int cells_y1                    ,
                           int cells_y2                    ,
                           int cells_z2                    ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nz       = coupler.get_nz();
    auto ny       = coupler.get_ny();
    auto nx       = coupler.get_nx();
    auto i_beg    = coupler.get_i_beg();
    auto j_beg    = coupler.get_j_beg();
    auto nx_glob  = coupler.get_nx_glob();
    auto ny_glob  = coupler.get_ny_glob();
    auto &dm      = coupler.get_data_manager_readwrite();
    int  num_vars = vnames.size();

    core::MultiField<real,3> fields;
    for (int ivar=0; ivar < num_vars; ivar++) { fields.add_field( dm.get<real,3>(vnames.at(ivar)) ); }

    if (! dm.entry_exists("time_sponge_tavg")) {
      dm.register_and_allocate<real>("time_sponge_tavg","",{num_vars,nz,ny,nx});
      auto tavg = dm.get<real,4>("time_sponge_tavg");
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_vars,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        tavg(l,k,j,i) = fields(l,k,j,i);
      });
    }

    auto tavg = dm.get<real,4>("time_sponge_tavg");
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_vars,nz,ny,nx) ,
                                      KOKKOS_LAMBDA (int l, int k, int j, int i) {
      tavg(l,k,j,i) = dt/average_scale*fields(l,k,j,i) + (average_scale-dt)/average_scale*tavg(l,k,j,i);
    });

    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_vars,nz,ny,nx) ,
                                      KOKKOS_LAMBDA (int l, int k, int j, int i) {
      real wt = 0;
      if (i_beg+i             < cells_x1) wt = std::max(wt,(1+std::cos(M_PI*(i_beg+i)/cells_x1))/2);
      if (j_beg+j             < cells_y1) wt = std::max(wt,(1+std::cos(M_PI*(j_beg+j)/cells_y1))/2);
      if (nx_glob-1-(i_beg+i) < cells_x2) wt = std::max(wt,(1+std::cos(M_PI*(nx_glob-1-(i_beg+i))/cells_x2))/2);
      if (ny_glob-1-(j_beg+j) < cells_y2) wt = std::max(wt,(1+std::cos(M_PI*(ny_glob-1-(j_beg+j))/cells_y2))/2);
      if (nz     -1-(      k) < cells_z2) wt = std::max(wt,(1+std::cos(M_PI*(nz     -1-(      k))/cells_z2))/2);
      wt *= dt/forcing_scale;
      fields(l,k,j,i) = wt*tavg(l,k,j,i) + (1-wt)*fields(l,k,j,i);
    });
  }

}


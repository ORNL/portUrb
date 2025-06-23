
#pragma once

#include "coupler.h"

namespace custom_modules {
  
  inline void fluctuation_scaling( core::Coupler            & coupler ,
                                   real dt                            ,
                                   real frac                          ,
                                   real tscale                        ,
                                   std::vector<std::string>   vnames  ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    using yakl::componentwise::operator/;
    using yakl::componentwise::operator-;
    auto nx       = coupler.get_nx();
    auto ny       = coupler.get_ny();
    auto nz       = coupler.get_nz();
    auto i_beg    = coupler.get_i_beg();
    auto j_beg    = coupler.get_j_beg();
    auto nx_glob  = coupler.get_nx_glob();
    auto ny_glob  = coupler.get_ny_glob();
    auto &dm      = coupler.get_data_manager_readwrite();
    core::MultiField<real,3> fields;
    int numvars = vnames.size();
    for (int i=0; i < numvars; i++) { fields.add_field( dm.get<real,3>(vnames.at(i)) ); }

    real mult = std::pow(frac,dt/tscale);

    // Get the mean column for main values
    real2d col("main_col_mean",numvars,nz);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(numvars,nz) , KOKKOS_LAMBDA (int l, int k) {
      col(l,k) = 0;
      for (int j=0; j < ny; j++) {
        for (int i=0; i < nx; i++) {
          col(l,k) += fields(l,k,j,i);
        }
      }
    });

    col = coupler.get_parallel_comm().all_reduce(col,MPI_SUM)/(nx_glob*ny_glob);

    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(numvars,nz,ny,nx) , KOKKOS_LAMBDA (int l, int k, int j, int i) {
      fields(l,k,j,i) = col(l,k) + mult*(fields(l,k,j,i)-col(l,k));
    });

  }
}



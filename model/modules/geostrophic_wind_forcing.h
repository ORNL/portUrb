
#pragma once

#include "coupler.h"

namespace modules {

  inline real2d geostrophic_wind_forcing( core::Coupler &coupler , real dt , real lat_g , real u_g , real v_g ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nz      = coupler.get_nz();
    auto ny      = coupler.get_ny();
    auto nx      = coupler.get_nx();
    auto ny_glob = coupler.get_ny_glob();
    auto nx_glob = coupler.get_nx_glob();
    auto &dm     = coupler.get_data_manager_readwrite();
    auto uvel    = dm.get<real,3>("uvel");
    auto vvel    = dm.get<real,3>("vvel");
    real fcor    = 2*7.2921e-5*std::sin(lat_g/180*M_PI);
    int constexpr idU  = 0;
    int constexpr idV  = 1;
    int constexpr nfld = 2;
    real2d col("col",nfld,nz);
    real r_nx_ny = 1. / (ny_glob*nx_glob);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nfld,nz) , KOKKOS_LAMBDA (int v, int k) {
      col(v,k) = 0;
      for (int j=0; j < ny; j++) {
        for (int i=0; i < nx; i++) {
          if (v == idU) col(v,k) += uvel(k,j,i)*r_nx_ny;
          if (v == idV) col(v,k) += vvel(k,j,i)*r_nx_ny;
        }
      }
    });
    col = coupler.get_parallel_comm().all_reduce( col , MPI_SUM , "" );
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      uvel(k,j,i) += dt*( fcor*(col(idV,k)-v_g));
      vvel(k,j,i) += dt*(-fcor*(col(idU,k)-u_g));
    });
    return col;
  }

  inline void geostrophic_wind_forcing_specified( core::Coupler &coupler , real dt , real lat_g , real u_g , real v_g ,
                                                  real2d const &col ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nz      = coupler.get_nz();
    auto ny      = coupler.get_ny();
    auto nx      = coupler.get_nx();
    auto ny_glob = coupler.get_ny_glob();
    auto nx_glob = coupler.get_nx_glob();
    auto &dm     = coupler.get_data_manager_readwrite();
    auto uvel    = dm.get<real,3>("uvel");
    auto vvel    = dm.get<real,3>("vvel");
    real fcor    = 2*7.2921e-5*std::sin(lat_g/180*M_PI);
    int constexpr idU  = 0;
    int constexpr idV  = 1;
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      uvel(k,j,i) += dt*( fcor*(col(idV,k)-v_g));
      vvel(k,j,i) += dt*(-fcor*(col(idU,k)-u_g));
    });
  }

}


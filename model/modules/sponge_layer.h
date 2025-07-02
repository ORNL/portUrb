
#pragma once

#include "coupler.h"

namespace modules {

  inline void sponge_layer( core::Coupler &coupler , real dt , real time_scale , real top_prop ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    auto ny_glob = coupler.get_ny_glob();
    auto nx_glob = coupler.get_nx_glob();
    auto nz      = coupler.get_nz  ();
    auto ny      = coupler.get_ny  ();
    auto nx      = coupler.get_nx  ();
    auto zlen    = coupler.get_zlen();
    auto dz      = coupler.get_dz  ();
    auto &dm     = coupler.get_data_manager_readwrite();

    auto dm_u = dm.get<real,3>("uvel");
    auto dm_v = dm.get<real,3>("vvel");
    auto dm_w = dm.get<real,3>("wvel");
    auto dm_T = dm.get<real,3>("temp");

    int constexpr idU  = 0;
    int constexpr idV  = 1;
    int constexpr idT  = 2;
    int constexpr nfld = 3;

    real z1 = (1-top_prop)*zlen;
    real z2 = zlen-dz/2;
    real p  = 2;

    int k1 = (int) std::floor(z1/dz-0.5);
    int nzloc = nz-k1;

    real2d col("col",nfld,nzloc);
    real r_nx_ny = 1./(nx_glob*ny_glob);
    parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nfld,{k1,nz-1}) , KOKKOS_LAMBDA (int v, int k) {
      col(v,k-k1) = 0;
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
          if (v == idU) col(v,k-k1) += dm_u(k,j,i)*r_nx_ny;
          if (v == idV) col(v,k-k1) += dm_v(k,j,i)*r_nx_ny;
          if (v == idT) col(v,k-k1) += dm_T(k,j,i)*r_nx_ny;
        }
      }
    });

    col = coupler.get_parallel_comm().all_reduce( col , MPI_SUM , "" );

    parallel_for( YAKL_AUTO_LABEL() , Bounds<3>({k1,nz-1},ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      real z = (k+0.5_fp)*dz;
      if (z > z1) {
        real factor = std::pow((z-z1)/(z2-z1),p) * dt / time_scale;
        dm_u(k,j,i) = factor*col(idU,k-k1) + (1-factor)*dm_u(k,j,i);
        dm_v(k,j,i) = factor*col(idV,k-k1) + (1-factor)*dm_v(k,j,i);
        dm_w(k,j,i) = factor*0             + (1-factor)*dm_w(k,j,i);
        dm_T(k,j,i) = factor*col(idT,k-k1) + (1-factor)*dm_T(k,j,i);
      }
    });
  }

}


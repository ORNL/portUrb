
#pragma once

#include "coupler.h"

namespace modules {

  // Applies a sponge layer to the top of the model domain to force variables toward their horizontal averages
  //   and force vertical velocity to zero
  // coupler     : core::Coupler object containing the model state
  // dt          : Timestep size in seconds
  // time_scale  : Time scale for sponge layer damping in seconds
  // top_prop    : Proportion of the domain height to apply the sponge layer over
  inline void sponge_layer( core::Coupler &coupler , real dt , real time_scale , real top_prop ) {
    using yakl::c::parallel_for;
    using yakl::c::Bounds;

    auto ny_glob = coupler.get_ny_glob(); // global number of cells in y direction
    auto nx_glob = coupler.get_nx_glob(); // global number of cells in x direction
    auto nz      = coupler.get_nz  ();    // number of cells in z direction
    auto ny      = coupler.get_ny  ();    // local number of cells in y direction
    auto nx      = coupler.get_nx  ();    // local number of cells in x direction
    auto zlen    = coupler.get_zlen();    // domain length in z direction
    auto zint    = coupler.get_zint().createHostCopy(); // interface heights on host (1-D array of size nz+1)
    auto zmid    = coupler.get_zmid();    // mid-point heights on device (1-D array of size nz)
    auto &dm     = coupler.get_data_manager_readwrite(); // Get DataManager for read/write access

    auto dm_u = dm.get<real,3>("uvel"); // Get 3-D u-velocity array from DataManager
    auto dm_v = dm.get<real,3>("vvel"); // Get 3-D v-velocity array from DataManager
    auto dm_w = dm.get<real,3>("wvel"); // Get 3-D w-velocity array from DataManager
    auto dm_T = dm.get<real,3>("temp"); // Get 3-D temperature array from DataManager

    int constexpr idU  = 0;  // index for u-velocity in the column array
    int constexpr idV  = 1;  // index for v-velocity in the column array
    int constexpr idT  = 2;  // index for temperature in the column array
    int constexpr nfld = 3;  // number of fields to sponge

    real z1 = zlen*(1-top_prop);  // bottom height of sponge layer
    real z2 = zlen;               // top height of sponge layer
    real p  = 2;                  // exponent for sponge layer relaxation profile

    // Locate the index of the bottom of the sponge layer
    int k1 = -1;
    for (int k=nz-1; k >= 0; k--) {
      if (z1 >= zint(k)) {
        k1 = k;
        break;
      }
    }
    if (k1 == -1) k1 = 0;
    int nzloc = nz-k1;  // number of vertical levels in the sponge layer

    real2d col("col",nfld,nzloc);  // Allocate array to hold horizontal averages of u, v, and T
    real r_nx_ny = 1./(nx_glob*ny_glob);  // pre-compute reciprocal of total number of horizontal cells
    // Compute local contributions to horizontal averages
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

    // Accumulate global horizontal averages using MPI Allreduce
    col = coupler.get_parallel_comm().all_reduce( col , MPI_SUM , "" );

    // Apply sponge layer to relax u, v, and T toward their horizontal averages and w toward zero
    parallel_for( YAKL_AUTO_LABEL() , Bounds<3>({k1,nz-1},ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      real z = zmid(k);
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


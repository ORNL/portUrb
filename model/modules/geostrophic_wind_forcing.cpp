#include "geostrophic_wind_forcing.h"

namespace modules {

real2d geostrophic_wind_forcing( core::Coupler &coupler , real dt , real lat_g , real u_g , real v_g ) {
    using yakl::SimpleBounds;
    auto nz      = coupler.get_nz();       // number of vertical levels
    auto ny      = coupler.get_ny();       // number of local cells in the y-direction
    auto nx      = coupler.get_nx();       // number of local cells in the x-direction
    auto ny_glob = coupler.get_ny_glob();  // total global number of cells in the y-direction
    auto nx_glob = coupler.get_nx_glob();  // total global number of cells in the x-direction
    auto &dm     = coupler.get_data_manager_readwrite();  // get the data manager with read/write access
    auto uvel    = dm.get<real,3>("uvel");  // Get the 3D u-velocity field
    auto vvel    = dm.get<real,3>("vvel");  // Get the 3D v-velocity field
    auto imm     = dm.get<real const,3>("immersed_proportion"); // Get the immersed proportion field
    real fcor    = 2*7.2921e-5*std::sin(lat_g/180*M_PI);  // Compute coriolis parameter
    int constexpr idU  = 0;  // label for u-velocity in the column-averaged array
    int constexpr idV  = 1;  // label for v-velocity in the column-averaged array
    int constexpr nfld = 2;  // number of velocity fields (u and v)
    real2d col("col",nfld,nz);  // Allocate averaged column array for u and v velocities
    real r_nx_ny = 1. / (ny_glob*nx_glob); // precompute reciprocal of total global horizontal cells
    // Compute local contributions to the column-averaged velocities
    yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nfld,nz) , KOKKOS_LAMBDA (int v, int k) {
      col(v,k) = 0;
      for (int j=0; j < ny; j++) {
        for (int i=0; i < nx; i++) {
          if (v == idU) col(v,k) += uvel(k,j,i)*r_nx_ny;
          if (v == idV) col(v,k) += vvel(k,j,i)*r_nx_ny;
        }
      }
    });
    // Reduce across all MPI ranks to get the global column-averaged velocities
    col = coupler.get_parallel_comm().all_reduce( col , MPI_SUM , "" );
    // Apply geostrophic forcing to the u and v velocity fields based on averaged column forcing
    yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      uvel(k,j,i) += dt*( fcor*(col(idV,k)-v_g));
      vvel(k,j,i) += dt*(-fcor*(col(idU,k)-u_g));
    });
    return col; // Return the column-averaged velocities to be used by geostrophic_wind_forcing_specified
  }

void geostrophic_wind_forcing_specified( core::Coupler &coupler , real dt , real lat_g , real u_g , real v_g ,
                                                  real2d const &col ) {
    using yakl::SimpleBounds;
    auto nz      = coupler.get_nz();       // number of vertical levels
    auto ny      = coupler.get_ny();       // number of local cells in the y-direction
    auto nx      = coupler.get_nx();       // number of local cells in the x-direction
    auto ny_glob = coupler.get_ny_glob();  // total global number of cells in the y-direction
    auto nx_glob = coupler.get_nx_glob();  // total global number of cells in the x-direction
    auto &dm     = coupler.get_data_manager_readwrite();  // get the data manager with read/write access
    auto uvel    = dm.get<real,3>("uvel");  // Get the 3D u-velocity field
    auto vvel    = dm.get<real,3>("vvel");  // Get the 3D v-velocity field
    real fcor    = 2*7.2921e-5*std::sin(lat_g/180*M_PI);  // Compute coriolis parameter
    int constexpr idU  = 0;  // label for u-velocity in the column-averaged array
    int constexpr idV  = 1;  // label for v-velocity in the column-averaged array
    // Apply geostrophic forcing to the u and v velocity fields based on averaged column forcing
    yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      uvel(k,j,i) += dt*( fcor*(col(idV,k)-v_g));
      vvel(k,j,i) += dt*(-fcor*(col(idU,k)-u_g));
    });
  }

void geostrophic_wind_forcing_indiv( core::Coupler &coupler , real dt , real lat_g , real u_g , real v_g ) {
    using yakl::SimpleBounds;
    auto nz      = coupler.get_nz();        // number of vertical levels
    auto ny      = coupler.get_ny();        // number of local cells in the y-direction
    auto nx      = coupler.get_nx();        // number of local cells in the x-direction
    auto ny_glob = coupler.get_ny_glob();   // total global number of cells in the y-direction
    auto nx_glob = coupler.get_nx_glob();   // total global number of cells in the x-direction
    auto &dm     = coupler.get_data_manager_readwrite();  // get the data manager with read/write access
    auto uvel    = dm.get<real,3>("uvel");  // Get the 3D u-velocity field
    auto vvel    = dm.get<real,3>("vvel");  // Get the 3D v-velocity field
    auto imm     = dm.get<real const,3>("immersed_proportion");  // Get the immersed proportion field
    real fcor    = 2*7.2921e-5*std::sin(lat_g/180*M_PI);  // Compute coriolis parameter
    // Apply geostrophic forcing to the u and v velocity fields for each individual cell
    yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      uvel(k,j,i) += dt*( fcor*(vvel(k,j,i)-v_g));
      vvel(k,j,i) += dt*(-fcor*(uvel(k,j,i)-u_g));
    });
  }

} // namespace modules


#include "main_header.h"

namespace modules {

  // Penalize differences between domain-averaged wind and target wind at specified height
  // coupler : Coupler object containing the data manager and parallel communicator
  // dt      : Timestep size in seconds
  // height  : Height at which to compute the domain-averaged wind in meters
  // u0      : Target u-component of wind in m/s
  // v0      : Target v-component of wind in m/s
  // tau     : Relaxation timescale in seconds (default: 10s)
  // Returns the applied u and v wind forcings in m/s^2 as a tuple
  inline std::tuple<real,real> uniform_pg_wind_forcing_height( core::Coupler & coupler  ,
                                                               real            dt       ,
                                                               real            height   ,
                                                               real            u0       ,
                                                               real            v0       ,
                                                               real            tau = 10 ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto dz      = coupler.get_dz  ().createHostCopy(); // dz on host to find vertical index
    auto zint    = coupler.get_zint().createHostCopy(); // zint on host to find vertical index
    auto nz      = coupler.get_nz();        // number of vertical levels
    auto ny      = coupler.get_ny();        // local number of cells in y-direction
    auto nx      = coupler.get_nx();        // local number of cells in x-direction
    auto ny_glob = coupler.get_ny_glob();   // global number of cells in y-direction
    auto nx_glob = coupler.get_nx_glob();   // global number of cells in x-direction  
    auto &dm     = coupler.get_data_manager_readwrite();  // data manager for read/write access
    auto uvel    = dm.get<real,3>("uvel");  // get u-velocity field
    auto vvel    = dm.get<real,3>("vvel");  // get v-velocity field
    // Find the cell whose midpoint forms the lower bound for interpolation
    int k1_search = -1;
    if (height < dz(0)/2) {
      k1_search = -1;
    } else if (height > zint(nz-1)+dz(nz-1)/2) {
      k1_search = nz-1;
    } else {
      for (int k=1; k < nz; k++) {
        if (height >= zint(k-1)+dz(k-1)/2 && height <= zint(k)+dz(k)/2) {
          k1_search = k-1;
          break;
        }
      }
    }
    // Limit to inside the vertical domain
    int k1 = std::max(0   ,k1_search  );
    int k2 = std::min(nz-1,k1_search+1);
    // Get local MPI task's horizontal sums of u and v above and below the target height
    SArray<real,2,2,2> u_v;
    u_v(0,0) = yakl::intrinsics::sum(uvel.slice<2>(k1,0,0));
    u_v(0,1) = yakl::intrinsics::sum(vvel.slice<2>(k1,0,0));
    u_v(1,0) = yakl::intrinsics::sum(uvel.slice<2>(k2,0,0));
    u_v(1,1) = yakl::intrinsics::sum(vvel.slice<2>(k2,0,0));
    // Accumulate overall sums across all MPI tasks
    u_v = coupler.get_parallel_comm().all_reduce( u_v , MPI_SUM , "uniform_pg_allreduce" );
    // Compute averages
    real u1 = u_v(0,0)/(ny_glob*nx_glob);
    real v1 = u_v(0,1)/(ny_glob*nx_glob);
    real u2 = u_v(1,0)/(ny_glob*nx_glob);
    real v2 = u_v(1,1)/(ny_glob*nx_glob);
    // Interpolate to the target height
    real u, v;
    if (k1 == k2) {
      u = u1;
      v = v1;
    } else {
      real z1 = zint(k2)-dz(k1)/2;
      real z2 = zint(k2)+dz(k2)/2;
      real w1 = (z2-height)/((dz(k1)+dz(k2))/2);
      real w2 = (height-z1)/((dz(k1)+dz(k2))/2);
      u = w1*u1 + w2*u2;
      v = w1*v1 + w2*v2;
    }
    // Apply the wind forcing uniformly to nudge the domain-averaged wind toward the target wind
    real u_forcing = dt / tau*(u0-u);
    real v_forcing = dt / tau*(v0-v);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      uvel(k,j,i) += u_forcing;
      vvel(k,j,i) += v_forcing;
    });
    // Return tendencies to be used later in uniform_pg_wind_forcing_specified for forced simulations
    return std::make_tuple(u_forcing/dt,v_forcing/dt);
  }


  // Penalize differences between domain-averaged wind and given wind values
  // coupler : Coupler object containing the data manager and parallel communicator
  // dt      : Timestep size in seconds
  // u_in    : Input u-component of wind in m/s to force toward target value
  // v_in    : Input v-component of wind in m/s to force toward target value
  // u0      : Target u-component of wind in m/s
  // v0      : Target v-component of wind in m/s
  // tau     : Relaxation timescale in seconds (default: 10s)
  // Returns the applied u and v wind forcings in m/s^2 as a tuple
  inline std::tuple<real,real> uniform_pg_wind_forcing_given( core::Coupler & coupler  ,
                                                              real            dt       ,
                                                              real            u_in     ,
                                                              real            v_in     ,
                                                              real            u0       ,
                                                              real            v0       ,
                                                              real            tau = 10 ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nz   = coupler.get_nz();  // number of vertical levels
    auto ny   = coupler.get_ny();  // local number of cells in y-direction
    auto nx   = coupler.get_nx();  // local number of cells in x-direction
    auto &dm  = coupler.get_data_manager_readwrite();  // data manager for read/write access
    auto uvel = dm.get<real,3>("uvel");   // get u-velocity field
    auto vvel = dm.get<real,3>("vvel");   // get v-velocity field
    real u_forcing = dt / tau*(u0-u_in);  // compute u forcing
    real v_forcing = dt / tau*(v0-v_in);  // compute v forcing
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      uvel(k,j,i) += u_forcing;
      vvel(k,j,i) += v_forcing;
    });
    // Return tendencies to be used later in uniform_pg_wind_forcing_specified for forced simulations
    return std::make_tuple(u_forcing/dt,v_forcing/dt);
  }


  // Apply specified uniform pressure-gradient wind forcing for simulations forced by precursor
  // coupler : Coupler object containing the data manager and parallel communicator
  // dt      : Timestep size in seconds
  // utend   : Specified u-component of wind tendency in m/s^2
  // vtend   : Specified v-component of wind tendency in m/s^2
  inline void uniform_pg_wind_forcing_specified( core::Coupler & coupler ,
                                                 real            dt      ,
                                                 real            utend   ,
                                                 real            vtend   ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nz      = coupler.get_nz();  // number of vertical levels
    auto ny      = coupler.get_ny();  // local number of cells in y-direction
    auto nx      = coupler.get_nx();  // local number of cells in x-direction
    auto &dm     = coupler.get_data_manager_readwrite();  // data manager for read/write access
    auto uvel    = dm.get<real,3>("uvel");  // get u-velocity field
    auto vvel    = dm.get<real,3>("vvel");  // get v-velocity field
    // Apply the specified wind tendencies uniformly
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      uvel(k,j,i) += dt*utend;
      vvel(k,j,i) += dt*vtend;
    });
  }

}


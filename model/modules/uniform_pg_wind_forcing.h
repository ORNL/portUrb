
#include "main_header.h"

namespace modules {

  inline std::tuple<real,real> uniform_pg_wind_forcing_height( core::Coupler & coupler  ,
                                                               real            dt       ,
                                                               real            height   ,
                                                               real            u0       ,
                                                               real            v0       ,
                                                               real            tau = 10 ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto dz      = coupler.get_dz  ().createHostCopy();
    auto zint    = coupler.get_zint().createHostCopy();
    auto nz      = coupler.get_nz();
    auto ny      = coupler.get_ny();
    auto nx      = coupler.get_nx();
    auto ny_glob = coupler.get_ny_glob();
    auto nx_glob = coupler.get_nx_glob();
    auto &dm     = coupler.get_data_manager_readwrite();
    auto uvel    = dm.get<real,3>("uvel");
    auto vvel    = dm.get<real,3>("vvel");
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
    int k1 = std::max(0   ,k1_search  );
    int k2 = std::min(nz-1,k1_search+1);
    SArray<real,2,2,2> u_v;
    u_v(0,0) = yakl::intrinsics::sum(uvel.slice<2>(k1,0,0));
    u_v(0,1) = yakl::intrinsics::sum(vvel.slice<2>(k1,0,0));
    u_v(1,0) = yakl::intrinsics::sum(uvel.slice<2>(k2,0,0));
    u_v(1,1) = yakl::intrinsics::sum(vvel.slice<2>(k2,0,0));
    u_v = coupler.get_parallel_comm().all_reduce( u_v , MPI_SUM , "uniform_pg_allreduce" );
    real u1 = u_v(0,0)/(ny_glob*nx_glob);
    real v1 = u_v(0,1)/(ny_glob*nx_glob);
    real u2 = u_v(1,0)/(ny_glob*nx_glob);
    real v2 = u_v(1,1)/(ny_glob*nx_glob);
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
    real u_forcing = dt / tau*(u0-u);
    real v_forcing = dt / tau*(v0-v);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      uvel(k,j,i) += u_forcing;
      vvel(k,j,i) += v_forcing;
    });
    return std::make_tuple(u_forcing/dt,v_forcing/dt);
  }


  inline std::tuple<real,real> uniform_pg_wind_forcing_given( core::Coupler & coupler  ,
                                                              real            dt       ,
                                                              real            u_in     ,
                                                              real            v_in     ,
                                                              real            u0       ,
                                                              real            v0       ,
                                                              real            tau = 10 ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nz   = coupler.get_nz();
    auto ny   = coupler.get_ny();
    auto nx   = coupler.get_nx();
    auto &dm  = coupler.get_data_manager_readwrite();
    auto uvel = dm.get<real,3>("uvel");
    auto vvel = dm.get<real,3>("vvel");
    real u_forcing = dt / tau*(u0-u_in);
    real v_forcing = dt / tau*(v0-v_in);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      uvel(k,j,i) += u_forcing;
      vvel(k,j,i) += v_forcing;
    });
    return std::make_tuple(u_forcing/dt,v_forcing/dt);
  }


  inline void uniform_pg_wind_forcing_specified( core::Coupler & coupler ,
                                                 real            dt      ,
                                                 real            utend   ,
                                                 real            vtend   ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nz      = coupler.get_nz();
    auto ny      = coupler.get_ny();
    auto nx      = coupler.get_nx();
    auto &dm     = coupler.get_data_manager_readwrite();
    auto uvel    = dm.get<real,3>("uvel");
    auto vvel    = dm.get<real,3>("vvel");
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      uvel(k,j,i) += dt*utend;
      vvel(k,j,i) += dt*vtend;
    });
  }

}


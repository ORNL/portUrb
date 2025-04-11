
#include "main_header.h"

namespace modules {

  inline void uniform_pg_wind_forcing( core::Coupler &coupler , real dt , real tau = 10 , size_t update_cycles = 1 ) {
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
    auto counter = coupler.get_option<size_t>("uniform_pg_wind_forcing_counter",0);
    if (counter%update_cycles == 0) {
      SArray<real,1,2> u_v;
      u_v(0) = yakl::intrinsics::sum(uvel);
      u_v(1) = yakl::intrinsics::sum(vvel);
      u_v = coupler.get_parallel_comm().all_reduce( u_v , MPI_SUM , "uniform_pg_allreduce" );
      coupler.set_option<real>("uniform_pg_wind_forcing_u",u_v(0)/(nz*ny_glob*nx_glob));
      coupler.set_option<real>("uniform_pg_wind_forcing_v",u_v(1)/(nz*ny_glob*nx_glob));
      if (counter == 0) {
        coupler.set_option<real>("uniform_pg_wind_forcing_u0",u_v(0)/(nz*ny_glob*nx_glob));
        coupler.set_option<real>("uniform_pg_wind_forcing_v0",u_v(1)/(nz*ny_glob*nx_glob));
      }
    }
    auto u0 = coupler.get_option<real>("uniform_pg_wind_forcing_u0");
    auto v0 = coupler.get_option<real>("uniform_pg_wind_forcing_v0");
    auto u  = coupler.get_option<real>("uniform_pg_wind_forcing_u");
    auto v  = coupler.get_option<real>("uniform_pg_wind_forcing_v");
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      uvel(k,j,i) += dt/tau*(u0-u);
      vvel(k,j,i) += dt/tau*(v0-v);
    });
    counter++;
    coupler.set_option<size_t>("uniform_pg_wind_forcing_counter",counter);
  }


  inline std::tuple<real,real> uniform_pg_wind_forcing_height( core::Coupler & coupler  ,
                                                               real            dt       ,
                                                               real            height   ,
                                                               real            u0       ,
                                                               real            v0       ,
                                                               real            tau = 10 ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto dz      = coupler.get_dz();
    auto nz      = coupler.get_nz();
    auto ny      = coupler.get_ny();
    auto nx      = coupler.get_nx();
    auto ny_glob = coupler.get_ny_glob();
    auto nx_glob = coupler.get_nx_glob();
    auto &dm     = coupler.get_data_manager_readwrite();
    auto uvel    = dm.get<real,3>("uvel");
    auto vvel    = dm.get<real,3>("vvel");
    int k1 = 0;
    for (int k=0; k < nz; k++) { if ((k+0.5)*dz > height) { k1 = k-1; break; } }
    SArray<real,2,2,2> u_v;
    u_v(0,0) = yakl::intrinsics::sum(uvel.slice<2>(k1  ,0,0));
    u_v(0,1) = yakl::intrinsics::sum(vvel.slice<2>(k1  ,0,0));
    u_v(1,0) = yakl::intrinsics::sum(uvel.slice<2>(k1+1,0,0));
    u_v(1,1) = yakl::intrinsics::sum(vvel.slice<2>(k1+1,0,0));
    u_v = coupler.get_parallel_comm().all_reduce( u_v , MPI_SUM , "uniform_pg_allreduce" );
    real u1 = u_v(0,0)/(ny_glob*nx_glob);
    real v1 = u_v(0,1)/(ny_glob*nx_glob);
    real u2 = u_v(1,0)/(ny_glob*nx_glob);
    real v2 = u_v(1,1)/(ny_glob*nx_glob);
    real z1 = (k1+0.5)*dz;
    real z2 = (k1+1.5)*dz;
    real w1 = (z2-height)/dz;
    real w2 = (height-z1)/dz;
    real u  = w1*u1 + w2*u2;
    real v  = w1*v1 + w2*v2;
    real u_forcing = dt / tau*(u0-u);
    real v_forcing = dt / tau*(v0-v);
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


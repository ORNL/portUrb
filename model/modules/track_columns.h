#pragma once

#include "coupler.h"

namespace modules {

  struct TrackColMaxAbs {
    inline static std::string const prefix = "MaxAbsCol_";
    real static constexpr           init   = std::numeric_limits<real>::lowest();

    static void update( core::Coupler                  const & coupler ,
                        core::MultiField<real const,3> const & in      ,
                        core::MultiField<real      ,1> const & prev    ) {
      auto nz = coupler.get_nz();
      auto ny = coupler.get_ny();
      auto nx = coupler.get_nx();
      real2d col("col",in.size(),nz);
      col = init;
      yakl::parallel_for( YAKL_AUTO_LABEL() , yakl::SimpleBounds<4>(in.size(),nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int l, int k, int j, int i) {
        Kokkos::atomic_max( &(col(l,k)) , std::abs(in(l,k,j,i)) );
      });
      col = coupler.get_parallel_comm().all_reduce( col , MPI_MAX , "TrackColMaxAbs_Reduce" );
      yakl::parallel_for( YAKL_AUTO_LABEL() , yakl::SimpleBounds<2>(in.size(),nz) ,
                                              KOKKOS_LAMBDA (int l, int k) {
        prev(l,k) = std::max( prev(l,k) , col(l,k) );
      });
    }
  };
  


  template <class OP> struct Track_Columns {

    std::vector<std::string> names;

    void init( core::Coupler &coupler , std::vector<std::string> names ) {
      this->names = names;
      auto nz   = coupler.get_nz();
      auto &dm  = coupler.get_data_manager_readwrite();
      for (auto & name : names) {
        std::string label = OP::prefix + name;
        dm.register_and_allocate<real>(label,{nz});
        dm.get<real,1>(label) = OP::init;
        coupler.register_output_variable<real>( label , core::Coupler::DIMS_COLUMN );
      }
      apply( coupler );
    }


    void apply( core::Coupler & coupler ) {
      auto &dm = coupler.get_data_manager_readwrite();
      core::MultiField<real      ,1> cols;
      core::MultiField<real const,3> fields;
      for (auto & name : names ) {
        fields.add_field( dm.get<real const,3>(             name) );
        cols  .add_field( dm.get<real      ,1>(OP::prefix + name) );
      }
      OP::update( coupler , fields , cols );
    }
  };

}


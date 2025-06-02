
#pragma once

#include "coupler.h"
#include "MultipleFields.h"

namespace modules {


  class ColumnNudger {
  public:
    std::vector<std::string> names;
    real2d column;

    void set_column( core::Coupler &coupler , std::vector<std::string> names_in = {"uvel"} ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int nx   = coupler.get_nx();
      int ny   = coupler.get_ny();
      int nz   = coupler.get_nz();
      names = names_in;
      column = real2d("column",names.size(),nz);
      auto &dm = coupler.get_data_manager_readonly();
      core::MultiField<real const,3> state;
      for (int i=0; i < names.size(); i++) { state.add_field( dm.get<real const,3>(names.at(i)) ); }
      column = get_column_average( coupler , state );
    }


    void nudge_to_column( core::Coupler &coupler , real dt , real time_scale = 900 ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int nx   = coupler.get_nx();
      int ny   = coupler.get_ny();
      int nz   = coupler.get_nz();
      auto &dm = coupler.get_data_manager_readwrite();
      auto immersed = dm.get<real const,3>("immersed_proportion");
      core::MultiField<real,3> state;
      for (int i=0; i < names.size(); i++) { state.add_field( dm.get<real,3>(names.at(i)) ); }
      auto state_col_avg = get_column_average( coupler , state );
      YAKL_SCOPE( column , this->column );
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(names.size(),nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (immersed(k,j,i) == 0) {
          state(l,k,j,i) += dt * ( column(l,k) - state_col_avg(l,k) ) / time_scale;
        }
      });
    }


    template <class T>
    real2d get_column_average( core::Coupler const &coupler , core::MultiField<T,3> &state ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int nx_glob = coupler.get_nx_glob();
      int ny_glob = coupler.get_ny_glob();
      int nx      = coupler.get_nx();
      int ny      = coupler.get_ny();
      int nz      = coupler.get_nz();
      real2d column("column",names.size(),nz);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(names.size(),nz) ,
                                        KOKKOS_LAMBDA (int l, int k) {
        column(l,k) = 0;
        for (int j=0; j < ny; j++) {
          for (int i=0; i < nx; i++) {
            column(l,k) += state(l,k,j,i);
          }
        }
      });
      column = coupler.get_parallel_comm().all_reduce( column , MPI_SUM , "column_nudging_Allreduce" );
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(names.size(),nz) , KOKKOS_LAMBDA (int l, int k) {
        column(l,k) /= (nx_glob*ny_glob);
      });
      return column;
    }

  };

}



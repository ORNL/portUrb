
#pragma once

#include "coupler.h"
#include "MultipleFields.h"

namespace modules {


  class EdgeSponge {
  public:
    std::vector<std::string>  names;
    real2d                    column;

    void set_column( core::Coupler &coupler ,
                     std::vector<std::string> names_in = {"density_dry","uvel","vvel","wvel","temp"} ) {
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


    void apply( core::Coupler &coupler , real prop_x1 = 0.1 ,
                                         real prop_x2 = 0.1 ,
                                         real prop_y1 = 0.1 ,
                                         real prop_y2 = 0.1 ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int nx_glob   = coupler.get_nx_glob();
      int ny_glob   = coupler.get_ny_glob();
      int i_beg     = coupler.get_i_beg();
      int j_beg     = coupler.get_j_beg();
      int nx        = coupler.get_nx();
      int ny        = coupler.get_ny();
      int nz        = coupler.get_nz();
      auto &dm      = coupler.get_data_manager_readwrite();
      float pwr = 5;
      core::MultiField<real,3> state;
      for (int i=0; i < names.size(); i++) { state.add_field( dm.get<real,3>(names.at(i)) ); }
      YAKL_SCOPE( column , this->column );
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(names.size(),nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        real prop_x = static_cast<real>(i_beg+i)/nx_glob;
        real prop_y = static_cast<real>(j_beg+j)/ny_glob;
        if (prop_x1 > 0 && prop_x <= prop_x1) {
          real wt = (prop_x1-prop_x)/prop_x1;
          wt = std::pow( wt , pwr );
          state(l,k,j,i) = wt*column(l,k) + (1-wt)*state(l,k,j,i);
        }
        if (prop_x2 > 0 && prop_x >= 1-prop_x2) {
          real wt = (prop_x-(1-prop_x2))/prop_x2;
          wt = std::pow( wt , pwr );
          state(l,k,j,i) = wt*column(l,k) + (1-wt)*state(l,k,j,i);
        }
        if (prop_y1 > 0 && prop_y <= prop_y1) {
          real wt = (prop_y1-prop_y)/prop_y1;
          wt = std::pow( wt , pwr );
          state(l,k,j,i) = wt*column(l,k) + (1-wt)*state(l,k,j,i);
        }
        if (prop_y2 > 0 && prop_y >= 1-prop_y2) {
          real wt = (prop_y-(1-prop_y2))/prop_y2;
          wt = std::pow( wt , pwr );
          state(l,k,j,i) = wt*column(l,k) + (1-wt)*state(l,k,j,i);
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
      column = 0;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(names.size(),nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        Kokkos::atomic_add( &column(l,k) , state(l,k,j,i) );
      });
      column = coupler.get_parallel_comm().all_reduce( column , MPI_SUM , "column_nudging_Allreduce" );
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(names.size(),nz) , KOKKOS_LAMBDA (int l, int k) {
        column(l,k) /= (nx_glob*ny_glob);
      });
      return column;
    }

  };

}




#pragma once

#include "coupler.h"

namespace custom_modules {
  
  inline void precursor_sponge( core::Coupler            & coupler_main      ,
                                core::Coupler      const & coupler_precursor ,
                                std::vector<std::string>   vnames            ,
                                int                        cells_x1 = 0      ,
                                int                        cells_x2 = 0      ,
                                int                        cells_y1 = 0      ,
                                int                        cells_y2 = 0      ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nx            = coupler_main     .get_nx();
    auto ny            = coupler_main     .get_ny();
    auto nz            = coupler_main     .get_nz();
    auto i_beg         = coupler_main     .get_i_beg();
    auto j_beg         = coupler_main     .get_j_beg();
    auto nx_glob       = coupler_main     .get_nx_glob();
    auto ny_glob       = coupler_main     .get_ny_glob();
    auto &dm_main      = coupler_main     .get_data_manager_readwrite();
    auto &dm_precursor = coupler_precursor.get_data_manager_readonly();
    core::MultiField<real      ,3> fields_main;
    core::MultiField<real const,3> fields_precursor;
    int numvars = vnames.size();
    for (int i=0; i < numvars; i++) {
      fields_main     .add_field( dm_main     .get<real      ,3>(vnames.at(i)) );
      fields_precursor.add_field( dm_precursor.get<real const,3>(vnames.at(i)) );
    }

    real p = 5;

    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(numvars,nz,ny,nx) , KOKKOS_LAMBDA (int l, int k, int j, int i) {
      real weight = 0;
      if (i_beg+i < cells_x1) {
        weight = std::max(weight,1-std::pow((real)(i_beg+i)/(real)(cells_x1-1),p));
      }
      if (j_beg+j < cells_y1) {
        weight = std::max(weight,1-std::pow((real)(j_beg+j)/(real)(cells_y1-1),p));
      }
      if (nx_glob-1-(i_beg+i) < cells_x2) {
        weight = std::max(weight,1-std::pow((real)(nx_glob-1-(i_beg+i))/(real)(cells_x2-1),p));
      }
      if (ny_glob-1-(j_beg+j) < cells_y2) {
        weight = std::max(weight,1-std::pow((real)(ny_glob-1-(j_beg+j))/(real)(cells_y2-1),p));
      }
      fields_main(l,k,j,i) = weight*fields_precursor(l,k,j,i) + (1-weight)*fields_main(l,k,j,i);
    });



    // if (cells_x1 > 0) {
    //   parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(numvars,nz,ny,nx) , KOKKOS_LAMBDA (int l, int k, int j, int i) {
    //     if (i_beg+i < cells_x1) {
    //       fields_main(l,k,j,i) = fields_precursor(l,k,j,i);
    //     }
    //   });
    // }
    // if (cells_x2 > 0) {
    //   parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(numvars,nz,ny,nx) , KOKKOS_LAMBDA (int l, int k, int j, int i) {
    //     if (nx_glob-1-(i_beg+i) < cells_x2) {
    //       fields_main(l,k,j,i) = fields_precursor(l,k,j,i);
    //     }
    //   });
    // }
    // if (cells_y1 > 0) {
    //   parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(numvars,nz,ny,nx) , KOKKOS_LAMBDA (int l, int k, int j, int i) {
    //     if (j_beg+j < cells_y1) {
    //       fields_main(l,k,j,i) = fields_precursor(l,k,j,i);
    //     }
    //   });
    // }
    // if (cells_y2 > 0) {
    //   parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(numvars,nz,ny,nx) , KOKKOS_LAMBDA (int l, int k, int j, int i) {
    //     if (ny_glob-1-(j_beg+j) < cells_y2) {
    //       fields_main(l,k,j,i) = fields_precursor(l,k,j,i);
    //     }
    //   });
    // }

  }
}



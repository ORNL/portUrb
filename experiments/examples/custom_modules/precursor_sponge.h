
#pragma once

#include "coupler.h"

namespace custom_modules {
  
  inline void precursor_sponge( core::Coupler            & coupler_main ,
                                core::Coupler      const & coupler_prec ,
                                std::vector<std::string>   vnames       ,
                                int                        cells_x1 = 0 ,
                                int                        cells_x2 = 0 ,
                                int                        cells_y1 = 0 ,
                                int                        cells_y2 = 0 ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    using yakl::componentwise::operator/;
    using yakl::componentwise::operator-;
    auto nx       = coupler_main.get_nx();
    auto ny       = coupler_main.get_ny();
    auto nz       = coupler_main.get_nz();
    auto i_beg    = coupler_main.get_i_beg();
    auto j_beg    = coupler_main.get_j_beg();
    auto nx_glob  = coupler_main.get_nx_glob();
    auto ny_glob  = coupler_main.get_ny_glob();
    auto &dm_main = coupler_main.get_data_manager_readwrite();
    auto &dm_prec = coupler_prec.get_data_manager_readonly();
    core::MultiField<real      ,3> fields_main;
    core::MultiField<real const,3> fields_prec;
    int numvars = vnames.size();
    int idR = -1, idT=-1;
    for (int i=0; i < numvars; i++) {
      if (vnames.at(i) == "density_dry") idR = i;
      if (vnames.at(i) == "temp"       ) idT = i;
      fields_main.add_field( dm_main.get<real      ,3>(vnames.at(i)) );
      fields_prec.add_field( dm_prec.get<real const,3>(vnames.at(i)) );
    }

    // Get the mean column for main values
    real2d col_main("main_col_mean",numvars,nz);
    real2d col_prec("main_col_prec",numvars,nz);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(numvars,nz) , KOKKOS_LAMBDA (int l, int k) {
      col_main(l,k) = 0;
      col_prec(l,k) = 0;
      for (int j=0; j < ny; j++) {
        for (int i=0; i < nx; i++) {
          col_main(l,k) += fields_main(l,k,j,i);
          col_prec(l,k) += fields_prec(l,k,j,i);
        }
      }
    });

    col_main = coupler_main.get_parallel_comm().all_reduce(col_main,MPI_SUM)/(nx_glob*ny_glob);
    col_prec = coupler_prec.get_parallel_comm().all_reduce(col_prec,MPI_SUM)/(nx_glob*ny_glob);

    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(numvars,nz,ny,nx) , KOKKOS_LAMBDA (int l, int k, int j, int i) {
      real weight = 0;
      if (i_beg+i < cells_x1) {
        int nfull = cells_x1/3;
        int npart = cells_x1-nfull;
        if (i_beg+i < nfull) { weight = 1; }
        else                 { weight = std::max(weight,(1+std::cos(M_PI*(i_beg+i-nfull)/npart))/2); }
      }
      if (j_beg+j < cells_y1) {
        int nfull = cells_y1/3;
        int npart = cells_y1-nfull;
        if (j_beg+j < nfull) { weight = 1; }
        else                 { weight = std::max(weight,(1+std::cos(M_PI*(j_beg+j-nfull)/npart))/2); }
      }
      if (nx_glob-1-(i_beg+i) < cells_x2) {
        int nfull = cells_x2/3;
        int npart = cells_x2-nfull;
        if (nx_glob-1-(i_beg+i) < nfull) { weight = 1; }
        else                             { weight = std::max(weight,(1+std::cos(M_PI*(nx_glob-1-(i_beg+i)-nfull)/npart))/2); }
      }
      if (ny_glob-1-(j_beg+j) < cells_y2) {
        int nfull = cells_y2/3;
        int npart = cells_y2-nfull;
        if (ny_glob-1-(j_beg+j) < nfull) { weight = 1; }
        else                             { weight = std::max(weight,(1+std::cos(M_PI*(ny_glob-1-(j_beg+j)-nfull)/npart))/2); }
      }
      if (l==idR || l==idT) {
        fields_main(l,k,j,i) = weight*(fields_prec(l,k,j,i)-col_prec(l,k)+col_main(l,k)) + (1-weight)*fields_main(l,k,j,i);
      } else {
        fields_main(l,k,j,i) = weight*(fields_prec(l,k,j,i)                            ) + (1-weight)*fields_main(l,k,j,i);
      }
    });

  }
}



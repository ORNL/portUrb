
#pragma once

#include "coupler.h"

namespace modules {
  
  // Sponges in precursor data into a forced simulation near horizontal boundaries
  // coupler_main : Coupler object for the main simulation to be sponged
  // coupler_prec : Coupler object for the precursor simulation to provide data
  // vnames      : Vector of variable names to sponge
  // cells_x1    : Number of cells to sponge in from the left x-boundary
  // cells_x2    : Number of cells to sponge in from the right x-boundary
  // cells_y1    : Number of cells to sponge in from the bottom y-boundary
  // cells_y2    : Number of cells to sponge in from the top y-boundary
  // A cosine weighting is used over the sponge regions, with the first third of each region
  //  being fully sponged and the remaining two-thirds transitioning to no sponge at all.
  inline void precursor_sponge( core::Coupler            & coupler_main ,
                                core::Coupler      const & coupler_prec ,
                                std::vector<std::string>   vnames       ,
                                int                        cells_x1 = 0 ,
                                int                        cells_x2 = 0 ,
                                int                        cells_y1 = 0 ,
                                int                        cells_y2 = 0 ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    using yakl::componentwise::operator/;  // Allows componentwise '/' operator on yakl::Array
    using yakl::componentwise::operator-;  // Allows componentwise '-' operator on yakl::Array
    auto nx       = coupler_main.get_nx();      // Local number of cells in the x-direction
    auto ny       = coupler_main.get_ny();      // Local number of cells in the y-direction
    auto nz       = coupler_main.get_nz();      // Number of cells in the z-direction
    auto i_beg    = coupler_main.get_i_beg();   // Global starting index in the x-direction
    auto j_beg    = coupler_main.get_j_beg();   // Global starting index in the y-direction
    auto nx_glob  = coupler_main.get_nx_glob(); // Total global number of cells in the x-direction
    auto ny_glob  = coupler_main.get_ny_glob(); // Total global number of cells in the y-direction
    auto &dm_main = coupler_main.get_data_manager_readwrite(); // DataManager for the main simulation (read-write)
    auto &dm_prec = coupler_prec.get_data_manager_readonly();  // DataManager for the precursor simulation (read-only)
    // Accrue fields to be sponged into MultipleFields objects for easy access
    core::MultiField<real      ,3> fields_main;
    core::MultiField<real const,3> fields_prec;
    int numvars = vnames.size();
    for (int i=0; i < numvars; i++) {
      fields_main.add_field( dm_main.get<real      ,3>(vnames.at(i)) );
      fields_prec.add_field( dm_prec.get<real const,3>(vnames.at(i)) );
    }
    // Perform the sponge operation
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(numvars,nz,ny,nx) , KOKKOS_LAMBDA (int l, int k, int j, int i) {
      real weight = 0;
      if (i_beg+i < cells_x1) {
        int nfull = cells_x1/3;     // Full weight for a third of the sponge region
        int npart = cells_x1-nfull; // Partial weight for the remaining two-thirds
        if (i_beg+i < nfull) { weight = 1; }
        else                 { weight = std::max(weight,(1+std::cos(M_PI*(i_beg+i-nfull)/npart))/2); }
      }
      if (j_beg+j < cells_y1) {
        int nfull = cells_y1/3;     // Full weight for a third of the sponge region
        int npart = cells_y1-nfull; // Partial weight for the remaining two-thirds
        if (j_beg+j < nfull) { weight = 1; }
        else                 { weight = std::max(weight,(1+std::cos(M_PI*(j_beg+j-nfull)/npart))/2); }
      }
      if (nx_glob-1-(i_beg+i) < cells_x2) {
        int nfull = cells_x2/3;     // Full weight for a third of the sponge region
        int npart = cells_x2-nfull; // Partial weight for the remaining two-thirds
        if (nx_glob-1-(i_beg+i) < nfull) { weight = 1; }
        else                             { weight = std::max(weight,(1+std::cos(M_PI*(nx_glob-1-(i_beg+i)-nfull)/npart))/2); }
      }
      if (ny_glob-1-(j_beg+j) < cells_y2) {
        int nfull = cells_y2/3;     // Full weight for a third of the sponge region
        int npart = cells_y2-nfull; // Partial weight for the remaining two-thirds
        if (ny_glob-1-(j_beg+j) < nfull) { weight = 1; }
        else                             { weight = std::max(weight,(1+std::cos(M_PI*(ny_glob-1-(j_beg+j)-nfull)/npart))/2); }
      }
      // Apply the sponge using convex weighting
      fields_main(l,k,j,i) = weight*fields_prec(l,k,j,i) + (1-weight)*fields_main(l,k,j,i);
    });

  }
}



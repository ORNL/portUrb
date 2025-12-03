
#pragma once

#include "coupler.h"

namespace modules {
  
  // This function scales the fluctuations about the mean of the specified fields
  //   by a factor determined by frac, dt, and tscale
  // coupler : Coupler object to access domain and data information
  // dt      : Timestep size
  // frac    : Fractional reduction in fluctuations over the timescale tscale
  // tscale  : Timescale over which fluctuations are reduced by the fraction frac
  // vnames  : Vector of variable names to apply fluctuation scaling to
  // This is typically used to modify turbulent precursor inflow to different turbulence intensities
  // Typically you'll save the precursor data, then apply this routine to scale the turbulence,
  //   and then restore the precursor data to its original values so that only the inflow is modified
  inline void fluctuation_scaling( core::Coupler            & coupler ,
                                   real dt                            ,
                                   real frac                          ,
                                   real tscale                        ,
                                   std::vector<std::string>   vnames  ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    using yakl::componentwise::operator/;  // Allows use of '/' on yakl::Array objects
    using yakl::componentwise::operator-;  // Allows use of '-' on yakl::Array objects
    auto nx       = coupler.get_nx();      // Get local number of cells in x-direction
    auto ny       = coupler.get_ny();      // Get local number of cells in y-direction
    auto nz       = coupler.get_nz();      // Get local number of cells in z-direction
    auto i_beg    = coupler.get_i_beg();   // Get global starting index in x-direction
    auto j_beg    = coupler.get_j_beg();   // Get global starting index in y-direction
    auto nx_glob  = coupler.get_nx_glob(); // Get global number of cells in x-direction
    auto ny_glob  = coupler.get_ny_glob(); // Get global number of cells in y-direction
    auto &dm      = coupler.get_data_manager_readwrite(); // Get DataManager for read/write access
    // Accure the specified fields into a MultiField of 3-D arrays
    core::MultiField<real,3> fields;
    int numvars = vnames.size();
    for (int i=0; i < numvars; i++) { fields.add_field( dm.get<real,3>(vnames.at(i)) ); }

    real mult = std::pow(frac,dt/tscale);  // Pre-compute the fluctuation scaling multiplier

    // Compute the locally summed column values for each variable
    real2d col("main_col_mean",numvars,nz);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(numvars,nz) , KOKKOS_LAMBDA (int l, int k) {
      col(l,k) = 0;
      for (int j=0; j < ny; j++) {
        for (int i=0; i < nx; i++) {
          col(l,k) += fields(l,k,j,i);
        }
      }
    });
    // Reduce the column sums across all MPI ranks and compute the global mean
    col = coupler.get_parallel_comm().all_reduce(col,MPI_SUM)/(nx_glob*ny_glob);
    // Now scale the fluctuations about the mean for each variable
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(numvars,nz,ny,nx) , KOKKOS_LAMBDA (int l, int k, int j, int i) {
      fields(l,k,j,i) = col(l,k) + mult*(fields(l,k,j,i)-col(l,k));
    });

  }
}



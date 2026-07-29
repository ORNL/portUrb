
#pragma once

#include "coupler.h"
#include "MultipleFields.h"

namespace modules {


  // This class exists to sponge into the domain essentially laminar column averages of specified fields
  class EdgeSponge {
  public:
    std::vector<std::string>  names;  // Names of the fields to sponge
    real2d                    column; // 2D array holding the column averages used for forcing domain edges

    // Compute the average column that should be used for forcing
    // coupler    : Coupler object holding the data manager and domain information
    // names_in   : Names of the fields to sponge (default: {"density_dry","uvel","vvel","wvel","temperature"})
    // The column averages are computed and stored in the column member variable
    void set_column( core::Coupler &coupler ,
                     std::vector<std::string> names_in = {"density_dry","uvel","vvel","wvel","temperature"} );


    // Apply the edge sponge to the specified fields in the coupler's data manager
    // coupler : Coupler object holding the data manager and domain information
    // prop_x1 : Proportion of the domain in west  x to sponge (default: 0.1)
    // prop_x2 : Proportion of the domain in east  x to sponge (default: 0.1)
    // prop_y1 : Proportion of the domain in south y to sponge (default: 0.1)
    // prop_y2 : Proportion of the domain in north y to sponge (default: 0.1)
    // The sponge is applied with a weighting that increases with a power of 5 towards the domain edges
    void apply( core::Coupler &coupler , real prop_x1 = 0.1 ,
                                         real prop_x2 = 0.1 ,
                                         real prop_y1 = 0.1 ,
                                         real prop_y2 = 0.1 );


    // Compute the average column from the 3-D fields in the MultiField object
    // coupler : Coupler object holding the data manager and domain information
    // state   : MultiField object holding the 3-D fields to average
    // Returns a 2-D array holding the column averages for each field in state
    template <class MF>
    real2d get_column_average( core::Coupler const & coupler , MF & state ) const requires (MF::view_type::rank()==3) {
      using yakl::SimpleBounds;
      int nx_glob = coupler.get_nx_glob(); // Global number of cells in x-direction
      int ny_glob = coupler.get_ny_glob(); // Global number of cells in y-direction
      int nx      = coupler.get_nx();      // Local number of cells in x-direction
      int ny      = coupler.get_ny();      // Local number of cells in y-direction
      int nz      = coupler.get_nz();      // Number of cells in z-direction
      real2d column("column",names.size(),nz);  // Allocate the column averages array
      // Compute local summed column for each field
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(names.size(),nz) ,
                                              KOKKOS_LAMBDA (int l, int k) {
        column(l,k) = 0;
        for (int j=0; j < ny; j++) {
          for (int i=0; i < nx; i++) {
            column(l,k) += state(l,k,j,i);
          }
        }
      });
      // Accumulate global summed column across all MPI ranks
      column = coupler.get_parallel_comm().all_reduce( column , MPI_SUM , "column_nudging_Allreduce" );
      // Compute the average by dividing by the total number of cells globally
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(names.size(),nz) , KOKKOS_LAMBDA (int l, int k) {
        column(l,k) /= (nx_glob*ny_glob);
      });
      return column; // return the computed column averages
    }

  };

}



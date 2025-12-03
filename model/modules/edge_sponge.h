
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
    // names_in   : Names of the fields to sponge (default: {"density_dry","uvel","vvel","wvel","temp"})
    // The column averages are computed and stored in the column member variable
    void set_column( core::Coupler &coupler ,
                     std::vector<std::string> names_in = {"density_dry","uvel","vvel","wvel","temp"} ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int nx   = coupler.get_nx(); // Get number of cells in x-direction
      int ny   = coupler.get_ny(); // Get number of cells in y-direction
      int nz   = coupler.get_nz(); // Get number of cells in z-direction
      names = names_in;            // Store the names of the fields to sponge
      column = real2d("column",names.size(),nz);      // Allocate the column averages array
      auto &dm = coupler.get_data_manager_readonly(); // Get read-only data manager
      // Accrue 3-D fields for the specified names for averaging
      core::MultiField<real const,3> state;
      for (int i=0; i < names.size(); i++) { state.add_field( dm.get<real const,3>(names.at(i)) ); }
      // Compute and store the column averages of those fields
      column = get_column_average( coupler , state );
    }


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
                                         real prop_y2 = 0.1 ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int nx_glob   = coupler.get_nx_glob(); // Global number of cells in x-direction
      int ny_glob   = coupler.get_ny_glob(); // Global number of cells in y-direction
      int i_beg     = coupler.get_i_beg();   // Beginning index in x-direction for this MPI rank
      int j_beg     = coupler.get_j_beg();   // Beginning index in y-direction for this MPI rank
      int nx        = coupler.get_nx();      // Local number of cells in x-direction
      int ny        = coupler.get_ny();      // Local number of cells in y-direction
      int nz        = coupler.get_nz();      // Number of cells in z-direction
      auto &dm      = coupler.get_data_manager_readwrite(); // Get read-write data manager
      float pwr = 5;     // Power to use for weighting towards edges
      // Accrue 3-D fields for the specified names to apply the sponge
      core::MultiField<real,3> state;
      for (int i=0; i < names.size(); i++) { state.add_field( dm.get<real,3>(names.at(i)) ); }
      // Bring the column member variable into local scope for the parallel_for
      YAKL_SCOPE( column , this->column );
      // Apply the sponge towards each edge of the domain
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(names.size(),nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        real prop_x = static_cast<real>(i_beg+i)/nx_glob; // This cell's proportional x location in the global domain
        real prop_y = static_cast<real>(j_beg+j)/ny_glob; // This cell's proportional y location in the global domain
        // Use convex weightings determined by the distance into the edge sponge to the fifth power
        //  with a stronger forcing closer to the edge
        if (prop_x1 > 0 && prop_x <= prop_x1) { // West edge sponge
          real wt = (prop_x1-prop_x)/prop_x1;
          wt = std::pow( wt , pwr );
          state(l,k,j,i) = wt*column(l,k) + (1-wt)*state(l,k,j,i);
        }
        if (prop_x2 > 0 && prop_x >= 1-prop_x2) { // East edge sponge
          real wt = (prop_x-(1-prop_x2))/prop_x2;
          wt = std::pow( wt , pwr );
          state(l,k,j,i) = wt*column(l,k) + (1-wt)*state(l,k,j,i);
        }
        if (prop_y1 > 0 && prop_y <= prop_y1) { // South edge sponge
          real wt = (prop_y1-prop_y)/prop_y1;
          wt = std::pow( wt , pwr );
          state(l,k,j,i) = wt*column(l,k) + (1-wt)*state(l,k,j,i);
        }
        if (prop_y2 > 0 && prop_y >= 1-prop_y2) { // North edge sponge
          real wt = (prop_y-(1-prop_y2))/prop_y2;
          wt = std::pow( wt , pwr );
          state(l,k,j,i) = wt*column(l,k) + (1-wt)*state(l,k,j,i);
        }
      });
    }


    // Compute the average column from the 3-D fields in the MultiField object
    // coupler : Coupler object holding the data manager and domain information
    // state   : MultiField object holding the 3-D fields to average
    // Returns a 2-D array holding the column averages for each field in state
    template <class T>
    real2d get_column_average( core::Coupler const &coupler , core::MultiField<T,3> &state ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int nx_glob = coupler.get_nx_glob(); // Global number of cells in x-direction
      int ny_glob = coupler.get_ny_glob(); // Global number of cells in y-direction
      int nx      = coupler.get_nx();      // Local number of cells in x-direction
      int ny      = coupler.get_ny();      // Local number of cells in y-direction
      int nz      = coupler.get_nz();      // Number of cells in z-direction
      real2d column("column",names.size(),nz);  // Allocate the column averages array
      // Compute local summed column for each field
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(names.size(),nz) ,
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
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(names.size(),nz) , KOKKOS_LAMBDA (int l, int k) {
        column(l,k) /= (nx_glob*ny_glob);
      });
      return column; // return the computed column averages
    }

  };

}



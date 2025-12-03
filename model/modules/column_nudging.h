
#pragma once

#include "coupler.h"
#include "MultipleFields.h"

namespace modules {


  class ColumnNudger {
  public:
    std::vector<std::string> names; // names of fields to nudge
    real2d column;                  // target column averages for each field (names.size() , nz)

    // Set the desired column averages for the specified fields
    // coupler    : Coupler object to access data manager and grid information
    // names_in   : Vector of field names to compute column averages for (default: {"uvel"})
    // Allocates the column array and computes the column averages for the specified fields
    void set_column( core::Coupler &coupler , std::vector<std::string> names_in = {"uvel"} ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int nx   = coupler.get_nx();
      int ny   = coupler.get_ny();
      int nz   = coupler.get_nz();
      names = names_in;
      column = real2d("column",names.size(),nz);      // Allocate average column array
      auto &dm = coupler.get_data_manager_readonly();
      // Accumulate desired fields for column averaging
      core::MultiField<real const,3> state;
      for (int i=0; i < names.size(); i++) { state.add_field( dm.get<real const,3>(names.at(i)) ); }
      column = get_column_average( coupler , state ); // Compute column averages
    }


    // Nudge the specified fields toward the target column averages over the given time scale
    // coupler     : Coupler object to access data manager and grid information
    // dt          : Timestep size
    // time_scale  : Time scale over which to nudge toward the target column averages (default: 900 seconds)
    // For each specified field, compute the difference between the current column average and the target column average
    //  and adjust the field values accordingly, scaled by dt and time_scale
    void nudge_to_column( core::Coupler &coupler , real dt , real time_scale = 900 ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int nx   = coupler.get_nx();
      int ny   = coupler.get_ny();
      int nz   = coupler.get_nz();
      auto &dm = coupler.get_data_manager_readwrite();
      auto immersed = dm.get<real const,3>("immersed_proportion"); // Proportion of cell that is immersed
      // Accumulate desired fields for column averaging for current state
      core::MultiField<real,3> state;
      for (int i=0; i < names.size(); i++) { state.add_field( dm.get<real,3>(names.at(i)) ); }
      auto state_col_avg = get_column_average( coupler , state ); // Compute current column averages
      YAKL_SCOPE( column , this->column ); // Capture target column averages into local scope
      // Nudge desired fields toward target column averages if not immersed
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(names.size(),nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (immersed(k,j,i) == 0) {
          state(l,k,j,i) += dt * ( column(l,k) - state_col_avg(l,k) ) / time_scale;
        }
      });
    }


    // Nudge individual cells toward the target column averages over the given time scale
    //  rather than the current column average
    // coupler     : Coupler object to access data manager and grid information
    // dt          : Timestep size
    // time_scale  : Time scale over which to nudge toward the target column averages (default: 900 seconds)
    // For each specified field, compute the difference between the current cell value and the target column average
    //  and adjust the field values accordingly, scaled by dt and time_scale
    // This differs from nudge_to_column in that it nudges each cell toward the target column average
    //  rather than the current column average
    void nudge_to_column_strict( core::Coupler &coupler , real dt , real time_scale = 900 ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int nx   = coupler.get_nx();
      int ny   = coupler.get_ny();
      int nz   = coupler.get_nz();
      auto &dm = coupler.get_data_manager_readwrite();
      auto immersed = dm.get<real const,3>("immersed_proportion"); // Proportion of cell that is immersed
      // Accumulate desired fields for column averaging for current state
      core::MultiField<real,3> state;
      for (int i=0; i < names.size(); i++) { state.add_field( dm.get<real,3>(names.at(i)) ); }
      YAKL_SCOPE( column , this->column ); // Capture target column averages into local scope
      // Nudge desired fields toward target column averages if not immersed
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(names.size(),nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (immersed(k,j,i) == 0) {
          state(l,k,j,i) += dt * ( column(l,k) - state(l,k,j,i) ) / time_scale;
        }
      });
    }


    // Compute the column averages for the specified fields
    // coupler : Coupler object to access data manager and grid information
    // state   : MultiField containing the fields to compute column averages for
    // Returns a real2d array of size (names.size() , nz) containing the column average
    template <class T>
    real2d get_column_average( core::Coupler const &coupler , core::MultiField<T,3> &state ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int nx_glob = coupler.get_nx_glob(); // Global number of cells in x-direction
      int ny_glob = coupler.get_ny_glob(); // Global number of cells in y-direction
      int nx      = coupler.get_nx(); // Local number of cells in x-direction for this MPI rank
      int ny      = coupler.get_ny(); // Local number of cells in y-direction for this MPI rank
      int nz      = coupler.get_nz(); // Number of cells in z-direction
      real2d column_loc("column_loc",names.size(),nz); // Allocate column average array
      // Compute local column sums (avoiding atomics for reproducibility)
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(names.size(),nz) ,
                                        KOKKOS_LAMBDA (int l, int k) {
        column_loc(l,k) = 0;
        for (int j=0; j < ny; j++) {
          for (int i=0; i < nx; i++) {
            column_loc(l,k) += state(l,k,j,i);
          }
        }
      });
      // Reduce to global column sums
      column_loc = coupler.get_parallel_comm().all_reduce( column_loc , MPI_SUM , "column_nudging_Allreduce" );
      // Divide by total number of horizontal cells to get averaged column
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(names.size(),nz) , KOKKOS_LAMBDA (int l, int k) {
        column_loc(l,k) /= (nx_glob*ny_glob);
      });
      return column_loc;
    }

  };

}



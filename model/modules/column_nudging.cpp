#include "column_nudging.h"

namespace modules {

void ColumnNudger::set_column( core::Coupler &coupler , std::vector<std::string> names_in ) {
      using yakl::SimpleBounds;
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

void ColumnNudger::nudge_to_column( core::Coupler &coupler , real dt , real time_scale ) {
      using yakl::SimpleBounds;
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
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(names.size(),nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (immersed(k,j,i) == 0) {
          state(l,k,j,i) += dt * ( column(l,k) - state_col_avg(l,k) ) / time_scale;
        }
      });
    }

void ColumnNudger::nudge_to_column_strict( core::Coupler &coupler , real dt , real time_scale ) {
      using yakl::SimpleBounds;
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
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(names.size(),nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (immersed(k,j,i) == 0) {
          state(l,k,j,i) += dt * ( column(l,k) - state(l,k,j,i) ) / time_scale;
        }
      });
    }

} // namespace modules

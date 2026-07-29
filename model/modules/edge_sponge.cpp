#include "edge_sponge.h"

namespace modules {

void EdgeSponge::set_column( core::Coupler &coupler ,
                     std::vector<std::string> names_in ) {
      using yakl::SimpleBounds;
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

void EdgeSponge::apply( core::Coupler &coupler , real prop_x1 ,
                                         real prop_x2 ,
                                         real prop_y1 ,
                                         real prop_y2 ) {
      using yakl::SimpleBounds;
      int  nx_glob = coupler.get_nx_glob(); // Global number of cells in x-direction
      int  ny_glob = coupler.get_ny_glob(); // Global number of cells in y-direction
      int  i_beg   = coupler.get_i_beg();   // Beginning index in x-direction for this MPI rank
      int  j_beg   = coupler.get_j_beg();   // Beginning index in y-direction for this MPI rank
      int  nx      = coupler.get_nx();      // Local number of cells in x-direction
      int  ny      = coupler.get_ny();      // Local number of cells in y-direction
      int  nz      = coupler.get_nz();      // Number of cells in z-direction
      auto &dm     = coupler.get_data_manager_readwrite(); // Get read-write data manager
      real pwr     = 3;     // Power to use for weighting towards edges
      // Accrue 3-D fields for the specified names to apply the sponge
      core::MultiField<real,3> state;
      for (int i=0; i < names.size(); i++) { state.add_field( dm.get<real,3>(names.at(i)) ); }
      // Bring the column member variable into local scope for the parallel_for
      YAKL_SCOPE( column , this->column );
      // Apply the sponge towards each edge of the domain
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(names.size(),nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int l, int k, int j, int i) {
        real prop_x = static_cast<real>(i_beg+i)/(nx_glob-1); // This cell's proportional x location in the global domain
        real prop_y = static_cast<real>(j_beg+j)/(ny_glob-1); // This cell's proportional y location in the global domain
        // Use convex weightings determined by the distance into the edge sponge to the fifth power
        //  with a stronger forcing closer to the edge
        if (prop_x1 > 0 && prop_x <= prop_x1) { // West edge sponge
          real wt = (prop_x)/prop_x1;
          wt = std::pow((std::cos(M_PI*wt)+1)/2,pwr);
          state(l,k,j,i) = wt*column(l,k) + (1-wt)*state(l,k,j,i);
        }
        if (prop_x2 > 0 && prop_x >= 1-prop_x2) { // East edge sponge
          real wt = (1-prop_x)/prop_x2;
          wt = std::pow((std::cos(M_PI*wt)+1)/2,pwr);
          state(l,k,j,i) = wt*column(l,k) + (1-wt)*state(l,k,j,i);
        }
        if (prop_y1 > 0 && prop_y <= prop_y1) { // South edge sponge
          real wt = (prop_y)/prop_y1;
          wt = std::pow((std::cos(M_PI*wt)+1)/2,pwr);
          state(l,k,j,i) = wt*column(l,k) + (1-wt)*state(l,k,j,i);
        }
        if (prop_y2 > 0 && prop_y >= 1-prop_y2) { // North edge sponge
          real wt = (1-prop_y)/prop_y2;
          wt = std::pow((std::cos(M_PI*wt)+1)/2,pwr);
          state(l,k,j,i) = wt*column(l,k) + (1-wt)*state(l,k,j,i);
        }
      });
    }

} // namespace modules


#pragma once

#include "coupler.h"
#include <functional>

namespace core {

  // The Ensembler class helps create ensemble couplers by registering dimensions
  //  and their associated rank counts and coupler modification functions
  // The create_coupler_comm function creates a ParallelComm for the current MPI task's ensemble
  //  and modifies the coupler according to the registered dimensions
  // The class supports up to 3 dimensions for ensemble creation
  // Each dimension requires:
  //  - size: number of entries in the dimension
  //  - func_nranks: function that takes an index in the dimension and returns the number of ranks for that index
  //  - func_coupler: function that takes an index in the dimension and a Coupler reference, and modifies the coupler accordingly
  // The create_coupler_comm function iterates over all combinations of dimension indices
  //  and determines if the current MPI rank is active for that combination
  //  If active, it creates the ParallelComm and applies the coupler modification functions
  // The base_ranks parameter allows specifying a base number of ranks to multiply with the dimensional rank counts
  struct Ensembler {

    // This holds information about a single ensemble dimension
    struct Dimension {
      int                                       size;         // Number of entries in the dimension
      std::function<int(int)>                   func_nranks;  // Function to get number of ranks for a given index
                                                              //  to multiply with base_ranks
      std::function<void(int,core::Coupler &)>  func_coupler; // Function to modify the coupler for a given index
                                                              //  typically by setting coupler options that will be
                                                              //  retrieved later to run this ensemble member's simulation
    };

    std::vector<Dimension> dimensions; // Holds all registered dimensions. Only up to 3 are supported currently


    // Register a new ensemble dimension
    // size : Number of entries in the dimension
    // func_nranks : Function that takes an index in the dimension and returns the number of ranks for that index
    // func_coupler : Function that takes an index in the dimension and a Coupler reference, and modifies the coupler accordingly
    // This function appends the new dimension to the dimensions vector
    // The order of registration determines the nesting order when creating the ensemble communicator
    // Only up to 3 dimensions are supported currently
    // The functions can be lambda functions or std::function objects
    void register_dimension( int size ,
                             std::function<int(int)> func_nranks ,
                             std::function<void(int,core::Coupler &)> func_coupler ) {
      dimensions.push_back({size,func_nranks,func_coupler}); // Append the new dimension
    }


    // A convenience function to append a string to an existing coupler option string with an underscore separator
    // coupler : Coupler reference to modify
    // label : Name of the coupler option to modify
    // val : String value to append
    // If the option does not exist or is empty, it is set to val
    // If the option exists and is non-empty, val is appended with an underscore separator
    // This is useful for building descriptive names for ensemble members based on their dimension indices
    // Typically this is used for output file nameing or stdout, stderr redirecting to unique files per ensemble member
    void append_coupler_string( core::Coupler &coupler , std::string label , std::string val ) const {
      auto option = coupler.get_option<std::string>(label,"");
      if (option.empty()) { coupler.set_option<std::string>(label,val); }
      else                { coupler.set_option<std::string>(label,option+std::string("_")+val); }
      
    }


    // Create the ParallelComm for the ensemble member corresponding to the current MPI rank
    // coupler : Coupler reference to modify for the current ensemble member
    // base_ranks : Base number of ranks to multiply with the dimensional rank counts (default: 1)
    // comm_in : MPI_Comm to use as the parent communicator (default: MPI_COMM_WORLD)
    // Returns the ParallelComm for the current ensemble member
    // The function iterates over all combinations of dimension indices
    //  and determines if the current MPI rank is active for that combination
    //  If active, it creates the ParallelComm and applies the coupler modification functions
    //  If inactive, the ParallelComm is not created and remains invalid, and the user can test
    //   if it is valid using the valid() method of the ParallelComm class
    //  It will only be inactive / invalid if the current MPI rank exceeds the total number of ranks needed.
    // The coupler is modified in-place to set options for the current ensemble member, typically by 
    //  setting options that will be retrieved later to run this ensemble member's simulation
    // Only up to 3 dimensions are supported currently
    // The user is responsible for ensuring that the total number of ranks requested
    //  across all ensemble members does not exceed the total number of MPI ranks in comm_in. If it does,
    //  then some ensemble members will not run correctly as some of their ranks will be inactive.
    // The user can check if the returned ParallelComm is valid to determine if the current MPI rank
    //  is active for an ensemble member.
    ParallelComm create_coupler_comm( Coupler & coupler , int base_ranks = 1 , MPI_Comm comm_in = MPI_COMM_WORLD) {
      if (dimensions.size() == 0) Kokkos::abort("Trying to tensor ensemble with no dimensions");
      int nranks_tot, myrank; // Total number of ranks and current rank in comm_in
      MPI_Comm_rank( comm_in , &myrank );     // Get current rank
      MPI_Comm_size( comm_in , &nranks_tot ); // Get total number of ranks
      int rank_beg = 0; // Beginning rank index for the current ensemble member
      ParallelComm par_comm; // ParallelComm to return (inactive by default)
      if        (dimensions.size() == 1) {
        // Loop over all entries in the only specified dimension
        for (int i0=0; i0 < dimensions.at(0).size; i0++) {
          int nranks = base_ranks; // Start with base ranks
          nranks *= dimensions.at(0).func_nranks(i0); // Multiply by dimension 0 rank count
          bool active = myrank >= rank_beg && myrank < rank_beg+nranks; // Check if current rank is active for this ensemble member
          par_comm.create( active , comm_in ); // Create the ParallelComm if active
          // Modify the coupler if active using the dimension's coupler modification function
          if (active) {
            dimensions.at(0).func_coupler(i0,coupler);
          }
          rank_beg += nranks; // Update the beginning rank index for the next ensemble member
        }
      } else if (dimensions.size() == 2) {
        // Same as before except now nested loops over 2 dimensions
        for (int i0=0; i0 < dimensions.at(0).size; i0++) {
        for (int i1=0; i1 < dimensions.at(1).size; i1++) {
          int nranks = base_ranks;
          nranks *= dimensions.at(0).func_nranks(i0);
          nranks *= dimensions.at(1).func_nranks(i1);
          bool active = myrank >= rank_beg && myrank < rank_beg+nranks;
          par_comm.create( active , comm_in );
          if (active) {
            dimensions.at(0).func_coupler(i0,coupler);
            dimensions.at(1).func_coupler(i1,coupler);
          }
          rank_beg += nranks;
        } }
      } else if (dimensions.size() == 3) {
        // Same as before except now nested loops over 3 dimensions
        for (int i0=0; i0 < dimensions.at(0).size; i0++) {
        for (int i1=0; i1 < dimensions.at(1).size; i1++) {
        for (int i2=0; i2 < dimensions.at(2).size; i2++) {
          int nranks = base_ranks;
          nranks *= dimensions.at(0).func_nranks(i0);
          nranks *= dimensions.at(1).func_nranks(i1);
          nranks *= dimensions.at(2).func_nranks(i2);
          bool active = myrank >= rank_beg && myrank < rank_beg+nranks;
          par_comm.create( active , comm_in );
          if (active) {
            dimensions.at(0).func_coupler(i0,coupler);
            dimensions.at(1).func_coupler(i1,coupler);
            dimensions.at(2).func_coupler(i2,coupler);
          }
          rank_beg += nranks;
        } } }
      } else {
        // More than 3 dimensions not implemented yet
        Kokkos::abort("Requesting more ensemble dimensions than the 1-3 implemented");
      }
      return par_comm; // Return the created ParallelComm (may be invalid if inactive)
    }
  };

}

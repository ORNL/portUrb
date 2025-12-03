
#pragma once

#include "main_header.h"

namespace core {

  // The ParallelComm struct wraps an MPI_Comm and provides utility functions
  //  for creating communicators, sending/receiving data, and performing reductions
  // The struct manages the lifetime of the MPI_Comm if it was created by this instance
  // It provides methods to create communicators based on active ranks,
  //  send/receive YAKL arrays, and perform reductions and allreductions
  // The struct uses YAKL for array management and timing
  // The struct provides detailed error checking for MPI operations
  struct ParallelComm {
    bool             comm_was_created;  // Whether the communicator was created by this ParallelComm instance
    MPI_Comm         comm;              // The MPI communicator
    int              nranks;            // Number of ranks in the communicator
    int              rank_id;           // Rank ID of this rank within this communicator
    MPI_Group        group;             // The MPI group associated with the communicator

    // Struct to hold information for a send/receive operation
    template <class T, int N>
    struct SendRecvPack {
      yakl::Array<T,N,yakl::memDevice,yakl::styleC>  arr;  // YAKL array to send/receive
      int                                            them; // Rank ID of the other rank to send to / receive from
      int                                            tag;  // MPI tag for the send/receive operation
    };


    // Nullify the communicator and reset all members
    void nullify() {
      comm_was_created = false;
      comm             = MPI_COMM_NULL;
      nranks           = 0;
      rank_id          = -1;
      group            = MPI_GROUP_NULL;
    }
    
    // Default constructor and constructor from existing MPI_Comm
    ParallelComm () { nullify(); }
    ParallelComm (MPI_Comm comm_in) { wrap(comm_in); }
    ~ParallelComm() { }


    // Wrap an existing MPI_Comm without creating a new one
    ParallelComm wrap( MPI_Comm comm_in = MPI_COMM_WORLD ) {
      comm = comm_in;
      check( MPI_Comm_size ( comm , &nranks  ) );
      check( MPI_Comm_rank ( comm , &rank_id ) );
      check( MPI_Comm_group( comm , &group   ) );
      comm_was_created = false;
      return *this;
    }


    // Create a new MPI_Comm based on whether this rank is active
    // b : If true, this rank is active and will be included in the new communicator
    //     If false, this rank will not be included in the new communicator
    // parent_comm : Parent MPI_Comm to create the new communicator from (default: MPI_COMM_WORLD)
    // Returns the ParallelComm instance with the created communicator if active, or invalid if not active
    // You can check if the communicator is valid using the valid() method or the bool operator after creation
    // The false communicator is freed immediately, and only the true communicator is stored
    ParallelComm create( bool b , MPI_Comm parent_comm = MPI_COMM_WORLD ) {
      int parent_rank;
      MPI_Comm newcomm;
      check( MPI_Comm_rank( parent_comm , &parent_rank ) );
      check( MPI_Comm_split( parent_comm , b ? 1 : 0 , parent_rank , &newcomm ) );
      if (b) {
        comm = newcomm;
        check( MPI_Comm_size ( comm , &nranks  ) );
        check( MPI_Comm_rank ( comm , &rank_id  ) );
        check( MPI_Comm_group( comm , &group ) );
      } else {
        check( MPI_Comm_free( &newcomm ) );
      }
      comm_was_created = true;
      return *this;
    }


    // Get the MPI_Comm associated with this ParallelComm
    MPI_Comm         get_mpi_comm () const { return comm;      }

    // Get the number of ranks in the communicator
    int              get_size     () const { return nranks;    }

    // Get the number of ranks in the communicator (same as get_size())
    int              size         () const { return nranks;    }

    // Get the rank ID of this rank within the communicator
    int              get_rank_id  () const { return rank_id;   }

    // Get the MPI_Group associated with the communicator
    MPI_Group        get_group    () const { return group;     }

    // Check if the communicator is valid
    bool             valid        () const { return comm != MPI_COMM_NULL; }

    // Bool operator to check if the communicator is valid
    explicit operator bool        () const { return comm != MPI_COMM_NULL; }


    // Destroy the communicator if it was created by this ParallelComm instance
    // Frees the MPI_Comm if it was created and is not MPI_COMM_NULL or MPI_COMM_WORLD
    void destroy() {
      if (comm_was_created && comm != MPI_COMM_NULL && comm != MPI_COMM_WORLD) {
        check( MPI_Comm_free(&comm) );
      }
      nullify();
    }


    ////////////////////////
    // Sends and Receives
    ////////////////////////
    // Perform multiple non-blocking sends and receives using MPI Isend and Irecv
    // receives : Vector of SendRecvPack structs for the receive operations
    // sends    : Vector of SendRecvPack structs for the send operations
    // lab      : Optional label for timing the operation using YAKL timers
    template <class T, int N>
    void send_receive( std::vector<SendRecvPack<T,N>> receives , std::vector<SendRecvPack<T,N>> sends ,
                       std::string lab = "" ) const {
      int n = receives.size();  // Number of receives and sends
      std::vector<MPI_Request> sReq (n); // MPI requests for sends
      std::vector<MPI_Request> rReq (n); // MPI requests for receives
      std::vector<MPI_Status > sStat(n); // MPI statuses for sends
      std::vector<MPI_Status > rStat(n); // MPI statuses for receives
      #ifdef PORTURB_GPU_AWARE_MPI
        if (lab != "") yakl::timer_start(lab.c_str()); // Start timer if label provided
        Kokkos::fence(); // Ensure all device operations are complete before MPI calls
        for (int i=0; i < n; i++) { // Post all receives
          auto arr = receives.at(i).arr; // Alias for receive array
          check( MPI_Irecv( arr.data() , arr.size() , get_type<T>() , receives.at(i).them , receives.at(i).tag , comm , &(rReq.at(i)) ) );
        }
        for (int i=0; i < n; i++) { // Post all sends
          auto arr = sends.at(i).arr; // Alias for send array
          check( MPI_Isend( arr.data() , arr.size() , get_type<T>() , sends   .at(i).them , sends   .at(i).tag , comm , &(sReq.at(i)) ) );
        }
        check( MPI_Waitall(n, sReq.data(), sStat.data()) ); // Wait for all sends to complete
        check( MPI_Waitall(n, rReq.data(), rStat.data()) ); // Wait for all receives to complete
        if (lab != "") yakl::timer_stop(lab.c_str()); // Stop timer if label provided
      #else
        if (lab != "") yakl::timer_start(lab.c_str()); // Start timer if label provided
        std::vector<yakl::Array<T,N,yakl::memHost,yakl::styleC>> receive_host_arrays(n); // Host arrays for receives
        std::vector<yakl::Array<T,N,yakl::memHost,yakl::styleC>> send_host_arrays(n);    // Host arrays for sends
        for (int i=0; i < n; i++) { // Post all receives
          receive_host_arrays.at(i) = receives.at(i).arr.createHostObject(); // Create host copy for receive
          check( MPI_Irecv( receive_host_arrays.at(i).data() , receive_host_arrays.at(i).size() , get_type<T>() ,
                            receives.at(i).them , receives.at(i).tag , comm , &(rReq.at(i)) ) );
        }
        for (int i=0; i < n; i++) { // Post all sends
          send_host_arrays   .at(i) = sends   .at(i).arr.createHostCopy(); // Create host copy for send
          check( MPI_Isend( send_host_arrays.at(i).data() , send_host_arrays.at(i).size() , get_type<T>() ,
                            sends.at(i).them , sends.at(i).tag , comm , &(sReq.at(i)) ) );
        }
        check( MPI_Waitall(n, sReq.data(), sStat.data()) ); // Wait for all sends to complete
        check( MPI_Waitall(n, rReq.data(), rStat.data()) ); // Wait for all receives to complete
        for (int i=0; i < n; i++) { receive_host_arrays.at(i).deep_copy_to(receives.at(i).arr);} // Copy received data to device arrays
        Kokkos::fence(); // Ensure all device operations are complete
        if (lab != "") yakl::timer_stop(lab.c_str()); // Stop timer if label provided
      #endif
    }


    ////////////////////
    // Allgather
    ////////////////////
    // Allgather for YAKL CSArray objects
    // arr : YAKL CSArray to allgather from each rank
    // lab : Optional label for timing the operation using YAKL timers
    template <class T, typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = false>
    yakl::Array<T,1,yakl::memHost,yakl::styleC> all_gather( T val , std::string lab = "" ) const {
      if (lab != "") yakl::timer_start( lab.c_str() );
      yakl::Array<T,1,yakl::memHost,yakl::styleC> ret("all_gather_result",nranks); // Result array
      check( MPI_Allgather( &val , 1 , get_type<T>() , ret.data() , 1 , get_type<T>() , comm ) );
      if (lab != "") yakl::timer_stop( lab.c_str() );
      return ret;
    }


    ////////////////////
    // Barrier
    ////////////////////
    // Perform an MPI Barrier on the communicator to wait for all ranks to reach this point
    void barrier() const { check( MPI_Barrier(comm) ); }


    ////////////////////
    // Broadcast
    ////////////////////
    // Broadcast for YAKL CSArray and Array objects, as well as single arithmetic values
    // arr : YAKL CSArray to broadcast (will be modified on non-root ranks)
    // root : Rank ID of the root rank that holds the array to broadcast
    // lab : Optional label for timing the operation using YAKL timers
    template <class T, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    void broadcast( yakl::CSArray<T,N,D0,D1,D2,D3> const & arr , int root = 0 , std::string lab = "" ) const {
      if (nranks == 1) return;
      if (lab != "") yakl::timer_start( lab.c_str() );
      check( MPI_Bcast( arr.data()  , arr.size() , get_type<T>() , root , comm ) );
      if (lab != "") yakl::timer_stop( lab.c_str() );
    }

    // Broadcast for YAKL Array objects
    // arr : YAKL Array to broadcast (will be modified on non-root ranks)
    // root : Rank ID of the root rank that holds the array to broadcast
    // lab : Optional label for timing the operation using YAKL timers
    template <class T, int N, int MEM, int STYLE>
    void broadcast( yakl::Array<T,N,MEM,STYLE> const & arr , int root = 0 , std::string lab = "" ) const {
      if (nranks == 1) return;
      if constexpr (MEM == yakl::memHost) {
        if (lab != "") yakl::timer_start( lab.c_str() );
        check( MPI_Bcast( arr.data() , arr.size() , get_type<T>() , root , comm ) );
        if (lab != "") yakl::timer_stop( lab.c_str() );
      } else {
        #ifdef PORTURB_GPU_AWARE_MPI
          if (lab != "") yakl::timer_start( lab.c_str() );
          Kokkos::fence();
          check( MPI_Bcast( arr.data() , arr.size() , get_type<T>() , root , comm ) );
          if (lab != "") yakl::timer_stop( lab.c_str() );
        #else
          if (lab != "") yakl::timer_start( lab.c_str() );
          auto arr_host  = arr.createHostCopy();
          check( MPI_Bcast( arr_host.data() , arr.size() , get_type<T>() , root , comm ) );
          arr_host.deep_copy_to(arr);
          if (lab != "") yakl::timer_stop( lab.c_str() );
        #endif
      }
    }

    // Broadcast for single arithmetic values
    // T must be an arithmetic type (int, float, double, etc.)
    // val : Value to broadcast (will be modified on non-root ranks)
    // root : Rank ID of the root rank that holds the value to broadcast
    // lab : Optional label for timing the operation using YAKL timers
    template <class T, typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = false>
    void broadcast( T & val , int root = 0 , std::string lab = "" ) const {
      if (nranks == 1) return;
      if (lab != "") yakl::timer_start( lab.c_str() );
      check( MPI_Bcast( &val , 1 , get_type<T>() , root , comm ) );
      if (lab != "") yakl::timer_stop( lab.c_str() );
    }


    ////////////////////
    // Reduce
    ////////////////////
    // Reduce for YAKL CSArray and Array objects, as well as single arithmetic values
    // arr : YAKL CSArray to reduce from each rank
    // op : MPI_Op operation to use for the reduction (e.g., MPI_SUM, MPI_MAX, etc.)
    // root : Rank ID of the root rank that will hold the reduced result
    // lab : Optional label for timing the operation using YAKL timers
    template <class T, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    yakl::CSArray<T,N,D0,D1,D2,D3> reduce( yakl::CSArray<T,N,D0,D1,D2,D3> loc , MPI_Op op , int root = 0 , std::string lab = "" ) const {
      if (nranks == 1) return loc;
      if (lab != "") yakl::timer_start( lab.c_str() );
      yakl::CSArray<T,N,D0,D1,D2,D3> glob;
      check( MPI_Reduce( loc.data() , glob.data() , loc.size() , get_type<T>() , op , root , comm ) );
      if (lab != "") yakl::timer_stop( lab.c_str() );
      return glob;
    }

    // Reduce for YAKL Array objects
    // arr : YAKL Array to reduce from each rank
    // op : MPI_Op operation to use for the reduction (e.g., MPI_SUM, MPI_MAX, etc.)
    // root : Rank ID of the root rank that will hold the reduced result
    // lab : Optional label for timing the operation using YAKL timers
    template <class T, int N, int MEM, int STYLE>
    yakl::Array<T,N,MEM,STYLE> reduce( yakl::Array<T,N,MEM,STYLE> loc , MPI_Op op , int root = 0 , std::string lab = "" ) const {
      if (nranks == 1) return loc;
      if constexpr (MEM == yakl::memHost) {
        if (lab != "") yakl::timer_start( lab.c_str() );
        auto glob = loc.createHostObject();
        check( MPI_Reduce( loc.data() , glob.data() , loc.size() , get_type<T>() , op , root , comm ) );
        if (lab != "") yakl::timer_stop( lab.c_str() );
        return glob;
      } else {
        #ifdef PORTURB_GPU_AWARE_MPI
          if (lab != "") yakl::timer_start( lab.c_str() );
          auto glob = loc.createDeviceObject();
          Kokkos::fence();
          check( MPI_Reduce( loc.data() , glob.data() , loc.size() , get_type<T>() , op , root , comm ) );
          if (lab != "") yakl::timer_stop( lab.c_str() );
          return glob;
        #else
          if (lab != "") yakl::timer_start( lab.c_str() );
          auto loc_host  = loc.createHostCopy  ();
          auto glob_host = loc.createHostObject();
          check( MPI_Reduce( loc_host.data() , glob_host.data() , loc.size() , get_type<T>() , op , root , comm ) );
          if (lab != "") yakl::timer_stop( lab.c_str() );
          return glob_host.createDeviceCopy();
        #endif
      }
    }

    // Reduce for single arithmetic values
    // T must be an arithmetic type (int, float, double, etc.)
    // val : Value to reduce from each rank
    // op : MPI_Op operation to use for the reduction (e.g., MPI_SUM, MPI_MAX, etc.)
    // root : Rank ID of the root rank that will hold the reduced result
    // lab : Optional label for timing the operation using YAKL timers
    template <class T, typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = false>
    T reduce( T loc , MPI_Op op , int root = 0 , std::string lab = "" ) const {
      if (nranks == 1) return loc;
      if (lab != "") yakl::timer_start( lab.c_str() );
      T glob;
      check( MPI_Reduce( &loc , &glob , 1 , get_type<T>() , op , root , comm ) );
      if (lab != "") yakl::timer_stop( lab.c_str() );
      return glob;
    }


    ////////////////////
    // Allreduce
    ////////////////////
    // Allreduce for YAKL CSArray and Array objects, as well as single arithmetic values
    // arr : YAKL CSArray to allreduce from each rank
    // op : MPI_Op operation to use for the allreduction (e.g., MPI_SUM, MPI_MAX, etc.)
    // lab : Optional label for timing the operation using YAKL timers
    template <class T, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    yakl::CSArray<T,N,D0,D1,D2,D3> all_reduce( yakl::CSArray<T,N,D0,D1,D2,D3> loc , MPI_Op op , std::string lab = "" ) const {
      if (nranks == 1) return loc;
      if (lab != "") yakl::timer_start( lab.c_str() );
      yakl::CSArray<T,N,D0,D1,D2,D3> glob;
      check( MPI_Allreduce( loc.data() , glob.data() , loc.size() , get_type<T>() , op , comm ) );
      if (lab != "") yakl::timer_stop( lab.c_str() );
      return glob;
    }

    // Allreduce for YAKL Array objects
    // arr : YAKL Array to allreduce from each rank
    // op : MPI_Op operation to use for the allreduction (e.g., MPI_SUM, MPI_MAX, etc.)
    // lab : Optional label for timing the operation using YAKL timers
    template <class T, int N, int MEM, int STYLE>
    yakl::Array<T,N,MEM,STYLE> all_reduce( yakl::Array<T,N,MEM,STYLE> loc , MPI_Op op , std::string lab = "" ) const {
      if (nranks == 1) return loc;
      if constexpr (MEM == yakl::memHost) {
        if (lab != "") yakl::timer_start( lab.c_str() );
        auto glob = loc.createHostObject();
        check( MPI_Allreduce( loc.data() , glob.data() , loc.size() , get_type<T>() , op , comm ) );
        if (lab != "") yakl::timer_stop( lab.c_str() );
        return glob;
      } else {
        #ifdef PORTURB_GPU_AWARE_MPI
          if (lab != "") yakl::timer_start( lab.c_str() );
          auto glob = loc.createDeviceObject();
          Kokkos::fence();
          check( MPI_Allreduce( loc.data() , glob.data() , loc.size() , get_type<T>() , op , comm ) );
          if (lab != "") yakl::timer_stop( lab.c_str() );
          return glob;
        #else
          if (lab != "") yakl::timer_start( lab.c_str() );
          auto loc_host  = loc.createHostCopy  ();
          auto glob_host = loc.createHostObject();
          check( MPI_Allreduce( loc_host.data() , glob_host.data() , loc.size() , get_type<T>() , op , comm ) );
          if (lab != "") yakl::timer_stop( lab.c_str() );
          return glob_host.createDeviceCopy();
        #endif
      }
    }

    // Allreduce for single arithmetic values
    // T must be an arithmetic type (int, float, double, etc.)
    // val : Value to allreduce from each rank
    // op : MPI_Op operation to use for the allreduction (e.g., MPI_SUM, MPI_MAX, etc.)
    // lab : Optional label for timing the operation using YAKL timers
    template <class T, typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = false>
    T all_reduce( T loc , MPI_Op op , std::string lab = "" ) const {
      if (nranks == 1) return loc;
      if (lab != "") yakl::timer_start( lab.c_str() );
      T glob;
      check( MPI_Allreduce( &loc , &glob , 1 , get_type<T>() , op , comm ) );
      if (lab != "") yakl::timer_stop( lab.c_str() );
      return glob;
    }


    // INTERNAL USE: Get the MPI_Datatype corresponding to the templated type T
    // T : Template parameter for the type to get the MPI_Datatype of
    // Returns the MPI_Datatype corresponding to type T
    template <class T> static MPI_Datatype get_type() {
      if      (std::is_same<T,         char         >::value) { return MPI_CHAR;                  }
      else if (std::is_same<T,unsigned char         >::value) { return MPI_UNSIGNED_CHAR;         }
      else if (std::is_same<T,         short        >::value) { return MPI_SHORT;                 }
      else if (std::is_same<T,unsigned short        >::value) { return MPI_UNSIGNED_SHORT;        }
      else if (std::is_same<T,         int          >::value) { return MPI_INT;                   }
      else if (std::is_same<T,unsigned int          >::value) { return MPI_UNSIGNED;              }
      else if (std::is_same<T,         long int     >::value) { return MPI_LONG;                  }
      else if (std::is_same<T,unsigned long int     >::value) { return MPI_UNSIGNED_LONG;         }
      else if (std::is_same<T,         long long int>::value) { return MPI_LONG_LONG;             }
      else if (std::is_same<T,unsigned long long int>::value) { return MPI_UNSIGNED_LONG_LONG;    }
      else if (std::is_same<T,                 float>::value) { return MPI_FLOAT;                 }
      else if (std::is_same<T,                double>::value) { return MPI_DOUBLE;                }
      else if (std::is_same<T,           long double>::value) { return MPI_LONG_DOUBLE;           }
      else if (std::is_same<T,                  bool>::value) { return MPI_C_BOOL;                }
      else { Kokkos::abort("Invalid type for MPI operations"); }
    }


    // INTERNAL USE: Check the return code of an MPI function and abort if there was an error
    // e : Return code from an MPI function
    // If e is not MPI_SUCCESS, prints the error string and aborts the program
    static void check(int e) {
      if (e == MPI_SUCCESS ) return;
      char estring[MPI_MAX_ERROR_STRING];
      int len;
      MPI_Error_string(e, estring, &len);
      printf("MPI Error: %s\n", estring);
      std::cout << std::endl;
      Kokkos::abort("MPI Error");
    }
  };

}


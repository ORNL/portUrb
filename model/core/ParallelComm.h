
#pragma once

#include "main_header.h"

namespace core {

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


    // Internal use: Nullify the communicator and reset all members
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


    ParallelComm wrap( MPI_Comm comm_in = MPI_COMM_WORLD ) {
      comm = comm_in;
      check( MPI_Comm_size ( comm , &nranks  ) );
      check( MPI_Comm_rank ( comm , &rank_id ) );
      check( MPI_Comm_group( comm , &group   ) );
      comm_was_created = false;
      return *this;
    }


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


    MPI_Comm         get_mpi_comm () const { return comm;      }
    int              get_size     () const { return nranks;    }
    int              size         () const { return nranks;    }
    int              get_rank_id  () const { return rank_id;   }
    MPI_Group        get_group    () const { return group;     }
    bool             valid        () const { return comm != MPI_COMM_NULL; }
    explicit operator bool        () const { return comm != MPI_COMM_NULL; }


    void destroy() {
      if (comm_was_created && comm != MPI_COMM_NULL && comm != MPI_COMM_WORLD) {
        check( MPI_Comm_free(&comm) );
      }
      nullify();
    }


    ////////////////////////
    // Sends and Receives
    ////////////////////////
    template <class T, int N>
    void send_receive( std::vector<SendRecvPack<T,N>> receives , std::vector<SendRecvPack<T,N>> sends ,
                       std::string lab = "" ) const {
      int n = receives.size();
      std::vector<MPI_Request> sReq (n);
      std::vector<MPI_Request> rReq (n);
      std::vector<MPI_Status > sStat(n);
      std::vector<MPI_Status > rStat(n);
      #ifdef PORTURB_GPU_AWARE_MPI
        if (lab != "") yakl::timer_start(lab.c_str());
        Kokkos::fence();
        for (int i=0; i < n; i++) {
          auto arr = receives.at(i).arr;
          check( MPI_Irecv( arr.data() , arr.size() , get_type<T>() , receives.at(i).them , receives.at(i).tag , comm , &(rReq.at(i)) ) );
        }
        for (int i=0; i < n; i++) {
          auto arr = sends.at(i).arr;
          check( MPI_Isend( arr.data() , arr.size() , get_type<T>() , sends   .at(i).them , sends   .at(i).tag , comm , &(sReq.at(i)) ) );
        }
        check( MPI_Waitall(n, sReq.data(), sStat.data()) );
        check( MPI_Waitall(n, rReq.data(), rStat.data()) );
        if (lab != "") yakl::timer_stop(lab.c_str());
      #else
        if (lab != "") yakl::timer_start(lab.c_str());
        std::vector<yakl::Array<T,N,yakl::memHost,yakl::styleC>> receive_host_arrays(n);
        std::vector<yakl::Array<T,N,yakl::memHost,yakl::styleC>> send_host_arrays(n);
        for (int i=0; i < n; i++) {
          receive_host_arrays.at(i) = receives.at(i).arr.createHostObject();
          check( MPI_Irecv( receive_host_arrays.at(i).data() , receive_host_arrays.at(i).size() , get_type<T>() ,
                            receives.at(i).them , receives.at(i).tag , comm , &(rReq.at(i)) ) );
        }
        for (int i=0; i < n; i++) {
          send_host_arrays   .at(i) = sends   .at(i).arr.createHostCopy();
          check( MPI_Isend( send_host_arrays.at(i).data() , send_host_arrays.at(i).size() , get_type<T>() ,
                            sends.at(i).them , sends.at(i).tag , comm , &(sReq.at(i)) ) );
        }
        check( MPI_Waitall(n, sReq.data(), sStat.data()) );
        check( MPI_Waitall(n, rReq.data(), rStat.data()) );
        for (int i=0; i < n; i++) { receive_host_arrays.at(i).deep_copy_to(receives.at(i).arr);}
        Kokkos::fence();
        if (lab != "") yakl::timer_stop(lab.c_str());
      #endif
    }


    ////////////////////
    // Allgather
    ////////////////////
    template <class T, typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = false>
    yakl::Array<T,1,yakl::memHost,yakl::styleC> all_gather( T val , std::string lab = "" ) const {
      if (lab != "") yakl::timer_start( lab.c_str() );
      yakl::Array<T,1,yakl::memHost,yakl::styleC> ret("all_gather_result",nranks);
      check( MPI_Allgather( &val , 1 , get_type<T>() , ret.data() , 1 , get_type<T>() , comm ) );
      if (lab != "") yakl::timer_stop( lab.c_str() );
      return ret;
    }


    ////////////////////
    // Barrier
    ////////////////////
    void barrier() const { check( MPI_Barrier(comm) ); }


    ////////////////////
    // Broadcast
    ////////////////////
    template <class T, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    void broadcast( yakl::CSArray<T,N,D0,D1,D2,D3> const & arr , int root = 0 , std::string lab = "" ) const {
      if (nranks == 1) return;
      if (lab != "") yakl::timer_start( lab.c_str() );
      check( MPI_Bcast( arr.data()  , arr.size() , get_type<T>() , root , comm ) );
      if (lab != "") yakl::timer_stop( lab.c_str() );
    }

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
    template <class T, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    yakl::CSArray<T,N,D0,D1,D2,D3> reduce( yakl::CSArray<T,N,D0,D1,D2,D3> loc , MPI_Op op , int root = 0 , std::string lab = "" ) const {
      if (nranks == 1) return loc;
      if (lab != "") yakl::timer_start( lab.c_str() );
      yakl::CSArray<T,N,D0,D1,D2,D3> glob;
      check( MPI_Reduce( loc.data() , glob.data() , loc.size() , get_type<T>() , op , root , comm ) );
      if (lab != "") yakl::timer_stop( lab.c_str() );
      return glob;
    }

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
    template <class T, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    yakl::CSArray<T,N,D0,D1,D2,D3> all_reduce( yakl::CSArray<T,N,D0,D1,D2,D3> loc , MPI_Op op , std::string lab = "" ) const {
      if (nranks == 1) return loc;
      if (lab != "") yakl::timer_start( lab.c_str() );
      yakl::CSArray<T,N,D0,D1,D2,D3> glob;
      check( MPI_Allreduce( loc.data() , glob.data() , loc.size() , get_type<T>() , op , comm ) );
      if (lab != "") yakl::timer_stop( lab.c_str() );
      return glob;
    }

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

    template <class T, typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = false>
    T all_reduce( T loc , MPI_Op op , std::string lab = "" ) const {
      if (nranks == 1) return loc;
      if (lab != "") yakl::timer_start( lab.c_str() );
      T glob;
      check( MPI_Allreduce( &loc , &glob , 1 , get_type<T>() , op , comm ) );
      if (lab != "") yakl::timer_stop( lab.c_str() );
      return glob;
    }


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



#pragma once

#include "mpi.h"
#include "YAKL.h"
#include "yaml-cpp/yaml.h"
#include "Counter.h"
#include <stdexcept>


using yakl::memHost;
using yakl::memDevice;
using yakl::styleC;
using yakl::Array;
using yakl::SArray;

template <class T, int N>
inline void check_for_nan_inf(Array<T,N,memDevice,styleC> arr , std::string file , int line) {
  yakl::ScalarLiveOut<bool> nan_present(false);
  yakl::c::parallel_for( YAKL_AUTO_LABEL() , arr.size() , KOKKOS_LAMBDA (int i) {
    if (std::isnan(arr.data()[i]) || !std::isfinite(arr.data()[i])) nan_present = true;
  });
  if ( nan_present.hostRead() ) std::cerr << file << ":" << line << ":" << arr.label() << ": has NaN or inf" << std::endl;
}

template <class T, typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = false>
inline void check_for_nan_inf(T val , std::string file , int line) {
  if ( std::isnan(val) || !std::isfinite(val) ) std::cerr << file << ":" << line << " is NaN or inf" << std::endl;
}

template <class T, int N, size_t D0, size_t D1, size_t D2, size_t D3>
inline void check_for_nan_inf(SArray<T,N,D0,D1,D2,D3> const & arr , std::string file , int line) {
  bool nan_present = false;
  for (int i=0; i < arr.size(); i++) {
    if (std::isnan(arr.data()[i]) || !std::isfinite(arr.data()[i])) nan_present = true;
  }
  if ( nan_present ) std::cerr << file << ":" << line << " has NaN or inf" << std::endl;
}

inline void debug_print( char const * file , int line ) {
  MPI_Barrier(MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0) std::cout << "*** DEBUG: " << file << ": " << line << std::endl;
}

template <class T> inline void debug_print_sum( T var , char const * file , int line , char const * varname ) {
  MPI_Barrier(MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0) std::cout << "*** DEBUG: " << file << ": " << line << ": sum(" << varname << ")  -->  " << yakl::intrinsics::sum( var ) << std::endl;
}

template <class T> inline void debug_print_avg( T var , char const * file , int line , char const * varname ) {
  MPI_Barrier(MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0) std::cout << "*** DEBUG: " << file << ": " << line << ": avg(" << varname << ")  -->  " << std::scientific << std::setprecision(17) << yakl::intrinsics::sum( var )/var.size() << std::endl;
}

template <class T> inline void debug_print_min( T var , char const * file , int line , char const * varname ) {
  MPI_Barrier(MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0) std::cout << "*** DEBUG: " << file << ": " << line << ": minval(" << varname << ")  -->  " << yakl::intrinsics::minval( var ) << std::endl;
}

template <class T> inline void debug_print_max( T var , char const * file , int line , char const * varname ) {
  MPI_Barrier(MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0) std::cout << "*** DEBUG: " << file << ": " << line << ": maxval(" << varname << ")  -->  " << yakl::intrinsics::maxval( var ) << std::endl;
}

template <class T> inline void debug_print_val( T var , char const * file , int line , char const * varname ) {
  MPI_Barrier(MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0) std::cout << "*** DEBUG: " << file << ": " << line << ": " << varname << "  -->  " << var << std::endl;
}

#define DEBUG_PRINT_MAIN() { debug_print(__FILE__,__LINE__); }
#define DEBUG_PRINT_MAIN_SUM(var) { debug_print_sum((var),__FILE__,__LINE__,#var); }
#define DEBUG_PRINT_MAIN_AVG(var) { debug_print_avg((var),__FILE__,__LINE__,#var); }
#define DEBUG_PRINT_MAIN_MIN(var) { debug_print_min((var),__FILE__,__LINE__,#var); }
#define DEBUG_PRINT_MAIN_MAX(var) { debug_print_max((var),__FILE__,__LINE__,#var); }
#define DEBUG_PRINT_MAIN_VAL(var) { debug_print_val((var),__FILE__,__LINE__,#var); }

#define DEBUG_NAN_INF_VAL(var) { if (std::isnan(var) || !std::isfinite(var)) { printf("WARNING: " #var " has a NaN or inf\n"); } }


int constexpr max_fields = 50;

typedef double real;

KOKKOS_INLINE_FUNCTION real constexpr operator"" _fp( long double x ) {
  return static_cast<real>(x);
}


KOKKOS_INLINE_FUNCTION void endrun(char const * msg = "") {
  Kokkos::abort(msg);
};


inline real timer_last(std::string label) {
  #ifdef YAKL_PROFILE
    return yakl::get_yakl_instance().timer.get_last_duration(label);
  #else
    return 0;
  #endif
}


typedef Array<real,1,memDevice,styleC> real1d;
typedef Array<real,2,memDevice,styleC> real2d;
typedef Array<real,3,memDevice,styleC> real3d;
typedef Array<real,4,memDevice,styleC> real4d;
typedef Array<real,5,memDevice,styleC> real5d;
typedef Array<real,6,memDevice,styleC> real6d;
typedef Array<real,7,memDevice,styleC> real7d;

typedef Array<real const,1,memDevice,styleC> realConst1d;
typedef Array<real const,2,memDevice,styleC> realConst2d;
typedef Array<real const,3,memDevice,styleC> realConst3d;
typedef Array<real const,4,memDevice,styleC> realConst4d;
typedef Array<real const,5,memDevice,styleC> realConst5d;
typedef Array<real const,6,memDevice,styleC> realConst6d;
typedef Array<real const,7,memDevice,styleC> realConst7d;

typedef Array<real,1,memHost,styleC> realHost1d;
typedef Array<real,2,memHost,styleC> realHost2d;
typedef Array<real,3,memHost,styleC> realHost3d;
typedef Array<real,4,memHost,styleC> realHost4d;
typedef Array<real,5,memHost,styleC> realHost5d;
typedef Array<real,6,memHost,styleC> realHost6d;
typedef Array<real,7,memHost,styleC> realHost7d;

typedef Array<real const,1,memHost,styleC> realConstHost1d;
typedef Array<real const,2,memHost,styleC> realConstHost2d;
typedef Array<real const,3,memHost,styleC> realConstHost3d;
typedef Array<real const,4,memHost,styleC> realConstHost4d;
typedef Array<real const,5,memHost,styleC> realConstHost5d;
typedef Array<real const,6,memHost,styleC> realConstHost6d;
typedef Array<real const,7,memHost,styleC> realConstHost7d;



typedef Array<int,1,memDevice,styleC> int1d;
typedef Array<int,2,memDevice,styleC> int2d;
typedef Array<int,3,memDevice,styleC> int3d;
typedef Array<int,4,memDevice,styleC> int4d;
typedef Array<int,5,memDevice,styleC> int5d;
typedef Array<int,6,memDevice,styleC> int6d;
typedef Array<int,7,memDevice,styleC> int7d;

typedef Array<int const,1,memDevice,styleC> intConst1d;
typedef Array<int const,2,memDevice,styleC> intConst2d;
typedef Array<int const,3,memDevice,styleC> intConst3d;
typedef Array<int const,4,memDevice,styleC> intConst4d;
typedef Array<int const,5,memDevice,styleC> intConst5d;
typedef Array<int const,6,memDevice,styleC> intConst6d;
typedef Array<int const,7,memDevice,styleC> intConst7d;

typedef Array<int,1,memHost,styleC> intHost1d;
typedef Array<int,2,memHost,styleC> intHost2d;
typedef Array<int,3,memHost,styleC> intHost3d;
typedef Array<int,4,memHost,styleC> intHost4d;
typedef Array<int,5,memHost,styleC> intHost5d;
typedef Array<int,6,memHost,styleC> intHost6d;
typedef Array<int,7,memHost,styleC> intHost7d;

typedef Array<int const,1,memHost,styleC> intConstHost1d;
typedef Array<int const,2,memHost,styleC> intConstHost2d;
typedef Array<int const,3,memHost,styleC> intConstHost3d;
typedef Array<int const,4,memHost,styleC> intConstHost4d;
typedef Array<int const,5,memHost,styleC> intConstHost5d;
typedef Array<int const,6,memHost,styleC> intConstHost6d;
typedef Array<int const,7,memHost,styleC> intConstHost7d;



typedef Array<char,1,memDevice,styleC> char1d;
typedef Array<char,2,memDevice,styleC> char2d;
typedef Array<char,3,memDevice,styleC> char3d;
typedef Array<char,4,memDevice,styleC> char4d;
typedef Array<char,5,memDevice,styleC> char5d;
typedef Array<char,6,memDevice,styleC> char6d;
typedef Array<char,7,memDevice,styleC> char7d;

typedef Array<char const,1,memDevice,styleC> charConst1d;
typedef Array<char const,2,memDevice,styleC> charConst2d;
typedef Array<char const,3,memDevice,styleC> charConst3d;
typedef Array<char const,4,memDevice,styleC> charConst4d;
typedef Array<char const,5,memDevice,styleC> charConst5d;
typedef Array<char const,6,memDevice,styleC> charConst6d;
typedef Array<char const,7,memDevice,styleC> charConst7d;

typedef Array<char,1,memHost,styleC> charHost1d;
typedef Array<char,2,memHost,styleC> charHost2d;
typedef Array<char,3,memHost,styleC> charHost3d;
typedef Array<char,4,memHost,styleC> charHost4d;
typedef Array<char,5,memHost,styleC> charHost5d;
typedef Array<char,6,memHost,styleC> charHost6d;
typedef Array<char,7,memHost,styleC> charHost7d;

typedef Array<char const,1,memHost,styleC> charConstHost1d;
typedef Array<char const,2,memHost,styleC> charConstHost2d;
typedef Array<char const,3,memHost,styleC> charConstHost3d;
typedef Array<char const,4,memHost,styleC> charConstHost4d;
typedef Array<char const,5,memHost,styleC> charConstHost5d;
typedef Array<char const,6,memHost,styleC> charConstHost6d;
typedef Array<char const,7,memHost,styleC> charConstHost7d;



typedef Array<bool,1,memDevice,styleC> bool1d;
typedef Array<bool,2,memDevice,styleC> bool2d;
typedef Array<bool,3,memDevice,styleC> bool3d;
typedef Array<bool,4,memDevice,styleC> bool4d;
typedef Array<bool,5,memDevice,styleC> bool5d;
typedef Array<bool,6,memDevice,styleC> bool6d;
typedef Array<bool,7,memDevice,styleC> bool7d;

typedef Array<bool const,1,memDevice,styleC> boolConst1d;
typedef Array<bool const,2,memDevice,styleC> boolConst2d;
typedef Array<bool const,3,memDevice,styleC> boolConst3d;
typedef Array<bool const,4,memDevice,styleC> boolConst4d;
typedef Array<bool const,5,memDevice,styleC> boolConst5d;
typedef Array<bool const,6,memDevice,styleC> boolConst6d;
typedef Array<bool const,7,memDevice,styleC> boolConst7d;

typedef Array<bool,1,memHost,styleC> boolHost1d;
typedef Array<bool,2,memHost,styleC> boolHost2d;
typedef Array<bool,3,memHost,styleC> boolHost3d;
typedef Array<bool,4,memHost,styleC> boolHost4d;
typedef Array<bool,5,memHost,styleC> boolHost5d;
typedef Array<bool,6,memHost,styleC> boolHost6d;
typedef Array<bool,7,memHost,styleC> boolHost7d;

typedef Array<bool const,1,memHost,styleC> boolConstHost1d;
typedef Array<bool const,2,memHost,styleC> boolConstHost2d;
typedef Array<bool const,3,memHost,styleC> boolConstHost3d;
typedef Array<bool const,4,memHost,styleC> boolConstHost4d;
typedef Array<bool const,5,memHost,styleC> boolConstHost5d;
typedef Array<bool const,6,memHost,styleC> boolConstHost6d;
typedef Array<bool const,7,memHost,styleC> boolConstHost7d;



typedef Array<float,1,memDevice,styleC> float1d;
typedef Array<float,2,memDevice,styleC> float2d;
typedef Array<float,3,memDevice,styleC> float3d;
typedef Array<float,4,memDevice,styleC> float4d;
typedef Array<float,5,memDevice,styleC> float5d;
typedef Array<float,6,memDevice,styleC> float6d;
typedef Array<float,7,memDevice,styleC> float7d;

typedef Array<float const,1,memDevice,styleC> floatConst1d;
typedef Array<float const,2,memDevice,styleC> floatConst2d;
typedef Array<float const,3,memDevice,styleC> floatConst3d;
typedef Array<float const,4,memDevice,styleC> floatConst4d;
typedef Array<float const,5,memDevice,styleC> floatConst5d;
typedef Array<float const,6,memDevice,styleC> floatConst6d;
typedef Array<float const,7,memDevice,styleC> floatConst7d;

typedef Array<float,1,memHost,styleC> floatHost1d;
typedef Array<float,2,memHost,styleC> floatHost2d;
typedef Array<float,3,memHost,styleC> floatHost3d;
typedef Array<float,4,memHost,styleC> floatHost4d;
typedef Array<float,5,memHost,styleC> floatHost5d;
typedef Array<float,6,memHost,styleC> floatHost6d;
typedef Array<float,7,memHost,styleC> floatHost7d;

typedef Array<float const,1,memHost,styleC> floatConstHost1d;
typedef Array<float const,2,memHost,styleC> floatConstHost2d;
typedef Array<float const,3,memHost,styleC> floatConstHost3d;
typedef Array<float const,4,memHost,styleC> floatConstHost4d;
typedef Array<float const,5,memHost,styleC> floatConstHost5d;
typedef Array<float const,6,memHost,styleC> floatConstHost6d;
typedef Array<float const,7,memHost,styleC> floatConstHost7d;



typedef Array<double,1,memDevice,styleC> double1d;
typedef Array<double,2,memDevice,styleC> double2d;
typedef Array<double,3,memDevice,styleC> double3d;
typedef Array<double,4,memDevice,styleC> double4d;
typedef Array<double,5,memDevice,styleC> double5d;
typedef Array<double,6,memDevice,styleC> double6d;
typedef Array<double,7,memDevice,styleC> double7d;

typedef Array<double const,1,memDevice,styleC> doubleConst1d;
typedef Array<double const,2,memDevice,styleC> doubleConst2d;
typedef Array<double const,3,memDevice,styleC> doubleConst3d;
typedef Array<double const,4,memDevice,styleC> doubleConst4d;
typedef Array<double const,5,memDevice,styleC> doubleConst5d;
typedef Array<double const,6,memDevice,styleC> doubleConst6d;
typedef Array<double const,7,memDevice,styleC> doubleConst7d;

typedef Array<double,1,memHost,styleC> doubleHost1d;
typedef Array<double,2,memHost,styleC> doubleHost2d;
typedef Array<double,3,memHost,styleC> doubleHost3d;
typedef Array<double,4,memHost,styleC> doubleHost4d;
typedef Array<double,5,memHost,styleC> doubleHost5d;
typedef Array<double,6,memHost,styleC> doubleHost6d;
typedef Array<double,7,memHost,styleC> doubleHost7d;

typedef Array<double const,1,memHost,styleC> doubleConstHost1d;
typedef Array<double const,2,memHost,styleC> doubleConstHost2d;
typedef Array<double const,3,memHost,styleC> doubleConstHost3d;
typedef Array<double const,4,memHost,styleC> doubleConstHost4d;
typedef Array<double const,5,memHost,styleC> doubleConstHost5d;
typedef Array<double const,6,memHost,styleC> doubleConstHost6d;
typedef Array<double const,7,memHost,styleC> doubleConstHost7d;




typedef Array<size_t,1,memDevice,styleC> size_t1d;
typedef Array<size_t,2,memDevice,styleC> size_t2d;
typedef Array<size_t,3,memDevice,styleC> size_t3d;
typedef Array<size_t,4,memDevice,styleC> size_t4d;
typedef Array<size_t,5,memDevice,styleC> size_t5d;
typedef Array<size_t,6,memDevice,styleC> size_t6d;
typedef Array<size_t,7,memDevice,styleC> size_t7d;

typedef Array<size_t const,1,memDevice,styleC> size_tConst1d;
typedef Array<size_t const,2,memDevice,styleC> size_tConst2d;
typedef Array<size_t const,3,memDevice,styleC> size_tConst3d;
typedef Array<size_t const,4,memDevice,styleC> size_tConst4d;
typedef Array<size_t const,5,memDevice,styleC> size_tConst5d;
typedef Array<size_t const,6,memDevice,styleC> size_tConst6d;
typedef Array<size_t const,7,memDevice,styleC> size_tConst7d;

typedef Array<size_t,1,memHost,styleC> size_tHost1d;
typedef Array<size_t,2,memHost,styleC> size_tHost2d;
typedef Array<size_t,3,memHost,styleC> size_tHost3d;
typedef Array<size_t,4,memHost,styleC> size_tHost4d;
typedef Array<size_t,5,memHost,styleC> size_tHost5d;
typedef Array<size_t,6,memHost,styleC> size_tHost6d;
typedef Array<size_t,7,memHost,styleC> size_tHost7d;

typedef Array<size_t const,1,memHost,styleC> size_tConstHost1d;
typedef Array<size_t const,2,memHost,styleC> size_tConstHost2d;
typedef Array<size_t const,3,memHost,styleC> size_tConstHost3d;
typedef Array<size_t const,4,memHost,styleC> size_tConstHost4d;
typedef Array<size_t const,5,memHost,styleC> size_tConstHost5d;
typedef Array<size_t const,6,memHost,styleC> size_tConstHost6d;
typedef Array<size_t const,7,memHost,styleC> size_tConstHost7d;



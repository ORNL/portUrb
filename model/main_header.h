
#pragma once

#include "mpi.h"
#include "YAKL.h"
#include "yaml-cpp/yaml.h"
#include "Counter.h"
#include <stdexcept>


using yakl::Array;
using yakl::SArray;

template <class ViewType>
inline void check_for_nan_inf(ViewType const & arr , std::string file , int line) {
  using yakl::componentwise::operator||;
  using yakl::componentwise::isnan;
  using yakl::componentwise::isinf;
  if ( yakl::intrinsics::any(isnan(arr) || isinf(arr)) ) std::cerr << file << ":" << line << ":"
                                                                   << arr.label() << ": has NaN or inf" << std::endl;
}

template <class T> requires std::is_arithmetic_v<T>
inline void check_for_nan_inf(T val , std::string file , int line) {
  if ( std::isnan(val) || !std::isfinite(val) ) std::cerr << file << ":" << line << " is NaN or inf" << std::endl;
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

KOKKOS_INLINE_FUNCTION real constexpr operator""_fp( long double x ) {
  return static_cast<real>(x);
}


KOKKOS_INLINE_FUNCTION void endrun(char const * msg = "") {
  Kokkos::abort(msg);
};


typedef Array<real *            ,yakl::DeviceSpace> real1d;
typedef Array<real **           ,yakl::DeviceSpace> real2d;
typedef Array<real ***          ,yakl::DeviceSpace> real3d;
typedef Array<real ****         ,yakl::DeviceSpace> real4d;
typedef Array<real *****        ,yakl::DeviceSpace> real5d;
typedef Array<real ******       ,yakl::DeviceSpace> real6d;
typedef Array<real *******      ,yakl::DeviceSpace> real7d;
typedef Array<real const *      ,yakl::DeviceSpace> realConst1d;
typedef Array<real const **     ,yakl::DeviceSpace> realConst2d;
typedef Array<real const ***    ,yakl::DeviceSpace> realConst3d;
typedef Array<real const ****   ,yakl::DeviceSpace> realConst4d;
typedef Array<real const *****  ,yakl::DeviceSpace> realConst5d;
typedef Array<real const ****** ,yakl::DeviceSpace> realConst6d;
typedef Array<real const *******,yakl::DeviceSpace> realConst7d;
typedef Array<real *            ,Kokkos::HostSpace> realHost1d;
typedef Array<real **           ,Kokkos::HostSpace> realHost2d;
typedef Array<real ***          ,Kokkos::HostSpace> realHost3d;
typedef Array<real ****         ,Kokkos::HostSpace> realHost4d;
typedef Array<real *****        ,Kokkos::HostSpace> realHost5d;
typedef Array<real ******       ,Kokkos::HostSpace> realHost6d;
typedef Array<real *******      ,Kokkos::HostSpace> realHost7d;
typedef Array<real const *      ,Kokkos::HostSpace> realConstHost1d;
typedef Array<real const **     ,Kokkos::HostSpace> realConstHost2d;
typedef Array<real const ***    ,Kokkos::HostSpace> realConstHost3d;
typedef Array<real const ****   ,Kokkos::HostSpace> realConstHost4d;
typedef Array<real const *****  ,Kokkos::HostSpace> realConstHost5d;
typedef Array<real const ****** ,Kokkos::HostSpace> realConstHost6d;
typedef Array<real const *******,Kokkos::HostSpace> realConstHost7d;

typedef Array<int *            ,yakl::DeviceSpace> int1d;
typedef Array<int **           ,yakl::DeviceSpace> int2d;
typedef Array<int ***          ,yakl::DeviceSpace> int3d;
typedef Array<int ****         ,yakl::DeviceSpace> int4d;
typedef Array<int *****        ,yakl::DeviceSpace> int5d;
typedef Array<int ******       ,yakl::DeviceSpace> int6d;
typedef Array<int *******      ,yakl::DeviceSpace> int7d;
typedef Array<int const *      ,yakl::DeviceSpace> intConst1d;
typedef Array<int const **     ,yakl::DeviceSpace> intConst2d;
typedef Array<int const ***    ,yakl::DeviceSpace> intConst3d;
typedef Array<int const ****   ,yakl::DeviceSpace> intConst4d;
typedef Array<int const *****  ,yakl::DeviceSpace> intConst5d;
typedef Array<int const ****** ,yakl::DeviceSpace> intConst6d;
typedef Array<int const *******,yakl::DeviceSpace> intConst7d;
typedef Array<int *            ,Kokkos::HostSpace> intHost1d;
typedef Array<int **           ,Kokkos::HostSpace> intHost2d;
typedef Array<int ***          ,Kokkos::HostSpace> intHost3d;
typedef Array<int ****         ,Kokkos::HostSpace> intHost4d;
typedef Array<int *****        ,Kokkos::HostSpace> intHost5d;
typedef Array<int ******       ,Kokkos::HostSpace> intHost6d;
typedef Array<int *******      ,Kokkos::HostSpace> intHost7d;
typedef Array<int const *      ,Kokkos::HostSpace> intConstHost1d;
typedef Array<int const **     ,Kokkos::HostSpace> intConstHost2d;
typedef Array<int const ***    ,Kokkos::HostSpace> intConstHost3d;
typedef Array<int const ****   ,Kokkos::HostSpace> intConstHost4d;
typedef Array<int const *****  ,Kokkos::HostSpace> intConstHost5d;
typedef Array<int const ****** ,Kokkos::HostSpace> intConstHost6d;
typedef Array<int const *******,Kokkos::HostSpace> intConstHost7d;

typedef Array<char *            ,yakl::DeviceSpace> char1d;
typedef Array<char **           ,yakl::DeviceSpace> char2d;
typedef Array<char ***          ,yakl::DeviceSpace> char3d;
typedef Array<char ****         ,yakl::DeviceSpace> char4d;
typedef Array<char *****        ,yakl::DeviceSpace> char5d;
typedef Array<char ******       ,yakl::DeviceSpace> char6d;
typedef Array<char *******      ,yakl::DeviceSpace> char7d;
typedef Array<char const *      ,yakl::DeviceSpace> charConst1d;
typedef Array<char const **     ,yakl::DeviceSpace> charConst2d;
typedef Array<char const ***    ,yakl::DeviceSpace> charConst3d;
typedef Array<char const ****   ,yakl::DeviceSpace> charConst4d;
typedef Array<char const *****  ,yakl::DeviceSpace> charConst5d;
typedef Array<char const ****** ,yakl::DeviceSpace> charConst6d;
typedef Array<char const *******,yakl::DeviceSpace> charConst7d;
typedef Array<char *            ,Kokkos::HostSpace> charHost1d;
typedef Array<char **           ,Kokkos::HostSpace> charHost2d;
typedef Array<char ***          ,Kokkos::HostSpace> charHost3d;
typedef Array<char ****         ,Kokkos::HostSpace> charHost4d;
typedef Array<char *****        ,Kokkos::HostSpace> charHost5d;
typedef Array<char ******       ,Kokkos::HostSpace> charHost6d;
typedef Array<char *******      ,Kokkos::HostSpace> charHost7d;
typedef Array<char const *      ,Kokkos::HostSpace> charConstHost1d;
typedef Array<char const **     ,Kokkos::HostSpace> charConstHost2d;
typedef Array<char const ***    ,Kokkos::HostSpace> charConstHost3d;
typedef Array<char const ****   ,Kokkos::HostSpace> charConstHost4d;
typedef Array<char const *****  ,Kokkos::HostSpace> charConstHost5d;
typedef Array<char const ****** ,Kokkos::HostSpace> charConstHost6d;
typedef Array<char const *******,Kokkos::HostSpace> charConstHost7d;

typedef Array<bool *            ,yakl::DeviceSpace> bool1d;
typedef Array<bool **           ,yakl::DeviceSpace> bool2d;
typedef Array<bool ***          ,yakl::DeviceSpace> bool3d;
typedef Array<bool ****         ,yakl::DeviceSpace> bool4d;
typedef Array<bool *****        ,yakl::DeviceSpace> bool5d;
typedef Array<bool ******       ,yakl::DeviceSpace> bool6d;
typedef Array<bool *******      ,yakl::DeviceSpace> bool7d;
typedef Array<bool const *      ,yakl::DeviceSpace> boolConst1d;
typedef Array<bool const **     ,yakl::DeviceSpace> boolConst2d;
typedef Array<bool const ***    ,yakl::DeviceSpace> boolConst3d;
typedef Array<bool const ****   ,yakl::DeviceSpace> boolConst4d;
typedef Array<bool const *****  ,yakl::DeviceSpace> boolConst5d;
typedef Array<bool const ****** ,yakl::DeviceSpace> boolConst6d;
typedef Array<bool const *******,yakl::DeviceSpace> boolConst7d;
typedef Array<bool *            ,Kokkos::HostSpace> boolHost1d;
typedef Array<bool **           ,Kokkos::HostSpace> boolHost2d;
typedef Array<bool ***          ,Kokkos::HostSpace> boolHost3d;
typedef Array<bool ****         ,Kokkos::HostSpace> boolHost4d;
typedef Array<bool *****        ,Kokkos::HostSpace> boolHost5d;
typedef Array<bool ******       ,Kokkos::HostSpace> boolHost6d;
typedef Array<bool *******      ,Kokkos::HostSpace> boolHost7d;
typedef Array<bool const *      ,Kokkos::HostSpace> boolConstHost1d;
typedef Array<bool const **     ,Kokkos::HostSpace> boolConstHost2d;
typedef Array<bool const ***    ,Kokkos::HostSpace> boolConstHost3d;
typedef Array<bool const ****   ,Kokkos::HostSpace> boolConstHost4d;
typedef Array<bool const *****  ,Kokkos::HostSpace> boolConstHost5d;
typedef Array<bool const ****** ,Kokkos::HostSpace> boolConstHost6d;
typedef Array<bool const *******,Kokkos::HostSpace> boolConstHost7d;

typedef Array<float *            ,yakl::DeviceSpace> float1d;
typedef Array<float **           ,yakl::DeviceSpace> float2d;
typedef Array<float ***          ,yakl::DeviceSpace> float3d;
typedef Array<float ****         ,yakl::DeviceSpace> float4d;
typedef Array<float *****        ,yakl::DeviceSpace> float5d;
typedef Array<float ******       ,yakl::DeviceSpace> float6d;
typedef Array<float *******      ,yakl::DeviceSpace> float7d;
typedef Array<float const *      ,yakl::DeviceSpace> floatConst1d;
typedef Array<float const **     ,yakl::DeviceSpace> floatConst2d;
typedef Array<float const ***    ,yakl::DeviceSpace> floatConst3d;
typedef Array<float const ****   ,yakl::DeviceSpace> floatConst4d;
typedef Array<float const *****  ,yakl::DeviceSpace> floatConst5d;
typedef Array<float const ****** ,yakl::DeviceSpace> floatConst6d;
typedef Array<float const *******,yakl::DeviceSpace> floatConst7d;
typedef Array<float *            ,Kokkos::HostSpace> floatHost1d;
typedef Array<float **           ,Kokkos::HostSpace> floatHost2d;
typedef Array<float ***          ,Kokkos::HostSpace> floatHost3d;
typedef Array<float ****         ,Kokkos::HostSpace> floatHost4d;
typedef Array<float *****        ,Kokkos::HostSpace> floatHost5d;
typedef Array<float ******       ,Kokkos::HostSpace> floatHost6d;
typedef Array<float *******      ,Kokkos::HostSpace> floatHost7d;
typedef Array<float const *      ,Kokkos::HostSpace> floatConstHost1d;
typedef Array<float const **     ,Kokkos::HostSpace> floatConstHost2d;
typedef Array<float const ***    ,Kokkos::HostSpace> floatConstHost3d;
typedef Array<float const ****   ,Kokkos::HostSpace> floatConstHost4d;
typedef Array<float const *****  ,Kokkos::HostSpace> floatConstHost5d;
typedef Array<float const ****** ,Kokkos::HostSpace> floatConstHost6d;
typedef Array<float const *******,Kokkos::HostSpace> floatConstHost7d;

typedef Array<double *            ,yakl::DeviceSpace> double1d;
typedef Array<double **           ,yakl::DeviceSpace> double2d;
typedef Array<double ***          ,yakl::DeviceSpace> double3d;
typedef Array<double ****         ,yakl::DeviceSpace> double4d;
typedef Array<double *****        ,yakl::DeviceSpace> double5d;
typedef Array<double ******       ,yakl::DeviceSpace> double6d;
typedef Array<double *******      ,yakl::DeviceSpace> double7d;
typedef Array<double const *      ,yakl::DeviceSpace> doubleConst1d;
typedef Array<double const **     ,yakl::DeviceSpace> doubleConst2d;
typedef Array<double const ***    ,yakl::DeviceSpace> doubleConst3d;
typedef Array<double const ****   ,yakl::DeviceSpace> doubleConst4d;
typedef Array<double const *****  ,yakl::DeviceSpace> doubleConst5d;
typedef Array<double const ****** ,yakl::DeviceSpace> doubleConst6d;
typedef Array<double const *******,yakl::DeviceSpace> doubleConst7d;
typedef Array<double *            ,Kokkos::HostSpace> doubleHost1d;
typedef Array<double **           ,Kokkos::HostSpace> doubleHost2d;
typedef Array<double ***          ,Kokkos::HostSpace> doubleHost3d;
typedef Array<double ****         ,Kokkos::HostSpace> doubleHost4d;
typedef Array<double *****        ,Kokkos::HostSpace> doubleHost5d;
typedef Array<double ******       ,Kokkos::HostSpace> doubleHost6d;
typedef Array<double *******      ,Kokkos::HostSpace> doubleHost7d;
typedef Array<double const *      ,Kokkos::HostSpace> doubleConstHost1d;
typedef Array<double const **     ,Kokkos::HostSpace> doubleConstHost2d;
typedef Array<double const ***    ,Kokkos::HostSpace> doubleConstHost3d;
typedef Array<double const ****   ,Kokkos::HostSpace> doubleConstHost4d;
typedef Array<double const *****  ,Kokkos::HostSpace> doubleConstHost5d;
typedef Array<double const ****** ,Kokkos::HostSpace> doubleConstHost6d;
typedef Array<double const *******,Kokkos::HostSpace> doubleConstHost7d;

typedef Array<size_t *            ,yakl::DeviceSpace> size_t1d;
typedef Array<size_t **           ,yakl::DeviceSpace> size_t2d;
typedef Array<size_t ***          ,yakl::DeviceSpace> size_t3d;
typedef Array<size_t ****         ,yakl::DeviceSpace> size_t4d;
typedef Array<size_t *****        ,yakl::DeviceSpace> size_t5d;
typedef Array<size_t ******       ,yakl::DeviceSpace> size_t6d;
typedef Array<size_t *******      ,yakl::DeviceSpace> size_t7d;
typedef Array<size_t const *      ,yakl::DeviceSpace> size_tConst1d;
typedef Array<size_t const **     ,yakl::DeviceSpace> size_tConst2d;
typedef Array<size_t const ***    ,yakl::DeviceSpace> size_tConst3d;
typedef Array<size_t const ****   ,yakl::DeviceSpace> size_tConst4d;
typedef Array<size_t const *****  ,yakl::DeviceSpace> size_tConst5d;
typedef Array<size_t const ****** ,yakl::DeviceSpace> size_tConst6d;
typedef Array<size_t const *******,yakl::DeviceSpace> size_tConst7d;
typedef Array<size_t *            ,Kokkos::HostSpace> size_tHost1d;
typedef Array<size_t **           ,Kokkos::HostSpace> size_tHost2d;
typedef Array<size_t ***          ,Kokkos::HostSpace> size_tHost3d;
typedef Array<size_t ****         ,Kokkos::HostSpace> size_tHost4d;
typedef Array<size_t *****        ,Kokkos::HostSpace> size_tHost5d;
typedef Array<size_t ******       ,Kokkos::HostSpace> size_tHost6d;
typedef Array<size_t *******      ,Kokkos::HostSpace> size_tHost7d;
typedef Array<size_t const *      ,Kokkos::HostSpace> size_tConstHost1d;
typedef Array<size_t const **     ,Kokkos::HostSpace> size_tConstHost2d;
typedef Array<size_t const ***    ,Kokkos::HostSpace> size_tConstHost3d;
typedef Array<size_t const ****   ,Kokkos::HostSpace> size_tConstHost4d;
typedef Array<size_t const *****  ,Kokkos::HostSpace> size_tConstHost5d;
typedef Array<size_t const ****** ,Kokkos::HostSpace> size_tConstHost6d;
typedef Array<size_t const *******,Kokkos::HostSpace> size_tConstHost7d;



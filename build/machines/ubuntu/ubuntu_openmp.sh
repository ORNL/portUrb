#!/bin/bash

export CC=mpicc
export FC=mpif90
export CXX=mpic++

myinclude="-I`nc-config --includedir`;-I`pnetcdf-config --includedir`"
mylibstmp=`nc-config --libs`
mylibs=`echo $mylibstmp | sed 's/[ ]\+/;/g'`
mylibstmp=`pnetcdf-config --libs`
mylibs+=";`echo $mylibstmp | sed 's/[ ]\+/;/g'`"

export PORTURB_BACKEND="Kokkos_ENABLE_OPENMP"
export PORTURB_ARCH="Kokkos_ARCH_NATIVE"
export PORTURB_CXX_FLAGS="-DHAVE_MPI;-DYAKL_PROFILE;-O3;${myinclude};-fopenmp"
export PORTURB_F90_FLAGS="-cpp;-ffree-line-length-none;-O3;-fdefault-real-8;-fdefault-double-8"
export PORTURB_LINK_FLAGS="-L`pnetcdf-config --libdir` -lpnetcdf `nc-config --libs` -fopenmp"
export PORTURB_DEBUG=OFF

unset CXXFLAGS
unset FFLAGS
unset F77FLAGS
unset F90FLAGS

# -DPORTURB_FUNCTION_TIMERS
# -DYAKL_AUTO_PROFILE
# -DPORTURB_FUNCTION_TIMER_BARRIER 
# -DPORTURB_GPU_AWARE_MPI
# -DPORTURB_NAN_CHECKS

#!/bin/bash

./cmakeclean.sh

if [ $# -ne 1 ]; then
  echo "Error: must pass exactly one parameter giving the directory to build"
  exit -1
fi

if [ ! -d $1 ]; then
  echo "Error: Passed directory does not exist"
  exit -1
fi

if [ ! -f $1/CMakeLists.txt ]; then
  echo "Error: Passed directory does not contain a CMakeLists.txt file"
  exit -1
fi

[[ "$PORTURB_DEBUG" == "ON" ]] && PORTURB_CXX_FLAGS+=";-DYAKL_DEBUG"
[[ ! "$PORTURB_DEBUG" == "ON" ]] && PORTURB_CXX_FLAGS+=";-DNDEBUG"
CMAKE_COMMAND=(cmake -Wno-dev -DPORTURB_HOME="`pwd`/..")
CMAKE_COMMAND+=(-DPORTURB_CXX_FLAGS="$PORTURB_CXX_FLAGS")
CMAKE_COMMAND+=(-DPORTURB_LINK_FLAGS="$PORTURB_LINK_FLAGS")
CMAKE_COMMAND+=(-DPORTURB_F90_FLAGS="$PORTURB_F90_FLAGS")
[[ ! "$PORTURB_BACKEND" == "" ]] && CMAKE_COMMAND+=(-D${PORTURB_BACKEND}=ON)
[[ ! "$PORTURB_ARCH"    == "" ]] && CMAKE_COMMAND+=(-D${PORTURB_ARCH}=ON)
[[ "$PORTURB_BACKEND" == "Kokkos_ENABLE_CUDA" ]] && CMAKE_COMMAND+=(-DKokkos_ENABLE_CUDA_CONSTEXPR=ON)
[[ "$PORTURB_BACKEND" == "Kokkos_ENABLE_CUDA"  && "$KOKKOS_USE_NVHPC" == "1" ]] && CMAKE_COMMAND+=(-DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON)
[[ "$PORTURB_DEBUG" == "ON" ]] && CMAKE_COMMAND+=(-DKokkos_ENABLE_DEBUG=ON -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON)
[[ "$PORTURB_INDEX_BITS" == "32" ]] && CMAKE_COMMAND+=(-DYAKL_INDEX_BITS=32)
CMAKE_COMMAND+=($1)

echo "${CMAKE_COMMAND[@]}"

"${CMAKE_COMMAND[@]}"

ln -sf $1/inputs .


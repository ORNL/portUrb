
#pragma once

#include "coupler.h"
#include "Mp_morr_two_moment.h"


namespace modules {

  // Implements the interface to the 2-moment Morrison microphysics scheme, which is in MP_morr_two_moment.h
  struct Microphysics_Morrison {
    // Declare Fortran-style YAKL arrays for use in microphysics calculations, which are indexed with the first
    //   index varying the fastest and have lower bounds that default to 1 but can be changed.
    typedef yakl::Array_F<double       * ,yakl::DeviceSpace> double1d_F;
    typedef yakl::Array_F<double       **,yakl::DeviceSpace> double2d_F;
    typedef yakl::Array_F<double const * ,yakl::DeviceSpace> doubleConst1d_F;
    typedef yakl::Array_F<double const **,yakl::DeviceSpace> doubleConst2d_F;
    typedef yakl::Array_F<double       * ,Kokkos::HostSpace> doubleHost1d_F;
    typedef yakl::Array_F<double       **,Kokkos::HostSpace> doubleHost2d_F;
    typedef yakl::Array_F<double const * ,Kokkos::HostSpace> doubleHostConst1d_F;
    typedef yakl::Array_F<double const **,Kokkos::HostSpace> doubleHostConst2d_F;
    typedef yakl::Array_F<int          * ,yakl::DeviceSpace> int1d_F;
    typedef yakl::Array_F<int          **,yakl::DeviceSpace> int2d_F;
    typedef yakl::Array_F<int    const * ,yakl::DeviceSpace> intConst1d_F;
    typedef yakl::Array_F<int    const **,yakl::DeviceSpace> intConst2d_F;
    typedef yakl::Array_F<int          * ,Kokkos::HostSpace> intHost1d_F;
    typedef yakl::Array_F<int          **,Kokkos::HostSpace> intHost2d_F;
    typedef yakl::Array_F<int    const * ,Kokkos::HostSpace> intHostConst1d_F;
    typedef yakl::Array_F<int    const **,Kokkos::HostSpace> intHostConst2d_F;
    typedef yakl::Array_F<bool         * ,yakl::DeviceSpace> bool1d_F;
    typedef yakl::Array_F<bool         **,yakl::DeviceSpace> bool2d_F;
    typedef yakl::Array_F<bool   const * ,yakl::DeviceSpace> boolConst1d_F;
    typedef yakl::Array_F<bool   const **,yakl::DeviceSpace> boolConst2d_F;
    typedef yakl::Array_F<bool         * ,Kokkos::HostSpace> boolHost1d_F;
    typedef yakl::Array_F<bool         **,Kokkos::HostSpace> boolHost2d_F;
    typedef yakl::Array_F<bool   const * ,Kokkos::HostSpace> boolHostConst1d_F;
    typedef yakl::Array_F<bool   const **,Kokkos::HostSpace> boolHostConst2d_F;
    // Doesn't actually have to be static or constexpr. Could be assigned in the constructor
    int static constexpr num_tracers = 10;

    // Create an instance of the Morrison microphysics class to use its methods
    // This actually implements the port of the 2-moment Morrison microphysics scheme to portable C++
    Mp_morr_two_moment  micro;


    // Returns the number of tracers used by this microphysics scheme
    KOKKOS_INLINE_FUNCTION static int get_num_tracers() { return num_tracers; }


    // Initializes the microphysics module within the coupler by registering tracers and persistent variables,
    //   initializing tracers to zero, and setting microphysics-related options in the coupler
    void init(core::Coupler &coupler);



    // Advances the microphysics scheme by one time step of length dt within the coupler
    // coupler : Reference to the coupler object
    // dt      : Time step length in seconds
    void time_step( core::Coupler &coupler , real dt );


    // Returns the name of this microphysics scheme
    std::string micro_name() const;


  };

}



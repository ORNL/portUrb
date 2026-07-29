
#pragma once

#include "coupler.h"

namespace modules {
  
  // This function scales the fluctuations about the mean of the specified fields
  //   by a factor determined by frac, dt, and tscale
  // coupler : Coupler object to access domain and data information
  // dt      : Timestep size
  // frac    : Fractional reduction in fluctuations over the timescale tscale
  // tscale  : Timescale over which fluctuations are reduced by the fraction frac
  // vnames  : Vector of variable names to apply fluctuation scaling to
  // This is typically used to modify turbulent precursor inflow to different turbulence intensities
  // Typically you'll save the precursor data, then apply this routine to scale the turbulence,
  //   and then restore the precursor data to its original values so that only the inflow is modified
  void fluctuation_scaling( core::Coupler            & coupler ,
                                   real dt                            ,
                                   real frac                          ,
                                   real tscale                        ,
                                   std::vector<std::string>   vnames  );
}



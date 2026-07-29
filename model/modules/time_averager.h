#pragma once

#include "coupler.h"

namespace modules {
  
  // This class keeps track of time-averaged u, v, w
  struct Time_Averager {

    std::vector<std::string> add_3d_varnames;

    // Allocate and initialize time-averaged fields since the last reset
    // Also, register these fields as output variables with the coupler for output and restart
    void init( core::Coupler &coupler , std::vector<std::string> add_3d_varnames = {} );


    // Reset time-averaged fields since the last reset to zero
    void reset( core::Coupler &coupler );


    // Accumulate time-averaged fields since the last reset
    void accumulate( core::Coupler &coupler , real dt );
  };

}



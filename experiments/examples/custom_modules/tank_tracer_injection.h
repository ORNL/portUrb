
#pragma once

#include "coupler.h"

namespace custom_modules {
  
  void tank_tracer_injection( core::Coupler & coupler     ,
                                     real            dt          ,
                                     real            x1          ,
                                     real            x2          ,
                                     real            y1          ,
                                     real            y2          ,
                                     real            z1          ,
                                     real            z2          ,
                                     real            conc        ,
                                     real            wvel        ,
                                     std::string     tracer_name );
}



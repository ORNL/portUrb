
#pragma once

#include "coupler.h"
#include "MultipleFields.h"

namespace modules {


  void overwrite_interpolate( core::Coupler                  & coupler  ,
                                     std::string              const & fname    ,
                                     std::vector<std::string> const & varnames );


}



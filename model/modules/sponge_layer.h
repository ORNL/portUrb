
#pragma once

#include "coupler.h"

namespace modules {

  // Applies a sponge layer to the top of the model domain to force variables toward their horizontal averages
  //   and force vertical velocity to zero
  // coupler     : core::Coupler object containing the model state
  // dt          : Timestep size in seconds
  // time_scale  : Time scale for sponge layer damping in seconds
  // top_prop    : Proportion of the domain height to apply the sponge layer over
  void sponge_layer( core::Coupler &coupler , real dt , real time_scale , real top_prop );


  // Applies a sponge layer to the top of the model domain to force variables toward their horizontal averages
  //   and force vertical velocity to zero
  // coupler     : core::Coupler object containing the model state
  // dt          : Timestep size in seconds
  // time_scale  : Time scale for sponge layer damping in seconds
  // top_prop    : Proportion of the domain height to apply the sponge layer over
  void sponge_layer_w( core::Coupler &coupler , real dt , real time_scale , real top_prop );

}


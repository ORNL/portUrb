
#pragma once

#include "coupler.h"

namespace modules {
  
  // Sponges in precursor data into a forced simulation near horizontal boundaries
  // coupler_main : Coupler object for the main simulation to be sponged
  // coupler_prec : Coupler object for the precursor simulation to provide data
  // vnames      : Vector of variable names to sponge
  // cells_x1    : Number of cells to sponge in from the left x-boundary
  // cells_x2    : Number of cells to sponge in from the right x-boundary
  // cells_y1    : Number of cells to sponge in from the bottom y-boundary
  // cells_y2    : Number of cells to sponge in from the top y-boundary
  // A cosine weighting is used over the sponge regions, with the first third of each region
  //  being fully sponged and the remaining two-thirds transitioning to no sponge at all.
  void precursor_sponge( core::Coupler            & coupler_main ,
                                core::Coupler      const & coupler_prec ,
                                std::vector<std::string>   vnames       ,
                                int                        cells_x1 = 0 ,
                                int                        cells_x2 = 0 ,
                                int                        cells_y1 = 0 ,
                                int                        cells_y2 = 0 );
}



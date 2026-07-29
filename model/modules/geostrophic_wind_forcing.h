
#pragma once

#include "coupler.h"

namespace modules {

  // Apply geostrophic forcing to the velocity fields
  // coupler : core::Coupler object containing the data manager and parallel comm info
  // dt      : Timestep size in seconds
  // lat_g   : Geostrophic latitude in degrees
  // u_g     : Geostrophic zonal wind speed in m/s
  // v_g     : Geostrophic meridional wind speed in m/s
  // Returns a real2d array of shape (2,nz) containing the domain-averaged u and v velocities at each vertical
  //   to be later used as specified forcing for simulations forced by turbulent precursor using the
  //   geostrophic_wind_forcing_specified routine.
  // This routine applyies geostrophic forcing to the overall model column rather than each individual cell.
  real2d geostrophic_wind_forcing( core::Coupler &coupler , real dt , real lat_g , real u_g , real v_g );


  // Re-use a specified average column to perform geostrophic forcing that was applied to a precursor simulation
  // coupler : core::Coupler object containing the data manager and parallel comm info
  // dt      : Timestep size in seconds
  // lat_g   : Geostrophic latitude in degrees
  // u_g     : Geostrophic zonal wind speed in m/s
  // v_g     : Geostrophic meridional wind speed in m/s
  // col     : real2d array of shape (2,nz) containing the domain-averaged u and v velocities at each vertical level
  //           to be used as specified forcing
  // This routine applies geostrophic forcing to the overall model column rather than each individual cell.
  void geostrophic_wind_forcing_specified( core::Coupler &coupler , real dt , real lat_g , real u_g , real v_g ,
                                                  real2d const &col );


  // Apply geostrophic forcing to the velocity fields for each individual cell
  // coupler : core::Coupler object containing the data manager and parallel comm info
  // dt      : Timestep size in seconds
  // lat_g   : Geostrophic latitude in degrees
  // u_g     : Geostrophic zonal wind speed in m/s
  // v_g     : Geostrophic meridional wind speed in m/s
  // This routine applies geostrophic forcing to each individual cell rather than the overall model column.
  void geostrophic_wind_forcing_indiv( core::Coupler &coupler , real dt , real lat_g , real u_g , real v_g );

}


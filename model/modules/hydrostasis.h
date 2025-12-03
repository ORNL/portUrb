
#pragma once

#include "main_header.h"
#include "coupler.h"
#include "TransformMatrices.h"

namespace modules {

  // Function type for vertical profiles (input of z-coordinate, output of variable at z)
  typedef std::function<real(real)> FUNC_Z;

  // Integrate hydrostatic pressure using GLL quadrature given temperature and water vapor dry mixing ratio
  // nq: number of quadrature points (default 9)
  // func_T: function providing temperature profile as a function of height
  // func_qv: function providing water vapor dry mixing ratio profile as a function of height
  // zint_d: 1D array of interface heights
  // dz_d:   1D array of layer thicknesses
  // p0:     surface pressure (default 1.e5 Pa)
  // grav:   gravitational acceleration (default 9.81 m/s^2)
  // R_d:    specific gas constant for dry air (default 287 J/kg/K)
  // R_v:    specific gas constant for water vapor (default 461 J/kg/K)
  // The approach is to use two levels of quadrature: One level over each cell, and another in between
  //   each interval within a cell. The finer level integrates the exponential term, while the coarser level
  //   steps aggregates the pressure from one quadrature point to the next.
  template <size_t nq = 9>
  inline realHost2d integrate_hydrostatic_pressure_gll_temp_qv( FUNC_Z         func_T      ,
                                                                FUNC_Z         func_qv     ,
                                                                real1d const & zint_d      ,
                                                                real1d const & dz_d        ,
                                                                real           p0   = 1.e5 ,
                                                                real           grav = 9.81 ,
                                                                real           R_d  = 287  ,
                                                                real           R_v  = 461  ) {
    SArray<real,1,nq> qpoints;  // Gausse-Legendre-Lobatto quadrature points
    SArray<real,1,nq> qweights; // Gausse-Legendre-Lobatto quadrature weights
    TransformMatrices::get_gll_points (qpoints ); // Get GLL points
    TransformMatrices::get_gll_weights(qweights); // Get GLL weights
    auto zint = zint_d.createHostCopy();  // Copy interface heights to host
    auto dz   = dz_d  .createHostCopy();  // Copy layer thicknesses to host
    int  nz   = dz.size();                // Number of vertical levels
    realHost2d pgll("pressure_hydrostatic_gll",nz,nq);  // Pressure at GLL points within each cell
    for (int k1=0; k1 < nz; k1++) {  // Loop over vertical levels
      if (k1 == 0) { pgll(k1,0) = p0;              } // Set surface pressure at first level
      else         { pgll(k1,0) = pgll(k1-1,nq-1); } // Carry over pressure from top of previous level
      for (int k2=1; k2 < nq; k2++) {  // Loop over intervals between GLL points within the cell
        real z1    = (zint(k1)+zint(k1+1))/2 + qpoints(k2-1)*dz(k1); // Lower bound of interval
        real z2    = (zint(k1)+zint(k1+1))/2 + qpoints(k2  )*dz(k1); // Upper bound of interval
        real dzloc = z2-z1;   // Thickness of the interval
        real tot   = 0;       // Accumulator for integral
        for (int k3=0; k3 < nq; k3++) {  // Loop over quadrature points within the interval
          real z  = z1 + dzloc/2 + qpoints(k3)*dzloc;  // Compute height at quadrature point
          real T  = func_T(z);                         // Get temperature at height z
          real qv = func_qv(z);                        // Get water vapor mixing ratio at height z
          tot += (1+qv)*grav / ((R_d+qv*R_v)*T) * qweights(k3); // Accumulate weighted integrand
        }
        pgll(k1,k2) = pgll(k1,k2-1) * std::exp(-tot*dzloc); // Update pressure using exponential of integral
      }
    }
    return pgll;
  }



  // Integrate hydrostatic pressure using GLL quadrature given potential temperature profile
  // nq: number of quadrature points (default 9)
  // func_th: function providing potential temperature profile as a function of height
  // zint_d: 1D array of interface heights
  // dz_d:   1D array of layer thicknesses
  // p0:     surface pressure (default 1.e5 Pa)
  // grav:   gravitational acceleration (default 9.81 m/s^2)
  // R_d:    specific gas constant for dry air (default 287 J/kg/K)
  // c_p:    specific heat capacity at constant pressure for dry air (default 1003 J/kg/K)
  // The approach is to use two levels of quadrature: One level over each cell, and another in between
  //   each interval within a cell. The finer level integrates the 1/theta term, while the coarser level
  //   steps aggregates the pressure from one quadrature point to the next.
  template <size_t nq = 9>
  inline realHost2d integrate_hydrostatic_pressure_gll_theta( FUNC_Z         func_th     ,
                                                              real1d const & zint_d      ,
                                                              real1d const & dz_d        ,
                                                              real           p0   = 1.e5 ,
                                                              real           grav = 9.81 ,
                                                              real           R_d  = 287  ,
                                                              real           c_p  = 1003 ) {
    SArray<real,1,nq> qpoints;  // Gausse-Legendre-Lobatto quadrature points
    SArray<real,1,nq> qweights; // Gausse-Legendre-Lobatto quadrature weights
    TransformMatrices::get_gll_points (qpoints ); // Get GLL points
    TransformMatrices::get_gll_weights(qweights); // Get GLL weights
    auto zint  = zint_d.createHostCopy(); // Copy interface heights to host
    auto dz    = dz_d  .createHostCopy(); // Copy layer thicknesses to host
    int  nz    = dz.size();               // Number of vertical levels
    real c_v   = c_p - R_d;               // Specific heat capacity at constant volume
    real gamma = c_p / c_v;               // ratio of specific heats
    real C0    = std::pow(R_d,c_p/c_v)*std::pow(p0,-R_d/c_v); // Constant in pressure equation
    real cnst  = grav*(1-gamma)/(gamma*std::pow(C0,1/gamma)); // Constant multiplier for integral
    realHost2d pgll("pressure_hydrostatic_gll",nz,nq); // Pressure at GLL points within each cell
    for (int k1=0; k1 < nz; k1++) {  // Loop over vertical levels
      if (k1 == 0) { pgll(k1,0) = std::pow(p0,(gamma-1)/gamma); } // Set transformed surface pressure at first level
      else         { pgll(k1,0) = pgll(k1-1,nq-1);      }         // Carry over transformed pressure from top of previous level
      for (int k2=1; k2 < nq; k2++) {  // Loop over intervals between GLL points within the cell
        real z1    = (zint(k1)+zint(k1+1))/2 + qpoints(k2-1)*dz(k1); // Lower bound of interval
        real z2    = (zint(k1)+zint(k1+1))/2 + qpoints(k2  )*dz(k1); // Upper bound of interval
        real dzloc = z2-z1;  // Thickness of the interval
        real tot   = 0;      // Accumulator for integral
        for (int k3=0; k3 < nq; k3++) {  // Loop over quadrature points within the interval
          real z  = z1 + dzloc/2 + qpoints(k3)*dzloc; // Compute height at quadrature point
          real th = func_th(z);                       // Get potential temperature at height z
          tot += 1/th * qweights(k3);                 // Accumulate weighted integrand
        }
        pgll(k1,k2) = pgll(k1,k2-1) + cnst*tot*dzloc; // Update transformed pressure using integral
      }
    }
    // Final transformation to get actual pressure values
    for (int k1=0; k1 < nz; k1++) {
      for (int k2=0; k2 < nq; k2++) {
        pgll(k1,k2) = std::pow( pgll(k1,k2) , gamma / (gamma-1) );
      }
    }
    return pgll;
  }

}



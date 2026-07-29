
#pragma once

#include "main_header.h"

namespace modules {
namespace profiles {


  // Computes a hydrostatic background density and potential temperature using c constant potential temperature
  // backgrounda for a single vertical location
  KOKKOS_INLINE_FUNCTION void hydro_const_theta( real z, real grav, real C0, real cp, real p0, real gamma, real rd,
                                      real &r, real &t ) {
    const real theta0 = 300.;  //Background potential temperature
    const real exner0 = 1.;    //Surface-level Exner pressure
    t = theta0;                                       //Potential Temperature at z
    real exner = exner0 - grav * z / (cp * theta0);   //Exner pressure at z
    real p = p0 * std::pow(exner,(cp/rd));            //Pressure at z
    real rt = std::pow((p / C0),(1._fp / gamma));     //rho*theta at z
    r = rt / t;                                       //Density at z
  }


  KOKKOS_INLINE_FUNCTION void hydro_const_bvf( real z, real grav, real C0, real cp, real p0, real gamma, real rd,
                                    real &r, real &t ) {
    const real theta0 = 300.;  //Background potential temperature
    const real bvf    = 1.e-2; //Brunt-Vaisaila Frequency
    const real exner0 = 1.;    //Surface-level Exner pressure
    t = theta0 * std::exp(bvf*bvf/grav*z);            //Potential temperature at z
    real exner = exner0 - grav*grav/(cp*bvf*bvf)*(t-theta0)/(t*theta0); //Exner pressure at z
    real p = p0 * std::pow(exner,(cp/rd));            //Pressure at z
    real rt = std::pow((p / C0),(1._fp / gamma));     //rho*theta at z
    r = rt / t;                                       //Density at z
  }


  // Samples a 3-D ellipsoid at a point in space
  KOKKOS_INLINE_FUNCTION real sample_ellipse_cosine(real amp, real x   , real y   , real z   ,
                                                   real x0  , real y0  , real z0  ,
                                                   real xrad, real yrad, real zrad) {
    //Compute distance from bubble center
    real dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) +
                      ((y-y0)/yrad)*((y-y0)/yrad) +
                      ((z-z0)/zrad)*((z-z0)/zrad) ) * M_PI / 2.;
    //If the distance from bubble center is less than the radius, create a cos**2 profile
    if (dist <= M_PI / 2.) {
      return amp * std::pow(cos(dist),2._fp);
    } else {
      return 0.;
    }
  }


  KOKKOS_INLINE_FUNCTION real saturation_vapor_pressure(real temp) {
    real tc = temp - 273.15;
    return 610.94 * std::exp( 17.625*tc / (243.04+tc) );
  }


  // Creates initial data at a point in space for the rising moist thermal test case
  KOKKOS_INLINE_FUNCTION void thermal(real x, real y, real z, real xlen, real ylen, real grav, real C0, real gamma,
                           real cp, real p0, real R_d, real R_v, real &rho, real &u, real &v, real &w,
                           real &theta, real &rho_v, real &hr, real &ht) {
    hydro_const_theta(z,grav,C0,cp,p0,gamma,R_d,hr,ht);
    real rho_d   = hr;
    u            = 0.;
    v            = 0.;
    w            = 0.;
    real theta_d = ht + sample_ellipse_cosine(2._fp  ,  x,y,z  ,  xlen/2,ylen/2,2000.  ,  2000.,2000.,2000.);
    real p_d     = C0 * pow( rho_d*theta_d , gamma );
    real temp    = p_d / rho_d / R_d;
    real sat_pv  = saturation_vapor_pressure(temp);
    real sat_rv  = sat_pv / R_v / temp;
    rho_v        = sample_ellipse_cosine(0.8_fp  ,  x,y,z  ,  xlen/2,ylen/2,2000.  ,  2000.,2000.,2000.) * sat_rv;
    real p       = rho_d * R_d * temp + rho_v * R_v * temp;
    rho          = rho_d + rho_v;
    theta        = std::pow( p / C0 , 1._fp / gamma ) / rho;
  }


  // Compute supercell temperature profile at a vertical location
  KOKKOS_INLINE_FUNCTION real init_supercell_temperature(real z, real z_0, real z_trop, real z_top,
                                              real T_0, real T_trop, real T_top) {
    if (z <= z_trop) {
      real lapse = - (T_trop - T_0) / (z_trop - z_0);
      return T_0 - lapse * (z - z_0);
    } else {
      real lapse = - (T_top - T_trop) / (z_top - z_trop);
      return T_trop - lapse * (z - z_trop);
    }
  }


  // Compute supercell dry pressure profile at a vertical location
  KOKKOS_INLINE_FUNCTION real init_supercell_pressure_dry(real z, real z_0, real z_trop, real z_top,
                                               real T_0, real T_trop, real T_top,
                                               real p_0, real R_d, real grav) {
    if (z <= z_trop) {
      real lapse = - (T_trop - T_0) / (z_trop - z_0);
      real T = init_supercell_temperature(z, z_0, z_trop, z_top, T_0, T_trop, T_top);
      return p_0 * pow( T / T_0 , grav/(R_d*lapse) );
    } else {
      // Get pressure at the tropopause
      real lapse = - (T_trop - T_0) / (z_trop - z_0);
      real p_trop = p_0 * pow( T_trop / T_0 , grav/(R_d*lapse) );
      // Get pressure at requested height
      lapse = - (T_top - T_trop) / (z_top - z_trop);
      if (lapse != 0) {
        real T = init_supercell_temperature(z, z_0, z_trop, z_top, T_0, T_trop, T_top);
        return p_trop * pow( T / T_trop , grav/(R_d*lapse) );
      } else {
        return p_trop * exp(-grav*(z-z_trop)/(R_d*T_trop));
      }
    }
  }

  
  // Compute supercell relative humidity profile at a vertical location
  KOKKOS_INLINE_FUNCTION real init_supercell_relhum(real z, real z_0, real z_trop) {
    if (z <= z_trop) {
      return 1._fp - 0.75_fp * pow(z / z_trop , 1.25_fp );
    } else {
      return 0.25_fp;
    }
  }


  // Computes dry saturation mixing ratio
  KOKKOS_INLINE_FUNCTION real init_supercell_sat_mix_dry( real press , real T ) {
    return 380/(press) * exp( 17.27_fp * (T-273)/(T-36) );
  }



}
}



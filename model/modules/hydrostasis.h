
#pragma once

#include "main_header.h"
#include "coupler.h"
#include "TransformMatrices.h"

namespace modules {

  typedef std::function<real(real)> FUNC_Z;

  template <size_t nq = 9>
  inline realHost2d integrate_hydrostatic_pressure_gll_temp_qv( FUNC_Z         func_T      ,
                                                                FUNC_Z         func_qv     ,
                                                                real1d const & zint_d      ,
                                                                real1d const & dz_d        ,
                                                                real           p0   = 1.e5 ,
                                                                real           grav = 9.81 ,
                                                                real           R_d  = 287  ,
                                                                real           R_v  = 461  ) {
    SArray<real,1,nq> qpoints;
    SArray<real,1,nq> qweights;
    TransformMatrices::get_gll_points (qpoints );
    TransformMatrices::get_gll_weights(qweights);
    auto zint = zint_d.createHostCopy();
    auto dz   = dz_d  .createHostCopy();
    int  nz   = dz.size();
    realHost2d pgll("pressure_hydrostatic_gll",nz,nq);
    for (int k1=0; k1 < nz; k1++) {
      if (k1 == 0) { pgll(k1,0) = p0;              }
      else         { pgll(k1,0) = pgll(k1-1,nq-1); }
      for (int k2=1; k2 < nq; k2++) {
        real z1    = (zint(k1)+zint(k1+1))/2 + qpoints(k2-1)*dz(k1);
        real z2    = (zint(k1)+zint(k1+1))/2 + qpoints(k2  )*dz(k1);
        real dzloc = z2-z1;
        real tot   = 0;
        for (int k3=0; k3 < nq; k3++) {
          real z  = z1 + dzloc/2 + qpoints(k3)*dzloc;
          real T  = func_T(z);
          real qv = func_qv(z);
          tot += (1+qv)*grav / ((R_d+qv*R_v)*T) * qweights(k3);
        }
        pgll(k1,k2) = pgll(k1,k2-1) * std::exp(-tot*dzloc);
      }
    }
    return pgll;
  }



  template <size_t nq = 9>
  inline realHost2d integrate_hydrostatic_pressure_gll_theta( FUNC_Z         func_th     ,
                                                              real1d const & zint_d      ,
                                                              real1d const & dz_d        ,
                                                              real           p0   = 1.e5 ,
                                                              real           grav = 9.81 ,
                                                              real           R_d  = 287  ,
                                                              real           c_p  = 1003 ) {
    SArray<real,1,nq> qpoints;
    SArray<real,1,nq> qweights;
    TransformMatrices::get_gll_points (qpoints );
    TransformMatrices::get_gll_weights(qweights);
    auto zint  = zint_d.createHostCopy();
    auto dz    = dz_d  .createHostCopy();
    int  nz    = dz.size();
    real c_v   = c_p - R_d;
    real gamma = c_p / c_v;
    real C0    = std::pow(R_d,c_p/c_v)*std::pow(p0,-R_d/c_v);
    real cnst  = grav*(1-gamma)/(gamma*std::pow(C0,1/gamma));
    realHost2d pgll("pressure_hydrostatic_gll",nz,nq);
    for (int k1=0; k1 < nz; k1++) {
      if (k1 == 0) { pgll(k1,0) = std::pow(p0,(gamma-1)/gamma); }
      else         { pgll(k1,0) = pgll(k1-1,nq-1);      }
      for (int k2=1; k2 < nq; k2++) {
        real z1    = (zint(k1)+zint(k1+1))/2 + qpoints(k2-1)*dz(k1);
        real z2    = (zint(k1)+zint(k1+1))/2 + qpoints(k2  )*dz(k1);
        real dzloc = z2-z1;
        real tot   = 0;
        for (int k3=0; k3 < nq; k3++) {
          real z  = z1 + dzloc/2 + qpoints(k3)*dzloc;
          real th = func_th(z);
          tot += 1/th * qweights(k3);
        }
        pgll(k1,k2) = pgll(k1,k2-1) + cnst*tot*dzloc;
      }
    }
    for (int k1=0; k1 < nz; k1++) {
      for (int k2=0; k2 < nq; k2++) {
        pgll(k1,k2) = std::pow( pgll(k1,k2) , gamma / (gamma-1) );
      }
    }
    return pgll;
  }

}



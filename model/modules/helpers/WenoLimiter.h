
#pragma once

#include "main_header.h"
#include "TransformMatrices.h"


template <class real, int ord> struct WenoLimiter {
  int static constexpr NL = (ord+1)/2;

  struct Params {
    yakl::SArray<real,NL,2,NL>  s2g  ;
    yakl::SArray<real,NL>       idl_L;
    yakl::SArray<real,NL,NL,NL> ATV  ;
  };

  Params params;

  static KOKKOS_INLINE_FUNCTION void compute_limited_edges( yakl::SArray<real,ord> const & s      ,
                                                            real                         & pL     ,
                                                            real                         & pR     ,
                                                            bool                           imm_L  ,
                                                            bool                           imm_R  ,
                                                            Params                 const & params ) {
    auto &s2g   = params.s2g;
    auto &idl_L = params.idl_L;
    auto &ATV   = params.ATV;
    yakl::SArray<real,NL> TV;
    yakl::SArray<real,NL> wt;
    // Left point
    real wt_sum = 0;
    for (int ip=0; ip < NL; ip++) {
      TV(ip) = 0;
      for (int jj=0; jj < NL; jj++) {
        for (int ii=0; ii < NL; ii++) { TV(ip) += ATV(ip,jj,ii)*s(ip+ii)*s(ip+jj); }
      }
      wt(ip) = idl_L(ip) / (TV(ip)*TV(ip)+1.e-20);
      wt_sum += wt(ip);
    }
    // if (imm_L) {
    //   real mx = TV(0);
    //   for (int i=1; i < TV.size(); i++) { mx = std::max(mx,TV(i)); }
    //   TV_sum -= TV(NL-1);
    //   TV(NL-1) = mx;
    //   TV_sum += TV(NL-1);
    // }
    // if (imm_R) {
    //   real mx = TV(0);
    //   for (int i=1; i < TV.size(); i++) { mx = std::max(mx,TV(i)); }
    //   TV_sum -= TV(0);
    //   TV(0) = mx;
    //   TV_sum += TV(0);
    // }
    for (int ip=0; ip < NL; ip++) { wt(ip) /= std::max(wt_sum,static_cast<real>(1.e-20)); }
    pL = 0;
    for (int ip=0; ip < NL; ip++) {
      for (int ii=0; ii < NL; ii++) { pL += wt(ip)*s2g(ip,0,ii)*s(ip+ii); }
    }
    // Right point
    wt_sum = 0;
    for (int ip=0; ip < NL; ip++) {
      wt(ip) = idl_L(NL-1-ip) / (TV(ip)*TV(ip)+1.e-20);
      wt_sum += wt(ip);
    }
    for (int ip=0; ip < NL; ip++) { wt(ip) /= std::max(wt_sum,static_cast<real>(1.e-20)); }
    pR = 0;
    for (int ip=0; ip < NL; ip++) {
      for (int ii=0; ii < NL; ii++) { pR += wt(ip)*s2g(ip,1,ii)*s(ip+ii); }
    }
  }

};



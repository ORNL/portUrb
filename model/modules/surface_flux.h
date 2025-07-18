
#pragma once

#include "coupler.h"

namespace modules {

  // Currently ignoring stability / universal functions
  inline void apply_surface_fluxes( core::Coupler &coupler , real dt ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    using yakl::c::Bounds;
    auto nx        = coupler.get_nx  ();
    auto ny        = coupler.get_ny  ();
    auto nz        = coupler.get_nz  ();
    auto dx        = coupler.get_dx  ();
    auto dy        = coupler.get_dy  ();
    auto dz        = coupler.get_dz  ();
    auto p0        = coupler.get_option<real>("p0");
    auto R_d       = coupler.get_option<real>("R_d");
    auto cp_d      = coupler.get_option<real>("cp_d");
    auto &dm       = coupler.get_data_manager_readwrite();
    auto dm_r      = dm.get<real const,3>("density_dry");
    auto dm_u      = dm.get<real      ,3>("uvel");
    auto dm_v      = dm.get<real      ,3>("vvel");
    auto dm_w      = dm.get<real      ,3>("wvel");
    auto dm_T      = dm.get<real      ,3>("temp");
    auto imm_prop  = dm.get<real const,3>("immersed_proportion_halos");
    auto imm_rough = dm.get<real const,3>("immersed_roughness_halos" );
    auto imm_temp  = dm.get<real const,3>("immersed_temp_halos"      );
    int  hs        = 1;

    real3d tend_u("tend_u",nz,ny,nx);
    real3d tend_v("tend_v",nz,ny,nx);
    real3d tend_w("tend_w",nz,ny,nx);
    real3d tend_T("tend_T",nz,ny,nx);

    real vk = 0.40;   // von karman constant

    parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      real r = dm_r(k,j,i);
      real u = dm_u(k,j,i);
      real v = dm_v(k,j,i);
      real w = dm_w(k,j,i);
      real T = dm_T(k,j,i);
      tend_u(k,j,i) = 0;
      tend_v(k,j,i) = 0;
      tend_w(k,j,i) = 0;
      tend_T(k,j,i) = 0;
      int indk, indj, indi;
      indk = hs+k;  indj = hs+j;  indi = hs+i-1;
      if (imm_prop(indk,indj,indi) > 0) {
        real z0   = imm_rough(indk,indj,indi);
        real lgx  = std::log((dx/2+z0)/z0);
        real c_dx = vk*vk/(lgx*lgx);
        real T0   = imm_temp(indk,indj,indi);
        real mag  = std::sqrt(v*v+w*w);
        tend_v(k,j,i) += -c_dx*(v-0 )*mag/dx;
        tend_w(k,j,i) += -c_dx*(w-0 )*mag/dx;
        if (T0 != 0) {
          real z0T  = 0.1*z0;
          real lgxT = std::log((dx/2+z0T)/z0T);
          real c_dxT = vk*vk/(lgx*lgxT);
          tend_T(k,j,i) += -c_dxT*(T-T0)*mag/dx;
        }
      }
      indk = hs+k;  indj = hs+j;  indi = hs+i+1;
      if (imm_prop(indk,indj,indi) > 0) {
        real z0   = imm_rough(indk,indj,indi);
        real lgx  = std::log((dx/2+z0)/z0);
        real c_dx = vk*vk/(lgx*lgx);
        real T0   = imm_temp(indk,indj,indi);
        real mag  = std::sqrt(v*v+w*w);
        tend_v(k,j,i) += -c_dx*(v-0 )*mag/dx;
        tend_w(k,j,i) += -c_dx*(w-0 )*mag/dx;
        if (T0 != 0) {
          real z0T  = 0.1*z0;
          real lgxT = std::log((dx/2+z0T)/z0T);
          real c_dxT = vk*vk/(lgx*lgxT);
          tend_T(k,j,i) += -c_dxT*(T-T0)*mag/dx;
        }
      }
      indk = hs+k;  indj = hs+j-1;  indi = hs+i;
      if (imm_prop(indk,indj,indi) > 0) {
        real z0   = imm_rough(indk,indj,indi);
        real lgy  = std::log((dy/2+z0)/z0);
        real c_dy = vk*vk/(lgy*lgy);
        real T0   = imm_temp(indk,indj,indi);
        real mag  = std::sqrt(u*u+w*w);
        tend_u(k,j,i) += -c_dy*(u-0 )*mag/dy;
        tend_w(k,j,i) += -c_dy*(w-0 )*mag/dy;
        if (T0 != 0) {
          real z0T  = 0.1*z0;
          real lgyT = std::log((dy/2+z0T)/z0T);
          real c_dyT = vk*vk/(lgy*lgyT);
          tend_T(k,j,i) += -c_dyT*(T-T0)*mag/dy;
        }
      }
      indk = hs+k;  indj = hs+j+1;  indi = hs+i;
      if (imm_prop(indk,indj,indi) > 0) {
        real z0   = imm_rough(indk,indj,indi);
        real lgy  = std::log((dy/2+z0)/z0);
        real c_dy = vk*vk/(lgy*lgy);
        real T0   = imm_temp(indk,indj,indi);
        real mag  = std::sqrt(u*u+w*w);
        tend_u(k,j,i) += -c_dy*(u-0 )*mag/dy;
        tend_w(k,j,i) += -c_dy*(w-0 )*mag/dy;
        if (T0 != 0) {
          real z0T  = 0.1*z0;
          real lgyT = std::log((dy/2+z0T)/z0T);
          real c_dyT = vk*vk/(lgy*lgyT);
          tend_T(k,j,i) += -c_dyT*(T-T0)*mag/dy;
        }
      }
      indk = hs+k-1;  indj = hs+j;  indi = hs+i;
      if (imm_prop(indk,indj,indi) > 0) {
        real z0   = imm_rough(indk,indj,indi);
        real lgz  = std::log((dz/2+z0)/z0);
        real c_dz = vk*vk/(lgz*lgz);
        real T0   = imm_temp(indk,indj,indi);
        real mag  = std::sqrt(u*u+v*v);
        tend_u(k,j,i) += -c_dz*(u-0 )*mag/dz;
        tend_v(k,j,i) += -c_dz*(v-0 )*mag/dz;
        if (T0 != 0) {
          real z0T  = 0.1*z0;
          real lgzT = std::log((dz/2+z0T)/z0T);
          real c_dzT = vk*vk/(lgz*lgzT);
          tend_T(k,j,i) += -c_dzT*(T-T0)*mag/dz;
        }
      }
      indk = hs+k+1;  indj = hs+j;  indi = hs+i;
      if (imm_prop(indk,indj,indi) > 0) {
        real z0   = imm_rough(indk,indj,indi);
        real lgz  = std::log((dz/2+z0)/z0);
        real c_dz = vk*vk/(lgz*lgz);
        real T0   = imm_temp(indk,indj,indi);
        real mag  = std::sqrt(u*u+v*v);
        tend_u(k,j,i) += -c_dz*(u-0 )*mag/dz;
        tend_v(k,j,i) += -c_dz*(v-0 )*mag/dz;
        if (T0 != 0) {
          real z0T  = 0.1*z0;
          real lgzT = std::log((dz/2+z0T)/z0T);
          real c_dzT = vk*vk/(lgz*lgzT);
          tend_T(k,j,i) += -c_dzT*(T-T0)*mag/dz;
        }
      }
    });

    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      dm_u(k,j,i) += dt*tend_u(k,j,i);
      dm_v(k,j,i) += dt*tend_v(k,j,i);
      dm_w(k,j,i) += dt*tend_w(k,j,i);
      dm_T(k,j,i) += dt*tend_T(k,j,i);
    });

  };

}


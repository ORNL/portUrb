
#pragma once

#include "main_header.h"
#include "profiles.h"
#include "coupler.h"
#include "TransformMatrices.h"
#include "hydrostasis.h"

namespace custom_modules {

  inline void sc_perturb( core::Coupler & coupler ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nx       = coupler.get_nx();
    auto ny       = coupler.get_ny();
    auto nz       = coupler.get_nz();
    auto dx       = coupler.get_dx();
    auto dy       = coupler.get_dy();
    auto dz       = coupler.get_dz();
    auto zint     = coupler.get_zint();
    auto zmid     = coupler.get_zmid();
    auto xlen     = coupler.get_xlen();
    auto ylen     = coupler.get_ylen();
    auto i_beg    = coupler.get_i_beg();
    auto j_beg    = coupler.get_j_beg();
    auto nx_glob  = coupler.get_nx_glob();
    auto ny_glob  = coupler.get_ny_glob();
    auto sim2d    = coupler.is_sim2d();
    auto R_d      = coupler.get_option<real>("R_d" );
    auto cp_d     = coupler.get_option<real>("cp_d");
    auto R_v      = coupler.get_option<real>("R_v" );
    auto cp_v     = coupler.get_option<real>("cp_v");
    auto p0       = coupler.get_option<real>("p0"  );
    auto grav     = coupler.get_option<real>("grav");
    auto cv_d     = coupler.get_option<real>("cv_d");
    auto gamma    = coupler.get_option<real>("gamma_d");
    auto kappa    = coupler.get_option<real>("kappa_d");
    auto C0       = coupler.get_option<real>("C0");
    auto &dm      = coupler.get_data_manager_readwrite();
    auto dm_rho_d = dm.get<real,3>("density_dry");
    auto dm_uvel  = dm.get<real,3>("uvel"       );
    auto dm_vvel  = dm.get<real,3>("vvel"       );
    auto dm_wvel  = dm.get<real,3>("wvel"       );
    auto dm_temp  = dm.get<real,3>("temp"       );
    auto dm_rho_v = dm.get<real,3>("water_vapor");
    auto dm_imm   = dm.get<real,3>("immersed_proportion");

    const int nqpoints = 9;
    SArray<real,1,nqpoints> qpoints;
    SArray<real,1,nqpoints> qweights;
    TransformMatrices::get_gll_points (qpoints );
    TransformMatrices::get_gll_weights(qweights);

    auto enable_gravity = coupler.get_option<bool>("enable_gravity",true);

    if        (coupler.get_option<std::string>("init_data") == "city") {

    } else if (coupler.get_option<std::string>("init_data") == "city_stretched") {

    } else if (coupler.get_option<std::string>("init_data") == "building") {

    } else if (coupler.get_option<std::string>("init_data") == "buildings_periodic") {

    } else if (coupler.get_option<std::string>("init_data") == "cubes_periodic") {

      real u0 = 10;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        if (dm_imm(k,j,i) == 0) {
          yakl::Random rng(3*(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + i_beg+i));
          dm_uvel(k,j,i) += rng.genFP<real>(-0.1,0.1)*u0;
          dm_vvel(k,j,i) += rng.genFP<real>(-0.1,0.1)*u0;
          dm_wvel(k,j,i) += rng.genFP<real>(-0.1,0.1)*u0;
        }
      });

    } else if (coupler.get_option<std::string>("init_data") == "constant") {

    } else if (coupler.get_option<std::string>("init_data") == "channel") {

      real u0 = coupler.get_option<real>( "constant_uvel" , 1. );
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        yakl::Random rng(3*(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + i_beg+i));
        dm_uvel(k,j,i) += rng.genFP<real>(-0.2,0.2)*u0;
        dm_vvel(k,j,i) += rng.genFP<real>(-0.2,0.2)*u0;
        dm_wvel(k,j,i) += rng.genFP<real>(-0.2,0.2)*u0;
      });

    } else if (coupler.get_option<std::string>("init_data") == "nrel_5mw_convective") {

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        yakl::Random rand(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + (i_beg+i));
        real z    = zmid(k);
        real ztop = 50;
        if (z <= ztop)  dm_temp(k,j,i) += rand.genFP<real>(-1.4,1.4);
      });

    } else if (coupler.get_option<std::string>("init_data") == "shallow_convection") {

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        yakl::Random rand(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + (i_beg+i));
        real z = zmid(k);
        if (z <= 1600) dm_temp (k,j,i) += rand.genFP<real>(-0.1,0.1);
        if (z <= 1600) dm_rho_v(k,j,i) += rand.genFP<real>(-2.5e-5,2.5e-5)*dm_rho_d(k,j,i);
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_neutral") {

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        yakl::Random rand(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + (i_beg+i));
        real z = zmid(k);
        if (z <= 400) dm_temp(k,j,i) += rand.genFP<real>(-0.25,0.25);
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_convective") {

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        yakl::Random rand(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + (i_beg+i));
        real z = zmid(k);
        if (z <= 400) dm_temp(k,j,i) += rand.genFP<real>(-0.25,0.25);
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_convective2") {

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        yakl::Random rand(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + (i_beg+i));
        real z = zmid(k);
        if (z <= 400) dm_temp(k,j,i) += rand.genFP<real>(-0.25,0.25);
      });


    } else if (coupler.get_option<std::string>("init_data") == "ABL_stable") {

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        yakl::Random rand(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + (i_beg+i));
        real z = zmid(k);
        if (z <= 50) dm_temp(k,j,i) += rand.genFP<real>(-0.10,0.10);
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_stable_bvf") {

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        yakl::Random rand(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + (i_beg+i));
        real z = zmid(k);
        if (z <= 80) dm_temp(k,j,i) += rand.genFP<real>(-0.5,0.5);
        if (z <= 80) dm_uvel(k,j,i) += rand.genFP<real>(-0.5,0.5);
        if (z <= 80) dm_vvel(k,j,i) += rand.genFP<real>(-0.5,0.5);
        if (z <= 80) dm_wvel(k,j,i) += rand.genFP<real>(-0.1,0.1);
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_neutral2") {

      auto wind = coupler.get_option<real>("hub_height_wind_mag");
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        yakl::Random rand(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + (i_beg+i));
        real z    = zmid(k);
        real ztop = 100;
        if (z <= ztop)  dm_temp(k,j,i) += rand.genFP<real>(-0.25,0.25);
        if (z <= ztop)  dm_uvel(k,j,i) *= (1+rand.genFP<real>(-0.03,0.03));
        if (z <= ztop)  dm_vvel(k,j,i) *= (1+rand.genFP<real>(-0.03,0.03));
      });

    } else if (coupler.get_option<std::string>("init_data") == "AWAKEN_neutral") {

    } else if (coupler.get_option<std::string>("init_data") == "supercell") {

      real x0    = xlen / 2;
      real y0    = ylen / 2;
      real z0    = 1500;
      real radx  = 10000;
      real rady  = 10000;
      real radz  = 1500;
      real amp   = 3;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real Tpert = 0;
        for (int kk=0; kk<nqpoints; kk++) {
          for (int jj=0; jj<nqpoints; jj++) {
            for (int ii=0; ii<nqpoints; ii++) {
              real x    = (i_beg+i+0.5)*dx + qpoints(ii)*dx;
              real y    = (j_beg+j+0.5)*dy + qpoints(jj)*dy;
              real z    = zmid(k)          + qpoints(kk)*dz(k);
              real xn   = (x-x0)/radx;
              real yn   = (y-y0)/rady;
              real zn   = (z-z0)/radz;
              real rad  = sqrt( xn*xn + yn*yn + zn*zn );
              Tpert    += (rad <= 1 ? amp*pow(cos(M_PI*rad/2),2._fp) : 0)*qweights(ii)*qweights(jj)*qweights(kk);
            }
          }
        }
        dm_temp(k,j,i) += Tpert;
      });

    } // if (init_data == ...)

  }

}



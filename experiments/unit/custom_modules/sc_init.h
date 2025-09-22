
#pragma once

#include "main_header.h"
#include "profiles.h"
#include "coupler.h"
#include "TransformMatrices.h"
#include "hydrostasis.h"
#include "TriMesh.h"
#include <random>

namespace custom_modules {

  inline void sc_init( core::Coupler & coupler ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    // Grid and variable parameters
    auto nx        = coupler.get_nx();
    auto ny        = coupler.get_ny();
    auto nz        = coupler.get_nz();
    auto dx        = coupler.get_dx();
    auto dy        = coupler.get_dy();
    auto dz        = coupler.get_dz();
    auto zint      = coupler.get_zint();
    auto zmid      = coupler.get_zmid();
    auto xlen      = coupler.get_xlen();
    auto ylen      = coupler.get_ylen();
    auto zlen      = coupler.get_zlen();
    auto i_beg     = coupler.get_i_beg();
    auto j_beg     = coupler.get_j_beg();
    auto nx_glob   = coupler.get_nx_glob();
    auto ny_glob   = coupler.get_ny_glob();
    auto roughness = coupler.get_option<real>("roughness",0.1);
    auto idWV      = coupler.get_option<int >("idWV"     ,-1 );
    if (idWV == -1) {
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < tracer_names.size(); tr++) { if (tracer_names.at(tr) == "water_vapor") idWV = tr; }
      coupler.set_option<int>("idWV",idWV);
    }
    // Physics parameters
    real R_d     = 287.     ;
    real cp_d    = 1003.    ;
    real R_v     = 461.     ;
    real cp_v    = 1859     ;
    real p0      = 1.e5     ;
    real grav    = 9.81     ;
    real cv_d    = cp_d-R_d ;
    real gamma_d = cp_d/cv_d;
    real kappa_d = R_d/cp_d ;
    real cv_v    = cp_v-R_v ;
    real C0      = pow(R_d*pow(p0,-kappa_d),gamma_d);
    if (! coupler.option_exists("R_d"    )) coupler.set_option<real>("R_d"    ,R_d    );
    if (! coupler.option_exists("cp_d"   )) coupler.set_option<real>("cp_d"   ,cp_d   );
    if (! coupler.option_exists("R_v"    )) coupler.set_option<real>("R_v"    ,R_v    );
    if (! coupler.option_exists("cp_v"   )) coupler.set_option<real>("cp_v"   ,cp_v   );
    if (! coupler.option_exists("p0"     )) coupler.set_option<real>("p0"     ,p0     );
    if (! coupler.option_exists("grav"   )) coupler.set_option<real>("grav"   ,grav   );
    if (! coupler.option_exists("cv_d"   )) coupler.set_option<real>("cv_d"   ,cv_d   );
    if (! coupler.option_exists("gamma_d")) coupler.set_option<real>("gamma_d",gamma_d);
    if (! coupler.option_exists("kappa_d")) coupler.set_option<real>("kappa_d",kappa_d);
    if (! coupler.option_exists("cv_v"   )) coupler.set_option<real>("cv_v"   ,cv_v   );
    if (! coupler.option_exists("C0"     )) coupler.set_option<real>("C0"     ,C0     );
    // Variables
    auto &dm = coupler.get_data_manager_readwrite();
    auto dims3d = {nz,ny,nx};
    auto dims2d = {   ny,nx};
    if (! dm.entry_exists("density_dry"        )) dm.register_and_allocate<real>("density_dry"        ,"",dims3d);
    if (! dm.entry_exists("uvel"               )) dm.register_and_allocate<real>("uvel"               ,"",dims3d);
    if (! dm.entry_exists("vvel"               )) dm.register_and_allocate<real>("vvel"               ,"",dims3d);
    if (! dm.entry_exists("wvel"               )) dm.register_and_allocate<real>("wvel"               ,"",dims3d);
    if (! dm.entry_exists("temp"               )) dm.register_and_allocate<real>("temp"               ,"",dims3d);
    if (! dm.entry_exists("water_vapor"        )) dm.register_and_allocate<real>("water_vapor"        ,"",dims3d);
    if (! dm.entry_exists("immersed_proportion")) dm.register_and_allocate<real>("immersed_proportion","",dims3d);
    if (! dm.entry_exists("immersed_roughness" )) dm.register_and_allocate<real>("immersed_roughness" ,"",dims3d);
    if (! dm.entry_exists("immersed_temp"      )) dm.register_and_allocate<real>("immersed_temp"      ,"",dims3d);
    if (! dm.entry_exists("surface_roughness"  )) dm.register_and_allocate<real>("surface_roughness"  ,"",dims2d);
    if (! dm.entry_exists("surface_temp"       )) dm.register_and_allocate<real>("surface_temp"       ,"",dims2d);
    auto dm_rho_d          = dm.get<real,3>("density_dry"        );
    auto dm_uvel           = dm.get<real,3>("uvel"               );
    auto dm_vvel           = dm.get<real,3>("vvel"               );
    auto dm_wvel           = dm.get<real,3>("wvel"               );
    auto dm_temp           = dm.get<real,3>("temp"               );
    auto dm_rho_v          = dm.get<real,3>("water_vapor"        );
    auto dm_immersed_prop  = dm.get<real,3>("immersed_proportion");
    auto dm_immersed_rough = dm.get<real,3>("immersed_roughness" );
    auto dm_immersed_temp  = dm.get<real,3>("immersed_temp"      );
    auto dm_surface_rough  = dm.get<real,2>("surface_roughness"  );
    auto dm_surface_temp   = dm.get<real,2>("surface_temp"       );
    dm_immersed_prop  = 0;
    dm_immersed_rough = roughness;
    dm_immersed_temp  = 0;
    dm_surface_rough  = roughness;
    dm_surface_temp   = 0;
    dm_rho_v          = 0;
    // Quadrature parameters
    const int nqpoints = 9;
    SArray<real,1,nqpoints> qpoints;
    SArray<real,1,nqpoints> qweights;
    TransformMatrices::get_gll_points (qpoints );
    TransformMatrices::get_gll_weights(qweights);

    coupler.add_option<std::string>("bc_x1","periodic");
    coupler.add_option<std::string>("bc_x2","periodic");
    coupler.add_option<std::string>("bc_y1","periodic");
    coupler.add_option<std::string>("bc_y2","periodic");
    coupler.add_option<std::string>("bc_z1","wall_free_slip");
    coupler.add_option<std::string>("bc_z2","wall_free_slip");
    auto enable_gravity = coupler.get_option<bool>("enable_gravity",true);


    if (coupler.get_option<std::string>("init_data") == "city") {

      dm_immersed_rough = coupler.get_option<real>("building_roughness");
      real uref = 20;
      real href = 500;
      auto faces = coupler.get_data_manager_readwrite().get<float,3>("mesh_faces");
      auto compute_theta = KOKKOS_LAMBDA (real z) -> real { return 300; };
      auto pressGLL = modules::integrate_hydrostatic_pressure_gll_theta(compute_theta,zint,dz,p0,grav,R_d,cp_d).createDeviceCopy();
      auto t1 = std::chrono::high_resolution_clock::now();
      if (coupler.is_mainproc()) std::cout << "*** Beginning setup ***" << std::endl;
      float4d zmesh("zmesh",ny,nx,nqpoints,nqpoints);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(ny,nx,nqpoints,nqpoints) ,
                                        KOKKOS_LAMBDA (int j, int i, int jj, int ii) {
        real x           = (i_beg+i+0.5)*dx + qpoints(ii)*dx;
        real y           = (j_beg+j+0.5)*dy + qpoints(jj)*dy;
        zmesh(j,i,jj,ii) = modules::TriMesh::max_height(x,y,faces,0);
        if (zmesh(j,i,jj,ii) == 0) zmesh(j,i,jj,ii) = -1;
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        dm_rho_d        (k,j,i) = 0;
        dm_uvel         (k,j,i) = 0;
        dm_vvel         (k,j,i) = 0;
        dm_wvel         (k,j,i) = 0;
        dm_temp         (k,j,i) = 0;
        dm_rho_v        (k,j,i) = 0;
        dm_immersed_prop(k,j,i) = 0;
        for (int kk=0; kk<nqpoints; kk++) {
          for (int jj=0; jj<nqpoints; jj++) {
            for (int ii=0; ii<nqpoints; ii++) {
              real x         = (i_beg+i+0.5)*dx + qpoints(ii)*dx;
              real y         = (j_beg+j+0.5)*dy + qpoints(jj)*dy;
              real z         = zmid(k)          + qpoints(kk)*dz(k);
              real theta     = compute_theta(z);
              real p         = pressGLL(k,kk);
              real rho_theta = std::pow( p/C0 , 1._fp/gamma_d );
              real rho       = rho_theta / theta;
              real umag      = uref*std::log((z+roughness)/roughness)/std::log((href+roughness)/roughness);
              real ang       = 29./180.*M_PI;
              real u         = umag*std::cos(ang);
              real v         = umag*std::sin(ang);
              real w         = 0;
              real T         = p/(rho*R_d);
              real rho_v     = 0;
              real wt = qweights(kk)*qweights(jj)*qweights(ii);
              dm_immersed_prop(k,j,i) += (z<=zmesh(j,i,jj,ii) ? 1 : 0) * wt;
              dm_rho_d        (k,j,i) += rho                           * wt;
              dm_uvel         (k,j,i) += (z<=zmesh(j,i,jj,ii) ? 0 : u) * wt;
              dm_vvel         (k,j,i) += (z<=zmesh(j,i,jj,ii) ? 0 : v) * wt;
              dm_wvel         (k,j,i) += (z<=zmesh(j,i,jj,ii) ? 0 : w) * wt;
              dm_temp         (k,j,i) += T                             * wt;
              dm_rho_v        (k,j,i) += rho_v                         * wt;
            }
          }
        }
        // if (k == 0) dm_surface_temp(j,i) = dm_temp(k,j,i);
      });
      std::chrono::duration<double> dur = std::chrono::high_resolution_clock::now() - t1;
      if (coupler.is_mainproc()) std::cout << "*** Finished setup in [" << dur.count() << "] seconds ***" << std::endl;

    } else if (coupler.get_option<std::string>("init_data") == "ABL_convective") {

      auto compute_theta = KOKKOS_LAMBDA (real z) -> real {
        if   (z <  600) { return 309;               }
        else            { return 309+0.004*(z-600); }
      };
      auto pressGLL = modules::integrate_hydrostatic_pressure_gll_theta(compute_theta,zint,dz,p0,grav,R_d,cp_d).createDeviceCopy();
      auto u_g = coupler.get_option<real>("geostrophic_u",10.);
      auto v_g = coupler.get_option<real>("geostrophic_v",0.);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        dm_rho_d(k,j,i) = 0;
        dm_uvel (k,j,i) = 0;
        dm_vvel (k,j,i) = 0;
        dm_wvel (k,j,i) = 0;
        dm_temp (k,j,i) = 0;
        dm_rho_v(k,j,i) = 0;
        for (int kk=0; kk<nqpoints; kk++) {
          real z         = zmid(k) + qpoints(kk)*dz(k);
          real theta     = compute_theta(z);
          real p         = pressGLL(k,kk);
          real rho_theta = std::pow( p/C0 , 1._fp/gamma_d );
          real rho       = rho_theta / theta;
          real u         = u_g;
          real v         = v_g;
          real w         = 0;
          real T         = p/(rho*R_d);
          real rho_v     = 0;
          real wt = qweights(kk);
          dm_rho_d(k,j,i) += rho   * wt;
          dm_uvel (k,j,i) += u     * wt;
          dm_vvel (k,j,i) += v     * wt;
          dm_wvel (k,j,i) += w     * wt;
          dm_temp (k,j,i) += T     * wt;
          dm_rho_v(k,j,i) += rho_v * wt;
        }
        // if (k == 0) dm_surface_temp(j,i) = dm_temp(k,j,i);
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_stable") {

      auto compute_theta = KOKKOS_LAMBDA (real z) -> real {
        if   (z <  100) { return 265;              }
        else            { return 265+0.01*(z-100); }
      };
      auto pressGLL = modules::integrate_hydrostatic_pressure_gll_theta(compute_theta,zint,dz,p0,grav,R_d,cp_d).createDeviceCopy();
      auto u_g = coupler.get_option<real>("geostrophic_u",8.);
      auto v_g = coupler.get_option<real>("geostrophic_v",0.);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        dm_rho_d(k,j,i) = 0;
        dm_uvel (k,j,i) = 0;
        dm_vvel (k,j,i) = 0;
        dm_wvel (k,j,i) = 0;
        dm_temp (k,j,i) = 0;
        dm_rho_v(k,j,i) = 0;
        for (int kk=0; kk<nqpoints; kk++) {
          real z         = zmid(k) + qpoints(kk)*dz(k);
          real theta     = compute_theta(z);
          real p         = pressGLL(k,kk);
          real rho_theta = std::pow( p/C0 , 1._fp/gamma_d );
          real rho       = rho_theta / theta;
          real u         = u_g;
          real v         = v_g;
          real w         = 0;
          real T         = p/(rho*R_d);
          real rho_v     = 0;
          real wt = qweights(kk);
          dm_rho_d(k,j,i) += rho   * wt;
          dm_uvel (k,j,i) += u     * wt;
          dm_vvel (k,j,i) += v     * wt;
          dm_wvel (k,j,i) += w     * wt;
          dm_temp (k,j,i) += T     * wt;
          dm_rho_v(k,j,i) += rho_v * wt;
        }
        if (k == 0) dm_surface_temp(j,i) = dm_temp(k,j,i);
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_neutral2") {

      auto compute_theta = KOKKOS_LAMBDA (real z) -> real {
        if      (z <  500)            { return 300;                        }
        else if (z >= 500 && z < 650) { return 300+0.08*(z-500);           }
        else                          { return 300+0.08*150+0.003*(z-650); }
      };
      real p0       = 1*R_d*300; // Assume a density of one
      real uref     = coupler.get_option<real>("hub_height_wind_mag",12); // Velocity at hub height
      real href     = coupler.get_option<real>("turbine_hub_height",90);  // Height of hub / center of windmills
      auto pressGLL = modules::integrate_hydrostatic_pressure_gll_theta(compute_theta,zint,dz,p0,grav,R_d,cp_d).createDeviceCopy();
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        dm_rho_d(k,j,i) = 0;
        dm_uvel (k,j,i) = 0;
        dm_vvel (k,j,i) = 0;
        dm_wvel (k,j,i) = 0;
        dm_temp (k,j,i) = 0;
        dm_rho_v(k,j,i) = 0;
        for (int kk=0; kk<nqpoints; kk++) {
          real z         = zmid(k) + qpoints(kk)*dz(k);
          real theta     = compute_theta(z);
          real p         = pressGLL(k,kk);
          real rho_theta = std::pow( p/C0 , 1._fp/gamma_d );
          real rho       = rho_theta / theta;
          real ustar     = uref / std::log((href+roughness)/roughness);
          real u         = ustar * std::log((z+roughness)/roughness);
          real v         = 0;
          real w         = 0;
          real T         = p/(rho*R_d);
          real rho_v     = 0;
          real wt = qweights(kk);
          dm_rho_d(k,j,i) += rho   * wt;
          dm_uvel (k,j,i) += u     * wt;
          dm_vvel (k,j,i) += v     * wt;
          dm_wvel (k,j,i) += w     * wt;
          dm_temp (k,j,i) += T     * wt;
          dm_rho_v(k,j,i) += rho_v * wt;
        }
        // if (k == 0) dm_surface_temp(j,i) = dm_temp(k,j,i);
      });

    } else if (coupler.get_option<std::string>("init_data") == "supercell") {

      YAML::Node config = YAML::LoadFile( "./inputs/wrf_supercell_sounding.yaml" );
      if ( !config ) { endrun("ERROR: Invalid turbine input file"); }
      auto sounding = config["sounding"].as<std::vector<std::vector<real>>>();
      int num_entries = sounding.size();
      realHost1d shost_height("s_height",num_entries);
      realHost1d shost_theta ("s_theta" ,num_entries);
      realHost1d shost_qv    ("s_qv"    ,num_entries);
      realHost1d shost_uvel  ("s_uvel"  ,num_entries);
      realHost1d shost_vvel  ("s_vvel"  ,num_entries);
      for (int i=0; i < num_entries; i++) {
        shost_height(i) = sounding[i][0];
        shost_theta (i) = sounding[i][1];
        shost_qv    (i) = sounding[i][2]/1000;
        shost_uvel  (i) = sounding[i][3];
        shost_vvel  (i) = sounding[i][4];
      }
      auto s_height = shost_height.createDeviceCopy();
      auto s_theta  = shost_theta .createDeviceCopy();
      auto s_qv     = shost_qv    .createDeviceCopy();
      auto s_uvel   = shost_uvel  .createDeviceCopy();
      auto s_vvel   = shost_vvel  .createDeviceCopy();
      // Linear interpolation in a reference variable based on u_infinity and reference u_infinity
      auto interp = KOKKOS_LAMBDA ( real1d ref_z , real1d ref_var , real z ) -> real {
        int imax = ref_z.size()-1; // Max index for the table
        if ( z < ref_z(0   ) ) return ref_var(0   );
        if ( z > ref_z(imax) ) return ref_var(imax);
        int i = 0;
        while (z > ref_z(i)) { i++; }
        if (i > 0) i--;
        real fac = (ref_z(i+1) - z) / (ref_z(i+1)-ref_z(i));
        return fac*ref_var(i) + (1-fac)*ref_var(i+1);
      };
      auto interp_host = KOKKOS_LAMBDA ( realHost1d ref_z , realHost1d ref_var , real z ) -> real {
        int imax = ref_z.size()-1; // Max index for the table
        if ( z < ref_z(0   ) ) return ref_var(0   );
        if ( z > ref_z(imax) ) return ref_var(imax);
        int i = 0;
        while (z > ref_z(i)) { i++; }
        if (i > 0) i--;
        real fac = (ref_z(i+1) - z) / (ref_z(i+1)-ref_z(i));
        return fac*ref_var(i) + (1-fac)*ref_var(i+1);
      };
      real T0  = 300;
      real Ttr = 213;
      real ztr = 12000;
      auto c_T = KOKKOS_LAMBDA (real z) -> real {
        if (z <= 12000) { return T0 + z/ztr*(Ttr-T0); }
        else            { return Ttr; }
      };
      auto c_qv = KOKKOS_LAMBDA (real z) -> real { return interp_host( shost_height , shost_qv , z ); };
      using modules::integrate_hydrostatic_pressure_gll_temp_qv;
      auto pressGLL = integrate_hydrostatic_pressure_gll_temp_qv(c_T,c_qv,zint,dz,p0,grav,R_d,R_v).createDeviceCopy();
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        dm_rho_d(k,j,i) = 0;
        dm_uvel (k,j,i) = 0;
        dm_vvel (k,j,i) = 0;
        dm_wvel (k,j,i) = 0;
        dm_temp (k,j,i) = 0;
        dm_rho_v(k,j,i) = 0;
        for (int kk=0; kk<nqpoints; kk++) {
          for (int jj=0; jj<nqpoints; jj++) {
            for (int ii=0; ii<nqpoints; ii++) {
              real x     = (i_beg+i+0.5)*dx + qpoints(ii)*dx;
              real y     = (j_beg+j+0.5)*dy + qpoints(jj)*dy;
              real z     = zmid(k)          + qpoints(kk)*dz(k);
              real T     = c_T(z);
              real qv    = interp( s_height , s_qv    , z );
              real u     = interp( s_height , s_uvel  , z );
              real v     = interp( s_height , s_vvel  , z );
              real p     = pressGLL(k,kk);
              real rho_d = p/((R_d+qv*R_v)*T);
              real rho_v = qv*rho_d;
              real w     = 0;
              real wt = qweights(kk)*qweights(jj)*qweights(ii);
              dm_rho_d(k,j,i) += rho_d * wt;
              dm_uvel (k,j,i) += u     * wt;
              dm_vvel (k,j,i) += v     * wt;
              dm_wvel (k,j,i) += w     * wt;
              dm_temp (k,j,i) += T     * wt;
              dm_rho_v(k,j,i) += rho_v * wt;
            }
          }
        }
        // if (k == 0) dm_surface_temp(j,i) = dm_temp(k,j,i);
      });

    } // if (init_data == ...)

    int hs = 1;
    {
      core::MultiField<real,3> fields;
      fields.add_field( dm_immersed_prop  );
      fields.add_field( dm_immersed_rough );
      fields.add_field( dm_immersed_temp  );
      auto fields_halos = coupler.create_and_exchange_halos( fields , hs );
      std::vector<std::string> dim_names = {"z_halo1","y_halo1","x_halo1"};
      dm.register_and_allocate<real>("immersed_proportion_halos","",{nz+2*hs,ny+2*hs,nx+2*hs},dim_names);
      dm.register_and_allocate<real>("immersed_roughness_halos" ,"",{nz+2*hs,ny+2*hs,nx+2*hs},dim_names);
      dm.register_and_allocate<real>("immersed_temp_halos"      ,"",{nz+2*hs,ny+2*hs,nx+2*hs},dim_names);
      fields_halos.get_field(0).deep_copy_to( dm.get<real,3>("immersed_proportion_halos") );
      fields_halos.get_field(1).deep_copy_to( dm.get<real,3>("immersed_roughness_halos" ) );
      fields_halos.get_field(2).deep_copy_to( dm.get<real,3>("immersed_temp_halos"      ) );
    }
    {
      core::MultiField<real,2> fields;
      fields.add_field( dm_surface_rough );
      fields.add_field( dm_surface_temp  );
      auto fields_halos = coupler.create_and_exchange_halos( fields , hs );
      std::vector<std::string> dim_names = {"y_halo1","x_halo1"};
      dm.register_and_allocate<real>("surface_roughness_halos" ,"",{ny+2*hs,nx+2*hs},dim_names);
      dm.register_and_allocate<real>("surface_temp_halos"      ,"",{ny+2*hs,nx+2*hs},dim_names);
      fields_halos.get_field(0).deep_copy_to( dm.get<real,2>("surface_roughness_halos" ) );
      fields_halos.get_field(1).deep_copy_to( dm.get<real,2>("surface_temp_halos"      ) );
    }

    auto imm_prop  = dm.get<real,3>("immersed_proportion_halos");
    auto imm_rough = dm.get<real,3>("immersed_roughness_halos" );
    auto imm_temp  = dm.get<real,3>("immersed_temp_halos"      );
    auto sfc_rough = dm.get<real,2>("surface_roughness_halos"  );
    auto sfc_temp  = dm.get<real,2>("surface_temp_halos"       );
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) , KOKKOS_LAMBDA (int kk, int j, int i) {
      imm_prop (      kk,j,i) = 1;
      imm_rough(      kk,j,i) = sfc_rough(j,i);
      imm_temp (      kk,j,i) = sfc_temp (j,i);
      imm_prop (hs+nz+kk,j,i) = 0;
      imm_rough(hs+nz+kk,j,i) = 0;
      imm_temp (hs+nz+kk,j,i) = 0;
    });
  }

}



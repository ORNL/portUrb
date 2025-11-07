
#pragma once

#include "main_header.h"
#include "coupler.h"

namespace modules {

  struct TurbineFitch {


    struct RefTurbine {
      std::vector<real> velmag;        // Velocity magnitude at infinity (m/s)
      std::vector<real> thrust_coef;   // Thrust coefficient             (dimensionless)
      std::vector<real> power_coef;    // Power coefficient              (dimensionless)
      std::vector<real> power;         // Power generation               (MW)
      real              hub_height;    // Hub height                     (m)
      real              blade_radius;  // Blade radius                   (m)
      void init( std::string fname ) {
        YAML::Node config = YAML::LoadFile( fname );
        if ( !config ) { endrun("ERROR: Invalid turbine input file"); }
        this->velmag       = config["velocity_magnitude"].as<std::vector<real>>();
        this->thrust_coef  = config["thrust_coef"       ].as<std::vector<real>>();
        this->power_coef   = config["power_coef"        ].as<std::vector<real>>();
        this->power        = config["power_megawatts"   ].as<std::vector<real>>();
        this->hub_height   = config["hub_height"        ].as<real>();
        this->blade_radius = config["blade_radius"      ].as<real>();
      }
    };


    struct Turbine {
      bool               active;       // Whether this turbine affects this MPI task
      real               base_loc_x;   // x location of the tower base
      real               base_loc_y;   // y location of the tower base
      RefTurbine         ref_turbine;  // The reference turbine to use for this turbine
      std::vector<real>  power_trace;  // Time trace of power generation
      std::vector<real>  mag_trace;    // Time trace of inflow wind magnitude normal to turbine plane
      real1d             prop;         // Proportion of the turbine in each vertical level
    };


    struct TurbineGroup {
      std::vector<Turbine> turbines;
      void add_turbine( core::Coupler       & coupler     ,
                        real                  base_loc_x  ,
                        real                  base_loc_y  ,
                        RefTurbine    const & ref_turbine ) {
        auto dx    = coupler.get_dx();
        auto dy    = coupler.get_dy();
        auto nx    = coupler.get_nx();
        auto ny    = coupler.get_ny();
        auto nz    = coupler.get_nz();
        auto zint  = coupler.get_zint().createHostCopy();
        auto i_beg = coupler.get_i_beg();
        auto j_beg = coupler.get_j_beg();
        real dom_x1 = (i_beg+0 )*dx;
        real dom_x2 = (i_beg+nx)*dx;
        real dom_y1 = (j_beg+0 )*dy;
        real dom_y2 = (j_beg+ny)*dy;
        bool active = base_loc_x >= dom_x1 && base_loc_x < dom_x2 && base_loc_y >= dom_y1 && base_loc_y < dom_y2;
        Turbine loc;
        loc.active      = active;
        loc.base_loc_x  = base_loc_x;
        loc.base_loc_y  = base_loc_y;
        loc.ref_turbine = ref_turbine;
        if (active) {
          realHost1d prop("prop",nz);
          int nsamp = 5;
          for (int k=0; k < nz; k++) {
            prop(k) = 0;
            for (int kk=0; kk < nsamp; kk++) {
              real z  = zint(k)+(kk+0.5)*(zint(k+1)-zint(k))/nsamp;
              real z0 = (z-ref_turbine.hub_height)/ref_turbine.blade_radius;
              if (std::abs(z0) < 1) prop(k) += std::sqrt(1-z0*z0);
            }
          }
          using yakl::componentwise::operator/;
          loc.prop = (prop/yakl::intrinsics::sum(prop)).createDeviceCopy();
        }
        turbines.push_back(loc);
      }
    };


    TurbineGroup  turbine_group;
    int           trace_size;


    void init( core::Coupler &coupler ) {
      RefTurbine ref_turbine;
      ref_turbine.init( coupler.get_option<std::string>("turbine_file") );
      if (coupler.option_exists("turbine_x_locs") && coupler.option_exists("turbine_y_locs")) {
        auto x_locs = coupler.get_option<std::vector<real>>("turbine_x_locs");
        auto y_locs = coupler.get_option<std::vector<real>>("turbine_y_locs");
        for (int iturb = 0; iturb < x_locs.size(); iturb++) {
          turbine_group.add_turbine( coupler , x_locs.at(iturb) , y_locs.at(iturb) , ref_turbine );
        }
      }
      trace_size = 0;
      coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        if (trace_size > 0) {
          nc.redef();
          nc.create_dim( "num_time_steps" , trace_size );
          for (int iturb=0; iturb < turbine_group.turbines.size(); iturb++) {
            nc.create_var<real>( std::string("power_trace_turb_")+std::to_string(iturb) , {"num_time_steps"} );
            nc.create_var<real>( std::string("mag_trace_turb_"  )+std::to_string(iturb) , {"num_time_steps"} );
          }
          nc.enddef();
          nc.begin_indep_data();
          for (int iturb=0; iturb < turbine_group.turbines.size(); iturb++) {
            auto &turbine = turbine_group.turbines.at(iturb);
            if (turbine.active) {
              realHost1d power_arr("power_arr",trace_size);
              realHost1d mag_arr  ("mag_arr"  ,trace_size);
              for (int i=0; i < trace_size; i++) { power_arr(i) = turbine.power_trace.at(i); }
              for (int i=0; i < trace_size; i++) { mag_arr  (i) = turbine.mag_trace  .at(i); }
              nc.write( power_arr , std::string("power_trace_turb_")+std::to_string(iturb) );
              nc.write( mag_arr   , std::string("mag_trace_turb_"  )+std::to_string(iturb) );
            }
            coupler.get_parallel_comm().barrier();
            turbine.power_trace.clear();
            turbine.mag_trace  .clear();
          }
          nc.end_indep_data();
        }
        trace_size = 0;
      });
    }


    void apply( core::Coupler & coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx    = coupler.get_nx   ();
      auto ny    = coupler.get_ny   ();
      auto nz    = coupler.get_nz   ();
      auto dx    = coupler.get_dx   ();
      auto dy    = coupler.get_dy   ();
      auto dz    = coupler.get_dz   ();
      auto i_beg = coupler.get_i_beg();
      auto j_beg = coupler.get_j_beg();
      auto &dm   = coupler.get_data_manager_readwrite();
      auto rho_d = dm.get<real const,3>("density_dry"  );
      auto uvel  = dm.get<real      ,3>("uvel"         );
      auto vvel  = dm.get<real      ,3>("vvel"         );
      auto tke   = dm.get<real      ,3>("TKE"          );

      real3d tend_u  ("tend_u"  ,nz,ny,nx);
      real3d tend_v  ("tend_v"  ,nz,ny,nx);
      real3d tend_tke("tend_tke",nz,ny,nx);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        tend_u  (k,j,i) = 0;
        tend_v  (k,j,i) = 0;
        tend_tke(k,j,i) = 0;
      });

      for (int iturb = 0; iturb < turbine_group.turbines.size(); iturb++) {
        auto &turbine = turbine_group.turbines.at(iturb);
        if (turbine.active) {
          // Get reference data for later computations
          real rad             = turbine.ref_turbine.blade_radius;
          real base_x          = turbine.base_loc_x              ;
          real base_y          = turbine.base_loc_y              ;
          auto prop            = turbine.prop                    ;
          auto ref_velmag      = turbine.ref_turbine.velmag      ;
          auto ref_thrust_coef = turbine.ref_turbine.thrust_coef ;
          auto ref_power_coef  = turbine.ref_turbine.power_coef  ;
          auto ref_power       = turbine.ref_turbine.power       ;
          int  i               = std::min(nx-1,std::max(0,(int)std::floor(base_x/dx)-(int)i_beg));
          int  j               = std::min(ny-1,std::max(0,(int)std::floor(base_y/dy)-(int)j_beg));
          yakl::ScalarLiveOut<real> u_d(0);
          yakl::ScalarLiveOut<real> v_d(0);
          parallel_for( YAKL_AUTO_LABEL() , nz , KOKKOS_LAMBDA (int k) {
            if (prop(k) > 0) {
              Kokkos::atomic_add( &(u_d()) , prop(k)*uvel(k,j,i) );
              Kokkos::atomic_add( &(v_d()) , prop(k)*vvel(k,j,i) );
            }
          });
          // Computation of disk properties
          real u0      = u_d.hostRead();
          real v0      = v_d.hostRead();
          real yaw     = std::atan2(v0,u0);
          real cos_yaw = std::cos(yaw);
          real sin_yaw = std::sin(yaw);
          real mag0    = sqrt(u0*u0+v0*v0);
          real C_T     = std::min( 1.  , interp( ref_velmag , ref_thrust_coef , mag0 ) );
          real C_P     = std::min( C_T , interp( ref_velmag , ref_power_coef  , mag0 ) );
          real pwr     =                 interp( ref_velmag , ref_power       , mag0 );
          real C_TKE   = coupler.get_option<real>("turbine_f_TKE",0.25) * (C_T - C_P);
          // Application of disk onto tendencies
          parallel_for( YAKL_AUTO_LABEL() , nz , KOKKOS_LAMBDA (int k) {
            if (prop(k) > 0) {
              real wt = prop(k)*M_PI*rad*rad/(dx*dy*dz(k));
              // Compute tendencies implied by actuator disk thoery; Only apply TKE for disk, not blades
              real t_u    = -0.5f             *C_T  *mag0*mag0*cos_yaw*wt;
              real t_v    = -0.5f             *C_T  *mag0*mag0*sin_yaw*wt;
              real t_tke  =  0.5f*rho_d(k,j,i)*C_TKE*mag0*mag0*mag0   *wt;
              tend_u  (k,j,i) += t_u;
              tend_v  (k,j,i) += t_v;
              tend_tke(k,j,i) += t_tke;
            }
          });
          turbine.mag_trace  .push_back( mag0 );
          turbine.power_trace.push_back( pwr  );
        } // if (turbine.active)
      } // for (int iturb = 0; iturb < turbine_group.turbines.size(); iturb++)

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        uvel(k,j,i) += dt * tend_u  (k,j,i);
        vvel(k,j,i) += dt * tend_v  (k,j,i);
        tke (k,j,i) += dt * tend_tke(k,j,i);
      });

      trace_size++;
    }


    void disk_average_wind( core::Coupler const & coupler     ,
                            RefTurbine    const & ref_turbine ,
                            real                & avg_u       ,
                            real                & avg_v       ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx      = coupler.get_nx();
      auto ny      = coupler.get_ny();
      auto nz      = coupler.get_nz();
      auto zint    = coupler.get_zint().createHostCopy();
      auto nx_glob = coupler.get_nx_glob();
      auto ny_glob = coupler.get_ny_glob();
      auto &dm     = coupler.get_data_manager_readonly();
      auto uvel    = dm.get<real const,3>("uvel");
      auto vvel    = dm.get<real const,3>("vvel");

      realHost1d prop_h("prop",nz);
      int nsamp = 5;
      for (int k=0; k < nz; k++) {
        prop_h(k) = 0;
        for (int kk=0; kk < nsamp; kk++) {
          real z  = zint(k)+(kk+0.5)*(zint(k+1)-zint(k))/nsamp;
          real z0 = (z-ref_turbine.hub_height)/ref_turbine.blade_radius;
          if (std::abs(z0) < 1) prop_h(k) += std::sqrt(1-z0*z0);
        }
      }
      using yakl::componentwise::operator/;
      auto prop = (prop_h/yakl::intrinsics::sum(prop_h)).createDeviceCopy();

      real2d udisk("udisk",ny,nx);
      real2d vdisk("vdisk",ny,nx);
      udisk = 0;
      vdisk = 0;
      real r_nx_ny = 1./(nx_glob*ny_glob);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        if (prop(k) > 0) {
          Kokkos::atomic_add( &udisk(j,i) , prop(k)*uvel(k,j,i)*r_nx_ny );
          Kokkos::atomic_add( &vdisk(j,i) , prop(k)*vvel(k,j,i)*r_nx_ny );
        }
      });
      avg_u = coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(udisk),MPI_SUM);
      avg_v = coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(vdisk),MPI_SUM);
    }


    // Linear interpolation in a reference variable based on u_infinity and reference u_infinity
    real interp( std::vector<real> const &ref_umag , std::vector<real> const &ref_var , real umag ) {
      int imax = ref_umag.size()-1; // Max index for the table
      if ( umag < ref_umag.at(0) || umag > ref_umag.at(imax) ) return 0;
      int i = 0;
      while (umag > ref_umag.at(i)) { i++; }
      if (i > 0) i--;
      real fac = (ref_umag.at(i+1) - umag) / (ref_umag.at(i+1)-ref_umag.at(i));
      return fac*ref_var.at(i) + (1-fac)*ref_var.at(i+1);
    }

  };

}



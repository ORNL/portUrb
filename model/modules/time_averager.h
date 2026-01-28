#pragma once

#include "coupler.h"

namespace modules {
  
  // This class keeps track of time-averaged u, v, w, and TKE fields since the last reset
  struct Time_Averager {

    // Allocate and initialize time-averaged fields since the last reset
    // Also, register these fields as output variables with the coupler for output and restart
    void init( core::Coupler &coupler ) {
      auto nx   = coupler.get_nx();  // Local number of cells in the x-direction
      auto ny   = coupler.get_ny();  // Local number of cells in the y-direction
      auto nz   = coupler.get_nz();  // Number of cells in the z-direction
      auto &dm  = coupler.get_data_manager_readwrite();  // Get DataManager for read/write access
      coupler.set_option<real>("time_averager_etime",0); // Initialize elapsed time time-averaging option to zero
      // Allocate time-averaged fields to the coupler's DataManager, and initialize to zero
      dm.register_and_allocate<real>("avg_u"     ,"",{nz,ny,nx});    dm.get<real,3>("avg_u"     ) = 0;
      dm.register_and_allocate<real>("avg_v"     ,"",{nz,ny,nx});    dm.get<real,3>("avg_v"     ) = 0;
      dm.register_and_allocate<real>("avg_w"     ,"",{nz,ny,nx});    dm.get<real,3>("avg_w"     ) = 0;
      dm.register_and_allocate<real>("avg_tke"   ,"",{nz,ny,nx});    dm.get<real,3>("avg_tke"   ) = 0;
      // Register these fields as output variables with the coupler for output / restart
      coupler.register_output_variable<real>( "avg_u"   , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_v"   , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_w"   , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_tke" , core::Coupler::DIMS_3D );
      if (coupler.get_option<bool>("output_correlations",false)) {
        dm.register_and_allocate<real>("corr_avg_u","",{nz,ny,nx});    dm.get<real,3>("corr_avg_u") = 0;
        dm.register_and_allocate<real>("corr_avg_v","",{nz,ny,nx});    dm.get<real,3>("corr_avg_v") = 0;
        dm.register_and_allocate<real>("corr_avg_w","",{nz,ny,nx});    dm.get<real,3>("corr_avg_w") = 0;
        dm.register_and_allocate<real>("avg_up_up" ,"",{nz,ny,nx});    dm.get<real,3>("avg_up_up" ) = 0;
        dm.register_and_allocate<real>("avg_up_vp" ,"",{nz,ny,nx});    dm.get<real,3>("avg_up_vp" ) = 0;
        dm.register_and_allocate<real>("avg_up_wp" ,"",{nz,ny,nx});    dm.get<real,3>("avg_up_wp" ) = 0;
        dm.register_and_allocate<real>("avg_vp_vp" ,"",{nz,ny,nx});    dm.get<real,3>("avg_vp_vp" ) = 0;
        dm.register_and_allocate<real>("avg_vp_wp" ,"",{nz,ny,nx});    dm.get<real,3>("avg_vp_wp" ) = 0;
        dm.register_and_allocate<real>("avg_wp_wp" ,"",{nz,ny,nx});    dm.get<real,3>("avg_wp_wp" ) = 0;
        coupler.register_output_variable<real>( "corr_avg_u" , core::Coupler::DIMS_3D );
        coupler.register_output_variable<real>( "corr_avg_v" , core::Coupler::DIMS_3D );
        coupler.register_output_variable<real>( "corr_avg_w" , core::Coupler::DIMS_3D );
        coupler.register_output_variable<real>( "avg_up_up"  , core::Coupler::DIMS_3D );
        coupler.register_output_variable<real>( "avg_up_vp"  , core::Coupler::DIMS_3D );
        coupler.register_output_variable<real>( "avg_up_wp"  , core::Coupler::DIMS_3D );
        coupler.register_output_variable<real>( "avg_vp_vp"  , core::Coupler::DIMS_3D );
        coupler.register_output_variable<real>( "avg_vp_wp"  , core::Coupler::DIMS_3D );
        coupler.register_output_variable<real>( "avg_wp_wp"  , core::Coupler::DIMS_3D );
      }
    }


    // Reset time-averaged fields since the last reset to zero
    void reset( core::Coupler &coupler ) {
      coupler.set_option<real>("time_averager_etime",0);
      auto &dm  = coupler.get_data_manager_readwrite(); // Get DataManager for read/write access
      dm.get<real,3>("avg_u"    ) = 0;
      dm.get<real,3>("avg_v"    ) = 0;
      dm.get<real,3>("avg_w"    ) = 0;
      dm.get<real,3>("avg_tke"  ) = 0;
      dm.get<real,3>("avg_up_up") = 0;
      dm.get<real,3>("avg_up_vp") = 0;
      dm.get<real,3>("avg_up_wp") = 0;
      dm.get<real,3>("avg_vp_vp") = 0;
      dm.get<real,3>("avg_vp_wp") = 0;
      dm.get<real,3>("avg_wp_wp") = 0;
    }


    // Accumulate time-averaged fields since the last reset
    void accumulate( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx_glob    = coupler.get_nx_glob();  // Get the global number of cells in the x-direction
      auto ny_glob    = coupler.get_ny_glob();  // Get the global number of cells in the y-direction
      auto nx         = coupler.get_nx();       // Local number of cells in the x-direction
      auto ny         = coupler.get_ny();       // Local number of cells in the y-direction
      auto nz         = coupler.get_nz();       // Number of cells in the z-direction
      auto &dm        = coupler.get_data_manager_readwrite();  // Get DataManager for read/write access
      auto uvel       = dm.get<real const,3>("uvel"       );  // Get u-velocity field
      auto vvel       = dm.get<real const,3>("vvel"       );  // Get v-velocity field
      auto wvel       = dm.get<real const,3>("wvel"       );  // Get w-velocity field
      auto tke        = dm.get<real const,3>("TKE"        );  // Get TKE field
      auto avg_u      = dm.get<real      ,3>("avg_u"      );
      auto avg_v      = dm.get<real      ,3>("avg_v"      );
      auto avg_w      = dm.get<real      ,3>("avg_w"      );
      auto avg_tke    = dm.get<real      ,3>("avg_tke"    );
      auto corr_avg_u = dm.get<real      ,3>("corr_avg_u" );
      auto corr_avg_v = dm.get<real      ,3>("corr_avg_v" );
      auto corr_avg_w = dm.get<real      ,3>("corr_avg_w" );
      auto avg_up_up  = dm.get<real      ,3>("avg_up_up"  );
      auto avg_up_vp  = dm.get<real      ,3>("avg_up_vp"  );
      auto avg_up_wp  = dm.get<real      ,3>("avg_up_wp"  );
      auto avg_vp_vp  = dm.get<real      ,3>("avg_vp_vp"  );
      auto avg_vp_wp  = dm.get<real      ,3>("avg_vp_wp"  );
      auto avg_wp_wp  = dm.get<real      ,3>("avg_wp_wp"  );
      auto etime      = coupler.get_option<real>("time_averager_etime"); // Get elapsed time for time-averaging since last reset
      real inertia = etime / (etime + dt);  // Compute inertia
      // Update time-averaged fields using moving average formula
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        avg_u  (k,j,i) = inertia*avg_u  (k,j,i) + (1-inertia)*uvel(k,j,i);
        avg_v  (k,j,i) = inertia*avg_v  (k,j,i) + (1-inertia)*vvel(k,j,i);
        avg_w  (k,j,i) = inertia*avg_w  (k,j,i) + (1-inertia)*wvel(k,j,i);
        avg_tke(k,j,i) = inertia*avg_tke(k,j,i) + (1-inertia)*tke (k,j,i);
      });
      coupler.set_option<real>("time_averager_etime",etime+dt);  // Update elapsed time for time-averaging since last reset
      if (coupler.get_option<bool>("output_correlations",false)) {
        real tau = coupler.get_option<real>("correlation_time_scale");
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
          corr_avg_u(k,j,i) = (tau-dt)/tau*corr_avg_u(k,j,i) + dt/tau*uvel(k,j,i);
          corr_avg_v(k,j,i) = (tau-dt)/tau*corr_avg_v(k,j,i) + dt/tau*vvel(k,j,i);
          corr_avg_w(k,j,i) = (tau-dt)/tau*corr_avg_w(k,j,i) + dt/tau*wvel(k,j,i);
          real up = uvel(k,j,i)-corr_avg_u(k,j,i);
          real vp = vvel(k,j,i)-corr_avg_v(k,j,i);
          real wp = wvel(k,j,i)-corr_avg_w(k,j,i);
          avg_up_up(k,j,i) = inertia*avg_up_up(k,j,i) + (1-inertia)*up*up;
          avg_up_vp(k,j,i) = inertia*avg_up_vp(k,j,i) + (1-inertia)*up*vp;
          avg_up_wp(k,j,i) = inertia*avg_up_wp(k,j,i) + (1-inertia)*up*wp;
          avg_vp_vp(k,j,i) = inertia*avg_vp_vp(k,j,i) + (1-inertia)*vp*vp;
          avg_vp_wp(k,j,i) = inertia*avg_vp_wp(k,j,i) + (1-inertia)*vp*wp;
          avg_wp_wp(k,j,i) = inertia*avg_wp_wp(k,j,i) + (1-inertia)*wp*wp;
        });
      }
    }
  };

}



#pragma once

#include "coupler.h"

namespace modules {
  
  // This class keeps track of time-averaged u, v, w, and TKE fields since the last reset and also long-term averages
  struct Time_Averager {

    // Allocate and initialize time-averaged fields since the last reset and long-term averages
    // Also, register these fields as output variables with the coupler for output and restart
    void init( core::Coupler &coupler ) {
      auto nx   = coupler.get_nx();  // Local number of cells in the x-direction
      auto ny   = coupler.get_ny();  // Local number of cells in the y-direction
      auto nz   = coupler.get_nz();  // Number of cells in the z-direction
      auto &dm  = coupler.get_data_manager_readwrite();  // Get DataManager for read/write access
      coupler.set_option<real>("time_averager_etime",0); // Initialize elapsed time time-averaging option to zero
      // Allocate time-averaged fields to the coupler's DataManager, and initialize to zero
      dm.register_and_allocate<real>("avg_u"      ,"",{nz,ny,nx});    dm.get<real,3>("avg_u"      ) = 0;
      dm.register_and_allocate<real>("avg_v"      ,"",{nz,ny,nx});    dm.get<real,3>("avg_v"      ) = 0;
      dm.register_and_allocate<real>("avg_w"      ,"",{nz,ny,nx});    dm.get<real,3>("avg_w"      ) = 0;
      dm.register_and_allocate<real>("avg_tke"    ,"",{nz,ny,nx});    dm.get<real,3>("avg_tke"    ) = 0;
      dm.register_and_allocate<real>("long_avg_u" ,"",{nz,ny,nx});    dm.get<real,3>("long_avg_u" ) = 0;
      dm.register_and_allocate<real>("long_avg_v" ,"",{nz,ny,nx});    dm.get<real,3>("long_avg_v" ) = 0;
      dm.register_and_allocate<real>("long_avg_w" ,"",{nz,ny,nx});    dm.get<real,3>("long_avg_w" ) = 0;
      // Register these fields as output variables with the coupler for output / restart
      coupler.register_output_variable<real>( "avg_u"       , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_v"       , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_w"       , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_tke"     , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "long_avg_u"  , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "long_avg_v"  , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "long_avg_w"  , core::Coupler::DIMS_3D );
    }


    // Reset time-averaged fields since the last reset to zero and non-long-term averages to zero
    void reset( core::Coupler &coupler ) {
      coupler.set_option<real>("time_averager_etime",0);
      auto &dm  = coupler.get_data_manager_readwrite(); // Get DataManager for read/write access
      dm.get<real,3>("avg_u"  ) = 0;
      dm.get<real,3>("avg_v"  ) = 0;
      dm.get<real,3>("avg_w"  ) = 0;
      dm.get<real,3>("avg_tke") = 0;
    }


    // Accumulate time-averaged fields since the last reset and long-term averages given the current dt
    void accumulate( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx_glob = coupler.get_nx_glob();  // Get the global number of cells in the x-direction
      auto ny_glob = coupler.get_ny_glob();  // Get the global number of cells in the y-direction
      auto nx      = coupler.get_nx();       // Local number of cells in the x-direction
      auto ny      = coupler.get_ny();       // Local number of cells in the y-direction
      auto nz      = coupler.get_nz();       // Number of cells in the z-direction
      auto &dm     = coupler.get_data_manager_readwrite();  // Get DataManager for read/write access
      auto uvel        = dm.get<real const,3>("uvel"       );  // Get u-velocity field
      auto vvel        = dm.get<real const,3>("vvel"       );  // Get v-velocity field
      auto wvel        = dm.get<real const,3>("wvel"       );  // Get w-velocity field
      auto tke         = dm.get<real const,3>("TKE"        );  // Get TKE field
      auto avg_u       = dm.get<real      ,3>("avg_u"      );  // Get time-averaged u-velocity field since last reset
      auto avg_v       = dm.get<real      ,3>("avg_v"      );  // Get time-averaged v-velocity field since last reset
      auto avg_w       = dm.get<real      ,3>("avg_w"      );  // Get time-averaged w-velocity field since last reset
      auto long_avg_u  = dm.get<real      ,3>("long_avg_u" );  // Get long-term averaged u-velocity field
      auto long_avg_v  = dm.get<real      ,3>("long_avg_v" );  // Get long-term averaged v-velocity field
      auto long_avg_w  = dm.get<real      ,3>("long_avg_w" );  // Get long-term averaged w-velocity field
      auto avg_tke     = dm.get<real      ,3>("avg_tke"    );  // Get time-averaged TKE field since last reset
      auto etime       = coupler.get_option<real>("time_averager_etime"); // Get elapsed time for time-averaging since last reset
      real inertia = etime / (etime + dt);  // Compute inertia for non-long-term averages
      real etime_long = coupler.get_option<real>("elapsed_time");  // Get total elapsed simulation time for long-term averages
      real inertia_long = etime_long / (etime_long + dt);  // Compute inertia for long-term averages
      // Update time-averaged fields using moving average formula
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        long_avg_u(k,j,i) = inertia_long * long_avg_u(k,j,i) + (1-inertia_long) * uvel(k,j,i);
        long_avg_v(k,j,i) = inertia_long * long_avg_v(k,j,i) + (1-inertia_long) * vvel(k,j,i);
        long_avg_w(k,j,i) = inertia_long * long_avg_w(k,j,i) + (1-inertia_long) * wvel(k,j,i);
        avg_u     (k,j,i) = inertia      * avg_u     (k,j,i) + (1-inertia     ) * uvel(k,j,i);
        avg_v     (k,j,i) = inertia      * avg_v     (k,j,i) + (1-inertia     ) * vvel(k,j,i);
        avg_w     (k,j,i) = inertia      * avg_w     (k,j,i) + (1-inertia     ) * wvel(k,j,i);
        avg_tke   (k,j,i) = inertia      * avg_tke   (k,j,i) + (1-inertia     ) * tke (k,j,i);
      });
      coupler.set_option<real>("time_averager_etime",etime+dt);  // Update elapsed time for time-averaging since last reset
    }
  };

}



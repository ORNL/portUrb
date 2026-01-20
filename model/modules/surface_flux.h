
#pragma once

#include "coupler.h"

namespace modules {


  struct SurfaceFlux {

    int  static constexpr idR = 0;
    int  static constexpr idU = 1;
    int  static constexpr idV = 2;
    int  static constexpr idW = 3;
    int  static constexpr idT = 4;
    int  static constexpr num_state = 5;
    int  static constexpr hs = 1;
    real static constexpr neut_thresh = 0.05;



    void init( core::Coupler & coupler ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx          = coupler.get_nx();  // Get local number of grid points in x-direction
      auto ny          = coupler.get_ny();  // Get local number of grid points in y-direction
      auto nz          = coupler.get_nz();  // Get number of grid points in z-direction
      auto num_tracers = coupler.get_num_tracers();
      auto &dm         = coupler.get_data_manager_readwrite(); // Get reference to the data manager (read/write)
      real4d state  ("state"  ,num_state  ,nz,ny,nx);
      real4d tracers("tracers",num_tracers,nz,ny,nx);
      convert_coupler_to_dynamics( coupler , state , tracers );
      dm.register_and_allocate<real>("surface_flux_imm_theta","",{nz+2*hs,ny+2*hs,nx+2*hs});
      auto imm_theta = dm.get<real,3>("surface_flux_imm_theta");
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        imm_theta(hs+k,hs+j,hs+i) = state(idT,k,j,i);
      });
      core::MultiField<real,3> fields;
      fields.add_field(imm_theta);
      coupler.halo_exchange( fields , hs );
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny+2*hs,nx+2*hs) , KOKKOS_LAMBDA (int j, int i) {
        imm_theta(0    ,j,i) = imm_theta(1      ,j,i);
        imm_theta(hs+nz,j,i) = imm_theta(hs+nz-1,j,i);
      });
      coupler.register_write_output_module( [=] (core::Coupler &coupler , yakl::SimplePNetCDF &nc) {
        using yakl::c::parallel_for;
        using yakl::c::SimpleBounds;
        auto nz        = coupler.get_nz();
        auto ny        = coupler.get_ny();
        auto nx        = coupler.get_nx();
        auto i_beg     = coupler.get_i_beg();
        auto j_beg     = coupler.get_j_beg();
        auto &dm       = coupler.get_data_manager_readonly(); // Get reference to the data manager (read/write)
        auto imm_theta = dm.get<real const,3>("surface_flux_imm_theta");
        nc.redef();
        if (! nc.dim_exists("nzp2")) nc.create_dim( "zp2" , nz+2 );
        nc.create_var<real>( "surface_flux_imm_theta" , {"zp2","y","x"} );
        nc.enddef();
        real3d imm_theta_loc("imm_theta_loc",nz+2,ny,nx);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+2,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
          imm_theta_loc(k,j,i) = imm_theta(k,hs+j,hs+i);
        });
        std::vector<MPI_Offset> start = {(MPI_Offset)0,(MPI_Offset)j_beg,(MPI_Offset)i_beg};
        nc.write_all( imm_theta_loc , "surface_flux_imm_theta" , start );
      });
      coupler.register_overwrite_with_restart_module( [=] (core::Coupler &coupler , yakl::SimplePNetCDF &nc) {
        auto nz        = coupler.get_nz();
        auto ny        = coupler.get_ny();
        auto nx        = coupler.get_nx();
        auto i_beg     = coupler.get_i_beg();
        auto j_beg     = coupler.get_j_beg();
        auto &dm       = coupler.get_data_manager_readwrite(); // Get reference to the data manager (read/write)
        auto imm_theta = dm.get<real,3>("surface_flux_imm_theta");
        real3d imm_theta_loc("imm_theta_loc",nz+2,ny,nx);
        std::vector<MPI_Offset> start = {(MPI_Offset)0,(MPI_Offset)j_beg,(MPI_Offset)i_beg};
        nc.read_all( imm_theta_loc , "surface_flux_imm_theta" , start );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+2,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
          imm_theta(k,hs+j,hs+i) = imm_theta_loc(k,j,i);
        });
        core::MultiField<real,3> fields;
        fields.add_field(imm_theta);
        coupler.halo_exchange( fields , hs );
      });
    }
    


    // Applies surface fluxes of momenta and temperature from the model surface as well as
    //   immersed boundaries using Monin-Obukhov similarity theory
    // coupler : Coupler object containing the data manager and options
    // dt      : Timestep size in seconds
    void apply( core::Coupler &coupler   ,
                real dt                  ,
                bool force_theta = false ,
                bool stab_corr   = false ,
                real nu = 1.5e-5         ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx          = coupler.get_nx();  // Get local number of grid points in x-direction
      auto ny          = coupler.get_ny();  // Get local number of grid points in y-direction
      auto nz          = coupler.get_nz();  // Get number of grid points in z-direction
      auto dx          = coupler.get_dx();  // Get grid spacing in x-direction
      auto dy          = coupler.get_dy();  // Get grid spacing in y-direction
      auto dz          = coupler.get_dz();  // Get grid spacing in z-direction
      auto num_tracers = coupler.get_num_tracers();
      auto p0          = coupler.get_option<real>("p0");       // Reference pressure in Pa
      auto R_d         = coupler.get_option<real>("R_d");      // Gas constant for dry air in J/(kg·K)
      auto cp_d        = coupler.get_option<real>("cp_d");     // Specific heat at constant pressure for dry air in J/(kg·K)
      auto grav        = coupler.get_option<real>("grav");
      auto &dm         = coupler.get_data_manager_readwrite(); // Get reference to the data manager (read/write)
      auto imm_prop    = dm.get<real const,3>("immersed_proportion_halos"); // Get immersed boundary proportion array
      auto imm_rough   = dm.get<real const,3>("immersed_roughness_halos" ); // Get immersed boundary roughness array
      auto imm_theta   = dm.get<real const,3>("surface_flux_imm_theta"   );
      real4d state  ("state"  ,num_state  ,nz,ny,nx);
      real4d tracers("tracers",num_tracers,nz,ny,nx);
      convert_coupler_to_dynamics( coupler , state , tracers );

      // Allocate arrays to hold surface flux tendencies
      real3d tend_u ("tend_u" ,nz,ny,nx);
      real3d tend_v ("tend_v" ,nz,ny,nx);
      real3d tend_w ("tend_w" ,nz,ny,nx);
      real3d tend_th("tend_th",nz,ny,nx);

      real vk   = 0.40;   // von karman constant
      real Czil = 0.1;

      // Compute surface flux tendencies using Monin-Obukhov similarity theory
      // This applies surface friction to neighboring cells if they are the surface or if they are
      //   immersed. 
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real u  = state(idU,k,j,i);  // u-velocity at this grid point
        real v  = state(idV,k,j,i);  // v-velocity at this grid point
        real w  = state(idW,k,j,i);  // w-velocity at this grid point
        real th = state(idT,k,j,i);  // Potential temperature at this grid point
        // Initialize tendencies to zero prior to accumulation
        tend_u (k,j,i) = 0;
        tend_v (k,j,i) = 0;
        tend_w (k,j,i) = 0;
        tend_th(k,j,i) = 0;
        int indk, indj, indi;  // These indices will index into neighboring cells

        // West neighbor
        indk = hs+k;  indj = hs+j;  indi = hs+i-1;
        if (imm_prop(indk,indj,indi) > 0) {
          // Compute roughness length, log of height ratio, drag coefficient, immersed temperature,
          //   and adjacent transverse velocity magnitude
          real z0        = imm_rough(indk,indj,indi);
          real mag       = std::max( std::sqrt(v*v+w*w) , 1.e-10 );
          real ustar     = vk*mag/std::log((dx/2+z0)/z0);
          tend_v(k,j,i) += -ustar*ustar*(v-0)/mag/dx;
          tend_w(k,j,i) += -ustar*ustar*(w-0)/mag/dx;
          if (force_theta) {
            real th0        = imm_theta(indk,indj,indi);
            real z0h        = z0*std::exp(-vk*Czil*std::sqrt(ustar*z0/nu));
            real thstar     = vk*(th-th0)/std::log((dx/2+z0h)/z0h);
            tend_th(k,j,i) += -ustar*thstar/dx;
          }
        }

        // East neighbor
        indk = hs+k;  indj = hs+j;  indi = hs+i+1;
        if (imm_prop(indk,indj,indi) > 0) {
          // Compute roughness length, log of height ratio, drag coefficient, immersed temperature,
          //   and adjacent transverse velocity magnitude
          real z0        = imm_rough(indk,indj,indi);
          real mag       = std::max( std::sqrt(v*v+w*w) , 1.e-10 );
          real ustar     = vk*mag/std::log((dx/2+z0)/z0);
          tend_v(k,j,i) += -ustar*ustar*(v-0)/mag/dx;
          tend_w(k,j,i) += -ustar*ustar*(w-0)/mag/dx;
          if (force_theta) {
            real th0        = imm_theta(indk,indj,indi);
            real z0h        = z0*std::exp(-vk*Czil*std::sqrt(ustar*z0/nu));
            real thstar     = vk*(th-th0)/std::log((dx/2+z0h)/z0h);
            tend_th(k,j,i) += -ustar*thstar/dx;
          }
        }

        // South neighbor
        indk = hs+k;  indj = hs+j-1;  indi = hs+i;
        if (imm_prop(indk,indj,indi) > 0) {
          // Compute roughness length, log of height ratio, drag coefficient, immersed temperature,
          //   and adjacent transverse velocity magnitude
          real z0        = imm_rough(indk,indj,indi);
          real mag       = std::max( std::sqrt(u*u+w*w) , 1.e-10 );
          real ustar     = vk*mag/std::log((dy/2+z0)/z0);
          tend_u(k,j,i) += -ustar*ustar*(u-0)/mag/dy;
          tend_w(k,j,i) += -ustar*ustar*(w-0)/mag/dy;
          if (force_theta) {
            real th0        = imm_theta(indk,indj,indi);
            real z0h        = z0*std::exp(-vk*Czil*std::sqrt(ustar*z0/nu));
            real thstar     = vk*(th-th0)/std::log((dy/2+z0h)/z0h);
            tend_th(k,j,i) += -ustar*thstar/dy;
          }
        }

        // North neighbor
        indk = hs+k;  indj = hs+j+1;  indi = hs+i;
        if (imm_prop(indk,indj,indi) > 0) {
          // Compute roughness length, log of height ratio, drag coefficient, immersed temperature,
          //   and adjacent transverse velocity magnitude
          real z0        = imm_rough(indk,indj,indi);
          real mag       = std::max( std::sqrt(u*u+w*w) , 1.e-10 );
          real ustar     = vk*mag/std::log((dy/2+z0)/z0);
          tend_u(k,j,i) += -ustar*ustar*(u-0)/mag/dy;
          tend_w(k,j,i) += -ustar*ustar*(w-0)/mag/dy;
          if (force_theta) {
            real th0        = imm_theta(indk,indj,indi);
            real z0h        = z0*std::exp(-vk*Czil*std::sqrt(ustar*z0/nu));
            real thstar     = vk*(th-th0)/std::log((dy/2+z0h)/z0h);
            tend_th(k,j,i) += -ustar*thstar/dy;
          }
        }

        // Bottom neighbor
        indk = hs+k-1;  indj = hs+j;  indi = hs+i;
        if (imm_prop(indk,indj,indi) > 0) {
          // Compute roughness length, log of height ratio, drag coefficient, immersed temperature,
          //   and adjacent transverse velocity magnitude
          real z0     = imm_rough(indk,indj,indi);
          real mag    = std::max( std::sqrt(u*u+v*v) , 1.e-10 );
          real ustar  = vk*mag/std::log((dz(k)/2+z0)/z0);
          real th0    = imm_theta(indk,indj,indi);
          real z0h    = z0*std::exp(-vk*Czil*std::sqrt(ustar*z0/nu));
          real thstar = vk*(th-th0)/std::log((dz(k)/2+z0h)/z0h);
          if (stab_corr && force_theta) stability_correction(vk,mag,z0,th,th0,grav,Czil,nu,dz(k),false,ustar,thstar);
          tend_u(k,j,i) += -ustar*ustar*(u-0)/mag/dz(k);
          tend_v(k,j,i) += -ustar*ustar*(v-0)/mag/dz(k);
          if (force_theta) tend_th(k,j,i) += -ustar*thstar/dz(k);
        }
        
        // Top neighbor
        indk = hs+k+1;  indj = hs+j;  indi = hs+i;
        if (imm_prop(indk,indj,indi) > 0) {
          // Compute roughness length, log of height ratio, drag coefficient, immersed temperature,
          //   and adjacent transverse velocity magnitude
          real z0     = imm_rough(indk,indj,indi);
          real mag    = std::max( std::sqrt(u*u+v*v) , 1.e-10 );
          real ustar  = vk*mag/std::log((dz(k)/2+z0)/z0);
          real th0    = imm_theta(indk,indj,indi);
          real z0h    = z0*std::exp(-vk*Czil*std::sqrt(ustar*z0/nu));
          real thstar = vk*(th-th0)/std::log((dz(k)/2+z0h)/z0h);
          if (stab_corr && force_theta) stability_correction(vk,mag,z0,th,th0,grav,Czil,nu,dz(k),true,ustar,thstar);
          tend_u(k,j,i) += -ustar*ustar*(u-0)/mag/dz(k);
          tend_v(k,j,i) += -ustar*ustar*(v-0)/mag/dz(k);
          if (force_theta) tend_th(k,j,i) += -ustar*thstar/dz(k);
        }
      });

      // Apply the accumulated tendencies to the state variables
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        state(idU,k,j,i) += dt*tend_u (k,j,i);
        state(idV,k,j,i) += dt*tend_v (k,j,i);
        state(idW,k,j,i) += dt*tend_w (k,j,i);
        state(idT,k,j,i) += dt*tend_th(k,j,i);
      });

      convert_dynamics_to_coupler( coupler , state , tracers );
    }



    KOKKOS_INLINE_FUNCTION static void stability_correction( real vk , real mag , real z0 , real th , real th0 ,
                                                             real grav , real Czil , real nu , real dzloc ,
                                                             bool wall_at_top ,
                                                             real & ustar , real & thstar ) {
      using std::sqrt;
      using std::log;
      using std::atan;
      int  max_iter = 20;
      real beta_m   = 5;
      real beta_h   = 5;
      real gamma_m  = 16;
      real gamma_h  = 16;
      real zmin     = -5;
      real zmax     =  5;
      real tol      = 1e-6;
      for (int iter = 0; iter < max_iter; iter++) {
        real ustar_prev  = ustar ;
        real thstar_prev = thstar;
        real z0h   = z0*std::exp(-vk*Czil*std::sqrt(ustar*z0/nu));
        real wpthp = -ustar*thstar;
        if (wall_at_top) wpthp *= -1;
        wpthp      = std::copysign( std::max( std::abs(wpthp) , 1.e-6 ) , wpthp );
        real L     = -ustar*ustar*ustar*th/(vk*grav*wpthp);
        real zeta  = std::max(zmin,std::min(zmax,(dzloc/2+z0)/L));
        real psi_m_1, psi_h_1;
        if (std::abs(zeta) < neut_thresh) {
          psi_m_1 = 0;
          psi_h_1 = 0;
        } else if (zeta >= 0) {
          psi_m_1 = -beta_m*zeta;
          psi_h_1 = -beta_h*zeta;
        } else {
          real xm = sqrt(sqrt(1-gamma_m*zeta));
          real xh = sqrt(sqrt(1-gamma_h*zeta));
          psi_m_1 = 2*log((1+xm)/2) + log((1+xm*xm)/2) - 2*atan(xm) + M_PI/2;
          psi_h_1 = 2*log((1+xh*xh)/2);
        }
        zeta = std::max(zmin,std::min(zmax,z0/L));
        real psi_m_2;
        if (std::abs(zeta) < neut_thresh) {
          psi_m_2 = 0;
        } else if (zeta >= 0) {
          psi_m_2 = -beta_m*zeta;
        } else {
          real xm = sqrt(sqrt(1-gamma_m*zeta));
          psi_m_2 = 2*log((1+xm)/2) + log((1+xm*xm)/2) - 2*atan(xm) + M_PI/2;
        }
        zeta = std::max(zmin,std::min(zmax,z0h/L));
        real psi_h_2;
        if (std::abs(zeta) < neut_thresh) {
          psi_h_2 = 0;
        } else if (zeta >= 0) {
          psi_h_2 = -beta_h*zeta;
        } else {
          real xh = sqrt(sqrt(1-gamma_h*zeta));
          psi_h_2 = 2*log((1+xh*xh)/2);
        }
        ustar  = vk*mag     /std::max(1.e-3,log((dzloc/2+z0 )/z0 ) - psi_m_1 + psi_m_2);
        thstar = vk*(th-th0)/std::max(1.e-3,log((dzloc/2+z0h)/z0h) - psi_h_1 + psi_h_2);
        if (std::abs(ustar-ustar_prev) <= tol && std::abs(thstar-thstar_prev) <= tol) break;
      }
    }



    void change_surface_theta( core::Coupler & coupler , real dt , real rate ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      int  nx           = coupler.get_nx();  // Get the number of grid points in the x-direction
      int  ny           = coupler.get_ny();  // Get the number of grid points in the y-direction
      auto &dm          = coupler.get_data_manager_readwrite(); // Get reference to the data manager (read/write)
      auto imm_theta    = dm.get<real,3>("surface_flux_imm_theta");
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny+2*hs,nx+2*hs) , KOKKOS_LAMBDA (int j, int i) {
        imm_theta(0,j,i) += dt*rate;
      });
    }



    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    // coupler : reference to the coupler object
    // state   : dynamics state array
    // tracers : dynamics tracers array
    void convert_dynamics_to_coupler( core::Coupler &coupler ,
                                      realConst4d    state   ,
                                      realConst4d    tracers ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto  nx          = coupler.get_nx();                     // Number of cells in x-direction (not including halos)
      auto  ny          = coupler.get_ny();                     // Number of cells in y-direction (not including halos)
      auto  nz          = coupler.get_nz();                     // Number of cells in z-direction (not including halos)
      auto  R_d         = coupler.get_option<real>("R_d"    );  // Gas constant for dry air
      auto  R_v         = coupler.get_option<real>("R_v"    );  // Gas constant for water vapor
      auto  gamma       = coupler.get_option<real>("gamma_d");  // Ratio of specific heats for dry air
      auto  C0          = coupler.get_option<real>("C0"     );  // p = C0 * (rho*theta)^gamma
      auto  idWV        = coupler.get_option<int >("idWV"   );  // Tracer index for water vapor
      auto  num_tracers = coupler.get_num_tracers();            // Number of tracers
      auto  &dm         = coupler.get_data_manager_readwrite(); // Get data manager as read-write
      auto  dm_rho_d    = dm.get<real,3>("density_dry");        // Get coupler dry density array
      auto  dm_uvel     = dm.get<real,3>("uvel"       );        // Get coupler u-velocity array
      auto  dm_vvel     = dm.get<real,3>("vvel"       );        // Get coupler v-velocity array
      auto  dm_wvel     = dm.get<real,3>("wvel"       );        // Get coupler w-velocity array
      auto  dm_temp     = dm.get<real,3>("temp"       );        // Get coupler temperature array
      // Get array that determines whether each tracer adds to the mass of the air mixture
      auto  tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
      // Accrue the tracer fields from the coupler data manager
      core::MultiField<real,3> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real,3>(tracer_names.at(tr)) ); }
      // Loop over all grid cells to compute dry density, velocities, temperature, and store in coupler arrays
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real rho   = state(idR,k,j,i);              // Total density
        real u     = state(idU,k,j,i);              // u-velocity
        real v     = state(idV,k,j,i);              // v-velocity
        real w     = state(idW,k,j,i);              // w-velocity
        real theta = state(idT,k,j,i);              // Potential temperature
        real press = C0 * pow( rho*theta , gamma ); // Full pressure
        real rho_v = tracers(idWV,k,j,i);           // Water vapor density
        real rho_d = rho;                           // Dry air density starting value
        // Subtract mass-adding tracers from total density to get dry air density
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho_d -= tracers(tr,k,j,i); }
        // Use equation of state to compute temperature from pressure, dry density, and water vapor density
        real temp = press / ( rho_d * R_d + rho_v * R_v );
        dm_uvel (k,j,i) = u;      // Store u-velocity in coupler array
        dm_vvel (k,j,i) = v;      // Store v-velocity in coupler array
        dm_wvel (k,j,i) = w;      // Store w-velocity in coupler array
        dm_temp (k,j,i) = temp;   // Store temperature in coupler array
        // Store tracer densities in coupler arrays
        for (int tr=0; tr < num_tracers; tr++) { dm_tracers(tr,k,j,i) = tracers(tr,k,j,i); }
      });
    }



    // Convert coupler's data to dynamics format of state and tracers arrays
    // coupler : reference to the coupler object
    // state   : dynamics state array
    // tracers : dynamics tracers array
    void convert_coupler_to_dynamics( core::Coupler const &coupler ,
                                      real4d              &state   ,
                                      real4d              &tracers ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto  nx          = coupler.get_nx();                    // Number of cells in x-direction (not including halos)
      auto  ny          = coupler.get_ny();                    // Number of cells in y-direction (not including halos)
      auto  nz          = coupler.get_nz();                    // Number of cells in z-direction (not including halos)
      auto  R_d         = coupler.get_option<real>("R_d"    ); // Gas constant for dry air
      auto  R_v         = coupler.get_option<real>("R_v"    ); // Gas constant for water vapor
      auto  gamma       = coupler.get_option<real>("gamma_d"); // Ratio of specific heats for dry air
      auto  C0          = coupler.get_option<real>("C0"     ); // p = C0 * (rho*theta)^gamma
      auto  idWV        = coupler.get_option<int >("idWV"   ); // Tracer index for water vapor
      auto  num_tracers = coupler.get_num_tracers();           // Number of tracers
      auto  &dm         = coupler.get_data_manager_readonly(); // Get data manager as read-only
      auto  dm_rho_d    = dm.get<real const,3>("density_dry"); // Get coupler dry density array
      auto  dm_uvel     = dm.get<real const,3>("uvel"       ); // Get coupler u-velocity array
      auto  dm_vvel     = dm.get<real const,3>("vvel"       ); // Get coupler v-velocity array
      auto  dm_wvel     = dm.get<real const,3>("wvel"       ); // Get coupler w-velocity array
      auto  dm_temp     = dm.get<real const,3>("temp"       ); // Get coupler temperature array
      // Get array that determines whether each tracer adds to the mass of the air mixture
      auto  tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
      // Accrue the tracer fields from the coupler data manager
      core::MultiField<real const,3> dm_tracers;
      auto tracer_names = coupler.get_tracer_names(); // Get the tracer names
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real const,3>(tracer_names.at(tr)) ); }
      // Loop over all grid cells to compute dynamics state and tracers arrays from coupler data
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real rho_d = dm_rho_d(k,j,i);                         // Dry air density
        real u     = dm_uvel (k,j,i);                         // u-velocity
        real v     = dm_vvel (k,j,i);                         // v-velocity
        real w     = dm_wvel (k,j,i);                         // w-velocity
        real temp  = dm_temp (k,j,i);                         // Temperature
        real rho_v = dm_tracers(idWV,k,j,i);                  // Water vapor density
        real press = rho_d * R_d * temp + rho_v * R_v * temp; // Full pressure
        real rho = rho_d;                                     // Total density starting value
        // Add mass-adding tracers to dry density to get total density
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho += dm_tracers(tr,k,j,i); }
        // Compute potential temperature from pressure and total density
        real theta = pow( press/C0 , 1._fp / gamma ) / rho;
        state(idR,k,j,i) = rho;   // Store total density in dynamics state array
        state(idU,k,j,i) = u;     // Store momentum in dynamics state array
        state(idV,k,j,i) = v;     // Store momentum in dynamics state array
        state(idW,k,j,i) = w;     // Store momentum in dynamics state array
        state(idT,k,j,i) = theta; // Store total potential temperature in dynamics state array
        // Store tracer densities in dynamics tracers array
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,k,j,i) = dm_tracers(tr,k,j,i); }
      });
    }

  };

}


#include "dynamics_edge_centered.h"

namespace modules {

void EulerEdgeCentered::compute_tendencies( core::Coupler       & coupler      ,
                             real4d        const & state        ,
                             real4d        const & state_tend   ,
                             real4d        const & tracers      ,
                             real4d        const & tracers_tend ,
                             real                  dt           ,
                             int                   istage       ,
                             int                   icycle       ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("compute_tendencies");
      #endif
      using yakl::SimpleBounds;
      auto nx                = coupler.get_nx();           // Number of cells in x-direction (excluding halos)
      auto ny                = coupler.get_ny();           // Number of cells in y-direction (excluding halos)
      auto nz                = coupler.get_nz();           // Number of cells in z-direction (excluding halos)
      auto dx                = coupler.get_dx();           // Grid spacing in x-direction
      auto dy                = coupler.get_dy();           // Grid spacing in y-direction
      auto dz                = coupler.get_dz();           // Grid spacing in z-direction
      auto px                = coupler.get_px();           // Grid spacing in x-direction
      auto py                = coupler.get_py();           // Grid spacing in y-direction
      auto nproc_x           = coupler.get_nproc_x();      // Grid spacing in x-direction
      auto nproc_y           = coupler.get_nproc_y();      // Grid spacing in y-direction
      auto i_beg             = coupler.get_i_beg();
      auto j_beg             = coupler.get_j_beg();
      auto num_tracers       = coupler.get_num_tracers();  // Total number of tracers
      auto enable_gravity    = coupler.get_option<bool>("enable_gravity",true); // Whether to enable gravity
      auto C0                = coupler.get_option<real>("C0"     );    // pressure = C0*pow(rho*theta,gamma)
      auto grav              = coupler.get_option<real>("grav"   );    // Gravity
      auto gamma             = coupler.get_option<real>("gamma_d");    // cp_dry / cv_dry (about 1.4)
      auto latitude          = coupler.get_option<real>("latitude",0); // For coriolis
      auto &dm               = coupler.get_data_manager_readonly();    // Grab read-only data manager
      auto immersed_prop     = dm.get<real const,3>("dycore_immersed_proportion_halos"); // Immersed Proportion
      auto immersed_dist     = dm.get<real const,3>("dycore_immersed_distance"); // Distance to the nearest immersed cell
      auto hy_dens_cells     = dm.get<real const,1>("hy_dens_cells"        ); // Hydrostatic density in cells with halos
      auto hy_theta_cells    = dm.get<real const,1>("hy_theta_cells"       ); // Hydrostatic potential temperature in cells with halos
      auto hy_theta_edges    = dm.get<real const,1>("hy_theta_edges"       ); // Hydrostatic potential temperature at edges (no halos)
      auto hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells"    ); // Hydrostatic pressure in cells with halos
      auto hy_dens_edges     = dm.get<real const,1>("hy_dens_edges"        ); // Hydrostatic density in cells with halos
      auto metjac_edges      = dm.get<real const,1>("dycore_metjac_edges"  ); // Vertical metric jacobian at edges
      // Compute matrices to convert polynomial coefficients to 2 GLL points and stencil values to 2 GLL points
      // These matrices will be in column-row format. That performed better than row-column format in performance tests
      real r_dx = 1./dx; // reciprocal of grid spacing
      real r_dy = 1./dy; // reciprocal of grid spacing
      real fcor = 2*7.2921e-5*std::sin(latitude/180*M_PI);  // For coriolis: 2*Omega*sin(latitude)

      FLOC constexpr imm_th = 0.5;

      FLOC cs = coupler.get_option<real>("dycore_cs",350);  // Speed of sound

      int nfields = num_state+1+num_tracers;

      // The main working array that holds all prognostic variables plus pressure
      yakl::Array<FLOC ****> fields_loc("fields_loc",nfields,nz+2*hs,ny+2*hs,nx+2*hs);
      bool rsst = coupler.get_option<bool>("dycore_rsst",false) || (coupler.get_option<real>("dycore_cs",350) != 350);

      // Load state and tracers into working array, dividing by density to get specific quantities, computing pressure,
      //  and subtracting hydrostatic values from density, potential temperature, and pressure
      // If Reduced Speed of Sound Technique (RSST) is being used, set pressure using cs^2 * (rho - rho_hydrostatic)
      //  Otherwise, use true pressure from equation of state
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        // Perturbation pressure if RSST is not used
        if (!rsst) fields_loc(idP,hs+k,hs+j,hs+i) = C0*std::pow(state(idT,k,j,i),gamma) - hy_pressure_cells(hs+k);
        real r_r = 1._fp / state(idR,k,j,i); // Reciprocal of density
        fields_loc(idR,hs+k,hs+j,hs+i) = state(idR,k,j,i) - hy_dens_cells(hs+k);
        // Load in state and tracers as specific quantities
        for (int l=1; l < num_state  ; l++) {
          if (l == idT) { fields_loc(l,hs+k,hs+j,hs+i) = state(l,k,j,i)*r_r - hy_theta_cells(hs+k); }
          else          { fields_loc(l,hs+k,hs+j,hs+i) = state(l,k,j,i)*r_r; }
        }
        for (int l=0; l < num_tracers; l++) { fields_loc(num_state+1+l,hs+k,hs+j,hs+i) = tracers(l,k,j,i)*r_r; }
        // Perturbation pressure if RSST is used
        if (rsst) { fields_loc(idP,hs+k,hs+j,hs+i) = cs*cs*(state(idR,k,j,i) - hy_dens_cells(hs+k)); }
      });

      // Perform periodic halo exchange in the horizontal, and implement vertical no-slip solid wall boundary conditions
      #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
      yakl::timer_start("dycore_halo_exchange_x");
      #endif
      if (ord > 1) coupler.halo_exchange_x( fields_loc , hs ); // Halo exchange in x-direction
      #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
      yakl::timer_stop("dycore_halo_exchange_x");
      yakl::timer_start("dycore_halo_exchange_y");
      #endif
      if (ord > 1) coupler.halo_exchange_y( fields_loc , hs ); // Halo exchange in y-direction
      #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
      yakl::timer_stop("dycore_halo_exchange_y");
      #endif
      // Set all boundary conditions. istage and icycle are needed for proper halo exchanges between
      //  precursor and forced simulations
      halo_boundary_conditions( coupler , fields_loc , istage , icycle );

      // Storage for cell-edge fluxes in each direction
      yakl::Array<FLOC ****> val_x ("val_x" ,nfields,nz,ny,nx+1);
      yakl::Array<FLOC ****> val_y ("val_y" ,nfields,nz,ny+1,nx);
      yakl::Array<FLOC ****> val_z ("val_z" ,nfields,nz+1,ny,nx);
      yakl::Array<FLOC ****> flux_x("flux_x",nfields,nz,ny,nx+1);
      yakl::Array<FLOC ****> flux_y("flux_y",nfields,nz,ny+1,nx);
      yakl::Array<FLOC ****> flux_z("flux_z",nfields,nz+1,ny,nx);

      // Determine if the bottom and top boundaries are solid walls
      auto wall_z1 = coupler.get_option<std::string>("bc_z1") == "wall_free_slip";
      auto wall_z2 = coupler.get_option<std::string>("bc_z2") == "wall_free_slip";
      auto wall_y1 = coupler.get_option<std::string>("bc_y1") == "wall_free_slip";
      auto wall_y2 = coupler.get_option<std::string>("bc_y2") == "wall_free_slip";

      FLOC hvbeta = 0.01;
      FLOC hvcoef = hvbeta/dt/std::pow(2.0,(double)(ord));
      if ((ord/2)%2==1) hvcoef *= -1;

      FLOC immbeta_amp = 10;
      FLOC immbeta_pow = 1;

      // Interpolate needed quantities at cell edges in the x, y, and z directions
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nfields,nz,ny,nx+1) ,
                                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        SArray<FLOC,ord> s;        // Stencil values
        for (int ii = 0; ii < ord; ii++) { s(ii) = fields_loc(l,hs+k,hs+j,i+ii); }
        SArray<bool,ord> imm;        // Stencil values for immersed boundary
        for (int ii = 0; ii < ord; ii++) { imm(ii) = immersed_prop(hs+k,hs+j,i+ii) > imm_th; }
        if (l==idV || l==idW || l==idP) modify_stencil_immersed_der0( s , imm);
        val_x(l,k,j,i) = TransformMatrices::edge_val(s);
        if (l != idP) {
          FLOC hvcoefloc = hvcoef;
          FLOC imm_dist = static_cast<FLOC>( std::min( immersed_dist(k,j,std::min(nx-1,i)), immersed_dist(k,j,std::max(0,i-1)) ) );
          if (imm_dist <= 12) {
            FLOC mult = 2.*imm_dist*imm_dist*imm_dist/1331. - 39.*imm_dist*imm_dist/1331. + 72.*imm_dist/1331. + 1296./1331.;
            hvcoefloc *= 1 + immbeta_amp*std::pow( std::max(FLOC(0),mult) , immbeta_pow );
          }
          flux_x(l,k,j,i) = hvcoefloc*dx*TransformMatrices::edge_hvder(s);
          if (l != idR) flux_x(l,k,j,i) *= hy_dens_cells(hs+k);
        }
      });
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nfields,nz,ny+1,nx) ,
                                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        SArray<FLOC,ord> s;        // Stencil values
        for (int jj = 0; jj < ord; jj++) { s(jj) = fields_loc(l,hs+k,j+jj,hs+i); }
        SArray<bool,ord> imm;        // Stencil values for immersed boundary
        for (int jj = 0; jj < ord; jj++) { imm(jj) = immersed_prop(hs+k,j+jj,hs+i) > imm_th; }
        if (l==idU || l==idW || l==idP) modify_stencil_immersed_der0( s , imm);
        val_y(l,k,j,i) = TransformMatrices::edge_val(s);
        if (l != idP) {
          FLOC hvcoefloc = hvcoef;
          FLOC imm_dist = static_cast<FLOC>( std::min( immersed_dist(k,std::min(ny-1,j),i), immersed_dist(k,std::max(0,j-1),i) ) );
          if (imm_dist <= 12) {
            FLOC mult = 2.*imm_dist*imm_dist*imm_dist/1331. - 39.*imm_dist*imm_dist/1331. + 72.*imm_dist/1331. + 1296./1331.;
            hvcoefloc *= 1 + immbeta_amp*std::pow( std::max(FLOC(0),mult) , immbeta_pow );
          }
          flux_y(l,k,j,i) = hvcoefloc*dy*TransformMatrices::edge_hvder(s);
          if (l != idR) flux_y(l,k,j,i) *= hy_dens_cells(hs+k);
          if (py==0         && j==0  && wall_y1) flux_y(l,k,j,i) = 0;
          if (py==nproc_y-1 && j==ny && wall_y2) flux_y(l,k,j,i) = 0;
        }
      });
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nfields,nz+1,ny,nx) ,
                                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        SArray<FLOC,ord> s;         // Stencil values
        for (int kk = 0; kk < ord; kk++) { s(kk) = fields_loc(l,k+kk,hs+j,hs+i); }
        SArray<bool,ord> imm;        // Stencil values for immersed boundary
        for (int kk = 0; kk < ord; kk++) { imm(kk) = immersed_prop(k+kk,hs+j,hs+i) > imm_th; }
        if (l==idU || l==idV || l==idP) modify_stencil_immersed_der0( s , imm);
        if (l != idP) {
          FLOC hvcoefloc = hvcoef;
          FLOC imm_dist = static_cast<FLOC>( std::min( immersed_dist(std::min(nz-1,k),j,i), immersed_dist(std::max(0,k-1),j,i) ) );
          if (imm_dist <= 12) {
            FLOC mult = 2.*imm_dist*imm_dist*imm_dist/1331. - 39.*imm_dist*imm_dist/1331. + 72.*imm_dist/1331. + 1296./1331.;
            hvcoefloc *= 1 + immbeta_amp*std::pow( std::max(FLOC(0),mult) , immbeta_pow );
          }
          real dzloc = 0.5*(dz(std::max(0,k-1)) + dz(std::min(nz-1,k)));
          flux_z(l,k,j,i) = hvcoefloc*dzloc*TransformMatrices::edge_hvder(s);
          if (l != idR) flux_z(l,k,j,i) *= hy_dens_edges(k);
          if (k==0  && wall_z1) flux_z(l,k,j,i) = 0;
          if (k==nz && wall_z2) flux_z(l,k,j,i) = 0;
        }
        for (int kk = 0; kk < ord; kk++) { s(kk) *= dz(std::max(0,std::min(nz-1,k-hs+kk))); }
        val_z(l,k,j,i) = TransformMatrices::edge_val(s) / metjac_edges(k);
      });
      // Construct fluxes from interpolated values
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx+1) ,
                                                        KOKKOS_LAMBDA (int k, int j, int i) {
        FLOC r  = val_x(idR,k,j,i) + hy_dens_cells(hs+k);
        FLOC u  = val_x(idU,k,j,i);
        FLOC v  = val_x(idV,k,j,i);
        FLOC w  = val_x(idW,k,j,i);
        FLOC th = val_x(idT,k,j,i) + hy_theta_cells(hs+k);
        FLOC p  = val_x(idP,k,j,i);
        flux_x(idR,k,j,i) += r*u;
        flux_x(idU,k,j,i) += r*u*u+p;
        flux_x(idV,k,j,i) += r*u*v;
        flux_x(idW,k,j,i) += r*u*w;
        flux_x(idT,k,j,i) += r*u*th;
        for (int l=0; l < num_tracers; l++) { flux_x(num_state+1+l,k,j,i) += r*u*val_x(num_state+1+l,k,j,i); }
      });
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny+1,nx) ,
                                                        KOKKOS_LAMBDA (int k, int j, int i) {
        FLOC r  = val_y(idR,k,j,i) + hy_dens_cells(hs+k);
        FLOC u  = val_y(idU,k,j,i);
        FLOC v  = val_y(idV,k,j,i);
        FLOC w  = val_y(idW,k,j,i);
        FLOC th = val_y(idT,k,j,i) + hy_theta_cells(hs+k);
        FLOC p  = val_y(idP,k,j,i);
        if (j==0  && wall_y1) v = 0;
        if (j==ny && wall_y2) v = 0;
        flux_y(idR,k,j,i) += r*v;
        flux_y(idU,k,j,i) += r*v*u;
        flux_y(idV,k,j,i) += r*v*v+p;
        flux_y(idW,k,j,i) += r*v*w;
        flux_y(idT,k,j,i) += r*v*th;
        for (int l=0; l < num_tracers; l++) { flux_y(num_state+1+l,k,j,i) += r*v*val_y(num_state+1+l,k,j,i); }
      });
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny,nx) ,
                                                        KOKKOS_LAMBDA (int k, int j, int i) {
        FLOC r  = val_z(idR,k,j,i) + hy_dens_edges(k);
        FLOC u  = val_z(idU,k,j,i);
        FLOC v  = val_z(idV,k,j,i);
        FLOC w  = val_z(idW,k,j,i);
        FLOC th = val_z(idT,k,j,i) + hy_theta_edges(k);
        FLOC p  = val_z(idP,k,j,i);
        if (k==0  && wall_z1) w = 0;
        if (k==nz && wall_z2) w = 0;
        flux_z(idR,k,j,i) += r*w;
        flux_z(idU,k,j,i) += r*w*u;
        flux_z(idV,k,j,i) += r*w*v;
        flux_z(idW,k,j,i) += r*w*w+p;
        flux_z(idT,k,j,i) += r*w*th;
        for (int l=0; l < num_tracers; l++) { flux_z(num_state+1+l,k,j,i) += r*w*val_z(num_state+1+l,k,j,i); }
      });

      //////////////////////////////////////////////////////////////////////////////////////////////
      // COMPUTE TENDENCIES FROM FLUX DIVERGENCES AND SOURCE TERMS
      //////////////////////////////////////////////////////////////////////////////////////////////

      // Use g*rho*theta'/theta0 for buoyancy if desired or if RSST is being used
      auto buoy_theta = coupler.get_option<bool>("dycore_buoyancy_theta",false) || rsst;
      int mx = std::max(num_state,num_tracers);
      // Compute tendencies as the flux divergence + gravity source term + coriolis
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(mx,nz,ny,nx) ,
                                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          // Compute tendencies as the flux divergence
          state_tend(l,k,j,i) = -( flux_x(l,k,j,i+1) - flux_x(l,k,j,i) ) * r_dx
                                -( flux_y(l,k,j+1,i) - flux_y(l,k,j,i) ) * r_dy
                                -( flux_z(l,k+1,j,i) - flux_z(l,k,j,i) ) / dz(k);
          // Add gravity term to vertical momentum
          if (l == idW && enable_gravity) {
            if (buoy_theta) { // theta-based buoyancy
              FLOC rho    = state(idR,k,j,i);
              FLOC thetap = fields_loc(idT,hs+k,hs+j,hs+i);
              FLOC theta  = thetap + hy_theta_cells(hs+k);
              FLOC pp, p;
              pp = fields_loc(idP,hs+k,hs+j,hs+i);
              p  = pp + hy_pressure_cells(hs+k);
              // state_tend(l,k,j,i) += grav*rho*thetap/hy_theta_cells(hs+k);
              state_tend(l,k,j,i) += grav*rho*(thetap/theta - pp/(gamma*p));
            } else {          // density-based buoyancy
              state_tend(l,k,j,i) += -grav*fields_loc(idR,hs+k,hs+j,hs+i);
            }
          }
          // Add Coriolis terms to horizontal momenta
          if (latitude != 0 && l == idU) state_tend(l,k,j,i) += fcor*state(idV,k,j,i);
          if (latitude != 0 && l == idV) state_tend(l,k,j,i) -= fcor*state(idU,k,j,i);
        }
        if (l < num_tracers) {
          // Compute tendencies as the flux divergence
          tracers_tend(l,k,j,i) = -( flux_x(num_state+1+l,k,j,i+1) - flux_x(num_state+1+l,k,j,i) ) * r_dx
                                  -( flux_y(num_state+1+l,k,j+1,i) - flux_y(num_state+1+l,k,j,i) ) * r_dy
                                  -( flux_z(num_state+1+l,k+1,j,i) - flux_z(num_state+1+l,k,j,i) ) / dz(k);
        }
      });

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("compute_tendencies");
      #endif
    }

} // namespace modules

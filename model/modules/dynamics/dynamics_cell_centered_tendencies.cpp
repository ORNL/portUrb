#include "dynamics_cell_centered.h"

namespace modules {

void EulerCellCentered::compute_tendencies( core::Coupler       & coupler      ,
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
      auto metjac_edges      = dm.get<real const,2>("dycore_metjac_edges"  ); // Vertical metric jacobian at edges
      // Compute matrices to convert polynomial coefficients to 2 GLL points and stencil values to 2 GLL points
      // These matrices will be in column-row format. That performed better than row-column format in performance tests
      real r_dx = 1./dx; // reciprocal of grid spacing
      real r_dy = 1./dy; // reciprocal of grid spacing
      real fcor = 2*7.2921e-5*std::sin(latitude/180*M_PI);  // For coriolis: 2*Omega*sin(latitude)

      real constexpr imm_th = 0.5;

      FLOC cs = coupler.get_option<real>("dycore_cs",350);  // Speed of sound

      int constexpr hsm1 = hs-1; // Halo size minus one

      // The main working array that holds all prognostic variables plus pressure
      yakl::Array<FLOC ****> fields_loc("fields_loc",num_state+num_tracers+1,nz+2*hs,ny+2*hs,nx+2*hs);
      bool rsst = coupler.get_option<bool>("dycore_rsst",false) || (coupler.get_option<real>("dycore_cs",350) != 350);

      // Load state and tracers into working array, dividing by density to get specific quantities, computing pressure,
      //  and subtracting hydrostatic values from density, potential temperature, and pressure
      // If Reduced Speed of Sound Technique (RSST) is being used, set pressure using cs^2 * (rho - rho_hydrostatic)
      //  Otherwise, use true pressure from equation of state
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        // Perturbation pressure if RSST is not used
        if (!rsst) fields_loc(idP,hs+k,hs+j,hs+i) = C0*std::pow(state(idT,k,j,i),gamma) - hy_pressure_cells(hs+k);
        real r_r = 1._fp / state(idR,k,j,i); // Reciprocal of density
        fields_loc(idR,hs+k,hs+j,hs+i) = state(idR,k,j,i);
        // Load in state and tracers as specific quantities
        for (int l=1; l < num_state  ; l++) { fields_loc(            l,hs+k,hs+j,hs+i) = state  (l,k,j,i)*r_r; }
        for (int l=0; l < num_tracers; l++) { fields_loc(num_state+1+l,hs+k,hs+j,hs+i) = tracers(l,k,j,i)*r_r; }
        // Remove hydrostasis from density and potential temperature
        fields_loc(idR,hs+k,hs+j,hs+i) -= hy_dens_cells (hs+k);
        fields_loc(idT,hs+k,hs+j,hs+i) -= hy_theta_cells(hs+k);
        // Perturbation pressure if RSST is used
        if (rsst) { fields_loc(idP,hs+k,hs+j,hs+i) = cs*cs*fields_loc(idR,hs+k,hs+j,hs+i); }
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
      yakl::Array<FLOC ****> flux_x("flux_x",num_state+num_tracers,nz,ny,nx+1);
      yakl::Array<FLOC ****> flux_y("flux_y",num_state+num_tracers,nz,ny+1,nx);
      yakl::Array<FLOC ****> flux_z("flux_z",num_state+num_tracers,nz+1,ny,nx);

      // Storage for cell-edge pressure in each direction
      yakl::Array<FLOC ***> p_x("p_x",nz,ny,nx+1);
      yakl::Array<FLOC ***> p_y("p_y",nz,ny+1,nx);
      yakl::Array<FLOC ***> p_z("p_z",nz+1,ny,nx);

      // Storage for cell-edge momentum in each direction
      yakl::Array<FLOC ***> ru_x("ru_x",nz,ny,nx+1);
      yakl::Array<FLOC ***> rv_y("rv_y",nz,ny+1,nx);
      yakl::Array<FLOC ***> rw_z("rw_z",nz+1,ny,nx);

      // Determine if the bottom and top boundaries are solid walls
      auto wall_z1 = coupler.get_option<std::string>("bc_z1") == "wall_free_slip";
      auto wall_z2 = coupler.get_option<std::string>("bc_z2") == "wall_free_slip";
      auto wall_y1 = coupler.get_option<std::string>("bc_y1") == "wall_free_slip";
      auto wall_y2 = coupler.get_option<std::string>("bc_y2") == "wall_free_slip";
      typedef WenoLimiter<FLOC,ord> Limiter; // Declare the WENO limiter
      auto use_weno = coupler.get_option<bool>("dycore_use_weno",true); // Whether to use WENO limiter
      auto imm_weno = coupler.get_option<bool>("dycore_use_weno_immersed",false); // Whether to use WENO limiter

      /////////////////////////////////////////////////////////////////////////////////////////
      // COMPUTE UPWIND CELL_EDGE PRESSURE AND MOMENTUM (ACOUSTIC UPWINDING)
      /////////////////////////////////////////////////////////////////////////////////////////
      // Reconstruct upwind cell-edge pressure and momentum in x-direction
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx+1) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool,ord> immersed; // Whether a stencil cell is immersed
        SArray<FLOC,ord> s;        // Stencil values

        // Load the stencils for cell immersion and pressure with the cell to the left of the edge as the center cell
        for (int ii = 0; ii < ord; ii++) { immersed(ii) = immersed_prop (hs+k,hs+j,i+ii) > imm_th; }
        for (int ii = 0; ii < ord; ii++) { s       (ii) = fields_loc(idP,hs+k,hs+j,i+ii); }
        // Upon encountering an immersed boundary, set zero derivative boundary conditions from there out in that direction
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_L, dummy;  // To hold left pressure and dummy right pressure
        if (use_weno || (imm_weno && immersed_dist(k,j,std::min(nx-1,i)) <= 6)) { Limiter::value_based(s,dummy,p_L,false,false); }
        else                                                               { p_L = TransformMatrices::sampR(s); }

        // Load the stencil for momentum with the cell to the left of the edge as the center cell
        for (int ii = 0; ii < ord; ii++) { s(ii) = (fields_loc(idR,hs+k,hs+j,i+ii)+hy_dens_cells(hs+k))*
                                                    fields_loc(idU,hs+k,hs+j,i+ii); }
        // Non-WENO reconstruction of momentum at this edge from the left side
        FLOC ru_L = 0;
        if (use_weno || (imm_weno && immersed_dist(k,j,std::min(nx-1,i)) <= 6)) {
          Limiter::value_based(s,dummy,ru_L,immersed(hsm1-1),immersed(hsm1+1));
        }
        else                                                               { ru_L = TransformMatrices::sampR(s); }

        // Load the stencils for cell immersion and pressure with the cell to the right of the edge as the center cell
        for (int ii = 0; ii < ord; ii++) { immersed(ii) = immersed_prop (hs+k,hs+j,i+ii+1) > imm_th; }
        for (int ii = 0; ii < ord; ii++) { s       (ii) = fields_loc(idP,hs+k,hs+j,i+ii+1); }
        // Upon encountering an immersed boundary, set zero derivative boundary conditions from there out in that direction
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_R; // To hold right pressure
        if (use_weno || (imm_weno && immersed_dist(k,j,std::min(nx-1,i)) <= 6)) { Limiter::value_based(s,p_R,dummy,false,false); }
        else                                                               { p_R = TransformMatrices::sampL(s);      }

        // Load the stencil for momentum with the cell to the right of the edge as the center cell
        for (int ii = 0; ii < ord; ii++) { s(ii) = (fields_loc(idR,hs+k,hs+j,i+ii+1)+hy_dens_cells(hs+k))*
                                                    fields_loc(idU,hs+k,hs+j,i+ii+1); }
        // Non-WENO reconstruction of momentum at this edge from the right side
        FLOC ru_R = 0;
        if (use_weno || (imm_weno && immersed_dist(k,j,std::min(nx-1,i)) <= 6)) {
          Limiter::value_based(s,ru_R,dummy,immersed(hsm1-1),immersed(hsm1+1));
        }
        else                                                               { ru_R = TransformMatrices::sampL(s);      }
        // Compute the upwind state of pressure and momentum at this edge
        p_x (k,j,i) = 0.5f*(p_L  + p_R  - cs*(ru_R-ru_L)   );
        ru_x(k,j,i) = 0.5f*(ru_L + ru_R -    (p_R -p_L )/cs);
      });

      // Reconstruct upwind cell-edge pressure and momentum in y-direction
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny+1,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool,ord> immersed; // Whether a stencil cell is immersed
        SArray<FLOC,ord> s;         // Stencil values

        // Load the stencils for cell immersion and pressure with the cell left of the edge as the center cell
        for (int jj = 0; jj < ord; jj++) { immersed(jj) = immersed_prop (hs+k,j+jj,hs+i) > imm_th; }
        for (int jj = 0; jj < ord; jj++) { s       (jj) = fields_loc(idP,hs+k,j+jj,hs+i); }
        // Upon encountering an immersed boundary, set zero derivative boundary conditions from there out in that direction
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_L, dummy; // To hold left pressure and dummy right pressure
        if (use_weno || (imm_weno && immersed_dist(k,std::min(ny-1,j),i) <= 6)) { Limiter::value_based(s,dummy,p_L,false,false); }
        else                                                               { p_L = TransformMatrices::sampR(s);      }

        // Load the stencil for momentum with the cell left of the edge as the center cell
        for (int jj = 0; jj < ord; jj++) { s(jj) = (fields_loc(idR,hs+k,j+jj,hs+i)+hy_dens_cells(hs+k))*
                                                    fields_loc(idV,hs+k,j+jj,hs+i); }
        // Non-WENO reconstruction of momentum at this edge from the left side
        FLOC rv_L;
        if (use_weno || (imm_weno && immersed_dist(k,std::min(ny-1,j),i) <= 6)) {
          Limiter::value_based(s,dummy,rv_L,immersed(hsm1-1),immersed(hsm1+1));
        }
        else                                                               { rv_L = TransformMatrices::sampR(s);      }
        if (wall_y1 && py == 0         && j == 0 ) rv_L = 0; // Impose wall boundary condition
        if (wall_y2 && py == nproc_y-1 && j == ny) rv_L = 0; // Impose wall boundary condition

        // Load the stencils for cell immersion and pressure with the cell right of the edge as the center cell
        for (int jj = 0; jj < ord; jj++) { immersed(jj) = immersed_prop (hs+k,j+jj+1,hs+i) > imm_th; }
        for (int jj = 0; jj < ord; jj++) { s       (jj) = fields_loc(idP,hs+k,j+jj+1,hs+i); }
        // Upon encountering an immersed boundary, set zero derivative boundary conditions from there out in that direction
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_R; // To hold right pressure
        if (use_weno || (imm_weno && immersed_dist(k,std::min(ny-1,j),i) <= 6)) { Limiter::value_based(s,p_R,dummy,false,false); }
        else                                                               { p_R = TransformMatrices::sampL(s);      }

        // Load the stencil for momentum with the cell right of the edge as the center cell
        for (int jj = 0; jj < ord; jj++) { s(jj) = (fields_loc(idR,hs+k,j+jj+1,hs+i)+hy_dens_cells(hs+k))*
                                                    fields_loc(idV,hs+k,j+jj+1,hs+i); }
        // Non-WENO reconstruction of momentum at this edge from the right side
        FLOC rv_R;
        if (use_weno || (imm_weno && immersed_dist(k,std::min(ny-1,j),i) <= 6)) {
          Limiter::value_based(s,rv_R,dummy,immersed(hsm1-1),immersed(hsm1+1));
        }
        else                                                               { rv_R = TransformMatrices::sampL(s);      }
        if (wall_y1 && py == 0         && j == 0 ) rv_R = 0; // Impose wall boundary condition
        if (wall_y2 && py == nproc_y-1 && j == ny) rv_R = 0; // Impose wall boundary condition
        // Compute the upwind state of pressure and momentum at this edge
        p_y (k,j,i) = 0.5f*(p_L  + p_R  - cs*(rv_R-rv_L)   );
        rv_y(k,j,i) = 0.5f*(rv_L + rv_R -    (p_R -p_L )/cs);
        if (wall_y1 && py == 0         && j == 0 ) rv_y(k,j,i) = 0; // Impose wall boundary condition
        if (wall_y2 && py == nproc_y-1 && j == ny) rv_y(k,j,i) = 0; // Impose wall boundary condition
      });

      // Reconstruct upwind cell-edge pressure and momentum in z-direction
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool,ord> immersed; // Whether a stencil cell is immersed
        SArray<FLOC,ord> s;         // Stencil values

        // Load the stencils for cell immersion and pressure with the cell left of the edge as the center cell
        for (int kk = 0; kk < ord; kk++) { immersed(kk) = immersed_prop (k+kk,hs+j,hs+i) > imm_th; }
        for (int kk = 0; kk < ord; kk++) { s       (kk) = fields_loc(idP,k+kk,hs+j,hs+i); }
        for (int kk = 0; kk < ord; kk++) { s       (kk) *= dz(std::max(0,std::min(nz-1,k-hsm1-1+kk)))/dz(std::max(0,k-1)); }
        // Upon encountering an immersed boundary, set zero derivative boundary conditions from there out in that direction
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_L, dummy; // To hold left pressure and dummy right pressure
        if (use_weno || (imm_weno && immersed_dist(std::min(nz-1,k),j,i) <= 6)) { Limiter::value_based(s,dummy,p_L,false,false); }
        else                                                               { p_L = TransformMatrices::sampR(s);      }
        p_L /= metjac_edges(1+k-1,1);

        // Load the stencil for momentum with the cell left of the edge as the center cell
        for (int kk = 0; kk < ord; kk++) { s(kk) = (fields_loc(idR,k+kk,hs+j,hs+i)+hy_dens_cells(k+kk))*
                                                    fields_loc(idW,k+kk,hs+j,hs+i); }
        // Multiply by normalized grid spacing to transform into zeta space
        for (int kk = 0; kk < ord; kk++) { s(kk) *= dz(std::max(0,std::min(nz-1,k-hsm1-1+kk)))/dz(std::max(0,k-1)); }
        // Non-WENO reconstruction of momentum at this edge from the left side
        FLOC rw_L;
        if (use_weno || (imm_weno && immersed_dist(std::min(nz-1,k),j,i) <= 6)) {
          Limiter::value_based(s,dummy,rw_L,immersed(hsm1-1),immersed(hsm1+1));
        }
        else                                                               { rw_L = TransformMatrices::sampR(s);      }
        rw_L /= metjac_edges(1+k-1,1);  // Divide by metric jacobian at this edge to transform to physical space
        if (wall_z1 && k == 0 ) rw_L = 0; // Impose wall boundary condition
        if (wall_z2 && k == nz) rw_L = 0; // Impose wall boundary condition

        // Load the stencils for cell immersion and pressure with the cell right of the edge as the center cell
        for (int kk = 0; kk < ord; kk++) { immersed(kk) = immersed_prop (k+kk+1,hs+j,hs+i) > imm_th; }
        for (int kk = 0; kk < ord; kk++) { s       (kk) = fields_loc(idP,k+kk+1,hs+j,hs+i); }
        // Multiply by normalized grid spacing to transform into zeta space
        for (int kk = 0; kk < ord; kk++) { s       (kk) *= dz(std::max(0,std::min(nz-1,k-hsm1+kk)))/dz(std::min(nz-1,k)); }
        // Upon encountering an immersed boundary, set zero derivative boundary conditions from there out in that direction
        modify_stencil_immersed_der0( s , immersed );
        FLOC p_R; // To hold right pressure
        if (use_weno || (imm_weno && immersed_dist(std::min(nz-1,k),j,i) <= 6)) { Limiter::value_based(s,p_R,dummy,false,false); }
        else                                                               { p_R = TransformMatrices::sampL(s);      }
        p_R /= metjac_edges(1+k,0); // Divide by metric jacobian at this edge to transform to physical space

        // Load the stencil for momentum with the cell right of the edge as the center cell
        for (int kk = 0; kk < ord; kk++) { s(kk) = (fields_loc(idR,k+kk+1,hs+j,hs+i)+hy_dens_cells(k+kk+1))*
                                                    fields_loc(idW,k+kk+1,hs+j,hs+i); }
        // Multiply by normalized grid spacing to transform into zeta space
        for (int kk = 0; kk < ord; kk++) { s(kk) *= dz(std::max(0,std::min(nz-1,k-hsm1+kk)))/dz(std::min(nz-1,k)); }
        // Non-WENO reconstruction of momentum at this edge from the right side
        FLOC rw_R;
        if (use_weno || (imm_weno && immersed_dist(std::min(nz-1,k),j,i) <= 6)) {
          Limiter::value_based(s,rw_R,dummy,immersed(hsm1-1),immersed(hsm1+1));
        }
        else                                                               { rw_R = TransformMatrices::sampL(s);      }
        rw_R /= metjac_edges(1+k,0); // Divide by metric jacobian at this edge to transform to physical space
        if (wall_z1 && k == 0 ) rw_R = 0; // Impose wall boundary condition
        if (wall_z2 && k == nz) rw_R = 0; // Impose wall boundary condition
        // Compute the upwind state of pressure and momentum at this edge
        p_z (k,j,i) = 0.5f*(p_L  + p_R  - cs*(rw_R-rw_L)   );
        rw_z(k,j,i) = 0.5f*(rw_L + rw_R -    (p_R -p_L )/cs);
        if (wall_z1 && k == 0 ) rw_z(k,j,i) = 0; // Impose wall boundary condition
        if (wall_z2 && k == nz) rw_z(k,j,i) = 0; // Impose wall boundary condition
      });

      //////////////////////////////////////////////////////////////////////////////////////////////
      // COMPUTE UPWIND ADVECTED QUANTITIES, AND COMPUTE TOTAL UPWIND FLUXES (ADVECTIVE UPWINDING)
      //////////////////////////////////////////////////////////////////////////////////////////////

      // Pressure will not be included in the advected fields, so accure a MultiField without pressure
      core::MultiField<FLOC,3> advect_fields;
      advect_fields.add_field( fields_loc.slice<3>(idR,0,0,0) ); // Think of these 0 indices as Fortran's (:,:,:)
      advect_fields.add_field( fields_loc.slice<3>(idU,0,0,0) );
      advect_fields.add_field( fields_loc.slice<3>(idV,0,0,0) );
      advect_fields.add_field( fields_loc.slice<3>(idW,0,0,0) );
      advect_fields.add_field( fields_loc.slice<3>(idT,0,0,0) );
      for (int tr=0; tr < num_tracers; tr++) { advect_fields.add_field( fields_loc.slice<3>(num_state+1+tr,0,0,0) ); }
      int num_fields = advect_fields.get_num_fields(); // This will be num_state+num_tracers

      // Reconstruct cell-edge advectively upwind advected quantities and compute total fluxes in x-direction
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx+1) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool,ord> immersed; // Whether a stencil cell is immersed
        FLOC ru = ru_x(k,j,i);        // Acoustically upwinded momentum in x-direction
        int ind = ru > 0 ? 0 : 1;     // Determine index offset based on flow direction
        // Load the cell immersersion stencil based on upwind offset
        for (int ii = 0; ii < ord; ii++) { immersed(ii) = immersed_prop(hs+k,hs+j,i+ii+ind) > imm_th; }
        for (int l=1; l < num_fields; l++) { // Loop over all advected fields except density
          // Gather the stencil values based on upwind offset
          SArray<FLOC,ord> s;
          for (int ii = 0; ii < ord; ii++) { s(ii) = advect_fields(l,hs+k,hs+j,i+ii+ind); }
          bool immL = immersed(hsm1-1);
          bool immR = immersed(hsm1+1);
          // For transverse velocities, modify stencil for immersed boundary zero-derivative condition (free-slip)
          if (l == idV || l == idW) {
            modify_stencil_immersed_der0( s , immersed );
            immL = false;
            immR = false;
          }
          FLOC val_L, val_R;
          if (use_weno || (imm_weno && immersed_dist(k,j,std::min(nx-1,i)) <= 6)) {
            Limiter::value_based(s,val_L,val_R,immL,immR);
          } else {
            val_L = TransformMatrices::sampL(s);
            val_R = TransformMatrices::sampR(s);
          }
          FLOC val = ru > 0 ? val_R : val_L;
          if (l == idT) val += hy_theta_cells(hs+k); // Add hydrostatic potential temperature back in
          flux_x(l,k,j,i) = ru*val;      // Compute total flux vector for advected fields
        }
        flux_x(idR,k,j,i)  = ru;         // Mass flux
        flux_x(idU,k,j,i) += p_x(k,j,i); // Momentum flux includes pressure
      });

      // Reconstruct cell-edge advectively upwind advected quantities and compute total fluxes in y-direction
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny+1,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool,ord> immersed; // Whether a stencil cell is immersed
        FLOC rv = rv_y(k,j,i);        // Acoustically upwinded momentum in y-direction
        int ind = rv > 0 ? 0 : 1;     // Determine index offset based on flow direction
        // Load the cell immersion stencil based on upwind offset
        for (int jj = 0; jj < ord; jj++) { immersed(jj) = immersed_prop(hs+k,j+jj+ind,hs+i) > imm_th; }
        for (int l=1; l < num_fields; l++) { // Loop over all advected fields except density
          // Gather the stencil values based on upwind offset
          SArray<FLOC,ord> s;
          for (int jj = 0; jj < ord; jj++) { s(jj) = advect_fields(l,hs+k,j+jj+ind,hs+i); }
          bool immL = immersed(hsm1-1);
          bool immR = immersed(hsm1+1);
          // For transverse velocities, modify stencil for immersed boundary zero-derivative condition (free-slip)
          if (l == idU || l == idW) {
            modify_stencil_immersed_der0( s , immersed );
            immL = false;
            immR = false;
          }
          FLOC val_L, val_R;
          if (use_weno || (imm_weno && immersed_dist(k,std::min(ny-1,j),i) <= 6)) {
            Limiter::value_based(s,val_L,val_R,immL,immR);
          } else {
            val_L = TransformMatrices::sampL(s);
            val_R = TransformMatrices::sampR(s);
          }
          FLOC val = rv > 0 ? val_R : val_L; // Choose value based on flow direction
          if (l == idT) val += hy_theta_cells(hs+k); // Add hydrostatic potential temperature back in
          flux_y(l,k,j,i) = rv*val;       // Compute total flux vector for advected fields
        }
        flux_y(idR,k,j,i)  = rv;          // Mass flux
        flux_y(idV,k,j,i) += p_y(k,j,i);  // Momentum flux includes pressure
      });

      // Reconstruct cell-edge advectively upwind advected quantities and compute total fluxes in z-direction
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        SArray<bool,ord> immersed; // Whether a stencil cell is immersed
        FLOC rw = rw_z(k,j,i);        // Acoustically upwinded momentum in z-direction
        int ind = rw > 0 ? 0 : 1;     // Determine index offset based on flow direction
        // Load the cell immersion stencil based on upwind offset
        for (int kk = 0; kk < ord; kk++) { immersed(kk) = immersed_prop(k+kk+ind,hs+j,hs+i) > imm_th; }
        for (int l=1; l < num_fields; l++) { // Loop over all advected fields except density
          // Gather the stencil values based on upwind offset
          SArray<FLOC,ord> s;
          for (int kk = 0; kk < ord; kk++) { s(kk) = advect_fields(l,k+kk+ind,hs+j,hs+i); }
          bool immL = immersed(hsm1-1);
          bool immR = immersed(hsm1+1);
          // For transverse velocities, modify stencil for immersed boundary zero-derivative condition (free-slip)
          if (l == idU || l == idV) {
            modify_stencil_immersed_der0( s , immersed );
            immL = false;
            immR = false;
          }
          // Multiply by normalized grid spacing to transform into zeta space
          for (int kk = 0; kk < ord; kk++) { s(kk) *= dz(std::max(0,std::min(nz-1,k-hs+ind+kk)))/
                                                      dz(std::max(0,std::min(nz-1,k-1 +ind   ))); }
          FLOC val_L, val_R;
          if (use_weno || (imm_weno && immersed_dist(std::min(nz-1,k),j,i) <= 6)) {
            Limiter::value_based(s,val_L,val_R,immL,immR);
          } else {
            val_L = TransformMatrices::sampL(s);
            val_R = TransformMatrices::sampR(s);
          }
          FLOC val = rw > 0 ? val_R : val_L; // Choose value based on flow direction
          // Divide by metric jacobian at this edge to transform to physical space
          val /= rw > 0 ? metjac_edges(1+k-1,1) : metjac_edges(1+k,0);
          if (l == idT)  val += hy_theta_edges(k); // Add hydrostatic potential temperature back in
          flux_z(l,k,j,i) = rw*val;       // Compute total flux vector for advected fields
        }
        flux_z(idR,k,j,i)  = rw;          // Mass flux
        flux_z(idW,k,j,i) += p_z(k,j,i);  // Momentum flux includes pressure
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
          tracers_tend(l,k,j,i) = -( flux_x(num_state+l,k,j,i+1) - flux_x(num_state+l,k,j,i) ) * r_dx
                                  -( flux_y(num_state+l,k,j+1,i) - flux_y(num_state+l,k,j,i) ) * r_dy
                                  -( flux_z(num_state+l,k+1,j,i) - flux_z(num_state+l,k,j,i) ) / dz(k);
        }
      });

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("compute_tendencies");
      #endif
    }

} // namespace modules

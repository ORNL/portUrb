#include "dynamics_cell_centered.h"

namespace modules {

void EulerCellCentered::enforce_immersed_boundaries( core::Coupler       & coupler ,
                                      real4d        const & state   ,
                                      real4d        const & tracers ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("enforce_immersed_boundaries");
      #endif
      using yakl::SimpleBounds;
      auto num_tracers     = coupler.get_num_tracers();                    // Total number of tracers
      auto nx              = coupler.get_nx();                             // Number of cells in x-direction (excluding halos)
      auto ny              = coupler.get_ny();                             // Number of cells in y-direction (excluding halos)
      auto nz              = coupler.get_nz();                             // Number of cells in z-direction (excluding halos)
      auto immersed_power  = coupler.get_option<real>("immersed_power",5); // Power for immersed boundary relaxation
      auto &dm             = coupler.get_data_manager_readonly();          // Get data manager for read-only access
      auto hy_dens_cells   = dm.get<real const,1>("hy_dens_cells" );       // Hydrostatic density
      auto hy_theta_cells  = dm.get<real const,1>("hy_theta_cells");       // Hydrostatic potential temperature
      auto immersed_prop   = dm.get<real const,3>("dycore_immersed_proportion_halos"); // Immersed Proportion (with halos)
      auto tracer_positive = dm.get<bool const,1>("tracer_positive");      // Whether each tracer is positive definite

      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real mult = std::pow( immersed_prop(hs+k,hs+j,hs+i) , immersed_power ); // Pre-compute multiplier
        // TODO: Find a way to calculate drag in here
        // Density
        {
          auto &var = state(idR,k,j,i);
          real  target = hy_dens_cells(hs+k);
          var = var + (target - var)*mult;
        }
        // u-momentum
        {
          auto &var = state(idU,k,j,i);
          real  target = 0;
          var = var + (target - var)*mult;
        }
        // v-momentum
        {
          auto &var = state(idV,k,j,i);
          real  target = 0;
          var = var + (target - var)*mult;
        }
        // w-momentum
        {
          auto &var = state(idW,k,j,i);
          real  target = 0;
          var = var + (target - var)*mult;
        }
        // density*theta
        {
          auto &var = state(idT,k,j,i);
          real  target = hy_dens_cells(hs+k)*hy_theta_cells(hs+k);
          var = var + (target - var)*mult;
        }
        // Tracers
        for (int tr=0; tr < num_tracers; tr++) {
          auto &var = tracers(tr,k,j,i);
          real  target = 0;
          var = var + (target - var)*mult;
          if (tracer_positive(tr))  var = std::max( 0._fp , var ); // Keep positive tracers positive
        }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("enforce_immersed_boundaries");
      #endif
    }

void EulerCellCentered::halo_boundary_conditions( core::Coupler & coupler               ,
                                   yakl::Array<FLOC ****> const & fields ,
                                   int istage                            ,
                                   int icycle                            ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("halo_boundary_conditions");
      #endif
      using yakl::SimpleBounds;
      auto nx           = coupler.get_nx(); // Local number of cells in x-direction (not including halos)
      auto ny           = coupler.get_ny(); // Local number of cells in y-direction (not including halos)
      auto nz           = coupler.get_nz(); // Local number of cells in z-direction (not including halos)
      auto px           = coupler.get_px(); // MPI rank in x-direction
      auto py           = coupler.get_py(); // MPI rank in y-direction
      auto bc_x1        = coupler.get_option<std::string>("bc_x1"); // Boundary condition in west   x direction
      auto bc_x2        = coupler.get_option<std::string>("bc_x2"); // Boundary condition in east   x direction
      auto bc_y1        = coupler.get_option<std::string>("bc_y1"); // Boundary condition in south  y direction
      auto bc_y2        = coupler.get_option<std::string>("bc_y2"); // Boundary condition in north  y direction
      auto bc_z1        = coupler.get_option<std::string>("bc_z1"); // Boundary condition in bottom z direction
      auto bc_z2        = coupler.get_option<std::string>("bc_z2"); // Boundary condition in top    z direction
      auto nproc_x      = coupler.get_nproc_x();               // Number of MPI ranks in x-direction
      auto nproc_y      = coupler.get_nproc_y();               // Number of MPI ranks in y-direction
      auto num_tracers  = coupler.get_num_tracers();           // Number of tracer fields
      auto &dm          = coupler.get_data_manager_readonly(); // Get data manager as read-only

      // The halo exchange called before this has already handled periodic BCs
      // If this is a precursor-forced simulation, the ghost cells must have been copied to this coupler object
      //  before this function is called, so here we just need to copy them into the halo cells for inflow boundaries

      if (px == 0) { // If my rank is on the west edge of the domain
        if (bc_x1 == "periodic") { // Already handled in halo_exchange
        } else if (bc_x1 == "open") {
          // Simple zero-gradient extrapolation for open boundary
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,hs) ,
                                                  KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            fields(l,hs+k,hs+j,hs-1-ii) = fields(l,hs+k,hs+j,hs+0);
          });
        } else if (bc_x1 == "precursor") {
          // For inflow boundaries, use precursor data in ghost cells except for pressure field
          // For outflow boundaries, use zero-gradient extrapolation
          auto prec_x1 = dm.get<FLOC const,6>("dycore_ghost_x1");
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,hs) ,
                                                  KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            if (l!=idP) {
              auto u = fields(idU,hs+k,hs+j,hs);
              fields(l,hs+k,hs+j,hs-1-ii) = u > 0 ? prec_x1(icycle,istage,l,k,j,ii) : fields(l,hs+k,hs+j,hs+0);
            } else {
              fields(l,hs+k,hs+j,hs-1-ii) = fields(l,hs+k,hs+j,hs+0);
            }
          });
        } else {
          std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_x1 can only be periodic or open";
          Kokkos::abort("");
        }
      }

      if (px == nproc_x-1) { // If my rank is on the east edge of the domain
        if (bc_x2 == "periodic") { // Already handled in halo_exchange
        } else if (bc_x2 == "open") {
          // Simple zero-gradient extrapolation for open boundary
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,hs) ,
                                            KOKKOS_LAMBDA (int l, int k, int j, int ii) {
                  fields(l,hs+k,hs+j,hs+nx+ii) = fields(l,hs+k,hs+j,hs+nx-1);
          });
        } else if (bc_x2 == "precursor") {
          // For inflow boundaries, use precursor data in ghost cells except for pressure field
          // For outflow boundaries, use zero-gradient extrapolation
          auto prec_x2 = dm.get<FLOC const,6>("dycore_ghost_x2");
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,hs) ,
                                                  KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            if (l!=idP) {
              auto u = fields(idU,hs+k,hs+j,hs+nx-1);
              fields(l,hs+k,hs+j,hs+nx+ii) = u > 0 ? fields(l,hs+k,hs+j,hs+nx-1) : prec_x2(icycle,istage,l,k,j,ii);
            } else {
              fields(l,hs+k,hs+j,hs+nx+ii) = fields(l,hs+k,hs+j,hs+nx-1);
            }
          });
        } else {
          std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_x2 can only be periodic or open";
          Kokkos::abort("");
        }
      }

      if (py == 0) { // If my rank is on the south edge of the domain
        if (bc_y1 == "periodic") { // Already handled in halo_exchange
        } else if (bc_y1 == "open") {
          // Simple zero-gradient extrapolation for open boundary
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                                  KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            fields(l,hs+k,hs-1-jj,hs+i) = fields(l,hs+k,hs+0,hs+i);
          });
        } else if (bc_y1 == "wall_free_slip") {
          // Simple zero-gradient extrapolation for open boundary
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                                  KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            if (l == idV) { fields(l,hs+k,hs-1-jj,hs+i) = 0; }
            else          { fields(l,hs+k,hs-1-jj,hs+i) = fields(l,hs+k,hs+0,hs+i); }
          });
        } else if (bc_y1 == "precursor") {
          // For inflow boundaries, use precursor data in ghost cells except for pressure field
          // For outflow boundaries, use zero-gradient extrapolation
          auto prec_y1 = dm.get<FLOC const,6>("dycore_ghost_y1");
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                                  KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            if (l!=idP) {
              auto v = fields(idV,hs+k,hs,hs+i);
              fields(l,hs+k,hs-1-jj,hs+i) = v > 0 ? prec_y1(icycle,istage,l,k,jj,i) : fields(l,hs+k,hs+0,hs+i);
            } else {
              fields(l,hs+k,hs-1-jj,hs+i) = fields(l,hs+k,hs+0,hs+i);
            }
          });
        } else {
          std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_y1 can only be periodic, wall_free_slip, or open";
          Kokkos::abort("");
        }
      }

      if (py == nproc_y-1) { // If my rank is on the north edge of the domain
        if (bc_y2 == "periodic") { // Already handled in halo_exchange
        } else if (bc_y2 == "open") {
          // Simple zero-gradient extrapolation for open boundary
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                                  KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            fields(l,hs+k,hs+ny+jj,hs+i) = fields(l,hs+k,hs+ny-1,hs+i);
          });
        } else if (bc_y2 == "wall_free_slip") {
          // Simple zero-gradient extrapolation for open boundary
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                                  KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            if (l == idV) { fields(l,hs+k,hs+ny+jj,hs+i) = 0; }
            else          { fields(l,hs+k,hs+ny+jj,hs+i) = fields(l,hs+k,hs+ny-1,hs+i); }
          });
        } else if (bc_y2 == "precursor") {
          // For inflow boundaries, use precursor data in ghost cells except for pressure field
          // For outflow boundaries, use zero-gradient extrapolation
          auto prec_y2 = dm.get<FLOC const,6>("dycore_ghost_y2");
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                                  KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            if (l!=idP) {
              auto v = fields(idV,hs+k,hs+ny-1,hs+i);
              fields(l,hs+k,hs+ny+jj,hs+i) = v > 0 ? fields(l,hs+k,hs+ny-1,hs+i) : prec_y2(icycle,istage,l,k,jj,i);
            } else {
              fields(l,hs+k,hs+ny+jj,hs+i) = fields(l,hs+k,hs+ny-1,hs+i);
            }
          });
        } else {
          std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_y2 can only be periodic, wall_free_slip, or open";
          Kokkos::abort("");
        }
      }

      if (bc_z1 == "wall_free_slip") {
        // Free-slip wall boundary condition at bottom boundary
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,hs,ny,nx) ,
                                                KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          if (l == idW) {
            fields(l,kk,hs+j,hs+i) = 0;
          } else {
            fields(l,hs-1-kk,hs+j,hs+i) = fields(l,hs+0,hs+j,hs+i);
          }
        });
      } else if (bc_z1 == "periodic") {
        // Periodic boundary condition at bottom boundary
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,hs,ny,nx) ,
                                                KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          fields(l,kk,hs+j,hs+i) = fields(l,nz+kk,hs+j,hs+i);
        });
      } else {
        std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_z1 can only be periodic or wall_free_slip";
        Kokkos::abort("");
      }

      if (bc_z2 == "wall_free_slip") {
        // Free-slip wall boundary condition at top boundary
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,hs,ny,nx) ,
                                                KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          if (l == idW) {
            fields(l,hs+nz+kk,hs+j,hs+i) = 0;
          } else {
            fields(l,hs+nz+kk,hs+j,hs+i) = fields(l,hs+nz-1,hs+j,hs+i);
          }
        });
      } else if (bc_z2 == "open") {
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,hs,ny,nx) ,
                                                KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          fields(l,hs+nz+kk,hs+j,hs+i) = fields(l,hs+nz-1,hs+j,hs+i);
        });
      } else if (bc_z2 == "periodic") {
        // Periodic boundary condition at top boundary
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,hs,ny,nx) ,
                                                KOKKOS_LAMBDA (int l, int kk, int j, int i) {
          fields(l,hs+nz+kk,hs+j,hs+i) = fields(l,hs+kk,hs+j,hs+i);
        });
      } else {
        std::cout << __FILE__ << ":" << __LINE__ << ": ERROR: bc_z2 can only be periodic or wall_free_slip";
        Kokkos::abort("");
      }

      // If this is a precursor simualtion forcing another coupler object, then store the ghost cells
      //  into the coupler object for use by the other coupler object
      if (coupler.get_option<bool>("dycore_is_precursor",false)) {
        if (px == 0) {
          auto ghost_x1 = coupler.get_data_manager_readwrite().get<FLOC,6>("dycore_ghost_x1");
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,hs) ,
                                                  KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            ghost_x1(icycle,istage,l,k,j,ii) = fields(l,hs+k,hs+j,hs-1-ii);
          });
        }
        if (px == nproc_x-1) {
          auto ghost_x2 = coupler.get_data_manager_readwrite().get<FLOC,6>("dycore_ghost_x2");
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,ny,hs) ,
                                                  KOKKOS_LAMBDA (int l, int k, int j, int ii) {
            ghost_x2(icycle,istage,l,k,j,ii) = fields(l,hs+k,hs+j,hs+nx+ii);
          });
        }
        if (py == 0) {
          auto ghost_y1 = coupler.get_data_manager_readwrite().get<FLOC,6>("dycore_ghost_y1");
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                                  KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            ghost_y1(icycle,istage,l,k,jj,i) = fields(l,hs+k,hs-1-jj,hs+i);
          });
        }
        if (py == nproc_y-1) {
          auto ghost_y2 = coupler.get_data_manager_readwrite().get<FLOC,6>("dycore_ghost_y2");
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers+1,nz,hs,nx) ,
                                                  KOKKOS_LAMBDA (int l, int k, int jj, int i) {
            ghost_y2(icycle,istage,l,k,jj,i) = fields(l,hs+k,hs+ny+jj,hs+i);
          });
        }
      }

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("halo_boundary_conditions");
      #endif
    }

real2d EulerCellCentered::compute_average_ghost_column( core::Coupler & coupler ) {
      using yakl::SimpleBounds;
      auto nx_glob           = coupler.get_nx_glob();  // Global number of cells in x-direction
      auto ny_glob           = coupler.get_ny_glob();  // Global number of cells in y-direction
      auto nx                = coupler.get_nx();       // Local number of cells in x-direction (not including halos)
      auto ny                = coupler.get_ny();       // Local number of cells in y-direction (not including halos)
      auto nz                = coupler.get_nz();       // Number of cells in z-direction (not including halos)
      auto C0                = coupler.get_option<real>("C0"     );  // pressure = C0*pow(rho*theta,gamma)
      auto gamma             = coupler.get_option<real>("gamma_d");  // cp_dry / cv_dry (about 1.4)
      auto cs                = coupler.get_option<real>("dycore_cs",350); // Speed of sound
      auto num_tracers       = coupler.get_num_tracers();   // Number of tracer fields
      // Hydrostatic pressure, density, and potential temperature over cells with halos
      auto hy_pressure_cells = coupler.get_data_manager_readonly().get<real const,1>("hy_pressure_cells");
      auto hy_dens_cells     = coupler.get_data_manager_readonly().get<real const,1>("hy_dens_cells");
      auto hy_theta_cells    = coupler.get_data_manager_readonly().get<real const,1>("hy_theta_cells");
      real4d state  ("state"  ,num_state  ,nz,ny,nx); // State variables
      real4d tracers("tracers",num_tracers,nz,ny,nx); // Tracer variables
      convert_coupler_to_dynamics( coupler , state , tracers ); // Convert coupler data to dynamics format
      real4d fields_loc("fields_loc",num_state+num_tracers+1,nz+2*hs,ny+2*hs,nx+2*hs); // Local fields with halos
      bool rsst = coupler.get_option<bool>("dycore_rsst",false) || (coupler.get_option<real>("dycore_cs",350) != 350);
      // Replicate the working array computation from compute_tendencies to get fields_loc populated
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        // Compute pressure perturbation if not using RSST
        if (!rsst) fields_loc(idP,hs+k,hs+j,hs+i) = C0*std::pow(state(idT,k,j,i),gamma) - hy_pressure_cells(hs+k);
        real r_r = 1._fp / state(idR,k,j,i); // Reciprocal of density
        fields_loc(idR,hs+k,hs+j,hs+i) = state(idR,k,j,i);
        // Store velocity, potential temperature, and tracers as specific quantities
        for (int l=1; l < num_state  ; l++) { fields_loc(            l,hs+k,hs+j,hs+i) = state  (l,k,j,i)*r_r; }
        for (int l=0; l < num_tracers; l++) { fields_loc(num_state+1+l,hs+k,hs+j,hs+i) = tracers(l,k,j,i)*r_r; }
        // Subtract hydrostatic contributions from density and potential temperature
        fields_loc(idR,hs+k,hs+j,hs+i) -= hy_dens_cells (hs+k);
        fields_loc(idT,hs+k,hs+j,hs+i) -= hy_theta_cells(hs+k);
        // If using RSST, compute perturbed pressure from perturbation density
        if (rsst) { fields_loc(idP,hs+k,hs+j,hs+i) = cs*cs*fields_loc(idR,hs+k,hs+j,hs+i); }
      });
      real2d ghost_col("ghost_col",num_state+num_tracers+1,nz); // Average column to return
      real r_nx_ny = 1./(nx_glob*ny_glob); // Reciprocal of global horizontal cell count
      // Compute average column for fields_loc
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(num_state+num_tracers+1,nz) , KOKKOS_LAMBDA (int l, int k) {
        ghost_col(l,k) = 0;
        for (int j=0; j < ny; j++) {
          for (int i=0; i < nx; i++) {
            ghost_col(l,k) += fields_loc(l,hs+k,hs+j,hs+i)*r_nx_ny;
          }
        }
      });
      // Sum across all MPI ranks to get global average column
      coupler.get_parallel_comm().all_reduce( ghost_col , MPI_SUM , "" ).deep_copy_to(ghost_col);
      Kokkos::fence();
      return ghost_col;
    }

void EulerCellCentered::copy_precursor_ghost_cells( core::Coupler & coupler_prec , core::Coupler & coupler_main ) {
      auto const prec_max_cycles = coupler_prec.get_option<int>("dycore_max_cycles");
      auto const main_max_cycles = coupler_main.get_option<int>("dycore_max_cycles");
      if (prec_max_cycles != main_max_cycles) {
        if (prec_max_cycles < main_max_cycles) {
          ensure_dycore_max_cycles(coupler_prec,main_max_cycles-1);
        } else {
          ensure_dycore_max_cycles(coupler_main,prec_max_cycles-1);
        }
      }
      int  px          = coupler_main.get_px();       // MPI rank in x-direction
      int  py          = coupler_main.get_py();       // MPI rank in y-direction
      int  npx         = coupler_main.get_nproc_x();  // Number of MPI ranks in x-direction
      int  npy         = coupler_main.get_nproc_y();  // Number of MPI ranks in y-direction
      auto &dm_prec    = coupler_prec.get_data_manager_readonly (); // Get precursor data manager as read-only
      auto &dm_main    = coupler_main.get_data_manager_readwrite(); // Get main data manager as read-write
      if (px == 0    ) dm_prec.get<FLOC const,6>("dycore_ghost_x1").deep_copy_to(dm_main.get<FLOC,6>("dycore_ghost_x1"));
      if (px == npx-1) dm_prec.get<FLOC const,6>("dycore_ghost_x2").deep_copy_to(dm_main.get<FLOC,6>("dycore_ghost_x2"));
      if (py == 0    ) dm_prec.get<FLOC const,6>("dycore_ghost_y1").deep_copy_to(dm_main.get<FLOC,6>("dycore_ghost_y1"));
      if (py == npy-1) dm_prec.get<FLOC const,6>("dycore_ghost_y2").deep_copy_to(dm_main.get<FLOC,6>("dycore_ghost_y2"));
    }

void EulerCellCentered::copy_column_to_precursor_ghost_cells( core::Coupler & coupler , real2d const & col ) {
      using yakl::SimpleBounds;
      int  px          = coupler.get_px();       // MPI rank in x-direction
      int  py          = coupler.get_py();       // MPI rank in y-direction
      int  npx         = coupler.get_nproc_x();  // Number of MPI ranks in x-direction
      int  npy         = coupler.get_nproc_y();  // Number of MPI ranks in y-direction
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();
      auto num_stages  = coupler.get_option<int>("dycore_num_stages");
      auto max_cycles  = coupler.get_option<int>("dycore_max_cycles");
      if (px == 0) {
        auto ghost_x1 = coupler.get_data_manager_readwrite().get<FLOC,6>("dycore_ghost_x1");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<6>(max_cycles,num_stages,num_state+num_tracers+1,nz,ny,hs) ,
                                                KOKKOS_LAMBDA (int icycle, int istage, int l, int k, int j, int ii) {
          ghost_x1(icycle,istage,l,k,j,ii) = col(l,k);
        });
      }
      if (px == npx-1) {
        auto ghost_x2 = coupler.get_data_manager_readwrite().get<FLOC,6>("dycore_ghost_x2");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<6>(max_cycles,num_stages,num_state+num_tracers+1,nz,ny,hs) ,
                                                KOKKOS_LAMBDA (int icycle, int istage, int l, int k, int j, int ii) {
          ghost_x2(icycle,istage,l,k,j,ii) = col(l,k);
        });
      }
      if (py == 0) {
        auto ghost_y1 = coupler.get_data_manager_readwrite().get<FLOC,6>("dycore_ghost_y1");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<6>(max_cycles,num_stages,num_state+num_tracers+1,nz,hs,nx) ,
                                                KOKKOS_LAMBDA (int icycle, int istage, int l, int k, int jj, int i) {
          ghost_y1(icycle,istage,l,k,jj,i) = col(l,k);
        });
      }
      if (py == npy-1) {
        auto ghost_y2 = coupler.get_data_manager_readwrite().get<FLOC,6>("dycore_ghost_y2");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<6>(max_cycles,num_stages,num_state+num_tracers+1,nz,hs,nx) ,
                                                KOKKOS_LAMBDA (int icycle, int istage, int l, int k, int jj, int i) {
          ghost_y2(icycle,istage,l,k,jj,i) = col(l,k);
        });
      }
    }

void EulerCellCentered::create_immersed_proportion_halos(core::Coupler &coupler) const {
      using yakl::SimpleBounds;
      auto nz     = coupler.get_nz();
      auto ny     = coupler.get_ny();
      auto nx     = coupler.get_nx();
      auto &dm    = coupler.get_data_manager_readwrite();
      auto wall_B = coupler.get_option<std::string>("bc_z1") == "wall_free_slip";
      auto wall_T = coupler.get_option<std::string>("bc_z2") == "wall_free_slip";

      if (! dm.entry_exists("dycore_immersed_proportion_halos")) {
        dm.register_and_allocate<real>("dycore_immersed_proportion_halos",{nz+2*hs,ny+2*hs,nx+2*hs});
      }

      auto immersed_prop       = dm.get<real const,3>("immersed_proportion");
      auto immersed_prop_halos = dm.get<real,3>("dycore_immersed_proportion_halos");
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int k, int j, int i) {
        immersed_prop_halos(hs+k,hs+j,hs+i) = immersed_prop(k,j,i);
      });

      // Exchanging x before y propagates the physical-domain values into the horizontal corner halos.
      core::MultiField<real,3> fields_halos;
      fields_halos.add_field( immersed_prop_halos );
      coupler.halo_exchange( fields_halos , hs );

      // Vertical boundaries span the full horizontal allocation so their corner and edge halos are also initialized.
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) ,
                                              KOKKOS_LAMBDA (int kk, int j, int i) {
        immersed_prop_halos(      kk,j,i) = wall_B ? 1 : 0;
        immersed_prop_halos(hs+nz+kk,j,i) = wall_T ? 1 : 0;
      });

      // Compute the Chebyshev distance to the nearest immersed cell within 12 cells.
      if (! dm.entry_exists("dycore_immersed_distance")) {
        dm.register_and_allocate<real>("dycore_immersed_distance",{nz,ny,nx});
        coupler.register_output_variable<real>("dycore_immersed_distance",core::Coupler::DIMS_3D);
      }
      int constexpr hsnew = 12;
      auto immersed_prop_copy = immersed_prop.createDeviceCopy();
      core::MultiField<real,3> fields;
      fields.add_field( immersed_prop_copy );
      auto fields_halos_larger = coupler.create_and_exchange_halos( fields , hsnew );
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hsnew,ny+2*hsnew,nx+2*hsnew) ,
                                              KOKKOS_LAMBDA (int kk, int j, int i) {
        fields_halos_larger(0,         kk,j,i) = wall_B ? 1 : 0;
        fields_halos_larger(0,hsnew+nz+kk,j,i) = wall_T ? 1 : 0;
      });
      auto immersed_distance = dm.get<real,3>("dycore_immersed_distance");
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int k, int j, int i) {
        real distance = 1000;
        for (int kk=-hsnew; kk <= hsnew; kk++) {
          for (int jj=-hsnew; jj <= hsnew; jj++) {
            for (int ii=-hsnew; ii <= hsnew; ii++) {
              if (fields_halos_larger(0,hsnew+k+kk,hsnew+j+jj,hsnew+i+ii) > 0) {
                int distance_loc = std::max(std::abs(kk),std::max(std::abs(jj),std::abs(ii)));
                distance = std::min(distance,static_cast<real>(std::max(1,distance_loc)));
              }
            }
          }
        }
        immersed_distance(k,j,i) = distance;
      });
    }

} // namespace modules

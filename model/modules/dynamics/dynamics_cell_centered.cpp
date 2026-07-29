#include "dynamics_cell_centered.h"

namespace modules {

void EulerCellCentered::ensure_dycore_max_cycles(core::Coupler &coupler, int icycle) const {
      auto max_cycles = coupler.get_option<int>("dycore_max_cycles");
      if (icycle < max_cycles) return;

      using yakl::SimpleBounds;
      auto &dm         = coupler.get_data_manager_readwrite();
      auto new_cycles  = icycle+1;
      auto num_stages  = coupler.get_option<int>("dycore_num_stages");
      auto num_tracers = coupler.get_num_tracers();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();

      auto resize = [&](std::string const & name, std::vector<int> dims) {
        if (! dm.entry_exists(name)) return;
        auto old_arr    = dm.get_collapsed<FLOC const>(name);
        auto old_size   = old_arr.extent(0);
        auto cycle_size = old_size / max_cycles;
        yakl::Array<FLOC *,yakl::DeviceSpace> saved(name+"_saved",old_size);
        old_arr.deep_copy_to(saved);
        Kokkos::fence();
        dm.unregister_and_deallocate(name);
        dims.at(0) = new_cycles;
        dm.register_and_allocate<FLOC>(name,dims);
        auto new_arr = dm.get_collapsed<FLOC>(name);
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<1>(new_arr.extent(0)) , KOKKOS_LAMBDA (int i) {
          new_arr(i) = saved(i < old_size ? i : i % cycle_size);
        });
        Kokkos::fence();
      };

      resize("dycore_ghost_x1",{max_cycles,num_stages,num_state+num_tracers+1,nz,ny,hs});
      resize("dycore_ghost_x2",{max_cycles,num_stages,num_state+num_tracers+1,nz,ny,hs});
      resize("dycore_ghost_y1",{max_cycles,num_stages,num_state+num_tracers+1,nz,hs,nx});
      resize("dycore_ghost_y2",{max_cycles,num_stages,num_state+num_tracers+1,nz,hs,nx});
      coupler.set_option("dycore_max_cycles",new_cycles);
    }

std::tuple<real,real> EulerCellCentered::compute_mass( core::Coupler & coupler , real4d const & state ) const {
      using yakl::SimpleBounds;
      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      auto dx = coupler.get_dx(); // grid spacing in x-direction
      auto dy = coupler.get_dy(); // grid spacing in y-direction
      auto dz = coupler.get_dz(); // 1D array of vertical cell grid spacing
      real3d r("r",nz,ny,nx); // Array for local mass
      real3d t("t",nz,ny,nx); // Array for local virtual potential temperature mass
      // Accumulate local mass
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j , int i) {
        r(k,j,i) = state(idR,k,j,i)*dx*dy*dz(k);
        t(k,j,i) = state(idT,k,j,i)*dx*dy*dz(k);
      });
      // Reduce the global mass across all MPI ranks
      real rmass = coupler.get_parallel_comm().all_reduce( yakl::intrinsics::sum(r) , MPI_SUM );
      real tmass = coupler.get_parallel_comm().all_reduce( yakl::intrinsics::sum(t) , MPI_SUM );
      return std::make_tuple(rmass,tmass);
    }

real EulerCellCentered::compute_time_step( core::Coupler const &coupler ) const {
      using yakl::intrinsics::minval;
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      auto cs = coupler.get_option<real>( "dycore_cs" , 350 ); // Speed of sound in m/s
      real maxwave = cs + coupler.get_option<real>( "dycore_max_wind" , 100 ); // Max wave speed in m/s (cs+wind)
      real cfl = coupler.get_option<real>("cfl",0.60);         // CFL number
      // Return the maximum stable time step based on the minimum cell size in the domain, max wave speed, and CFL number
      return cfl * std::min( std::min( dx , dy ) , minval(dz) ) / maxwave;
    }

void EulerCellCentered::time_step(core::Coupler &coupler, real dt_phys) const {
      if (dt_phys <= 0) {
        endrun("ERROR: dynamics time_step requires dt_phys > 0");
      }
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step");
      #endif
      using yakl::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers(); // Total number of tracers
      auto nx          = coupler.get_nx(); // Number of cells in x-direction (excluding halos)
      auto ny          = coupler.get_ny(); // Number of cells in y-direction (excluding halos)
      auto nz          = coupler.get_nz(); // Number of cells in z-direction (excluding halos)
      auto &dm         = coupler.get_data_manager_readwrite(); // Get data manager for read/write access
      real4d state  ("state"  ,num_state  ,nz,ny,nx); // State array for the dynamical core
      real4d tracers("tracers",num_tracers,nz,ny,nx); // Tracer array for the dynamical core
      convert_coupler_to_dynamics( coupler , state , tracers ); // Convert coupler data to dynamical core format
      real dt_dyn = compute_time_step( coupler );        // Compute maximum stable dynamical core time step
      int ncycles = (int) std::ceil( dt_phys / dt_dyn ); // Determine number of sub-cycles needed for stability
      dt_dyn = dt_phys / ncycles;                        // Make sure individual sub-step time steps are equal

      // auto mass1 = compute_mass( coupler , state );
      // Get the desired time stepper from the coupler options and perform the sub-cycled time stepping
      // Must pass the icycle number to the time stepper for proper ghost cell exchanges with precursor simulations
      auto time_stepper = coupler.get_option<std::string>("dycore_time_stepper","ssprk3");
      for (int icycle = 0; icycle < ncycles; icycle++) {
        ensure_dycore_max_cycles(coupler,icycle);
        if      (time_stepper == "linrk3") { time_step_rk3   (coupler,state,tracers,dt_dyn,icycle); }
        else if (time_stepper == "linrk4") { time_step_rk4   (coupler,state,tracers,dt_dyn,icycle); }
        else if (time_stepper == "ssprk3") { time_step_ssprk3(coupler,state,tracers,dt_dyn,icycle); }
        else                               { throw std::runtime_error(std::string("ERROR: Unknown time stepper: ") + time_stepper); }
      }
      // auto mass2 = compute_mass( coupler , state );
      // if (coupler.is_mainproc()) std::cout << "Mass change: "
      //                                      << (std::get<0>(mass2)-std::get<0>(mass1))/std::get<0>(mass1) << " , "
      //                                      << (std::get<1>(mass2)-std::get<1>(mass1))/std::get<1>(mass1) << std::endl;
      // Convert the dynamical core state and tracer arrays back to the coupler format
      convert_dynamics_to_coupler( coupler , state , tracers );
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step");
      #endif
    }

void EulerCellCentered::time_step_rk3( core::Coupler & coupler ,
                        real4d const  & state   ,
                        real4d const  & tracers ,
                        real            dt_dyn  ,
                        int             icycle  ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step_rk_3_3");
      #endif
      using yakl::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers(); // Total number of tracers
      auto nx          = coupler.get_nx();          // Number of cells in x-direction (excluding halos)
      auto ny          = coupler.get_ny();          // Number of cells in y-direction (excluding halos)
      auto nz          = coupler.get_nz();          // Number of cells in z-direction (excluding halos)
      auto &dm         = coupler.get_data_manager_readonly(); // Get data manager for read-only access
      auto tracer_positive = dm.get<bool const,1>("tracer_positive"); // Whether each tracer is positive definite
      // RK3 requires temporary arrays to hold intermediate state and tracers arrays
      real4d state_tmp   ("state_tmp"   ,num_state  ,nz,ny,nx);
      real4d tracers_tmp ("tracers_tmp" ,num_tracers,nz,ny,nx);
      // To hold tendencies (time derivatives of state and tracers)
      real4d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx);
      real4d tracers_tend("tracers_tend",num_tracers,nz,ny,nx);

      // Set immersed boundaries in state and tracers to hydrostasis at rest
      enforce_immersed_boundaries( coupler , state , tracers );

      // Stage 1
      // Compute time derivatives of the state and tracers using a time steyp of dt/3
      compute_tendencies(coupler,state    ,state_tend,tracers    ,tracers_tend,dt_dyn/3,0,icycle);
      // Apply tendencies for the first stage for state and tracers
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = state  (l,k,j,i) + dt_dyn/3 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = tracers(l,k,j,i) + dt_dyn/3 * tracers_tend(l,k,j,i);
        }
      });
      enforce_immersed_boundaries( coupler , state_tmp , tracers_tmp );

      // Stage 2
      // Compute time derivatives of the state and tracers using a time step of dt/2
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/2,1,icycle);
      // Apply tendencies for the second stage for state and tracers
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = state  (l,k,j,i) + dt_dyn/2 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = tracers(l,k,j,i) + dt_dyn/2 * tracers_tend(l,k,j,i);
        }
      });
      enforce_immersed_boundaries( coupler , state_tmp , tracers_tmp );

      // Stage 3
      // Compute time derivatives of the state and tracers using a time step of dt/1
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/1,2,icycle);
      // Apply tendencies for the third stage for state and tracers
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state      (l,k,j,i) = state  (l,k,j,i) + dt_dyn/1 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers    (l,k,j,i) = tracers(l,k,j,i) + dt_dyn/1 * tracers_tend(l,k,j,i);
          // Correct tracer values to be positive definite if needed
          if (tracer_positive(l))  tracers(l,k,j,i) = std::max( 0._fp , tracers(l,k,j,i) );
        }
      });

      // Set immersed boundaries in state and tracers to hydrostasis at rest
      enforce_immersed_boundaries( coupler , state , tracers );

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step_rk_3_3");
      #endif
    }

void EulerCellCentered::time_step_rk4( core::Coupler & coupler ,
                        real4d const  & state   ,
                        real4d const  & tracers ,
                        real            dt_dyn  ,
                        int             icycle  ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step_rk_3_3");
      #endif
      using yakl::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers();           // Total number of tracers
      auto nx          = coupler.get_nx();                    // Number of cells in x-direction (excluding halos)
      auto ny          = coupler.get_ny();                    // Number of cells in y-direction (excluding halos)
      auto nz          = coupler.get_nz();                    // Number of cells in z-direction (excluding halos)
      auto &dm         = coupler.get_data_manager_readonly(); // Get data manager for read-only access
      auto tracer_positive = dm.get<bool const,1>("tracer_positive"); // Whether each tracer is positive definite
      // SSPRK3 requires temporary arrays to hold intermediate state and tracers arrays
      real4d state_tmp   ("state_tmp"   ,num_state  ,nz,ny,nx);
      real4d tracers_tmp ("tracers_tmp" ,num_tracers,nz,ny,nx);
      // To hold tendencies (time derivatives of state and tracers)
      real4d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx);
      real4d tracers_tend("tracers_tend",num_tracers,nz,ny,nx);

      // Set immersed boundaries in state and tracers to hydrostasis at rest
      enforce_immersed_boundaries( coupler , state , tracers );

      // Stage 1
      // Compute time derivatives of the state and tracers using a time steyp of dt/4
      compute_tendencies(coupler,state    ,state_tend,tracers    ,tracers_tend,dt_dyn/4,0,icycle);
      // Apply tendencies for the first stage for state and tracers
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = state  (l,k,j,i) + dt_dyn/4 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = tracers(l,k,j,i) + dt_dyn/4 * tracers_tend(l,k,j,i);
        }
      });
      enforce_immersed_boundaries( coupler , state_tmp , tracers_tmp );

      // Stage 2
      // Compute time derivatives of the state and tracers using a time step of dt/3
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/3,1,icycle);
      // Apply tendencies for the second stage for state and tracers
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = state  (l,k,j,i) + dt_dyn/3 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = tracers(l,k,j,i) + dt_dyn/3 * tracers_tend(l,k,j,i);
        }
      });
      enforce_immersed_boundaries( coupler , state_tmp , tracers_tmp );

      // Stage 3
      // Compute time derivatives of the state and tracers using a time step of dt/2
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/2,2,icycle);
      // Apply tendencies for the third stage for state and tracers
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = state  (l,k,j,i) + dt_dyn/2 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = tracers(l,k,j,i) + dt_dyn/2 * tracers_tend(l,k,j,i);
        }
      });
      enforce_immersed_boundaries( coupler , state_tmp , tracers_tmp );

      // Stage 4
      // Compute time derivatives of the state and tracers using a time step of dt/1
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/1,3,icycle);
      // Apply tendencies for the fourth stage for state and tracers
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state      (l,k,j,i) = state  (l,k,j,i) + dt_dyn/1 * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers    (l,k,j,i) = tracers(l,k,j,i) + dt_dyn/1 * tracers_tend(l,k,j,i);
          // Correct tracer values to be positive definite if needed
          if (tracer_positive(l))  tracers(l,k,j,i) = std::max( 0._fp , tracers(l,k,j,i) );
        }
      });

      // Set immersed boundaries in state and tracers to hydrostasis at rest
      enforce_immersed_boundaries( coupler , state , tracers );

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step_rk_3_3");
      #endif
    }

void EulerCellCentered::time_step_ssprk3( core::Coupler & coupler ,
                           real4d const  & state   ,
                           real4d const  & tracers ,
                           real            dt_dyn  ,
                           int             icycle  ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step_rk_3_3");
      #endif
      using yakl::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers();            // Total number of tracers
      auto nx          = coupler.get_nx();                     // Number of cells in x-direction (excluding halos)
      auto ny          = coupler.get_ny();                     // Number of cells in y-direction (excluding halos)
      auto nz          = coupler.get_nz();                     // Number of cells in z-direction (excluding halos)
      auto &dm         = coupler.get_data_manager_readonly();  // Get data manager for read-only access
      auto tracer_positive = dm.get<bool const,1>("tracer_positive"); // Whether each tracer is positive definite
      // SSPRK3 requires temporary arrays to hold intermediate state and tracers arrays
      real4d state_tmp   ("state_tmp"   ,num_state  ,nz,ny,nx);
      real4d tracers_tmp ("tracers_tmp" ,num_tracers,nz,ny,nx);
      // To hold tendencies (time derivatives of state and tracers)
      real4d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx);
      real4d tracers_tend("tracers_tend",num_tracers,nz,ny,nx);

      // Set immersed boundaries in state and tracers to hydrostasis at rest
      enforce_immersed_boundaries( coupler , state , tracers );

      // Stage 1
      // Compute time derivatives of the state and tracers using a time steyp of dt
      compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn,0,icycle);
      // Apply tendencies for the first stage for state and tracers
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = state  (l,k,j,i) + dt_dyn * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = tracers(l,k,j,i) + dt_dyn * tracers_tend(l,k,j,i);
        }
      });
      enforce_immersed_boundaries( coupler , state_tmp , tracers_tmp );

      // Stage 2
      // Compute time derivatives of the state and tracers using a time step of dt/4
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/4.,1,icycle);
      // Apply tendencies for the second stage for state and tracers
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,k,j,i) = (3._fp/4._fp) * state      (l,k,j,i) +
                                 (1._fp/4._fp) * state_tmp  (l,k,j,i) +
                                 (1._fp/4._fp) * dt_dyn * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,k,j,i) = (3._fp/4._fp) * tracers    (l,k,j,i) +
                                 (1._fp/4._fp) * tracers_tmp(l,k,j,i) +
                                 (1._fp/4._fp) * dt_dyn * tracers_tend(l,k,j,i);
        }
      });
      enforce_immersed_boundaries( coupler , state_tmp , tracers_tmp );

      // Stage 3
      // Compute time derivatives of the state and tracers using a time step of dt*2/3
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn*2./3.,2,icycle);
      // Apply tendencies for the third stage for state and tracers
      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state  (l,k,j,i) = (1._fp/3._fp) * state      (l,k,j,i) +
                             (2._fp/3._fp) * state_tmp  (l,k,j,i) +
                             (2._fp/3._fp) * dt_dyn * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers(l,k,j,i) = (1._fp/3._fp) * tracers    (l,k,j,i) +
                             (2._fp/3._fp) * tracers_tmp(l,k,j,i) +
                             (2._fp/3._fp) * dt_dyn * tracers_tend(l,k,j,i);
          // Ensure positive tracers stay positive
          if (tracer_positive(l))  tracers(l,k,j,i) = std::max( 0._fp , tracers(l,k,j,i) );
        }
      });

      // Set immersed boundaries in state and tracers to hydrostasis at rest
      enforce_immersed_boundaries( coupler , state , tracers );

      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step_rk_3_3");
      #endif
    }

} // namespace modules


#pragma once

#include "main_header.h"
#include "coupler.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "acoustic_projection.h"
#include <memory>

namespace modules {


  struct Dynamics_Euler_Stratified {
    mutable std::shared_ptr<ConnectivityGalerkinMultigrid<float>> anelastic_multigrid;
    mutable std::shared_ptr<GeometricMultigrid<float>> anelastic_geometric_multigrid;
    // Order of accuracy (numerical convergence rate for smooth flows) for the dynamical core
    #ifndef PORTURB_ORD
      int static constexpr ord = 8;
    #else
      int static constexpr ord = PORTURB_ORD;
    #endif
    static_assert(ord == 2 || ord == 4 || ord == 6 || ord == 8 || ord == 10,
                  "dynamics_rk_fast requires ord to be 2, 4, 6, 8, or 10");
    int static constexpr hs  = ord/2; // Number of halo cells ("hs" == "halo size")
    int static constexpr num_state = 5;   // Number of state variables
    // IDs for the variables in the state vector
    int  static constexpr idR = 0;  // Density
    int  static constexpr idU = 1;  // u-momentum
    int  static constexpr idV = 2;  // v-momentum
    int  static constexpr idW = 3;  // w-momentum
    int  static constexpr idT = 4;  // Density * potential temperature

    typedef float FLOC; // Use single precision locally


    // Increase precursor ghost-cell storage when the current sub-cycle exceeds its capacity
    void ensure_dycore_max_cycles(core::Coupler &coupler, int icycle) const {
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



    // Compute total mass of density and total mass of virtual potential temperature in the domain
    //  for verification purposes
    // coupler : Coupler instance
    // state   : State array from the dynamical core
    // Returns a tuple of summed density mass and virtual potential temperature mass
    std::tuple<real,real> compute_mass( core::Coupler & coupler , real4d const & state ) const {
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



    AcousticProjectionConfig acoustic_projection_config(core::Coupler const &coupler) const {
      AcousticProjectionConfig config;
      config.diagnostics = coupler.get_option<bool>("dycore_anelastic_projection_diagnostics",false);
      config.check_linearity = coupler.get_option<bool>("dycore_anelastic_check_linearity",false);
      config.check_cg_compatibility =
          coupler.get_option<bool>("dycore_anelastic_check_cg_compatibility",true);
      config.use_conjugate_gradient =
          coupler.get_option<bool>("dycore_anelastic_use_cg",config.check_cg_compatibility);
      config.time_linear_solver = coupler.get_option<bool>("dycore_anelastic_time_linear_solver",false);
      config.screening = coupler.get_option<bool>("dycore_anelastic_screening",false);
      config.sound_speed = coupler.get_option<real>("dycore_cs",350);
      config.momentum_hyperviscosity = coupler.get_option<real>("dycore_anelastic_projection_beta",0.1);
      config.pressure_hyperviscosity =
          coupler.get_option<real>("dycore_anelastic_projection_pressure_beta",0);
      bool const use_jacobi = coupler.get_option<bool>("dycore_anelastic_use_jacobi_preconditioner",true);
      config.preconditioner = coupler.get_option<std::string>("dycore_anelastic_preconditioner",
                                                              use_jacobi ? "Jacobi" : "none");
      config.gmres_restart = coupler.get_option<int>("dycore_anelastic_gmres_restart",30);
      config.linear_solver_max_iterations = coupler.get_option<int>("dycore_anelastic_gmres_max_iters",200);
      config.linear_solver_relative_tolerance =
          coupler.get_option<real>("dycore_anelastic_gmres_rel_tol",1.e-6);
      config.linear_solver_absolute_tolerance = coupler.get_option<real>("dycore_anelastic_gmres_abs_tol",0);
      config.linear_solver_verbose = coupler.get_option<bool>("dycore_anelastic_gmres_verbose",false);
      config.gmres_reorthogonalize =
          coupler.get_option<bool>("dycore_anelastic_gmres_reorthogonalize",true);
      config.schwarz_tile_nx = coupler.get_option<int>("dycore_anelastic_schwarz_tile_nx",8);
      config.schwarz_tile_ny = coupler.get_option<int>("dycore_anelastic_schwarz_tile_ny",8);
      config.schwarz_overlap = coupler.get_option<int>("dycore_anelastic_schwarz_overlap",2);
      config.schwarz_chebyshev_degree =
          coupler.get_option<int>("dycore_anelastic_schwarz_chebyshev_degree",8);
      config.schwarz_chebyshev_lambda_min =
          coupler.get_option<real>("dycore_anelastic_schwarz_chebyshev_lambda_min",0.02);
      config.schwarz_chebyshev_lambda_max =
          coupler.get_option<real>("dycore_anelastic_schwarz_chebyshev_lambda_max",2);
      config.multigrid = anelastic_multigrid;
      config.multigrid_vcycles = coupler.get_option<int>("dycore_anelastic_multigrid_vcycles",1);
      config.multigrid_pre_smooth = coupler.get_option<int>("dycore_anelastic_multigrid_pre_smooth",1);
      config.multigrid_post_smooth = coupler.get_option<int>("dycore_anelastic_multigrid_post_smooth",1);
      config.multigrid_aggregate_size = coupler.get_option<int>("dycore_anelastic_multigrid_aggregate_size",8);
      config.multigrid_max_levels = coupler.get_option<int>("dycore_anelastic_multigrid_max_levels",24);
      config.multigrid_coarse_max_dofs =
          coupler.get_option<int>("dycore_anelastic_multigrid_coarse_max_dofs",256);
      config.multigrid_coarse_smooth =
          coupler.get_option<int>("dycore_anelastic_multigrid_coarse_smooth",16);
      config.multigrid_jacobi_weight =
          coupler.get_option<real>("dycore_anelastic_multigrid_jacobi_weight",2._fp/3._fp);
      config.geometric_multigrid = anelastic_geometric_multigrid;
      config.geometric_multigrid_vcycles =
          coupler.get_option<int>("dycore_anelastic_geometric_multigrid_vcycles",1);
      config.geometric_multigrid_pre_smooth =
          coupler.get_option<int>("dycore_anelastic_geometric_multigrid_pre_smooth",2);
      config.geometric_multigrid_post_smooth =
          coupler.get_option<int>("dycore_anelastic_geometric_multigrid_post_smooth",2);
      config.geometric_multigrid_coarse_smooth =
          coupler.get_option<int>("dycore_anelastic_geometric_multigrid_coarse_smooth",24);
      config.geometric_multigrid_max_levels =
          coupler.get_option<int>("dycore_anelastic_geometric_multigrid_max_levels",20);
      config.geometric_multigrid_coarse_cells =
          coupler.get_option<int>("dycore_anelastic_geometric_multigrid_coarse_cells",32768);
      config.geometric_multigrid_min_cells_per_rank =
          coupler.get_option<int>("dycore_anelastic_geometric_multigrid_min_cells_per_rank",131072);
      config.geometric_multigrid_jacobi_weight =
          coupler.get_option<real>("dycore_anelastic_geometric_multigrid_jacobi_weight",2._fp/3._fp);
      return config;
    }


    // Compute the time step from the advective CFL. The anelastic projection removes acoustic propagation.
    real compute_time_step( core::Coupler const &coupler ) const {
      using yakl::intrinsics::minval;
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      real maxwave = coupler.get_option<real>( "dycore_max_wind" , 100 ); // Configured advective scale in m/s
      if (maxwave <= 0) endrun("ERROR: dycore_max_wind must be positive for the anelastic dycore");
      real cfl = coupler.get_option<real>("cfl",0.60);         // CFL number
      // Return the maximum stable time step based on the minimum cell size in the domain, max wave speed, and CFL number
      return cfl * std::min( std::min( dx , dy ) , minval(dz) ) / maxwave;
    }



    // Perform a time step
    // coupler : Coupler instance
    // dt_phys : Desired physical time step to advance the solution (may be sub-cycled internally for stability)
    // Advances the solution in the coupler's data manager state and tracer arrays by dt_phys
    // Uses sub-cycling with stable dynamical core time steps as needed
    void time_step(core::Coupler &coupler, real dt_phys) const {
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
      real4d state("state",num_state,nz,ny,nx); // State array for the dynamical core
      real4d tracers;
      if (num_tracers > 0) tracers = real4d("tracers",num_tracers,nz,ny,nx);
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
        else { throw std::runtime_error(std::string("ERROR: Unknown time stepper: ") + time_stepper); }
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



    // Max CFL: 0.72
    // This CFL is smaller than normal because dimensions are split within each RK stage
    // This is the linearly third-order, non-linearly second-order quasi-Runge-Kutta method used by WRF
    // coupler : Coupler instance
    // state   : State array from the dynamical core
    // tracers : Tracer array from the dynamical core
    // dt_dyn  : Dynamical core time step to use for this sub-step
    // icycle  : Current sub-cycle index (from 0 to ncycles-1)
    // Advances the solution in state and tracers by dt_dyn using the linRK3 method
    // The icycle number is used for proper ghost cell exchanges between precursor and forced simulations
    void time_step_rk3( core::Coupler & coupler ,
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
      real4d state_tmp("state_tmp",num_state,nz,ny,nx);
      real4d tracers_tmp;
      // To hold tendencies (time derivatives of state and tracers)
      real4d state_tend("state_tend",num_state,nz,ny,nx);
      real4d tracers_tend;
      if (num_tracers > 0) {
        tracers_tmp  = real4d("tracers_tmp" ,num_tracers,nz,ny,nx);
        tracers_tend = real4d("tracers_tend",num_tracers,nz,ny,nx);
      }

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



    // Max CFL: 0.99
    // This CFL is smaller than normal because dimensions are split within each RK stage
    // This is the linearly fourth-order, non-linearly second-order quasi-Runge-Kutta method used by WRF
    // coupler : Coupler instance
    // state   : State array from the dynamical core
    // tracers : Tracer array from the dynamical core
    // dt_dyn  : Dynamical core time step to use for this sub-step
    // icycle  : Current sub-cycle index (from 0 to ncycles-1)
    // Advances the solution in state and tracers by dt_dyn using the linRK3 method
    // The icycle number is used for proper ghost cell exchanges between precursor and forced simulations
    void time_step_rk4( core::Coupler & coupler ,
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
      real4d state_tmp("state_tmp",num_state,nz,ny,nx);
      real4d tracers_tmp;
      // To hold tendencies (time derivatives of state and tracers)
      real4d state_tend("state_tend",num_state,nz,ny,nx);
      real4d tracers_tend;
      if (num_tracers > 0) {
        tracers_tmp  = real4d("tracers_tmp" ,num_tracers,nz,ny,nx);
        tracers_tend = real4d("tracers_tend",num_tracers,nz,ny,nx);
      }

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



    // Max CFL: 0.72
    // This CFL is smaller than normal because dimensions are split within each RK stage
    // This is the optimal three-stage third-order Strong Stability Preserving Runge-Kutta method
    // coupler : Coupler instance
    // state   : State array from the dynamical core
    // tracers : Tracer array from the dynamical core
    // dt_dyn  : Dynamical core time step to use for this sub-step
    // icycle  : Current sub-cycle index (from 0 to ncycles-1)
    // Advances the solution in state and tracers by dt_dyn using the linRK3 method
    // The icycle number is used for proper ghost cell exchanges between precursor and forced simulations
    void time_step_ssprk3( core::Coupler & coupler ,
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
      real4d state_tmp("state_tmp",num_state,nz,ny,nx);
      real4d tracers_tmp;
      // To hold tendencies (time derivatives of state and tracers)
      real4d state_tend("state_tend",num_state,nz,ny,nx);
      real4d tracers_tend;
      if (num_tracers > 0) {
        tracers_tmp  = real4d("tracers_tmp" ,num_tracers,nz,ny,nx);
        tracers_tend = real4d("tracers_tend",num_tracers,nz,ny,nx);
      }

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



    // Enforce immersed boundary conditions by relaxing variables toward hydrostasis at rest
    // coupler : Coupler instance
    // state   : State array from the dynamical core
    // tracers : Tracer array from the dynamical core
    void enforce_immersed_boundaries( core::Coupler       & coupler ,
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
      auto immersed_thresh = coupler.get_option<real>("immersed_threshold",0.5); // Threshold for immersed cells
      auto &dm             = coupler.get_data_manager_readonly();          // Get data manager for read-only access
      auto hy_dens_cells   = dm.get<real const,1>("hy_dens_cells" );       // Hydrostatic density
      auto hy_theta_cells  = dm.get<real const,1>("hy_theta_cells");       // Hydrostatic potential temperature
      auto immersed_prop   = dm.get<real const,3>("dycore_immersed_proportion_halos"); // Immersed Proportion (with halos)
      auto tracer_positive = dm.get<bool const,1>("tracer_positive");      // Whether each tracer is positive definite

      yakl::autotune::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real mult = immersed_prop(hs+k,hs+j,hs+i) > immersed_thresh ? 1 : 0;
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






    int static constexpr idP = 5; // Index of pressure in total array of num_state+1+num_tracers in compute_tendencies


    // One anelastic RK forcing evaluation: pressureless conservative FE advection, theta buoyancy, and projection.
    void compute_tendencies( core::Coupler       & coupler      ,
                             real4d        const & state        ,
                             real4d        const & state_tend   ,
                             real4d        const & tracers      ,
                             real4d        const & tracers_tend ,
                             real                  dt           ,
                             int                   istage       ,
                             int                   icycle       ) const {
      using yakl::SimpleBounds;
      if (dt <= 0) endrun("ERROR: anelastic tendency forcing interval must be positive");

      auto const nx          = coupler.get_nx();
      auto const ny          = coupler.get_ny();
      auto const nz          = coupler.get_nz();
      auto const dx          = coupler.get_dx();
      auto const dy          = coupler.get_dy();
      auto const dz          = coupler.get_dz();
      auto const num_tracers = coupler.get_num_tracers();
      bool diagnostics = false;
      if constexpr (yakl::kokkos_debug) {
        diagnostics = coupler.get_option<bool>("dycore_anelastic_projection_diagnostics",false);
      }
      auto const grav        = coupler.get_option<real>("grav");
      auto const gravity     = coupler.get_option<bool>("enable_gravity",true);
      auto const imm_th      = coupler.get_option<real>("immersed_threshold",0.5);
      auto       &dm         = coupler.get_data_manager_readwrite();
      auto const immersed    = dm.get<real const,3>("dycore_immersed_proportion_halos");
      auto const imm_dist    = dm.get<real const,3>("dycore_immersed_distance");
      auto const rho_h       = dm.get<real const,1>("hy_dens_cells");
      auto const rho_h_edge  = dm.get<real const,1>("hy_dens_edges");
      auto const theta_h     = dm.get<real const,1>("hy_theta_cells");
      auto const theta_h_edge = dm.get<real const,1>("hy_theta_edges");
      auto const metjac_edge = dm.get<real const,1>("dycore_metjac_edges");
      int const nfields = num_state + 1 + num_tracers;
      FLOC const dt_loc = static_cast<FLOC>(dt);
      FLOC const dx_loc = static_cast<FLOC>(dx);
      FLOC const dy_loc = static_cast<FLOC>(dy);
      FLOC const r_dx_loc = FLOC(1)/dx_loc;
      FLOC const r_dy_loc = FLOC(1)/dy_loc;

      // The temporary conservative system always starts from fixed hydrostatic density.
      yakl::Array<FLOC ****> fields_loc("anelastic_adv_fields",nfields,nz+2*hs,ny+2*hs,nx+2*hs);
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        FLOC const r_rho_h = FLOC(1)/static_cast<FLOC>(rho_h(hs+k));
        fields_loc(idR,hs+k,hs+j,hs+i) = 0;
        fields_loc(idU,hs+k,hs+j,hs+i) = static_cast<FLOC>(state(idU,k,j,i))*r_rho_h;
        fields_loc(idV,hs+k,hs+j,hs+i) = static_cast<FLOC>(state(idV,k,j,i))*r_rho_h;
        fields_loc(idW,hs+k,hs+j,hs+i) = static_cast<FLOC>(state(idW,k,j,i))*r_rho_h;
        fields_loc(idT,hs+k,hs+j,hs+i) = static_cast<FLOC>(state(idT,k,j,i))*r_rho_h -
                                               static_cast<FLOC>(theta_h(hs+k));
        fields_loc(idP,hs+k,hs+j,hs+i) = 0;
        for (int tr = 0; tr < num_tracers; tr++) {
          fields_loc(num_state+1+tr,hs+k,hs+j,hs+i) = static_cast<FLOC>(tracers(tr,k,j,i))*r_rho_h;
        }
      });
      if (ord > 1) coupler.halo_exchange_x(fields_loc,hs);
      if (ord > 1) coupler.halo_exchange_y(fields_loc,hs);
      halo_boundary_conditions(coupler,fields_loc,istage,icycle);

      yakl::Array<FLOC ****> val_x ("anelastic_adv_val_x" ,nfields,nz,ny,nx+1);
      yakl::Array<FLOC ****> val_y ("anelastic_adv_val_y" ,nfields,nz,ny+1,nx);
      yakl::Array<FLOC ****> val_z ("anelastic_adv_val_z" ,nfields,nz+1,ny,nx);
      yakl::Array<FLOC ****> flux_x("anelastic_adv_flux_x",nfields,nz,ny,nx+1);
      yakl::Array<FLOC ****> flux_y("anelastic_adv_flux_y",nfields,nz,ny+1,nx);
      yakl::Array<FLOC ****> flux_z("anelastic_adv_flux_z",nfields,nz+1,ny,nx);
      bool const wall_x1 = coupler.get_option<std::string>("bc_x1") == "wall_free_slip";
      bool const wall_x2 = coupler.get_option<std::string>("bc_x2") == "wall_free_slip";
      bool const wall_y1 = coupler.get_option<std::string>("bc_y1") == "wall_free_slip";
      bool const wall_y2 = coupler.get_option<std::string>("bc_y2") == "wall_free_slip";
      bool const wall_z1 = coupler.get_option<std::string>("bc_z1") == "wall_free_slip";
      bool const wall_z2 = coupler.get_option<std::string>("bc_z2") == "wall_free_slip";
      auto const px      = coupler.get_px();
      auto const py      = coupler.get_py();
      auto const nproc_x = coupler.get_nproc_x();
      auto const nproc_y = coupler.get_nproc_y();
      FLOC constexpr hvbeta = 0.01;
      FLOC hvcoef = hvbeta/dt_loc/std::pow(FLOC(2),ord);
      if ((ord/2)%2 == 1) hvcoef *= -1;
      FLOC constexpr immbeta_amp = 20;

      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<4>(nfields,nz,ny,nx+1),
                         KOKKOS_LAMBDA (int l, int k, int j, int i) {
        SArray<FLOC,ord> s;
        SArray<bool,ord> imm;
        for (int ii = 0; ii < ord; ii++) {
          s(ii)   = fields_loc(l,hs+k,hs+j,i+ii);
          imm(ii) = immersed(hs+k,hs+j,i+ii) > imm_th;
        }
        if (l != idU) modify_stencil_immersed_der0(s,imm);
        val_x(l,k,j,i) = TransformMatrices::edge_val(s);
        if (l != idP) {
          FLOC coef = hvcoef;
          FLOC const dist = std::min(static_cast<FLOC>(imm_dist(k,j,std::min(nx-1,i))),
                                     static_cast<FLOC>(imm_dist(k,j,std::max(0,i-1))));
          if (dist <= 12) {
            FLOC const mult = 2*dist*dist*dist/1331 - 39*dist*dist/1331 + 72*dist/1331 + 1296._fp/1331._fp;
            coef *= 1 + immbeta_amp*std::max(FLOC(0),mult);
          }
          flux_x(l,k,j,i) = coef*dx_loc*TransformMatrices::edge_hvder(s);
          if (l != idR) flux_x(l,k,j,i) *= static_cast<FLOC>(rho_h(hs+k));
        }
      });
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<4>(nfields,nz,ny+1,nx),
                         KOKKOS_LAMBDA (int l, int k, int j, int i) {
        SArray<FLOC,ord> s;
        SArray<bool,ord> imm;
        for (int jj = 0; jj < ord; jj++) {
          s(jj)   = fields_loc(l,hs+k,j+jj,hs+i);
          imm(jj) = immersed(hs+k,j+jj,hs+i) > imm_th;
        }
        if (l != idV) modify_stencil_immersed_der0(s,imm);
        val_y(l,k,j,i) = TransformMatrices::edge_val(s);
        if (l != idP) {
          FLOC coef = hvcoef;
          FLOC const dist = std::min(static_cast<FLOC>(imm_dist(k,std::min(ny-1,j),i)),
                                     static_cast<FLOC>(imm_dist(k,std::max(0,j-1),i)));
          if (dist <= 12) {
            FLOC const mult = 2*dist*dist*dist/1331 - 39*dist*dist/1331 + 72*dist/1331 + 1296._fp/1331._fp;
            coef *= 1 + immbeta_amp*std::max(FLOC(0),mult);
          }
          flux_y(l,k,j,i) = coef*dy_loc*TransformMatrices::edge_hvder(s);
          if (l != idR) flux_y(l,k,j,i) *= static_cast<FLOC>(rho_h(hs+k));
          if ((py == 0 && j == 0 && wall_y1) || (py == nproc_y-1 && j == ny && wall_y2)) flux_y(l,k,j,i) = 0;
        }
      });
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<4>(nfields,nz+1,ny,nx),
                         KOKKOS_LAMBDA (int l, int k, int j, int i) {
        SArray<FLOC,ord> s;
        SArray<bool,ord> imm;
        for (int kk = 0; kk < ord; kk++) {
          s(kk)   = fields_loc(l,k+kk,hs+j,hs+i);
          imm(kk) = immersed(k+kk,hs+j,hs+i) > imm_th;
        }
        if (l != idW) modify_stencil_immersed_der0(s,imm);
        if (l != idP) {
          FLOC coef = hvcoef;
          FLOC const dist = std::min(static_cast<FLOC>(imm_dist(std::min(nz-1,k),j,i)),
                                     static_cast<FLOC>(imm_dist(std::max(0,k-1),j,i)));
          if (dist <= 12) {
            FLOC const mult = 2*dist*dist*dist/1331 - 39*dist*dist/1331 + 72*dist/1331 + 1296._fp/1331._fp;
            coef *= 1 + immbeta_amp*std::max(FLOC(0),mult);
          }
          FLOC const dzloc = FLOC(0.5)*(static_cast<FLOC>(dz(std::max(0,k-1))) +
                                        static_cast<FLOC>(dz(std::min(nz-1,k))));
          flux_z(l,k,j,i) = coef*dzloc*TransformMatrices::edge_hvder(s);
          if (l != idR) flux_z(l,k,j,i) *= static_cast<FLOC>(rho_h_edge(k));
          if ((k == 0 && wall_z1) || (k == nz && wall_z2)) flux_z(l,k,j,i) = 0;
        }
        for (int kk = 0; kk < ord; kk++) {
          s(kk) *= static_cast<FLOC>(dz(std::max(0,std::min(nz-1,k-hs+kk))));
        }
        val_z(l,k,j,i) = TransformMatrices::edge_val(s)/static_cast<FLOC>(metjac_edge(k));
      });

      // Construct each direction's mass flux once, then reuse it for every transported specific quantity.
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx+1), KOKKOS_LAMBDA (int k, int j, int i) {
        FLOC u = val_x(idU,k,j,i);
        if (immersed(hs+k,hs+j,hs+i-1) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) u = 0;
        FLOC const mass_flux = (val_x(idR,k,j,i)+static_cast<FLOC>(rho_h(hs+k)))*u;
        flux_x(idR,k,j,i) += mass_flux;
        flux_x(idU,k,j,i) += mass_flux*u;
        flux_x(idV,k,j,i) += mass_flux*val_x(idV,k,j,i);
        flux_x(idW,k,j,i) += mass_flux*val_x(idW,k,j,i);
        flux_x(idT,k,j,i) += mass_flux*(val_x(idT,k,j,i)+static_cast<FLOC>(theta_h(hs+k)));
        for (int tr = 0; tr < num_tracers; tr++) {
          flux_x(num_state+1+tr,k,j,i) += mass_flux*val_x(num_state+1+tr,k,j,i);
        }
      });
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny+1,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        FLOC v = val_y(idV,k,j,i);
        if (immersed(hs+k,hs+j-1,hs+i) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) v = 0;
        if ((py == 0 && j == 0 && wall_y1) || (py == nproc_y-1 && j == ny && wall_y2)) v = 0;
        FLOC const mass_flux = (val_y(idR,k,j,i)+static_cast<FLOC>(rho_h(hs+k)))*v;
        flux_y(idR,k,j,i) += mass_flux;
        flux_y(idU,k,j,i) += mass_flux*val_y(idU,k,j,i);
        flux_y(idV,k,j,i) += mass_flux*v;
        flux_y(idW,k,j,i) += mass_flux*val_y(idW,k,j,i);
        flux_y(idT,k,j,i) += mass_flux*(val_y(idT,k,j,i)+static_cast<FLOC>(theta_h(hs+k)));
        for (int tr = 0; tr < num_tracers; tr++) {
          flux_y(num_state+1+tr,k,j,i) += mass_flux*val_y(num_state+1+tr,k,j,i);
        }
      });
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz+1,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        FLOC w = val_z(idW,k,j,i);
        if (immersed(hs+k-1,hs+j,hs+i) > imm_th || immersed(hs+k,hs+j,hs+i) > imm_th) w = 0;
        if ((k == 0 && wall_z1) || (k == nz && wall_z2)) w = 0;
        FLOC const mass_flux = (val_z(idR,k,j,i)+static_cast<FLOC>(rho_h_edge(k)))*w;
        flux_z(idR,k,j,i) += mass_flux;
        flux_z(idU,k,j,i) += mass_flux*val_z(idU,k,j,i);
        flux_z(idV,k,j,i) += mass_flux*val_z(idV,k,j,i);
        flux_z(idW,k,j,i) += mass_flux*w;
        flux_z(idT,k,j,i) += mass_flux*(val_z(idT,k,j,i)+static_cast<FLOC>(theta_h_edge(k)));
        for (int tr = 0; tr < num_tracers; tr++) {
          flux_z(num_state+1+tr,k,j,i) += mass_flux*val_z(num_state+1+tr,k,j,i);
        }
      });

      yakl::Array<FLOC ****> adv("anelastic_adv_state",num_state,nz,ny,nx);
      yakl::Array<FLOC ****> qstar;
      if (num_tracers > 0) qstar = yakl::Array<FLOC ****>("anelastic_qstar",num_tracers,nz,ny,nx);
      yakl::Array<FLOC ****> star ("anelastic_star",4,nz,ny,nx);
      int3d invalid("anelastic_invalid_density",nz,ny,nx);
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        auto divergence = [&] (int l) {
          return (flux_x(l,k,j,i+1)-flux_x(l,k,j,i))*r_dx_loc +
                 (flux_y(l,k,j+1,i)-flux_y(l,k,j,i))*r_dy_loc +
                 (flux_z(l,k+1,j,i)-flux_z(l,k,j,i))/static_cast<FLOC>(dz(k));
        };
        adv(idR,k,j,i) = static_cast<FLOC>(rho_h(hs+k))       - dt_loc*divergence(idR);
        adv(idU,k,j,i) = static_cast<FLOC>(state(idU,k,j,i)) - dt_loc*divergence(idU);
        adv(idV,k,j,i) = static_cast<FLOC>(state(idV,k,j,i)) - dt_loc*divergence(idV);
        adv(idW,k,j,i) = static_cast<FLOC>(state(idW,k,j,i)) - dt_loc*divergence(idW);
        adv(idT,k,j,i) = static_cast<FLOC>(state(idT,k,j,i)) - dt_loc*divergence(idT);
        invalid(k,j,i) = !std::isfinite(adv(idR,k,j,i)) || adv(idR,k,j,i) <= std::numeric_limits<FLOC>::min();
        FLOC const r_rho = invalid(k,j,i) ? 0 : FLOC(1)/adv(idR,k,j,i);
        star(0,k,j,i) = adv(idU,k,j,i)*r_rho;
        star(1,k,j,i) = adv(idV,k,j,i)*r_rho;
        star(2,k,j,i) = adv(idW,k,j,i)*r_rho;
        star(3,k,j,i) = adv(idT,k,j,i)*r_rho;
        if (gravity) {
          FLOC const theta_h_loc = static_cast<FLOC>(theta_h(hs+k));
          star(2,k,j,i) += dt_loc*static_cast<FLOC>(grav)*(star(3,k,j,i)-theta_h_loc)/theta_h_loc;
        }
        for (int tr = 0; tr < num_tracers; tr++) {
          FLOC const rhoq = static_cast<FLOC>(tracers(tr,k,j,i))-dt_loc*divergence(num_state+1+tr);
          qstar(tr,k,j,i) = rhoq*r_rho;
        }
      });
      int const invalid_global = coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(invalid),MPI_SUM);
      if (invalid_global != 0) endrun("ERROR: pressureless anelastic advection produced invalid temporary density");
      if constexpr (yakl::kokkos_debug) {
        if (diagnostics) {
          yakl::Array<FLOC ***> density_change("anelastic_temporary_density_change",nz,ny,nx);
          yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
            density_change(k,j,i) = std::abs(adv(idR,k,j,i)-static_cast<FLOC>(rho_h(hs+k)));
          });
          FLOC const max_change =
              coupler.get_parallel_comm().all_reduce(yakl::intrinsics::maxval(density_change),MPI_MAX);
          coupler.set_option<real>("dycore_anelastic_last_temporary_density_change",static_cast<real>(max_change));
        }
      }

      float4d momentum_in ("anelastic_projection_momentum_in" ,3,nz,ny,nx);
      float4d momentum_out("anelastic_projection_momentum_out",3,nz,ny,nx);
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        FLOC const rho = static_cast<FLOC>(rho_h(hs+k));
        momentum_in(0,k,j,i) = rho*star(0,k,j,i);
        momentum_in(1,k,j,i) = rho*star(1,k,j,i);
        momentum_in(2,k,j,i) = rho*star(2,k,j,i);
      });
      auto pressure = dm.get<real,3>("anelastic_pressure_pert");
      acoustic_projection<ord>(coupler,momentum_in,momentum_out,pressure,dt,acoustic_projection_config(coupler));
      yakl::parallel_for(YAKL_AUTO_LABEL(), SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        real const rho = rho_h(hs+k);
        state_tend(idR,k,j,i) = 0;
        state_tend(idU,k,j,i) = (momentum_out(0,k,j,i)-state(idU,k,j,i))/dt;
        state_tend(idV,k,j,i) = (momentum_out(1,k,j,i)-state(idV,k,j,i))/dt;
        state_tend(idW,k,j,i) = (momentum_out(2,k,j,i)-state(idW,k,j,i))/dt;
        state_tend(idT,k,j,i) = rho*(star(3,k,j,i)-state(idT,k,j,i)/rho)/dt;
        for (int tr = 0; tr < num_tracers; tr++) {
          tracers_tend(tr,k,j,i) = rho*(qstar(tr,k,j,i)-tracers(tr,k,j,i)/rho)/dt;
        }
      });
    }


    // Apply halo boundary conditions to the fields
    // Precursor BCs assume that the ghost cell data has already been copied into this coupler object
    //  before this function is called
    // coupler : reference to the coupler object
    // fields  : array of fields with halos
    // istage  : current RK stage
    // icycle  : current cycle number (for precursor data lookup)
    void halo_boundary_conditions( core::Coupler & coupler               ,
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



    // This computes the average column of the fields for ghost cell filling in idealized non-tubulent forcing simulations
    // coupler : reference to the coupler object
    // returns : average column of fields_loc from compute_tendencies
    real2d compute_average_ghost_column( core::Coupler & coupler ) {
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
      real4d state("state",num_state,nz,ny,nx); // State variables
      real4d tracers;
      if (num_tracers > 0) tracers = real4d("tracers",num_tracers,nz,ny,nx);
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



    // For simulations forced by a concurrent turbulent precursor, copy the ghost cell data from the precursor coupler to the main coupler
    // coupler_prec : reference to the precursor coupler object
    // coupler_main : reference to the main coupler object
    void copy_precursor_ghost_cells( core::Coupler & coupler_prec , core::Coupler & coupler_main ) {
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



    // For simulations forced by a concurrent turbulent precursor, copy the ghost cell data from the precursor coupler to the main coupler
    // coupler_prec : reference to the precursor coupler object
    // coupler_main : reference to the main coupler object
    void copy_column_to_precursor_ghost_cells( core::Coupler & coupler , real2d const & col ) {
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



    // Refresh the dycore immersed-proportion field from the coupler and populate all halo cells.
    void create_immersed_proportion_halos(core::Coupler &coupler) const {
      using yakl::SimpleBounds;
      auto nz     = coupler.get_nz();
      auto ny     = coupler.get_ny();
      auto nx     = coupler.get_nx();
      auto &dm    = coupler.get_data_manager_readwrite();
      auto wall_B = coupler.get_option<std::string>("bc_z1") == "wall_free_slip";
      auto wall_T = coupler.get_option<std::string>("bc_z2") == "wall_free_slip";
      auto immersed_thresh = coupler.get_option<real>("immersed_threshold",0.5);

      if (! dm.entry_exists("dycore_immersed_proportion_halos")) {
        dm.register_and_allocate<real>("dycore_immersed_proportion_halos",{nz+2*hs,ny+2*hs,nx+2*hs});
      }

      auto immersed_prop       = dm.get<real const,3>("immersed_proportion");
      auto immersed_prop_halos = dm.get<real,3>("dycore_immersed_proportion_halos");
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int k, int j, int i) {
        immersed_prop_halos(hs+k,hs+j,hs+i) = immersed_prop(k,j,i) > immersed_thresh ? 1 : 0;
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
              if (fields_halos_larger(0,hsnew+k+kk,hsnew+j+jj,hsnew+i+ii) > immersed_thresh) {
                int distance_loc = std::max(std::abs(kk),std::max(std::abs(jj),std::abs(ii)));
                distance = std::min(distance,static_cast<real>(std::max(1,distance_loc)));
              }
            }
          }
        }
        immersed_distance(k,j,i) = distance;
      });
    }



    // Initialize the class data as well as the state and tracers arrays and convert them back into the coupler state
    // coupler : reference to the coupler object
    // Make sure that all tracers are registered in the coupler before calling this function
    // This should be called after initializing the model data but before perturbing the initial conditions for
    //  things like thermals or initial potential temperature perturbations to initiate turbulence
    //  so that the hydrostatic profiles are accurately computed
    void init(core::Coupler &coupler) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("init");
      #endif
      using yakl::SimpleBounds;
      auto nx             = coupler.get_nx();       // Local number of cells in x-direction (not including halos)
      auto ny             = coupler.get_ny();       // Local number of cells in y-direction (not including halos)
      auto nz             = coupler.get_nz();       // Local number of cells in z-direction (not including halos)
      auto dx             = coupler.get_dx();       // Grid spacing in x-direction
      auto dy             = coupler.get_dy();       // Grid spacing in y-direction
      auto dz             = coupler.get_dz();       // Cell thicknesses in z-direction (1-D array of length nz)
      auto zmid           = coupler.get_zmid();     // Cell-center heights in z-direction
      auto px             = coupler.get_px();       // MPI rank in x-direction
      auto py             = coupler.get_py();       // MPI rank in y-direction
      auto nproc_x        = coupler.get_nproc_x();  // Number of MPI ranks in x-direction
      auto nproc_y        = coupler.get_nproc_y();  // Number of MPI ranks in y-direction
      auto nx_glob        = coupler.get_nx_glob();  // Global number of cells in x-direction
      auto ny_glob        = coupler.get_ny_glob();  // Global number of cells in y-direction
      auto num_tracers    = coupler.get_num_tracers();  // Number of tracer fields
      auto gamma          = coupler.get_option<real>("gamma_d"); // cp_dry / cv_dry (about 1.4)
      auto C0             = coupler.get_option<real>("C0"     ); // pressure = C0*pow(rho*theta,gamma)
      auto grav           = coupler.get_option<real>("grav"   ); // Gravitational acceleration
      auto enable_gravity = coupler.get_option<bool>("enable_gravity",true); // Whether gravity is enabled
      auto &dm            = coupler.get_data_manager_readwrite(); // Get data manager as read-write
      auto tracer_names   = coupler.get_tracer_names();           // Get tracer names from coupler (std::vector<std::string>)
      // Get the time stepping scheme to set num_stages
      auto time_stepper   = coupler.get_option<std::string>("dycore_time_stepper","ssprk3");

      // Set the number of stages based on the time stepping scheme
      if      (time_stepper == "ssprk3") { coupler.set_option("dycore_num_stages",3);    }
      else if (time_stepper == "linrk3") { coupler.set_option("dycore_num_stages",3);    }
      else if (time_stepper == "linrk4") { coupler.set_option("dycore_num_stages",4);    }
      else                               { Kokkos::abort("Invalid dycore_time_stepper"); }
      coupler.set_option("dycore_max_cycles",4);

      // If the current coupler object is a precursor for another simulation, or the current coupler is using
      //  precursor BC's, then allocate ghost cell storage for exchanging data between the precursor and forced
      //  simulation. This storage is only needed on the MPI ranks at the domain boundaries where precursor BC's
      //  are applied.
      // We need storage for all variables, all Runge-Kutta stages, all sub-cycles, and the halo size in
      //  the horizontal directions.
      // The array that is halo exchanged has 5 state variables (num_state), all tracers (num_tracers),
      //  and a pressure variable, so num_state+num_tracers+1.
      if ( coupler.get_option<bool>("dycore_is_precursor",false)   ||
           coupler.get_option<std::string>("bc_x1") == "precursor" ||
           coupler.get_option<std::string>("bc_x2") == "precursor" ||
           coupler.get_option<std::string>("bc_y1") == "precursor" ||
           coupler.get_option<std::string>("bc_y2") == "precursor" ) {
        auto nstage     = coupler.get_option<int>("dycore_num_stages"); // Number of Runge-Kutta stages
        auto max_cycles = coupler.get_option<int>("dycore_max_cycles");
        if (px == 0) { // If we're at the west edge process of the domain
          dm.register_and_allocate<FLOC>("dycore_ghost_x1",{max_cycles,nstage,num_state+num_tracers+1,nz,ny,hs});
        }
        if (px == nproc_x-1) { // If we're at the east edge process of the domain
          dm.register_and_allocate<FLOC>("dycore_ghost_x2",{max_cycles,nstage,num_state+num_tracers+1,nz,ny,hs});
        }
        if (py == 0) { // If we're at the south edge process of the domain
          dm.register_and_allocate<FLOC>("dycore_ghost_y1",{max_cycles,nstage,num_state+num_tracers+1,nz,hs,nx});
        }
        if (py == nproc_y-1) { // If we're at the north edge process of the domain
          dm.register_and_allocate<FLOC>("dycore_ghost_y2",{max_cycles,nstage,num_state+num_tracers+1,nz,hs,nx});
        }
        if (dm.entry_exists("dycore_ghost_x1")) dm.get<FLOC,6>("dycore_ghost_x1") = 0;
        if (dm.entry_exists("dycore_ghost_x2")) dm.get<FLOC,6>("dycore_ghost_x2") = 0;
        if (dm.entry_exists("dycore_ghost_y1")) dm.get<FLOC,6>("dycore_ghost_y1") = 0;
        if (dm.entry_exists("dycore_ghost_y2")) dm.get<FLOC,6>("dycore_ghost_y2") = 0;
      }

      // Compute the metric jacobian (dz/dzeta) where zeta is the k interface index
      // import sympy as sp
      // def gen_coefs(N,lab,i0=0) :
      //   return sp.Matrix(sp.symbols(f"{lab}{i0+0}:{i0+N}"))
      // def gen_poly(coefs) :
      //   x = sp.symbols('x')
      //   return sum([ coefs[i]*x**i for i in range(len(coefs)) ])
      // N      = 7
      // x      = sp.symbols('x')
      // hs     = N//2
      // coefs  = gen_coefs(N,'a')
      // p      = gen_poly(coefs)
      // constr = sp.Matrix([ p.subs(x,i) for i in range(-hs,hs+1) ])
      // Ainv   = constr.jacobian(coefs).inv()
      // vals   = gen_coefs(N,'v')
      // p      = gen_poly(Ainv*vals)
      // dp     = p.diff(x,1)
      // print(dp.subs(x,0))
      dm.register_and_allocate<real>("dycore_metjac_edges",{nz+1});
      auto metjac_edges = dm.get<real,1>("dycore_metjac_edges");
      yakl::parallel_for( YAKL_AUTO_LABEL() , nz+1 , KOKKOS_LAMBDA (int k) {
        SArray<real,7> s;
        s(0) = -dz(std::max(0,k-1))-dz(std::max(0,k-2))-dz(std::max(0,k-3));
        for (int kk=1; kk < 7; kk++) { s(kk) = s(kk-1) + dz(std::max(0,std::min(nz-1,k-4+kk))); }
        metjac_edges(k) = -s(0)/60 + 3*s(1)/20 - 3*s(2)/4 + 3*s(4)/4 - 3*s(5)/20 + s(6)/60;
      });

      coupler.set_option<int>("dycore_hs",hs); // Let other modules know the dycore halo size

      // Accumulate arrays that determine whethe each tracer adds mass and whether each tracer is positive definite
      // Do this on the host at first since it involves std::string operations
      bool1d tracer_adds_mass("tracer_adds_mass",std::max(1,num_tracers));
      bool1d tracer_positive ("tracer_positive" ,std::max(1,num_tracers));
      auto tracer_adds_mass_host = tracer_adds_mass.createHostCopy();
      auto tracer_positive_host  = tracer_positive .createHostCopy();
      tracer_adds_mass_host = false;
      tracer_positive_host  = false;
      for (int tr=0; tr < num_tracers; tr++) {
        std::string tracer_desc;
        bool        tracer_found, positive, adds_mass, diffuse;
        coupler.get_tracer_info( tracer_names.at(tr) , tracer_desc, tracer_found , positive , adds_mass , diffuse );
        tracer_positive_host (tr) = positive;
        tracer_adds_mass_host(tr) = adds_mass;
      }
      // Copy to device, register in coupler data manager, and store in data manager memory
      tracer_positive_host .deep_copy_to(tracer_positive );
      tracer_adds_mass_host.deep_copy_to(tracer_adds_mass);
      dm.register_and_allocate<bool>("tracer_adds_mass",{std::max(1,num_tracers)});
      auto dm_tracer_adds_mass = dm.get<bool,1>("tracer_adds_mass");
      tracer_adds_mass.deep_copy_to(dm_tracer_adds_mass);
      dm.register_and_allocate<bool>("tracer_positive",{std::max(1,num_tracers)});
      auto dm_tracer_positive = dm.get<bool,1>("tracer_positive");
      tracer_positive.deep_copy_to(dm_tracer_positive);

      // Allocate state and tracer arrays, and convert coupler data to dynamics format for
      //  computing the initial hydrostatic profiles of density, potential temperature, and pressure
      real4d state("state",num_state,nz,ny,nx);
      real4d tracers;
      state = 0;
      if (num_tracers > 0) {
        tracers = real4d("tracers",num_tracers,nz,ny,nx);
        tracers = 0;
      }
      convert_coupler_to_dynamics( coupler , state , tracers );
      // Compute the average column of density, potential temperature, and pressure for use
      //  in initializing the hydrostatic profiles including halo cells
      // The computation being here is why init should be called after initializing initial data
      //  but before applying perturbations to the flow
      dm.register_and_allocate<real>("hy_dens_cells"    ,{nz+2*hs});
      dm.register_and_allocate<real>("hy_theta_cells"   ,{nz+2*hs});
      dm.register_and_allocate<real>("hy_pressure_cells",{nz+2*hs});
      auto r = dm.get<real,1>("hy_dens_cells"    );    r = 0;
      auto t = dm.get<real,1>("hy_theta_cells"   );    t = 0;
      auto p = dm.get<real,1>("hy_pressure_cells");    p = 0;
      // Local accumulations
      yakl::parallel_for( YAKL_AUTO_LABEL() , nz , KOKKOS_LAMBDA (int k) {
        for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
            r(hs+k) += state(idR,k,j,i);
            t(hs+k) += state(idT,k,j,i) / state(idR,k,j,i);
            p(hs+k) += C0 * std::pow( state(idT,k,j,i) , gamma );
          }
        }
      });
      // Global aggregations of sums
      coupler.get_parallel_comm().all_reduce( r , MPI_SUM ).deep_copy_to(r);
      coupler.get_parallel_comm().all_reduce( t , MPI_SUM ).deep_copy_to(t);
      coupler.get_parallel_comm().all_reduce( p , MPI_SUM ).deep_copy_to(p);
      // Computation of averages
      real r_nx_ny = 1./(nx_glob*ny_glob);
      yakl::parallel_for( YAKL_AUTO_LABEL() , nz , KOKKOS_LAMBDA (int k) {
        r(hs+k) *= r_nx_ny;
        t(hs+k) *= r_nx_ny;
        p(hs+k) *= r_nx_ny;
      });
      // Extend theta with the constant physical gradient from the nearest two interior cell centers.
      // For q = rho*theta, hydrostatic balance and p = C0*q^gamma give
      // d(q^(gamma-1))/dz = -grav*(gamma-1)/(gamma*C0*theta).
      // Integrating 1/theta exactly for linear theta keeps rho, theta, and pressure hydrostatically consistent.
      if (hs > 0 && nz < 2) {
        endrun("ERROR: Hydrostatic ghost-cell extension requires nz >= 2");
      }
      real const B = grav*(gamma-1)/(gamma*C0);
      yakl::parallel_for( YAKL_AUTO_LABEL(), hs,
                          KOKKOS_LAMBDA (int kk) {
        { // Extend below the first interior cell; boundary ghost cells retain the first-cell thickness.
          int  const k0         = hs;
          int  const k          = k0-1-kk;
          real const theta0     = t(k0);
          real const q0         = r(k0)*theta0;
          real const Q0         = std::pow(q0,gamma-1);
          real const delta_z    = -dz(0)*(kk+1);
          real const grad_theta = (t(k0+1)-theta0)/(zmid(1)-zmid(0));
          real const theta_g    = theta0 + grad_theta*delta_z;
          real const x          = (theta_g-theta0)/theta0;
          real fac;
          // log1p(x)/x evaluates the linear-theta integral; use its series near x=0.
          if (std::abs(x) < 1.e-6_fp) { fac = 1._fp - 0.5_fp*x + x*x/3._fp; }
          else                        { fac = std::log1p(x)/x;              }
          real const integral = delta_z/theta0*fac;
          real const Qg       = Q0 - B*integral;
          real const qg       = std::pow(Qg,1._fp/(gamma-1));
          t(k) = theta_g;
          r(k) = qg/theta_g;
          p(k) = C0*std::pow(qg,gamma);
        }
        { // Extend above the last interior cell; boundary ghost cells retain the last-cell thickness.
          int  const k0         = hs+nz-1;
          int  const k          = k0+1+kk;
          real const theta0     = t(k0);
          real const q0         = r(k0)*theta0;
          real const Q0         = std::pow(q0,gamma-1);
          real const delta_z    = dz(nz-1)*(kk+1);
          real const grad_theta = (theta0-t(k0-1))/(zmid(nz-1)-zmid(nz-2));
          real const theta_g    = theta0 + grad_theta*delta_z;
          real const x          = (theta_g-theta0)/theta0;
          real fac;
          // This expression also tends smoothly to delta_z/theta0 for zero theta gradient.
          if (std::abs(x) < 1.e-6_fp) { fac = 1._fp - 0.5_fp*x + x*x/3._fp; }
          else                        { fac = std::log1p(x)/x;              }
          real const integral = delta_z/theta0*fac;
          real const Qg       = Q0 - B*integral;
          real const qg       = std::pow(Qg,1._fp/(gamma-1));
          t(k) = theta_g;
          r(k) = qg/theta_g;
          p(k) = C0*std::pow(qg,gamma);
        }
      });

      // This lambda function is to interpolate hydrostatic profiles from cell centers to edges
      //  (linear for theta, and log-linear for rho and pressure)
      auto compute_hydrostasis_edges = [] (core::Coupler &coupler) {
        using yakl::SimpleBounds;
        auto nz   = coupler.get_nz  (); // Number of cells in z-direction (not including halos)
        auto ny   = coupler.get_ny  (); // Number of cells in y-direction (not including halos)
        auto nx   = coupler.get_nx  (); // Number of cells in x-direction (not including halos)
        auto &dm  = coupler.get_data_manager_readwrite(); // Get data manager as read-write
        // Register edge hydrostatic values if they do not already exist
        if (! dm.entry_exists("hy_dens_edges"    )) dm.register_and_allocate<real>("hy_dens_edges"    ,{nz+1});
        if (! dm.entry_exists("hy_theta_edges"   )) dm.register_and_allocate<real>("hy_theta_edges"   ,{nz+1});
        if (! dm.entry_exists("hy_pressure_edges")) dm.register_and_allocate<real>("hy_pressure_edges",{nz+1});
        // Obtain the cells (with halos) and edges hydrostatic values
        auto hy_dens_cells     = dm.get<real const,1>("hy_dens_cells"    );
        auto hy_theta_cells    = dm.get<real const,1>("hy_theta_cells"   );
        auto hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells");
        auto hy_dens_edges     = dm.get<real      ,1>("hy_dens_edges"    );
        auto hy_theta_edges    = dm.get<real      ,1>("hy_theta_edges"   );
        auto hy_pressure_edges = dm.get<real      ,1>("hy_pressure_edges");
        // Interpolate from cell centers to edges
        if (ord < 5) {
          yakl::parallel_for( YAKL_AUTO_LABEL() , nz+1 , KOKKOS_LAMBDA (int k) {
            hy_dens_edges    (k) = std::exp( 0.5_fp*std::log(hy_dens_cells(hs+k-1)) +
                                             0.5_fp*std::log(hy_dens_cells(hs+k  )) );
            hy_theta_edges   (k) =           0.5_fp*hy_theta_cells(hs+k-1) +
                                             0.5_fp*hy_theta_cells(hs+k  ) ;
            hy_pressure_edges(k) = std::exp( 0.5_fp*std::log(hy_pressure_cells(hs+k-1)) +
                                             0.5_fp*std::log(hy_pressure_cells(hs+k  )) );
          });
        } else {
          yakl::parallel_for( YAKL_AUTO_LABEL() , nz+1 , KOKKOS_LAMBDA (int k) {
            hy_dens_edges    (k) = std::exp( -1./12.*std::log(hy_dens_cells(hs+k-2)) +
                                              7./12.*std::log(hy_dens_cells(hs+k-1)) +
                                              7./12.*std::log(hy_dens_cells(hs+k  )) +
                                             -1./12.*std::log(hy_dens_cells(hs+k+1)) );
            hy_theta_edges   (k) =           -1./12.*hy_theta_cells(hs+k-2) +
                                              7./12.*hy_theta_cells(hs+k-1) +
                                              7./12.*hy_theta_cells(hs+k  ) +
                                             -1./12.*hy_theta_cells(hs+k+1);
            hy_pressure_edges(k) = std::exp( -1./12.*std::log(hy_pressure_cells(hs+k-2)) +
                                              7./12.*std::log(hy_pressure_cells(hs+k-1)) +
                                              7./12.*std::log(hy_pressure_cells(hs+k  )) +
                                             -1./12.*std::log(hy_pressure_cells(hs+k+1)) );
          });
        }
      };

      // Call the two lambda functions created above to set up immersed proportion halos,
      //  immersed-distance data, and compute hydrostatic edge values
      create_immersed_proportion_halos( coupler );
      compute_hydrostasis_edges       ( coupler );

      auto projection_config = acoustic_projection_config(coupler);
      if (projection_config.preconditioner == "Multigrid") {
        anelastic_multigrid = std::make_shared<ConnectivityGalerkinMultigrid<float>>();
        projection_config.multigrid = anelastic_multigrid;
      } else if (projection_config.preconditioner == "GeometricMultigrid") {
        anelastic_geometric_multigrid = std::make_shared<GeometricMultigrid<float>>();
        projection_config.geometric_multigrid = anelastic_geometric_multigrid;
      }
      initialize_acoustic_projection<ord>(coupler,projection_config);

      // Projection pressure is a diagnostic constraint pressure, not thermodynamic EOS pressure. It is mean-zero only
      // for the unscreened operator, whose pressure has a constant nullspace.
      dm.register_and_allocate<real>("anelastic_pressure_pert",{nz,ny,nx});
      dm.get<real,3>("anelastic_pressure_pert") = 0;
      coupler.register_output_variable<real>("anelastic_pressure_pert",core::Coupler::DIMS_3D,
                                             {{"units",std::string("Pa")}});

      // Register immersed_proportion as an output and restart variable
      coupler.register_output_variable<real>( "immersed_proportion" , core::Coupler::DIMS_3D      );


      // Create an output module to be called during coupler.write_output() to write hydrostatic profiles
      //   and write perturbations of potential temperature, pressure, and density to file
      // coupler : reference to the coupler object
      // nc      : reference to the FileIO object for writing output (open and not in define mode)
      coupler.register_write_output_module( [=] (core::Coupler &coupler, core::FileIO &nc) {
        auto i_beg = coupler.get_i_beg(); // Get local starting indices in x and y directions
        auto j_beg = coupler.get_j_beg(); // Get local starting indices in x and y directions
        auto nz    = coupler.get_nz();    // Get local number of cells in z-direction (not including halos)
        auto ny    = coupler.get_ny();    // Get local number of cells in y-direction (not including halos)
        auto nx    = coupler.get_nx();    // Get local number of cells in x-direction (not including halos)
        nc.redef();  // re-enter define mode to add new dimensions and variables
        nc.create_dim( "z_halo" , coupler.get_nz()+2*hs );         // Vertical dimension with halos
        nc.create_var<real>( "z_halo"             , {"z_halo"});    // Define haloed vertical coordinate
        nc.create_var<real>( "hy_dens_cells"     , {"z_halo"});    // Define hydrostatic density variable
        nc.create_var<real>( "hy_theta_cells"    , {"z_halo"});    // Define hydrostatic potential temperature variable
        nc.create_var<real>( "hy_pressure_cells" , {"z_halo"});    // Define hydrostatic pressure variable
        nc.writeVariableAttribute(std::string("m")       ,"z_halo"            ,"units");
        nc.writeVariableAttribute(std::string("z_halo")  ,"hy_dens_cells"     ,"coordinates");
        nc.writeVariableAttribute(std::string("kg/m^3")  ,"hy_dens_cells"     ,"units");
        nc.writeVariableAttribute(std::string("z_halo")  ,"hy_theta_cells"    ,"coordinates");
        nc.writeVariableAttribute(std::string("K")       ,"hy_theta_cells"    ,"units");
        nc.writeVariableAttribute(std::string("z_halo")  ,"hy_pressure_cells" ,"coordinates");
        nc.writeVariableAttribute(std::string("Pa")      ,"hy_pressure_cells" ,"units");
        nc.writeGlobalAttribute(hs,"dycore_hs");
        // nc.create_var<real>( "theta_pert"        , {"z","y","x"}); // Define potential temperature perturbation variable
        // nc.create_var<real>( "pressure_pert"     , {"z","y","x"}); // Define pressure perturbation variable
        // nc.create_var<real>( "density_pert"      , {"z","y","x"}); // Define density perturbation variable
        nc.enddef(); // Exit define mode to write data
        auto const zmid = coupler.get_zmid();
        auto const dz   = coupler.get_dz();
        real1d z_halo("z_halo_output",nz+2*hs);
        yakl::parallel_for(YAKL_AUTO_LABEL(),nz+2*hs,KOKKOS_LAMBDA (int k) {
          if      (k < hs   ) z_halo(k) = zmid(0    )-(hs-k)*dz(0   );
          else if (k >= hs+nz) z_halo(k) = zmid(nz-1)+(k-hs-nz+1)*dz(nz-1);
          else                  z_halo(k) = zmid(k-hs);
        });
        nc.begin_indep_data(); // Enter independent data mode to write 1-D arrays from main task only
        auto &dm = coupler.get_data_manager_readonly(); // Get data manager as read-only
        // Write hydrostatic profiles from main task only
        if (coupler.is_mainproc()) nc.write( z_halo                                      , "z_halo"           );
        if (coupler.is_mainproc()) nc.write_data_manager<real,1>(dm,"hy_dens_cells"    ,"hy_dens_cells"    );
        if (coupler.is_mainproc()) nc.write_data_manager<real,1>(dm,"hy_theta_cells"   ,"hy_theta_cells"   );
        if (coupler.is_mainproc()) nc.write_data_manager<real,1>(dm,"hy_pressure_cells","hy_pressure_cells");
        nc.end_indep_data(); // Exit independent data mode to write 3-D perturbation arrays
        // // Allocate state and tracer arrays, and convert coupler data to dynamics format to compute perturbations
        // real4d state  ("state"  ,num_state  ,nz,ny,nx);
        // real4d tracers("tracers",num_tracers,nz,ny,nx);
        // convert_coupler_to_dynamics( coupler , state , tracers );
        // // Define the offset for writing the 3-D perturbation arrays for this MPI rank
        // std::vector<MPI_Offset> start_3d = {0,(MPI_Offset)j_beg,(MPI_Offset)i_beg};
        // real3d data("data",nz,ny,nx); // Holds local 3-D perturbation data before writing
        // auto hy_dens_cells = dm.get<real const,1>("hy_dens_cells");
        // // Compute and write perturbation density
        // yakl::parallel_for( yakl::Bounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        //   data(k,j,i) = state(idR,k,j,i) - hy_dens_cells(hs+k);
        // });
        // nc.write_all(data,"density_pert",start_3d);
        // // Compute and write perturbation potential temperature
        // auto hy_theta_cells = dm.get<real const,1>("hy_theta_cells");
        // yakl::parallel_for( yakl::Bounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        //   data(k,j,i) = state(idT,k,j,i) / state(idR,k,j,i) - hy_theta_cells(hs+k);
        // });
        // nc.write_all(data,"theta_pert",start_3d);
        // // Compute and write perturbation pressure
        // auto hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells");
        // yakl::parallel_for( yakl::Bounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        //   data(k,j,i) = C0 * std::pow( state(idT,k,j,i) , gamma ) - hy_pressure_cells(hs+k);
        // });
        // nc.write_all(data,"pressure_pert",start_3d);
      } );

      // Register a restart module to read in hydrostatic profiles from file
      // coupler : reference to the coupler object
      // nc      : reference to the FileIO object for reading restart data (opened)
      coupler.register_overwrite_with_restart_module( [=, this] (core::Coupler &coupler, core::FileIO &nc) {
        auto &dm = coupler.get_data_manager_readwrite();
        nc.read_all(dm.get<real,1>("hy_dens_cells"    ),"hy_dens_cells"    ,{0});
        nc.read_all(dm.get<real,1>("hy_theta_cells"   ),"hy_theta_cells"   ,{0});
        nc.read_all(dm.get<real,1>("hy_pressure_cells"),"hy_pressure_cells",{0});
        create_immersed_proportion_halos( coupler );
        compute_hydrostasis_edges       ( coupler );
      } );
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("init");
      #endif
    }



    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    // coupler : reference to the coupler object
    // state   : dynamics state array
    // tracers : dynamics tracers array
    void convert_dynamics_to_coupler( core::Coupler &coupler ,
                                      realConst4d    state   ,
                                      realConst4d    tracers ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("convert_dynamics_to_coupler");
      #endif
      using yakl::SimpleBounds;
      auto nx          = coupler.get_nx();  // Number of cells in x-direction (not including halos)
      auto ny          = coupler.get_ny();  // Number of cells in y-direction (not including halos)
      auto nz          = coupler.get_nz();  // Number of cells in z-direction (not including halos)
      auto R_d         = coupler.get_option<real>("R_d"    ); // Gas constant for dry air
      auto R_v         = coupler.get_option<real>("R_v"    ); // Gas constant for water vapor
      auto cp_d        = coupler.get_option<real>("cp_d"   ); // Gas constant for dry air
      auto gamma       = coupler.get_option<real>("gamma_d"); // Ratio of specific heats for dry air
      auto C0          = coupler.get_option<real>("C0"     ); // p = C0 * (rho*theta)^gamma
      auto p0          = coupler.get_option<real>("p0"     ); // p0
      auto num_tracers = coupler.get_num_tracers(); // Number of tracers
      auto &dm         = coupler.get_data_manager_readwrite(); // Get data manager as read-write
      auto dm_rho_d          = dm.get<real,3>("density_dry"); // Get coupler dry density array
      auto dm_uvel           = dm.get<real,3>("uvel"       ); // Get coupler u-velocity array
      auto dm_vvel           = dm.get<real,3>("vvel"       ); // Get coupler v-velocity array
      auto dm_wvel           = dm.get<real,3>("wvel"       ); // Get coupler w-velocity array
      auto dm_temp           = dm.get<real,3>("temperature"); // Get coupler temperature array
      auto hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells");
      auto tracer_adds_mass  = dm.get<bool const,1>("tracer_adds_mass" );
      bool rsst = coupler.get_option<bool>("dycore_rsst",false) || (coupler.get_option<real>("dycore_cs",350) != 350);
      // Accrue the tracer fields from the coupler data manager
      core::MultiField<real,3> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      int idWV = -1;
      for (int tr=0; tr < num_tracers; tr++) { if (tracer_names.at(tr) == "water_vapor") idWV = tr; }
      bool rho_v_exists = idWV >= 0;
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real,3>(tracer_names.at(tr)) ); }
      // Loop over all grid cells to compute dry density, velocities, temperature, and store in coupler arrays
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real rho   = state(idR,k,j,i);        // Total density
        real u     = state(idU,k,j,i) / rho;  // u-velocity
        real v     = state(idV,k,j,i) / rho;  // v-velocity
        real w     = state(idW,k,j,i) / rho;  // w-velocity
        real theta = state(idT,k,j,i) / rho;  // Potential temperature
        real rho_v = rho_v_exists ? tracers(idWV,k,j,i) : 0; // Water vapor density
        real rho_d = rho;                     // Dry air density starting value
        // Subtract mass-adding tracers from total density to get dry air density
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho_d -= tracers(tr,k,j,i); }
        // Use equation of state to compute temperature from pressure, dry density, and water vapor density
        real temp;
        real press = C0 * pow( rho*theta , gamma ); // Full pressure
        temp = press / ( rho_d * R_d + rho_v * R_v );
        dm_rho_d(k,j,i) = rho_d;  // Store dry air density in coupler array
        dm_uvel (k,j,i) = u;      // Store u-velocity in coupler array
        dm_vvel (k,j,i) = v;      // Store v-velocity in coupler array
        dm_wvel (k,j,i) = w;      // Store w-velocity in coupler array
        dm_temp (k,j,i) = temp;   // Store temperature in coupler array
        // Store tracer densities in coupler arrays
        for (int tr=0; tr < num_tracers; tr++) { dm_tracers(tr,k,j,i) = tracers(tr,k,j,i); }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("convert_dynamics_to_coupler");
      #endif
    }



    // Convert coupler's data to dynamics format of state and tracers arrays
    // coupler : reference to the coupler object
    // state   : dynamics state array
    // tracers : dynamics tracers array
    void convert_coupler_to_dynamics( core::Coupler const &coupler ,
                                      real4d              &state   ,
                                      real4d              &tracers ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("convert_coupler_to_dynamics");
      #endif
      using yakl::SimpleBounds;
      auto nx          = coupler.get_nx(); // Number of cells in x-direction (not including halos)
      auto ny          = coupler.get_ny(); // Number of cells in y-direction (not including halos)
      auto nz          = coupler.get_nz(); // Number of cells in z-direction (not including halos)
      auto R_d         = coupler.get_option<real>("R_d"    ); // Gas constant for dry air
      auto R_v         = coupler.get_option<real>("R_v"    ); // Gas constant for water vapor
      auto cp_d        = coupler.get_option<real>("cp_d"   ); // Gas constant for dry air
      auto gamma       = coupler.get_option<real>("gamma_d"); // Ratio of specific heats for dry air
      auto C0          = coupler.get_option<real>("C0"     ); // p = C0 * (rho*theta)^gamma
      auto p0          = coupler.get_option<real>("p0"     ); // p0
      auto num_tracers = coupler.get_num_tracers(); // Number of tracers
      auto &dm         = coupler.get_data_manager_readonly(); // Get data manager as read-only
      auto dm_rho_d         = dm.get<real const,3>("density_dry"); // Get coupler dry density array
      auto dm_uvel          = dm.get<real const,3>("uvel"       ); // Get coupler u-velocity array
      auto dm_vvel          = dm.get<real const,3>("vvel"       ); // Get coupler v-velocity array
      auto dm_wvel          = dm.get<real const,3>("wvel"       ); // Get coupler w-velocity array
      auto dm_temp          = dm.get<real const,3>("temperature"); // Get coupler temperature array
      auto tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
      bool const anelastic_ready = dm.entry_exists("hy_dens_cells");
      realConst1d hy_dens_cells;
      if (anelastic_ready) hy_dens_cells = dm.get<real const,1>("hy_dens_cells");
      // Accrue the tracer fields from the coupler data manager
      core::MultiField<real const,3> dm_tracers;
      auto tracer_names = coupler.get_tracer_names(); // Get the tracer names
      int idWV = -1;
      for (int tr=0; tr < num_tracers; tr++) { if (tracer_names.at(tr) == "water_vapor") idWV = tr; }
      bool rho_v_exists = idWV >= 0;
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real const,3>(tracer_names.at(tr)) ); }
      // Loop over all grid cells to compute dynamics state and tracers arrays from coupler data
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real rho_d = dm_rho_d(k,j,i); // Dry air density
        real u     = dm_uvel (k,j,i); // u-velocity
        real v     = dm_vvel (k,j,i); // v-velocity
        real w     = dm_wvel (k,j,i); // w-velocity
        real temp  = dm_temp (k,j,i); // Temperature
        real rho_v = rho_v_exists ? dm_tracers(idWV,k,j,i) : 0; // Water vapor density
        real rho   = rho_d;           // Total density starting value
        // Add mass-adding tracers to dry density to get total density
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho += dm_tracers(tr,k,j,i); }
        // Compute potential temperature from pressure and total density
        real theta;
        real press = rho_d * R_d * temp + rho_v * R_v * temp; // Full pressure
        theta = std::pow( press/C0 , 1._fp / gamma ) / rho;
        // During init, physical density bootstraps rho_H. Afterwards, persistent dynamics storage is rho_H weighted.
        real const rho_dyn = anelastic_ready ? hy_dens_cells(hs+k) : rho;
        state(idR,k,j,i) = rho_dyn;
        state(idU,k,j,i) = rho_dyn*u;
        state(idV,k,j,i) = rho_dyn*v;
        state(idW,k,j,i) = rho_dyn*w;
        state(idT,k,j,i) = rho_dyn*theta;
        for (int tr = 0; tr < num_tracers; tr++) {
          tracers(tr,k,j,i) = rho_dyn*dm_tracers(tr,k,j,i)/rho;
        }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("convert_coupler_to_dynamics");
      #endif
    }


  };

}

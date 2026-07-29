#include "dynamics_cell_centered.h"

namespace modules {

void EulerCellCentered::init(core::Coupler &coupler) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("init");
      #endif
      using yakl::SimpleBounds;
      auto nx             = coupler.get_nx();       // Local number of cells in x-direction (not including halos)
      auto ny             = coupler.get_ny();       // Local number of cells in y-direction (not including halos)
      auto nz             = coupler.get_nz();       // Local number of cells in z-direction (not including halos)
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
      }

      // Compute the metric jacobian (dz/dzeta) where zeta is the k interface index
      //
      // # Sagemath code
      // def coefs_1d(N,N0,lab) :
      //     return vector([ var(lab+'%s'%i) for i in range(N0,N0+N) ])
      // def poly_1d(N,coefs,x) :
      //     return sum( vector([ coefs[i]*x^i for i in range(N) ]) )
      // N      = 6
      // coefs  = coefs_1d(N,0,'a')
      // p      = poly_1d(N,coefs,x)
      // constr = vector([ p.subs(x=i-N/2+1) for i in range(N) ])
      // p      = poly_1d(N,jacobian(constr,coefs)^-1*coefs_1d(N,0,'s'),x)
      // print( vector([ i-N/2+1 for i in range(N) ]) )
      // print( 60*p.diff(x).subs(x=0) )
      // print( 60*p.diff(x).subs(x=1) )
      //
      dm.register_and_allocate<real>("dycore_metjac_edges",{nz+2,2});
      auto metjac_edges = dm.get<real,2>("dycore_metjac_edges");
      yakl::parallel_for( YAKL_AUTO_LABEL() , nz+2 , KOKKOS_LAMBDA (int k_in) {
        int k = k_in-1;
        SArray<real,6> s;
        s(0) = -dz(std::max(0,k-1))-dz(std::max(0,k-2));
        for (int kk=1; kk < 6; kk++) { s(kk) = s(kk-1) + dz(std::max(0,std::min(nz-1,k-3+kk))); }
        for (int kk=0; kk < 6; kk++) { s(kk) /= dz(std::max(0,std::min(nz-1,k))); }
        metjac_edges(k+1,0) = ( 3*s(0)-30*s(1)-20*s(2)+60*s(3)-15*s(4)+2*s(5))/60.;
        metjac_edges(k+1,1) = (-2*s(0)+15*s(1)-60*s(2)+20*s(3)+30*s(4)-3*s(5))/60.;
      });

      coupler.set_option<int>("dycore_hs",hs); // Let other modules know the dycore halo size

      // Accumulate arrays that determine whethe each tracer adds mass and whether each tracer is positive definite
      // Do this on the host at first since it involves std::string operations
      bool1d tracer_adds_mass("tracer_adds_mass",num_tracers);
      bool1d tracer_positive ("tracer_positive" ,num_tracers);
      auto tracer_adds_mass_host = tracer_adds_mass.createHostCopy();
      auto tracer_positive_host  = tracer_positive .createHostCopy();
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
      dm.register_and_allocate<bool>("tracer_adds_mass",{num_tracers});
      auto dm_tracer_adds_mass = dm.get<bool,1>("tracer_adds_mass");
      tracer_adds_mass.deep_copy_to(dm_tracer_adds_mass);
      dm.register_and_allocate<bool>("tracer_positive",{num_tracers});
      auto dm_tracer_positive = dm.get<bool,1>("tracer_positive");
      tracer_positive.deep_copy_to(dm_tracer_positive);

      // Allocate state and tracer arrays, and convert coupler data to dynamics format for
      //  computing the initial hydrostatic profiles of density, potential temperature, and pressure
      real4d state  ("state"  ,num_state  ,nz,ny,nx);  state   = 0;
      real4d tracers("tracers",num_tracers,nz,ny,nx);  tracers = 0;
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
        using yakl::SimpleBounds;;
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

      // Register immersed_proportion as an output and restart variable
      coupler.register_output_variable<real>( "immersed_proportion" , core::Coupler::DIMS_3D      );

      // Create an output module to be called during coupler.write_output() to write hydrostatic profiles
      //   and write perturbations of potential temperature, pressure, and density to file
      // coupler : reference to the coupler object
      // nc      : reference to the SimplePNetCDF object for writing output (open and not in define mode)
      coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        auto i_beg = coupler.get_i_beg(); // Get local starting indices in x and y directions
        auto j_beg = coupler.get_j_beg(); // Get local starting indices in x and y directions
        auto nz    = coupler.get_nz();    // Get local number of cells in z-direction (not including halos)
        auto ny    = coupler.get_ny();    // Get local number of cells in y-direction (not including halos)
        auto nx    = coupler.get_nx();    // Get local number of cells in x-direction (not including halos)
        nc.redef();  // re-enter define mode to add new dimensions and variables
        nc.create_dim( "z_halo" , coupler.get_nz()+2*hs );         // Vertical dimension with halos
        nc.create_var<real>( "hy_dens_cells"     , {"z_halo"});    // Define hydrostatic density variable
        nc.create_var<real>( "hy_theta_cells"    , {"z_halo"});    // Define hydrostatic potential temperature variable
        nc.create_var<real>( "hy_pressure_cells" , {"z_halo"});    // Define hydrostatic pressure variable
        // nc.create_var<real>( "theta_pert"        , {"z","y","x"}); // Define potential temperature perturbation variable
        // nc.create_var<real>( "pressure_pert"     , {"z","y","x"}); // Define pressure perturbation variable
        // nc.create_var<real>( "density_pert"      , {"z","y","x"}); // Define density perturbation variable
        nc.enddef(); // Exit define mode to write data
        nc.begin_indep_data(); // Enter independent data mode to write 1-D arrays from main task only
        auto &dm = coupler.get_data_manager_readonly(); // Get data manager as read-only
        // Write hydrostatic profiles from main task only
        if (coupler.is_mainproc()) nc.write( dm.get<real const,1>("hy_dens_cells"    ) , "hy_dens_cells"     );
        if (coupler.is_mainproc()) nc.write( dm.get<real const,1>("hy_theta_cells"   ) , "hy_theta_cells"    );
        if (coupler.is_mainproc()) nc.write( dm.get<real const,1>("hy_pressure_cells") , "hy_pressure_cells" );
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
      // nc      : reference to the SimplePNetCDF object for reading restart data (opened)
      coupler.register_overwrite_with_restart_module( [=, this] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
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

} // namespace modules

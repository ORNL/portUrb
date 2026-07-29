#include "dynamics_cell_centered.h"

namespace modules {

void EulerCellCentered::convert_dynamics_to_coupler( core::Coupler &coupler ,
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

void EulerCellCentered::convert_coupler_to_dynamics( core::Coupler const &coupler ,
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
      bool rsst = coupler.get_option<bool>("dycore_rsst",false) || (coupler.get_option<real>("dycore_cs",350) != 350);
      realConst1d hy_pressure_cells;
      if (dm.entry_exists("hy_pressure_cells"))  hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells");
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
        state(idR,k,j,i) = rho;         // Store total density in dynamics state array
        state(idU,k,j,i) = rho * u;     // Store momentum in dynamics state array
        state(idV,k,j,i) = rho * v;     // Store momentum in dynamics state array
        state(idW,k,j,i) = rho * w;     // Store momentum in dynamics state array
        state(idT,k,j,i) = rho * theta; // Store total potential temperature in dynamics state array
        // Store tracer densities in dynamics tracers array
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,k,j,i) = dm_tracers(tr,k,j,i); }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("convert_coupler_to_dynamics");
      #endif
    }

} // namespace modules

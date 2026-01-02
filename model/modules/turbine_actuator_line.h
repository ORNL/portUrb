
#pragma once

#include "main_header.h"
#include "coupler.h"

namespace YAML {
  template<> struct convert<std::tuple<real,real,real,real,std::string>> {
    static Node encode(const std::tuple<real,real,real,real,std::string>& rhs) {
      Node node;
      node.push_back(std::get<0>(rhs));
      node.push_back(std::get<1>(rhs));
      node.push_back(std::get<2>(rhs));
      node.push_back(std::get<3>(rhs));
      node.push_back(std::get<4>(rhs));
      return node;
    }

    static bool decode(const Node& node, std::tuple<real,real,real,real,std::string>& rhs) {
      if (!node.IsSequence() || node.size() != 5) {
        return false;
      }
      rhs = std::tuple<real,real,real,real,std::string>(node[0].as<real>(),
                                                        node[1].as<real>(),
                                                        node[2].as<real>(),
                                                        node[3].as<real>(),
                                                        node[4].as<std::string>());
      return true;
    }
  };
} // namespace YAML


namespace modules {

  // For simplicity and for parallel efficiency, the TurbineActuatorLine class will assume that:
  //   * The turbine is always yawed to face the west direction (yaw of zero degrees)
  //   * There is a constant upwind direction that does not change (coupler option)
  //   * There is only one type of reference turbine for all turbines simulated
  //   * The inflow wind speed is a constant coupler option to determine the turbine's rotation rate
  //   * Grid spacing is the same in all three directions, x, y, and z
  struct TurbineActuatorLine {
    int static constexpr MAX_FIELDS = 100;
    typedef yakl::SArray<realHost1d,1,MAX_FIELDS> MultiFieldHost;
    typedef yakl::SArray<real1d    ,1,MAX_FIELDS> MultiFieldDev;

    // This class holds information about a reference wind turbine, including lookup tables for various properties
    //   and turbine geometric properties
    struct RefTurbine {
      int            B                ;
      real           R                ;
      real           R_hub            ;
      real           H                ;
      real           overhang         ; // Offset of blades from tower center (m). This is also the length of the hub flange
      real           hub_radius       ; // Radius of the hub, where there is no blade (m)
      real           hub_flange_height; // Height (and width) of the hub flange (m)
      real           tower_base_rad   ; // Radius of the tower base at ground or water level (m)
      real           tower_top_rad    ; // Radius of the tower top connected to hub flange (m)
      real           shaft_tilt       ; // Shaft tilt in radians
      realHost1d     host_rad_locs    ;
      realHost1d     host_foil_mid    ;
      realHost1d     host_foil_len    ;
      realHost1d     host_foil_twist  ;
      realHost1d     host_foil_chord  ;
      realHost1d     host_foil_id     ;
      MultiFieldHost host_foil_alpha  ;
      MultiFieldHost host_foil_clift  ;
      MultiFieldHost host_foil_cdrag  ;
      realHost1d     host_rwt_mag     ;
      realHost1d     host_rwt_ct      ;
      realHost1d     host_rwt_cp      ;
      realHost1d     host_rwt_pwr_mw  ;
      realHost1d     host_rwt_rot     ;
      real1d         dev_rad_locs     ;
      real1d         dev_foil_mid     ;
      real1d         dev_foil_len     ;
      real1d         dev_foil_twist   ;
      real1d         dev_foil_chord   ;
      real1d         dev_foil_id      ;
      MultiFieldDev  dev_foil_alpha   ;
      MultiFieldDev  dev_foil_clift   ;
      MultiFieldDev  dev_foil_cdrag   ;
      real1d         dev_rwt_mag      ;
      real1d         dev_rwt_ct       ;
      real1d         dev_rwt_cp       ;
      real1d         dev_rwt_pwr_mw   ;
      real1d         dev_rwt_rot      ;
      void init( core::Coupler const & coupler ) {
        typedef std::tuple<real,real,real,real,std::string> FOIL_LINE;
        auto dx        = coupler.get_dx();
        auto eps_fixed = coupler.get_option<real>("turbine_eps_fixed",-1);
        // GET YAML DATA
        YAML::Node node   = YAML::LoadFile(coupler.get_option<std::string>("turbine_file"));
        R                 = node["blade_radius"      ].as<real>();
        R_hub             = node["hub_radius"        ].as<real>();
        H                 = node["hub_height"        ].as<real>();
        B                 = node["num_blades"        ].as<int>(3);
        overhang          = node["overhang"          ].as<real>(-0.1 *R);
        hub_radius        = node["hub_radius"        ].as<real>( 0.03*R);
        hub_flange_height = node["hub_flange_height" ].as<real>( 0.04*R);
        tower_base_rad    = node["tower_base_radius" ].as<real>(5);
        tower_top_rad     = node["tower_top_radius"  ].as<real>(3);
        shaft_tilt        = node["shaft_tilt_deg"    ].as<real>(0)/180.*M_PI;
        auto foil_summary = node["airfoil_summary"   ].as<std::vector<FOIL_LINE>>();
        auto foil_names   = node["airfoil_names"     ].as<std::vector<std::string>>();
        auto velmag       = node["velocity_magnitude"].as<std::vector<real>>();
        auto cthrust      = node["thrust_coef"       ].as<std::vector<real>>();
        auto cpower       = node["power_coef"        ].as<std::vector<real>>();
        auto power_mw     = node["power_megawatts"   ].as<std::vector<real>>();
        auto rot_rpm      = node["rotation_rpm"      ].as<std::vector<real>>();
        std::vector<std::vector<std::vector<real>>> foil_vals;
        for (int ifoil=0; ifoil < foil_names.size(); ifoil++) {
          foil_vals.push_back( node[foil_names.at(ifoil)].as<std::vector<std::vector<real>>>() );
        }
        // COPY YAML DATA TO YAKL ARRAYS
        int nseg  = foil_summary.size();
        int nfoil = foil_names.size();
        host_foil_mid   = realHost1d("foil_mid"  ,nseg);
        host_foil_len   = realHost1d("foil_len"  ,nseg);
        host_foil_twist = realHost1d("foil_twist",nseg);
        host_foil_chord = realHost1d("foil_chord",nseg);
        host_foil_id    = realHost1d("foil_id"   ,nseg);
        for (int iseg=0; iseg < nseg ; iseg++) {
          host_foil_mid  (iseg) = std::get<0>(foil_summary.at(iseg));
          host_foil_twist(iseg) = std::get<1>(foil_summary.at(iseg))/180.*M_PI;
          host_foil_len  (iseg) = std::get<2>(foil_summary.at(iseg));
          host_foil_chord(iseg) = std::get<3>(foil_summary.at(iseg));
          int id = -1;
          for (int ifoil = 0; ifoil < nfoil; ifoil++) {
            if (std::get<4>(foil_summary.at(iseg)) == foil_names.at(ifoil)) { id = ifoil; break; }
          }
          host_foil_id   (iseg) = id;
        }
        for (int ifoil = 0; ifoil < nfoil; ifoil++) {
          int nalpha = foil_vals.at(ifoil).size();
          realHost1d loc_alpha("foil_alpha",nalpha);
          realHost1d loc_clift("foil_clift",nalpha);
          realHost1d loc_cdrag("foil_cdrag",nalpha);
          for (int ialpha = 0; ialpha < nalpha; ialpha++) {
            loc_alpha(ialpha) = foil_vals.at(ifoil).at(ialpha).at(0);
            loc_clift(ialpha) = foil_vals.at(ifoil).at(ialpha).at(1);
            loc_cdrag(ialpha) = foil_vals.at(ifoil).at(ialpha).at(2);
          }
          host_foil_alpha(ifoil) = loc_alpha;
          host_foil_clift(ifoil) = loc_clift;
          host_foil_cdrag(ifoil) = loc_cdrag;
        }
        int nrwt = velmag.size();
        host_rwt_mag    = realHost1d("rwt_mag"   ,nrwt);
        host_rwt_ct     = realHost1d("rwt_ct"    ,nrwt);
        host_rwt_cp     = realHost1d("rwt_cp"    ,nrwt);
        host_rwt_pwr_mw = realHost1d("rwt_pwr_mw",nrwt);
        host_rwt_rot    = realHost1d("rwt_rot"   ,nrwt);
        for (int irwt = 0; irwt < nrwt; irwt++) {
          host_rwt_mag   (irwt) = velmag  .at(irwt);
          host_rwt_ct    (irwt) = cthrust .at(irwt);
          host_rwt_cp    (irwt) = cpower  .at(irwt);
          host_rwt_pwr_mw(irwt) = power_mw.at(irwt);
          host_rwt_rot   (irwt) = rot_rpm .at(irwt)*2.*M_PI/60.;
        }
        real deps      = dx/4;
        int  nrad      = (int) std::ceil((R-R_hub)/deps);
        host_rad_locs  = realHost1d("rad_locs",nrad);
        for (int irad=0; irad < nrad; irad++) { host_rad_locs(irad) = R_hub + (R-R_hub)*(irad+0.5)/nrad; }
        // CREATE DEVICE COPIES
        dev_rad_locs   = host_rad_locs  .createDeviceCopy();
        dev_foil_mid   = host_foil_mid  .createDeviceCopy();
        dev_foil_len   = host_foil_len  .createDeviceCopy();
        dev_foil_twist = host_foil_twist.createDeviceCopy();
        dev_foil_chord = host_foil_chord.createDeviceCopy();
        dev_foil_id    = host_foil_id   .createDeviceCopy();
        dev_rwt_mag    = host_rwt_mag   .createDeviceCopy();
        dev_rwt_ct     = host_rwt_ct    .createDeviceCopy();
        dev_rwt_cp     = host_rwt_cp    .createDeviceCopy();
        dev_rwt_pwr_mw = host_rwt_pwr_mw.createDeviceCopy();
        dev_rwt_rot    = host_rwt_rot   .createDeviceCopy();
        for (int i=0; i < nfoil; i++) {
          dev_foil_alpha(i) = host_foil_alpha(i).createDeviceCopy();
          dev_foil_clift(i) = host_foil_clift(i).createDeviceCopy();
          dev_foil_cdrag(i) = host_foil_cdrag(i).createDeviceCopy();
        }
      }
    };



    // This holds information about an individual turbine in the simulation (there can be multiple turbines)
    struct Turbine {
      bool                    active;             // Whether this turbine affects this MPI task
      real                    base_loc_x;         // x location of the tower base
      real                    base_loc_y;         // y location of the tower base
      real                    rot_angle;          // Rotation angle in radians
      real                    pitch;              // blade pitch angle in radians
      core::ParallelComm      par_comm;           // MPI communicator for this turbine
      std::vector<realHost2d> force_axial_trace;  // Time trace axial force along radius for all blades
      std::vector<realHost2d> force_tang_trace;   // Time trace tangental force along radius for all blades
      std::vector<realHost2d> inflow_axial_trace; // Time trace axial inflow speed along radius for all blades
      std::vector<realHost2d> inflow_tang_trace;  // Time trace tangential inflow speed along radius for all blades
      std::vector<real>       power_trace;  
    };



    // This holds information about all turbines in the simulation
    struct TurbineGroup {
      std::vector<Turbine> turbines;  // All turbines in the simulation
      // This routine adds a turbine to the group based on its base location and reference turbine data
      // The coupler is needed in order to determine whether the turbine is active on this MPI task
      void add_turbine( core::Coupler       & coupler     ,
                        real                  base_loc_x  ,
                        real                  base_loc_y  ,
                        RefTurbine const    & ref_turbine ) {
        auto i_beg     = coupler.get_i_beg();  // Get the beginning x-direction index for this MPI task
        auto j_beg     = coupler.get_j_beg();  // Get the beginning y-direction index for this MPI task
        auto nx        = coupler.get_nx();     // Get the number of x-direction cells
        auto ny        = coupler.get_ny();     // Get the number of y-direction cells
        auto dx        = coupler.get_dx();     // Get the grid spacing in the x-direction
        auto dy        = coupler.get_dy();     // Get the grid spacing in the y-direction
        auto eps_fixed = coupler.get_option<real>("turbine_eps_fixed",-1);
        real max_chord = yakl::intrinsics::maxval(ref_turbine.host_foil_chord);
        real max_eps   = eps_fixed > 0 ? eps_fixed : std::max( max_chord/2 , 2*dx );
        // bounds of this MPI task's domain
        real dom_x1  = (i_beg+0 )*dx;
        real dom_x2  = (i_beg+nx)*dx;
        real dom_y1  = (j_beg+0 )*dy;
        real dom_y2  = (j_beg+ny)*dy;
        // Rectangular bounds of this turbine's potential influence
        real turb_x1 = base_loc_x-20*max_eps;
        real turb_x2 = base_loc_x+20*max_eps;
        real turb_y1 = base_loc_y-20*max_eps;
        real turb_y2 = base_loc_y+20*max_eps;
        bool active = !( turb_x1 > dom_x2 || // Turbine's to the right
                         turb_x2 < dom_x1 || // Turbine's to the left
                         turb_y1 > dom_y2 || // Turbine's above
                         turb_y2 < dom_y1 ); // Turbine's below
        // Create a local turbine object, assign its properties, and add it to the list
        Turbine loc;
        loc.active      = active;
        loc.base_loc_x  = base_loc_x;
        loc.base_loc_y  = base_loc_y;
        loc.rot_angle   = 0;
        loc.pitch       = 0;
        loc.par_comm.create( active , coupler.get_parallel_comm().get_mpi_comm() );
        turbines.push_back(loc);
      }
    };



    RefTurbine    ref_turbine;    // The reference turbine information
    TurbineGroup  turbine_group;  // Holds all turbines in the simulation
    int           trace_size;     // Number of time steps recorded in the turbine traces so far
                                  // This is reset to zero after writing output each time



    template <class T, int MEM>
    static KOKKOS_INLINE_FUNCTION T linear_interp( yakl::Array<T,1,MEM> const & aref                ,
                                                   yakl::Array<T,1,MEM> const & vref                ,
                                                   T                            a                   ,
                                                   bool                         const_extrap = true ) {
      int n = aref.size();
      if ( n==0 || aref.size() != vref.size() ) Kokkos::abort("Invalid input vectors");
      if ( a < aref(0) || aref.size() == 1 ) return const_extrap ? vref(0)   : 0.;
      if ( a > aref(n-1)                   ) return const_extrap ? vref(n-1) : 0.;
      for (int i=0; i < n-1; i++) {
        if (a >= aref(i) && a <= aref(i+1)) return vref(i)+(a-aref(i))/(aref(i+1)-aref(i))*(vref(i+1)-vref(i));
      }
      return 0.; // Doesn't get here, but gotta keep that compiler happy..
    }



    template <class T, int MEM>
    static KOKKOS_INLINE_FUNCTION T linear_interp( yakl::Array<T,1,MEM> const & aref                ,
                                                   yakl::Array<T,2,MEM> const & vref                ,
                                                   T                            a                   ,
                                                   int                          iblade              ,
                                                   bool                         const_extrap = true ) {
      int n = aref.size();
      if ( n==0 || aref.size() != vref.extent(1) ) Kokkos::abort("Invalid input vectors");
      if ( a < aref(0) || aref.size() == 1 ) return const_extrap ? vref(iblade,0)   : 0.;
      if ( a > aref(n-1)                   ) return const_extrap ? vref(iblade,n-1) : 0.;
      for (int i=0; i < n-1; i++) {
        if (a >= aref(i) && a <= aref(i+1)) {
          return vref(iblade,i)+(a-aref(i))/(aref(i+1)-aref(i))*(vref(iblade,i+1)-vref(iblade,i));
        }
      }
      return 0.; // Doesn't get here, but gotta keep that compiler happy..
    }



    // Initialize the turbine actuator disc module, adding all the specified turbines from the coupler options
    void init( core::Coupler &coupler ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx    = coupler.get_nx();    // Get the local number of x-direction cells
      auto ny    = coupler.get_ny();    // Get the local number of y-direction cells
      auto nz    = coupler.get_nz();    // Get the number of z-direction cells
      auto dx    = coupler.get_dx();    // Get the grid spacing in the x-direction
      auto dy    = coupler.get_dy();    // Get the grid spacing in the y-direction
      auto i_beg = coupler.get_i_beg(); // Get the beginning x-direction index for this MPI task
      auto j_beg = coupler.get_j_beg(); // Get the beginning y-direction index for this MPI task
      auto &dm   = coupler.get_data_manager_readwrite();
      ref_turbine.init( coupler );
      // Add turbines based on the specified x and y locations from coupler options
      if (coupler.option_exists("turbine_x_locs") && coupler.option_exists("turbine_y_locs")) {
        auto x_locs = coupler.get_option<std::vector<real>>("turbine_x_locs");
        auto y_locs = coupler.get_option<std::vector<real>>("turbine_y_locs");
        for (int iturb = 0; iturb < x_locs.size(); iturb++) {
          turbine_group.add_turbine( coupler , x_locs.at(iturb) , y_locs.at(iturb) , ref_turbine );
        }
      }
      trace_size = 0;  // Initialize trace size to zero
      dm.register_and_allocate<real>("turbine_tend_u","",{nz,ny,nx});
      dm.register_and_allocate<real>("turbine_tend_v","",{nz,ny,nx});
      dm.register_and_allocate<real>("turbine_tend_w","",{nz,ny,nx});
      coupler.register_output_variable<real>( "turbine_tend_u" , coupler.DIMS_3D );
      coupler.register_output_variable<real>( "turbine_tend_v" , coupler.DIMS_3D );
      coupler.register_output_variable<real>( "turbine_tend_w" , coupler.DIMS_3D );
      // Register output writing function to write turbine traces
      coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        if (trace_size > 0) {
          int nrad = ref_turbine.host_rad_locs.size();
          nc.redef();  // Enter define mode in order to add new variables and dimensions
          nc.create_dim( "ndt"     , trace_size    );
          nc.create_dim( "nblades" , ref_turbine.B );
          nc.create_dim( "nrad"    , nrad          );
          nc.create_var<real>( std::string("radial_points") , {"nrad"} );
          // Add a trace variable for each turbine for power, and inflow wind speed
          for (int iturb=0; iturb < turbine_group.turbines.size(); iturb++) {
            std::string turb_str = std::to_string(iturb);
            nc.create_var<real>( std::string("force_axial_" )+turb_str , {"ndt","nblades","nrad"} );
            nc.create_var<real>( std::string("force_tang_"  )+turb_str , {"ndt","nblades","nrad"} );
            nc.create_var<real>( std::string("inflow_axial_")+turb_str , {"ndt","nblades","nrad"} );
            nc.create_var<real>( std::string("inflow_tang_" )+turb_str , {"ndt","nblades","nrad"} );
            nc.create_var<real>( std::string("power_"       )+turb_str , {"ndt"                 } );
          }
          nc.enddef();  // Exit define mode to write data
          nc.begin_indep_data();  // Begin independent data section for writing from only the owning MPI task
          if (coupler.is_mainproc()) {
            nc.write( ref_turbine.host_rad_locs , std::string("radial_points") );
          }
          for (int iturb=0; iturb < turbine_group.turbines.size(); iturb++) {  // Loop over all turbines
            auto &turbine = turbine_group.turbines.at(iturb); // Grab a reference to this turbine for convenience
            // Only write this data from the MPI task that contains the base location
            if (turbine.base_loc_x >= i_beg*dx && turbine.base_loc_x < (i_beg+nx)*dx &&
                turbine.base_loc_y >= j_beg*dy && turbine.base_loc_y < (j_beg+ny)*dy ) {
              realHost3d force_axial_arr ("force_axial_arr" ,trace_size,ref_turbine.B,nrad);
              realHost3d force_tang_arr  ("force_tang_arr"  ,trace_size,ref_turbine.B,nrad);
              realHost3d inflow_axial_arr("inflow_axial_arr",trace_size,ref_turbine.B,nrad);
              realHost3d inflow_tang_arr ("inflow_tang_arr" ,trace_size,ref_turbine.B,nrad);
              realHost1d power_arr       ("power_arr"       ,trace_size                   );
              // Load data from std:;vector to yakl::Array for writing
              for (int i1=0; i1 < trace_size; i1++) {
                for (int i2=0; i2 < ref_turbine.B; i2++) {
                  for (int i3=0; i3 < nrad; i3++) {
                    force_axial_arr (i1,i2,i3) = turbine.force_axial_trace .at(i1)(i2,i3);
                    force_tang_arr  (i1,i2,i3) = turbine.force_tang_trace  .at(i1)(i2,i3);
                    inflow_axial_arr(i1,i2,i3) = turbine.inflow_axial_trace.at(i1)(i2,i3);
                    inflow_tang_arr (i1,i2,i3) = turbine.inflow_tang_trace .at(i1)(i2,i3);
                  }
                }
                power_arr(i1) = turbine.power_trace.at(i1);
              }
              // Write the data arrays to file
              nc.write( force_axial_arr  , std::string("force_axial_" )+std::to_string(iturb) );
              nc.write( force_tang_arr   , std::string("force_tang_"  )+std::to_string(iturb) );
              nc.write( inflow_axial_arr , std::string("inflow_axial_")+std::to_string(iturb) );
              nc.write( inflow_tang_arr  , std::string("inflow_tang_" )+std::to_string(iturb) );
              nc.write( power_arr        , std::string("power_"       )+std::to_string(iturb) );
            }
            coupler.get_parallel_comm().barrier();
            // Clear the trace data after writing
            turbine.force_axial_trace .clear();
            turbine.force_tang_trace  .clear();
            turbine.inflow_axial_trace.clear();
            turbine.inflow_tang_trace .clear();
            turbine.power_trace       .clear();
          }
          nc.end_indep_data();  // End independent data section
        }
        trace_size = 0;  // Reset trace size to zero after writing
      });
    }


    // Apply the turbine actuator disc forces and yaw updates for all turbines, accumulating tendencies from
    //   thrust and torque forces. Keep traces of the power, yaw angle, and inflow wind speed normal to the turbine plane.
    // Injects a portion of the unused thrust energy back into the flow as SGS/unresolved TKE.
    void apply( core::Coupler & coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx                = coupler.get_nx   ();  // Get the local number of x-direction cells
      auto ny                = coupler.get_ny   ();  // Get the local number of y-direction cells
      auto nz                = coupler.get_nz   ();  // Get the number of z-direction cells
      auto dx                = coupler.get_dx   ();  // Get the grid spacing in the x-direction
      auto dy                = coupler.get_dy   ();  // Get the grid spacing in the y-direction
      auto dz                = coupler.get_dz   ().createHostCopy()(0);  // Get the grid spacing in the z-direction (1-D array of size nz)
      auto i_beg             = coupler.get_i_beg();  // Get the beginning x-direction index for this MPI task
      auto j_beg             = coupler.get_j_beg();  // Get the beginning y-direction index for this MPI task
      auto &dm               = coupler.get_data_manager_readwrite();  // Get the data manager for read/write access
      auto dm_rho_d          = dm.get<real const,3>("density_dry"   );  // Dry density
      auto dm_uvel           = dm.get<real      ,3>("uvel"          );  // u-velocity
      auto dm_vvel           = dm.get<real      ,3>("vvel"          );  // v-velocity
      auto dm_wvel           = dm.get<real      ,3>("wvel"          );  // w-velocity
      auto dm_tend_u         = dm.get<real      ,3>("turbine_tend_u");
      auto dm_tend_v         = dm.get<real      ,3>("turbine_tend_v");
      auto dm_tend_w         = dm.get<real      ,3>("turbine_tend_w");
      auto B                 = ref_turbine.B                ;
      auto R                 = ref_turbine.R                ;
      auto R_hub             = ref_turbine.R_hub            ;
      auto H                 = ref_turbine.H                ;
      auto overhang          = ref_turbine.overhang         ;
      auto hub_radius        = ref_turbine.hub_radius       ;
      auto hub_flange_height = ref_turbine.hub_flange_height;
      auto tower_base_rad    = ref_turbine.tower_base_rad   ;
      auto tower_top_rad     = ref_turbine.tower_top_rad    ;
      auto shaft_tilt        = ref_turbine.shaft_tilt       ;
      auto host_rwt_mag      = ref_turbine.host_rwt_mag     ;
      auto host_rwt_rot      = ref_turbine.host_rwt_rot     ;
      auto host_rad_locs     = ref_turbine.host_rad_locs    ;
      auto host_foil_mid     = ref_turbine.host_foil_mid    ;
      auto host_foil_twist   = ref_turbine.host_foil_twist  ;
      auto host_foil_chord   = ref_turbine.host_foil_chord  ;
      auto host_foil_id      = ref_turbine.host_foil_id     ;
      auto host_foil_alpha   = ref_turbine.host_foil_alpha  ;
      auto host_foil_clift   = ref_turbine.host_foil_clift  ;
      auto host_foil_cdrag   = ref_turbine.host_foil_cdrag  ;
      auto dev_rad_locs      = ref_turbine.dev_rad_locs     ;
      auto dev_foil_mid      = ref_turbine.dev_foil_mid     ;
      auto dev_foil_chord    = ref_turbine.dev_foil_chord   ;
      auto U_inf             = coupler.get_option<real>("turbine_inflow_mag");
      auto gen_eff           = coupler.get_option<real>("turbine_gen_eff");
      auto max_power         = coupler.get_option<real>("turbine_max_power");
      auto omega             = coupler.get_option<real>("turbine_omega_rad_sec",-999);
      auto eps_fixed         = coupler.get_option<real>("turbine_eps_fixed",-1);
      int  nrad              = host_rad_locs.size();

      if (omega == -999) omega = linear_interp( host_rwt_mag , host_rwt_rot , U_inf , true );
      real cos_tlt = std::cos(shaft_tilt);
      real sin_tlt = std::sin(shaft_tilt);
      real pi_3_2 = std::pow(M_PI,3./2.);

      // This integrates the function: std::exp(-(x*x+y*y+z*z)/(eps*eps))/(pi_3_2*eps*eps*eps)
      auto int_proj_3d = KOKKOS_LAMBDA (real x1, real x2, real y1, real y2, real z1, real z2, real eps) -> real {
        real erf_x1 = std::erf(x1/eps);
        real erf_x2 = std::erf(x2/eps);
        real erf_y1 = std::erf(y1/eps);
        real erf_y2 = std::erf(y2/eps);
        real erf_z1 = std::erf(z1/eps);
        real erf_z2 = std::erf(z2/eps);
        return -1./8.*((erf_x1 - erf_x2)*erf_y1 - (erf_x1 - erf_x2)*erf_y2)*erf_z1 +
                1./8.*((erf_x1 - erf_x2)*erf_y1 - (erf_x1 - erf_x2)*erf_y2)*erf_z2;
      };

      // Allocate tendency arrays for u, v, w and initialize to zero
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        dm_tend_u(k,j,i) = 0;
        dm_tend_v(k,j,i) = 0;
        dm_tend_w(k,j,i) = 0;
      });

      for (int iturb = 0; iturb < turbine_group.turbines.size(); iturb++) {  // Loop over all turbines
        auto &turbine = turbine_group.turbines.at(iturb);  // Grab a reference to this turbine for convenience
        if (turbine.active) {  // If this MPI task is potentially involved with the turbine
          real base_x    = turbine.base_loc_x;  // Tower base x-location
          real base_y    = turbine.base_loc_y;  // Tower base y-location
          real rot_angle = turbine.rot_angle;   // Rotation angle about the hub
          ////////////////////////////////////////////////////////////////
          // Sample inflow wind speed ahead of each turbine blade point 
          ////////////////////////////////////////////////////////////////
          int constexpr ID_AXIAL = 0;  // Axial inflow wind speed
          int constexpr ID_TANG  = 1;  // Tangential inflow wind speed
          int constexpr ID_RHO   = 2;  // Density
          real3d inflow_props("inflow_props",3,B,nrad);
          inflow_props = 0;
          real max_chord = yakl::intrinsics::maxval(host_foil_chord);
          real max_eps   = eps_fixed > 0 ? eps_fixed : std::max( max_chord/2 , 2*dx );
          int num_z = (int) std::ceil(max_eps/dz*4);
          int num_y = (int) std::ceil(max_eps/dy*4);
          int num_x = (int) std::ceil(max_eps/dx*4);
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(B,nrad,2*num_z,2*num_y,2*num_x) ,
                                            KOKKOS_LAMBDA (int iblade, int irad, int kk, int jj, int ii) {
            // Get center point for this Gaussian point projection
            real x      = 0;
            real y      = 0;
            real z      = dev_rad_locs(irad);
            real bl_ang = rot_angle + 2*M_PI*((real)iblade)/((real)B);
            real hub_x  = base_x + overhang;
            real hub_y  = base_y;
            real hub_z  = H;
            real x_rot  = x;
            real y_rot  = std::cos(bl_ang)*y - std::sin(bl_ang)*z;
            real z_rot  = std::sin(bl_ang)*y + std::cos(bl_ang)*z;
            real x_tilt =  cos_tlt*x_rot + sin_tlt*z_rot;
            real y_tilt =  y_rot;
            real z_tilt = -sin_tlt*x_rot + cos_tlt*z_rot;
            real x0     = hub_x + x_tilt;
            real y0     = hub_y + y_tilt;
            real z0     = hub_z + z_tilt;
            int  i0     = static_cast<int>(std::floor(x0/dx))-i_beg;
            int  j0     = static_cast<int>(std::floor(y0/dy))-j_beg;
            int  k0     = static_cast<int>(std::floor(z0/dz));
            int  ti     = i0-num_x+ii;
            int  tj     = j0-num_y+jj;
            int  tk     = k0-num_z+kk;
            if ( ti >= 0 && ti < nx && tj >= 0 && tj < ny && tk >= 0 && tk < nz) {
              // Get chord length and epsilon
              real c     = linear_interp( dev_foil_mid , dev_foil_chord , dev_rad_locs(irad) , true );
              real eps   = eps_fixed > 0 ? eps_fixed : std::max( c/2 , (real)(2*dx) );
              real proj  = int_proj_3d( (i_beg+ti)*dx-x0 , (i_beg+ti+1)*dx-x0 ,
                                        (j_beg+tj)*dy-y0 , (j_beg+tj+1)*dy-y0 ,
                                        (      tk)*dz-z0 , (      tk+1)*dz-z0 , eps );
              real rho   = dm_rho_d(tk,tj,ti);
              real u     = dm_uvel (tk,tj,ti);
              real v     = dm_vvel (tk,tj,ti);
              real w     = dm_wvel (tk,tj,ti);
              real az    = std::atan2(z_rot,-y_rot);
              real tang  = -(w*cos_tlt + u*sin_tlt)*std::cos(az) - v*std::sin(az);
              real axial = u*cos_tlt - w*sin_tlt;
              Kokkos::atomic_add( &(inflow_props(ID_AXIAL,iblade,irad)) , proj*axial);
              Kokkos::atomic_add( &(inflow_props(ID_TANG ,iblade,irad)) , proj*tang );
              Kokkos::atomic_add( &(inflow_props(ID_RHO  ,iblade,irad)) , proj*rho  );
            }
          });
          inflow_props = turbine.par_comm.all_reduce( inflow_props , MPI_SUM , "sum1" );
          auto inflow_props_host = inflow_props.createHostCopy();
          realHost2d inflow_axial("inflow_axial",B,nrad);
          realHost2d inflow_tang ("inflow_tang" ,B,nrad);
          inflow_props_host.slice<2>(ID_AXIAL,0,0).deep_copy_to(inflow_axial);
          inflow_props_host.slice<2>(ID_TANG ,0,0).deep_copy_to(inflow_tang );
          turbine.inflow_axial_trace.push_back(inflow_axial);
          turbine.inflow_tang_trace .push_back(inflow_tang );
          /////////////////////////////////////////////////////////////////////////////
          // Compute thrust and torque at each blade point using blade element theory
          /////////////////////////////////////////////////////////////////////////////
          int        nseg = host_foil_id.size();
          realHost2d host_force_axial("force_axial",B,nrad);
          realHost2d host_force_tang ("force_tang" ,B,nrad);
          real       total_power = 0;
          for (int iblade = 0; iblade < B; iblade++) {
            for (int irad = 0; irad < nrad; irad++) {
              real r     = host_rad_locs(irad);
              real dr    = (R-R_hub)/nrad;
              real twist = linear_interp(host_foil_mid,host_foil_twist,r,true);
              real chord = linear_interp(host_foil_mid,host_foil_chord,r,true);
              real mult1, mult2;
              realHost1d ref_alpha1;
              realHost1d ref_clift1;
              realHost1d ref_cdrag1;
              realHost1d ref_alpha2;
              realHost1d ref_clift2;
              realHost1d ref_cdrag2;
              if (r <= host_foil_mid(0)) {
                mult1 = 0.5;
                mult2 = 0.5;
                ref_alpha1 = host_foil_alpha(host_foil_id(0));
                ref_clift1 = host_foil_clift(host_foil_id(0));
                ref_cdrag1 = host_foil_cdrag(host_foil_id(0));
                ref_alpha2 = host_foil_alpha(host_foil_id(0));
                ref_clift2 = host_foil_clift(host_foil_id(0));
                ref_cdrag2 = host_foil_cdrag(host_foil_id(0));
              } else if (r >= host_foil_mid(nseg-1)) {
                mult1 = 0.5;
                mult2 = 0.5;
                ref_alpha1 = host_foil_alpha(host_foil_id(nseg-1));
                ref_clift1 = host_foil_clift(host_foil_id(nseg-1));
                ref_cdrag1 = host_foil_cdrag(host_foil_id(nseg-1));
                ref_alpha2 = host_foil_alpha(host_foil_id(nseg-1));
                ref_clift2 = host_foil_clift(host_foil_id(nseg-1));
                ref_cdrag2 = host_foil_cdrag(host_foil_id(nseg-1));
              } else {
                int iseg1 = 0;
                for (int i=0; i < nseg-1; i++) { if (r >= host_foil_mid(i) && r <= host_foil_mid(i+1)) { iseg1 = i; break; } }
                int iseg2 = iseg1+1;
                mult1 = (host_foil_mid(iseg2) - r)/(host_foil_mid(iseg2)-host_foil_mid(iseg1));
                mult2 = (r - host_foil_mid(iseg1))/(host_foil_mid(iseg2)-host_foil_mid(iseg1));
                ref_alpha1 = host_foil_alpha(host_foil_id(iseg1));
                ref_clift1 = host_foil_clift(host_foil_id(iseg1));
                ref_cdrag1 = host_foil_cdrag(host_foil_id(iseg1));
                ref_alpha2 = host_foil_alpha(host_foil_id(iseg2));
                ref_clift2 = host_foil_clift(host_foil_id(iseg2));
                ref_cdrag2 = host_foil_cdrag(host_foil_id(iseg2));
              }
              real theta     = twist + turbine.pitch;               // Blade section angle (twist + pitch) (rad)
              real U_tang    = omega*r - inflow_props_host(ID_TANG,iblade,irad);
              real U_ax      = inflow_props_host(ID_AXIAL,iblade,irad);
              real W         = std::sqrt(U_ax*U_ax + U_tang*U_tang);
              real phi       = std::atan2( U_ax , U_tang );
              real alpha     = phi - theta;
              real Cl1       = linear_interp( ref_alpha1 , ref_clift1 , alpha/M_PI*180. ,true ); // Coefficient of lift
              real Cd1       = linear_interp( ref_alpha1 , ref_cdrag1 , alpha/M_PI*180. ,true ); // Coefficient of drag
              real Cl2       = linear_interp( ref_alpha2 , ref_clift2 , alpha/M_PI*180. ,true ); // Coefficient of lift
              real Cd2       = linear_interp( ref_alpha2 , ref_cdrag2 , alpha/M_PI*180. ,true ); // Coefficient of drag
              real Cl        = mult1*Cl1 + mult2*Cl2;
              real Cd        = mult1*Cd1 + mult2*Cd2;
              real Cn        = Cl * std::cos(phi) + Cd * std::sin(phi);
              real Ct        = Cl * std::sin(phi) - Cd * std::cos(phi);
              real dT_dr     = 0.5 * inflow_props_host(ID_RHO,iblade,irad) * W*W * chord * Cn;
              real dQ_dr     = 0.5 * inflow_props_host(ID_RHO,iblade,irad) * W*W * chord * Ct * r;
              real d         = 0.95;
              real x         = std::clamp(r/R,d,(real)1);
              real tip_decay = (3*(d+1)*x*x-2*x*x*x- 6*d*x+3*d-1)/(d*d*d-3*d*d+3*d-1);
              host_force_axial(iblade,irad) = dT_dr*dr              *tip_decay;
              host_force_tang (iblade,irad) = dQ_dr*dr/r            *tip_decay;
              total_power                  += dQ_dr*dr*omega*gen_eff*tip_decay;
            }
          }
          turbine.power_trace      .push_back(total_power     );
          turbine.force_axial_trace.push_back(host_force_axial);
          turbine.force_tang_trace .push_back(host_force_tang );
          ////////////////////////////////////////////////////////////////
          // Change blade pitch to reach max power if possible
          ////////////////////////////////////////////////////////////////
          real pitch_rate = 0.1; // radians per second
          if (total_power > max_power) { turbine.pitch += pitch_rate * dt; }
          if (total_power < max_power) { turbine.pitch -= pitch_rate * dt; }
          turbine.pitch = std::max(turbine.pitch,(real)0); // Don't let pitch get negative
          ////////////////////////////////////////////////////////////////
          // Apply thrust and torque to the flow for each blade
          ////////////////////////////////////////////////////////////////
          auto force_axial = host_force_axial.createDeviceCopy();
          auto force_tang  = host_force_tang .createDeviceCopy();
          num_z = (int) std::ceil(max_eps/dz*4);
          num_y = (int) std::ceil(max_eps/dy*4);
          num_x = (int) std::ceil(max_eps/dx*4);
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(B,nrad,2*num_z,2*num_y,2*num_x) ,
                                            KOKKOS_LAMBDA (int iblade, int irad, int kk, int jj, int ii) {
            // Get center point for this Gaussian point projection
            real x      = 0;
            real y      = 0;
            real z      = dev_rad_locs(irad);
            real bl_ang = rot_angle + 2*M_PI*((real)iblade)/((real)B);
            real hub_x  = base_x + overhang;
            real hub_y  = base_y;
            real hub_z  = H;
            real x_rot  = x;
            real y_rot  = std::cos(bl_ang)*y - std::sin(bl_ang)*z;
            real z_rot  = std::sin(bl_ang)*y + std::cos(bl_ang)*z;
            real x_tilt =  cos_tlt*x_rot + sin_tlt*z_rot;
            real y_tilt =  y_rot;
            real z_tilt = -sin_tlt*x_rot + cos_tlt*z_rot;
            real x0     = hub_x + x_tilt;
            real y0     = hub_y + y_tilt;
            real z0     = hub_z + z_tilt;
            int  i0     = static_cast<int>(std::floor(x0/dx))-i_beg;
            int  j0     = static_cast<int>(std::floor(y0/dy))-j_beg;
            int  k0     = static_cast<int>(std::floor(z0/dz));
            int  ti     = i0-num_x+ii;
            int  tj     = j0-num_y+jj;
            int  tk     = k0-num_z+kk;
            if ( ti >= 0 && ti < nx && tj >= 0 && tj < ny && tk >= 0 && tk < nz) {
              real c       = linear_interp( dev_foil_mid , dev_foil_chord , dev_rad_locs(irad) , true );
              real eps     = eps_fixed > 0 ? eps_fixed : std::max( c/2 , (real)(2*dx) );
              real proj    = int_proj_3d( (i_beg+ti)*dx-x0 , (i_beg+ti+1)*dx-x0 ,
                                          (j_beg+tj)*dy-y0 , (j_beg+tj+1)*dy-y0 ,
                                          (      tk)*dz-z0 , (      tk+1)*dz-z0 , eps );
              real F_axial = force_axial(iblade,irad)/inflow_props(ID_RHO,iblade,irad);
              real F_tang  = force_tang (iblade,irad)/inflow_props(ID_RHO,iblade,irad);
              real tend_u  = -F_axial*cos_tlt/(dx*dy*dz);
              real tend_v  =  0;
              real tend_w  =  F_axial*sin_tlt/(dx*dy*dz);
              real az      = std::atan2(z_rot,-y_rot);
              tend_u      +=  F_tang*std::cos(az)*sin_tlt/(dx*dy*dz);
              tend_v      +=  F_tang*std::sin(az)/(dx*dy*dz);
              tend_w      +=  F_tang*std::cos(az)*cos_tlt/(dx*dy*dz);
              Kokkos::atomic_add( &(dm_tend_u(tk,tj,ti)) , proj*tend_u );
              Kokkos::atomic_add( &(dm_tend_v(tk,tj,ti)) , proj*tend_v );
              Kokkos::atomic_add( &(dm_tend_w(tk,tj,ti)) , proj*tend_w );
            }
          });
          turbine.rot_angle -= omega*dt;
        } // if (turbine.active)
      } // for (int iturb = 0; iturb < turbine_group.turbines.size(); iturb++)

      // Apply accumulated tendencies to the flow field
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        dm_uvel(k,j,i) += dt * dm_tend_u(k,j,i);
        dm_vvel(k,j,i) += dt * dm_tend_v(k,j,i);
        dm_wvel(k,j,i) += dt * dm_tend_w(k,j,i);
      });

      for (int iturb = 0; iturb < turbine_group.turbines.size(); iturb++) {  // Loop over all turbines
        auto &turbine = turbine_group.turbines.at(iturb);  // Grab a reference to this turbine for convenience
        if (turbine.active) {  // If this MPI task is potentially involved with the turbine
          real base_x = turbine.base_loc_x;  // Tower base x-location
          real base_y = turbine.base_loc_y;  // Tower base y-location
          if (coupler.get_option<bool>("turbine_immerse_material",false)) {
            int constexpr N = 3;
            real ov  = overhang;
            real hr  = R_hub;
            real fh  = hub_flange_height;
            real bx  = base_x;
            real by  = base_y;
            real s0x = bx + ov;
            real s0y = by;
            real s0z = H;
            real tower_top = H - fh/2;
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
              real imm = 0;
              for (int kk=0; kk < N; kk++) {
                for (int jj=0; jj < N; jj++) {
                  for (int ii=0; ii < N; ii++) {
                    real x   = (i_beg+i)*dx + (ii+0.5)*dx/N;
                    real y   = (j_beg+j)*dy + (jj+0.5)*dy/N;
                    real z   = (      k)*dz + (kk+0.5)*dz/N;
                    real xp  = x-bx;
                    real yp  = y-by;
                    real zp  = z-H;
                    real rad = tower_base_rad + (tower_top_rad-tower_base_rad)*(z/tower_top);
                    if ( ((x-s0x)*(x-s0x) + (y-s0y)*(y-s0y) + (z-s0z)*(z-s0z) < hr*hr) ||                   // Hub (sphere)
                         (xp > ov/2 && xp < -ov/2 && yp > -fh/2 && yp < fh/2 && zp > -fh/2 && zp < fh/2) || // Hub Flange (hexahedron)
                         (x-bx)*(x-bx) + (y-by)*(y-by) <= rad*rad  && z <= tower_top  ) {                   // Tower
                      imm += 1;
                    }
                  }
                }
              }
              imm /= (N*N*N);
              real mult = imm*imm*imm*imm*imm;  // imm^5
              dm_uvel(k,j,i) += (0-dm_uvel(k,j,i))*mult;
              dm_vvel(k,j,i) += (0-dm_vvel(k,j,i))*mult;
              dm_wvel(k,j,i) += (0-dm_wvel(k,j,i))*mult;
            });
          }
        }
      }

      trace_size++;
    }

  };

}



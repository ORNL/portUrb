
#pragma once

#include "main_header.h"
#include "coupler.h"
#include "Betti_simplified.h"

namespace modules {

  // Uses disk actuators to represent wind turbines in an LES model by applying friction terms to horizontal
  //   velocities and adding a portion of the thrust not generating power to TKE.
  struct WindmillActuators {

    typedef real F;


    // Stores information needed to imprint a turbine actuator disk onto the grid.
    struct RefTurbine {
      // Reference wind turbine (RWT) tables
      realHost1d velmag_host;       // Velocity magnitude at infinity (m/s)
      realHost1d thrust_coef_host;  // Thrust coefficient             (dimensionless)
      realHost1d power_coef_host;   // Power coefficient              (dimensionless)
      realHost1d power_host;        // Power generation               (MW)
      realHost1d rotation_host;     // Rotation speed                 (radians / sec)
      // Turbine properties
      real       hub_height;        // Hub height                     (m)
      real       blade_radius;      // Blade radius                   (m)
      real       max_yaw_speed;     // Angular active yawing speed    (radians / sec)
      real       overhang;          // Offset of blades from tower center (m)
                                    // This is also the length of the hub flange
      real       hub_radius;        // Radius of the hub, where there is no blade (m)
      real       hub_flange_height; // Height (and width) of the hub flange (m)
      real       tower_base_rad;    // Radius of the tower base at ground or water level (m)
      real       tower_top_rad;     // Radius of the tower top connected to hub flange (m)
      real       shaft_tilt;        // Shaft tilt in radians
      void init( std::string fname ) {
        YAML::Node config = YAML::LoadFile( fname );
        if ( !config ) { endrun("ERROR: Invalid turbine input file"); }
        auto velmag_vec      = config["velocity_magnitude"].as<std::vector<real>>();
        auto thrust_coef_vec = config["thrust_coef"       ].as<std::vector<real>>();
        auto power_coef_vec  = config["power_coef"        ].as<std::vector<real>>();
        auto power_vec       = config["power_megawatts"   ].as<std::vector<real>>();
        bool do_blades_loc   = false;
        if ( config["rotation_rpm"] ) do_blades_loc = true;
        auto rotation_vec = do_blades_loc ? config["rotation_rpm"].as<std::vector<real>>() : std::vector<real>();
        // Allocate YAKL arrays to ensure the data is contiguous and to load into the data manager later
        velmag_host      = realHost1d("velmag"     ,velmag_vec     .size());
        thrust_coef_host = realHost1d("thrust_coef",thrust_coef_vec.size());
        power_coef_host  = realHost1d("power_coef" ,power_coef_vec .size());
        power_host       = realHost1d("power"      ,power_vec      .size());
        if (do_blades_loc) rotation_host = realHost1d("rotation",rotation_vec.size());
        // Make sure the sizes match
        if ( velmag_host.size() != thrust_coef_host.size() ||
             velmag_host.size() != power_coef_host .size() ||
             velmag_host.size() != power_host      .size() ||
             (do_blades_loc && (velmag_host.size() != rotation_host.size())) ) {
          Kokkos::abort("ERROR: turbine arrays not all the same size");
        }
        // Move from std::vectors into YAKL arrays
        for (int i=0; i < velmag_host.size(); i++) {
          velmag_host     (i) = velmag_vec     .at(i);
          thrust_coef_host(i) = thrust_coef_vec.at(i);
          power_coef_host (i) = power_coef_vec .at(i);
          power_host      (i) = power_vec      .at(i);
          if (do_blades_loc) rotation_host(i) = rotation_vec.at(i)*2*M_PI/60; // Convert from rpm to radians/sec
        }
        // Copy from host to device and set other parameters
        this->hub_height        = config["hub_height"       ].as<real>();
        this->blade_radius      = config["blade_radius"     ].as<real>();
        this->max_yaw_speed     = config["max_yaw_speed"    ].as<real>(0.5)/180.*M_PI; // Convert from deg/sec to rad/sec
        this->overhang          = config["overhang"         ].as<real>(-0.1 *blade_radius);
        this->hub_radius        = config["hub_radius"       ].as<real>(0.03*blade_radius);
        this->hub_flange_height = config["hub_flange_height"].as<real>(0.04*blade_radius);
        this->tower_base_rad    = config["tower_base_radius"].as<real>(5);
        this->tower_top_rad     = config["tower_top_radius" ].as<real>(3);
        this->shaft_tilt        = config["shaft_tilt_deg"   ].as<real>(0)/180.*M_PI;
      }
    };


    // Yaw will change as if it were an active yaw system that moves at a certain max speed. It will react
    //   to some time average of the wind velocities. The operator() outputs the new yaw angle in radians.
    struct YawTend {
      real tau, uavg, vavg;
      YawTend( real tau_in=60 , real uavg_in=0, real vavg_in=0 ) { tau=tau_in; uavg=uavg_in; vavg=vavg_in; }
      real operator() ( real uvel , real vvel , real dt , real yaw , real max_yaw_speed ) {
        // Update the moving average by weighting according using time scale as inertia
        uavg = (tau-dt)/tau*uavg + dt/tau*uvel;
        vavg = (tau-dt)/tau*vavg + dt/tau*vvel;
        // atan2 gives [-pi,pi] with zero representing moving toward the east
        // But we're using a coordinate system rotated by pi such that zero faces west.
        // That is, we're using an "upwind" coordinate system
        real dir_upwind = std::atan2(vavg,uavg);
        // Compute difference between time-averaged upwind direction and current yaw
        real diff = dir_upwind - yaw;
        if (diff >  M_PI) diff -= 2*M_PI;
        if (diff < -M_PI) diff += 2*M_PI;
        // Limit to the max yaw speed of the turbine
        real tend = diff / dt;
        if (tend > 0) { tend = std::min(  max_yaw_speed , tend ); }
        else          { tend = std::max( -max_yaw_speed , tend ); }
        // Return the new yaw angle
        return yaw+dt*tend;
      }
    };


    // Holds information about a turbine (location, reference_type, yaw, etc)
    struct Turbine {
      bool                    active;            // Whether this turbine affects this MPI task
      real                    base_loc_x;        // x location of the tower base
      real                    base_loc_y;        // y location of the tower base
      std::vector<real>       power_trace;       // Time trace of power generation
      std::vector<real>       yaw_trace;         // Time trace of yaw of the turbine
      std::vector<real>       u_samp_trace;      // Time trace of disk-integrated inflow u velocity
      std::vector<real>       v_samp_trace;      // Time trace of disk-integrated inflow v velocity
      std::vector<real>       mag195_trace;      // Time trace of disk-integrated 19.5m infoat velocity
      std::vector<real>       betti_trace;       // Time trace of floating motions perturbations
      std::vector<real>       surge_pos_trace;   // Time trace of floating surge position
      std::vector<real>       surge_vel_trace;   // Time trace of floating surge velocity
      std::vector<real>       heave_pos_trace;   // Time trace of floating heave position
      std::vector<real>       heave_vel_trace;   // Time trace of floating heave velocity
      std::vector<real>       pitch_pos_trace;   // Time trace of floating pitch position
      std::vector<real>       pitch_vel_trace;   // Time trace of floating pitch velocity
      std::vector<real>       cp_trace;          // Time trace of coefficient of power
      std::vector<real>       ct_trace;          // Time trace of coefficient of thrust
      real                    u_samp_inertial;   // Intertial inflow u-velocity normal to the turbine plane
      real                    v_samp_inertial;   // Intertial inflow u-velocity normal to the turbine plane
      real                    yaw_angle;         // Current yaw angle (radians counter-clockwise from facing west)
      real                    rot_angle;         // Current rotation angle (radians)
      YawTend                 yaw_tend;          // Functor to compute the change in yaw
      RefTurbine              ref_turbine;       // The reference turbine to use for this turbine
      core::ParallelComm      par_comm;          // MPI communicator for this turbine
      int                     nranks;            // Number of MPI ranks involved with this turbine
      int                     sub_rankid;        // My process's rank ID in the sub communicator
      int                     owning_sub_rankid; // Subcommunicator rank ID of the owner of this turbine
      bool                    apply_thrust;      // Whether to apply the thrust to the simulation or not
      Floating_motions_betti  floating_motions;  // Class to handle floating motions due to waves, thrust, etc
    };


    struct TurbineGroup {
      std::vector<Turbine> turbines;
      void add_turbine( core::Coupler       & coupler     ,
                        real                  base_loc_x  ,
                        real                  base_loc_y  ,
                        RefTurbine    const & ref_turbine ,
                        bool                  apply_thrust = true ) {
        using yakl::c::parallel_for;
        using yakl::c::SimpleBounds;
        auto i_beg  = coupler.get_i_beg();
        auto j_beg  = coupler.get_j_beg();
        auto nx     = coupler.get_nx();
        auto ny     = coupler.get_ny();
        auto nz     = coupler.get_nz();
        auto dx     = coupler.get_dx();
        auto dy     = coupler.get_dy();
        auto dz     = coupler.get_dz();
        auto zint   = coupler.get_zint();
        auto zmid   = coupler.get_zmid();
        auto myrank = coupler.get_myrank();
        auto imm    = coupler.get_data_manager_readwrite().get<real,3>("immersed_proportion");
        auto imm_h  = coupler.get_data_manager_readwrite().get<real,3>("immersed_proportion_halos");
        // bounds of this MPI task's domain
        real dom_x1  = (i_beg+0 )*dx;
        real dom_x2  = (i_beg+nx)*dx;
        real dom_y1  = (j_beg+0 )*dy;
        real dom_y2  = (j_beg+ny)*dy;
        // Rectangular bounds of this turbine's potential influence
        real turb_x1 = base_loc_x-6*ref_turbine.blade_radius-6*std::sqrt(dx*dy);
        real turb_x2 = base_loc_x+6*ref_turbine.blade_radius+6*std::sqrt(dx*dy);
        real turb_y1 = base_loc_y-6*ref_turbine.blade_radius-6*std::sqrt(dx*dy);
        real turb_y2 = base_loc_y+6*ref_turbine.blade_radius+6*std::sqrt(dx*dy);
        // Determine if the two domains overlap
        bool active = !( turb_x1 > dom_x2 || // Turbine's to the right
                         turb_x2 < dom_x1 || // Turbine's to the left
                         turb_y1 > dom_y2 || // Turbine's above
                         turb_y2 < dom_y1 ); // Turbine's below
        Turbine loc;
        loc.active          = active;
        loc.base_loc_x      = base_loc_x;
        loc.base_loc_y      = base_loc_y;
        loc.yaw_angle       = coupler.get_option<real>("turbine_initial_yaw",0);
        loc.rot_angle       = 0.;
        loc.yaw_tend        = YawTend();
        loc.ref_turbine     = ref_turbine;
        loc.u_samp_inertial = 0;
        loc.v_samp_inertial = 0;
        loc.apply_thrust    = apply_thrust;
        // loc.floating_motions.init("./inputs/Betti_NREL_5MW.nc");
        loc.floating_motions.init(ref_turbine.shaft_tilt);
        loc.par_comm.create( active , coupler.get_parallel_comm().get_mpi_comm() );
        if (active) {
          // Get subcommunicator size and rank id
          loc.nranks     = loc.par_comm.get_size();
          loc.sub_rankid = loc.par_comm.get_rank_id();
          // Determine if I "own" the turbine (if the hub's in my domain)
          bool owner = base_loc_x >= i_beg*dx && base_loc_x < (i_beg+nx)*dx &&
                       base_loc_y >= j_beg*dy && base_loc_y < (j_beg+ny)*dy ;
          // Gather who owns the turbine, so yaw angles can be broadcast later
          if ( loc.nranks == 1) {
            loc.owning_sub_rankid = 0;
          } else {
            bool owner = base_loc_x >= i_beg*dx && base_loc_x < (i_beg+nx)*dx &&
                         base_loc_y >= j_beg*dy && base_loc_y < (j_beg+ny)*dy ;
            auto owner_arr = loc.par_comm.all_gather( owner );
            for (int i=0; i < loc.nranks; i++) { if (owner_arr(i)) loc.owning_sub_rankid = i; }
          }
        } else {
          // Don't want comparisons to give true for any of these
          loc.nranks = -1;
          loc.sub_rankid = -2;
          loc.owning_sub_rankid = -3;
        }
        // Add the turbine
        turbines.push_back(loc);
        // Add the base to immersed_proportion
        if (loc.apply_thrust && coupler.get_option<bool>("turbine_immerse_material",false)) {
          real tower_top      = ref_turbine.hub_height - ref_turbine.hub_flange_height/2;
          real tower_base_rad = ref_turbine.tower_base_rad;
          real tower_top_rad  = ref_turbine.tower_top_rad ;
          int N = 10;
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
            int count = 0;
            for (int kk=0; kk < N; kk++) {
              for (int jj=0; jj < N; jj++) {
                for (int ii=0; ii < N; ii++) {
                  int x = (i_beg+i)*dx + ii*dx   /(N-1);
                  int y = (j_beg+j)*dy + jj*dy   /(N-1);
                  int z = zmid(k)      + kk*dz(k)/(N-1);
                  auto bx  = base_loc_x;
                  auto by  = base_loc_y;
                  auto rad = tower_base_rad + (tower_top_rad-tower_base_rad)*(z/tower_top);
                  if ( (x-bx)*(x-bx) + (y-by)*(y-by) <= rad*rad  && z <= tower_top ) count++;
                }
              }
            }
            // Express the base as an immersed boundary
            imm  (  k,  j,  i) += static_cast<real>(count)/(N*N*N);
            imm_h(1+k,1+j,1+i) += static_cast<real>(count)/(N*N*N);
          });
        }
      }
    };


    // Sagemath code producing the function used in DefaultThrustShape
    // def c_scalar(val,coeflab) :
    //     import re
    //     s = str(val).replace(' ','')
    //     s = re.sub("([a-zA-Z0-9_]*)\\^2","(\\1*\\1)",s,0,re.DOTALL)
    //     s = re.sub("([a-zA-Z0-9_]*)\\^3","(\\1*\\1*\\1)",s,0,re.DOTALL)
    //     return s
    // def coefs_1d(N,N0,lab) :
    //     return vector([ var(lab+'%s'%i) for i in range(N0,N0+N) ])
    // def poly_1d(N,coefs) :
    //     return sum( vector([ coefs[i]*x^i for i in range(N) ]) )
    // var('x2,x3,a')
    // coefs = coefs_1d(3,0,'a')
    // p = poly_1d(3,coefs)
    // constr = vector([p.subs(x=0),p.subs(x=x2),p.diff(x).subs(x=x2)])
    // p1 = poly_1d(3,(jacobian(constr,coefs)^-1)*vector([0,1,0]))
    // coefs = coefs_1d(4,0,'a')
    // p = poly_1d(4,coefs)
    // constr = vector([p.subs(x=x2),p.diff(x).subs(x=x2),p.subs(x=x3),p.diff(x).subs(x=x3)])
    // p2 = poly_1d(4,(jacobian(constr,coefs)^-1)*vector([1,0,0,0]))
    // print("p1 = pow(",c_scalar(p1.simplify_full(),'none'),", a );")
    // print("p2 = ",c_scalar(p2.simplify_full(),'none'),";")
    // x2 = 0.9;    x3 = 1;    a = 0.5
    // ( plot(p1.subs(x2=x2)^a,x,0 ,x2) + plot(p2.subs(x2=x2,x3=x3),x,x2,x3) ).show()
    // a = 0.5 reproduces: A comparison of actuator disk and actuator line wind turbine models and best practices for their use
    struct DefaultThrustShape {
      KOKKOS_INLINE_FUNCTION F operator() ( F x , F x2 = 0.9 , F x3 = 1.0 , F a = 2 ) const {
        using std::pow;
        if (x < x2) return pow(-1.0*((x*x)-2*x*x2)/(x2*x2),a);
        if (x < x3) return -1.0*(2*(x*x*x)-3*(x*x)*x2-3*x2*(x3*x3)+(x3*x3*x3)-3*((x*x)-2*x*x2)*x3)/((x2*x2*x2)-3*(x2*x2)*x3+3*x2*(x3*x3)-(x3*x3*x3));
        return 0;
      }
    };


    struct DefaultProjectionShape1D {
      KOKKOS_INLINE_FUNCTION F operator() ( F x , F xr , int p = 2 ) const {
        F term = 1-(x/xr)*(x/xr);
        if (term <= 0) return 0;
        F term_p = term;
        for (int i = 0; i < p-1; i++) { term_p *= term; }
        return term_p;
      }
    };


    // Class data members
    TurbineGroup  turbine_group;
    int           trace_size;
    int           sample_counter;


    void init( core::Coupler &coupler ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx   = coupler.get_nx  ();
      auto ny   = coupler.get_ny  ();
      auto nz   = coupler.get_nz  ();
      auto xlen = coupler.get_xlen();
      auto ylen = coupler.get_ylen();
      auto &dm  = coupler.get_data_manager_readwrite();

      trace_size = 0;
      
      RefTurbine ref_turbine;
      ref_turbine.init( coupler.get_option<std::string>("turbine_file") );
      if (coupler.option_exists("override_shaft_tilt_deg")) {
        ref_turbine.shaft_tilt = coupler.get_option<real>("override_shaft_tilt_deg");
      }

      int num_x = (int) std::round( xlen / 10 / (2*ref_turbine.blade_radius) );
      int num_y = (int) std::round( ylen / 10 / (2*ref_turbine.blade_radius) );
      real xinc = xlen/num_x;
      real yinc = ylen/num_y;
      // Determine the x and y bounds of this MPI task's domain
      if (coupler.option_exists("turbine_x_locs") && coupler.option_exists("turbine_y_locs")) {
        auto x_locs = coupler.get_option<std::vector<real>>("turbine_x_locs");
        auto y_locs = coupler.get_option<std::vector<real>>("turbine_y_locs");
        std::vector<bool> apply_thrust;
        apply_thrust.assign(x_locs.size(),true);
        if (coupler.option_exists("turbine_apply_thrust")) {
          apply_thrust = coupler.get_option<std::vector<bool>>("turbine_apply_thrust");
        }
        for (int iturb = 0; iturb < x_locs.size(); iturb++) {
          turbine_group.add_turbine( coupler , x_locs.at(iturb) , y_locs.at(iturb) , ref_turbine ,
                                     apply_thrust.at(iturb) );
        }
      } else {
        for (real y = yinc/2; y < ylen; y += yinc) {
          for (real x = xinc/2; x < xlen; x += xinc) {
            turbine_group.add_turbine( coupler , x , y , ref_turbine );
          }
        }
      }

      dm.register_and_allocate<real>("windmill_proj_weight","",{nz,ny,nx});
      coupler.register_output_variable<real>( "windmill_proj_weight" , core::Coupler::DIMS_3D );
      dm.register_and_allocate<real>("windmill_samp_weight","",{nz,ny,nx});
      coupler.register_output_variable<real>( "windmill_samp_weight" , core::Coupler::DIMS_3D );
      // Create an output module in the coupler to dump the windmill portions and the power trace from task zero
      coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        if (trace_size > 0) {
          nc.redef();
          nc.create_dim( "num_time_steps" , trace_size );
          for (int iturb=0; iturb < turbine_group.turbines.size(); iturb++) {
            std::string pow_vname       = std::string("power_trace_turb_"    ) + std::to_string(iturb);
            std::string yaw_vname       = std::string("yaw_trace_turb_"      ) + std::to_string(iturb);
            std::string u_samp_vname    = std::string("u_samp_trace_turb_"   ) + std::to_string(iturb);
            std::string v_samp_vname    = std::string("v_samp_trace_turb_"   ) + std::to_string(iturb);
            std::string mag195_vname    = std::string("mag195_trace_turb_"   ) + std::to_string(iturb);
            std::string betti_vname     = std::string("betti_trace_turb_"    ) + std::to_string(iturb);
            std::string surge_pos_vname = std::string("surge_pos_trace_turb_") + std::to_string(iturb);
            std::string surge_vel_vname = std::string("surge_vel_trace_turb_") + std::to_string(iturb);
            std::string heave_pos_vname = std::string("heave_pos_trace_turb_") + std::to_string(iturb);
            std::string heave_vel_vname = std::string("heave_vel_trace_turb_") + std::to_string(iturb);
            std::string pitch_pos_vname = std::string("pitch_pos_trace_turb_") + std::to_string(iturb);
            std::string pitch_vel_vname = std::string("pitch_vel_trace_turb_") + std::to_string(iturb);
            std::string cp_vname        = std::string("cp_trace_turb_"       ) + std::to_string(iturb);
            std::string ct_vname        = std::string("ct_trace_turb_"       ) + std::to_string(iturb);
            nc.create_var<real>( pow_vname       , {"num_time_steps"} );
            nc.create_var<real>( yaw_vname       , {"num_time_steps"} );
            nc.create_var<real>( u_samp_vname    , {"num_time_steps"} );
            nc.create_var<real>( v_samp_vname    , {"num_time_steps"} );
            nc.create_var<real>( mag195_vname    , {"num_time_steps"} );
            nc.create_var<real>( betti_vname     , {"num_time_steps"} );
            nc.create_var<real>( surge_pos_vname , {"num_time_steps"} );
            nc.create_var<real>( surge_vel_vname , {"num_time_steps"} );
            nc.create_var<real>( heave_pos_vname , {"num_time_steps"} );
            nc.create_var<real>( heave_vel_vname , {"num_time_steps"} );
            nc.create_var<real>( pitch_pos_vname , {"num_time_steps"} );
            nc.create_var<real>( pitch_vel_vname , {"num_time_steps"} );
            nc.create_var<real>( cp_vname        , {"num_time_steps"} );
            nc.create_var<real>( ct_vname        , {"num_time_steps"} );
          }
          nc.enddef();
          nc.begin_indep_data();
          for (int iturb=0; iturb < turbine_group.turbines.size(); iturb++) {
            auto &turbine = turbine_group.turbines.at(iturb);
            if (turbine.active && turbine.sub_rankid == turbine.owning_sub_rankid) {
              realHost1d power_arr    ("power_arr"    ,trace_size);
              realHost1d yaw_arr      ("yaw_arr"      ,trace_size);
              realHost1d u_samp_arr   ("u_samp_arr"   ,trace_size);
              realHost1d v_samp_arr   ("v_samp_arr"   ,trace_size);
              realHost1d mag195_arr   ("mag195_arr"   ,trace_size);
              realHost1d betti_arr    ("betti_arr"    ,trace_size);
              realHost1d surge_pos_arr("surge_pos_arr",trace_size);
              realHost1d surge_vel_arr("surge_vel_arr",trace_size);
              realHost1d heave_pos_arr("heave_pos_arr",trace_size);
              realHost1d heave_vel_arr("heave_vel_arr",trace_size);
              realHost1d pitch_pos_arr("pitch_pos_arr",trace_size);
              realHost1d pitch_vel_arr("pitch_vel_arr",trace_size);
              realHost1d cp_arr       ("cp_arr"       ,trace_size);
              realHost1d ct_arr       ("ct_arr"       ,trace_size);
              for (int i=0; i < trace_size; i++) { power_arr    (i) = turbine.power_trace    .at(i); }
              for (int i=0; i < trace_size; i++) { yaw_arr      (i) = turbine.yaw_trace      .at(i)/M_PI*180; }
              for (int i=0; i < trace_size; i++) { u_samp_arr   (i) = turbine.u_samp_trace   .at(i); }
              for (int i=0; i < trace_size; i++) { v_samp_arr   (i) = turbine.v_samp_trace   .at(i); }
              for (int i=0; i < trace_size; i++) { mag195_arr   (i) = turbine.mag195_trace   .at(i); }
              for (int i=0; i < trace_size; i++) { betti_arr    (i) = turbine.betti_trace    .at(i); }
              for (int i=0; i < trace_size; i++) { surge_pos_arr(i) = turbine.surge_pos_trace.at(i); }
              for (int i=0; i < trace_size; i++) { surge_vel_arr(i) = turbine.surge_vel_trace.at(i); }
              for (int i=0; i < trace_size; i++) { heave_pos_arr(i) = turbine.heave_pos_trace.at(i); }
              for (int i=0; i < trace_size; i++) { heave_vel_arr(i) = turbine.heave_vel_trace.at(i); }
              for (int i=0; i < trace_size; i++) { pitch_pos_arr(i) = turbine.pitch_pos_trace.at(i)/M_PI*180; }
              for (int i=0; i < trace_size; i++) { pitch_vel_arr(i) = turbine.pitch_vel_trace.at(i); }
              for (int i=0; i < trace_size; i++) { cp_arr       (i) = turbine.cp_trace       .at(i); }
              for (int i=0; i < trace_size; i++) { ct_arr       (i) = turbine.ct_trace       .at(i); }
              std::string pow_vname       = std::string("power_trace_turb_"    ) + std::to_string(iturb);
              std::string yaw_vname       = std::string("yaw_trace_turb_"      ) + std::to_string(iturb);
              std::string u_samp_vname    = std::string("u_samp_trace_turb_"   ) + std::to_string(iturb);
              std::string v_samp_vname    = std::string("v_samp_trace_turb_"   ) + std::to_string(iturb);
              std::string mag195_vname    = std::string("mag195_trace_turb_"   ) + std::to_string(iturb);
              std::string betti_vname     = std::string("betti_trace_turb_"    ) + std::to_string(iturb);
              std::string surge_pos_vname = std::string("surge_pos_trace_turb_") + std::to_string(iturb);
              std::string surge_vel_vname = std::string("surge_vel_trace_turb_") + std::to_string(iturb);
              std::string heave_pos_vname = std::string("heave_pos_trace_turb_") + std::to_string(iturb);
              std::string heave_vel_vname = std::string("heave_vel_trace_turb_") + std::to_string(iturb);
              std::string pitch_pos_vname = std::string("pitch_pos_trace_turb_") + std::to_string(iturb);
              std::string pitch_vel_vname = std::string("pitch_vel_trace_turb_") + std::to_string(iturb);
              std::string cp_vname        = std::string("cp_trace_turb_"       ) + std::to_string(iturb);
              std::string ct_vname        = std::string("ct_trace_turb_"       ) + std::to_string(iturb);
              nc.write( power_arr     , pow_vname       );
              nc.write( yaw_arr       , yaw_vname       );
              nc.write( u_samp_arr    , u_samp_vname    );
              nc.write( v_samp_arr    , v_samp_vname    );
              nc.write( mag195_arr    , mag195_vname    );
              nc.write( betti_arr     , betti_vname     );
              nc.write( surge_pos_arr , surge_pos_vname );
              nc.write( surge_vel_arr , surge_vel_vname );
              nc.write( heave_pos_arr , heave_pos_vname );
              nc.write( heave_vel_arr , heave_vel_vname );
              nc.write( pitch_pos_arr , pitch_pos_vname );
              nc.write( pitch_vel_arr , pitch_vel_vname );
              nc.write( cp_arr        , cp_vname        );
              nc.write( ct_arr        , ct_vname        );
            }
            coupler.get_parallel_comm().barrier();
            turbine.power_trace    .clear();
            turbine.yaw_trace      .clear();
            turbine.u_samp_trace   .clear();
            turbine.v_samp_trace   .clear();
            turbine.mag195_trace   .clear();
            turbine.betti_trace    .clear();
            turbine.surge_pos_trace.clear();
            turbine.surge_vel_trace.clear();
            turbine.heave_pos_trace.clear();
            turbine.heave_vel_trace.clear();
            turbine.pitch_pos_trace.clear();
            turbine.pitch_vel_trace.clear();
            turbine.cp_trace       .clear();
            turbine.ct_trace       .clear();
          }
          nc.end_indep_data();
        }
        trace_size = 0;
      });
      sample_counter = 0;
    }


    void apply( core::Coupler & coupler , F dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx              = coupler.get_nx   ();
      auto ny              = coupler.get_ny   ();
      auto nz              = coupler.get_nz   ();
      auto dx              = coupler.get_dx   ();
      auto dy              = coupler.get_dy   ();
      auto dz              = coupler.get_dz   ();
      auto zint            = coupler.get_zint ();
      auto zmid            = coupler.get_zmid();
      auto i_beg           = coupler.get_i_beg();
      auto j_beg           = coupler.get_j_beg();
      auto &dm             = coupler.get_data_manager_readwrite();
      auto rho_d           = dm.get<real const,3>("density_dry"  );
      auto uvel            = dm.get<real      ,3>("uvel"         );
      auto vvel            = dm.get<real      ,3>("vvel"         );
      auto wvel            = dm.get<real      ,3>("wvel"         );
      auto tke             = dm.get<real      ,3>("TKE"          );
      auto proj_weight_tot = dm.get<real      ,3>("windmill_proj_weight");
      auto samp_weight_tot = dm.get<real      ,3>("windmill_samp_weight");
      auto thrust_shape    = DefaultThrustShape();
      auto proj_shape_1d   = DefaultProjectionShape1D();

      yakl::Array<F,3,yakl::memDevice> tend_u  ("tend_u"  ,nz,ny,nx);
      yakl::Array<F,3,yakl::memDevice> tend_v  ("tend_v"  ,nz,ny,nx);
      yakl::Array<F,3,yakl::memDevice> tend_w  ("tend_w"  ,nz,ny,nx);
      yakl::Array<F,3,yakl::memDevice> tend_tke("tend_tke",nz,ny,nx);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        tend_u         (k,j,i) = 0;
        tend_v         (k,j,i) = 0;
        tend_w         (k,j,i) = 0;
        tend_tke       (k,j,i) = 0;
        proj_weight_tot(k,j,i) = 0;
        samp_weight_tot(k,j,i) = 0;
      });

      for (int iturb = 0; iturb < turbine_group.turbines.size(); iturb++) {
        auto &turbine = turbine_group.turbines.at(iturb);
        if (turbine.active) {
          ///////////////////////////////////////////////////
          // Sampling of turbine disk
          ///////////////////////////////////////////////////
          // Pre-compute rotation matrix terms
          F cos_yaw = std::cos(turbine.yaw_angle);
          F sin_yaw = std::sin(turbine.yaw_angle);
          F cos_tlt = std::cos(turbine.ref_turbine.shaft_tilt);
          F sin_tlt = std::sin(turbine.ref_turbine.shaft_tilt);
          // Get reference data for later computations
          F rad                 = turbine.ref_turbine.blade_radius    ; // Radius of the blade plane
          F hub_height          = turbine.ref_turbine.hub_height      ; // height of the hub
          F base_x              = turbine.base_loc_x;
          F base_y              = turbine.base_loc_y;
          F rot_angle           = turbine.rot_angle;
          auto  ref_velmag      = turbine.ref_turbine.velmag_host     ; // For interpolation
          auto  ref_thrust_coef = turbine.ref_turbine.thrust_coef_host; // For interpolation
          auto  ref_power_coef  = turbine.ref_turbine.power_coef_host ; // For interpolation
          auto  ref_power       = turbine.ref_turbine.power_host      ; // For interpolation
          auto  ref_rotation    = turbine.ref_turbine.rotation_host   ; // For interpolation
          bool  do_blades       = coupler.get_option<bool>("turbine_do_blades",true) &&
                                  ( ref_rotation.initialized() || coupler.option_exists("turbine_rot_fixed") ) &&
                                  ( dx*(63/rad) < 16 );
          F overhang            = turbine.ref_turbine.overhang;
          F hub_radius          = turbine.ref_turbine.hub_radius;
          F hub_flange_ht       = turbine.ref_turbine.hub_flange_height;
          F max_yaw_speed       = turbine.ref_turbine.max_yaw_speed;
          F decay               = 2*dx/rad; // Length of decay of thrust after the end of the blade radius (relative)

          // First, apply tendencies due to immersed hub & hub flange
          if (turbine.apply_thrust && coupler.get_option<bool>("turbine_immerse_material",false)) {
            int constexpr N = 10;
            F ov = -overhang;
            F hr =  hub_radius;
            F fh =  hub_flange_ht;
            F s0x = base_x - ov*cos_yaw;
            F s0y = base_y - ov*sin_yaw;
            F s0z = hub_height;
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
              F imm = 0;
              for (int kk=0; kk < N; kk++) {
                for (int jj=0; jj < N; jj++) {
                  for (int ii=0; ii < N; ii++) {
                    F x = (i_beg+i)*dx + ii*dx   /(N-1);
                    F y = (j_beg+j)*dy + jj*dy   /(N-1);
                    F z = zmid(k)      + kk*dz(k)/(N-1);
                    // Rotate in (-yaw) direction to compare against vanilla x and y
                    F cos_nyaw =  cos_yaw;
                    F sin_nyaw = -sin_yaw;
                    F xp = cos_nyaw*(x-base_x) - sin_nyaw*(y-base_y);
                    F yp = sin_nyaw*(x-base_x) + cos_nyaw*(y-base_y);
                    F zp = (z-hub_height);
                    // Hub (sphere)
                    if ( ((x-s0x)*(x-s0x) + (y-s0y)*(y-s0y) + (z-s0z)*(z-s0z) < hr*hr) ||                    // Hub (sphere)
                         (xp > -ov/2 && xp < ov/2 && yp > -fh/2 && yp < fh/2 && zp > -fh/2 && zp < fh/2) ) { // Hub Flange (hexahedron)
                      imm += 1;
                    }
                  }
                }
              }
              imm /= (N*N*N);
              F mult = imm*imm*imm*imm*imm;  // imm^5
              uvel(k,j,i) += (0-uvel(k,j,i))*mult;
              vvel(k,j,i) += (0-vvel(k,j,i))*mult;
              wvel(k,j,i) += (0-wvel(k,j,i))*mult;
            });
          }

          // Zero out disk weights for projection and sampling
          // Compute average winds in a 3-D tet around the turbine hub to compute upstream direction
          yakl::Array<F,3,yakl::memDevice> disk_weight_angle("disk_weight_angle",nz,ny,nx);
          yakl::Array<F,3,yakl::memDevice> disk_weight_proj ("disk_weight_proj" ,nz,ny,nx);
          yakl::Array<F,3,yakl::memDevice> disk_weight_samp ("disk_weight_samp" ,nz,ny,nx);
          yakl::Array<F,3,yakl::memDevice> blade_weight_proj("blade_weight_proj",nz,ny,nx);
          yakl::Array<F,3,yakl::memDevice> uvel_3d          ("uvel_3d"          ,nz,ny,nx);
          yakl::Array<F,3,yakl::memDevice> vvel_3d          ("vvel_3d"          ,nz,ny,nx);
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
            disk_weight_angle(k,j,i) = 0;
            disk_weight_proj (k,j,i) = 0;
            disk_weight_samp (k,j,i) = 0;
            blade_weight_proj(k,j,i) = 0;
            F x = (i_beg+i+0.5f)*dx;
            F y = (j_beg+j+0.5f)*dy;
            F z = zmid(k);
            if ( z >= hub_height-rad && z <= hub_height+rad &&
                 y >= base_y    -rad && y <= base_y    +rad &&
                 x >= base_x    -rad && x <= base_x    +rad ) {
              uvel_3d(k,j,i) = uvel(k,j,i);
              vvel_3d(k,j,i) = vvel(k,j,i);
            } else {
              uvel_3d(k,j,i) = 0;
              vvel_3d(k,j,i) = 0;
            }
          });
          yakl::SArray<F,1,2> weights_tot;
          weights_tot(0) = yakl::intrinsics::sum(uvel_3d);
          weights_tot(1) = yakl::intrinsics::sum(vvel_3d);
          weights_tot = turbine.par_comm.all_reduce( weights_tot , MPI_SUM , "windmill_Allreduce1" );
          F upstream_uvel = weights_tot(0);
          F upstream_vvel = weights_tot(1);
          // Compute upstream direction
          F upstream_dir;
          if (coupler.option_exists("turbine_upstream_dir")) {
            upstream_dir = coupler.get_option<real>("turbine_upstream_dir");
          } else {
            upstream_dir = std::atan2( upstream_vvel , upstream_uvel );  // theta=tan^-1(v/u)
          }
          // Compute upstream offset at two turbine diameters upstream based on wind direction at the turbine
          F upstream_x_offset = -4*rad*std::cos(upstream_dir);
          F upstream_y_offset = -4*rad*std::sin(upstream_dir);
          // Compute and sum weights for disk projection and upstream sampling projection
          yakl::Array<F,2,yakl::memDevice> umag_19_5m_2d("umag_19_5m_2d",ny,nx);
          {
            // Project disks
            // Reference space is centered about the origin with the turbine disk facing toward the west
            // Reference proj is rotated & translated to its base location, hub height, yaw angle, shaft tilt
            F xr = std::max(5.,5*dx);  // This is the thickness of the disk in the x-direction in reference space
            // Define number of cells in 3-D to sample over for reference space disk projection
            int num_x = std::ceil(20/dx*xr             *2);
            int num_y = std::ceil(20/dx*rad*(1+decay/2)*2);
            int num_z = std::ceil(20/dx*rad*(1+decay/2)*2);
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_z,num_y,num_x) , KOKKOS_LAMBDA (int k, int j, int i) {
              // Initial point in the y-z plane facing the negative x direction
              F x = -xr              + (2*xr             *i)/(num_x-1);
              F y = -rad*(1+decay/2) + (2*rad*(1+decay/2)*j)/(num_y-1);
              F z = -rad*(1+decay/2) + (2*rad*(1+decay/2)*k)/(num_z-1);
              F rloc = std::sqrt(y*y+z*z);  // radius of the point about the origin
              if (rloc <= rad*(1+decay/2)) {
                // Compute the 3-D shaping function for this point in reference space
                F shp = thrust_shape(rloc/rad,1-decay/2,1+decay/2,0.5)*proj_shape_1d(x,xr);
                // Rotate about y-axis for shaft tilt
                F x1 =  cos_tlt*(x+overhang) + sin_tlt*z;
                F y1 =  y;
                F z1 = -sin_tlt*(x+overhang) + cos_tlt*z;
                // Rotate about z-axis for yaw angle, and translate to base location
                F xp = base_x     + cos_yaw*x1 - sin_yaw*y1;
                F yp = base_y     + sin_yaw*x1 + cos_yaw*y1;
                F zp = hub_height + z1;
                // If it's in this task's domain, atomically add to the cell's total shape function sum for projection
                //     Also, add the average angle for torque application later
                int ti = static_cast<int>(std::floor(xp/dx))-i_beg;
                int tj = static_cast<int>(std::floor(yp/dy))-j_beg;
                int tk = 0;
                for (int kk=0; kk < nz; kk++) {
                  if (zp >= zint(kk) && zp < zint(kk+1)) {
                    tk = kk;
                    break;
                  }
                }
                if ( ti >= 0 && ti < nx && tj >= 0 && tj < ny && tk >= 0 && tk < nz) {
                  Kokkos::atomic_add( &disk_weight_angle(tk,tj,ti) , shp*std::atan2(z,-y) );
                  Kokkos::atomic_add( &disk_weight_proj (tk,tj,ti) , shp );
                }
                // Now do the same thing for the upwind sampling disk for computing inflow velocity
                xp += upstream_x_offset;
                yp += upstream_y_offset;
                ti = static_cast<int>(std::floor(xp/dx))-i_beg;
                tj = static_cast<int>(std::floor(yp/dy))-j_beg;
                // tk is the same because only a horizontal translation was applied
                if ( ti >= 0 && ti < nx && tj >= 0 && tj < ny && tk >= 0 && tk < nz) {
                  Kokkos::atomic_add( &disk_weight_samp(tk,tj,ti) , shp );
                }
              }
            });
            // Project blades if they're active
            if (do_blades) {
              // Each blade's reference space is centered about the origin, pointed upward, facing westward.
              // Zero out shaping function for projection of each blade
              yakl::Array<F,3,yakl::memDevice> blade_1("blade_1",nz,ny,nx);
              yakl::Array<F,3,yakl::memDevice> blade_2("blade_2",nz,ny,nx);
              yakl::Array<F,3,yakl::memDevice> blade_3("blade_3",nz,ny,nx);
              parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
                blade_1(k,j,i) = 0;
                blade_2(k,j,i) = 0;
                blade_3(k,j,i) = 0;
              });
              // xr is the thickness of the blade's projeciton in the x- and y-directions in ref space
              F   xr    = std::max(5.,5*dx);
              // Number of cells to sample over in each direction
              int     num_x = std::ceil(20/dx*xr*2);
              int     num_y = std::ceil(20/dx*xr*2);
              int     num_z = std::ceil(20/dx*rad*(1+decay/2));
              // Define the angles of the three blades about the hub, and compute trig functions on the angles
              F   th1 = rot_angle;
              F   th2 = rot_angle + 2.*M_PI/3.;
              F   th3 = rot_angle + 4.*M_PI/3.;
              F   cos_th1 = std::cos(th1);
              F   cos_th2 = std::cos(th2);
              F   cos_th3 = std::cos(th3);
              F   sin_th1 = std::sin(th1);
              F   sin_th2 = std::sin(th2);
              F   sin_th3 = std::sin(th3);
              parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_z,num_y,num_x) ,
                                                KOKKOS_LAMBDA (int k, int j, int i) {
                // Compute coords for this sample point in reference space
                F x = -xr  + (2*xr           *i)/(num_x-1);
                F y = -xr  + (2*xr           *j)/(num_y-1);
                F z = 0    + (rad*(1+decay/2)*k)/(num_z-1);
                F rloc = std::sqrt(x*x+y*y);
                if (rloc <= xr) {
                  // Compute 3-D shaping function for this point in reference space
                  F shp = z <= hub_radius ? 0 : thrust_shape(z/rad,1-decay/2,1+decay/2)*proj_shape_1d(rloc,xr);
                  // BLADE 1
                  {
                    // Rotate about x-axis for rotation angle
                    F x1 = x;
                    F y1 = cos_th1*y - sin_th1*z;
                    F z1 = sin_th1*y + cos_th1*z;
                    // Rotate about y-axis for shaft tilt
                    F x2 =  cos_tlt*(x1+overhang) + sin_tlt*z1;
                    F y2 =  y1;
                    F z2 = -sin_tlt*(x1+overhang) + cos_tlt*z1;
                    // Rotate about z-axis for yaw angle, and translate to base location
                    F xp = base_x     + cos_yaw*x2 - sin_yaw*y2;
                    F yp = base_y     + sin_yaw*x2 + cos_yaw*y2;
                    F zp = hub_height + z2;
                    // if it's in this task's domain, then atomically add shape to this cell's total
                    int ti = static_cast<int>(std::floor(xp/dx))-i_beg;
                    int tj = static_cast<int>(std::floor(yp/dy))-j_beg;
                    int tk = 0;
                    for (int kk=0; kk < nz; kk++) {
                      if (zp >= zint(kk) && zp < zint(kk+1)) {
                        tk = kk;
                        break;
                      }
                    }
                    if ( ti >= 0 && ti < nx && tj >= 0 && tj < ny && tk >= 0 && tk < nz) {
                      Kokkos::atomic_add( &blade_1(tk,tj,ti) , shp );
                    }
                  }
                  // BLADE 2
                  {
                    // Rotate about x-axis for rotation angle
                    F x1 = x;
                    F y1 = cos_th2*y - sin_th2*z;
                    F z1 = sin_th2*y + cos_th2*z;
                    // Rotate about y-axis for shaft tilt
                    F x2 =  cos_tlt*(x1+overhang) + sin_tlt*z1;
                    F y2 =  y1;
                    F z2 = -sin_tlt*(x1+overhang) + cos_tlt*z1;
                    // Rotate about z-axis for yaw angle, and translate to base location
                    F xp = base_x     + cos_yaw*x2 - sin_yaw*y2;
                    F yp = base_y     + sin_yaw*x2 + cos_yaw*y2;
                    F zp = hub_height + z2;
                    // if it's in this task's domain, then atomically add shape to this cell's total
                    int ti = static_cast<int>(std::floor(xp/dx))-i_beg;
                    int tj = static_cast<int>(std::floor(yp/dy))-j_beg;
                    int tk = 0;
                    for (int kk=0; kk < nz; kk++) {
                      if (zp >= zint(kk) && zp < zint(kk+1)) {
                        tk = kk;
                        break;
                      }
                    }
                    if ( ti >= 0 && ti < nx && tj >= 0 && tj < ny && tk >= 0 && tk < nz) {
                      Kokkos::atomic_add( &blade_2(tk,tj,ti) , shp );
                    }
                  }
                  // BLADE 3
                  {
                    // Rotate about x-axis for rotation angle
                    F x1 = x;
                    F y1 = cos_th3*y - sin_th3*z;
                    F z1 = sin_th3*y + cos_th3*z;
                    // Rotate about y-axis for shaft tilt
                    F x2 =  cos_tlt*(x1+overhang) + sin_tlt*z1;
                    F y2 =  y1;
                    F z2 = -sin_tlt*(x1+overhang) + cos_tlt*z1;
                    // Rotate about z-axis for yaw angle, and translate to base location
                    F xp = base_x     + cos_yaw*x2 - sin_yaw*y2;
                    F yp = base_y     + sin_yaw*x2 + cos_yaw*y2;
                    F zp = hub_height + z2;
                    // if it's in this task's domain, then atomically add shape to this cell's total
                    int ti = static_cast<int>(std::floor(xp/dx))-i_beg;
                    int tj = static_cast<int>(std::floor(yp/dy))-j_beg;
                    int tk = 0;
                    for (int kk=0; kk < nz; kk++) {
                      if (zp >= zint(kk) && zp < zint(kk+1)) {
                        tk = kk;
                        break;
                      }
                    }
                    if ( ti >= 0 && ti < nx && tj >= 0 && tj < ny && tk >= 0 && tk < nz) {
                      Kokkos::atomic_add( &blade_3(tk,tj,ti) , shp );
                    }
                  }
                }
              });
              yakl::SArray<F,1,3> blade_sum;
              blade_sum(0) = yakl::intrinsics::sum(blade_1);
              blade_sum(1) = yakl::intrinsics::sum(blade_2);
              blade_sum(2) = yakl::intrinsics::sum(blade_3);
              blade_sum = turbine.par_comm.all_reduce( blade_sum , MPI_SUM , "blade_reduce" );
              F r_sum1 = 1./blade_sum(0);
              F r_sum2 = 1./blade_sum(1);
              F r_sum3 = 1./blade_sum(2);
              // Normalize blade weights by sum
              // Compute the max over each cell as the "disk weight" to avoid double counting multiple blades
              parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
                F n1 = blade_1(k,j,i) * r_sum1;
                F n2 = blade_2(k,j,i) * r_sum2;
                F n3 = blade_3(k,j,i) * r_sum3;
                blade_weight_proj(k,j,i) = std::max(n1,std::max(n2,n3));
              });
              // Disk weights for projection will be normalized by sum later
            }
            // Compute 19.5m winds for floating motions parameterization
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , KOKKOS_LAMBDA (int j, int i) {
              F x = (i_beg+i+0.5f)*dx;
              F y = (j_beg+j+0.5f)*dy;
              if (std::abs(x-(base_x+upstream_x_offset)) <= rad && std::abs(y-(base_y+upstream_y_offset)) <= rad) {
                real zp = 19.5;
                int k19_5 = 0;
                for (int kk=0; kk < nz; kk++) {
                  if (zp >= zint(kk) && zp < zint(kk+1)) {
                    k19_5 = kk;
                    break;
                  }
                }
                F u = uvel(k19_5,j,i);
                F v = vvel(k19_5,j,i);
                umag_19_5m_2d(j,i) = std::sqrt(u*u + v*v);
              } else {
                umag_19_5m_2d(j,i) = 0;
              }
            });
          }
          // Reduce projection and 19.5m wind sums for normalization
          using yakl::componentwise::operator>;
          yakl::SArray<F,1,5> weights_tot2;
          weights_tot2(0) = yakl::intrinsics::sum(umag_19_5m_2d);
          weights_tot2(1) = (F) yakl::intrinsics::count(umag_19_5m_2d > 0.f);
          weights_tot2(2) = yakl::intrinsics::sum(disk_weight_proj);
          weights_tot2(3) = yakl::intrinsics::sum(disk_weight_samp);
          weights_tot2(4) = yakl::intrinsics::sum(blade_weight_proj);
          weights_tot2 = turbine.par_comm.all_reduce( weights_tot2 , MPI_SUM , "windmill_Allreduce1" );
          F umag_19_5m     = weights_tot2(0) / weights_tot2(1);
          F disk_proj_tot  = weights_tot2(2);
          F disk_samp_tot  = weights_tot2(3);
          F blade_proj_tot = weights_tot2(4);
          // Compute the blending weight for blades versus disk
          F blade_wt;
          F dxloc = dx*(63/rad);
          if (dxloc <= 2) {
            blade_wt = 1;
          } else {
            F x = (std::log2(dxloc)-1)/3; // Interpolate based on grid spacing in log space between 2 and 16
            blade_wt = -2*x*x*x + 3*x*x;   // p(0)=0; p'(0)=0; p(1)=1; p'(1)=0  defined in [0,1]
          }
          turbine.mag195_trace.push_back( umag_19_5m ); // Save trace of 19.5m wind speed
          // Blend blades and disk based on grid spacing, and normalize cell angle by projected weights
          if (do_blades) {
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
              if (disk_weight_proj(k,j,i) > 1.e-10) disk_weight_angle(k,j,i) /= disk_weight_proj(k,j,i);
              disk_weight_proj (k,j,i) /= disk_proj_tot;
              blade_weight_proj(k,j,i) /= blade_proj_tot;
              disk_weight_proj (k,j,i) = blade_wt*blade_weight_proj(k,j,i) + (1-blade_wt)*disk_weight_proj(k,j,i);
            });
            disk_proj_tot = turbine.par_comm.all_reduce( yakl::intrinsics::sum(disk_weight_proj) , MPI_SUM , "disk2" );
          }
          ///////////////////////////////////////////////////
          // Aggregation of disk integrals
          ///////////////////////////////////////////////////
          // Normalize disk weights for projection and upstream sampling so they sum to one
          // Aggregate disk-averaged wind velocities in upstream sampling region
          // Normalize cell angle by projected weights if blades are not used
          yakl::Array<F,3,yakl::memDevice> samp_u("samp_u",nz,ny,nx);
          yakl::Array<F,3,yakl::memDevice> samp_v("samp_v",nz,ny,nx);
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
            if (disk_weight_proj(k,j,i) > 0) {
              if (! do_blades && disk_weight_proj(k,j,i) > 1.e-10) disk_weight_angle(k,j,i) /= disk_weight_proj(k,j,i);
              disk_weight_proj(k,j,i) /= disk_proj_tot;
              proj_weight_tot (k,j,i) += disk_weight_proj(k,j,i);  // proj_weight_tot is for I/O vis
            }
            if (disk_weight_samp(k,j,i) > 0) {
              disk_weight_samp(k,j,i) /= disk_samp_tot;
              samp_weight_tot (k,j,i) += disk_weight_samp(k,j,i);  // samp_weight_tot is for I/O vis
              samp_u          (k,j,i)  = disk_weight_samp(k,j,i)*uvel(k,j,i);
              samp_v          (k,j,i)  = disk_weight_samp(k,j,i)*vvel(k,j,i);
            } else {
              samp_u          (k,j,i) = 0;
              samp_v          (k,j,i) = 0;
            }
          });
          // Reduce weighted sums of sampled upstream inflow u and v velocities
          SArray<F,1,2> sums;
          sums(0) = yakl::intrinsics::sum( samp_u );
          sums(1) = yakl::intrinsics::sum( samp_v );
          sums = turbine.par_comm.all_reduce( sums , MPI_SUM , "windmill_Allreduce2" );
          turbine.u_samp_trace.push_back( sums(0) );
          turbine.v_samp_trace.push_back( sums(1) );
          // Compute instantaneous wind magnitude for applying forces
          F instant_u0   = sums(0)*cos_yaw;  // instantaneous u-velocity normal to the turbine plane
          F instant_v0   = sums(1)*sin_yaw;  // instantaneous v-velocity normal to the turbine plane
          F instant_mag0 = std::max( (F)0. , instant_u0 + instant_v0 );
          // Compute inertial wind magnitude for computing turbine properties
          F inertial_u0   = turbine.u_samp_inertial;  // inertial u-velocity normal to the turbine plane
          F inertial_v0   = turbine.v_samp_inertial;  // inertial v-velocity normal to the turbine plane
          F inertial_mag0 = std::max( (F)0. , inertial_u0 + inertial_v0 );
          ///////////////////////////////////////////////////
          // Computation of disk properties
          ///////////////////////////////////////////////////
          F C_T = interp( ref_velmag , ref_thrust_coef , inertial_mag0 ); // Interpolate thrust coefficient
          F C_P = interp( ref_velmag , ref_power_coef  , inertial_mag0 ); // Interpolate power coefficient
          F pwr = interp( ref_velmag , ref_power       , inertial_mag0 ); // Interpolate power generation
          C_T = std::min((F)1.,C_T);
          F rot_speed = 0;
          if (ref_rotation.initialized()) rot_speed = interp( ref_velmag , ref_rotation , inertial_mag0 );
          if (coupler.option_exists("turbine_rot_fixed")) rot_speed = coupler.get_option<real>("turbine_rot_fixed");
          if (inertial_mag0 > 1.e-10) {
            if ( ! coupler.get_option<bool>("turbine_orig_C_T",true) ) {
              F a = std::max( (F)0. , std::min( (F)1. , 1 - C_P / (C_T+1.e-10f) ) );
              C_T     = 4*a*(1-a);
            }
            C_P = std::min( C_T , C_P );
          } else {
            C_T = 0;
            C_P = 0;
          }
          F C_Q = rot_speed == 0 ? 0 : C_P * inertial_mag0 / (rot_speed * rad);
          //////////////////////////////////////////////////////////////////
          // Application of floating turbine motion perturbation
          //////////////////////////////////////////////////////////////////
          if (coupler.get_option<bool>("turbine_floating_motions",false)) {
            F betti_pert, surge_pos, surge_vel, heave_pos, heave_vel, pitch_pos, pitch_vel;
            if (coupler.get_option<bool>( "turbine_floating_sine"  , false )) {
              auto amp   = coupler.get_option<real>( "turbine_floating_sine_amp"  );
              auto freq  = coupler.get_option<real>( "turbine_floating_sine_freq" );
              auto etime = coupler.get_option<real>( "elapsed_time"               );
              betti_pert = freq*amp*std::cos(freq*etime);
              surge_pos  = 0;
              surge_vel  = 0;
              heave_pos  = 0;
              heave_vel  = 0;
              pitch_pos  = 0;
              pitch_vel  = 0;
            } else {
              auto vect  = turbine.floating_motions.time_step( dt , instant_mag0 , umag_19_5m , C_T );
              surge_pos  = vect.at(0); // surge (x) position
              surge_vel  = vect.at(1); // surge velocity
              heave_pos  = vect.at(2); // heave (y) position
              heave_vel  = vect.at(3); // heave velocity
              pitch_pos  = vect.at(4); // pitch position
              pitch_vel  = vect.at(5); // pitch velocity    
              betti_pert = vect.at(6); // Induced velocity normal to disk
            }
            turbine.betti_trace    .push_back( betti_pert );
            turbine.surge_pos_trace.push_back( surge_pos  );
            turbine.surge_vel_trace.push_back( surge_vel  );
            turbine.heave_pos_trace.push_back( heave_pos  );
            turbine.heave_vel_trace.push_back( heave_vel  );
            turbine.pitch_pos_trace.push_back( pitch_pos  );
            turbine.pitch_vel_trace.push_back( pitch_vel  );
            F mult = 1;
            if ( instant_mag0 > 1.e-10 ) mult = std::max((F)0.,instant_mag0+betti_pert)/instant_mag0;
            instant_mag0 *= mult;
            instant_u0   *= mult;
            instant_v0   *= mult;
          } else {
            turbine.betti_trace    .push_back( 0 );
            turbine.surge_pos_trace.push_back( 0 );
            turbine.surge_vel_trace.push_back( 0 );
            turbine.heave_pos_trace.push_back( 0 );
            turbine.heave_vel_trace.push_back( 0 );
            turbine.pitch_pos_trace.push_back( 0 );
            turbine.pitch_vel_trace.push_back( 0 );
          }
          // Compute inertial u and v at sampling disk
          F inertial_tau = 30;
          turbine.u_samp_inertial = instant_u0*dt/inertial_tau + (inertial_tau-dt)/inertial_tau*turbine.u_samp_inertial;
          turbine.v_samp_inertial = instant_v0*dt/inertial_tau + (inertial_tau-dt)/inertial_tau*turbine.v_samp_inertial;
          // Keep track of the turbine yaw angle and the power production for this time step
          turbine.yaw_trace  .push_back( turbine.yaw_angle );
          turbine.power_trace.push_back( pwr               );
          turbine.cp_trace   .push_back( C_P               );
          turbine.ct_trace   .push_back( C_T               );
          // Fraction of thrust that didn't generate power to send into TKE
          F f_TKE = coupler.get_option<real>("turbine_f_TKE",0.25);
          F C_TKE = f_TKE * (C_T - C_P);
          ///////////////////////////////////////////////////
          // Application of disk onto tendencies
          ///////////////////////////////////////////////////
          if (turbine.apply_thrust) {
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
              if (disk_weight_proj(k,j,i) > 0) {
                // This is needed to compute the thrust force based on windmill proportion in each cell
                F turb_factor = M_PI*rad*rad/(dx*dy*dz(k));
                F az = disk_weight_angle(k,j,i);
                F wt = disk_weight_proj(k,j,i)*turb_factor;
                F un = uvel(k,j,i)*cos_yaw + vvel(k,j,i)*sin_yaw;
                // If normal wind is negative, don't do anything.
                if (un > 0) {
                  // Compute tendencies implied by actuator disk thoery; Only apply TKE for disk, not blades
                  F t_u    = -0.5f             *C_T  *instant_mag0*instant_mag0*cos_yaw                  *wt;
                  F t_v    = -0.5f             *C_T  *instant_mag0*instant_mag0*sin_yaw                  *wt;
                  F t_tke  =  0.5f*rho_d(k,j,i)*C_TKE*instant_mag0*instant_mag0*instant_mag0*(1-blade_wt)*wt;
                  // Compute tendencies for swirl
                  F t_w    =  0.5f             *C_Q  *instant_mag0*instant_mag0*std::cos(az)         *wt;
                    t_u   += -0.5f             *C_Q  *instant_mag0*instant_mag0*std::sin(az)*sin_yaw *wt;
                    t_v   +=  0.5f             *C_Q  *instant_mag0*instant_mag0*std::sin(az)*cos_yaw *wt;
                  tend_u  (k,j,i) += t_u;
                  tend_v  (k,j,i) += t_v;
                  tend_w  (k,j,i) += t_w;
                  tend_tke(k,j,i) += t_tke;
                }
              }
            });
          }
          ///////////////////////////////////////////////////
          // Update the disk's yaw angle and rot angle
          ///////////////////////////////////////////////////
          // Using only the hub cell's velocity leads to odd behavior. I'm going to use the disk-averaged
          // u and v velocity instead (note it's *not* normal u an v velocity but just plain u and v)
          if (! coupler.get_option<bool>("turbine_fixed_yaw",false)) {
            turbine.yaw_angle = turbine.yaw_tend( upstream_uvel , upstream_vvel , dt ,
                                                  turbine.yaw_angle , max_yaw_speed );
          }
          turbine.rot_angle -= rot_speed*dt;
          if (turbine.rot_angle < -2*M_PI) turbine.rot_angle += 2*M_PI;
        } // if (turbine.active)
      } // for (int iturb = 0; iturb < turbine_group.turbines.size(); iturb++)

      ///////////////////////////////////////////////////
      // Application of tendencies onto model variables
      ///////////////////////////////////////////////////
      // Update velocities and TKE based on tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        uvel(k,j,i) += dt * tend_u  (k,j,i);
        vvel(k,j,i) += dt * tend_v  (k,j,i);
        wvel(k,j,i) += dt * tend_w  (k,j,i);
        tke (k,j,i) += dt * tend_tke(k,j,i);
      });

      // So all tasks know how large the trace is. Makes PNetCDF output easier to manage
      trace_size++;
    }


    // Linear interpolation in a reference variable based on u_infinity and reference u_infinity
    real interp( realHost1d const &ref_umag , realHost1d const &ref_var , real umag ) {
      int imax = ref_umag.extent(0)-1; // Max index for the table
      // If umag exceeds the bounds of the reference data, the turbine is idle and producing no power
      if ( umag < ref_umag(0) || umag > ref_umag(imax) ) return 0;
      // Find the index such that umag lies between ref_umag(i) and ref_umag(i+1)
      int i = 0;
      // Increment past the cell it needs to be in (unless it stops at cell zero)
      while (umag > ref_umag(i)) { i++; }
      // Decrement to make it correct if not task zero
      if (i > 0) i--;
      // Linear interpolation: higher weight for left if it's closer to left
      real fac = (ref_umag(i+1) - umag) / (ref_umag(i+1)-ref_umag(i));
      return fac*ref_var(i) + (1-fac)*ref_var(i+1);
    }

  };

}



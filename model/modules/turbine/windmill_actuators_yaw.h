
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
      void init( std::string fname );
    };


    // Yaw will change as if it were an active yaw system that moves at a certain max speed. It will react
    //   to some time average of the wind velocities. The operator() outputs the new yaw angle in radians.
    struct YawTend {
      real tau, uavg, vavg;
      YawTend( real tau_in=60 , real uavg_in=0, real vavg_in=0 );
      real operator() ( real uvel , real vvel , real dt , real yaw , real max_yaw_speed );
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
                        bool                  apply_thrust = true );
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


    void init( core::Coupler &coupler );


    void apply( core::Coupler & coupler , F dt );


    // Linear interpolation in a reference variable based on u_infinity and reference u_infinity
    real interp( realHost1d const &ref_umag , realHost1d const &ref_var , real umag );

  };

}



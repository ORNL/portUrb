
#include "main_header.h"
#include "coupler.h"

namespace modules {

  // Penalize differences between domain-averaged wind and target wind at specified height
  // coupler : Coupler object containing the data manager and parallel communicator
  // dt      : Timestep size in seconds
  // height  : Height at which to compute the domain-averaged wind in meters
  // u0      : Target u-component of wind in m/s
  // v0      : Target v-component of wind in m/s
  // tau     : Relaxation timescale in seconds (default: 10s)
  // Returns the applied u and v wind forcings in m/s^2 as a tuple
  std::tuple<real,real> uniform_pg_wind_forcing_height( core::Coupler & coupler  ,
                                                               real            dt       ,
                                                               real            height   ,
                                                               real            u0       ,
                                                               real            v0       ,
                                                               real            tau = 10 );


  // Penalize differences between domain-averaged wind and given wind values
  // coupler : Coupler object containing the data manager and parallel communicator
  // dt      : Timestep size in seconds
  // u_in    : Input u-component of wind in m/s to force toward target value
  // v_in    : Input v-component of wind in m/s to force toward target value
  // u0      : Target u-component of wind in m/s
  // v0      : Target v-component of wind in m/s
  // tau     : Relaxation timescale in seconds (default: 10s)
  // Returns the applied u and v wind forcings in m/s^2 as a tuple
  std::tuple<real,real> uniform_pg_wind_forcing_given( core::Coupler & coupler  ,
                                                              real            dt       ,
                                                              real            u_in     ,
                                                              real            v_in     ,
                                                              real            u0       ,
                                                              real            v0       ,
                                                              real            tau = 10 );


  // Apply specified uniform pressure-gradient wind forcing for simulations forced by precursor
  // coupler : Coupler object containing the data manager and parallel communicator
  // dt      : Timestep size in seconds
  // utend   : Specified u-component of wind tendency in m/s^2
  // vtend   : Specified v-component of wind tendency in m/s^2
  void uniform_pg_wind_forcing_specified( core::Coupler & coupler ,
                                                 real            dt      ,
                                                 real            utend   ,
                                                 real            vtend   );



  std::pair<real,real> uniform_pg_wind_forcing_yzplane( core::Coupler & coupler ,
                                                               real            dt      ,
                                                               real            z1      ,
                                                               real            z2      ,
                                                               real            y1      ,
                                                               real            y2      ,
                                                               real            x0      ,
                                                               bool            force_v ,
                                                               real            u0      ,
                                                               real            v0      ,
                                                               real            tau     );

}


#include "main_header.h"
#include <random>

namespace modules {

  struct Floating_motions_betti {
    int    static constexpr nfreq  = 400;      // Number of frequency intervals to sum over in PM spectrum
    real   static constexpr dt_max = 0.01;     // Maximum timestep for RK4 integration (s)
    size_t static constexpr rand_pool_size = 1024*100; // Size of random phase pool
    SArray<real,6>          state;             // Current state vector
                                               // state(0): surge (x) position
                                               // state(1): surge velocity
                                               // state(2): heave (y) position
                                               // state(3): heave velocity
                                               // state(4): pitch angular position
                                               // state(5): pitch angular velocity    
    real                    etime;             // Current elapsed time
    std::vector<real>       rand_pool;         // Pool of random phases for PM spectrum
    int                     rand_pool_counter; // Offset into random phase pool
    real                    shaft_tilt;        // Shaft tilt angle in degrees


    // Assume arr is ordered lowest to highest. Return index of arr(index) nearest to "val"
    // If two values are equally close, return the lower index
    // arr : 1D array of reals
    // val : value to find nearest to
    // returns : index of arr nearest to val
    int nearest_index(realHost1d const & arr, real val);


    // Initialize the Betti floating motions module
    // shaft_tilt : Shaft tilt angle in degrees (default 0)
    void init(real shaft_tilt = 0);


    // Computes Pierson Moskowitz spectrum outputs
    // wind_19_5m: average wind velocity at 19.5m (m/s)
    // zeta      : "the x component to evaluate"
    // eta       : "the y component to evaluate .Note: the coordinate system here is different
    //                from the Betti model. The downward is negative in this case"
    // t         : the time to evaluate (s)
    // N         : The number of frequency intervals to use
    // returns  : Tuple of [wave_eta,v_x,v_y,a_x,a_y]
    // wave_eta : Wave elevation (m)
    // v_x      : x-direction wave velocity
    // v_y      : y-direction wave velocity
    // a_x      : x-direction wave acceleration
    // a_y      : y-direction wave acceleration
    // N = 400 does the job well enough
    // Fully develped oceans have essentially random wave phases at different frequencies. This spectrum empirically
    //   provides significant wave heights and pitch, roll, surge, heave, sway patterns. This approximates local wind
    //   waves and not long term swells from larger systems.
    // The random phases are precomputed and stored in rand_pool to avoid recomputing them each time this function
    //   is called.
    // Returns a tuple of five reals: (wave elevation, wave x-velocity, wave y-velocity, 
    //                                 wave x-acceleration, wave y-acceleration)
    std::tuple<real,real,real,real,real>
    pierson_moskowitz_spectrum( real wind_19_5m , real zeta , real eta , real t , int N);


    // Compute the structure dynamics given the current state and environmental inputs
    // x_1          : Current state vector
    // t            : Current time (s)
    // turbine_wind : Average wind speed at turbine hub height (m/s)
    // wind_19_5m   : Average wind speed at 19.5m (m/s)
    // Ct           : Thrust coefficient of the wind turbine
    // returns : Tuple of (state derivative vector, surge force, heave force)
    std::tuple<SArray<real,6>,real,real>
    structure( SArray<real,6> const & x_1 , real t , real turbine_wind , real wind_19_5m , real Ct );


    // Compute the state derivatives for RK4 integration
    // x           : Current state vector
    // t           : Current time (s)
    // turbine_wind: Average wind speed at turbine hub height (m/s)
    // wind_19_5m  : Average wind speed at 19.5m (m/s)
    // Ct          : Thrust coefficient of the wind turbine
    // returns : state derivative vector after forward integration step
    // This is for all intents and purposes a wrapper around the structure function
    SArray<real,6>
    Betti_tend( SArray<real,6> const & x , real t , real turbine_wind , real wind_19_5m , real Ct );


    std::array<real,7> time_step( real dt , real turbine_wind , real wind_19_5m , real Ct );

  };

}



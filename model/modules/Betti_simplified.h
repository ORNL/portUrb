
#include "main_header.h"
#include <random>

namespace modules {

  struct Floating_motions_betti {
    int    static constexpr nfreq  = 400;      // Number of frequency intervals to sum over in PM spectrum
    real   static constexpr dt_max = 0.01;     // Maximum timestep for RK4 integration (s)
    size_t static constexpr rand_pool_size = 1024*100; // Size of random phase pool
    SArray<real,1,6>        state;             // Current state vector
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
    int nearest_index(realHost1d const & arr, real val) {
      int  ind      = 0;
      real min_diff = std::abs(arr(0)-val);
      for (int i=1; i < arr.size(); i++) {
        real diff = std::abs(arr(i)-val);
        if (diff < min_diff) { ind = i; min_diff = diff; }
      }
      return ind;
    }


    // Initialize the Betti floating motions module
    // shaft_tilt : Shaft tilt angle in degrees (default 0)
    void init(real shaft_tilt = 0) {
      // Initialize state positions and velocities to zero
      this->state(0)  = 0;
      this->state(1)  = 0;
      this->state(2)  = 0;
      this->state(3)  = 0;
      this->state(4)  = 0;
      this->state(5)  = 0;
      this->etime     = 0;
      // Initialize random phase pool for PM spectrum
      rand_pool = std::vector<real>(rand_pool_size);
      std::mt19937 generator(12345);
      std::uniform_real_distribution<real> distribution(0.,2.*M_PI);
      for (int i=0; i < rand_pool_size; i++) { rand_pool.at(i) = distribution(generator); }
      rand_pool_counter = 0;  // Offset into random phase pool
      this->shaft_tilt = shaft_tilt;
    }


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
    pierson_moskowitz_spectrum( real wind_19_5m , real zeta , real eta , real t , int N) {
      real constexpr g       = 9.81;      // gravity acceleration
      real constexpr alpha   = 0.0081;    // Phillip's constant
      real constexpr start_f = 0.1;       // starting frequency
      real f_pm    = 0.14*(g/wind_19_5m); // peak frequency
      real cutof_f = 3*f_pm;              // cutoff frequency
      real sum_wave_eta = 0;              // wave elevation
      real sum_v_x      = 0;              // wave x-velocity
      real sum_v_y      = 0;              // wave y-velocity
      real sum_a_x      = 0;              // wave x-acceleration
      real sum_a_y      = 0;              // wave y-acceleration
      for (int i=0; i < N; i++) {         // loop over frequency intervals
        real random_phase  = rand_pool.at(rand_pool_counter++);         // get random phase from pool
        if (rand_pool_counter >= rand_pool_size) rand_pool_counter = 0; // wrap around pool if needed
        real delta_f       = (cutof_f - start_f) / (N-1);  // frequency interval width
        real f             = start_f + i*delta_f;          // current frequency
        real omega         = f*2*M_PI;                     // angular frequency
        real pi_term       = std::pow(2.*M_PI,4.);         // precompute constant term
        real f_term1       = std::pow(f,5.);               // f^5 term
        real f_term2       = std::pow(f_pm/f,4.);          // (f_pm/f)^4 term
        real S_pm          = (alpha*g*g /(pi_term*f_term1))*std::exp(-(5./4.)*f_term2); // PM spectrum value
        real a             = std::sqrt(2*S_pm*delta_f);    // wave amplitude for this frequency interval
        real k             = omega*omega/g;                // wave number from deep water dispersion relation
        real sin_component = std::sin(omega*t - k*zeta + random_phase); // sine component for this frequency interval
        real cos_component = std::cos(omega*t - k*zeta + random_phase); // cosine component for this frequency interval
        real exp_component = std::exp(k*eta);              // exponential decay with depth
        sum_wave_eta += a*sin_component;                           // accumulate wave elevation
        sum_v_x      += omega*a*exp_component*sin_component;       // accumulate wave x-velocity
        sum_v_y      += omega*a*exp_component*cos_component;       // accumulate wave y-velocity
        sum_a_x      += omega*omega*a*exp_component*cos_component; // accumulate wave x-acceleration
        sum_a_y      -= omega*omega*a*exp_component*sin_component; // accumulate wave y-acceleration
      }
      // Return tuple of results
      return std::make_tuple( sum_wave_eta , sum_v_x , sum_v_y , sum_a_x , sum_a_y );
    }


    // Compute the structure dynamics given the current state and environmental inputs
    // x_1          : Current state vector
    // t            : Current time (s)
    // turbine_wind : Average wind speed at turbine hub height (m/s)
    // wind_19_5m   : Average wind speed at 19.5m (m/s)
    // Ct           : Thrust coefficient of the wind turbine
    // returns : Tuple of (state derivative vector, surge force, heave force)
    std::tuple<SArray<real,1,6>,real,real>
    structure( SArray<real,1,6> const & x_1 , real t , real turbine_wind , real wind_19_5m , real Ct ) {
      real constexpr g       = 9.80665 ;         // (m/s^2)  gravity acceleration
      real constexpr rho_w   = 1025    ;         // (kg/m^3) water density
      real constexpr M_N     = 240000  ;         // (kg)     Mass of nacelle
      real constexpr M_P     = 110000  ;         // (kg)     Mass of blades and hub
      real constexpr M_S     = 8947870 ;         // (kg)     Mass of "structure" (tower and floater)
      real constexpr m_x     = 11127000;         // (kg)     Added mass in horizontal direction
      real constexpr m_y     = 1504400 ;         // (kg)     Added mass in vertical direction
      real constexpr d_Nh    = -1.8    ;         // (m)      Horizontal distance between BS and BN
      real constexpr d_Nv    = 126.9003;         // (m)      Vertical distance between BS and BN
      real constexpr d_Ph    = 5.4305  ;         // (m)      Horizontal distance between BS and BP
      real constexpr d_Pv    = 127.5879;         // (m)      Vertical distance between BS and BP
      real constexpr J_S     = 3.4917e9;         // (kg*m^2) "Structure" moment of inertia
      real constexpr J_N     = 2607890 ;         // (kg*m^2) Nacelle moment of inertia
      real constexpr J_P     = 50365000;         // (kg*m^2) Blades, hub and low speed shaft moment of inertia
      real constexpr h       = 200     ;         // (m)      Depth of water
      real constexpr h_pt    = 47.89   ;         // (m)      Height of the floating structure
      real constexpr r_g     = 9       ;         // (m)      Radius of floater
      real constexpr d_Sbott = 10.3397 ;         // (m)      Vertical distance between BS and floater bottom
      real constexpr r_tb    = 3       ;         // (m)      Maximum radius of the tower
      real constexpr d_t     = 10.3397 ;         // (m)      Vertical distance between BS and hooks of tie rods
      real constexpr l_a     = 27      ;         // (m)      Distance between the hooks of tie rods
      real constexpr l_0     = 151.73  ;         // (m)      Rest length of tie rods
      real constexpr K_T1    = 2*(1.5/l_0)*1.e9; // (N/m)    Spring constant of lateral tie rods
      real constexpr K_T2    = 2*(1.5/l_0)*1.e9; // (N/m)    Spring constant of lateral tie rods
      real constexpr K_T3    = 4*(1.5/l_0)*1.e9; // (N/m)    Spring constant of central tie rod
      real constexpr d_T     = 75.7843 ;         // (m)      Vertical distance between BS and BT
      real constexpr rho     = 1.225   ;         // (kg/m^3) Density of air
      real constexpr C_dN    = 1       ;         // (-)      Nacelle drag coefficient
      real constexpr A_N     = 9.62    ;         // (m^2)    Nacelle area
      real constexpr C_dT    = 1       ;         // (-)      tower drag coefficient
      real constexpr A       = 12469   ;         // (m^2)    Rotor area
      real constexpr n_dg    = 2       ;         //（-）     Number of floater sub-cylinders
      real constexpr C_dgper = 1       ;         // (-)      Perpendicular cylinder drag coefficient
      real constexpr C_dgpar = 0.006   ;         // (-)      Parallel cylinder drag coefficient
      real constexpr C_dgb   = 1.9     ;         // (-)      Floater bottom drag coefficient
      real constexpr den_l   = 116.027 ;         // (kg/m)   the mass density of the mooring lines
      real constexpr dia_l   = 0.127   ;         // (m)      the diameter of the mooring lines
      real constexpr h_T     = 87.6    ;         // (m)      the height of the tower
      real constexpr D_T     = 4.935   ;         // (m)      the main diameter of the tower
      real zeta   = x_1(0);                                     // surge (x) position
      real v_zeta = x_1(1);                                     // surge velocity
      real eta    = x_1(2);                                     // heave (y) position
      real v_eta  = x_1(3);                                     // heave velocity
      real alpha  = x_1(4);                                     // pitch position
      real omega  = x_1(5);                                     // pitch velocity    
      real M_X   = M_S + m_x + M_N + M_P;                       // effective mass in x direction
      real M_Y   = M_S + m_y + M_N + M_P;                       // effective mass in y direction
      real d_N   = std::sqrt(d_Nh*d_Nh + d_Nv*d_Nv);            // distance from BS to BN
      real d_P   = std::sqrt(d_Ph*d_Ph + d_Pv*d_Pv);            // distance from BS to BP
      real M_d   = M_N*d_N + M_P*d_P;                           // first moment of mass about BS
      real J_TOT = J_S + J_N + J_P + M_N*d_N*d_N + M_P*d_P*d_P; // total moment of inertia about BS
      real sin_alpha = std::sin(alpha);                         // sine of pitch angle
      real cos_alpha = std::cos(alpha);                         // cosine of pitch angle
      SArray<real,2,6,6> E;     // Mass/Inertia matrix
      E = 0;
      E(0,0) = 1            ;   // surge
      E(1,1) = M_X          ;   // surge inertia
      E(1,5) = M_d*cos_alpha;   // surge-pitch coupling
      E(2,2) = 1            ;   // heave
      E(3,3) = M_Y          ;   // heave inertia
      E(3,5) = M_d*sin_alpha;   // heave-pitch coupling
      E(4,4) = 1            ;   // pitch
      E(5,1) = M_d*cos_alpha;   // pitch-surge coupling
      E(5,3) = M_d*sin_alpha;   // pitch-heave coupling
      E(5,5) = J_TOT        ;   // pitch inertia

      // Weight Forces
      real Qwe_zeta  = 0;                    // no weight force in x direction
      real Qwe_eta   = (M_N + M_P + M_S)*g;  // weight force in y direction
      // Moment due to weight about BS
      real Qwe_alpha = ((M_N*d_Nv + M_P*d_Pv)*sin_alpha + (M_N*d_Nh + M_P*d_Ph )*cos_alpha)*g;

      // Buoyancy Forces d00 - d12 are unused dummy variables
      // They are included to match the return signature of pierson_moskowitz_spectrum
      // Compute wave eleveation and kinematics at three points in x direction: BS, BP(+r_g), BP(-r_g)
      auto [h_wave,d01,d02,d03,d04] = pierson_moskowitz_spectrum( wind_19_5m , zeta       , 0 , t , nfreq );
      auto [h_p_rg,d05,d06,d07,d08] = pierson_moskowitz_spectrum( wind_19_5m , zeta + r_g , 0 , t , nfreq );
      auto [h_n_rg,d09,d10,d11,d12] = pierson_moskowitz_spectrum( wind_19_5m , zeta - r_g , 0 , t , nfreq );
      h_wave += h; // add water depth to wave elevation at BS
      h_p_rg += h; // add water depth to wave elevation at BP(+r_g)
      h_n_rg += h; // add water depth to wave elevation at BP(-r_g)
      real h_w      = (h_wave + h_p_rg + h_n_rg)/3;               // average water height at BS and both BP points
      real h_sub    = std::min( h_w - h + eta + d_Sbott , h_pt ); // submerged height of floater
      real d_G      = eta - h_sub/2;                              // vertical distance between BS and floater center of buoyancy
      // Buoyancy volume
      real V_g      = h_sub*M_PI*r_g*r_g + std::max( (h_w - h + eta + d_Sbott) - h_pt , 0._fp )*M_PI*r_tb*r_tb;
      real Qb_zeta  = 0;                          // no buoyancy force in x direction
      real Qb_eta   = -rho_w*V_g*g;               // buoyancy force in y direction
      real Qb_alpha = -rho_w*V_g*g*d_G*sin_alpha; // moment due to buoyancy about BS

      // Tie Rod Force
      real D_x        = l_a; // horizontal distance between the hooks of tie rods
      // lengths of the three tie rods
      real l_1        = std::sqrt(std::pow(h   - eta  - l_a*sin_alpha - d_t*cos_alpha,2._fp) +
                                  std::pow(D_x - zeta - l_a*cos_alpha + d_t*sin_alpha,2._fp));
      real l_2        = std::sqrt(std::pow(h   - eta  + l_a*sin_alpha - d_t*cos_alpha,2._fp) +
                                  std::pow(D_x + zeta - l_a*cos_alpha - d_t*sin_alpha,2._fp));
      real l_3        = std::sqrt(std::pow(h   - eta  - d_t*cos_alpha,2._fp) +
                                  std::pow(      zeta - d_t*sin_alpha,2._fp));
      // tension forces in the three tie rods
      real f_1        = std::max( 0._fp , K_T1*(l_1 - l_0) );
      real f_2        = std::max( 0._fp , K_T2*(l_2 - l_0) );
      real f_3        = std::max( 0._fp , K_T3*(l_3 - l_0) );
      // angles of the three tie rods
      real theta_1    = std::atan((D_x - zeta - l_a*cos_alpha + d_t*sin_alpha) /
                                  (h   - eta  - l_a*sin_alpha - d_t*cos_alpha));
      real theta_2    = std::atan((D_x + zeta - l_a*cos_alpha - d_t*sin_alpha) /
                                  (h   - eta  + l_a*sin_alpha - d_t*cos_alpha));
      real theta_3    = std::atan((      zeta - d_t*sin_alpha) /
                                  (h   - eta  - d_t*cos_alpha));
      real v_tir      = (0.5*dia_l)*(0.5*dia_l)*M_PI; // volume per unit length of the tie rods
      real w_tir      = den_l*g;                      // weight per unit length of the tie rods
      real b_tir      = rho_w*g*v_tir;                // buoyancy per unit length of the tie rods
      real lambda_tir = w_tir - b_tir;                // effective weight per unit length of the tie rods
      // Tie rod contributions to forces and moment
      real Qt_zeta    = f_1*std::sin(theta_1) - f_2*std::sin(theta_2) - f_3*std::sin(theta_3);
      real Qt_eta     = f_1*std::cos(theta_1) + f_2*std::cos(theta_2) + f_3*std::cos(theta_3) + 4*lambda_tir*l_0;
      real Qt_alpha   = ( f_1*(l_a*std::cos(theta_1 + alpha) - d_t*std::sin(theta_1 + alpha)) - 
                          f_2*(l_a*std::cos(theta_2 - alpha) - d_t*std::sin(theta_2 - alpha)) +
                          f_3* d_t*std::sin(theta_3 - alpha) +
                           lambda_tir*l_0*(l_a*cos_alpha - d_t*sin_alpha) -
                           lambda_tir*l_0*(l_a*cos_alpha + d_t*sin_alpha) -
                         2*lambda_tir*l_0*                       d_t*sin_alpha  );
                     
      // Wind Force
      real v_in      = turbine_wind + v_zeta + d_P*omega*cos_alpha; // effective wind speed at rotor
      real FA        = 0.5*rho*A*Ct*v_in*v_in;                      // rotor thrust force
      // Nacelle and Tower Wind Drag Forces
      real FAN       = 0.5*rho*C_dN*A_N    *cos_alpha*std::pow(turbine_wind+v_zeta+d_N*omega*cos_alpha,2._fp);
      real FAT       = 0.5*rho*C_dT*h_T*D_T*cos_alpha*std::pow(turbine_wind+v_zeta+d_T*omega*cos_alpha,2._fp);
      // 5 degree shaft tilt of the 5MW turbine taken into account in the two LOCs below
      real Qwi_zeta  = -(FAN+FAT) + std::cos(shaft_tilt/180.*M_PI)*(-FA);
      // Negative should be correct because downward is negative
      // eta       : "the y component to evaluate .Note: the coordinate system here is different
      //                from the Betti model. The downward is negative in this case"
      real Qwi_eta   =              std::sin(shaft_tilt/180.*M_PI)*(-FA);
      real Qwi_alpha = (-FA *(d_Pv*cos_alpha - d_Ph*sin_alpha)
                        -FAN*(d_Nv*cos_alpha - d_Nh*sin_alpha)
                        -FAT  *d_T*cos_alpha);

      // Wave and Drag Forces
      real Qh_zeta   = 0; // initialize wave drag force in x direction
      real Qh_eta    = 0; // initialize wave drag force in y direction
      real Qwa_zeta  = 0; // initialize wave added mass force in x direction
      real Qwa_eta   = 0; // initialize wave added mass force in y direction
      real Qh_alpha  = 0; // initialize wave drag moment in pitch
      real Qwa_alpha = 0; // initialize wave added mass moment in pitch
      real v_par0;        // parallel velocity at first sub-cylinder
      for (int i=0; i < n_dg; i++) { // loop over sub-cylinders of the floater
        real h_pg         = (i + 1 - 0.5)*h_sub/n_dg; // height of the current sub-cylinder center
        real height       = -(h_sub - h_pg);          // depth of the current sub-cylinder center
        // Get wave kinematics at the sub-cylinder center
        auto [d1,v_x,v_y,a_x,a_y] = pierson_moskowitz_spectrum(wind_19_5m, zeta, height, t, nfreq);
        // Compute velocities in the parallel and perpendicular directions
        real v_par        = ((v_zeta + (h_pg - d_Sbott)*omega*cos_alpha - v_x)*-sin_alpha +
                             (v_eta  + (h_pg - d_Sbott)*omega*sin_alpha - v_y)*cos_alpha);
        if (i == 0) v_par0 = v_par; // store parallel velocity at first sub-cylinder
        // Compute forces and moments from this sub-cylinder
        real v_per        = ((v_zeta + (h_pg - d_Sbott)*omega*cos_alpha - v_x)*cos_alpha +
                             (v_eta  + (h_pg - d_Sbott)*omega*sin_alpha - v_y)*sin_alpha);
        real a_per        = a_x*cos_alpha + a_y*sin_alpha;
        // Accumulate forces and moments from each sub-cylinder
        real tempQh_zeta  = (-0.5*C_dgper*rho_w     *2*r_g*(h_sub/n_dg)* std::abs(v_per)*v_per*cos_alpha -
                              0.5*C_dgpar*rho_w*M_PI*2*r_g*(h_sub/n_dg)* std::abs(v_par)*v_par*sin_alpha);
        real tempQh_eta   = (-0.5*C_dgper*rho_w     *2*r_g*(h_sub/n_dg)* std::abs(v_per)*v_per*sin_alpha -
                              0.5*C_dgpar*rho_w*M_PI*2*r_g*(h_sub/n_dg)* std::abs(v_par)*v_par*cos_alpha);
        real tempQwa_zeta = (rho_w*V_g + m_x)*a_per*cos_alpha/n_dg;
        real tempQwa_eta  = (rho_w*V_g + m_x)*a_per*sin_alpha/n_dg;
        Qh_zeta   +=  tempQh_zeta ; // accumulate wave drag force in x direction
        Qh_eta    +=  tempQh_eta  ; // accumulate wave drag force in y direction
        Qwa_zeta  +=  tempQwa_zeta; // accumulate wave added mass force in x direction
        Qwa_eta   +=  tempQwa_eta ; // accumulate wave added mass force in y direction
        // accumulate wave drag moment in pitch
        Qh_alpha  += (tempQh_zeta *(h_pg - d_Sbott)*cos_alpha + tempQh_eta *(h_pg - d_Sbott)*sin_alpha);
        // accumulate wave added mass moment in pitch
        Qwa_alpha += (tempQwa_zeta*(h_pg - d_Sbott)*cos_alpha + tempQwa_eta*(h_pg - d_Sbott)*sin_alpha);
      }

      Qh_zeta -= 0.5*C_dgb*rho_w*M_PI*r_g*r_g*std::abs(v_par0)*v_par0*sin_alpha; // bottom drag force in x direction
      Qh_eta  -= 0.5*C_dgb*rho_w*M_PI*r_g*r_g*std::abs(v_par0)*v_par0*cos_alpha; // bottom drag force in y direction
      real Q_zeta  = Qwe_zeta  + Qb_zeta  + Qt_zeta  + Qh_zeta  + Qwa_zeta  + Qwi_zeta  + Qh_zeta ; // net force in x
      real Q_eta   = Qwe_eta   + Qb_eta   + Qt_eta   + Qh_eta   + Qwa_eta   + Qwi_eta   + Qh_eta  ; // net force in y
      real Q_alpha = Qwe_alpha + Qb_alpha + Qt_alpha + Qh_alpha + Qwa_alpha + Qwi_alpha + Qh_alpha; // net torque in pitch
      SArray<real,1,6> F; // Right-hand side vector of forces and moments
      F(0) = v_zeta;                             // surge velocity
      F(1) = Q_zeta + M_d*omega*omega*sin_alpha; // surge acceleration
      F(2) = v_eta;                              // heave velocity
      F(3) = Q_eta  - M_d*omega*omega*cos_alpha; // heave acceleration
      F(4) = omega;                              // pitch angular velocity
      F(5) = Q_alpha;                            // pitch angular acceleration
      real avegQ_t = std::sqrt(Qt_zeta*Qt_zeta+Qt_eta*Qt_eta)/8; // average tension in the tie rods
      // Solve for derivatives by inverting E and multiplying by F
      auto deriv = yakl::intrinsics::matmul_rc( yakl::intrinsics::matinv_ge(E) , F );
      // Return tuple of derivatives, effective wind speed at rotor, and average tie rod tension
      return std::make_tuple( deriv , v_in , avegQ_t );
    }


    // Compute the state derivatives for RK4 integration
    // x           : Current state vector
    // t           : Current time (s)
    // turbine_wind: Average wind speed at turbine hub height (m/s)
    // wind_19_5m  : Average wind speed at 19.5m (m/s)
    // Ct          : Thrust coefficient of the wind turbine
    // returns : state derivative vector after forward integration step
    // This is for all intents and purposes a wrapper around the structure function
    SArray<real,1,6>
    Betti_tend( SArray<real,1,6> const & x , real t , real turbine_wind , real wind_19_5m , real Ct ) {
      // Get the state derivatives from the structure function
      auto [dx1dt,v_in,Q_t] = structure( x , t , turbine_wind , wind_19_5m , Ct);
      return dx1dt;
    }


    std::array<real,7> time_step( real dt , real turbine_wind , real wind_19_5m , real Ct ) {
      using yakl::componentwise::operator+; // enable componentwise '+' for yakl::SArray
      using yakl::componentwise::operator*; // enable componentwise '*' for yakl::SArray
      using yakl::componentwise::operator/; // enable componentwise '/' for yakl::SArray
      int niter = std::ceil( dt / dt_max ); // number of RK4 iterations needed for stable integration
      dt = dt / niter; // adjust dt for number of iterations
      real wind = 0;   // accumulate effective perturbation to wind speed at rotor over sub-steps
      for (int iter=0; iter < niter; iter++) { // Iterate over sub-steps (RK4 at each sub-step)
        real constexpr d_Ph = 5.4305    ; // (m) Horizontal distance between BS and BP
        real constexpr d_Pv = 127.5879  ; // (m) Vertical distance between BS and BP
        // The four lines below implement the RK4 integration scheme
        auto k1 = Betti_tend( state         , etime      ,  turbine_wind , wind_19_5m , Ct );
        auto k2 = Betti_tend( state+dt/2*k1 , etime+dt/2 ,  turbine_wind , wind_19_5m , Ct );
        auto k3 = Betti_tend( state+dt/2*k2 , etime+dt/2 ,  turbine_wind , wind_19_5m , Ct );
        auto k4 = Betti_tend( state+dt  *k3 , etime+dt   ,  turbine_wind , wind_19_5m , Ct );
        state  = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6; // Apply b-weighting to update state
        etime += dt;  // Advance elapsed time
        // Accumulate effective wind speed perturbation to be applied to inflow wind speed
        wind  += state(1) + std::sqrt(d_Ph*d_Ph + d_Pv*d_Pv)*state(5)*std::cos(state(4));
      }
      std::array<real,7> ret;
      ret.at(0) = state(0); // surge (x) position
      ret.at(1) = state(1); // surge velocity
      ret.at(2) = state(2); // heave (y) position
      ret.at(3) = state(3); // heave velocity
      ret.at(4) = state(4); // pitch position
      ret.at(5) = state(5); // pitch velocity    
      ret.at(6) = wind / niter; // average effective wind speed perturbation at rotor over the full time step
      return ret;
    }

  };

}



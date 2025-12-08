
#include "coupler.h"
#include <fstream>
#include "YAKL_netcdf.h"


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




struct turbine_BEM {
  int static constexpr MAX_FIELDS = 50;
  typedef yakl::SArray<realHost1d,1,MAX_FIELDS> MultiField;

  real       R         ;
  real       R_hub     ;
  realHost1d foil_mid  ;
  realHost1d foil_len  ;
  realHost1d foil_twist;
  realHost1d foil_chord;
  realHost1d foil_id   ;
  MultiField foil_alpha;
  MultiField foil_clift;
  MultiField foil_cdrag;
  realHost1d rwt_mag   ;
  realHost1d rwt_ct    ;
  realHost1d rwt_cp    ;
  realHost1d rwt_pwr_mw;
  realHost1d rwt_rot   ;

  void init(std::string turbine_file) {
    typedef std::tuple<real,real,real,real,std::string> FOIL_LINE;
    // GET YAML DATA
    YAML::Node node   = YAML::LoadFile(turbine_file);
    R                 = node["blade_radius"      ].as<real>();
    R_hub             = node["tower_top_radius"  ].as<real>();
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
    foil_mid   = realHost1d("foil_mid"  ,nseg);
    foil_len   = realHost1d("foil_len"  ,nseg);
    foil_twist = realHost1d("foil_twist",nseg);
    foil_chord = realHost1d("foil_chord",nseg);
    foil_id    = realHost1d("foil_id"   ,nseg);
    for (int iseg=0; iseg < nseg ; iseg++) {
      foil_mid  (iseg) = std::get<0>(foil_summary.at(iseg));
      foil_twist(iseg) = std::get<1>(foil_summary.at(iseg))/180.*M_PI;
      foil_len  (iseg) = std::get<2>(foil_summary.at(iseg));
      foil_chord(iseg) = std::get<3>(foil_summary.at(iseg));
      int id = -1;
      for (int ifoil = 0; ifoil < nfoil; ifoil++) {
        if (std::get<4>(foil_summary.at(iseg)) == foil_names.at(ifoil)) { id = ifoil; break; }
      }
      foil_id   (iseg) = id;
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
      foil_alpha(ifoil) = loc_alpha;
      foil_clift(ifoil) = loc_clift;
      foil_cdrag(ifoil) = loc_cdrag;
    }
    int nrwt = velmag.size();
    rwt_mag    = realHost1d("rwt_mag"   ,nrwt);
    rwt_ct     = realHost1d("rwt_ct"    ,nrwt);
    rwt_cp     = realHost1d("rwt_cp"    ,nrwt);
    rwt_pwr_mw = realHost1d("rwt_pwr_mw",nrwt);
    rwt_rot    = realHost1d("rwt_rot"   ,nrwt);
    for (int irwt = 0; irwt < nrwt; irwt++) {
      rwt_mag   (irwt) = velmag  .at(irwt);
      rwt_ct    (irwt) = cthrust .at(irwt);
      rwt_cp    (irwt) = cpower  .at(irwt);
      rwt_pwr_mw(irwt) = power_mw.at(irwt);
      rwt_rot   (irwt) = rot_rpm .at(irwt)*2.*M_PI/60.;
    }
  }

  static real linear_interp( realHost1d const & aref                ,
                             realHost1d const & vref                ,
                             real               a                   ,
                             bool               const_extrap = true ) {
    int n = aref.size();
    if ( n==0 || aref.size() != vref.size() ) Kokkos::abort("Invalid input vectors");
    if ( a < aref(0) || aref.size() == 1 ) return const_extrap ? vref(0)   : 0.;
    if ( a > aref(n-1)                   ) return const_extrap ? vref(n-1) : 0.;
    for (int i=0; i < n-1; i++) {
      if (a >= aref(i) && a <= aref(i+1)) return vref(i)+(a-aref(i))/(aref(i+1)-aref(i))*(vref(i+1)-vref(i));
    }
    return 0.; // Doesn't get here, but gotta keep that compiler happy..
  }



  // TODO: Near rated wind speed, this routine limits convergence of alpha to 1e-3 at the last blade segment
  static real prandtl_tip_loss(real r, real R, int num_blades, real phi_rad) {
    if (std::abs(std::sin(phi_rad)) < 1e-6) return 1;
    real f_tip = (num_blades / 2.0) * (R - r) / (r * std::sin(phi_rad));
    if (f_tip > 50) return 1;
    real F_tip = (2.0 / M_PI) * std::acos(std::exp(-f_tip));
    return std::max(F_tip, (real)0.0001);
  }



  static real prandtl_hub_loss(real r, real R_hub, int num_blades, real phi_rad) {
      if (std::abs(std::sin(phi_rad)) < 1e-6) return 1;
      real f_hub = (num_blades / 2.0) * (r - R_hub) / (r * std::sin(phi_rad));
      if (f_hub > 50) return 1;
      real F_hub = (2.0 / M_PI) * std::acos(std::exp(-f_hub));
      return std::max(F_hub, (real) 0.0001);
  }



  real blade_element( real         r                 ,         // input : radial position (m)
                      real         twist             ,         // input : twist angle (radians)
                      real         chord             ,         // input : chord length (m)
                      real         omega             ,         // input : rotation rate (radians / sec)
                      realHost1d & ref_alpha         ,         // input : look-up alpha for coefficients of lift and drag
                      realHost1d & ref_clift         ,         // input : look-up for coefficient of lift based on alpha
                      realHost1d & ref_cdrag         ,         // input : look-up for coefficient of drag based on alpha
                      real         U_inf             ,         // input : inflow wind speed
                      real         pitch             ,         // input : pitch of the blades (radians)
                      int          num_blades        ,         // input : number of blades
                      real         rho               ,         // input : total air density (kg/m^3)
                      bool         tip_loss          ,         // input : whether to include tip loss
                      bool         hub_loss          ,         // input : whether to include hub loss
                      int          max_iter          ,         // input : maximum number of iterations for convergence
                      real         tol               ,         // input : tolerance for convergence
                      real       & a                 ,         // output: axial induction factor (-)
                      real       & a_prime           ,         // output: tangential induction factor (-)
                      real       & phi               ,         // output: inflow angle (rad)
                      real       & alpha             ,         // output: angle of attack (rad)
                      real       & Cl                ,         // output: lift coefficient (-)
                      real       & Cd                ,         // output: drag coefficient (-)
                      real       & Cn                ,         // output: normal force coefficient (-)
                      real       & Ct                ,         // output: tangential force coefficient (-)
                      real       & W                 ,         // output: relative velocity magnitude (m/s)
                      real       & dT_dr             ,         // output: thrust per unit span (N/m) (all blades)
                      real       & dQ_dr             ,         // output: torque per unit span (N·m/m) (all blades)
                      real       & F                 ,         // output: combined tip-hub loss factor (-)
                      real       & sigma             ) const { // output: local solidarity factor
    a          = 0;                                    // Axial induction factor
    a_prime    = 0;                                    // Tangential induction factor
    sigma      = num_blades * chord / (2. * M_PI * r); // Local solidity
    real theta = twist + pitch;                        // Blade section angle (twist + pitch) (rad)
    real a_new;       // Next predicted iteration for axial induction factor
    real a_prime_new; // Next predicted iteration for tangential induction factor
    for (int iter = 0; iter < max_iter; iter++) {
      real U_axial = U_inf * (1 - a);                            // Axial velocity
      real U_tang  = omega * r * (1 + a_prime);                  // Tangential velocity
      W            = std::sqrt(U_axial*U_axial + U_tang*U_tang); // Relative velocity magnitude
      phi          = std::atan2(U_axial, U_tang);                // Flow angle (rad)
      alpha        = phi - theta;                                // Angle of attack (rad)
      Cl           = linear_interp( ref_alpha , ref_clift , alpha/M_PI*180. ,true ); // Coefficient of lift
      Cd           = linear_interp( ref_alpha , ref_cdrag , alpha/M_PI*180. ,true ); // Coefficient of drag
      Cn           = Cl * std::cos(phi) + Cd * std::sin(phi); // Normal force coefficient
      Ct           = Cl * std::sin(phi) - Cd * std::cos(phi); // Tangential force coefficient
      F = 1;  // Total loss from tip and hub
      if (tip_loss) F *= prandtl_tip_loss( r , R     , num_blades , phi );  // Tip loss
      if (hub_loss) F *= prandtl_hub_loss( r , R_hub , num_blades , phi );  // Hub loss
      real sin_phi = std::sin(phi);    if (std::abs(sin_phi) < 1e-6) sin_phi = 1e-6;
      real cos_phi = std::cos(phi);    if (std::abs(cos_phi) < 1e-6) cos_phi = 1e-6;
      a_new = 1. / ((4 * F * sin_phi*sin_phi) / (sigma * Cn + 1e-10) + 1); // New axial induction factor
      // Glauert correction if needed
      if (a_new > 0.4) {
        real ac = 0.2;
        real K  = 4 * F * sin_phi*sin_phi / (sigma * Cn + 1e-10);
        a_new   = 0.5 * (2 + K * (1 - 2*ac) - std::sqrt((K*(1 - 2*ac) + 2)*(K*(1 - 2*ac) + 2) + 4*(K*ac*ac - 1)));
      }
      // Compute new tangential induction factor
      real denom_ap = (4 * F * sin_phi * cos_phi) / (sigma * Ct + 1e-10) - 1;
      if (abs(denom_ap) < 1e-10) { a_prime_new = 0.0; }
      else                       { a_prime_new = 1 / denom_ap; }
      // Clip new axial and tangential induction factors to realistic values
      a_new       = std::max((real)-0.5,std::min((real)0.95,a_new      ));
      a_prime_new = std::max((real)-0.5,std::min((real)0.95,a_prime_new));
      // If converged, then exit and compute final values
      if (std::abs(a_new - a) < tol && std::abs(a_prime_new - a_prime) < tol) {
        a       = a_new;
        a_prime = a_prime_new;
        break;
      }
      // If not converged, then proceed to the next iteration
      real relax = 0.25;
      // Use under-relaxation for stability
      a       = relax * a_new       + (1 - relax) * a      ;
      a_prime = relax * a_prime_new + (1 - relax) * a_prime;
    }
    real U_axial = U_inf * (1 - a);
    real U_tang  = omega * r * (1 + a_prime);
    W            = std::sqrt(U_axial*U_axial + U_tang*U_tang);
    phi          = std::atan2(U_axial, U_tang);
    alpha        = phi - theta;
    Cl           = linear_interp( ref_alpha , ref_clift , alpha/M_PI*180. ,true ); // Coefficient of lift
    Cd           = linear_interp( ref_alpha , ref_cdrag , alpha/M_PI*180. ,true ); // Coefficient of drag
    Cn           = Cl * std::cos(phi) + Cd * std::sin(phi);
    Ct           = Cl * std::sin(phi) - Cd * std::cos(phi);
    // dT_dr        = 0.5 * rho * W*W * chord * Cn * num_blades;
    // dQ_dr        = 0.5 * rho * W*W * chord * Ct * num_blades * r;
    dT_dr        = 4*M_PI*r*rho*U_inf*U_inf*(1-a)*a*F;
    dQ_dr        = 4*M_PI*r*r*r*rho*U_inf*omega*(1-a)*a_prime*F;
    return std::max( std::abs(a_new - a) , std::abs(a_prime_new - a_prime) );
  }



  void auto_pitch( realHost1d & winds           ,   // input : inflow wind speeds (m/s)
                   int          num_blades      ,   // input : number of blades
                   real         rho             ,   // input : total air density (kg/m^3)
                   real         gen_eff         ,   // input : Proportion of torque that generates power
                   bool         tloss           ,   // input : whether to include tip loss
                   bool         hloss           ,   // input : whether to include hub loss
                   real         max_power       ,   // input : whether to include hub loss
                   real         max_thrust_prop ,   // input : whether to include hub loss
                   realHost1d & out_pitch       ,   // output: pitch angle (radians)
                   realHost2d & out_dT_dr       ,   // output: thrust values at section mid points
                   realHost2d & out_dQ_dr       ,   // output: torque values at section mid points
                   realHost2d & out_a_r         ,   // output: axial induction factors at section mid points
                   realHost2d & out_ap_r        ,   // output: tangential induction factors at section mid points
                   realHost2d & out_alpha_r     ,   // output: angle of attack at section mid points
                   realHost2d & out_phi_r       ,   // output: flow angle at section mid points
                   realHost1d & out_thrust      ,   // output: total thrust (N)
                   realHost1d & out_torque      ,   // output: total torque (N m)
                   realHost1d & out_power       ,   // output: power generation (W)
                   realHost1d & out_C_T         ,   // output: thrust coefficient
                   realHost1d & out_C_P         ,   // output: power coefficient
                   realHost1d & out_C_Q         ) { // output: torque coefficient
    int  nwinds = winds.size();
    int  nseg   = foil_mid.size();
    out_pitch   = realHost1d("out_pitch  ",nwinds);
    out_dT_dr   = realHost2d("out_dT_dr  ",nwinds,nseg);
    out_dQ_dr   = realHost2d("out_dQ_dr  ",nwinds,nseg);
    out_a_r     = realHost2d("out_a_r    ",nwinds,nseg);
    out_ap_r    = realHost2d("out_ap_r   ",nwinds,nseg);
    out_alpha_r = realHost2d("out_alpha_r",nwinds,nseg);
    out_phi_r   = realHost2d("out_phi_r  ",nwinds,nseg);
    out_thrust  = realHost1d("out_thrust ",nwinds);
    out_torque  = realHost1d("out_torque ",nwinds);
    out_power   = realHost1d("out_power  ",nwinds);
    out_C_T     = realHost1d("out_C_T    ",nwinds);
    out_C_P     = realHost1d("out_C_P    ",nwinds);
    out_C_Q     = realHost1d("out_C_Q    ",nwinds);
    real pitch_min = 0;
    real pitch_max = M_PI/2.;

    // Determine the maximum thrust among the input wind speeds
    for (int iwind = 0; iwind < nwinds; iwind++) {
      int  mxiter  = 200;            // Maximum number of iterations
      real tol     = 1.e-6;          // Tolerance for convergence
      real U_inf   = winds(iwind);   // Inflow wind speed
      real omega   = linear_interp(rwt_mag,rwt_rot,U_inf,false); // Rotation rate (rad/sec)
      real thrust  = 0;              // For accumulating the total thrust
      real torque  = 0;              // For accumulating the total torque
      real pitch   = 0;              // Initial pitch assumption
      // Loop over blade segments and compute maximum
      for (int iseg = 0; iseg < nseg; iseg++) {
        real       r         = foil_mid  (iseg);
        real       dr        = foil_len  (iseg);
        real       twist     = foil_twist(iseg);
        real       chord     = foil_chord(iseg);
        realHost1d ref_alpha = foil_alpha(foil_id(iseg));
        realHost1d ref_clift = foil_clift(foil_id(iseg));
        realHost1d ref_cdrag = foil_cdrag(foil_id(iseg));
        real a, ap, phi, alpha, Cl, Cd, Cn, Ct, W, dT_dr, dQ_dr, F, sigma; // outputs
        real conv = blade_element( r , twist , chord , omega , ref_alpha , ref_clift , ref_cdrag , U_inf ,     // inputs
                                   pitch , num_blades , rho , tloss , hloss , mxiter , tol ,       // inputs
                                   a , ap , phi , alpha , Cl , Cd , Cn , Ct , W , dT_dr , dQ_dr , F , sigma ); // outputs
        if (conv > tol) std::cout << "NOT CONVERGED: r: " << r << ";  conv == " << conv << std::endl;
        out_dT_dr  (iwind,iseg) = dT_dr;
        out_dQ_dr  (iwind,iseg) = dQ_dr;
        out_a_r    (iwind,iseg) = a;
        out_ap_r   (iwind,iseg) = ap;
        out_alpha_r(iwind,iseg) = alpha/M_PI*180.;
        out_phi_r  (iwind,iseg) = phi/M_PI*180.;
        thrust += dT_dr*dr;
        torque += dQ_dr*dr;
      }
      real power = torque*omega*gen_eff;
      out_pitch (iwind) = pitch ;
      out_thrust(iwind) = thrust;
      out_torque(iwind) = torque;
      out_power (iwind) = power ;
      out_C_T   (iwind) = thrust/(0.5*rho*M_PI*R*R*U_inf*U_inf)      ;
      out_C_Q   (iwind) = torque/(0.5*rho*M_PI*R*R*R*U_inf*U_inf)    ;
      out_C_P   (iwind) = power /(0.5*rho*M_PI*R*R*U_inf*U_inf*U_inf);
      if (power > max_power) {
        real p1 = pitch_min;
        real p2 = pitch_max;
        while ( std::abs(power-max_power) > 1 ) {
          real pitch = (p1+p2)/2;
          real thrust  = 0;              // For accumulating the total thrust
          real torque  = 0;              // For accumulating the total torque
          // Loop over blade segments and compute maximum
          for (int iseg = 0; iseg < nseg; iseg++) {
            real       r         = foil_mid  (iseg);
            real       dr        = foil_len  (iseg);
            real       twist     = foil_twist(iseg);
            real       chord     = foil_chord(iseg);
            realHost1d ref_alpha = foil_alpha(foil_id(iseg));
            realHost1d ref_clift = foil_clift(foil_id(iseg));
            realHost1d ref_cdrag = foil_cdrag(foil_id(iseg));
            real a, ap, phi, alpha, Cl, Cd, Cn, Ct, W, dT_dr, dQ_dr, F, sigma; // outputs
            real conv = blade_element( r , twist , chord , omega , ref_alpha , ref_clift , ref_cdrag , U_inf ,     // inputs
                                       pitch , num_blades , rho , tloss , hloss , mxiter , tol ,       // inputs
                                       a , ap , phi , alpha , Cl , Cd , Cn , Ct , W , dT_dr , dQ_dr , F , sigma ); // outputs
            if (conv > tol) std::cout << "NOT CONVERGED: r: " << r << ";  conv == " << conv << std::endl;
            out_dT_dr  (iwind,iseg) = dT_dr;
            out_dQ_dr  (iwind,iseg) = dQ_dr;
            out_a_r    (iwind,iseg) = a;
            out_ap_r   (iwind,iseg) = ap;
            out_alpha_r(iwind,iseg) = alpha/M_PI*180.;
            out_phi_r  (iwind,iseg) = phi/M_PI*180.;
            thrust += dT_dr*dr;
            torque += dQ_dr*dr;
          }
          power = torque*omega*gen_eff;
          out_pitch (iwind) = pitch ;
          out_thrust(iwind) = thrust;
          out_torque(iwind) = torque;
          out_power (iwind) = power ;
          out_C_T   (iwind) = thrust/(0.5*rho*M_PI*R*R*U_inf*U_inf)      ;
          out_C_Q   (iwind) = torque/(0.5*rho*M_PI*R*R*R*U_inf*U_inf)    ;
          out_C_P   (iwind) = power /(0.5*rho*M_PI*R*R*U_inf*U_inf*U_inf);
          if (power > max_power) { p1 = pitch; }
          else                   { p2 = pitch; }
        }
      }
    }

    real max_thrust = yakl::intrinsics::maxval(out_thrust)*max_thrust_prop;

    // Determine the maximum thrust among the input wind speeds
    for (int iwind = 0; iwind < nwinds; iwind++) {
      real thrust = out_thrust(iwind);
      real pitch  = out_pitch (iwind);
      while (thrust > max_thrust) {
        pitch += 0.01/180.*M_PI;
        int  mxiter  = 200;            // Maximum number of iterations
        real tol     = 1.e-6;          // Tolerance for convergence
        real U_inf   = winds(iwind);   // Inflow wind speed
        real omega   = linear_interp(rwt_mag,rwt_rot,U_inf,false); // Rotation rate (rad/sec)
        thrust       = 0;              // For accumulating the total thrust
        real torque  = 0;              // For accumulating the total torque
        // Loop over blade segments and compute maximum
        for (int iseg = 0; iseg < nseg; iseg++) {
          real       r         = foil_mid  (iseg);
          real       dr        = foil_len  (iseg);
          real       twist     = foil_twist(iseg);
          real       chord     = foil_chord(iseg);
          realHost1d ref_alpha = foil_alpha(foil_id(iseg));
          realHost1d ref_clift = foil_clift(foil_id(iseg));
          realHost1d ref_cdrag = foil_cdrag(foil_id(iseg));
          real a, ap, phi, alpha, Cl, Cd, Cn, Ct, W, dT_dr, dQ_dr, F, sigma; // outputs
          real conv = blade_element( r , twist , chord , omega , ref_alpha , ref_clift , ref_cdrag , U_inf ,     // inputs
                                     pitch , num_blades , rho , tloss , hloss , mxiter , tol ,       // inputs
                                     a , ap , phi , alpha , Cl , Cd , Cn , Ct , W , dT_dr , dQ_dr , F , sigma ); // outputs
          if (conv > tol) std::cout << "NOT CONVERGED: r: " << r << ";  conv == " << conv << std::endl;
          out_dT_dr  (iwind,iseg) = dT_dr;
          out_dQ_dr  (iwind,iseg) = dQ_dr;
          out_a_r    (iwind,iseg) = a;
          out_ap_r   (iwind,iseg) = ap;
          out_alpha_r(iwind,iseg) = alpha/M_PI*180.;
          out_phi_r  (iwind,iseg) = phi/M_PI*180.;
          thrust += dT_dr*dr;
          torque += dQ_dr*dr;
        }
        real power = torque*omega*gen_eff;
        out_pitch (iwind) = pitch ;
        out_thrust(iwind) = thrust;
        out_torque(iwind) = torque;
        out_power (iwind) = power ;
        out_C_T   (iwind) = thrust/(0.5*rho*M_PI*R*R*U_inf*U_inf)      ;
        out_C_Q   (iwind) = torque/(0.5*rho*M_PI*R*R*R*U_inf*U_inf)    ;
        out_C_P   (iwind) = power /(0.5*rho*M_PI*R*R*U_inf*U_inf*U_inf);
      }
    }

  }

};
                    

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    turbine_BEM bem;
    bem.init("./inputs/NREL_5MW_126_RWT.yaml");

    real cut_in  = bem.rwt_mag(0);
    real cut_out = bem.rwt_mag(bem.rwt_mag.size()-1);
    realHost1d winds("winds",(int)((cut_out-cut_in)*10)+1);
    for (int i=0; i < winds.size(); i++) {
      winds(i) = cut_in + (cut_out-cut_in)*((real)i)/(winds.size()-1.);
    }

    bool tloss           = true;   // Use tip loss?
    bool hloss           = true;   // Use hub loss?
    int  mxiter          = 200;    // Maximum number of iterations
    real tol             = 1.e-6;  // Tolerance for convergence
    real pitch           = 0;      // Blade pitch
    real U_inf           = 3;      // Inflow wind speed
    real gen_eff         = 0.944;  // Efficiency of power generation
    int  num_blades      = 3;
    real rho             = 1.225;  // Air density
    real max_power       = 5.29661e6;
    real max_thrust_prop = 0.8;

    realHost2d out_dT_dr, out_dQ_dr, out_a_r, out_ap_r, out_alpha_r, out_phi_r;
    realHost1d out_pitch, out_thrust, out_torque, out_power, out_C_T, out_C_P, out_C_Q;

    bem.auto_pitch( winds , num_blades , rho , gen_eff , tloss , hloss , max_power , max_thrust_prop ,
                    out_pitch , out_dT_dr , out_dQ_dr , out_a_r , out_ap_r , out_alpha_r , out_phi_r , out_thrust , out_torque , out_power ,
                    out_C_T , out_C_P , out_C_Q );

    yakl::SimpleNetCDF nc;
    nc.create( "bem.nc" , yakl::NETCDF_MODE_REPLACE );
    nc.write( winds        , "wind"    , {"wind"});
    nc.write( bem.foil_mid , "segment" , {"segment"});
    nc.write( out_dT_dr    , "dT_dr"   , {"wind","segment"});
    nc.write( out_dQ_dr    , "dQ_dr"   , {"wind","segment"});
    nc.write( out_a_r      , "a_r"     , {"wind","segment"});
    nc.write( out_ap_r     , "ap_r"    , {"wind","segment"});
    nc.write( out_alpha_r  , "alpha_r" , {"wind","segment"});
    nc.write( out_phi_r    , "phi_r"   , {"wind","segment"});
    nc.write( out_pitch    , "pitch"   , {"wind"});
    nc.write( out_thrust   , "thrust"  , {"wind"});
    nc.write( out_torque   , "torque"  , {"wind"});
    nc.write( out_power    , "power"   , {"wind"});
    nc.write( out_C_T      , "C_T"     , {"wind"});
    nc.write( out_C_P      , "C_P"     , {"wind"});
    nc.write( out_C_Q      , "C_Q"     , {"wind"});

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}


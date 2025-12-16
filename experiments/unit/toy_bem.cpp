
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



  static real prandtl_tip_loss(real r, real R, int B, real phi) {
    real s = std::abs(std::sin(phi));
    if (s < 1e-10) return 1.0;           // phi ~ 0 => F ~ 1 (limit)
    real f = (B * 0.5) * (R - r) / (r * s);
    // if f is huge, exp(-f) ~ 0 => acos(0)=pi/2 => F=1 (good)
    real F = (2.0/M_PI) * std::acos(std::exp(-f));
    return std::clamp(F, (real)1e-10, (real)1.0);
  }

  static real prandtl_hub_loss(real r, real Rhub, int B, real phi) {
    real s = std::abs(std::sin(phi));
    if (s < 1e-10) return 1.0;
    real f = (B * 0.5) * (r - Rhub) / (r * s);
    real F = (2.0/M_PI) * std::acos(std::exp(-f));
    return std::clamp(F, (real)1e-10, (real)1.0);
  }



  real blade_element( real         r                 ,         // input : radial position (m)
                      real         twist             ,         // input : twist angle (radians)
                      real         chord             ,         // input : chord length (m)
                      real         omega             ,         // input : rotation rate (radians / sec)
                      real         mult1             ,
                      real         mult2             ,
                      realHost1d & ref_alpha1        ,         // input : look-up alpha for coefficients of lift and drag
                      realHost1d & ref_clift1        ,         // input : look-up for coefficient of lift based on alpha
                      realHost1d & ref_cdrag1        ,         // input : look-up for coefficient of drag based on alpha
                      realHost1d & ref_alpha2        ,         // input : look-up alpha for coefficients of lift and drag
                      realHost1d & ref_clift2        ,         // input : look-up for coefficient of lift based on alpha
                      realHost1d & ref_cdrag2        ,         // input : look-up for coefficient of drag based on alpha
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
    sigma         = num_blades * chord / (2. * M_PI * r); // Local solidity
    real theta    = twist + pitch;                        // Blade section angle (twist + pitch) (rad)
    real lambda_r = omega*r/U_inf;
    a             = 0.3; // (2+M_PI*lambda_r*sigma-std::sqrt(4-4*M_PI*lambda_r*sigma+M_PI*lambda_r*lambda_r*sigma*(8*theta+M_PI*sigma)))/4;
    a_prime       = 0;                                    // Tangential induction factor
    real a_new;       // Next predicted iteration for axial induction factor
    real a_prime_new; // Next predicted iteration for tangential induction factor
    real relax = 0.25;
    bool max_reached = false;
    for (int iter = 0; iter < max_iter; iter++) {
      real U_axial = U_inf * (1 - a);                            // Axial velocity
      real U_tang  = omega * r * (1 + a_prime);                  // Tangential velocity
      phi          = std::atan2(U_axial, U_tang);                // Flow angle (rad)
      alpha        = phi - theta;                                // Angle of attack (rad)
      real Cl1     = linear_interp( ref_alpha1 , ref_clift1 , alpha/M_PI*180. , true ); // Coefficient of lift
      real Cd1     = linear_interp( ref_alpha1 , ref_cdrag1 , alpha/M_PI*180. , true ); // Coefficient of drag
      real Cl2     = linear_interp( ref_alpha2 , ref_clift2 , alpha/M_PI*180. , true ); // Coefficient of lift
      real Cd2     = linear_interp( ref_alpha2 , ref_cdrag2 , alpha/M_PI*180. , true ); // Coefficient of drag
      Cl           = mult1*Cl1 + mult2*Cl2;
      Cd           = mult1*Cd1 + mult2*Cd2;
      Cn           = Cl * std::cos(phi) + Cd * std::sin(phi); // Normal force coefficient
      Ct           = Cl * std::sin(phi) - Cd * std::cos(phi); // Tangential force coefficient
      F = 1;  // Total loss from tip and hub
      if (tip_loss) F *= prandtl_tip_loss( r , R     , num_blades , phi );  // Tip loss
      if (hub_loss) F *= prandtl_hub_loss( r , R_hub , num_blades , phi );  // Hub loss
      real sin_phi = std::sin(phi);
      real cos_phi = std::cos(phi);
      real C_T = (sigma*(1-a)*(1-a)*Cn)/(sin_phi*sin_phi);
      if (C_T > 0.96*F) {
        a_new = (18*F-20-3*std::sqrt(C_T*(50-36*F)+12*F*(3*F-4))) / (36*F-50);
      } else {
        a_new = 1./(1+(4*F*sin_phi*sin_phi)/(sigma*Cn));
      }
      // Compute new tangential induction factor
      a_prime_new = 1./(-1.+(4*F*sin_phi*cos_phi)/(sigma*Ct));
      // Clip new axial and tangential induction factors to realistic values
      // a_new       = std::max((real)-0.5,std::min((real)0.95,a_new      ));
      // a_prime_new = std::max((real)-0.5,std::min((real)0.50,a_prime_new));
      // If converged, then exit and compute final values
      if (std::abs(a_new - a) < tol && std::abs(a_prime_new - a_prime) < tol) {
        a       = a_new;
        a_prime = a_prime_new;
        break;
      }
      if (iter == max_iter-1) max_reached = true;
      // If not converged, then proceed to the next iteration
      // Use under-relaxation for stability
      a       = relax * a_new       + (1 - relax) * a      ;
      a_prime = relax * a_prime_new + (1 - relax) * a_prime;
    }
    real U_axial = U_inf * (1 - a);
    real U_tang  = omega * r * (1 + a_prime);
    W            = std::sqrt(U_axial*U_axial + U_tang*U_tang);
    phi          = std::atan2(U_axial, U_tang);
    alpha        = phi - theta;
    real Cl1     = linear_interp( ref_alpha1 , ref_clift1 , alpha/M_PI*180. ,true ); // Coefficient of lift
    real Cd1     = linear_interp( ref_alpha1 , ref_cdrag1 , alpha/M_PI*180. ,true ); // Coefficient of drag
    real Cl2     = linear_interp( ref_alpha2 , ref_clift2 , alpha/M_PI*180. ,true ); // Coefficient of lift
    real Cd2     = linear_interp( ref_alpha2 , ref_cdrag2 , alpha/M_PI*180. ,true ); // Coefficient of drag
    Cl           = mult1*Cl1 + mult2*Cl2;
    Cd           = mult1*Cd1 + mult2*Cd2;
    Cn           = Cl * std::cos(phi) + Cd * std::sin(phi);
    Ct           = Cl * std::sin(phi) - Cd * std::cos(phi);
    dT_dr        = 0.5 * rho * W*W * chord * Cn * num_blades;
    dQ_dr        = 0.5 * rho * W*W * chord * Ct * num_blades * r;
    if (max_reached) std::cout << "NOT CONVERGED: " 
                               << r << " , "
                               << a << " , "
                               << a_new << " , "
                               << std::endl;
    // abort();
    return std::max( std::abs(a_new - a) , std::abs(a_prime_new - a_prime) );
  }



  void blade_integral( real         U_inf           ,   // input : inflow wind speed (m/s)
                       int          num_blades      ,   // input : number of blades
                       real         rho             ,   // input : total air density (kg/m^3)
                       real         gen_eff         ,   // input : Proportion of torque that generates power
                       bool         tloss           ,   // input : whether to include tip loss
                       bool         hloss           ,   // input : whether to include hub loss
                       real         pitch           ,   // input : pitch angle (radians)
                       real         omega           ,   // input : rotation rate (radians / second)
                       int          nrad            ,   // input : Number of points to sample along the blade
                       realHost1d & out_dT_dr       ,   // output: thrust values at section mid points
                       realHost1d & out_dQ_dr       ,   // output: torque values at section mid points
                       real       & out_thrust      ,   // output: total thrust (N)
                       real       & out_torque      ,   // output: total torque (N m)
                       real       & out_power       ,   // output: power generation (W)
                       real       & out_C_T         ,   // output: thrust coefficient
                       real       & out_C_P         ,   // output: power coefficient
                       real       & out_C_Q         ,
                       realHost1d & out_phi_r       ,
                       realHost1d & out_alpha_r     ,
                       realHost1d & out_Cn_r        ,
                       realHost1d & out_Ct_r        ,
                       realHost1d & out_a_r         ,
                       realHost1d & out_ap_r
                       ) {
    int  mxiter  = 200;            // Maximum number of iterations
    real tol     = 1.e-6;          // Tolerance for convergence
    real thrust  = 0;              // For accumulating the total thrust
    real torque  = 0;              // For accumulating the total torque
    int  nseg    = foil_mid.size();
    // Loop over blade segments and compute maximum
    for (int irad = 0; irad < nrad; irad++) {
      real       r          = R_hub + (R-R_hub) * (irad+0.5) / nrad;
      real       dr         = (R-R_hub)/nrad;
      real       twist      = linear_interp(foil_mid,foil_twist,r,true);
      real       chord      = linear_interp(foil_mid,foil_chord,r,true);
      real mult1, mult2;
      realHost1d ref_alpha1;
      realHost1d ref_clift1;
      realHost1d ref_cdrag1;
      realHost1d ref_alpha2;
      realHost1d ref_clift2;
      realHost1d ref_cdrag2;
      if (r <= foil_mid(0)) {
        mult1 = 0.5;
        mult2 = 0.5;
        ref_alpha1 = foil_alpha(foil_id(0));
        ref_clift1 = foil_clift(foil_id(0));
        ref_cdrag1 = foil_cdrag(foil_id(0));
        ref_alpha2 = foil_alpha(foil_id(0));
        ref_clift2 = foil_clift(foil_id(0));
        ref_cdrag2 = foil_cdrag(foil_id(0));
      } else if (r >= foil_mid(nseg-1)) {
        mult1 = 0.5;
        mult2 = 0.5;
        ref_alpha1 = foil_alpha(foil_id(nseg-1));
        ref_clift1 = foil_clift(foil_id(nseg-1));
        ref_cdrag1 = foil_cdrag(foil_id(nseg-1));
        ref_alpha2 = foil_alpha(foil_id(nseg-1));
        ref_clift2 = foil_clift(foil_id(nseg-1));
        ref_cdrag2 = foil_cdrag(foil_id(nseg-1));
      } else {
        int iseg1 = 0;
        for (int i=0; i < nseg-1; i++) { if (r >= foil_mid(i) && r <= foil_mid(i+1)) { iseg1 = i; break; } }
        int iseg2 = iseg1+1;
        mult1 = (foil_mid(iseg2) - r)/(foil_mid(iseg2)-foil_mid(iseg1));
        mult2 = (r - foil_mid(iseg1))/(foil_mid(iseg2)-foil_mid(iseg1));
        ref_alpha1 = foil_alpha(foil_id(iseg1));
        ref_clift1 = foil_clift(foil_id(iseg1));
        ref_cdrag1 = foil_cdrag(foil_id(iseg1));
        ref_alpha2 = foil_alpha(foil_id(iseg2));
        ref_clift2 = foil_clift(foil_id(iseg2));
        ref_cdrag2 = foil_cdrag(foil_id(iseg2));
      }
      real a, ap, phi, alpha, Cl, Cd, Cn, Ct, W, dT_dr, dQ_dr, F, sigma; // outputs
      real conv = blade_element( r , twist , chord , omega , mult1 , mult2 ,                                 // inputs
                                 ref_alpha1 , ref_clift1 , ref_cdrag1 ,                                      // inputs
                                 ref_alpha2 , ref_clift2 , ref_cdrag2 ,                                      // inputs
                                 U_inf , pitch , num_blades , rho , tloss , hloss , mxiter , tol ,           // inputs
                                 a , ap , phi , alpha , Cl , Cd , Cn , Ct , W , dT_dr , dQ_dr , F , sigma ); // outputs
      // if (conv > tol) std::cout << "NOT CONVERGED: r: " << r << ";  conv == " << conv << std::endl;
      out_dT_dr  (irad) = dT_dr;
      out_dQ_dr  (irad) = dQ_dr;
      out_phi_r  (irad) = phi;
      out_alpha_r(irad) = alpha;
      out_Cn_r   (irad) = Cn;
      out_Ct_r   (irad) = Ct;
      out_a_r    (irad) = a;
      out_ap_r   (irad) = ap;
      thrust += dT_dr*dr;
      torque += dQ_dr*dr;
    }
    real power = torque*omega*gen_eff;
    out_thrust = thrust;
    out_torque = torque;
    out_power  = power ;
    out_C_T    = thrust/(0.5*rho*M_PI*R*R*U_inf*U_inf)      ;
    out_C_Q    = torque/(0.5*rho*M_PI*R*R*R*U_inf*U_inf)    ;
    out_C_P    = power /(0.5*rho*M_PI*R*R*U_inf*U_inf*U_inf);
    // abort();
  }



  void auto_pitch( realHost1d & winds           ,   // input : inflow wind speeds (m/s)
                   int          num_blades      ,   // input : number of blades
                   real         rho             ,   // input : total air density (kg/m^3)
                   real         gen_eff         ,   // input : Proportion of torque that generates power
                   bool         tloss           ,   // input : whether to include tip loss
                   bool         hloss           ,   // input : whether to include hub loss
                   real         max_power       ,   // input : whether to include hub loss
                   real         max_thrust_prop ,   // input : whether to include hub loss
                   int          nrad            ,   // input : number of points to sample along blade
                   realHost1d & out_pitch       ,   // output: pitch angle (radians)
                   realHost1d & out_omega       ,   // output: rotation rate (radians / second)
                   realHost2d & out_dT_dr       ,   // output: thrust values at section mid points
                   realHost2d & out_dQ_dr       ,   // output: torque values at section mid points
                   realHost1d & out_thrust      ,   // output: total thrust (N)
                   realHost1d & out_torque      ,   // output: total torque (N m)
                   realHost1d & out_power       ,   // output: power generation (W)
                   realHost1d & out_C_T         ,   // output: thrust coefficient
                   realHost1d & out_C_P         ,   // output: power coefficient
                   realHost1d & out_C_Q         ,
                   realHost2d & out_phi_r       ,
                   realHost2d & out_alpha_r,
                   realHost2d & out_Cn_r,
                   realHost2d & out_Ct_r,
                   realHost2d & out_a_r,
                   realHost2d & out_ap_r
                   ) { // output: torque coefficient
    int  nwinds = winds.size();
    out_pitch    = realHost1d("out_pitch  ",nwinds);
    out_omega    = realHost1d("out_omega  ",nwinds);
    out_dT_dr    = realHost2d("out_dT_dr  ",nwinds,nrad); // I'm adding a zero and radius point
    out_dQ_dr    = realHost2d("out_dQ_dr  ",nwinds,nrad); // I'm adding a zero and radius point
    out_thrust   = realHost1d("out_thrust ",nwinds);
    out_torque   = realHost1d("out_torque ",nwinds);
    out_power    = realHost1d("out_power  ",nwinds);
    out_C_T      = realHost1d("out_C_T    ",nwinds);
    out_C_P      = realHost1d("out_C_P    ",nwinds);
    out_C_Q      = realHost1d("out_C_Q    ",nwinds);
    out_phi_r    = realHost2d("out_phi    ",nwinds,nrad);
    out_alpha_r  = realHost2d("out_alpha  ",nwinds,nrad);
    out_Cn_r     = realHost2d("out_Cn     ",nwinds,nrad);
    out_Ct_r     = realHost2d("out_Ct     ",nwinds,nrad);
    out_a_r      = realHost2d("out_a      ",nwinds,nrad);
    out_ap_r     = realHost2d("out_ap     ",nwinds,nrad);
    real pitch_min = 0;
    real pitch_max = M_PI/2.;

    // Determine the maximum thrust among the input wind speeds
    for (int iwind = 0; iwind < nwinds; iwind++) {
      std::cout << winds(iwind) << std::endl;
      real U_inf = winds(iwind);
      real omega0 = linear_interp(rwt_mag,rwt_rot,U_inf,false); // Rotation rate (rad/sec)
      real omega = omega0;
      real pitch = 0;
      auto dT_dr_loc   = out_dT_dr  .slice<1>(iwind,0);
      auto dQ_dr_loc   = out_dQ_dr  .slice<1>(iwind,0);
      auto phi_r_loc   = out_phi_r  .slice<1>(iwind,0);
      auto alpha_r_loc = out_alpha_r.slice<1>(iwind,0);
      auto Cn_r_loc    = out_Cn_r   .slice<1>(iwind,0);
      auto Ct_r_loc    = out_Ct_r   .slice<1>(iwind,0);
      auto a_r_loc     = out_a_r    .slice<1>(iwind,0);
      auto ap_r_loc    = out_ap_r   .slice<1>(iwind,0);
      blade_integral( U_inf , num_blades , rho , gen_eff , tloss , hloss , pitch , omega , nrad ,
                      dT_dr_loc , dQ_dr_loc ,
                      out_thrust(iwind) , out_torque(iwind) , out_power(iwind) , out_C_T(iwind) ,
                      out_C_P(iwind) , out_C_Q(iwind) , phi_r_loc , alpha_r_loc , Cn_r_loc , Ct_r_loc ,
                      a_r_loc , ap_r_loc );
      out_pitch(iwind) = pitch;
      out_omega(iwind) = omega;
      while (out_power(iwind) > max_power) {
        // if (yakl::intrinsics::minval(dT_dr_loc) >= 0) { pitch += 0.01/180.*M_PI; }
        // else                                          { omega -= 0.01*2*M_PI/60.; }
        pitch += 0.01/180*M_PI;
        blade_integral( U_inf , num_blades , rho , gen_eff , tloss , hloss , pitch , omega , nrad ,
                        dT_dr_loc , dQ_dr_loc ,
                        out_thrust(iwind) , out_torque(iwind) , out_power(iwind) , out_C_T(iwind) ,
                        out_C_P(iwind) , out_C_Q(iwind) , phi_r_loc , alpha_r_loc , Cn_r_loc , Ct_r_loc ,
                        a_r_loc , ap_r_loc );
        out_pitch(iwind) = pitch;
        out_omega(iwind) = omega;
      }
    }

    real max_thrust = yakl::intrinsics::maxval(out_thrust)*max_thrust_prop;

    // Determine the maximum thrust among the input wind speeds
    for (int iwind = 0; iwind < nwinds; iwind++) {
      real U_inf  = winds(iwind);
      real pitch  = out_pitch(iwind);
      real omega  = out_omega(iwind);
      while (out_thrust(iwind) > max_thrust) {
        pitch += 0.01/180.*M_PI;
        auto dT_dr_loc = out_dT_dr.slice<1>(iwind,0);
        auto dQ_dr_loc = out_dQ_dr.slice<1>(iwind,0);
        auto phi_r_loc   = out_phi_r  .slice<1>(iwind,0);
        auto alpha_r_loc = out_alpha_r.slice<1>(iwind,0);
        auto Cn_r_loc    = out_Cn_r   .slice<1>(iwind,0);
        auto Ct_r_loc    = out_Ct_r   .slice<1>(iwind,0);
        auto a_r_loc     = out_a_r    .slice<1>(iwind,0);
        auto ap_r_loc    = out_ap_r   .slice<1>(iwind,0);
        blade_integral( U_inf , num_blades , rho , gen_eff , tloss , hloss , pitch , omega , nrad ,
                        dT_dr_loc , dQ_dr_loc ,
                        out_thrust(iwind) , out_torque(iwind) , out_power(iwind) , out_C_T(iwind) ,
                        out_C_P(iwind) , out_C_Q(iwind) , phi_r_loc , alpha_r_loc , Cn_r_loc , Ct_r_loc ,
                        a_r_loc , ap_r_loc );
        out_pitch(iwind) = pitch;
        out_omega(iwind) = omega;
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

    bool tloss           = true;   // Use tip loss?
    bool hloss           = true;   // Use hub loss?
    real gen_eff         = 0.944;  // Efficiency of power generation
    int  num_blades      = 3;
    real rho             = 1.225;  // Air density
    real max_power       = 5.29661e6;
    real max_thrust_prop = 1.0;

    // real U_inf = 11;
    // int  nrad  = 200;
    // real pitch = 0;
    // realHost1d out_dT_dr("out_dT_dr",nrad);
    // realHost1d out_dQ_dr("out_dQ_dr",nrad);
    // real out_thrust,out_torque,out_power,out_C_T, out_C_P, out_C_Q;
    // bem.blade_integral( U_inf , num_blades , rho , gen_eff , tloss , hloss , pitch , nrad , out_dT_dr , out_dQ_dr ,
    //                     out_thrust , out_torque , out_power , out_C_T , out_C_P , out_C_Q );
    // std::ofstream fh("output.txt");
    // for (int irad = 0; irad < nrad; irad++) {
    //   real r = bem.R * (irad+0.5) / nrad;
    //   fh << std::scientific << std::setw(15) <<     r           << "  "
    //      << std::scientific << std::setw(15) << out_dT_dr(irad) << "  "
    //      << std::scientific << std::setw(15) << out_dQ_dr(irad) << std::endl;
    // }
    // fh.close();


    real cut_in  = bem.rwt_mag(0);
    real cut_out = bem.rwt_mag(bem.rwt_mag.size()-1);
    realHost1d winds("winds",(int)((cut_out-cut_in))+1);
    for (int i=0; i < winds.size(); i++) {
      winds(i) = cut_in + (cut_out-cut_in)*((real)i)/(winds.size()-1.);
    }

    int nrad = 100;

    realHost2d out_dT_dr, out_dQ_dr, out_phi_r, out_alpha_r, out_Cn_r, out_Ct_r, out_a_r, out_ap_r;
    realHost1d out_pitch, out_omega, out_thrust, out_torque, out_power, out_C_T, out_C_P, out_C_Q;

    bem.auto_pitch( winds , num_blades , rho , gen_eff , tloss , hloss , max_power , max_thrust_prop , nrad ,
                    out_pitch , out_omega , out_dT_dr , out_dQ_dr , out_thrust , out_torque , out_power ,
                    out_C_T , out_C_P , out_C_Q , out_phi_r , out_alpha_r , out_Cn_r , out_Ct_r , out_a_r , out_ap_r );

    realHost1d segment("segment",nrad);
    for (int i=0; i < nrad; i++) { segment(i) = bem.R_hub + (bem.R-bem.R_hub) * (i+0.5) / nrad; }
    yakl::SimpleNetCDF nc;
    nc.create( "bem.nc" , yakl::NETCDF_MODE_REPLACE );
    nc.write( winds        , "wind"    , {"wind"});
    nc.write( segment      , "segment" , {"segment"});
    nc.write( out_dT_dr    , "dT_dr"   , {"wind","segment"});
    nc.write( out_dQ_dr    , "dQ_dr"   , {"wind","segment"});
    nc.write( out_phi_r    , "phi_r"   , {"wind","segment"});
    nc.write( out_alpha_r  , "alpha_r" , {"wind","segment"});
    nc.write( out_Cn_r     , "Cn_r"    , {"wind","segment"});
    nc.write( out_Ct_r     , "Ct_r"    , {"wind","segment"});
    nc.write( out_a_r      , "a_r"     , {"wind","segment"});
    nc.write( out_ap_r     , "ap_r"    , {"wind","segment"});
    nc.write( out_pitch    , "pitch"   , {"wind"});
    nc.write( out_omega    , "omega"   , {"wind"});
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


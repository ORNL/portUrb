
#include "coupler.h"

int constexpr MAX_FIELDS = 50;
typedef yakl::SArray<realHost1d,1,MAX_FIELDS> MultiField;

typedef std::tuple<real, real, real, real, std::string> FOIL_LINE;

namespace YAML {
  template<> struct convert<FOIL_LINE> {
    static Node encode(const FOIL_LINE& rhs) {
      Node node;
      node.push_back(std::get<0>(rhs));
      node.push_back(std::get<1>(rhs));
      node.push_back(std::get<2>(rhs));
      node.push_back(std::get<3>(rhs));
      node.push_back(std::get<4>(rhs));
      return node;
    }

    static bool decode(const Node& node, FOIL_LINE& rhs) {
      if (!node.IsSequence() || node.size() != 5) {
        return false;
      }
      rhs = FOIL_LINE(node[0].as<real>(),
                      node[1].as<real>(),
                      node[2].as<real>(),
                      node[3].as<real>(),
                      node[4].as<std::string>());
      return true;
    }
  };
} // namespace YAML


real linear_interp( realHost1d const & aref                ,
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


real clip( real v , real mn , real mx ) { return std::max(mn,std::min(mx,v)); }



void bem_aerodyn( real         r                 ,     // input : radial position (m)
                  realHost1d & foil_sec          ,     // input : airfoil sections markers (m)
                  realHost1d & foil_twist        ,     // input : airfoil section twist angle (deg)
                  realHost1d & foil_chord        ,     // input : airfoil section chord length (m)
                  realHost1d & foil_id           ,     // input : airfoil section coef of lift & drag lookup index
                  MultiField & foil_alpha        ,     // input : angle of attack for coef of lift & drag lookup
                  MultiField & foil_clift        ,     // input : coefficient of lift based on alpha
                  MultiField & foil_cdrag        ,     // input : coefficient of drag based on alpha
                  realHost1d & rwt_mag           ,     // input : controller inflow wind speed (m/s)
                  realHost1d & rwt_rot           ,     // input : controller rotation rate (rpm)
                  real         U_inf             ,     // input : inflow wind speed
                  real         pitch             ,     // input : pitch of the blades (radians)
                  int          B                 ,     // input : number of blades
                  real         R                 ,     // input : blade radius (m)
                  real         R_hub             ,     // input : hub radius (m)
                  real         rho               ,     // input : total air density (kg/m^3)
                  bool         use_tip_loss      ,     // input : whether to include tip loss
                  bool         use_hub_loss      ,     // input : whether to include hub loss
                  int          max_iter          ,     // input : maximum number of iterations for convergence
                  real         tol               ,     // input : tolerance for convergence
                  real       & a                 ,     // output: axial induction factor (-)
                  real       & ap                ,     // output: tangential induction factor (-)
                  real       & phi               ,     // output: inflow angle (rad)
                  real       & alpha             ,     // output: angle of attack (rad)
                  real       & Cl                ,     // output: lift coefficient (-)
                  real       & Cd                ,     // output: drag coefficient (-)
                  real       & Cn                ,     // output: normal force coefficient (-) (in rotor plane coordinates)
                  real       & Ct                ,     // output: tangential force coefficient (-) (in rotor plane coordinates)
                  real       & W                 ,     // output: relative velocity magnitude (m/s)
                  real       & dT_dr             ,     // output: thrust per unit span (N/m) (all B blades)
                  real       & dQ_dr             ,     // output: torque per unit span (N·m/m) (all B blades)
                  real       & F                 ) {   // output: combined tip-hub loss factor (-)
  // Get the chord length, twist angle, and coefficients of lift and drag lookup arrays from the foil segment arrays
  real       chord;     // Chord length (m)
  real       twist;     // Twist angle (rad)
  realHost1d ref_alpha; // alpha array for clift, cdrag lookup
  realHost1d ref_clift; // clift array for coefficient of lift lookup
  realHost1d ref_cdrag; // cdrag array for coefficient of drag lookup
  int        id;
  for (int iseg = 0; iseg < foil_sec.size()-1; iseg++) {
    if (r >= foil_sec(iseg) && r < foil_sec(iseg+1)) {
      twist     = foil_twist(iseg)/180*M_PI;
      chord     = foil_chord(iseg);
      id        = foil_id(iseg);
      ref_alpha = foil_alpha(id);
      ref_clift = foil_clift(id);
      ref_cdrag = foil_cdrag(id);
    }
  }
  if (r == foil_sec(foil_sec.size()-1)) {
    int iseg  = foil_sec.size()-2;
    twist     = foil_twist(iseg)/180*M_PI;
    chord     = foil_chord(iseg);
    id        = foil_id(iseg);
    ref_alpha = foil_alpha(id);
    ref_clift = foil_clift(id);
    ref_cdrag = foil_cdrag(id);
  }
  // Get rotation rate in radians per second from U_inf
  real omega    = linear_interp( rwt_mag , rwt_rot , U_inf , false ) * 2*M_PI / 60; // rpm to rad/sec conversion
  real sigma    = B * chord / (2 * M_PI * r); // Local solidity - ratio of blade area to annular area at this radius
  real lambda_r = omega * r / U_inf;          // Local tip speed ratio at this radial station
  real beta     = twist + pitch;              // Total blade pitch angle (twist + pitch) (radians)
  // Next three lines compute initial axial induction factor
  real t1   = 4 + M_PI * lambda_r * sigma;
  real disc = t1*t1 - 4*M_PI*sigma + 8*M_PI*lambda_r*sigma*beta;
  a         = disc > 0 ? clip( (t1 - std::sqrt(std::max(disc,(real)0)))/2 , (real) 0 , (real) 0.5 ) : 0.3;
  ap        = 0.0; // tangential induction factor
  real a_old, a_new, ap_old, ap_new;
  for (int iter = 0; iter < max_iter; iter++) {
    a_old  = a;
    ap_old = ap;
    // Eq 21: Inflow angle
    phi      = std::atan2( U_inf*(1-a) , omega*r*(1+ap) );
    real sp  = std::sin(phi);
    real cp  = std::cos(phi);
    sp = std::max( abs(sp) , (real) 1e-6 ) * (sp >= 0 ? 1 : -1);
    // Interpolate coefficients of lift and drag from alpha in degrees
    alpha = phi - beta;  // Angle of attack
    Cl    = linear_interp( ref_alpha , ref_clift , alpha/M_PI*180 , true ); // Coefficient of lift
    Cd    = linear_interp( ref_alpha , ref_cdrag , alpha/M_PI*180 , true ); // Coefficient of drag
    // Eqs 23-25: Loss factors
    real F_tip = (2./M_PI) * std::acos(std::exp(-(B/2.)*(R-r    )/(r*sp))); // Tip loss
    real F_hub = (2./M_PI) * std::acos(std::exp(-(B/2.)*(r-R_hub)/(r*sp))); // Hub loss
    real F     = std::max( F_tip*F_hub , (real) 1e-10 ); // Total loss
        // F_tip = (2/np.pi) * np.arccos(np.exp(-(B/2)*(R-r    )/(r*sp)))
        // F_hub = (2/np.pi) * np.arccos(np.exp(-(B/2)*(r-R_hub)/(r*sp)))
        // F = max(F_tip * F_hub, 1e-10)
    Cn = Cl*cp + Cd*sp; // Normal force coefficient (perpendicular to rotor plane, thrust direction)
    // Eq 27: Standard BEM
    a_new = std::abs(Cn) > 1e-10 ? 1./(1 + 4*F*sp*sp/(sigma*Cn)) : 0.; // Axial induction from standard BEM theory
    // Eq 26: Glauert correction
    real CT = 1 + sigma * (1 - a)*(1 - a) * Cn / (sp*sp); // Coefficient of thrust
    if (CT > 0.96*F && a_new > 0.4) {
      real inner = CT*(50-36*F) + 12*F*(3*F-4);
      if (inner >= 0) a_new = (18*F - 20 - 3*std::sqrt(inner))/(36*F - 50);
    }
    // Eq 28: Tangential induction
    Ct = Cl*sp - Cd*cp;
    ap_new = 0;
    if (abs(Ct) > 1e-10 && abs(cp) > 1e-10) {
      real d = 4*F*sp*cp/(sigma*Ct) - 1;
      ap_new = abs(d) > 1e-10 ? 1./d : 0.0;
    }
    // Safety limits on induction factors
    a_new  = clip( a_new  , -0.5 , 0.95 );
    ap_new = clip( ap_new , -0.5 , 0.5  );
    if ( abs(a_new-a_old) < tol && abs(ap_new-ap_old) < tol ) {
      a  = a_new;
      ap = ap_new;
      break;
    }
    // Under-relaxation
    a  = 0.25*a_new  + 0.75*a_old;  // Axial induction factor
    ap = 0.25*ap_new + 0.75*ap_old; // Tangential induction factor
  }
  phi     = std::atan2( U_inf*(1-a) , omega*r*(1+ap) );
  alpha   = phi - beta;                                 // Angle of attack
  Cl      = linear_interp( ref_alpha , ref_clift , alpha/M_PI*180 , true ); // Coefficient of lift
  Cd      = linear_interp( ref_alpha , ref_cdrag , alpha/M_PI*180 , true ); // Coefficient of drag
  W       = std::sqrt( (U_inf*(1-a))*(U_inf*(1-a)) + (omega*r*(1+ap))*(omega*r*(1+ap)) ); // relative velocity magnitude
  real sp = std::sin(phi);
  real cp = std::cos(phi);
  Cn      = Cl*cp + Cd*sp; // Normal force
  Ct      = Cl*sp - Cd*cp; // Tangential force
  dT_dr   = 0.5*rho*W*W*chord*Cn*B;
  dQ_dr   = 0.5*rho*W*W*chord*Ct*B*r;
}


int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    int  B            = 3;      // Number of blades
    real rho          = 1.225;  // Air density
    // GET YAML DATA
    std::string turbine_file = "./inputs/NREL_5MW_126_RWT.yaml";
    YAML::Node  node         = YAML::LoadFile(turbine_file);
    auto R            = node["blade_radius"      ].as<real>();
    auto R_hub        = (real) 1.5; // node["tower_top_radius"  ].as<real>();
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
    int nseg  = foil_summary.size();
    int nfoil = foil_names.size();
    realHost1d foil_sec  ("foil_sec"  ,nseg+1);
    realHost1d foil_twist("foil_twist",nseg);
    realHost1d foil_chord("foil_chord",nseg);
    realHost1d foil_id   ("foil_id"   ,nseg);
    for (int iseg=0; iseg < nseg ; iseg++) {
      real mid = std::get<0>(foil_summary.at(iseg));
      real len = std::get<2>(foil_summary.at(iseg));
      foil_sec  (iseg) = mid - len/2;
      foil_twist(iseg) = std::get<1>(foil_summary.at(iseg));
      foil_chord(iseg) = std::get<3>(foil_summary.at(iseg));
      int id = -1;
      for (int ifoil = 0; ifoil < nfoil; ifoil++) {
        if (std::get<4>(foil_summary.at(iseg)) == foil_names.at(ifoil)) { id = ifoil; break; }
      }
      foil_id   (iseg) = id;
    }
    foil_sec(nseg) = R;
    MultiField foil_alpha;
    MultiField foil_clift;
    MultiField foil_cdrag;
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
    realHost1d rwt_mag    ("rwt_mag"    ,nrwt);
    realHost1d rwt_ct     ("rwt_ct"     ,nrwt);
    realHost1d rwt_cp     ("rwt_cp"     ,nrwt);
    realHost1d rwt_pwr_mw ("rwt_pwr_mw" ,nrwt);
    realHost1d rwt_rot_rpm("rwt_rot_rpm",nrwt);
    for (int irwt = 0; irwt < nrwt; irwt++) {
      rwt_mag    (irwt) = velmag  .at(irwt);
      rwt_ct     (irwt) = cthrust .at(irwt);
      rwt_cp     (irwt) = cpower  .at(irwt);
      rwt_pwr_mw (irwt) = power_mw.at(irwt);
      rwt_rot_rpm(irwt) = rot_rpm .at(irwt);
    }



    real r1     = foil_sec(0)   ;
    real r2     = foil_sec(nseg);
    bool tloss  = true;  // Use tip loss?
    bool hloss  = true;  // Use hub loss?
    int  mxiter = 100;   // Maximum number of iterations
    real tol    = 1.e-6; // Tolerance for convergence
    real pitch  = 0;     // Blade pitch
    real U_inf  = 11.4;  // Inflow wind speed
    int  nrad   = 100;
    real thrust = 0;
    for (int i=0; i < nrad; i++) {
      real r     = r1 + (r2-r1)*((real)i)/(nrad-1.);
      real a, ap, phi, alpha, Cl, Cd, Cn, Ct, W, dT_dr, dQ_dr, F;
      bem_aerodyn( r , foil_sec , foil_twist , foil_chord , foil_id , foil_alpha , foil_clift , foil_cdrag ,    // inputs
                   rwt_mag , rwt_rot_rpm , U_inf , pitch , B , R , R_hub , rho , tloss , hloss , mxiter , tol , // inputs
                   a , ap , phi , alpha , Cl , Cd , Cn , Ct , W , dT_dr , dQ_dr , F );                          // outputs
      // std::cout << r << " , " << dT_dr << " , " << dQ_dr << std::endl;
      thrust += dT_dr*(r2-r1)/99;
    }
    std::cout << "Thrust: " << thrust/(0.5*rho*M_PI*R*R*U_inf*U_inf) << std::endl;


    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}


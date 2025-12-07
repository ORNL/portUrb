
#include "coupler.h"
#include <fstream>

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



real prandtl_tip_loss(real r, real R, int num_blades, real phi_rad) {
  if (std::abs(std::sin(phi_rad)) < 1e-6) return 1;
  real f_tip = (num_blades / 2.0) * (R - r) / (r * std::sin(phi_rad));
  if (f_tip > 50) return 1;
  real F_tip = (2.0 / M_PI) * std::acos(std::exp(-f_tip));
  return std::max(F_tip, (real)0.0001);
}



real prandtl_hub_loss(real r, real R_hub, int num_blades, real phi_rad) {
    if (std::abs(std::sin(phi_rad)) < 1e-6) return 1;
    real f_hub = (num_blades / 2.0) * (r - R_hub) / (r * std::sin(phi_rad));
    if (f_hub > 50) return 1;
    real F_hub = (2.0 / M_PI) * std::acos(std::exp(-f_hub));
    return std::max(F_hub, (real) 0.0001);
}



real bem_aerodyn( real         r                 ,   // input : radial position (m)
                  real         twist             ,   // input : twist angle (radians)
                  real         chord             ,   // input : 
                  realHost1d & ref_alpha         ,   // input : 
                  realHost1d & ref_clift         ,   // input : 
                  realHost1d & ref_cdrag         ,   // input : 
                  realHost1d & rwt_mag           ,   // input : controller inflow wind speed (m/s)
                  realHost1d & rwt_rot           ,   // input : controller rotation rate (radians / sec)
                  real         U_inf             ,   // input : inflow wind speed
                  real         pitch             ,   // input : pitch of the blades (radians)
                  int          num_blades        ,   // input : number of blades
                  real         R                 ,   // input : blade radius (m)
                  real         R_hub             ,   // input : hub radius (m)
                  real         rho               ,   // input : total air density (kg/m^3)
                  bool         tip_loss          ,   // input : whether to include tip loss
                  bool         hub_loss          ,   // input : whether to include hub loss
                  int          max_iter          ,   // input : maximum number of iterations for convergence
                  real         tol               ,   // input : tolerance for convergence
                  real       & a                 ,   // output: axial induction factor (-)
                  real       & a_prime           ,   // output: tangential induction factor (-)
                  real       & phi               ,   // output: inflow angle (rad)
                  real       & alpha             ,   // output: angle of attack (rad)
                  real       & Cl                ,   // output: lift coefficient (-)
                  real       & Cd                ,   // output: drag coefficient (-)
                  real       & Cn                ,   // output: normal force coefficient (-)
                  real       & Ct                ,   // output: tangential force coefficient (-)
                  real       & W                 ,   // output: relative velocity magnitude (m/s)
                  real       & dT_dr             ,   // output: thrust per unit span (N/m) (all blades)
                  real       & dQ_dr             ,   // output: torque per unit span (N·m/m) (all blades)
                  real       & F                 ,   // output: combined tip-hub loss factor (-)
                  real       & sigma             ) { // output: local solidarity factor
  a              = 0;                                    // Axial induction factor
  a_prime        = 0;                                    // Tangential induction factor
  sigma          = num_blades * chord / (2. * M_PI * r); // Local solidity
  real theta     = twist + pitch;                        // Blade section angle (twist + pitch) (rad)
  real omega     = linear_interp( rwt_mag , rwt_rot , U_inf , true ); // Rotation rate (rad/sec)
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
    real F = 1;  // Total loss from tip and hub
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
  dT_dr        = 0.5 * rho * W*W * chord * Cn * num_blades;
  dQ_dr        = 0.5 * rho * W*W * chord * Ct * num_blades * r;
  return std::max( std::abs(a_new - a) , std::abs(a_prime_new - a_prime) );
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
    auto R_hub        = node["tower_top_radius"  ].as<real>();
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
    realHost1d foil_mid  ("foil_mid"  ,nseg);
    realHost1d foil_len  ("foil_len"  ,nseg);
    realHost1d foil_twist("foil_twist",nseg);
    realHost1d foil_chord("foil_chord",nseg);
    realHost1d foil_id   ("foil_id"   ,nseg);
    for (int iseg=0; iseg < nseg ; iseg++) {
      foil_mid  (iseg) = std::get<0>(foil_summary.at(iseg));
      foil_twist(iseg) = std::get<1>(foil_summary.at(iseg));
      foil_len  (iseg) = std::get<2>(foil_summary.at(iseg));
      foil_chord(iseg) = std::get<3>(foil_summary.at(iseg));
      int id = -1;
      for (int ifoil = 0; ifoil < nfoil; ifoil++) {
        if (std::get<4>(foil_summary.at(iseg)) == foil_names.at(ifoil)) { id = ifoil; break; }
      }
      foil_id   (iseg) = id;
    }
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
    realHost1d rwt_mag    ("rwt_mag"   ,nrwt);
    realHost1d rwt_ct     ("rwt_ct"    ,nrwt);
    realHost1d rwt_cp     ("rwt_cp"    ,nrwt);
    realHost1d rwt_pwr_mw ("rwt_pwr_mw",nrwt);
    realHost1d rwt_rot    ("rwt_rot"   ,nrwt);
    for (int irwt = 0; irwt < nrwt; irwt++) {
      rwt_mag   (irwt) = velmag  .at(irwt);
      rwt_ct    (irwt) = cthrust .at(irwt);
      rwt_cp    (irwt) = cpower  .at(irwt);
      rwt_pwr_mw(irwt) = power_mw.at(irwt);
      rwt_rot   (irwt) = rot_rpm .at(irwt)*2.*M_PI/60.;
    }

    // SET PARAMETERS FOR BEM COMPUTATIONS AND FORCE ACCUMULATIONS
    bool tloss   = true;   // Use tip loss?
    bool hloss   = true;   // Use hub loss?
    int  mxiter  = 1000;    // Maximum number of iterations
    real tol     = 1.e-6;  // Tolerance for convergence
    real pitch   = 10.45/180*M_PI;      // Blade pitch
    real U_inf   = 15;   // Inflow wind speed
    real gen_eff = 0.944;  // Efficiency of power generation
    real thrust  = 0;
    real torque  = 0;
    std::ofstream of("output.txt");
    of << std::scientific << std::setprecision(5) << std::setw(15) << "r    " << "  " <<
          std::scientific << std::setprecision(5) << std::setw(15) << "dT_dr" << "  " <<
          std::scientific << std::setprecision(5) << std::setw(15) << "dQ_dr" << "  " <<
          std::scientific << std::setprecision(5) << std::setw(15) << "a    " << "  " <<
          std::scientific << std::setprecision(5) << std::setw(15) << "ap   " << std::endl;
    // Loop over blade segments and compute thrust and torque properties at each
    for (int i=0; i < nseg; i++) {
      real       r         = foil_mid(i);
      real       dr        = foil_len(i);
      real       twist     = foil_twist(i)/180.*M_PI;
      real       chord     = foil_chord(i);
      realHost1d ref_alpha = foil_alpha(foil_id(i));
      realHost1d ref_clift = foil_clift(foil_id(i));
      realHost1d ref_cdrag = foil_cdrag(foil_id(i));

      real a, ap, phi, alpha, Cl, Cd, Cn, Ct, W, dT_dr, dQ_dr, F, sigma; // outputs
      real conv = bem_aerodyn( r , twist , chord , ref_alpha , ref_clift , ref_cdrag , rwt_mag , rwt_rot , // inputs
                               U_inf , pitch , B , R , R_hub , rho , tloss , hloss , mxiter , tol ,        // inputs
                               a , ap , phi , alpha , Cl , Cd , Cn , Ct , W , dT_dr , dQ_dr , F , sigma ); // outputs
      if (conv > tol) std::cout << "NOT CONVERGED: r: " << r << ";  conv == " << conv << std::endl;

      of << std::scientific << std::setprecision(5) << std::setw(15) << r     << "  " <<
            std::scientific << std::setprecision(5) << std::setw(15) << dT_dr << "  " <<
            std::scientific << std::setprecision(5) << std::setw(15) << dQ_dr << "  " <<
            std::scientific << std::setprecision(5) << std::setw(15) << a     << "  " <<
            std::scientific << std::setprecision(5) << std::setw(15) << ap    << std::endl;
      thrust += dT_dr*dr;
      torque += dQ_dr*dr;
    }
    of.close();
    real power = torque*linear_interp(rwt_mag,rwt_rot,U_inf,false)*gen_eff;
    std::cout << "Thrust (kN)   : " << thrust/1e3                                  << std::endl;
    std::cout << "Torque (MN m) : " << torque/1e6                                  << std::endl;
    std::cout << "Power (MW)    : " << power/1e6                                   << std::endl;
    std::cout << "C_Thrust      : " << thrust/(0.5*rho*M_PI*R*R*U_inf*U_inf)       << std::endl;
    std::cout << "C_Torque      : " << torque/(0.5*rho*M_PI*R*R*R*U_inf*U_inf)     << std::endl;
    std::cout << "C_Power       : " << power /(0.5*rho*M_PI*R*R*U_inf*U_inf*U_inf) << std::endl;



    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}


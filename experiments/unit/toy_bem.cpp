
#include "coupler.h"


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


int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    std::string turbine_file = "./inputs/NREL_5MW_126_RWT.yaml";
    YAML::Node  node         = YAML::LoadFile(turbine_file);

    int  B            = 3;      // Number of blades
    real rho          = 1.225;  // Air density
    auto R            = node["blade_radius"    ].as<real>();
    auto R_hub        = node["tower_top_radius"].as<real>();
    auto foil_summary = node["airfoil_summary" ].as<std::vector<FOIL_LINE>>();
    auto foil_names   = node["airfoil_names"   ].as<std::vector<std::string>>();
    std::vector<std::vector<std::vector<real>>> foil_vals;
    for (int ifoil=0; ifoil < foil_names.size(); ifoil++) {
      foil_vals.push_back( node[foil_names.at(ifoil)].as<std::vector<std::vector<real>>>() );
    }
    int nseg  = foil_summary.size();
    int nfoil = foil_names.size();
    realHost1d foil_beg  ("foil_beg"  ,nseg);
    realHost1d foil_end  ("foil_end"  ,nseg);
    realHost1d foil_twist("foil_twist",nseg);
    realHost1d foil_chord("foil_chord",nseg);
    realHost1d foil_id   ("foil_id"   ,nseg);
    for (int iseg=0; iseg < nseg ; iseg++) {
      real mid = std::get<0>(foil_summary.at(iseg));
      real len = std::get<2>(foil_summary.at(iseg));
      foil_beg  (iseg) = mid - len/2;
      foil_end  (iseg) = mid + len/2;
      foil_twist(iseg) = std::get<1>(foil_summary.at(iseg));
      foil_chord(iseg) = std::get<3>(foil_summary.at(iseg));
      int id = -1;
      for (int ifoil = 0; ifoil < nfoil; ifoil++) {
        if (std::get<4>(foil_summary.at(iseg)) == foil_names.at(ifoil)) { id = ifoil; break; }
      }
      foil_id   (iseg) = id;
    }
    core::MultiField<real,1> foil_alpha;
    core::MultiField<real,1> foil_clift;
    core::MultiField<real,1> foil_cdrag;
    for (int ifoil = 0; ifoil < nfoil; ifoil++) {
      int nalpha = foil_vals.at(ifoil).size();
      real1d loc_alpha("foil_alpha",nalpha);
      real1d loc_clift("foil_clift",nalpha);
      real1d loc_cdrag("foil_cdrag",nalpha);
      for (int ialpha = 0; ialpha < nalpha; ialpha++) {
        loc_alpha(ialpha) = foil_vals.at(ifoil).at(ialpha).at(0);
        loc_clift(ialpha) = foil_vals.at(ifoil).at(ialpha).at(1);
        loc_cdrag(ialpha) = foil_vals.at(ifoil).at(ialpha).at(2);
      }
      foil_alpha.add_field( loc_alpha );
      foil_clift.add_field( loc_clift );
      foil_cdrag.add_field( loc_cdrag );
    }

    foil_end(nseg-1) = R;
    std::cout << foil_beg   << std::endl;
    std::cout << foil_end   << std::endl;
    std::cout << foil_twist << std::endl;
    std::cout << foil_chord << std::endl;
    std::cout << foil_id    << std::endl;
    for (int ifoil = 0; ifoil < nfoil; ifoil++) {
      std::cout << foil_alpha.get_field(ifoil) << std::endl;
      std::cout << foil_clift.get_field(ifoil) << std::endl;
      std::cout << foil_cdrag.get_field(ifoil) << std::endl;
    }

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}


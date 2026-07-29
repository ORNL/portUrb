
#pragma once

#include <unordered_map>
#include "main_header.h"
#include "coupler.h"

namespace YAML {
  template<> struct convert<std::tuple<real,real,real,std::string>> {
    static Node encode(const std::tuple<real,real,real,std::string>& rhs);

    static bool decode(const Node& node, std::tuple<real,real,real,std::string>& rhs);
  };
} // namespace YAML


namespace modules {

  // For simplicity and for parallel efficiency, the TurbineActuatorLine class will assume that:
  //   * The turbine is always yawed to face the west direction (yaw of zero degrees)
  //   * There is a constant upwind direction that does not change (coupler option)
  //   * There is only one type of reference turbine for all turbines simulated
  //   * The inflow wind speed is a constant coupler option to determine the turbine's rotation rate
  //   * Grid spacing is the same in all three directions, x, y, and z
  struct TurbineActuatorLine {
    int static constexpr MAX_FIELDS = 100;
    typedef yakl::SArray<realHost1d,MAX_FIELDS> MultiFieldHost;
    typedef yakl::SArray<real1d    ,MAX_FIELDS> MultiFieldDev;

    // This class holds information about a reference wind turbine, including lookup tables for various properties
    //   and turbine geometric properties
    struct RefTurbine {
      int            B                ;
      real           R                ;
      real           R_hub            ;
      real           H                ;
      real           overhang         ; // Offset of blades from tower center (m). This is also the length of the hub flange
      real           hub_flange_height; // Height (and width) of the hub flange (m)
      real           tower_base_rad   ; // Radius of the tower base at ground or water level (m)
      real           tower_top_rad    ; // Radius of the tower top connected to hub flange (m)
      real           shaft_tilt       ; // Shaft tilt in radians
      realHost1d     host_rad_locs    ;
      realHost1d     host_foil_mid    ;
      realHost1d     host_foil_twist  ;
      realHost1d     host_foil_chord  ;
      intHost1d      host_foil_id     ;
      MultiFieldHost host_foil_alpha  ;
      MultiFieldHost host_foil_clift  ;
      MultiFieldHost host_foil_cdrag  ;
      realHost1d     host_rwt_mag     ;
      realHost1d     host_rwt_ct      ;
      realHost1d     host_rwt_cp      ;
      realHost1d     host_rwt_pwr_mw  ;
      realHost1d     host_rwt_rot     ;
      real1d         dev_rad_locs     ;
      real1d         dev_foil_mid     ;
      real1d         dev_foil_twist   ;
      real1d         dev_foil_chord   ;
      int1d          dev_foil_id      ;
      MultiFieldDev  dev_foil_alpha   ;
      MultiFieldDev  dev_foil_clift   ;
      MultiFieldDev  dev_foil_cdrag   ;
      real1d         dev_rwt_mag      ;
      real1d         dev_rwt_ct       ;
      real1d         dev_rwt_cp       ;
      real1d         dev_rwt_pwr_mw   ;
      real1d         dev_rwt_rot      ;
      void init( core::Coupler const & coupler );
    };



    struct TraceEntry {
      std::string             name;
      bool                    dims_pnts;
      std::vector<real      > vals_1;
      std::vector<realHost2d> vals_pnts;
    };



    struct Traces {
      std::vector<TraceEntry> entries;
      void register_entry( std::string name , bool dims_pnts );
      std::vector<real> & get_1( std::string name );
      std::vector<realHost2d> & get_pnts( std::string name );
      void clear_all();
    };



    // This holds information about an individual turbine in the simulation (there can be multiple turbines)
    struct Turbine {
      bool                    active;             // Whether this turbine affects this MPI task
      real                    base_loc_x;         // x location of the tower base
      real                    base_loc_y;         // y location of the tower base
      real                    rot_angle;          // Rotation angle in radians
      real                    pitch;              // blade pitch angle in radians
      core::ParallelComm      par_comm;           // MPI communicator for this turbine
      Traces                  traces;             // 
    };



    // This holds information about all turbines in the simulation
    struct TurbineGroup {
      std::vector<Turbine> turbines;  // All turbines in the simulation
      // This routine adds a turbine to the group based on its base location and reference turbine data
      // The coupler is needed in order to determine whether the turbine is active on this MPI task
      void add_turbine( core::Coupler       & coupler     ,
                        real                  base_loc_x  ,
                        real                  base_loc_y  ,
                        RefTurbine const    & ref_turbine );
    };



    RefTurbine    ref_turbine;    // The reference turbine information
    TurbineGroup  turbine_group;  // Holds all turbines in the simulation
    int           trace_size;     // Number of time steps recorded in the turbine traces so far
                                  // This is reset to zero after writing output each time



    template <class T, class MEM>
    static KOKKOS_INLINE_FUNCTION T linear_interp( yakl::Array<T *,MEM> const & aref                ,
                                                   yakl::Array<T *,MEM> const & vref                ,
                                                   T                            a                   ,
                                                   bool                         const_extrap = true ) {
      int n = aref.size();
      if ( n==0 || aref.size() != vref.size() ) Kokkos::abort("Invalid input vectors");
      if ( a < aref(0) || aref.size() == 1 ) return const_extrap ? vref(0)   : 0.;
      if ( a > aref(n-1)                   ) return const_extrap ? vref(n-1) : 0.;
      for (int i=0; i < n-1; i++) {
        if (a >= aref(i) && a <= aref(i+1)) return vref(i)+(a-aref(i))/(aref(i+1)-aref(i))*(vref(i+1)-vref(i));
      }
      return 0.; // Doesn't get here, but gotta keep that compiler happy..
    }



    template <class T, class MEM>
    static KOKKOS_INLINE_FUNCTION T linear_interp( yakl::Array<T * ,MEM> const & aref                ,
                                                   yakl::Array<T **,MEM> const & vref                ,
                                                   T                             a                   ,
                                                   int                           iblade              ,
                                                   bool                          const_extrap = true ) {
      int n = aref.size();
      if ( n==0 || aref.size() != vref.extent(1) ) Kokkos::abort("Invalid input vectors");
      if ( a < aref(0) || aref.size() == 1 ) return const_extrap ? vref(iblade,0)   : 0.;
      if ( a > aref(n-1)                   ) return const_extrap ? vref(iblade,n-1) : 0.;
      for (int i=0; i < n-1; i++) {
        if (a >= aref(i) && a <= aref(i+1)) {
          return vref(iblade,i)+(a-aref(i))/(aref(i+1)-aref(i))*(vref(iblade,i+1)-vref(iblade,i));
        }
      }
      return 0.; // Doesn't get here, but gotta keep that compiler happy..
    }



    // Initialize the turbine actuator disc module, adding all the specified turbines from the coupler options
    void init( core::Coupler &coupler );


    // Apply the turbine actuator disc forces and yaw updates for all turbines, accumulating tendencies from
    //   thrust and torque forces. Keep traces of the power, yaw angle, and inflow wind speed normal to the turbine plane.
    // Injects a portion of the unused thrust energy back into the flow as SGS/unresolved TKE.
    void apply( core::Coupler & coupler , real dt );

  };

}



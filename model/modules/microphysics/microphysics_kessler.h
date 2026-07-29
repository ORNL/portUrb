
#pragma once

#include "main_header.h"
#include "coupler.h"

namespace modules {

  // This class implements the Kessler microphysics scheme
  class Microphysics_Kessler {
  public:
    int static constexpr num_tracers = 3;

    real R_d    ;  // Dry air ideal gas constant
    real cp_d   ;  // Specific heat of dry air at constant pressure
    real cv_d   ;  // Specific heat of dry air at constant volume
    real gamma_d;  // Ratio of specific heats for dry air
    real kappa_d;  // R_d / cp_d
    real R_v    ;  // Water vapor ideal gas constant
    real cp_v   ;  // Specific heat of water vapor at constant pressure
    real cv_v   ;  // Specific heat of water vapor at constant volume
    real p0     ;  // Reference pressure (10^5 Pa)
    real grav   ;  // Gravitational acceleration (9.81 m/s^2)

    int static constexpr ID_V = 0;  // Local index for water vapor
    int static constexpr ID_C = 1;  // Local index for cloud liquid
    int static constexpr ID_R = 2;  // Local index for precipitated liquid (rain)



    // Initialize physical constants
    Microphysics_Kessler();



    // Returns the number of tracers introduced by this microphysics scheme
    KOKKOS_INLINE_FUNCTION static int get_num_tracers() {
      return num_tracers;
    }



    // Initialize the microphysics module within the coupler
    // Adds three tracers: water_vapor, cloud_liquid, precip_liquid
    // Call this before the dynamics module's init so that tracers are registered properly
    // Also allocates non-tracer microphysics variable: precl
    void init(core::Coupler &coupler);



    // Advance the microphysics module by one time step
    // Applies latent heating, hydrometeor creation, and sedimentation fallout
    // coupler : Reference to the coupler object
    // dt : Time step size in seconds
    void time_step( core::Coupler &coupler , real dt ) const;



    ///////////////////////////////////////////////////////////////////////////////
    //
    //  Version:  2.0
    //
    //  Date:  January 22nd, 2015
    //
    //  Change log:
    //  v2 - Added sub-cycling of rain sedimentation so as not to violate
    //       CFL condition.
    //
    //  The KESSLER subroutine implements the Kessler (1969) microphysics
    //  parameterization as described by Soong and Ogura (1973) and Klemp
    //  and Wilhelmson (1978, KW). KESSLER is called at the end of each
    //  time step and makes the final adjustments to the potential
    //  temperature and moisture variables due to microphysical processes
    //  occurring during that time step. KESSLER is called once for each
    //  vertical column of grid cells. Increments are computed and added
    //  into the respective variables. The Kessler scheme contains three
    //  moisture categories: water vapor, cloud water (liquid water that
    //  moves with the flow), and rain water (liquid water that falls
    //  relative to the surrounding air). There  are no ice categories.
    //  
    //  Variables in the column are ordered from the surface to the top.
    //
    //  Parameters:
    //     theta (inout) - dry potential temperature (K)
    //     qv    (inout) - water vapor mixing ratio (gm/gm) (dry mixing ratio)
    //     qc    (inout) - cloud water mixing ratio (gm/gm) (dry mixing ratio)
    //     qr    (inout) - rain  water mixing ratio (gm/gm) (dry mixing ratio)
    //     rho   (in   ) - dry air density (not mean state as in KW) (kg/m^3)
    //     pk    (in   ) - Exner function  (not mean state as in KW) (p/p0)**(R/cp)
    //     dt    (in   ) - time step (s)
    //     z     (in   ) - heights of thermodynamic levels in the grid column (m)
    //     precl (  out) - Precipitation rate (m_water/s)
    //     Rd    (in   ) - Dry air ideal gas constant
    //     cp    (in   ) - Specific heat of dry air at constant pressure
    //     p0    (in   ) - Reference pressure (Pa)
    //
    // Output variables:
    //     Increments are added into t, qv, qc, qr, and precl which are
    //     returned to the routine from which KESSLER was called. To obtain
    //     the total precip qt, after calling the KESSLER routine, compute:
    //
    //       qt = sum over surface grid cells of (precl * cell area)  (kg)
    //       [here, the conversion to kg uses (10^3 kg/m^3)*(10^-3 m/mm) = 1]
    //
    //
    //  Written in Fortran by: Paul Ullrich
    //                         University of California, Davis
    //                         Email: paullrich@ucdavis.edu
    //
    //  Ported to C++ / YAKL by: Matt Norman
    //                           Oak Ridge National Laboratory
    //                           normanmr@ornl.gov
    //                           https://mrnorman.github.io
    //
    //  Based on a code by Joseph Klemp
    //  (National Center for Atmospheric Research)
    //
    //  Reference:
    //
    //    Klemp, J. B., W. C. Skamarock, W. C., and S.-H. Park, 2015:
    //    Idealized Global Nonhydrostatic Atmospheric Test Cases on a Reduced
    //    Radius Sphere. Journal of Advances in Modeling Earth Systems. 
    //    doi:10.1002/2015MS000435
    //
    ///////////////////////////////////////////////////////////////////////////////
    void kessler(real2d const &theta, real2d const &qv, real2d const &qc,
                 real2d const &qr, realConst2d rho,
                 real1d const &precl, realConst2d z, realConst2d pk, real dt,
                 real Rd, real cp, real p0, bool no_rain) const;


    std::string micro_name() const;

  };

}




#pragma once

#include "main_header.h"

namespace modules {

  struct Mp_morr_two_moment {
    typedef yakl::Array_F<double       * ,yakl::DeviceSpace> double1d_F;
    typedef yakl::Array_F<double       **,yakl::DeviceSpace> double2d_F;
    typedef yakl::Array_F<double const **,yakl::DeviceSpace> doubleConst2d_F;
    typedef yakl::Array_F<int          * ,yakl::DeviceSpace> int1d_F;
    typedef yakl::Array_F<bool         * ,yakl::DeviceSpace> bool1d_F;
    typedef yakl::Array_F<bool         **,yakl::DeviceSpace> bool2d_F;

    int   static constexpr inum    = 1;
    // switches for microphysics scheme
    // iact = 1, use std::power-law ccn spectra, nccn = cs^k
    // iact = 2, use lognormal aerosol size dist to derive ccn spectra
    // iact = 3, activation calculated in module_mixactivate
    int   static constexpr iact    = 2;
    // ibase = 1, neglect droplet activation at lateral cloud edges due to 
    //             unresolved entrainment and mixing, activate
    //             at cloud base or in region with little cloud water using 
    //             non-equlibrium supersaturation, 
    //             in cloud interior activate using equilibrium supersaturation
    // ibase = 2, assume droplet activation at lateral cloud edges due to 
    //             unresolved entrainment and mixing dominates,
    //             activate droplets everywhere in the cloud using non-equilibrium
    //             supersaturation, based on the 
    //             local sub-grid and/or grid-scale vertical velocity 
    //             at the grid point
    // note: only used for predicted droplet concentration (inum = 0) in non-wrf-chem version of code
    int   static constexpr ibase   = 2;
    // include sub-grid vertical velocity in droplet activation
    // isub = 0, include sub-grid w (recommended for lower resolution)
    // isub = 1, exclude sub-grid w, only use grid-scale w
    // note: only used for predicted droplet concentration (inum = 0) in non-wrf-chem version of code
    int   static constexpr isub    = 0;
    // switch for liquid-only run
    // iliq = 0, include ice
    // iliq = 1, liquid only, no ice
    int   static constexpr iliq    = 0;
    // switch for ice nucleation
    // inuc = 0, use formula from rasmussen et al. 2002 (mid-latitude)
    //      = 1, use mpace observations
    int   static constexpr inuc    = 0;
    // switch for graupel/no graupel
    // igraup = 0, include graupel
    // igraup = 1, no graupel
    int   static constexpr igraup  = 0;
    double static constexpr pi      = 3.1415926535897932384626434;
    double static constexpr xxx     = 0.9189385332046727417803297;
    double static constexpr r       = 287.;
    double static constexpr rv      = 461.6;
    double static constexpr g       = 9.81;
    double static constexpr cp      = 1004.5;
    double static constexpr ep_2    = 0.621750433;
    // for inum = 1, set constant droplet concentration (cm-3)
    double static constexpr ndcnst  = 250.;
    double static constexpr ai      = 700.;        // 'a' parameter in fallspeed-diam relationship
    double static constexpr ac      = 3.e7;        // 'a' parameter in fallspeed-diam relationship
    double static constexpr as      = 11.72;       // 'a' parameter in fallspeed-diam relationship
    double static constexpr ar      = 841.99667;   // 'a' parameter in fallspeed-diam relationship
    double static constexpr bi      = 1.;          // 'b' parameter in fallspeed-diam relationship
    double static constexpr bc      = 2.;          // 'b' parameter in fallspeed-diam relationship
    double static constexpr bs      = 0.41;        // 'b' parameter in fallspeed-diam relationship
    double static constexpr br      = 0.8;         // 'b' parameter in fallspeed-diam relationship
    double static constexpr rhosu   = 85000./(287.15*273.15); // standard air density at 850 mb
    double static constexpr rhow    = 997.;        // density of liquid water
    double static constexpr rhoi    = 500.;        // bulk density of cloud ice
    double static constexpr rhosn   = 100.;        // bulk density of snow
    double static constexpr aimm    = 0.66;        // parameter in bigg immersion freezing
    double static constexpr bimm    = 100.;        // parameter in bigg immersion freezing
    double static constexpr ecr     = 1.;          // collection efficiency between droplets/rain and snow/rain
    double static constexpr dcs     = 125.e-6;     // threshold size for cloud ice autoconversion
    double static constexpr mg0     = 1.6e-10;     // mass of embryo graupel
    double static constexpr f1s     = 0.86;        // ventilation parameter for snow
    double static constexpr f2s     = 0.28;        // ventilation parameter for snow
    double static constexpr f1r     = 0.78;        // ventilation parameter for rain
    double static constexpr f2r     = 0.308;       // ventilation parameter for rain
    double static constexpr qsmall  = 1.e-14;      // smallest allowed hydrometeor mixing ratio
    double static constexpr eii     = 0.1;         // collection efficiency, ice-ice collisions
    double static constexpr eci     = 0.7;         // collection efficiency, ice-droplet collisions
    #ifdef MICRO_MORR_2011_02_20
        double static constexpr cpw = 4218.;       // specific heat of liquid water
    #else
        double static constexpr cpw = 4187.;       // specific heat of liquid water
    #endif
    double static constexpr di      = 3.;          // size distribution parameters for cloud ice, snow, graupel
    double static constexpr ds      = 3.;          // size distribution parameters for cloud ice, snow, graupel
    double static constexpr dg      = 3.;          // size distribution parameters for cloud ice, snow, graupel
    double static constexpr rin     = 0.1e-6;      // radius of contact nuclei (m)
    double static constexpr lammaxi = 1./1.e-6;    // No documentation
    double static constexpr lammaxr = 1./20.e-6;   // No documentation
    double static constexpr lamminr = 1./2800.e-6; // No documentation
    double static constexpr lammaxs = 1./10.e-6;   // No documentation
    double static constexpr lammins = 1./2000.e-6; // No documentation
    double static constexpr lammaxg = 1./20.e-6;   // No documentation
    double static constexpr lamming = 1./2000.e-6; // No documentation
    double static constexpr mw      = 0.018;       // molecular weight water (kg/mol)
    double static constexpr osm     = 1.;          // osmotic coefficient
    double static constexpr vi      = 3.;          // number of ion dissociated in solution
    double static constexpr epsm    = 0.7;         // aerosol soluble fraction
    double static constexpr rhoa    = 1777.;       // aerosol bulk density (kg/m3)
    double static constexpr map     = 0.132;       // molecular weight aerosol (kg/mol)
    double static constexpr ma      = 0.0284;      // molecular weight of 'air' (kg/mol)
    #ifdef MICRO_MORR_2011_02_20
        double static constexpr rr  = 8.3187;      // universal gas constant
    #else
        double static constexpr rr  = 8.3145;      // universal gas constant
    #endif
    double static constexpr rm1     = 0.052e-6;    // activation parameter
    double static constexpr sig1    = 2.04;        // geometric mean radius, mode 1 (m)
    double static constexpr nanew1  = 72.2e6;      // total aerosol concentration, mode 1 (m^-3)
    double static constexpr rm2     = 1.3e-6;      // geometric mean radius, mode 2 (m)
    double static constexpr sig2    = 2.5;         // standard deviation of aerosol s.d., mode 2
    double static constexpr nanew2  = 1.8e6;       // total aerosol concentration, mode 2 (m^-3)
    // Docuementation for these is in init_two_moment
    double ag,bg,rhog,mi0,ci,cs,cg,mmult,lammini,bact,f11,f21,f12,f22,cons1,cons2,cons3,cons4,cons5,
          cons6,cons7,cons8,cons9,cons10,cons11,cons12,cons13,cons14,cons15,cons16,cons17,cons18,cons19,
          cons20,cons21,cons22,cons23,cons24,cons25,cons26,cons27,cons28,cons29,cons30,cons31,cons32,
          cons33,cons34,cons35,cons36,cons37,cons38,cons39,cons40,cons41;


    // hm added new option for hail
    // switch for hail/graupel
    // ihail = 0, dense precipitating ice is graupel
    // ihail = 1, dense precipitating gice is hail
    void init(int ihail);



    // qv - water vapor mixing ratio (kg/kg)
    // qc - cloud water mixing ratio (kg/kg)
    // qr - rain water mixing ratio (kg/kg)
    // qi - cloud ice mixing ratio (kg/kg)
    // qs - snow mixing ratio (kg/kg)
    // qg - graupel mixing ratio (kg/kg)
    // ni - cloud ice number concentration (1/kg)
    // ns - snow number concentration (1/kg)
    // nr - rain number concentration (1/kg)
    // ng - graupel number concentration (1/kg)
    // p - air pressure (pa)
    // w - vertical air velocity (m/s)
    // t - temperature (k)
    // pii - exner function - used to convert potential temp to temp
    // dz - difference in height over interface (m)
    // dt_in - model time step (sec)
    // itimestep - time step counter
    // rainnc - accumulated grid-scale precipitation (mm)
    // rainncv - one time step grid scale precipitation (mm/time step)
    // snownc - accumulated grid-scale snow plus cloud ice (mm)
    // snowncv - one time step grid scale snow plus cloud ice (mm/time step)
    // graupelnc - accumulated grid-scale graupel (mm)
    // graupelncv - one time step grid scale graupel (mm/time step)
    // sr - one time step mass ratio of snow to total precip
    // qrcuten, rain tendency from parameterized cumulus convection
    // qscuten, snow tendency from parameterized cumulus convection
    // qicuten, cloud ice tendency from parameterized cumulus convection
    // variables below currently not in use, not coupled to pbl or radiation codes
    // tke - turbulence kinetic energy (m^2 s-2), needed for droplet activation (see code below)
    // nctend - droplet concentration tendency from pbl (kg-1 s-1)
    // nctend - cloud ice concentration tendency from pbl (kg-1 s-1)
    // kzh - heat eddy diffusion coefficient from ysu scheme (m^2 s-1), needed for droplet activation (see code below)
    // effcs - cloud droplet effective radius output to radiation code (micron)
    // effis - cloud droplet effective radius output to radiation code (micron)
    // hm, added for wrf-chem coupling
    // qlsink - tendency of cloud water to rain, snow, graupel (kg/kg/s)
    // csed,ised,ssed,gsed,rsed - sedimentation fluxes (kg/m^2/s) for cloud water, ice, snow, graupel, rain
    // preci,precs,precg,precr - sedimentation fluxes (kg/m^2/s) for ice, snow, graupel, rain
    // rainprod - total tendency of conversion of cloud water/ice and graupel to rain (kg kg-1 s-1)
    // evapprod - tendency of evaporation of rain (kg kg-1 s-1)
    void run(double2d_F const &t, double2d_F const &qv, double2d_F const &qc, double2d_F const &qr,
             double2d_F const &qi, double2d_F const &qs, double2d_F const &qg, double2d_F const &ni,
             double2d_F const &ns, double2d_F const &nr, double2d_F const &ng, 
             doubleConst2d_F p, double dt_in, doubleConst2d_F dz, double1d_F const &rainnc,
             double1d_F const &rainncv, double1d_F const &sr, double1d_F const &snownc,
             double1d_F const &snowncv, double1d_F const &graupelnc, double1d_F const &graupelncv,
             doubleConst2d_F qrcuten, doubleConst2d_F qscuten, doubleConst2d_F qicuten, int ncol,
             int nz, double2d_F const &qlsink, double2d_F const &precr, double2d_F const &preci,
             double2d_F const &precs, double2d_F const &precg);



    // wrf:model_layer:physics
    // 
    //  this module contains the two-moment microphysics code described by
    //      morrison et al. (2009, mwr)
    //  changes for v3.2, relative to most recent (bug-fix) code for v3.1
    //  1) added accelerated melting of graupel/snow due to collision with rain, following lin et al. (1983)
    //  2) increased minimum lambda for rain, and added rain drop breakup following modified version
    //      of verlinde and cotton (1993)
    //  3) change minimum allowed mixing ratios in dry conditions (rh < 90%), this improves radar reflectiivity
    //      in low reflectivity regions
    //  4) bug fix to maximum allowed particle fallspeeds as a function of air density
    //  5) bug fix to calculation of liquid water saturation vapor pressure (change is very minor)
    //  6) include wrf constants per suggestion of jimy
    //  bug fix, 5/12/10
    //  7) bug fix for saturation vapor pressure in low pressure, to avoid division by zero
    //  8) include 'ep2' wrf constant for saturation mixing ratio calculation, instead of hardwire constant
    //  changes for v3.3
    //  1) modification for coupling with wrf-chem (predicted droplet number concentration) as an option
    //  2) modify fallspeed below the lowest level of precipitation, which prevents
    //       potential for spurious accumulation of precipitation during sub-stepping for sedimentation
    //  3) bug fix to latent heat release due to collisions of cloud ice with rain
    //  4) clean up of comments in the code
    //  additional minor bug fixes and small changes, 5/30/2011
    //  minor revisions by a. ackerman april 2011:
    //  1) replaced kinematic with dynamic viscosity 
    //  2) replaced scaling by air density for cloud droplet sedimentation
    //     with viscosity-dependent stokes expression
    //  3) use ikawa and saito (1991) air-density scaling for cloud ice
    //  4) corrected typo in 2nd digit of ventilation constant f2r
    //  additional fixes:
    //  5) temperature for accelerated melting due to colliions of snow and graupel
    //     with rain should use celsius, not kelvin (bug reported by k. van weverberg)
    //  6) npracs is not subtracted from snow number concentration, since
    //     decrease in snow number is already accounted for by nsmlts 
    //  7) fix for switch for running w/o graupel/hail (cloud ice and snow only)
    //  hm bug fix 3/16/12
    //  1) very minor change to limits on autoconversion source of rain number when cloud water is depleted
    //  wrfv3.5
    //  hm/a. ackerman bug fix 11/08/12
    //  1) for accelerated melting from collisions, should use rain mass collected by snow, not snow mass 
    //     collected by rain
    //  2) minor changes to some comments
    //  3) reduction of maximum-allowed ice concentration from 10 cm-3 to 0.3
    //     cm-3. this was done to address the problem of excessive and persistent
    //     anvil cirrus produced by the scheme.
    //  changes for wrfv3.5.1
    //  1) added output for snow+cloud ice and graupel time step and accumulated
    //     surface precipitation
    //  2) bug fix to option w/o graupel/hail (igraup = 1), include praci, pgsacw,
    //     and pgracs as sources for snow instead of graupel/hail, bug reported by
    //     hailong wang (pnnl)
    //  3) very minor fix to immersion freezing rate formulation (negligible impact)
    //  4) clarifications to code comments
    //  5) minor change to shedding of rain, remove limit so that the number of 
    //     collected drops can smaller than number of shed drops
    //  6) change of specific heat of liquid water from 4218 to 4187 j/kg/k
    //  changes for wrfv3.6.1
    //  1) minor bug fix to melting of snow and graupel, an extra factor of air density (rho) was removed
    //     from the calculation of psmlt and pgmlt
    //  2) redundant initialization of psmlt (non answer-changing)
    //  changes for wrfv3.8.1
    //  1) changes and cleanup of code comments
    //  2) correction to universal gas constant (very small change)
    //  changes for wrfv4.3
    //  1) fix to saturation vapor pressure polysvp to work at t < -80 c
    // this scheme is a bulk double-moment scheme that predicts mixing
    // ratios and number concentrations of five hydrometeor species:
    // cloud droplets, cloud (small) ice, rain, snow, and graupel/hail.
    // code structure: main subroutine is 'morr_two_moment'. also included in this file is
    // 'function polysvp'
    // note: this subroutine uses 1d array in vertical (column), even though variables are called '3d'......
    // qc3dten  : cloud water mixing ratio tendency (kg/kg/s)
    // qi3dten  : cloud ice mixing ratio tendency (kg/kg/s)
    // qni3dten : snow mixing ratio tendency (kg/kg/s)
    // qr3dten  : rain mixing ratio tendency (kg/kg/s)
    // ni3dten  : cloud ice number concentration (1/kg/s)
    // ns3dten  : snow number concentration (1/kg/s)
    // nr3dten  : rain number concentration (1/kg/s)
    // qc3d     : cloud water mixing ratio (kg/kg)
    // qi3d     : cloud ice mixing ratio (kg/kg)
    // qni3d    : snow mixing ratio (kg/kg)
    // qr3d     : rain mixing ratio (kg/kg)
    // ni3d     : cloud ice number concentration (1/kg)
    // ns3d     : snow number concentration (1/kg)
    // nr3d     : rain number concentration (1/kg)
    // t3dten   : temperature tendency (k/s)
    // qv3dten  : water vapor mixing ratio tendency (kg/kg/s)
    // t3d      : temperature (k)
    // qv3d     : water vapor mixing ratio (kg/kg)
    // pres     : atmospheric pressure (pa)
    // dzq      : difference in height across level (m)
    // w3d      : grid-scale vertical velocity (m/s)
    // wvar     : sub-grid vertical velocity (m/s)
    // qg3dten  : graupel mix ratio tendency (kg/kg/s)
    // ng3dten  : graupel numb conc tendency (1/kg/s)
    // qg3d     : graupel mix ratio (kg/kg)
    // ng3d     : graupel number conc (1/kg)
    // qgsten   : graupel sed tend (kg/kg/s)
    // qrsten   : rain sed tend (kg/kg/s)
    // qisten   : cloud ice sed tend (kg/kg/s)
    // qnisten  : snow sed tend (kg/kg/s)
    // qcsten   : cloud wat sed tend (kg/kg/s)      
    // precrt   : total precip per time step (mm)
    // snowrt   : snow per time step (mm)
    // snowprt  : total cloud ice plus snow per time step (mm)
    // dt       : model time step (sec)
    void run_two_mom(double2d_F const &qc3d, double2d_F const &qi3d, double2d_F const &qni3d, double2d_F const &qr3d,
                     double2d_F const &ni3d, double2d_F const &ns3d, double2d_F const &nr3d, double2d_F const &t3d,
                     double2d_F const &qv3d, doubleConst2d_F pres, doubleConst2d_F dzq, double1d_F const &precrt,
                     double1d_F const &snowrt, double1d_F const &snowprt, double1d_F const &grplprt, double dt, int ncol,
                     int nz, double2d_F const &qg3d, double2d_F const &ng3d, doubleConst2d_F qrcu1d,
                     doubleConst2d_F qscu1d, doubleConst2d_F qicu1d, int iinum, double2d_F const &c2prec,
                     double2d_F const &ised, double2d_F const &ssed, double2d_F const &gsed, double2d_F const &rsed);


    template <class T1, class T2> KOKKOS_INLINE_FUNCTION static double min(T1 a, T2 b) { return a < b ? a : b; }
    template <class T1, class T2> KOKKOS_INLINE_FUNCTION static double max(T1 a, T2 b) { return a > b ? a : b; }



    // compute saturation vapor pressure
    // polysvp returned in units of pa.
    // t is input in units of k.
    // type refers to saturation with respect to liquid (0) or ice (1)
    // replace goff-gratch with faster formulation from flatau et al. 1992, table 4 (right-hand column)
    #ifdef MICRO_MORR_2011_02_20

      KOKKOS_INLINE_FUNCTION static double polysvp( double t , int type) {
        // liquid
        double a0 = 6.11239921;
        double a1 = 0.443987641;
        double a2 = 0.142986287e-1;
        double a3 = 0.264847430e-3;
        double a4 = 0.302950461e-5;
        double a5 = 0.206739458e-7;
        double a6 = 0.640689451e-10;
        double a7 = -0.952447341e-13;
        double a8 = -0.976195544e-15;
        // ice
        double a0i = 6.11147274;
        double a1i = 0.503160820;
        double a2i = 0.188439774e-1;
        double a3i = 0.420895665e-3;
        double a4i = 0.615021634e-5;
        double a5i = 0.602588177e-7;
        double a6i = 0.385852041e-9;
        double a7i = 0.146898966e-11;
        double a8i = 0.252751365e-14;
        double ret;
        // ICE
        if (type == 1) {
          real dt = max(-80.,t-273.16);
          ret = a0i + dt*(a1i+dt*(a2i+dt*(a3i+dt*(a4i+dt*(a5i+dt*(a6i+dt*(a7i+a8i*dt)))))));
          ret = ret*100.;
        }
        // liquid
        if (type == 0) {
          real dt = max(-80.,t-273.16);
          ret = a0 + dt*(a1+dt*(a2+dt*(a3+dt*(a4+dt*(a5+dt*(a6+dt*(a7+a8*dt)))))));
          ret = ret*100.;
        }
        return ret;
      }

    #else

      KOKKOS_INLINE_FUNCTION static double polysvp( double t , int type) {
        // liquid
        double a0 = 6.11239921;
        double a1 = 0.443987641;
        double a2 = 0.142986287e-1;
        double a3 = 0.264847430e-3;
        double a4 = 0.302950461e-5;
        double a5 = 0.206739458e-7;
        double a6 = 0.640689451e-10;
        double a7 = -0.952447341e-13;
        double a8 = -0.976195544e-15;
        // ice
        double a0i = 6.11147274;
        double a1i = 0.503160820;
        double a2i = 0.188439774e-1;
        double a3i = 0.420895665e-3;
        double a4i = 0.615021634e-5;
        double a5i = 0.602588177e-7;
        double a6i = 0.385852041e-9;
        double a7i = 0.146898966e-11;
        double a8i = 0.252751365e-14;
        double ret;
        // ice
        if (type==1) {
          // hm 11/16/20, use goff-gratch for t < 195.8 k and flatau et al. equal or above 195.8 k
          if (t >= 195.8) {
            double dt=t-273.15;
            ret = a0i + dt*(a1i+dt*(a2i+dt*(a3i+dt*(a4i+dt*(a5i+dt*(a6i+dt*(a7i+a8i*dt))))))) ;
            ret = ret*100.;
          } else {
            ret = std::pow(10.,-9.09718*(273.16/t-1.)-3.56654*std::log10(273.16/t)+0.876793*
                  (1.-t/273.16)+std::log10(6.1071))*100.;
          }
        }
        // liquid
        if (type==0) {
          // hm 11/16/20, use goff-gratch for t < 202.0 k and flatau et al. equal or above 202.0 k
          if (t >= 202.0) {
            double dt = t-273.15;
            ret = a0 + dt*(a1+dt*(a2+dt*(a3+dt*(a4+dt*(a5+dt*(a6+dt*(a7+a8*dt)))))));
            ret = ret*100.;
          } else {
            // note: uncertain below -70 c, but produces physical values (non-negative) unlike flatau
            ret = std::pow(10.,-7.90298*(373.16/t-1.)+5.02808*std::log10(373.16/t)-1.3816e-7*
                  (std::pow(10.,11.344*(1.-t/373.16))-1.)+
                  8.1328e-3*(std::pow(10.,-3.49149*(373.16/t-1.))-1.)+std::log10(1013.246))*100.;
          }
        }
        return ret;
      }

    #endif

    struct RunContext;
    void run_two_mom_initialize(RunContext const &context);
    void run_two_mom_warm_processes(RunContext const &context);
    void run_two_mom_cold_processes(RunContext const &context);
    void run_two_mom_tendencies(RunContext const &context);
    void run_two_mom_sedimentation(RunContext const &context);
    void run_two_mom_finalize(RunContext const &context);

  };

}

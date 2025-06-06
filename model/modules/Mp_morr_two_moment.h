
#pragma once

namespace modules {

  struct Mp_morr_two_moment {
    typedef yakl::Array<double      ,1,yakl::memDevice,yakl::styleFortran> double1d_F;
    typedef yakl::Array<double      ,2,yakl::memDevice,yakl::styleFortran> double2d_F;
    typedef yakl::Array<double const,2,yakl::memDevice,yakl::styleFortran> doubleConst2d_F;
    typedef yakl::Array<int        ,1,yakl::memDevice,yakl::styleFortran> int1d_F;
    typedef yakl::Array<bool       ,1,yakl::memDevice,yakl::styleFortran> bool1d_F;
    typedef yakl::Array<bool       ,2,yakl::memDevice,yakl::styleFortran> bool2d_F;

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
    void init(int ihail) {
      ag      = ihail == 0 ? 19.3 : 114.5;               // 'a' parameter in fallspeed-diam relationship
      bg      = ihail == 0 ? 0.37 : 0.5  ;               // 'b' parameter in fallspeed-diam relationship
      rhog    = ihail == 0 ? 400. : 900. ;               //  bulk density of graupel
      mi0     = 4./3.*pi*rhoi*std::pow(10.e-6,3);             // initial size of nucleated crystal
      ci      = rhoi*pi/6.;                              // size distribution parameters for cloud ice, snow, graupel
      cs      = rhosn*pi/6.;                             // size distribution parameters for cloud ice, snow, graupel
      cg      = rhog*pi/6.;                              // size distribution parameters for cloud ice, snow, graupel
      mmult   = 4./3.*pi*rhoi*std::pow(5.e-6,3);              // mass of splintered ice particle
      lammini = 1./(2.*dcs+100.e-6);
      bact    = vi*osm*epsm*mw*rhoa/(map*rhow);          // activation parameter
      f11     = 0.5*std::exp(2.5*std::pow(std::log(sig1),2)); // correction factor for activation, mode 1
      f21     = 1.+0.25*std::log(sig1);                  // correction factor for activation, mode 1
      f12     = 0.5*std::exp(2.5*std::pow(std::log(sig2),2)); // correction factor for activation, mode 2
      f22     = 1.+0.25*std::log(sig2);                  // correction factor for activation, mode 2     
      // constants for efficiency
      cons1   = std::tgamma(1.+ds)*cs;
      cons2   = std::tgamma(1.+dg)*cg;
      cons3   = std::tgamma(4.+bs)/6.;
      cons4   = std::tgamma(4.+br)/6.;
      cons5   = std::tgamma(1.+bs);
      cons6   = std::tgamma(1.+br);
      cons7   = std::tgamma(4.+bg)/6.;
      cons8   = std::tgamma(1.+bg);
      cons9   = std::tgamma(5./2.+br/2.);
      cons10  = std::tgamma(5./2.+bs/2.);
      cons11  = std::tgamma(5./2.+bg/2.);
      cons12  = std::tgamma(1.+di)*ci;
      cons13  = std::tgamma(bs+3.)*pi/4.*eci;
      cons14  = std::tgamma(bg+3.)*pi/4.*eci;
      cons15  = -1108.*eii*std::pow(pi,(1.-bs)/3.)*std::pow(rhosn,(-2.-bs)/3.)/(4.*720.);
      cons16  = std::tgamma(bi+3.)*pi/4.*eci;
      cons17  = 4.*2.*3.*rhosu*pi*eci*eci*std::tgamma(2.*bs+2.)/(8.*(rhog-rhosn));
      cons18  = rhosn*rhosn;
      cons19  = rhow*rhow;
      cons20  = 20.*pi*pi*rhow*bimm;
      cons21  = 4./(dcs*rhoi);
      cons22  = pi*rhoi*std::pow(dcs,3)/6.;
      cons23  = pi/4.*eii*std::tgamma(bs+3.);
      cons24  = pi/4.*ecr*std::tgamma(br+3.);
      cons25  = pi*pi/24.*rhow*ecr*std::tgamma(br+6.);
      cons26  = pi/6.*rhow;
      cons27  = std::tgamma(1.+bi);
      cons28  = std::tgamma(4.+bi)/6.;
      cons29  = 4./3.*pi*rhow*std::pow(25.e-6,3);
      cons30  = 4./3.*pi*rhow;
      cons31  = pi*pi*ecr*rhosn;
      cons32  = pi/2.*ecr;
      cons33  = pi*pi*ecr*rhog;
      cons34  = 5./2.+br/2.;
      cons35  = 5./2.+bs/2.;
      cons36  = 5./2.+bg/2.;
      cons37  = 4.*pi*1.38e-23/(6.*pi*rin);
      cons38  = pi*pi/3.*rhow;
      cons39  = pi*pi/36.*rhow*bimm;
      cons40  = pi/6.*bimm;
      cons41  = pi*pi*ecr*rhow;
    }



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
             double2d_F const &precs, double2d_F const &precg) {
      using yakl::fortran::parallel_for;
      using yakl::fortran::SimpleBounds;
      double2d_F c2prec("c2prec",ncol,nz);
      double dt    = dt_in;
      // inum = 0, predict droplet concentration
      // inum = 1, assume constant droplet concentration   
      // !!!note: predicted droplet concentration not available in this version
      // contact hugh morrison (morrison@ucar.edu) for further information
      int  iinum = 1;
      run_two_mom(qc, qi, qs, qr ,ni, ns, nr, t, qv, p, dz, rainncv, sr, snowncv, graupelncv, dt, ncol, nz, qg,
                  ng, qrcuten, qscuten, qicuten, iinum, c2prec, preci, precs, precg, precr);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (qc(i,k) > 1.e-10) { qlsink(i,k) = c2prec(i,k)/qc(i,k); }
          else                  { qlsink(i,k) = 0.0;                 }
          if (k == 1) {
            rainnc    (i) = rainnc(i)+rainncv(i);
            snownc    (i) = snownc(i)+snowncv(i);
            graupelnc (i) = graupelnc(i)+graupelncv(i);
            sr        (i) = sr(i)/(rainncv(i)+1.e-12);
          }
      });
    }



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
                     double2d_F const &ised, double2d_F const &ssed, double2d_F const &gsed, double2d_F const &rsed) {
      using yakl::fortran::parallel_for;
      using yakl::fortran::SimpleBounds;
      YAKL_SCOPE( ag      , this->ag      );
      YAKL_SCOPE( bg      , this->bg      );     
      YAKL_SCOPE( rhog    , this->rhog    );       
      YAKL_SCOPE( mi0     , this->mi0     );      
      YAKL_SCOPE( ci      , this->ci      );     
      YAKL_SCOPE( cs      , this->cs      );     
      YAKL_SCOPE( cg      , this->cg      );     
      YAKL_SCOPE( mmult   , this->mmult   );
      YAKL_SCOPE( lammini , this->lammini );          
      YAKL_SCOPE( bact    , this->bact    );
      YAKL_SCOPE( f11     , this->f11     );      
      YAKL_SCOPE( f21     , this->f21     );      
      YAKL_SCOPE( f12     , this->f12     );      
      YAKL_SCOPE( f22     , this->f22     );      
      YAKL_SCOPE( cons1   , this->cons1   );        
      YAKL_SCOPE( cons2   , this->cons2   );        
      YAKL_SCOPE( cons3   , this->cons3   );        
      YAKL_SCOPE( cons4   , this->cons4   );        
      YAKL_SCOPE( cons5   , this->cons5   );        
      YAKL_SCOPE( cons6   , this->cons6   );        
      YAKL_SCOPE( cons7   , this->cons7   );        
      YAKL_SCOPE( cons8   , this->cons8   );        
      YAKL_SCOPE( cons9   , this->cons9   );        
      YAKL_SCOPE( cons10  , this->cons10  );         
      YAKL_SCOPE( cons11  , this->cons11  );         
      YAKL_SCOPE( cons12  , this->cons12  );         
      YAKL_SCOPE( cons13  , this->cons13  );         
      YAKL_SCOPE( cons14  , this->cons14  );         
      YAKL_SCOPE( cons15  , this->cons15  );         
      YAKL_SCOPE( cons16  , this->cons16  );         
      YAKL_SCOPE( cons17  , this->cons17  );         
      YAKL_SCOPE( cons18  , this->cons18  );         
      YAKL_SCOPE( cons19  , this->cons19  );         
      YAKL_SCOPE( cons20  , this->cons20  );         
      YAKL_SCOPE( cons21  , this->cons21  );         
      YAKL_SCOPE( cons22  , this->cons22  );         
      YAKL_SCOPE( cons23  , this->cons23  );         
      YAKL_SCOPE( cons24  , this->cons24  );         
      YAKL_SCOPE( cons25  , this->cons25  );         
      YAKL_SCOPE( cons26  , this->cons26  );         
      YAKL_SCOPE( cons27  , this->cons27  );         
      YAKL_SCOPE( cons28  , this->cons28  );         
      YAKL_SCOPE( cons29  , this->cons29  );         
      YAKL_SCOPE( cons30  , this->cons30  );         
      YAKL_SCOPE( cons31  , this->cons31  );         
      YAKL_SCOPE( cons32  , this->cons32  );         
      YAKL_SCOPE( cons33  , this->cons33  );         
      YAKL_SCOPE( cons34  , this->cons34  );         
      YAKL_SCOPE( cons35  , this->cons35  );         
      YAKL_SCOPE( cons36  , this->cons36  );         
      YAKL_SCOPE( cons37  , this->cons37  );         
      YAKL_SCOPE( cons38  , this->cons38  );         
      YAKL_SCOPE( cons39  , this->cons39  );         
      YAKL_SCOPE( cons40  , this->cons40  );         
      YAKL_SCOPE( cons41  , this->cons41  );         
      double2d_F ng3dten   ("ng3dten   ",ncol,nz);  // graupel numb conc tendency (1/kg/s)
      double2d_F qg3dten   ("qg3dten   ",ncol,nz);  // graupel mix ratio tendency (kg/kg/s)
      double2d_F effc      ("effc      ",ncol,nz);  // droplet effective radius (micron)
      double2d_F effi      ("effi      ",ncol,nz);  // cloud ice effective radius (micron)
      double2d_F effs      ("effs      ",ncol,nz);  // snow effective radius (micron)
      double2d_F effr      ("effr      ",ncol,nz);  // rain effective radius (micron)
      double2d_F effg      ("effg      ",ncol,nz);  // graupel effective radius (micron)
      double2d_F t3dten    ("t3dten    ",ncol,nz);  // temperature tendency (k/s)
      double2d_F qv3dten   ("qv3dten   ",ncol,nz);  // water vapor mixing ratio tendency (kg/kg/s)
      double2d_F qc3dten   ("qc3dten   ",ncol,nz);  // cloud water mixing ratio tendency (kg/kg/s)
      double2d_F qi3dten   ("qi3dten   ",ncol,nz);  // cloud ice mixing ratio tendency (kg/kg/s)
      double2d_F qni3dten  ("qni3dten  ",ncol,nz);  // snow mixing ratio tendency (kg/kg/s)
      double2d_F qr3dten   ("qr3dten   ",ncol,nz);  // rain mixing ratio tendency (kg/kg/s)
      double2d_F ni3dten   ("ni3dten   ",ncol,nz);  // cloud ice number concentration (1/kg/s)
      double2d_F ns3dten   ("ns3dten   ",ncol,nz);  // snow number concentration (1/kg/s)
      double2d_F nr3dten   ("nr3dten   ",ncol,nz);  // rain number concentration (1/kg/s)
      double2d_F csed      ("csed      ",ncol,nz);  // sedimentation fluxes (kg/m^2/s) for cloud water, ice, snow, graupel, rain
      double2d_F qgsten    ("qgsten    ",ncol,nz);  // graupel sed tend (kg/kg/s)
      double2d_F qrsten    ("qrsten    ",ncol,nz);  // rain sed tend (kg/kg/s)
      double2d_F qisten    ("qisten    ",ncol,nz);  // cloud ice sed tend (kg/kg/s)
      double2d_F qnisten   ("qnisten   ",ncol,nz);  // snow sed tend (kg/kg/s)
      double2d_F qcsten    ("qcsten    ",ncol,nz);  // cloud wat sed tend (kg/kg/s)      
      double2d_F nc3d      ("nc3d      ",ncol,nz);  //
      double2d_F nc3dten   ("nc3dten   ",ncol,nz);  //
      double2d_F lamc      ("lamc      ",ncol,nz);  // slope parameter for droplets (m-1)
      double2d_F lami      ("lami      ",ncol,nz);  // slope parameter for cloud ice (m-1)
      double2d_F lams      ("lams      ",ncol,nz);  // slope parameter for snow (m-1)
      double2d_F lamr      ("lamr      ",ncol,nz);  // slope parameter for rain (m-1)
      double2d_F lamg      ("lamg      ",ncol,nz);  // slope parameter for graupel (m-1)
      double2d_F cdist1    ("cdist1    ",ncol,nz);  // psd parameter for droplets
      double2d_F n0i       ("n0i       ",ncol,nz);  // intercept parameter for cloud ice (kg-1 m-1)
      double2d_F n0s       ("n0s       ",ncol,nz);  // intercept parameter for snow (kg-1 m-1)
      double2d_F n0rr      ("n0rr      ",ncol,nz);  // intercept parameter for rain (kg-1 m-1)
      double2d_F n0g       ("n0g       ",ncol,nz);  // intercept parameter for graupel (kg-1 m-1)
      double2d_F pgam      ("pgam      ",ncol,nz);  // spectral shape parameter for droplets
      double2d_F nsubc     ("nsubc     ",ncol,nz);  // loss of nc during evap
      double2d_F nsubi     ("nsubi     ",ncol,nz);  // loss of ni during sub.
      double2d_F nsubs     ("nsubs     ",ncol,nz);  // loss of ns during sub.
      double2d_F nsubr     ("nsubr     ",ncol,nz);  // loss of nr during evap
      double2d_F prd       ("prd       ",ncol,nz);  // dep cloud ice
      double2d_F pre       ("pre       ",ncol,nz);  // evap of rain
      double2d_F prds      ("prds      ",ncol,nz);  // dep snow
      double2d_F nnuccc    ("nnuccc    ",ncol,nz);  // change n due to contact freez droplets
      double2d_F mnuccc    ("mnuccc    ",ncol,nz);  // change q due to contact freez droplets
      double2d_F pra       ("pra       ",ncol,nz);  // accretion droplets by rain
      double2d_F prc       ("prc       ",ncol,nz);  // autoconversion droplets
      double2d_F pcc       ("pcc       ",ncol,nz);  // cond/evap droplets
      double2d_F nnuccd    ("nnuccd    ",ncol,nz);  // change n freezing aerosol (prim ice nucleation)
      double2d_F mnuccd    ("mnuccd    ",ncol,nz);  // change q freezing aerosol (prim ice nucleation)
      double2d_F mnuccr    ("mnuccr    ",ncol,nz);  // change q due to contact freez rain
      double2d_F nnuccr    ("nnuccr    ",ncol,nz);  // change n due to contact freez rain
      double2d_F npra      ("npra      ",ncol,nz);  // change in n due to droplet acc by rain
      double2d_F nragg     ("nragg     ",ncol,nz);  // self-collection/breakup of rain
      double2d_F nsagg     ("nsagg     ",ncol,nz);  // self-collection of snow
      double2d_F nprc      ("nprc      ",ncol,nz);  // change nc autoconversion droplets
      double2d_F nprc1     ("nprc1     ",ncol,nz);  // change nr autoconversion droplets
      double2d_F prai      ("prai      ",ncol,nz);  // change q accretion cloud ice by snow
      double2d_F prci      ("prci      ",ncol,nz);  // change q autoconversin cloud ice to snow
      double2d_F psacws    ("psacws    ",ncol,nz);  // change q droplet accretion by snow
      double2d_F npsacws   ("npsacws   ",ncol,nz);  // change n droplet accretion by snow
      double2d_F psacwi    ("psacwi    ",ncol,nz);  // change q droplet accretion by cloud ice
      double2d_F npsacwi   ("npsacwi   ",ncol,nz);  // change n droplet accretion by cloud ice
      double2d_F nprci     ("nprci     ",ncol,nz);  // change n autoconversion cloud ice by snow
      double2d_F nprai     ("nprai     ",ncol,nz);  // change n accretion cloud ice
      double2d_F nmults    ("nmults    ",ncol,nz);  // ice mult due to riming droplets by snow
      double2d_F nmultr    ("nmultr    ",ncol,nz);  // ice mult due to riming rain by snow
      double2d_F qmults    ("qmults    ",ncol,nz);  // change q due to ice mult droplets/snow
      double2d_F qmultr    ("qmultr    ",ncol,nz);  // change q due to ice rain/snow
      double2d_F pracs     ("pracs     ",ncol,nz);  // change q rain-snow collection
      double2d_F npracs    ("npracs    ",ncol,nz);  // change n rain-snow collection
      double2d_F pccn      ("pccn      ",ncol,nz);  // change q droplet activation
      double2d_F psmlt     ("psmlt     ",ncol,nz);  // change q melting snow to rain
      double2d_F evpms     ("evpms     ",ncol,nz);  // chnage q melting snow evaporating
      double2d_F nsmlts    ("nsmlts    ",ncol,nz);  // change n melting snow
      double2d_F nsmltr    ("nsmltr    ",ncol,nz);  // change n melting snow to rain
      double2d_F piacr     ("piacr     ",ncol,nz);  // change qr, ice-rain collection
      double2d_F niacr     ("niacr     ",ncol,nz);  // change n, ice-rain collection
      double2d_F praci     ("praci     ",ncol,nz);  // change qi, ice-rain collection
      double2d_F piacrs    ("piacrs    ",ncol,nz);  // change qr, ice rain collision, added to snow
      double2d_F niacrs    ("niacrs    ",ncol,nz);  // change n, ice rain collision, added to snow
      double2d_F pracis    ("pracis    ",ncol,nz);  // change qi, ice rain collision, added to snow
      double2d_F eprd      ("eprd      ",ncol,nz);  // sublimation cloud ice
      double2d_F eprds     ("eprds     ",ncol,nz);  // sublimation snow
      double2d_F pracg     ("pracg     ",ncol,nz);  // change in q collection rain by graupel
      double2d_F psacwg    ("psacwg    ",ncol,nz);  // change in q collection droplets by graupel
      double2d_F pgsacw    ("pgsacw    ",ncol,nz);  // conversion q to graupel due to collection droplets by snow
      double2d_F pgracs    ("pgracs    ",ncol,nz);  // conversion q to graupel due to collection rain by snow
      double2d_F prdg      ("prdg      ",ncol,nz);  // dep of graupel
      double2d_F eprdg     ("eprdg     ",ncol,nz);  // sub of graupel
      double2d_F evpmg     ("evpmg     ",ncol,nz);  // change q melting of graupel and evaporation
      double2d_F pgmlt     ("pgmlt     ",ncol,nz);  // change q melting of graupel
      double2d_F npracg    ("npracg    ",ncol,nz);  // change n collection rain by graupel
      double2d_F npsacwg   ("npsacwg   ",ncol,nz);  // change n collection droplets by graupel
      double2d_F nscng     ("nscng     ",ncol,nz);  // change n conversion to graupel due to collection droplets by snow
      double2d_F ngracs    ("ngracs    ",ncol,nz);  // change n conversion to graupel due to collection rain by snow
      double2d_F ngmltg    ("ngmltg    ",ncol,nz);  // change n melting graupel
      double2d_F ngmltr    ("ngmltr    ",ncol,nz);  // change n melting graupel to rain
      double2d_F nsubg     ("nsubg     ",ncol,nz);  // change n sub/dep of graupel
      double2d_F psacr     ("psacr     ",ncol,nz);  // conversion due to coll of snow by rain
      double2d_F nmultg    ("nmultg    ",ncol,nz);  // ice mult due to acc droplets by graupel
      double2d_F nmultrg   ("nmultrg   ",ncol,nz);  // ice mult due to acc rain by graupel
      double2d_F qmultg    ("qmultg    ",ncol,nz);  // change q due to ice mult droplets/graupel
      double2d_F qmultrg   ("qmultrg   ",ncol,nz);  // change q due to ice mult rain/graupel
      double2d_F kap       ("kap       ",ncol,nz);  // thermal conductivity of air
      double2d_F evs       ("evs       ",ncol,nz);  // saturation vapor pressure
      double2d_F eis       ("eis       ",ncol,nz);  // ice saturation vapor pressure
      double2d_F qvs       ("qvs       ",ncol,nz);  // saturation mixing ratio
      double2d_F qvi       ("qvi       ",ncol,nz);  // ice saturation mixing ratio
      double2d_F qvqvs     ("qvqvs     ",ncol,nz);  // sautration ratio
      double2d_F qvqvsi    ("qvqvsi    ",ncol,nz);  // ice saturaion ratio
      double2d_F dv        ("dv        ",ncol,nz);  // diffusivity of water vapor in air
      double2d_F xxls      ("xxls      ",ncol,nz);  // latent heat of sublimation
      double2d_F xxlv      ("xxlv      ",ncol,nz);  // latent heat of vaporization
      double2d_F cpm       ("cpm       ",ncol,nz);  // specific heat at const pressure for moist air
      double2d_F mu        ("mu        ",ncol,nz);  // viscocity of air
      double2d_F sc        ("sc        ",ncol,nz);  // schmidt number
      double2d_F xlf       ("xlf       ",ncol,nz);  // latent heat of freezing
      double2d_F rho       ("rho       ",ncol,nz);  // air density
      double2d_F ab        ("ab        ",ncol,nz);  // correction to condensation rate due to latent heating
      double2d_F abi       ("abi       ",ncol,nz);  // correction to deposition rate due to latent heating
      double2d_F dap       ("dap       ",ncol,nz);  // diffusivity of aerosol
      double2d_F dumi      ("dumi      ",ncol,nz);  //
      double2d_F dumr      ("dumr      ",ncol,nz);  //
      double2d_F dumfni    ("dumfni    ",ncol,nz);  //
      double2d_F dumg      ("dumg      ",ncol,nz);  //
      double2d_F dumfng    ("dumfng    ",ncol,nz);  //
      double2d_F fr        ("fr        ",ncol,nz);  //
      double2d_F fi        ("fi        ",ncol,nz);  //
      double2d_F fni       ("fni       ",ncol,nz);  //
      double2d_F fg        ("fg        ",ncol,nz);  //
      double2d_F fng       ("fng       ",ncol,nz);  //
      double2d_F faloutr   ("faloutr   ",ncol,nz);  //
      double2d_F falouti   ("falouti   ",ncol,nz);  //
      double2d_F faloutni  ("faloutni  ",ncol,nz);  //
      double2d_F dumqs     ("dumqs     ",ncol,nz);  //
      double2d_F dumfns    ("dumfns    ",ncol,nz);  //
      double2d_F fs        ("fs        ",ncol,nz);  //
      double2d_F fns       ("fns       ",ncol,nz);  //
      double2d_F falouts   ("falouts   ",ncol,nz);  //
      double2d_F faloutns  ("faloutns  ",ncol,nz);  //
      double2d_F faloutg   ("faloutg   ",ncol,nz);  //
      double2d_F faloutng  ("faloutng  ",ncol,nz);  //
      double2d_F dumc      ("dumc      ",ncol,nz);  //
      double2d_F dumfnc    ("dumfnc    ",ncol,nz);  //
      double2d_F fc        ("fc        ",ncol,nz);  //
      double2d_F faloutc   ("faloutc   ",ncol,nz);  //
      double2d_F faloutnc  ("faloutnc  ",ncol,nz);  //
      double2d_F fnc       ("fnc       ",ncol,nz);  //
      double2d_F dumfnr    ("dumfnr    ",ncol,nz);  //
      double2d_F faloutnr  ("faloutnr  ",ncol,nz);  //
      double2d_F fnr       ("fnr       ",ncol,nz);  //
      double2d_F ain       ("ain       ",ncol,nz);  //
      double2d_F arn       ("arn       ",ncol,nz);  //
      double2d_F asn       ("asn       ",ncol,nz);  //
      double2d_F acn       ("acn       ",ncol,nz);  //
      double2d_F agn       ("agn       ",ncol,nz);  //
      double2d_F tqimelt   ("tqimelt   ",ncol,nz);  // melting of cloud ice (tendency)
      bool2d_F  skip_micro("skip_micro",ncol,nz);  // Nothing to do: skip the microphysics for this cell
      bool2d_F  t_ge_273  ("t_ge_273"  ,ncol,nz);  // The cell's temperature is >= 273.15
      bool2d_F  no_cirg   ("no_cirg"   ,ncol,nz);  // There is no cloud, ice, rain, or graupel
      int1d_F   nstep     ("nstep     ",ncol);     // Number of fallout substeps due to large terminal velocity
      bool1d_F  hydro_pres("hydro_pres",ncol);     // Hydrometeors are present in this column: do fallout
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<1>(ncol) , KOKKOS_LAMBDA (int i) {
        hydro_pres(i) = false;
        nstep     (i) = 1;
        precrt    (i) = 0.;
        snowrt    (i) = 0.;
        snowprt   (i) = 0.;
        grplprt   (i) = 0.;
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          double dum;
          skip_micro(i,k) = false;
          nc3d    (i,k) = 0.;
          ng3dten (i,k) = 0.;
          qg3dten (i,k) = 0.;
          t3dten  (i,k) = 0.;
          qv3dten (i,k) = 0.;
          qc3dten (i,k) = 0.;
          qi3dten (i,k) = 0.;
          qni3dten(i,k) = 0.;
          qr3dten (i,k) = 0.;
          ni3dten (i,k) = 0.;
          ns3dten (i,k) = 0.;
          nr3dten (i,k) = 0.;
          nc3dten (i,k) = 0.;
          c2prec  (i,k) = 0.;
          csed    (i,k) = 0.;
          ised    (i,k) = 0.;
          ssed    (i,k) = 0.;
          gsed    (i,k) = 0.;
          rsed    (i,k) = 0.;
          xxlv    (i,k) = 3.1484e6-2370.*t3d(i,k);                 // latent heat of vaporation
          xxls    (i,k) = 3.15e6-2370.*t3d(i,k)+0.3337e6;          // latent heat of sublimation
          cpm     (i,k) = cp*(1.+0.887*qv3d(i,k));
            #ifdef MICRO_MORR_2011_02_20
              evs (i,k) = polysvp(t3d(i,k),0);
              eis (i,k) = polysvp(t3d(i,k),1);
            #else
              evs (i,k) = min(0.99*pres(i,k),polysvp(t3d(i,k),0)); // saturation vapor pressure and mixing ratio
              eis (i,k) = min(0.99*pres(i,k),polysvp(t3d(i,k),1));
            #endif
          if (eis(i,k) > evs(i,k)) eis(i,k) = evs(i,k); // make sure ice saturation doesn't exceed water sat. near freezing
          #ifdef MICRO_MORR_2011_02_20
            qvs   (i,k) = .622*evs(i,k)/(pres(i,k)-evs(i,k));
            qvi   (i,k) = .622*eis(i,k)/(pres(i,k)-eis(i,k));
          #else
            qvs   (i,k) = ep_2*evs(i,k)/(pres(i,k)-evs(i,k));
            qvi   (i,k) = ep_2*eis(i,k)/(pres(i,k)-eis(i,k));
          #endif
          qvqvs   (i,k) = qv3d(i,k)/qvs(i,k);
          qvqvsi  (i,k) = qv3d(i,k)/qvi(i,k);
          rho     (i,k) = pres(i,k)/(r*t3d(i,k));
          // add number concentration due to cumulus tendency
          // assume n0 associated with cumulus param rain is 10^7 m^-4
          // assume n0 associated with cumulus param snow is 2 x 10^7 m^-4
          // for detrained cloud ice, assume mean volume diam of 80 micron
          if (qrcu1d(i,k) >= 1.e-10) nr3d(i,k) = nr3d(i,k)+1.8e5*std::pow(qrcu1d(i,k)*dt/(pi*rhow*std::pow(rho(i,k),3)),0.25);
          if (qscu1d(i,k) >= 1.e-10) ns3d(i,k) = ns3d(i,k)+3.e5*std::pow(qscu1d(i,k)*dt/(cons1*std::pow(rho(i,k),3)),1./(ds+1.));
          if (qicu1d(i,k) >= 1.e-10) ni3d(i,k) = ni3d(i,k)+qicu1d(i,k)*dt/(ci*std::pow(80.e-6,di));
          // at subsaturation, remove small amounts of cloud/precip water
          if (qvqvs(i,k) < 0.9) {
            if (qr3d(i,k) < 1.e-8) {
               qv3d(i,k)=qv3d(i,k)+qr3d(i,k);
               t3d (i,k)=t3d(i,k)-qr3d(i,k)*xxlv(i,k)/cpm(i,k);
               qr3d(i,k)=0.;
            }
            if (qc3d(i,k) < 1.e-8) {
               qv3d(i,k)=qv3d(i,k)+qc3d(i,k);
               t3d (i,k)=t3d(i,k)-qc3d(i,k)*xxlv(i,k)/cpm(i,k);
               qc3d(i,k)=0.;
            }
          }
          if (qvqvsi(i,k) < 0.9) {
            if (qi3d(i,k) < 1.e-8) {
               qv3d(i,k)=qv3d(i,k)+qi3d(i,k);
               t3d (i,k)=t3d(i,k)-qi3d(i,k)*xxls(i,k)/cpm(i,k);
               qi3d(i,k)=0.;
            }
            if (qni3d(i,k) < 1.e-8) {
               qv3d (i,k)=qv3d(i,k)+qni3d(i,k);
               t3d  (i,k)=t3d(i,k)-qni3d(i,k)*xxls(i,k)/cpm(i,k);
               qni3d(i,k)=0.;
            }
            if (qg3d(i,k) < 1.e-8) {
               qv3d(i,k)=qv3d(i,k)+qg3d(i,k);
               t3d (i,k)=t3d(i,k)-qg3d(i,k)*xxls(i,k)/cpm(i,k);
               qg3d(i,k)=0.;
            }
          }
          xlf(i,k) = xxls(i,k)-xxlv(i,k);  // heat of fusion
          // if mixing ratio < qsmall set mixing ratio and number conc to zero
          if (qc3d(i,k) < qsmall) {
            qc3d(i,k) = 0.;
            nc3d(i,k) = 0.;
            effc(i,k) = 0.;
          }
          if (qr3d(i,k) < qsmall) {
            qr3d(i,k) = 0.;
            nr3d(i,k) = 0.;
            effr(i,k) = 0.;
          }
          if (qi3d(i,k) < qsmall) {
            qi3d(i,k) = 0.;
            ni3d(i,k) = 0.;
            effi(i,k) = 0.;
          }
          if (qni3d(i,k) < qsmall) {
            qni3d(i,k) = 0.;
            ns3d (i,k) = 0.;
            effs (i,k) = 0.;
          }
          if (qg3d(i,k) < qsmall) {
            qg3d(i,k) = 0.;
            ng3d(i,k) = 0.;
            effg(i,k) = 0.;
          }
          qrsten (i,k) = 0.;
          qisten (i,k) = 0.;
          qnisten(i,k) = 0.;
          qcsten (i,k) = 0.;
          qgsten (i,k) = 0.;
          // microphysics parameters varying in time/height
          mu     (i,k) = 1.496e-6*std::pow(t3d(i,k),1.5)/(t3d(i,k)+120.);
          // fall speed with density correction (heymsfield and benssemer 2006)
          dum          = std::pow(rhosu/rho(i,k),0.54);
          // ikawa and saito 1991 air-density correction 
          #ifdef MICRO_MORR_2011_02_20
            ain  (i,k) = dum*ai;
          #else
            ain  (i,k) = std::pow(rhosu/rho(i,k),0.35)*ai;
          #endif
          arn    (i,k) = dum*ar;
          asn    (i,k) = dum*as;
          // temperature-dependent stokes fall speed
          #ifdef MICRO_MORR_2011_02_20
            acn  (i,k) = dum*ac;
          #else
            acn  (i,k) = g*rhow/(18.*mu(i,k));
          #endif
          agn    (i,k) = dum*ag;
          lami   (i,k) = 0.;
          // if there is no cloud/precip water, and if subsaturated, then skip microphysics for this cell
          if (qc3d(i,k) < qsmall && qi3d(i,k) < qsmall && qni3d(i,k) < qsmall && qr3d(i,k) < qsmall && qg3d(i,k) < qsmall) {
            if (t3d(i,k) <  273.15 && qvqvsi(i,k) < 0.999) skip_micro(i,k) = true;
            if (t3d(i,k) >= 273.15 && qvqvs (i,k) < 0.999) skip_micro(i,k) = true;
          }
          if (! skip_micro(i,k)) {
            // thermal conductivity for air
            #ifdef MICRO_MORR_2011_02_20
              dum = 1.496e-6*std::pow(t3d(i,k),1.5)/(t3d(i,k)+120.);
              kap (i,k) = 1.414e3*dum;
            #else
              kap (i,k) = 1.414e3*mu(i,k);
            #endif
            // diffusivity of water vapor
            dv    (i,k) = 8.794e-5*std::pow(t3d(i,k),1.81)/pres(i,k);
            // schmit number
            #ifdef MICRO_MORR_2011_02_20
              mu(i,k) = dum/rho(i,k);
              sc(i,k) = mu(i,k)/dv(i,k);
            #else
              sc  (i,k) = mu(i,k)/(rho(i,k)*dv(i,k));
            #endif
            // psychometic corrections
            // rate of change sat. mix. ratio with temperature
            dum         = (rv*std::pow(t3d(i,k),2));
            double dqsdt  = xxlv(i,k)*qvs(i,k)/dum;
            double dqsidt = xxls(i,k)*qvi(i,k)/dum;
            abi   (i,k) = 1.+dqsidt*xxls(i,k)/cpm(i,k);
            ab    (i,k) = 1.+dqsdt*xxlv(i,k)/cpm(i,k);
            t_ge_273(i,k) = t3d(i,k) >= 273.15;  // This is a primary code split, so save to bool array for fissioning
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (t_ge_273(i,k)) {
              if (iinum==1) {
                // convert ndcnst from cm-3 to kg-1
                nc3d(i,k) = ndcnst*1.e6/rho(i,k);
              }
              // get size distribution parameters
              // melt very small snow and graupel mixing ratios, add to rain
              if (qni3d(i,k) < 1.e-6) {
                qr3d (i,k) = qr3d(i,k)+qni3d(i,k);
                nr3d (i,k) = nr3d(i,k)+ns3d (i,k);
                t3d  (i,k) = t3d (i,k)-qni3d(i,k)*xlf(i,k)/cpm(i,k);
                qni3d(i,k) = 0.;
                ns3d (i,k) = 0.;
              }
              if (qg3d(i,k) < 1.e-6) {
                qr3d(i,k) = qr3d(i,k)+qg3d(i,k);
                nr3d(i,k) = nr3d(i,k)+ng3d(i,k);
                t3d (i,k) = t3d (i,k)-qg3d(i,k)*xlf(i,k)/cpm(i,k);
                qg3d(i,k) = 0.;
                ng3d(i,k) = 0.;
              }
              // True if there's no cloud, ice, rain, or graupel
              no_cirg(i,k) = qc3d(i,k) < qsmall && qni3d(i,k) < 1.e-8 && qr3d(i,k) < qsmall && qg3d(i,k) < 1.e-8;
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (t_ge_273(i,k)) {
              if ( ! no_cirg(i,k)) {  // If there's cloud or ice or rain or graupel
                double dum;
                // make sure number concentrations aren't negative
                ns3d(i,k) = max(0.,ns3d(i,k));
                nc3d(i,k) = max(0.,nc3d(i,k));
                nr3d(i,k) = max(0.,nr3d(i,k));
                ng3d(i,k) = max(0.,ng3d(i,k));
                // rain
                if (qr3d(i,k) >= qsmall) {
                  lamr(i,k) = std::pow(pi*rhow*nr3d(i,k)/qr3d(i,k),1./3.);
                  n0rr(i,k) = nr3d(i,k)*lamr(i,k);
                  // check for slope
                  // adjust vars
                  if (lamr(i,k) < lamminr) {
                    lamr(i,k) = lamminr;
                    n0rr(i,k) = std::pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow);
                    nr3d(i,k) = n0rr(i,k)/lamr(i,k);
                  } else if (lamr(i,k) > lammaxr) {
                    lamr(i,k) = lammaxr;
                    n0rr(i,k) = std::pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow);
                    nr3d(i,k) = n0rr(i,k)/lamr(i,k);
                  }
                  // cloud droplets
                  // martin et al. (1994) formula for pgam
                  if (qc3d(i,k) >= qsmall) {
                    dum     =  pres(i,k)/(287.15*t3d(i,k));
                    pgam(i,k) = 0.0005714*(nc3d(i,k)/1.e6*dum)+0.2714;
                    pgam(i,k) = 1./(std::pow(pgam(i,k),2))-1.;
                    pgam(i,k) = max(pgam(i,k),2.);
                    pgam(i,k) = min(pgam(i,k),10.);
                    lamc(i,k) = std::pow(cons26*nc3d(i,k)*std::tgamma(pgam(i,k)+4.)/(qc3d(i,k)*std::tgamma(pgam(i,k)+1.)),1./3.);
                    // lammin, 60 micron diameter
                    // lammax, 1 micron
                    double lammin  = (pgam(i,k)+1.)/60.e-6;
                    double lammax  = (pgam(i,k)+1.)/1.e-6;
                    if (lamc(i,k) < lammin) {
                      lamc(i,k) = lammin;
                      nc3d(i,k) = std::exp(3.*std::log(lamc(i,k))+std::log(qc3d(i,k))+std::log(std::tgamma(pgam(i,k)+1.))-std::log(std::tgamma(pgam(i,k)+4.)))/cons26;
                    } else if (lamc(i,k) > lammax) {
                      lamc(i,k) = lammax;
                      nc3d(i,k) = std::exp(3.*std::log(lamc(i,k))+std::log(qc3d(i,k))+std::log(std::tgamma(pgam(i,k)+1.))-std::log(std::tgamma(pgam(i,k)+4.)))/cons26;
                    }
                  }
                }
                // snow
                if (qni3d(i,k) >= qsmall) {
                  lams(i,k) = std::pow(cons1*ns3d(i,k)/qni3d(i,k),1./ds);
                  n0s(i,k) = ns3d(i,k)*lams(i,k);
                  if (lams(i,k) < lammins) {
                    lams(i,k) = lammins;
                    n0s(i,k) = std::pow(lams(i,k),4)*qni3d(i,k)/cons1;
                    ns3d(i,k) = n0s(i,k)/lams(i,k);
                  } else if (lams(i,k) > lammaxs) {
                    lams(i,k) = lammaxs;
                    n0s(i,k) = std::pow(lams(i,k),4)*qni3d(i,k)/cons1;
                    ns3d(i,k) = n0s(i,k)/lams(i,k);
                  }
                }
                // graupel
                if (qg3d(i,k) >= qsmall) {
                  lamg(i,k) = std::pow(cons2*ng3d(i,k)/qg3d(i,k),1./dg);
                  n0g(i,k) = ng3d(i,k)*lamg(i,k);
                  if (lamg(i,k) < lamming) {
                    lamg(i,k) = lamming;
                    n0g(i,k) = std::pow(lamg(i,k),4)*qg3d(i,k)/cons2;
                    ng3d(i,k) = n0g(i,k)/lamg(i,k);
                  } else if (lamg(i,k) > lammaxg) {
                    lamg(i,k) = lammaxg;
                    n0g(i,k) = std::pow(lamg(i,k),4)*qg3d(i,k)/cons2;
                    ng3d(i,k) = n0g(i,k)/lamg(i,k);
                  }
                }
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          double dum;
          if (! skip_micro(i,k)) {
            if (t_ge_273(i,k)) {
              if ( ! no_cirg(i,k)) {  // If there's cloud or ice or rain or graupel
                prc(i,k) = 0.;
                nprc(i,k) = 0.;
                nprc1(i,k) = 0.;
                pra(i,k) = 0.;
                npra(i,k) = 0.;
                nragg(i,k) = 0.;
                nsmlts(i,k) = 0.;
                nsmltr(i,k) = 0.;
                evpms(i,k) = 0.;
                pcc(i,k) = 0.;
                pre(i,k) = 0.;
                nsubc(i,k) = 0.;
                nsubr(i,k) = 0.;
                pracg(i,k) = 0.;
                npracg(i,k) = 0.;
                psmlt(i,k) = 0.;
                pgmlt(i,k) = 0.;
                evpmg(i,k) = 0.;
                pracs(i,k) = 0.;
                npracs(i,k) = 0.;
                ngmltg(i,k) = 0.;
                ngmltr(i,k) = 0.;
                // calculation of microphysical process rates, t > 273.15 k
                // autoconversion of cloud liquid water to rain
                // formula from beheng (1994)
                // using numerical simulation of stochastic collection equation
                // and initial cloud droplet size distribution specified
                // as a std::tgamma distribution
                // use minimum value of 1.e-6 to prevent doubleing point error
                if (qc3d(i,k) >= 1.e-6) {
                  // from khairoutdinov and kogan 2000, mwr
                  prc(i,k)=1350.*std::pow(qc3d(i,k),2.47)*std::pow(nc3d(i,k)/1.e6*rho(i,k),-1.79);
                  // nprc is change in nc
                  nprc1(i,k) = prc(i,k)/cons29;
                  nprc(i,k) = prc(i,k)/(qc3d(i,k)/nc3d(i,k));
                  nprc(i,k) = min( nprc(i,k) , nc3d(i,k)/dt );
                  #ifdef MICRO_MORR_2011_02_20
                  #else
                    nprc1(i,k) = min( nprc1(i,k) , nprc(i,k) );
                  #endif
                }
                // formula from ikawa and saito (1991)
                if (qr3d(i,k) >= 1.e-8 && qni3d(i,k) >= 1.e-8) {
                  double ums = asn(i,k)*cons3/std::pow(lams(i,k),bs);
                  double umr = arn(i,k)*cons4/std::pow(lamr(i,k),br);
                  double uns = asn(i,k)*cons5/std::pow(lams(i,k),bs);
                  double unr = arn(i,k)*cons6/std::pow(lamr(i,k),br);
                  // set reaslistic limits on fallspeeds
                  double dum = std::pow(rhosu/rho(i,k),0.54);
                  ums = min( ums , 1.2*dum );
                  uns = min( uns , 1.2*dum );
                  umr = min( umr , 9.1*dum );
                  unr = min( unr , 9.1*dum );
                  // for above freezing conditions to get accelerated melting of snow,
                  // we need collection of rain by snow (following lin et al. 1983)
                  #ifdef MICRO_MORR_2011_02_20
                    pracs(i,k) = cons31*(std::pow(std::pow(1.2*umr-0.95*ums,2)+0.08*ums*umr,0.5)*rho(i,k)*n0rr(i,k)*n0s(i,k)/std::pow(lams(i,k),3)*
                                (5./(std::pow(lams(i,k),3)*lamr(i,k))+2./(std::pow(lams(i,k),2)*std::pow(lamr(i,k),2))+0.5/(lams(i,k)*std::pow(lamr(i,k),3))));
                  #else
                    pracs(i,k) = cons41*(std::pow(std::pow(1.2*umr-0.95*ums,2)+0.08*ums*umr,0.5)*rho(i,k)*n0rr(i,k)*n0s(i,k)/std::pow(lamr(i,k),3)*
                                (5./(std::pow(lamr(i,k),3)*lams(i,k))+2./(std::pow(lamr(i,k),2)*std::pow(lams(i,k),2))+0.5/(lamr(i,k)*std::pow(lams(i,k),3))));
                  #endif
                  #ifdef MICRO_MORR_2011_02_20
                    npracs(i,k) = cons32*rho(i,k)*std::pow(1.7*std::pow(unr-uns,2)+0.3*unr*uns,0.5)*n0rr(i,k)*n0s(i,k)*(1./(std::pow(lamr(i,k),3)*lams(i,k))+
                                  1./(std::pow(lamr(i,k),2)*std::pow(lams(i,k),2))+1./(lamr(i,k)*std::pow(lams(i,k),3)));
                  #else
                  #endif
                }
                // add collection of graupel by rain above freezing
                // assume all rain collection by graupel above freezing is shed
                // assume shed drops are 1 mm in size
                if (qr3d(i,k) >= 1.e-8 && qg3d(i,k) >= 1.e-8) {
                  double umg = agn(i,k)*cons7/std::pow(lamg(i,k),bg);
                  double umr = arn(i,k)*cons4/std::pow(lamr(i,k),br);
                  double ung = agn(i,k)*cons8/std::pow(lamg(i,k),bg);
                  double unr = arn(i,k)*cons6/std::pow(lamr(i,k),br);
                  // set reaslistic limits on fallspeeds
                  double dum = std::pow(rhosu/rho(i,k),0.54);
                  umg = min( umg , 20.*dum );
                  ung = min( ung , 20.*dum );
                  umr = min( umr , 9.1*dum );
                  unr = min( unr , 9.1*dum );
                  // pracg is mixing ratio of rain per sec collected by graupel/hail
                  pracg(i,k)  = cons41*(std::pow(std::pow(1.2*umr-0.95*umg,2)+0.08*umg*umr,0.5)*
                                rho(i,k)*n0rr(i,k)*n0g(i,k)/std::pow(lamr(i,k),3)*
                                (5./(std::pow(lamr(i,k),3)*lamg(i,k))+2./(std::pow(lamr(i,k),2)*
                                std::pow(lamg(i,k),2))+0.5/(lamr(i,k)*std::pow(lamg(i,k),3))));
                  dum       = pracg(i,k)/5.2e-7;
                  // assume 1 mm drops are shed, get number shed per sec
                  npracg(i,k) = cons32*rho(i,k)*std::pow(1.7*std::pow(unr-ung,2)+0.3*unr*ung,0.5)*n0rr(i,k)*n0g(i,k)*
                                (1./(std::pow(lamr(i,k),3)*lamg(i,k))+1./(std::pow(lamr(i,k),2)*
                                std::pow(lamg(i,k),2))+1./(lamr(i,k)*std::pow(lamg(i,k),3)));
                  #ifdef MICRO_MORR_2011_02_20
                    npracg(i,k) = max(npracg(i,k)-dum,0.);
                  #else
                    npracg(i,k) = npracg(i,k)-dum;
                  #endif
                }
                // accretion of cloud liquid water by rain
                // continuous collection equation with
                // gravitational collection kernel, droplet fall speed neglected
                if (qr3d(i,k) >= 1.e-8  &&  qc3d(i,k) >= 1.e-8) {
                  // khairoutdinov and kogan 2000, mwr
                  dum     = (qc3d(i,k)*qr3d(i,k));
                  pra(i,k) = 67.*std::pow(dum,1.15);
                  npra(i,k) = pra(i,k)/(qc3d(i,k)/nc3d(i,k));
                }
                // self-collection of rain drops
                // from beheng(1994)
                // from numerical simulation of the stochastic collection equation
                // as descrined above for autoconversion
                if (qr3d(i,k) >= 1.e-8) {
                  double dum1=300.e-6;
                  if (1./lamr(i,k) < dum1) {
                    dum=1.;
                  } else if (1./lamr(i,k) >= dum1) {
                    dum=2.-std::exp(2300.*(1./lamr(i,k)-dum1));
                  }
                  nragg(i,k) = -5.78*dum*nr3d(i,k)*qr3d(i,k)*rho(i,k);
                }
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          double dum;
          if (! skip_micro(i,k)) {
            if (t_ge_273(i,k)) {
              if ( ! no_cirg(i,k)) {  // If there's cloud or ice or rain or graupel
                double epsr;
                // calculate evap of rain (rutledge and hobbs 1983)
                if (qr3d(i,k) >= qsmall) {
                  epsr = 2.*pi*n0rr(i,k)*rho(i,k)*dv(i,k)*(f1r/(lamr(i,k)*lamr(i,k))+f2r*std::pow(arn(i,k)*
                         rho(i,k)/mu(i,k),0.5)*std::pow(sc(i,k),1./3.)*cons9/(std::pow(lamr(i,k),cons34)));
                } else {
                  epsr = 0.;
                }
                // no condensation onto rain, only evap allowed
                if (qv3d(i,k) < qvs(i,k)) {
                  pre(i,k) = epsr*(qv3d(i,k)-qvs(i,k))/ab(i,k);
                  pre(i,k) = min(pre(i,k),0.);
                } else {
                  pre(i,k) = 0.;
                }
                // melting of snow
                // snow may persits above freezing, formula from rutledge and hobbs, 1984
                // if water supersaturation, snow melts to form rain
                if (qni3d(i,k) >= 1.e-8) {
                  #ifdef MICRO_MORR_2011_02_20
                    dum = -cpw/xlf(i,k)*t3d(i,k)*pracs(i,k);
                  #else
                    dum = -cpw/xlf(i,k)*(t3d(i,k)-273.15)*pracs(i,k);
                  #endif
                  #ifdef MICRO_MORR_2011_02_20
                    psmlt(i,k) = 2.*pi*n0s(i,k)*kap(i,k)*(273.15-t3d(i,k))/xlf(i,k)*rho(i,k)*(f1s/(lams(i,k)*lams(i,k))+f2s*
                                 std::pow(asn(i,k)*rho(i,k)/mu(i,k),0.5)*std::pow(sc(i,k),1./3.)*cons10/(std::pow(lams(i,k),cons35)))+dum;
                  #else
                    psmlt(i,k) = 2.*pi*n0s(i,k)*kap(i,k)*(273.15-t3d(i,k))/xlf(i,k)*(f1s/(lams(i,k)*lams(i,k))+f2s*
                                 std::pow(asn(i,k)*rho(i,k)/mu(i,k),0.5)*std::pow(sc(i,k),1./3.)*cons10/(std::pow(lams(i,k),cons35)))+dum;
                  #endif
                  // in water subsaturation, snow melts and evaporates
                  if (qvqvs(i,k) < 1.) {
                    double epss = 2.*pi*n0s(i,k)*rho(i,k)*dv(i,k)*(f1s/(lams(i,k)*lams(i,k))+f2s*
                                 std::pow(asn(i,k)*rho(i,k)/mu(i,k),0.5)*std::pow(sc(i,k),1./3.)*cons10/(std::pow(lams(i,k),cons35)));
                    evpms(i,k) = (qv3d(i,k)-qvs(i,k))*epss/ab(i,k)    ;
                    evpms(i,k) = max(evpms(i,k),psmlt(i,k));
                    psmlt(i,k) = psmlt(i,k)-evpms(i,k);
                  }
                }
                // melting of graupel
                // graupel may persits above freezing, formula from rutledge and hobbs, 1984
                // if water supersaturation, graupel melts to form rain
                if (qg3d(i,k) >= 1.e-8) {
                  #ifdef MICRO_MORR_2011_02_20
                    dum = -cpw/xlf(i,k)*t3d(i,k)*pracg(i,k);
                  #else
                    dum = -cpw/xlf(i,k)*(t3d(i,k)-273.15)*pracg(i,k);
                  #endif
                  #ifdef MICRO_MORR_2011_02_20
                    pgmlt(i,k) = 2.*pi*n0g(i,k)*kap(i,k)*(273.15-t3d(i,k))/xlf(i,k)*rho(i,k)*(f1s/(lamg(i,k)*lamg(i,k))+f2s*
                                std::pow(agn(i,k)*rho(i,k)/mu(i,k),0.5)*std::pow(sc(i,k),1./3.)*cons11/(std::pow(lamg(i,k),cons36)))+dum;
                  #else
                    pgmlt(i,k) = 2.*pi*n0g(i,k)*kap(i,k)*(273.15-t3d(i,k))/xlf(i,k)*(f1s/(lamg(i,k)*lamg(i,k))+f2s*
                                std::pow(agn(i,k)*rho(i,k)/mu(i,k),0.5)*std::pow(sc(i,k),1./3.)*cons11/(std::pow(lamg(i,k),cons36)))+dum;
                  #endif
                  if (qvqvs(i,k) < 1.) {
                    double epsg = 2.*pi*n0g(i,k)*rho(i,k)*dv(i,k)*(f1s/(lamg(i,k)*lamg(i,k))+f2s*std::pow(agn(i,k)*
                                 rho(i,k)/mu(i,k),0.5)*std::pow(sc(i,k),1./3.)*cons11/(std::pow(lamg(i,k),cons36)));
                    evpmg(i,k) = (qv3d(i,k)-qvs(i,k))*epsg/ab(i,k);
                    evpmg(i,k) = max(evpmg(i,k),pgmlt(i,k));
                    pgmlt(i,k) = pgmlt(i,k)-evpmg(i,k);
                  }
                }
                pracg(i,k) = 0.;
                pracs(i,k) = 0.;
                dum = (prc(i,k)+pra(i,k))*dt;
                if (dum > qc3d(i,k) && qc3d(i,k) >= qsmall) {
                  double ratio = qc3d(i,k)/dum;
                  prc(i,k) = prc(i,k)*ratio;
                  pra(i,k) = pra(i,k)*ratio;
                }
                // conservation of snow
                dum = (-psmlt(i,k)-evpms(i,k)+pracs(i,k))*dt;
                if (dum > qni3d(i,k) && qni3d(i,k) >= qsmall) {
                  // no source terms for snow at t > freezing
                  double ratio    = qni3d(i,k)/dum;
                  psmlt(i,k) = psmlt(i,k)*ratio;
                  evpms(i,k) = evpms(i,k)*ratio;
                  pracs(i,k) = pracs(i,k)*ratio;
                }
                // conservation of graupel
                dum = (-pgmlt(i,k)-evpmg(i,k)+pracg(i,k))*dt;
                if (dum > qg3d(i,k) && qg3d(i,k) >= qsmall) {
                  // no source term for graupel above freezing
                  double ratio    = qg3d (i,k)/dum;
                  pgmlt(i,k) = pgmlt(i,k)*ratio;
                  evpmg(i,k) = evpmg(i,k)*ratio;
                  pracg(i,k) = pracg(i,k)*ratio;
                }
                // conservation of qr
                dum = (-pracs(i,k)-pracg(i,k)-pre(i,k)-pra(i,k)-prc(i,k)+psmlt(i,k)+pgmlt(i,k))*dt;
                if (dum > qr3d(i,k) && qr3d(i,k) >= qsmall) {
                  double ratio  = (qr3d(i,k)/dt+pracs(i,k)+pracg(i,k)+pra(i,k)+prc(i,k)-psmlt(i,k)-pgmlt(i,k))/(-pre(i,k));
                  pre(i,k) = pre(i,k)*ratio;
                }
                qv3dten (i,k) = qv3dten (i,k) + (-pre(i,k)-evpms(i,k)-evpmg(i,k));
                t3dten  (i,k) = t3dten  (i,k) + (pre(i,k)*xxlv(i,k)+(evpms(i,k)+evpmg(i,k))*xxls(i,k)+
                                                (psmlt(i,k)+pgmlt(i,k)-pracs(i,k)-pracg(i,k))*xlf(i,k))/cpm(i,k);
                qc3dten (i,k) = qc3dten (i,k) + (-pra(i,k)-prc(i,k));;
                qr3dten (i,k) = qr3dten (i,k) + (pre(i,k)+pra(i,k)+prc(i,k)-psmlt(i,k)-pgmlt(i,k)+pracs(i,k)+pracg(i,k));
                qni3dten(i,k) = qni3dten(i,k) + (psmlt(i,k)+evpms(i,k)-pracs(i,k));
                qg3dten (i,k) = qg3dten (i,k) + (pgmlt(i,k)+evpmg(i,k)-pracg(i,k));
                #ifdef MICRO_MORR_2011_02_20
                  ns3dten(i,k) = ns3dten(i,k)-npracs(i,k);
                #else
                #endif
                nc3dten (i,k) = nc3dten (i,k) + (-npra(i,k)-nprc(i,k));
                nr3dten (i,k) = nr3dten (i,k) + (nprc1(i,k)+nragg(i,k)-npracg(i,k));
                c2prec  (i,k) = pra(i,k)+prc(i,k);
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          double dum;
          if (! skip_micro(i,k)) {
            if (t_ge_273(i,k)) {
              if ( ! no_cirg(i,k)) {  // If there's cloud or ice or rain or graupel
                if (pre(i,k) < 0.) {
                  dum      = pre(i,k)*dt/qr3d(i,k);
                  dum      = max(-1.,dum);
                  nsubr(i,k) = dum*nr3d(i,k)/dt;
                }
                if (evpms(i,k)+psmlt(i,k) < 0.) {
                  dum       = (evpms(i,k)+psmlt(i,k))*dt/qni3d(i,k);
                  dum       = max(-1.,dum);
                  nsmlts(i,k) = dum*ns3d(i,k)/dt;
                }
                if (psmlt(i,k) < 0.) {
                  dum       = psmlt(i,k)*dt/qni3d(i,k);
                  dum       = max(-1.0,dum);
                  nsmltr(i,k) = dum*ns3d(i,k)/dt;
                }
                if (evpmg(i,k)+pgmlt(i,k) < 0.) {
                  dum       = (evpmg(i,k)+pgmlt(i,k))*dt/qg3d(i,k);
                  dum       = max(-1.,dum);
                  ngmltg(i,k) = dum*ng3d(i,k)/dt;
                }
                if (pgmlt(i,k) < 0.) {
                  dum       = pgmlt(i,k)*dt/qg3d(i,k);
                  dum       = max(-1.0,dum);
                  ngmltr(i,k) = dum*ng3d(i,k)/dt;
                }
                ns3dten(i,k) = ns3dten(i,k)+(nsmlts(i,k));
                ng3dten(i,k) = ng3dten(i,k)+(ngmltg(i,k));
                nr3dten(i,k) = nr3dten(i,k)+(nsubr(i,k)-nsmltr(i,k)-ngmltr(i,k));
              } // if ( ! no_cirg(i,k)) 
              // now calculate saturation adjustment to condense extra vapor above
              // water saturation
              double dumt   = t3d(i,k)+dt*t3dten(i,k);
              double dumqv  = qv3d(i,k)+dt*qv3dten(i,k);
              double dum=min(0.99*pres(i,k),polysvp(dumt,0));
              #ifdef MICRO_MORR_2011_02_20
                double dumqss = 0.622*polysvp(dumt,0)/ (pres(i,k)-polysvp(dumt,0));
              #else
                double dumqss = ep_2*dum/(pres(i,k)-dum);
              #endif
              double dumqc  = qc3d(i,k)+dt*qc3dten(i,k);
              dumqc  = max(dumqc,0.);
              // saturation adjustment for liquid
              double dums   = dumqv-dumqss;
              pcc    (i,k) = dums/(1.+std::pow(xxlv(i,k),2)*dumqss/(cpm(i,k)*rv*std::pow(dumt,2)))/dt;
              if (pcc(i,k)*dt+dumqc < 0.)  pcc(i,k) = -dumqc/dt;
              qv3dten(i,k) = qv3dten(i,k)-pcc(i,k);
              t3dten (i,k) = t3dten (i,k)+pcc(i,k)*xxlv(i,k)/cpm(i,k);
              qc3dten(i,k) = qc3dten(i,k)+pcc(i,k);
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              //hm add, allow for constant droplet number
              // inum = 0, predict droplet number
              // inum = 1, set constant droplet number
              if (iinum==1) {
                nc3d(i,k)=ndcnst*1.e6/rho(i,k);
              }
              ni3d(i,k) = max(0.,ni3d(i,k));
              ns3d(i,k) = max(0.,ns3d(i,k));
              nc3d(i,k) = max(0.,nc3d(i,k));
              nr3d(i,k) = max(0.,nr3d(i,k));
              ng3d(i,k) = max(0.,ng3d(i,k));
              // cloud ice
              if (qi3d(i,k) >= qsmall) {
                lami(i,k) = std::pow(cons12*ni3d(i,k)/qi3d(i,k),1./di);
                n0i(i,k) = ni3d(i,k)*lami(i,k);
                if (lami(i,k) < lammini) {
                  lami(i,k) = lammini;
                  n0i(i,k) = std::pow(lami(i,k),4)*qi3d(i,k)/cons12;
                  ni3d(i,k) = n0i(i,k)/lami(i,k);
                } else if (lami(i,k) > lammaxi) {
                  lami(i,k) = lammaxi;
                  n0i(i,k) = std::pow(lami(i,k),4)*qi3d(i,k)/cons12;
                  ni3d(i,k) = n0i(i,k)/lami(i,k);
                }
              }
              // rain
              if (qr3d(i,k) >= qsmall) {
                lamr(i,k) = std::pow(pi*rhow*nr3d(i,k)/qr3d(i,k),1./3.);
                n0rr(i,k) = nr3d(i,k)*lamr(i,k);
                if (lamr(i,k) < lamminr) {
                  lamr(i,k) = lamminr;
                  n0rr(i,k) = std::pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow);
                  nr3d(i,k) = n0rr(i,k)/lamr(i,k);
                } else if (lamr(i,k) > lammaxr) {
                  lamr(i,k) = lammaxr;
                  n0rr(i,k) = std::pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow);
                  nr3d(i,k) = n0rr(i,k)/lamr(i,k);
                }
              }
              // cloud droplets
              if (qc3d(i,k) >= qsmall) {
                double dum     = pres(i,k)/(287.15*t3d(i,k));
                pgam(i,k) = 0.0005714*(nc3d(i,k)/1.e6*dum)+0.2714;
                pgam(i,k) = 1./(std::pow(pgam(i,k),2))-1.;
                pgam(i,k) = max(pgam(i,k),2.);
                pgam(i,k) = min(pgam(i,k),10.);
                lamc(i,k) = std::pow(cons26*nc3d(i,k)*std::tgamma(pgam(i,k)+4.)/(qc3d(i,k)*std::tgamma(pgam(i,k)+1.)),1./3.);
                // lammin, 60 micron diameter
                // lammax, 1 micron
                double lammin  = (pgam(i,k)+1.)/60.e-6;
                double lammax  = (pgam(i,k)+1.)/1.e-6;
                if (lamc(i,k) < lammin) {
                  lamc(i,k) = lammin;
                  nc3d(i,k) = std::exp(3.*std::log(lamc(i,k))+std::log(qc3d(i,k))+std::log(std::tgamma(pgam(i,k)+1.))-std::log(std::tgamma(pgam(i,k)+4.)))/cons26;
                } else if (lamc(i,k) > lammax) {
                  lamc(i,k) = lammax;
                  nc3d(i,k) = std::exp(3.*std::log(lamc(i,k))+std::log(qc3d(i,k))+std::log(std::tgamma(pgam(i,k)+1.))-std::log(std::tgamma(pgam(i,k)+4.)))/cons26;
                }
                // to calculate droplet freezing
                cdist1(i,k) = nc3d(i,k)/std::tgamma(pgam(i,k)+1.);
              }
              // snow
              if (qni3d(i,k) >= qsmall) {
                lams(i,k) = std::pow(cons1*ns3d(i,k)/qni3d(i,k),1./ds);
                n0s(i,k) = ns3d(i,k)*lams(i,k);
                if (lams(i,k) < lammins) {
                  lams(i,k) = lammins;
                  n0s(i,k) = std::pow(lams(i,k),4)*qni3d(i,k)/cons1;
                  ns3d(i,k) = n0s(i,k)/lams(i,k);
                } else if (lams(i,k) > lammaxs) {
                  lams(i,k) = lammaxs;
                  n0s(i,k) = std::pow(lams(i,k),4)*qni3d(i,k)/cons1;
                  ns3d(i,k) = n0s(i,k)/lams(i,k);
                }
              }
              // graupel
              if (qg3d(i,k) >= qsmall) {
                lamg(i,k) = std::pow(cons2*ng3d(i,k)/qg3d(i,k),1./dg);
                n0g(i,k) = ng3d(i,k)*lamg(i,k);
                if (lamg(i,k) < lamming) {
                  lamg(i,k) = lamming;
                  n0g(i,k) = std::pow(lamg(i,k),4)*qg3d(i,k)/cons2;
                  ng3d(i,k) = n0g(i,k)/lamg(i,k);
                } else if (lamg(i,k) > lammaxg) {
                  lamg(i,k) = lammaxg;
                  n0g(i,k) = std::pow(lamg(i,k),4)*qg3d(i,k)/cons2;
                  ng3d(i,k) = n0g(i,k)/lamg(i,k);
                }
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              mnuccc(i,k) = 0.;
              nnuccc(i,k) = 0.;
              prc(i,k) = 0.;
              nprc(i,k) = 0.;
              nprc1(i,k) = 0.;
              nsagg(i,k) = 0.;
              psacws(i,k) = 0.;
              npsacws(i,k) = 0.;
              psacwi(i,k) = 0.;
              npsacwi(i,k) = 0.;
              pracs(i,k) = 0.;
              npracs(i,k) = 0.;
              nmults(i,k) = 0.;
              qmults(i,k) = 0.;
              nmultr(i,k) = 0.;
              qmultr(i,k) = 0.;
              nmultg(i,k) = 0.;
              qmultg(i,k) = 0.;
              nmultrg(i,k) = 0.;
              qmultrg(i,k) = 0.;
              mnuccr(i,k) = 0.;
              nnuccr(i,k) = 0.;
              pra(i,k) = 0.;
              npra(i,k) = 0.;
              nragg(i,k) = 0.;
              prci(i,k) = 0.;
              nprci(i,k) = 0.;
              prai(i,k) = 0.;
              nprai(i,k) = 0.;
              nnuccd(i,k) = 0.;
              mnuccd(i,k) = 0.;
              pcc(i,k) = 0.;
              pre(i,k) = 0.;
              prd(i,k) = 0.;
              prds(i,k) = 0.;
              eprd(i,k) = 0.;
              eprds(i,k) = 0.;
              nsubc(i,k) = 0.;
              nsubi(i,k) = 0.;
              nsubs(i,k) = 0.;
              nsubr(i,k) = 0.;
              piacr(i,k) = 0.;
              niacr(i,k) = 0.;
              praci(i,k) = 0.;
              piacrs(i,k) = 0.;
              niacrs(i,k) = 0.;
              pracis(i,k) = 0.;
              pracg(i,k) = 0.;
              psacr(i,k) = 0.;
              psacwg(i,k) = 0.;
              pgsacw(i,k) = 0.;
              pgracs(i,k) = 0.;
              prdg(i,k) = 0.;
              eprdg(i,k) = 0.;
              npracg(i,k) = 0.;
              npsacwg(i,k) = 0.;
              nscng(i,k) = 0.;
              ngracs(i,k) = 0.;
              nsubg(i,k) = 0.;
              // calculation of microphysical process rates
              // accretion/autoconversion/freezing/melting/coag.
              // freezing of cloud droplets
              // only allowed below -4 c
              if (qc3d(i,k) >= qsmall  &&  t3d(i,k) < 269.15) {
                // number of contact nuclei (m^-3) from meyers et al., 1992
                // factor of 1000 is to convert from l^-1 to m^-3
                // meyers curve
                double nacnt     = std::exp(-2.80+0.262*(273.15-t3d(i,k)))*1000.;
                // cooper curve
                //        nacnt =  5.*exp(0.304*(273.15-t3d(k)))
                // flecther
                //        nacnt = 0.01*exp(0.6*(273.15-t3d(k)))
                // contact freezing
                // mean free path
                double dum       = 7.37*t3d(i,k)/(288.*10.*pres(i,k))/100.;
                // effective diffusivity of contact nuclei
                // based on brownian diffusion
                dap(i,k) = cons37*t3d(i,k)*(1.+dum/rin)/mu(i,k);
                // immersion freezing (bigg 1953)
                mnuccc(i,k) = cons38*dap(i,k)*nacnt*std::exp(std::log(cdist1(i,k))+std::log(std::tgamma(pgam(i,k)+5.))-4.*std::log(lamc(i,k)));
                nnuccc(i,k) = 2.*pi*dap(i,k)*nacnt*cdist1(i,k)*std::tgamma(pgam(i,k)+2.)/lamc(i,k);
                #ifdef MICRO_MORR_2011_02_20
                  mnuccc(i,k) = mnuccc(i,k)+cons39*std::exp(std::log(cdist1(i,k))+std::log(std::tgamma(7.+pgam(i,k)))-6.*std::log(lamc(i,k)))*(std::exp(aimm*(273.15-t3d(i,k)))   );
                  nnuccc(i,k) = nnuccc(i,k)+cons40*std::exp(std::log(cdist1(i,k))+std::log(std::tgamma(pgam(i,k)+4.))-3.*std::log(lamc(i,k)))*(std::exp(aimm*(273.15-t3d(i,k)))   );
                #else
                  mnuccc(i,k) = mnuccc(i,k)+cons39*std::exp(std::log(cdist1(i,k))+std::log(std::tgamma(7.+pgam(i,k)))-6.*std::log(lamc(i,k)))*(std::exp(aimm*(273.15-t3d(i,k)))-1.);
                  nnuccc(i,k) = nnuccc(i,k)+cons40*std::exp(std::log(cdist1(i,k))+std::log(std::tgamma(pgam(i,k)+4.))-3.*std::log(lamc(i,k)))*(std::exp(aimm*(273.15-t3d(i,k)))-1.);
                #endif
                // put in a catch here to prevent divergence between number conc. and
                // mixing ratio, since strict conservation not checked for number conc
                nnuccc(i,k) = min(nnuccc(i,k),nc3d(i,k)/dt);
              }
              // autoconversion of cloud liquid water to rain
              // formula from beheng (1994)
              // using numerical simulation of stochastic collection equation
              // and initial cloud droplet size distribution specified
              // as a std::tgamma distribution
              // use minimum value of 1.e-6 to prevent doubleing point error
              if (qc3d(i,k) >= 1.e-6) {
                // from khairoutdinov and kogan 2000, mwr
                prc(i,k) = 1350.*std::pow(qc3d(i,k),2.47)*std::pow(nc3d(i,k)/1.e6*rho(i,k),-1.79);
                // nprc is change in nc
                nprc1(i,k) = prc(i,k)/cons29;
                nprc(i,k) = prc(i,k)/(qc3d(i,k)/nc3d(i,k));
                nprc(i,k) = min( nprc(i,k) , nc3d(i,k)/dt );
                #ifdef MICRO_MORR_2011_02_20
                #else
                  nprc1(i,k) = min( nprc1(i,k) , nprc(i,k)    );
                #endif
              }
              // self-collection of droplet not included in kk2000 scheme
              // snow aggregation from passarelli, 1978, used by reisner, 1998
              // this is hard-wired for bs = 0.4 for now
              if (qni3d(i,k) >= 1.e-8) {
                nsagg(i,k) = cons15*asn(i,k)*std::pow(rho(i,k),(2.+bs)/3.)*std::pow(qni3d(i,k),(2.+bs)/3.)*std::pow(ns3d(i,k)*rho(i,k),(4.-bs)/3.)/(rho(i,k));
              }
              // accretion of cloud droplets onto snow/graupel
              // here use continuous collection equation with
              // simple gravitational collection kernel ignoring
              // snow
              if (qni3d(i,k) >= 1.e-8  &&  qc3d(i,k) >= qsmall) {
                psacws(i,k) = cons13*asn(i,k)*qc3d(i,k)*rho(i,k)*n0s(i,k)/std::pow(lams(i,k),bs+3.);
                npsacws(i,k) = cons13*asn(i,k)*nc3d(i,k)*rho(i,k)*n0s(i,k)/std::pow(lams(i,k),bs+3.);
              }
              // collection of cloud water by graupel
              if (qg3d(i,k) >= 1.e-8  &&  qc3d(i,k) >= qsmall) {
                psacwg(i,k) = cons14*agn(i,k)*qc3d(i,k)*rho(i,k)*n0g(i,k)/std::pow(lamg(i,k),bg+3.);
                npsacwg(i,k) = cons14*agn(i,k)*nc3d(i,k)*rho(i,k)*n0g(i,k)/std::pow(lamg(i,k),bg+3.);
              }
              // cloud ice collecting droplets, assume that cloud ice mean diam > 100 micron
              // before riming can occur
              // assume that rime collected on cloud ice does not lead
              // to hallet-mossop splintering
              if (qi3d(i,k) >= 1.e-8  &&  qc3d(i,k) >= qsmall) {
                // put in size dependent collection efficiency based on stokes law
                // from thompson et al. 2004, mwr
                if (1./lami(i,k) >= 100.e-6) {
                  psacwi(i,k) = cons16*ain(i,k)*qc3d(i,k)*rho(i,k)*n0i(i,k)/std::pow(lami(i,k),bi+3.);
                  npsacwi(i,k) = cons16*ain(i,k)*nc3d(i,k)*rho(i,k)*n0i(i,k)/std::pow(lami(i,k),bi+3.);
                }
              }
              // accretion of rain water by snow
              // formula from ikawa and saito, 1991, used by reisner et al, 1998
              if (qr3d(i,k) >= 1.e-8 && qni3d(i,k) >= 1.e-8) {
                double ums = asn(i,k)*cons3/std::pow(lams(i,k),bs);
                double umr = arn(i,k)*cons4/std::pow(lamr(i,k),br);
                double uns = asn(i,k)*cons5/std::pow(lams(i,k),bs);
                double unr = arn(i,k)*cons6/std::pow(lamr(i,k),br);
                // set reaslistic limits on fallspeeds
                double dum = std::pow(rhosu/rho(i,k),0.54);
                ums = min( ums , 1.2*dum );
                uns = min( uns , 1.2*dum );
                umr = min( umr , 9.1*dum );
                unr = min( unr , 9.1*dum );
                // make sure pracs doesn't exceed total rain mixing ratio
                // as this may otherwise result in too much transfer of water during
                // rime-splintering
                pracs(i,k) = cons41*(std::pow(std::pow(1.2*umr-0.95*ums,2)+0.08*ums*umr,0.5)*rho(i,k)*n0rr(i,k)*n0s(i,k)/std::pow(lamr(i,k),3)*
                           (5./(std::pow(lamr(i,k),3)*lams(i,k))+2./(std::pow(lamr(i,k),2)*std::pow(lams(i,k),2))+0.5/(lamr(i,k)*std::pow(lams(i,k),3))));
                npracs(i,k) = cons32*rho(i,k)*std::pow(1.7*std::pow(unr-uns,2)+0.3*unr*uns,0.5)*n0rr(i,k)*n0s(i,k)*(1./(std::pow(lamr(i,k),3)*lams(i,k))+
                            1./(std::pow(lamr(i,k),2)*std::pow(lams(i,k),2))+1./(lamr(i,k)*std::pow(lams(i,k),3)));
                pracs(i,k) = min(pracs(i,k),qr3d(i,k)/dt);
                // collection of snow by rain - needed for graupel conversion calculations
                // only calculate if snow and rain mixing ratios exceed 0.1 g/kg
                if (qni3d(i,k) >= 0.1e-3 && qr3d(i,k) >= 0.1e-3) {
                  psacr(i,k) = cons31*(std::pow(std::pow(1.2*umr-0.95*ums,2)+0.08*ums*umr,0.5)*rho(i,k)*n0rr(i,k)*n0s(i,k)/std::pow(lams(i,k),3)* 
                             (5./(std::pow(lams(i,k),3)*lamr(i,k))+2./(std::pow(lams(i,k),2)*std::pow(lamr(i,k),2))+0.5/(lams(i,k)*std::pow(lamr(i,k),3))))            ;
                }
              }
              // collection of rainwater by graupel, from ikawa and saito 1990, 
              // used by reisner et al 1998
              if (qr3d(i,k) >= 1.e-8 && qg3d(i,k) >= 1.e-8) {
                double umg = agn(i,k)*cons7/std::pow(lamg(i,k),bg);
                double umr = arn(i,k)*cons4/std::pow(lamr(i,k),br);
                double ung = agn(i,k)*cons8/std::pow(lamg(i,k),bg);
                double unr = arn(i,k)*cons6/std::pow(lamr(i,k),br);
                // set reaslistic limits on fallspeeds
                double dum = std::pow(rhosu/rho(i,k),0.54);
                umg = min( umg , 20.*dum );
                ung = min( ung , 20.*dum );
                umr = min( umr , 9.1*dum );
                unr = min( unr , 9.1*dum );
                pracg(i,k) = cons41*(std::pow(std::pow(1.2*umr-0.95*umg,2)+0.08*umg*umr,0.5)*rho(i,k)*n0rr(i,k)*n0g(i,k)/std::pow(lamr(i,k),3)* 
                            (5./(std::pow(lamr(i,k),3)*lamg(i,k))+2./(std::pow(lamr(i,k),2)*std::pow(lamg(i,k),2))+0.5/(lamr(i,k)*std::pow(lamg(i,k),3))));
                npracg(i,k) = cons32*rho(i,k)*std::pow(1.7*std::pow(unr-ung,2)+0.3*unr*ung,0.5)*n0rr(i,k)*n0g(i,k)*(1./(std::pow(lamr(i,k),3)*lamg(i,k))+
                            1./(std::pow(lamr(i,k),2)*std::pow(lamg(i,k),2))+1./(lamr(i,k)*std::pow(lamg(i,k),3)));
                // make sure pracg doesn't exceed total rain mixing ratio
                // as this may otherwise result in too much transfer of water during
                // rime-splintering
                pracg(i,k) = min(pracg(i,k),qr3d(i,k)/dt);
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              // rime-splintering - snow
              // hallet-mossop (1974)
              // number of splinters formed is based on mass of rimed water
              // dum1 = mass of individual splinters
              // hm add threshold snow and droplet mixing ratio for rime-splintering
              // to limit rime-splintering in stratiform clouds
              // these thresholds correspond with graupel thresholds in rh 1984
              if (qni3d(i,k) >= 0.1e-3) {
                if (qc3d(i,k) >= 0.5e-3 || qr3d(i,k) >= 0.1e-3) {
                  if (psacws(i,k) > 0. || pracs(i,k) > 0.) {
                    if (t3d(i,k) < 270.16  &&  t3d(i,k) > 265.16) {
                      double fmult;
                      if (t3d(i,k) > 270.16) {
                        fmult = 0.;
                      } else if (t3d(i,k) <= 270.16 && t3d(i,k) > 268.16)  {
                        fmult = (270.16-t3d(i,k))/2.;
                      } else if (t3d(i,k) >= 265.16 && t3d(i,k) <= 268.16)   {
                        fmult = (t3d(i,k)-265.16)/3.;
                      } else if (t3d(i,k) < 265.16) {
                        fmult = 0.;
                      }
                      // 1000 is to convert from kg to g
                      // splintering from droplets accreted onto snow
                      if (psacws(i,k) > 0.) {
                        nmults(i,k) = 35.e4*psacws(i,k)*fmult*1000.;
                        qmults(i,k) = nmults(i,k)*mmult;
                        // constrain so that transfer of mass from snow to ice cannot be more mass
                        // than was rimed onto snow
                        qmults(i,k) = min(qmults(i,k),psacws(i,k));
                        psacws(i,k) = psacws(i,k)-qmults(i,k);
                      }
                      // riming and splintering from accreted raindrops
                      if (pracs(i,k) > 0.) {
                        nmultr(i,k) = 35.e4*pracs(i,k)*fmult*1000.;
                        qmultr(i,k) = nmultr(i,k)*mmult;
                        // constrain so that transfer of mass from snow to ice cannot be more mass
                        // than was rimed onto snow
                        qmultr(i,k) = min(qmultr(i,k),pracs(i,k));
                        pracs(i,k) = pracs(i,k)-qmultr(i,k);
                      }
                    }
                  }
                }
              }
              // rime-splintering - graupel 
              // hallet-mossop (1974)
              // number of splinters formed is based on mass of rimed water
              // dum1 = mass of individual splinters
              // hm add threshold snow mixing ratio for rime-splintering
              // to limit rime-splintering in stratiform clouds
              if (qg3d(i,k) >= 0.1e-3) {
                if (qc3d(i,k) >= 0.5e-3 || qr3d(i,k) >= 0.1e-3) {
                  if (psacwg(i,k) > 0. || pracg(i,k) > 0.) {
                    if (t3d(i,k) < 270.16  &&  t3d(i,k) > 265.16) {
                      double fmult;
                      if (t3d(i,k) > 270.16) {
                        fmult = 0.;
                      } else if (t3d(i,k) <= 270.16 && t3d(i,k) > 268.16)  {
                        fmult = (270.16-t3d(i,k))/2.;
                      } else if (t3d(i,k) >= 265.16 && t3d(i,k) <= 268.16)   {
                        fmult = (t3d(i,k)-265.16)/3.;
                      } else if (t3d(i,k) < 265.16) {
                        fmult = 0.;
                      }
                      // 1000 is to convert from kg to g
                      // splintering from droplets accreted onto graupel
                      if (psacwg(i,k) > 0.) {
                        nmultg(i,k) = 35.e4*psacwg(i,k)*fmult*1000.;
                        qmultg(i,k) = nmultg(i,k)*mmult;
                        // constrain so that transfer of mass from graupel to ice cannot be more mass
                        // than was rimed onto graupel
                        qmultg(i,k) = min(qmultg(i,k),psacwg(i,k));
                        psacwg(i,k) = psacwg(i,k)-qmultg(i,k);
                      }
                      // riming and splintering from accreted raindrops
                      if (pracg(i,k) > 0.) {
                        nmultrg(i,k) = 35.e4*pracg(i,k)*fmult*1000.;
                        qmultrg(i,k) = nmultrg(i,k)*mmult;
                        // constrain so that transfer of mass from graupel to ice cannot be more mass
                        // than was rimed onto graupel
                        qmultrg(i,k) = min(qmultrg(i,k),pracg(i,k));
                        pracg(i,k) = pracg(i,k)-qmultrg(i,k);
                      }
                    }
                  }
                }
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              // conversion of rimed cloud water onto snow to graupel/hail
              if (psacws(i,k) > 0.) {
                // only allow conversion if qni > 0.1 and qc > 0.5 g/kg following rutledge and hobbs (1984)
                if (qni3d(i,k) >= 0.1e-3 && qc3d(i,k) >= 0.5e-3) {
                  // portion of riming converted to graupel (reisner et al. 1998, originally is1991)
                  pgsacw(i,k) = min(psacws(i,k),cons17*dt*n0s(i,k)*qc3d(i,k)*qc3d(i,k)*asn(i,k)*asn(i,k)/(rho(i,k)*std::pow(lams(i,k),2.*bs+2.)));
                  // mix rat converted into graupel as embryo (reisner et al. 1998, orig m1990)
                  double dum       = max(rhosn/(rhog-rhosn)*pgsacw(i,k),0.) ;
                  // number concentraiton of embryo graupel from riming of snow
                  nscng(i,k) = dum/mg0*rho(i,k);
                  // limit max number converted to snow number
                  nscng(i,k) = min(nscng(i,k),ns3d(i,k)/dt);
                  // portion of riming left for snow
                  psacws(i,k) = psacws(i,k) - pgsacw(i,k);
                }
              }
              // conversion of rimed rainwater onto snow converted to graupel
              if (pracs(i,k) > 0.) {
                // only allow conversion if qni > 0.1 and qr > 0.1 g/kg following rutledge and hobbs (1984)
                if (qni3d(i,k) >= 0.1e-3 && qr3d(i,k) >= 0.1e-3) {
                  // portion of collected rainwater converted to graupel (reisner et al. 1998)
                  double dum = cons18*std::pow(4./lams(i,k),3)*std::pow(4./lams(i,k),3)/(cons18*std::pow(4./lams(i,k),3)*std::pow(4./lams(i,k),3)+
                              cons19*std::pow(4./lamr(i,k),3)*std::pow(4./lamr(i,k),3));
                  dum       = min( dum , 1. );
                  dum       = max( dum , 0. );
                  pgracs(i,k) = (1.-dum)*pracs(i,k);
                  ngracs(i,k) = (1.-dum)*npracs(i,k);
                  // limit max number converted to min of either rain or snow number concentration
                  ngracs(i,k) = min(ngracs(i,k),nr3d(i,k)/dt);
                  ngracs(i,k) = min(ngracs(i,k),ns3d(i,k)/dt);
                  // amount left for snow production
                  pracs(i,k) = pracs(i,k) - pgracs(i,k);
                  npracs(i,k) = npracs(i,k) - ngracs(i,k);
                  // conversion to graupel due to collection of snow by rain
                  psacr(i,k) = psacr(i,k)*(1.-dum);
                }
              }
              // freezing of rain drops
              // freezing allowed below -4 c
              if (t3d(i,k) < 269.15 && qr3d(i,k) >= qsmall) {
                // immersion freezing (bigg 1953)
                mnuccr(i,k) = cons20*nr3d(i,k)*(std::exp(aimm*(273.15-t3d(i,k)))-1.)/std::pow(lamr(i,k),3)/std::pow(lamr(i,k),3);
                #ifdef MICRO_MORR_2011_02_20
                  nnuccr(i,k) = pi*nr3d(i,k)*bimm*(std::exp(aimm*(273.15-t3d(i,k)))   )/std::pow(lamr(i,k),3);
                #else
                  nnuccr(i,k) = pi*nr3d(i,k)*bimm*(std::exp(aimm*(273.15-t3d(i,k)))-1.)/std::pow(lamr(i,k),3);
                #endif
                // prevent divergence between mixing ratio and number conc
                nnuccr(i,k) = min(nnuccr(i,k),nr3d(i,k)/dt);
              }
              // accretion of cloud liquid water by rain
              // continuous collection equation with
              // gravitational collection kernel, droplet fall speed neglected
              if (qr3d(i,k) >= 1.e-8  &&  qc3d(i,k) >= 1.e-8) {
                // khairoutdinov and kogan 2000, mwr
                double dum     = (qc3d(i,k)*qr3d(i,k));
                pra(i,k) = 67.*std::pow(dum,1.15);
                npra(i,k) = pra(i,k)/(qc3d(i,k)/nc3d(i,k));
              }
              // self-collection of rain drops
              // from beheng(1994)
              // from numerical simulation of the stochastic collection equation
              // as descrined above for autoconversion
              if (qr3d(i,k) >= 1.e-8) {
                double dum1=300.e-6;
                double dum;
                if (1./lamr(i,k) < dum1) {
                  dum=1.;
                } else if (1./lamr(i,k) >= dum1) {
                  dum=2.-std::exp(2300.*(1./lamr(i,k)-dum1));
                }
                nragg(i,k) = -5.78*dum*nr3d(i,k)*qr3d(i,k)*rho(i,k);
              }
              // autoconversion of cloud ice to snow
              // following harrington et al. (1995) with modification
              // here it is assumed that autoconversion can only occur when the
              // ice is growing, i.e. in conditions of ice supersaturation
              if (qi3d(i,k) >= 1.e-8  && qvqvsi(i,k) >= 1.) {
                nprci(i,k) = cons21*(qv3d(i,k)-qvi(i,k))*rho(i,k)*n0i(i,k)*std::exp(-lami(i,k)*dcs)*dv(i,k)/abi(i,k);
                prci(i,k) = cons22*nprci(i,k);
                nprci(i,k) = min(nprci(i,k),ni3d(i,k)/dt);
              }
              // accretion of cloud ice by snow
              // for this calculation, it is assumed that the vs >> vi
              // and ds >> di for continuous collection
              if (qni3d(i,k) >= 1.e-8  &&  qi3d(i,k) >= qsmall) {
                prai(i,k) = cons23*asn(i,k)*qi3d(i,k)*rho(i,k)*n0s(i,k)/std::pow(lams(i,k),bs+3.);
                nprai(i,k) = cons23*asn(i,k)*ni3d(i,k)*rho(i,k)*n0s(i,k)/std::pow(lams(i,k),bs+3.);
                nprai(i,k) = min( nprai(i,k) , ni3d(i,k)/dt );
              }
              // hm, add 12/13/06, collision of rain and ice to produce snow or graupel
              // follows reisner et al. 1998
              // assumed fallspeed and size of ice crystal << than for rain
              if (qr3d(i,k) >= 1.e-8  &&  qi3d(i,k) >= 1.e-8  &&  t3d(i,k) <= 273.15) {
                // allow graupel formation from rain-ice collisions only if rain mixing ratio > 0.1 g/kg,
                // otherwise add to snow
                if (qr3d(i,k) >= 0.1e-3) {
                  niacr(i,k)=cons24*ni3d(i,k)*n0rr(i,k)*arn(i,k)/std::pow(lamr(i,k),br+3.)*rho(i,k);
                  piacr(i,k)=cons25*ni3d(i,k)*n0rr(i,k)*arn(i,k)/std::pow(lamr(i,k),br+3.)/std::pow(lamr(i,k),3)*rho(i,k);
                  praci(i,k)=cons24*qi3d(i,k)*n0rr(i,k)*arn(i,k)/std::pow(lamr(i,k),br+3.)*rho(i,k);
                  niacr(i,k)=min(niacr(i,k),nr3d(i,k)/dt);
                  niacr(i,k)=min(niacr(i,k),ni3d(i,k)/dt);
                } else {
                  niacrs(i,k)=cons24*ni3d(i,k)*n0rr(i,k)*arn(i,k)/std::pow(lamr(i,k),br+3.)*rho(i,k);
                  piacrs(i,k)=cons25*ni3d(i,k)*n0rr(i,k)*arn(i,k)/std::pow(lamr(i,k),br+3.)/std::pow(lamr(i,k),3)*rho(i,k);
                  pracis(i,k)=cons24*qi3d(i,k)*n0rr(i,k)*arn(i,k)/std::pow(lamr(i,k),br+3.)*rho(i,k);
                  niacrs(i,k)=min(niacrs(i,k),nr3d(i,k)/dt);
                  niacrs(i,k)=min(niacrs(i,k),ni3d(i,k)/dt);
                }
              }
              // nucleation of cloud ice from homogeneous and heterogeneous freezing on aerosol
              if (inuc==0) {
                // add threshold according to greg thomspon
                if ((qvqvs(i,k) >= 0.999  &&  t3d(i,k) <= 265.15)  ||  qvqvsi(i,k) >= 1.08) {
                  // hm, modify dec. 5, 2006, replace with cooper curve
                  double kc2 = 0.005*std::exp(0.304*(273.15-t3d(i,k)))*1000.;
                  kc2 = min( kc2 ,500.e3 );
                  kc2 = max( kc2/rho(i,k) , 0. );
                  if (kc2 > ni3d(i,k)+ns3d(i,k)+ng3d(i,k)) {
                    nnuccd(i,k) = (kc2-ni3d(i,k)-ns3d(i,k)-ng3d(i,k))/dt;
                    mnuccd(i,k) = nnuccd(i,k)*mi0;
                  }
                }
              } else if (inuc==1) {
                if (t3d(i,k) < 273.15 && qvqvsi(i,k) > 1.) {
                  double kc2 = 0.16*1000./rho(i,k);
                  if (kc2 > ni3d(i,k)+ns3d(i,k)+ng3d(i,k)) {
                    nnuccd(i,k) = (kc2-ni3d(i,k)-ns3d(i,k)-ng3d(i,k))/dt;
                    mnuccd(i,k) = nnuccd(i,k)*mi0;
                  }
                }
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              double epsi;
              // calculate evap/sub/dep terms for qi,qni,qr
              // no ventilation for cloud ice
              if (qi3d(i,k) >= qsmall) {
                 epsi = 2.*pi*n0i(i,k)*rho(i,k)*dv(i,k)/(lami(i,k)*lami(i,k));
              } else {
                 epsi = 0.;
              }
              double epss;
              if (qni3d(i,k) >= qsmall) {
                epss = 2.*pi*n0s(i,k)*rho(i,k)*dv(i,k)*(f1s/(lams(i,k)*lams(i,k))+f2s*std::pow(asn(i,k)*rho(i,k)/mu(i,k),0.5)*std::pow(sc(i,k),1./3.)*cons10/(std::pow(lams(i,k),cons35)));
              } else {
                epss = 0.;
              }
              double epsg;
              if (qg3d(i,k) >= qsmall) {
                epsg = 2.*pi*n0g(i,k)*rho(i,k)*dv(i,k)*(f1s/(lamg(i,k)*lamg(i,k))+f2s*std::pow(agn(i,k)*rho(i,k)/mu(i,k),0.5)*std::pow(sc(i,k),1./3.)*cons11/(std::pow(lamg(i,k),cons36)));
              } else {
                epsg = 0.;
              }
              double epsr;
              if (qr3d(i,k) >= qsmall) {
                epsr = 2.*pi*n0rr(i,k)*rho(i,k)*dv(i,k)*(f1r/(lamr(i,k)*lamr(i,k))+f2r*std::pow(arn(i,k)*rho(i,k)/mu(i,k),0.5)*std::pow(sc(i,k),1./3.)*cons9/(std::pow(lamr(i,k),cons34)));
              } else {
                epsr = 0.;
              }
              double dum;
              // only include region of ice size dist < dcs
              // dum is fraction of d*n(d) < dcs
              // logic below follows that of harrington et al. 1995 (jas)
              if (qi3d(i,k) >= qsmall) {              
                dum    = (1.-std::exp(-lami(i,k)*dcs)*(1.+lami(i,k)*dcs));
                prd(i,k) = epsi*(qv3d(i,k)-qvi(i,k))/abi(i,k)*dum;
              } else {
                dum=0.;
              }
              // add deposition in tail of ice size dist to snow if snow is present
              if (qni3d(i,k) >= qsmall) {
                prds(i,k) = epss*(qv3d(i,k)-qvi(i,k))/abi(i,k)+epsi*(qv3d(i,k)-qvi(i,k))/abi(i,k)*(1.-dum);
              } else { // otherwise add to cloud ice
                prd(i,k) = prd(i,k)+epsi*(qv3d(i,k)-qvi(i,k))/abi(i,k)*(1.-dum);
              }
              // vapor dpeosition on graupel
              prdg(i,k) = epsg*(qv3d(i,k)-qvi(i,k))/abi(i,k);
              // no condensation onto rain, only evap
              if (qv3d(i,k) < qvs(i,k)) {
                pre(i,k) = epsr*(qv3d(i,k)-qvs(i,k))/ab(i,k);
                pre(i,k) = min( pre(i,k) , 0. );
              } else {
                pre(i,k) = 0.;
              }
              // make sure not pushed into ice supersat/subsat
              // formula from reisner 2 scheme
              dum = (qv3d(i,k)-qvi(i,k))/dt;
              double fudgef = 0.9999;
              double sum_dep = prd(i,k)+prds(i,k)+mnuccd(i,k)+prdg(i,k);
              if( (dum > 0.  &&  sum_dep > dum*fudgef)  ||  (dum < 0.  &&  sum_dep < dum*fudgef) ) {
                mnuccd(i,k) = fudgef*mnuccd(i,k)*dum/sum_dep;
                prd(i,k) = fudgef*prd(i,k)*dum/sum_dep;
                prds(i,k) = fudgef*prds(i,k)*dum/sum_dep;
                prdg(i,k) = fudgef*prdg(i,k)*dum/sum_dep;
              }
              // if cloud ice/snow/graupel vap deposition is neg, then assign to sublimation processes
              if (prd(i,k) < 0.) {
                eprd(i,k)=prd(i,k);
                prd(i,k)=0.;
              }
              if (prds(i,k) < 0.) {
                eprds(i,k)=prds(i,k);
                prds(i,k)=0.;
              }
              if (prdg(i,k) < 0.) {
                eprdg(i,k)=prdg(i,k);
                prdg(i,k)=0.;
              }
              // conservation of water
              // this is adopted loosely from mm5 resiner code. however, here we
              // only adjust processes that are negative, rather than all processes.
              // if mixing ratios less than qsmall, then no depletion of water
              // through microphysical processes, skip conservation
              // note: conservation check not applied to number concentration species. additional catch
              // below will prevent negative number concentration
              // for each microphysical process which provides a source for number, there is a check
              // to make sure that can't exceed total number of depleted species with the time
              // step
              // ****sensitivity - no ice
              if (iliq==1) {
                mnuccc(i,k)=0.;
                nnuccc(i,k)=0.;
                mnuccr(i,k)=0.;
                nnuccr(i,k)=0.;
                mnuccd(i,k)=0.;
                nnuccd(i,k)=0.;
              }
              // ****sensitivity - no graupel
              if (igraup==1) {
                pracg(i,k) = 0.;
                psacr(i,k) = 0.;
                psacwg(i,k) = 0.;
                #ifdef MICRO_MORR_2011_02_20
                  pgsacw(i,k) = 0.;
                  pgracs(i,k) = 0.;
                #else
                #endif
                prdg(i,k) = 0.;
                eprdg(i,k) = 0.;
                evpmg(i,k) = 0.;
                pgmlt(i,k) = 0.;
                npracg(i,k) = 0.;
                npsacwg(i,k) = 0.;
                nscng(i,k) = 0.;
                ngracs(i,k) = 0.;
                nsubg(i,k) = 0.;
                ngmltg(i,k) = 0.;
                ngmltr(i,k) = 0.;
                #ifdef MICRO_MORR_2011_02_20
                #else
                  piacrs(i,k) = piacrs(i,k)+piacr(i,k);
                  piacr(i,k) = 0.;
                  pracis(i,k) = pracis(i,k)+praci(i,k);
                  praci(i,k) = 0.;
                  psacws(i,k) = psacws(i,k)+pgsacw(i,k);
                  pgsacw(i,k) = 0.;
                  pracs(i,k) = pracs(i,k)+pgracs(i,k);
                  pgracs(i,k) = 0.;
                #endif
              }
              // conservation of qc
              dum = (prc(i,k)+pra(i,k)+mnuccc(i,k)+psacws(i,k)+psacwi(i,k)+qmults(i,k)+psacwg(i,k)+
                     pgsacw(i,k)+qmultg(i,k))*dt;
              if (dum > qc3d(i,k)  &&  qc3d(i,k) >= qsmall) {
                double ratio = qc3d(i,k)/dum;
                prc(i,k) = prc(i,k)*ratio;
                pra(i,k) = pra(i,k)*ratio;
                mnuccc(i,k) = mnuccc(i,k)*ratio;
                psacws(i,k) = psacws(i,k)*ratio;
                psacwi(i,k) = psacwi(i,k)*ratio;
                qmults(i,k) = qmults(i,k)*ratio;
                qmultg(i,k) = qmultg(i,k)*ratio;
                psacwg(i,k) = psacwg(i,k)*ratio;
                pgsacw(i,k) = pgsacw(i,k)*ratio;
              }
              // conservation of qi
              dum = (-prd(i,k)-mnuccc(i,k)+prci(i,k)+prai(i,k)-qmults(i,k)-qmultg(i,k)-qmultr(i,k)-qmultrg(i,k)-
                               mnuccd(i,k)+praci(i,k)+pracis(i,k)-eprd(i,k)-psacwi(i,k))*dt;
              if (dum > qi3d(i,k)  &&  qi3d(i,k) >= qsmall) {
                double ratio = (qi3d(i,k)/dt+prd(i,k)+mnuccc(i,k)+qmults(i,k)+qmultg(i,k)+qmultr(i,k)+qmultrg(i,k)+
                              mnuccd(i,k)+psacwi(i,k))/(prci(i,k)+prai(i,k)+praci(i,k)+pracis(i,k)-eprd(i,k));
                prci(i,k) = prci(i,k)*ratio;
                prai(i,k) = prai(i,k)*ratio;
                praci(i,k) = praci(i,k)*ratio;
                pracis(i,k) = pracis(i,k)*ratio;
                eprd(i,k) = eprd(i,k)*ratio;
              }
              // conservation of qr
              dum = ((pracs(i,k)-pre(i,k))+(qmultr(i,k)+qmultrg(i,k)-prc(i,k))+(mnuccr(i,k)-pra(i,k))+
                    piacr(i,k)+piacrs(i,k)+pgracs(i,k)+pracg(i,k))*dt;
              if (dum > qr3d(i,k) && qr3d(i,k) >= qsmall) {
                double ratio = (qr3d(i,k)/dt+prc(i,k)+pra(i,k))/(-pre(i,k)+qmultr(i,k)+qmultrg(i,k)+pracs(i,k)+
                                mnuccr(i,k)+piacr(i,k)+piacrs(i,k)+pgracs(i,k)+pracg(i,k));
                pre(i,k) = pre(i,k)*ratio;
                pracs(i,k) = pracs(i,k)*ratio;
                qmultr(i,k) = qmultr(i,k)*ratio;
                qmultrg(i,k) = qmultrg(i,k)*ratio;
                mnuccr(i,k) = mnuccr(i,k)*ratio;
                piacr(i,k) = piacr(i,k)*ratio;
                piacrs(i,k) = piacrs(i,k)*ratio;
                pgracs(i,k) = pgracs(i,k)*ratio;
                pracg(i,k) = pracg(i,k)*ratio;
              }
              // conservation of qni
              // conservation for graupel scheme
              if (igraup==0) {
                dum = (-prds(i,k)-psacws(i,k)-prai(i,k)-prci(i,k)-pracs(i,k)-eprds(i,k)+psacr(i,k)-
                      piacrs(i,k)-pracis(i,k))*dt;
                if (dum > qni3d(i,k) && qni3d(i,k) >= qsmall) {
                  double ratio = (qni3d(i,k)/dt+prds(i,k)+psacws(i,k)+prai(i,k)+prci(i,k)+pracs(i,k)+piacrs(i,k)+
                                pracis(i,k))/(-eprds(i,k)+psacr(i,k));
                  eprds(i,k) = eprds(i,k)*ratio;
                  psacr(i,k) = psacr(i,k)*ratio;
                }
              // for no graupel, need to include freezing of rain for snow
              } else if (igraup==1) {
                dum = (-prds(i,k)-psacws(i,k)-prai(i,k)-prci(i,k)-pracs(i,k)-eprds(i,k)+psacr(i,k)-piacrs(i,k)-
                      pracis(i,k)-mnuccr(i,k))*dt;
                if (dum > qni3d(i,k) && qni3d(i,k) >= qsmall) {
                  double ratio = (qni3d(i,k)/dt+prds(i,k)+psacws(i,k)+prai(i,k)+prci(i,k)+pracs(i,k)+piacrs(i,k)+
                                pracis(i,k)+mnuccr(i,k))/(-eprds(i,k)+psacr(i,k));
                  eprds(i,k) = eprds(i,k)*ratio;
                  psacr(i,k) = psacr(i,k)*ratio;
                }
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          double dum;
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              // conservation of qg
              dum = (-psacwg(i,k)-pracg(i,k)-pgsacw(i,k)-pgracs(i,k)-prdg(i,k)-mnuccr(i,k)-eprdg(i,k)-piacr(i,k)-
                    praci(i,k)-psacr(i,k))*dt;
              if (dum > qg3d(i,k) && qg3d(i,k) >= qsmall) {
                double ratio = (qg3d(i,k)/dt+psacwg(i,k)+pracg(i,k)+pgsacw(i,k)+pgracs(i,k)+prdg(i,k)+mnuccr(i,k)+
                              psacr(i,k)+piacr(i,k)+praci(i,k))/(-eprdg(i,k));
                eprdg(i,k) = eprdg(i,k)*ratio;
              }
              qv3dten(i,k) = qv3dten(i,k)+(-pre(i,k)-prd(i,k)-prds(i,k)-mnuccd(i,k)-eprd(i,k)-eprds(i,k)-
                             prdg(i,k)-eprdg(i,k));
              #ifdef MICRO_MORR_2011_02_20
                t3dten(i,k) = t3dten(i,k)+(pre(i,k)*xxlv(i,k)+(prd(i,k)+prds(i,k)+mnuccd(i,k)+eprd(i,k)+eprds(i,k)+
                              prdg(i,k)+eprdg(i,k))*xxls(i,k)+(psacws(i,k)+psacwi(i,k)+mnuccc(i,k)+mnuccr(i,k)+
                              qmults(i,k)+qmultg(i,k)+qmultr(i,k)+qmultrg(i,k)+pracs(i,k)+psacwg(i,k)+pracg(i,k)+
                              pgsacw(i,k)+pgracs(i,k)                       )*xlf(i,k))/cpm(i,k);
              #else
                t3dten(i,k) = t3dten(i,k)+(pre(i,k)*xxlv(i,k)+(prd(i,k)+prds(i,k)+mnuccd(i,k)+eprd(i,k)+eprds(i,k)+
                              prdg(i,k)+eprdg(i,k))*xxls(i,k)+(psacws(i,k)+psacwi(i,k)+mnuccc(i,k)+mnuccr(i,k)+
                              qmults(i,k)+qmultg(i,k)+qmultr(i,k)+qmultrg(i,k)+pracs(i,k)+psacwg(i,k)+pracg(i,k)+
                              pgsacw(i,k)+pgracs(i,k)+piacr(i,k)+piacrs(i,k))*xlf(i,k))/cpm(i,k);
              #endif
              qc3dten(i,k) = qc3dten(i,k)+(-pra(i,k)-prc(i,k)-mnuccc(i,k)+pcc(i,k)-psacws(i,k)-psacwi(i,k)-
                             qmults(i,k)-qmultg(i,k)-psacwg(i,k)-pgsacw(i,k));
              qi3dten(i,k) = qi3dten(i,k)+(prd(i,k)+eprd(i,k)+psacwi(i,k)+mnuccc(i,k)-prci(i,k)-prai(i,k)+
                             qmults(i,k)+qmultg(i,k)+qmultr(i,k)+qmultrg(i,k)+mnuccd(i,k)-praci(i,k)-pracis(i,k));
              qr3dten(i,k) = qr3dten(i,k)+(pre(i,k)+pra(i,k)+prc(i,k)-pracs(i,k)-mnuccr(i,k)-qmultr(i,k)-
                             qmultrg(i,k)-piacr(i,k)-piacrs(i,k)-pracg(i,k)-pgracs(i,k));
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              if (igraup==0) {
                qni3dten(i,k) = qni3dten(i,k)+(prai(i,k)+psacws(i,k)+prds(i,k)+pracs(i,k)+prci(i,k)+
                                eprds(i,k)-psacr(i,k)+piacrs(i,k)+pracis(i,k));
                ns3dten(i,k) = ns3dten(i,k)+(nsagg(i,k)+nprci(i,k)-nscng(i,k)-ngracs(i,k)+niacrs(i,k));
                qg3dten(i,k) = qg3dten(i,k)+(pracg(i,k)+psacwg(i,k)+pgsacw(i,k)+pgracs(i,k)+prdg(i,k)+eprdg(i,k)+
                               mnuccr(i,k)+piacr(i,k)+praci(i,k)+psacr(i,k));
                ng3dten(i,k) = ng3dten(i,k)+(nscng(i,k)+ngracs(i,k)+nnuccr(i,k)+niacr(i,k));
              // for no graupel, need to include freezing of rain for snow
              } else if (igraup==1) {
                qni3dten(i,k) = qni3dten(i,k)+(prai(i,k)+psacws(i,k)+prds(i,k)+pracs(i,k)+prci(i,k)+eprds(i,k)-
                                psacr(i,k)+piacrs(i,k)+pracis(i,k)+mnuccr(i,k));
                ns3dten(i,k) = ns3dten(i,k)+(nsagg(i,k)+nprci(i,k)-nscng(i,k)-ngracs(i,k)+niacrs(i,k)+nnuccr(i,k));
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              nc3dten(i,k) = nc3dten(i,k)+(-nnuccc(i,k)-npsacws(i,k)-npra(i,k)-nprc(i,k)-npsacwi(i,k)-npsacwg(i,k));
              ni3dten(i,k) = ni3dten(i,k)+(nnuccc(i,k)-nprci(i,k)-nprai(i,k)+nmults(i,k)+nmultg(i,k)+nmultr(i,k)+
                             nmultrg(i,k)+nnuccd(i,k)-niacr(i,k)-niacrs(i,k));
              nr3dten(i,k) = nr3dten(i,k)+(nprc1(i,k)-npracs(i,k)-nnuccr(i,k)+nragg(i,k)-niacr(i,k)-niacrs(i,k)-
                             npracg(i,k)-ngracs(i,k));
              #ifdef MICRO_MORR_2011_02_20
              #else
                c2prec (i,k) = pra(i,k)+prc(i,k)+psacws(i,k)+qmults(i,k)+qmultg(i,k)+psacwg(i,k)+pgsacw(i,k)+
                               mnuccc(i,k)+psacwi(i,k);
              #endif
              double dumt       = t3d(i,k)+dt*t3dten(i,k);
              double dumqv      = qv3d(i,k)+dt*qv3dten(i,k);
              double dum        = min( 0.99*pres(i,k) , polysvp(dumt,0) );
              #ifdef MICRO_MORR_2011_02_20
                double dumqss = 0.622*polysvp(dumt,0)/ (pres(i,k)-polysvp(dumt,0));
              #else
                double dumqss     = ep_2*dum/(pres(i,k)-dum);
              #endif
              double dumqc      = qc3d(i,k)+dt*qc3dten(i,k);
              dumqc            = max( dumqc , 0. );
              // saturation adjustment for liquid
              double dums       = dumqv-dumqss;
              pcc(i,k)     = dums/(1.+std::pow(xxlv(i,k),2)*dumqss/(cpm(i,k)*rv*std::pow(dumt,2)))/dt;
              if (pcc(i,k)*dt+dumqc < 0.) pcc(i,k) = -dumqc/dt;
              qv3dten(i,k) = qv3dten(i,k)-pcc(i,k);
              t3dten (i,k) = t3dten (i,k)+pcc(i,k)*xxlv(i,k)/cpm(i,k);
              qc3dten(i,k) = qc3dten(i,k)+pcc(i,k);
              // activation of cloud droplets
              // activation of droplet currently not calculated
              // droplet concentration is specified !!!!!
              // sublimate, melt, or evaporate number concentration
              // this formulation assumes 1:1 ratio between mass loss and
              // loss of number concentration
              if (eprd(i,k) < 0.) {
                dum      = eprd(i,k)*dt/qi3d(i,k);
                dum      = max(-1.,dum);
                nsubi(i,k) = dum*ni3d(i,k)/dt;
              }
              if (eprds(i,k) < 0.) {
                dum      = eprds(i,k)*dt/qni3d(i,k);
                dum      = max(-1.,dum);
                nsubs(i,k) = dum*ns3d(i,k)/dt;
              }
              if (pre(i,k) < 0.) {
                dum      = pre(i,k)*dt/qr3d(i,k);
                dum      = max(-1.,dum);
                nsubr(i,k) = dum*nr3d(i,k)/dt;
              }
              if (eprdg(i,k) < 0.) {
                dum      = eprdg(i,k)*dt/qg3d(i,k);
                dum      = max(-1.,dum);
                nsubg(i,k) = dum*ng3d(i,k)/dt;
              }
              ni3dten(i,k) = ni3dten(i,k)+nsubi(i,k);
              ns3dten(i,k) = ns3dten(i,k)+nsubs(i,k);
              ng3dten(i,k) = ng3dten(i,k)+nsubg(i,k);
              nr3dten(i,k) = nr3dten(i,k)+nsubr(i,k);
            } // temperature
            hydro_pres(i) = true; // No hydrometeors are present. Skip the rest of the routine
          } // if (! skip_micro(i,k))
      });

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (hydro_pres(i)) {
            // calculate sedimenation
            // the numerics here follow from reisner et al. (1998)
            // fallout terms are calculated on split time steps to ensure numerical
            // stability, i.e. courant# < 1
            dumi  (i,k) = qi3d (i,k)+qi3dten (i,k)*dt;
            dumqs (i,k) = qni3d(i,k)+qni3dten(i,k)*dt;
            dumr  (i,k) = qr3d (i,k)+qr3dten (i,k)*dt;
            dumfni(i,k) = ni3d (i,k)+ni3dten (i,k)*dt;
            dumfns(i,k) = ns3d (i,k)+ns3dten (i,k)*dt;
            dumfnr(i,k) = nr3d (i,k)+nr3dten (i,k)*dt;
            dumc  (i,k) = qc3d (i,k)+qc3dten (i,k)*dt;
            dumfnc(i,k) = nc3d (i,k)+nc3dten (i,k)*dt;
            dumg  (i,k) = qg3d (i,k)+qg3dten (i,k)*dt;
            dumfng(i,k) = ng3d (i,k)+ng3dten (i,k)*dt;
            // switch for constant droplet number
            if (iinum==1) dumfnc(i,k) = nc3d(i,k);
            // make sure number concentrations are positive
            dumfni(i,k) = max( 0. , dumfni(i,k) );
            dumfns(i,k) = max( 0. , dumfns(i,k) );
            dumfnc(i,k) = max( 0. , dumfnc(i,k) );
            dumfnr(i,k) = max( 0. , dumfnr(i,k) );
            dumfng(i,k) = max( 0. , dumfng(i,k) );
            // cloud ice
            double dlami;
            if (dumi(i,k) >= qsmall) {
              dlami = std::pow(cons12*dumfni(i,k)/dumi(i,k),1./di);
              dlami = max( dlami , lammini );
              dlami = min( dlami , lammaxi );
            }
            // rain
            double dlamr;
            if (dumr(i,k) >= qsmall) {
              dlamr = std::pow(pi*rhow*dumfnr(i,k)/dumr(i,k),1./3.);
              dlamr = max( dlamr , lamminr );
              dlamr = min( dlamr , lammaxr );
            }
            // cloud droplets
            double dlamc;
            if (dumc(i,k) >= qsmall) {
              double dum     = pres(i,k)/(287.15*t3d(i,k));
              pgam(i,k) = 0.0005714*(nc3d(i,k)/1.e6*dum)+0.2714;
              pgam(i,k) = 1./(std::pow(pgam(i,k),2))-1.;
              pgam(i,k) = max(pgam(i,k),2.);
              pgam(i,k) = min(pgam(i,k),10.);
              dlamc   = std::pow(cons26*dumfnc(i,k)*std::tgamma(pgam(i,k)+4.)/(dumc(i,k)*std::tgamma(pgam(i,k)+1.)),1./3.);
              double lammin  = (pgam(i,k)+1.)/60.e-6;
              double lammax  = (pgam(i,k)+1.)/1.e-6;
              dlamc   = max(dlamc,lammin);
              dlamc   = min(dlamc,lammax);
            }
            // snow
            double dlams;
            if (dumqs(i,k) >= qsmall) {
              dlams = std::pow(cons1*dumfns(i,k)/ dumqs(i,k),1./ds);
              dlams=max(dlams,lammins);
              dlams=min(dlams,lammaxs);
            }
            // graupel
            double dlamg;
            if (dumg(i,k) >= qsmall) {
              dlamg = std::pow(cons2*dumfng(i,k)/ dumg(i,k),1./dg);
              dlamg=max(dlamg,lamming);
              dlamg=min(dlamg,lammaxg);
            }
            // calculate number-weighted and mass-weighted terminal fall speeds
            // cloud water
            double unc, umc;
            if (dumc(i,k) >= qsmall) {
              unc =  acn(i,k)*std::tgamma(1.+bc+pgam(i,k))/ (std::pow(dlamc,bc)*std::tgamma(pgam(i,k)+1.));
              umc = acn(i,k)*std::tgamma(4.+bc+pgam(i,k))/  (std::pow(dlamc,bc)*std::tgamma(pgam(i,k)+4.));
            } else {
              umc = 0.;
              unc = 0.;
            }
            double uni, umi;
            if (dumi(i,k) >= qsmall) {
              uni = ain(i,k)*cons27/std::pow(dlami,bi);
              umi = ain(i,k)*cons28/std::pow(dlami,bi);
            } else {
              umi = 0.;
              uni = 0.;
            }
            double umr, unr;
            if (dumr(i,k) >= qsmall) {
              unr = arn(i,k)*cons6/std::pow(dlamr,br);
              umr = arn(i,k)*cons4/std::pow(dlamr,br);
            } else {
              umr = 0.;
              unr = 0.;
            }
            double ums, uns;
            if (dumqs(i,k) >= qsmall) {
              ums = asn(i,k)*cons3/std::pow(dlams,bs);
              uns = asn(i,k)*cons5/std::pow(dlams,bs);
            } else {
              ums = 0.;
              uns = 0.;
            }
            double umg, ung;
            if (dumg(i,k) >= qsmall) {
              umg = agn(i,k)*cons7/std::pow(dlamg,bg);
              ung = agn(i,k)*cons8/std::pow(dlamg,bg);
            } else {
              umg = 0.;
              ung = 0.;
            }
            // set realistic limits on fallspeed
            double dum    = std::pow(rhosu/rho(i,k),0.54);
            ums    = min(ums,1.2*dum);
            uns    = min(uns,1.2*dum);
            #ifdef MICRO_MORR_2011_02_20
              umi  = min(umi,1.2*dum);
              uni  = min(uni,1.2*dum);
            #else
              umi  = min(umi,1.2*std::pow(rhosu/rho(i,k),0.35));
              uni  = min(uni,1.2*std::pow(rhosu/rho(i,k),0.35));
            #endif
            umr    = min(umr,9.1*dum);
            unr    = min(unr,9.1*dum);
            umg    = min(umg,20.*dum);
            ung    = min(ung,20.*dum);
            fr (i,k) = umr;
            fi (i,k) = umi;
            fni(i,k) = uni;
            fs (i,k) = ums;
            fns(i,k) = uns;
            fnr(i,k) = unr;
            fc (i,k) = umc;
            fnc(i,k) = unc;
            fg (i,k) = umg;
            fng(i,k) = ung;
            dumr  (i,k) = dumr  (i,k)*rho(i,k);
            dumi  (i,k) = dumi  (i,k)*rho(i,k);
            dumfni(i,k) = dumfni(i,k)*rho(i,k);
            dumqs (i,k) = dumqs (i,k)*rho(i,k);
            dumfns(i,k) = dumfns(i,k)*rho(i,k);
            dumfnr(i,k) = dumfnr(i,k)*rho(i,k);
            dumc  (i,k) = dumc  (i,k)*rho(i,k);
            dumfnc(i,k) = dumfnc(i,k)*rho(i,k);
            dumg  (i,k) = dumg  (i,k)*rho(i,k);
            dumfng(i,k) = dumfng(i,k)*rho(i,k);
          }
      });

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<1>(ncol) , KOKKOS_LAMBDA (int i) {
        if (hydro_pres(i)) {
          nstep(i) = 1;
          for (int k = nz; k >= 1; k--) {
            #ifdef MICRO_MORR_2011_02_20
            #else
              if (k <= nz-1) {
                //  v3.3 modify fallspeed below level of precip
                if (fr (i,k) < 1.e-10) fr (i,k) = fr (i,k+1);
                if (fi (i,k) < 1.e-10) fi (i,k) = fi (i,k+1);
                if (fni(i,k) < 1.e-10) fni(i,k) = fni(i,k+1);
                if (fs (i,k) < 1.e-10) fs (i,k) = fs (i,k+1);
                if (fns(i,k) < 1.e-10) fns(i,k) = fns(i,k+1);
                if (fnr(i,k) < 1.e-10) fnr(i,k) = fnr(i,k+1);
                if (fc (i,k) < 1.e-10) fc (i,k) = fc (i,k+1);
                if (fnc(i,k) < 1.e-10) fnc(i,k) = fnc(i,k+1);
                if (fg (i,k) < 1.e-10) fg (i,k) = fg (i,k+1);
                if (fng(i,k) < 1.e-10) fng(i,k) = fng(i,k+1);
              } // k le nz-1
            #endif
            // calculate number of split time steps
            double rgvm = max(fr(i,k),max(fi(i,k),max(fs(i,k),max(fc(i,k),max(fni(i,k),max(fnr(i,k),
                              max(fns(i,k),max(fnc(i,k),max(fg(i,k),fng(i,k))))))))));
            nstep(i) = max(int(rgvm*dt/dzq(i,k)+1.),nstep(i));
          }
        }
      });

      auto max_nstep = yakl::intrinsics::maxval( nstep );

      for (int n = 1; n <= max_nstep; n++) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (hydro_pres(i) && nstep(i) <= n) {
              faloutr (i,k) = fr (i,k)*dumr  (i,k);
              falouti (i,k) = fi (i,k)*dumi  (i,k);
              faloutni(i,k) = fni(i,k)*dumfni(i,k);
              falouts (i,k) = fs (i,k)*dumqs (i,k);
              faloutns(i,k) = fns(i,k)*dumfns(i,k);
              faloutnr(i,k) = fnr(i,k)*dumfnr(i,k);
              faloutc (i,k) = fc (i,k)*dumc  (i,k);
              faloutnc(i,k) = fnc(i,k)*dumfnc(i,k);
              faloutg (i,k) = fg (i,k)*dumg  (i,k);
              faloutng(i,k) = fng(i,k)*dumfng(i,k);
          }
        });
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<1>(ncol) , KOKKOS_LAMBDA (int i) {
          if (hydro_pres(i) && nstep(i) <= n) {
            {
              // top of model
              int k = nz;
              double faltndr  = faloutr (i,k)/dzq(i,k);
              double faltndi  = falouti (i,k)/dzq(i,k);
              double faltndni = faloutni(i,k)/dzq(i,k);
              double faltnds  = falouts (i,k)/dzq(i,k);
              double faltndns = faloutns(i,k)/dzq(i,k);
              double faltndnr = faloutnr(i,k)/dzq(i,k);
              double faltndc  = faloutc (i,k)/dzq(i,k);
              double faltndnc = faloutnc(i,k)/dzq(i,k);
              double faltndg  = faloutg (i,k)/dzq(i,k);
              double faltndng = faloutng(i,k)/dzq(i,k);
              // add fallout terms to eulerian tendencies
              qrsten (i,k) = qrsten (i,k)-faltndr /nstep(i)/rho(i,k);
              qisten (i,k) = qisten (i,k)-faltndi /nstep(i)/rho(i,k);
              ni3dten(i,k) = ni3dten(i,k)-faltndni/nstep(i)/rho(i,k);
              qnisten(i,k) = qnisten(i,k)-faltnds /nstep(i)/rho(i,k);
              ns3dten(i,k) = ns3dten(i,k)-faltndns/nstep(i)/rho(i,k);
              nr3dten(i,k) = nr3dten(i,k)-faltndnr/nstep(i)/rho(i,k);
              qcsten (i,k) = qcsten (i,k)-faltndc /nstep(i)/rho(i,k);
              nc3dten(i,k) = nc3dten(i,k)-faltndnc/nstep(i)/rho(i,k);
              qgsten (i,k) = qgsten (i,k)-faltndg /nstep(i)/rho(i,k);
              ng3dten(i,k) = ng3dten(i,k)-faltndng/nstep(i)/rho(i,k);
              dumr  (i,k) = dumr  (i,k)-faltndr *dt/nstep(i);
              dumi  (i,k) = dumi  (i,k)-faltndi *dt/nstep(i);
              dumfni(i,k) = dumfni(i,k)-faltndni*dt/nstep(i);
              dumqs (i,k) = dumqs (i,k)-faltnds *dt/nstep(i);
              dumfns(i,k) = dumfns(i,k)-faltndns*dt/nstep(i);
              dumfnr(i,k) = dumfnr(i,k)-faltndnr*dt/nstep(i);
              dumc  (i,k) = dumc  (i,k)-faltndc *dt/nstep(i);
              dumfnc(i,k) = dumfnc(i,k)-faltndnc*dt/nstep(i);
              dumg  (i,k) = dumg  (i,k)-faltndg *dt/nstep(i);
              dumfng(i,k) = dumfng(i,k)-faltndng*dt/nstep(i);
            }
          }
        });
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz-1,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (hydro_pres(i) && nstep(i) <= n) {
              double faltndr  = (faloutr (i,k+1)-faloutr (i,k))/dzq(i,k);
              double faltndi  = (falouti (i,k+1)-falouti (i,k))/dzq(i,k);
              double faltndni = (faloutni(i,k+1)-faloutni(i,k))/dzq(i,k);
              double faltnds  = (falouts (i,k+1)-falouts (i,k))/dzq(i,k);
              double faltndns = (faloutns(i,k+1)-faloutns(i,k))/dzq(i,k);
              double faltndnr = (faloutnr(i,k+1)-faloutnr(i,k))/dzq(i,k);
              double faltndc  = (faloutc (i,k+1)-faloutc (i,k))/dzq(i,k);
              double faltndnc = (faloutnc(i,k+1)-faloutnc(i,k))/dzq(i,k);
              double faltndg  = (faloutg (i,k+1)-faloutg (i,k))/dzq(i,k);
              double faltndng = (faloutng(i,k+1)-faloutng(i,k))/dzq(i,k);
              // add fallout terms to eulerian tendencies
              qrsten (i,k) = qrsten (i,k)+faltndr    /nstep(i)/rho(i,k);
              qisten (i,k) = qisten (i,k)+faltndi    /nstep(i)/rho(i,k);
              ni3dten(i,k) = ni3dten(i,k)+faltndni   /nstep(i)/rho(i,k);
              qnisten(i,k) = qnisten(i,k)+faltnds    /nstep(i)/rho(i,k);
              ns3dten(i,k) = ns3dten(i,k)+faltndns   /nstep(i)/rho(i,k);
              nr3dten(i,k) = nr3dten(i,k)+faltndnr   /nstep(i)/rho(i,k);
              qcsten (i,k) = qcsten (i,k)+faltndc    /nstep(i)/rho(i,k);
              nc3dten(i,k) = nc3dten(i,k)+faltndnc   /nstep(i)/rho(i,k);
              qgsten (i,k) = qgsten (i,k)+faltndg    /nstep(i)/rho(i,k);
              ng3dten(i,k) = ng3dten(i,k)+faltndng   /nstep(i)/rho(i,k);
              dumr   (i,k) = dumr   (i,k)+faltndr *dt/nstep(i);
              dumi   (i,k) = dumi   (i,k)+faltndi *dt/nstep(i);
              dumfni (i,k) = dumfni (i,k)+faltndni*dt/nstep(i);
              dumqs  (i,k) = dumqs  (i,k)+faltnds *dt/nstep(i);
              dumfns (i,k) = dumfns (i,k)+faltndns*dt/nstep(i);
              dumfnr (i,k) = dumfnr (i,k)+faltndnr*dt/nstep(i);
              dumc   (i,k) = dumc   (i,k)+faltndc *dt/nstep(i);
              dumfnc (i,k) = dumfnc (i,k)+faltndnc*dt/nstep(i);
              dumg   (i,k) = dumg   (i,k)+faltndg *dt/nstep(i);
              dumfng (i,k) = dumfng (i,k)+faltndng*dt/nstep(i);
              // for wrf-chem, need precip rates (units of kg/m^2/s)
              csed(i,k)=csed(i,k)+faloutc(i,k)/nstep(i);
              ised(i,k)=ised(i,k)+falouti(i,k)/nstep(i);
              ssed(i,k)=ssed(i,k)+falouts(i,k)/nstep(i);
              gsed(i,k)=gsed(i,k)+faloutg(i,k)/nstep(i);
              rsed(i,k)=rsed(i,k)+faloutr(i,k)/nstep(i);
              if (k == 1) {
                precrt (i) = precrt (i)+(faloutr(i,1)+faloutc(i,1)+falouts(i,1)+falouti(i,1)+faloutg(i,1))*dt/nstep(i);
                snowrt (i) = snowrt (i)+(falouts(i,1)+falouti(i,1)+faloutg(i,1))*dt/nstep(i);
                #ifdef MICRO_MORR_2011_02_20
                #else
                  snowprt(i) = snowprt(i)+(falouti(i,1)+falouts(i,1))*dt/nstep(i);
                  grplprt(i) = grplprt(i)+(faloutg(i,1))*dt/nstep(i);
                #endif
              }
          }
        });
      } // nstep(i)


      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (hydro_pres(i)) {
            // add on sedimentation tendencies for mixing ratio to rest of tendencies
            qr3dten (i,k) = qr3dten (i,k) + qrsten (i,k);
            qi3dten (i,k) = qi3dten (i,k) + qisten (i,k);
            qc3dten (i,k) = qc3dten (i,k) + qcsten (i,k);
            qg3dten (i,k) = qg3dten (i,k) + qgsten (i,k);
            qni3dten(i,k) = qni3dten(i,k) + qnisten(i,k);;
            // put all cloud ice in snow category if mean diameter exceeds 2 * dcs
            if (qi3d(i,k) >= qsmall && t3d(i,k) < 273.15 && lami(i,k) >= 1.e-10) {
              if (1./lami(i,k) >= 2.*dcs) {
                qni3dten(i,k) = qni3dten(i,k)+qi3d(i,k)/dt+ qi3dten(i,k);
                ns3dten(i,k) = ns3dten(i,k)+ni3d(i,k)/dt+   ni3dten(i,k);
                qi3dten(i,k) = -qi3d(i,k)/dt;
                ni3dten(i,k) = -ni3d(i,k)/dt;
              }
            }
            // hm add tendencies here, then call sizeparameter
            // to ensure consisitency between mixing ratio and number concentration
            qc3d (i,k) = qc3d (i,k)+qc3dten (i,k)*dt;
            qi3d (i,k) = qi3d (i,k)+qi3dten (i,k)*dt;
            qni3d(i,k) = qni3d(i,k)+qni3dten(i,k)*dt;
            qr3d (i,k) = qr3d (i,k)+qr3dten (i,k)*dt;
            #ifdef MICRO_MORR_2011_02_20
            #else
              nc3d (i,k) = nc3d (i,k)+nc3dten (i,k)*dt;
            #endif
            ni3d (i,k) = ni3d (i,k)+ni3dten (i,k)*dt;
            ns3d (i,k) = ns3d (i,k)+ns3dten (i,k)*dt;
            nr3d (i,k) = nr3d (i,k)+nr3dten (i,k)*dt;
            if (igraup==0) {
              qg3d(i,k) = qg3d(i,k)+qg3dten(i,k)*dt;
              ng3d(i,k) = ng3d(i,k)+ng3dten(i,k)*dt;
            }
            // add temperature and water vapor tendencies from microphysics
            t3d (i,k) = t3d (i,k)+t3dten (i,k)*dt;
            qv3d(i,k) = qv3d(i,k)+qv3dten(i,k)*dt;
            // saturation vapor pressure and mixing ratio
            #ifdef MICRO_MORR_2011_02_20
              evs(i,k) = polysvp(t3d(i,k),0);
              eis(i,k) = polysvp(t3d(i,k),1);
            #else
              evs(i,k) = min( 0.99*pres(i,k) , polysvp(t3d(i,k),0) )  ; // pa
              eis(i,k) = min( 0.99*pres(i,k) , polysvp(t3d(i,k),1) ) ;  // pa
            #endif
            // make sure ice saturation doesn't exceed water sat. near freezing
            if (eis(i,k) > evs(i,k)) eis(i,k) = evs(i,k);
            #ifdef MICRO_MORR_2011_02_20
              qvs(i,k) = .622*evs(i,k)/(pres(i,k)-evs(i,k));
              qvi(i,k) = .622*eis(i,k)/(pres(i,k)-eis(i,k));
            #else
              qvs(i,k) = ep_2*evs(i,k)/(pres(i,k)-evs(i,k));
              qvi(i,k) = ep_2*eis(i,k)/(pres(i,k)-eis(i,k));
            #endif
            qvqvs(i,k) = qv3d(i,k)/qvs(i,k);
            qvqvsi(i,k) = qv3d(i,k)/qvi(i,k);
            // at subsaturation, remove small amounts of cloud/precip water
            if (qvqvs(i,k) < 0.9) {
              if (qr3d(i,k) < 1.e-8) {
                qv3d(i,k)=qv3d(i,k)+qr3d(i,k);
                t3d (i,k)=t3d (i,k)-qr3d(i,k)*xxlv(i,k)/cpm(i,k);
                qr3d(i,k)=0.;
              }
              if (qc3d(i,k) < 1.e-8) {
                qv3d(i,k)=qv3d(i,k)+qc3d(i,k);
                t3d (i,k)=t3d (i,k)-qc3d(i,k)*xxlv(i,k)/cpm(i,k);
                qc3d(i,k)=0.;
              }
            }
            if (qvqvsi(i,k) < 0.9) {
              if (qi3d(i,k) < 1.e-8) {
                qv3d(i,k)=qv3d(i,k)+qi3d(i,k);
                t3d (i,k)=t3d (i,k)-qi3d(i,k)*xxls(i,k)/cpm(i,k);
                qi3d(i,k)=0.;
              }
              if (qni3d(i,k) < 1.e-8) {
                qv3d (i,k)=qv3d(i,k)+qni3d(i,k);
                t3d  (i,k)=t3d (i,k)-qni3d(i,k)*xxls(i,k)/cpm(i,k);
                qni3d(i,k)=0.;
              }
              if (qg3d(i,k) < 1.e-8) {
                qv3d(i,k)=qv3d(i,k)+qg3d(i,k);
                t3d (i,k)=t3d (i,k)-qg3d(i,k)*xxls(i,k)/cpm(i,k);
                qg3d(i,k)=0.;
              }
            }
            // if mixing ratio < qsmall set mixing ratio and number conc to zero
            if (qc3d(i,k) < qsmall) {
              qc3d(i,k) = 0.;
              nc3d(i,k) = 0.;
              effc(i,k) = 0.;
            }
            if (qr3d(i,k) < qsmall) {
              qr3d(i,k) = 0.;
              nr3d(i,k) = 0.;
              effr(i,k) = 0.;
            }
            if (qi3d(i,k) < qsmall) {
              qi3d(i,k) = 0.;
              ni3d(i,k) = 0.;
              effi(i,k) = 0.;
            }
            if (qni3d(i,k) < qsmall) {
              qni3d(i,k) = 0.;
              ns3d (i,k) = 0.;
              effs (i,k) = 0.;
            }
            if (qg3d(i,k) < qsmall) {
              qg3d(i,k) = 0.;
              ng3d(i,k) = 0.;
              effg(i,k) = 0.;
            }
            // if there is no cloud/precip water, then skip calculations
            if ( !  (qc3d(i,k) < qsmall && qi3d(i,k) < qsmall && qni3d(i,k) < qsmall  && 
                     qr3d(i,k) < qsmall && qg3d(i,k) < qsmall)) {
              // calculate instantaneous processes
              // add melting of cloud ice to form rain
              if (qi3d(i,k) >= qsmall && t3d(i,k) >= 273.15) {
                qr3d(i,k) = qr3d(i,k)+qi3d(i,k);
                t3d(i,k) = t3d(i,k)-qi3d(i,k)*xlf(i,k)/cpm(i,k);
                qi3d(i,k) = 0.;
                nr3d(i,k) = nr3d(i,k)+ni3d(i,k);
                ni3d(i,k) = 0.;
              }
              // ****sensitivity - no ice
              if (iliq != 1) {
                // homogeneous freezing of cloud water
                if (t3d(i,k) <= 233.15 && qc3d(i,k) >= qsmall) {
                  qi3d(i,k)=qi3d(i,k)+qc3d(i,k);
                  t3d (i,k)=t3d (i,k)+qc3d(i,k)*xlf(i,k)/cpm(i,k);
                  qc3d(i,k)=0.;
                  ni3d(i,k)=ni3d(i,k)+nc3d(i,k);
                  nc3d(i,k)=0.;
                }
                // homogeneous freezing of rain
                if (igraup==0) {
                  if (t3d(i,k) <= 233.15 && qr3d(i,k) >= qsmall) {
                     qg3d(i,k) = qg3d(i,k)+qr3d(i,k);
                     t3d (i,k) = t3d (i,k)+qr3d(i,k)*xlf(i,k)/cpm(i,k);
                     qr3d(i,k) = 0.;
                     ng3d(i,k) = ng3d(i,k)+ nr3d(i,k);
                     nr3d(i,k) = 0.;
                  }
                } else if (igraup==1) {
                  if (t3d(i,k) <= 233.15 && qr3d(i,k) >= qsmall) {
                    qni3d(i,k) = qni3d(i,k)+qr3d(i,k);
                    t3d  (i,k) = t3d  (i,k)+qr3d(i,k)*xlf(i,k)/cpm(i,k);
                    qr3d (i,k) = 0.;
                    ns3d (i,k) = ns3d (i,k)+nr3d(i,k);
                    nr3d (i,k) = 0.;
                  }
                }
              }
              // make sure number concentrations aren't negative
              ni3d(i,k) = max( 0. , ni3d(i,k) );
              ns3d(i,k) = max( 0. , ns3d(i,k) );
              nc3d(i,k) = max( 0. , nc3d(i,k) );
              nr3d(i,k) = max( 0. , nr3d(i,k) );
              ng3d(i,k) = max( 0. , ng3d(i,k) );
              // cloud ice
              if (qi3d(i,k) >= qsmall) {
                lami(i,k) = std::pow(cons12*ni3d(i,k)/qi3d(i,k),1./di);
                if (lami(i,k) < lammini) {
                  lami(i,k) = lammini;
                  n0i(i,k) = std::pow(lami(i,k),4)*qi3d(i,k)/cons12;
                  ni3d(i,k) = n0i(i,k)/lami(i,k);
                } else if (lami(i,k) > lammaxi) {
                  lami(i,k) = lammaxi;
                  n0i(i,k) = std::pow(lami(i,k),4)*qi3d(i,k)/cons12;
                  ni3d(i,k) = n0i(i,k)/lami(i,k);
                }
              }
              // rain
              if (qr3d(i,k) >= qsmall) {
                lamr(i,k) = std::pow(pi*rhow*nr3d(i,k)/qr3d(i,k),1./3.);
                if (lamr(i,k) < lamminr) {
                  lamr(i,k) = lamminr;
                  n0rr(i,k) = std::pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow);
                  nr3d(i,k) = n0rr(i,k)/lamr(i,k);
                } else if (lamr(i,k) > lammaxr) {
                  lamr(i,k) = lammaxr;
                  n0rr(i,k) = std::pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow);
                  nr3d(i,k) = n0rr(i,k)/lamr(i,k);
                }
              }
              // cloud droplets
              if (qc3d(i,k) >= qsmall) {
                double dum = pres(i,k)/(287.15*t3d(i,k));
                pgam(i,k)=0.0005714*(nc3d(i,k)/1.e6*dum)+0.2714;
                pgam(i,k)=1./(std::pow(pgam(i,k),2))-1.;
                pgam(i,k)=max(pgam(i,k),2.);
                pgam(i,k)=min(pgam(i,k),10.);
                lamc(i,k) = std::pow(cons26*nc3d(i,k)*std::tgamma(pgam(i,k)+4.)/(qc3d(i,k)*std::tgamma(pgam(i,k)+1.)),1./3.);
                // lammin, 60 micron diameter
                // lammax, 1 micron
                double lammin = (pgam(i,k)+1.)/60.e-6;
                double lammax = (pgam(i,k)+1.)/1.e-6;
                if (lamc(i,k) < lammin) {
                  lamc(i,k) = lammin;
                  nc3d(i,k) = std::exp(3.*std::log(lamc(i,k))+std::log(qc3d(i,k))+std::log(std::tgamma(pgam(i,k)+1.))-
                              std::log(std::tgamma(pgam(i,k)+4.)))/cons26;
                } else if (lamc(i,k) > lammax) {
                  lamc(i,k) = lammax;
                  nc3d(i,k) = std::exp(3.*std::log(lamc(i,k))+std::log(qc3d(i,k))+std::log(std::tgamma(pgam(i,k)+1.))-
                              std::log(std::tgamma(pgam(i,k)+4.)))/cons26;
                }
              }
              // snow
              if (qni3d(i,k) >= qsmall) {
                lams(i,k) = std::pow(cons1*ns3d(i,k)/qni3d(i,k),1./ds);
                if (lams(i,k) < lammins) {
                  lams(i,k) = lammins;
                  n0s(i,k) = std::pow(lams(i,k),4)*qni3d(i,k)/cons1;
                  ns3d(i,k) = n0s(i,k)/lams(i,k);
                } else if (lams(i,k) > lammaxs) {
                  lams(i,k) = lammaxs;
                  n0s(i,k) = std::pow(lams(i,k),4)*qni3d(i,k)/cons1;
                  ns3d(i,k) = n0s(i,k)/lams(i,k);
                }
              }
              // graupel
              if (qg3d(i,k) >= qsmall) {
                lamg(i,k) = std::pow(cons2*ng3d(i,k)/qg3d(i,k),1./dg);
                if (lamg(i,k) < lamming) {
                  lamg(i,k) = lamming;
                  n0g(i,k) = std::pow(lamg(i,k),4)*qg3d(i,k)/cons2;
                  ng3d(i,k) = n0g(i,k)/lamg(i,k);
                } else if (lamg(i,k) > lammaxg) {
                  lamg(i,k) = lammaxg;
                  n0g(i,k) = std::pow(lamg(i,k),4)*qg3d(i,k)/cons2;
                  ng3d(i,k) = n0g(i,k)/lamg(i,k);
                }
              }
            }
            // calculate effective radius
            if (qi3d(i,k) >= qsmall) {
              effi(i,k) = 3./lami(i,k)/2.*1.e6;
            } else {
              effi(i,k) = 25.;
            }
            if (qni3d(i,k) >= qsmall) {
              effs(i,k) = 3./lams(i,k)/2.*1.e6;
            } else {
              effs(i,k) = 25.;
            }
            if (qr3d(i,k) >= qsmall) {
              effr(i,k) = 3./lamr(i,k)/2.*1.e6;
            } else {
              effr(i,k) = 25.;
            }
            if (qc3d(i,k) >= qsmall) {
              effc(i,k) = std::tgamma(pgam(i,k)+4.)/std::tgamma(pgam(i,k)+3.)/lamc(i,k)/2.*1.e6;
            } else {
              effc(i,k) = 25.;
            }
            if (qg3d(i,k) >= qsmall) {
              effg(i,k) = 3./lamg(i,k)/2.*1.e6;
            } else {
              effg(i,k) = 25.;
            }
            // hm add 1/10/06, add upper bound on ice number, this is needed
            // to prevent very large ice number due to homogeneous freezing
            // of droplets, especially when inum = 1, set max at 10 cm-3
            //          ni3d(k) = min(ni3d(k),10.e6/rho(k))
            // hm, 12/28/12, lower maximum ice concentration to address problem
            // of excessive and persistent anvil
            // note: this may change/reduce sensitivity to aerosol/ccn concentration
            #ifdef MICRO_MORR_2011_02_20
              ni3d(i,k) = min( ni3d(i,k) , 10.e6/rho(i,k) );
            #else
              ni3d(i,k) = min( ni3d(i,k) , 0.3e6/rho(i,k) );
            #endif
            // add bound on droplet number - cannot exceed aerosol concentration
            if (iinum==0 && iact==2) {
              nc3d(i,k) = min( nc3d(i,k) , (nanew1+nanew2)/rho(i,k) );
            }
            // switch for constant droplet number
            if (iinum==1) { 
              // change ndcnst from cm-3 to kg-1
              nc3d(i,k) = ndcnst*1.e6/rho(i,k);
            }
          } // If (hydro_pres(i))
      });
    }


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

  };

}


#include "Mp_morr_two_moment_context.h"

namespace modules {

void Mp_morr_two_moment::init(int ihail) {
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

void Mp_morr_two_moment::run(double2d_F const &t, double2d_F const &qv, double2d_F const &qc, double2d_F const &qr,
             double2d_F const &qi, double2d_F const &qs, double2d_F const &qg, double2d_F const &ni,
             double2d_F const &ns, double2d_F const &nr, double2d_F const &ng,
             doubleConst2d_F p, double dt_in, doubleConst2d_F dz, double1d_F const &rainnc,
             double1d_F const &rainncv, double1d_F const &sr, double1d_F const &snownc,
             double1d_F const &snowncv, double1d_F const &graupelnc, double1d_F const &graupelncv,
             doubleConst2d_F qrcuten, doubleConst2d_F qscuten, doubleConst2d_F qicuten, int ncol,
             int nz, double2d_F const &qlsink, double2d_F const &precr, double2d_F const &preci,
             double2d_F const &precs, double2d_F const &precg) {
      using yakl::parallel_for_F;
      using yakl::SimpleBounds_F;
      double2d_F c2prec("c2prec",ncol,nz);
      double dt    = dt_in;
      // inum = 0, predict droplet concentration
      // inum = 1, assume constant droplet concentration
      // !!!note: predicted droplet concentration not available in this version
      // contact hugh morrison (morrison@ucar.edu) for further information
      int  iinum = 1;
      run_two_mom(qc, qi, qs, qr ,ni, ns, nr, t, qv, p, dz, rainncv, sr, snowncv, graupelncv, dt, ncol, nz, qg,
                  ng, qrcuten, qscuten, qicuten, iinum, c2prec, preci, precs, precg, precr);
      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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

void Mp_morr_two_moment::run_two_mom(double2d_F const &qc3d, double2d_F const &qi3d, double2d_F const &qni3d, double2d_F const &qr3d,
                     double2d_F const &ni3d, double2d_F const &ns3d, double2d_F const &nr3d, double2d_F const &t3d,
                     double2d_F const &qv3d, doubleConst2d_F pres, doubleConst2d_F dzq, double1d_F const &precrt,
                     double1d_F const &snowrt, double1d_F const &snowprt, double1d_F const &grplprt, double dt, int ncol,
                     int nz, double2d_F const &qg3d, double2d_F const &ng3d, doubleConst2d_F qrcu1d,
                     doubleConst2d_F qscu1d, doubleConst2d_F qicu1d, int iinum, double2d_F const &c2prec,
                     double2d_F const &ised, double2d_F const &ssed, double2d_F const &gsed, double2d_F const &rsed) {
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
      RunContext context;
      context.qc3d = qc3d;
      context.qi3d = qi3d;
      context.qni3d = qni3d;
      context.qr3d = qr3d;
      context.ni3d = ni3d;
      context.ns3d = ns3d;
      context.nr3d = nr3d;
      context.t3d = t3d;
      context.qv3d = qv3d;
      context.pres = pres;
      context.dzq = dzq;
      context.precrt = precrt;
      context.snowrt = snowrt;
      context.snowprt = snowprt;
      context.grplprt = grplprt;
      context.dt = dt;
      context.ncol = ncol;
      context.nz = nz;
      context.qg3d = qg3d;
      context.ng3d = ng3d;
      context.qrcu1d = qrcu1d;
      context.qscu1d = qscu1d;
      context.qicu1d = qicu1d;
      context.iinum = iinum;
      context.c2prec = c2prec;
      context.ised = ised;
      context.ssed = ssed;
      context.gsed = gsed;
      context.rsed = rsed;
      context.ag = ag;
      context.bg = bg;
      context.rhog = rhog;
      context.mi0 = mi0;
      context.ci = ci;
      context.cs = cs;
      context.cg = cg;
      context.mmult = mmult;
      context.lammini = lammini;
      context.bact = bact;
      context.f11 = f11;
      context.f21 = f21;
      context.f12 = f12;
      context.f22 = f22;
      context.cons1 = cons1;
      context.cons2 = cons2;
      context.cons3 = cons3;
      context.cons4 = cons4;
      context.cons5 = cons5;
      context.cons6 = cons6;
      context.cons7 = cons7;
      context.cons8 = cons8;
      context.cons9 = cons9;
      context.cons10 = cons10;
      context.cons11 = cons11;
      context.cons12 = cons12;
      context.cons13 = cons13;
      context.cons14 = cons14;
      context.cons15 = cons15;
      context.cons16 = cons16;
      context.cons17 = cons17;
      context.cons18 = cons18;
      context.cons19 = cons19;
      context.cons20 = cons20;
      context.cons21 = cons21;
      context.cons22 = cons22;
      context.cons23 = cons23;
      context.cons24 = cons24;
      context.cons25 = cons25;
      context.cons26 = cons26;
      context.cons27 = cons27;
      context.cons28 = cons28;
      context.cons29 = cons29;
      context.cons30 = cons30;
      context.cons31 = cons31;
      context.cons32 = cons32;
      context.cons33 = cons33;
      context.cons34 = cons34;
      context.cons35 = cons35;
      context.cons36 = cons36;
      context.cons37 = cons37;
      context.cons38 = cons38;
      context.cons39 = cons39;
      context.cons40 = cons40;
      context.cons41 = cons41;
      context.ng3dten = ng3dten;
      context.qg3dten = qg3dten;
      context.effc = effc;
      context.effi = effi;
      context.effs = effs;
      context.effr = effr;
      context.effg = effg;
      context.t3dten = t3dten;
      context.qv3dten = qv3dten;
      context.qc3dten = qc3dten;
      context.qi3dten = qi3dten;
      context.qni3dten = qni3dten;
      context.qr3dten = qr3dten;
      context.ni3dten = ni3dten;
      context.ns3dten = ns3dten;
      context.nr3dten = nr3dten;
      context.csed = csed;
      context.qgsten = qgsten;
      context.qrsten = qrsten;
      context.qisten = qisten;
      context.qnisten = qnisten;
      context.qcsten = qcsten;
      context.nc3d = nc3d;
      context.nc3dten = nc3dten;
      context.lamc = lamc;
      context.lami = lami;
      context.lams = lams;
      context.lamr = lamr;
      context.lamg = lamg;
      context.cdist1 = cdist1;
      context.n0i = n0i;
      context.n0s = n0s;
      context.n0rr = n0rr;
      context.n0g = n0g;
      context.pgam = pgam;
      context.nsubc = nsubc;
      context.nsubi = nsubi;
      context.nsubs = nsubs;
      context.nsubr = nsubr;
      context.prd = prd;
      context.pre = pre;
      context.prds = prds;
      context.nnuccc = nnuccc;
      context.mnuccc = mnuccc;
      context.pra = pra;
      context.prc = prc;
      context.pcc = pcc;
      context.nnuccd = nnuccd;
      context.mnuccd = mnuccd;
      context.mnuccr = mnuccr;
      context.nnuccr = nnuccr;
      context.npra = npra;
      context.nragg = nragg;
      context.nsagg = nsagg;
      context.nprc = nprc;
      context.nprc1 = nprc1;
      context.prai = prai;
      context.prci = prci;
      context.psacws = psacws;
      context.npsacws = npsacws;
      context.psacwi = psacwi;
      context.npsacwi = npsacwi;
      context.nprci = nprci;
      context.nprai = nprai;
      context.nmults = nmults;
      context.nmultr = nmultr;
      context.qmults = qmults;
      context.qmultr = qmultr;
      context.pracs = pracs;
      context.npracs = npracs;
      context.pccn = pccn;
      context.psmlt = psmlt;
      context.evpms = evpms;
      context.nsmlts = nsmlts;
      context.nsmltr = nsmltr;
      context.piacr = piacr;
      context.niacr = niacr;
      context.praci = praci;
      context.piacrs = piacrs;
      context.niacrs = niacrs;
      context.pracis = pracis;
      context.eprd = eprd;
      context.eprds = eprds;
      context.pracg = pracg;
      context.psacwg = psacwg;
      context.pgsacw = pgsacw;
      context.pgracs = pgracs;
      context.prdg = prdg;
      context.eprdg = eprdg;
      context.evpmg = evpmg;
      context.pgmlt = pgmlt;
      context.npracg = npracg;
      context.npsacwg = npsacwg;
      context.nscng = nscng;
      context.ngracs = ngracs;
      context.ngmltg = ngmltg;
      context.ngmltr = ngmltr;
      context.nsubg = nsubg;
      context.psacr = psacr;
      context.nmultg = nmultg;
      context.nmultrg = nmultrg;
      context.qmultg = qmultg;
      context.qmultrg = qmultrg;
      context.kap = kap;
      context.evs = evs;
      context.eis = eis;
      context.qvs = qvs;
      context.qvi = qvi;
      context.qvqvs = qvqvs;
      context.qvqvsi = qvqvsi;
      context.dv = dv;
      context.xxls = xxls;
      context.xxlv = xxlv;
      context.cpm = cpm;
      context.mu = mu;
      context.sc = sc;
      context.xlf = xlf;
      context.rho = rho;
      context.ab = ab;
      context.abi = abi;
      context.dap = dap;
      context.dumi = dumi;
      context.dumr = dumr;
      context.dumfni = dumfni;
      context.dumg = dumg;
      context.dumfng = dumfng;
      context.fr = fr;
      context.fi = fi;
      context.fni = fni;
      context.fg = fg;
      context.fng = fng;
      context.faloutr = faloutr;
      context.falouti = falouti;
      context.faloutni = faloutni;
      context.dumqs = dumqs;
      context.dumfns = dumfns;
      context.fs = fs;
      context.fns = fns;
      context.falouts = falouts;
      context.faloutns = faloutns;
      context.faloutg = faloutg;
      context.faloutng = faloutng;
      context.dumc = dumc;
      context.dumfnc = dumfnc;
      context.fc = fc;
      context.faloutc = faloutc;
      context.faloutnc = faloutnc;
      context.fnc = fnc;
      context.dumfnr = dumfnr;
      context.faloutnr = faloutnr;
      context.fnr = fnr;
      context.ain = ain;
      context.arn = arn;
      context.asn = asn;
      context.acn = acn;
      context.agn = agn;
      context.tqimelt = tqimelt;
      context.skip_micro = skip_micro;
      context.t_ge_273 = t_ge_273;
      context.no_cirg = no_cirg;
      context.nstep = nstep;
      context.hydro_pres = hydro_pres;
      run_two_mom_initialize(context);
      run_two_mom_warm_processes(context);
      run_two_mom_cold_processes(context);
      run_two_mom_tendencies(context);
      run_two_mom_sedimentation(context);
      run_two_mom_finalize(context);
    }


} // namespace modules

#include "Mp_morr_two_moment_context.h"

namespace modules {

  void Mp_morr_two_moment::run_two_mom_warm_processes(RunContext const &context) {
    using yakl::parallel_for_F;
    using yakl::SimpleBounds_F;
    auto qc3d = context.qc3d;
    auto qni3d = context.qni3d;
    auto qr3d = context.qr3d;
    auto ns3d = context.ns3d;
    auto nr3d = context.nr3d;
    auto t3d = context.t3d;
    auto qv3d = context.qv3d;
    auto pres = context.pres;
    auto dt = context.dt;
    auto ncol = context.ncol;
    auto nz = context.nz;
    auto qg3d = context.qg3d;
    auto ng3d = context.ng3d;
    auto iinum = context.iinum;
    auto c2prec = context.c2prec;
    auto bg = context.bg;
    auto cons1 = context.cons1;
    auto cons2 = context.cons2;
    auto cons3 = context.cons3;
    auto cons4 = context.cons4;
    auto cons5 = context.cons5;
    auto cons6 = context.cons6;
    auto cons7 = context.cons7;
    auto cons8 = context.cons8;
    auto cons9 = context.cons9;
    auto cons10 = context.cons10;
    auto cons11 = context.cons11;
    auto cons26 = context.cons26;
    auto cons29 = context.cons29;
    auto cons32 = context.cons32;
    auto cons34 = context.cons34;
    auto cons35 = context.cons35;
    auto cons36 = context.cons36;
    auto cons41 = context.cons41;
    auto qg3dten = context.qg3dten;
    auto t3dten = context.t3dten;
    auto qv3dten = context.qv3dten;
    auto qc3dten = context.qc3dten;
    auto qni3dten = context.qni3dten;
    auto qr3dten = context.qr3dten;
    auto ns3dten = context.ns3dten;
    auto nr3dten = context.nr3dten;
    auto nc3d = context.nc3d;
    auto nc3dten = context.nc3dten;
    auto lamc = context.lamc;
    auto lams = context.lams;
    auto lamr = context.lamr;
    auto lamg = context.lamg;
    auto n0s = context.n0s;
    auto n0rr = context.n0rr;
    auto n0g = context.n0g;
    auto pgam = context.pgam;
    auto nsubc = context.nsubc;
    auto nsubr = context.nsubr;
    auto pre = context.pre;
    auto pra = context.pra;
    auto prc = context.prc;
    auto pcc = context.pcc;
    auto npra = context.npra;
    auto nragg = context.nragg;
    auto nprc = context.nprc;
    auto nprc1 = context.nprc1;
    auto pracs = context.pracs;
    auto npracs = context.npracs;
    auto psmlt = context.psmlt;
    auto evpms = context.evpms;
    auto nsmlts = context.nsmlts;
    auto nsmltr = context.nsmltr;
    auto pracg = context.pracg;
    auto evpmg = context.evpmg;
    auto pgmlt = context.pgmlt;
    auto npracg = context.npracg;
    auto ngmltg = context.ngmltg;
    auto ngmltr = context.ngmltr;
    auto kap = context.kap;
    auto qvs = context.qvs;
    auto qvqvs = context.qvqvs;
    auto dv = context.dv;
    auto xxls = context.xxls;
    auto xxlv = context.xxlv;
    auto cpm = context.cpm;
    auto mu = context.mu;
    auto sc = context.sc;
    auto xlf = context.xlf;
    auto rho = context.rho;
    auto ab = context.ab;
    auto arn = context.arn;
    auto asn = context.asn;
    auto agn = context.agn;
    auto skip_micro = context.skip_micro;
    auto t_ge_273 = context.t_ge_273;
    auto no_cirg = context.no_cirg;

      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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
      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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
      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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
      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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
  }

} // namespace modules

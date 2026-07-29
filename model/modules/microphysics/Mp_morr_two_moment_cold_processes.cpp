#include "Mp_morr_two_moment_context.h"

namespace modules {

  void Mp_morr_two_moment::run_two_mom_cold_processes(RunContext const &context) {
    using yakl::parallel_for_F;
    using yakl::SimpleBounds_F;
    auto qc3d = context.qc3d;
    auto qi3d = context.qi3d;
    auto qni3d = context.qni3d;
    auto qr3d = context.qr3d;
    auto ni3d = context.ni3d;
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
    auto bg = context.bg;
    auto rhog = context.rhog;
    auto mi0 = context.mi0;
    auto mmult = context.mmult;
    auto lammini = context.lammini;
    auto cons1 = context.cons1;
    auto cons2 = context.cons2;
    auto cons3 = context.cons3;
    auto cons4 = context.cons4;
    auto cons5 = context.cons5;
    auto cons6 = context.cons6;
    auto cons7 = context.cons7;
    auto cons8 = context.cons8;
    auto cons12 = context.cons12;
    auto cons13 = context.cons13;
    auto cons14 = context.cons14;
    auto cons15 = context.cons15;
    auto cons16 = context.cons16;
    auto cons17 = context.cons17;
    auto cons18 = context.cons18;
    auto cons19 = context.cons19;
    auto cons20 = context.cons20;
    auto cons21 = context.cons21;
    auto cons22 = context.cons22;
    auto cons23 = context.cons23;
    auto cons24 = context.cons24;
    auto cons25 = context.cons25;
    auto cons26 = context.cons26;
    auto cons29 = context.cons29;
    auto cons31 = context.cons31;
    auto cons32 = context.cons32;
    auto cons37 = context.cons37;
    auto cons38 = context.cons38;
    auto cons39 = context.cons39;
    auto cons40 = context.cons40;
    auto cons41 = context.cons41;
    auto ng3dten = context.ng3dten;
    auto t3dten = context.t3dten;
    auto qv3dten = context.qv3dten;
    auto qc3dten = context.qc3dten;
    auto ns3dten = context.ns3dten;
    auto nr3dten = context.nr3dten;
    auto nc3d = context.nc3d;
    auto lamc = context.lamc;
    auto lami = context.lami;
    auto lams = context.lams;
    auto lamr = context.lamr;
    auto lamg = context.lamg;
    auto cdist1 = context.cdist1;
    auto n0i = context.n0i;
    auto n0s = context.n0s;
    auto n0rr = context.n0rr;
    auto n0g = context.n0g;
    auto pgam = context.pgam;
    auto nsubc = context.nsubc;
    auto nsubi = context.nsubi;
    auto nsubs = context.nsubs;
    auto nsubr = context.nsubr;
    auto prd = context.prd;
    auto pre = context.pre;
    auto prds = context.prds;
    auto nnuccc = context.nnuccc;
    auto mnuccc = context.mnuccc;
    auto pra = context.pra;
    auto prc = context.prc;
    auto pcc = context.pcc;
    auto nnuccd = context.nnuccd;
    auto mnuccd = context.mnuccd;
    auto mnuccr = context.mnuccr;
    auto nnuccr = context.nnuccr;
    auto npra = context.npra;
    auto nragg = context.nragg;
    auto nsagg = context.nsagg;
    auto nprc = context.nprc;
    auto nprc1 = context.nprc1;
    auto prai = context.prai;
    auto prci = context.prci;
    auto psacws = context.psacws;
    auto npsacws = context.npsacws;
    auto psacwi = context.psacwi;
    auto npsacwi = context.npsacwi;
    auto nprci = context.nprci;
    auto nprai = context.nprai;
    auto nmults = context.nmults;
    auto nmultr = context.nmultr;
    auto qmults = context.qmults;
    auto qmultr = context.qmultr;
    auto pracs = context.pracs;
    auto npracs = context.npracs;
    auto psmlt = context.psmlt;
    auto evpms = context.evpms;
    auto nsmlts = context.nsmlts;
    auto nsmltr = context.nsmltr;
    auto piacr = context.piacr;
    auto niacr = context.niacr;
    auto praci = context.praci;
    auto piacrs = context.piacrs;
    auto niacrs = context.niacrs;
    auto pracis = context.pracis;
    auto eprd = context.eprd;
    auto eprds = context.eprds;
    auto pracg = context.pracg;
    auto psacwg = context.psacwg;
    auto pgsacw = context.pgsacw;
    auto pgracs = context.pgracs;
    auto prdg = context.prdg;
    auto eprdg = context.eprdg;
    auto evpmg = context.evpmg;
    auto pgmlt = context.pgmlt;
    auto npracg = context.npracg;
    auto npsacwg = context.npsacwg;
    auto nscng = context.nscng;
    auto ngracs = context.ngracs;
    auto ngmltg = context.ngmltg;
    auto ngmltr = context.ngmltr;
    auto nsubg = context.nsubg;
    auto psacr = context.psacr;
    auto nmultg = context.nmultg;
    auto nmultrg = context.nmultrg;
    auto qmultg = context.qmultg;
    auto qmultrg = context.qmultrg;
    auto qvi = context.qvi;
    auto qvqvs = context.qvqvs;
    auto qvqvsi = context.qvqvsi;
    auto dv = context.dv;
    auto xxlv = context.xxlv;
    auto cpm = context.cpm;
    auto mu = context.mu;
    auto rho = context.rho;
    auto abi = context.abi;
    auto dap = context.dap;
    auto ain = context.ain;
    auto arn = context.arn;
    auto asn = context.asn;
    auto agn = context.agn;
    auto skip_micro = context.skip_micro;
    auto t_ge_273 = context.t_ge_273;
    auto no_cirg = context.no_cirg;

      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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
      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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
      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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
      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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
      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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
  }

} // namespace modules

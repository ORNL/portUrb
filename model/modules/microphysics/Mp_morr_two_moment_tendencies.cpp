#include "Mp_morr_two_moment_context.h"

namespace modules {

  void Mp_morr_two_moment::run_two_mom_tendencies(RunContext const &context) {
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
    auto c2prec = context.c2prec;
    auto bg = context.bg;
    auto lammini = context.lammini;
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
    auto cons12 = context.cons12;
    auto cons26 = context.cons26;
    auto cons27 = context.cons27;
    auto cons28 = context.cons28;
    auto cons34 = context.cons34;
    auto cons35 = context.cons35;
    auto cons36 = context.cons36;
    auto ng3dten = context.ng3dten;
    auto qg3dten = context.qg3dten;
    auto t3dten = context.t3dten;
    auto qv3dten = context.qv3dten;
    auto qc3dten = context.qc3dten;
    auto qi3dten = context.qi3dten;
    auto qni3dten = context.qni3dten;
    auto qr3dten = context.qr3dten;
    auto ni3dten = context.ni3dten;
    auto ns3dten = context.ns3dten;
    auto nr3dten = context.nr3dten;
    auto nc3d = context.nc3d;
    auto nc3dten = context.nc3dten;
    auto lami = context.lami;
    auto lams = context.lams;
    auto lamr = context.lamr;
    auto lamg = context.lamg;
    auto n0i = context.n0i;
    auto n0s = context.n0s;
    auto n0rr = context.n0rr;
    auto n0g = context.n0g;
    auto pgam = context.pgam;
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
    auto qvs = context.qvs;
    auto qvi = context.qvi;
    auto dv = context.dv;
    auto xxls = context.xxls;
    auto xxlv = context.xxlv;
    auto cpm = context.cpm;
    auto mu = context.mu;
    auto sc = context.sc;
    auto xlf = context.xlf;
    auto rho = context.rho;
    auto ab = context.ab;
    auto abi = context.abi;
    auto dumi = context.dumi;
    auto dumr = context.dumr;
    auto dumfni = context.dumfni;
    auto dumg = context.dumg;
    auto dumfng = context.dumfng;
    auto fr = context.fr;
    auto fi = context.fi;
    auto fni = context.fni;
    auto fg = context.fg;
    auto fng = context.fng;
    auto dumqs = context.dumqs;
    auto dumfns = context.dumfns;
    auto fs = context.fs;
    auto fns = context.fns;
    auto dumc = context.dumc;
    auto dumfnc = context.dumfnc;
    auto fc = context.fc;
    auto fnc = context.fnc;
    auto dumfnr = context.dumfnr;
    auto fnr = context.fnr;
    auto ain = context.ain;
    auto arn = context.arn;
    auto asn = context.asn;
    auto acn = context.acn;
    auto agn = context.agn;
    auto skip_micro = context.skip_micro;
    auto t_ge_273 = context.t_ge_273;
    auto hydro_pres = context.hydro_pres;

      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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
      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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
      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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
      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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

      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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

  }

} // namespace modules

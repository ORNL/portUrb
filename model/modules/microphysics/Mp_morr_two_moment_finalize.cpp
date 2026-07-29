#include "Mp_morr_two_moment_context.h"

namespace modules {

  void Mp_morr_two_moment::run_two_mom_finalize(RunContext const &context) {
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
    auto lammini = context.lammini;
    auto cons1 = context.cons1;
    auto cons2 = context.cons2;
    auto cons12 = context.cons12;
    auto cons26 = context.cons26;
    auto ng3dten = context.ng3dten;
    auto qg3dten = context.qg3dten;
    auto effc = context.effc;
    auto effi = context.effi;
    auto effs = context.effs;
    auto effr = context.effr;
    auto effg = context.effg;
    auto t3dten = context.t3dten;
    auto qv3dten = context.qv3dten;
    auto qc3dten = context.qc3dten;
    auto qi3dten = context.qi3dten;
    auto qni3dten = context.qni3dten;
    auto qr3dten = context.qr3dten;
    auto ni3dten = context.ni3dten;
    auto ns3dten = context.ns3dten;
    auto nr3dten = context.nr3dten;
    auto qgsten = context.qgsten;
    auto qrsten = context.qrsten;
    auto qisten = context.qisten;
    auto qnisten = context.qnisten;
    auto qcsten = context.qcsten;
    auto nc3d = context.nc3d;
    auto nc3dten = context.nc3dten;
    auto lamc = context.lamc;
    auto lami = context.lami;
    auto lams = context.lams;
    auto lamr = context.lamr;
    auto lamg = context.lamg;
    auto n0i = context.n0i;
    auto n0s = context.n0s;
    auto n0rr = context.n0rr;
    auto n0g = context.n0g;
    auto pgam = context.pgam;
    auto evs = context.evs;
    auto eis = context.eis;
    auto qvs = context.qvs;
    auto qvi = context.qvi;
    auto qvqvs = context.qvqvs;
    auto qvqvsi = context.qvqvsi;
    auto xxls = context.xxls;
    auto xxlv = context.xxlv;
    auto cpm = context.cpm;
    auto xlf = context.xlf;
    auto rho = context.rho;
    auto hydro_pres = context.hydro_pres;

      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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

} // namespace modules

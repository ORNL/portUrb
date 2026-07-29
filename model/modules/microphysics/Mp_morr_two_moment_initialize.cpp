#include "Mp_morr_two_moment_context.h"

namespace modules {

  void Mp_morr_two_moment::run_two_mom_initialize(RunContext const &context) {
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
    auto precrt = context.precrt;
    auto snowrt = context.snowrt;
    auto snowprt = context.snowprt;
    auto grplprt = context.grplprt;
    auto dt = context.dt;
    auto ncol = context.ncol;
    auto nz = context.nz;
    auto qg3d = context.qg3d;
    auto ng3d = context.ng3d;
    auto qrcu1d = context.qrcu1d;
    auto qscu1d = context.qscu1d;
    auto qicu1d = context.qicu1d;
    auto c2prec = context.c2prec;
    auto ised = context.ised;
    auto ssed = context.ssed;
    auto gsed = context.gsed;
    auto rsed = context.rsed;
    auto ag = context.ag;
    auto ci = context.ci;
    auto cons1 = context.cons1;
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
    auto csed = context.csed;
    auto qgsten = context.qgsten;
    auto qrsten = context.qrsten;
    auto qisten = context.qisten;
    auto qnisten = context.qnisten;
    auto qcsten = context.qcsten;
    auto nc3d = context.nc3d;
    auto nc3dten = context.nc3dten;
    auto lami = context.lami;
    auto kap = context.kap;
    auto evs = context.evs;
    auto eis = context.eis;
    auto qvs = context.qvs;
    auto qvi = context.qvi;
    auto qvqvs = context.qvqvs;
    auto qvqvsi = context.qvqvsi;
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
    auto ain = context.ain;
    auto arn = context.arn;
    auto asn = context.asn;
    auto acn = context.acn;
    auto agn = context.agn;
    auto skip_micro = context.skip_micro;
    auto t_ge_273 = context.t_ge_273;
    auto nstep = context.nstep;
    auto hydro_pres = context.hydro_pres;

      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<1>(ncol) , KOKKOS_LAMBDA (int i) {
        hydro_pres(i) = false;
        nstep     (i) = 1;
        precrt    (i) = 0.;
        snowrt    (i) = 0.;
        snowprt   (i) = 0.;
        grplprt   (i) = 0.;
      });
      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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
  }

} // namespace modules

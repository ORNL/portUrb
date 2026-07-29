#include "Mp_morr_two_moment_context.h"

namespace modules {

  void Mp_morr_two_moment::run_two_mom_sedimentation(RunContext const &context) {
    using yakl::parallel_for_F;
    using yakl::SimpleBounds_F;
    auto dzq = context.dzq;
    auto precrt = context.precrt;
    auto snowrt = context.snowrt;
    auto snowprt = context.snowprt;
    auto grplprt = context.grplprt;
    auto dt = context.dt;
    auto ncol = context.ncol;
    auto nz = context.nz;
    auto ised = context.ised;
    auto ssed = context.ssed;
    auto gsed = context.gsed;
    auto rsed = context.rsed;
    auto ng3dten = context.ng3dten;
    auto ni3dten = context.ni3dten;
    auto ns3dten = context.ns3dten;
    auto nr3dten = context.nr3dten;
    auto csed = context.csed;
    auto qgsten = context.qgsten;
    auto qrsten = context.qrsten;
    auto qisten = context.qisten;
    auto qnisten = context.qnisten;
    auto qcsten = context.qcsten;
    auto nc3dten = context.nc3dten;
    auto rho = context.rho;
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
    auto faloutr = context.faloutr;
    auto falouti = context.falouti;
    auto faloutni = context.faloutni;
    auto dumqs = context.dumqs;
    auto dumfns = context.dumfns;
    auto fs = context.fs;
    auto fns = context.fns;
    auto falouts = context.falouts;
    auto faloutns = context.faloutns;
    auto faloutg = context.faloutg;
    auto faloutng = context.faloutng;
    auto dumc = context.dumc;
    auto dumfnc = context.dumfnc;
    auto fc = context.fc;
    auto faloutc = context.faloutc;
    auto faloutnc = context.faloutnc;
    auto fnc = context.fnc;
    auto dumfnr = context.dumfnr;
    auto faloutnr = context.faloutnr;
    auto fnr = context.fnr;
    auto nstep = context.nstep;
    auto hydro_pres = context.hydro_pres;

      parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<1>(ncol) , KOKKOS_LAMBDA (int i) {
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
        parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
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
        parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<1>(ncol) , KOKKOS_LAMBDA (int i) {
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
        parallel_for_F( YAKL_AUTO_LABEL() , SimpleBounds_F<2>(nz-1,ncol) , KOKKOS_LAMBDA (int k, int i) {
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


  }

} // namespace modules

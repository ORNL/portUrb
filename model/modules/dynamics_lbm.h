
#pragma once

#include "main_header.h"
#include "coupler.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "WenoLimiter.h"
#include <random>
#include <sstream>
#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Dense>

namespace modules {

  struct Dynamics_Euler_LBM {

    int   static constexpr hs  = 1;
    float static constexpr cs2 = 1./3.;

    real compute_time_step( core::Coupler const &coupler ) const {
      auto dx    = coupler.get_dx();    // grid spacing
      return 0.1*dx/coupler.get_option<real>("dycore_max_wind",100);
    }


    void time_step(core::Coupler &coupler, real dt) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx    = coupler.get_nx();    // Proces-local number of cells
      auto ny    = coupler.get_ny();    // Proces-local number of cells
      auto nz    = coupler.get_nz();    // Total vertical cells
      auto nx_glob = coupler.get_nx_glob();
      auto ny_glob = coupler.get_ny_glob();
      auto i_beg   = coupler.get_i_beg();
      auto j_beg   = coupler.get_j_beg();
      auto dx    = coupler.get_dx();    // grid spacing
      auto nq    = coupler.get_option<int>("dycore_nq" ,27);
      auto ord   = coupler.get_option<int>("dycore_ord",3 );
      auto &dm   = coupler.get_data_manager_readwrite();  // Grab read-only data manager
      auto f     = dm.get<float     ,4>("dycore_lbm_f"    );
      auto c     = dm.get<int       ,2>("dycore_lbm_c"    );
      auto wt    = dm.get<float     ,1>("dycore_lbm_w"    );
      auto rho_d = dm.get<real      ,3>("density_dry"     );
      auto uvel  = dm.get<real      ,3>("uvel"            );
      auto vvel  = dm.get<real      ,3>("vvel"            );
      auto wvel  = dm.get<real      ,3>("wvel"            );
      auto opp_x = dm.get<int       ,1>("dycore_lbm_opp_x");
      auto opp_y = dm.get<int       ,1>("dycore_lbm_opp_y");
      auto opp_z = dm.get<int       ,1>("dycore_lbm_opp_z");
      auto opp   = dm.get<int       ,1>("dycore_lbm_opp"  );
      auto imm   = dm.get<real      ,3>("immersed_proportion_halos");

      // Construct full f
      float4d feq("feq",nq,nz,ny,nx);
      compute_equ( coupler , feq , rho_d , uvel , vvel , wvel , c , wt , dx , dt , ord );
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nq,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        f(l,k,j,i) += feq(l,k,j,i);
      });

      // Collision
      auto fcoll = collision_srt( coupler , f , feq , 0.505 );

      // Exchange halos, and apply boundary conditions
      coupler.halo_exchange( fcoll , hs );
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nq,nz,ny+2*hs,nx+2*hs) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        fcoll(l,hs+nz,j,i) = fcoll(l,hs+0   ,j,i);
        fcoll(l,0    ,j,i) = fcoll(l,hs+nz-1,j,i);
        if (l==0) {
          imm(hs+nz,j,i) = 1;
          imm(0    ,j,i) = 1;
        }
      });

      // Stream
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nq,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (imm(hs+k,hs+j,hs+i) > 0.5) { return; }
        int l0 = l;
        int i0 = i-c(l,0);
        int j0 = j-c(l,1);
        int k0 = k-c(l,2);
        // if (imm(hs+k0,hs+j0,hs+i0) > 0.5) {
        //   if (c(l,0) == -1) { i0--; l0 = opp_x(l0); }
        //   if (c(l,0) ==  1) { i0++; l0 = opp_x(l0); }
        // }
        // if (imm(hs+k0,hs+j0,hs+i0) > 0.5) {
        //   if (c(l,1) == -1) { j0--; l0 = opp_y(l0); }
        //   if (c(l,1) ==  1) { j0++; l0 = opp_y(l0); }
        // }
        // if (imm(hs+k0,hs+j0,hs+i0) > 0.5) {
        //   if (c(l,2) == -1) { k0--; l0 = opp_z(l0); }
        //   if (c(l,2) ==  1) { k0++; l0 = opp_z(l0); }
        // }
        if (imm(hs+k0, hs+j0, hs+i0) > 0.5) {
          i0 = i;
          j0 = j;
          k0 = k;
          l0 = opp(l);
        }
        f(l,k,j,i) = fcoll(l0,hs+k0,hs+j0,hs+i0);
      });

      compute_rho_vel( coupler , f   , rho_d , uvel , vvel , wvel , c      , dx , dt , ord );
      compute_equ    ( coupler , feq , rho_d , uvel , vvel , wvel , c , wt , dx , dt , ord );
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nq,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        f(l,k,j,i) -= feq(l,k,j,i);
      });
    }


    void init(core::Coupler &coupler) const {
      using yakl::intrinsics::sum;
      using yakl::componentwise::operator/;
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx  = coupler.get_nx();
      auto ny  = coupler.get_ny();
      auto nz  = coupler.get_nz();
      auto nq  = coupler.get_option<int>("dycore_nq",27);
      if (nq != 19 && nq != 27) Kokkos::abort("ERROR: nq must be 19 or 27");
      auto &dm = coupler.get_data_manager_readwrite();

      dm.register_and_allocate<float>("dycore_lbm_f","",{nq,nz,ny,nx});
      dm.get<float,4>("dycore_lbm_f") = 0;

      dm.register_and_allocate<int  >("dycore_lbm_c","",{nq,3});
      intHost2d c("lbm_c",nq,3);
      c(0 ,0) =  0;    c(0 ,1) =  0;    c(0 ,2) =  0;
      c(1 ,0) = +1;    c(1 ,1) =  0;    c(1 ,2) =  0;
      c(2 ,0) = -1;    c(2 ,1) =  0;    c(2 ,2) =  0;
      c(3 ,0) =  0;    c(3 ,1) = +1;    c(3 ,2) =  0;
      c(4 ,0) =  0;    c(4 ,1) = -1;    c(4 ,2) =  0;
      c(5 ,0) =  0;    c(5 ,1) =  0;    c(5 ,2) = +1;
      c(6 ,0) =  0;    c(6 ,1) =  0;    c(6 ,2) = -1;
      c(7 ,0) = +1;    c(7 ,1) = +1;    c(7 ,2) =  0;
      c(8 ,0) = -1;    c(8 ,1) = -1;    c(8 ,2) =  0;
      c(9 ,0) = +1;    c(9 ,1) =  0;    c(9 ,2) = +1;
      c(10,0) = -1;    c(10,1) =  0;    c(10,2) = -1;
      c(11,0) =  0;    c(11,1) = +1;    c(11,2) = +1;
      c(12,0) =  0;    c(12,1) = -1;    c(12,2) = -1;
      c(13,0) = +1;    c(13,1) = -1;    c(13,2) =  0;
      c(14,0) = -1;    c(14,1) = +1;    c(14,2) =  0;
      c(15,0) = +1;    c(15,1) =  0;    c(15,2) = -1;
      c(16,0) = -1;    c(16,1) =  0;    c(16,2) = +1;
      c(17,0) =  0;    c(17,1) = +1;    c(17,2) = -1;
      c(18,0) =  0;    c(18,1) = -1;    c(18,2) = +1;
      if (nq == 27) {
        c(19,0) = +1;    c(19,1) = +1;    c(19,2) = +1;
        c(20,0) = -1;    c(20,1) = -1;    c(20,2) = -1;
        c(21,0) = +1;    c(21,1) = +1;    c(21,2) = -1;
        c(22,0) = -1;    c(22,1) = -1;    c(22,2) = +1;
        c(23,0) = +1;    c(23,1) = -1;    c(23,2) = +1;
        c(24,0) = -1;    c(24,1) = +1;    c(24,2) = -1;
        c(25,0) = -1;    c(25,1) = +1;    c(25,2) = +1;
        c(26,0) = +1;    c(26,1) = -1;    c(26,2) = -1;
      }
      c.deep_copy_to(dm.get<int,2>("dycore_lbm_c"));

      dm.register_and_allocate<float>("dycore_lbm_w","",{nq});
      floatHost1d w("lbm_w",nq);
      if (nq == 19) {
        w(0) = 1./3.;
        for (int i = 1; i <= 6 ; i++) { w(i) = 1./18.; }
        for (int i = 7; i <= 18; i++) { w(i) = 1./36.; }
      } else if (nq == 27) {
        w(0) = 8./27.;
        for (int i = 1 ; i <= 6 ; i++) { w(i) = 2./27. ; }
        for (int i = 7 ; i <= 18; i++) { w(i) = 1./54. ; }
        for (int i = 19; i <= 26; i++) { w(i) = 1./216.; }
      }
      w.deep_copy_to(dm.get<float,1>("dycore_lbm_w"));

      dm.register_and_allocate<int  >("dycore_lbm_opp_x","",{nq});
      dm.register_and_allocate<int  >("dycore_lbm_opp_y","",{nq});
      dm.register_and_allocate<int  >("dycore_lbm_opp_z","",{nq});
      dm.register_and_allocate<int  >("dycore_lbm_opp"  ,"",{nq});
      intHost1d opp_x("opp_x",nq); opp_x = -1;
      intHost1d opp_y("opp_z",nq); opp_y = -1;
      intHost1d opp_z("opp_z",nq); opp_z = -1;
      intHost1d opp  ("opp"  ,nq); opp   = -1;
      for (int i=0; i < nq; i++) {
        for (int ii=0; ii < nq; ii++) {
          if ( c(i,0) == -c(ii,0) && c(i,1) ==  c(ii,1) && c(i,2) ==  c(ii,2) ) opp_x(i) = ii;
          if ( c(i,0) ==  c(ii,0) && c(i,1) == -c(ii,1) && c(i,2) ==  c(ii,2) ) opp_y(i) = ii;
          if ( c(i,0) ==  c(ii,0) && c(i,1) ==  c(ii,1) && c(i,2) == -c(ii,2) ) opp_z(i) = ii;
          if ( c(i,0) == -c(ii,0) && c(i,1) == -c(ii,1) && c(i,2) == -c(ii,2) ) opp  (i) = ii;
        }
      }
      opp_x.deep_copy_to(dm.get<int,1>("dycore_lbm_opp_x"));
      opp_y.deep_copy_to(dm.get<int,1>("dycore_lbm_opp_y"));
      opp_z.deep_copy_to(dm.get<int,1>("dycore_lbm_opp_z"));
      opp  .deep_copy_to(dm.get<int,1>("dycore_lbm_opp"  ));
    }


    void compute_equ( core::Coupler &coupler ,
                      float4d const & feq  ,
                      real3d  const & rho  ,
                      real3d  const & uvel ,
                      real3d  const & vvel ,
                      real3d  const & wvel ,
                      int2d   const & c    ,
                      float1d const & wt   ,
                      float            dx   ,
                      float            dt   ,
                      int ord              ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      if (ord != 2 && ord != 3) Kokkos::abort("ERROR: ord must be 2 or 3");
      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      auto nq = wt.size();
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nq,nz,ny,nx) , KOKKOS_LAMBDA (int l, int k, int j, int i) {
        float r   = rho (k,j,i);
        float u   = uvel(k,j,i)*dt/dx;
        float v   = vvel(k,j,i)*dt/dx;
        float w   = wvel(k,j,i)*dt/dx;
        float u2  = u*u+v*v+w*w;
        float cdu = c(l,0)*u+c(l,1)*v+c(l,2)*w;
        feq(l,k,j,i) = 1 + cdu/cs2 + cdu*cdu/(2*cs2*cs2) - u2/(2*cs2);
        if (ord == 3) feq(l,k,j,i) += cdu*cdu*cdu/(6*cs2*cs2*cs2) - cdu*u2/(2*cs2*cs2);
        feq(l,k,j,i) *= r*wt(l);
      });
    }


    void compute_rho_vel( core::Coupler &coupler ,
                          float4d const & f    ,
                          real3d  const & rho  ,
                          real3d  const & uvel ,
                          real3d  const & vvel ,
                          real3d  const & wvel ,
                          int2d   const & c    ,
                          float            dx   ,
                          float            dt   ,
                          int ord              ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      if (ord != 2 && ord != 3) Kokkos::abort("ERROR: ord must be 2 or 3");
      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      auto nq = f.extent(0);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        float r  = 0;
        float ru = 0;
        float rv = 0;
        float rw = 0;
        for (int l=0; l < nq; l++) {
          r  += f(l,k,j,i);
          ru += f(l,k,j,i)*c(l,0);
          rv += f(l,k,j,i)*c(l,1);
          rw += f(l,k,j,i)*c(l,2);
        }
        rho (k,j,i) = r;
        uvel(k,j,i) = (ru/r)*dx/dt;
        vvel(k,j,i) = (rv/r)*dx/dt;
        wvel(k,j,i) = (rw/r)*dx/dt;
      });
    }


    float4d collision_srt( core::Coupler const & coupler ,
                           float4d       const & f       ,
                           float4d       const & feq     ,
                           float                 tau     ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      auto nq = f.extent(0);
      float4d fcoll("fcoll",nq,nz+2*hs,ny+2*hs,nx+2*hs);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nq,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        fcoll(l,hs+k,hs+j,hs+i) = f(l,k,j,i) - (f(l,k,j,i)-feq(l,k,j,i))/tau;
      });
      return fcoll;
    }


    float4d collision_trt( core::Coupler const & coupler ,
                           float4d       const & f       ,
                           float4d       const & feq     ,
                           float                 tau_p   ,
                           float                 Lambda  ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx    = coupler.get_nx();
      auto ny    = coupler.get_ny();
      auto nz    = coupler.get_nz();
      auto nq    = f.extent(0);
      auto &dm   = coupler.get_data_manager_readonly();
      auto opp   = dm.get<int  const,1>("dycore_lbm_opp"       );
      float tau_m = Lambda/(tau_p-0.5)+0.5;
      float4d fcoll("fcoll",nq,nz+2*hs,ny+2*hs,nx+2*hs);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nq,nz,ny,nx) ,
                                        KOKKOS_LAMBDA (int l, int k, int j, int i) {
        int lo = opp(l);
        float f_p   = 0.5*(f  (l,k,j,i)+f  (lo,k,j,i));
        float f_m   = 0.5*(f  (l,k,j,i)-f  (lo,k,j,i));
        float feq_p = 0.5*(feq(l,k,j,i)+feq(lo,k,j,i));
        float feq_m = 0.5*(feq(l,k,j,i)-feq(lo,k,j,i));
        fcoll(l,hs+k,hs+j,hs+i) = f(l,k,j,i) - (f_p-feq_p)/tau_p - (f_m-feq_m)/tau_m;
      });
      return fcoll;
    }

  };

}




#include "GMRES.h"
#include "YAKL_pnetcdf.h"

typedef double real;

typedef yakl::Array<real *   > real1d;
typedef yakl::Array<real **  > real2d;
typedef yakl::Array<real *** > real3d;
typedef yakl::Array<real ****> real4d;

typedef yakl::Array<real *   ,Kokkos::HostSpace> realHost1d;

int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  Kokkos::initialize();
  yakl::init();
  {
    int nx = 100;
    real xlen = 1.;
    real dx = xlen/nx;
    real dt = 0.6*dx;
    real2d soln("soln",2,nx);
    int constexpr idP = 0;
    int constexpr idU = 1;
    auto pp = soln.slice<1>(idP,yakl::COLON);
    auto ru = soln.slice<1>(idU,yakl::COLON);
    yakl::parallel_for( YAKL_AUTO_LABEL(), nx, KOKKOS_LAMBDA (int i) {
      real x = (i+0.5)*dx;
      /////////// TEST 1
      ru(i) = 0;
      pp(i) = i==49 ? 1 : 0;
      /////////// TEST 2
      // ru(i) = 0.1*std::sin(2*M_PI*x) + 0.03*std::sin(6*M_PI*x);
      // pp(i) = 0;
      /////////// TEST 3
      // ru(i) = 0.05 + 0.1*std::sin(2*M_PI*x);
      // pp(i) = 0;
      /////////// TEST 4
      // ru(i) = 0.08*std::sin(2*M_PI*x) + 0.04*std::cos(8*M_PI*x) + 0.02*std::sin(22*M_PI*x);
      // pp(i) = 0;
    });
    auto soln_init_h = soln.createHostCopy();
    auto forcing   = soln.createDeviceObject();
    forcing.slice<1>(idP,yakl::COLON) = 0;
    ru.deep_copy_to(forcing.slice<1>(idU,yakl::COLON));
    YaklRestartedGMRES<real> gmres;
    YaklRestartedGMRES<real>::Options opts;
    opts.verbose = true;
    opts.rel_tol = 1.e-14;
    opts.restart = 50;
    opts.max_iters = 1000;
    auto compute_Ax = [=] ( yakl::Array<real *> const & x_in ,  
                            yakl::Array<real *> const & Ax_out ,
                            MPI_Comm comm ) {
      auto x_in_2d   = x_in  .reshape(2,nx);
      auto Ax_out_2d = Ax_out.reshape(2,nx);
      auto pp        = x_in_2d  .slice<1>(idP,yakl::COLON);
      auto ru        = x_in_2d  .slice<1>(idU,yakl::COLON);
      auto lhs_pp    = Ax_out_2d.slice<1>(idP,yakl::COLON);
      auto lhs_ru    = Ax_out_2d.slice<1>(idU,yakl::COLON);
      yakl::parallel_for( YAKL_AUTO_LABEL() , nx , KOKKOS_LAMBDA (int i) {
        int im1 = i-1;    if (im1 < 0   ) im1 += nx;
        int ip1 = i+1;    if (ip1 > nx-1) ip1 -= nx;
        real constexpr cs = 100;
        real constexpr cs2 = cs*cs;
        // Left interface
        real pp_L = pp(im1);    real pp_R = pp(i);
        real ru_L = ru(im1);    real ru_R = ru(i);
        real cs2_ru_L_upw = 0.5*(cs2*ru_L + cs2*ru_R - cs*(pp_R - pp_L));
        real pp_L_upw     = 0.5*(    pp_L +     pp_R - cs*(ru_R - ru_L));
        real ru_L_upw = cs2_ru_L_upw / cs2;
        // Right interface
        pp_L = pp(i);      pp_R = pp(ip1);
        ru_L = ru(i);      ru_R = ru(ip1);
        real cs2_ru_R_upw = 0.5*(cs2*ru_L + cs2*ru_R - cs*(pp_R - pp_L));
        real pp_R_upw     = 0.5*(    pp_L +     pp_R - cs*(ru_R - ru_L));
        real ru_R_upw = cs2_ru_R_upw / cs2;
        // Compute LHS
        lhs_pp(i) =            (ru_R_upw - ru_L_upw) / dx;
        lhs_ru(i) = ru(i) + dt*(pp_R_upw - pp_L_upw) / dx;
        // lhs_pp(i) =            (ru(ip1) - ru(im1)) / (2*dx);
        // lhs_ru(i) = ru(i) + dt*(pp(ip1) - pp(im1)) / (2*dx);
      });
    };
    auto result = gmres.solve( soln.collapse() , forcing.collapse() , compute_Ax , opts );
    auto soln_final_h = soln.createHostCopy();
    std::cout << std::setw(15) << "x" << "  "
              << std::setw(15) << "Init pp"  << "  "
              << std::setw(15) << "Init ru"  << "  "
              << std::setw(15) << "Final pp" << "  "
              << std::setw(15) << "Final ru" << std::endl;
    for (int i=0; i < nx; i++) {
      std::cout << std::setprecision(8) << std::scientific << std::setw(15) << (i+0.5)*dx          << "  "
                << std::setprecision(8) << std::scientific << std::setw(15) << soln_init_h (idP,i) << "  "
                << std::setprecision(8) << std::scientific << std::setw(15) << soln_init_h (idU,i) << "  "
                << std::setprecision(8) << std::scientific << std::setw(15) << soln_final_h(idP,i) << "  "
                << std::setprecision(8) << std::scientific << std::setw(15) << soln_final_h(idU,i) << std::endl;
    }
  }



  {
    int nx = 100;
    int ny = 100;
    int nz = 100;
    real xlen = 1.;
    real ylen = 1.;
    real zlen = 1.;
    real dx = xlen/nx;
    real dy = ylen/ny;
    real dz = zlen/nz;
    real dt = 0.6*dx;
    real4d soln("soln",4,nz,ny,nx);
    int constexpr idP = 0;
    int constexpr idU = 1;
    int constexpr idV = 2;
    int constexpr idW = 3;
    auto forcing = soln.createDeviceObject();

    yakl::parallel_for( YAKL_AUTO_LABEL(), yakl::SimpleBounds<3>(nz,ny,nx),
                                           KOKKOS_LAMBDA (int k, int j, int i) {
      real x = (i+0.5) * dx;
      real y = (j+0.5) * dy;
      real z = (k+0.5) * dz;

      real constexpr pi    = M_PI;
      real constexpr a1    = 0.01;
      real constexpr a2    = 0.004;
      real constexpr b1    = 0.01;
      real constexpr b2    = 0.004;

      // Compressible part: grad(phi)
      real mx_comp =  a1*2*pi*std::cos(2*pi*x)*std::cos(2*pi*y)*std::sin(2*pi*z) -
                      a2*4*pi*std::sin(4*pi*x)*std::sin(2*pi*y)*std::cos(6*pi*z);
      real my_comp = -a1*2*pi*std::sin(2*pi*x)*std::sin(2*pi*y)*std::sin(2*pi*z) +
                      a2*2*pi*std::cos(4*pi*x)*std::cos(2*pi*y)*std::cos(6*pi*z);
      real mz_comp =  a1*2*pi*std::sin(2*pi*x)*std::cos(2*pi*y)*std::cos(2*pi*z) -
                      a2*6*pi*std::cos(4*pi*x)*std::sin(2*pi*y)*std::sin(6*pi*z);

      // Solenoidal part: curl(0,0,psi)
      real mx_sol =  b1*2*pi*std::sin(2*pi*x)*std::cos(2*pi*y) -
                     b2*2*pi*std::cos(4*pi*x)*std::sin(2*pi*y)*std::sin(2*pi*z);
      real my_sol = -b1*2*pi*std::cos(2*pi*x)*std::sin(2*pi*y) +
                     b2*4*pi*std::sin(4*pi*x)*std::cos(2*pi*y)*std::sin(2*pi*z);
      real mz_sol = 0;

      forcing(idP,k,j,i) = 0;
      forcing(idU,k,j,i) = mx_comp + mx_sol;
      forcing(idV,k,j,i) = my_comp + my_sol;
      forcing(idW,k,j,i) = mz_comp + mz_sol;

      soln   (idP,k,j,i) = 0;
      soln   (idU,k,j,i) = 0;
      soln   (idV,k,j,i) = 0;
      soln   (idW,k,j,i) = 0;
    });
    auto soln_init_h = soln.createHostCopy();
    YaklRestartedGMRES<real> gmres;
    YaklRestartedGMRES<real>::Options opts;
    opts.verbose = true;
    opts.rel_tol = 1.e-14;
    opts.restart = 50;
    opts.max_iters = 1000;
    auto compute_Ax = [=] ( yakl::Array<real *> const & x_in ,
                            yakl::Array<real *> const & Ax_out ,
                            MPI_Comm comm ) {
      auto x_in_4d   = x_in  .reshape(4,nz,ny,nx);
      auto Ax_out_4d = Ax_out.reshape(4,nz,ny,nx);
      using yakl::COLON;
      auto pp     = x_in_4d  .slice<3>(idP,COLON,COLON,COLON);
      auto ru     = x_in_4d  .slice<3>(idU,COLON,COLON,COLON);
      auto rv     = x_in_4d  .slice<3>(idV,COLON,COLON,COLON);
      auto rw     = x_in_4d  .slice<3>(idW,COLON,COLON,COLON);
      auto lhs_pp = Ax_out_4d.slice<3>(idP,COLON,COLON,COLON);
      auto lhs_ru = Ax_out_4d.slice<3>(idU,COLON,COLON,COLON);
      auto lhs_rv = Ax_out_4d.slice<3>(idV,COLON,COLON,COLON);
      auto lhs_rw = Ax_out_4d.slice<3>(idW,COLON,COLON,COLON);
      yakl::parallel_for( YAKL_AUTO_LABEL() , yakl::SimpleBounds<3>(nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int k, int j, int i) {
        ///////////////////
        // East-West fluxes
        ///////////////////
        int im1 = i-1;    if (im1 < 0   ) im1 += nx;
        int ip1 = i+1;    if (ip1 > nx-1) ip1 -= nx;
        real constexpr cs = 100;
        real constexpr cs2 = cs*cs;
        // West interface
        real pp_L = pp(k,j,im1);    real pp_R = pp(k,j,i);
        real ru_L = ru(k,j,im1);    real ru_R = ru(k,j,i);
        real cs2_ru_west_upw = 0.5*(cs2*ru_L + cs2*ru_R - cs*(pp_R - pp_L));
        real pp_west_upw     = 0.5*(    pp_L +     pp_R - cs*(ru_R - ru_L));
        real ru_west_upw = cs2_ru_west_upw / cs2;
        // East interface
        pp_L = pp(k,j,i);      pp_R = pp(k,j,ip1);
        ru_L = ru(k,j,i);      ru_R = ru(k,j,ip1);
        real cs2_ru_east_upw = 0.5*(cs2*ru_L + cs2*ru_R - cs*(pp_R - pp_L));
        real pp_east_upw     = 0.5*(    pp_L +     pp_R - cs*(ru_R - ru_L));
        real ru_east_upw = cs2_ru_east_upw / cs2;
        /////////////////////
        // North-South fluxes
        /////////////////////
        int jm1 = j-1;    if (jm1 < 0   ) jm1 += ny;
        int jp1 = j+1;    if (jp1 > ny-1) jp1 -= ny;
        // South interface
             pp_L = pp(k,jm1,i);         pp_R = pp(k,j,i);
        real rv_L = rv(k,jm1,i);    real rv_R = rv(k,j,i);
        real cs2_rv_south_upw = 0.5*(cs2*rv_L + cs2*rv_R - cs*(pp_R - pp_L));
        real pp_south_upw     = 0.5*(    pp_L +     pp_R - cs*(rv_R - rv_L));
        real rv_south_upw = cs2_rv_south_upw / cs2;
        // North interface
        pp_L = pp(k,j,i);      pp_R = pp(k,jp1,i);
        rv_L = rv(k,j,i);      rv_R = rv(k,jp1,i);
        real cs2_rv_north_upw = 0.5*(cs2*rv_L + cs2*rv_R - cs*(pp_R - pp_L));
        real pp_north_upw     = 0.5*(    pp_L +     pp_R - cs*(rv_R - rv_L));
        real rv_north_upw = cs2_rv_north_upw / cs2;
        /////////////////
        // Up-Down fluxes
        /////////////////
        int km1 = k-1;    if (km1 < 0   ) km1 += nz;
        int kp1 = k+1;    if (kp1 > nz-1) kp1 -= nz;
        // Down interface
             pp_L = pp(km1,j,i);         pp_R = pp(k,j,i);
        real rw_L = rw(km1,j,i);    real rw_R = rw(k,j,i);
        real cs2_rw_down_upw = 0.5*(cs2*rw_L + cs2*rw_R - cs*(pp_R - pp_L));
        real pp_down_upw     = 0.5*(    pp_L +     pp_R - cs*(rw_R - rw_L));
        real rw_down_upw = cs2_rw_down_upw / cs2;
        // Up interface
        pp_L = pp(k,j,i);      pp_R = pp(kp1,j,i);
        rw_L = rw(k,j,i);      rw_R = rw(kp1,j,i);
        real cs2_rw_up_upw = 0.5*(cs2*rw_L + cs2*rw_R - cs*(pp_R - pp_L));
        real pp_up_upw     = 0.5*(    pp_L +     pp_R - cs*(rw_R - rw_L));
        real rw_up_upw = cs2_rw_up_upw / cs2;

        // Update LHS with North-South fluxes
        lhs_pp(k,j,i) = (ru_east_upw  - ru_west_upw ) / dx +
                        (rv_north_upw - rv_south_upw) / dy +
                        (rw_up_upw    - rw_down_upw ) / dz;
        lhs_ru(k,j,i) = ru(k,j,i) + dt*(pp_east_upw  - pp_west_upw ) / dx;
        lhs_rv(k,j,i) = rv(k,j,i) + dt*(pp_north_upw - pp_south_upw) / dy;
        lhs_rw(k,j,i) = rw(k,j,i) + dt*(pp_up_upw    - pp_down_upw ) / dz;
      });
    };
    auto result = gmres.solve( soln.collapse() , forcing.collapse() , compute_Ax , opts );
    auto soln_final_h = soln.createHostCopy();
    realHost1d x_host("x_host",nx);
    realHost1d y_host("y_host",ny);
    realHost1d z_host("z_host",nz);
    for (int i=0; i<nx; i++) { x_host(i) = (i+0.5)*dx; }
    for (int j=0; j<ny; j++) { y_host(j) = (j+0.5)*dy; }
    for (int k=0; k<nz; k++) { z_host(k) = (k+0.5)*dz; }
    yakl::SimplePNetCDF nc;
    nc.create("periodic3d.nc");
    nc.create_dim("x",(MPI_Offset)nx);
    nc.create_dim("y",(MPI_Offset)ny);
    nc.create_dim("z",(MPI_Offset)nz);
    nc.create_var<real>("x",{"x"});
    nc.create_var<real>("y",{"y"});
    nc.create_var<real>("z",{"z"});
    nc.create_var<real>("pp_init" ,{"z","y","x"});
    nc.create_var<real>("ru_init" ,{"z","y","x"});
    nc.create_var<real>("rv_init" ,{"z","y","x"});
    nc.create_var<real>("rw_init" ,{"z","y","x"});
    nc.create_var<real>("pp_final",{"z","y","x"});
    nc.create_var<real>("ru_final",{"z","y","x"});
    nc.create_var<real>("rv_final",{"z","y","x"});
    nc.create_var<real>("rw_final",{"z","y","x"});
    nc.enddef();
    using yakl::COLON;
    std::vector<MPI_Offset> start = {0,0,0};
    nc.write_all( x_host , "x" , {(MPI_Offset)0} );
    nc.write_all( y_host , "y" , {(MPI_Offset)0} );
    nc.write_all( z_host , "z" , {(MPI_Offset)0} );
    nc.write_all( soln_init_h .slice<3>(idP,COLON,COLON,COLON) , "pp_init"  , start );
    nc.write_all( soln_init_h .slice<3>(idU,COLON,COLON,COLON) , "ru_init"  , start );
    nc.write_all( soln_init_h .slice<3>(idV,COLON,COLON,COLON) , "rv_init"  , start );
    nc.write_all( soln_init_h .slice<3>(idW,COLON,COLON,COLON) , "rw_init"  , start );
    nc.write_all( soln_final_h.slice<3>(idP,COLON,COLON,COLON) , "pp_final" , start );
    nc.write_all( soln_final_h.slice<3>(idU,COLON,COLON,COLON) , "ru_final" , start );
    nc.write_all( soln_final_h.slice<3>(idV,COLON,COLON,COLON) , "rv_final" , start );
    nc.write_all( soln_final_h.slice<3>(idW,COLON,COLON,COLON) , "rw_final" , start );
    nc.close();
  }



  {
    int nx = 100;
    int ny = 100;
    int nz = 100;
    real xlen = 1.;
    real ylen = 1.;
    real zlen = 1.;
    real dx = xlen/nx;
    real dy = ylen/ny;
    real dz = zlen/nz;
    real dt = 0.6*dx;
    real4d soln("soln",4,nz,ny,nx);
    int constexpr idP = 0;
    int constexpr idU = 1;
    int constexpr idV = 2;
    int constexpr idW = 3;
    auto forcing = soln.createDeviceObject();

    yakl::parallel_for( YAKL_AUTO_LABEL(), yakl::SimpleBounds<3>(nz,ny,nx),
                                           KOKKOS_LAMBDA (int k, int j, int i) {
      real x = (i+0.5) * dx;
      real y = (j+0.5) * dy;
      real z = (k+0.5) * dz;

      real pi  = M_PI;
      real kx1 = 2*pi/xlen;
      real ky1 = 2*pi/ylen;
      real kz1 =   pi/zlen;
      real kx2 = 4*pi/xlen;
      real ky2 = 2*pi/ylen;
      real kz2 = 2*pi/zlen;
      real a1  = 0.01;
      real a2  = 0.004;
      real b1  = 0.01;
      real b2  = 0.004;

      // Compressible part: grad(phi)
      real mx_comp  = a1*kx1*std::cos(kx1*x)*std::cos(ky1*y)*std::cos(kz1*z) -
                      a2*kx2*std::sin(kx2*x)*std::sin(ky2*y)*std::cos(kz2*z);
      real my_comp = -a1*ky1*std::sin(kx1*x)*std::sin(ky1*y)*std::cos(kz1*z) +
                      a2*ky2*std::cos(kx2*x)*std::cos(ky2*y)*std::cos(kz2*z);
      real mz_comp = -a1*kz1*std::sin(kx1*x)*std::cos(ky1*y)*std::sin(kz1*z) -
                      a2*kz2*std::cos(kx2*x)*std::sin(ky2*y)*std::sin(kz2*z);

      // Solenoidal part: curl(0,0,psi)
      real mx_sol =  b1*ky1*std::sin(kx1*x)*std::cos(ky1*y) -
                     b2*ky2*std::cos(kx2*x)*std::sin(ky2*y)*std::cos(kz1*z);
      real my_sol = -b1*kx1*std::cos(kx1*x)*std::sin(ky1*y) +
                     b2*kx2*std::sin(kx2*x)*std::cos(ky2*y)*std::cos(kz1*z);
      real mz_sol = 0;

      forcing(idP,k,j,i) = 0;
      forcing(idU,k,j,i) = mx_comp + mx_sol;
      forcing(idV,k,j,i) = my_comp + my_sol;
      forcing(idW,k,j,i) = mz_comp + mz_sol;

      soln   (idP,k,j,i) = 0;
      soln   (idU,k,j,i) = 0;
      soln   (idV,k,j,i) = 0;
      soln   (idW,k,j,i) = 0;
    });
    auto soln_init_h = soln.createHostCopy();
    YaklRestartedGMRES<real> gmres;
    YaklRestartedGMRES<real>::Options opts;
    opts.verbose = true;
    opts.rel_tol = 1.e-14;
    opts.restart = 50;
    opts.max_iters = 1000;
    auto compute_Ax = [=] ( yakl::Array<real *> const & x_in ,
                            yakl::Array<real *> const & Ax_out ,
                            MPI_Comm comm ) {
      auto x_in_4d   = x_in  .reshape(4,nz,ny,nx);
      auto Ax_out_4d = Ax_out.reshape(4,nz,ny,nx);
      using yakl::COLON;
      auto pp     = x_in_4d  .slice<3>(idP,COLON,COLON,COLON);
      auto ru     = x_in_4d  .slice<3>(idU,COLON,COLON,COLON);
      auto rv     = x_in_4d  .slice<3>(idV,COLON,COLON,COLON);
      auto rw     = x_in_4d  .slice<3>(idW,COLON,COLON,COLON);
      auto lhs_pp = Ax_out_4d.slice<3>(idP,COLON,COLON,COLON);
      auto lhs_ru = Ax_out_4d.slice<3>(idU,COLON,COLON,COLON);
      auto lhs_rv = Ax_out_4d.slice<3>(idV,COLON,COLON,COLON);
      auto lhs_rw = Ax_out_4d.slice<3>(idW,COLON,COLON,COLON);
      yakl::parallel_for( YAKL_AUTO_LABEL() , yakl::SimpleBounds<3>(nz,ny,nx) ,
                                              KOKKOS_LAMBDA (int k, int j, int i) {
        ///////////////////
        // East-West fluxes
        ///////////////////
        int im1 = i-1;    if (im1 < 0   ) im1 += nx;
        int ip1 = i+1;    if (ip1 > nx-1) ip1 -= nx;
        real constexpr cs = 100;
        real constexpr cs2 = cs*cs;
        // West interface
        real pp_L = pp(k,j,im1);    real pp_R = pp(k,j,i);
        real ru_L = ru(k,j,im1);    real ru_R = ru(k,j,i);
        real cs2_ru_west_upw = 0.5*(cs2*ru_L + cs2*ru_R - cs*(pp_R - pp_L));
        real pp_west_upw     = 0.5*(    pp_L +     pp_R - cs*(ru_R - ru_L));
        real ru_west_upw = cs2_ru_west_upw / cs2;
        // East interface
        pp_L = pp(k,j,i);      pp_R = pp(k,j,ip1);
        ru_L = ru(k,j,i);      ru_R = ru(k,j,ip1);
        real cs2_ru_east_upw = 0.5*(cs2*ru_L + cs2*ru_R - cs*(pp_R - pp_L));
        real pp_east_upw     = 0.5*(    pp_L +     pp_R - cs*(ru_R - ru_L));
        real ru_east_upw = cs2_ru_east_upw / cs2;
        /////////////////////
        // North-South fluxes
        /////////////////////
        int jm1 = j-1;    if (jm1 < 0   ) jm1 += ny;
        int jp1 = j+1;    if (jp1 > ny-1) jp1 -= ny;
        // South interface
             pp_L = pp(k,jm1,i);         pp_R = pp(k,j,i);
        real rv_L = rv(k,jm1,i);    real rv_R = rv(k,j,i);
        real cs2_rv_south_upw = 0.5*(cs2*rv_L + cs2*rv_R - cs*(pp_R - pp_L));
        real pp_south_upw     = 0.5*(    pp_L +     pp_R - cs*(rv_R - rv_L));
        real rv_south_upw = cs2_rv_south_upw / cs2;
        // North interface
        pp_L = pp(k,j,i);      pp_R = pp(k,jp1,i);
        rv_L = rv(k,j,i);      rv_R = rv(k,jp1,i);
        real cs2_rv_north_upw = 0.5*(cs2*rv_L + cs2*rv_R - cs*(pp_R - pp_L));
        real pp_north_upw     = 0.5*(    pp_L +     pp_R - cs*(rv_R - rv_L));
        real rv_north_upw = cs2_rv_north_upw / cs2;
        /////////////////
        // Up-Down fluxes
        /////////////////
        int km1 = std::max(0   ,k-1);
        int kp1 = std::min(nz-1,k+1);
        // Down interface
             pp_L = pp(km1,j,i);         pp_R = pp(k,j,i);
        real rw_L = rw(km1,j,i);    real rw_R = rw(k,j,i);
        if (k==0) rw_L = 0;
        real cs2_rw_down_upw = 0.5*(cs2*rw_L + cs2*rw_R - cs*(pp_R - pp_L));
        real pp_down_upw     = 0.5*(    pp_L +     pp_R - cs*(rw_R - rw_L));
        real rw_down_upw = cs2_rw_down_upw / cs2;
        if (k==0) rw_down_upw = 0;
        // Up interface
        pp_L = pp(k,j,i);      pp_R = pp(kp1,j,i);
        rw_L = rw(k,j,i);      rw_R = rw(kp1,j,i);
        if (k==nz-1) rw_R = 0;
        real cs2_rw_up_upw = 0.5*(cs2*rw_L + cs2*rw_R - cs*(pp_R - pp_L));
        real pp_up_upw     = 0.5*(    pp_L +     pp_R - cs*(rw_R - rw_L));
        real rw_up_upw = cs2_rw_up_upw / cs2;
        if (k==nz-1) rw_up_upw = 0;

        // Update LHS with North-South fluxes
        lhs_pp(k,j,i) = (ru_east_upw  - ru_west_upw ) / dx +
                        (rv_north_upw - rv_south_upw) / dy +
                        (rw_up_upw    - rw_down_upw ) / dz;
        lhs_ru(k,j,i) = ru(k,j,i) + dt*(pp_east_upw  - pp_west_upw ) / dx;
        lhs_rv(k,j,i) = rv(k,j,i) + dt*(pp_north_upw - pp_south_upw) / dy;
        lhs_rw(k,j,i) = rw(k,j,i) + dt*(pp_up_upw    - pp_down_upw ) / dz;
      });
    };
    auto result = gmres.solve( soln.collapse() , forcing.collapse() , compute_Ax , opts );
    auto soln_final_h = soln.createHostCopy();
    realHost1d x_host("x_host",nx);
    realHost1d y_host("y_host",ny);
    realHost1d z_host("z_host",nz);
    for (int i=0; i<nx; i++) { x_host(i) = (i+0.5)*dx; }
    for (int j=0; j<ny; j++) { y_host(j) = (j+0.5)*dy; }
    for (int k=0; k<nz; k++) { z_host(k) = (k+0.5)*dz; }
    yakl::SimplePNetCDF nc;
    nc.create("wallz.nc");
    nc.create_dim("x",(MPI_Offset)nx);
    nc.create_dim("y",(MPI_Offset)ny);
    nc.create_dim("z",(MPI_Offset)nz);
    nc.create_var<real>("x",{"x"});
    nc.create_var<real>("y",{"y"});
    nc.create_var<real>("z",{"z"});
    nc.create_var<real>("pp_init" ,{"z","y","x"});
    nc.create_var<real>("ru_init" ,{"z","y","x"});
    nc.create_var<real>("rv_init" ,{"z","y","x"});
    nc.create_var<real>("rw_init" ,{"z","y","x"});
    nc.create_var<real>("pp_final",{"z","y","x"});
    nc.create_var<real>("ru_final",{"z","y","x"});
    nc.create_var<real>("rv_final",{"z","y","x"});
    nc.create_var<real>("rw_final",{"z","y","x"});
    nc.enddef();
    using yakl::COLON;
    std::vector<MPI_Offset> start = {0,0,0};
    nc.write_all( x_host , "x" , {(MPI_Offset)0} );
    nc.write_all( y_host , "y" , {(MPI_Offset)0} );
    nc.write_all( z_host , "z" , {(MPI_Offset)0} );
    nc.write_all( soln_init_h .slice<3>(idP,COLON,COLON,COLON) , "pp_init"  , start );
    nc.write_all( soln_init_h .slice<3>(idU,COLON,COLON,COLON) , "ru_init"  , start );
    nc.write_all( soln_init_h .slice<3>(idV,COLON,COLON,COLON) , "rv_init"  , start );
    nc.write_all( soln_init_h .slice<3>(idW,COLON,COLON,COLON) , "rw_init"  , start );
    nc.write_all( soln_final_h.slice<3>(idP,COLON,COLON,COLON) , "pp_final" , start );
    nc.write_all( soln_final_h.slice<3>(idU,COLON,COLON,COLON) , "ru_final" , start );
    nc.write_all( soln_final_h.slice<3>(idV,COLON,COLON,COLON) , "rv_final" , start );
    nc.write_all( soln_final_h.slice<3>(idW,COLON,COLON,COLON) , "rw_final" , start );
    nc.close();
  }

  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}
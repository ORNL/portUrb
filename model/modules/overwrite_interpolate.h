
#pragma once

#include "coupler.h"
#include "MultipleFields.h"

namespace modules {


  inline void overwrite_interpolate( core::Coupler                  & coupler  ,
                                     std::string              const & fname    ,
                                     std::vector<std::string> const & varnames ) {
    auto nx     = coupler.get_nx();
    auto ny     = coupler.get_ny();
    auto nz     = coupler.get_nz();
    auto dx     = coupler.get_dx();
    auto dy     = coupler.get_dy();
    auto dz     = coupler.get_dz();
    auto i_beg  = coupler.get_i_beg();
    auto j_beg  = coupler.get_j_beg();
    auto zmid   = coupler.get_zmid();
    auto zint   = coupler.get_zint();
    auto dz_h   = dz.createHostCopy();
    auto zint_h = zint.createHostCopy();
    auto &dm    = coupler.get_data_manager_readwrite();
    auto hy_r   = dm.get<real,1>("hy_dens_cells");
    auto hy_p   = dm.get<real,1>("hy_pressure_cells");
    auto R_d    = coupler.get_option<real>("R_d");
    // Open the file
    auto par_comm = coupler.get_parallel_comm();
    yakl::SimplePNetCDF nc(par_comm.get_mpi_comm());
    nc.open(fname,NC_NOWRITE);
    // Get the x, y, and z coordinates from the file
    int nx_glob_f = (int) nc.get_dim_size("x");
    int ny_glob_f = (int) nc.get_dim_size("y");
    int nz_f      = (int) nc.get_dim_size("z");
    int nz_halo_f = (int) nc.get_dim_size("z_halo");
    int hs_f      = (nz_halo_f-nz_f)/2;
    int hs        = (hy_r.size()-nz)/2;
    nc.begin_indep_data();
    real1d x_f   ("x_f"   ,nx_glob_f);
    real1d y_f   ("y_f"   ,ny_glob_f);
    real1d z_f   ("z_f"   ,nz_f     );
    real1d zint_f("zint_f",nz_f+1   );
    real1d hy_r_f("hy_dens_cells"    ,nz_halo_f);
    real1d hy_p_f("hy_pressure_cells",nz_halo_f);
    if (coupler.is_mainproc()) nc.read(x_f   ,"x" );
    if (coupler.is_mainproc()) nc.read(y_f   ,"y" );
    if (coupler.is_mainproc()) nc.read(z_f   ,"z" );
    if (coupler.is_mainproc()) nc.read(zint_f,"zi");
    if (coupler.is_mainproc()) nc.read(hy_r_f,"hy_dens_cells"    );
    if (coupler.is_mainproc()) nc.read(hy_p_f,"hy_pressure_cells");
    par_comm.broadcast(x_f   );
    par_comm.broadcast(y_f   );
    par_comm.broadcast(z_f   );
    par_comm.broadcast(zint_f);
    par_comm.broadcast(hy_r_f);
    par_comm.broadcast(hy_p_f);
    nc.end_indep_data();
    auto x_f_h    = x_f   .createHostCopy();
    auto y_f_h    = y_f   .createHostCopy();
    auto z_f_h    = z_f   .createHostCopy();
    auto zint_f_h = zint_f.createHostCopy();
    auto dx_f = x_f_h(1)-x_f_h(0);
    auto dy_f = y_f_h(1)-y_f_h(0);
    realHost1d dz_f_h("dz_f",nz_f);
    for (int i=0; i < nz_f; i++) { dz_f_h(i) = zint_f_h(i+1)-zint_f_h(i); }
    auto dz_f = dz_f_h.createDeviceCopy();
    // Get extents this task needs from the file
    real x1 = (i_beg   )*dx;
    real x2 = (i_beg+nx)*dx;
    real y1 = (j_beg   )*dy;
    real y2 = (j_beg+ny)*dy;
    real z1 = zint_h(0 );
    real z2 = zint_h(nz);
    int i1_f = std::max(0          ,(int)std::floor(x1/dx_f)-1);
    int i2_f = std::min(nx_glob_f-1,(int)std::ceil (x2/dx_f)+1);
    int j1_f = std::max(0          ,(int)std::floor(y1/dy_f)-1);
    int j2_f = std::min(ny_glob_f-1,(int)std::ceil (y2/dy_f)+1);
    int nx_f = i2_f - i1_f + 1;
    int ny_f = j2_f - j1_f + 1;
    std::vector<MPI_Offset> start = {(MPI_Offset)0,(MPI_Offset)j1_f,(MPI_Offset)i1_f};
    core::MultiField<real,3> fields_f;
    core::MultiField<real,3> fields;
    int idR = -1;
    int idT = -1;
    for (int i=0; i < varnames.size(); i++) {
      real3d field(varnames.at(i),nz_f,ny_f,nx_f);
      nc.read_all(field,varnames.at(i),start);
      fields_f.add_field(field);
      fields  .add_field(dm.get<real,3>(varnames.at(i)));
      if (varnames.at(i) == "density_dry") idR = i;
      if (varnames.at(i) == "temperature") idT = i;
    }
    yakl::parallel_for( YAKL_AUTO_LABEL() , yakl::SimpleBounds<4>(varnames.size(),nz,ny,nx) ,
                                            KOKKOS_LAMBDA (int l, int k, int j, int i) {
      int i_f = std::max(0,std::min(nx_f-1,(int) std::round((i_beg+i+0.5)*dx/dx_f-0.5) - i1_f));
      int j_f = std::max(0,std::min(ny_f-1,(int) std::round((j_beg+j+0.5)*dy/dy_f-0.5) - j1_f));
      int k_f;
      real zdist = std::numeric_limits<real>::max();
      for (int k2=0; k2 < nz_f; k2++) {
        if (std::abs(zmid(k) - z_f(k2)) < zdist) { zdist = std::abs(zmid(k) - z_f(k2)); k_f = k2; }
      }
      if (l == idR) {
        fields(l,k,j,i) = (fields_f(l,k_f,j_f,i_f)-hy_r_f(hs_f+k_f)) + hy_r(hs+k);
      } else if (l == idT) {
        auto T_f = hy_p_f(hs_f+k_f)/R_d/hy_r_f(hs_f+k_f);
        auto T   = hy_p  (hs  +k  )/R_d/hy_r  (hs  +k  );
        fields(l,k,j,i) = (fields_f(l,k_f,j_f,i_f)-T_f) + T;
      } else {
        fields(l,k,j,i) = fields_f(l,k_f,j_f,i_f);
      }
    });
  }


}



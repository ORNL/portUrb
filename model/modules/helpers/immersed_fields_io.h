#pragma once

#include "coupler.h"

namespace modules {

  inline void register_immersed_fields_io(core::Coupler &coupler) {
    int constexpr imm_hs = 1;
    std::vector<std::string> const dim_names = {
      "z_imm_halo1", "y_imm_halo1", "x_imm_halo1"
    };

    coupler.register_write_output_module([=](core::Coupler &coupler, core::FileIO &nc) {
      auto nz = coupler.get_nz();
      auto ny = coupler.get_ny();
      auto nx = coupler.get_nx();
      auto i_beg = coupler.get_i_beg();
      auto j_beg = coupler.get_j_beg();
      auto px = coupler.get_px();
      auto py = coupler.get_py();
      auto nproc_x = coupler.get_nproc_x();
      auto nproc_y = coupler.get_nproc_y();
      auto &dm = coupler.get_data_manager_readonly();
      auto immersed_prop = dm.get<real const,3>("immersed_proportion");
      auto immersed_rough = dm.get<real const,3>("immersed_roughness");
      auto i_src = px == 0 ? 0 : imm_hs;
      auto j_src = py == 0 ? 0 : imm_hs;
      auto nx_out = nx + (px == 0 ? imm_hs : 0) + (px == nproc_x-1 ? imm_hs : 0);
      auto ny_out = ny + (py == 0 ? imm_hs : 0) + (py == nproc_y-1 ? imm_hs : 0);
      float3d immersed_prop_out("immersed_prop_out",nz+2*imm_hs,ny_out,nx_out);
      float3d immersed_rough_out("immersed_rough_out",nz+2*imm_hs,ny_out,nx_out);
      yakl::parallel_for(YAKL_AUTO_LABEL(), yakl::SimpleBounds<3>(nz+2*imm_hs,ny_out,nx_out),
                         KOKKOS_LAMBDA(int k, int j, int i) {
        immersed_prop_out(k,j,i) = immersed_prop(k,j_src+j,i_src+i);
        immersed_rough_out(k,j,i) = immersed_rough(k,j_src+j,i_src+i);
      });

      nc.redef();
      if (!nc.dim_exists(dim_names.at(0))) {
        nc.create_dim(dim_names.at(0),nz+2*imm_hs);
        nc.create_dim(dim_names.at(1),coupler.get_ny_glob()+2*imm_hs);
        nc.create_dim(dim_names.at(2),coupler.get_nx_glob()+2*imm_hs);
      }
      if (!nc.var_exists("immersed_proportion")) {
        nc.create_var<float>("immersed_proportion",dim_names);
        nc.writeVariableAttribute(std::string("1"),"immersed_proportion","units");
      }
      if (!nc.var_exists("immersed_roughness")) {
        nc.create_var<float>("immersed_roughness",dim_names);
        nc.writeVariableAttribute(std::string("m"),"immersed_roughness","units");
      }
      nc.enddef();
      std::vector<MPI_Offset> start = {
        0,
        static_cast<MPI_Offset>(j_beg + (py == 0 ? 0 : imm_hs)),
        static_cast<MPI_Offset>(i_beg + (px == 0 ? 0 : imm_hs))
      };
      nc.write_all(immersed_prop_out,"immersed_proportion",start);
      nc.write_all(immersed_rough_out,"immersed_roughness",start);
    });

    coupler.register_overwrite_with_restart_module([=](core::Coupler &coupler, core::FileIO &nc) {
      auto &dm = coupler.get_data_manager_readwrite();
      auto immersed_prop = dm.get<real,3>("immersed_proportion");
      auto immersed_rough = dm.get<real,3>("immersed_roughness");
      auto nz = coupler.get_nz();
      auto ny = coupler.get_ny();
      auto nx = coupler.get_nx();
      float3d immersed_prop_in("immersed_prop_in",nz+2*imm_hs,ny+2*imm_hs,nx+2*imm_hs);
      float3d immersed_rough_in("immersed_rough_in",nz+2*imm_hs,ny+2*imm_hs,nx+2*imm_hs);
      std::vector<MPI_Offset> start = {
        0,
        static_cast<MPI_Offset>(coupler.get_j_beg()),
        static_cast<MPI_Offset>(coupler.get_i_beg())
      };
      nc.read_all(immersed_prop_in,"immersed_proportion",start);
      nc.read_all(immersed_rough_in,"immersed_roughness",start);
      yakl::parallel_for(YAKL_AUTO_LABEL(),
                         yakl::SimpleBounds<3>(nz+2*imm_hs,ny+2*imm_hs,nx+2*imm_hs),
                         KOKKOS_LAMBDA(int k, int j, int i) {
        immersed_prop(k,j,i) = immersed_prop_in(k,j,i);
        immersed_rough(k,j,i) = immersed_rough_in(k,j,i);
      });
    });
  }

}
#pragma once

#include "coupler.h"

namespace custom_modules {

  template <class Dycore>
  inline void register_tank_tracer_injection( core::Coupler const & coupler     ,
                                              Dycore             & dycore       ,
                                              real                 x0           ,
                                              real                 y0           ,
                                              real                 radius       ,
                                              real                 tracer_in    ,
                                              real                 wvel         ,
                                              std::string          tracer_name ) {
    using yakl::SimpleBounds;
    using FLOC      = typename Dycore::FLOC;
    using FluxArray = typename Dycore::FluxArray;

    auto tracer_id = coupler.get_tracer_index(tracer_name);
    if (tracer_id < 0) { endrun("ERROR: tank tracer injection tracer is not registered"); }
    if (radius <= 0)   { endrun("ERROR: tank tracer injection radius must be positive"); }

    dycore.register_flux_addition_callback(
      [=] ( core::Coupler const & c                  ,
            real4d        const & /* state */        ,
            real4d        const & /* tracers */      ,
            FluxArray     const & /* flux_x */       ,
            FluxArray     const & /* flux_y */       ,
            FluxArray     const & flux_z             ,
            int                   tracer_flux_offset ,
            real                  /* dt */           ,
            int                   /* istage */       ,
            int                   /* icycle */       ) {
        auto nx             = c.get_nx();
        auto ny             = c.get_ny();
        auto dx             = c.get_dx();
        auto dy             = c.get_dy();
        auto i_beg          = c.get_i_beg();
        auto j_beg          = c.get_j_beg();
        auto imm_th         = c.get_option<real>("immersed_threshold",0.5);
        auto &dm            = c.get_data_manager_readonly();
        auto immersed_prop  = dm.get<real const,3>("immersed_proportion");
        auto hy_dens_edges  = dm.get<real const,1>("hy_dens_edges"      );
        auto hy_theta_edges = dm.get<real const,1>("hy_theta_edges"     );
        int constexpr idR   = Dycore::idR;
        int constexpr idW   = Dycore::idW;
        int constexpr idT   = Dycore::idT;

        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , KOKKOS_LAMBDA (int j, int i) {
          real x = (i_beg+i+0.5_fp)*dx;
          real y = (j_beg+j+0.5_fp)*dy;
          real r = std::sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0));
          if (r <= radius && immersed_prop(0,j,i) <= imm_th) {
            FLOC w_in      = static_cast<FLOC>(wvel*std::max(0._fp,1._fp-(r/radius)*(r/radius)));
            FLOC mass_flux = static_cast<FLOC>(hy_dens_edges(0))*w_in;
            flux_z(idR                         ,0,j,i) += mass_flux;
            flux_z(idW                         ,0,j,i) += mass_flux*w_in;
            flux_z(idT                         ,0,j,i) += mass_flux*static_cast<FLOC>(hy_theta_edges(0));
            flux_z(tracer_flux_offset+tracer_id,0,j,i) += mass_flux*static_cast<FLOC>(tracer_in);
          }
        });
      }
    );
  }

}

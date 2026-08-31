#pragma once

#include "coupler.h"

namespace custom_modules {

  template <class Dycore>
  inline void register_tank_tracer_injection( core::Coupler const & coupler      ,
                                              Dycore              & dycore       ,
                                              real                 x0            ,
                                              real                 y0            ,
                                              real                 tracer_radius ,
                                              real                 wvel_radius   ,
                                              real                 conc          ,
                                              real                 wvel          ,
                                              std::string          tracer_name   ) {
    using yakl::SimpleBounds;
    using FLOC      = typename Dycore::FLOC;
    using FluxArray = typename Dycore::FluxArray;

    auto tracer_id = coupler.get_tracer_index(tracer_name);
    if (tracer_id < 0) { endrun("ERROR: tank tracer injection tracer is not registered"); }
    if (wvel_radius <= 0)   { endrun("ERROR: tank tracer injection wvel radius must be positive"); }
    if (tracer_radius <= 0)   { endrun("ERROR: tank tracer injection tracer radius must be positive"); }

    dycore.register_flux_addition_callback(
      [=] ( core::Coupler       & c                  ,
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
        auto nx_glob        = c.get_nx_glob();
        auto ny_glob        = c.get_ny_glob();
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
        // auto seed = c.get_option<size_t>("tracer_injection_seed",0);
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , KOKKOS_LAMBDA (int j, int i) {
          // size_t i_stream = size_t(j_beg+j)*size_t(nx_glob) + size_t(i_beg+i);
          // yakl::Random prng(seed,i_stream);
          real x = (i_beg+i+0.5)*dx;
          real y = (j_beg+j+0.5)*dy;
          real r = std::sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0));
          if (r <= std::max(wvel_radius,tracer_radius) && immersed_prop(0,j,i) <= imm_th) {
            real wvel_shape   = 1; // std::max( 0. , 1.-(r/wvel_radius)*(r/wvel_radius) );
            real tracer_shape = 1; // std::max( 0. , 1.-(r/tracer_radius)*(r/tracer_radius) );
            real w_in         = wvel*wvel_shape;
            real conc_in      = conc*tracer_shape;
            // w_in    *= (1 + prng.gen_normal(0.,0.1));
            // conc_in *= (1 + prng.gen_normal(0.,0.1));
            real mass_flux    = hy_dens_edges(0)*w_in;
            flux_z(idR                         ,0,j,i) += mass_flux;
            flux_z(idW                         ,0,j,i) += mass_flux*w_in;
            flux_z(idT                         ,0,j,i) += mass_flux*hy_theta_edges(0);
            flux_z(tracer_flux_offset+tracer_id,0,j,i) += mass_flux*conc_in;
          }
        });
        // c.set_option<size_t>("tracer_injection_seed",seed+size_t(ny_glob)*size_t(nx_glob));
        // c.register_output_option("tracer_injection_seed");
      }
    );
  }

}

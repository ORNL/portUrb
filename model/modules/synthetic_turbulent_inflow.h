#pragma once

#include "coupler.h"

#include <cstdint>

namespace modules {


  // Generates a temporally coherent, multi-scale turbulent inflow in the existing x1 precursor storage.
  //
  // Each eddy is represented by a compact vector potential. The velocity is evaluated as its analytic curl, so
  // the continuum field is divergence free even though sampling it on the dycore grid is not discretely solenoidal.
  // Eddy bands cover only the truncated inertial range from outer_length through the globally resolvable four-cell
  // cutoff. Their target energies are exact band integrals of E(k) proportional to k^(-5/3); no production or
  // dissipation range is modeled.
  // The module perturbs velocity only; density, potential temperature, pressure, and tracers use a fixed primitive
  // column captured during initialization. This deliberately avoids injecting thermodynamic noise at a compressible
  // boundary.
  //
  // The module is stateful and registers output/restart callbacks that retain a pointer to it. Construct it in the
  // same scope as the Coupler, initialize it once, and do not move it afterward.
  class SyntheticTurbulentInflow {
  public:
    struct Config {
      int  num_eddies        = 256; // Total compact eddies distributed approximately evenly among scale bands
      int  random_seed       = 0;   // Seed for decomposition-independent deterministic eddy properties
      real outer_length      = -1;  // Largest eddy diameter [m]; <= 0 uses 1/4 of the smaller inlet dimension
      real wall_decay_length = -1;  // Free-slip normal-velocity decay distance [m]; <= 0 uses outer_length/2
    };

  private:
    int static constexpr num_eddy_properties = 9;
    int static constexpr num_device_properties = 11;
    int static constexpr id_x       = 0;
    int static constexpr id_y       = 1;
    int static constexpr id_z       = 2;
    int static constexpr id_radius  = 3;
    int static constexpr id_ex      = 4;
    int static constexpr id_ey      = 5;
    int static constexpr id_ez      = 6;
    int static constexpr id_scale   = 7;
    int static constexpr id_recycle = 8;

    Config config;
    bool initialized = false;
    bool wall_y1     = false;
    bool wall_y2     = false;
    bool wall_z1     = false;
    bool wall_z2     = false;
    bool periodic_y  = false;
    int  num_scales  = 0;
    int  halo_size   = 0;
    real outer_length = 0;
    real smallest_length = 0;
    real wall_decay_length = 0;
    real calibration = 1;
    real x_ghost_min = 0;

    real1d u_mean;
    real1d v_mean;
    real1d turbulence_intensity;
    real1d scale_diameter;
    real1d scale_fraction;
    real1d scale_amplitude;
    real2d base_column;
    real2d eddies;
    realHost2d eddy_state_host;

    // SplitMix64 provides inexpensive reproducible random bits without retaining a standard-library RNG state.
    // Eddy recycling therefore depends only on the user seed, eddy index, recycle count, and property stream.
    static std::uint64_t mix_bits(std::uint64_t value) {
      value += 0x9e3779b97f4a7c15ULL;
      value  = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
      value  = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
      return value ^ (value >> 31);
    }

    real uniform(int eddy, int recycle, int stream) const {
      auto value = static_cast<std::uint64_t>(config.random_seed);
      value ^= static_cast<std::uint64_t>(eddy    + 1) * 0xd2b74407b1ce6e93ULL;
      value ^= static_cast<std::uint64_t>(recycle + 1) * 0xca5a826395121157ULL;
      value ^= static_cast<std::uint64_t>(stream  + 1) * 0x9e3779b97f4a7c15ULL;
      return static_cast<real>(mix_bits(value) >> 11) * (1._fp / 9007199254740992._fp);
    }

    static int nearest_level(real z, realHost1d const &zmid) {
      int level = 0;
      real distance = std::abs(z-zmid(0));
      for (int k = 1; k < zmid.extent(0); k++) {
        real const candidate = std::abs(z-zmid(k));
        if (candidate < distance) {
          distance = candidate;
          level = k;
        }
      }
      return level;
    }

    void randomize_eddy(int eddy, int recycle, realHost1d const &diameters, real ylen, real zlen, bool initial) {
      int const scale = eddy % num_scales;
      real const y = uniform(eddy,recycle,1)*ylen;
      real const z = uniform(eddy,recycle,2)*zlen;
      real const radius = diameters(scale)/2;

      real const ez = 2*uniform(eddy,recycle,3)-1;
      real const azimuth = 2*M_PI*uniform(eddy,recycle,4);
      real const eh = std::sqrt(std::max(0._fp,1-ez*ez));
      real const ex = eh*std::cos(azimuth);
      real const ey = eh*std::sin(azimuth);

      // Initially distribute centers across the complete slab that can influence x1 ghosts. Recycled eddies enter
      // where their leading edge first touches the outermost ghost-cell face.
      real const xmin = x_ghost_min-radius;
      real const xmax = radius;
      real const x = initial ? xmin + uniform(eddy,recycle,0)*(xmax-xmin) : xmin;
      eddy_state_host(eddy,id_x      ) = x;
      eddy_state_host(eddy,id_y      ) = y;
      eddy_state_host(eddy,id_z      ) = z;
      eddy_state_host(eddy,id_radius ) = radius;
      eddy_state_host(eddy,id_ex     ) = ex;
      eddy_state_host(eddy,id_ey     ) = ey;
      eddy_state_host(eddy,id_ez     ) = ez;
      eddy_state_host(eddy,id_scale  ) = scale;
      eddy_state_host(eddy,id_recycle) = recycle;
    }

    void update_device_eddies(realHost1d const &zmid_host) {
      auto u_host  = u_mean.createHostCopy();
      auto v_host  = v_mean.createHostCopy();
      auto ti_host = turbulence_intensity.createHostCopy();
      auto amplitudes_host = scale_amplitude.createHostCopy();
      realHost2d eddies_host("synthetic_turbulent_inflow_eddies_host",config.num_eddies,num_device_properties);
      for (int n = 0; n < config.num_eddies; n++) {
        int const k = nearest_level(eddy_state_host(n,id_z),zmid_host);
        int const scale = static_cast<int>(eddy_state_host(n,id_scale));
        real const mean_speed = std::sqrt(u_host(k)*u_host(k) + v_host(k)*v_host(k));
        real const amplitude = ti_host(k)*mean_speed*amplitudes_host(scale);
        for (int p = 0; p < 7; p++) eddies_host(n,p) = eddy_state_host(n,p);
        eddies_host(n,7) = amplitude;
        eddies_host(n,8) = u_host(k);
        eddies_host(n,9) = v_host(k);
        eddies_host(n,10) = scale;
      }
      eddies_host.deep_copy_to(eddies);
    }

    // Quintic smoothstep and its derivative. Both the first and second derivatives vanish at xi=0 and xi=1,
    // preventing the wall taper itself from introducing a sharp feature into high-order reconstruction stencils.
    KOKKOS_INLINE_FUNCTION static void smooth_wall(real distance, real decay_length, real derivative_sign,
                                                    real &factor, real &derivative) {
      if (distance >= decay_length) {
        factor = 1;
        derivative = 0;
        return;
      }
      real const xi = std::max(0._fp,distance/decay_length);
      factor = xi*xi*xi*(10 + xi*(-15 + 6*xi));
      derivative = derivative_sign*30*xi*xi*(xi-1)*(xi-1)/decay_length;
    }

    KOKKOS_INLINE_FUNCTION static void wall_factors(real y, real z, real ylen, real zlen, real decay_length,
                                                    bool wall_y1, bool wall_y2, bool wall_z1, bool wall_z2,
                                                    real &Y, real &dYdy, real &Z, real &dZdz) {
      real y1 = 1, y2 = 1, z1 = 1, z2 = 1;
      real dy1 = 0, dy2 = 0, dz1 = 0, dz2 = 0;
      if (wall_y1) smooth_wall(y,      decay_length,  1,y1,dy1);
      if (wall_y2) smooth_wall(ylen-y, decay_length, -1,y2,dy2);
      if (wall_z1) smooth_wall(z,      decay_length,  1,z1,dz1);
      if (wall_z2) smooth_wall(zlen-z, decay_length, -1,z2,dz2);
      Y = y1*y2;
      Z = z1*z2;
      dYdy = dy1*y2 + y1*dy2;
      dZdz = dz1*z2 + z1*dz2;
    }

    // Evaluate the analytic curl of
    //   A_x = Y Z A0_x,  A_y = Z A0_y,  A_z = Y A0_z.
    // This component-selective taper makes v vanish at spanwise free-slip walls and w vanish at vertical free-slip
    // walls. Tangential fluctuations need not vanish. Since the final field remains a curl, it is analytically
    // divergence free everywhere that the compact polynomial is differentiable.
    KOKKOS_INLINE_FUNCTION static void evaluate_velocity(real x, real y, real z, real ylen, real zlen,
                                                         real decay_length, bool wall_y1, bool wall_y2,
                                                         bool wall_z1, bool wall_z2, bool periodic_y,
                                                         real calibration, realConst2d const &eddies,
                                                         real time_offset, int requested_scale,
                                                         real &u, real &v, real &w) {
      u = 0;
      v = 0;
      w = 0;
      real Y, dYdy, Z, dZdz;
      wall_factors(y,z,ylen,zlen,decay_length,wall_y1,wall_y2,wall_z1,wall_z2,Y,dYdy,Z,dZdz);
      for (int n = 0; n < eddies.extent(0); n++) {
        if (requested_scale >= 0 && static_cast<int>(eddies(n,10)) != requested_scale) continue;
        real rx = x-eddies(n,0)-eddies(n,8)*time_offset;
        real ry = y-eddies(n,1)-eddies(n,9)*time_offset;
        real rz = z-eddies(n,2);
        if (periodic_y) {
          if      (ry >  ylen/2) ry -= ylen;
          else if (ry < -ylen/2) ry += ylen;
        }
        real const radius = eddies(n,3);
        real const q2 = (rx*rx + ry*ry + rz*rz)/(radius*radius);
        if (q2 >= 1) continue;

        real const one_minus_q2 = 1-q2;
        real const phi = one_minus_q2*one_minus_q2*one_minus_q2;
        real const gradient_coefficient = -6*one_minus_q2*one_minus_q2/(radius*radius);
        real const coefficient = calibration*eddies(n,7)*radius;
        real const ax = coefficient*eddies(n,4)*phi;
        real const ay = coefficient*eddies(n,5)*phi;
        real const az = coefficient*eddies(n,6)*phi;
        real const dax_dy = coefficient*eddies(n,4)*gradient_coefficient*ry;
        real const dax_dz = coefficient*eddies(n,4)*gradient_coefficient*rz;
        real const day_dx = coefficient*eddies(n,5)*gradient_coefficient*rx;
        real const day_dz = coefficient*eddies(n,5)*gradient_coefficient*rz;
        real const daz_dx = coefficient*eddies(n,6)*gradient_coefficient*rx;
        real const daz_dy = coefficient*eddies(n,6)*gradient_coefficient*ry;

        u += dYdy*az + Y*daz_dy - dZdz*ay - Z*day_dz;
        v += Y*dZdz*ax + Y*Z*dax_dz - Y*daz_dx;
        w += Z*day_dx - dYdy*Z*ax - Y*Z*dax_dy;
      }
    }

    // Equal numbers of compact eddies at different diameters do not have equal plane-averaged response: their
    // support volumes and the Fourier response of the curl kernel differ. Measure that response for the retained
    // ensemble one scale at a time, then choose the amplitude of each scale so its variance equals the exact
    // integral of k^(-5/3) over that wavenumber band. This is an initialization-only calculation.
    void calibrate_scales(core::Coupler const &coupler) {
      using yakl::SimpleBounds;
      int const px      = coupler.get_px();
      int const nz      = coupler.get_nz();
      int const ny      = coupler.get_ny();
      int const j_beg   = coupler.get_j_beg();
      int const ny_glob = coupler.get_ny_glob();
      real const dx     = coupler.get_dx();
      real const dy     = coupler.get_dy();
      real const ylen   = coupler.get_ylen();
      real const zlen   = coupler.get_zlen();
      auto const zmid = coupler.get_zmid();
      auto const eddies_device = realConst2d(eddies);
      real4d velocity("synthetic_turbulent_inflow_scale_velocity",num_scales,3,nz,ny);
      real3d means("synthetic_turbulent_inflow_scale_means",num_scales,2,nz);
      real1d actual("synthetic_turbulent_inflow_scale_actual",num_scales);
      bool const wy1 = wall_y1;
      bool const wy2 = wall_y2;
      bool const wz1 = wall_z1;
      bool const wz2 = wall_z2;
      bool const py  = periodic_y;
      real const decay = wall_decay_length;
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_scales,nz,ny) ,
                                              KOKKOS_LAMBDA (int scale, int k, int j) {
        velocity(scale,0,k,j) = 0;
        velocity(scale,1,k,j) = 0;
        velocity(scale,2,k,j) = 0;
        if (px != 0) return;
        real const y = (j_beg+j+0.5_fp)*dy;
        real u, v, w;
        evaluate_velocity(-0.5_fp*dx,y,zmid(k),ylen,zlen,decay,wy1,wy2,wz1,wz2,py,1,eddies_device,0,scale,u,v,w);
        velocity(scale,0,k,j) = u;
        velocity(scale,1,k,j) = v;
        velocity(scale,2,k,j) = w;
      });
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_scales,2,nz) ,
                                              KOKKOS_LAMBDA (int scale, int component, int k) {
        means(scale,component,k) = 0;
        if (px != 0) return;
        for (int j = 0; j < ny; j++) means(scale,component,k) += velocity(scale,component,k,j);
      });
      means = coupler.get_parallel_comm().all_reduce(means,MPI_SUM,"synthetic_inflow_scale_means");
      bool const correct_v = !wall_y1 && !wall_y2;
      yakl::parallel_for( YAKL_AUTO_LABEL() , num_scales , KOKKOS_LAMBDA (int scale) {
        actual(scale) = 0;
        if (px != 0) return;
        for (int k = 0; k < nz; k++) {
          real const mean_u = means(scale,0,k)/ny_glob;
          real const mean_v = correct_v ? means(scale,1,k)/ny_glob : 0;
          for (int j = 0; j < ny; j++) {
            real const u = velocity(scale,0,k,j)-mean_u;
            real const v = velocity(scale,1,k,j)-mean_v;
            real const w = velocity(scale,2,k,j);
            actual(scale) += u*u + v*v + w*w;
          }
        }
      });
      actual = coupler.get_parallel_comm().all_reduce(actual,MPI_SUM,"synthetic_inflow_scale_variance");

      auto actual_host    = actual.createHostCopy();
      auto fraction_host  = scale_fraction.createHostCopy();
      auto amplitude_host = scale_amplitude.createHostCopy();
      auto u_host  = u_mean.createHostCopy();
      auto v_host  = v_mean.createHostCopy();
      auto ti_host = turbulence_intensity.createHostCopy();
      real target_total = 0;
      for (int k = 0; k < nz; k++) {
        real const mean_speed_squared = u_host(k)*u_host(k) + v_host(k)*v_host(k);
        target_total += ny_glob*3*ti_host(k)*ti_host(k)*mean_speed_squared;
      }
      for (int scale = 0; scale < num_scales; scale++) {
        real const target = fraction_host(scale)*target_total;
        if (target > 0 && actual_host(scale) <= 0) {
          endrun("Synthetic turbulent inflow scale produced zero calibration variance");
        }
        amplitude_host(scale) = target > 0 ? std::sqrt(target/actual_host(scale)) : 0;
      }
      amplitude_host.deep_copy_to(scale_amplitude);
    }

    // Independent scale responses add only in the ensemble mean. A finite retained ensemble has small cross-scale
    // correlations, so apply one common residual factor after scale calibration. Because it multiplies every band
    // equally, this enforces the requested total turbulence intensity without changing the intended band ratios.
    void calibrate_total(core::Coupler const &coupler) {
      using yakl::SimpleBounds;
      int const px      = coupler.get_px();
      int const nz      = coupler.get_nz();
      int const ny      = coupler.get_ny();
      int const j_beg   = coupler.get_j_beg();
      int const ny_glob = coupler.get_ny_glob();
      real const dx     = coupler.get_dx();
      real const dy     = coupler.get_dy();
      real const ylen   = coupler.get_ylen();
      real const zlen   = coupler.get_zlen();
      auto const zmid = coupler.get_zmid();
      auto const eddies_device = realConst2d(eddies);
      real3d velocity("synthetic_turbulent_inflow_total_velocity",3,nz,ny);
      real2d means("synthetic_turbulent_inflow_total_means",2,nz);
      real2d actual("synthetic_turbulent_inflow_total_actual",nz,ny);
      bool const wy1 = wall_y1;
      bool const wy2 = wall_y2;
      bool const wz1 = wall_z1;
      bool const wz2 = wall_z2;
      bool const py  = periodic_y;
      real const decay = wall_decay_length;
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ny) , KOKKOS_LAMBDA (int k, int j) {
        velocity(0,k,j) = 0;
        velocity(1,k,j) = 0;
        velocity(2,k,j) = 0;
        if (px != 0) return;
        real const y = (j_beg+j+0.5_fp)*dy;
        real u, v, w;
        evaluate_velocity(-0.5_fp*dx,y,zmid(k),ylen,zlen,decay,wy1,wy2,wz1,wz2,py,1,eddies_device,0,-1,u,v,w);
        velocity(0,k,j) = u;
        velocity(1,k,j) = v;
        velocity(2,k,j) = w;
      });
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(2,nz) , KOKKOS_LAMBDA (int component, int k) {
        means(component,k) = 0;
        if (px != 0) return;
        for (int j = 0; j < ny; j++) means(component,k) += velocity(component,k,j);
      });
      means = coupler.get_parallel_comm().all_reduce(means,MPI_SUM,"synthetic_inflow_total_means");
      bool const correct_v = !wall_y1 && !wall_y2;
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ny) , KOKKOS_LAMBDA (int k, int j) {
        actual(k,j) = 0;
        if (px != 0) return;
        real const u = velocity(0,k,j)-means(0,k)/ny_glob;
        real const v = velocity(1,k,j)-(correct_v ? means(1,k)/ny_glob : 0);
        real const w = velocity(2,k,j);
        actual(k,j) = u*u + v*v + w*w;
      });
      real const actual_sum = coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(actual),MPI_SUM);
      auto u_host  = u_mean.createHostCopy();
      auto v_host  = v_mean.createHostCopy();
      auto ti_host = turbulence_intensity.createHostCopy();
      real target_sum = 0;
      for (int k = 0; k < nz; k++) {
        real const mean_speed_squared = u_host(k)*u_host(k) + v_host(k)*v_host(k);
        target_sum += ny_glob*3*ti_host(k)*ti_host(k)*mean_speed_squared;
      }
      if (target_sum > 0 && actual_sum <= 0) endrun("Synthetic turbulent inflow produced zero calibration variance");
      calibration = target_sum > 0 ? std::sqrt(target_sum/actual_sum) : 0;
    }

    void advance_eddies(core::Coupler const &coupler, real dt) {
      auto zmid_host = coupler.get_zmid().createHostCopy();
      auto diameters_host = scale_diameter.createHostCopy();
      auto u_host = u_mean.createHostCopy();
      auto v_host = v_mean.createHostCopy();
      real const ylen = coupler.get_ylen();
      real const zlen = coupler.get_zlen();
      for (int n = 0; n < config.num_eddies; n++) {
        int const k = nearest_level(eddy_state_host(n,id_z),zmid_host);
        eddy_state_host(n,id_x) += u_host(k)*dt;
        eddy_state_host(n,id_y) += v_host(k)*dt;
        if (periodic_y) {
          while (eddy_state_host(n,id_y) < 0   ) eddy_state_host(n,id_y) += ylen;
          while (eddy_state_host(n,id_y) >= ylen) eddy_state_host(n,id_y) -= ylen;
        }
        real const radius = eddy_state_host(n,id_radius);
        if (eddy_state_host(n,id_x)-radius > 0) {
          int const recycle = static_cast<int>(eddy_state_host(n,id_recycle))+1;
          randomize_eddy(n,recycle,diameters_host,ylen,zlen,false);
        }
      }
    }

    void register_restart(core::Coupler &coupler) {
      coupler.register_write_output_module( [this] (core::Coupler &coupler, core::FileIO &nc) {
        auto base_host = base_column.createHostCopy();
        auto scale_amplitude_host = scale_amplitude.createHostCopy();
        nc.redef();
        nc.create_dim("synthetic_inflow_eddy",config.num_eddies);
        nc.create_dim("synthetic_inflow_eddy_property",num_eddy_properties);
        nc.create_dim("synthetic_inflow_base_field",base_column.extent(0));
        nc.create_dim("synthetic_inflow_scale",num_scales);
        nc.create_var<real>("synthetic_inflow_eddy_state",{"synthetic_inflow_eddy","synthetic_inflow_eddy_property"});
        nc.create_var<real>("synthetic_inflow_base_column",{"synthetic_inflow_base_field","z"});
        nc.create_var<real>("synthetic_inflow_scale_amplitude",{"synthetic_inflow_scale"});
        nc.create_var<real>("synthetic_inflow_calibration",{});
        nc.enddef();
        nc.begin_indep_data();
        if (coupler.is_mainproc()) {
          nc.write(eddy_state_host,"synthetic_inflow_eddy_state");
          nc.write(base_host,"synthetic_inflow_base_column");
          nc.write(scale_amplitude_host,"synthetic_inflow_scale_amplitude");
          nc.write(calibration,"synthetic_inflow_calibration");
        }
        nc.end_indep_data();
      });
      coupler.register_overwrite_with_restart_module( [this] (core::Coupler &coupler, core::FileIO &nc) {
        realHost2d base_host("synthetic_inflow_restart_base",base_column.extent(0),base_column.extent(1));
        realHost1d scale_amplitude_host("synthetic_inflow_restart_scale_amplitude",num_scales);
        nc.begin_indep_data();
        if (coupler.is_mainproc()) {
          nc.read(eddy_state_host,"synthetic_inflow_eddy_state");
          nc.read(base_host,"synthetic_inflow_base_column");
          nc.read(scale_amplitude_host,"synthetic_inflow_scale_amplitude");
          nc.read(calibration,"synthetic_inflow_calibration");
        }
        nc.end_indep_data();
        auto const comm = coupler.get_parallel_comm();
        comm.broadcast(eddy_state_host);
        comm.broadcast(base_host);
        comm.broadcast(scale_amplitude_host);
        comm.broadcast(calibration);
        base_host.deep_copy_to(base_column);
        scale_amplitude_host.deep_copy_to(scale_amplitude);
        update_device_eddies(coupler.get_zmid().createHostCopy());
      });
    }

  public:
    SyntheticTurbulentInflow() = default;
    SyntheticTurbulentInflow(SyntheticTurbulentInflow const &) = delete;
    SyntheticTurbulentInflow &operator=(SyntheticTurbulentInflow const &) = delete;
    SyntheticTurbulentInflow(SyntheticTurbulentInflow &&) = delete;
    SyntheticTurbulentInflow &operator=(SyntheticTurbulentInflow &&) = delete;

    // Host diagnostics expose the realized scale model for unit tests and experiment metadata. Fractions sum to
    // one and are exact integrals of the truncated k^(-5/3) target; amplitudes additionally account for the compact
    // kernel, finite eddy population, mean correction, and wall taper.
    int get_num_scales() const {
      if (! initialized) endrun("Synthetic turbulent inflow scale diagnostics requested before init");
      return num_scales;
    }

    real get_smallest_length() const {
      if (! initialized) endrun("Synthetic turbulent inflow scale diagnostics requested before init");
      return smallest_length;
    }

    realHost1d get_scale_energy_fraction() const {
      if (! initialized) endrun("Synthetic turbulent inflow scale diagnostics requested before init");
      return scale_fraction.createHostCopy();
    }

    realHost1d get_scale_amplitude() const {
      if (! initialized) endrun("Synthetic turbulent inflow scale diagnostics requested before init");
      return scale_amplitude.createHostCopy();
    }

    template <class Dycore>
    void init(core::Coupler &coupler, Dycore &dycore, real1d const &u_mean_in, real1d const &v_mean_in,
              real1d const &turbulence_intensity_in, Config const &config_in = Config()) {
      if (initialized) endrun("Synthetic turbulent inflow may only be initialized once");
      if (coupler.get_option<std::string>("bc_x1") != "precursor") {
        endrun("Synthetic turbulent inflow requires bc_x1=precursor");
      }
      int const nz = coupler.get_nz();
      if (u_mean_in.extent(0) != nz || v_mean_in.extent(0) != nz || turbulence_intensity_in.extent(0) != nz) {
        endrun("Synthetic turbulent inflow profiles must have extent nz");
      }
      if (config_in.num_eddies <= 0) endrun("Synthetic turbulent inflow requires num_eddies > 0");
      config = config_in;
      halo_size = Dycore::hs;
      x_ghost_min = -halo_size*coupler.get_dx();
      wall_y1 = coupler.get_option<std::string>("bc_y1") == "wall_free_slip";
      wall_y2 = coupler.get_option<std::string>("bc_y2") == "wall_free_slip";
      wall_z1 = coupler.get_option<std::string>("bc_z1") == "wall_free_slip";
      wall_z2 = coupler.get_option<std::string>("bc_z2") == "wall_free_slip";
      periodic_y = coupler.get_option<std::string>("bc_y1") == "periodic" &&
                   coupler.get_option<std::string>("bc_y2") == "periodic";
      outer_length = config.outer_length > 0 ? config.outer_length :
                                             0.25_fp*std::min(coupler.get_ylen(),coupler.get_zlen());
      wall_decay_length = config.wall_decay_length > 0 ? config.wall_decay_length : outer_length/2;
      if (outer_length <= 0 || wall_decay_length <= 0) endrun("Synthetic turbulent inflow lengths must be positive");

      u_mean = real1d("synthetic_turbulent_inflow_u_mean",nz);
      v_mean = real1d("synthetic_turbulent_inflow_v_mean",nz);
      turbulence_intensity = real1d("synthetic_turbulent_inflow_intensity",nz);
      u_mean_in.deep_copy_to(u_mean);
      v_mean_in.deep_copy_to(v_mean);
      turbulence_intensity_in.deep_copy_to(turbulence_intensity);
      auto u_host  = u_mean.createHostCopy();
      auto v_host  = v_mean.createHostCopy();
      auto ti_host = turbulence_intensity.createHostCopy();
      for (int k = 0; k < nz; k++) {
        if (u_host(k) <= 0) endrun("Synthetic turbulent inflow requires positive u_mean at every level");
        if (ti_host(k) < 0) endrun("Synthetic turbulent inflow intensity must be nonnegative");
        if ((wall_y1 || wall_y2) && std::abs(v_host(k)) > 1.e-12) {
          endrun("Synthetic turbulent inflow requires v_mean=0 when a spanwise boundary is wall_free_slip");
        }
      }

      auto dz_host = coupler.get_dz().createHostCopy();
      smallest_length = 4*std::max(coupler.get_dx(),coupler.get_dy());
      for (int k = 0; k < nz; k++) {
        real const cutoff = 4*std::max(coupler.get_dx(),std::max(coupler.get_dy(),dz_host(k)));
        smallest_length = std::max(smallest_length,cutoff);
      }
      if (outer_length <= smallest_length) {
        endrun("Synthetic turbulent inflow outer_length must exceed the four-cell cutoff");
      }
      num_scales = 0;
      for (real band_outer = outer_length; band_outer > smallest_length; band_outer /= 2) num_scales++;
      if (config.num_eddies < num_scales) {
        endrun("Synthetic turbulent inflow requires at least one eddy per inertial-range scale");
      }
      realHost1d diameters_host("synthetic_turbulent_inflow_scale_diameter_host",num_scales);
      realHost1d fractions_host("synthetic_turbulent_inflow_scale_fraction_host",num_scales);
      realHost1d amplitudes_host("synthetic_turbulent_inflow_scale_amplitude_host",num_scales);
      real const denominator = std::pow(outer_length,2._fp/3._fp)-std::pow(smallest_length,2._fp/3._fp);
      for (int scale = 0; scale < num_scales; scale++) {
        real const band_outer = outer_length/std::pow(2._fp,scale);
        real const band_inner = std::max(smallest_length,band_outer/2);
        diameters_host(scale) = band_outer;
        fractions_host(scale) = (std::pow(band_outer,2._fp/3._fp)-std::pow(band_inner,2._fp/3._fp))/denominator;
        amplitudes_host(scale) = 1;
      }
      scale_diameter = diameters_host.createDeviceCopy();
      scale_fraction = fractions_host.createDeviceCopy();
      scale_amplitude = amplitudes_host.createDeviceCopy();

      base_column = dycore.compute_average_ghost_column(coupler);
      eddy_state_host = realHost2d("synthetic_turbulent_inflow_state",config.num_eddies,num_eddy_properties);
      eddies = real2d("synthetic_turbulent_inflow_eddies",config.num_eddies,num_device_properties);
      auto zmid_host = coupler.get_zmid().createHostCopy();
      for (int n = 0; n < config.num_eddies; n++) {
        randomize_eddy(n,0,diameters_host,coupler.get_ylen(),coupler.get_zlen(),true);
      }
      update_device_eddies(zmid_host);
      calibrate_scales(coupler);
      update_device_eddies(zmid_host);
      calibrate_total(coupler);
      register_restart(coupler);
      initialized = true;
    }

    // Fill dycore_ghost_x1 for the complete physical step, including every acoustic subcycle and RK stage, then
    // advance the retained eddy ensemble to the end of the physical step. Call immediately before dycore.time_step.
    template <class Dycore>
    void apply(core::Coupler &coupler, Dycore &dycore, real dt) {
      if (! initialized) endrun("Synthetic turbulent inflow apply called before init");
      if (dt <= 0) endrun("Synthetic turbulent inflow requires dt > 0");
      using yakl::SimpleBounds;
      using FLOC = typename Dycore::FLOC;
      real const dt_stable = dycore.compute_time_step(coupler);
      int const ncycles = static_cast<int>(std::ceil(dt/dt_stable));
      real const dt_dyn = dt/ncycles;
      dycore.ensure_dycore_max_cycles(coupler,ncycles-1);
      int const nstages = coupler.get_option<int>("dycore_num_stages");
      std::string const time_stepper = coupler.get_option<std::string>("dycore_time_stepper","ssprk3");
      int stepper = 0;
      if      (time_stepper == "linrk3" ) stepper = 1;
      else if (time_stepper == "linrk4" ) stepper = 2;
      else if (time_stepper == "ssprk3") stepper = 3;
      else endrun("Synthetic turbulent inflow encountered an unsupported dycore time stepper");

      int const px = coupler.get_px();
      int const nz = coupler.get_nz();
      int const ny = coupler.get_ny();
      int const j_beg = coupler.get_j_beg();
      int const num_fields = base_column.extent(0);
      real const dx = coupler.get_dx();
      real const dy = coupler.get_dy();
      real const ylen = coupler.get_ylen();
      real const zlen = coupler.get_zlen();
      auto const zmid = coupler.get_zmid();
      auto const base = realConst2d(base_column);
      auto const u_profile = realConst1d(u_mean);
      auto const v_profile = realConst1d(v_mean);
      auto const eddies_device = realConst2d(eddies);
      bool const wy1 = wall_y1;
      bool const wy2 = wall_y2;
      bool const wz1 = wall_z1;
      bool const wz2 = wall_z2;
      bool const py  = periodic_y;
      real const decay = wall_decay_length;
      real const scale = calibration;

      if (px == 0) {
        auto ghost = coupler.get_data_manager_readwrite().template get<FLOC,6>("dycore_ghost_x1");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<6>(ncycles,nstages,num_fields,nz,ny,halo_size) ,
                                                KOKKOS_LAMBDA (int cycle, int stage, int l, int k, int j, int ii) {
          ghost(cycle,stage,l,k,j,ii) = static_cast<FLOC>(base(l,k));
        });
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(ncycles,nstages,nz,ny,halo_size) ,
                                                KOKKOS_LAMBDA (int cycle, int stage, int k, int j, int ii) {
          real stage_fraction = 0;
          if (stepper == 1) {
            if      (stage == 1) stage_fraction = 1._fp/3._fp;
            else if (stage == 2) stage_fraction = 1._fp/2._fp;
          } else if (stepper == 2) {
            if      (stage == 1) stage_fraction = 1._fp/4._fp;
            else if (stage == 2) stage_fraction = 1._fp/3._fp;
            else if (stage == 3) stage_fraction = 1._fp/2._fp;
          } else {
            if      (stage == 1) stage_fraction = 1;
            else if (stage == 2) stage_fraction = 1._fp/2._fp;
          }
          real const time_offset = (cycle+stage_fraction)*dt_dyn;
          real const x = -(ii+0.5_fp)*dx;
          real const y = (j_beg+j+0.5_fp)*dy;
          real u, v, w;
          evaluate_velocity(x,y,zmid(k),ylen,zlen,decay,wy1,wy2,wz1,wz2,py,scale,eddies_device,
                            time_offset,-1,u,v,w);
          ghost(cycle,stage,Dycore::idU,k,j,ii) = static_cast<FLOC>(u_profile(k)+u);
          ghost(cycle,stage,Dycore::idV,k,j,ii) = static_cast<FLOC>(v_profile(k)+v);
          ghost(cycle,stage,Dycore::idW,k,j,ii) = static_cast<FLOC>(w);
        });
      }

      // A height-dependent correction to u that is constant in x and y has zero divergence and removes finite-eddy
      // fluctuations in bulk inflow. Compute it at the innermost ghost center and apply the same value throughout
      // the normal ghost thickness; computing a separate correction at each ii would introduce an x derivative.
      // The analogous v correction is omitted when y is a wall-normal direction.
      real4d means("synthetic_turbulent_inflow_means",ncycles,nstages,2,nz);
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(ncycles,nstages,2,nz) ,
                                              KOKKOS_LAMBDA (int cycle, int stage, int l, int k) {
        means(cycle,stage,l,k) = 0;
      });
      if (px == 0) {
        auto ghost = coupler.get_data_manager_readonly().template get<FLOC const,6>("dycore_ghost_x1");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(ncycles,nstages,2,nz) ,
                                                KOKKOS_LAMBDA (int cycle, int stage, int l, int k) {
          real const profile = l == 0 ? u_profile(k) : v_profile(k);
          int const field = l == 0 ? Dycore::idU : Dycore::idV;
          for (int j = 0; j < ny; j++) means(cycle,stage,l,k) += ghost(cycle,stage,field,k,j,0)-profile;
        });
      }
      means = coupler.get_parallel_comm().all_reduce(means,MPI_SUM,"synthetic_inflow_mean");
      if (px == 0) {
        auto ghost = coupler.get_data_manager_readwrite().template get<FLOC,6>("dycore_ghost_x1");
        bool const correct_v = !wall_y1 && !wall_y2;
        real const inverse_ny = 1._fp/coupler.get_ny_glob();
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(ncycles,nstages,nz,ny,halo_size) ,
                                                KOKKOS_LAMBDA (int cycle, int stage, int k, int j, int ii) {
          ghost(cycle,stage,Dycore::idU,k,j,ii) -= static_cast<FLOC>(means(cycle,stage,0,k)*inverse_ny);
          if (correct_v) {
            ghost(cycle,stage,Dycore::idV,k,j,ii) -= static_cast<FLOC>(means(cycle,stage,1,k)*inverse_ny);
          }
        });
      }
      advance_eddies(coupler,dt);
      update_device_eddies(coupler.get_zmid().createHostCopy());
    }
  };

}

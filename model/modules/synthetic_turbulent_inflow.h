#pragma once

#include "coupler.h"

#include <cstdint>
#include <vector>

namespace modules {

  // Generates homogeneous tank-style turbulence from smooth, overlapping two-dimensional random octaves. The
  // finest octave is cellwise white noise; coarser octaves are translation-invariant moving-window smooths of
  // independent full-resolution white noise.
  // A single fixed-width digital filter is applied to every octave before three vector potentials and their smooth
  // time derivatives define velocity as a space-time curl.
  // Taylor's hypothesis supplies d/dx=-d/dt/U, making the convected extension discretely divergence free while
  // requiring only transverse derivatives at the inlet.
  //
  // Octave spacings are 1,2,4,... finest-grid cells and stop at a user-selected fraction of the shorter transverse
  // domain length. The conservative default is one tenth, avoiding synthetic energy-producing scales that approach
  // the wall separation.
  // Random noise and stochastic potential state extend one curl halo beyond nonperiodic transverse boundaries, so
  // no octave stencil stops or changes shape at a wall. Ten stationary realizations measure the complete
  // filtered discrete-curl response and calibrate each octave to its Kolmogorov energy share. The implementation
  // deliberately targets uniform, shear-free wind- and water-tank configurations.
  //
  // This class registers callbacks that retain its address. Construct it in the same
  // scope as the Coupler, initialize it once, and do not move it afterward.
  class DigitalFilterTurbulentInflow {
  public:
    struct Config {
      int random_seed                  = 0;    // Decomposition-independent deterministic random seed
      int streamwise_length_cells      = 5;    // Shared streamwise filter length [finest-grid cells]
      int spanwise_length_cells        = 5;    // Shared spanwise filter length [finest-grid cells]
      int vertical_length_cells        = 5;    // Shared vertical filter length [finest-grid cells]
      real maximum_length_fraction     = 0.10; // Largest octave length / shorter transverse domain length
      int calibration_realizations     = 10;   // Sequential stationary samples used for TI calibration
    };

  private:
    Config config;
    bool initialized = false;
    int halo_size    = 0;
    int random_count = 0;
    int num_octaves  = 0;
    bool periodic_y  = false;
    bool periodic_z  = false;
    real mean_u      = 0;
    real intensity   = 0;
    real time_scale  = 0;

    struct OctaveKernel {
      int octave_ny = 0;
      int octave_nz = 0;
      int combined_ny = 0;
      int combined_nz = 0;
      real1d octave_y;
      real1d octave_z;
      real1d combined_y;
      real1d combined_z;
    };
    std::vector<OctaveKernel> octave_kernels;
    real1d octave_lambdas;
    real1d octave_axial_scales;
    real1d octave_transverse_scales;
    real2d base_column;
    // (octave,Ax,Ay,Az,dAx/dt,dAy/dt,dAz/dt,z with exterior halos,local y with exterior halos)
    real4d state;

    KOKKOS_INLINE_FUNCTION static std::uint64_t mix_bits_device(std::uint64_t value) {
      value += 0x9e3779b97f4a7c15ULL;
      value  = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
      value  = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
      return value ^ (value >> 31);
    }

    KOKKOS_INLINE_FUNCTION static real random_unit_variance(int seed, int count, int octave, int field, int k, int j) {
      auto value = static_cast<std::uint64_t>(seed);
      value ^= static_cast<std::uint64_t>(count + 1) * 0xd2b74407b1ce6e93ULL;
      value ^= static_cast<std::uint64_t>(octave + 1) * 0xa24baed4963ee407ULL;
      value ^= static_cast<std::uint64_t>(field + 1) * 0xca5a826395121157ULL;
      value ^= static_cast<std::uint64_t>(static_cast<std::int64_t>(k)) * 0x9e3779b97f4a7c15ULL;
      value ^= static_cast<std::uint64_t>(static_cast<std::int64_t>(j)) * 0x94d049bb133111ebULL;
      real const uniform = static_cast<real>(mix_bits_device(value) >> 11) *
                           (static_cast<real>(1) / static_cast<real>(9007199254740992ULL));
      return std::sqrt(static_cast<real>(3))*(2*uniform-1);
    }

    KOKKOS_INLINE_FUNCTION static int wrap_index(int index, int extent) {
      int wrapped = index % extent;
      if (wrapped < 0) wrapped += extent;
      return wrapped;
    }

    KOKKOS_INLINE_FUNCTION static real indexed_noise(int seed, int count, int octave, int field, int k, int j,
                                                      int nz, int ny, bool wrap_z, bool wrap_y) {
      int const source_k = wrap_z ? wrap_index(k,nz) : k;
      int const source_j = wrap_y ? wrap_index(j,ny) : j;
      return random_unit_variance(seed,count,octave,field,source_k,source_j);
    }

    static bool nearly_equal(real lhs, real rhs) {
      return std::abs(lhs-rhs) <= 1.e-12*std::max(static_cast<real>(1),std::max(std::abs(lhs),std::abs(rhs)));
    }

    realHost1d initialize_filter(real length_cells, int &half_width, std::string const &label) {
      half_width = static_cast<int>(std::ceil(2*length_cells));
      realHost1d host(label+"_host",2*half_width+1);
      real sum_squared = 0;
      for (int offset = -half_width; offset <= half_width; offset++) {
        real const value = std::exp(-M_PI*std::abs(offset)/length_cells);
        host(offset+half_width) = value;
        sum_squared += value*value;
      }
      for (int offset = 0; offset < host.extent(0); offset++) host(offset) /= std::sqrt(sum_squared);
      return host;
    }

    realHost1d initialize_octave_filter(int level, int &half_width, std::string const &label) {
      half_width = level == 1 ? 0 : 2*level-1;
      realHost1d host(label,2*half_width+1);
      real sum_squared = 0;
      for (int offset = -half_width; offset <= half_width; offset++) {
        real const distance = std::abs(static_cast<real>(offset))/level;
        real value;
        if (distance < 1) {
          value = static_cast<real>(2)/3-distance*distance+static_cast<real>(0.5)*distance*distance*distance;
        } else {
          value = std::pow(2-distance,3)/6;
        }
        host(offset+half_width) = value;
        sum_squared += value*value;
      }
      for (int index = 0; index < host.extent(0); index++) host(index) /= std::sqrt(sum_squared);
      return host;
    }

    realHost1d convolve_filters(realHost1d const &first, int first_half, realHost1d const &second, int second_half,
                                std::string const &label) {
      int const half_width = first_half+second_half;
      realHost1d result(label,2*half_width+1);
      result = 0;
      for (int first_offset = -first_half; first_offset <= first_half; first_offset++) {
        for (int second_offset = -second_half; second_offset <= second_half; second_offset++) {
          result(first_offset+second_offset+half_width) +=
            first(first_offset+first_half)*second(second_offset+second_half);
        }
      }
      return result;
    }

  public:
    // NVCC requires functions enclosing extended host/device lambdas to be publicly accessible.
    real3d create_smoothed_octave(core::Coupler const &coupler, int count, int octave, real1d const &kernel_y,
                                  int ny_pad, real1d const &kernel_z, int nz_pad, bool include_exterior) const {
      using yakl::SimpleBounds;
      int const nz      = coupler.get_nz();
      int const ny      = coupler.get_ny();
      int const ny_glob = coupler.get_ny_glob();
      int const j_beg   = coupler.get_j_beg();
      int const seed    = config.random_seed;
      int const exterior = include_exterior ? 1 : 0;
      int const output_nz = nz+2*exterior;
      int const output_ny = ny+2*exterior;
      bool const wrap_y = periodic_y;
      bool const wrap_z = periodic_z;
      auto const weights_y = realConst1d(kernel_y);
      auto const weights_z = realConst1d(kernel_z);

      real3d filtered_y("digital_filter_inflow_filtered_y",6,output_nz+2*nz_pad,output_ny);
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(6,output_nz+2*nz_pad,output_ny) ,
                                              KOKKOS_LAMBDA (int field, int kp, int jp) {
        int const target_k = kp-nz_pad-exterior;
        int const target_j = j_beg+jp-exterior;
        real value = 0;
        for (int offset = -ny_pad; offset <= ny_pad; offset++) {
          value += weights_y(offset+ny_pad)*indexed_noise(seed,count,octave,field,target_k,target_j+offset,
                                                          nz,ny_glob,wrap_z,wrap_y);
        }
        filtered_y(field,kp,jp) = value;
      });
      real3d innovation("digital_filter_inflow_smoothed_octave",6,output_nz,output_ny);
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(6,output_nz,output_ny) ,
                                              KOKKOS_LAMBDA (int field, int k, int j) {
        real value = 0;
        for (int offset = -nz_pad; offset <= nz_pad; offset++) {
          value += weights_z(offset+nz_pad)*filtered_y(field,k+offset+nz_pad,j);
        }
        innovation(field,k,j) = value;
      });
      return innovation;
    }

    real4d create_unfiltered_innovation(core::Coupler const &coupler, int count) const {
      using yakl::SimpleBounds;
      int const nz = coupler.get_nz();
      int const ny = coupler.get_ny();
      real4d innovation("digital_filter_inflow_unfiltered_innovation",num_octaves,6,nz,ny);
      for (int octave = 0; octave < num_octaves; octave++) {
        auto const &kernel = octave_kernels[octave];
        auto filtered = create_smoothed_octave(coupler,count,octave,kernel.octave_y,kernel.octave_ny,
                                                kernel.octave_z,kernel.octave_nz,false);
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(6,nz,ny) ,
                                                KOKKOS_LAMBDA (int field, int k, int j) {
          innovation(octave,field,k,j) = filtered(field,k,j);
        });
      }
      return innovation;
    }

    real4d create_filtered_innovation(core::Coupler const &coupler, int count) const {
      using yakl::SimpleBounds;
      int const nz = coupler.get_nz();
      int const ny = coupler.get_ny();
      real4d innovation("digital_filter_inflow_innovation",num_octaves,6,nz+2,ny+2);
      for (int octave = 0; octave < num_octaves; octave++) {
        auto const &kernel = octave_kernels[octave];
        auto filtered = create_smoothed_octave(coupler,count,octave,kernel.combined_y,kernel.combined_ny,
                                                kernel.combined_z,kernel.combined_nz,true);
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(6,nz+2,ny+2) ,
                                                KOKKOS_LAMBDA (int field, int k, int j) {
          innovation(octave,field,k,j) = filtered(field,k,j);
        });
      }
      return innovation;
    }

    real3d compute_raw_velocity(core::Coupler const &coupler, int selected_octave = -1) {
      using yakl::SimpleBounds;
      int const nz      = coupler.get_nz();
      int const ny      = coupler.get_ny();
      int const octave_count = num_octaves;
      real const dy     = coupler.get_dy();
      real const zlen   = coupler.get_zlen();
      bool const wrap_z = periodic_z;
      auto const dz = coupler.get_dz();
      auto const zint = coupler.get_zint();
      auto const zmid = coupler.get_zmid();
      auto const axial_scales = realConst1d(octave_axial_scales);
      auto const transverse_scales = realConst1d(octave_transverse_scales);
      auto state_device = state;
      real3d potential("digital_filter_inflow_potential",6,nz+2,ny+2);
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(6,nz+2,ny+2) ,
                                              KOKKOS_LAMBDA (int field, int k, int j) {
        real value = 0;
        for (int octave = 0; octave < octave_count; octave++) {
          if (selected_octave < 0 || octave == selected_octave) {
            real const scale = selected_octave >= 0 ? 1 :
                               (field % 3 == 0 ? axial_scales(octave) : transverse_scales(octave));
            value += scale*state_device(octave,field,k,j);
          }
        }
        potential(field,k,j) = value;
      });

      // gradient(0,...)=d/dy and gradient(1,...)=d/dz. Applying identical linear operators to all potential
      // components makes the discrete mixed derivatives commute in the interior. The vertical finite-volume
      // operator uses actual interface and center heights, so it remains conservative on a stretched grid.
      real4d gradient("digital_filter_inflow_potential_gradient",2,6,nz,ny);
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(6,nz,ny) ,
                                              KOKKOS_LAMBDA (int field, int k, int j) {
        real const center = potential(field,k+1,j+1);
        real lower;
        real upper;
        real lower_face;
        real upper_face;
        if (k > 0) {
          lower = potential(field,k,j+1);
          real const weight = (zint(k)-zmid(k-1))/(zmid(k)-zmid(k-1));
          lower_face = lower+weight*(center-lower);
        } else {
          lower = potential(field,0,j+1);
          real const lower_z = wrap_z ? zmid(nz-1)-zlen : zmid(0)-dz(0);
          real const weight = (zint(0)-lower_z)/(zmid(0)-lower_z);
          lower_face = lower+weight*(center-lower);
        }
        if (k+1 < nz) {
          upper = potential(field,k+2,j+1);
          real const weight = (zint(k+1)-zmid(k))/(zmid(k+1)-zmid(k));
          upper_face = center+weight*(upper-center);
        } else {
          upper = potential(field,nz+1,j+1);
          real const upper_z = wrap_z ? zmid(0)+zlen : zmid(nz-1)+dz(nz-1);
          real const weight = (zint(nz)-zmid(nz-1))/(upper_z-zmid(nz-1));
          upper_face = center+weight*(upper-center);
        }
        gradient(0,field,k,j) = (potential(field,k+1,j+2)-potential(field,k+1,j))/(2*dy);
        gradient(1,field,k,j) = (upper_face-lower_face)/dz(k);
      });

      // Extra components retain the two contributions to v and w for initialization-only energy calibration.
      real3d velocity("digital_filter_inflow_raw_velocity",8,nz,ny);
      real const inverse_u = static_cast<real>(1)/mean_u;
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ny) , KOKKOS_LAMBDA (int k, int j) {
        real const v_axial     =  gradient(1,0,k,j);
        real const v_temporal  =  potential(5,k+1,j+1)*inverse_u;
        real const w_temporal  = -potential(4,k+1,j+1)*inverse_u;
        real const w_axial     = -gradient(0,0,k,j);
        velocity(0,k,j) = gradient(0,2,k,j)-gradient(1,1,k,j);
        velocity(1,k,j) = v_axial+v_temporal;
        velocity(2,k,j) = w_temporal+w_axial;
        velocity(3,k,j) = gradient(0,5,k,j)-gradient(1,4,k,j); // du'/dt
        velocity(4,k,j) = v_axial;
        velocity(5,k,j) = v_temporal;
        velocity(6,k,j) = w_temporal;
        velocity(7,k,j) = w_axial;
      });
      return velocity;
    }

    void set_stationary_state(core::Coupler const &coupler, int count) {
      using yakl::SimpleBounds;
      auto innovation = create_filtered_innovation(coupler,count);
      int const nz = coupler.get_nz();
      int const ny = coupler.get_ny();
      auto const lambdas = realConst1d(octave_lambdas);
      auto state_device = state;
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_octaves,3,nz+2,ny+2) ,
                                              KOKKOS_LAMBDA (int octave, int component, int k, int j) {
        state_device(octave,component  ,k,j) = innovation(octave,component  ,k,j);
        state_device(octave,component+3,k,j) = lambdas(octave)*innovation(octave,component+3,k,j);
      });
    }

    void calibrate(core::Coupler const &coupler) {
      using yakl::SimpleBounds;
      int const px = coupler.get_px();
      int const nz = coupler.get_nz();
      int const ny = coupler.get_ny();
      realHost1d local_statistics_host("digital_filter_inflow_local_calibration_host",4*num_octaves);
      local_statistics_host = 0;
      for (int sample = 0; sample < config.calibration_realizations; sample++) {
        // Negative counters provide calibration-only random streams and leave production/restart sequencing unchanged.
        set_stationary_state(coupler,-sample-1);
        for (int octave = 0; octave < num_octaves; octave++) {
          auto velocity = compute_raw_velocity(coupler,octave);
          real2d statistics("digital_filter_inflow_calibration_statistics",4,nz*ny);
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ny) , KOKKOS_LAMBDA (int k, int j) {
            int const index = k*ny+j;
            statistics(0,index) = px == 0 ? velocity(0,k,j)*velocity(0,k,j) : 0;
            statistics(1,index) = px == 0 ? velocity(4,k,j)*velocity(4,k,j) +
                                            velocity(7,k,j)*velocity(7,k,j) : 0;
            statistics(2,index) = px == 0 ? velocity(5,k,j)*velocity(5,k,j) +
                                            velocity(6,k,j)*velocity(6,k,j) : 0;
            statistics(3,index) = px == 0 ? 2*(velocity(4,k,j)*velocity(5,k,j) +
                                               velocity(7,k,j)*velocity(6,k,j)) : 0;
          });
          for (int statistic = 0; statistic < 4; statistic++) {
            local_statistics_host(4*octave+statistic) +=
              yakl::intrinsics::sum(statistics.slice<1>(statistic,0));
          }
        }
      }
      auto local_statistics = local_statistics_host.createDeviceCopy();
      auto global_statistics = coupler.get_parallel_comm().all_reduce(
                                 local_statistics,MPI_SUM,"digital_filter_inflow_calibration");
      auto global_statistics_host = global_statistics.createHostCopy();
      real const count = config.calibration_realizations*nz*coupler.get_ny_glob();
      real const total_target = intensity*intensity*mean_u*mean_u;
      real fraction_sum = 0;
      for (int octave = 0; octave < num_octaves; octave++) {
        fraction_sum += std::pow(static_cast<real>(1 << octave),static_cast<real>(2)/3);
      }
      realHost1d axial_scales_host("digital_filter_inflow_axial_scales_host",num_octaves);
      realHost1d transverse_scales_host("digital_filter_inflow_transverse_scales_host",num_octaves);
      for (int octave = 0; octave < num_octaves; octave++) {
        real const target = total_target*std::pow(static_cast<real>(1 << octave),static_cast<real>(2)/3)/fraction_sum;
        real const actual_u   = global_statistics_host(4*octave  )/count;
        real const axial      = global_statistics_host(4*octave+1)/count;
        real const transverse = global_statistics_host(4*octave+2)/count;
        real const cross       = global_statistics_host(4*octave+3)/count;
        if (target == 0) {
          axial_scales_host(octave) = 0;
          transverse_scales_host(octave) = 0;
          continue;
        }
        if (actual_u <= 0 || axial <= 0) {
          endrun("Digital-filter turbulent inflow produced zero per-octave calibration variance");
        }
        real const transverse_scale = std::sqrt(target/actual_u);
        real const linear = cross*transverse_scale;
        real const constant = transverse*transverse_scale*transverse_scale-2*target;
        real const discriminant = linear*linear-4*axial*constant;
        if (discriminant < 0) endrun("Digital-filter turbulent inflow could not calibrate octave energy");
        real const axial_scale = (-linear+std::sqrt(discriminant))/(2*axial);
        if (axial_scale <= 0) endrun("Digital-filter turbulent inflow calibration produced a nonpositive scale");
        axial_scales_host(octave) = axial_scale;
        transverse_scales_host(octave) = transverse_scale;
      }
      axial_scales_host.deep_copy_to(octave_axial_scales);
      transverse_scales_host.deep_copy_to(octave_transverse_scales);
    }

    void advance(core::Coupler const &coupler, real dt) {
      using yakl::SimpleBounds;
      auto innovation = create_filtered_innovation(coupler,random_count);
      int const nz = coupler.get_nz();
      int const ny = coupler.get_ny();
      // This critically damped second-order stochastic process has stationary Var(A)=1, Var(dA/dt)=lambda^2,
      // and correlation (1+lambda*t)exp(-lambda*t). Each octave uses the Taylor decay rate lambda=U/L.
      auto const lambdas = realConst1d(octave_lambdas);
      auto state_device = state;
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_octaves,3,nz+2,ny+2) ,
                                              KOKKOS_LAMBDA (int octave, int component, int k, int j) {
        real const lambda = lambdas(octave);
        real const decay = std::exp(-lambda*dt);
        real const phi00 = decay*(1+lambda*dt);
        real const phi01 = decay*dt;
        real const phi10 = -decay*lambda*lambda*dt;
        real const phi11 = decay*(1-lambda*dt);
        real const covariance00 = std::max(static_cast<real>(0),1-(phi00*phi00+lambda*lambda*phi01*phi01));
        real const covariance01 = -(phi00*phi10+lambda*lambda*phi01*phi11);
        real const covariance11 = std::max(static_cast<real>(0),lambda*lambda-
                                                               (phi10*phi10+lambda*lambda*phi11*phi11));
        real const noise00 = std::sqrt(covariance00);
        real const noise10 = noise00 > 0 ? covariance01/noise00 : 0;
        real const noise11 = std::sqrt(std::max(static_cast<real>(0),covariance11-noise10*noise10));
        real const potential = state_device(octave,component  ,k,j);
        real const rate      = state_device(octave,component+3,k,j);
        state_device(octave,component  ,k,j) = phi00*potential + phi01*rate +
                                                  noise00*innovation(octave,component,k,j);
        state_device(octave,component+3,k,j) = phi10*potential + phi11*rate +
                                                  noise10*innovation(octave,component  ,k,j) +
                                                  noise11*innovation(octave,component+3,k,j);
      });
      random_count++;
    }

    real4d gather_global_state(core::Coupler const &coupler) const {
      using yakl::SimpleBounds;
      int const px      = coupler.get_px();
      int const nz      = coupler.get_nz();
      int const ny      = coupler.get_ny();
      int const ny_glob = coupler.get_ny_glob();
      int const j_beg   = coupler.get_j_beg();
      auto state_device = state;
      real4d local("digital_filter_inflow_global_state_local",num_octaves,6,nz+2,ny_glob+2);
      local = 0;
      if (px == 0) {
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_octaves,6,nz+2,ny) ,
                                                KOKKOS_LAMBDA (int octave, int field, int k, int j) {
          local(octave,field,k,j_beg+j+1) = state_device(octave,field,k,j+1);
        });
        if (j_beg == 0) {
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_octaves,6,nz+2) ,
                                                  KOKKOS_LAMBDA (int octave, int field, int k) {
            local(octave,field,k,0) = state_device(octave,field,k,0);
          });
        }
        if (j_beg+ny == ny_glob) {
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_octaves,6,nz+2) ,
                                                  KOKKOS_LAMBDA (int octave, int field, int k) {
            local(octave,field,k,ny_glob+1) = state_device(octave,field,k,ny+1);
          });
        }
      }
      return coupler.get_parallel_comm().all_reduce(local,MPI_SUM,"digital_filter_inflow_gather_state");
    }

    void scatter_global_state(core::Coupler const &coupler, real4d const &global) {
      using yakl::SimpleBounds;
      int const nz    = coupler.get_nz();
      int const ny    = coupler.get_ny();
      int const j_beg = coupler.get_j_beg();
      auto state_device = state;
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_octaves,6,nz+2,ny+2) ,
                                              KOKKOS_LAMBDA (int octave, int field, int k, int j) {
        state_device(octave,field,k,j) = global(octave,field,k,j_beg+j);
      });
    }

  private:
    void register_restart(core::Coupler &coupler) {
      coupler.register_write_output_module( [this] (core::Coupler &coupler, core::FileIO &nc) {
        auto global_state = gather_global_state(coupler);
        auto base_host = base_column.createHostCopy();
        auto axial_scales_host = octave_axial_scales.createHostCopy();
        auto transverse_scales_host = octave_transverse_scales.createHostCopy();
        nc.redef();
        nc.create_dim("digital_filter_inflow_octave",num_octaves);
        nc.create_dim("digital_filter_inflow_field",6);
        nc.create_dim("digital_filter_inflow_base_field",base_column.extent(0));
        nc.create_dim("digital_filter_inflow_state_z",coupler.get_nz()+2);
        nc.create_dim("digital_filter_inflow_state_y",coupler.get_ny_glob()+2);
        nc.create_var<real>("digital_filter_inflow_state",
                            {"digital_filter_inflow_octave","digital_filter_inflow_field",
                             "digital_filter_inflow_state_z","digital_filter_inflow_state_y"});
        nc.create_var<real>("digital_filter_inflow_base_column",{"digital_filter_inflow_base_field","z"});
        nc.create_var<real>("digital_filter_inflow_axial_potential_scale",{"digital_filter_inflow_octave"});
        nc.create_var<real>("digital_filter_inflow_transverse_potential_scale",{"digital_filter_inflow_octave"});
        nc.create_var<int>("digital_filter_inflow_random_count",{});
        nc.enddef();
        nc.begin_indep_data();
        if (coupler.is_mainproc()) {
          nc.write(global_state,"digital_filter_inflow_state");
          nc.write(base_host,"digital_filter_inflow_base_column");
          nc.write(axial_scales_host,"digital_filter_inflow_axial_potential_scale");
          nc.write(transverse_scales_host,"digital_filter_inflow_transverse_potential_scale");
          nc.write(random_count,"digital_filter_inflow_random_count");
        }
        nc.end_indep_data();
      });
      coupler.register_overwrite_with_restart_module( [this] (core::Coupler &coupler, core::FileIO &nc) {
        real4d global_state("digital_filter_inflow_restart_state",num_octaves,6,coupler.get_nz()+2,
                            coupler.get_ny_glob()+2);
        realHost2d base_host("digital_filter_inflow_restart_base",base_column.extent(0),base_column.extent(1));
        realHost1d axial_scales_host("digital_filter_inflow_restart_axial_scales",num_octaves);
        realHost1d transverse_scales_host("digital_filter_inflow_restart_transverse_scales",num_octaves);
        nc.begin_indep_data();
        if (coupler.is_mainproc()) {
          nc.read(global_state,"digital_filter_inflow_state");
          nc.read(base_host,"digital_filter_inflow_base_column");
          nc.read(axial_scales_host,"digital_filter_inflow_axial_potential_scale");
          nc.read(transverse_scales_host,"digital_filter_inflow_transverse_potential_scale");
          nc.read(random_count,"digital_filter_inflow_random_count");
        }
        nc.end_indep_data();
        auto const comm = coupler.get_parallel_comm();
        comm.broadcast(global_state);
        comm.broadcast(base_host);
        comm.broadcast(axial_scales_host);
        comm.broadcast(transverse_scales_host);
        comm.broadcast(random_count);
        base_host.deep_copy_to(base_column);
        axial_scales_host.deep_copy_to(octave_axial_scales);
        transverse_scales_host.deep_copy_to(octave_transverse_scales);
        scatter_global_state(coupler,global_state);
      });
    }

  public:
    DigitalFilterTurbulentInflow() = default;
    DigitalFilterTurbulentInflow(DigitalFilterTurbulentInflow const &) = delete;
    DigitalFilterTurbulentInflow &operator=(DigitalFilterTurbulentInflow const &) = delete;
    DigitalFilterTurbulentInflow(DigitalFilterTurbulentInflow &&) = delete;
    DigitalFilterTurbulentInflow &operator=(DigitalFilterTurbulentInflow &&) = delete;

    real get_time_scale() const {
      if (!initialized) endrun("Digital-filter turbulent inflow diagnostics requested before init");
      return time_scale;
    }

    int get_num_octaves() const {
      if (!initialized) endrun("Digital-filter turbulent inflow diagnostics requested before init");
      return num_octaves;
    }

    int get_maximum_octave_level() const {
      if (!initialized) endrun("Digital-filter turbulent inflow diagnostics requested before init");
      return 1 << (num_octaves-1);
    }

    static int compute_maximum_octave_level(real ylen, real zlen, real dy, real maximum_dz,
                                             real maximum_length_fraction) {
      real const cutoff_length = maximum_length_fraction*std::min(ylen,zlen);
      int const cutoff = static_cast<int>(std::floor(cutoff_length/std::max(dy,maximum_dz)));
      if (cutoff < 1) return 0;
      int level = 1;
      while (level <= cutoff/2) level *= 2;
      return level;
    }

    static int compute_num_octaves(real ylen, real zlen, real dy, real maximum_dz, real maximum_length_fraction) {
      int const maximum_level = compute_maximum_octave_level(ylen,zlen,dy,maximum_dz,maximum_length_fraction);
      int count = 0;
      for (int level = 1; level <= maximum_level; level *= 2) {
        count++;
        if (level > maximum_level/2) break;
      }
      return count;
    }

    static real compute_capped_filter_length(real requested_length, real xlen, real ylen, real zlen) {
      return std::min(requested_length,static_cast<real>(0.25)*std::min({xlen,ylen,zlen}));
    }

    template <class Dycore>
    void init(core::Coupler &coupler, Dycore &dycore, real1d const &u_mean_in, real1d const &v_mean_in,
              real1d const &turbulence_intensity_in) {
      init(coupler,dycore,u_mean_in,v_mean_in,turbulence_intensity_in,Config());
    }

    template <class Dycore>
    void init(core::Coupler &coupler, Dycore &dycore, real1d const &u_mean_in, real1d const &v_mean_in,
              real1d const &turbulence_intensity_in, Config const &config_in) {
      if (initialized) endrun("Digital-filter turbulent inflow may only be initialized once");
      if (coupler.get_option<std::string>("bc_x1") != "precursor") {
        endrun("Digital-filter turbulent inflow requires bc_x1=precursor");
      }
      int const nz = coupler.get_nz();
      if (u_mean_in.extent(0) != nz || v_mean_in.extent(0) != nz || turbulence_intensity_in.extent(0) != nz) {
        endrun("Digital-filter turbulent inflow profiles must have extent nz");
      }
      config = config_in;
      if (config.streamwise_length_cells <= 0 || config.spanwise_length_cells <= 0 ||
          config.vertical_length_cells <= 0) {
        endrun("Digital-filter turbulent inflow base filter lengths must be positive cell counts");
      }
      if (config.maximum_length_fraction <= 0 || config.maximum_length_fraction > 1) {
        endrun("Digital-filter turbulent inflow maximum length fraction must be in (0,1]");
      }
      if (config.calibration_realizations <= 0) {
        endrun("Digital-filter turbulent inflow calibration realizations must be positive");
      }
      auto u_host  = u_mean_in.createHostCopy();
      auto v_host  = v_mean_in.createHostCopy();
      auto ti_host = turbulence_intensity_in.createHostCopy();
      mean_u = u_host(0);
      intensity = ti_host(0);
      if (mean_u <= 0) endrun("Digital-filter turbulent inflow requires positive uniform u_mean");
      if (intensity < 0) endrun("Digital-filter turbulent inflow intensity must be nonnegative");
      for (int k = 0; k < nz; k++) {
        if (!nearly_equal(u_host(k),mean_u) || !nearly_equal(v_host(k),0) || !nearly_equal(ti_host(k),intensity)) {
          endrun("Digital-filter turbulent inflow requires constant u_mean, zero v_mean, and constant intensity");
        }
      }
      std::string const bc_y1 = coupler.get_option<std::string>("bc_y1");
      std::string const bc_y2 = coupler.get_option<std::string>("bc_y2");
      std::string const bc_z1 = coupler.get_option<std::string>("bc_z1");
      std::string const bc_z2 = coupler.get_option<std::string>("bc_z2");
      bool const valid_y = (bc_y1 == "periodic" && bc_y2 == "periodic") ||
                           (bc_y1 == "wall_free_slip" && bc_y2 == "wall_free_slip");
      bool const valid_z = (bc_z1 == "periodic" && bc_z2 == "periodic") ||
                           (bc_z1 == "wall_free_slip" && bc_z2 == "wall_free_slip");
      if (!valid_y || !valid_z) {
        endrun("Digital-filter turbulent inflow requires periodic or wall_free_slip transverse boundary pairs");
      }
      periodic_y = bc_y1 == "periodic";
      periodic_z = bc_z1 == "periodic";

      halo_size = Dycore::hs;
      real const dy = coupler.get_dy();
      auto dz_host = coupler.get_dz().createHostCopy();
      real maximum_dz = 0;
      for (int k = 0; k < nz; k++) maximum_dz = std::max(maximum_dz,dz_host(k));
      int const maximum_level = compute_maximum_octave_level(coupler.get_ylen(),coupler.get_zlen(),dy,maximum_dz,
                                                              config.maximum_length_fraction);
      if (maximum_level < 1) {
        endrun("Digital-filter turbulent inflow maximum length cutoff is smaller than one cell");
      }
      num_octaves = compute_num_octaves(coupler.get_ylen(),coupler.get_zlen(),dy,maximum_dz,
                                        config.maximum_length_fraction);
      real const dx = coupler.get_dx();
      real const streamwise_filter_length = compute_capped_filter_length(
                                               config.streamwise_length_cells*dx,coupler.get_xlen(),
                                               coupler.get_ylen(),coupler.get_zlen());
      real const spanwise_filter_length = compute_capped_filter_length(
                                             config.spanwise_length_cells*dy,coupler.get_xlen(),
                                             coupler.get_ylen(),coupler.get_zlen());
      real const vertical_filter_length = compute_capped_filter_length(
                                             config.vertical_length_cells*maximum_dz,coupler.get_xlen(),
                                             coupler.get_ylen(),coupler.get_zlen());
      int fixed_ny;
      int fixed_nz;
      auto fixed_y = initialize_filter(spanwise_filter_length/dy,fixed_ny,"digital_filter_inflow_filter_y_host");
      auto fixed_z = initialize_filter(vertical_filter_length/maximum_dz,fixed_nz,
                                       "digital_filter_inflow_filter_z_host");
      octave_kernels.clear();
      octave_kernels.reserve(num_octaves);
      for (int octave = 0; octave < num_octaves; octave++) {
        int const level = 1 << octave;
        OctaveKernel kernel;
        auto octave_y = initialize_octave_filter(level,kernel.octave_ny,
                                                  "digital_filter_inflow_octave_y_host");
        auto octave_z = initialize_octave_filter(level,kernel.octave_nz,
                                                  "digital_filter_inflow_octave_z_host");
        auto combined_y = convolve_filters(octave_y,kernel.octave_ny,fixed_y,fixed_ny,
                                            "digital_filter_inflow_combined_y_host");
        auto combined_z = convolve_filters(octave_z,kernel.octave_nz,fixed_z,fixed_nz,
                                            "digital_filter_inflow_combined_z_host");
        kernel.combined_ny = kernel.octave_ny+fixed_ny;
        kernel.combined_nz = kernel.octave_nz+fixed_nz;
        kernel.octave_y = octave_y.createDeviceCopy();
        kernel.octave_z = octave_z.createDeviceCopy();
        kernel.combined_y = combined_y.createDeviceCopy();
        kernel.combined_z = combined_z.createDeviceCopy();
        octave_kernels.push_back(kernel);
      }

      realHost1d lambdas_host("digital_filter_inflow_octave_lambdas_host",num_octaves);
      for (int octave = 0; octave < num_octaves; octave++) {
        real const octave_length = (1 << octave)*dx;
        real const correlation_length = std::max(octave_length,streamwise_filter_length);
        lambdas_host(octave) = mean_u/correlation_length;
      }
      octave_lambdas = lambdas_host.createDeviceCopy();
      time_scale = std::max(maximum_level*dx,streamwise_filter_length)/mean_u;
      octave_axial_scales = real1d("digital_filter_inflow_axial_scales",num_octaves);
      octave_transverse_scales = real1d("digital_filter_inflow_transverse_scales",num_octaves);

      base_column = dycore.compute_average_ghost_column(coupler);
      state = real4d("digital_filter_inflow_state",num_octaves,6,nz+2,coupler.get_ny()+2);
      calibrate(coupler);
      set_stationary_state(coupler,random_count);
      random_count++;
      register_restart(coupler);
      initialized = true;
    }

    // Populate every x1 ghost needed by the dycore, advancing the stochastic state at acoustic-subcycle cadence.
    // A single correlated plane is held fixed across all RK stages within each acoustic subcycle.
    template <class Dycore>
    void apply(core::Coupler &coupler, Dycore &dycore, real dt) {
      using yakl::SimpleBounds;
      using FLOC = typename Dycore::FLOC;
      if (!initialized) endrun("Digital-filter turbulent inflow apply called before init");
      if (dt <= 0) endrun("Digital-filter turbulent inflow requires dt > 0");
      real const dt_stable = dycore.compute_time_step(coupler);
      int const ncycles = static_cast<int>(std::ceil(dt/dt_stable));
      real const dt_dyn = dt/ncycles;
      dycore.ensure_dycore_max_cycles(coupler,ncycles-1);
      int const nstages   = coupler.get_option<int>("dycore_num_stages");
      int const px        = coupler.get_px();
      int const nz        = coupler.get_nz();
      int const ny        = coupler.get_ny();
      int const num_fields = base_column.extent(0);
      auto const base = realConst2d(base_column);

      if (px == 0) {
        auto ghost = coupler.get_data_manager_readwrite().template get<FLOC,6>("dycore_ghost_x1");
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<6>(ncycles,nstages,num_fields,nz,ny,halo_size) ,
                                                KOKKOS_LAMBDA (int cycle, int stage, int field, int k, int j, int ii) {
          ghost(cycle,stage,field,k,j,ii) = static_cast<FLOC>(base(field,k));
        });
      }

      for (int cycle = 0; cycle < ncycles; cycle++) {
        auto velocity = compute_raw_velocity(coupler);
        real const um = mean_u;
        if (px == 0) {
          auto ghost = coupler.get_data_manager_readwrite().template get<FLOC,6>("dycore_ghost_x1");
          yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nstages,nz,ny,halo_size) ,
                                                  KOKKOS_LAMBDA (int stage, int k, int j, int ii) {
            ghost(cycle,stage,Dycore::idU,k,j,ii) = static_cast<FLOC>(um+velocity(0,k,j));
            ghost(cycle,stage,Dycore::idV,k,j,ii) = static_cast<FLOC>(velocity(1,k,j));
            ghost(cycle,stage,Dycore::idW,k,j,ii) = static_cast<FLOC>(velocity(2,k,j));
          });
        }
        advance(coupler,dt_dyn);
      }
    }
  };

}

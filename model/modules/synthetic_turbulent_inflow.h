#pragma once

#include "coupler.h"

#include <cstdint>

namespace modules {

  // Generates homogeneous tank-style turbulence with exponential correlations based on the two-dimensional
  // digital-filter method of Xie and Castro (2008). Three filtered vector potentials and their smooth time
  // derivatives define velocity as a space-time curl. Taylor's hypothesis supplies d/dx=-d/dt/U, making the
  // convected extension discretely divergence free while requiring only transverse derivatives at the inlet.
  //
  // Correlation and wall-decay lengths are specified in cells so their defaults follow the grid's resolvable scales.
  // The streamwise cell count is converted to a Lagrangian time scale with Taylor's hypothesis. The first
  // implementation deliberately targets uniform, shear-free wind- and water-tank configurations.
  //
  // This class registers callbacks that retain its address. Construct it in the same
  // scope as the Coupler, initialize it once, and do not move it afterward.
  class DigitalFilterTurbulentInflow {
  public:
    struct Config {
      int random_seed             = 0; // Decomposition-independent deterministic random seed
      int streamwise_length_cells = 5; // Streamwise integral correlation length [cells]
      int spanwise_length_cells   = 5; // Spanwise integral correlation length [cells]
      int vertical_length_cells   = 5; // Vertical integral correlation length [cells]
      int wall_decay_cells        = 3; // Normal-velocity decay distance [cells]
    };

  private:
    Config config;
    bool initialized = false;
    bool periodic_y  = false;
    bool periodic_z  = false;
    bool wall_y1     = false;
    bool wall_y2     = false;
    bool wall_z1     = false;
    bool wall_z2     = false;
    int halo_size    = 0;
    int filter_ny    = 0;
    int filter_nz    = 0;
    int random_count = 0;
    real mean_u      = 0;
    real intensity   = 0;
    real time_scale  = 0;
    real axial_potential_scale      = 0;
    real transverse_potential_scale = 0;

    real1d filter_y;
    real1d filter_z;
    real2d base_column;
    // (Ax,Ay,Az,dAx/dt,dAy/dt,dAz/dt,z,local y with one halo)
    real3d state;

    KOKKOS_INLINE_FUNCTION static std::uint64_t mix_bits_device(std::uint64_t value) {
      value += 0x9e3779b97f4a7c15ULL;
      value  = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
      value  = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
      return value ^ (value >> 31);
    }

    KOKKOS_INLINE_FUNCTION static real random_unit_variance(int seed, int count, int field, int k, int j) {
      auto value = static_cast<std::uint64_t>(seed);
      value ^= static_cast<std::uint64_t>(count + 1) * 0xd2b74407b1ce6e93ULL;
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

    KOKKOS_INLINE_FUNCTION static real smooth_factor(real distance, real decay_length) {
      if (distance >= decay_length) return 1;
      real const xi = std::max(static_cast<real>(0),distance/decay_length);
      return xi*xi*xi*(10 + xi*(-15 + 6*xi));
    }

    KOKKOS_INLINE_FUNCTION static real wall_factor(real location, real length, real decay_length,
                                                   bool wall_lower, bool wall_upper) {
      real factor = 1;
      if (wall_lower) factor *= smooth_factor(location,decay_length);
      if (wall_upper) factor *= smooth_factor(length-location,decay_length);
      return factor;
    }

    static bool nearly_equal(real lhs, real rhs) {
      return std::abs(lhs-rhs) <= 1.e-12*std::max(static_cast<real>(1),std::max(std::abs(lhs),std::abs(rhs)));
    }

    void initialize_filter(int length_cells, int half_width, real1d &filter, std::string const &label) {
      realHost1d host(label+"_host",2*half_width+1);
      real sum_squared = 0;
      for (int offset = -half_width; offset <= half_width; offset++) {
        real const value = std::exp(-M_PI*std::abs(offset)/length_cells);
        host(offset+half_width) = value;
        sum_squared += value*value;
      }
      for (int offset = 0; offset < host.extent(0); offset++) host(offset) /= std::sqrt(sum_squared);
      filter = host.createDeviceCopy();
    }

  public:
    // NVCC requires functions enclosing extended host/device lambdas to be publicly accessible.
    real3d create_filtered_innovation(core::Coupler const &coupler, int count) const {
      using yakl::SimpleBounds;
      int const nz      = coupler.get_nz();
      int const ny      = coupler.get_ny();
      int const ny_glob = coupler.get_ny_glob();
      int const j_beg   = coupler.get_j_beg();
      int const ny_pad  = filter_ny;
      int const nz_pad  = filter_nz;
      int const seed    = config.random_seed;
      bool const py     = periodic_y;
      bool const pz     = periodic_z;
      auto const by = realConst1d(filter_y);
      auto const bz = realConst1d(filter_z);

      // The wide random padding is generated from global indices on every rank. It therefore costs no communication,
      // remains independent of MPI decomposition, and permits a filter wider than a rank's local y slab.
      real3d filtered_y("digital_filter_inflow_filtered_y",6,nz+2*nz_pad,ny);
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(6,nz+2*nz_pad,ny) ,
                                              KOKKOS_LAMBDA (int field, int kp, int j) {
        int const source_k = kp-nz_pad;
        int const random_k = pz ? wrap_index(source_k,nz) : source_k;
        real value = 0;
        for (int offset = -ny_pad; offset <= ny_pad; offset++) {
          int const source_j = j_beg+j+offset;
          int const random_j = py ? wrap_index(source_j,ny_glob) : source_j;
          value += by(offset+ny_pad)*random_unit_variance(seed,count,field,random_k,random_j);
        }
        filtered_y(field,kp,j) = value;
      });

      real3d innovation("digital_filter_inflow_innovation",6,nz,ny);
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(6,nz,ny) , KOKKOS_LAMBDA (int field, int k, int j) {
        real value = 0;
        for (int offset = -nz_pad; offset <= nz_pad; offset++) {
          value += bz(offset+nz_pad)*filtered_y(field,k+offset+nz_pad,j);
        }
        innovation(field,k,j) = value;
      });
      return innovation;
    }

    void exchange_state_y(core::Coupler const &coupler) {
      using yakl::SimpleBounds;
      int const nz = coupler.get_nz();
      int const ny = coupler.get_ny();
      real2d send_south("digital_filter_inflow_send_south",6,nz);
      real2d send_north("digital_filter_inflow_send_north",6,nz);
      real2d recv_south("digital_filter_inflow_recv_south",6,nz);
      real2d recv_north("digital_filter_inflow_recv_north",6,nz);
      auto state_device = state;
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(6,nz) , KOKKOS_LAMBDA (int field, int k) {
        send_south(field,k) = state_device(field,k,1);
        send_north(field,k) = state_device(field,k,ny);
      });
      auto const &neighbors = coupler.get_neighbor_rankid_matrix();
      coupler.get_parallel_comm().send_receive<real,2>( { {recv_south,neighbors(0,1),20},
                                                          {recv_north,neighbors(2,1),21} },
                                                        { {send_south,neighbors(0,1),21},
                                                          {send_north,neighbors(2,1),20} },
                                                        "digital_filter_inflow_y_halo" );
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(6,nz) , KOKKOS_LAMBDA (int field, int k) {
        state_device(field,k,0   ) = recv_south(field,k);
        state_device(field,k,ny+1) = recv_north(field,k);
      });
    }

    real3d compute_raw_velocity(core::Coupler const &coupler, real axial_scale, real transverse_scale) {
      using yakl::SimpleBounds;
      exchange_state_y(coupler);
      int const nz      = coupler.get_nz();
      int const ny      = coupler.get_ny();
      int const ny_glob = coupler.get_ny_glob();
      int const j_beg   = coupler.get_j_beg();
      real const dy     = coupler.get_dy();
      real const zlen   = coupler.get_zlen();
      bool const py     = periodic_y;
      bool const pz     = periodic_z;
      bool const wy1    = wall_y1;
      bool const wy2    = wall_y2;
      bool const wz1    = wall_z1;
      bool const wz2    = wall_z2;
      real const decay_cells = config.wall_decay_cells;
      auto const dz = coupler.get_dz();
      auto const zint = coupler.get_zint();
      auto const zmid = coupler.get_zmid();
      auto state_device = state;
      real3d potential("digital_filter_inflow_tapered_potential",6,nz,ny+2);
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(6,nz,ny) ,
                                              KOKKOS_LAMBDA (int field, int k, int j) {
        real const y_cell = j_beg+j+static_cast<real>(0.5);
        real const z_cell = k+static_cast<real>(0.5);
        real const taper = wall_factor(y_cell,ny_glob,decay_cells,wy1,wy2)*
                           wall_factor(z_cell,nz,decay_cells,wz1,wz2);
        real const scale = field % 3 == 0 ? axial_scale : transverse_scale;
        potential(field,k,j+1) = scale*taper*state_device(field,k,j+1);
      });
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(6,nz) , KOKKOS_LAMBDA (int field, int k) {
        if (j_beg == 0 && !py) {
          potential(field,k,0) = wy1 ? -potential(field,k,1) : potential(field,k,1);
        } else {
          real const y_cell = j_beg-static_cast<real>(0.5);
          real const z_cell = k+static_cast<real>(0.5);
          real const scale = field % 3 == 0 ? axial_scale : transverse_scale;
          potential(field,k,0) = scale*wall_factor(y_cell,ny_glob,decay_cells,wy1,wy2)*
                                 wall_factor(z_cell,nz,decay_cells,wz1,wz2)*state_device(field,k,0);
        }
        if (j_beg+ny == ny_glob && !py) {
          potential(field,k,ny+1) = wy2 ? -potential(field,k,ny) : potential(field,k,ny);
        } else {
          real const y_cell = j_beg+ny+static_cast<real>(0.5);
          real const z_cell = k+static_cast<real>(0.5);
          real const scale = field % 3 == 0 ? axial_scale : transverse_scale;
          potential(field,k,ny+1) = scale*wall_factor(y_cell,ny_glob,decay_cells,wy1,wy2)*
                                    wall_factor(z_cell,nz,decay_cells,wz1,wz2)*state_device(field,k,ny+1);
        }
      });

      // gradient(0,...)=d/dy and gradient(1,...)=d/dz. Applying identical linear operators to all potential
      // components makes the discrete mixed derivatives commute in the interior. The vertical finite-volume
      // operator uses actual interface and center heights, so it remains conservative on a stretched grid.
      real4d gradient("digital_filter_inflow_potential_gradient",2,6,nz,ny);
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(6,nz,ny) ,
                                              KOKKOS_LAMBDA (int field, int k, int j) {
        real const center = potential(field,k,j+1);
        real lower;
        real upper;
        real lower_face;
        real upper_face;
        if (k > 0) {
          lower = potential(field,k-1,j+1);
          real const weight = (zint(k)-zmid(k-1))/(zmid(k)-zmid(k-1));
          lower_face = lower+weight*(center-lower);
        } else if (pz) {
          lower = potential(field,nz-1,j+1);
          real const lower_z = zmid(nz-1)-zlen;
          real const weight = (zint(0)-lower_z)/(zmid(0)-lower_z);
          lower_face = lower+weight*(center-lower);
        } else {
          lower_face = wz1 ? 0 : center;
        }
        if (k+1 < nz) {
          upper = potential(field,k+1,j+1);
          real const weight = (zint(k+1)-zmid(k))/(zmid(k+1)-zmid(k));
          upper_face = center+weight*(upper-center);
        } else if (pz) {
          upper = potential(field,0,j+1);
          real const upper_z = zmid(0)+zlen;
          real const weight = (zint(nz)-zmid(nz-1))/(upper_z-zmid(nz-1));
          upper_face = center+weight*(upper-center);
        } else {
          upper_face = wz2 ? 0 : center;
        }
        gradient(0,field,k,j) = (potential(field,k,j+2)-potential(field,k,j))/(2*dy);
        gradient(1,field,k,j) = (upper_face-lower_face)/dz(k);
      });

      // Extra components retain the two contributions to v and w for initialization-only energy calibration.
      real3d velocity("digital_filter_inflow_raw_velocity",8,nz,ny);
      real const inverse_u = static_cast<real>(1)/mean_u;
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ny) , KOKKOS_LAMBDA (int k, int j) {
        real const v_axial     =  gradient(1,0,k,j);
        real const v_temporal  =  potential(5,k,j+1)*inverse_u;
        real const w_temporal  = -potential(4,k,j+1)*inverse_u;
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

    real3d compute_raw_velocity(core::Coupler const &coupler) {
      return compute_raw_velocity(coupler,axial_potential_scale,transverse_potential_scale);
    }

    void calibrate(core::Coupler const &coupler) {
      using yakl::SimpleBounds;
      auto velocity = compute_raw_velocity(coupler,1,1);
      int const px = coupler.get_px();
      int const nz = coupler.get_nz();
      int const ny = coupler.get_ny();
      real2d u_variance("digital_filter_inflow_u_variance",nz,ny);
      real2d axial_variance("digital_filter_inflow_axial_potential_variance",nz,ny);
      real2d transverse_variance("digital_filter_inflow_transverse_potential_variance",nz,ny);
      real2d transverse_cross("digital_filter_inflow_transverse_cross",nz,ny);
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ny) , KOKKOS_LAMBDA (int k, int j) {
        u_variance(k,j) = px == 0 ? velocity(0,k,j)*velocity(0,k,j) : 0;
        axial_variance(k,j) = px == 0 ? velocity(4,k,j)*velocity(4,k,j) +
                                               velocity(7,k,j)*velocity(7,k,j) : 0;
        transverse_variance(k,j) = px == 0 ? velocity(5,k,j)*velocity(5,k,j) +
                                                    velocity(6,k,j)*velocity(6,k,j) : 0;
        transverse_cross(k,j) = px == 0 ? 2*(velocity(4,k,j)*velocity(5,k,j) +
                                              velocity(7,k,j)*velocity(6,k,j)) : 0;
      });
      real const count = nz*coupler.get_ny_glob();
      real const actual_u = coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(u_variance),MPI_SUM)/count;
      real const axial = coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(axial_variance),MPI_SUM)/count;
      real const transverse = coupler.get_parallel_comm().all_reduce(
                                   yakl::intrinsics::sum(transverse_variance),MPI_SUM)/count;
      real const cross = coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(transverse_cross),MPI_SUM)/count;
      real const target = intensity*intensity*mean_u*mean_u;
      if (target > 0 && (actual_u <= 0 || axial <= 0)) {
        endrun("Digital-filter turbulent inflow produced zero calibration variance");
      }
      if (target == 0) {
        axial_potential_scale = 0;
        transverse_potential_scale = 0;
        return;
      }
      transverse_potential_scale = std::sqrt(target/actual_u);
      real const linear = cross*transverse_potential_scale;
      real const constant = transverse*transverse_potential_scale*transverse_potential_scale-2*target;
      real const discriminant = linear*linear-4*axial*constant;
      if (discriminant < 0) endrun("Digital-filter turbulent inflow could not calibrate vector-potential energy");
      real const root_positive = (-linear+std::sqrt(discriminant))/(2*axial);
      real const root_negative = (-linear-std::sqrt(discriminant))/(2*axial);
      axial_potential_scale = std::max(root_positive,root_negative);
      if (axial_potential_scale <= 0) {
        endrun("Digital-filter turbulent inflow vector-potential calibration produced a nonpositive scale");
      }
    }

    void advance(core::Coupler const &coupler, real dt) {
      using yakl::SimpleBounds;
      auto innovation = create_filtered_innovation(coupler,random_count);
      int const nz = coupler.get_nz();
      int const ny = coupler.get_ny();
      // This critically damped second-order stochastic process has stationary Var(A)=1, Var(dA/dt)=lambda^2,
      // and correlation (1+lambda*t)exp(-lambda*t). lambda=2/T makes its integral time scale equal to T=Lx/U.
      real const lambda = static_cast<real>(2)/time_scale;
      real const decay = std::exp(-lambda*dt);
      real const phi00 = decay*(1+lambda*dt);
      real const phi01 = decay*dt;
      real const phi10 = -decay*lambda*lambda*dt;
      real const phi11 = decay*(1-lambda*dt);
      real const covariance00 = std::max(static_cast<real>(0),1-(phi00*phi00+lambda*lambda*phi01*phi01));
      real const covariance01 = -(phi00*phi10+lambda*lambda*phi01*phi11);
      real const covariance11 = std::max(static_cast<real>(0),lambda*lambda-(phi10*phi10+lambda*lambda*phi11*phi11));
      real const noise00 = std::sqrt(covariance00);
      real const noise10 = noise00 > 0 ? covariance01/noise00 : 0;
      real const noise11 = std::sqrt(std::max(static_cast<real>(0),covariance11-noise10*noise10));
      auto state_device = state;
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(3,nz,ny) ,
                                              KOKKOS_LAMBDA (int component, int k, int j) {
        real const potential = state_device(component  ,k,j+1);
        real const rate      = state_device(component+3,k,j+1);
        state_device(component  ,k,j+1) = phi00*potential + phi01*rate + noise00*innovation(component,k,j);
        state_device(component+3,k,j+1) = phi10*potential + phi11*rate + noise10*innovation(component  ,k,j) +
                                                                            noise11*innovation(component+3,k,j);
      });
      random_count++;
    }

    real3d gather_global_state(core::Coupler const &coupler) const {
      using yakl::SimpleBounds;
      int const px      = coupler.get_px();
      int const nz      = coupler.get_nz();
      int const ny      = coupler.get_ny();
      int const ny_glob = coupler.get_ny_glob();
      int const j_beg   = coupler.get_j_beg();
      auto state_device = state;
      real3d local("digital_filter_inflow_global_state_local",6,nz,ny_glob);
      local = 0;
      if (px == 0) {
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(6,nz,ny) , KOKKOS_LAMBDA (int field, int k, int j) {
          local(field,k,j_beg+j) = state_device(field,k,j+1);
        });
      }
      return coupler.get_parallel_comm().all_reduce(local,MPI_SUM,"digital_filter_inflow_gather_state");
    }

    void scatter_global_state(core::Coupler const &coupler, real3d const &global) {
      using yakl::SimpleBounds;
      int const nz    = coupler.get_nz();
      int const ny    = coupler.get_ny();
      int const j_beg = coupler.get_j_beg();
      auto state_device = state;
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(6,nz,ny) , KOKKOS_LAMBDA (int field, int k, int j) {
        state_device(field,k,j+1) = global(field,k,j_beg+j);
      });
    }

  private:
    void register_restart(core::Coupler &coupler) {
      coupler.register_write_output_module( [this] (core::Coupler &coupler, core::FileIO &nc) {
        auto global_state = gather_global_state(coupler);
        auto base_host = base_column.createHostCopy();
        nc.redef();
        nc.create_dim("digital_filter_inflow_field",6);
        nc.create_dim("digital_filter_inflow_base_field",base_column.extent(0));
        nc.create_var<real>("digital_filter_inflow_state",{"digital_filter_inflow_field","z","y"});
        nc.create_var<real>("digital_filter_inflow_base_column",{"digital_filter_inflow_base_field","z"});
        nc.create_var<real>("digital_filter_inflow_axial_potential_scale",{});
        nc.create_var<real>("digital_filter_inflow_transverse_potential_scale",{});
        nc.create_var<int>("digital_filter_inflow_random_count",{});
        nc.enddef();
        nc.begin_indep_data();
        if (coupler.is_mainproc()) {
          nc.write(global_state,"digital_filter_inflow_state");
          nc.write(base_host,"digital_filter_inflow_base_column");
          nc.write(axial_potential_scale,"digital_filter_inflow_axial_potential_scale");
          nc.write(transverse_potential_scale,"digital_filter_inflow_transverse_potential_scale");
          nc.write(random_count,"digital_filter_inflow_random_count");
        }
        nc.end_indep_data();
      });
      coupler.register_overwrite_with_restart_module( [this] (core::Coupler &coupler, core::FileIO &nc) {
        real3d global_state("digital_filter_inflow_restart_state",6,coupler.get_nz(),coupler.get_ny_glob());
        realHost2d base_host("digital_filter_inflow_restart_base",base_column.extent(0),base_column.extent(1));
        nc.begin_indep_data();
        if (coupler.is_mainproc()) {
          nc.read(global_state,"digital_filter_inflow_state");
          nc.read(base_host,"digital_filter_inflow_base_column");
          nc.read(axial_potential_scale,"digital_filter_inflow_axial_potential_scale");
          nc.read(transverse_potential_scale,"digital_filter_inflow_transverse_potential_scale");
          nc.read(random_count,"digital_filter_inflow_random_count");
        }
        nc.end_indep_data();
        auto const comm = coupler.get_parallel_comm();
        comm.broadcast(global_state);
        comm.broadcast(base_host);
        comm.broadcast(axial_potential_scale);
        comm.broadcast(transverse_potential_scale);
        comm.broadcast(random_count);
        base_host.deep_copy_to(base_column);
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
        endrun("Digital-filter turbulent inflow correlation lengths must be positive cell counts");
      }
      if (config.wall_decay_cells <= 0) {
        endrun("Digital-filter turbulent inflow wall decay must be a positive cell count");
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
      periodic_y = bc_y1 == "periodic" && bc_y2 == "periodic";
      periodic_z = bc_z1 == "periodic" && bc_z2 == "periodic";
      wall_y1 = bc_y1 == "wall_free_slip";
      wall_y2 = bc_y2 == "wall_free_slip";
      wall_z1 = bc_z1 == "wall_free_slip";
      wall_z2 = bc_z2 == "wall_free_slip";
      if (!(periodic_y || (wall_y1 && wall_y2)) || !(periodic_z || (wall_z1 && wall_z2))) {
        endrun("Digital-filter turbulent inflow requires periodic or wall_free_slip transverse boundary pairs");
      }

      halo_size = Dycore::hs;
      time_scale = config.streamwise_length_cells*coupler.get_dx()/mean_u;
      filter_ny = 2*config.spanwise_length_cells;
      filter_nz = 2*config.vertical_length_cells;
      initialize_filter(config.spanwise_length_cells,filter_ny,filter_y,"digital_filter_inflow_filter_y");
      initialize_filter(config.vertical_length_cells,filter_nz,filter_z,"digital_filter_inflow_filter_z");

      base_column = dycore.compute_average_ghost_column(coupler);
      state = real3d("digital_filter_inflow_state",6,nz,coupler.get_ny()+2);
      auto initial = create_filtered_innovation(coupler,random_count);
      auto state_device = state;
      real const lambda = static_cast<real>(2)/time_scale;
      yakl::parallel_for( YAKL_AUTO_LABEL() , yakl::SimpleBounds<3>(3,nz,coupler.get_ny()) ,
                                              KOKKOS_LAMBDA (int component, int k, int j) {
        state_device(component  ,k,j+1) = initial(component  ,k,j);
        state_device(component+3,k,j+1) = lambda*initial(component+3,k,j);
      });
      random_count++;
      calibrate(coupler);
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

#include "coupler.h"
#include "dynamics_cell_centered.h"
#include "edge_sponge.h"
#include "sc_init.h"
#include "synthetic_turbulent_inflow.h"

namespace {

  using Dycore = modules::Dynamics_Euler_Stratified;
  using FLOC = Dycore::FLOC;

  void require(core::Coupler const &coupler, bool condition, std::string const &message) {
    int const valid = coupler.get_parallel_comm().all_reduce(condition ? 1 : 0,MPI_MIN);
    if (valid == 0) endrun(message.c_str());
  }

  real ghost_signature(core::Coupler const &coupler) {
    real local = 0;
    if (coupler.get_px() == 0) {
      auto ghost = coupler.get_data_manager_readonly().get<FLOC const,6>("dycore_ghost_x1");
      real1d values("digital_filter_inflow_signature",coupler.get_nz()*coupler.get_ny());
      int const ny = coupler.get_ny();
      yakl::parallel_for( YAKL_AUTO_LABEL() , values.extent(0) , KOKKOS_LAMBDA (int index) {
        int const j = index % ny;
        int const k = index / ny;
        values(index) = ghost(0,0,Dycore::idU,k,j,0) + 2*ghost(0,0,Dycore::idV,k,j,0) +
                        3*ghost(0,0,Dycore::idW,k,j,0);
      });
      local = yakl::intrinsics::sum(values);
    }
    return coupler.get_parallel_comm().all_reduce(local,MPI_SUM);
  }

  real2d copy_u_fluctuation(core::Coupler const &coupler, real mean_u) {
    real2d result("digital_filter_inflow_u_snapshot",coupler.get_nz(),coupler.get_ny());
    result = 0;
    if (coupler.get_px() == 0) {
      auto ghost = coupler.get_data_manager_readonly().get<FLOC const,6>("dycore_ghost_x1");
      yakl::parallel_for( YAKL_AUTO_LABEL() , yakl::SimpleBounds<2>(coupler.get_nz(),coupler.get_ny()) ,
                                              KOKKOS_LAMBDA (int k, int j) {
        result(k,j) = ghost(0,0,Dycore::idU,k,j,0)-mean_u;
      });
    }
    return result;
  }

  void check_statistics_and_structure(core::Coupler const &coupler, modules::DigitalFilterTurbulentInflow &inflow,
                                      real mean_u, real intensity) {
    int const px      = coupler.get_px();
    int const nz      = coupler.get_nz();
    int const ny      = coupler.get_ny();
    int const ny_glob = coupler.get_ny_glob();
    int const j_beg   = coupler.get_j_beg();
    real const dy     = coupler.get_dy();
    auto const dz     = coupler.get_dz();
    auto const zint   = coupler.get_zint();
    auto const zmid   = coupler.get_zmid();
    real local_sum_u = 0;
    real local_energy = 0;
    real local_divergence = 0;
    real local_v_wall = 0;
    real local_v_center = 0;
    real local_w_wall = 0;
    real local_w_center = 0;
    auto raw_velocity = inflow.compute_raw_velocity(coupler);
    if (px == 0) {
      auto ghost = coupler.get_data_manager_readonly().get<FLOC const,6>("dycore_ghost_x1");
      real1d work("digital_filter_inflow_test_work",nz*ny);
      yakl::parallel_for( YAKL_AUTO_LABEL() , work.extent(0) , KOKKOS_LAMBDA (int index) {
        int const j = index % ny;
        int const k = index / ny;
        real const up = ghost(0,0,Dycore::idU,k,j,0)-mean_u;
        real const vp = ghost(0,0,Dycore::idV,k,j,0);
        real const wp = ghost(0,0,Dycore::idW,k,j,0);
        work(index) = up*up + vp*vp + wp*wp;
      });
      local_energy = yakl::intrinsics::sum(work);
      yakl::parallel_for( YAKL_AUTO_LABEL() , work.extent(0) , KOKKOS_LAMBDA (int index) {
        int const j = index % ny;
        int const k = index / ny;
        work(index) = ghost(0,0,Dycore::idU,k,j,0)*dz(k);
      });
      local_sum_u = yakl::intrinsics::sum(work);

      int const kcenter = nz/2;
      int const jcenter_glob = ny_glob/2;
      int const jcenter = jcenter_glob-j_beg;
      real1d samples("digital_filter_inflow_wall_samples",4);
      yakl::parallel_for( YAKL_AUTO_LABEL() , 4 , KOKKOS_LAMBDA (int sample) {
        if (sample == 0) {
          real const south = j_beg == 0 ? ghost(0,0,Dycore::idV,kcenter,0,0) : 0;
          real const north = j_beg+ny == ny_glob ? ghost(0,0,Dycore::idV,kcenter,ny-1,0) : 0;
          samples(sample) = south*south + north*north;
        } else if (sample == 1) {
          real const value = jcenter >= 0 && jcenter < ny ? ghost(0,0,Dycore::idV,kcenter,jcenter,0) : 0;
          samples(sample) = value*value;
        } else if (sample == 2) {
          real const bottom = jcenter >= 0 && jcenter < ny ? ghost(0,0,Dycore::idW,0,jcenter,0) : 0;
          real const top = jcenter >= 0 && jcenter < ny ? ghost(0,0,Dycore::idW,nz-1,jcenter,0) : 0;
          samples(sample) = bottom*bottom + top*top;
        } else {
          real const value = jcenter >= 0 && jcenter < ny ? ghost(0,0,Dycore::idW,kcenter,jcenter,0) : 0;
          samples(sample) = value*value;
        }
      });
      auto samples_host = samples.createHostCopy();
      local_v_wall   = samples_host(0);
      local_v_center = samples_host(1);
      local_w_wall   = samples_host(2);
      local_w_center = samples_host(3);
    }

    // Taylor's hypothesis gives du/dx=-du/dt/U. The complete space-time curl must therefore satisfy
    // -du/dt/U+dv/dy+dw/dz=0, rather than only cancelling the transverse divergence.
    real1d divergence_values("digital_filter_inflow_divergence",nz*ny);
    divergence_values = 0;
    yakl::parallel_for( YAKL_AUTO_LABEL() , yakl::SimpleBounds<2>(nz-2,std::max(0,ny-2)) ,
                                            KOKKOS_LAMBDA (int kk, int jj) {
      int const k = kk+1;
      int const j = jj+1;
      real const lower_weight = (zint(k)-zmid(k-1))/(zmid(k)-zmid(k-1));
      real const upper_weight = (zint(k+1)-zmid(k))/(zmid(k+1)-zmid(k));
      real const lower_face = raw_velocity(2,k-1,j) +
                              lower_weight*(raw_velocity(2,k,j)-raw_velocity(2,k-1,j));
      real const upper_face = raw_velocity(2,k,j) +
                              upper_weight*(raw_velocity(2,k+1,j)-raw_velocity(2,k,j));
      real const value = -raw_velocity(3,k,j)/mean_u +
                         (raw_velocity(1,k,j+1)-raw_velocity(1,k,j-1))/(2*dy) +
                         (upper_face-lower_face)/dz(k);
      divergence_values(k*ny+j) = std::abs(value);
    });
    local_divergence = yakl::intrinsics::maxval(divergence_values);

    auto const comm = coupler.get_parallel_comm();
    real const count = nz*ny_glob;
    real const average_u = comm.all_reduce(local_sum_u,MPI_SUM)/(ny_glob*coupler.get_zlen());
    real const energy = comm.all_reduce(local_energy,MPI_SUM);
    real const target_energy = count*3*intensity*intensity*mean_u*mean_u;
    real const divergence = comm.all_reduce(local_divergence,MPI_MAX);
    real const v_wall = comm.all_reduce(local_v_wall,MPI_SUM);
    real const v_center = comm.all_reduce(local_v_center,MPI_SUM);
    real const w_wall = comm.all_reduce(local_w_wall,MPI_SUM);
    real const w_center = comm.all_reduce(local_w_center,MPI_SUM);
    require(coupler,std::abs(average_u-mean_u) < 2.e-5,"Digital-filter inflow did not preserve the bulk mean wind");
    require(coupler,std::abs(energy-target_energy) < 5.e-4*target_energy,
            "Digital-filter inflow did not realize its isotropic total-energy target");
    require(coupler,divergence < 2.e-6,"Digital-filter inflow space-time curl was not discretely divergence free");
    require(coupler,v_wall < v_center,"Digital-filter inflow v did not decay toward spanwise walls");
    require(coupler,w_wall < w_center,"Digital-filter inflow w did not decay toward vertical walls");
  }

  void check_temporal_correlation(core::Coupler const &coupler, real2d const &first, real2d const &second) {
    int const px = coupler.get_px();
    real2d product("digital_filter_inflow_temporal_product",coupler.get_nz(),coupler.get_ny());
    real2d first_squared("digital_filter_inflow_temporal_first",coupler.get_nz(),coupler.get_ny());
    real2d second_squared("digital_filter_inflow_temporal_second",coupler.get_nz(),coupler.get_ny());
    yakl::parallel_for( YAKL_AUTO_LABEL() , yakl::SimpleBounds<2>(coupler.get_nz(),coupler.get_ny()) ,
                                            KOKKOS_LAMBDA (int k, int j) {
      product(k,j) = px == 0 ? first(k,j)*second(k,j) : 0;
      first_squared(k,j) = px == 0 ? first(k,j)*first(k,j) : 0;
      second_squared(k,j) = px == 0 ? second(k,j)*second(k,j) : 0;
    });
    auto const comm = coupler.get_parallel_comm();
    real const covariance = comm.all_reduce(yakl::intrinsics::sum(product),MPI_SUM);
    real const variance_1 = comm.all_reduce(yakl::intrinsics::sum(first_squared),MPI_SUM);
    real const variance_2 = comm.all_reduce(yakl::intrinsics::sum(second_squared),MPI_SUM);
    real const correlation = covariance/std::sqrt(variance_1*variance_2);
    require(coupler,correlation > 0.9,"Digital-filter inflow did not retain its streamwise correlation");
  }

  void check_interior_fluctuations(core::Coupler const &coupler, real mean_u) {
    auto const &dm = coupler.get_data_manager_readonly();
    auto uvel = dm.get<real const,3>("uvel");
    auto vvel = dm.get<real const,3>("vvel");
    auto wvel = dm.get<real const,3>("wvel");
    int const nx = coupler.get_nx();
    int const ny = coupler.get_ny();
    int const nz = coupler.get_nz();
    real1d kinetic_energy("digital_filter_inflow_interior_fluctuations",nz*ny*nx);
    yakl::parallel_for( YAKL_AUTO_LABEL() , yakl::SimpleBounds<3>(nz,ny,nx) ,
                                            KOKKOS_LAMBDA (int k, int j, int i) {
      real const up = uvel(k,j,i)-mean_u;
      kinetic_energy((k*ny+j)*nx+i) = up*up + vvel(k,j,i)*vvel(k,j,i) + wvel(k,j,i)*wvel(k,j,i);
    });
    real const maximum = coupler.get_parallel_comm().all_reduce(yakl::intrinsics::maxval(kinetic_energy),MPI_MAX);
    require(coupler,maximum > 1.e-10,"Digital-filter inflow fluctuations did not enter the model interior");
  }

}

int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  Kokkos::initialize();
  yakl::init();
  {
    bool visualize      = false;
    bool use_edge_sponge = true;
    for (int arg = 1; arg < argc; arg++) {
      if (std::string(argv[arg]) == "--visualize") visualize = true;
      if (std::string(argv[arg]) == "--no-sponge") use_edge_sponge = false;
    }
    core::Coupler coupler;
    int  const     nx_glob = visualize ? 64 : 20;
    int  const     ny_glob = visualize ? 64 : 20;
    int  const     nz      = visualize ? 64 : 20;
    real constexpr spacing = 10;
    real constexpr mean_u  = 8;
    real constexpr ti      = 0.10;
    real constexpr dt      = 0.1;
    coupler.set_option<std::string>("init_data","constant");
    coupler.set_option<real>("constant_uvel",mean_u);
    coupler.set_option<real>("constant_vvel",0);
    coupler.set_option<real>("constant_temp",300);
    coupler.set_option<real>("constant_press",1.e5);
    coupler.set_option<real>("cfl",0.6);
    coupler.set_option<real>("dycore_cs",80);
    coupler.set_option<real>("dycore_max_wind",20);
    coupler.set_option<int>("dycore_max_cycles",4);
    coupler.set_option<std::string>("dycore_time_stepper","ssprk3");
    auto levels = visualize ? coupler.generate_levels_equal(nz,nz*spacing) :
                              coupler.generate_levels_exp(nz,nz*spacing,static_cast<real>(0.7)*spacing);
    coupler.init(core::ParallelComm(MPI_COMM_WORLD),levels,
                 ny_glob,nx_glob,ny_glob*spacing,nx_glob*spacing);
    coupler.add_tracer("water_vapor","water_vapor",true,true,true);
    coupler.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;
    custom_modules::sc_init(coupler);
    coupler.set_option<std::string>("bc_x1","precursor");
    coupler.set_option<std::string>("bc_x2","open");
    coupler.set_option<std::string>("bc_y1","wall_free_slip");
    coupler.set_option<std::string>("bc_y2","wall_free_slip");
    coupler.set_option<std::string>("bc_z1","wall_free_slip");
    coupler.set_option<std::string>("bc_z2","wall_free_slip");

    Dycore dycore;
    modules::DigitalFilterTurbulentInflow inflow;
    modules::EdgeSponge edge_sponge;
    dycore.init(coupler);
    real1d u_profile("digital_filter_inflow_test_u",nz);
    real1d v_profile("digital_filter_inflow_test_v",nz);
    real1d intensity("digital_filter_inflow_test_ti",nz);
    u_profile = mean_u;
    v_profile = 0;
    intensity = ti;
    modules::DigitalFilterTurbulentInflow::Config config;
    config.random_seed = 8123;
    inflow.init(coupler,dycore,u_profile,v_profile,intensity,config);
    real const expected_time_scale = config.streamwise_length_cells*coupler.get_dx()/mean_u;
    require(coupler,std::abs(inflow.get_time_scale()-expected_time_scale) < 1.e-12,
            "Digital-filter inflow used the wrong Taylor time scale");
    if (use_edge_sponge) edge_sponge.set_column(coupler,{"density_dry","temperature"});

    if (visualize) {
      // Stop after one mean-flow transit through the 640 m domain. One-second snapshots resolve passage of the
      // five-cell transverse structures while keeping the visualization compact.
      int const visualization_steps = static_cast<int>(std::ceil(coupler.get_xlen()/(mean_u*dt)));
      int constexpr output_stride = 10;
      std::string const output_prefix = use_edge_sponge ? "digital_filter_turbulent_inflow_visualization_64cube" :
                                                          "digital_filter_turbulent_inflow_visualization_64cube_no_sponge";
      real elapsed_time = 0;
      for (int step = 0; step < visualization_steps; step++) {
        inflow.apply(coupler,dycore,dt);
        if (use_edge_sponge) edge_sponge.apply(coupler,0.1,0.1,0,0);
        dycore.time_step(coupler,dt);
        elapsed_time += dt;
        coupler.set_option<real>("elapsed_time",elapsed_time);
        if ((step+1) % output_stride == 0) {
          coupler.write_output_file(output_prefix,false);
        }
      }
      check_interior_fluctuations(coupler,mean_u);
    } else {
      inflow.apply(coupler,dycore,dt);
      check_statistics_and_structure(coupler,inflow,mean_u,ti);
      auto first_u = copy_u_fluctuation(coupler,mean_u);
      if (use_edge_sponge) edge_sponge.apply(coupler,0.1,0.1,0,0);
      dycore.time_step(coupler,dt);
      coupler.set_option<real>("elapsed_time",dt);
      check_interior_fluctuations(coupler,mean_u);
      coupler.write_output_file("digital_filter_turbulent_inflow",false);

      inflow.apply(coupler,dycore,dt);
      auto second_u = copy_u_fluctuation(coupler,mean_u);
      check_temporal_correlation(coupler,first_u,second_u);
      real const uninterrupted_signature = ghost_signature(coupler);
      std::string const extension = core::FileIO::default_backend() == "adios2" ? ".bp" : ".nc";
      coupler.set_option<std::string>("restart_file","digital_filter_turbulent_inflow_00000000"+extension);
      coupler.overwrite_with_restart();
      inflow.apply(coupler,dycore,dt);
      real const restarted_signature = ghost_signature(coupler);
      real const signature_scale = std::max(static_cast<real>(1),std::abs(uninterrupted_signature));
      require(coupler,std::abs(restarted_signature-uninterrupted_signature) < 1.e-11*signature_scale,
              "Digital-filter inflow restart did not reproduce uninterrupted evolution");
    }

    auto const &dm = coupler.get_data_manager_readonly();
    auto density = dm.get<real const,3>("density_dry");
    auto temperature = dm.get<real const,3>("temperature");
    require(coupler,std::isfinite(yakl::intrinsics::minval(density)) && yakl::intrinsics::minval(density) > 0,
            "Digital-filter inflow integration produced invalid density");
    require(coupler,std::isfinite(yakl::intrinsics::minval(temperature)) && yakl::intrinsics::minval(temperature) > 0,
            "Digital-filter inflow integration produced invalid temperature");
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}

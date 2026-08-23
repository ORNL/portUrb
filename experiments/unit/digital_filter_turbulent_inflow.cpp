#include "coupler.h"
#include "dynamics_cell_centered.h"
#include "edge_sponge.h"
#include "sc_init.h"
#include "synthetic_turbulent_inflow.h"

#include <limits>

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
    real const dy     = coupler.get_dy();
    auto const dz     = coupler.get_dz();
    auto const zint   = coupler.get_zint();
    auto const zmid   = coupler.get_zmid();
    real local_sum_u = 0;
    real local_energy = 0;
    real local_divergence = 0;
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
    require(coupler,std::abs(average_u-mean_u) < 0.5*intensity*mean_u,
            "Digital-filter inflow realization was too far from its ensemble-mean wind target");
    // Calibration targets the ensemble mean rather than rescaling every production plane. A single realization is
    // therefore allowed sampling variability while still being required to remain close to the requested TI.
    require(coupler,std::abs(energy-target_energy) < 0.5*target_energy,
            "Digital-filter inflow realization was too far from its isotropic total-energy target");
    require(coupler,divergence < 2.e-6,"Digital-filter inflow space-time curl was not discretely divergence free");
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

  void check_octave_structure(core::Coupler const &coupler, modules::DigitalFilterTurbulentInflow const &inflow) {
    using yakl::SimpleBounds;
    int const px    = coupler.get_px();
    int const nz    = coupler.get_nz();
    int const ny    = coupler.get_ny();
    int const j_beg = coupler.get_j_beg();
    auto innovation = inflow.create_unfiltered_innovation(coupler,314159);
    real previous_gradient = std::numeric_limits<real>::max();
    for (int octave = 0; octave < inflow.get_num_octaves(); octave++) {
      int const level = 1 << octave;
      int const pair_count = 6*nz*std::max(0,ny-1);
      real2d statistics("digital_filter_inflow_octave_statistics",6,pair_count);
      statistics = 0;
      yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(6,nz,std::max(0,ny-1)) ,
                                              KOKKOS_LAMBDA (int field, int k, int jj) {
        int const j = jj+1;
        int const index = (field*nz+k)*(ny-1)+jj;
        real const difference = innovation(octave,field,k,j)-innovation(octave,field,k,j-1);
        bool const panel_edge = octave > 0 && (j_beg+j) % level == 0;
        statistics(panel_edge ? 0 : 1,index) = px == 0 ? difference*difference : 0;
        statistics(panel_edge ? 2 : 3,index) = px == 0 ? 1 : 0;
        statistics(4,index) = px == 0 ? innovation(octave,field,k,j)*innovation(octave,field,k,j) : 0;
        statistics(5,index) = px == 0 ? innovation(octave,field,k,j)*innovation(octave,field,k,j-1) : 0;
      });
      realHost1d local_host("digital_filter_inflow_octave_local_host",6);
      for (int statistic = 0; statistic < 6; statistic++) {
        local_host(statistic) = yakl::intrinsics::sum(statistics.slice<1>(statistic,0));
      }
      auto local = local_host.createDeviceCopy();
      auto global = coupler.get_parallel_comm().all_reduce(local,MPI_SUM,"digital_filter_inflow_octave_test");
      auto global_host = global.createHostCopy();
      if (octave > 0) {
        real const boundary_mean = global_host(0)/global_host(2);
        real const interior_mean = global_host(1)/global_host(3);
        require(coupler,boundary_mean > 0.25*interior_mean && boundary_mean < 4*interior_mean,
                "Digital-filter inflow octave retained panel-boundary jump energy");
      }
      real const gradient_variance = (global_host(0)+global_host(1))/(global_host(2)+global_host(3));
      real const sample_count = global_host(2)+global_host(3);
      if (octave == 0) {
        real const variance = global_host(4)/sample_count;
        real const neighbor_correlation = global_host(5)/global_host(4);
        require(coupler,variance > 0.8 && variance < 1.2,
                "Digital-filter inflow finest octave was not unit-variance cellwise noise");
        require(coupler,std::abs(neighbor_correlation) < 0.1,
                "Digital-filter inflow finest octave was not spatially white before filtering");
      } else {
        require(coupler,gradient_variance < previous_gradient,
                "Digital-filter inflow coarse octave did not become spatially smoother");
      }
      previous_gradient = gradient_variance;
    }
  }

  void check_exterior_octave_stationarity(core::Coupler const &coupler,
                                           modules::DigitalFilterTurbulentInflow const &inflow) {
    using yakl::SimpleBounds;
    int const px = coupler.get_px();
    int const nz = coupler.get_nz();
    int const ny = coupler.get_ny();
    int const j_beg = coupler.get_j_beg();
    int const ny_glob = coupler.get_ny_glob();
    for (int octave = 0; octave < inflow.get_num_octaves(); octave++) {
      realHost1d totals_host("digital_filter_inflow_boundary_totals_host",4);
      totals_host = 0;
      for (int sample = 0; sample < 10; sample++) {
        auto innovation = inflow.create_filtered_innovation(coupler,271828+sample);
        real2d statistics("digital_filter_inflow_boundary_statistics",4,6*nz*ny);
        statistics = 0;
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(6,nz) , KOKKOS_LAMBDA (int field, int k) {
          if (px == 0 && j_beg == 0) {
            real const difference = innovation(octave,field,k+1,1)-innovation(octave,field,k+1,0);
            statistics(0,field*nz+k) = difference*difference;
            statistics(1,field*nz+k) = 1;
          }
          if (px == 0 && j_beg+ny == ny_glob) {
            real const difference = innovation(octave,field,k+1,ny+1)-innovation(octave,field,k+1,ny);
            statistics(0,field*nz+k) += difference*difference;
            statistics(1,field*nz+k) += 1;
          }
        });
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(6,ny) , KOKKOS_LAMBDA (int field, int j) {
          if (px == 0) {
            real const lower = innovation(octave,field,1,j+1)-innovation(octave,field,0,j+1);
            real const upper = innovation(octave,field,nz+1,j+1)-innovation(octave,field,nz,j+1);
            statistics(0,6*nz+field*ny+j) = lower*lower+upper*upper;
            statistics(1,6*nz+field*ny+j) = 2;
          }
        });
        yakl::parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(6,std::max(0,nz-1),ny) ,
                                                KOKKOS_LAMBDA (int field, int k, int j) {
          int const index = (field*(nz-1)+k)*ny+j;
          real const difference = innovation(octave,field,k+2,j+1)-innovation(octave,field,k+1,j+1);
          statistics(2,index) = px == 0 ? difference*difference : 0;
          statistics(3,index) = px == 0 ? 1 : 0;
        });
        for (int statistic = 0; statistic < 4; statistic++) {
          totals_host(statistic) += yakl::intrinsics::sum(statistics.slice<1>(statistic,0));
        }
      }
      auto totals = totals_host.createDeviceCopy();
      auto global = coupler.get_parallel_comm().all_reduce(totals,MPI_SUM,"digital_filter_inflow_boundary_test");
      auto global_host = global.createHostCopy();
      real const boundary_variance = global_host(0)/global_host(1);
      real const interior_variance = global_host(2)/global_host(3);
      require(coupler,boundary_variance > 0.5*interior_variance && boundary_variance < 2*interior_variance,
              "Digital-filter inflow octave lost stationarity at an exterior boundary");
    }
  }

}

int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  Kokkos::initialize();
  yakl::init();
  {
    bool visualize      = false;
    bool octave_only_64 = false;
    bool use_edge_sponge = true;
    bool periodic_transverse = false;
    for (int arg = 1; arg < argc; arg++) {
      if (std::string(argv[arg]) == "--visualize") visualize = true;
      if (std::string(argv[arg]) == "--octave-only-64") octave_only_64 = true;
      if (std::string(argv[arg]) == "--no-sponge") use_edge_sponge = false;
      if (std::string(argv[arg]) == "--periodic-transverse") periodic_transverse = true;
    }
    core::Coupler coupler;
    int  const     nx_glob = visualize || octave_only_64 ? 64 : 20;
    int  const     ny_glob = visualize || octave_only_64 ? 64 : 20;
    int  const     nz      = visualize || octave_only_64 ? 64 : 20;
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
    auto levels = visualize || octave_only_64 ? coupler.generate_levels_equal(nz,nz*spacing) :
                                                coupler.generate_levels_exp(nz,nz*spacing,
                                                                            static_cast<real>(0.7)*spacing);
    coupler.init(core::ParallelComm(MPI_COMM_WORLD),levels,
                 ny_glob,nx_glob,ny_glob*spacing,nx_glob*spacing);
    coupler.add_tracer("water_vapor","water_vapor",true,true,true);
    coupler.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;
    custom_modules::sc_init(coupler);
    coupler.set_option<std::string>("bc_x1","precursor");
    coupler.set_option<std::string>("bc_x2","open");
    std::string const transverse_bc = periodic_transverse ? "periodic" : "wall_free_slip";
    coupler.set_option<std::string>("bc_y1",transverse_bc);
    coupler.set_option<std::string>("bc_y2",transverse_bc);
    coupler.set_option<std::string>("bc_z1",transverse_bc);
    coupler.set_option<std::string>("bc_z2",transverse_bc);

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
    require(coupler,modules::DigitalFilterTurbulentInflow::compute_maximum_octave_level(640,640,10,10,
                                                                                       config.maximum_length_fraction) == 4,
            "A default 64-cubed inflow did not select octave levels 1,2,4");
    require(coupler,modules::DigitalFilterTurbulentInflow::compute_num_octaves(640,640,10,10,
                                                                               config.maximum_length_fraction) == 3,
            "A default 64-cubed inflow did not create exactly three dyadic octaves");
    require(coupler,modules::DigitalFilterTurbulentInflow::compute_maximum_octave_level(640,640,10,10,0.25) == 16,
            "A user-selected quarter-domain cutoff did not retain octave levels 1,2,4,8,16");
    require(coupler,modules::DigitalFilterTurbulentInflow::compute_maximum_octave_level(640,640,10,20,0.10) == 2,
            "A stretched vertical grid did not conservatively limit the maximum physical octave length");
    require(coupler,modules::DigitalFilterTurbulentInflow::compute_capped_filter_length(1000,640,800,960) == 160,
            "Digital-filter inflow did not cap its filter at one quarter of the shortest domain length");
    inflow.init(coupler,dycore,u_profile,v_profile,intensity,config);
    auto dz_host = coupler.get_dz().createHostCopy();
    real maximum_dz = 0;
    for (int k = 0; k < nz; k++) maximum_dz = std::max(maximum_dz,dz_host(k));
    int const maximum_multiplier = modules::DigitalFilterTurbulentInflow::compute_maximum_octave_level(
                                     coupler.get_ylen(),coupler.get_zlen(),coupler.get_dy(),maximum_dz,
                                     config.maximum_length_fraction);
    int const maximum_streamwise_length = std::max(config.streamwise_length_cells,maximum_multiplier);
    real const expected_time_scale = maximum_streamwise_length*coupler.get_dx()/mean_u;
    require(coupler,std::abs(inflow.get_time_scale()-expected_time_scale) < 1.e-12,
            "Digital-filter inflow used the wrong Taylor time scale");
    require(coupler,inflow.get_maximum_octave_level() == maximum_multiplier,
            "Digital-filter inflow did not honor its maximum octave cutoff");
    int const expected_octaves = modules::DigitalFilterTurbulentInflow::compute_num_octaves(
                                   coupler.get_ylen(),coupler.get_zlen(),coupler.get_dy(),maximum_dz,
                                   config.maximum_length_fraction);
    require(coupler,inflow.get_num_octaves() == expected_octaves,
            "Digital-filter inflow created the wrong number of dyadic octaves");
    check_octave_structure(coupler,inflow);
    if (!periodic_transverse) check_exterior_octave_stationarity(coupler,inflow);
    if (use_edge_sponge) edge_sponge.set_column(coupler,{"density_dry","temperature"});

    if (octave_only_64) {
      // Initialization plus the octave and exterior-stationarity checks above are the complete focused test.
      coupler.write_output_file("digital_filter_turbulent_inflow_octaves_64",false);
    } else if (visualize) {
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
      std::string const output_prefix = periodic_transverse ? "digital_filter_turbulent_inflow_periodic" :
                                                              "digital_filter_turbulent_inflow";
      coupler.write_output_file(output_prefix,false);

      inflow.apply(coupler,dycore,dt);
      auto second_u = copy_u_fluctuation(coupler,mean_u);
      check_temporal_correlation(coupler,first_u,second_u);
      real const uninterrupted_signature = ghost_signature(coupler);
      std::string const extension = core::FileIO::default_backend() == "adios2" ? ".bp" : ".nc";
      coupler.set_option<std::string>("restart_file",output_prefix+"_00000000"+extension);
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

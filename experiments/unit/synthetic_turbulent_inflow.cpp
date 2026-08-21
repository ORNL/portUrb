#include "coupler.h"
#include "dynamics_cell_centered.h"
#include "edge_sponge.h"
#include "sc_init.h"
#include "synthetic_turbulent_inflow.h"

namespace {

  void require(core::Coupler const &coupler, bool condition, std::string const &message) {
    int const valid = coupler.get_parallel_comm().all_reduce(condition ? 1 : 0,MPI_MIN);
    if (valid == 0) endrun(message.c_str());
  }

  real ghost_signature(core::Coupler const &coupler) {
    using FLOC = modules::Dynamics_Euler_Stratified::FLOC;
    if (coupler.get_px() != 0) return coupler.get_parallel_comm().all_reduce(0._fp,MPI_SUM);
    auto ghost = coupler.get_data_manager_readonly().get<FLOC const,6>("dycore_ghost_x1");
    real1d values("synthetic_inflow_signature",ghost.extent(3)*ghost.extent(4)*ghost.extent(5));
    int const nz = ghost.extent(3);
    int const ny = ghost.extent(4);
    int const hs = ghost.extent(5);
    yakl::parallel_for(YAKL_AUTO_LABEL(),values.extent(0),KOKKOS_LAMBDA (int index) {
      int const ii = index % hs;
      int const j  = (index/hs) % ny;
      int const k  = index/(hs*ny);
      values(index) = ghost(0,0,modules::Dynamics_Euler_Stratified::idU,k,j,ii) +
                      2*ghost(0,0,modules::Dynamics_Euler_Stratified::idV,k,j,ii) +
                      3*ghost(0,0,modules::Dynamics_Euler_Stratified::idW,k,j,ii);
    });
    return coupler.get_parallel_comm().all_reduce(yakl::intrinsics::sum(values),MPI_SUM);
  }

  void check_generated_ghosts(core::Coupler const &coupler, real1d const &u_mean, real1d const &intensity) {
    using FLOC = modules::Dynamics_Euler_Stratified::FLOC;
    int const nz = coupler.get_nz();
    int const ny = coupler.get_ny();
    int const ny_glob = coupler.get_ny_glob();
    int const j_beg = coupler.get_j_beg();
    real1d mean_sums("synthetic_inflow_mean_sums",nz);
    mean_sums = 0;
    real local_velocity_energy = 0;
    real local_inner_energy = 0;
    real local_target_energy = 0;
    real local_density_range = 0;
    real local_v_wall = 0;
    real local_v_center = 0;
    real local_w_wall = 0;
    real local_w_center = 0;
    if (coupler.get_px() == 0) {
      auto ghost = coupler.get_data_manager_readonly().get<FLOC const,6>("dycore_ghost_x1");
      int const hs = modules::Dynamics_Euler_Stratified::hs;
      real1d work("synthetic_inflow_ghost_work",nz*ny*hs);
      yakl::parallel_for(YAKL_AUTO_LABEL(),nz,KOKKOS_LAMBDA (int k) {
        for (int j = 0; j < ny; j++) {
          mean_sums(k) += ghost(0,0,modules::Dynamics_Euler_Stratified::idU,k,j,0);
        }
      });

      yakl::parallel_for(YAKL_AUTO_LABEL(),work.extent(0),KOKKOS_LAMBDA (int index) {
        int const ii = index % hs;
        int const j  = (index/hs) % ny;
        int const k  = index/(hs*ny);
        real const up = ghost(0,0,modules::Dynamics_Euler_Stratified::idU,k,j,ii)-u_mean(k);
        real const vp = ghost(0,0,modules::Dynamics_Euler_Stratified::idV,k,j,ii);
        real const wp = ghost(0,0,modules::Dynamics_Euler_Stratified::idW,k,j,ii);
        work(index) = up*up + vp*vp + wp*wp;
      });
      local_velocity_energy = yakl::intrinsics::sum(work);

      real1d inner_energy("synthetic_inflow_inner_energy",nz*ny);
      real1d target_energy("synthetic_inflow_target_energy",nz*ny);
      yakl::parallel_for(YAKL_AUTO_LABEL(),inner_energy.extent(0),KOKKOS_LAMBDA (int index) {
        int const j = index % ny;
        int const k = index/ny;
        real const up = ghost(0,0,modules::Dynamics_Euler_Stratified::idU,k,j,0)-u_mean(k);
        real const vp = ghost(0,0,modules::Dynamics_Euler_Stratified::idV,k,j,0);
        real const wp = ghost(0,0,modules::Dynamics_Euler_Stratified::idW,k,j,0);
        inner_energy(index) = up*up + vp*vp + wp*wp;
        target_energy(index) = 3*intensity(k)*intensity(k)*u_mean(k)*u_mean(k);
      });
      local_inner_energy  = yakl::intrinsics::sum(inner_energy);
      local_target_energy = yakl::intrinsics::sum(target_energy);

      real1d density_values("synthetic_inflow_density_values",nz*ny*hs);
      yakl::parallel_for(YAKL_AUTO_LABEL(),density_values.extent(0),KOKKOS_LAMBDA (int index) {
        int const ii = index % hs;
        int const j  = (index/hs) % ny;
        int const k  = index/(hs*ny);
        density_values(index) = ghost(0,0,modules::Dynamics_Euler_Stratified::idR,k,j,ii);
      });
      local_density_range = yakl::intrinsics::maxval(density_values)-yakl::intrinsics::minval(density_values);

      int const kcenter = nz/2;
      int const jcenter_glob = ny_glob/2;
      int const jcenter = jcenter_glob-j_beg;
      real1d samples("synthetic_inflow_wall_samples",hs);
      yakl::parallel_for(YAKL_AUTO_LABEL(),hs,KOKKOS_LAMBDA (int ii) {
        real const vy1 = j_beg == 0 ? ghost(0,0,modules::Dynamics_Euler_Stratified::idV,kcenter,0,ii) : 0;
        real const vy2 = j_beg+ny == ny_glob ?
                         ghost(0,0,modules::Dynamics_Euler_Stratified::idV,kcenter,ny-1,ii) : 0;
        samples(ii) = vy1*vy1 + vy2*vy2;
      });
      local_v_wall = yakl::intrinsics::sum(samples);
      yakl::parallel_for(YAKL_AUTO_LABEL(),hs,KOKKOS_LAMBDA (int ii) {
        real const value = jcenter >= 0 && jcenter < ny ?
                           ghost(0,0,modules::Dynamics_Euler_Stratified::idV,kcenter,jcenter,ii) : 0;
        samples(ii) = value*value;
      });
      local_v_center = yakl::intrinsics::sum(samples);
      yakl::parallel_for(YAKL_AUTO_LABEL(),hs,KOKKOS_LAMBDA (int ii) {
        real const wz1 = jcenter >= 0 && jcenter < ny ?
                         ghost(0,0,modules::Dynamics_Euler_Stratified::idW,0,jcenter,ii) : 0;
        real const wz2 = jcenter >= 0 && jcenter < ny ?
                         ghost(0,0,modules::Dynamics_Euler_Stratified::idW,nz-1,jcenter,ii) : 0;
        samples(ii) = wz1*wz1 + wz2*wz2;
      });
      local_w_wall = yakl::intrinsics::sum(samples);
      yakl::parallel_for(YAKL_AUTO_LABEL(),hs,KOKKOS_LAMBDA (int ii) {
        real const value = jcenter >= 0 && jcenter < ny ?
                           ghost(0,0,modules::Dynamics_Euler_Stratified::idW,kcenter,jcenter,ii) : 0;
        samples(ii) = value*value;
      });
      local_w_center = yakl::intrinsics::sum(samples);
    }
    auto const comm = coupler.get_parallel_comm();
    mean_sums = comm.all_reduce(mean_sums,MPI_SUM,"synthetic_inflow_test_mean");
    real1d mean_errors("synthetic_inflow_mean_errors",nz);
    yakl::parallel_for(YAKL_AUTO_LABEL(),nz,KOKKOS_LAMBDA (int k) {
      mean_errors(k) = std::abs(mean_sums(k)/ny_glob-u_mean(k));
    });
    real const mean_error = yakl::intrinsics::maxval(mean_errors);
    real const velocity_energy = comm.all_reduce(local_velocity_energy,MPI_SUM);
    real const inner_energy = comm.all_reduce(local_inner_energy,MPI_SUM);
    real const target_energy = comm.all_reduce(local_target_energy,MPI_SUM);
    real const density_range = comm.all_reduce(local_density_range,MPI_MAX);
    real const v_wall = comm.all_reduce(local_v_wall,MPI_SUM);
    real const v_center = comm.all_reduce(local_v_center,MPI_SUM);
    real const w_wall = comm.all_reduce(local_w_wall,MPI_SUM);
    real const w_center = comm.all_reduce(local_w_center,MPI_SUM);
    require(coupler,mean_error < 2.e-5,"Synthetic inflow did not preserve the requested mean u profile");
    require(coupler,velocity_energy > 0,"Synthetic inflow did not generate velocity fluctuations");
    require(coupler,std::abs(inner_energy-target_energy) < 5.e-4*target_energy,
            "Synthetic inflow did not match the requested plane-averaged turbulence intensity");
    require(coupler,density_range < 1.e-6,"Synthetic inflow introduced density fluctuations into its ghost column");
    require(coupler,v_wall < v_center,"Spanwise wall-normal turbulent velocity did not decay toward y walls");
    require(coupler,w_wall < w_center,"Vertical wall-normal turbulent velocity did not decay toward z walls");
  }

  void check_scale_model(core::Coupler const &coupler, modules::SyntheticTurbulentInflow const &inflow,
                         real outer_length) {
    int const num_scales = inflow.get_num_scales();
    real const cutoff = inflow.get_smallest_length();
    auto fractions = inflow.get_scale_energy_fraction();
    auto amplitudes = inflow.get_scale_amplitude();
    auto counts = inflow.get_scale_eddy_count();
    require(coupler,num_scales == 2,"Synthetic inflow test did not exercise multiple inertial-range scales");
    real fraction_sum = 0;
    int population_sum = 0;
    for (int scale = 0; scale < num_scales; scale++) {
      real const band_outer = outer_length/std::pow(2._fp,scale);
      real const band_inner = std::max(cutoff,band_outer/2);
      real const expected = (std::pow(band_outer,2._fp/3._fp)-std::pow(band_inner,2._fp/3._fp)) /
                            (std::pow(outer_length,2._fp/3._fp)-std::pow(cutoff,2._fp/3._fp));
      require(coupler,std::abs(fractions(scale)-expected) < 1.e-12,
              "Synthetic inflow scale energy is not the exact integral of k^(-5/3)");
      require(coupler,std::isfinite(amplitudes(scale)) && amplitudes(scale) > 0,
              "Synthetic inflow produced an invalid scale-response calibration");
      fraction_sum += fractions(scale);
      population_sum += counts(scale);
    }
    require(coupler,std::abs(fraction_sum-1) < 1.e-12,
            "Synthetic inflow inertial-range energy fractions do not sum to one");
    require(coupler,population_sum == inflow.get_num_eddies(),
            "Synthetic inflow band populations do not sum to the total population");
    require(coupler,counts(1) > counts(0),
            "Synthetic inflow did not assign more eddies to the smaller projected support");
    real const support_0 = counts(0)*outer_length*outer_length;
    real const support_1 = counts(1)*(outer_length/2)*(outer_length/2);
    require(coupler,std::abs(support_0-support_1) < 0.03_fp*std::max(support_0,support_1),
            "Synthetic inflow band populations do not provide comparable projected coverage");
  }

}

int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  Kokkos::initialize();
  yakl::init();
  {
    bool visualize = false;
    for (int arg = 1; arg < argc; arg++) {
      if (std::string(argv[arg]) == "--visualize") visualize = true;
    }
    core::Coupler coupler;
    int  constexpr nx_glob = 24;
    int  constexpr ny_glob = 16;
    int  constexpr nz      = 16;
    real constexpr dx      = 10;
    real constexpr xlen    = nx_glob*dx;
    real constexpr ylen    = ny_glob*dx;
    real constexpr zlen    = nz*dx;
    real constexpr dt      = 0.1;

    coupler.set_option<std::string>("init_data","constant");
    coupler.set_option<real>("constant_uvel",8);
    coupler.set_option<real>("constant_vvel",0);
    coupler.set_option<real>("constant_temp",300);
    coupler.set_option<real>("constant_press",1.e5);
    coupler.set_option<real>("cfl",0.6);
    coupler.set_option<real>("dycore_cs",80);
    coupler.set_option<real>("dycore_max_wind",20);
    coupler.set_option<int>("dycore_max_cycles",4);
    coupler.set_option<std::string>("dycore_time_stepper","ssprk3");
    coupler.init(core::ParallelComm(MPI_COMM_WORLD),coupler.generate_levels_equal(nz,zlen),
                 ny_glob,nx_glob,ylen,xlen);
    coupler.add_tracer("water_vapor","water_vapor",true,true,true);
    coupler.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;
    custom_modules::sc_init(coupler);
    coupler.set_option<std::string>("bc_x1","precursor");
    coupler.set_option<std::string>("bc_x2","open");
    coupler.set_option<std::string>("bc_y1","wall_free_slip");
    coupler.set_option<std::string>("bc_y2","wall_free_slip");
    coupler.set_option<std::string>("bc_z1","wall_free_slip");
    coupler.set_option<std::string>("bc_z2","wall_free_slip");

    modules::Dynamics_Euler_Stratified dycore;
    modules::SyntheticTurbulentInflow inflow;
    modules::EdgeSponge edge_sponge;
    dycore.init(coupler);

    real1d u_mean("synthetic_inflow_test_u_mean",nz);
    real1d v_mean("synthetic_inflow_test_v_mean",nz);
    real1d intensity("synthetic_inflow_test_intensity",nz);
    auto zmid = coupler.get_zmid();
    yakl::parallel_for(YAKL_AUTO_LABEL(),nz,KOKKOS_LAMBDA (int k) {
      u_mean(k) = 8 + 2*zmid(k)/zlen;
      v_mean(k) = 0;
      intensity(k) = 0.10;
    });
    modules::SyntheticTurbulentInflow::Config config;
    config.minimum_eddies = 1;
    config.cells_per_eddy = 0.5;
    config.random_seed = 731;
    config.outer_length = 120;
    config.wall_decay_length = 30;
    inflow.init(coupler,dycore,u_mean,v_mean,intensity,config);
    require(coupler,inflow.get_num_eddies() == 2*ny_glob*nz,
            "Synthetic inflow automatic population did not scale with the global inlet slab cells");
    check_scale_model(coupler,inflow,config.outer_length);
    edge_sponge.set_column(coupler,{"density_dry","temperature"});

    if (visualize) {
      // Forty-five seconds carries even the slowest prescribed mean flow through more than one complete 240 m
      // domain length. One-second snapshots resolve passage of the smallest 40 m nominal structures with several
      // frames while keeping the visualization data set compact.
      int  constexpr visualization_steps = 450;
      int  constexpr output_stride       = 10;
      real etime = 0;
      coupler.write_output_file("synthetic_turbulent_inflow_visualization",false);
      for (int step = 0; step < visualization_steps; step++) {
        inflow.apply(coupler,dycore,dt);
        edge_sponge.apply(coupler,0.1,0.1,0,0);
        dycore.time_step(coupler,dt);
        etime += dt;
        coupler.set_option<real>("elapsed_time",etime);
        if ((step+1) % output_stride == 0) {
          coupler.write_output_file("synthetic_turbulent_inflow_visualization",false);
        }
      }
    } else {
      inflow.apply(coupler,dycore,dt);
      check_generated_ghosts(coupler,u_mean,intensity);
      edge_sponge.apply(coupler,0.1,0.1,0,0);
      dycore.time_step(coupler,dt);
      coupler.set_option<real>("elapsed_time",dt);
      coupler.write_output_file("synthetic_turbulent_inflow",false);

      inflow.apply(coupler,dycore,dt);
      real const uninterrupted_signature = ghost_signature(coupler);
      std::string const extension = core::FileIO::default_backend() == "adios2" ? ".bp" : ".nc";
      coupler.set_option<std::string>("restart_file","synthetic_turbulent_inflow_00000000"+extension);
      coupler.overwrite_with_restart();
      inflow.apply(coupler,dycore,dt);
      real const restarted_signature = ghost_signature(coupler);
      real const signature_scale = std::max(1._fp,std::abs(uninterrupted_signature));
      require(coupler,std::abs(restarted_signature-uninterrupted_signature) < 1.e-11*signature_scale,
              "Synthetic inflow restart did not reproduce uninterrupted eddy evolution");
    }

    auto const &dm = coupler.get_data_manager_readonly();
    auto density = dm.get<real const,3>("density_dry");
    auto temperature = dm.get<real const,3>("temperature");
    require(coupler,std::isfinite(yakl::intrinsics::minval(density)) &&
                    std::isfinite(yakl::intrinsics::maxval(density)) && yakl::intrinsics::minval(density) > 0,
            "Synthetic inflow integration produced invalid density");
    require(coupler,std::isfinite(yakl::intrinsics::minval(temperature)) &&
                    std::isfinite(yakl::intrinsics::maxval(temperature)) && yakl::intrinsics::minval(temperature) > 0,
            "Synthetic inflow integration produced invalid temperature");
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}

#pragma once

#include "coupler.h"

namespace custom_modules {

  // Accumulate scalar diagnostics from device-resident fields. Only reduction results cross to the host.
  // The bounds intentionally describe broad physical regimes rather than a particular chaotic realization.
  class FinalStateAssessment {
  public:
    FinalStateAssessment(core::Coupler const & coupler, std::string const & test_name) :
      coupler(coupler), test_name(test_name), dm(coupler.get_data_manager_readonly()),
      comm(coupler.get_parallel_comm()), valid(true), checks(0) {
      auto const global_cells = coupler.get_nx_glob() * coupler.get_ny_glob() * coupler.get_nz();
      if (global_cells > 50000) {
        if (coupler.is_mainproc()) {
          std::cerr << test_name << ": grid has " << global_cells
                    << " cells; integration tests are limited to 50000" << std::endl;
        }
        endrun("Integration-test grid is too large");
      }
      dm.validate_all(true, __FILE__, __LINE__, coupler.get_myrank());
    }

    void check_field_range(std::string const & name, real lower, real upper) {
      auto const field      = dm.get_collapsed<real const>(name);
      auto const global_min = comm.all_reduce(yakl::intrinsics::minval(field), MPI_MIN);
      auto const global_max = comm.all_reduce(yakl::intrinsics::maxval(field), MPI_MAX);
      check_interval(name + " range", global_min, global_max, lower, upper);
    }

    void check_field_range_if_present(std::string const & name, real lower, real upper) {
      if (dm.entry_exists(name)) check_field_range(name, lower, upper);
    }

    void check_field_mean(std::string const & name, real lower, real upper) {
      auto const field        = dm.get_collapsed<real const>(name);
      auto const global_sum   = comm.all_reduce(yakl::intrinsics::sum(field), MPI_SUM);
      auto const global_count = comm.all_reduce(field.size(), MPI_SUM);
      check_scalar(name + " mean", global_sum / global_count, lower, upper);
    }

    void check_volume_weighted_mean(std::string const & name, real lower, real upper) {
      auto const nx    = coupler.get_nx();
      auto const ny    = coupler.get_ny();
      auto const nz    = coupler.get_nz();
      auto const dz    = coupler.get_dz();
      auto const field = dm.get<real const,3>(name);
      real3d weighted_field("integration_test_weighted_field", nz, ny, nx);
      yakl::parallel_for(YAKL_AUTO_LABEL(), yakl::SimpleBounds<3>(nz,ny,nx), KOKKOS_LAMBDA (int k, int j, int i) {
        weighted_field(k,j,i) = field(k,j,i) * dz(k);
      });
      auto const global_sum = comm.all_reduce(yakl::intrinsics::sum(weighted_field), MPI_SUM);
      auto const weight_sum = coupler.get_nx_glob() * coupler.get_ny_glob() * yakl::intrinsics::sum(dz);
      check_scalar(name + " volume-weighted mean", global_sum / weight_sum, lower, upper);
    }

    void check_field_max(std::string const & name, real lower, real upper) {
      auto const field      = dm.get_collapsed<real const>(name);
      auto const global_max = comm.all_reduce(yakl::intrinsics::maxval(field), MPI_MAX);
      check_scalar(name + " maximum", global_max, lower, upper);
    }

    void check_velocity_magnitude(real upper) {
      auto const uvel = dm.get_collapsed<real const>("uvel");
      auto const vvel = dm.get_collapsed<real const>("vvel");
      auto const wvel = dm.get_collapsed<real const>("wvel");
      real1d velocity_magnitude("integration_test_velocity_magnitude", uvel.size());
      yakl::parallel_for(YAKL_AUTO_LABEL(), yakl::SimpleBounds<1>(uvel.size()), KOKKOS_LAMBDA (int i) {
        velocity_magnitude(i) = std::sqrt(uvel(i) * uvel(i) + vvel(i) * vvel(i) + wvel(i) * wvel(i));
      });
      auto const global_max = comm.all_reduce(yakl::intrinsics::maxval(velocity_magnitude), MPI_MAX);
      check_scalar("velocity magnitude maximum", global_max, 0, upper);
    }

    void check_max_ratio(std::string const & numerator_name, std::string const & denominator_name, real upper) {
      auto const numerator   = dm.get_collapsed<real const>(numerator_name);
      auto const denominator = dm.get_collapsed<real const>(denominator_name);
      real1d ratio("integration_test_ratio", numerator.size());
      yakl::parallel_for(YAKL_AUTO_LABEL(), yakl::SimpleBounds<1>(numerator.size()), KOKKOS_LAMBDA (int i) {
        ratio(i) = numerator(i) / denominator(i);
      });
      auto const global_max = comm.all_reduce(yakl::intrinsics::maxval(ratio), MPI_MAX);
      check_scalar(numerator_name + " / " + denominator_name + " maximum", global_max, 0, upper);
    }

    void check_positive_tracers() {
      for (auto const & tracer_name : coupler.get_tracer_names()) {
        std::string tracer_desc;
        bool tracer_found;
        bool positive;
        bool adds_mass;
        bool diffuse;
        coupler.get_tracer_info(tracer_name, tracer_desc, tracer_found, positive, adds_mass, diffuse);
        if (tracer_found && positive) check_field_range(tracer_name, 0, std::numeric_limits<real>::max());
      }
    }

    void finish() {
      int const globally_valid = comm.all_reduce(valid ? 1 : 0, MPI_MIN);
      if (globally_valid == 0) endrun("Integration test produced a physically implausible solution");

      if (coupler.is_mainproc()) {
        auto const global_cells = coupler.get_nx_glob() * coupler.get_ny_glob() * coupler.get_nz();
        std::cout << test_name << ": PASS (" << global_cells << " cells; " << checks
                  << " final-state physical checks)" << std::endl;
      }
    }

  private:
    core::Coupler const &       coupler;
    std::string const &         test_name;
    core::DataManager<> const & dm;
    core::ParallelComm const    comm;
    bool                        valid;
    int                         checks;

    void check_scalar(std::string const & description, real value, real lower, real upper) {
      checks++;
      if (value < lower || value > upper) {
        valid = false;
        if (coupler.is_mainproc()) {
          std::cerr << test_name << ": " << description << " = " << value
                    << " is outside [" << lower << ", " << upper << "]" << std::endl;
        }
      }
    }

    void check_interval(std::string const & description, real minimum, real maximum, real lower, real upper) {
      checks++;
      if (minimum < lower || maximum > upper) {
        valid = false;
        if (coupler.is_mainproc()) {
          std::cerr << test_name << ": " << description << " = [" << minimum << ", " << maximum
                    << "] is outside [" << lower << ", " << upper << "]" << std::endl;
        }
      }
    }
  };


  inline void check_atmosphere(FinalStateAssessment & check, real density_min, real density_max,
                               real temperature_min, real temperature_max, real velocity_max,
                               real tke_max) {
    check.check_field_range("density_dry", density_min, density_max);
    check.check_field_range("temperature", temperature_min, temperature_max);
    check.check_velocity_magnitude(velocity_max);
    check.check_field_range_if_present("TKE", 0, tke_max);
    check.check_positive_tracers();
  }


  inline void check_dry_abl_solution(core::Coupler const & coupler, std::string const & test_name,
                                     real density_min, real density_max, real temperature_min,
                                     real temperature_max, real velocity_max, real mean_u_min,
                                     real mean_u_max, real mean_w_max, real tke_activity_min) {
    FinalStateAssessment check(coupler, test_name);
    check_atmosphere(check, density_min, density_max, temperature_min, temperature_max, velocity_max, 5);
    check.check_volume_weighted_mean("uvel", mean_u_min, mean_u_max);
    check.check_volume_weighted_mean("wvel", -mean_w_max, mean_w_max);
    check.check_field_max("TKE", tke_activity_min, 5);
    check.check_field_range_if_present("water_vapor", 0, 1.e-14);
    check.check_field_range("immersed_proportion", 0, 0);
    check.finish();
  }


  inline void check_abl_convective_solution(core::Coupler const & coupler, std::string const & test_name) {
    check_dry_abl_solution(coupler, test_name, 0.7, 1.3, 270, 320, 20, 8, 12, 0.1, 1.e-6);
  }


  inline void check_abl_neutral_solution(core::Coupler const & coupler, std::string const & test_name) {
    check_dry_abl_solution(coupler, test_name, 0.8, 1.4, 280, 315, 15, 8, 12, 0.1, 1.e-6);
  }


  inline void check_abl_stable_solution(core::Coupler const & coupler, std::string const & test_name) {
    check_dry_abl_solution(coupler, test_name, 1.1, 1.5, 250, 280, 10, 6, 9, 0.05, 1.e-6);
  }


  inline void check_city_solution(core::Coupler const & coupler, std::string const & test_name) {
    FinalStateAssessment check(coupler, test_name);
    check_atmosphere(check, 0.8, 1.4, 280, 310, 30, 50);
    check.check_volume_weighted_mean("uvel", 5, 20);
    check.check_volume_weighted_mean("wvel", -0.5, 0.5);
    check.check_field_max("TKE", 0.01, 50);
    check.check_field_range_if_present("water_vapor", 0, 1.e-14);
    check.check_field_range("immersed_proportion", 0, 1);
    check.check_field_mean("immersed_proportion", 0.005, 0.25);
    check.check_field_max("immersed_proportion", 0.99, 1);
    check.finish();
  }


  inline void check_kessler_solution(core::Coupler const & coupler, std::string const & test_name) {
    FinalStateAssessment check(coupler, test_name);
    check_atmosphere(check, 0.05, 1.3, 190, 310, 100, 10);
    check.check_volume_weighted_mean("uvel", 10, 25);
    check.check_volume_weighted_mean("wvel", -0.1, 0.1);
    check.check_max_ratio("water_vapor", "density_dry", 0.05);
    check.check_max_ratio("cloud_liquid", "density_dry", 0.05);
    check.check_max_ratio("precip_liquid", "density_dry", 0.05);
    check.check_field_range("immersed_proportion", 0, 0);
    check.finish();
  }


  inline void check_morrison_solution(core::Coupler const & coupler, std::string const & test_name) {
    FinalStateAssessment check(coupler, test_name);
    check_atmosphere(check, 0.05, 1.3, 190, 310, 100, 10);
    check.check_volume_weighted_mean("uvel", 10, 25);
    check.check_volume_weighted_mean("wvel", -0.1, 0.1);
    for (auto const & name : {"water_vapor", "cloud_water", "rain_water", "cloud_ice", "snow", "graupel"}) {
      check.check_max_ratio(name, "density_dry", 0.05);
    }
    check.check_field_range_if_present("micro_rainnc", 0, std::numeric_limits<real>::max());
    check.check_field_range_if_present("micro_snownc", 0, std::numeric_limits<real>::max());
    check.check_field_range_if_present("micro_graupelnc", 0, std::numeric_limits<real>::max());
    check.check_field_range("immersed_proportion", 0, 0);
    check.finish();
  }


  inline void check_turbine_simple_solution(core::Coupler const & coupler, std::string const & test_name) {
    FinalStateAssessment check(coupler, test_name);
    check_atmosphere(check, 0.8, 1.4, 280, 315, 20, 20);
    check.check_volume_weighted_mean("uvel", 7, 12);
    check.check_volume_weighted_mean("wvel", -0.5, 0.5);
    check.check_field_max("TKE", 0.01, 20);
    check.check_field_range_if_present("water_vapor", 0, 1.e-14);
    check.check_field_range("immersed_proportion", 0, 0);
    check.finish();
  }


  inline void check_turbine_ensemble_solution(core::Coupler const & coupler, std::string const & test_name,
                                              real hub_wind, bool turbine_active) {
    FinalStateAssessment check(coupler, test_name);
    check_atmosphere(check, 0.7, 1.3, 260, 320, 50, 25);
    check.check_volume_weighted_mean("uvel", 0.5 * hub_wind, 1.5 * hub_wind);
    check.check_volume_weighted_mean("wvel", -0.5, 0.5);
    check.check_field_max("TKE", turbine_active ? 0.01 : 1.e-6, 25);
    check.check_field_range_if_present("water_vapor", 0, 1.e-14);
    check.check_field_range("immersed_proportion", 0, 0);
    check.finish();
  }

} // namespace custom_modules

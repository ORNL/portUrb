#pragma once

#include "coupler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace custom_modules {

  inline bool restart_check_requested(int argc, char **argv, std::string const &test_name) {
    bool restart_check = false;
    for (int i=1; i < argc; i++) {
      if (std::string(argv[i]) == "--restart-check") restart_check = true;
      else endrun(("Unknown "+test_name+" argument: "+argv[i]).c_str());
    }
    return restart_check;
  }


  inline std::string restart_check_filename(std::string const &output_prefix) {
    std::string const extension = core::FileIO::default_backend() == "adios2" ? ".bp" : ".nc";
    return output_prefix+"_00000001"+extension;
  }


  inline std::string restart_check_snapshot_prefix(std::string const &output_prefix) {
    return output_prefix+"_restart_reference";
  }

  namespace restart_consistency {

    // This deliberately simple rank-local format is independent of FileIO so restart reads are checked against an
    // exact reference obtained through a separate host-copy path.

    inline std::string snapshot_filename(std::string const &prefix, int rank) {
      std::ostringstream filename;
      filename << prefix << ".rank" << std::setfill('0') << std::setw(6) << rank << ".bin";
      return filename.str();
    }

    inline std::string diff_filename(std::string const &prefix, int rank) {
      std::ostringstream filename;
      filename << prefix << ".rank" << std::setfill('0') << std::setw(6) << rank << ".diff";
      return filename.str();
    }

    template <class T>
    void write_scalar(std::ostream &stream, T const &value) {
      stream.write(reinterpret_cast<char const *>(&value),sizeof(T));
      if (!stream) throw std::runtime_error("Unable to write DataManager snapshot metadata");
    }

    template <class T>
    T read_scalar(std::istream &stream) {
      T value;
      stream.read(reinterpret_cast<char *>(&value),sizeof(T));
      if (!stream) throw std::runtime_error("Unable to read DataManager snapshot metadata");
      return value;
    }

    inline std::uint64_t checksum(char const *data, std::uint64_t bytes) {
      std::uint64_t value = 1469598103934665603ULL;
      for (std::uint64_t i=0; i < bytes; i++) {
        value ^= static_cast<unsigned char>(data[i]);
        value *= 1099511628211ULL;
      }
      return value;
    }

    inline void write_string(std::ostream &stream, std::string const &value) {
      write_scalar(stream,static_cast<std::uint64_t>(value.size()));
      stream.write(value.data(),value.size());
      if (!stream) throw std::runtime_error("Unable to write DataManager snapshot string");
    }

    inline std::string read_string(std::istream &stream) {
      auto const size = read_scalar<std::uint64_t>(stream);
      std::string value(size,'\0');
      stream.read(value.data(),size);
      if (!stream) throw std::runtime_error("Unable to read DataManager snapshot string");
      return value;
    }

    struct FloatingDiff {
      long double error_l2  = 0;
      long double limit_l2  = 0;
      long double error_linf = 0;
      long double limit_linf = 0;
      bool finite = true;
      bool valid  = true;
    };

    template <class T>
    FloatingDiff compare_floating(std::vector<char> const &reference_bytes, T const *current, std::uint64_t count,
                                  real relative_tolerance) {
      long double reference_l2_squared = 0;
      long double current_l2_squared   = 0;
      long double error_l2_squared     = 0;
      long double reference_linf       = 0;
      long double current_linf         = 0;
      long double error_linf           = 0;
      bool finite = true;
      for (std::uint64_t i=0; i < count; i++) {
        T reference;
        std::memcpy(&reference,reference_bytes.data()+i*sizeof(T),sizeof(T));
        long double const reference_value = reference;
        long double const current_value   = current[i];
        long double const error           = current_value-reference_value;
        finite = finite && std::isfinite(reference_value) && std::isfinite(current_value);
        reference_l2_squared += reference_value*reference_value;
        current_l2_squared   += current_value*current_value;
        error_l2_squared     += error*error;
        reference_linf = std::max(reference_linf,std::abs(reference_value));
        current_linf   = std::max(current_linf  ,std::abs(current_value  ));
        error_linf     = std::max(error_linf    ,std::abs(error          ));
      }
      if (count == 0) return {};
      long double const reference_l2 = std::sqrt(reference_l2_squared);
      long double const current_l2   = std::sqrt(current_l2_squared);
      long double const error_l2     = std::sqrt(error_l2_squared);
      // Use only the representable numerical floor as the absolute term. This keeps the mixed criterion sensitive to
      // physically tiny fields while relative tolerance handles ordinary single-precision restart quantization.
      long double const absolute_tolerance = std::numeric_limits<T>::denorm_min();
      long double const l2_limit = absolute_tolerance*std::sqrt(static_cast<long double>(count)) +
                                   relative_tolerance*std::max(reference_l2,current_l2);
      long double const linf_limit = absolute_tolerance +
                                     relative_tolerance*std::max(reference_linf,current_linf);
      return {error_l2,l2_limit,error_linf,linf_limit,finite,
              finite && error_l2 <= l2_limit && error_linf <= linf_limit};
    }

    inline std::string format_floating_diff(std::string const &name, FloatingDiff const &diff) {
      std::ostringstream line;
      line << "  " << name << std::setprecision(17)
           << ": L2 diff / limit = " << diff.error_l2 << " / " << diff.limit_l2
           << ", L_inf diff / limit = " << diff.error_linf << " / " << diff.limit_linf
           << ", finite = " << diff.finite;
      if (!diff.valid) line << " [FAILED]";
      return line.str();
    }

    inline std::string format_failure(std::string const &name, std::string const &description) {
      return "  "+name+": "+description+" [FAILED]";
    }

  } // namespace restart_consistency


  inline void write_data_manager_snapshot(core::Coupler const &coupler, std::string const &prefix) {
    using namespace restart_consistency;
    auto const &dm = coupler.get_data_manager_readonly();
    int const rank = coupler.get_myrank();
    std::ofstream stream(snapshot_filename(prefix,rank),std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Unable to create DataManager snapshot file");

    std::array<char,8> constexpr magic = {'P','U','R','B','D','M','0','1'};
    stream.write(magic.data(),magic.size());
    write_scalar(stream,std::uint32_t(1));
    write_scalar(stream,std::int32_t(rank));
    write_scalar(stream,std::int32_t(coupler.get_nranks()));

    std::vector<std::string> names;
    names.reserve(dm.entries.size());
    for (auto const &[name,entry] : dm.entries) names.push_back(name);
    std::sort(names.begin(),names.end());
    write_scalar(stream,static_cast<std::uint64_t>(names.size()));

    Kokkos::fence();
    for (auto const &name : names) {
      write_string(stream,name);
      auto const &value = dm.entries.at(name);
      write_scalar(stream,static_cast<std::uint32_t>(value.index()));
      std::visit([&](auto const &entry) {
        using T = typename std::remove_cvref_t<decltype(entry)>::value_type;
        write_scalar(stream,static_cast<std::uint32_t>(entry.dims.size()));
        for (int dim : entry.dims) write_scalar(stream,static_cast<std::int64_t>(dim));
        write_scalar(stream,static_cast<std::uint8_t>(entry.positive));
        write_scalar(stream,static_cast<std::uint64_t>(sizeof(T)));
        write_scalar(stream,static_cast<std::uint64_t>(entry.bytes/sizeof(T)));
        write_scalar(stream,static_cast<std::uint64_t>(entry.bytes));
        auto host = dm.template get_collapsed<T const>(name).createHostCopy();
        auto const payload_checksum = checksum(reinterpret_cast<char const *>(host.data()),entry.bytes);
        write_scalar(stream,payload_checksum);
        stream.write(reinterpret_cast<char const *>(host.data()),entry.bytes);
        if (!stream) throw std::runtime_error("Unable to write DataManager snapshot payload for "+name);
      },value);
    }
  }


  inline bool compare_data_manager_snapshot(core::Coupler const &coupler, std::string const &prefix,
                                            std::string const &test_name,
                                            std::map<std::string,std::string> const &ignored_entries,
                                            real relative_tolerance = 1.e-5) {
    using namespace restart_consistency;
    auto const &dm = coupler.get_data_manager_readonly();
    int const rank = coupler.get_myrank();
    bool local_valid = true;
    int compared_entries = 0;
    int ignored_common_entries = 0;
    std::set<std::string> snapshot_names;
    std::vector<std::string> report_lines;

    try {
      std::ifstream stream(snapshot_filename(prefix,rank),std::ios::binary);
      if (!stream) throw std::runtime_error("Unable to open DataManager snapshot file");
      std::array<char,8> magic;
      std::array<char,8> constexpr expected_magic = {'P','U','R','B','D','M','0','1'};
      stream.read(magic.data(),magic.size());
      if (magic != expected_magic) throw std::runtime_error("Invalid DataManager snapshot magic");
      if (read_scalar<std::uint32_t>(stream) != 1) throw std::runtime_error("Unsupported DataManager snapshot version");
      if (read_scalar<std::int32_t>(stream) != rank) throw std::runtime_error("DataManager snapshot rank mismatch");
      if (read_scalar<std::int32_t>(stream) != coupler.get_nranks()) {
        throw std::runtime_error("DataManager snapshot MPI decomposition mismatch");
      }

      auto const num_entries = read_scalar<std::uint64_t>(stream);
      for (std::uint64_t entry_index=0; entry_index < num_entries; entry_index++) {
        std::string const name = read_string(stream);
        auto const type_index = read_scalar<std::uint32_t>(stream);
        auto const num_dims = read_scalar<std::uint32_t>(stream);
        std::vector<int> dims(num_dims);
        for (int &dim : dims) dim = static_cast<int>(read_scalar<std::int64_t>(stream));
        bool const positive = read_scalar<std::uint8_t>(stream);
        auto const element_bytes = read_scalar<std::uint64_t>(stream);
        auto const count = read_scalar<std::uint64_t>(stream);
        auto const bytes = read_scalar<std::uint64_t>(stream);
        auto const expected_checksum = read_scalar<std::uint64_t>(stream);
        std::vector<char> reference_bytes(bytes);
        stream.read(reference_bytes.data(),bytes);
        if (!stream) throw std::runtime_error("Unable to read DataManager snapshot payload for "+name);
        if (checksum(reference_bytes.data(),bytes) != expected_checksum) {
          throw std::runtime_error("DataManager snapshot checksum mismatch for "+name);
        }
        if (!snapshot_names.insert(name).second) throw std::runtime_error("Duplicate snapshot entry "+name);

        auto const current = dm.entries.find(name);
        if (current == dm.entries.end()) {
          report_lines.push_back(format_failure(name,"missing from restarted DataManager"));
          local_valid = false;
          continue;
        }
        if (current->second.index() != type_index) {
          report_lines.push_back(format_failure(name,"type mismatch"));
          local_valid = false;
          continue;
        }

        bool entry_schema_valid = true;
        std::visit([&](auto const &entry) {
          using T = typename std::remove_cvref_t<decltype(entry)>::value_type;
          entry_schema_valid = entry.dims == dims && entry.positive == positive && element_bytes == sizeof(T) &&
                               count == entry.bytes/sizeof(T) && bytes == entry.bytes;
          if (!entry_schema_valid) return;
          if (ignored_entries.contains(name)) return;
          auto host = dm.template get_collapsed<T const>(name).createHostCopy();
          if constexpr (std::is_floating_point_v<T>) {
            auto const diff = compare_floating(reference_bytes,host.data(),count,relative_tolerance);
            report_lines.push_back(format_floating_diff(name,diff));
            local_valid = diff.valid && local_valid;
          } else {
            std::uint64_t differing_bytes = 0;
            auto const *current_bytes = reinterpret_cast<char const *>(host.data());
            for (std::uint64_t i=0; i < bytes; i++) differing_bytes += reference_bytes[i] != current_bytes[i];
            bool const valid = differing_bytes == 0;
            std::ostringstream line;
            line << "  " << name << ": differing bytes = " << differing_bytes << " / " << bytes;
            if (!valid) line << " [FAILED]";
            report_lines.push_back(line.str());
            local_valid = valid && local_valid;
          }
          compared_entries++;
        },current->second);
        if (!entry_schema_valid) {
          report_lines.push_back(format_failure(name,"shape or metadata mismatch"));
          local_valid = false;
        }
      }

      for (auto const &[name,entry] : dm.entries) {
        if (!snapshot_names.contains(name)) {
          report_lines.push_back(format_failure(name,"extra entry in restarted DataManager"));
          local_valid = false;
        }
      }
      for (auto const &[name,reason] : ignored_entries) {
        if (!snapshot_names.contains(name)) {
          report_lines.push_back(format_failure(name,"ignore list names an absent entry"));
          local_valid = false;
        } else if (dm.entries.contains(name)) {
          report_lines.push_back("  "+name+": ignored ("+reason+")");
          ignored_common_entries++;
        }
      }
    } catch (std::exception const &error) {
      report_lines.push_back(format_failure("snapshot",error.what()));
      local_valid = false;
    }

    auto const par_comm = coupler.get_parallel_comm();
    // Entry sets can differ on boundary ranks, so each rank checks its own complete snapshot before one final consensus.
    int const globally_valid = par_comm.all_reduce(local_valid ? 1 : 0,MPI_MIN);

    std::ostringstream report;
    report << test_name << " restart consistency differences";
    if (coupler.get_nranks() > 1) report << " (rank " << rank << ")";
    report << ":" << std::endl;
    for (auto const &line : report_lines) report << line << std::endl;
    report << test_name << " restart consistency: " << (globally_valid == 1 ? "PASS" : "FAIL") << " ("
           << compared_entries << " entries compared, " << ignored_common_entries << " ignored)";
    if (globally_valid == 0) report << " [FAILED]";
    report << std::endl;

    std::ofstream diff_stream(diff_filename(prefix,rank),std::ios::trunc);
    if (!diff_stream) throw std::runtime_error("Unable to create DataManager restart difference file");
    diff_stream << report.str();
    if (!diff_stream) throw std::runtime_error("Unable to write DataManager restart difference file");

    for (int report_rank=0; report_rank < coupler.get_nranks(); report_rank++) {
      par_comm.barrier();
      if (rank == report_rank) std::cout << report.str();
    }
    par_comm.barrier();
    return globally_valid == 1;
  }

} // namespace custom_modules

#include "coupler.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <ctime>
#include <filesystem>
#include <fcntl.h>
#include <iomanip>
#include <system_error>
#include <unistd.h>

namespace {

int constexpr num_cells   = 512;
int constexpr num_octaves = 8;
float constexpr mean_wind = 1.f;
float constexpr turbulence_intensity = 0.1f;

struct CompressionCase {
  std::string compressor;
  int clevel;
};

struct BenchmarkResult {
  CompressionCase configuration;
  std::vector<double> wall_seconds;
  std::vector<double> cpu_seconds;
  std::vector<std::uintmax_t> bytes;
};

struct RunResult {
  double wall_seconds;
  double cpu_seconds;
  std::uintmax_t bytes;
};

std::uintmax_t directory_size(std::filesystem::path const &path) {
  std::uintmax_t bytes = 0;
  for (auto const &entry : std::filesystem::recursive_directory_iterator(path)) {
    if (entry.is_regular_file()) bytes += entry.file_size();
  }
  return bytes;
}

double process_cpu_seconds() {
  timespec value;
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&value) != 0) {
    throw std::system_error(errno,std::generic_category(),"clock_gettime(CLOCK_PROCESS_CPUTIME_ID)");
  }
  return value.tv_sec+1.e-9*value.tv_nsec;
}

void sync_descriptor(int descriptor, std::string const &description, bool data_only) {
  int const status = data_only ? fdatasync(descriptor) : fsync(descriptor);
  if (status != 0) throw std::system_error(errno,std::generic_category(),"Synchronizing "+description);
}

void sync_directory(std::filesystem::path const &path) {
  int const descriptor = open(path.c_str(),O_RDONLY | O_DIRECTORY);
  if (descriptor < 0) throw std::system_error(errno,std::generic_category(),"Opening directory "+path.string());
  try {
    sync_descriptor(descriptor,path.string(),false);
  } catch (...) {
    close(descriptor);
    throw;
  }
  if (close(descriptor) != 0) throw std::system_error(errno,std::generic_category(),"Closing directory "+path.string());
}

void sync_output(std::filesystem::path const &path) {
  std::vector<std::filesystem::path> directories;
  directories.push_back(path);
  for (auto const &entry : std::filesystem::recursive_directory_iterator(path)) {
    if (entry.is_directory()) {
      directories.push_back(entry.path());
    } else if (entry.is_regular_file()) {
      int const descriptor = open(entry.path().c_str(),O_RDWR);
      if (descriptor < 0) {
        throw std::system_error(errno,std::generic_category(),"Opening output file "+entry.path().string());
      }
      try {
        sync_descriptor(descriptor,entry.path().string(),true);
      } catch (...) {
        close(descriptor);
        throw;
      }
      if (close(descriptor) != 0) {
        throw std::system_error(errno,std::generic_category(),"Closing output file "+entry.path().string());
      }
    }
  }
  for (auto const &directory : directories) sync_directory(directory);
  auto const parent = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
  sync_directory(parent);
}

template <class T>
double median(std::vector<T> values) {
  if (values.empty()) throw std::runtime_error("Cannot calculate a median of an empty sample");
  std::sort(values.begin(),values.end());
  auto const middle = values.size()/2;
  if (values.size()%2 == 1) return static_cast<double>(values[middle]);
  return 0.5*(static_cast<double>(values[middle-1])+static_cast<double>(values[middle]));
}

double median_absolute_deviation(std::vector<double> const &values) {
  double const center = median(values);
  std::vector<double> deviations;
  deviations.reserve(values.size());
  for (auto const value : values) deviations.push_back(std::abs(value-center));
  return median(std::move(deviations));
}

void generate_velocity(floatHost3d const &uvel, floatHost3d const &vvel, floatHost3d const &wvel,
                       int global_k_begin, MPI_Comm comm) {
  double constexpr pi = 3.141592653589793238462643383279502884;
  std::array<float,num_octaves> amplitudes;
  double amplitude_square_sum = 0;
  for (int octave = 0; octave < num_octaves; octave++) {
    int const wavenumber = 1 << octave;
    amplitudes[octave] = std::pow(static_cast<float>(wavenumber),-1.f/3.f);
    amplitude_square_sum += amplitudes[octave]*amplitudes[octave];
  }
  float const normalization = turbulence_intensity*mean_wind/std::sqrt(0.5*amplitude_square_sum);
  for (auto &amplitude : amplitudes) amplitude *= normalization;

  std::array<std::array<float,num_cells>,num_octaves> sin_x;
  std::array<std::array<float,num_cells>,num_octaves> cos_x;
  std::array<std::array<float,num_cells>,num_octaves> sin_y;
  std::array<std::array<float,num_cells>,num_octaves> cos_y;
  std::array<std::array<float,num_cells>,num_octaves> sin_z;
  std::array<std::array<float,num_cells>,num_octaves> cos_z;
  for (int octave = 0; octave < num_octaves; octave++) {
    int const wavenumber = 1 << octave;
    double const phase_x = 0.37*(octave+1);
    double const phase_y = 0.53*(octave+1);
    double const phase_z = 0.71*(octave+1);
    for (int i = 0; i < num_cells; i++) {
      double const angle_x = 2*pi*wavenumber*(i+0.5)/num_cells+phase_x;
      double const angle_y = 2*pi*wavenumber*(i+0.5)/num_cells+phase_y;
      double const angle_z = 2*pi*wavenumber*(i+0.5)/num_cells+phase_z;
      sin_x[octave][i] = std::sin(angle_x);
      cos_x[octave][i] = std::cos(angle_x);
      sin_y[octave][i] = std::sin(angle_y);
      cos_y[octave][i] = std::cos(angle_y);
      sin_z[octave][i] = std::sin(angle_z);
      cos_z[octave][i] = std::cos(angle_z);
    }
  }

  double local_perturbation_square_sum = 0;
  for (int k = 0; k < uvel.extent(0); k++) {
    int const global_k = global_k_begin+k;
    for (int j = 0; j < num_cells; j++) {
      for (int i = 0; i < num_cells; i++) {
        float up = 0;
        float vp = 0;
        float wp = 0;
        for (int octave = 0; octave < num_octaves; octave++) {
          float const amplitude = amplitudes[octave];
          float const sx = sin_x[octave][i];
          float const sy = sin_y[octave][j];
          float const sz = sin_z[octave][global_k];
          float const cx = cos_x[octave][i];
          float const cy = cos_y[octave][j];
          float const cz = cos_z[octave][global_k];
          // This cyclic curl construction is divergence-free for every octave.
          up += amplitude*sx*(cy-cz);
          vp += amplitude*sy*(cz-cx);
          wp += amplitude*sz*(cx-cy);
        }
        uvel(k,j,i) = mean_wind+up;
        vvel(k,j,i) = vp;
        wvel(k,j,i) = wp;
        local_perturbation_square_sum += static_cast<double>(up)*up+
                                         static_cast<double>(vp)*vp+
                                         static_cast<double>(wp)*wp;
      }
    }
  }

  double global_perturbation_square_sum = 0;
  MPI_Allreduce(&local_perturbation_square_sum,&global_perturbation_square_sum,1,MPI_DOUBLE,MPI_SUM,comm);
  double const cell_count = static_cast<double>(num_cells)*num_cells*num_cells;
  double const measured_intensity = std::sqrt(global_perturbation_square_sum/(3*cell_count))/mean_wind;
  int rank = 0;
  MPI_Comm_rank(comm,&rank);
  if (rank == 0) {
    std::cout << "Synthetic velocity field: " << num_cells << "^3 cells, three float components, "
              << "wavelengths 512 through 4 cells, measured turbulence intensity = "
              << std::setprecision(8) << measured_intensity << std::endl;
  }
  if (std::abs(measured_intensity-turbulence_intensity) > 1.e-6) {
    throw std::runtime_error("Synthetic velocity field does not have the requested turbulence intensity");
  }
}

RunResult run_case(core::Coupler &coupler, floatHost3d const &uvel, floatHost3d const &vvel,
                   floatHost3d const &wvel, int global_k_begin, CompressionCase const &configuration,
                   std::string const &run_name, bool keep_file, bool timed) {
  MPI_Comm const comm = MPI_COMM_WORLD;
  int rank = 0;
  MPI_Comm_rank(comm,&rank);
  coupler.set_option<std::string>("adios2_compression_compressor",configuration.compressor);
  coupler.set_option<int>("adios2_compression_clevel",configuration.clevel);

  std::string const stem = "file_io_turbulence_"+run_name+"_"+configuration.compressor+"_c"+
                           std::to_string(configuration.clevel);
  std::string const filename = stem+".bp";
  MPI_Barrier(comm);
  auto const begin = std::chrono::steady_clock::now();
  double const cpu_begin = process_cpu_seconds();
  {
    core::FileIO file(comm,coupler.get_option<std::string>("file_io_backend"),
                      coupler.get_option<int>("adios2_compression_min_bytes"),
                      coupler.get_option<std::string>("adios2_compression_compressor"),
                      coupler.get_option<int>("adios2_compression_clevel"));
    file.create(filename);
    file.create_dim("z",num_cells);
    file.create_dim("y",num_cells);
    file.create_dim("x",num_cells);
    std::vector<std::string> const dimensions = {"z","y","x"};
    file.create_var<float>("uvel",dimensions);
    file.create_var<float>("vvel",dimensions);
    file.create_var<float>("wvel",dimensions);
    file.writeGlobalAttribute(std::string("periodic divergence-free multiscale curl"),"synthetic_eddy_method");
    file.writeGlobalAttribute(turbulence_intensity,"turbulence_intensity");
    file.writeGlobalAttribute(4,"minimum_eddy_wavelength_cells");
    file.writeGlobalAttribute(configuration.compressor,"adios2_compression_compressor");
    file.writeGlobalAttribute(configuration.clevel,"adios2_compression_clevel");
    file.enddef();
    std::vector<MPI_Offset> const start = {global_k_begin,0,0};
    file.write_all(uvel,"uvel",start);
    file.write_all(vvel,"vvel",start);
    file.write_all(wvel,"wvel",start);
    file.close();
  }
  double const local_cpu_seconds = process_cpu_seconds()-cpu_begin;
  MPI_Barrier(comm);

  int sync_failed = 0;
  std::string sync_error;
  std::uintmax_t bytes = 0;
  if (rank == 0) {
    try {
      sync_output(filename);
      bytes = directory_size(filename);
    } catch (std::exception const &error) {
      sync_failed = 1;
      sync_error = error.what();
    }
  }
  MPI_Bcast(&sync_failed,1,MPI_INT,0,comm);
  if (sync_failed != 0) {
    if (rank == 0) std::cerr << "Durability synchronization failed: " << sync_error << std::endl;
    throw std::runtime_error("Unable to synchronize BP5 output to durable storage");
  }
  MPI_Barrier(comm);
  double const local_wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
  double wall_seconds = 0;
  double cpu_seconds = 0;
  MPI_Reduce(&local_wall_seconds,&wall_seconds,1,MPI_DOUBLE,MPI_MAX,0,comm);
  MPI_Reduce(&local_cpu_seconds,&cpu_seconds,1,MPI_DOUBLE,MPI_MAX,0,comm);

  {
    core::FileIO file(comm,"adios2");
    file.open(filename);
    bool const compressed = file.variable_has_operations("uvel") &&
                            file.variable_has_operations("vvel") &&
                            file.variable_has_operations("wvel");
    file.close();
    int const local_valid = compressed ? 1 : 0;
    int global_valid = 0;
    MPI_Allreduce(&local_valid,&global_valid,1,MPI_INT,MPI_MIN,comm);
    if (global_valid == 0) throw std::runtime_error("A velocity component is missing its ADIOS2 compression operation");
  }
  MPI_Barrier(comm);

  if (!keep_file && rank == 0) std::filesystem::remove_all(filename);
  MPI_Barrier(comm);
  if (!timed && rank == 0) {
    std::cout << "Warm-up complete: " << configuration.compressor << " level " << configuration.clevel << std::endl;
  }
  return {wall_seconds,cpu_seconds,bytes};
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  Kokkos::initialize();
  yakl::init();
  int result = 0;
  try {
    bool keep_files = false;
    if (argc == 2 && std::string(argv[1]) == "--keep-files") {
      keep_files = true;
    } else if (argc != 1) {
      throw std::runtime_error("Usage: file_io_turbulence_compression [--keep-files]");
    }
    int rank = 0;
    int nranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nranks);
    if (nranks > num_cells) throw std::runtime_error("The benchmark supports no more than 512 MPI ranks");

    int const global_k_begin = num_cells*rank/nranks;
    int const global_k_end   = num_cells*(rank+1)/nranks;
    int const local_nz       = global_k_end-global_k_begin;
    floatHost3d uvel("synthetic_uvel",local_nz,num_cells,num_cells);
    floatHost3d vvel("synthetic_vvel",local_nz,num_cells,num_cells);
    floatHost3d wvel("synthetic_wvel",local_nz,num_cells,num_cells);
    generate_velocity(uvel,vvel,wvel,global_k_begin,MPI_COMM_WORLD);

    core::Coupler coupler;
    coupler.set_option<int>("adios2_compression_min_bytes",1);
    std::array<CompressionCase,4> const cases = {{{"lz4",5},{"lz4hc",9},{"zstd",5},{"zstd",9}}};
    std::array<BenchmarkResult,4> results;
    for (int i = 0; i < cases.size(); i++) results[i].configuration = cases[i];

    int benchmark_id = 0;
    if (rank == 0) benchmark_id = static_cast<int>(getpid());
    MPI_Bcast(&benchmark_id,1,MPI_INT,0,MPI_COMM_WORLD);
    std::string const benchmark_name = std::to_string(benchmark_id);

    // Exercise every codec once before collecting samples, including a durability sync, to remove first-use effects.
    for (int i = 0; i < cases.size(); i++) {
      run_case(coupler,uvel,vvel,wvel,global_k_begin,cases[i],
               benchmark_name+"_warmup"+std::to_string(i),false,false);
    }

    // A cyclic Latin square places every configuration once in each sequence position.
    for (int rotation = 0; rotation < cases.size(); rotation++) {
      for (int position = 0; position < cases.size(); position++) {
        int const case_index = (position+rotation)%cases.size();
        bool const keep_sample = keep_files && rotation == cases.size()-1;
        std::string const run_name = keep_sample ? "kept" :
                                     benchmark_name+"_r"+std::to_string(rotation)+"p"+std::to_string(position);
        auto const sample = run_case(coupler,uvel,vvel,wvel,global_k_begin,cases[case_index],
                                     run_name,keep_sample,true);
        results[case_index].wall_seconds.push_back(sample.wall_seconds);
        results[case_index].cpu_seconds.push_back(sample.cpu_seconds);
        results[case_index].bytes.push_back(sample.bytes);
      }
    }

    if (rank == 0) {
      double constexpr gib = 1024.*1024.*1024.;
      double const raw_gib = 3.*num_cells*num_cells*num_cells*sizeof(float)/gib;
      std::cout << "\ncompressor,clevel,wall_median_s,wall_mad_s,wall_min_s,wall_max_s,cpu_median_s,cpu_mad_s,"
                << "size_median_GiB,compression_ratio\n";
      for (auto const &entry : results) {
        double const wall_median = median(entry.wall_seconds);
        double const wall_mad = median_absolute_deviation(entry.wall_seconds);
        double const cpu_median = median(entry.cpu_seconds);
        double const cpu_mad = median_absolute_deviation(entry.cpu_seconds);
        double const size_gib = median(entry.bytes)/gib;
        auto const wall_bounds = std::minmax_element(entry.wall_seconds.begin(),entry.wall_seconds.end());
        std::cout << entry.configuration.compressor << "," << entry.configuration.clevel << ","
                  << std::fixed << std::setprecision(6) << wall_median << "," << wall_mad << ","
                  << *wall_bounds.first << "," << *wall_bounds.second << ","
                  << cpu_median << "," << cpu_mad << "," << size_gib << "," << raw_gib/size_gib << "\n";
      }
    }
  } catch (std::exception const &error) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (rank == 0) std::cerr << "file_io_turbulence_compression: " << error.what() << std::endl;
    result = 1;
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
  return result;
}

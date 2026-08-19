#include "core/FileIO.h"

int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  Kokkos::initialize();
  yakl::init();
  int rank = 0;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nranks);
  if (nranks != 2) {
    if (rank == 0) std::cerr << "file_io_compression_mpi requires exactly two MPI ranks" << std::endl;
    yakl::finalize();
    Kokkos::finalize();
    MPI_Finalize();
    return 1;
  }

  std::array<std::string,4> const supported_compressors = {"blosclz","lz4","lz4hc","zstd"};
  for (auto const &compressor : supported_compressors) {
    core::FileIO file(MPI_COMM_WORLD,"adios2",1048576,compressor,1);
    if (file.compression_parameter("compressor") != compressor ||
        file.compression_parameter("clevel") != "1" ||
        file.compression_parameter("doshuffle") != "BLOSC_BITSHUFFLE") {
      throw std::runtime_error("FileIO rejected a supported native ADIOS2 Blosc2 configuration");
    }
  }
  for (auto const clevel : {0,10}) {
    bool rejected = false;
    try {
      core::FileIO file(MPI_COMM_WORLD,"adios2",1048576,"lz4",clevel);
    } catch (std::invalid_argument const &) {
      rejected = true;
    }
    if (!rejected) throw std::runtime_error("FileIO accepted an invalid Blosc2 compression level");
  }
  for (auto const &compressor : {std::string("zlib"),std::string("invalid")}) {
    bool rejected = false;
    try {
      core::FileIO file(MPI_COMM_WORLD,"adios2",1048576,compressor,5);
    } catch (std::invalid_argument const &) {
      rejected = true;
    }
    if (!rejected) throw std::runtime_error("FileIO accepted an unsupported Blosc2 compressor");
  }

  std::string const filename = "file_io_compression_mpi.bp";
  int constexpr nx = 131073;
  {
    core::FileIO file(MPI_COMM_WORLD,"adios2");
    file.create(filename);
    file.create_dim("y",2);
    file.create_dim("x",nx);
    file.create_var<float>("field",{"y","x"});
    if (file.compression_parameter("compressor") != "zstd" ||
        file.compression_parameter("clevel") != "5" ||
        file.compression_parameter("doshuffle") != "BLOSC_BITSHUFFLE") {
      throw std::runtime_error("FileIO did not configure the default native ADIOS2 Blosc2 operator");
    }
    file.writeVariableAttribute(std::string("test"),"field","units");
    floatHost2d local("file_io_compression_local",1,nx);
    for (int i = 0; i < nx; i++) local(0,i) = 10*rank+(i%17);
    file.write_all(local,"field",{rank,0});
    file.close();
  }

  int failed = 0;
  {
    core::FileIO file(MPI_COMM_WORLD,"adios2");
    file.open(filename);
    floatHost2d local("file_io_compression_restored",1,nx);
    file.read_all(local,"field",{rank,0});
    for (int i = 0; i < nx; i++) {
      if (local(0,i) != 10*rank+(i%17)) failed = 1;
    }
    if (!file.variable_has_operations("field")) failed = 1;
    file.close();
  }

  int global_failed = 0;
  MPI_Allreduce(&failed,&global_failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
  return global_failed;
}

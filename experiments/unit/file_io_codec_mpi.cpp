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
    if (rank == 0) std::cerr << "file_io_codec_mpi requires exactly two MPI ranks" << std::endl;
    yakl::finalize();
    Kokkos::finalize();
    MPI_Finalize();
    return 1;
  }

  std::string const filename = "file_io_codec_mpi.bp";
  int constexpr nx = 131073;
  {
    core::FileIO file(MPI_COMM_WORLD,"adios2");
    file.create(filename);
    file.create_dim("y",2);
    file.create_dim("x",nx);
    file.create_var<float>("field",{"y","x"});
    file.writeVariableAttribute(std::string("test"),"field","units");
    floatHost2d local("file_io_codec_local",1,nx);
    for (int i = 0; i < nx; i++) local(0,i) = 10*rank+(i%17);
    file.write_all(local,"field",{rank,0});
    file.close();
  }

  int failed = 0;
  {
    core::FileIO file(MPI_COMM_WORLD,"adios2");
    file.open(filename);
    floatHost2d local("file_io_codec_restored",1,nx);
    file.read_all(local,"field",{rank,0});
    for (int i = 0; i < nx; i++) {
      if (local(0,i) != 10*rank+(i%17)) failed = 1;
    }
    std::string codec;
    std::string compressor;
    file.readVariableAttribute(codec,"field","codec");
    file.readVariableAttribute(compressor,"field","codec_compressor");
    if (codec != "blosc2" || compressor != "lz4" || !file.var_exists("field/codec_block_directory")) failed = 1;
    file.close();
  }

  int global_failed = 0;
  MPI_Allreduce(&failed,&global_failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
  return global_failed;
}

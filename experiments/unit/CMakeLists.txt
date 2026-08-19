cmake_minimum_required(VERSION 3.18)
project(portUrb)

set(CMAKE_CXX_STANDARD 20)

enable_testing()

option(PORTURB_ENABLE_FILE_IO_TESTS
       "Build and register standalone file I/O tests and benchmarks"
       OFF)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

add_subdirectory(${PORTURB_HOME}/model model)

add_library(custom_modules INTERFACE)
target_include_directories(custom_modules INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)

set(mylibs modules custom_modules model "${PORTURB_LINK_FLAGS}")

macro(process_target target_name)
  add_executable(${target_name} ${target_name}.cpp)
  target_compile_options(${target_name} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PORTURB_CXX_FLAGS}>)
  target_link_libraries(${target_name} ${mylibs})
  add_test(NAME ${target_name} COMMAND ${target_name})
endmacro()

process_target(abl_convective)
process_target(abl_neutral_fixed)
process_target(abl_neutral_variable)
process_target(abl_stable)
process_target(city)
process_target(supercell_kessler_fixed)
process_target(supercell_kessler_variable)
process_target(supercell_morr2mom)
# process_target(toy_bem)
process_target(turbine_neutral_ensemble)
process_target(turbine_simple)
process_target(anelastic_dycore)

if (PORTURB_ENABLE_FILE_IO_TESTS)
  if (NOT PORTURB_FILE_IO_BACKEND STREQUAL "adios2")
    message(FATAL_ERROR "PORTURB_ENABLE_FILE_IO_TESTS requires PORTURB_FILE_IO_BACKEND=adios2")
  endif()

  add_executable(file_io_compression_mpi EXCLUDE_FROM_ALL file_io_compression_mpi.cpp)
  target_compile_options(file_io_compression_mpi PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PORTURB_CXX_FLAGS}>)
  target_link_libraries(file_io_compression_mpi ${mylibs})
  add_test(NAME file_io_compression_mpi COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                                               ${MPIEXEC_PREFLAGS} $<TARGET_FILE:file_io_compression_mpi>
                                               ${MPIEXEC_POSTFLAGS})
  set_tests_properties(file_io_compression_mpi PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=0;GATOR_INITIAL_MB=64")

  # This writes repeated 512^3, three-component velocity data sets. Keep it as an explicit benchmark target rather than
  # adding long-running compression and durable-storage work to the default CTest suite.
  add_executable(file_io_turbulence_compression EXCLUDE_FROM_ALL file_io_turbulence_compression.cpp)
  target_compile_options(file_io_turbulence_compression PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PORTURB_CXX_FLAGS}>)
  target_link_libraries(file_io_turbulence_compression ${mylibs})
endif()

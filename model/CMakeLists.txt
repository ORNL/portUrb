
set(KOKKOS_HOME            ${CMAKE_CURRENT_SOURCE_DIR}/../external/kokkos              )
set(KOKKOS_HOME            ${CMAKE_CURRENT_SOURCE_DIR}/../external/kokkos  PARENT_SCOPE)
set(KOKKOS_BIN             ${CMAKE_CURRENT_BINARY_DIR}/kokkos                          )
set(KOKKOS_BIN             ${CMAKE_CURRENT_BINARY_DIR}/kokkos              PARENT_SCOPE)
set(YAKL_HOME              ${CMAKE_CURRENT_SOURCE_DIR}/../external/YAKL                )
set(YAKL_HOME              ${CMAKE_CURRENT_SOURCE_DIR}/../external/YAKL    PARENT_SCOPE)
set(YAKL_BIN               ${CMAKE_CURRENT_BINARY_DIR}/yakl                            )
set(YAKL_BIN               ${CMAKE_CURRENT_BINARY_DIR}/yakl                PARENT_SCOPE)
set(PNETCDF_HOME           ${CMAKE_CURRENT_SOURCE_DIR}/../external/PnetCDF             )
set(ADIOS2_HOME            ${CMAKE_CURRENT_SOURCE_DIR}/../external/ADIOS2              )
set(BLOSC2_HOME            ${CMAKE_CURRENT_SOURCE_DIR}/../external/c-blosc2             )
set(PORTURB_FILE_IO_BACKEND "adios2" CACHE STRING "File I/O backend (adios2 or pnetcdf)")

add_subdirectory(${KOKKOS_HOME}                                    ${KOKKOS_BIN}                       )
include_directories(${Kokkos_INCLUDE_DIRS_RET})
add_subdirectory(${YAKL_HOME}                                      ${YAKL_BIN}                         )

if (Kokkos_ENABLE_HIP)
  option(PORTURB_HIP_KERNEL_RESOURCE_REPORT
         "Generate a per-executable AMD GPU kernel resource report"
         ON)
  if (PORTURB_HIP_KERNEL_RESOURCE_REPORT)
    find_program(PORTURB_BASH bash REQUIRED)
    find_program(PORTURB_ROC_OBJ_LS roc-obj-ls REQUIRED)
    find_program(PORTURB_LLVM_NM llvm-nm
                 HINTS "${ROCM_PATH}/llvm/bin" "$ENV{ROCM_PATH}/llvm/bin" REQUIRED)
    find_program(PORTURB_LLVM_READOBJ llvm-readobj
                 HINTS "${ROCM_PATH}/llvm/bin" "$ENV{ROCM_PATH}/llvm/bin" REQUIRED)
    find_program(PORTURB_LLVM_SYMBOLIZER llvm-symbolizer
                 HINTS "${ROCM_PATH}/llvm/bin" "$ENV{ROCM_PATH}/llvm/bin" REQUIRED)

    set(PORTURB_HIP_RESOURCE_REPORT_SCRIPT
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/hip_kernel_resources.sh"
        CACHE INTERNAL "portUrb HIP kernel resource report script")

    function(porturb_add_hip_kernel_resource_reports)
      get_property(porturb_targets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY BUILDSYSTEM_TARGETS)
      foreach(porturb_target IN LISTS porturb_targets)
        get_target_property(porturb_target_type "${porturb_target}" TYPE)
        if (porturb_target_type STREQUAL "EXECUTABLE")
          target_compile_options("${porturb_target}" PRIVATE
                                 $<$<COMPILE_LANGUAGE:CXX>:-gline-tables-only>)
          add_custom_command(
            TARGET "${porturb_target}" POST_BUILD
            COMMAND "${PORTURB_BASH}" "${PORTURB_HIP_RESOURCE_REPORT_SCRIPT}"
                    --executable "$<TARGET_FILE:${porturb_target}>"
                    --output "$<TARGET_FILE_DIR:${porturb_target}>/${porturb_target}.hip_kernel_resources.txt"
                    --source-root "${PORTURB_HOME}"
                    --roc-obj-ls "${PORTURB_ROC_OBJ_LS}"
                    --llvm-nm "${PORTURB_LLVM_NM}"
                    --llvm-readobj "${PORTURB_LLVM_READOBJ}"
                    --llvm-symbolizer "${PORTURB_LLVM_SYMBOLIZER}"
            COMMENT "Writing ${porturb_target}.hip_kernel_resources.txt"
            VERBATIM)
        endif()
      endforeach()
    endfunction()

    cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
                   CALL porturb_add_hip_kernel_resource_reports)
  endif()
endif()

set(YAML_CPP_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../external/yaml-cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../external/ponni     ponni )
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../external/eigen     eigen )

if (PORTURB_FILE_IO_BACKEND STREQUAL "adios2")
  set(_porturb_build_shared_libs "${BUILD_SHARED_LIBS}")
  set(_porturb_build_testing     "${BUILD_TESTING}")
  set(_porturb_fortran_loaded    "${CMAKE_Fortran_COMPILER_LOADED}")
  if (DEFINED BUILD_SHARED_LIBS)
    set(_porturb_build_shared_libs_defined ON)
  else()
    set(_porturb_build_shared_libs_defined OFF)
  endif()
  if (DEFINED BUILD_TESTING)
    set(_porturb_build_testing_defined ON)
  else()
    set(_porturb_build_testing_defined OFF)
  endif()

  # Build one static Blosc2 library with its fast LZ4/LZ4HC and Zstd compressors.
  # ADIOS2 links it into its native "blosc" operator, so no runtime plugin is needed.
  set(BUILD_STATIC     ON  CACHE BOOL "" FORCE)
  set(BUILD_SHARED     OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTS      OFF CACHE BOOL "" FORCE)
  set(BUILD_FUZZERS    OFF CACHE BOOL "" FORCE)
  set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
  set(BUILD_PLUGINS    OFF CACHE BOOL "" FORCE)
  set(BUILD_LITE       OFF CACHE BOOL "" FORCE)
  set(DEACTIVATE_ZLIB  ON  CACHE BOOL "" FORCE)
  set(DEACTIVATE_ZSTD  OFF CACHE BOOL "" FORCE)
  set(BLOSC_INSTALL    OFF CACHE BOOL "" FORCE)
  add_subdirectory(${BLOSC2_HOME} ${CMAKE_CURRENT_BINARY_DIR}/c-blosc2 EXCLUDE_FROM_ALL)
  target_include_directories(blosc2_static PUBLIC ${BLOSC2_HOME}/include)
  set_target_properties(blosc2_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

  set(BUILD_SHARED_LIBS              ON  CACHE BOOL "" FORCE)
  set(BUILD_TESTING                  OFF CACHE BOOL "" FORCE)
  set(ADIOS2_BUILD_EXAMPLES          OFF CACHE BOOL "" FORCE)
  # FindBlosc2 exposes a non-alias adapter under the shared target name because
  # ADIOS2 cannot create an alias to c-blosc2's add_subdirectory alias target.
  set(ADIOS2_Blosc2_PREFER_SHARED    ON  CACHE BOOL "" FORCE)
  set(ADIOS2_INSTALL_GENERATE_CONFIG OFF CACHE BOOL "" FORCE)
  set(ADIOS2_LIBADIOS_MODE           ON  CACHE BOOL "" FORCE)
  set(ADIOS2_RUN_INSTALL_TEST        OFF CACHE BOOL "" FORCE)
  set(ADIOS2_USE_EXTERNAL_YAMLCPP    ON  CACHE STRING "" FORCE)
  set(ADIOS2_USE_Blosc2              ON  CACHE STRING "" FORCE)
  set(ADIOS2_USE_MPI                 ON  CACHE STRING "" FORCE)
  foreach(feature IN ITEMS AWSSDK BigWhoop BZip2 CUDA Caliper Campaign Catalyst CURL DAOS
                           DataMan DataSpaces Derived_Variable Fortran HDF5 HDF5_VOL IME Kokkos
                           KVCACHE LIBPRESSIO MGARD MHS OpenSSL PIP PNG PRODM Profiling Python
                           PythonFreeThreaded SST SZ SZ3 SealKeygen Sodium SysVShMem UCX XRootD ZFP ZeroMQ)
    set(ADIOS2_USE_${feature} OFF CACHE STRING "" FORCE)
  endforeach()
  # ADIOS2 2.12.1 otherwise enables its Fortran bindings whenever its parent
  # project has already enabled Fortran, even with ADIOS2_USE_Fortran=OFF.
  set(CMAKE_Fortran_COMPILER_LOADED OFF)
  # ADIOS2 2.12.1's Unix capability check does not consult ADIOS2_USE_SysVShMem.
  set(HAVE_shmget OFF CACHE INTERNAL "Disable unused SysV shared-memory support" FORCE)
  set(_porturb_cmake_module_path "${CMAKE_MODULE_PATH}")
  list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  add_subdirectory(${ADIOS2_HOME} ${CMAKE_CURRENT_BINARY_DIR}/adios2 EXCLUDE_FROM_ALL)
  set(CMAKE_MODULE_PATH "${_porturb_cmake_module_path}")
  set(CMAKE_Fortran_COMPILER_LOADED "${_porturb_fortran_loaded}")

  if (_porturb_build_shared_libs_defined)
    set(BUILD_SHARED_LIBS "${_porturb_build_shared_libs}" CACHE BOOL "" FORCE)
  else()
    unset(BUILD_SHARED_LIBS CACHE)
    unset(BUILD_SHARED_LIBS)
  endif()
  if (_porturb_build_testing_defined)
    set(BUILD_TESTING "${_porturb_build_testing}" CACHE BOOL "" FORCE)
  else()
    unset(BUILD_TESTING CACHE)
    unset(BUILD_TESTING)
  endif()
endif()
add_subdirectory(core                                              core                                )
add_subdirectory(modules                                           modules                             )

add_library(model INTERFACE)
target_link_libraries(model INTERFACE core modules yakl yaml-cpp ponni eigen)
if (PORTURB_FILE_IO_BACKEND STREQUAL "adios2")
  target_link_libraries(model INTERFACE adios2::cxx_mpi)
  target_compile_definitions(model INTERFACE PORTURB_HAS_ADIOS2)
elseif (PORTURB_FILE_IO_BACKEND STREQUAL "pnetcdf")
  target_compile_definitions(model INTERFACE PORTURB_HAS_PNETCDF)
else()
  message(FATAL_ERROR "PORTURB_FILE_IO_BACKEND must be either adios2 or pnetcdf")
endif()
target_include_directories(model INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

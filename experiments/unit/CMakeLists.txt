cmake_minimum_required(VERSION 3.5)
project(portUrb)

set(CMAKE_CXX_STANDARD 20)

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
endmacro()

process_target(abl_convective)
process_target(abl_neutral_fixed)
process_target(abl_neutral_variable)
process_target(abl_stable)
process_target(city)
process_target(supercell_kessler_fixed)
process_target(supercell_kessler_variable)
process_target(supercell_morr2mom)
process_target(toy_bem)
process_target(turbine_neutral_ensemble)
process_target(turbine_simple)


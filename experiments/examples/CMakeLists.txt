cmake_minimum_required(VERSION 3.0)
project(portUrb)

set(CMAKE_CXX_STANDARD 17)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

include_directories(${PORTURB_HOME}/model
                    ${PORTURB_HOME}/model/core
                    ${PORTURB_HOME}/model/modules
                    ${PORTURB_HOME}/model/modules/helpers
                    ${PORTURB_HOME}/model/modules/helpers/microphysics_p3
                    ${PORTURB_HOME}/external/yaml-cpp/include
                    ${PORTURB_HOME}/external/ponni/src
                    ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules
                    )

add_subdirectory(${PORTURB_HOME}/model model)

add_library(custom_modules INTERFACE)
target_include_directories(custom_modules INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)

set(mylibs modules custom_modules model core ponni yaml-cpp yakl eigen "${PORTURB_LINK_FLAGS}")

macro(process_target target_name)
  add_executable(${target_name} ${target_name}.cpp)
  target_compile_options(${target_name} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PORTURB_CXX_FLAGS}>)
  target_link_libraries(${target_name} ${mylibs})
endmacro()

process_target(abl_convective)
process_target(abl_neutral)
process_target(abl_stable)
process_target(awaken_simplest)
process_target(building)
process_target(city)
process_target(city_dispersion)
process_target(city_ensemble)
process_target(cubes_periodic)
process_target(nrel_5mw_convective)
process_target(shallow_convection)
process_target(sphere)
process_target(supercell)
process_target(turbine_neutral_ensemble)
process_target(turbine_neutral_ensemble2)
process_target(turbine_simplest)
process_target(windfarm_ensemble_10x3)


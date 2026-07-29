
set(MORR_SRC module_mp_morr_two_moment.F90
             module_model_constants.F90
             module_wrf_error.F90
             wrf_debug.F90
             module_mp_radar.F90)

add_library(microphysics_morr ${MORR_SRC})
target_include_directories(microphysics_morr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(microphysics_morr PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_compile_options(microphysics_morr PUBLIC $<$<COMPILE_LANGUAGE:Fortran>:${PORTURB_F90_FLAGS}>)



add_subdirectory(utils/microphysics_morr microphysics_morr)

add_library(modules STATIC
            turbine/Betti_simplified.cpp
            microphysics/Mp_morr_two_moment.cpp
            microphysics/Mp_morr_two_moment_initialize.cpp
            microphysics/Mp_morr_two_moment_warm_processes.cpp
            microphysics/Mp_morr_two_moment_cold_processes.cpp
            microphysics/Mp_morr_two_moment_tendencies.cpp
            microphysics/Mp_morr_two_moment_sedimentation.cpp
            microphysics/Mp_morr_two_moment_finalize.cpp
            utils/TriMesh.cpp
            column_nudging.cpp
            dynamics/dynamics_cell_centered.cpp
            dynamics/dynamics_cell_centered_tendencies.cpp
            dynamics/dynamics_cell_centered_boundaries.cpp
            dynamics/dynamics_cell_centered_init.cpp
            dynamics/dynamics_cell_centered_conversion.cpp
            dynamics/dynamics_edge_centered.cpp
            dynamics/dynamics_edge_centered_tendencies.cpp
            dynamics/dynamics_edge_centered_boundaries.cpp
            dynamics/dynamics_edge_centered_init.cpp
            dynamics/dynamics_edge_centered_conversion.cpp
            edge_sponge.cpp
            fluctuation_scaling.cpp
            geostrophic_wind_forcing.cpp
            les_closure.cpp
            microphysics/microphysics_kessler.cpp
            microphysics/microphysics_morr.cpp
            overwrite_interpolate.cpp
            precursor_sponge.cpp
            sponge_layer.cpp
            surface_flux.cpp
            time_averager.cpp
            turbine/turbine_actuator_disc.cpp
            turbine/turbine_actuator_line.cpp
            turbine/turbine_fitch.cpp
            uniform_pg_wind_forcing.cpp
            turbine/windmill_actuators_yaw.cpp)
target_compile_options(modules PRIVATE ${PORTURB_CXX_FLAGS})
target_include_directories(modules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(modules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/dynamics)
target_include_directories(modules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/microphysics)
target_include_directories(modules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/turbine)
target_include_directories(modules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils)
target_include_directories(modules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(modules PUBLIC core microphysics_morr yakl yaml-cpp ponni eigen)

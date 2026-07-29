
add_subdirectory(helpers/microphysics_morr microphysics_morr)

add_library(modules INTERFACE)
target_include_directories(modules INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(modules INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/helpers)
target_link_libraries(modules INTERFACE microphysics_morr)


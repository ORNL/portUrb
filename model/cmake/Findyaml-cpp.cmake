# ADIOS2's external yaml-cpp path uses find_package even when its parent has
# already provided the yaml-cpp target with add_subdirectory.  Report that
# existing target as the package so ADIOS2 reuses portUrb's single library.
if (NOT TARGET yaml-cpp::yaml-cpp)
  set(yaml-cpp_FOUND FALSE)
  return()
endif()

set(yaml-cpp_VERSION 0.9.0)
set(YAML_CPP_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../external/yaml-cpp/include")
set(YAML_CPP_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp")
set(yaml-cpp_FOUND TRUE)

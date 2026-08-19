# ADIOS2 uses find_package even when its parent has already provided Blosc2
# with add_subdirectory. Report that existing target as the package so ADIOS2
# links its native Blosc operator to portUrb's single static Blosc2 library.
if (NOT TARGET Blosc2::blosc2_static)
  set(Blosc2_FOUND FALSE)
  return()
endif()

# c-blosc2's in-tree namespace target is itself an alias, and ADIOS2 attempts
# to create another alias to the target it selects. CMake forbids aliasing an
# alias, so expose a real imported interface target that carries the static
# library for this superbuild.
if (NOT TARGET Blosc2::blosc2_shared)
  add_library(Blosc2::blosc2_shared INTERFACE IMPORTED GLOBAL)
  set_property(TARGET Blosc2::blosc2_shared PROPERTY INTERFACE_LINK_LIBRARIES blosc2_static)
endif()

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../../external/c-blosc2/include/blosc2.h"
     _porturb_blosc2_version_line REGEX "^#define BLOSC2_VERSION_STRING")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" Blosc2_VERSION "${_porturb_blosc2_version_line}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Blosc2 REQUIRED_VARS Blosc2_VERSION VERSION_VAR Blosc2_VERSION)

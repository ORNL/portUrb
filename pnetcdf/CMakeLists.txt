cmake_minimum_required(VERSION 3.5)
include(CheckTypeSize)
include(CheckFunctionExists)
include(CheckIncludeFile)
include(CheckSymbolExists)
include(TestBigEndian)
include(CheckSymbolExists)
find_program(M4 m4 REQUIRED)
project(PnetCDF LANGUAGES C CXX)

file(STRINGS "${PNETCDF_HOME}/RELEASE_NOTES" PNETCDF_VERSION_STRING REGEX "Version[ ]*[1-9]" LIMIT_COUNT 1)
string(REGEX MATCH "[^ ]+ +([^ ]+)" _ "${PNETCDF_VERSION_STRING}")
set(PNETCDF_VERSION "${CMAKE_MATCH_1}")
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ "${PNETCDF_VERSION}")
set(PNETCDF_VERSION_MAJOR "${CMAKE_MATCH_1}")
set(PNETCDF_VERSION_MINOR "${CMAKE_MATCH_2}")
set(PNETCDF_VERSION_SUB   "${CMAKE_MATCH_3}")
set(PNETCDF_VERSION_PRE "")
set(ENABLE_FORTRAN                  0)
set(ENABLE_CXX                      0)
set(ENABLE_ERANGE_FILL              0)
set(ENABLE_SUBFILING                0)
set(RELAX_COORD_BOUND               0)
set(PNETCDF_DEBUG                   0)
set(PNETCDF_PROFILING               0)
set(ENABLE_NULL_BYTE_HEADER_PADDING 0)
set(IN_PLACE_SWAP                   0)
set(ENABLE_BURST_BUFFER             0)
set(ENABLE_THREAD_SAFE              0)
set(ENABLE_NETCDF4                  0)
set(ENABLE_ADIOS                    0)
check_type_size("size_t"    SIZEOF_SIZE_T)
check_type_size("off_t"     SIZEOF_OFF_T)
check_type_size("char"      SIZEOF_CHAR)
check_type_size("short"     SIZEOF_SHORT)
check_type_size("int"       SIZEOF_INT)
check_type_size("long"      SIZEOF_LONG)
check_type_size("long long" SIZEOF_LONG_LONG)
check_type_size("float"     SIZEOF_FLOAT)
check_type_size("double"    SIZEOF_DOUBLE)
check_type_size("unsigned char"      SIZEOF_UCHAR)
check_type_size("unsigned short"     SIZEOF_USHORT)
check_type_size("unsigned int"       SIZEOF_UINT)
check_type_size("unsigned long long" SIZEOF_ULONGLONG)
check_type_size("uint64_t"           SIZEOF_UINT64)
set(CMAKE_EXTRA_INCLUDE_FILES "mpi.h")
check_type_size("MPI_Offset" SIZEOF_MPI_OFFSET)
unset(CMAKE_EXTRA_INCLUDE_FILES)
check_symbol_exists(MPI_Count "mpi.h" HAVE_MPI_LARGE_COUNT)
check_function_exists(access   HAVE_ACCESS)
check_function_exists(close    HAVE_CLOSE)
check_function_exists(open     HAVE_OPEN)
check_function_exists(lstat    HAVE_LSTAT)
check_function_exists(strerror HAVE_STRERROR)
check_function_exists(truncate HAVE_TRUNCATE)
check_function_exists(unlink   HAVE_UNLINK)
check_function_exists(strtok_r USE_STRTOK_R)
check_include_file(inttypes.h HAVE_INTTYPES_H)
check_include_file(search.h   HAVE_SEARCH_H)
check_include_file(stdint.h   HAVE_STDINT_H)
check_include_file(stdlib.h   HAVE_STDLIB_H)
test_big_endian(WORDS_BIGENDIAN)
test_big_endian(FLOAT_WORDS_BIGENDIAN)
check_symbol_exists(MPI_LB                        "mpi.h" HAVE_DECL_MPI_LB)
check_symbol_exists(MPI_UB                        "mpi.h" HAVE_DECL_MPI_UB)
check_symbol_exists(MPI_Bcast_c                   "mpi.h" HAVE_MPI_BCAST_C)
check_symbol_exists(MPI_COMBINER_HINDEXED_INTEGER "mpi.h" HAVE_MPI_COMBINER_HINDEXED_INTEGER)
check_symbol_exists(MPI_COMBINER_HVECTOR_INTEGER  "mpi.h" HAVE_MPI_COMBINER_HVECTOR_INTEGER)
check_symbol_exists(MPI_COMBINER_STRUCT_INTEGER   "mpi.h" HAVE_MPI_COMBINER_STRUCT_INTEGER)
check_symbol_exists(MPI_Get_count_c               "mpi.h" HAVE_MPI_GET_COUNT_C)
check_symbol_exists(MPI_Type_get_true_extent_c    "mpi.h" HAVE_MPI_TYPE_GET_TRUE_EXTENT_C)
check_symbol_exists(MPI_Type_get_true_extent_x    "mpi.h" HAVE_MPI_TYPE_GET_TRUE_EXTENT_X)
check_symbol_exists(MPI_Type_size_c               "mpi.h" HAVE_MPI_TYPE_SIZE_C)
check_symbol_exists(MPI_Type_size_x               "mpi.h" HAVE_MPI_TYPE_SIZE_X)

configure_file(${PNETCDF_HOME}/src/include/pnetcdf.h.in ${CMAKE_CURRENT_BINARY_DIR}/pnetcdf.h @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in  ${CMAKE_CURRENT_BINARY_DIR}/config.h @ONLY)
set(M4_FILES
    ${PNETCDF_HOME}/src/dispatchers/attr_getput.m4
    ${PNETCDF_HOME}/src/dispatchers/var_getput.m4
    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_getput.m4
    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_i_varn.m4
    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_varn.m4
    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_attr.m4
    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_i_getput.m4
    ${PNETCDF_HOME}/src/drivers/common/ncx.m4
    ${PNETCDF_HOME}/src/drivers/common/convert_swap.m4
    ${PNETCDF_HOME}/src/drivers/include/ncx_h.m4
)
foreach(M4_FILE ${M4_FILES})
    file(RELATIVE_PATH M4_REL ${PNETCDF_HOME} ${M4_FILE})
    get_filename_component(M4_REL_DIR ${M4_REL} DIRECTORY)
    get_filename_component(M4_NAME    ${M4_FILE} NAME_WE)
    if(M4_NAME STREQUAL "ncx_h")
        set(OUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/ncx.h)
    else()
        set(OUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${M4_NAME}.c)
    endif()
    execute_process(
        COMMAND ${M4} -DPNETCDF -I ${PNETCDF_HOME}/m4 ${M4_FILE}
        OUTPUT_FILE ${OUT_FILE}
    )
endforeach()
include_directories(${PNETCDF_HOME}/src/include
                    ${PNETCDF_HOME}/src/drivers/include
                    ${PNETCDF_HOME}/src/drivers/common
                    ${PNETCDF_HOME}/src/drivers/ncmpio
                    ${PNETCDF_HOME}/src/drivers/pncio
                    ${PNETCDF_HOME}/src/libs
                    ${CMAKE_CURRENT_BINARY_DIR}
                    )
add_definitions(-DHAVE_CONFIG_H)
file(GLOB PNETCDF_SRC_DISPATCHERS "${PNETCDF_HOME}/src/dispatchers/*.c")
file(GLOB PNETCDF_SRC_M4 "${CMAKE_CURRENT_BINARY_DIR}/*.c")
add_library(pnetcdf 
                    ${PNETCDF_HOME}/src/drivers/common/check_name.c
                    ${PNETCDF_HOME}/src/drivers/common/create_imaptype.c
                    ${PNETCDF_HOME}/src/drivers/common/dtype_decode.c
                    ${PNETCDF_HOME}/src/drivers/common/error_mpi2nc.c
                    ${PNETCDF_HOME}/src/drivers/common/error_posix2nc.c
                    ${PNETCDF_HOME}/src/drivers/common/hash_map.c
                    ${PNETCDF_HOME}/src/drivers/common/mem_alloc.c
                    ${PNETCDF_HOME}/src/drivers/common/pack_unpack.c
                    ${PNETCDF_HOME}/src/drivers/common/utf8proc.c
                    ${PNETCDF_HOME}/src/drivers/common/utils.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_bput.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_file_io.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_create.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_hash_func.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_driver.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_enddef.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_dim.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_open.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_file_misc.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_intra_node.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_sync.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_fill.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_header_get.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_filetype.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_header_put.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_var.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_wait.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_close.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_vard.c
                    ${PNETCDF_HOME}/src/drivers/ncmpio/ncmpio_util.c
                    ${PNETCDF_SRC_DISPATCHERS}
                    ${PNETCDF_SRC_M4}
                    ${PNETCDF_HOME}/src/libs/strdup.c
                    ${PNETCDF_HOME}/src/libs/strcasecmp.c)
target_include_directories(pnetcdf PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_compile_options(pnetcdf PRIVATE -w)

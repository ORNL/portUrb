!WRF:DRIVER_LAYER:UTIL
!

MODULE module_wrf_error
CONTAINS
  LOGICAL FUNCTION wrf_at_debug_level ( level )
    IMPLICIT NONE
    INTEGER , INTENT(IN) :: level
  END FUNCTION wrf_at_debug_level

! ------------------------------------------------------------------------------

  SUBROUTINE init_module_wrf_error(on_io_server)
    IMPLICIT NONE
    LOGICAL,OPTIONAL,INTENT(IN) :: on_io_server
  END SUBROUTINE init_module_wrf_error

END MODULE module_wrf_error

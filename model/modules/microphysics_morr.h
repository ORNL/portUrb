
#pragma once

#include "coupler.h"
#include "Mp_morr_two_moment.h"


namespace modules {

  // Implements the interface to the 2-moment Morrison microphysics scheme, which is in MP_morr_two_moment.h
  struct Microphysics_Morrison {
    // Declare Fortran-style YAKL arrays for use in microphysics calculations, which are indexed with the first
    //   index varying the fastest and have lower bounds that default to 1 but can be changed.
    typedef yakl::Array<double      ,1,yakl::memDevice,yakl::styleFortran> double1d_F;
    typedef yakl::Array<double      ,2,yakl::memDevice,yakl::styleFortran> double2d_F;
    typedef yakl::Array<double const,1,yakl::memDevice,yakl::styleFortran> doubleConst1d_F;
    typedef yakl::Array<double const,2,yakl::memDevice,yakl::styleFortran> doubleConst2d_F;
    typedef yakl::Array<double      ,1,yakl::memHost  ,yakl::styleFortran> doubleHost1d_F;
    typedef yakl::Array<double      ,2,yakl::memHost  ,yakl::styleFortran> doubleHost2d_F;
    typedef yakl::Array<double const,1,yakl::memHost  ,yakl::styleFortran> doubleHostConst1d_F;
    typedef yakl::Array<double const,2,yakl::memHost  ,yakl::styleFortran> doubleHostConst2d_F;
    typedef yakl::Array<int        ,1,yakl::memDevice,yakl::styleFortran> int1d_F;
    typedef yakl::Array<int        ,2,yakl::memDevice,yakl::styleFortran> int2d_F;
    typedef yakl::Array<int   const,1,yakl::memDevice,yakl::styleFortran> intConst1d_F;
    typedef yakl::Array<int   const,2,yakl::memDevice,yakl::styleFortran> intConst2d_F;
    typedef yakl::Array<int        ,1,yakl::memHost  ,yakl::styleFortran> intHost1d_F;
    typedef yakl::Array<int        ,2,yakl::memHost  ,yakl::styleFortran> intHost2d_F;
    typedef yakl::Array<int   const,1,yakl::memHost  ,yakl::styleFortran> intHostConst1d_F;
    typedef yakl::Array<int   const,2,yakl::memHost  ,yakl::styleFortran> intHostConst2d_F;
    typedef yakl::Array<bool       ,1,yakl::memDevice,yakl::styleFortran> bool1d_F;
    typedef yakl::Array<bool       ,2,yakl::memDevice,yakl::styleFortran> bool2d_F;
    typedef yakl::Array<bool  const,1,yakl::memDevice,yakl::styleFortran> boolConst1d_F;
    typedef yakl::Array<bool  const,2,yakl::memDevice,yakl::styleFortran> boolConst2d_F;
    typedef yakl::Array<bool       ,1,yakl::memHost  ,yakl::styleFortran> boolHost1d_F;
    typedef yakl::Array<bool       ,2,yakl::memHost  ,yakl::styleFortran> boolHost2d_F;
    typedef yakl::Array<bool  const,1,yakl::memHost  ,yakl::styleFortran> boolHostConst1d_F;
    typedef yakl::Array<bool  const,2,yakl::memHost  ,yakl::styleFortran> boolHostConst2d_F;
    // Doesn't actually have to be static or constexpr. Could be assigned in the constructor
    int static constexpr num_tracers = 10;

    // Create an instance of the Morrison microphysics class to use its methods
    // This actually implements the port of the 2-moment Morrison microphysics scheme to portable C++
    Mp_morr_two_moment  micro;


    // Returns the number of tracers used by this microphysics scheme
    KOKKOS_INLINE_FUNCTION static int get_num_tracers() { return num_tracers; }


    // Initializes the microphysics module within the coupler by registering tracers and persistent variables,
    //   initializing tracers to zero, and setting microphysics-related options in the coupler
    void init(core::Coupler &coupler) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      // hm added new option for hail
      // switch for hail/graupel
      // ihail = 0, dense precipitating ice is graupel
      // ihail = 1, dense precipitating ice is hail
      int ihail = coupler.get_option<int>("micro_morr_ihail",1);
      micro.init( ihail );  // Initialize the Morrison microphysics class with ihail option

      int nx   = coupler.get_nx(); // Number of local cells in the x-direction
      int ny   = coupler.get_ny(); // Number of local cells in the y-direction
      int nz   = coupler.get_nz(); // Number of vertical levels

      // Register tracers in the coupler
      //                 name              description   positive   adds mass    diffuse
      coupler.add_tracer("water_vapor"   , ""          , true     , true       , true);
      coupler.add_tracer("cloud_water"   , ""          , true     , true       , true);
      coupler.add_tracer("rain_water"    , ""          , true     , true       , true);
      coupler.add_tracer("cloud_ice"     , ""          , true     , true       , true);
      coupler.add_tracer("snow"          , ""          , true     , true       , true);
      coupler.add_tracer("graupel"       , ""          , true     , true       , true);
      coupler.add_tracer("cloud_ice_num" , ""          , true     , false      , true);
      coupler.add_tracer("snow_num"      , ""          , true     , false      , true);
      coupler.add_tracer("rain_num"      , ""          , true     , false      , true);
      coupler.add_tracer("graupel_num"   , ""          , true     , false      , true);

      auto &dm = coupler.get_data_manager_readwrite(); // Get the DataManager with read/write access
      dm.register_and_allocate<real>("micro_rainnc"   ,"accumulated precipitation (mm)"      ,{ny,nx},{"y","x"});
      dm.register_and_allocate<real>("micro_snownc"   ,"accumulated snow plus cloud ice (mm)",{ny,nx},{"y","x"});
      dm.register_and_allocate<real>("micro_graupelnc","accumulated graupel (mm)"            ,{ny,nx},{"y","x"});

      // Register surface precipitation variables as output / restart variables.
      coupler.register_output_variable<real>( "micro_rainnc"    , core::Coupler::DIMS_SURFACE );
      coupler.register_output_variable<real>( "micro_snownc"    , core::Coupler::DIMS_SURFACE );
      coupler.register_output_variable<real>( "micro_graupelnc" , core::Coupler::DIMS_SURFACE );

      // Initialize tracers and persistent variables to zero
      dm.get_collapsed<real>( "water_vapor"     ) = 0;
      dm.get_collapsed<real>( "cloud_water"     ) = 0;
      dm.get_collapsed<real>( "rain_water"      ) = 0;
      dm.get_collapsed<real>( "cloud_ice"       ) = 0;
      dm.get_collapsed<real>( "snow"            ) = 0;
      dm.get_collapsed<real>( "graupel"         ) = 0;
      dm.get_collapsed<real>( "cloud_ice_num"   ) = 0;
      dm.get_collapsed<real>( "snow_num"        ) = 0;
      dm.get_collapsed<real>( "rain_num"        ) = 0;
      dm.get_collapsed<real>( "graupel_num"     ) = 0;
      dm.get_collapsed<real>( "micro_rainnc"    ) = 0;
      dm.get_collapsed<real>( "micro_snownc"    ) = 0;
      dm.get_collapsed<real>( "micro_graupelnc" ) = 0;

      coupler.set_option<std::string>("micro","p3");  // let the coupler know which microphysics scheme is being used
      real R_d        = 287.;        // ideal gas constant for dry air
      real cp_d       = 7.*R_d/2.;   // specific heat at constant pressure for dry air
      real cv_d       = cp_d - R_d;  // specific heat at constant volume for dry air
      real gamma_d    = cp_d / cv_d; // ratio of specific heats for dry air
      real kappa_d    = R_d  / cp_d; // Poisson constant for dry air
      real R_v        = 461.6;       // ideal gas constant for water vapor
      real cp_v       = 4.*R_v;      // specific heat at constant pressure for water vapor
      real cv_v       = cp_v - R_v;  // specific heat at constant volume for water vapor
      real p0         = 1.e5;        // reference pressure in Pa
      real grav       = 9.81;        // gravitational acceleration in m/s^2
      // Set microphysics-related physical constants in the coupler
      coupler.set_option<real>("R_d"    ,R_d    );
      coupler.set_option<real>("cp_d"   ,cp_d   );
      coupler.set_option<real>("cv_d"   ,cv_d   );
      coupler.set_option<real>("gamma_d",gamma_d);
      coupler.set_option<real>("kappa_d",kappa_d);
      coupler.set_option<real>("R_v"    ,R_v    );
      coupler.set_option<real>("cp_v"   ,cp_v   );
      coupler.set_option<real>("cv_v"   ,cv_v   );
      coupler.set_option<real>("p0"     ,p0     );
      coupler.set_option<real>("grav"   ,grav   );
    }



    // Advances the microphysics scheme by one time step of length dt within the coupler
    // coupler : Reference to the coupler object
    // dt      : Time step length in seconds
    void time_step( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      // Get the dimensions sizes
      int nz   = coupler.get_nz(); // Number of vertical levels
      int ny   = coupler.get_ny(); // Number of local cells in the y-direction
      int nx   = coupler.get_nx(); // Number of local cells in the x-direction
      int ncol = ny*nx;            // Number of columns (horizontal cells)

      // Get tracers and persistent variables dimensioned as (nz,ny*nx)
      auto &dm = coupler.get_data_manager_readwrite(); // Get the DataManager with read/write access
      auto dm_rho_v     = dm.get_lev_col  <real>("water_vapor"    );
      auto dm_rho_c     = dm.get_lev_col  <real>("cloud_water"    );
      auto dm_rho_r     = dm.get_lev_col  <real>("rain_water"     );
      auto dm_rho_i     = dm.get_lev_col  <real>("cloud_ice"      );
      auto dm_rho_s     = dm.get_lev_col  <real>("snow"           );
      auto dm_rho_g     = dm.get_lev_col  <real>("graupel"        );
      auto dm_rho_in    = dm.get_lev_col  <real>("cloud_ice_num"  );
      auto dm_rho_sn    = dm.get_lev_col  <real>("snow_num"       );
      auto dm_rho_rn    = dm.get_lev_col  <real>("rain_num"       );
      auto dm_rho_gn    = dm.get_lev_col  <real>("graupel_num"    );
      auto dm_rho_dry   = dm.get_lev_col  <real>("density_dry"    );
      auto dm_temp      = dm.get_lev_col  <real>("temp"           );
      auto dm_rainnc    = dm.get_collapsed<real>("micro_rainnc"   );
      auto dm_snownc    = dm.get_collapsed<real>("micro_snownc"   );
      auto dm_graupelnc = dm.get_collapsed<real>("micro_graupelnc");
      auto dz = coupler.get_dz();  // Get the vertical grid spacing array (1-D array of length nz)

      // Allocate inputs and outputs for the 2-moment Morrison microphysics scheme as Fortran-style YAKL arrays
      // Note that the data itself does not need to be transposed; only the indexing convention is changed
      double dt_in = dt;
      double2d_F qv        ("qv        ",ncol,nz); // water vapor dry mixing ratio
      double2d_F qc        ("qc        ",ncol,nz); // cloud liquid water dry mixing ratio
      double2d_F qr        ("qr        ",ncol,nz); // rain water dry mixing ratio
      double2d_F qi        ("qi        ",ncol,nz); // cloud ice dry mixing ratio
      double2d_F qs        ("qs        ",ncol,nz); // snow dry mixing ratio
      double2d_F qg        ("qg        ",ncol,nz); // graupel dry mixing ratio
      double2d_F ni        ("ni        ",ncol,nz); // cloud ice number concentration dry mixing ratio
      double2d_F ns        ("ns        ",ncol,nz); // snow number concentration dry mixing ratio
      double2d_F nr        ("nr        ",ncol,nz); // rain number concentration dry mixing ratio
      double2d_F t         ("t         ",ncol,nz); // temperature
      double2d_F ng        ("ng        ",ncol,nz); // graupel number concentration dry mixing ratio
      double2d_F qlsink    ("qlsink    ",ncol,nz); // cloud liquid water sink due to autoconversion and accretion
      double2d_F preci     ("preci     ",ncol,nz); // in-cloud ice precipitation rate
      double2d_F precs     ("precs     ",ncol,nz); // snow precipitation rate
      double2d_F precg     ("precg     ",ncol,nz); // graupel precipitation rate
      double2d_F precr     ("precr     ",ncol,nz); // rain precipitation rate
      double2d_F p         ("p         ",ncol,nz); // pressure
      double2d_F qrcuten   ("qrcuten   ",ncol,nz); // rain water dry mixing ratio tendencies
      double2d_F qscuten   ("qscuten   ",ncol,nz); // snow dry mixing ratio tendencies
      double2d_F qicuten   ("qicuten   ",ncol,nz); // cloud ice dry mixing ratio tendencies
      double2d_F dz_arr    ("dz_arr"    ,ncol,nz); // vertical grid spacing array (3-D instead of 1-D for microphysics)
      double1d_F rainncv   ("rainncv   ",ncol   ); // surface rain number concentration tendency
      double1d_F sr        ("sr        ",ncol   ); // surface rain water mixing ratio tendency
      double1d_F snowncv   ("snowncv   ",ncol   ); // surface snow number concentration tendency
      double1d_F graupelncv("graupelncv",ncol   ); // surface graupel number concentration tendency
      double1d_F rainnc    ("rainnc    ",ncol   ); // surface rain number concentration
      double1d_F snownc    ("snownc    ",ncol   ); // surface snow number concentration
      double1d_F graupelnc ("graupelnc ",ncol   ); // surface graupel number concentration

      //////////////////////////////////////////////////////////////////////////////
      // Compute quantities needed for inputs to Morrison 2-mom
      //////////////////////////////////////////////////////////////////////////////
      real R_d  = coupler.get_option<real>("R_d" );
      real R_v  = coupler.get_option<real>("R_v" );
      real cp_d = coupler.get_option<real>("cp_d");
      real p0   = coupler.get_option<real>("p0"  );

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
        // Compute dry mixing ratios for water vapor, cloud liquid, rain water, cloud ice, snow, graupel
        qv     (i+1,k+1) = dm_rho_v (k,i)/dm_rho_dry(k,i);
        qc     (i+1,k+1) = dm_rho_c (k,i)/dm_rho_dry(k,i);
        qr     (i+1,k+1) = dm_rho_r (k,i)/dm_rho_dry(k,i);
        qi     (i+1,k+1) = dm_rho_i (k,i)/dm_rho_dry(k,i);
        qs     (i+1,k+1) = dm_rho_s (k,i)/dm_rho_dry(k,i);
        qg     (i+1,k+1) = dm_rho_g (k,i)/dm_rho_dry(k,i);
        // Compute dry mixing ratios for cloud ice, snow, rain, and graupel.
        // It might seem odd to divide by density twice, but the coupler stores number concentration as
        //   a mass-weighted quantity, and the Morrison scheme expects a dry mixing ratio.
        ni     (i+1,k+1) = dm_rho_in(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        ns     (i+1,k+1) = dm_rho_sn(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        nr     (i+1,k+1) = dm_rho_rn(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        ng     (i+1,k+1) = dm_rho_gn(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        p      (i+1,k+1) = R_d*dm_rho_dry(k,i)*dm_temp(k,i) + R_v*dm_rho_v(k,i)*dm_temp(k,i); // pressure
        t      (i+1,k+1) = dm_temp(k,i);  // temperature
        qrcuten(i+1,k+1) = 0;
        qscuten(i+1,k+1) = 0;
        qicuten(i+1,k+1) = 0;
        dz_arr (i+1,k+1) = dz(k);  // vertical grid spacing
        if (k == 0) {
          rainnc   (i+1) = dm_rainnc   (i);
          snownc   (i+1) = dm_snownc   (i);
          graupelnc(i+1) = dm_graupelnc(i);
        }
      });

      // Run the portable C++ version of the Morrison 2-moment microphysics scheme
      micro.run(t, qv, qc, qr, qi, qs, qg, ni, ns, nr,
                ng, p, dt_in, dz_arr, rainnc, rainncv, sr, snownc,
                snowncv, graupelnc, graupelncv, qrcuten, qscuten, qicuten, ncol,
                nz, qlsink, precr, preci, precs, precg);
      
      // Convert Morrison 2mom outputs into dynamics coupler state, tracer masses, and mass-weighted number concentrations
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
        dm_rho_v (k,i) = qv(i+1,k+1)*dm_rho_dry(k,i);
        dm_rho_c (k,i) = qc(i+1,k+1)*dm_rho_dry(k,i);
        dm_rho_r (k,i) = qr(i+1,k+1)*dm_rho_dry(k,i);
        dm_rho_i (k,i) = qi(i+1,k+1)*dm_rho_dry(k,i);
        dm_rho_s (k,i) = qs(i+1,k+1)*dm_rho_dry(k,i);
        dm_rho_g (k,i) = qg(i+1,k+1)*dm_rho_dry(k,i);
        dm_rho_in(k,i) = ni(i+1,k+1)*dm_rho_dry(k,i)*dm_rho_dry(k,i);
        dm_rho_sn(k,i) = ns(i+1,k+1)*dm_rho_dry(k,i)*dm_rho_dry(k,i);
        dm_rho_rn(k,i) = nr(i+1,k+1)*dm_rho_dry(k,i)*dm_rho_dry(k,i);
        dm_rho_gn(k,i) = ng(i+1,k+1)*dm_rho_dry(k,i)*dm_rho_dry(k,i);
        dm_temp  (k,i) = t (i+1,k+1);
        if (k == 0) {
          dm_rainnc   (i) = rainnc   (i+1);
          dm_snownc   (i) = snownc   (i+1);
          dm_graupelnc(i) = graupelnc(i+1);
        }
      });
    }


    std::string micro_name() const { return "p3"; }


  };

}



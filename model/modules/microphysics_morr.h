
#pragma once

#include "coupler.h"
#include "Mp_morr_two_moment.h"


extern "C"
void mp_morr_two_moment(int *itimestep,double *th, double *qv, double *qc, double *qr, double *qi,
                        double *qs,double *qg, double *ni, double *ns, double *nr,
                        double *ng, double *rho, double *pii, double *p, double *dt_in, double *dz,
                        double *ht,
                        double *w, double *rainnc, double *rainncv, double *sr, double *snownc,
                        double *snowncv, double *graupelnc, double *graupelncv, double *refl_10cm,
                        bool *diagflag, int *do_radar_ref, double *qrcuten, double *qscuten, double *qicuten, 
                        bool *f_qndrop, double *qndrop,
                        int *ids, int *ide, int *jds, int *jde, int *kds, int *kde,
                        int *ims, int *ime, int *jms, int *jme, int *kms, int *kme,
                        int *its, int *ite, int *jts, int *jte, int *kts, int *kte, bool *wetscav_on, double *rainprod, double *evapprod,
                        double *qlsink, double *precr, double *preci, double *precs,
                        double *precg);

extern "C"
void morr_two_moment_init(int *morr_rimed_ice);


namespace modules {

  struct Microphysics_Morrison {
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
    #ifdef MICRO_MORR_FORTRAN
      int trace_size;
      std::vector<double> errs_qv       ;
      std::vector<double> errs_qc       ;
      std::vector<double> errs_qr       ;
      std::vector<double> errs_qi       ;
      std::vector<double> errs_qs       ;
      std::vector<double> errs_qg       ;
      std::vector<double> errs_ni       ;
      std::vector<double> errs_ns       ;
      std::vector<double> errs_nr       ;
      std::vector<double> errs_ng       ;
      std::vector<double> errs_t        ;
      std::vector<double> errs_rainnc   ;
      std::vector<double> errs_snownc   ;
      std::vector<double> errs_graupelnc;
    #endif
    Mp_morr_two_moment  micro;


    KOKKOS_INLINE_FUNCTION static int get_num_tracers() { return num_tracers; }


    // Can do whatever you want, but mainly for registering tracers and allocating data
    void init(core::Coupler &coupler) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      // hm added new option for hail
      // switch for hail/graupel
      // ihail = 0, dense precipitating ice is graupel
      // ihail = 1, dense precipitating ice is hail
      int ihail = coupler.get_option<int>("micro_morr_ihail",1);
      micro.init( ihail );

      #ifdef MICRO_MORR_FORTRAN
        trace_size = 0;
        morr_two_moment_init(&ihail);
      #endif

      int nx   = coupler.get_nx();
      int ny   = coupler.get_ny();
      int nz   = coupler.get_nz();

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

      auto &dm = coupler.get_data_manager_readwrite();
      dm.register_and_allocate<real>("micro_rainnc"   ,"accumulated precipitation (mm)"      ,{ny,nx},{"y","x"});
      dm.register_and_allocate<real>("micro_snownc"   ,"accumulated snow plus cloud ice (mm)",{ny,nx},{"y","x"});
      dm.register_and_allocate<real>("micro_graupelnc","accumulated graupel (mm)"            ,{ny,nx},{"y","x"});

      coupler.register_output_variable<real>( "micro_rainnc"    , core::Coupler::DIMS_SURFACE );
      coupler.register_output_variable<real>( "micro_snownc"    , core::Coupler::DIMS_SURFACE );
      coupler.register_output_variable<real>( "micro_graupelnc" , core::Coupler::DIMS_SURFACE );

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

      coupler.set_option<std::string>("micro","p3");
      real R_d        = 287.;
      real cp_d       = 7.*R_d/2.;
      real cv_d       = cp_d - R_d;
      real gamma_d    = cp_d / cv_d;
      real kappa_d    = R_d  / cp_d;
      real R_v        = 461.6;
      real cp_v       = 4.*R_v;
      real cv_v       = cp_v - R_v;
      real p0         = 1.e5;
      real grav       = 9.81;
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

      #ifdef MICRO_MORR_FORTRAN
        coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
          if (trace_size > 0) {
            nc.redef();
            nc.create_dim( "num_time_steps_phys" , trace_size );
            nc.create_var<real>( "micro_morr_errs_qv"        , {"num_time_steps_phys"} );
            nc.create_var<real>( "micro_morr_errs_qc"        , {"num_time_steps_phys"} );
            nc.create_var<real>( "micro_morr_errs_qr"        , {"num_time_steps_phys"} );
            nc.create_var<real>( "micro_morr_errs_qi"        , {"num_time_steps_phys"} );
            nc.create_var<real>( "micro_morr_errs_qs"        , {"num_time_steps_phys"} );
            nc.create_var<real>( "micro_morr_errs_qg"        , {"num_time_steps_phys"} );
            nc.create_var<real>( "micro_morr_errs_ni"        , {"num_time_steps_phys"} );
            nc.create_var<real>( "micro_morr_errs_ns"        , {"num_time_steps_phys"} );
            nc.create_var<real>( "micro_morr_errs_nr"        , {"num_time_steps_phys"} );
            nc.create_var<real>( "micro_morr_errs_ng"        , {"num_time_steps_phys"} );
            nc.create_var<real>( "micro_morr_errs_t"         , {"num_time_steps_phys"} );
            nc.create_var<real>( "micro_morr_errs_rainnc"    , {"num_time_steps_phys"} );
            nc.create_var<real>( "micro_morr_errs_snownc"    , {"num_time_steps_phys"} );
            nc.create_var<real>( "micro_morr_errs_graupelnc" , {"num_time_steps_phys"} );
            nc.enddef();
            nc.begin_indep_data();
            if (coupler.is_mainproc()) {
              realHost1d arr_qv       ("arr_qv       " , trace_size );
              realHost1d arr_qc       ("arr_qc       " , trace_size );
              realHost1d arr_qr       ("arr_qr       " , trace_size );
              realHost1d arr_qi       ("arr_qi       " , trace_size );
              realHost1d arr_qs       ("arr_qs       " , trace_size );
              realHost1d arr_qg       ("arr_qg       " , trace_size );
              realHost1d arr_ni       ("arr_ni       " , trace_size );
              realHost1d arr_ns       ("arr_ns       " , trace_size );
              realHost1d arr_nr       ("arr_nr       " , trace_size );
              realHost1d arr_ng       ("arr_ng       " , trace_size );
              realHost1d arr_t        ("arr_t        " , trace_size );
              realHost1d arr_rainnc   ("arr_rainnc   " , trace_size );
              realHost1d arr_snownc   ("arr_snownc   " , trace_size );
              realHost1d arr_graupelnc("arr_graupelnc" , trace_size );
              for (int i=0; i < trace_size; i++) { arr_qv       (i) = errs_qv       .at(i); }
              for (int i=0; i < trace_size; i++) { arr_qc       (i) = errs_qc       .at(i); }
              for (int i=0; i < trace_size; i++) { arr_qr       (i) = errs_qr       .at(i); }
              for (int i=0; i < trace_size; i++) { arr_qi       (i) = errs_qi       .at(i); }
              for (int i=0; i < trace_size; i++) { arr_qs       (i) = errs_qs       .at(i); }
              for (int i=0; i < trace_size; i++) { arr_qg       (i) = errs_qg       .at(i); }
              for (int i=0; i < trace_size; i++) { arr_ni       (i) = errs_ni       .at(i); }
              for (int i=0; i < trace_size; i++) { arr_ns       (i) = errs_ns       .at(i); }
              for (int i=0; i < trace_size; i++) { arr_nr       (i) = errs_nr       .at(i); }
              for (int i=0; i < trace_size; i++) { arr_ng       (i) = errs_ng       .at(i); }
              for (int i=0; i < trace_size; i++) { arr_t        (i) = errs_t        .at(i); }
              for (int i=0; i < trace_size; i++) { arr_rainnc   (i) = errs_rainnc   .at(i); }
              for (int i=0; i < trace_size; i++) { arr_snownc   (i) = errs_snownc   .at(i); }
              for (int i=0; i < trace_size; i++) { arr_graupelnc(i) = errs_graupelnc.at(i); }
              nc.write( arr_qv        , "micro_morr_errs_qv"        );
              nc.write( arr_qc        , "micro_morr_errs_qc"        );
              nc.write( arr_qr        , "micro_morr_errs_qr"        );
              nc.write( arr_qi        , "micro_morr_errs_qi"        );
              nc.write( arr_qs        , "micro_morr_errs_qs"        );
              nc.write( arr_qg        , "micro_morr_errs_qg"        );
              nc.write( arr_ni        , "micro_morr_errs_ni"        );
              nc.write( arr_ns        , "micro_morr_errs_ns"        );
              nc.write( arr_nr        , "micro_morr_errs_nr"        );
              nc.write( arr_ng        , "micro_morr_errs_ng"        );
              nc.write( arr_t         , "micro_morr_errs_t"         );
              nc.write( arr_rainnc    , "micro_morr_errs_rainnc"    );
              nc.write( arr_snownc    , "micro_morr_errs_snownc"    );
              nc.write( arr_graupelnc , "micro_morr_errs_graupelnc" );
              errs_qv       .clear();
              errs_qc       .clear();
              errs_qr       .clear();
              errs_qi       .clear();
              errs_qs       .clear();
              errs_qg       .clear();
              errs_ni       .clear();
              errs_ns       .clear();
              errs_nr       .clear();
              errs_ng       .clear();
              errs_t        .clear();
              errs_rainnc   .clear();
              errs_snownc   .clear();
              errs_graupelnc.clear();
              trace_size = 0;
            }
            nc.end_indep_data();
          }
          trace_size = 0;
        });
      #endif
    }



    void time_step( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      // Get the dimensions sizes
      int nz   = coupler.get_nz();
      int ny   = coupler.get_ny();
      int nx   = coupler.get_nx();
      int ncol = ny*nx;

      // Get tracers and persistent variables dimensioned as (nz,ny*nx)
      auto &dm = coupler.get_data_manager_readwrite();
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
      real dz = coupler.get_dz();

      // Allocates inputs and outputs
      double dt_in = dt;
      double2d_F qv        ("qv        ",ncol,nz);
      double2d_F qc        ("qc        ",ncol,nz);
      double2d_F qr        ("qr        ",ncol,nz);
      double2d_F qi        ("qi        ",ncol,nz);
      double2d_F qs        ("qs        ",ncol,nz);
      double2d_F qg        ("qg        ",ncol,nz);
      double2d_F ni        ("ni        ",ncol,nz);
      double2d_F ns        ("ns        ",ncol,nz);
      double2d_F nr        ("nr        ",ncol,nz);
      double2d_F t         ("t         ",ncol,nz);
      double2d_F ng        ("ng        ",ncol,nz);
      double2d_F qlsink    ("qlsink    ",ncol,nz);
      double2d_F preci     ("preci     ",ncol,nz);
      double2d_F precs     ("precs     ",ncol,nz);
      double2d_F precg     ("precg     ",ncol,nz);
      double2d_F precr     ("precr     ",ncol,nz);
      double2d_F p         ("p         ",ncol,nz);
      double2d_F qrcuten   ("qrcuten   ",ncol,nz);
      double2d_F qscuten   ("qscuten   ",ncol,nz);
      double2d_F qicuten   ("qicuten   ",ncol,nz);
      double2d_F dz_arr    ("dz_arr"    ,ncol,nz);
      double1d_F rainncv   ("rainncv   ",ncol   );
      double1d_F sr        ("sr        ",ncol   );
      double1d_F snowncv   ("snowncv   ",ncol   );
      double1d_F graupelncv("graupelncv",ncol   );
      double1d_F rainnc    ("rainnc    ",ncol   );
      double1d_F snownc    ("snownc    ",ncol   );
      double1d_F graupelnc ("graupelnc ",ncol   );

      //////////////////////////////////////////////////////////////////////////////
      // Compute quantities needed for inputs to Morrison 2-mom
      //////////////////////////////////////////////////////////////////////////////
      real R_d  = coupler.get_option<real>("R_d" );
      real R_v  = coupler.get_option<real>("R_v" );
      real cp_d = coupler.get_option<real>("cp_d");
      real p0   = coupler.get_option<real>("p0"  );

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
        qv     (i+1,k+1) = dm_rho_v (k,i)/dm_rho_dry(k,i);
        qc     (i+1,k+1) = dm_rho_c (k,i)/dm_rho_dry(k,i);
        qr     (i+1,k+1) = dm_rho_r (k,i)/dm_rho_dry(k,i);
        qi     (i+1,k+1) = dm_rho_i (k,i)/dm_rho_dry(k,i);
        qs     (i+1,k+1) = dm_rho_s (k,i)/dm_rho_dry(k,i);
        qg     (i+1,k+1) = dm_rho_g (k,i)/dm_rho_dry(k,i);
        ni     (i+1,k+1) = dm_rho_in(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        ns     (i+1,k+1) = dm_rho_sn(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        nr     (i+1,k+1) = dm_rho_rn(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        ng     (i+1,k+1) = dm_rho_gn(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        p      (i+1,k+1) = R_d*dm_rho_dry(k,i)*dm_temp(k,i) + R_v*dm_rho_v(k,i)*dm_temp(k,i);
        t      (i+1,k+1) = dm_temp(k,i);
        qrcuten(i+1,k+1) = 0;
        qscuten(i+1,k+1) = 0;
        qicuten(i+1,k+1) = 0;
        dz_arr (i+1,k+1) = dz;
        if (k == 0) {
          rainnc   (i+1) = dm_rainnc   (i);
          snownc   (i+1) = dm_snownc   (i);
          graupelnc(i+1) = dm_graupelnc(i);
        }
      });

      #ifdef MICRO_MORR_FORTRAN
        double2d_F th       ("th"       ,ncol,nz);
        double2d_F pii      ("pii"      ,ncol,nz);
        double2d_F rho      ("rho"      ,ncol,nz);  // not used
        double1d_F ht       ("ht"       ,ncol   );  // not used
        double2d_F w        ("w"        ,ncol,nz);  // not used
        double2d_F refl_10cm("refl_10cm",ncol,nz);  // not used
        double2d_F rainprod ("rainprod" ,ncol,nz);  // not used
        double2d_F evapprod ("evapprod" ,ncol,nz);  // not used
        double2d_F qndrop   ("qndrop"   ,ncol,nz);  // not used
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          pii(i+1,k+1) = std::pow( p(i+1,k+1) / p0 , R_d/cp_d );
          th (i+1,k+1) = t(i+1,k+1) / pii(i+1,k+1);
        });
        auto host_th         = th        .createHostCopy();
        auto host_qv         = qv        .createHostCopy();
        auto host_qc         = qc        .createHostCopy();
        auto host_qr         = qr        .createHostCopy();
        auto host_qi         = qi        .createHostCopy();
        auto host_qs         = qs        .createHostCopy();
        auto host_qg         = qg        .createHostCopy();
        auto host_ni         = ni        .createHostCopy();
        auto host_ns         = ns        .createHostCopy();
        auto host_nr         = nr        .createHostCopy();
        auto host_ng         = ng        .createHostCopy();
        auto host_rho        = rho       .createHostCopy();
        auto host_pii        = pii       .createHostCopy();
        auto host_p          = p         .createHostCopy();
        auto host_dz         = dz_arr    .createHostCopy();
        auto host_ht         = ht        .createHostCopy();
        auto host_w          = w         .createHostCopy();
        auto host_rainnc     = rainnc    .createHostCopy();
        auto host_rainncv    = rainncv   .createHostCopy();
        auto host_sr         = sr        .createHostCopy();
        auto host_snownc     = snownc    .createHostCopy();
        auto host_snowncv    = snowncv   .createHostCopy();
        auto host_graupelnc  = graupelnc .createHostCopy();
        auto host_graupelncv = graupelncv.createHostCopy();
        auto host_refl_10cm  = refl_10cm .createHostCopy();
        auto host_qrcuten    = qrcuten   .createHostCopy();
        auto host_qscuten    = qscuten   .createHostCopy();
        auto host_qicuten    = qicuten   .createHostCopy();
        auto host_rainprod   = rainprod  .createHostCopy();
        auto host_evapprod   = evapprod  .createHostCopy();
        auto host_qlsink     = qlsink    .createHostCopy();
        auto host_precr      = precr     .createHostCopy();
        auto host_preci      = preci     .createHostCopy();
        auto host_precs      = precs     .createHostCopy();
        auto host_precg      = precg     .createHostCopy();
        auto host_qndrop     = qndrop    .createHostCopy();
        // Unused
        bool diagflag     = false;
        int  do_radar_ref = 0;
        bool f_qndrop     = false;
        bool wetscav_on   = false;
        int  itimestep    = 1;
        int ids = 1   , jds=1, kds = 1 , ims = 1   , jms = 1, kms = 1 , its = 1   , jts = 1, kts = 1 ;
        int ide = ncol, jde=1, kde = nz, ime = ncol, jme = 1, kme = nz, ite = ncol, jte = 1, kte = nz;

        mp_morr_two_moment(&itimestep,host_th.data(), host_qv.data(), host_qc.data(), host_qr.data(), host_qi.data(),
                           host_qs.data(),host_qg.data(), host_ni.data(), host_ns.data(), host_nr.data(),
                           host_ng.data(), host_rho.data(), host_pii.data(), host_p.data(), &dt_in, host_dz.data(),
                           host_ht.data(),
                           host_w.data(), host_rainnc.data(), host_rainncv.data(), host_sr.data(), host_snownc.data(),
                           host_snowncv.data(), host_graupelnc.data(), host_graupelncv.data(), host_refl_10cm.data(),
                           &diagflag, &do_radar_ref, host_qrcuten.data(), host_qscuten.data(), host_qicuten.data(), 
                           &f_qndrop, host_qndrop.data(),
                           &ids, &ide, &jds, &jde, &kds, &kde,
                           &ims, &ime, &jms, &jme, &kms, &kme,
                           &its, &ite, &jts, &jte, &kts, &kte, &wetscav_on, host_rainprod.data(), host_evapprod.data(),
                           host_qlsink.data(), host_precr.data(), host_preci.data(), host_precs.data(),
                           host_precg.data());

        auto for_qv        = host_qv       .createDeviceCopy();
        auto for_qc        = host_qc       .createDeviceCopy();
        auto for_qr        = host_qr       .createDeviceCopy();
        auto for_qi        = host_qi       .createDeviceCopy();
        auto for_qs        = host_qs       .createDeviceCopy();
        auto for_qg        = host_qg       .createDeviceCopy();
        auto for_ni        = host_ni       .createDeviceCopy();
        auto for_ns        = host_ns       .createDeviceCopy();
        auto for_nr        = host_nr       .createDeviceCopy();
        auto for_ng        = host_ng       .createDeviceCopy();
        auto for_th        = host_th       .createDeviceCopy();
        auto for_rainnc    = host_rainnc   .createDeviceCopy();
        auto for_snownc    = host_snownc   .createDeviceCopy();
        auto for_graupelnc = host_graupelnc.createDeviceCopy();

        double2d_F for_t("fort_t",ncol,nz);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          for_t(i+1,k+1) = for_th(i+1,k+1) * pii(i+1,k+1);
        });

      #endif

      micro.run(t, qv, qc, qr, qi, qs, qg, ni, ns, nr,
                ng, p, dt_in, dz_arr, rainnc, rainncv, sr, snownc,
                snowncv, graupelnc, graupelncv, qrcuten, qscuten, qicuten, ncol,
                nz, qlsink, precr, preci, precs, precg);
      
      #ifdef MICRO_MORR_FORTRAN
        using yakl::componentwise::operator-;
        using yakl::intrinsics::abs;
        using yakl::intrinsics::sum;
        auto nx_glob = coupler.get_nx_glob();
        auto ny_glob = coupler.get_ny_glob();
        auto diff_qv        = coupler.get_parallel_comm().all_reduce( sum(abs(for_qv       -qv       )) , MPI_SUM , "" );
        auto diff_qc        = coupler.get_parallel_comm().all_reduce( sum(abs(for_qc       -qc       )) , MPI_SUM , "" );
        auto diff_qr        = coupler.get_parallel_comm().all_reduce( sum(abs(for_qr       -qr       )) , MPI_SUM , "" );
        auto diff_qi        = coupler.get_parallel_comm().all_reduce( sum(abs(for_qi       -qi       )) , MPI_SUM , "" );
        auto diff_qs        = coupler.get_parallel_comm().all_reduce( sum(abs(for_qs       -qs       )) , MPI_SUM , "" );
        auto diff_qg        = coupler.get_parallel_comm().all_reduce( sum(abs(for_qg       -qg       )) , MPI_SUM , "" );
        auto diff_ni        = coupler.get_parallel_comm().all_reduce( sum(abs(for_ni       -ni       )) , MPI_SUM , "" );
        auto diff_ns        = coupler.get_parallel_comm().all_reduce( sum(abs(for_ns       -ns       )) , MPI_SUM , "" );
        auto diff_nr        = coupler.get_parallel_comm().all_reduce( sum(abs(for_nr       -nr       )) , MPI_SUM , "" );
        auto diff_ng        = coupler.get_parallel_comm().all_reduce( sum(abs(for_ng       -ng       )) , MPI_SUM , "" );
        auto diff_t         = coupler.get_parallel_comm().all_reduce( sum(abs(for_t        -t        )) , MPI_SUM , "" );
        auto diff_rainnc    = coupler.get_parallel_comm().all_reduce( sum(abs(for_rainnc   -rainnc   )) , MPI_SUM , "" );
        auto diff_snownc    = coupler.get_parallel_comm().all_reduce( sum(abs(for_snownc   -snownc   )) , MPI_SUM , "" );
        auto diff_graupelnc = coupler.get_parallel_comm().all_reduce( sum(abs(for_graupelnc-graupelnc)) , MPI_SUM , "" );
        auto sum_qv        = coupler.get_parallel_comm().all_reduce( sum(abs(for_qv       )) , MPI_SUM , "" );
        auto sum_qc        = coupler.get_parallel_comm().all_reduce( sum(abs(for_qc       )) , MPI_SUM , "" );
        auto sum_qr        = coupler.get_parallel_comm().all_reduce( sum(abs(for_qr       )) , MPI_SUM , "" );
        auto sum_qi        = coupler.get_parallel_comm().all_reduce( sum(abs(for_qi       )) , MPI_SUM , "" );
        auto sum_qs        = coupler.get_parallel_comm().all_reduce( sum(abs(for_qs       )) , MPI_SUM , "" );
        auto sum_qg        = coupler.get_parallel_comm().all_reduce( sum(abs(for_qg       )) , MPI_SUM , "" );
        auto sum_ni        = coupler.get_parallel_comm().all_reduce( sum(abs(for_ni       )) , MPI_SUM , "" );
        auto sum_ns        = coupler.get_parallel_comm().all_reduce( sum(abs(for_ns       )) , MPI_SUM , "" );
        auto sum_nr        = coupler.get_parallel_comm().all_reduce( sum(abs(for_nr       )) , MPI_SUM , "" );
        auto sum_ng        = coupler.get_parallel_comm().all_reduce( sum(abs(for_ng       )) , MPI_SUM , "" );
        auto sum_t         = coupler.get_parallel_comm().all_reduce( sum(abs(for_t        )) , MPI_SUM , "" );
        auto sum_rainnc    = coupler.get_parallel_comm().all_reduce( sum(abs(for_rainnc   )) , MPI_SUM , "" );
        auto sum_snownc    = coupler.get_parallel_comm().all_reduce( sum(abs(for_snownc   )) , MPI_SUM , "" );
        auto sum_graupelnc = coupler.get_parallel_comm().all_reduce( sum(abs(for_graupelnc)) , MPI_SUM , "" );
        errs_qv       .push_back( sum_qv       > 1.e-14 ? diff_qv       /sum_qv        : 0 );
        errs_qc       .push_back( sum_qc       > 1.e-14 ? diff_qc       /sum_qc        : 0 );
        errs_qr       .push_back( sum_qr       > 1.e-14 ? diff_qr       /sum_qr        : 0 );
        errs_qi       .push_back( sum_qi       > 1.e-14 ? diff_qi       /sum_qi        : 0 );
        errs_qs       .push_back( sum_qs       > 1.e-14 ? diff_qs       /sum_qs        : 0 );
        errs_qg       .push_back( sum_qg       > 1.e-14 ? diff_qg       /sum_qg        : 0 );
        errs_ni       .push_back( sum_ni       > 1.e-14 ? diff_ni       /sum_ni        : 0 );
        errs_ns       .push_back( sum_ns       > 1.e-14 ? diff_ns       /sum_ns        : 0 );
        errs_nr       .push_back( sum_nr       > 1.e-14 ? diff_nr       /sum_nr        : 0 );
        errs_ng       .push_back( sum_ng       > 1.e-14 ? diff_ng       /sum_ng        : 0 );
        errs_t        .push_back( sum_t        > 1.e-14 ? diff_t        /sum_t         : 0 );
        errs_rainnc   .push_back( sum_rainnc   > 1.e-14 ? diff_rainnc   /sum_rainnc    : 0 );
        errs_snownc   .push_back( sum_snownc   > 1.e-14 ? diff_snownc   /sum_snownc    : 0 );
        errs_graupelnc.push_back( sum_graupelnc> 1.e-14 ? diff_graupelnc/sum_graupelnc : 0 );
        trace_size++;
        Kokkos::fence();
        qv        = for_qv       ;
        qc        = for_qc       ;
        qr        = for_qr       ;
        qi        = for_qi       ;
        qs        = for_qs       ;
        qg        = for_qg       ;
        ni        = for_ni       ;
        ns        = for_ns       ;
        nr        = for_nr       ;
        ng        = for_ng       ;
        t         = for_t        ;
        rainnc    = for_rainnc   ;
        snownc    = for_snownc   ;
        graupelnc = for_graupelnc;
      #endif
      
      ///////////////////////////////////////////////////////////////////////////////
      // Convert Morrison 2mom outputs into dynamics coupler state and tracer masses
      ///////////////////////////////////////////////////////////////////////////////
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



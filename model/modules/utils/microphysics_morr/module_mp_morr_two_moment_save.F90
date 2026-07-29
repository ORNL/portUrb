
module module_mp_morr_two_moment
  implicit none
  real, parameter :: pi   = 3.1415926535897932384626434
  real, parameter :: xxx  = 0.9189385332046727417803297
  real, parameter :: r    = 287.
  real, parameter :: rv   = 461.6
  real, parameter :: g    = 9.81
  real, parameter :: cp   = 1004.5
  real, parameter :: ep_2 = 0.621750433
  integer :: iact
  integer :: inum
  real    :: ndcnst
  integer :: iliq
  integer :: inuc
  integer :: ibase
  integer :: isub      
  integer :: igraup
  integer :: ihail
  real    :: ai,ac,as,ar,ag    ! 'a' parameter in fallspeed-diam relationship
  real    :: bi,bc,bs,br,bg    ! 'b' parameter in fallspeed-diam relationship
  real    :: rhosu             ! standard air density at 850 mb
  real    :: rhow              ! density of liquid water
  real    :: rhoi              ! bulk density of cloud ice
  real    :: rhosn             ! bulk density of snow
  real    :: rhog              ! bulk density of graupel
  real    :: aimm              ! parameter in bigg immersion freezing
  real    :: bimm              ! parameter in bigg immersion freezing
  real    :: ecr               ! collection efficiency between droplets/rain and snow/rain
  real    :: dcs               ! threshold size for cloud ice autoconversion
  real    :: mi0               ! initial size of nucleated crystal
  real    :: mg0               ! mass of embryo graupel
  real    :: f1s               ! ventilation parameter for snow
  real    :: f2s               ! ventilation parameter for snow
  real    :: f1r               ! ventilation parameter for rain
  real    :: f2r               ! ventilation parameter for rain
  real    :: qsmall            ! smallest allowed hydrometeor mixing ratio
  real    :: ci,di,cs,ds,cg,dg ! size distribution parameters for cloud ice, snow, graupel
  real    :: eii               ! collection efficiency, ice-ice collisions
  real    :: eci               ! collection efficiency, ice-droplet collisions
  real    :: rin               ! radius of contact nuclei (m)
  real    :: cpw               ! specific heat of liquid water
  real    :: c1                ! 'c' in nccn = cs^k (cm-3)
  real    :: k1                ! 'k' in nccn = cs^k
  real    :: mw                ! molecular weight water (kg/mol)
  real    :: osm               ! osmotic coefficient
  real    :: vi                ! number of ion dissociated in solution
  real    :: epsm              ! aerosol soluble fraction
  real    :: rhoa              ! aerosol bulk density (kg/m3)
  real    :: map               ! molecular weight aerosol (kg/mol)
  real    :: ma                ! molecular weight of 'air' (kg/mol)
  real    :: rr                ! universal gas constant
  real    :: bact              ! activation parameter
  real    :: rm1               ! geometric mean radius, mode 1 (m)
  real    :: rm2               ! geometric mean radius, mode 2 (m)
  real    :: nanew1            ! total aerosol concentration, mode 1 (m^-3)
  real    :: nanew2            ! total aerosol concentration, mode 2 (m^-3)
  real    :: sig1              ! standard deviation of aerosol s.d., mode 1
  real    :: sig2              ! standard deviation of aerosol s.d., mode 2
  real    :: f11               ! correction factor for activation, mode 1
  real    :: f12               ! correction factor for activation, mode 1
  real    :: f21               ! correction factor for activation, mode 2
  real    :: f22               ! correction factor for activation, mode 2     
  real    :: mmult             ! mass of splintered ice particle
  real    :: lammaxi,lammini,lammaxr,lamminr,lammaxs,lammins,lammaxg,lamming
  real    :: cons1,cons2,cons3,cons4,cons5,cons6,cons7,cons8,cons9,cons10
  real    :: cons11,cons12,cons13,cons14,cons15,cons16,cons17,cons18,cons19,cons20
  real    :: cons21,cons22,cons23,cons24,cons25,cons26,cons27,cons28,cons29,cons30
  real    :: cons31,cons32,cons33,cons34,cons35,cons36,cons37,cons38,cons39,cons40
  real    :: cons41
  interface pow
    module procedure pow_rr
    module procedure pow_ri
  end interface

contains

  subroutine morr_two_moment_init(morr_rimed_ice) bind(c,name="morr_two_moment_init") ! ras  
    implicit none
    integer, intent(in):: morr_rimed_ice ! ras  
    integer n,i
    inum    = 1
    ndcnst  = 250.
    iact    = 2
    ibase   = 2
    isub    = 0      
    iliq    = 0
    inuc    = 0
    igraup  = 0
    if (morr_rimed_ice == 1) then
      ihail = 1
    else
      ihail = 0
    endif
    ai      = 700.
    ac      = 3.e7
    as      = 11.72
    ar      = 841.99667
    bi      = 1.
    bc      = 2.
    bs      = 0.41
    br      = 0.8
    if (ihail==0) then
      ag    = 19.3
      bg    = 0.37
    else
      ag    = 114.5 
      bg    = 0.5
    endif
    rhosu   = 85000./(287.15*273.15)
    rhow    = 997.
    rhoi    = 500.
    rhosn   = 100.
    if (ihail==0) then
      rhog  = 400.
    else
      rhog  = 900.
    endif
    aimm    = 0.66
    bimm    = 100.
    ecr     = 1.
    dcs     = 125.e-6
    mi0     = 4./3.*pi*rhoi*pow(10.e-6,3)
    mg0     = 1.6e-10
    f1s     = 0.86
    f2s     = 0.28
    f1r     = 0.78
    f2r     = 0.308
    qsmall  = 1.e-14
    eii     = 0.1
    eci     = 0.7
    cpw     = 4187.
    ci      = rhoi*pi/6.
    di      = 3.
    cs      = rhosn*pi/6.
    ds      = 3.
    cg      = rhog*pi/6.
    dg      = 3.
    rin     = 0.1e-6
    mmult   = 4./3.*pi*rhoi*pow(5.e-6,3)
    lammaxi = 1./1.e-6
    lammini = 1./(2.*dcs+100.e-6)
    lammaxr = 1./20.e-6
    lamminr = 1./2800.e-6
    lammaxs = 1./10.e-6
    lammins = 1./2000.e-6
    lammaxg = 1./20.e-6
    lamming = 1./2000.e-6
    k1      = 0.4
    c1      = 120. 
    mw      = 0.018
    osm     = 1.
    vi      = 3.
    epsm    = 0.7
    rhoa    = 1777.
    map     = 0.132
    ma      = 0.0284
    rr      = 8.3145
    bact    = vi*osm*epsm*mw*rhoa/(map*rhow)
    rm1     = 0.052e-6
    sig1    = 2.04
    nanew1  = 72.2e6
    f11     = 0.5*exp(2.5*pow(log(sig1),2))
    f21     = 1.+0.25*log(sig1)
    rm2     = 1.3e-6
    sig2    = 2.5
    nanew2  = 1.8e6
    f12     = 0.5*exp(2.5*pow(log(sig2),2))
    f22     = 1.+0.25*log(sig2)
    cons1   = gamma(1.+ds)*cs
    cons2   = gamma(1.+dg)*cg
    cons3   = gamma(4.+bs)/6.
    cons4   = gamma(4.+br)/6.
    cons5   = gamma(1.+bs)
    cons6   = gamma(1.+br)
    cons7   = gamma(4.+bg)/6.
    cons8   = gamma(1.+bg)
    cons9   = gamma(5./2.+br/2.)
    cons10  = gamma(5./2.+bs/2.)
    cons11  = gamma(5./2.+bg/2.)
    cons12  = gamma(1.+di)*ci
    cons13  = gamma(bs+3.)*pi/4.*eci
    cons14  = gamma(bg+3.)*pi/4.*eci
    cons15  = -1108.*eii*pow(pi,(1.-bs)/3.)*pow(rhosn,(-2.-bs)/3.)/(4.*720.)
    cons16  = gamma(bi+3.)*pi/4.*eci
    cons17  = 4.*2.*3.*rhosu*pi*eci*eci*gamma(2.*bs+2.)/(8.*(rhog-rhosn))
    cons18  = rhosn*rhosn
    cons19  = rhow*rhow
    cons20  = 20.*pi*pi*rhow*bimm
    cons21  = 4./(dcs*rhoi)
    cons22  = pi*rhoi*pow(dcs,3)/6.
    cons23  = pi/4.*eii*gamma(bs+3.)
    cons24  = pi/4.*ecr*gamma(br+3.)
    cons25  = pi*pi/24.*rhow*ecr*gamma(br+6.)
    cons26  = pi/6.*rhow
    cons27  = gamma(1.+bi)
    cons28  = gamma(4.+bi)/6.
    cons29  = 4./3.*pi*rhow*pow(25.e-6,3)
    cons30  = 4./3.*pi*rhow
    cons31  = pi*pi*ecr*rhosn
    cons32  = pi/2.*ecr
    cons33  = pi*pi*ecr*rhog
    cons34  = 5./2.+br/2.
    cons35  = 5./2.+bs/2.
    cons36  = 5./2.+bg/2.
    cons37  = 4.*pi*1.38e-23/(6.*pi*rin)
    cons38  = pi*pi/3.*rhow
    cons39  = pi*pi/36.*rhow*bimm
    cons40  = pi/6.*bimm
    cons41  = pi*pi*ecr*rhow
  end subroutine morr_two_moment_init



  subroutine mp_morr_two_moment(t, qv, qc, qr, qi, qs, qg, ni, ns, nr, ng, rho, p, dt_in, dz, rainnc, rainncv, &
                                sr, snownc, snowncv, graupelnc, graupelncv, qrcuten, qscuten, qicuten,    &
                                ncol, nz,       &
                                qlsink, precr, preci, precs, precg) bind(c,name="mp_morr_two_moment")
    use iso_c_binding, only: c_int, c_float
    implicit none
    integer(c_int  ), intent(in   ) :: ncol, nz
    real   (c_float), dimension(ncol,nz), intent(inout) :: qv, qc, qr, qi, qs, qg, ni, ns, nr, t, ng   
    real   (c_float), dimension(ncol,nz), intent(inout) :: qlsink, preci, precs, precg, precr
    real   (c_float), dimension(ncol,nz), intent(in   ) :: p, dz, rho
    real   (c_float), dimension(ncol   ), intent(inout) :: rainnc, rainncv, sr, snownc, snowncv, graupelnc, graupelncv
    real   (c_float), dimension(ncol,nz), intent(in   ) :: qrcuten, qscuten, qicuten
    real   (c_float), intent(in) :: dt_in
    real, dimension(ncol,nz) :: c2prec
    integer :: iinum, i, k
    real    :: dt
    dt    = dt_in   
    iinum = 1

    call morr_two_moment_micro(qc, qi, qs, qr ,ni, ns, nr, t, qv, p, dz, rainncv, sr, snowncv, graupelncv,      &
                               dt, ncol, nz, qg, ng, qrcuten, qscuten, qicuten, iinum, c2prec, preci, &
                               precs, precg, precr)
    do k = 1 , nz
      do i = 1 , ncol
        if (qc(i,k) > 1.e-10) then
           qlsink(i,k) = c2prec(i,k)/qc(i,k)
        else
           qlsink(i,k) = 0.0
        endif
        if (k == 1) then
          rainnc    (i) = rainnc(i)+rainncv(i)
          snownc    (i) = snownc(i)+snowncv(i)
          graupelnc (i) = graupelnc(i)+graupelncv(i)
          sr        (i) = sr(i)/(rainncv(i)+1.e-12)
        endif
      enddo
    enddo   

  end subroutine mp_morr_two_moment



  subroutine morr_two_moment_micro(qc3d, qi3d, qni3d, qr3d, ni3d, ns3d, nr3d, t3d, qv3d, pres, dzq, precrt, snowrt, &
                                   snowprt, grplprt, dt, ncol, nz, qg3d, ng3d, qrcu1d, qscu1d, qicu1d,    &
                                   iinum, c2prec, ised, ssed, gsed, rsed)
    implicit none
    integer, intent(in   ) :: ncol, nz
    integer, intent(in   ) :: iinum
    real   , intent(in   ) :: dt                            ! model time step (sec)
    real   , intent(  out), dimension(:) :: precrt            ! total precip per time step (mm)
    real   , intent(  out), dimension(:) :: snowrt            ! snow per time step (mm)
    real   , intent(  out), dimension(:) :: snowprt           ! total cloud ice plus snow per time step (mm)
    real   , intent(  out), dimension(:) :: grplprt           ! total graupel per time step (mm)
    real   , intent(inout), dimension(:,:) :: qc3d      ! cloud water mixing ratio (kg/kg)
    real   , intent(inout), dimension(:,:) :: qi3d      ! cloud ice mixing ratio (kg/kg)
    real   , intent(inout), dimension(:,:) :: qni3d     ! snow mixing ratio (kg/kg)
    real   , intent(inout), dimension(:,:) :: qr3d      ! rain mixing ratio (kg/kg)
    real   , intent(inout), dimension(:,:) :: ni3d      ! cloud ice number concentration (1/kg)
    real   , intent(inout), dimension(:,:) :: ns3d      ! snow number concentration (1/kg)
    real   , intent(inout), dimension(:,:) :: nr3d      ! rain number concentration (1/kg)
    real   , intent(inout), dimension(:,:) :: t3d       ! temperature (k)
    real   , intent(inout), dimension(:,:) :: qv3d      ! water vapor mixing ratio (kg/kg)
    real   , intent(in   ), dimension(:,:) :: pres      ! atmospheric pressure (pa)
    real   , intent(in   ), dimension(:,:) :: dzq       ! difference in height across level (m)
    real   , intent(inout), dimension(:,:) :: qg3d      ! graupel mix ratio (kg/kg)
    real   , intent(inout), dimension(:,:) :: ng3d      ! graupel number conc (1/kg)
    real   , intent(in   ), dimension(:,:) :: qrcu1d
    real   , intent(in   ), dimension(:,:) :: qscu1d
    real   , intent(in   ), dimension(:,:) :: qicu1d
    real   , intent(  out), dimension(:,:) :: c2prec
    real   , intent(  out), dimension(:,:) :: ised
    real   , intent(  out), dimension(:,:) :: ssed
    real   , intent(  out), dimension(:,:) :: gsed
    real   , intent(  out), dimension(:,:) :: rsed
    real   , dimension(ncol,nz) :: ng3dten   ! graupel numb conc tendency (1/kg/s)
    real   , dimension(ncol,nz) :: qg3dten   ! graupel mix ratio tendency (kg/kg/s)
    real   , dimension(ncol,nz) :: effc      ! droplet effective radius (micron)
    real   , dimension(ncol,nz) :: effi      ! cloud ice effective radius (micron)
    real   , dimension(ncol,nz) :: effs      ! snow effective radius (micron)
    real   , dimension(ncol,nz) :: effr      ! rain effective radius (micron)
    real   , dimension(ncol,nz) :: effg      ! graupel effective radius (micron)
    real   , dimension(ncol,nz) :: t3dten    ! temperature tendency (k/s)
    real   , dimension(ncol,nz) :: qv3dten   ! water vapor mixing ratio tendency (kg/kg/s)
    real   , dimension(ncol,nz) :: qc3dten   ! cloud water mixing ratio tendency (kg/kg/s)
    real   , dimension(ncol,nz) :: qi3dten   ! cloud ice mixing ratio tendency (kg/kg/s)
    real   , dimension(ncol,nz) :: qni3dten  ! snow mixing ratio tendency (kg/kg/s)
    real   , dimension(ncol,nz) :: qr3dten   ! rain mixing ratio tendency (kg/kg/s)
    real   , dimension(ncol,nz) :: ni3dten   ! cloud ice number concentration (1/kg/s)
    real   , dimension(ncol,nz) :: ns3dten   ! snow number concentration (1/kg/s)
    real   , dimension(ncol,nz) :: nr3dten   ! rain number concentration (1/kg/s)
    real   , dimension(ncol,nz) :: csed
    real   , dimension(ncol,nz) :: qgsten    ! graupel sed tend (kg/kg/s)
    real   , dimension(ncol,nz) :: qrsten    ! rain sed tend (kg/kg/s)
    real   , dimension(ncol,nz) :: qisten    ! cloud ice sed tend (kg/kg/s)
    real   , dimension(ncol,nz) :: qnisten   ! snow sed tend (kg/kg/s)
    real   , dimension(ncol,nz) :: qcsten    ! cloud wat sed tend (kg/kg/s)      
    real   , dimension(ncol,nz) :: nc3d
    real   , dimension(ncol,nz) :: nc3dten
    real   , dimension(ncol,nz) :: lamc      ! slope parameter for droplets (m-1)
    real   , dimension(ncol,nz) :: lami      ! slope parameter for cloud ice (m-1)
    real   , dimension(ncol,nz) :: lams      ! slope parameter for snow (m-1)
    real   , dimension(ncol,nz) :: lamr      ! slope parameter for rain (m-1)
    real   , dimension(ncol,nz) :: lamg      ! slope parameter for graupel (m-1)
    real   , dimension(ncol,nz) :: cdist1    ! psd parameter for droplets
    real   , dimension(ncol,nz) :: n0i       ! intercept parameter for cloud ice (kg-1 m-1)
    real   , dimension(ncol,nz) :: n0s       ! intercept parameter for snow (kg-1 m-1)
    real   , dimension(ncol,nz) :: n0rr      ! intercept parameter for rain (kg-1 m-1)
    real   , dimension(ncol,nz) :: n0g       ! intercept parameter for graupel (kg-1 m-1)
    real   , dimension(ncol,nz) :: pgam      ! spectral shape parameter for droplets
    real   , dimension(ncol,nz) :: nsubc     ! loss of nc during evap
    real   , dimension(ncol,nz) :: nsubi     ! loss of ni during sub.
    real   , dimension(ncol,nz) :: nsubs     ! loss of ns during sub.
    real   , dimension(ncol,nz) :: nsubr     ! loss of nr during evap
    real   , dimension(ncol,nz) :: prd       ! dep cloud ice
    real   , dimension(ncol,nz) :: pre       ! evap of rain
    real   , dimension(ncol,nz) :: prds      ! dep snow
    real   , dimension(ncol,nz) :: nnuccc    ! change n due to contact freez droplets
    real   , dimension(ncol,nz) :: mnuccc    ! change q due to contact freez droplets
    real   , dimension(ncol,nz) :: pra       ! accretion droplets by rain
    real   , dimension(ncol,nz) :: prc       ! autoconversion droplets
    real   , dimension(ncol,nz) :: pcc       ! cond/evap droplets
    real   , dimension(ncol,nz) :: nnuccd    ! change n freezing aerosol (prim ice nucleation)
    real   , dimension(ncol,nz) :: mnuccd    ! change q freezing aerosol (prim ice nucleation)
    real   , dimension(ncol,nz) :: mnuccr    ! change q due to contact freez rain
    real   , dimension(ncol,nz) :: nnuccr    ! change n due to contact freez rain
    real   , dimension(ncol,nz) :: npra      ! change in n due to droplet acc by rain
    real   , dimension(ncol,nz) :: nragg     ! self-collection/breakup of rain
    real   , dimension(ncol,nz) :: nsagg     ! self-collection of snow
    real   , dimension(ncol,nz) :: nprc      ! change nc autoconversion droplets
    real   , dimension(ncol,nz) :: nprc1     ! change nr autoconversion droplets
    real   , dimension(ncol,nz) :: prai      ! change q accretion cloud ice by snow
    real   , dimension(ncol,nz) :: prci      ! change q autoconversin cloud ice to snow
    real   , dimension(ncol,nz) :: psacws    ! change q droplet accretion by snow
    real   , dimension(ncol,nz) :: npsacws   ! change n droplet accretion by snow
    real   , dimension(ncol,nz) :: psacwi    ! change q droplet accretion by cloud ice
    real   , dimension(ncol,nz) :: npsacwi   ! change n droplet accretion by cloud ice
    real   , dimension(ncol,nz) :: nprci     ! change n autoconversion cloud ice by snow
    real   , dimension(ncol,nz) :: nprai     ! change n accretion cloud ice
    real   , dimension(ncol,nz) :: nmults    ! ice mult due to riming droplets by snow
    real   , dimension(ncol,nz) :: nmultr    ! ice mult due to riming rain by snow
    real   , dimension(ncol,nz) :: qmults    ! change q due to ice mult droplets/snow
    real   , dimension(ncol,nz) :: qmultr    ! change q due to ice rain/snow
    real   , dimension(ncol,nz) :: pracs     ! change q rain-snow collection
    real   , dimension(ncol,nz) :: npracs    ! change n rain-snow collection
    real   , dimension(ncol,nz) :: pccn      ! change q droplet activation
    real   , dimension(ncol,nz) :: psmlt     ! change q melting snow to rain
    real   , dimension(ncol,nz) :: evpms     ! chnage q melting snow evaporating
    real   , dimension(ncol,nz) :: nsmlts    ! change n melting snow
    real   , dimension(ncol,nz) :: nsmltr    ! change n melting snow to rain
    real   , dimension(ncol,nz) :: piacr     ! change qr, ice-rain collection
    real   , dimension(ncol,nz) :: niacr     ! change n, ice-rain collection
    real   , dimension(ncol,nz) :: praci     ! change qi, ice-rain collection
    real   , dimension(ncol,nz) :: piacrs    ! change qr, ice rain collision, added to snow
    real   , dimension(ncol,nz) :: niacrs    ! change n, ice rain collision, added to snow
    real   , dimension(ncol,nz) :: pracis    ! change qi, ice rain collision, added to snow
    real   , dimension(ncol,nz) :: eprd      ! sublimation cloud ice
    real   , dimension(ncol,nz) :: eprds     ! sublimation snow
    real   , dimension(ncol,nz) :: pracg     ! change in q collection rain by graupel
    real   , dimension(ncol,nz) :: psacwg    ! change in q collection droplets by graupel
    real   , dimension(ncol,nz) :: pgsacw    ! conversion q to graupel due to collection droplets by snow
    real   , dimension(ncol,nz) :: pgracs    ! conversion q to graupel due to collection rain by snow
    real   , dimension(ncol,nz) :: prdg      ! dep of graupel
    real   , dimension(ncol,nz) :: eprdg     ! sub of graupel
    real   , dimension(ncol,nz) :: evpmg     ! change q melting of graupel and evaporation
    real   , dimension(ncol,nz) :: pgmlt     ! change q melting of graupel
    real   , dimension(ncol,nz) :: npracg    ! change n collection rain by graupel
    real   , dimension(ncol,nz) :: npsacwg   ! change n collection droplets by graupel
    real   , dimension(ncol,nz) :: nscng     ! change n conversion to graupel due to collection droplets by snow
    real   , dimension(ncol,nz) :: ngracs    ! change n conversion to graupel due to collection rain by snow
    real   , dimension(ncol,nz) :: ngmltg    ! change n melting graupel
    real   , dimension(ncol,nz) :: ngmltr    ! change n melting graupel to rain
    real   , dimension(ncol,nz) :: nsubg     ! change n sub/dep of graupel
    real   , dimension(ncol,nz) :: psacr     ! conversion due to coll of snow by rain
    real   , dimension(ncol,nz) :: nmultg    ! ice mult due to acc droplets by graupel
    real   , dimension(ncol,nz) :: nmultrg   ! ice mult due to acc rain by graupel
    real   , dimension(ncol,nz) :: qmultg    ! change q due to ice mult droplets/graupel
    real   , dimension(ncol,nz) :: qmultrg   ! change q due to ice mult rain/graupel
    real   , dimension(ncol,nz) :: kap       ! thermal conductivity of air
    real   , dimension(ncol,nz) :: evs       ! saturation vapor pressure
    real   , dimension(ncol,nz) :: eis       ! ice saturation vapor pressure
    real   , dimension(ncol,nz) :: qvs       ! saturation mixing ratio
    real   , dimension(ncol,nz) :: qvi       ! ice saturation mixing ratio
    real   , dimension(ncol,nz) :: qvqvs     ! sautration ratio
    real   , dimension(ncol,nz) :: qvqvsi    ! ice saturaion ratio
    real   , dimension(ncol,nz) :: dv        ! diffusivity of water vapor in air
    real   , dimension(ncol,nz) :: xxls      ! latent heat of sublimation
    real   , dimension(ncol,nz) :: xxlv      ! latent heat of vaporization
    real   , dimension(ncol,nz) :: cpm       ! specific heat at const pressure for moist air
    real   , dimension(ncol,nz) :: mu        ! viscocity of air
    real   , dimension(ncol,nz) :: sc        ! schmidt number
    real   , dimension(ncol,nz) :: xlf       ! latent heat of freezing
    real   , dimension(ncol,nz) :: rho       ! air density
    real   , dimension(ncol,nz) :: ab        ! correction to condensation rate due to latent heating
    real   , dimension(ncol,nz) :: abi       ! correction to deposition rate due to latent heating
    real   , dimension(ncol,nz) :: dap       ! diffusivity of aerosol
    real   , dimension(ncol,nz) :: dumi
    real   , dimension(ncol,nz) :: dumr
    real   , dimension(ncol,nz) :: dumfni
    real   , dimension(ncol,nz) :: dumg
    real   , dimension(ncol,nz) :: dumfng
    real   , dimension(ncol,nz) :: fr
    real   , dimension(ncol,nz) :: fi
    real   , dimension(ncol,nz) :: fni
    real   , dimension(ncol,nz) :: fg
    real   , dimension(ncol,nz) :: fng
    real   , dimension(ncol,nz) :: faloutr
    real   , dimension(ncol,nz) :: falouti
    real   , dimension(ncol,nz) :: faloutni
    real   , dimension(ncol,nz) :: dumqs
    real   , dimension(ncol,nz) :: dumfns
    real   , dimension(ncol,nz) :: fs
    real   , dimension(ncol,nz) :: fns
    real   , dimension(ncol,nz) :: falouts
    real   , dimension(ncol,nz) :: faloutns
    real   , dimension(ncol,nz) :: faloutg
    real   , dimension(ncol,nz) :: faloutng
    real   , dimension(ncol,nz) :: dumc
    real   , dimension(ncol,nz) :: dumfnc
    real   , dimension(ncol,nz) :: fc
    real   , dimension(ncol,nz) :: faloutc
    real   , dimension(ncol,nz) :: faloutnc
    real   , dimension(ncol,nz) :: fnc
    real   , dimension(ncol,nz) :: dumfnr
    real   , dimension(ncol,nz) :: faloutnr
    real   , dimension(ncol,nz) :: fnr
    real   , dimension(ncol,nz) :: ain
    real   , dimension(ncol,nz) :: arn
    real   , dimension(ncol,nz) :: asn
    real   , dimension(ncol,nz) :: acn
    real   , dimension(ncol,nz) :: agn
    real   , dimension(ncol,nz) :: tqimelt        ! melting of cloud ice (tendency)
    real   , dimension(ncol   ) :: nacnt     ! number of contact in
    real   , dimension(ncol   ) :: fmult     ! temp.-dep. parameter for rime-splintering
    real   , dimension(ncol   ) :: uni
    real   , dimension(ncol   ) :: umi
    real   , dimension(ncol   ) :: umr
    real   , dimension(ncol   ) :: rgvm
    real   , dimension(ncol   ) :: faltndr
    real   , dimension(ncol   ) :: faltndi
    real   , dimension(ncol   ) :: faltndni
    real   , dimension(ncol   ) :: ums
    real   , dimension(ncol   ) :: uns
    real   , dimension(ncol   ) :: faltnds
    real   , dimension(ncol   ) :: faltndns
    real   , dimension(ncol   ) :: unr
    real   , dimension(ncol   ) :: faltndg
    real   , dimension(ncol   ) :: faltndng
    real   , dimension(ncol   ) :: unc
    real   , dimension(ncol   ) :: umc
    real   , dimension(ncol   ) :: ung
    real   , dimension(ncol   ) :: umg
    real   , dimension(ncol   ) :: faltndc
    real   , dimension(ncol   ) :: faltndnc
    real   , dimension(ncol   ) :: faltndnr
    real   , dimension(ncol   ) :: dum1
    real   , dimension(ncol   ) :: dumt
    real   , dimension(ncol   ) :: dumqv
    real   , dimension(ncol   ) :: dumqss
    real   , dimension(ncol   ) :: dums
    real   , dimension(ncol   ) :: dqsdt    ! change of sat. mix. rat. with temperature
    real   , dimension(ncol   ) :: dqsidt   ! change in ice sat. mixing rat. with t
    real   , dimension(ncol   ) :: epsi     ! 1/phase rel. time (see m2005), ice
    real   , dimension(ncol   ) :: epss     ! 1/phase rel. time (see m2005), snow
    real   , dimension(ncol   ) :: epsr     ! 1/phase rel. time (see m2005), rain
    real   , dimension(ncol   ) :: epsg     ! 1/phase rel. time (see m2005), graupel
    real   , dimension(ncol   ) :: kc2       ! total ice nucleation rate
    real   , dimension(ncol   ) :: dumqc
    real   , dimension(ncol   ) :: ratio
    real   , dimension(ncol   ) :: sum_dep
    real   , dimension(ncol   ) :: fudgef
    real   , dimension(ncol   ) :: dlams
    real   , dimension(ncol   ) :: dlamr
    real   , dimension(ncol   ) :: dlami
    real   , dimension(ncol   ) :: dlamc
    real   , dimension(ncol   ) :: dlamg
    real   , dimension(ncol   ) :: lammax
    real   , dimension(ncol   ) :: lammin
    integer, dimension(ncol   ) :: ltrue
    integer, dimension(ncol   ) :: nstep
    real                        :: dum
    integer                     :: i, k, n
    do i = 1 , ncol
      ltrue(i) = 0
      do k = 1 , nz
        nc3d    (i,k)   = 0.
        ng3dten (i,k) = 0.
        qg3dten (i,k) = 0.
        t3dten  (i,k) = 0.
        qv3dten (i,k) = 0.
        qc3dten (i,k) = 0.
        qi3dten (i,k) = 0.
        qni3dten(i,k) = 0.
        qr3dten (i,k) = 0.
        ni3dten (i,k) = 0.
        ns3dten (i,k) = 0.
        nr3dten (i,k) = 0.
        nc3dten(i,k) = 0.
        c2prec (i,k) = 0.
        csed   (i,k) = 0.
        ised   (i,k) = 0.
        ssed   (i,k) = 0.
        gsed   (i,k) = 0.
        rsed   (i,k) = 0.
        xxlv(i,k) = 3.1484e6-2370.*t3d(i,k)
        xxls(i,k) = 3.15e6-2370.*t3d(i,k)+0.3337e6
        cpm(i,k) = cp*(1.+0.887*qv3d(i,k))
        evs(i,k) = min(0.99*pres(i,k),polysvp(t3d(i,k),0))   ! pa
        eis(i,k) = min(0.99*pres(i,k),polysvp(t3d(i,k),1))   ! pa
        if (eis(i,k) > evs(i,k)) eis(i,k) = evs(i,k)
        qvs(i,k) = ep_2*evs(i,k)/(pres(i,k)-evs(i,k))
        qvi(i,k) = ep_2*eis(i,k)/(pres(i,k)-eis(i,k))
        qvqvs(i,k) = qv3d(i,k)/qvs(i,k)
        qvqvsi(i,k) = qv3d(i,k)/qvi(i,k)
        rho(i,k) = pres(i,k)/(r*t3d(i,k))
        if (qrcu1d(i,k) >= 1.e-10) then
          nr3d(i,k) = nr3d(i,k)+1.8e5*pow(qrcu1d(i,k)*dt/(pi*rhow*pow(rho(i,k),3)),0.25)
        endif
        if (qscu1d(i,k) >= 1.e-10) then
          ns3d(i,k) = ns3d(i,k)+3.e5*pow(qscu1d(i,k)*dt/(cons1*pow(rho(i,k),3)),1./(ds+1.))
        endif
        if (qicu1d(i,k) >= 1.e-10) then
          ni3d(i,k) = ni3d(i,k)+qicu1d(i,k)*dt/(ci*pow(80.e-6,di))
        endif
        if (qvqvs(i,k) < 0.9) then
          if (qr3d(i,k) < 1.e-8) then
             qv3d(i,k)=qv3d(i,k)+qr3d(i,k)
             t3d (i,k)=t3d(i,k)-qr3d(i,k)*xxlv(i,k)/cpm(i,k)
             qr3d(i,k)=0.
          endif
          if (qc3d(i,k) < 1.e-8) then
             qv3d(i,k)=qv3d(i,k)+qc3d(i,k)
             t3d (i,k)=t3d(i,k)-qc3d(i,k)*xxlv(i,k)/cpm(i,k)
             qc3d(i,k)=0.
          endif
        endif

        if (qvqvsi(i,k) < 0.9) then
          if (qi3d(i,k) < 1.e-8) then
             qv3d(i,k)=qv3d(i,k)+qi3d(i,k)
             t3d (i,k)=t3d(i,k)-qi3d(i,k)*xxls(i,k)/cpm(i,k)
             qi3d(i,k)=0.
          endif
          if (qni3d(i,k) < 1.e-8) then
             qv3d (i,k)=qv3d(i,k)+qni3d(i,k)
             t3d  (i,k)=t3d(i,k)-qni3d(i,k)*xxls(i,k)/cpm(i,k)
             qni3d(i,k)=0.
          endif
          if (qg3d(i,k) < 1.e-8) then
             qv3d(i,k)=qv3d(i,k)+qg3d(i,k)
             t3d (i,k)=t3d(i,k)-qg3d(i,k)*xxls(i,k)/cpm(i,k)
             qg3d(i,k)=0.
          endif
        endif
        xlf(i,k) = xxls(i,k)-xxlv(i,k)
        if (qc3d(i,k) < qsmall) then
          qc3d(i,k) = 0.
          nc3d(i,k) = 0.
          effc(i,k) = 0.
        endif
        if (qr3d(i,k) < qsmall) then
          qr3d(i,k) = 0.
          nr3d(i,k) = 0.
          effr(i,k) = 0.
        endif
        if (qi3d(i,k) < qsmall) then
          qi3d(i,k) = 0.
          ni3d(i,k) = 0.
          effi(i,k) = 0.
        endif
        if (qni3d(i,k) < qsmall) then
          qni3d(i,k) = 0.
          ns3d(i,k) = 0.
          effs(i,k) = 0.
        endif
        if (qg3d(i,k) < qsmall) then
          qg3d(i,k) = 0.
          ng3d(i,k) = 0.
          effg(i,k) = 0.
        endif
        qrsten (i,k) = 0.
        qisten (i,k) = 0.
        qnisten(i,k) = 0.
        qcsten (i,k) = 0.
        qgsten (i,k) = 0.
        mu (i,k) = 1.496e-6*pow(t3d(i,k),1.5)/(t3d(i,k)+120.)
        dum      = pow(rhosu/rho(i,k),0.54)
        ain(i,k) = pow(rhosu/rho(i,k),0.35)*ai
        arn(i,k) = dum*ar
        asn(i,k) = dum*as
        acn(i,k) = g*rhow/(18.*mu(i,k))
        agn(i,k) = dum*ag
        lami   (i,k) = 0.
        if (qc3d(i,k) < qsmall.and.qi3d(i,k) < qsmall.and.qni3d(i,k) < qsmall .and.qr3d(i,k) < qsmall.and.qg3d(i,k) < qsmall) then
          if (t3d(i,k) <  273.15.and.qvqvsi(i,k) < 0.999) cycle
          if (t3d(i,k) >= 273.15.and.qvqvs(i,k) < 0.999) cycle
        endif
        kap   (i,k) = 1.414e3*mu(i,k)
        dv    (i,k) = 8.794e-5*pow(t3d(i,k),1.81)/pres(i,k)
        sc    (i,k) = mu(i,k)/(rho(i,k)*dv(i,k))
        dum         = (rv*pow(t3d(i,k),2))
        dqsdt (i)   = xxlv(i,k)*qvs(i,k)/dum
        dqsidt(i)   = xxls(i,k)*qvi(i,k)/dum
        abi   (i,k) = 1.+dqsidt(i)*xxls(i,k)/cpm(i,k)
        ab    (i,k) = 1.+dqsdt(i)*xxlv(i,k)/cpm(i,k)
        if (t3d(i,k) >= 273.15) then
          if (iinum==1) then
            nc3d(i,k) = ndcnst*1.e6/rho(i,k)
          endif
          if (qni3d(i,k) < 1.e-6) then
            qr3d (i,k) = qr3d(i,k)+qni3d(i,k)
            nr3d (i,k) = nr3d(i,k)+ns3d (i,k)
            t3d  (i,k) = t3d (i,k)-qni3d(i,k)*xlf(i,k)/cpm(i,k)
            qni3d(i,k) = 0.
            ns3d (i,k) = 0.
          endif
          if (qg3d(i,k) < 1.e-6) then
            qr3d(i,k) = qr3d(i,k)+qg3d(i,k)
            nr3d(i,k) = nr3d(i,k)+ng3d(i,k)
            t3d (i,k) = t3d (i,k)-qg3d(i,k)*xlf(i,k)/cpm(i,k)
            qg3d(i,k) = 0.
            ng3d(i,k) = 0.
          endif
          if (.not. (qc3d(i,k) < qsmall.and.qni3d(i,k) < 1.e-8.and.qr3d(i,k) < qsmall.and.qg3d(i,k) < 1.e-8)) then
            ns3d(i,k) = max(0.,ns3d(i,k))
            nc3d(i,k) = max(0.,nc3d(i,k))
            nr3d(i,k) = max(0.,nr3d(i,k))
            ng3d(i,k) = max(0.,ng3d(i,k))
            if (qr3d(i,k) >= qsmall) then
              lamr(i,k) = pow(pi*rhow*nr3d(i,k)/qr3d(i,k),1./3.)
              n0rr(i,k) = nr3d(i,k)*lamr(i,k)
              if (lamr(i,k) < lamminr) then
                lamr(i,k) = lamminr
                n0rr(i,k) = pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow)
                nr3d(i,k) = n0rr(i,k)/lamr(i,k)
              else if (lamr(i,k) > lammaxr) then
                lamr(i,k) = lammaxr
                n0rr(i,k) = pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow)
                nr3d(i,k) = n0rr(i,k)/lamr(i,k)
              endif
            endif
            if (qc3d(i,k) >= qsmall) then
              dum     =  pres(i,k)/(287.15*t3d(i,k))
              pgam(i,k) = 0.0005714*(nc3d(i,k)/1.e6*dum)+0.2714
              pgam(i,k) = 1./(pow(pgam(i,k),2))-1.
              pgam(i,k) = max(pgam(i,k),2.)
              pgam(i,k) = min(pgam(i,k),10.)
              lamc(i,k) = pow(cons26*nc3d(i,k)*gamma(pgam(i,k)+4.)/(qc3d(i,k)*gamma(pgam(i,k)+1.)),1./3.)
              lammin(i)  = (pgam(i,k)+1.)/60.e-6
              lammax(i)  = (pgam(i,k)+1.)/1.e-6
              if (lamc(i,k) < lammin(i)) then
                lamc(i,k) = lammin(i)
                nc3d(i,k) = exp(3.*log(lamc(i,k))+log(qc3d(i,k))+log(gamma(pgam(i,k)+1.))-log(gamma(pgam(i,k)+4.)))/cons26
              else if (lamc(i,k) > lammax(i)) then
                lamc(i,k) = lammax(i)
                nc3d(i,k) = exp(3.*log(lamc(i,k))+log(qc3d(i,k))+log(gamma(pgam(i,k)+1.))-log(gamma(pgam(i,k)+4.)))/cons26
              endif
            endif
            if (qni3d(i,k) >= qsmall) then
              lams(i,k) = pow(cons1*ns3d(i,k)/qni3d(i,k),1./ds)
              n0s(i,k) = ns3d(i,k)*lams(i,k)
              if (lams(i,k) < lammins) then
                lams(i,k) = lammins
                n0s(i,k) = pow(lams(i,k),4)*qni3d(i,k)/cons1
                ns3d(i,k) = n0s(i,k)/lams(i,k)
              else if (lams(i,k) > lammaxs) then
                lams(i,k) = lammaxs
                n0s(i,k) = pow(lams(i,k),4)*qni3d(i,k)/cons1
                ns3d(i,k) = n0s(i,k)/lams(i,k)
              endif
            endif
            if (qg3d(i,k) >= qsmall) then
              lamg(i,k) = pow(cons2*ng3d(i,k)/qg3d(i,k),1./dg)
              n0g(i,k) = ng3d(i,k)*lamg(i,k)
              if (lamg(i,k) < lamming) then
                lamg(i,k) = lamming
                n0g(i,k) = pow(lamg(i,k),4)*qg3d(i,k)/cons2
                ng3d(i,k) = n0g(i,k)/lamg(i,k)
              else if (lamg(i,k) > lammaxg) then
                lamg(i,k) = lammaxg
                n0g(i,k) = pow(lamg(i,k),4)*qg3d(i,k)/cons2
                ng3d(i,k) = n0g(i,k)/lamg(i,k)
              endif
            endif
            prc(i,k) = 0.
            nprc(i,k) = 0.
            nprc1(i,k) = 0.
            pra(i,k) = 0.
            npra(i,k) = 0.
            nragg(i,k) = 0.
            nsmlts(i,k) = 0.
            nsmltr(i,k) = 0.
            evpms(i,k) = 0.
            pcc(i,k) = 0.
            pre(i,k) = 0.
            nsubc(i,k) = 0.
            nsubr(i,k) = 0.
            pracg(i,k) = 0.
            npracg(i,k) = 0.
            psmlt(i,k) = 0.
            pgmlt(i,k) = 0.
            evpmg(i,k) = 0.
            pracs(i,k) = 0.
            npracs(i,k) = 0.
            ngmltg(i,k) = 0.
            ngmltr(i,k) = 0.
            if (qc3d(i,k) >= 1.e-6) then
              prc(i,k)=1350.*pow(qc3d(i,k),2.47)*pow(nc3d(i,k)/1.e6*rho(i,k),-1.79)
              nprc1(i,k) = prc(i,k)/cons29
              nprc(i,k) = prc(i,k)/(qc3d(i,k)/nc3d(i,k))
              nprc(i,k) = min( nprc(i,k) , nc3d(i,k)/dt )
              nprc1(i,k) = min( nprc1(i,k) , nprc(i,k)    )
            endif
            if (qr3d(i,k) >= 1.e-8.and.qni3d(i,k) >= 1.e-8) then
              ums(i) = asn(i,k)*cons3/pow(lams(i,k),bs)
              umr(i) = arn(i,k)*cons4/pow(lamr(i,k),br)
              uns(i) = asn(i,k)*cons5/pow(lams(i,k),bs)
              unr(i) = arn(i,k)*cons6/pow(lamr(i,k),br)
              dum = pow(rhosu/rho(i,k),0.54)
              ums(i) = min( ums(i) , 1.2*dum )
              uns(i) = min( uns(i) , 1.2*dum )
              umr(i) = min( umr(i) , 9.1*dum )
              unr(i) = min( unr(i) , 9.1*dum )
              pracs(i,k) = cons41*(pow(pow(1.2*umr(i)-0.95*ums(i),2)+0.08*ums(i)*umr(i),0.5)*rho(i,k)*n0rr(i,k)*n0s(i,k)/pow(lamr(i,k),3)* &
                          (5./(pow(lamr(i,k),3)*lams(i,k))+2./(pow(lamr(i,k),2)*pow(lams(i,k),2))+0.5/(lamr(i,k)*pow(lams(i,k),3))))
            endif
            if (qr3d(i,k) >= 1.e-8.and.qg3d(i,k) >= 1.e-8) then
              umg(i) = agn(i,k)*cons7/pow(lamg(i,k),bg)
              umr(i) = arn(i,k)*cons4/pow(lamr(i,k),br)
              ung(i) = agn(i,k)*cons8/pow(lamg(i,k),bg)
              unr(i) = arn(i,k)*cons6/pow(lamr(i,k),br)
              dum = pow(rhosu/rho(i,k),0.54)
              umg(i) = min( umg(i) , 20.*dum )
              ung(i) = min( ung(i) , 20.*dum )
              umr(i) = min( umr(i) , 9.1*dum )
              unr(i) = min( unr(i) , 9.1*dum )
              pracg(i,k)  = cons41*(pow(pow(1.2*umr(i)-0.95*umg(i),2)+0.08*umg(i)*umr(i),0.5)*rho(i,k)*n0rr(i,k)*n0g(i,k)/pow(lamr(i,k),3)* &
                          (5./(pow(lamr(i,k),3)*lamg(i,k))+2./(pow(lamr(i,k),2)*pow(lamg(i,k),2))+0.5/(lamr(i,k)*pow(lamg(i,k),3))))
              dum       = pracg(i,k)/5.2e-7
              npracg(i,k) = cons32*rho(i,k)*pow(1.7*pow(unr(i)-ung(i),2)+0.3*unr(i)*ung(i),0.5)*n0rr(i,k)*n0g(i,k)* &
                          (1./(pow(lamr(i,k),3)*lamg(i,k))+1./(pow(lamr(i,k),2)*pow(lamg(i,k),2))+1./(lamr(i,k)*pow(lamg(i,k),3)))
              npracg(i,k) = npracg(i,k)-dum
            endif
            if (qr3d(i,k) >= 1.e-8 .and. qc3d(i,k) >= 1.e-8) then
              dum     = (qc3d(i,k)*qr3d(i,k))
              pra(i,k) = 67.*pow(dum,1.15)
              npra(i,k) = pra(i,k)/(qc3d(i,k)/nc3d(i,k))
            endif
            if (qr3d(i,k) >= 1.e-8) then
              dum1(i)=300.e-6
              if (1./lamr(i,k) < dum1(i)) then
                dum=1.
              else if (1./lamr(i,k) >= dum1(i)) then
                dum=2.-exp(2300.*(1./lamr(i,k)-dum1(i)))
              endif
              nragg(i,k) = -5.78*dum*nr3d(i,k)*qr3d(i,k)*rho(i,k)
            endif
            if (qr3d(i,k) >= qsmall) then
              epsr(i) = 2.*pi*n0rr(i,k)*rho(i,k)*dv(i,k)*(f1r/(lamr(i,k)*lamr(i,k))+f2r*pow(arn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons9/(pow(lamr(i,k),cons34)))
            else
              epsr(i) = 0.
            endif
            if (qv3d(i,k) < qvs(i,k)) then
              pre(i,k) = epsr(i)*(qv3d(i,k)-qvs(i,k))/ab(i,k)
              pre(i,k) = min(pre(i,k),0.)
            else
              pre(i,k) = 0.
            endif
            if (qni3d(i,k) >= 1.e-8) then
              dum      = -cpw/xlf(i,k)*(t3d(i,k)-273.15)*pracs(i,k)
              psmlt(i,k) = 2.*pi*n0s(i,k)*kap(i,k)*(273.15-t3d(i,k))/xlf(i,k)*(f1s/(lams(i,k)*lams(i,k))+f2s*pow(asn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons10/(pow(lams(i,k),cons35)))+dum
              if (qvqvs(i,k) < 1.) then
                epss(i)     = 2.*pi*n0s(i,k)*rho(i,k)*dv(i,k)*(f1s/(lams(i,k)*lams(i,k))+f2s*pow(asn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons10/(pow(lams(i,k),cons35)))
                evpms(i,k) = (qv3d(i,k)-qvs(i,k))*epss(i)/ab(i,k)    
                evpms(i,k) = max(evpms(i,k),psmlt(i,k))
                psmlt(i,k) = psmlt(i,k)-evpms(i,k)
              endif
            endif
            if (qg3d(i,k) >= 1.e-8) then
              dum      = -cpw/xlf(i,k)*(t3d(i,k)-273.15)*pracg(i,k)
              pgmlt(i,k) = 2.*pi*n0g(i,k)*kap(i,k)*(273.15-t3d(i,k))/xlf(i,k)*(f1s/(lamg(i,k)*lamg(i,k))+f2s*pow(agn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons11/(pow(lamg(i,k),cons36)))+dum
              if (qvqvs(i,k) < 1.) then
                epsg(i)     = 2.*pi*n0g(i,k)*rho(i,k)*dv(i,k)*(f1s/(lamg(i,k)*lamg(i,k))+f2s*pow(agn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons11/(pow(lamg(i,k),cons36)))
                evpmg(i,k) = (qv3d(i,k)-qvs(i,k))*epsg(i)/ab(i,k)
                evpmg(i,k) = max(evpmg(i,k),pgmlt(i,k))
                pgmlt(i,k) = pgmlt(i,k)-evpmg(i,k)
              endif
            endif
            pracg(i,k) = 0.
            pracs(i,k) = 0.
            dum = (prc(i,k)+pra(i,k))*dt
            if (dum > qc3d(i,k).and.qc3d(i,k) >= qsmall) then
              ratio(i) = qc3d(i,k)/dum
              prc(i,k) = prc(i,k)*ratio(i)
              pra(i,k) = pra(i,k)*ratio(i)
            endif
            dum = (-psmlt(i,k)-evpms(i,k)+pracs(i,k))*dt
            if (dum > qni3d(i,k).and.qni3d(i,k) >= qsmall) then
              ratio(i)    = qni3d(i,k)/dum
              psmlt(i,k) = psmlt(i,k)*ratio(i)
              evpms(i,k) = evpms(i,k)*ratio(i)
              pracs(i,k) = pracs(i,k)*ratio(i)
            endif
            dum = (-pgmlt(i,k)-evpmg(i,k)+pracg(i,k))*dt
            if (dum > qg3d(i,k).and.qg3d(i,k) >= qsmall) then
              ratio(i)    = qg3d (i,k)/dum
              pgmlt(i,k) = pgmlt(i,k)*ratio(i)
              evpmg(i,k) = evpmg(i,k)*ratio(i)
              pracg(i,k) = pracg(i,k)*ratio(i)
            endif
            dum = (-pracs(i,k)-pracg(i,k)-pre(i,k)-pra(i,k)-prc(i,k)+psmlt(i,k)+pgmlt(i,k))*dt
            if (dum > qr3d(i,k).and.qr3d(i,k) >= qsmall) then
              ratio(i)  = (qr3d(i,k)/dt+pracs(i,k)+pracg(i,k)+pra(i,k)+prc(i,k)-psmlt(i,k)-pgmlt(i,k))/(-pre(i,k))
              pre(i,k) = pre(i,k)*ratio(i)
            endif
            qv3dten (i,k) = qv3dten (i,k) + (-pre(i,k)-evpms(i,k)-evpmg(i,k))
            t3dten  (i,k) = t3dten  (i,k) + (pre(i,k)*xxlv(i,k)+(evpms(i,k)+evpmg(i,k))*xxls(i,k)+(psmlt(i,k)+pgmlt(i,k)-pracs(i,k)-pracg(i,k))*xlf(i,k))/cpm(i,k)
            qc3dten (i,k) = qc3dten (i,k) + (-pra(i,k)-prc(i,k))
            qr3dten (i,k) = qr3dten (i,k) + (pre(i,k)+pra(i,k)+prc(i,k)-psmlt(i,k)-pgmlt(i,k)+pracs(i,k)+pracg(i,k))
            qni3dten(i,k) = qni3dten(i,k) + (psmlt(i,k)+evpms(i,k)-pracs(i,k))
            qg3dten (i,k) = qg3dten (i,k) + (pgmlt(i,k)+evpmg(i,k)-pracg(i,k))
            nc3dten (i,k) = nc3dten (i,k) + (-npra(i,k)-nprc(i,k))
            nr3dten (i,k) = nr3dten (i,k) + (nprc1(i,k)+nragg(i,k)-npracg(i,k))
            c2prec  (i,k) = pra(i,k)+prc(i,k)
            if (pre(i,k) < 0.) then
              dum      = pre(i,k)*dt/qr3d(i,k)
              dum      = max(-1.,dum)
              nsubr(i,k) = dum*nr3d(i,k)/dt
            endif
            if (evpms(i,k)+psmlt(i,k) < 0.) then
              dum       = (evpms(i,k)+psmlt(i,k))*dt/qni3d(i,k)
              dum       = max(-1.,dum)
              nsmlts(i,k) = dum*ns3d(i,k)/dt
            endif
            if (psmlt(i,k) < 0.) then
              dum       = psmlt(i,k)*dt/qni3d(i,k)
              dum       = max(-1.0,dum)
              nsmltr(i,k) = dum*ns3d(i,k)/dt
            endif
            if (evpmg(i,k)+pgmlt(i,k) < 0.) then
              dum       = (evpmg(i,k)+pgmlt(i,k))*dt/qg3d(i,k)
              dum       = max(-1.,dum)
              ngmltg(i,k) = dum*ng3d(i,k)/dt
            endif
            if (pgmlt(i,k) < 0.) then
              dum       = pgmlt(i,k)*dt/qg3d(i,k)
              dum       = max(-1.0,dum)
              ngmltr(i,k) = dum*ng3d(i,k)/dt
            endif
            ns3dten(i,k) = ns3dten(i,k)+(nsmlts(i,k))
            ng3dten(i,k) = ng3dten(i,k)+(ngmltg(i,k))
            nr3dten(i,k) = nr3dten(i,k)+(nsubr(i,k)-nsmltr(i,k)-ngmltr(i,k))
          endif

          dumt(i) = t3d(i,k)+dt*t3dten(i,k)
          dumqv(i) = qv3d(i,k)+dt*qv3dten(i,k)
          dum=min(0.99*pres(i,k),polysvp(dumt(i),0))
          dumqss(i) = ep_2*dum/(pres(i,k)-dum)
          dumqc(i) = qc3d(i,k)+dt*qc3dten(i,k)
          dumqc(i) = max(dumqc(i),0.)
          dums(i) = dumqv(i)-dumqss(i)
          pcc(i,k) = dums(i)/(1.+pow(xxlv(i,k),2)*dumqss(i)/(cpm(i,k)*rv*pow(dumt(i),2)))/dt
          if (pcc(i,k)*dt+dumqc(i) < 0.) then
            pcc(i,k) = -dumqc(i)/dt
          endif
          qv3dten(i,k) = qv3dten(i,k)-pcc(i,k)
          t3dten(i,k) = t3dten(i,k)+pcc(i,k)*xxlv(i,k)/cpm(i,k)
          qc3dten(i,k) = qc3dten(i,k)+pcc(i,k)
        else  ! temperature < 273.15
          if (iinum==1) then
            nc3d(i,k)=ndcnst*1.e6/rho(i,k)
          endif
          ni3d(i,k) = max(0.,ni3d(i,k))
          ns3d(i,k) = max(0.,ns3d(i,k))
          nc3d(i,k) = max(0.,nc3d(i,k))
          nr3d(i,k) = max(0.,nr3d(i,k))
          ng3d(i,k) = max(0.,ng3d(i,k))
          if (qi3d(i,k) >= qsmall) then
            lami(i,k) = pow(cons12*ni3d(i,k)/qi3d(i,k),1./di)
            n0i(i,k) = ni3d(i,k)*lami(i,k)
            if (lami(i,k) < lammini) then
              lami(i,k) = lammini
              n0i(i,k) = pow(lami(i,k),4)*qi3d(i,k)/cons12
              ni3d(i,k) = n0i(i,k)/lami(i,k)
            else if (lami(i,k) > lammaxi) then
              lami(i,k) = lammaxi
              n0i(i,k) = pow(lami(i,k),4)*qi3d(i,k)/cons12
              ni3d(i,k) = n0i(i,k)/lami(i,k)
            endif
          endif
          if (qr3d(i,k) >= qsmall) then
            lamr(i,k) = pow(pi*rhow*nr3d(i,k)/qr3d(i,k),1./3.)
            n0rr(i,k) = nr3d(i,k)*lamr(i,k)
            if (lamr(i,k) < lamminr) then
              lamr(i,k) = lamminr
              n0rr(i,k) = pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow)
              nr3d(i,k) = n0rr(i,k)/lamr(i,k)
            else if (lamr(i,k) > lammaxr) then
              lamr(i,k) = lammaxr
              n0rr(i,k) = pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow)
              nr3d(i,k) = n0rr(i,k)/lamr(i,k)
            endif
          endif
          if (qc3d(i,k) >= qsmall) then
            dum     = pres(i,k)/(287.15*t3d(i,k))
            pgam(i,k) = 0.0005714*(nc3d(i,k)/1.e6*dum)+0.2714
            pgam(i,k) = 1./(pow(pgam(i,k),2))-1.
            pgam(i,k) = max(pgam(i,k),2.)
            pgam(i,k) = min(pgam(i,k),10.)
            lamc(i,k) = pow(cons26*nc3d(i,k)*gamma(pgam(i,k)+4.)/(qc3d(i,k)*gamma(pgam(i,k)+1.)),1./3.)
            lammin(i)  = (pgam(i,k)+1.)/60.e-6
            lammax(i)  = (pgam(i,k)+1.)/1.e-6
            if (lamc(i,k) < lammin(i)) then
              lamc(i,k) = lammin(i)
              nc3d(i,k) = exp(3.*log(lamc(i,k))+log(qc3d(i,k))+log(gamma(pgam(i,k)+1.))-log(gamma(pgam(i,k)+4.)))/cons26
            else if (lamc(i,k) > lammax(i)) then
              lamc(i,k) = lammax(i)
              nc3d(i,k) = exp(3.*log(lamc(i,k))+log(qc3d(i,k))+log(gamma(pgam(i,k)+1.))-log(gamma(pgam(i,k)+4.)))/cons26
            endif
            cdist1(i,k) = nc3d(i,k)/gamma(pgam(i,k)+1.)
          endif
          if (qni3d(i,k) >= qsmall) then
            lams(i,k) = pow(cons1*ns3d(i,k)/qni3d(i,k),1./ds)
            n0s(i,k) = ns3d(i,k)*lams(i,k)
            if (lams(i,k) < lammins) then
              lams(i,k) = lammins
              n0s(i,k) = pow(lams(i,k),4)*qni3d(i,k)/cons1
              ns3d(i,k) = n0s(i,k)/lams(i,k)
            else if (lams(i,k) > lammaxs) then
              lams(i,k) = lammaxs
              n0s(i,k) = pow(lams(i,k),4)*qni3d(i,k)/cons1
              ns3d(i,k) = n0s(i,k)/lams(i,k)
            endif
          endif
          if (qg3d(i,k) >= qsmall) then
            lamg(i,k) = pow(cons2*ng3d(i,k)/qg3d(i,k),1./dg)
            n0g(i,k) = ng3d(i,k)*lamg(i,k)
            if (lamg(i,k) < lamming) then
              lamg(i,k) = lamming
              n0g(i,k) = pow(lamg(i,k),4)*qg3d(i,k)/cons2
              ng3d(i,k) = n0g(i,k)/lamg(i,k)
            else if (lamg(i,k) > lammaxg) then
              lamg(i,k) = lammaxg
              n0g(i,k) = pow(lamg(i,k),4)*qg3d(i,k)/cons2
              ng3d(i,k) = n0g(i,k)/lamg(i,k)
            endif
          endif
          mnuccc(i,k) = 0.
          nnuccc(i,k) = 0.
          prc(i,k) = 0.
          nprc(i,k) = 0.
          nprc1(i,k) = 0.
          nsagg(i,k) = 0.
          psacws(i,k) = 0.
          npsacws(i,k) = 0.
          psacwi(i,k) = 0.
          npsacwi(i,k) = 0.
          pracs(i,k) = 0.
          npracs(i,k) = 0.
          nmults(i,k) = 0.
          qmults(i,k) = 0.
          nmultr(i,k) = 0.
          qmultr(i,k) = 0.
          nmultg(i,k) = 0.
          qmultg(i,k) = 0.
          nmultrg(i,k) = 0.
          qmultrg(i,k) = 0.
          mnuccr(i,k) = 0.
          nnuccr(i,k) = 0.
          pra(i,k) = 0.
          npra(i,k) = 0.
          nragg(i,k) = 0.
          prci(i,k) = 0.
          nprci(i,k) = 0.
          prai(i,k) = 0.
          nprai(i,k) = 0.
          nnuccd(i,k) = 0.
          mnuccd(i,k) = 0.
          pcc(i,k) = 0.
          pre(i,k) = 0.
          prd(i,k) = 0.
          prds(i,k) = 0.
          eprd(i,k) = 0.
          eprds(i,k) = 0.
          nsubc(i,k) = 0.
          nsubi(i,k) = 0.
          nsubs(i,k) = 0.
          nsubr(i,k) = 0.
          piacr(i,k) = 0.
          niacr(i,k) = 0.
          praci(i,k) = 0.
          piacrs(i,k) = 0.
          niacrs(i,k) = 0.
          pracis(i,k) = 0.
          pracg(i,k) = 0.
          psacr(i,k) = 0.
          psacwg(i,k) = 0.
          pgsacw(i,k) = 0.
          pgracs(i,k) = 0.
          prdg(i,k) = 0.
          eprdg(i,k) = 0.
          npracg(i,k) = 0.
          npsacwg(i,k) = 0.
          nscng(i,k) = 0.
          ngracs(i,k) = 0.
          nsubg(i,k) = 0.
          if (qc3d(i,k) >= qsmall .and. t3d(i,k) < 269.15) then
            nacnt(i)     = exp(-2.80+0.262*(273.15-t3d(i,k)))*1000.
            dum       = 7.37*t3d(i,k)/(288.*10.*pres(i,k))/100.
            dap(i,k) = cons37*t3d(i,k)*(1.+dum/rin)/mu(i,k)
            mnuccc(i,k) = cons38*dap(i,k)*nacnt(i)*exp(log(cdist1(i,k))+log(gamma(pgam(i,k)+5.))-4.*log(lamc(i,k)))
            nnuccc(i,k) = 2.*pi*dap(i,k)*nacnt(i)*cdist1(i,k)*gamma(pgam(i,k)+2.)/lamc(i,k)
            mnuccc(i,k) = mnuccc(i,k)+cons39*exp(log(cdist1(i,k))+log(gamma(7.+pgam(i,k)))-6.*log(lamc(i,k)))*(exp(aimm*(273.15-t3d(i,k)))-1.)
            nnuccc(i,k) = nnuccc(i,k)+cons40*exp(log(cdist1(i,k))+log(gamma(pgam(i,k)+4.))-3.*log(lamc(i,k)))*(exp(aimm*(273.15-t3d(i,k)))-1.)
            nnuccc(i,k) = min(nnuccc(i,k),nc3d(i,k)/dt)
          endif
          if (qc3d(i,k) >= 1.e-6) then
            prc(i,k) = 1350.*pow(qc3d(i,k),2.47)*pow(nc3d(i,k)/1.e6*rho(i,k),-1.79)
            nprc1(i,k) = prc(i,k)/cons29
            nprc(i,k) = prc(i,k)/(qc3d(i,k)/nc3d(i,k))
            nprc(i,k) = min( nprc(i,k) , nc3d(i,k)/dt )
            nprc1(i,k) = min( nprc1(i,k) , nprc(i,k)    )
          endif
          if (qni3d(i,k) >= 1.e-8) then
            nsagg(i,k) = cons15*asn(i,k)*pow(rho(i,k),(2.+bs)/3.)*pow(qni3d(i,k),(2.+bs)/3.)*pow(ns3d(i,k)*rho(i,k),(4.-bs)/3.)/(rho(i,k))
          endif
          if (qni3d(i,k) >= 1.e-8 .and. qc3d(i,k) >= qsmall) then
            psacws(i,k) = cons13*asn(i,k)*qc3d(i,k)*rho(i,k)*n0s(i,k)/pow(lams(i,k),bs+3.)
            npsacws(i,k) = cons13*asn(i,k)*nc3d(i,k)*rho(i,k)*n0s(i,k)/pow(lams(i,k),bs+3.)
          endif
          if (qg3d(i,k) >= 1.e-8 .and. qc3d(i,k) >= qsmall) then
            psacwg(i,k) = cons14*agn(i,k)*qc3d(i,k)*rho(i,k)*n0g(i,k)/pow(lamg(i,k),bg+3.)
            npsacwg(i,k) = cons14*agn(i,k)*nc3d(i,k)*rho(i,k)*n0g(i,k)/pow(lamg(i,k),bg+3.)
          endif
          if (qi3d(i,k) >= 1.e-8 .and. qc3d(i,k) >= qsmall) then
            if (1./lami(i,k) >= 100.e-6) then
              psacwi(i,k) = cons16*ain(i,k)*qc3d(i,k)*rho(i,k)*n0i(i,k)/pow(lami(i,k),bi+3.)
              npsacwi(i,k) = cons16*ain(i,k)*nc3d(i,k)*rho(i,k)*n0i(i,k)/pow(lami(i,k),bi+3.)
            endif
          endif
          if (qr3d(i,k) >= 1.e-8.and.qni3d(i,k) >= 1.e-8) then
            ums(i) = asn(i,k)*cons3/pow(lams(i,k),bs)
            umr(i) = arn(i,k)*cons4/pow(lamr(i,k),br)
            uns(i) = asn(i,k)*cons5/pow(lams(i,k),bs)
            unr(i) = arn(i,k)*cons6/pow(lamr(i,k),br)
            dum = pow(rhosu/rho(i,k),0.54)
            ums(i) = min( ums(i) , 1.2*dum )
            uns(i) = min( uns(i) , 1.2*dum )
            umr(i) = min( umr(i) , 9.1*dum )
            unr(i) = min( unr(i) , 9.1*dum )
            pracs(i,k) = cons41*(pow(pow(1.2*umr(i)-0.95*ums(i),2)+0.08*ums(i)*umr(i),0.5)*rho(i,k)*n0rr(i,k)*n0s(i,k)/pow(lamr(i,k),3)* &
                       (5./(pow(lamr(i,k),3)*lams(i,k))+2./(pow(lamr(i,k),2)*pow(lams(i,k),2))+0.5/(lamr(i,k)*pow(lams(i,k),3))))
            npracs(i,k) = cons32*rho(i,k)*pow(1.7*pow(unr(i)-uns(i),2)+0.3*unr(i)*uns(i),0.5)*n0rr(i,k)*n0s(i,k)*(1./(pow(lamr(i,k),3)*lams(i,k))+ &
                        1./(pow(lamr(i,k),2)*pow(lams(i,k),2))+1./(lamr(i,k)*pow(lams(i,k),3)))
            pracs(i,k) = min(pracs(i,k),qr3d(i,k)/dt)
            if (qni3d(i,k) >= 0.1e-3.and.qr3d(i,k) >= 0.1e-3) then
              psacr(i,k) = cons31*(pow(pow(1.2*umr(i)-0.95*ums(i),2)+0.08*ums(i)*umr(i),0.5)*rho(i,k)*n0rr(i,k)*n0s(i,k)/pow(lams(i,k),3)* &
                         (5./(pow(lams(i,k),3)*lamr(i,k))+2./(pow(lams(i,k),2)*pow(lamr(i,k),2))+0.5/(lams(i,k)*pow(lamr(i,k),3))))            
            endif
          endif
          if (qr3d(i,k) >= 1.e-8.and.qg3d(i,k) >= 1.e-8) then
            umg(i) = agn(i,k)*cons7/pow(lamg(i,k),bg)
            umr(i) = arn(i,k)*cons4/pow(lamr(i,k),br)
            ung(i) = agn(i,k)*cons8/pow(lamg(i,k),bg)
            unr(i) = arn(i,k)*cons6/pow(lamr(i,k),br)
            dum = pow(rhosu/rho(i,k),0.54)
            umg(i) = min( umg(i) , 20.*dum )
            ung(i) = min( ung(i) , 20.*dum )
            umr(i) = min( umr(i) , 9.1*dum )
            unr(i) = min( unr(i) , 9.1*dum )
            pracg(i,k) = cons41*(pow(pow(1.2*umr(i)-0.95*umg(i),2)+0.08*umg(i)*umr(i),0.5)*rho(i,k)*n0rr(i,k)*n0g(i,k)/pow(lamr(i,k),3)* &
                        (5./(pow(lamr(i,k),3)*lamg(i,k))+2./(pow(lamr(i,k),2)*pow(lamg(i,k),2))+0.5/(lamr(i,k)*pow(lamg(i,k),3))))
            npracg(i,k) = cons32*rho(i,k)*pow(1.7*pow(unr(i)-ung(i),2)+0.3*unr(i)*ung(i),0.5)*n0rr(i,k)*n0g(i,k)*(1./(pow(lamr(i,k),3)*lamg(i,k))+ &
                        1./(pow(lamr(i,k),2)*pow(lamg(i,k),2))+1./(lamr(i,k)*pow(lamg(i,k),3)))
            pracg(i,k) = min(pracg(i,k),qr3d(i,k)/dt)
          endif
          if (qni3d(i,k) >= 0.1e-3) then
            if (qc3d(i,k) >= 0.5e-3.or.qr3d(i,k) >= 0.1e-3) then
              if (psacws(i,k) > 0..or.pracs(i,k) > 0.) then
                if (t3d(i,k) < 270.16 .and. t3d(i,k) > 265.16) then
                  if (t3d(i,k) > 270.16) then
                    fmult(i) = 0.
                  else if (t3d(i,k) <= 270.16.and.t3d(i,k) > 268.16)  then
                    fmult(i) = (270.16-t3d(i,k))/2.
                  else if (t3d(i,k) >= 265.16.and.t3d(i,k) <= 268.16)   then
                    fmult(i) = (t3d(i,k)-265.16)/3.
                  else if (t3d(i,k) < 265.16) then
                    fmult(i) = 0.
                  endif
                  if (psacws(i,k) > 0.) then
                    nmults(i,k) = 35.e4*psacws(i,k)*fmult(i)*1000.
                    qmults(i,k) = nmults(i,k)*mmult
                    qmults(i,k) = min(qmults(i,k),psacws(i,k))
                    psacws(i,k) = psacws(i,k)-qmults(i,k)
                  endif
                  if (pracs(i,k) > 0.) then
                    nmultr(i,k) = 35.e4*pracs(i,k)*fmult(i)*1000.
                    qmultr(i,k) = nmultr(i,k)*mmult
                    qmultr(i,k) = min(qmultr(i,k),pracs(i,k))
                    pracs(i,k) = pracs(i,k)-qmultr(i,k)
                  endif
                endif
              endif
            endif
          endif
          if (qg3d(i,k) >= 0.1e-3) then
            if (qc3d(i,k) >= 0.5e-3.or.qr3d(i,k) >= 0.1e-3) then
              if (psacwg(i,k) > 0..or.pracg(i,k) > 0.) then
                if (t3d(i,k) < 270.16 .and. t3d(i,k) > 265.16) then
                  if (t3d(i,k) > 270.16) then
                    fmult(i) = 0.
                  else if (t3d(i,k) <= 270.16.and.t3d(i,k) > 268.16)  then
                    fmult(i) = (270.16-t3d(i,k))/2.
                  else if (t3d(i,k) >= 265.16.and.t3d(i,k) <= 268.16)   then
                    fmult(i) = (t3d(i,k)-265.16)/3.
                  else if (t3d(i,k) < 265.16) then
                    fmult(i) = 0.
                  endif
                  if (psacwg(i,k) > 0.) then
                    nmultg(i,k) = 35.e4*psacwg(i,k)*fmult(i)*1000.
                    qmultg(i,k) = nmultg(i,k)*mmult
                    qmultg(i,k) = min(qmultg(i,k),psacwg(i,k))
                    psacwg(i,k) = psacwg(i,k)-qmultg(i,k)
                  endif
                  if (pracg(i,k) > 0.) then
                    nmultrg(i,k) = 35.e4*pracg(i,k)*fmult(i)*1000.
                    qmultrg(i,k) = nmultrg(i,k)*mmult
                    qmultrg(i,k) = min(qmultrg(i,k),pracg(i,k))
                    pracg(i,k) = pracg(i,k)-qmultrg(i,k)
                  endif
                endif
              endif
            endif
          endif
          if (psacws(i,k) > 0.) then
            if (qni3d(i,k) >= 0.1e-3.and.qc3d(i,k) >= 0.5e-3) then
              pgsacw(i,k) = min(psacws(i,k),cons17*dt*n0s(i,k)*qc3d(i,k)*qc3d(i,k)*asn(i,k)*asn(i,k)/(rho(i,k)*pow(lams(i,k),2.*bs+2.)))
              dum       = max(rhosn/(rhog-rhosn)*pgsacw(i,k),0.) 
              nscng(i,k) = dum/mg0*rho(i,k)
              nscng(i,k) = min(nscng(i,k),ns3d(i,k)/dt)
              psacws(i,k) = psacws(i,k) - pgsacw(i,k)
            endif
          endif
          if (pracs(i,k) > 0.) then
            if (qni3d(i,k) >= 0.1e-3.and.qr3d(i,k) >= 0.1e-3) then
              dum       = cons18*pow(4./lams(i,k),3)*pow(4./lams(i,k),3)/(cons18*pow(4./lams(i,k),3)*pow(4./lams(i,k),3)+ &  
                          cons19*pow(4./lamr(i,k),3)*pow(4./lamr(i,k),3))
              dum       = min( dum , 1. )
              dum       = max( dum , 0. )
              pgracs(i,k) = (1.-dum)*pracs(i,k)
              ngracs(i,k) = (1.-dum)*npracs(i,k)
              ngracs(i,k) = min(ngracs(i,k),nr3d(i,k)/dt)
              ngracs(i,k) = min(ngracs(i,k),ns3d(i,k)/dt)
              pracs(i,k) = pracs(i,k) - pgracs(i,k)
              npracs(i,k) = npracs(i,k) - ngracs(i,k)
              psacr(i,k) = psacr(i,k)*(1.-dum)
            endif
          endif
          if (t3d(i,k) < 269.15.and.qr3d(i,k) >= qsmall) then
            mnuccr(i,k) = cons20*nr3d(i,k)*(exp(aimm*(273.15-t3d(i,k)))-1.)/pow(lamr(i,k),3)/pow(lamr(i,k),3)
            nnuccr(i,k) = pi*nr3d(i,k)*bimm*(exp(aimm*(273.15-t3d(i,k)))-1.)/pow(lamr(i,k),3)
            nnuccr(i,k) = min(nnuccr(i,k),nr3d(i,k)/dt)
          endif
          if (qr3d(i,k) >= 1.e-8 .and. qc3d(i,k) >= 1.e-8) then
            dum     = (qc3d(i,k)*qr3d(i,k))
            pra(i,k) = 67.*pow(dum,1.15)
            npra(i,k) = pra(i,k)/(qc3d(i,k)/nc3d(i,k))
          endif
          if (qr3d(i,k) >= 1.e-8) then
            dum1(i)=300.e-6
            if (1./lamr(i,k) < dum1(i)) then
              dum=1.
            else if (1./lamr(i,k) >= dum1(i)) then
              dum=2.-exp(2300.*(1./lamr(i,k)-dum1(i)))
            endif
            nragg(i,k) = -5.78*dum*nr3d(i,k)*qr3d(i,k)*rho(i,k)
          endif
          if (qi3d(i,k) >= 1.e-8 .and.qvqvsi(i,k) >= 1.) then
            nprci(i,k) = cons21*(qv3d(i,k)-qvi(i,k))*rho(i,k)*n0i(i,k)*exp(-lami(i,k)*dcs)*dv(i,k)/abi(i,k)
            prci(i,k) = cons22*nprci(i,k)
            nprci(i,k) = min(nprci(i,k),ni3d(i,k)/dt)
          endif
          if (qni3d(i,k) >= 1.e-8 .and. qi3d(i,k) >= qsmall) then
            prai(i,k) = cons23*asn(i,k)*qi3d(i,k)*rho(i,k)*n0s(i,k)/pow(lams(i,k),bs+3.)
            nprai(i,k) = cons23*asn(i,k)*ni3d(i,k)*rho(i,k)*n0s(i,k)/pow(lams(i,k),bs+3.)
            nprai(i,k) = min( nprai(i,k) , ni3d(i,k)/dt )
          endif
          if (qr3d(i,k) >= 1.e-8 .and. qi3d(i,k) >= 1.e-8 .and. t3d(i,k) <= 273.15) then
            if (qr3d(i,k) >= 0.1e-3) then
              niacr(i,k)=cons24*ni3d(i,k)*n0rr(i,k)*arn(i,k)/pow(lamr(i,k),br+3.)*rho(i,k)
              piacr(i,k)=cons25*ni3d(i,k)*n0rr(i,k)*arn(i,k)/pow(lamr(i,k),br+3.)/pow(lamr(i,k),3)*rho(i,k)
              praci(i,k)=cons24*qi3d(i,k)*n0rr(i,k)*arn(i,k)/pow(lamr(i,k),br+3.)*rho(i,k)
              niacr(i,k)=min(niacr(i,k),nr3d(i,k)/dt)
              niacr(i,k)=min(niacr(i,k),ni3d(i,k)/dt)
            else 
              niacrs(i,k)=cons24*ni3d(i,k)*n0rr(i,k)*arn(i,k)/pow(lamr(i,k),br+3.)*rho(i,k)
              piacrs(i,k)=cons25*ni3d(i,k)*n0rr(i,k)*arn(i,k)/pow(lamr(i,k),br+3.)/pow(lamr(i,k),3)*rho(i,k)
              pracis(i,k)=cons24*qi3d(i,k)*n0rr(i,k)*arn(i,k)/pow(lamr(i,k),br+3.)*rho(i,k)
              niacrs(i,k)=min(niacrs(i,k),nr3d(i,k)/dt)
              niacrs(i,k)=min(niacrs(i,k),ni3d(i,k)/dt)
            endif
          endif
          if (inuc==0) then
            if ((qvqvs(i,k) >= 0.999 .and. t3d(i,k) <= 265.15) .or. qvqvsi(i,k) >= 1.08) then
              kc2(i) = 0.005*exp(0.304*(273.15-t3d(i,k)))*1000. ! convert from l-1 to m-3
              kc2(i) = min( kc2(i) ,500.e3 )
              kc2(i) = max( kc2(i)/rho(i,k) , 0. )  ! convert to kg-1
              if (kc2(i) > ni3d(i,k)+ns3d(i,k)+ng3d(i,k)) then
                nnuccd(i,k) = (kc2(i)-ni3d(i,k)-ns3d(i,k)-ng3d(i,k))/dt
                mnuccd(i,k) = nnuccd(i,k)*mi0
              endif
            endif
          else if (inuc==1) then
            if (t3d(i,k) < 273.15.and.qvqvsi(i,k) > 1.) then
              kc2(i) = 0.16*1000./rho(i,k)  ! convert from l-1 to kg-1
              if (kc2(i) > ni3d(i,k)+ns3d(i,k)+ng3d(i,k)) then
                nnuccd(i,k) = (kc2(i)-ni3d(i,k)-ns3d(i,k)-ng3d(i,k))/dt
                mnuccd(i,k) = nnuccd(i,k)*mi0
              endif
            endif
          endif

          if (qi3d(i,k) >= qsmall) then
             epsi(i) = 2.*pi*n0i(i,k)*rho(i,k)*dv(i,k)/(lami(i,k)*lami(i,k))
          else
             epsi(i) = 0.
          endif
          if (qni3d(i,k) >= qsmall) then
            epss(i) = 2.*pi*n0s(i,k)*rho(i,k)*dv(i,k)*(f1s/(lams(i,k)*lams(i,k))+f2s*pow(asn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons10/(pow(lams(i,k),cons35)))
          else
            epss(i) = 0.
          endif
          if (qg3d(i,k) >= qsmall) then
            epsg(i) = 2.*pi*n0g(i,k)*rho(i,k)*dv(i,k)*(f1s/(lamg(i,k)*lamg(i,k))+f2s*pow(agn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons11/(pow(lamg(i,k),cons36)))
          else
            epsg(i) = 0.
          endif
          if (qr3d(i,k) >= qsmall) then
            epsr(i) = 2.*pi*n0rr(i,k)*rho(i,k)*dv(i,k)*(f1r/(lamr(i,k)*lamr(i,k))+f2r*pow(arn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons9/(pow(lamr(i,k),cons34)))
          else
            epsr(i) = 0.
          endif
          if (qi3d(i,k) >= qsmall) then              
            dum    = (1.-exp(-lami(i,k)*dcs)*(1.+lami(i,k)*dcs))
            prd(i,k) = epsi(i)*(qv3d(i,k)-qvi(i,k))/abi(i,k)*dum
          else
            dum=0.
          endif
          if (qni3d(i,k) >= qsmall) then
            prds(i,k) = epss(i)*(qv3d(i,k)-qvi(i,k))/abi(i,k)+epsi(i)*(qv3d(i,k)-qvi(i,k))/abi(i,k)*(1.-dum)
          else
            prd(i,k) = prd(i,k)+epsi(i)*(qv3d(i,k)-qvi(i,k))/abi(i,k)*(1.-dum)
          endif
          prdg(i,k) = epsg(i)*(qv3d(i,k)-qvi(i,k))/abi(i,k)
          if (qv3d(i,k) < qvs(i,k)) then
            pre(i,k) = epsr(i)*(qv3d(i,k)-qvs(i,k))/ab(i,k)
            pre(i,k) = min( pre(i,k) , 0. )
          else
            pre(i,k) = 0.
          endif
          dum = (qv3d(i,k)-qvi(i,k))/dt
          fudgef(i) = 0.9999
          sum_dep(i) = prd(i,k)+prds(i,k)+mnuccd(i,k)+prdg(i,k)
          if( (dum > 0. .and. sum_dep(i) > dum*fudgef(i)) .or. (dum < 0. .and. sum_dep(i) < dum*fudgef(i)) ) then
            mnuccd(i,k) = fudgef(i)*mnuccd(i,k)*dum/sum_dep(i)
            prd(i,k) = fudgef(i)*prd(i,k)*dum/sum_dep(i)
            prds(i,k) = fudgef(i)*prds(i,k)*dum/sum_dep(i)
            prdg(i,k) = fudgef(i)*prdg(i,k)*dum/sum_dep(i)
          endif
          if (prd(i,k) < 0.) then
            eprd(i,k)=prd(i,k)
            prd(i,k)=0.
          endif
          if (prds(i,k) < 0.) then
            eprds(i,k)=prds(i,k)
            prds(i,k)=0.
          endif
          if (prdg(i,k) < 0.) then
            eprdg(i,k)=prdg(i,k)
            prdg(i,k)=0.
          endif
          if (iliq==1) then
            mnuccc(i,k)=0.
            nnuccc(i,k)=0.
            mnuccr(i,k)=0.
            nnuccr(i,k)=0.
            mnuccd(i,k)=0.
            nnuccd(i,k)=0.
          endif
          if (igraup==1) then
            pracg(i,k) = 0.
            psacr(i,k) = 0.
            psacwg(i,k) = 0.
            prdg(i,k) = 0.
            eprdg(i,k) = 0.
            evpmg(i,k) = 0.
            pgmlt(i,k) = 0.
            npracg(i,k) = 0.
            npsacwg(i,k) = 0.
            nscng(i,k) = 0.
            ngracs(i,k) = 0.
            nsubg(i,k) = 0.
            ngmltg(i,k) = 0.
            ngmltr(i,k) = 0.
            piacrs(i,k) = piacrs(i,k)+piacr(i,k)
            piacr(i,k) = 0.
            pracis(i,k) = pracis(i,k)+praci(i,k)
            praci(i,k) = 0.
            psacws(i,k) = psacws(i,k)+pgsacw(i,k)
            pgsacw(i,k) = 0.
            pracs(i,k) = pracs(i,k)+pgracs(i,k)
            pgracs(i,k) = 0.
          endif
          dum = (prc(i,k)+pra(i,k)+mnuccc(i,k)+psacws(i,k)+psacwi(i,k)+qmults(i,k)+psacwg(i,k)+pgsacw(i,k)+qmultg(i,k))*dt
          if (dum > qc3d(i,k) .and. qc3d(i,k) >= qsmall) then
            ratio(i) = qc3d(i,k)/dum
            prc(i,k) = prc(i,k)*ratio(i)
            pra(i,k) = pra(i,k)*ratio(i)
            mnuccc(i,k) = mnuccc(i,k)*ratio(i)
            psacws(i,k) = psacws(i,k)*ratio(i)
            psacwi(i,k) = psacwi(i,k)*ratio(i)
            qmults(i,k) = qmults(i,k)*ratio(i)
            qmultg(i,k) = qmultg(i,k)*ratio(i)
            psacwg(i,k) = psacwg(i,k)*ratio(i)
            pgsacw(i,k) = pgsacw(i,k)*ratio(i)
          endif
          dum = (-prd(i,k)-mnuccc(i,k)+prci(i,k)+prai(i,k)-qmults(i,k)-qmultg(i,k)-qmultr(i,k)-qmultrg(i,k)-mnuccd(i,k)+praci(i,k)+pracis(i,k)-eprd(i,k)-psacwi(i,k))*dt
          if (dum > qi3d(i,k) .and. qi3d(i,k) >= qsmall) then
            ratio(i) = (qi3d(i,k)/dt+prd(i,k)+mnuccc(i,k)+qmults(i,k)+qmultg(i,k)+qmultr(i,k)+qmultrg(i,k)+mnuccd(i,k)+psacwi(i,k))/(prci(i,k)+prai(i,k)+praci(i,k)+pracis(i,k)-eprd(i,k))
            prci(i,k) = prci(i,k)*ratio(i)
            prai(i,k) = prai(i,k)*ratio(i)
            praci(i,k) = praci(i,k)*ratio(i)
            pracis(i,k) = pracis(i,k)*ratio(i)
            eprd(i,k) = eprd(i,k)*ratio(i)
          endif
          dum = ((pracs(i,k)-pre(i,k))+(qmultr(i,k)+qmultrg(i,k)-prc(i,k))+(mnuccr(i,k)-pra(i,k))+piacr(i,k)+piacrs(i,k)+pgracs(i,k)+pracg(i,k))*dt
          if (dum > qr3d(i,k).and.qr3d(i,k) >= qsmall) then
            ratio(i) = (qr3d(i,k)/dt+prc(i,k)+pra(i,k))/(-pre(i,k)+qmultr(i,k)+qmultrg(i,k)+pracs(i,k)+mnuccr(i,k)+piacr(i,k)+piacrs(i,k)+pgracs(i,k)+pracg(i,k))
            pre(i,k) = pre(i,k)*ratio(i)
            pracs(i,k) = pracs(i,k)*ratio(i)
            qmultr(i,k) = qmultr(i,k)*ratio(i)
            qmultrg(i,k) = qmultrg(i,k)*ratio(i)
            mnuccr(i,k) = mnuccr(i,k)*ratio(i)
            piacr(i,k) = piacr(i,k)*ratio(i)
            piacrs(i,k) = piacrs(i,k)*ratio(i)
            pgracs(i,k) = pgracs(i,k)*ratio(i)
            pracg(i,k) = pracg(i,k)*ratio(i)
          endif
          if (igraup==0) then
            dum = (-prds(i,k)-psacws(i,k)-prai(i,k)-prci(i,k)-pracs(i,k)-eprds(i,k)+psacr(i,k)-piacrs(i,k)-pracis(i,k))*dt
            if (dum > qni3d(i,k).and.qni3d(i,k) >= qsmall) then
              ratio(i) = (qni3d(i,k)/dt+prds(i,k)+psacws(i,k)+prai(i,k)+prci(i,k)+pracs(i,k)+piacrs(i,k)+pracis(i,k))/(-eprds(i,k)+psacr(i,k))
              eprds(i,k) = eprds(i,k)*ratio(i)
              psacr(i,k) = psacr(i,k)*ratio(i)
            endif
          else if (igraup==1) then
            dum = (-prds(i,k)-psacws(i,k)-prai(i,k)-prci(i,k)-pracs(i,k)-eprds(i,k)+psacr(i,k)-piacrs(i,k)-pracis(i,k)-mnuccr(i,k))*dt
            if (dum > qni3d(i,k).and.qni3d(i,k) >= qsmall) then
              ratio(i) = (qni3d(i,k)/dt+prds(i,k)+psacws(i,k)+prai(i,k)+prci(i,k)+pracs(i,k)+piacrs(i,k)+pracis(i,k)+mnuccr(i,k))/(-eprds(i,k)+psacr(i,k))
              eprds(i,k) = eprds(i,k)*ratio(i)
              psacr(i,k) = psacr(i,k)*ratio(i)
            endif
          endif
          dum = (-psacwg(i,k)-pracg(i,k)-pgsacw(i,k)-pgracs(i,k)-prdg(i,k)-mnuccr(i,k)-eprdg(i,k)-piacr(i,k)-praci(i,k)-psacr(i,k))*dt
          if (dum > qg3d(i,k).and.qg3d(i,k) >= qsmall) then
            ratio(i) = (qg3d(i,k)/dt+psacwg(i,k)+pracg(i,k)+pgsacw(i,k)+pgracs(i,k)+prdg(i,k)+mnuccr(i,k)+psacr(i,k)+piacr(i,k)+praci(i,k))/(-eprdg(i,k))
            eprdg(i,k) = eprdg(i,k)*ratio(i)
          endif
          qv3dten(i,k) = qv3dten(i,k)+(-pre(i,k)-prd(i,k)-prds(i,k)-mnuccd(i,k)-eprd(i,k)-eprds(i,k)-prdg(i,k)-eprdg(i,k))
          t3dten(i,k) = t3dten(i,k)+(pre(i,k)*xxlv(i,k)+(prd(i,k)+prds(i,k)+mnuccd(i,k)+eprd(i,k)+eprds(i,k)+prdg(i,k)+eprdg(i,k))*xxls(i,k)+ &
                      (psacws(i,k)+psacwi(i,k)+mnuccc(i,k)+mnuccr(i,k)+qmults(i,k)+qmultg(i,k)+qmultr(i,k)+qmultrg(i,k)+pracs(i,k) &
                       +psacwg(i,k)+pracg(i,k)+pgsacw(i,k)+pgracs(i,k)+piacr(i,k)+piacrs(i,k))*xlf(i,k))/cpm(i,k)
          qc3dten(i,k) = qc3dten(i,k)+(-pra(i,k)-prc(i,k)-mnuccc(i,k)+pcc(i,k)-psacws(i,k)-psacwi(i,k)-qmults(i,k)-qmultg(i,k)-psacwg(i,k)-pgsacw(i,k))
          qi3dten(i,k) = qi3dten(i,k)+(prd(i,k)+eprd(i,k)+psacwi(i,k)+mnuccc(i,k)-prci(i,k)- &
                       prai(i,k)+qmults(i,k)+qmultg(i,k)+qmultr(i,k)+qmultrg(i,k)+mnuccd(i,k)-praci(i,k)-pracis(i,k))
          qr3dten(i,k) = qr3dten(i,k)+(pre(i,k)+pra(i,k)+prc(i,k)-pracs(i,k)-mnuccr(i,k)-qmultr(i,k)-qmultrg(i,k) &
                       -piacr(i,k)-piacrs(i,k)-pracg(i,k)-pgracs(i,k))
          if (igraup==0) then
            qni3dten(i,k) = qni3dten(i,k)+(prai(i,k)+psacws(i,k)+prds(i,k)+pracs(i,k)+prci(i,k)+eprds(i,k)-psacr(i,k)+piacrs(i,k)+pracis(i,k))
            ns3dten(i,k) = ns3dten(i,k)+(nsagg(i,k)+nprci(i,k)-nscng(i,k)-ngracs(i,k)+niacrs(i,k))
            qg3dten(i,k) = qg3dten(i,k)+(pracg(i,k)+psacwg(i,k)+pgsacw(i,k)+pgracs(i,k)+prdg(i,k)+eprdg(i,k)+mnuccr(i,k)+piacr(i,k)+praci(i,k)+psacr(i,k))
            ng3dten(i,k) = ng3dten(i,k)+(nscng(i,k)+ngracs(i,k)+nnuccr(i,k)+niacr(i,k))
          else if (igraup==1) then
            qni3dten(i,k) = qni3dten(i,k)+(prai(i,k)+psacws(i,k)+prds(i,k)+pracs(i,k)+prci(i,k)+eprds(i,k)-psacr(i,k)+piacrs(i,k)+pracis(i,k)+mnuccr(i,k))
            ns3dten(i,k) = ns3dten(i,k)+(nsagg(i,k)+nprci(i,k)-nscng(i,k)-ngracs(i,k)+niacrs(i,k)+nnuccr(i,k))
          endif
          nc3dten(i,k) = nc3dten(i,k)+(-nnuccc(i,k)-npsacws(i,k)-npra(i,k)-nprc(i,k)-npsacwi(i,k)-npsacwg(i,k))
          ni3dten(i,k) = ni3dten(i,k)+(nnuccc(i,k)-nprci(i,k)-nprai(i,k)+nmults(i,k)+nmultg(i,k)+nmultr(i,k)+nmultrg(i,k)+nnuccd(i,k)-niacr(i,k)-niacrs(i,k))
          nr3dten(i,k) = nr3dten(i,k)+(nprc1(i,k)-npracs(i,k)-nnuccr(i,k)+nragg(i,k)-niacr(i,k)-niacrs(i,k)-npracg(i,k)-ngracs(i,k))
          c2prec (i,k) = pra(i,k)+prc(i,k)+psacws(i,k)+qmults(i,k)+qmultg(i,k)+psacwg(i,k)+pgsacw(i,k)+mnuccc(i,k)+psacwi(i,k)
          dumt(i)       = t3d(i,k)+dt*t3dten(i,k)
          dumqv(i)      = qv3d(i,k)+dt*qv3dten(i,k)
          dum        = min( 0.99*pres(i,k) , polysvp(dumt(i),0) )
          dumqss(i)     = ep_2*dum/(pres(i,k)-dum)
          dumqc(i)      = qc3d(i,k)+dt*qc3dten(i,k)
          dumqc(i)      = max( dumqc(i) , 0. )
          dums(i)       = dumqv(i)-dumqss(i)
          pcc(i,k)     = dums(i)/(1.+pow(xxlv(i,k),2)*dumqss(i)/(cpm(i,k)*rv*pow(dumt(i),2)))/dt
          if (pcc(i,k)*dt+dumqc(i) < 0.) pcc(i,k) = -dumqc(i)/dt
          qv3dten(i,k) = qv3dten(i,k)-pcc(i,k)
          t3dten (i,k) = t3dten (i,k)+pcc(i,k)*xxlv(i,k)/cpm(i,k)
          qc3dten(i,k) = qc3dten(i,k)+pcc(i,k)
          if (eprd(i,k) < 0.) then
            dum      = eprd(i,k)*dt/qi3d(i,k)
            dum      = max(-1.,dum)
            nsubi(i,k) = dum*ni3d(i,k)/dt
          endif
          if (eprds(i,k) < 0.) then
            dum      = eprds(i,k)*dt/qni3d(i,k)
            dum      = max(-1.,dum)
            nsubs(i,k) = dum*ns3d(i,k)/dt
          endif
          if (pre(i,k) < 0.) then
            dum      = pre(i,k)*dt/qr3d(i,k)
            dum      = max(-1.,dum)
            nsubr(i,k) = dum*nr3d(i,k)/dt
          endif
          if (eprdg(i,k) < 0.) then
            dum      = eprdg(i,k)*dt/qg3d(i,k)
            dum      = max(-1.,dum)
            nsubg(i,k) = dum*ng3d(i,k)/dt
          endif
          ni3dten(i,k) = ni3dten(i,k)+nsubi(i,k)
          ns3dten(i,k) = ns3dten(i,k)+nsubs(i,k)
          ng3dten(i,k) = ng3dten(i,k)+nsubg(i,k)
          nr3dten(i,k) = nr3dten(i,k)+nsubr(i,k)
        endif !!!!!! temperature
        ltrue(i) = 1
      enddo  !!! k
    enddo !!! i

    do i = 1 , ncol
      precrt (i) = 0.
      snowrt (i) = 0.
      snowprt(i) = 0.
      grplprt(i) = 0.

      if (ltrue(i)==0) cycle

      nstep(i) = 1
      do k = nz,1,-1
        dumi(i,k) = qi3d (i,k)+qi3dten (i,k)*dt
        dumqs(i,k) = qni3d(i,k)+qni3dten(i,k)*dt
        dumr(i,k) = qr3d (i,k)+qr3dten (i,k)*dt
        dumfni(i,k) = ni3d (i,k)+ni3dten (i,k)*dt
        dumfns(i,k) = ns3d (i,k)+ns3dten (i,k)*dt
        dumfnr(i,k) = nr3d (i,k)+nr3dten (i,k)*dt
        dumc(i,k) = qc3d (i,k)+qc3dten (i,k)*dt
        dumfnc(i,k) = nc3d (i,k)+nc3dten (i,k)*dt
        dumg(i,k) = qg3d (i,k)+qg3dten (i,k)*dt
        dumfng(i,k) = ng3d (i,k)+ng3dten (i,k)*dt
        if (iinum==1) dumfnc(i,k) = nc3d(i,k)
        dumfni(i,k) = max( 0. , dumfni(i,k) )
        dumfns(i,k) = max( 0. , dumfns(i,k) )
        dumfnc(i,k) = max( 0. , dumfnc(i,k) )
        dumfnr(i,k) = max( 0. , dumfnr(i,k) )
        dumfng(i,k) = max( 0. , dumfng(i,k) )
        if (dumi(i,k) >= qsmall) then
          dlami(i) = pow(cons12*dumfni(i,k)/dumi(i,k),1./di)
          dlami(i) = max( dlami(i) , lammini )
          dlami(i) = min( dlami(i) , lammaxi )
        endif
        if (dumr(i,k) >= qsmall) then
          dlamr(i) = pow(pi*rhow*dumfnr(i,k)/dumr(i,k),1./3.)
          dlamr(i) = max( dlamr(i) , lamminr )
          dlamr(i) = min( dlamr(i) , lammaxr )
        endif
        if (dumc(i,k) >= qsmall) then
          dum     = pres(i,k)/(287.15*t3d(i,k))
          pgam(i,k) = 0.0005714*(nc3d(i,k)/1.e6*dum)+0.2714
          pgam(i,k) = 1./(pow(pgam(i,k),2))-1.
          pgam(i,k) = max(pgam(i,k),2.)
          pgam(i,k) = min(pgam(i,k),10.)
          dlamc(i)   = pow(cons26*dumfnc(i,k)*gamma(pgam(i,k)+4.)/(dumc(i,k)*gamma(pgam(i,k)+1.)),1./3.)
          lammin(i)  = (pgam(i,k)+1.)/60.e-6
          lammax(i)  = (pgam(i,k)+1.)/1.e-6
          dlamc(i)   = max(dlamc(i),lammin(i))
          dlamc(i)   = min(dlamc(i),lammax(i))
        endif
        if (dumqs(i,k) >= qsmall) then
          dlams(i) = pow(cons1*dumfns(i,k)/ dumqs(i,k),1./ds)
          dlams(i)=max(dlams(i),lammins)
          dlams(i)=min(dlams(i),lammaxs)
        endif
        if (dumg(i,k) >= qsmall) then
          dlamg(i) = pow(cons2*dumfng(i,k)/ dumg(i,k),1./dg)
          dlamg(i)=max(dlamg(i),lamming)
          dlamg(i)=min(dlamg(i),lammaxg)
        endif
        if (dumc(i,k) >= qsmall) then
          unc(i) =  acn(i,k)*gamma(1.+bc+pgam(i,k))/ (pow(dlamc(i),bc)*gamma(pgam(i,k)+1.))
          umc(i) = acn(i,k)*gamma(4.+bc+pgam(i,k))/  (pow(dlamc(i),bc)*gamma(pgam(i,k)+4.))
        else
          umc(i) = 0.
          unc(i) = 0.
        endif
        if (dumi(i,k) >= qsmall) then
          uni(i) = ain(i,k)*cons27/pow(dlami(i),bi)
          umi(i) = ain(i,k)*cons28/pow(dlami(i),bi)
        else
          umi(i) = 0.
          uni(i) = 0.
        endif
        if (dumr(i,k) >= qsmall) then
          unr(i) = arn(i,k)*cons6/pow(dlamr(i),br)
          umr(i) = arn(i,k)*cons4/pow(dlamr(i),br)
        else
          umr(i) = 0.
          unr(i) = 0.
        endif
        if (dumqs(i,k) >= qsmall) then
          ums(i) = asn(i,k)*cons3/pow(dlams(i),bs)
          uns(i) = asn(i,k)*cons5/pow(dlams(i),bs)
        else
          ums(i) = 0.
          uns(i) = 0.
        endif
        if (dumg(i,k) >= qsmall) then
          umg(i) = agn(i,k)*cons7/pow(dlamg(i),bg)
          ung(i) = agn(i,k)*cons8/pow(dlamg(i),bg)
        else
          umg(i) = 0.
          ung(i) = 0.
        endif
        dum    = pow(rhosu/rho(i,k),0.54)
        ums(i)    = min(ums(i),1.2*dum)
        uns(i)    = min(uns(i),1.2*dum)
        umi(i)    = min(umi(i),1.2*pow(rhosu/rho(i,k),0.35))
        uni(i)    = min(uni(i),1.2*pow(rhosu/rho(i,k),0.35))
        umr(i)    = min(umr(i),9.1*dum)
        unr(i)    = min(unr(i),9.1*dum)
        umg(i)    = min(umg(i),20.*dum)
        ung(i)    = min(ung(i),20.*dum)
        fr(i,k) = umr(i)
        fi(i,k) = umi(i)
        fni(i,k) = uni(i)
        fs(i,k) = ums(i)
        fns(i,k) = uns(i)
        fnr(i,k) = unr(i)
        fc(i,k) = umc(i)
        fnc(i,k) = unc(i)
        fg(i,k) = umg(i)
        fng(i,k) = ung(i)
        if (k <= nz-1) then
          if (fr(i,k) < 1.e-10) fr(i,k) = fr (i,k+1)
          if (fi(i,k) < 1.e-10) fi(i,k) = fi (i,k+1)
          if (fni(i,k) < 1.e-10) fni(i,k) = fni(i,k+1)
          if (fs(i,k) < 1.e-10) fs(i,k) = fs (i,k+1)
          if (fns(i,k) < 1.e-10) fns(i,k) = fns(i,k+1)
          if (fnr(i,k) < 1.e-10) fnr(i,k) = fnr(i,k+1)
          if (fc(i,k) < 1.e-10) fc(i,k) = fc (i,k+1)
          if (fnc(i,k) < 1.e-10) fnc(i,k) = fnc(i,k+1)
          if (fg(i,k) < 1.e-10) fg(i,k) = fg (i,k+1)
          if (fng(i,k) < 1.e-10) fng(i,k) = fng(i,k+1)
        endif ! k le nz-1
        rgvm(i) = max(fr(i,k),fi(i,k),fs(i,k),fc(i,k),fni(i,k),fnr(i,k),fns(i,k),fnc(i,k),fg(i,k),fng(i,k))
        nstep(i) = max(int(rgvm(i)*dt/dzq(i,k)+1.),nstep(i))
        dumr(i,k) = dumr(i,k)*rho(i,k)
        dumi(i,k) = dumi(i,k)*rho(i,k)
        dumfni(i,k) = dumfni(i,k)*rho(i,k)
        dumqs(i,k) = dumqs(i,k)*rho(i,k)
        dumfns(i,k) = dumfns(i,k)*rho(i,k)
        dumfnr(i,k) = dumfnr(i,k)*rho(i,k)
        dumc(i,k) = dumc(i,k)*rho(i,k)
        dumfnc(i,k) = dumfnc(i,k)*rho(i,k)
        dumg(i,k) = dumg(i,k)*rho(i,k)
        dumfng(i,k) = dumfng(i,k)*rho(i,k)
      enddo

      do n = 1,nstep(i)
        do k = 1,nz
          faloutr(i,k) = fr(i,k)*dumr(i,k)
          falouti(i,k) = fi(i,k)*dumi(i,k)
          faloutni(i,k) = fni(i,k)*dumfni(i,k)
          falouts(i,k) = fs(i,k)*dumqs(i,k)
          faloutns(i,k) = fns(i,k)*dumfns(i,k)
          faloutnr(i,k) = fnr(i,k)*dumfnr(i,k)
          faloutc(i,k) = fc(i,k)*dumc(i,k)
          faloutnc(i,k) = fnc(i,k)*dumfnc(i,k)
          faloutg(i,k) = fg(i,k)*dumg(i,k)
          faloutng(i,k) = fng(i,k)*dumfng(i,k)
        enddo
        k        = nz
        faltndr(i)  = faloutr(i,k)/dzq(i,k)
        faltndi(i)  = falouti(i,k)/dzq(i,k)
        faltndni(i) = faloutni(i,k)/dzq(i,k)
        faltnds(i)  = falouts(i,k)/dzq(i,k)
        faltndns(i) = faloutns(i,k)/dzq(i,k)
        faltndnr(i) = faloutnr(i,k)/dzq(i,k)
        faltndc(i)  = faloutc(i,k)/dzq(i,k)
        faltndnc(i) = faloutnc(i,k)/dzq(i,k)
        faltndg(i)  = faloutg(i,k)/dzq(i,k)
        faltndng(i) = faloutng(i,k)/dzq(i,k)
        qrsten (i,k) = qrsten (i,k)-faltndr(i) /nstep(i)/rho(i,k)
        qisten (i,k) = qisten (i,k)-faltndi(i) /nstep(i)/rho(i,k)
        ni3dten(i,k) = ni3dten(i,k)-faltndni(i)/nstep(i)/rho(i,k)
        qnisten(i,k) = qnisten(i,k)-faltnds(i) /nstep(i)/rho(i,k)
        ns3dten(i,k) = ns3dten(i,k)-faltndns(i)/nstep(i)/rho(i,k)
        nr3dten(i,k) = nr3dten(i,k)-faltndnr(i)/nstep(i)/rho(i,k)
        qcsten (i,k) = qcsten (i,k)-faltndc(i) /nstep(i)/rho(i,k)
        nc3dten(i,k) = nc3dten(i,k)-faltndnc(i)/nstep(i)/rho(i,k)
        qgsten (i,k) = qgsten (i,k)-faltndg(i) /nstep(i)/rho(i,k)
        ng3dten(i,k) = ng3dten(i,k)-faltndng(i)/nstep(i)/rho(i,k)
        dumr(i,k) = dumr(i,k)-faltndr(i) *dt/nstep(i)
        dumi(i,k) = dumi(i,k)-faltndi(i) *dt/nstep(i)
        dumfni(i,k) = dumfni(i,k)-faltndni(i)*dt/nstep(i)
        dumqs(i,k) = dumqs(i,k)-faltnds(i) *dt/nstep(i)
        dumfns(i,k) = dumfns(i,k)-faltndns(i)*dt/nstep(i)
        dumfnr(i,k) = dumfnr(i,k)-faltndnr(i)*dt/nstep(i)
        dumc(i,k) = dumc(i,k)-faltndc(i) *dt/nstep(i)
        dumfnc(i,k) = dumfnc(i,k)-faltndnc(i)*dt/nstep(i)
        dumg(i,k) = dumg(i,k)-faltndg(i) *dt/nstep(i)
        dumfng(i,k) = dumfng(i,k)-faltndng(i)*dt/nstep(i)
        do k = nz-1,1,-1
          faltndr(i)  = (faloutr (i,k+1)-faloutr(i,k))/dzq(i,k)
          faltndi(i)  = (falouti (i,k+1)-falouti(i,k))/dzq(i,k)
          faltndni(i) = (faloutni(i,k+1)-faloutni(i,k))/dzq(i,k)
          faltnds(i)  = (falouts (i,k+1)-falouts(i,k))/dzq(i,k)
          faltndns(i) = (faloutns(i,k+1)-faloutns(i,k))/dzq(i,k)
          faltndnr(i) = (faloutnr(i,k+1)-faloutnr(i,k))/dzq(i,k)
          faltndc(i)  = (faloutc (i,k+1)-faloutc(i,k))/dzq(i,k)
          faltndnc(i) = (faloutnc(i,k+1)-faloutnc(i,k))/dzq(i,k)
          faltndg(i)  = (faloutg (i,k+1)-faloutg(i,k))/dzq(i,k)
          faltndng(i) = (faloutng(i,k+1)-faloutng(i,k))/dzq(i,k)
          qrsten (i,k) = qrsten (i,k)+faltndr(i) /nstep(i)/rho(i,k)
          qisten (i,k) = qisten (i,k)+faltndi(i) /nstep(i)/rho(i,k)
          ni3dten(i,k) = ni3dten(i,k)+faltndni(i)/nstep(i)/rho(i,k)
          qnisten(i,k) = qnisten(i,k)+faltnds(i) /nstep(i)/rho(i,k)
          ns3dten(i,k) = ns3dten(i,k)+faltndns(i)/nstep(i)/rho(i,k)
          nr3dten(i,k) = nr3dten(i,k)+faltndnr(i)/nstep(i)/rho(i,k)
          qcsten (i,k) = qcsten (i,k)+faltndc(i) /nstep(i)/rho(i,k)
          nc3dten(i,k) = nc3dten(i,k)+faltndnc(i)/nstep(i)/rho(i,k)
          qgsten (i,k) = qgsten (i,k)+faltndg(i) /nstep(i)/rho(i,k)
          ng3dten(i,k) = ng3dten(i,k)+faltndng(i)/nstep(i)/rho(i,k)
          dumr(i,k) = dumr(i,k)+faltndr(i) *dt/nstep(i)
          dumi(i,k) = dumi(i,k)+faltndi(i) *dt/nstep(i)
          dumfni(i,k) = dumfni(i,k)+faltndni(i)*dt/nstep(i)
          dumqs(i,k) = dumqs(i,k)+faltnds(i) *dt/nstep(i)
          dumfns(i,k) = dumfns(i,k)+faltndns(i)*dt/nstep(i)
          dumfnr(i,k) = dumfnr(i,k)+faltndnr(i)*dt/nstep(i)
          dumc(i,k) = dumc(i,k)+faltndc(i) *dt/nstep(i)
          dumfnc(i,k) = dumfnc(i,k)+faltndnc(i)*dt/nstep(i)
          dumg(i,k) = dumg(i,k)+faltndg(i) *dt/nstep(i)
          dumfng(i,k) = dumfng(i,k)+faltndng(i)*dt/nstep(i)
          csed(i,k)=csed(i,k)+faloutc(i,k)/nstep(i)
          ised(i,k)=ised(i,k)+falouti(i,k)/nstep(i)
          ssed(i,k)=ssed(i,k)+falouts(i,k)/nstep(i)
          gsed(i,k)=gsed(i,k)+faloutg(i,k)/nstep(i)
          rsed(i,k)=rsed(i,k)+faloutr(i,k)/nstep(i)
        enddo
        precrt (i) = precrt (i)+(faloutr(i,1)+faloutc(i,1)+falouts(i,1)+falouti(i,1)+faloutg(i,1))*dt/nstep(i)
        snowrt (i) = snowrt (i)+(falouts(i,1)+falouti(i,1)+faloutg(i,1))*dt/nstep(i)
        snowprt(i) = snowprt(i)+(falouti(i,1)+falouts(i,1))*dt/nstep(i)
        grplprt(i) = grplprt(i)+(faloutg(i,1))*dt/nstep(i)
      enddo ! nstep(i)

      do k=1,nz
        qr3dten (i,k) = qr3dten (i,k) + qrsten (i,k)
        qi3dten (i,k) = qi3dten (i,k) + qisten (i,k)
        qc3dten (i,k) = qc3dten (i,k) + qcsten (i,k)
        qg3dten (i,k) = qg3dten (i,k) + qgsten (i,k)
        qni3dten(i,k) = qni3dten(i,k) + qnisten(i,k)
        if (qi3d(i,k) >= qsmall.and.t3d(i,k) < 273.15.and.lami(i,k) >= 1.e-10) then
          if (1./lami(i,k) >= 2.*dcs) then
            qni3dten(i,k) = qni3dten(i,k)+qi3d(i,k)/dt+ qi3dten(i,k)
            ns3dten(i,k) = ns3dten(i,k)+ni3d(i,k)/dt+   ni3dten(i,k)
            qi3dten(i,k) = -qi3d(i,k)/dt
            ni3dten(i,k) = -ni3d(i,k)/dt
          endif
        endif
        qc3d (i,k) = qc3d (i,k)+qc3dten (i,k)*dt
        qi3d (i,k) = qi3d (i,k)+qi3dten (i,k)*dt
        qni3d(i,k) = qni3d(i,k)+qni3dten(i,k)*dt
        qr3d (i,k) = qr3d (i,k)+qr3dten (i,k)*dt
        nc3d (i,k) = nc3d (i,k)+nc3dten (i,k)*dt
        ni3d (i,k) = ni3d (i,k)+ni3dten (i,k)*dt
        ns3d (i,k) = ns3d (i,k)+ns3dten (i,k)*dt
        nr3d (i,k) = nr3d (i,k)+nr3dten (i,k)*dt
        if (igraup==0) then
          qg3d(i,k) = qg3d(i,k)+qg3dten(i,k)*dt
          ng3d(i,k) = ng3d(i,k)+ng3dten(i,k)*dt
        endif
        t3d (i,k) = t3d (i,k)+t3dten (i,k)*dt
        qv3d(i,k) = qv3d(i,k)+qv3dten(i,k)*dt
        evs(i,k) = min( 0.99*pres(i,k) , polysvp(t3d(i,k),0) )   ! pa
        eis(i,k) = min( 0.99*pres(i,k) , polysvp(t3d(i,k),1) )   ! pa
        if (eis(i,k) > evs(i,k)) eis(i,k) = evs(i,k)
        qvs(i,k) = ep_2*evs(i,k)/(pres(i,k)-evs(i,k))
        qvi(i,k) = ep_2*eis(i,k)/(pres(i,k)-eis(i,k))
        qvqvs(i,k) = qv3d(i,k)/qvs(i,k)
        qvqvsi(i,k) = qv3d(i,k)/qvi(i,k)
        if (qvqvs(i,k) < 0.9) then
          if (qr3d(i,k) < 1.e-8) then
            qv3d(i,k)=qv3d(i,k)+qr3d(i,k)
            t3d (i,k)=t3d (i,k)-qr3d(i,k)*xxlv(i,k)/cpm(i,k)
            qr3d(i,k)=0.
          endif
          if (qc3d(i,k) < 1.e-8) then
            qv3d(i,k)=qv3d(i,k)+qc3d(i,k)
            t3d (i,k)=t3d (i,k)-qc3d(i,k)*xxlv(i,k)/cpm(i,k)
            qc3d(i,k)=0.
          endif
        endif
        if (qvqvsi(i,k) < 0.9) then
          if (qi3d(i,k) < 1.e-8) then
            qv3d(i,k)=qv3d(i,k)+qi3d(i,k)
            t3d (i,k)=t3d (i,k)-qi3d(i,k)*xxls(i,k)/cpm(i,k)
            qi3d(i,k)=0.
          endif
          if (qni3d(i,k) < 1.e-8) then
            qv3d (i,k)=qv3d(i,k)+qni3d(i,k)
            t3d  (i,k)=t3d (i,k)-qni3d(i,k)*xxls(i,k)/cpm(i,k)
            qni3d(i,k)=0.
          endif
          if (qg3d(i,k) < 1.e-8) then
            qv3d(i,k)=qv3d(i,k)+qg3d(i,k)
            t3d (i,k)=t3d (i,k)-qg3d(i,k)*xxls(i,k)/cpm(i,k)
            qg3d(i,k)=0.
          endif
        endif
        if (qc3d(i,k) < qsmall) then
          qc3d(i,k) = 0.
          nc3d(i,k) = 0.
          effc(i,k) = 0.
        endif
        if (qr3d(i,k) < qsmall) then
          qr3d(i,k) = 0.
          nr3d(i,k) = 0.
          effr(i,k) = 0.
        endif
        if (qi3d(i,k) < qsmall) then
          qi3d(i,k) = 0.
          ni3d(i,k) = 0.
          effi(i,k) = 0.
        endif
        if (qni3d(i,k) < qsmall) then
          qni3d(i,k) = 0.
          ns3d (i,k) = 0.
          effs (i,k) = 0.
        endif
        if (qg3d(i,k) < qsmall) then
          qg3d(i,k) = 0.
          ng3d(i,k) = 0.
          effg(i,k) = 0.
        endif
        if (.not. (qc3d(i,k) < qsmall.and.qi3d(i,k) < qsmall.and.qni3d(i,k) < qsmall .and.qr3d(i,k) < qsmall.and.qg3d(i,k) < qsmall)) then
          if (qi3d(i,k) >= qsmall.and.t3d(i,k) >= 273.15) then
            qr3d(i,k) = qr3d(i,k)+qi3d(i,k)
            t3d(i,k) = t3d(i,k)-qi3d(i,k)*xlf(i,k)/cpm(i,k)
            qi3d(i,k) = 0.
            nr3d(i,k) = nr3d(i,k)+ni3d(i,k)
            ni3d(i,k) = 0.
          endif
          if (iliq /= 1) then
            if (t3d(i,k) <= 233.15.and.qc3d(i,k) >= qsmall) then
              qi3d(i,k)=qi3d(i,k)+qc3d(i,k)
              t3d (i,k)=t3d (i,k)+qc3d(i,k)*xlf(i,k)/cpm(i,k)
              qc3d(i,k)=0.
              ni3d(i,k)=ni3d(i,k)+nc3d(i,k)
              nc3d(i,k)=0.
            endif
            if (igraup==0) then
              if (t3d(i,k) <= 233.15.and.qr3d(i,k) >= qsmall) then
                 qg3d(i,k) = qg3d(i,k)+qr3d(i,k)
                 t3d (i,k) = t3d (i,k)+qr3d(i,k)*xlf(i,k)/cpm(i,k)
                 qr3d(i,k) = 0.
                 ng3d(i,k) = ng3d(i,k)+ nr3d(i,k)
                 nr3d(i,k) = 0.
              endif
            else if (igraup==1) then
              if (t3d(i,k) <= 233.15.and.qr3d(i,k) >= qsmall) then
                qni3d(i,k) = qni3d(i,k)+qr3d(i,k)
                t3d  (i,k) = t3d  (i,k)+qr3d(i,k)*xlf(i,k)/cpm(i,k)
                qr3d (i,k) = 0.
                ns3d (i,k) = ns3d (i,k)+nr3d(i,k)
                nr3d (i,k) = 0.
              endif
            endif
          endif
          ni3d(i,k) = max( 0. , ni3d(i,k) )
          ns3d(i,k) = max( 0. , ns3d(i,k) )
          nc3d(i,k) = max( 0. , nc3d(i,k) )
          nr3d(i,k) = max( 0. , nr3d(i,k) )
          ng3d(i,k) = max( 0. , ng3d(i,k) )
          if (qi3d(i,k) >= qsmall) then
            lami(i,k) = pow(cons12*ni3d(i,k)/qi3d(i,k),1./di)
            if (lami(i,k) < lammini) then
              lami(i,k) = lammini
              n0i(i,k) = pow(lami(i,k),4)*qi3d(i,k)/cons12
              ni3d(i,k) = n0i(i,k)/lami(i,k)
            else if (lami(i,k) > lammaxi) then
              lami(i,k) = lammaxi
              n0i(i,k) = pow(lami(i,k),4)*qi3d(i,k)/cons12
              ni3d(i,k) = n0i(i,k)/lami(i,k)
            endif
          endif
          if (qr3d(i,k) >= qsmall) then
            lamr(i,k) = pow(pi*rhow*nr3d(i,k)/qr3d(i,k),1./3.)
            if (lamr(i,k) < lamminr) then
              lamr(i,k) = lamminr
              n0rr(i,k) = pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow)
              nr3d(i,k) = n0rr(i,k)/lamr(i,k)
            else if (lamr(i,k) > lammaxr) then
              lamr(i,k) = lammaxr
              n0rr(i,k) = pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow)
              nr3d(i,k) = n0rr(i,k)/lamr(i,k)
            endif
          endif
          if (qc3d(i,k) >= qsmall) then
            dum = pres(i,k)/(287.15*t3d(i,k))
            pgam(i,k)=0.0005714*(nc3d(i,k)/1.e6*dum)+0.2714
            pgam(i,k)=1./(pow(pgam(i,k),2))-1.
            pgam(i,k)=max(pgam(i,k),2.)
            pgam(i,k)=min(pgam(i,k),10.)
            lamc(i,k) = pow(cons26*nc3d(i,k)*gamma(pgam(i,k)+4.)/(qc3d(i,k)*gamma(pgam(i,k)+1.)),1./3.)
            lammin(i) = (pgam(i,k)+1.)/60.e-6
            lammax(i) = (pgam(i,k)+1.)/1.e-6
            if (lamc(i,k) < lammin(i)) then
              lamc(i,k) = lammin(i)
              nc3d(i,k) = exp(3.*log(lamc(i,k))+log(qc3d(i,k))+log(gamma(pgam(i,k)+1.))-log(gamma(pgam(i,k)+4.)))/cons26
            else if (lamc(i,k) > lammax(i)) then
              lamc(i,k) = lammax(i)
              nc3d(i,k) = exp(3.*log(lamc(i,k))+log(qc3d(i,k))+log(gamma(pgam(i,k)+1.))-log(gamma(pgam(i,k)+4.)))/cons26
            endif
          endif
          if (qni3d(i,k) >= qsmall) then
            lams(i,k) = pow(cons1*ns3d(i,k)/qni3d(i,k),1./ds)
            if (lams(i,k) < lammins) then
              lams(i,k) = lammins
              n0s(i,k) = pow(lams(i,k),4)*qni3d(i,k)/cons1
              ns3d(i,k) = n0s(i,k)/lams(i,k)
            else if (lams(i,k) > lammaxs) then
              lams(i,k) = lammaxs
              n0s(i,k) = pow(lams(i,k),4)*qni3d(i,k)/cons1
              ns3d(i,k) = n0s(i,k)/lams(i,k)
            endif
          endif
          if (qg3d(i,k) >= qsmall) then
            lamg(i,k) = pow(cons2*ng3d(i,k)/qg3d(i,k),1./dg)
            if (lamg(i,k) < lamming) then
              lamg(i,k) = lamming
              n0g(i,k) = pow(lamg(i,k),4)*qg3d(i,k)/cons2
              ng3d(i,k) = n0g(i,k)/lamg(i,k)
            else if (lamg(i,k) > lammaxg) then
              lamg(i,k) = lammaxg
              n0g(i,k) = pow(lamg(i,k),4)*qg3d(i,k)/cons2
              ng3d(i,k) = n0g(i,k)/lamg(i,k)
            endif
          endif
        endif
        if (qi3d(i,k) >= qsmall) then
          effi(i,k) = 3./lami(i,k)/2.*1.e6
        else
          effi(i,k) = 25.
        endif
        if (qni3d(i,k) >= qsmall) then
          effs(i,k) = 3./lams(i,k)/2.*1.e6
        else
          effs(i,k) = 25.
        endif
        if (qr3d(i,k) >= qsmall) then
          effr(i,k) = 3./lamr(i,k)/2.*1.e6
        else
          effr(i,k) = 25.
        endif
        if (qc3d(i,k) >= qsmall) then
          effc(i,k) = gamma(pgam(i,k)+4.)/gamma(pgam(i,k)+3.)/lamc(i,k)/2.*1.e6
        else
          effc(i,k) = 25.
        endif
        if (qg3d(i,k) >= qsmall) then
          effg(i,k) = 3./lamg(i,k)/2.*1.e6
        else
          effg(i,k) = 25.
        endif
        ni3d(i,k) = min( ni3d(i,k) , 0.3e6/rho(i,k) )
        if (iinum==0.and.iact.eq.2) then
          nc3d(i,k) = min( nc3d(i,k) , (nanew1+nanew2)/rho(i,k) )
        endif
        if (iinum==1) then 
          nc3d(i,k) = ndcnst*1.e6/rho(i,k)
        endif
      enddo !!! k loop
    enddo !!! i loop
  end subroutine morr_two_moment_micro



  real function polysvp (t,type)
    implicit none
    real    :: t
    integer :: type
    ! ice
    real :: a0i,a1i,a2i,a3i,a4i,a5i,a6i,a7i,a8i 
    real :: a0,a1,a2,a3,a4,a5,a6,a7,a8 
    real :: dt
    a0 = 6.11239921
    a1 = 0.443987641
    a2 = 0.142986287e-1
    a3 = 0.264847430e-3
    a4 = 0.302950461e-5
    a5 = 0.206739458e-7
    a6 = 0.640689451e-10
    a7 = -0.952447341e-13
    a8 = -0.976195544e-15
    a0i = 6.11147274
    a1i = 0.503160820
    a2i = 0.188439774e-1
    a3i = 0.420895665e-3
    a4i = 0.615021634e-5
    a5i = 0.602588177e-7
    a6i = 0.385852041e-9
    a7i = 0.146898966e-11
    a8i = 0.252751365e-14
    ! ice
    if (type==1) then
      if (t >= 195.8) then
        dt=t-273.15
        polysvp = a0i + dt*(a1i+dt*(a2i+dt*(a3i+dt*(a4i+dt*(a5i+dt*(a6i+dt*(a7i+a8i*dt))))))) 
        polysvp = polysvp*100.
      else
        polysvp = pow(10.,-9.09718*(273.16/t-1.)-3.56654*alog10(273.16/t)+0.876793*(1.-t/273.16)+alog10(6.1071))*100.
      endif
    endif
    ! liquid
    if (type==0) then
      if (t >= 202.0) then
        dt = t-273.15
        polysvp = a0 + dt*(a1+dt*(a2+dt*(a3+dt*(a4+dt*(a5+dt*(a6+dt*(a7+a8*dt)))))))
        polysvp = polysvp*100.
      else
        polysvp = pow(10.,-7.90298*(373.16/t-1.)+5.02808*alog10(373.16/t)-1.3816e-7*(pow(10.,11.344*(1.-t/373.16))-1.)+ &
                  8.1328e-3*(pow(10.,-3.49149*(373.16/t-1.))-1.)+alog10(1013.246))*100.                      
      endif
    endif
  end function polysvp



  real function gamma(x)
    implicit none
    integer :: i,n
    logical :: parity
    real    :: conv,eps,fact,res,sum,x,xbig,xden,xinf,xminin,xnum,y,y1,ysq,z
    real, dimension(7) :: c
    real, dimension(8) :: p
    real, dimension(8) :: q
    xbig   = 35.040e0
    xminin = 1.18e-38
    eps    = 1.19e-7
    xinf   = 3.4e38
    q(1) = -3.08402300119738975254353e+1
    q(2) =  3.15350626979604161529144e+2
    q(3) = -1.01515636749021914166146e+3
    q(4) = -3.10777167157231109440444e+3
    q(5) =  2.25381184209801510330112e+4
    q(6) =  4.75584627752788110767815e+3
    q(7) = -1.34659959864969306392456e+5
    q(8) = -1.15132259675553483497211e+5
    c(1) = -1.910444077728e-03
    c(2) =  8.4171387781295e-04
    c(3) = -5.952379913043012e-04
    c(4) =  7.93650793500350248e-04
    c(5) = -2.777777777777681622553e-03
    c(6) =  8.333333333333333331554247e-02
    c(7) =  5.7083835261e-03
    p(1) = -1.71618513886549492533811e+0
    p(2) =  2.47656508055759199108314e+1
    p(3) = -3.79804256470945635097577e+2
    p(4) =  6.29331155312818442661052e+2
    p(5) =  8.66966202790413211295064e+2
    p(6) = -3.14512729688483675254357e+4
    p(7) = -3.61444134186911729807069e+4
    p(8) =  6.64561438202405440627855e+4
    parity  = .false.
    fact    = 1.
    n       = 0
    y       = x
    if (y <= 0) then
      y   = -x
      y1  = aint(y)
      res = y-y1
      if (res /= 0) then
        if (y1 /= aint(y1*0.5)*2) parity=.true.
        fact = -pi/sin(pi*res)
        y    = y+1.
      else
        res=xinf
        gamma = res
        return
      endif
    endif
    if (y < eps) then
      if (y >= xminin) then
        res=1./y
      else
        res=xinf
        gamma = res
        return
      endif
    else if (y < 12) then
      y1 = y
      if(y < 1.) then
        z = y
        y = y+1.
      else
        n = int(y)-1
        y = y-real(n)
        z = y-1.
      endif
      xnum = 0
      xden = 1.
      do i=1,8
        xnum = (xnum+p(i))*z
        xden = xden*z+q(i)
      enddo
      res = xnum/xden+1.
      if (y1 < y) then
        res = res/y1
      else if (y1 > y) then
        do i=1,n
          res = res*y
          y   = y+1.
        enddo
      endif
    else
      if(y <= xbig)then
        ysq = y*y
        sum = c(7)
        do i=1,6
          sum = sum/ysq+c(i)
        enddo
        sum = sum/y-y+xxx
        sum = sum+(y-0.5)*log(y)
        res = exp(sum)
      else
        res = xinf
        gamma = res
        return
      endif
    endif
    if (parity) res=-res
    if (fact /= 1.) res=fact/res
    gamma = res
  end function gamma



  real function pow_rr(a,b)
    real, intent(in) :: a, b
    pow_rr = a**b
  end function pow_rr
  real function pow_ri(a,b)
    real   , intent(in) :: a
    integer, intent(in) :: b
    pow_ri = a**b
  end function pow_ri


end module module_mp_morr_two_moment


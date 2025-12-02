# Main shared modules for portUrb simulations

For more information on a module, please see the individual module source file (`*.h`) itself and the comments therein. Some are classes that implement an `init` and a custom-named run function name. Some only implement a custom-named run function. Most accept the coupler object and time step size in their run functions, but some accept additional parameters.

Also, check out the `portUrb/experiments/examples` directory `*.cpp` files for examples of how to use the modules below.

* `helpers/`: Contains helper files, mainly for the dynamical core (`dynamics_rk_simpler.h`) for reconstruction (`TransformMatrices.h`), Weighted Essentially Non-Oscillatory reconstruction (`WenoLimiter.h`), and sagemath code to generate code for the reconstruction and WENO source files.
* `Betti_simplified.h`: Implements the `Floating_motions_betti` class to simulate floating turbine motions for the Tension Leg Platform of an NREL 5MW turbine using the equations and constants of "Development of a Control-Oriented Model of Floating Wind Turbines", Betti et al., 2014
* `column_nudging.h`: Implements the `ColumnNudger` class to set a column to nudge the entire coupler object's column toward and then nudge the coupler's average column toward that value according to a specified time scale.
* `dynamics_rk_simpler.h`: Implements the main dynamical core class `Dynamics_Euler_Stratified_WenoFV`, which performs 3-D compressible, non-hydrostatic, moist Euler equations of fluid dynamics without an LES closure or surface fluxes using high-order WENO interpolation for everything except momenta. Also the primary workhorse for inserting immersed boundaries. This also performs transport of mass-weighted tracer fields like water vapor, hydrometeors, SGS TKE, and other tracers registered with the dynamical core. The default order of accuracy is 9th-order. You'll want to have all tracers registered with the Coupler object before calling the dynamical core `init` function, which typically amounts to calling other modules' `init` functions before calling this class's `init` function. You'll want to allocate and initialize data before calling this class's `init` function, and you'll want to perform perturbations on the initial state *after* calling this function's `init` function. This uses a cell-centered upwind Finite-Volume method implemented on perturbations about a hydrostatic background state.
* `edge_sponge.h`: Implements the `EdgeSpone` class, which sets a column to nudge the horizontal edges of the simulation towards, and then has a function to nudge the horizontal edges of the domain toward the set column according to a time scale and domain lengths of nudging for each horizontal edge. 
* `fluctuation_scaling.h`: This will scale the fluctuations about the mean of the specified variables of a coupler object by the specified fraction by a time scale. This is primarily used to change turbulent precursor turbulent intensities to a smaller or higher value when feeding them into a forced simulation.
* `geostrophic_wind_forcing.h`: This implements a function to force the coupler object's horizontal winds with "geostrophic forcing" for atmospheric boundary layer simulations. This is a form of forcing akin to "pressure gradient forcing" except with an ekman spiral arising when used alongside surface friction. One can specify geostrophic forcing directly to a coupler object and then save the column forcing to apply identical forcing using a separate routine to a forced simulation from the direct forcing applied to a precursor coupler object.
* `hydrostasis.h`: This is not used directly during runtime but is, rather, only used during initialization to establish hydrostatic background states for a column using specified variables (either potential temperature profile or a combined specification of temperature and dry mixing ratio of water vapor). 
* `les_closure.h`: 
* `microphysics_kessler.h`
* `microphysics_morr.h`
* `Mp_morr_two_moment.h`
* `precursor_sponge.h`
* `profiles.h`
* `sponge_layer.h`
* `surface_cooling.h`
* `surface_flux.h`
* `surface_heat_flux.h`
* `time_averager.h`
* `TriMesh.h`
* `turbine_actuator_disc.h`
* `turbine_fitch.h`
* `uniform_pg_wind_forcing.h`
* `windmill_actuators_yaw.h`

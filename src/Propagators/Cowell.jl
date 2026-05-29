# src/Propagators/Cowell.jl

"""
    Cowell

Module responsible for the numerical propagation of orbital dynamics using Cowell's method. 

It orchestrates the direct integration of the Cartesian equations of motion, managing the setup 
of initial conditions, optional canonical unit normalization, perturbation forces (such as N-body 
and solar radiation pressure via SPICE ephemerides), and the safe persistence of high-volume 
trajectory data directly to the disk.
"""
module Cowell

using StaticArrays
using Unitful, UnitfulAstro
using RecursiveArrayTools: ArrayPartition
using LinearAlgebra 
using GeometryBasics

using ..Types
using ..Coordinates
using ..Dynamics
using ..Logging
using ..DataHandling
using ..PropagatorUtils
using DifferentialEquations
using SPICE 

export run_simulation

"""
    run_simulation(; ics, perturbation_params, spice_info, tspan, t_vector, propagator_options, output_directory)

Main orchestrator function that executes numerical orbit propagation using Cowell's method (direct integration in Cartesian coordinates).

This function iterates over a vector of initial conditions, setting up and solving the ODE problem for each case. 
It supports optional canonical unit normalization, dynamically handles N-body and solar radiation pressure ephemerides via `SPICE`, 
and uses disk-based persistence (`affect_disk!`) to stream trajectory data directly to CSV files, avoiding excessive RAM consumption during long-duration simulations.

# Keyword Arguments
- `ics::Vector{<:Types.InitialConditions}`: A vector containing one or more sets of initial Keplerian orbital elements.
- `perturbation_params::Types.PerturbationParameters`: Structure containing the physical parameters of the central body and the active perturbing forces (e.g., gravity harmonics, SRP).
- `spice_info::Types.SpiceInformations`: Configuration paths and reference frames for the SPICE ephemeris kernels.
- `tspan::Tuple`: The simulation time span `(t_start, t_end)`. The values must have physical units assigned (e.g., `(0.0u"s", 10.0u"d")`) which are dynamically unwrapped.
- `t_vector::AbstractVector`: A vector of specific time points. If `propagator_options.saveat` is true, the solver will specifically evaluate and save the solution at these points.
- `propagator_options::Types.PropagatorOptions`: Structure defining the numerical solver (e.g., `Vern7()`), absolute/relative tolerances, and normalization flags.
- `output_directory::String`: Directory path where the generated `.csv` files and logs will be saved. Defaults to `../../output`.

# Returns
- `Vector{Types.SimulationResult}`: A vector containing the fully populated simulation result object for each executed initial condition.
"""
function run_simulation(; 
    ics::Vector{<:Types.InitialConditions}, 
    perturbation_params::Types.PerturbationParameters,
    spice_info::Types.SpiceInformations = SpiceInformations(), # optional
    tspan::Tuple,
    t_vector::AbstractVector,
    propagator_options::Types.PropagatorOptions,
    output_directory::String = joinpath(pwd(), "output")
    )

    all_results = Types.SimulationResult[]
    sanitized_params = PropagatorUtils.sanitize_parameters(perturbation_params)
    Logging.log_simulation_setup(sanitized_params, spice_info)

    for (ic_index, ic) in enumerate(ics)

        # poincare helper call
        cb_poincare, p_data = PropagatorUtils.setup_poincare_callback(propagator_options, propagator_options.propagator)
        callback_to_use = cb_poincare
       
        a0_unit = ic.a0; e0_unit = ic.e0; i0_unit = ic.i0; h0_unit = ic.h0; g0_unit = ic.g0; f0_unit = ic.f0;
        
        @info "Starting Simulation for orbit: $ic_index"
        @info "Initial Conditions" a0=a0_unit e0=e0_unit i0=i0_unit h0=h0_unit g0=g0_unit f0=f0_unit
        
        # unwrap the units using the new variables. a0, e0, etc.
        a0 = ustrip(u"km", a0_unit); e0 = e0_unit; i0 = ustrip(u"rad", i0_unit);
        h0 = ustrip(u"rad", h0_unit); g0 = ustrip(u"rad", g0_unit); f0 = ustrip(u"rad", f0_unit);
        
        mu = ustrip(u"km^3/s^2", sanitized_params.mu); R = ustrip(u"km", sanitized_params.R);
        alpha = ustrip(u"m^2/kg", sanitized_params.alpha);
 
        function _integrate_orbit(
            a0::Float64, e0::Float64, i0::Float64, h0::Float64, g0::Float64, f0::Float64, mu::Float64, 
            tspan::Tuple, t_vector::AbstractVector, propagator_options::Types.PropagatorOptions, poincare_callback_var::Any
            )
            @info "Integrating the orbit..."

            # declares the function outside the scope to be the entire function
            local u0_solver, mu_solver, R_solver, omega_rot_solver, tspan_solver, t_vec_solver, units, n_bodies_solver, alpha_solver, dist_scale

            if propagator_options.canonical_unit_normalization

                # create an object using the canonical units.
                units = Coordinates.canonical_units(mu, R)
                dist_scale = units.DU
                @info "Canonical units: DU=$(units.DU) km, TU=$(units.TU) s, VU=$(units.VU) km/s"

                r0, v0 = Coordinates.orbital_elements_to_state_vectors(a0, e0, i0, h0, g0, f0, mu)
                u0_solver = Coordinates.normalize_state([r0; v0], units)

                mu_scale = (units.TU^2 / units.DU^3)
                
                # acceleration factor (srp): L/T^2 -> making it dimensionless requires multiplying by T^2/L
                acc_scale = (units.TU^2 / units.DU)

                # normalization of rotation
                omega_rot_solver = ustrip(u"rad/s", sanitized_params.omega_rot) * units.TU

                # normalizes the mu of each body in the list and removes the vector to be of the abstract type that the struct expects.
                # using positional arguments in the order: name, mu, spice_id (struct PerturbingBody)
                n_bodies_solver = Types.PerturbingBody[typeof(b)(b.name, b.mu * mu_scale, b.spice_id) for b in sanitized_params.n_bodies]
                
                # normalizes alpha (which makes up the acceleration term of the SRP)
                alpha_solver = alpha * acc_scale

                # mu_adim = mu_physical * (TU^3 / DU^3)
                mu_solver = mu * mu_scale
                
                # R_adim = R_physical / DU
                R_solver  = R / units.DU

                tspan_solver = tspan ./ units.TU
                t_vec_solver = t_vector ./ units.TU               

            else 
                r0, v0 = Coordinates.orbital_elements_to_state_vectors(a0, e0, i0, h0, g0, f0, mu)
                u0_solver = SVector(r0..., v0...)

                R_solver = R
                mu_solver = mu

                tspan_solver = tspan
                t_vec_solver = t_vector

                omega_rot_solver = ustrip(u"rad/s", sanitized_params.omega_rot)

                dist_scale = 1.0
                n_bodies_solver = sanitized_params.n_bodies
                alpha_solver = alpha

            end

            p_solver = Types.PerturbationParameters(
                mu=mu_solver, R=R_solver, omega_rot=omega_rot_solver, j2=sanitized_params.j2, j3=sanitized_params.j3, j4=sanitized_params.j4,
                j5=sanitized_params.j5, j6=sanitized_params.j6, j7=sanitized_params.j7, j8=sanitized_params.j8, 
                j9=sanitized_params.j9, j10=sanitized_params.j10, j11=sanitized_params.j11, j12=sanitized_params.j12,
                j13=sanitized_params.j13, j14=sanitized_params.j14, j15=sanitized_params.j15, j16=sanitized_params.j16,
                j17=sanitized_params.j17, j18=sanitized_params.j18,
                c22=sanitized_params.c22, c31=sanitized_params.c31, c32=sanitized_params.c32, c33=sanitized_params.c33,
                c42=sanitized_params.c42, c44=sanitized_params.c44,
                s22=sanitized_params.s22, s31=sanitized_params.s31, s32=sanitized_params.s32, s33=sanitized_params.s33,
                s42=sanitized_params.s42, s44=sanitized_params.s44,
                alpha=alpha_solver, cr=sanitized_params.cr,
                shadow_in_srp=sanitized_params.shadow_in_srp,
                n_bodies=n_bodies_solver
            )
            
            perturbation_func = Dynamics.set_perturbation(p_solver, spice_info, t_vec_solver, t_vector; dist_scale=dist_scale)
            p = Types.SimulationParameters(p_solver, perturbation_func)

            local prob
            if propagator_options.second_order
                # for second order, u0_solver[1:3] is position and [4:6] is velocity.
                prob = SecondOrderODEProblem(
                    Dynamics.cowell_equations_2nd, 
                    SVector(u0_solver[4:6]...), 
                    SVector(u0_solver[1:3]...), 
                    tspan_solver, p
                )
                @debug "Integrator configured" type=typeof(propagator_options.integrator) order=2

            else
                # first order
                prob = ODEProblem(Dynamics.cowell_equations, SVector(u0_solver...), tspan_solver, p)
                @debug "Integrator configured" type=typeof(propagator_options.integrator) order=1
            end

            solver_opts = Dict{Symbol, Any}(
                :reltol => propagator_options.reltol,
                :abstol => propagator_options.abstol,
                :maxiters => propagator_options.maxiters,
            )
            
            # add `saveat` if it is enabled (true) in `run_simulation`.
            if propagator_options.saveat==true
                solver_opts[:saveat] = t_vec_solver
            end

            # add 'dt' if it is a fixed step
            if !isnothing(propagator_options.dt)
                dt_s = propagator_options.dt isa Unitful.Time ? 
                    ustrip(u"s", propagator_options.dt) : # converts the entry ("minute", "hr", ...) to seconds
                    propagator_options.dt
                dt_solver = propagator_options.canonical_unit_normalization ? 
                            dt_s / units.TU : 
                            dt_s
                solver_opts[:dt] = dt_solver
                propagator_options.second_order && (solver_opts[:adaptive] = false)
            end
  
            # suggestion from Professor Rafael Sfair (FEG) - save the data as the integrator progresses and avoid unnecessarily occupying RAM.
            # 1. it defines the path and opens the file, going up two levels to reach the root and find 'output'.
            file_path = joinpath(output_directory, "orbit_$(ic_index).csv")
            mkpath(output_directory)
            io = open(file_path, "w")
            
            # write the heading
            println(io, "t,rx,ry,rz,vx,vy,vz")

            # 2. defines the function that the integrator will call at each step.
            # 'int' is the integrator object that contains the current state (u) and time (t)
            affect_disk! = (int) -> begin
                u = int.u
                t = int.t
                # order-based extraction (2nd order: [v, r] | 1st order: [r, v])
                #r, v = propagator_options.second_order ? (u[4:6], u[1:3]) : (u[1:3], u[4:6])
                r, v = propagator_options.second_order ? (u.x[2], u.x[1]) : (u[1:3], u[4:6])

                # standardized writing: t, rx, ry, rz, vx, vy, vz
                println(io, "$t,$(r[1]),$(r[2]),$(r[3]),$(v[1]),$(v[2]),$(v[3])")

            end

            # 3. creates the callback (condition 'true' means execute at every step)
            # save_positions=(false, false) so that Poincare events don't create extra points in the main trajectory 
            # (they are already saved separately in p_data)
            cb_disk = DiscreteCallback((u, t, int) -> true, affect_disk!; save_positions=(false, false))        

            # 4. it matches the Poincare callback. 
            # Julia uses CallbackSet to run multiple callbacks simultaneously.
            full_cb = isnothing(poincare_callback_var) ? cb_disk : CallbackSet(cb_disk, poincare_callback_var)

            # 5. for ram: save_everystep = false
            # this prevents Julia from storing the millions of points in the vector. 'sol.u'
            # since i'm already saving to a file, I don't need them in memory.

            local sol # necessary because the sun is going to enter a block by @elapid

            integration_time = @elapsed begin
                sol = solve(
                    prob, propagator_options.integrator; 
                    callback = full_cb, 
                    # if saveat is true, save_everystep is false (Clean Dataset)
                    # if saveat is false, then save_everystep is true (Poincaré/Integrator Steps)
                    save_everystep = !propagator_options.saveat,
                    solver_opts...
                )
            end

            # structured log showing the time and number of steps taken.
            @info "Integration completed successfully" time_sec=round(integration_time, digits=3) n_steps=length(sol.t)

            # 6. close the file after the simulation
            close(io)

            return sol, (propagator_options.canonical_unit_normalization ? units : nothing)
        end

        # function call to obtain the solution
        sol, units_used = _integrate_orbit(a0, e0, i0, h0, g0, f0, mu, tspan, t_vector, propagator_options, callback_to_use)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------
        
        # packs everything into the SimulationResult struct 
        result = Types.SimulationResult(
            solution = sol,
            units = units_used,
            initial_conditions = ic,
            parameters = sanitized_params,
            propagator = propagator_options.propagator,
            equation_type = nothing, # Cowell does not have a "symbolic equation" associated with it
            poincare_raw = p_data[:raw_states]
        )
        push!(all_results, result)

    end # end loop for
    
    return all_results
end

end # end of module
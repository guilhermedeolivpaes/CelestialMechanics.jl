# src/Propagators/Hamiltonian.jl


"""
    Hamiltonian

Module responsible for the numerical propagation of orbital dynamics using Hamilton canonical equations. 

It operates strictly in the phase space of Delaunay action-angle variables, enabling the direct numerical integration of average or osculating 
Hamiltonians symbolically derived in computer algebra systems.

"""
module Hamiltonian


using ..Types
using ..Coordinates
using ..ReadEquations
using ..PropagatorUtils
using Unitful, UnitfulAstro
using DifferentialEquations
using ForwardDiff
using StaticArrays
using RecursiveArrayTools: ArrayPartition
using DataFrames
using LinearAlgebra 

export run_hamiltonian_simulation

function _hamiltonian_to_ode(H::Function)
    function hamiltonian_rhs!(du, u, p, t)
        grad = ForwardDiff.gradient(x -> H(x, p, t), SVector{6}(u))
        @inbounds begin
            du[1] = -grad[4]
            du[2] = -grad[5]
            du[3] = -grad[6]
            du[4] =  grad[1]
            du[5] =  grad[2]
            du[6] =  grad[3]
        end
        return nothing
    end
    return hamiltonian_rhs!
end

"""
    run_hamiltonian_simulation(; ics, equation_func, perturbation_params, tspan, t_vector, propagator_options, use_hamiltonian=true)

Executes the numerical integration of an orbital system using Hamilton differential equations.

This function maps the initial planetary conditions (Keplerian elements and mean anomaly) into Delaunay 
momenta (L, G, H) and coordinates (l, g, h). It ensures type stability for the numerical solver by converting 
uninitialized physical parameters into zero values.

When `use_hamiltonian = true` (default), `equation_func` is interpreted as the scalar Hamiltonian `H(u, p, t)` 
with `u = [L, G, H, l, g, h]`, and Hamilton's canonical equations are derived automatically via `ForwardDiff`. 
When `use_hamiltonian = false`, `equation_func` is used directly as the pre-derived in-place ODE `f!(du, u, p, t)`.

# Keyword Arguments
- `ics::Vector{<:Types.InitialPlanetaryConditions}`: A vector containing the initial conditions defined by classical orbital elements and the mean anomaly.
- `equation_func::Function`: Either the scalar Hamiltonian `H(u, p, t)` (when `use_hamiltonian=true`) or the pre-derived in-place dynamics function `f!(du, u, p, t)` (when `use_hamiltonian=false`).
- `perturbation_params::Types.PhysicalParams`: Structure holding the physical parameters (gravity harmonics, third body data, etc.) required to evaluate the Hamiltonian or equations.
- `tspan::Tuple`: The simulation time span (t_start, t_end).
- `t_vector::AbstractVector`: A vector of specific time points. If `saveat` is enabled in the propagator options, the solver evaluates the solution exactly at these intervals.
- `propagator_options::Types.PropagatorOptions`: Structure defining the numerical solver (e.g., `Vern9()`), tolerances, and event callbacks (such as Poincare sections).
- `use_hamiltonian::Bool=true`: If `true`, derives Hamilton's equations from `equation_func(u, p, t)` via automatic differentiation; if `false`, uses `equation_func(du, u, p, t)` directly.

# Returns
- `Vector{Types.SimulationResult}`: A vector containing the complete simulation result object for each evaluated initial condition, including raw Poincare map data if requested.
"""
function run_hamiltonian_simulation(; 
    ics::Vector{<:Types.InitialPlanetaryConditions}, 
    equation_func::Function,           
    perturbation_params::Types.PhysicalParams, 
    tspan::Tuple,
    t_vector::AbstractVector,
    propagator_options::Types.PropagatorOptions,
    use_hamiltonian::Bool = true,
    coordinates::Symbol = :delaunay,
    analytic_ephemeris::Union{Nothing, Types.AnalyticEphemeris} = nothing
    )

    all_results = Types.SimulationResult[]

    coordinates in (:delaunay, :poincare) || throw(ArgumentError(
        "Unsupported coordinates ':$coordinates'. Expected :delaunay or :poincare."
    ))

    ode_func = use_hamiltonian ? _hamiltonian_to_ode(equation_func) : equation_func

    # Delaunay phase space supports apsidal and nodal sections.
    # :equatorial_crossing / :stroboscopic are Cartesian-only (they read
    # int.p.perturb_params and treat u[3] as z) -> reject them early.
    if coordinates == :delaunay
        for section in propagator_options.poincare_sections
            section in (:periapsis_map, :nodal_section, :apsidal_section) || error(
                "Hamiltonian (Delaunay) propagator supports :periapsis_map, :nodal_section, and :apsidal_section; " *
                "section; ':$section' is Cartesian-only. Use the Cowell propagator for it."
            )
        end
    elseif propagator_options.poincare_callback
        throw(ArgumentError("Poincare section callbacks are currently supported only for coordinates=:delaunay."))
    end

    # 1. prepares parameters: replaces Dict with NamedTuple (NECESSARY FOR THE SOLVER)
    # transforms the PhysicalParams struct into a named tuple of Float64
    # if the field is nothing, it becomes 0.0, as this ensures type stability.
    p_val = NamedTuple{fieldnames(typeof(perturbation_params))}(
        map(fieldnames(typeof(perturbation_params))) do f
            val = getfield(perturbation_params, f)
            isnothing(val) ? 0.0 : Float64(val)
        end
    )

    eq_type = coordinates == :poincare ? Types.PoincareEquations() : Types.HamiltonEquations()

    for (ic_index, ic) in enumerate(ics)
        println("--- Hamiltonian Simulation ($(coordinates)): Orbit $ic_index ---")
        
        # 2. extraction of initial conditions
        mu_val = p_val.mu 
        
        a0 = ustrip(u"km", ic.a0)
        e0 = ic.e0
        i0 = ustrip(u"rad", ic.i0)
        h0 = ustrip(u"rad", ic.h0) 
        g0 = ustrip(u"rad", ic.g0) 
        l0 = ustrip(u"rad", ic.l0) 

        # 3. calculation of canonical moments (L, G, H)
        L0 = sqrt(mu_val * a0)
        G0 = L0 * sqrt(max(0.0, 1.0 - e0^2))
        H0 = G0 * cos(i0)

        if coordinates == :poincare
            r2_0 = sqrt(max(0.0, 2.0 * (L0 - G0)))
            r3_0 = sqrt(max(0.0, 2.0 * (G0 - H0)))
            u0_state = [
                L0,
                r2_0 * cos(g0 + h0),
                r3_0 * cos(h0),
                l0 + g0 + h0,
                -r2_0 * sin(g0 + h0),
                -r3_0 * sin(h0),
            ]
            all_p_data = Dict{Symbol, Dict}()
            cb_all = nothing
        else
            # initial state: [L, G, H, l, g, h]
            u0_state = [L0, G0, H0, l0, g0, h0]
            cb_poincare, all_p_data = PropagatorUtils.setup_all_poincare_callbacks(
                propagator_options, Types.HamiltonEquations()
            )
            cb_collision, collision_data =
                PropagatorUtils.setup_delaunay_collision_callback(propagator_options)
            all_p_data[:collision] = collision_data
            callbacks = filter(!isnothing, (cb_poincare, cb_collision))
            cb_all = isempty(callbacks) ? nothing : CallbackSet(callbacks...)
        end
        
        prob = ODEProblem(ode_func, u0_state, tspan, p_val)
        propagator_options.section_dtmax > 0 || throw(ArgumentError(
            "section_dtmax must be positive"))
        
        solver_opts = Dict{Symbol, Any}(
            :reltol => propagator_options.reltol,
            :abstol => propagator_options.abstol,
            :maxiters => propagator_options.maxiters,
            :saveat => propagator_options.saveat ? t_vector : Float64[],
            :progress => true,
            :dtmax => propagator_options.section_dtmax,
            :callback => cb_all
        )

        if !isnothing(propagator_options.dt)
            dt_s = propagator_options.dt isa Unitful.Time ? 
                ustrip(u"s", propagator_options.dt) : 
                Float64(propagator_options.dt)
            solver_opts[:dt] = dt_s
        end
        
        sol = solve(prob, propagator_options.integrator; solver_opts...)
               
        p_data_periapsis = get(all_p_data, :periapsis_map, PropagatorUtils._new_p_data())

        result = Types.SimulationResult(
            solution = sol,
            elements = nothing,
            initial_conditions = ic,
            parameters = perturbation_params,
            propagator = propagator_options.propagator,
            equation_type = eq_type,
            poincare_raw = p_data_periapsis[:raw_states],
            poincare_all = all_p_data,
        )

        push!(all_results, result)        
    end # end for
    
    return all_results
end

end # end of module

# src/Propagators/CR3BP.jl
module CR3BP

using StaticArrays
using LinearAlgebra
using DifferentialEquations
using ..Types
using ..DataHandling
using ..PropagatorUtils
using ..Dynamics

export run_cr3bp_simulation

"""
    run_cr3bp_simulation(; u0::AbstractVector, mu::Float64, tspan::Tuple, propagator_options::Types.PropagatorOptions)

configures and executes the numerical integration for the circular restricted three-body problem (cr3bp).

this function sets up the ordinary differential equation (ode) problem utilizing static arrays for performance. it dynamically injects event callbacks, such as poincare sections, based on the provided configuration, and computes the trajectory using the specified solver settings.

# keyword arguments
- `u0::AbstractVector`: the initial state vector of the system containing dimensionless positions and velocities [x, y, z, vx, vy, vz].
- `mu::Float64`: the dimensionless mass parameter (mass ratio) of the primary bodies.
- `tspan::Tuple`: a tuple specifying the start and end times for the numerical integration.
- `propagator_options::Types.PropagatorOptions`: configuration structure containing the integrator choice, solver tolerances, maximum iterations, and event settings.

# returns
- `Types.SimulationResult`: a structured object containing the numerical solution, system parameters, the propagator dispatch type, and raw poincare section crossover states.
"""
function run_cr3bp_simulation(; u0::AbstractVector, mu::Float64, tspan::Tuple, propagator_options::Types.PropagatorOptions)
    
    p = Types.CR3BPParameters(mu)
    u0_svec = SVector{6, Float64}(u0)
    
    cb_poincare, p_data = PropagatorUtils.setup_poincare_callback(propagator_options, Types.CR3BPPropagator())
    
    prob = ODEProblem(Dynamics.cr3bp_equations, u0_svec, tspan, p)
    
    solver_opts = Dict{Symbol, Any}(
        :reltol => propagator_options.reltol,
        :abstol => propagator_options.abstol,
        :maxiters => propagator_options.maxiters,
        :callback => cb_poincare 
    )
    
    propagator_options.saveat && (solver_opts[:saveat] = collect(range(tspan[1], tspan[2]; length=2000)))

    sol = solve(prob, propagator_options.integrator; solver_opts...)
    
    return Types.SimulationResult(
        solution = sol,
        parameters = p,
        propagator = Types.CR3BPPropagator(),
        poincare_raw = p_data[:raw_states] 
    )
end

end # module
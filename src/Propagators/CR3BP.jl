# src/Propagators/CR3BP.jl
module CR3BP

using StaticArrays
using LinearAlgebra
using DifferentialEquations
using ..Types
using ..DataHandling
using ..PropagatorUtils

export run_cr3bp_simulation, jacobi_constant

"""
    cr3bp_equations(u::SVector{6, T}, p::Types.CR3BPParameters{T}, t::T) where {T}

evaluates the equations of motion for the circular restricted three-body problem (cr3bp) in a synodic (rotating) reference frame.

this function computes the derivatives of the state vector, including velocity and acceleration components, by incorporating the gravitational pull of the two primary bodies and the pseudo-forces (coriolis and centrifugal) inherent to the rotating frame. it is optimized using static arrays for high performance numerical integration.

# arguments
- `u::SVector{6, T}`: the current state vector of the third body, containing dimensionless positions and velocities [x, y, z, vx, vy, vz].
- `p::Types.CR3BPParameters{T}`: the parameter structure containing the mass ratio mu of the system.
- `t::T`: the current dimensionless time.

# returns
- `SVector{6, T}`: the derivative of the state vector [vx, vy, vz, ax, ay, az].
"""
function cr3bp_equations(u::SVector{6, T}, p::Types.CR3BPParameters{T}, t::T) where {T}
    x, y, z, vx, vy, vz = u[1], u[2], u[3], u[4], u[5], u[6]
    mu = p.mu

    # distances to primary bodies
    r1 = sqrt((x + mu)^2 + y^2 + z^2)
    r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2)

    # accelerations
    ax = 2*vy + x - ((1 - mu)*(x + mu))/(r1^3) - (mu*(x - 1 + mu))/(r2^3)
    ay = -2*vx + y - ((1 - mu)*y)/(r1^3) - (mu*y)/(r2^3)
    az = -((1 - mu)*z)/(r1^3) - (mu*z)/(r2^3)

    return SVector{6, T}(vx, vy, vz, ax, ay, az)
end

"""
    jacobi_constant(u::AbstractVector{T}, mu::T) where {T}

calculates the jacobi constant for a given state vector in the circular restricted three-body problem (cr3bp).

this constant represents the only known integral of motion in the cr3bp system, serving as an energy-like quantity evaluated in the synodic (rotating) reference frame. it is frequently used to validate the accuracy of numerical integration.

# arguments
- `u::AbstractVector{T}`: the state vector of the third body, containing dimensionless positions and velocities [x, y, z, vx, vy, vz].
- `mu::T`: the dimensionless mass parameter (mass ratio) of the two primary bodies.

# returns
- `T`: the calculated jacobi constant value.
"""
function jacobi_constant(u::AbstractVector{T}, mu::T) where {T}
    x, y, z, vx, vy, vz = u[1], u[2], u[3], u[4], u[5], u[6]
    r1 = sqrt((x + mu)^2 + y^2 + z^2)
    r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2)
    
    v2 = vx^2 + vy^2 + vz^2
    return (x^2 + y^2) + 2*(1 - mu)/r1 + 2*mu/r2 - v2
end

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
    
    prob = ODEProblem(cr3bp_equations, u0_svec, tspan, p)
    
    solver_opts = Dict{Symbol, Any}(
        :reltol => propagator_options.reltol,
        :abstol => propagator_options.abstol,
        :maxiters => propagator_options.maxiters,
        :callback => cb_poincare # 2. Injeta o callback no solver
    )
    
    propagator_options.saveat && (solver_opts[:saveat] = collect(range(tspan[1], tspan[2]; length=2000)))

    sol = solve(prob, propagator_options.integrator; solver_opts...)
    
    return Types.SimulationResult(
        solution = sol,
        parameters = p,
        propagator = Types.CR3BPPropagator(),
        poincare_raw = p_data[:raw_states] # 3. Salva os pontos exatos do cruzamento!
    )
end

end # module
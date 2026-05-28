# src/Propagators/NBodyIntegrator.jl

"""
    NBodyIntegrator

Self-consistent N-body integrator: all N particles are propagated
simultaneously, each acting as a source of gravity for all others.

# Design choices

| Concern | Decision |
|---|---|
| Equations of motion | Two canonical forms: first-order for explicit integrators, second-order (acceleration-only) for symplectic integrators |
| DiffEq interface | `ODEProblem` (explicit) or `SecondOrderODEProblem` (symplectic) |
| State layout (1st order) | flat `Vector{Float64}` of length 6N: `[r₁,v₁,r₂,v₂,…,rₙ,vₙ]` |
| State layout (2nd order) | two flat `Vector{Float64}` of length 3N each: positions `r` and velocities `v` |
| Performance | in-place (`!`) right-hand-side functions; `@inbounds`; inner loop O(N²) |
| Disk streaming | same `affect_disk!` / `DiscreteCallback` pattern as `Cowell.jl` |

# Recommended integrators (pass as `propagator_options.integrator`)

# Symplectic (use `use_symplectic = true`)
| Julia symbol | Order | Notes |
|---|---|---|
| `VerletLeapfrog()` | 2 | Störmer–Verlet, fastest, minimal drift |
| `Ruth3()` | 3 | Ruth's 3-stage method |
| `McAte4()` | 4 | McAte 4-stage |
| `KahanLi6()` | 6 | Kahan–Li 6-stage |
| `KahanLi8()` | 8 | Kahan–Li 8-stage, recommended for long integrations |
| `SofSpa10()` | 10 | Sofroniou & Spaletta |

# Explicit / adaptive (use `use_symplectic = false`)
| Julia symbol | Order | Notes |
|---|---|---|
| `Vern7()` | 7 | Fast, low memory, good for moderate accuracy |
| `Vern9()` | 9 | High accuracy, good for perturbed orbits |
| `DOP853()` | 8 | Dormand–Prince 8(5,3), widely tested |
| `TsitPap8()` | 8 | Tsitouras–Papakostas, competitive with DOP853 |

> Symplectic integrators require a **fixed time step** (`propagator_options.dt ≠ nothing`)
> and conserve the symplectic structure (hence energy over long periods).
> Explicit integrators are adaptive but do not conserve the symplectic structure.
"""
module NBodyIntegrator

using StaticArrays
using LinearAlgebra
using RecursiveArrayTools: ArrayPartition
using DifferentialEquations
using Unitful
using Printf

using ..Types
using ..Coordinates
using ..DataHandling
using ..Dynamics

export run_nbody_simulation


# ──────────────────────────────────────────────────────────────────────────────
# 1.  ENERGY & MOMENTUM DIAGNOSTICS  (optional, post-integration)
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_energy(u_flat, bodies) → Float64

Total mechanical energy of the N-body system from a flat first-order state vector.
Useful for checking energy conservation (or drift) across the integration.

Returns E = KE + PE where KE = ½Σᵢmᵢvᵢ² and PE = −Σᵢ<ⱼ mᵢmⱼ/rᵢⱼ
(note: we use μᵢ = G mᵢ, so all masses are in km³ s⁻² units).

# Arguments
- `u_flat::AbstractVector{Float64}` : 6N state vector `[r₁,v₁,…]`.
- `bodies::Vector{NBodyParticle}`   : Particle list with `.mu` values.
"""
function compute_energy(u_flat::AbstractVector{Float64},
                        bodies::Vector{Types.NBodyParticle})
    N = length(bodies)
    KE = 0.0
    PE = 0.0

    # Gravitational constant G is embedded in μ = G M; to compute
    # KE = ½ M v² we need mass M = μ/G.  We use G = 6.674e-20 km³ kg⁻¹ s⁻².
    G = 6.674e-20  # km³ kg⁻¹ s⁻²

    for i in 1:N
        base_ri = 6*(i-1)
        vx = u_flat[base_ri+4]; vy = u_flat[base_ri+5]; vz = u_flat[base_ri+6]
        mi = bodies[i].mu / G
        KE += 0.5 * mi * (vx^2 + vy^2 + vz^2)
    end

    for i in 1:N, j in (i+1):N
        base_i = 6*(i-1); base_j = 6*(j-1)
        dx = u_flat[base_j+1] - u_flat[base_i+1]
        dy = u_flat[base_j+2] - u_flat[base_i+2]
        dz = u_flat[base_j+3] - u_flat[base_i+3]
        rij = sqrt(dx^2 + dy^2 + dz^2)
        mi = bodies[i].mu / G
        mj = bodies[j].mu / G
        PE -= G * mi * mj / rij
    end

    return KE + PE
end

"""
    compute_momentum(u_flat, bodies) → SVector{3,Float64}

Total linear momentum of the N-body system (should be conserved).
"""
function compute_momentum(u_flat::AbstractVector{Float64},
                          bodies::Vector{Types.NBodyParticle})
    G  = 6.674e-20
    px = 0.0; py = 0.0; pz = 0.0
    for (i, b) in enumerate(bodies)
        base = 6*(i-1)
        m = b.mu / G
        px += m * u_flat[base+4]
        py += m * u_flat[base+5]
        pz += m * u_flat[base+6]
    end
    return SVector{3,Float64}(px, py, pz)
end


# ──────────────────────────────────────────────────────────────────────────────
# 4.  COLLISION DETECTION CALLBACK
# ──────────────────────────────────────────────────────────────────────────────

"""
    make_collision_callback(bodies; terminate)

Returns a `VectorContinuousCallback` that triggers when any two bodies come
within the sum of their radii (R₁ + R₂).

# Arguments
- `bodies`      : Particle list (radii in `.R` fields).
- `terminate`   : If `true` (default) the integrator halts at first contact.
                  If `false`, a warning is logged and integration continues.
"""
function make_collision_callback(bodies::Vector{Types.NBodyParticle};
                                 terminate::Bool = true)
    N = length(bodies)
    # Build all unique pairs (i,j) with i < j
    pairs = Tuple{Int,Int}[]
    for i in 1:N, j in (i+1):N
        push!(pairs, (i, j))
    end
    n_pairs = length(pairs)

    # condition: distance minus sum-of-radii (root at contact)
    function condition!(out, u, t, int)
        for (k, (i, j)) in enumerate(pairs)
            bi = 6*(i-1); bj = 6*(j-1)
            dx = u[bj+1] - u[bi+1]
            dy = u[bj+2] - u[bi+2]
            dz = u[bj+3] - u[bi+3]
            d  = sqrt(dx^2 + dy^2 + dz^2)
            Rij = int.p.bodies[i].R + int.p.bodies[j].R
            out[k] = d - Rij
        end
    end

    function affect!(int, idx)
        i, j = pairs[idx]
        @warn "Collision detected between body $i ($(int.p.bodies[i].name)) " *
              "and body $j ($(int.p.bodies[j].name)) at t = $(int.t)"
        terminate && terminate!(int)
    end

    return VectorContinuousCallback(condition!, affect!, n_pairs)
end


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MAIN SIMULATION FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

"""
    run_nbody_simulation(; ic, tspan, propagator_options,
                           use_symplectic, detect_collisions, output_directory)

Integrate an N-body system forward in time.

# Keyword Arguments
- `ic::Types.NBodySystemIC` :
      Initial conditions (positions, velocities, μ values) for all N bodies.
- `tspan::Tuple` :
      Time span as plain `Float64` **seconds** or as a `Tuple` of `Unitful`
      quantities, e.g. `(0.0u"s", 365.25u"d")`.
- `propagator_options::Types.PropagatorOptions` :
      Solver algorithm and tolerances.
      • Symplectic: set `use_symplectic=true` and `dt ≠ nothing`.
      • Explicit:   set `use_symplectic=false`; adaptive stepping works.
- `use_symplectic::Bool = false` :
      When `true` a `SecondOrderODEProblem` is built and the integrator must
      be a symplectic method (e.g. `KahanLi8()`).
      When `false` an `ODEProblem` is built and any explicit method works
      (e.g. `Vern9()`).
- `detect_collisions::Bool = false` :
      When `true`, a `VectorContinuousCallback` is added that halts integration
      upon physical contact (requires non-zero `.R` values in the particles).
- `output_directory::String` :
      Directory for the CSV output file `nbody_simulation.csv`.

# Returns
- `Types.SimulationResult` :
      `.solution` holds the raw DiffEq solution; `.propagator` is `NBodyPropagator()`.

# Notes on Symplectic Integrators
Symplectic methods conserve the symplectic 2-form exactly, which implies that
long-term energy drift is **bounded** (not accumulative) – a critical property
for billion-year solar system integrations.  They are fixed-step, so set
`propagator_options.dt` to an appropriate value (e.g. 1/20 of the shortest
orbital period in the system).

# Notes on Explicit Integrators
`Vern9()` and `DOP853()` are adaptive and naturally handle stiff phases
(close encounters).  Use them for short-to-medium integrations where you
need accurate trajectories rather than long-term statistical behaviour.

# Example – Solar System planets (symplectic)
```julia
using CelestialMechanics

bodies = [
    NBodyParticle(name=:sun,     r0=[0,0,0],   v0=[0,0,0],   mu=1.327e11, R=696_000.0),
    NBodyParticle(name=:earth,   r0=[1.496e8,0,0], v0=[0,29.78,0], mu=3.986e5, R=6_371.0),
    NBodyParticle(name=:jupiter, r0=[7.783e8,0,0], v0=[0,13.07,0], mu=1.267e8, R=69_911.0),
]

ic   = NBodySystemIC(bodies)
opts = PropagatorOptions(integrator=KahanLi8(), dt=86400.0)  # 1-day step

result = run_nbody_simulation(
    ic                = ic,
    tspan             = (0.0, 365.25*86400.0),  # 1 year in seconds
    propagator_options = opts,
    use_symplectic    = true,
    detect_collisions = false,
)
```
"""
function run_nbody_simulation(;
    ic::Types.NBodySystemIC,
    tspan::Tuple,
    propagator_options::Types.PropagatorOptions,
    use_symplectic::Bool    = false,
    detect_collisions::Bool = false,
    output_directory::String = joinpath(pwd(), "output")
    )

    bodies = ic.bodies
    N = length(bodies)
    @info "N-body integrator started" N=N symplectic=use_symplectic

    # ── unwrap tspan if Unitful ────────────────────────────────────────────────
    t0 = isa(tspan[1], Unitful.Quantity) ? ustrip(u"s", tspan[1]) : Float64(tspan[1])
    tf = isa(tspan[2], Unitful.Quantity) ? ustrip(u"s", tspan[2]) : Float64(tspan[2])
    tspan_s = (t0, tf)

    p = Types.NBodyParameters(bodies)

    # ── canonical unit normalization ───────────────────────────────────────────
    units = nothing
    solver_bodies = bodies
    tspan_solver = tspan_s

    if propagator_options.canonical_unit_normalization
        # first body is the central reference for du and tu
        ref_body = bodies[1]
        units = Coordinates.canonical_units(ustrip(ref_body.mu), ustrip(ref_body.R))
        mu_ref = ustrip(ref_body.mu)
        
        # normalize each particle relative to the reference body
        solver_bodies = [
            Types.NBodyParticle(
                name = b.name,
                r0   = b.r0 ./ units.DU,
                v0   = b.v0 ./ units.VU,
                mu   = ustrip(b.mu) / mu_ref,
                R    = ustrip(b.R) / units.DU
            ) for b in bodies
        ]
        
        # update time span and parameters for the solver
        tspan_solver = (tspan_s[1] / units.TU, tspan_s[2] / units.TU)
        p = Types.NBodyParameters(solver_bodies)
    end

    # ── disk output ────────────────────────────────────────────────────────────
    mkpath(output_directory)
    file_path = joinpath(output_directory, "nbody_simulation.csv")
    io        = open(file_path, "w")

    # CSV header: t, name_rx, name_ry, name_rz, name_vx, name_vy, name_vz, ...
    header_parts = ["t"]
    for b in bodies
        n = string(b.name)
        push!(header_parts, "$(n)_rx", "$(n)_ry", "$(n)_rz",
                             "$(n)_vx", "$(n)_vy", "$(n)_vz")
    end
    println(io, join(header_parts, ","))

    # ── build ode problem ──────────────────────────────────────────────────────
    local prob, affect_disk!

    if use_symplectic
        # ── secondorderodeproblem ──────────────────────────────────────────────
        # initial position vector: [r1x, r1y, r1z, r2x, ...] length 3n
        r0 = vcat([collect(b.r0) for b in solver_bodies]...)
        # initial velocity vector: [v1x, v1y, v1z, v2x, ...] length 3n
        v0 = vcat([collect(b.v0) for b in solver_bodies]...)

        prob = SecondOrderODEProblem(Dynamics.nbody_accelerations!, v0, r0, tspan_solver, p)
        @debug "Integrator configured" type=typeof(propagator_options.integrator) order=2

        # for secondorderode the integrator state is an arraypartition(v, r):
        #   int.u.x[1] = v  (length 3n)
        #   int.u.x[2] = r  (length 3n)
        affect_disk! = (int) -> begin
            t = int.t
            v_cur = int.u.x[1]
            r_cur = int.u.x[2]
            row = string(t)
            for i in 1:N
                bi = 3*(i-1)
                row *= ",$(r_cur[bi+1]),$(r_cur[bi+2]),$(r_cur[bi+3])," *
                        "$(v_cur[bi+1]),$(v_cur[bi+2]),$(v_cur[bi+3])"
            end
            println(io, row)
        end

    else
        # ── odeproblem (first-order form) ──────────────────────────────────────
        # initial state: [r1, v1, r2, v2, ...] length 6n
        # uses the original vcat logic mapped over the normalized bodies
        u0 = vcat([vcat(collect(b.r0), collect(b.v0)) for b in solver_bodies]...)

        prob = ODEProblem(Dynamics.nbody_equations!, u0, tspan_solver, p)
        @debug "Integrator configured" type=typeof(propagator_options.integrator) order=1

        affect_disk! = (int) -> begin
            u = int.u; t = int.t
            row = string(t)
            for i in 1:N
                base = 6*(i-1)
                row *= ",$(u[base+1]),$(u[base+2]),$(u[base+3])," *
                        "$(u[base+4]),$(u[base+5]),$(u[base+6])"
            end
            println(io, row)
        end
    end

    # ── callbacks ──────────────────────────────────────────────────────────────
    cb_disk = DiscreteCallback(
        (u, t, int) -> true,
        affect_disk!;
        save_positions = (false, false)
    )

    callbacks_list = Any[cb_disk]
    detect_collisions && push!(callbacks_list, make_collision_callback(bodies))
    full_cb = length(callbacks_list) == 1 ? callbacks_list[1] :
              CallbackSet(callbacks_list...)

    # ── solver options ─────────────────────────────────────────────────────────
    solver_opts = Dict{Symbol, Any}(
        :reltol        => propagator_options.reltol,
        :abstol        => propagator_options.abstol,
        :maxiters      => propagator_options.maxiters,
        :callback      => full_cb,
        :save_everystep => false,
    )

    if !isnothing(propagator_options.dt)
        solver_opts[:dt] = propagator_options.dt
        # Symplectic integrators need adaptive=false (fixed step)
        use_symplectic && (solver_opts[:adaptive] = false)
    elseif use_symplectic
        error("Symplectic integrators require a fixed time step. " *
              "Please set `propagator_options.dt` to a non-nothing value.")
    end

    # tspan_solver (canonical units)
    propagator_options.saveat && (solver_opts[:saveat] = collect(range(tspan_solver[1], tspan_solver[2]; length=1000)))

    # ── integrate ──────────────────────────────────────────────────────────────
    local sol
    elapsed = @elapsed begin
        sol = solve(prob, propagator_options.integrator; solver_opts...)
    end
    close(io)

    @info "N-body integration completed" time_s=round(elapsed; digits=3) retcode=sol.retcode

    # ── energy conservation check (first-order only) ───────────────────────────
    if !use_symplectic && length(sol.u) >= 2
        u_start = copy(sol.u[1])
        u_end   = copy(sol.u[end])
        
        # if canonical units were used, we must denormalize the states 
        # before feeding them to the physical energy equation
        if !isnothing(units)
            for i in 1:N
                idx = 6*(i-1)+1 : 6*i
                r_p_start, v_p_start = Coordinates.denormalize_state(u_start[idx], units)
                u_start[idx] = vcat(r_p_start, v_p_start)
                
                r_p_end, v_p_end = Coordinates.denormalize_state(u_end[idx], units)
                u_end[idx] = vcat(r_p_end, v_p_end)
            end
        end

        E0 = compute_energy(u_start, bodies)
        Ef = compute_energy(u_end, bodies)
        E_rel = abs((Ef - E0) / E0)
        @info "Energy conservation" E_rel=@sprintf("%.4e", E_rel)
    end

    return Types.SimulationResult(
        solution           = sol,
        units              = units,
        initial_conditions = ic,
        parameters         = nothing,
        propagator         = Types.NBodyPropagator(),
        equation_type      = nothing,
        poincare_raw       = nothing,
    )
end

end # module NBodyIntegrator
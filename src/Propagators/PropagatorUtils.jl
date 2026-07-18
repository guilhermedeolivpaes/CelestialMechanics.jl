# src/propagators/propagatorutils.jl

"""
    PropagatorUtils

Module containing utility functions shared across the different orbit propagators (Cowell, Hamiltonian, Lagrange). 
It handles parameter sanitization, high-performance local coordinate transformations, and the generation of ContinuousCallbacks for event detection (such as Poincare sections).
"""
module PropagatorUtils

using LinearAlgebra
using StaticArrays          
using Unitful               
using ..Types            
using DifferentialEquations
using DiffEqCallbacks
using RecursiveArrayTools
using Makie: Point2f

"""
    sanitize_parameters(p::Types.PerturbationParameters)

Ensures type stability by replacing any `nothing` fields within the perturbation parameters struct with properly typed zero values before they are injected into the numerical solver.

# Arguments
- `p::Types.PerturbationParameters`: The perturbation parameters structure.

# Returns
- `Types.PerturbationParameters`: A new parameter structure with sanitized values.
"""
function sanitize_parameters(p::Types.PerturbationParameters)
    return Types.PerturbationParameters(
        mu = isnothing(p.mu) ? 0.0u"km^3/s^2" : p.mu,
        R = isnothing(p.R) ? 0.0u"km" : p.R,
        omega_rot = isnothing(p.omega_rot) ? 0.0u"rad/s" : p.omega_rot,
        j2 = isnothing(p.j2) ? 0.0 : p.j2,
        j3 = isnothing(p.j3) ? 0.0 : p.j3,
        j4 = isnothing(p.j4) ? 0.0 : p.j4,
        j5 = isnothing(p.j5) ? 0.0 : p.j5,
        j6 = isnothing(p.j6) ? 0.0 : p.j6,
        j7 = isnothing(p.j7) ? 0.0 : p.j7,
        j8 = isnothing(p.j8) ? 0.0 : p.j8,
        j9 = isnothing(p.j9) ? 0.0 : p.j9,
        j10 = isnothing(p.j10) ? 0.0 : p.j10,
        j11 = isnothing(p.j11) ? 0.0 : p.j11,
        j12 = isnothing(p.j12) ? 0.0 : p.j12,
        j13 = isnothing(p.j13) ? 0.0 : p.j13,
        j14 = isnothing(p.j14) ? 0.0 : p.j14,
        j15 = isnothing(p.j15) ? 0.0 : p.j15,
        j16 = isnothing(p.j16) ? 0.0 : p.j16,
        j17 = isnothing(p.j17) ? 0.0 : p.j17,
        j18 = isnothing(p.j18) ? 0.0 : p.j18,
        c22 = isnothing(p.c22) ? 0.0 : p.c22,
        c31 = isnothing(p.c31) ? 0.0 : p.c31,
        c32 = isnothing(p.c32) ? 0.0 : p.c32,
        c33 = isnothing(p.c33) ? 0.0 : p.c33,
        c42 = isnothing(p.c42) ? 0.0 : p.c42,
        c44 = isnothing(p.c44) ? 0.0 : p.c44,
        s22 = isnothing(p.s22) ? 0.0 : p.s22,
        s31 = isnothing(p.s31) ? 0.0 : p.s31,
        s32 = isnothing(p.s32) ? 0.0 : p.s32,
        s33 = isnothing(p.s33) ? 0.0 : p.s33,
        s42 = isnothing(p.s42) ? 0.0 : p.s42,
        s44 = isnothing(p.s44) ? 0.0 : p.s44,
        alpha  = isnothing(p.alpha) ? 0.0u"m^2/kg" : p.alpha,
        cr = isnothing(p.cr) ? 0.0 : p.cr,
        shadow_in_srp = p.shadow_in_srp,
        cd = isnothing(p.cd) ? 0.0 : p.cd,
        am_drag = isnothing(p.am_drag) ? 0.0u"m^2/kg" : p.am_drag,
        n_bodies = p.n_bodies 
    )
end

# defines a constant for the z-axis unit vector to avoid allocations inside the function
const K_HAT = SVector{3}(0.0, 0.0, 1.0)

"""
    _cartesian_to_elements_snapshot(u::SVector{6,T}, mu::T; tol::Real = 1e-10) where {T}

Optimized in-place conversion from Cartesian state vectors to classical orbital elements. 

Designed specifically for use inside high-frequency numerical callbacks (e.g., extracting orbital parameters exactly at a Poincare section crossing) without triggering memory allocations.

# Arguments
- `u::SVector{6,T}`: The Cartesian state vector `[x, y, z, vx, vy, vz]`.
- `mu::T`: The standard gravitational parameter.
- `tol::Real`: Tolerance for singularity checks (circular or equatorial orbits). Defaults to 1e-10.

# Returns
- `Tuple{T, T, T, T, T}`: A tuple `(a, e, inc, RAAN, g)` containing the semi-major axis, eccentricity, inclination, right ascension of the ascending node, and argument of periapsis.
"""
function _cartesian_to_elements_snapshot(u::SVector{6,T}, mu::T; tol::Real = 1e-10) where {T}
    r_vec = SVector{3}(u[1], u[2], u[3])
    v_vec = SVector{3}(u[4], u[5], u[6])
    
    r_mag = norm(r_vec)
    v_mag = norm(v_vec)
    h_vec = cross(r_vec, v_vec) # specific angular momentum vector
    h_mag = norm(h_vec)

    n_vec = cross(K_HAT, h_vec) # node vector pointing to ascending node
    n_mag = norm(n_vec)

    e_vec = (1.0 / mu) * ((v_mag^2 - mu / r_mag) * r_vec - dot(r_vec, v_vec) * v_vec) # eccentricity vector
    e = norm(e_vec)

    energy = (v_mag^2 / 2.0) - (mu / r_mag) # specific orbital energy
    a = -mu / (2.0 * energy)
    # if the energy is near zero (parabolic), a = inf
    if abs(energy) < tol; a = T(Inf); end
    
    inc = acos(clamp(h_vec[3] / h_mag, -1.0, 1.0))

    # raan and pericenter argument logic
    local RAAN, g
    is_equatorial = (n_mag < tol)
    is_circular = (e < tol)

    if is_equatorial
        RAAN = 0.0 
        g = is_circular ? 0.0 : atan(e_vec[2], e_vec[1])
    elseif is_circular
        RAAN = atan(n_vec[2], n_vec[1])
        g = 0.0
    else
        RAAN = atan(n_vec[2], n_vec[1])
        cos_g = clamp(dot(n_vec, e_vec) / (n_mag * e), -1.0, 1.0)
        g = (e_vec[3] >= 0.0) ? acos(cos_g) : (2π - acos(cos_g))
    end

    return a, e, inc, RAAN, g
end

"""
    setup_poincare_callback(opts, eq_type)

Centralizes the event detection logic (e.g., pericenter passages or strobes) across the different propagators in the toolkit.
It uses multiple dispatch on `eq_type` to define the correct mathematical crossing condition (Cartesian dot product or angular roots).

# Arguments
- `opts::Types.PropagatorOptions`: Structure containing the `poincare_callback` boolean flag.
- `eq_type::Any`: The abstract type defining the current phase space (e.g., `CowellPropagator`, `HamiltonEquations`, `LagrangeEquations`).

# Returns
- `Tuple{Union{ContinuousCallback, Nothing}, Dict{Symbol, Any}}`: A tuple containing the `ContinuousCallback` object (or `nothing` if disabled) and a data dictionary `p_data` tracking raw states and crossing times.
"""
function setup_poincare_callback(opts::Types.PropagatorOptions, eq_type) 
    # data container
    p_data = Dict(
        :raw_states => Vector{Float64}[],
        :raw_snaps  => Vector{Float64}[],
        :times      => Float64[],
        :e_g        => Point2f[],
        :i_h        => Point2f[],
        :a_e        => Point2f[],
        :g_h        => Point2f[],
    )

    # if the poincare flag is inactive, return nothing
    if !(opts.poincare_callback)
        return nothing, p_data
    end

    # --- definition of the condition based on dispatch ---
    # we use an internal function to capture the specific behavior
    condition = _get_condition(eq_type)
    
    # --- definition of the event action ---
    affect!(int) = begin
        #if _is_pericenter(int.u, eq_type)
        if _is_pericenter(int.u, int, eq_type)
            # if cowell, convert on the fly and store the 5 elements
            if eq_type isa Types.CowellPropagator
                if int.u isa RecursiveArrayTools.ArrayPartition
                    u_flat = SVector{6}(vcat(int.u.x[2], int.u.x[1]))  # second_order: [r; v]
                else
                    u_flat = SVector{6}(int.u)  # first_order: its already [r; v]
                end
                elems = _cartesian_to_elements_snapshot(u_flat, int.p.perturb_params.mu)
                push!(p_data[:raw_states], collect(elems)) 
                # elems = [a, e, i, raan, omega]
                push!(p_data[:e_g], Point2f(elems[2], elems[5]))    # (e, g)
                push!(p_data[:i_h], Point2f(elems[3], elems[4]))    # (i, h)
                push!(p_data[:a_e], Point2f(elems[1], elems[2]))    # (a, e)
                push!(p_data[:g_h], Point2f(elems[5], elems[4]))    # (g, h)
            else
                # for hamilton/lagrange, `u` already holds the elements/momenta, store raw
                push!(p_data[:raw_states], copy(int.u))
            end
            push!(p_data[:times], int.t)
        end
    end

    cb = DiscreteCallback(
        (u, t, int) -> begin
            if int.u isa ArrayPartition
                prev_sign = dot(int.uprev.x[2], int.uprev.x[1])
                curr_sign = dot(int.u.x[2], int.u.x[1])
            else
                prev_sign = dot(int.uprev[1:3], int.uprev[4:6])
                curr_sign = dot(int.u[1:3], int.u[4:6])
            end
            return prev_sign * curr_sign < 0  # sign change
        end,
        affect!;
        save_positions=(false, false)
    )
                            
    return cb, p_data
end

# --- private helpers (multiple dispatch) ---

"""
    _get_condition(::Types.CowellPropagator)

Defines the continuous root-finding condition for Cartesian states (r.v = 0).

# Arguments
- `::Types.CowellPropagator`: Propagator type.

# Returns
- `Function`: A condition function evaluating `dot(r, v)`.
"""
_get_condition(::Types.CowellPropagator) = (u, t, int) -> dot(u[1:3], u[4:6]) # r.v

"""
    _get_condition(::Types.HamiltonEquations)

Defines the continuous root-finding condition for Delaunay variables (sin(l) = 0).

# Arguments
- `::Types.HamiltonEquations`: Equations type.

# Returns
- `Function`: A condition function evaluating `sin(l)`.
"""
_get_condition(::Types.HamiltonEquations) = (u, t, int) -> sin(u[4]) # sin(l)

"""
    _get_condition(::Types.LagrangeEquations)

Defines the continuous root-finding condition for Lagrange variables (sin(M) = 0).

# Arguments
- `::Types.LagrangeEquations`: Equations type.

# Returns
- `Function`: A condition function evaluating `sin(m)`.
"""
_get_condition(::Types.LagrangeEquations) = (u, t, int) -> sin(u[6]) # sin(m)


"""
    _get_condition(::Types.CR3BPPropagator)

Defines the continuous root-finding condition for Lagrange variables (sin(M) = 0).

# Arguments
- `::Types.LagrangeEquations`: Equations type.

# Returns
- `Function`: A condition function evaluating `eta=0`.

"""
_get_condition(::Types.CR3BPPropagator) = (u, t, int) -> u[2] # u[2] e a coordenada y (eta)

# --- pericenter vs apocenter verification ---
# (optional: ensures that we only save data at the closest approach)

"""
    _is_pericenter(u, ::Types.CowellPropagator)

Verifies if the Cartesian root corresponds to a pericenter passage.

# Arguments
- `u`: The state vector.
- `::Types.CowellPropagator`: Propagator type.

# Returns
- `Bool`: Always true for Cowell.
"""
#_is_pericenter(u, ::Types.CowellPropagator) = true # r.v is zero at both, requires extra check if needed
_is_pericenter(u, int, ::Types.CowellPropagator) = dot(int.uprev[1:3], int.uprev[4:6]) < 0

"""
    _is_pericenter(u, ::Types.HamiltonEquations)

Verifies if the Delaunay root corresponds to a pericenter passage (cos(l) > 0).

# Arguments
- `u`: The state vector.
- `::Types.HamiltonEquations`: Equations type.

# Returns
- `Bool`: True if pericenter, false otherwise.
"""
_is_pericenter(u, ::Types.HamiltonEquations) = cos(u[4]) > 0 # l=0 -> cos=1 (peri), l=pi -> cos=-1 (apo)

"""
    _is_pericenter(u, ::Types.LagrangeEquations)

Verifies if the Keplerian root corresponds to a pericenter passage (cos(M) > 0).

# Arguments
- `u`: The state vector.
- `::Types.LagrangeEquations`: Equations type.

# Returns
- `Bool`: True if pericenter, false otherwise.
"""
_is_pericenter(u, ::Types.LagrangeEquations) = cos(u[6]) > 0 # m=0 -> cos=1 (peri)


"""
    _is_pericenter(u, ::Types.CR3BPPropagator)

Filtra a direção do cruzamento.
Na sua função antiga: `eta_anterior < 0 && eta_atual >= 0`.
Isso significa que a partícula está cruzando de baixo para cima, logo, a velocidade em y deve ser positiva.
"""
_is_pericenter(u, ::Types.CR3BPPropagator) = u[5] > 0.0 # u[5] e a velocidade vy (eta_dot)

# ══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL POINCARE SECTIONS (Cartesian only)
# ══════════════════════════════════════════════════════════════════════════════

"""
    _store_elements!(p_data, elems, t)

Internal helper: converts a Keplerian tuple (a, e, i, Ω, ω) into the standard
Point2f projections and pushes them into `p_data`. Avoids code duplication
across the three section callbacks.
"""
function _store_elements!(p_data, elems, t)
    a, e, i, Ω, ω = elems
    push!(p_data[:raw_states], collect(elems))
    push!(p_data[:e_g],       Point2f(e, ω))
    push!(p_data[:i_h],       Point2f(i, Ω))
    push!(p_data[:a_e],       Point2f(a, e))
    push!(p_data[:g_h],       Point2f(ω, Ω))
    push!(p_data[:ecos_esin], Point2f(e * cos(ω), e * sin(ω)))
    push!(p_data[:icos_isin], Point2f(i * cos(Ω), i * sin(Ω)))
    push!(p_data[:times], t)
end

"""
    _extract_flat_state(u)

Returns an SVector{6} in [r; v] order regardless of whether the integrator
state is an ArrayPartition (second-order) or a plain vector (first-order).
"""
function _extract_flat_state(u)
    if u isa RecursiveArrayTools.ArrayPartition
        return SVector{6}(vcat(u.x[2], u.x[1]))  # second_order: x[2]=r, x[1]=v
    else
        return SVector{6}(u)
    end
end

"""
    _new_p_data()

Creates a fresh data container with the standard keys.
"""
function _new_p_data()
    return Dict(
        :raw_states => Vector{Float64}[],
        :raw_snaps  => Vector{Float64}[],
        :times      => Float64[],
        :e_g        => Point2f[],
        :i_h        => Point2f[],
        :a_e        => Point2f[],
        :g_h        => Point2f[],
        :ecos_esin  => Point2f[],   
        :icos_isin  => Point2f[],   
    )
end

# ── 1. Equatorial plane crossing: z = 0, ż > 0 (ascending node) ─────────────

"""
    setup_equatorial_crossing_callback(opts)

Creates a `DiscreteCallback` that detects ascending crossings of the equatorial
plane (z = 0, ż > 0) for Cartesian state vectors.

Surface of section:  Σ = { (r, v) ∈ ℝ⁶ : z = 0,  ż > 0 }

# Returns
- `Tuple{Union{DiscreteCallback, Nothing}, Dict}`: callback and data container.
"""
function setup_equatorial_crossing_callback(opts::Types.PropagatorOptions)
    p_data = _new_p_data()
    !opts.poincare_callback && return (nothing, p_data)

    affect!(int) = begin
        vz = int.u isa RecursiveArrayTools.ArrayPartition ? int.u.x[1][3] : int.u[6]
        if vz > 0  # ascending filter
            u_flat = _extract_flat_state(int.u)
            elems = _cartesian_to_elements_snapshot(u_flat, int.p.perturb_params.mu)
            _store_elements!(p_data, elems, int.t)
        end
    end

    cb = DiscreteCallback(
        (u, t, int) -> begin
            if int.u isa ArrayPartition
                prev_z = int.uprev.x[2][3]
                curr_z = int.u.x[2][3]
            else
                prev_z = int.uprev[3]
                curr_z = int.u[3]
            end
            return prev_z * curr_z < 0
        end,
        affect!;
        save_positions=(false, false)
    )

    return cb, p_data
end

# ── 2. Stroboscopic map: sample at every period T ────────────────────────────

"""
    setup_stroboscopic_callback(opts; T_perturbation::Real)

Creates a `PeriodicCallback` that samples the Cartesian state at times tₙ = n·T.

Natural Poincare section for systems with dominant periodic forcing (e.g.,
Didymos rotation period for tesserals, Dimorphos orbital period for third-body).

# Keyword Arguments
- `T_perturbation::Real`: forcing period in solver time units.

# Returns
- `Tuple{Union{PeriodicCallback, Nothing}, Dict}`: callback and data container.
"""
function setup_stroboscopic_callback(opts::Types.PropagatorOptions; T_perturbation::Real)
    p_data = _new_p_data()
    !opts.poincare_callback && return (nothing, p_data)

    affect!(int) = begin
        u_flat = _extract_flat_state(int.u)
        elems = _cartesian_to_elements_snapshot(u_flat, int.p.perturb_params.mu)
        _store_elements!(p_data, elems, int.t)
    end

    cb = DiffEqCallbacks.PeriodicCallback(affect!, T_perturbation; initial_affect=false, save_positions=(false, false))
    return cb, p_data
end

# ── 3. Orquestrador: monta todos os callbacks a partir de poincare_sections ──

"""
    setup_all_poincare_callbacks(opts, eq_type)

Iterates over `opts.poincare_sections` and builds a unified `CallbackSet`
plus a `Dict{Symbol, Dict}` mapping each section label to its `p_data`.

Supported section labels:
- `:periapsis_map`         — apsidal crossing (r·v = 0)
- `:equatorial_crossing`   — equatorial plane (z = 0, ż > 0)
- `:stroboscopic`          — periodic sampling at T_perturbation

# Returns
- `Tuple{Union{CallbackSet, Nothing}, Dict{Symbol, Dict}}`
"""
function setup_all_poincare_callbacks(opts::Types.PropagatorOptions, eq_type)
    callbacks = []
    all_p_data = Dict{Symbol, Dict}()

    for section in opts.poincare_sections
        if section == :periapsis_map
            cb, pd = setup_poincare_callback(opts, eq_type)
        elseif section == :equatorial_crossing
            cb, pd = setup_equatorial_crossing_callback(opts)
        elseif section == :stroboscopic
            isnothing(opts.T_perturbation) && error(
                "T_perturbation is required in PropagatorOptions when :stroboscopic section is active."
            )
            cb, pd = setup_stroboscopic_callback(opts; T_perturbation=opts.T_perturbation)
        else
            error("Unknown Poincare section: :$section. Options: :periapsis_map, :equatorial_crossing, :stroboscopic")
        end

        !isnothing(cb) && push!(callbacks, cb)
        all_p_data[section] = pd
    end

    cb_set = isempty(callbacks) ? nothing : CallbackSet(callbacks...)
    return cb_set, all_p_data
end

end # end of module

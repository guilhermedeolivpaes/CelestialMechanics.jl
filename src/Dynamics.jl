# src/dynamics.jl

"""
    Dynamics

This module orchestrates the orbital dynamics simulation. It is responsible for:
1.  Assembling the final perturbation function based on the activated physical models.
2.  Pre-calculating and interpolating ephemeris data (SPICE) for efficient use during integration.
3.  Defining the differential equation of motion (Cowell's equations) to be solved.
4.  Providing logging utilities to record the simulation configuration.
"""

module Dynamics

using SPICE
using StaticArrays
using LinearAlgebra
using RecursiveArrayTools: ArrayPartition
using DataInterpolations
using Unitful
using Unitful.DefaultSymbols # added for u"km", u"s", etc

# importing the necessary types and functions from other modules
using ..Types
using ..Ephemeris
using ..PerturbationEquations

function _collect_kernels(info::Types.SpiceInformations)
    kernels = String[]
    # regardless: if the path exists, it goes into the list.
    !isnothing(info.path_leapseconds_tls) && push!(kernels, info.path_leapseconds_tls)
    !isnothing(info.path_solar_system_bsp) && push!(kernels, info.path_solar_system_bsp)
    !isnothing(info.path_another_body)     && push!(kernels, info.path_another_body)
    !isnothing(info.path_binary_system)    && push!(kernels, info.path_binary_system)
    !isnothing(info.path_primary_body_bin_sys) && push!(kernels, info.path_primary_body_bin_sys)
    return kernels
end

# standard version
function _build_interpolators(::Types.StandardContext, info, p_params, t_vector, t_phys_vector, dist_scale)
    return _internal_spice_loop(info.primary_body_SPICE, info, p_params, t_vector, t_phys_vector, dist_scale)
end

# binary system version
function _build_interpolators(::Types.BinarySystemContext, info, p_params, t_vector, t_phys_vector, dist_scale)
    @warn "Binary system detected: Precedence for $(info.primary_body_bin_sys_SPICE)"
    return _internal_spice_loop(info.primary_body_bin_sys_SPICE, info, p_params, t_vector, t_phys_vector, dist_scale;
                                sun_observer = info.binary_system_SPICE)
end

# the processing loop (encapsulated to avoid repetition)
function _internal_spice_loop(observer, info, p_params, t_vector, t_phys_vector, dist_scale; sun_observer=nothing)
    interpolators = Types.BodyInterpolator{Float64}[]
    kernels = _collect_kernels(info)
    
    try
        SPICE.furnsh(kernels...)
        
        # identify necessary bodies (n-bodies + sun if srp is present)
        target_ids = Set(b.spice_id for b in p_params.n_bodies)
        (!isnothing(p_params.cr) && !iszero(p_params.cr)) && push!(target_ids, "SUN")

        for id in target_ids
            # take positions and create splines
            obs = (id == "SUN" && !isnothing(sun_observer)) ? sun_observer : observer
            raw = Ephemeris.get_body_position_vectors(info, t_phys_vector, id, obs)
            # convert unitful position vectors to matrix and normalize by distance scale
            mtx = hcat(ustrip.(raw)...) ./ dist_scale
            
            # find the corresponding mu if the body is in n_bodies.
            body_mu = 0.0
            idx = findfirst(x -> x.spice_id == id, p_params.n_bodies)
            if !isnothing(idx)
                body_mu = ustrip(p_params.n_bodies[idx].mu)
            end

            push!(interpolators, Types.BodyInterpolator(
                id, body_mu, 
                CubicSpline(mtx[1,:], t_vector),
                CubicSpline(mtx[2,:], t_vector),
                CubicSpline(mtx[3,:], t_vector)
            ))
        end
    finally
        SPICE.kclear()
    end
    return interpolators
end


"""
    set_perturbation(p_params, spice_info, t_vector) -> Function, Vector{String}, Vector{Float64}

Builds and returns the total perturbation function.

This is a "factory" function that performs the following steps:
1.  Loads the necessary SPICE kernels.
2.  Pre-calculates the positions of the perturbing bodies (Sun, etc.) for the entire time interval.
3.  Creates interpolation objects for these data, avoiding costly calls to SPICE inside the solver loop.
4.  Unloads the SPICE kernels.
5.  Returns a closure (the function `perturbation(r, t)`), which calculates the total perturbation acceleration for a given state and time, using the interpolated data.

# Returns
- `perturbation::Function`: The perturbation function `f(r, t)`.
- `eclipse_factors_history::Vector{Float64}`: The vector where the eclipse factors will be stored.
"""

function set_perturbation(p_params::Types.PerturbationParameters, spice_info::Types.SpiceInformations, t_vector::Vector{Float64}, t_phys_vector::Vector{Float64}; dist_scale=1.0)
    
    # decides the context
    context = isnothing(spice_info.primary_body_bin_sys_SPICE) ? Types.StandardContext() : Types.BinarySystemContext()
    
    # dispatches and receives the finished interpolators
    interpolators = _build_interpolators(context, spice_info, p_params, t_vector, t_phys_vector, dist_scale)

    # create the disturbance closure
    # will be called upon at each step of the integration
    function _perturbation(r_vector::SVector{3, T}, t::T) where {T} 
        # remembering that t is a type parameter and represents any numeric type that julia supports,
        # such as float64, float32, or even dual (used in automatic differentiation).
        # where {t} is the scope clause. it introduces the symbol t to the compiler and defines a consistency rule
        # ensuring that the type inside the svector is exactly the same type as the scalar t. if t is float64, the vector must be svector{3, float64}.
        # this allows julia to generate specialized and ultra-fast machine code for each different type you use,
        # without needing type checks during execution.
        P = @MVector zeros(T, 3) # mutable acceleration vector for performance
        p = p_params 

        # extract only once for the entire closure.
        mu_val = T(ustrip(p.mu))
        R_val  = T(ustrip(p.R))

        # zonal harmonics
        if !isnothing(p.j2) && !iszero(p.j2); P .+= PerturbationEquations.j2_perturbation(r_vector, mu_val, R_val, p.j2); end
        if !isnothing(p.j3) && !iszero(p.j3); P .+= PerturbationEquations.j3_perturbation(r_vector, mu_val, R_val, p.j3); end
        if !isnothing(p.j4) && !iszero(p.j4); P .+= PerturbationEquations.j4_perturbation(r_vector, mu_val, R_val, p.j4); end
        if !isnothing(p.j5) && !iszero(p.j5); P .+= PerturbationEquations.j5_perturbation(r_vector, mu_val, R_val, p.j5); end
        if !isnothing(p.j6) && !iszero(p.j6); P .+= PerturbationEquations.j6_perturbation(r_vector, mu_val, R_val, p.j6); end
        if !isnothing(p.j7) && !iszero(p.j7); P .+= PerturbationEquations.j7_perturbation(r_vector, mu_val, R_val, p.j7); end
        if !isnothing(p.j8) && !iszero(p.j8); P .+= PerturbationEquations.j8_perturbation(r_vector, mu_val, R_val, p.j8); end
        if !isnothing(p.j9) && !iszero(p.j9); P .+= PerturbationEquations.j9_perturbation(r_vector, mu_val, R_val, p.j9); end
        if !isnothing(p.j10) && !iszero(p.j10); P .+= PerturbationEquations.j10_perturbation(r_vector, mu_val, R_val, p.j10); end
        if !isnothing(p.j11) && !iszero(p.j11); P .+= PerturbationEquations.j11_perturbation(r_vector, mu_val, R_val, p.j11); end
        if !isnothing(p.j12) && !iszero(p.j12); P .+= PerturbationEquations.j12_perturbation(r_vector, mu_val, R_val, p.j12); end
        if !isnothing(p.j13) && !iszero(p.j13); P .+= PerturbationEquations.j13_perturbation(r_vector, mu_val, R_val, p.j13); end
        if !isnothing(p.j14) && !iszero(p.j14); P .+= PerturbationEquations.j14_perturbation(r_vector, mu_val, R_val, p.j14); end
        if !isnothing(p.j15) && !iszero(p.j15); P .+= PerturbationEquations.j15_perturbation(r_vector, mu_val, R_val, p.j15); end
        if !isnothing(p.j16) && !iszero(p.j16); P .+= PerturbationEquations.j16_perturbation(r_vector, mu_val, R_val, p.j16); end
        if !isnothing(p.j17) && !iszero(p.j17); P .+= PerturbationEquations.j17_perturbation(r_vector, mu_val, R_val, p.j17); end
        if !isnothing(p.j18) && !iszero(p.j18); P .+= PerturbationEquations.j18_perturbation(r_vector, mu_val, R_val, p.j18); end

        # sectoral harmonics (m = n)
        if (!isnothing(p.c22) && !isnothing(p.s22)) && (!iszero(p.c22) || !iszero(p.s22)) 
            P .+= PerturbationEquations.cs22_perturbation(r_vector, t, mu_val, R_val, p.c22, p.s22, p.omega_rot) # omega_rot without ustrip, because it is removed in the normalization decision.
        end
        
        if (!isnothing(p.c33) && !isnothing(p.s33)) && (!iszero(p.c33) || !iszero(p.s33))
            P .+= PerturbationEquations.cs33_perturbation(r_vector, t, mu_val, R_val, p.c33, p.s33, p.omega_rot)
        end
        if (!isnothing(p.c44) && !isnothing(p.s44)) && (!iszero(p.c44) || !iszero(p.s44))
            P .+= PerturbationEquations.cs44_perturbation(r_vector, t, mu_val, R_val, p.c44, p.s44, p.omega_rot)
        end

        # tesseral harmonics (m < n)
        if (!isnothing(p.c31) && !isnothing(p.s31)) && (!iszero(p.c31) || !iszero(p.s31))
            P .+= PerturbationEquations.cs31_perturbation(r_vector, t, mu_val, R_val, p.c31, p.s31, p.omega_rot)
        end
        if (!isnothing(p.c32) && !isnothing(p.s32)) && (!iszero(p.c32) || !iszero(p.s32))
            P .+= PerturbationEquations.cs32_perturbation(r_vector, t, mu_val, R_val, p.c32, p.s32, p.omega_rot)
        end
        if (!isnothing(p.c41) && !isnothing(p.s41)) && (!iszero(p.c41) || !iszero(p.s41))
            P .+= PerturbationEquations.cs41_perturbation(r_vector, t, mu_val, R_val, p.c41, p.s41, p.omega_rot)
        end
        if (!isnothing(p.c42) && !isnothing(p.s42)) && (!iszero(p.c42) || !iszero(p.s42))
            P .+= PerturbationEquations.cs42_perturbation(r_vector, t, mu_val, R_val, p.c42, p.s42, p.omega_rot)
        end
        if (!isnothing(p.c43) && !isnothing(p.s43)) && (!iszero(p.c43) || !iszero(p.s43))
            P .+= PerturbationEquations.cs43_perturbation(r_vector, t, mu_val, R_val, p.c43, p.s43, p.omega_rot)
        end
        
        for bi in interpolators
            # ultra-fast and typed call
            # creates the svector of the disturbing body's position at the given time.
            # if mu is 0.0 (as in the case of sun, used only for srp)
            if bi.mu > 0.0
                r_body = SVector{3, T}(bi.itp_x(t), bi.itp_y(t), bi.itp_z(t))
                P .+= PerturbationEquations.nbody_perturbation(r_vector, r_body, T(bi.mu))
            end

        end

        # srp
        if !isnothing(p.cr) && !iszero(p.cr)
            idx = findfirst(x -> x.spice_id == "SUN", interpolators)
            bi_sun = interpolators[idx]
            r_sun = SVector{3, T}(bi_sun.itp_x(t), bi_sun.itp_y(t), bi_sun.itp_z(t))
            P .+= PerturbationEquations.srp_perturbation(r_vector, ustrip(p.R), ustrip(p.alpha), p.cr, r_sun; 
                                    use_shadow_model=p.shadow_in_srp, dist_scale=dist_scale)
        end
        
        return SVector{3, T}(P)
    end
end


"""
    cowell_equations(u, p, t) -> SVector{6}

Defines the system of first-order differential equations for orbit
propagation (Cowell formulations), in the format `du/dt = f(u, p, t)` required
by `DifferentialEquations.jl`.

# Arguments
- `u::SVector{6}`: The state vector `[rx, ry, rz, vx, vy, vz]`.
- `p::SimulationParameters`: The struct containing the simulation parameters, including the perturbation function.
- `t::Float64`: The current time.

# Returns
- `SVector{6}`: The derivative of the state vector `[vx, vy, vz, ax, ay, az]`.
"""

# first-order cowell equations: u = (r,v) -> (v, a(r,t))
# generic t-shaped keys to work with float32/64, bigfloat, dual etc.
function cowell_equations(u::SVector{6,T}, p::Types.SimulationParameters, t::T) where {T}
    r = SVector{3,T}(u[1], u[2], u[3])
    v = SVector{3,T}(u[4], u[5], u[6])

    mu = T(p.perturb_params.mu)
    # efficient gravitational acceleration: -mu r / ||r||^3
    # calculate squared distance and its inverse cube root for gravitational acceleration
    r2    = dot(r, r)
    rinv3 = inv(r2 * sqrt(r2))
    U     = -mu * r * rinv3

    P_raw = p.perturb_func(r, t)              # must return svector{3, t}-compatible
    P     = SVector{3,T}(P_raw)               # type compatibility force

    a = U + P
    return SVector{6,T}(v[1], v[2], v[3], a[1], a[2], a[3])
end


function cowell_equations_2nd(v::SVector{3,T}, r::SVector{3,T}, p::Types.SimulationParameters, t::T) where {T}
    # now 'r' is the position and 'v' is the velocity
    
    mu = T(p.perturb_params.mu)
    
    # use 'r' for all position calculations
    # calculate squared distance and its inverse cube root for gravitational acceleration
    r2    = dot(r, r)
    rinv3 = inv(r2 * sqrt(r2))
    U     = -mu * r * rinv3

    # the disturbance should also use the 'r' position
    # the velocity 'v' is not used, which is correct for gravity
    P_raw = p.perturb_func(r, t)
    P     = SVector{3,T}(P_raw)

    return U + P # returns to acceleration
end

# ──────────────────────────────────────────────────────────────────────────────
# 1.  first-order equations  (for explicit / adaptive integrators)
# ──────────────────────────────────────────────────────────────────────────────
# state vector u has 6n components:
#   u[6(i-1)+1 : 6(i-1)+3] = r_i
#   u[6(i-1)+4 : 6(i-1)+6] = v_i

"""
    nbody_equations!(du, u, p, t)

In-place N-body right-hand side for first-order ODE solvers.

`du[6i-5:6i-3] = v_i`,  `du[6i-2:6i] = sum_j mu_j (r_j - r_i)/|r_j - r_i|^3`.
"""
function nbody_equations!(
    du::Vector{T}, u::Vector{T}, p::Types.NBodyParameters, t::T
    ) where {T}

    N = length(p.bodies)

    @inbounds for i in 1:N
        base_ri = 6*(i-1)        # offset of r_i in u
        base_vi = base_ri + 3    # offset of v_i in u

        # dr/dt = v  (copy velocity into du)
        du[base_ri+1] = u[base_vi+1]
        du[base_ri+2] = u[base_vi+2]
        du[base_ri+3] = u[base_vi+3]

        # dv/dt = gravitational acceleration from all other bodies
        rix = u[base_ri+1]; riy = u[base_ri+2]; riz = u[base_ri+3]
        ax  = zero(T);      ay  = zero(T);      az  = zero(T)

        for j in 1:N
            i == j && continue
            base_rj = 6*(j-1)

            # relative position vector from body i to body j
            dx = u[base_rj+1] - rix
            dy = u[base_rj+2] - riy
            dz = u[base_rj+3] - riz

            # calculate distance squared and inverse cube for newtonian gravity
            r2      = dx*dx + dy*dy + dz*dz
            inv_r3  = inv(r2 * sqrt(r2))
            muj     = T(p.bodies[j].mu)

            # accumulate gravitational acceleration
            ax += muj * dx * inv_r3
            ay += muj * dy * inv_r3
            az += muj * dz * inv_r3
        end

        du[base_vi+1] = ax
        du[base_vi+2] = ay
        du[base_vi+3] = az
    end

    return nothing
end


# ──────────────────────────────────────────────────────────────────────────────
# 2.  second-order equations  (for symplectic integrators)
# ──────────────────────────────────────────────────────────────────────────────
# differentialequations.jl symplectic methods require a `secondorderodeproblem`
# with the signature `f(dv, v, r, p, t)`, where:
#   r  = flat position vector  [r_1x, r_1y, r_1z, r_2x, ...]   (length 3n)
#   v  = flat velocity vector  [v_1x, v_1y, v_1z, v_2x, ...]   (length 3n)
#   dv = acceleration output   (length 3n)
#
# note: `v` is passed but not used in pure gravitational n-body (no velocity-
# dependent forces).  the signature is kept for diffeq compatibility.

"""
    nbody_accelerations!(dv, v, r, p, t)

In-place N-body acceleration for `SecondOrderODEProblem` / symplectic solvers.

`dv[3i-2:3i] = sum_j mu_j (r_j - r_i)/|r_j - r_i|^3`.
"""
function nbody_accelerations!(
    dv::Vector{T}, v::Vector{T}, r::Vector{T}, p::Types.NBodyParameters, t::T
    ) where {T}

    N = length(p.bodies)
    fill!(dv, zero(T))

    @inbounds for i in 1:N
        base_i = 3*(i-1)
        rix = r[base_i+1]; riy = r[base_i+2]; riz = r[base_i+3]

        ax = zero(T); ay = zero(T); az = zero(T)

        for j in 1:N
            i == j && continue
            base_j = 3*(j-1)

            # relative position between body i and j
            dx = r[base_j+1] - rix
            dy = r[base_j+2] - riy
            dz = r[base_j+3] - riz

            # compute inverse distance cubed
            r2     = dx*dx + dy*dy + dz*dz
            inv_r3 = inv(r2 * sqrt(r2))
            muj    = T(p.bodies[j].mu)

            # accumulate acceleration contribution
            ax += muj * dx * inv_r3
            ay += muj * dy * inv_r3
            az += muj * dz * inv_r3
        end

        dv[base_i+1] = ax
        dv[base_i+2] = ay
        dv[base_i+3] = az
    end

    return nothing
end

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

end # end of module

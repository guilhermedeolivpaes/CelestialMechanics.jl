# src/Models/SphericalHarmonics.jl

"""
    SphericalHarmonics

Module implementing spherical harmonic gravitational field evaluations using
Cunningham's (1970) recurrence relations in Cartesian coordinates, along with
loaders for standard planetary gravity field files (PDS SHA and ICGEM .gfc).

Supports:
- Arbitrary degree N and order M (0 <= M <= N).
- Fully normalized (geodesy 4π) or unnormalized (classical Kaula) harmonic coefficients.
- Body-fixed and inertial reference frame transformations with central body rotation rate.
- Optional Condon-Shortley phase adjustment (default: false, conforming to standard geodesy and GMAT).
- Zero-allocation evaluations during ODE integration via `CunninghamWorkspace`.
- Reading and denormalizing gravity field files: PDS SHA (.tab) and ICGEM (.gfc).

# References
- Cunningham, L. E. (1970). Celestial Mechanics, 2(2), 207-216.
- Montenbruck, O., & Gill, E. (2000). Satellite Orbits: Models, Methods and Applications. Springer, Sec 3.2.4 & 3.2.6.
- NASA Goddard Space Flight Center. GMAT Mathematical Specifications, Harmonic Gravity Field.
- ICGEM format specification: https://icgem.gfz.de/docs/ICGEM-Format-2023.pdf
"""
module SphericalHarmonics

using LinearAlgebra
using StaticArrays
using SpecialFunctions
using Unitful
using UnitfulAstro
using Logging

using ..Types

export GravityHarmonics,
       CunninghamWorkspace,
       normalization_factor,
       denormalize_coefficients,
       normalize_coefficients,
       cunningham_harmonics!,
       cunningham_acceleration_body_fixed!,
       cunningham_acceleration_body_fixed,
       cunningham_perturbation,
       cunningham_perturbation!,
       cunningham_potential,
       spherical_harmonic_perturbation,
       load_gravity_sha,
       sha_to_physical_params,
       sha_to_body_data,
       load_gravity_icgem,
       icgem_to_body_data

"""
    normalization_factor(n::Int, m::Int) -> Float64

Computes the geodetic 4π normalization factor N_nm relating fully normalized
coefficients (C̄_nm, S̄_nm) to unnormalized coefficients (C_nm, S_nm):

    C_nm = C̄_nm * N_nm
    C̄_nm = C_nm / N_nm

where:
    N_nm = sqrt((2 - δ_0m) * (2n + 1) * (n - m)! / (n + m)!)

Uses `SpecialFunctions.loggamma` to maintain precision and prevent overflow.
"""
function normalization_factor(n::Int, m::Int)::Float64
    if n < 0 || m < 0 || m > n
        throw(ArgumentError("normalization_factor: required 0 <= m <= n, got n=$n, m=$m"))
    end
    delta = (m == 0) ? 1.0 : 0.0
    log_N = 0.5 * (log(2.0 - delta) + log(2n + 1.0) + SpecialFunctions.loggamma(n - m + 1.0) - SpecialFunctions.loggamma(n + m + 1.0))
    return exp(log_N)
end

"""
    denormalize_coefficients(C_bar::AbstractMatrix, S_bar::AbstractMatrix;
                             degree::Int = size(C_bar, 1) - 1,
                             order::Int = min(degree, size(C_bar, 2) - 1)) -> (Matrix, Matrix)

Converts fully normalized coefficients (geodesy 4π convention) to unnormalized
coefficients (Kaula convention) up to `degree` and `order`.
"""
function denormalize_coefficients(C_bar::AbstractMatrix{<:Real}, S_bar::AbstractMatrix{<:Real};
                                 degree::Int = size(C_bar, 1) - 1,
                                 order::Int = min(degree, size(C_bar, 2) - 1))
    C = zeros(Float64, degree + 1, order + 1)
    S = zeros(Float64, degree + 1, order + 1)

    for n in 0:degree
        for m in 0:min(n, order)
            c_val = (n + 1 <= size(C_bar, 1) && m + 1 <= size(C_bar, 2)) ? Float64(C_bar[n + 1, m + 1]) : 0.0
            s_val = (n + 1 <= size(S_bar, 1) && m + 1 <= size(S_bar, 2)) ? Float64(S_bar[n + 1, m + 1]) : 0.0

            N = normalization_factor(n, m)
            C[n + 1, m + 1] = c_val * N
            S[n + 1, m + 1] = s_val * N
        end
    end
    return C, S
end

"""
    normalize_coefficients(C::AbstractMatrix, S::AbstractMatrix;
                           degree::Int = size(C, 1) - 1,
                           order::Int = min(degree, size(C, 2) - 1)) -> (Matrix, Matrix)

Converts unnormalized coefficients (Kaula convention) to fully normalized
coefficients (geodesy 4π convention) up to `degree` and `order`.
"""
function normalize_coefficients(C::AbstractMatrix{<:Real}, S::AbstractMatrix{<:Real};
                               degree::Int = size(C, 1) - 1,
                               order::Int = min(degree, size(C, 2) - 1))
    C_bar = zeros(Float64, degree + 1, order + 1)
    S_bar = zeros(Float64, degree + 1, order + 1)

    for n in 0:degree
        for m in 0:min(n, order)
            c_val = (n + 1 <= size(C, 1) && m + 1 <= size(C, 2)) ? Float64(C[n + 1, m + 1]) : 0.0
            s_val = (n + 1 <= size(S, 1) && m + 1 <= size(S, 2)) ? Float64(S[n + 1, m + 1]) : 0.0

            N = normalization_factor(n, m)
            C_bar[n + 1, m + 1] = c_val / N
            S_bar[n + 1, m + 1] = s_val / N
        end
    end
    return C_bar, S_bar
end

# ─────────────────────────────────────────────────────────────────────────────
# Constructors and Callables for Types.GravityHarmonics
# ─────────────────────────────────────────────────────────────────────────────

"""
    GravityHarmonics(C, S; max_degree, max_order, normalized=false, cs_phase=false)

Constructs a `GravityHarmonics` structure from coefficient matrices.

# Arguments
- `C::AbstractMatrix`: Cosine harmonic coefficients with 1-based indexing `C[n+1, m+1]`.
- `S::AbstractMatrix`: Sine harmonic coefficients with 1-based indexing `S[n+1, m+1]`.

# Keyword Arguments
- `max_degree::Int`: Maximum harmonic degree (default: `size(C, 1) - 1`).
- `max_order::Int`: Maximum harmonic order (default: `min(max_degree, size(C, 2) - 1)`).
- `normalized::Bool`: If `true`, input coefficients are assumed fully normalized (4π)
  and will be automatically denormalized for use in Cunningham recurrence. Default: `false`.
- `cs_phase::Bool`: Whether the coefficients include the Condon-Shortley phase (-1)^m.
  Default: `false` (standard in geodesy and GMAT).
"""
function Types.GravityHarmonics(
    C::AbstractMatrix{<:Real},
    S::AbstractMatrix{<:Real};
    max_degree::Int = size(C, 1) - 1,
    max_order::Int = min(max_degree, size(C, 2) - 1),
    normalized::Bool = false,
    cs_phase::Bool = false
)
    C_orig = zeros(Float64, max_degree + 1, max_order + 1)
    S_orig = zeros(Float64, max_degree + 1, max_order + 1)
    for n in 0:max_degree
        for m in 0:min(n, max_order)
            C_orig[n + 1, m + 1] = (n + 1 <= size(C, 1) && m + 1 <= size(C, 2)) ? Float64(C[n + 1, m + 1]) : 0.0
            S_orig[n + 1, m + 1] = (n + 1 <= size(S, 1) && m + 1 <= size(S, 2)) ? Float64(S[n + 1, m + 1]) : 0.0
        end
    end

    C_unnorm, S_unnorm = if normalized
        denormalize_coefficients(C_orig, S_orig; degree = max_degree, order = max_order)
    else
        (copy(C_orig), copy(S_orig))
    end

    return Types.GravityHarmonics(
        max_degree,
        max_order,
        normalized,
        cs_phase,
        C_unnorm,
        S_unnorm,
        C_orig,
        S_orig
    )
end

"""
    GravityHarmonics(grav::NamedTuple; max_degree=grav.max_degree, max_order=max_degree, cs_phase=false)

Constructs a `GravityHarmonics` structure from the output of `load_gravity_sha` or `load_gravity_icgem`.
"""
function Types.GravityHarmonics(
    grav::NamedTuple;
    max_degree::Int = grav.max_degree,
    max_order::Int = max_degree,
    cs_phase::Bool = false
)
    max_deg = min(max_degree, grav.max_degree)
    max_ord = min(max_order, max_deg)

    C = zeros(Float64, max_deg + 1, max_ord + 1)
    S = zeros(Float64, max_deg + 1, max_ord + 1)

    if hasproperty(grav, :coeffs)
        for ((n, m), (c_val, s_val)) in grav.coeffs
            if n <= max_deg && m <= max_ord
                C[n + 1, m + 1] = c_val
                S[n + 1, m + 1] = s_val
            end
        end
    end

    if hasproperty(grav, :Jn)
        for (n, j_val) in grav.Jn
            if n <= max_deg
                C[n + 1, 1] = -j_val
            end
        end
    end

    return Types.GravityHarmonics(
        C,
        S;
        max_degree = max_deg,
        max_order  = max_ord,
        normalized = grav.normalized,
        cs_phase   = cs_phase
    )
end

"""
    GravityHarmonics(; max_degree=2, max_order=0, Jn=Dict(), coeffs=Dict(), normalized=false, cs_phase=false)

Convenience constructor for `GravityHarmonics` from dictionaries of zonal and tesseral harmonics.
"""
function Types.GravityHarmonics(;
    max_degree::Int = 2,
    max_order::Int = 0,
    Jn = Dict{Int, Float64}(),
    coeffs = Dict{Tuple{Int, Int}, Tuple{Float64, Float64}}(),
    normalized::Bool = false,
    cs_phase::Bool = false
)
    C = zeros(Float64, max_degree + 1, max_order + 1)
    S = zeros(Float64, max_degree + 1, max_order + 1)

    if Jn isa AbstractDict
        for (n, j_val) in Jn
            if 2 <= n <= max_degree
                C[n + 1, 1] = -Float64(j_val)
            end
        end
    elseif Jn isa AbstractVector
        for (idx, j_val) in enumerate(Jn)
            n = idx + 1
            if 2 <= n <= max_degree
                C[n + 1, 1] = -Float64(j_val)
            end
        end
    end

    for ((n, m), (c_val, s_val)) in coeffs
        if n <= max_degree && m <= max_order
            C[n + 1, m + 1] = Float64(c_val)
            S[n + 1, m + 1] = Float64(s_val)
        end
    end

    return Types.GravityHarmonics(C, S; max_degree=max_degree, max_order=max_order, normalized=normalized, cs_phase=cs_phase)
end

# ─────────────────────────────────────────────────────────────────────────────
# Preallocated Workspace for Zero-Allocation Cunningham Integration
# ─────────────────────────────────────────────────────────────────────────────

"""
    CunninghamWorkspace{T}(degree, order, V, W, a_bf, C_work, S_work, cs_phase)

Preallocated buffers for evaluating Cunningham recurrence without heap allocations
inside ODE integrator loops.
"""
struct CunninghamWorkspace{T}
    degree::Int
    order::Int
    V::Matrix{T}
    W::Matrix{T}
    a_bf::Vector{T}
    C_work::Matrix{Float64}
    S_work::Matrix{Float64}
    cs_phase::Bool
end

function CunninghamWorkspace(degree::Int, order::Int = degree; T::Type = Float64,
                             C::AbstractMatrix = zeros(degree+1, order+1),
                             S::AbstractMatrix = zeros(degree+1, order+1),
                             normalized::Bool = false,
                             cs_phase::Bool = false)
    max_deg = degree + 1
    V = zeros(T, max_deg + 1, max_deg + 1)
    W = zeros(T, max_deg + 1, max_deg + 1)
    a_bf = zeros(T, 3)
    C_work, S_work = normalized ? denormalize_coefficients(C, S; degree=degree, order=order) : (Matrix{Float64}(C), Matrix{Float64}(S))
    return CunninghamWorkspace{T}(degree, order, V, W, a_bf, C_work, S_work, cs_phase)
end

function CunninghamWorkspace(gh::Types.GravityHarmonics; T::Type = Float64)
    return CunninghamWorkspace(gh.max_degree, gh.max_order; T=T, C=gh.C, S=gh.S, normalized=false, cs_phase=gh.cs_phase)
end

# ─────────────────────────────────────────────────────────────────────────────
# Cunningham Recurrence Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    cunningham_harmonics!(V, W, r_bf, R, degree, order=degree)

Computes Cunningham's V and W harmonic functions up to degree `degree + 1` in-place.
"""
function cunningham_harmonics!(
    V::AbstractMatrix{T},
    W::AbstractMatrix{T},
    r_bf::AbstractVector,
    R::Real,
    degree::Int,
    order::Int = degree
) where {T}
    x = T(r_bf[1])
    y = T(r_bf[2])
    z = T(r_bf[3])

    r2 = x * x + y * y + z * z
    r = sqrt(r2)
    b = T(R) / r2
    bz = b * z
    bR = b * T(R)

    max_deg = degree + 1

    # Base case (0, 0)
    V[1, 1] = T(R) / r
    W[1, 1] = zero(T)

    # Diagonal / sectoral terms (m = n)
    for m in 1:max_deg
        idx = m + 1
        prev = m
        fac = T(2m - 1) * b
        V[idx, idx] = fac * (x * V[prev, prev] - y * W[prev, prev])
        W[idx, idx] = fac * (x * W[prev, prev] + y * V[prev, prev])
    end

    # Off-diagonal terms (n > m)
    for m in 0:max_deg
        m_idx = m + 1
        for n in (m + 1):max_deg
            n_idx = n + 1
            fac1 = (T(2n - 1) / T(n - m)) * bz
            v_prev1 = V[n_idx - 1, m_idx]
            w_prev1 = W[n_idx - 1, m_idx]

            if n == m + 1
                V[n_idx, m_idx] = fac1 * v_prev1
                W[n_idx, m_idx] = fac1 * w_prev1
            else
                fac2 = (T(n + m - 1) / T(n - m)) * bR
                V[n_idx, m_idx] = fac1 * v_prev1 - fac2 * V[n_idx - 2, m_idx]
                W[n_idx, m_idx] = fac1 * w_prev1 - fac2 * W[n_idx - 2, m_idx]
            end
        end
    end

    return nothing
end

"""
    cunningham_acceleration_body_fixed!(a, V, W, mu, R, C, S, degree, order; cs_phase=false)

Computes the gravitational perturbing acceleration in the body-fixed frame using
precomputed Cunningham functions V and W and unnormalized coefficients C and S.
"""
function cunningham_acceleration_body_fixed!(
    a::AbstractVector{T},
    V::AbstractMatrix{T},
    W::AbstractMatrix{T},
    mu::Real,
    R::Real,
    C::AbstractMatrix{<:Real},
    S::AbstractMatrix{<:Real},
    degree::Int,
    order::Int = degree;
    cs_phase::Bool = false
) where {T}
    ax = zero(T)
    ay = zero(T)
    az = zero(T)

    for n in 2:degree
        n_idx = n + 1

        # m = 0 (zonal terms)
        C_n0 = T(C[n_idx, 1])
        if !iszero(C_n0)
            ax -= C_n0 * V[n_idx + 1, 2]
            ay -= C_n0 * W[n_idx + 1, 2]
            az -= T(n + 1) * C_n0 * V[n_idx + 1, 1]
        end

        # m > 0 (tesseral and sectoral terms)
        for m in 1:min(n, order)
            m_idx = m + 1
            C_nm = T(C[n_idx, m_idx])
            S_nm = T(S[n_idx, m_idx])

            if iszero(C_nm) && iszero(S_nm)
                continue
            end

            phase_fac = (cs_phase && isodd(m)) ? -one(T) : one(T)
            C_eff = C_nm * phase_fac
            S_eff = S_nm * phase_fac

            if m == 1
                # m = 1 boundary term
                ax += 0.5 * (-C_eff * V[n_idx + 1, 3] - S_eff * W[n_idx + 1, 3] +
                             T(n) * (T(n + 1)) * (C_eff * V[n_idx + 1, 1]))
                ay += 0.5 * (-C_eff * W[n_idx + 1, 3] + S_eff * V[n_idx + 1, 3] +
                             T(n) * (T(n + 1)) * (S_eff * V[n_idx + 1, 1]))
            else
                # m > 1 general terms
                fac_n_m = T(n - m + 1) * T(n - m + 2)
                ax += 0.5 * (-C_eff * V[n_idx + 1, m_idx + 1] - S_eff * W[n_idx + 1, m_idx + 1] +
                             fac_n_m * (C_eff * V[n_idx + 1, m_idx - 1] + S_eff * W[n_idx + 1, m_idx - 1]))
                ay += 0.5 * (-C_eff * W[n_idx + 1, m_idx + 1] + S_eff * V[n_idx + 1, m_idx + 1] +
                             fac_n_m * (-C_eff * W[n_idx + 1, m_idx - 1] + S_eff * V[n_idx + 1, m_idx - 1]))
            end

            az += T(n - m + 1) * (-C_eff * V[n_idx + 1, m_idx] - S_eff * W[n_idx + 1, m_idx])
        end
    end

    factor = T(mu) / (T(R) * T(R))
    a[1] = ax * factor
    a[2] = ay * factor
    a[3] = az * factor

    return nothing
end

"""
    cunningham_acceleration_body_fixed(r_bf, mu, R, C, S; degree, order, cs_phase=false) -> SVector{3}
"""
function cunningham_acceleration_body_fixed(
    r_bf::AbstractVector{T},
    mu::Real,
    R::Real,
    C::AbstractMatrix{<:Real},
    S::AbstractMatrix{<:Real};
    degree::Int = size(C, 1) - 1,
    order::Int = min(degree, size(C, 2) - 1),
    cs_phase::Bool = false
) where {T}
    max_deg = degree + 1
    V = zeros(T, max_deg + 1, max_deg + 1)
    W = zeros(T, max_deg + 1, max_deg + 1)
    cunningham_harmonics!(V, W, r_bf, R, degree, order)

    a = zeros(T, 3)
    cunningham_acceleration_body_fixed!(a, V, W, mu, R, C, S, degree, order; cs_phase=cs_phase)
    return SVector{3, T}(a[1], a[2], a[3])
end

"""
    cunningham_perturbation!(ws::CunninghamWorkspace{T}, r_inertial, t, mu, R, omega_rot=0.0) -> SVector{3}

Zero-allocation in-place evaluation of the gravitational perturbing acceleration in the inertial frame.
"""
function cunningham_perturbation!(
    ws::CunninghamWorkspace{T},
    r_inertial::AbstractVector{T},
    t::Real,
    mu::Real,
    R::Real,
    omega_rot::Union{Nothing, Real} = 0.0
) where {T}
    omega = isnothing(omega_rot) ? 0.0 : Float64(ustrip(omega_rot))
    time_val = Float64(ustrip(t))
    theta = omega * time_val

    c = cos(theta)
    s = sin(theta)

    x_i, y_i, z_i = T(r_inertial[1]), T(r_inertial[2]), T(r_inertial[3])
    x_bf = c * x_i + s * y_i
    y_bf = -s * x_i + c * y_i
    z_bf = z_i
    r_bf = SVector{3, T}(x_bf, y_bf, z_bf)

    cunningham_harmonics!(ws.V, ws.W, r_bf, R, ws.degree, ws.order)
    cunningham_acceleration_body_fixed!(ws.a_bf, ws.V, ws.W, mu, R, ws.C_work, ws.S_work, ws.degree, ws.order; cs_phase=ws.cs_phase)

    ax_bf, ay_bf, az_bf = ws.a_bf[1], ws.a_bf[2], ws.a_bf[3]
    ax_i = c * ax_bf - s * ay_bf
    ay_i = s * ax_bf + c * ay_bf
    az_i = az_bf

    return SVector{3, T}(ax_i, ay_i, az_i)
end

"""
    cunningham_perturbation(r_inertial, t, mu, R, C, S; degree, order, omega_rot=0.0, cs_phase=false, normalized=false) -> SVector{3}

Calculates the gravitational perturbing acceleration in the **inertial frame**.
"""
function cunningham_perturbation(
    r_inertial::AbstractVector{T},
    t::Real,
    mu::Real,
    R::Real,
    C::AbstractMatrix{<:Real},
    S::AbstractMatrix{<:Real};
    degree::Int = size(C, 1) - 1,
    order::Int = min(degree, size(C, 2) - 1),
    omega_rot::Union{Nothing, Real} = 0.0,
    cs_phase::Bool = false,
    normalized::Bool = false
) where {T}
    C_work, S_work = if normalized
        denormalize_coefficients(C, S; degree=degree, order=order)
    else
        (C, S)
    end

    omega = isnothing(omega_rot) ? 0.0 : Float64(ustrip(omega_rot))
    time_val = Float64(ustrip(t))
    theta = omega * time_val

    c = cos(theta)
    s = sin(theta)

    x_i, y_i, z_i = T(r_inertial[1]), T(r_inertial[2]), T(r_inertial[3])
    x_bf = c * x_i + s * y_i
    y_bf = -s * x_i + c * y_i
    z_bf = z_i
    r_bf = SVector{3, T}(x_bf, y_bf, z_bf)

    max_deg = degree + 1
    V = zeros(T, max_deg + 1, max_deg + 1)
    W = zeros(T, max_deg + 1, max_deg + 1)
    cunningham_harmonics!(V, W, r_bf, R, degree, order)

    a_bf = zeros(T, 3)
    cunningham_acceleration_body_fixed!(a_bf, V, W, mu, R, C_work, S_work, degree, order; cs_phase=cs_phase)

    ax_bf, ay_bf, az_bf = a_bf[1], a_bf[2], a_bf[3]
    ax_i = c * ax_bf - s * ay_bf
    ay_i = s * ax_bf + c * ay_bf
    az_i = az_bf

    return SVector{3, T}(ax_i, ay_i, az_i)
end

function cunningham_perturbation(
    r_inertial::AbstractVector{T},
    t::Real,
    mu::Real,
    R::Real,
    gh::Types.GravityHarmonics,
    omega_rot::Union{Nothing, Real} = 0.0
) where {T}
    return cunningham_perturbation(
        r_inertial,
        t,
        mu,
        R,
        gh.C,
        gh.S;
        degree = gh.max_degree,
        order = gh.max_order,
        omega_rot = omega_rot,
        cs_phase = gh.cs_phase,
        normalized = false
    )
end

function (gh::Types.GravityHarmonics)(
    r_inertial::AbstractVector,
    t::Real,
    mu::Real,
    R::Real,
    omega_rot::Union{Nothing, Real} = 0.0
)
    return cunningham_perturbation(r_inertial, t, mu, R, gh, omega_rot)
end

const spherical_harmonic_perturbation = cunningham_perturbation

"""
    cunningham_potential(r_inertial, t, mu, R, C, S; degree, order, omega_rot=0.0, normalized=false) -> Float64

Computes the perturbing gravitational potential energy U_pert (degrees n >= 2) in the inertial frame.
"""
function cunningham_potential(
    r_inertial::AbstractVector{T},
    t::Real,
    mu::Real,
    R::Real,
    C::AbstractMatrix{<:Real},
    S::AbstractMatrix{<:Real};
    degree::Int = size(C, 1) - 1,
    order::Int = min(degree, size(C, 2) - 1),
    omega_rot::Union{Nothing, Real} = 0.0,
    cs_phase::Bool = false,
    normalized::Bool = false
) where {T}
    C_work, S_work = normalized ? denormalize_coefficients(C, S; degree=degree, order=order) : (C, S)
    omega = isnothing(omega_rot) ? 0.0 : Float64(ustrip(omega_rot))
    theta = omega * Float64(ustrip(t))
    c, s = cos(theta), sin(theta)

    x_i, y_i, z_i = T(r_inertial[1]), T(r_inertial[2]), T(r_inertial[3])
    r_bf = SVector{3, T}(c * x_i + s * y_i, -s * x_i + c * y_i, z_i)

    max_deg = degree + 1
    V = zeros(T, max_deg + 1, max_deg + 1)
    W = zeros(T, max_deg + 1, max_deg + 1)
    cunningham_harmonics!(V, W, r_bf, R, degree, order)

    U = zero(T)
    factor = T(mu) / T(R)
    for n in 2:degree
        n_idx = n + 1
        C_n0 = T(C_work[n_idx, 1])
        if !iszero(C_n0)
            U += C_n0 * V[n_idx, 1]
        end
        for m in 1:min(n, order)
            m_idx = m + 1
            phase_fac = (cs_phase && isodd(m)) ? -one(T) : one(T)
            C_nm = T(C_work[n_idx, m_idx]) * phase_fac
            S_nm = T(S_work[n_idx, m_idx]) * phase_fac
            if !iszero(C_nm)
                U += C_nm * V[n_idx, m_idx]
            end
            if !iszero(S_nm)
                U += S_nm * W[n_idx, m_idx]
            end
        end
    end
    return Float64(factor * U)
end

function cunningham_potential(
    r_inertial::AbstractVector,
    t::Real,
    mu::Real,
    R::Real,
    gh::Types.GravityHarmonics,
    omega_rot::Union{Nothing, Real} = 0.0
)
    return cunningham_potential(
        r_inertial, t, mu, R, gh.C, gh.S;
        degree=gh.max_degree, order=gh.max_order, omega_rot=omega_rot,
        cs_phase=gh.cs_phase, normalized=false
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# File Parsers: PDS SHA (.tab) and ICGEM (.gfc)
# ─────────────────────────────────────────────────────────────────────────────

function _build_fields(coeffs::Dict{Tuple{Int,Int}, Tuple{Float64,Float64}},
                       Jn::Dict{Int,Float64},
                       max_degree::Int)
    fields = Dict{Symbol, Any}()

    for n in 2:max_degree
        if haskey(Jn, n)
            fields[Symbol("j$n")] = Jn[n]
        end

        for m in 1:n
            if haskey(coeffs, (n, m))
                C, S = coeffs[(n, m)]
                fields[Symbol("c$n$m")] = C
                fields[Symbol("s$n$m")] = S
            end
        end
    end

    return fields
end

"""
    load_gravity_sha(filepath::String; max_degree::Int = 18, denormalize=true, normalized=nothing) -> NamedTuple

Reads a PDS SHA-format gravity field file and returns harmonic coefficients.
"""
function load_gravity_sha(
    filepath::String;
    max_degree::Int = 18,
    denormalize::Bool = true,
    normalized::Union{Nothing, Bool} = nothing
)
    if !isfile(filepath)
        error("SphericalHarmonics: file not found at $filepath")
    end

    lines = readlines(filepath)
    header = replace(lines[1], r"[Dd]" => "e")
    header_fields = split(header, ',')

    R_km       = parse(Float64, strip(header_fields[1]))
    GM         = parse(Float64, strip(header_fields[2]))
    n_max_file = parse(Int,     strip(header_fields[4]))
    norm_flag  = parse(Int,     strip(header_fields[6]))

    is_normalized = (norm_flag == 1)
    n_load = min(max_degree, n_max_file)
    should_denorm = !isnothing(normalized) ? !normalized : denormalize

    coeffs = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
    Jn     = Dict{Int, Float64}()

    for line in lines[2:end]
        stripped = strip(replace(line, r"[Dd]" => "e"))
        isempty(stripped) && continue

        fields = split(stripped, ',')
        length(fields) < 4 && continue

        n = parse(Int, strip(fields[1]))
        m = parse(Int, strip(fields[2]))

        n > n_load && continue
        n < 2 && continue

        Cnm_bar = parse(Float64, strip(fields[3]))
        Snm_bar = parse(Float64, strip(fields[4]))

        if is_normalized && should_denorm
            N = normalization_factor(n, m)
            Cnm = Cnm_bar * N
            Snm = Snm_bar * N
        else
            Cnm = Cnm_bar
            Snm = Snm_bar
        end

        coeffs[(n, m)] = (Cnm, Snm)
        if m == 0
            Jn[n] = -Cnm
        end
    end

    is_output_normalized = is_normalized && !should_denorm
    @info "SphericalHarmonics: loaded $(length(coeffs)) coefficients up to degree $n_load" source=basename(filepath) R_km GM source_normalized=is_normalized output_normalized=is_output_normalized

    return (
        R_km              = R_km,
        GM                = GM,
        max_degree        = n_load,
        normalized        = is_output_normalized,
        source_normalized = is_normalized,
        coeffs            = coeffs,
        Jn                = Jn
    )
end

"""
    sha_to_physical_params(filepath::String; max_degree::Int = 6, kwargs...) -> PhysicalParams

Reads a SHA file and constructs a `PhysicalParams` struct with denormalized coefficients.
"""
function sha_to_physical_params(filepath::String; max_degree::Int = 6, kwargs...)
    grav = load_gravity_sha(filepath; max_degree = max_degree)
    fields = _build_fields(grav.coeffs, grav.Jn, grav.max_degree)

    pp_kwargs = Dict{Symbol, Any}(:mu => grav.GM, :R => grav.R_km)
    for fname in fieldnames(Types.PhysicalParams)
        if haskey(fields, fname)
            pp_kwargs[fname] = fields[fname]
        end
    end
    merge!(pp_kwargs, Dict(kwargs))

    return Types.PhysicalParams(; pp_kwargs...)
end

"""
    sha_to_body_data(filepath::String; max_degree::Int = 6, denormalize=true, normalized=nothing,
                     name=:custom, spice_id="", omega_rot=0.0, extra_fields...) -> NamedTuple

Reads a SHA file and constructs a `NamedTuple` compatible with `BODIES_DATA`.
"""
function sha_to_body_data(
        filepath::String; 
        max_degree::Int = 6,
        denormalize::Bool = true,
        normalized::Union{Nothing, Bool} = nothing,
        name::Symbol = :custom, 
        spice_id::String = "",
        omega_rot::Float64 = 0.0,
        extra_fields...
    )
    grav = load_gravity_sha(filepath; max_degree = max_degree, denormalize = denormalize, normalized = normalized)
    fields = _build_fields(grav.coeffs, grav.Jn, grav.max_degree)

    base = Dict{Symbol, Any}(
        :name               => name,
        :mu                 => grav.GM * 1.0u"km^3/s^2",
        :R                  => grav.R_km * 1.0u"km",
        :omega_rot          => omega_rot * 1.0u"rad/s",
        :spice_id           => spice_id,
        :normalized_gravity => grav.normalized
    )

    merge!(base, fields)
    for (k, v) in extra_fields
        base[k] = v
    end

    keys_sorted = sort(collect(keys(base)))
    nt_keys = tuple(keys_sorted...)
    nt_vals = tuple((base[k] for k in keys_sorted)...)
    return NamedTuple{nt_keys}(nt_vals)
end

"""
    load_gravity_icgem(filepath::String; max_degree::Int = 18, denormalize=true, normalized=nothing) -> NamedTuple

Reads an ICGEM-format (.gfc) gravity field file and returns parsed coefficients.
"""
function load_gravity_icgem(
    filepath::String;
    max_degree::Int = 18,
    denormalize::Bool = true,
    normalized::Union{Nothing, Bool} = nothing
)
    if !isfile(filepath)
        error("SphericalHarmonics: file not found at $filepath")
    end

    lines = readlines(filepath)
    GM = 0.0
    R_m = 0.0
    n_max_file = 0
    is_normalized = true
    header_end = 0

    for (idx, line) in enumerate(lines)
        stripped = strip(line)
        if startswith(stripped, "end_of_head")
            header_end = idx
            break
        end

        tokens = split(stripped)
        isempty(tokens) && continue
        key = lowercase(tokens[1])

        if key == "earth_gravity_constant" || key == "gravity_constant"
            GM = parse(Float64, tokens[2])
        elseif key == "radius"
            R_m = parse(Float64, tokens[2])
        elseif key == "max_degree"
            n_max_file = parse(Int, tokens[2])
        elseif key == "norm"
            norm_str = lowercase(tokens[2])
            is_normalized = (norm_str == "fully_normalized")
        end
    end

    if header_end == 0
        @warn "SphericalHarmonics: no 'end_of_head' found in ICGEM file; attempting to parse from line 1."
        header_end = 0
    end

    R_km = R_m / 1000.0
    GM_km = GM / 1.0e9

    n_load = n_max_file > 0 ? min(max_degree, n_max_file) : max_degree
    should_denorm = !isnothing(normalized) ? !normalized : denormalize

    coeffs = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
    Jn = Dict{Int, Float64}()

    for line in lines[(header_end + 1):end]
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, "#") && continue

        tokens = split(stripped)
        length(tokens) < 5 && continue
        keyword = lowercase(tokens[1])
        keyword != "gfc" && continue

        n = parse(Int, tokens[2])
        m = parse(Int, tokens[3])

        n > n_load && continue
        n < 2 && continue

        Cnm_bar = parse(Float64, replace(tokens[4], r"[Dd]" => "e"))
        Snm_bar = parse(Float64, replace(tokens[5], r"[Dd]" => "e"))

        if is_normalized && should_denorm
            N = normalization_factor(n, m)
            Cnm = Cnm_bar * N
            Snm = Snm_bar * N
        else
            Cnm = Cnm_bar
            Snm = Snm_bar
        end

        coeffs[(n, m)] = (Cnm, Snm)
        if m == 0
            Jn[n] = -Cnm
        end
    end

    is_output_normalized = is_normalized && !should_denorm
    @info "SphericalHarmonics: loaded $(length(coeffs)) coefficients up to degree $n_load" source=basename(filepath) R_km GM_km3s2=GM_km source_normalized=is_normalized output_normalized=is_output_normalized

    return (
        R_km              = R_km,
        GM                = GM_km,
        max_degree        = n_load,
        normalized        = is_output_normalized,
        source_normalized = is_normalized,
        coeffs            = coeffs,
        Jn                = Jn
    )
end

"""
    icgem_to_body_data(filepath::String; max_degree::Int = 6, denormalize=true, normalized=nothing,
                       name=:custom, spice_id="", omega_rot=0.0, extra_fields...) -> NamedTuple

Reads an ICGEM (.gfc) file and constructs a `NamedTuple` compatible with `BODIES_DATA`.
"""
function icgem_to_body_data(
        filepath::String;
        max_degree::Int = 6,
        denormalize::Bool = true,
        normalized::Union{Nothing, Bool} = nothing,
        name::Symbol = :custom,
        spice_id::String = "",
        omega_rot::Float64 = 0.0,
        extra_fields...
    )
    grav = load_gravity_icgem(filepath; max_degree = max_degree, denormalize = denormalize, normalized = normalized)
    fields = _build_fields(grav.coeffs, grav.Jn, grav.max_degree)

    base = Dict{Symbol, Any}(
        :name               => name,
        :mu                 => grav.GM * 1.0u"km^3/s^2",
        :R                  => grav.R_km * 1.0u"km",
        :omega_rot          => omega_rot * 1.0u"rad/s",
        :spice_id           => spice_id,
        :normalized_gravity => grav.normalized
    )

    merge!(base, fields)
    for (k, v) in extra_fields
        base[k] = v
    end

    keys_sorted = sort(collect(keys(base)))
    nt_keys = tuple(keys_sorted...)
    nt_vals = tuple((base[k] for k in keys_sorted)...)
    return NamedTuple{nt_keys}(nt_vals)
end

end # module SphericalHarmonics

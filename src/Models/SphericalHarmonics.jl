# src/Models/SphericalHarmonics.jl

"""
    SphericalHarmonics

Module implementing spherical harmonic gravitational field evaluations using
Cunningham's (1970) recurrence relations in Cartesian coordinates.

Supports:
- Arbitrary degree N and order M (0 <= M <= N).
- Fully normalized (geodesy 4π) or unnormalized (classical Kaula) harmonic coefficients.
- Body-fixed and inertial reference frame transformations with central body rotation rate.
- Optional Condon-Shortley phase adjustment (default: false, conforming to standard geodesy and GMAT).

# References
- Cunningham, L. E. (1970). Celestial Mechanics, 2(2), 207-216.
- Montenbruck, O., & Gill, E. (2000). Satellite Orbits: Models, Methods and Applications. Springer, Sec 3.2.4 & 3.2.6.
- NASA Goddard Space Flight Center. GMAT Mathematical Specifications, Harmonic Gravity Field.

# Note
- This module was written by Antigravity (Gemini) based on the author's drafts.
"""
module SphericalHarmonics

using LinearAlgebra
using StaticArrays
using SpecialFunctions
using Unitful

using ..Types

export GravityHarmonics,
       normalization_factor,
       denormalize_coefficients,
       normalize_coefficients,
       cunningham_harmonics!,
       cunningham_acceleration_body_fixed!,
       cunningham_acceleration_body_fixed,
       cunningham_perturbation,
       spherical_harmonic_perturbation

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
            # if 1-indexed vector, element 1 is J2, element 2 is J3, etc.
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
# Cunningham Recurrence Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    cunningham_harmonics!(V, W, r_bf, R, degree, order=degree)

Computes Cunningham's auxiliary harmonic functions V_nm and W_nm up to degree+1
and order+1 in the body-fixed frame coordinates `r_bf = (x, y, z)`.

# Arguments
- `V::AbstractMatrix{T}`: Matrix of minimum dimensions `(degree + 2, degree + 2)`.
- `W::AbstractMatrix{T}`: Matrix of minimum dimensions `(degree + 2, degree + 2)`.
- `r_bf::AbstractVector`: Position vector in the body-fixed rotating frame.
- `R::Real`: Equatorial reference radius of the central body.
- `degree::Int`: Maximum harmonic degree.
- `order::Int`: Maximum harmonic order (default: `degree`).
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
            C_raw = T(C[n_idx, m_idx])
            S_raw = T(S[n_idx, m_idx])
            (iszero(C_raw) && iszero(S_raw)) && continue

            # Optional Condon-Shortley phase (-1)^m adjustment
            sgn = (cs_phase && isodd(m)) ? -one(T) : one(T)
            C_nm = sgn * C_raw
            S_nm = sgn * S_raw

            k = T(n - m + 2) * T(n - m + 1)

            term_x1 = C_nm * V[n_idx + 1, m_idx + 1] + S_nm * W[n_idx + 1, m_idx + 1]
            term_x2 = C_nm * V[n_idx + 1, m_idx - 1] + S_nm * W[n_idx + 1, m_idx - 1]
            ax += T(0.5) * (-term_x1 + k * term_x2)

            term_y1 = -C_nm * W[n_idx + 1, m_idx + 1] + S_nm * V[n_idx + 1, m_idx + 1]
            term_y2 = -C_nm * W[n_idx + 1, m_idx - 1] + S_nm * V[n_idx + 1, m_idx - 1]
            ay += T(0.5) * (term_y1 + k * term_y2)

            az -= T(n - m + 1) * (C_nm * V[n_idx + 1, m_idx] + S_nm * W[n_idx + 1, m_idx])
        end
    end

    scale = T(mu) / (T(R) * T(R))
    a[1] = scale * ax
    a[2] = scale * ay
    a[3] = scale * az
    return a
end

"""
    cunningham_acceleration_body_fixed(r_bf, mu, R, C, S; degree, order, cs_phase=false) -> SVector{3}

Allocating convenience wrapper to compute body-fixed gravitational perturbation.
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
    cunningham_perturbation(r_inertial, t, mu, R, C, S; degree, order, omega_rot=0.0, cs_phase=false, normalized=false) -> SVector{3}

Calculates the gravitational perturbing acceleration in the **inertial frame**
by rotating the position vector to the body-fixed frame at epoch `t`, evaluating
Cunningham's spherical harmonic recurrence, and rotating the acceleration back to
the inertial frame.

# Arguments
- `r_inertial`: Position vector in the inertial frame [length].
- `t`: Epoch or time elapsed since reference orientation [time].
- `mu`: Gravitational parameter of the central body [length^3/time^2].
- `R`: Equatorial reference radius of the central body [length].
- `C`, `S`: Matrices of harmonic coefficients (1-based: index `[n+1, m+1]`).

# Keyword Arguments
- `degree::Int`: Maximum harmonic degree (default: `size(C, 1) - 1`).
- `order::Int`: Maximum harmonic order (default: `min(degree, size(C, 2) - 1)`).
- `omega_rot`: Angular rotation rate of the central body [rad/time]. Default: `0.0`.
- `cs_phase::Bool`: Include Condon-Shortley phase. Default: `false`.
- `normalized::Bool`: Whether coefficients `C` and `S` are fully normalized. Default: `false`.
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

    # Inertial -> Body-Fixed: R_z(theta) * r_inertial
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

    # Body-Fixed -> Inertial: R_z(-theta) * a_bf
    ax_bf, ay_bf, az_bf = a_bf[1], a_bf[2], a_bf[3]
    ax_i = c * ax_bf - s * ay_bf
    ay_i = s * ax_bf + c * ay_bf
    az_i = az_bf

    return SVector{3, T}(ax_i, ay_i, az_i)
end

"""
    cunningham_perturbation(r_inertial, t, mu, R, gh::GravityHarmonics, omega_rot=0.0) -> SVector{3}

Calculates the gravitational perturbing acceleration using a preconstructed `GravityHarmonics` object.
"""
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
        normalized = false # gh.C is already unnormalized
    )
end

# Callable syntax for GravityHarmonics
function (gh::Types.GravityHarmonics)(
    r_inertial::AbstractVector,
    t::Real,
    mu::Real,
    R::Real,
    omega_rot::Union{Nothing, Real} = 0.0
)
    return cunningham_perturbation(r_inertial, t, mu, R, gh, omega_rot)
end

# Alias
const spherical_harmonic_perturbation = cunningham_perturbation

end # module SphericalHarmonics

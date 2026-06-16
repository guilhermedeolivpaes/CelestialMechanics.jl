# src/Core/Coordinates.jl

"""
    Coordinates

This module provides functions for converting between different coordinate
systems used in orbital dynamics, specifically between classical Keplerian
orbital elements, Delaunay action-angle variables, and Cartesian state vectors 
(position and velocity).

"""
module Coordinates

using LinearAlgebra 
using StaticArrays

import ..Types

export orbital_elements_to_state_vectors, state_vectors_to_orbital_elements, mean_to_true_anomaly, true_to_mean_anomaly, delaunay_to_keplerian, keplerian_to_delaunay, delaunay_to_cartesian, lagrange_to_cartesian, unwrap_angle

"""
    orbital_elements_to_state_vectors(a::Real, e::Real, i::Real, h::Real, g::Real, f::Real, mu::Real)

Converts classical Keplerian orbital elements to Cartesian position and velocity vectors.

# Arguments
- `a::Real`: Semi-major axis.
- `e::Real`: Eccentricity.
- `i::Real`: Inclination in radians.
- `h::Real`: Right ascension of the ascending node (RAAN) in radians.
- `g::Real`: Argument of periapsis in radians.
- `f::Real`: True anomaly in radians.
- `mu::Real`: Standard gravitational parameter of the central body.

# Returns
- `Tuple{SVector{3}, SVector{3}}`: A tuple containing the Cartesian position vector `r_vec` and velocity vector `v_vec`.
"""
function orbital_elements_to_state_vectors(a::Real, e::Real, i::Real, h::Real, g::Real, f::Real, mu::Real)

    # mean motion
    n = sqrt(mu/a^3)

    # eccentric anomaly e
    E = 2*atan(tan(f/2)*sqrt((1-e)/(1+e)))

    # orbit radius
    r = a*(1 - (e*cos(E)))

    # preparation to calculate position and velocity vectors
    R11 = cos(g)*cos(h) - sin(g)*sin(h)*cos(i)
    R12 = -sin(g)*cos(h) - cos(g)*sin(h)*cos(i)
    R21 = cos(g)*sin(h) + sin(g)*cos(h)*cos(i)
    R22 = -sin(g)*sin(h) + cos(g)*cos(h)*cos(i)
    R31 = sin(g)*sin(i)
    R32 = cos(g)*sin(i)

    Ax = R11*a
    Ay = R21*a
    Az = R31*a
    
    Bx = R12*a*(sqrt(1-(e^2)))
    By = R22*a*(sqrt(1-(e^2)))
    Bz = R32*a*(sqrt(1-(e^2)))
   
    # position vector
    X = Ax*(cos(E)-e)+Bx*(sin(E))
    Y = Ay*(cos(E)-e)+By*(sin(E))
    Z = Az*(cos(E)-e)+Bz*(sin(E))

    # velocity vector
    Vx = (((a*n)/r)*(-Ax*sin(E)+Bx*cos(E)))
    Vy = (((a*n)/r)*(-Ay*sin(E)+By*cos(E)))
    Vz = (((a*n)/r)*(-Az*sin(E)+Bz*cos(E)))

    # uses svector to prevent julia from allocating new memory at each integration step, 
    # which can drastically speed up the simulation.
    r_vec = SVector(X, Y, Z)
    v_vec = SVector(Vx, Vy, Vz)
    
    return r_vec, v_vec    
end

# defines the vector k_hat (z-axis) outside the function as a static constant
# this avoids memory allocation at each iteration of the loop
const k_hat = SVector{3}(0.0, 0.0, 1.0)

"""
    state_vectors_to_orbital_elements(r_vectors::AbstractMatrix{<:Real}, v_vectors::AbstractMatrix{<:Real}, mu::Real; tol::Real=1e-10)

Converts a time series of Cartesian state vectors into Keplerian orbital elements and specific energy.

It includes singularity handling for circular and equatorial orbits.

# Arguments
- `r_vectors::AbstractMatrix{<:Real}`: Matrix of position vectors where each column is a 3D state at a given time step.
- `v_vectors::AbstractMatrix{<:Real}`: Matrix of velocity vectors corresponding to `r_vectors`.
- `mu::Real`: Standard gravitational parameter of the central body.

# Keyword Arguments
- `tol::Real`: Tolerance used to detect singularities (circular or equatorial orbits). Defaults to 1e-10.

# Returns
- `Tuple`: Returns seven arrays `(a, e, i, h, g, f, epsilon)` containing the semi-major axis, eccentricity, 
  inclination (degrees), RAAN (degrees), argument of periapsis (degrees), true anomaly (degrees), and specific energy.
"""
function state_vectors_to_orbital_elements(
    r_vectors::AbstractMatrix{<:Real}, 
    v_vectors::AbstractMatrix{<:Real}, 
    mu::Real;
    tol::Real = 1e-10 # tolerance for singularities
    )
    num_elements = size(r_vectors, 2)
    
    # pre-allocate output vectors
    a_vec       = zeros(num_elements)
    e_vec_out   = zeros(num_elements)
    i_vec       = zeros(num_elements)
    h_vec       = zeros(num_elements)
    g_vec       = zeros(num_elements)
    f_vec       = zeros(num_elements)
    epsilon_vec = zeros(num_elements) # specific energy

    for j in 1:num_elements
        # convert columns to svector.
        # from here on, operations do not allocate memory.
        r = SVector{3}(@view r_vectors[:, j])
        v = SVector{3}(@view v_vectors[:, j])
        
        r_norm = norm(r)
        v_norm = norm(v)
        
        # angular momentum vector
        h = cross(r, v)
        h_norm = norm(h)

        # node vector
        n_vec = cross(k_hat, h)
        n_norm = norm(n_vec)

        # eccentricity vector
        e_vec = (1.0 / mu) * ((v_norm^2 - mu / r_norm) * r - dot(r, v) * v)
        e = norm(e_vec)

        # 1. semi-major axis (a)
        # specific energy epsilon = v_norm^2 / 2 - mu / r_norm = -mu / 2a
        # adding energy to the dataframe avoids recalculating it later for plots
        epsilon = (v_norm^2 / 2.0) - (mu / r_norm) 
        epsilon_vec[j] = epsilon 
        a = -mu / (2.0 * epsilon)
        
        # handle parabolic case (energy approx 0)
        if abs(epsilon) < tol
            a = Inf
        end

        # 2. eccentricity (e)
        # (e was already calculated)

        # 3. inclination (i)
        inc = acos(clamp(h[3] / h_norm, -1.0, 1.0))

        # angle variables (local to the loop)
        local h, g, f

        # --- singularity handling ---
        # check for equatorial orbit (i approx 0 or i approx 180)
        is_equatorial = (n_norm < tol)

        # check for circular orbit (e approx 0)
        is_circular = (e < tol)

        if is_equatorial && is_circular
            # case 1: circular-equatorial
            h = 0.0  # undefined, convention
            g = 0.0  # undefined, convention
            f = atan(r[2], r[1]) # true longitude (l)

        elseif is_equatorial && !is_circular
            # case 2: equatorial, but not circular
            h = 0.0  # undefined, convention
            g = atan(e_vec[2], e_vec[1]) # longitude of periapsis (varpi)
            f = atan(dot(r, cross(h, e_vec))/(h_norm * e), dot(r, e_vec)/e) # true anomaly

        elseif !is_equatorial && is_circular
            # case 3: circular, but not equatorial
            h = atan(n_vec[2], n_vec[1]) # raan (defined)
            g = 0.0 # undefined, convention
            
            # argument of latitude (u)
            cos_u = clamp(dot(n_vec, r) / (n_norm * r_norm), -1.0, 1.0)
            f = (r[3] >= 0.0) ? acos(cos_u) : (2π - acos(cos_u))

        else
            # case 4: general (non-equatorial, non-circular)
            h = atan(n_vec[2], n_vec[1]) # raan

            # argument of periapsis (g)
            cos_g = clamp(dot(n_vec, e_vec) / (n_norm * e), -1.0, 1.0)
            g = (e_vec[3] >= 0.0) ? acos(cos_g) : (2π - acos(cos_g))

            # true anomaly (f)
            cos_f = clamp(dot(e_vec, r) / (e * r_norm), -1.0, 1.0)
            vr = dot(r, v) / r_norm # radial component of velocity
            f = (vr >= 0.0) ? acos(cos_f) : (2π - acos(cos_f))
        end

        # store results (converting to degrees)
        a_vec[j]     = a
        e_vec_out[j] = e
        i_vec[j]     = rad2deg(inc)
        h_vec[j] = rad2deg(h)
        g_vec[j] = rad2deg(g)
        f_vec[j]    = rad2deg(f)
    end

    return a_vec, e_vec_out, i_vec, h_vec, g_vec, f_vec, epsilon_vec
end

"""
    mean_to_true_anomaly(l_rad::Float64, e::Float64)

Converts the mean anomaly into the true anomaly by solving Kepler's equation 
using the Newton-Raphson method.

# Arguments
- `l_rad::Float64`: Mean anomaly in radians.
- `e::Float64`: Eccentricity.

# Returns
- `Float64`: True anomaly in radians.
"""
function mean_to_true_anomaly(l_rad::Float64, e::Float64)
    # solves kepler's equation for e (eccentric anomaly)
    # simple newton-raphson
    l = mod2pi(l_rad)
    E = l > pi ? l - e : l + e # initial guess
    for _ in 1:10  # converges very fast for e < 0.9
        f_E = E - e * sin(E) - l
        df_E = 1.0 - e * cos(E)
        E -= f_E / df_E
        if abs(f_E) < 1e-12 break end
    end
    
    # converts e to f (true anomaly)
    # using atan2 to avoid quadrant issues
    sin_f = (sqrt(1.0 - e^2) * sin(E)) / (1.0 - e * cos(E))
    cos_f = (cos(E) - e) / (1.0 - e * cos(E))
    
    return atan(sin_f, cos_f)
end

"""
    true_to_mean_anomaly(f_rad::Real, e::Real)

Converts the true anomaly into the mean anomaly through direct algebraic relations.

# Arguments
- `f_rad::Real`: True anomaly in radians.
- `e::Real`: Eccentricity.

# Returns
- `Real`: Mean anomaly in radians, bounded between 0 and 2π.
"""
function true_to_mean_anomaly(f_rad::Real, e::Real)
    # f -> e (eccentric anomaly)
    cosE = (e + cos(f_rad)) / (1.0 + e * cos(f_rad))
    sinE = (sqrt(1.0 - e^2) * sin(f_rad)) / (1.0 + e * cos(f_rad))
    E = atan(sinE, cosE)
    
    # e -> m (mean anomaly - direct kepler's equation)
    l = E - e * sin(E)
    return mod2pi(l)
end

"""
    canonical_units(mu::Number, R::Number)

Calculates the canonical normalization units (DU, TU, VU) based on the 
central body's gravitational parameter and radius (Reference: Vallado).

# Arguments
- `mu::Number`: Gravitational parameter.
- `R::Number`: Equatorial radius of the central body.

# Returns
- `Types.CanonicalUnits`: A structure containing the normalization factors.
"""
function canonical_units(mu::Number, R::Number)
    # vallado page 237
    DU = R
    TU = sqrt(DU^3 / mu)
    VU = DU / TU
    # calls the original constructor
    return Types.CanonicalUnits(DU, TU, VU)
end

"""
    normalize_state(y_dim::AbstractVector, units::Types.CanonicalUnits)

Normalizes a dimensional Cartesian state vector using canonical units.

# Arguments
- `y_dim::AbstractVector`: A 6-element vector containing dimensional position and velocity `[x, y, z, vx, vy, vz]`.
- `units::Types.CanonicalUnits`: The canonical normalization factors.

# Returns
- `Vector{Float64}`: A normalized 6-element state vector.
"""
function normalize_state(y_dim::AbstractVector, units::Types.CanonicalUnits)
    r_dim = y_dim[1:3]
    v_dim = y_dim[4:6]
    r_adim = r_dim / units.DU # non-dimensional radius
    v_adim = v_dim / units.VU
    
    return [r_adim; v_adim]
end

"""
    denormalize_state(y_adim::AbstractVector, units::Types.CanonicalUnits)

Transforms a non-dimensional (canonical) Cartesian state vector back into physical units (km and km/s).

# Arguments
- `y_adim::AbstractVector`: A 6-element normalized state vector `[rx, ry, rz, vx, vy, vz]`.
- `units::Types.CanonicalUnits`: The canonical normalization factors (DU, TU, VU).

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: A tuple containing the dimensional position (km) and velocity (km/s).
"""
function denormalize_state(y_adim::AbstractVector, units::Types.CanonicalUnits)
    r_adim = y_adim[1:3]
    v_adim = y_adim[4:6]
    r_phys = r_adim .* units.DU
    v_phys = v_adim .* units.VU
    return r_phys, v_phys
end

"""
    keplerian_to_delaunay(a, e, i, h, g, l, mu)

Converts Keplerian orbital elements to Delaunay action-angle variables.

# Arguments
- `a`, `e`, `i`, `h`, `g`, `l`: Semi-major axis, eccentricity, inclination, RAAN, argument of periapsis, and mean anomaly.
- `mu`: Standard gravitational parameter of the central body.

# Returns
- `Tuple`: Delaunay variables `(L, G, H, l, g, h)`.
"""
function keplerian_to_delaunay(a, e, i, h, g, l, mu)
    L = sqrt(mu * a)
    G = L * sqrt(1 - e^2)
    H = G * cos(i)
    return L, G, H, l, g, h
end

"""
    delaunay_to_keplerian(L, G, H, l, g, h, mu)

Converts Delaunay action-angle variables to classical Keplerian elements.

# Arguments
- `L`, `G`, `H`, `l`, `g`, `h`: The Delaunay momenta and coordinates.
- `mu`: Standard gravitational parameter of the central body.

# Returns
- `Tuple`: Keplerian elements `(a, e, i, h, g, l)` where `l` is the mean anomaly.
"""
function delaunay_to_keplerian(L, G, H, l, g, h, mu)
    a = L^2 / mu
    e = sqrt(max(0.0, 1.0 - (G/L)^2))
    i = acos(clamp(H/G, -1.0, 1.0))
    return a, e, i, h, g, l 
end

"""
    delaunay_to_cartesian(L, G, H, l, g, h, mu)

Transforms Delaunay variables directly into Cartesian position and velocity vectors.

# Arguments
- `L`, `G`, `H`, `l`, `g`, `h`: The Delaunay variables.
- `mu`: Standard gravitational parameter of the central body.

# Returns
- `Tuple{SVector{3}, SVector{3}}`: Cartesian position and velocity vectors.
"""
function delaunay_to_cartesian(L, G, H, l, g, h, mu)
    a, e, i, Om, om, M = delaunay_to_keplerian(L, G, H, l, g, h, mu)
    f = mean_to_true_anomaly(M, e)
     r, v = orbital_elements_to_state_vectors(a, e, i, Om, om, f, mu)
    return SVector{3}(r), SVector{3}(v)
end

"""
    lagrange_to_cartesian(a, e, i, Om, om, l, mu)

Transforms Lagrange planetary variables (Keplerian elements using mean anomaly) 
directly into Cartesian position and velocity vectors.

# Arguments
- `a`, `e`, `i`, `Om`, `om`, `l`: Semi-major axis, eccentricity, inclination, RAAN, argument of periapsis, and mean anomaly.
- `mu`: Standard gravitational parameter of the central body.

# Returns
- `Tuple{SVector{3}, SVector{3}}`: Cartesian position and velocity vectors.
"""
function lagrange_to_cartesian(a, e, i, Om, om, l, mu)
    f = mean_to_true_anomaly(l, e)
    r, v = orbital_elements_to_state_vectors(a, e, i, Om, om, f, mu)
    return SVector{3}(r), SVector{3}(v)
end


"""
    unwrap_angle(v::AbstractVector{<:Real})

Removes discontinuities (jumps of ±2π) from a time series of angular values in radians, 
producing a continuous, monotonic signal suitable for linear fitting of secular rates.

This is essential for extracting secular precession rates (e.g., dg/dt, dh/dt) from 
numerical propagation outputs where angles are bounded in [0, 2π) or [-π, π).

# Arguments
- `v::AbstractVector{<:Real}`: A vector of angular values in radians.

# Returns
- `Vector{Float64}`: A continuous (unwrapped) angular time series in radians.

# Example
```julia
g_rad = deg2rad.(df.g_deg)
g_continuous = unwrap_angle(g_rad)
slope = (g_continuous[end] - g_continuous[1]) / (t[end] - t[1])  # rad/s
```
"""
function unwrap_angle(v::AbstractVector{<:Real})
    n = length(v)
    out = Vector{Float64}(undef, n)
    out[1] = Float64(v[1])
    for i in 2:n
        d = v[i] - v[i-1]
        if d > π
            d -= 2π
        elseif d < -π
            d += 2π
        end
        out[i] = out[i-1] + d
    end
    return out
end

end # end of module

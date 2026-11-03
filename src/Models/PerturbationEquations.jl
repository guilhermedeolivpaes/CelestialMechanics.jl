# src/Models/PerturbationEquations.jl

module PerturbationEquations

using ..Constants
using ..Types
using ..SphericalHarmonics: cunningham_perturbation, spherical_harmonic_perturbation,
                            cunningham_acceleration_body_fixed, cunningham_acceleration_body_fixed!,
                            cunningham_harmonics!, GravityHarmonics
using LinearAlgebra, StaticArrays, Unitful, Unitful.DefaultSymbols, UnitfulAstro

export cunningham_perturbation, spherical_harmonic_perturbation, GravityHarmonics,
       j2_potential, j2_perturbation, nbody_perturbation,
       SolarRadiationPressure, eclipse_factor, srp_perturbation,
       harris_priester_density, drag_acceleration, drag_perturbation


"""
    j2_potential(r_vector, mu, R, j2) -> Float64

Calculates the perturbing potential energy due to the J2 zonal harmonic.
"""
function j2_potential(r_vector, mu, R, j2)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^2 * j2 * mu * (3*z^2 - r^2)) / (2 * r^5)
end

""" 
    j2_perturbation(r_vector, mu, R, j2) -> SVector{3}

Calculates the perturbing acceleration due to the J2 zonal harmonic (polar flattening).

# Arguments
- `r_vector`: Satellite position vector in the inertial frame [length].
- `mu`: Gravitational parameter of the central body [length^3/time^2].
- `R`: Equatorial radius of the central body [length].
- `j2`: J2 coefficient (dimensionless).

# Returns
- `SVector{3}`: Perturbing acceleration vector [length/time^2].
"""
function j2_perturbation(r_vector, mu, R, j2)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (3*j2*R^2*mu*x*(5*z^2-r^2))/(2*r^7)
    ay = (3*j2*R^2*mu*y*(5*z^2-r^2))/(2*r^7)
    az = (3*j2*R^2*mu*z*(5*z^2-3*r^2))/(2*r^7)
    return SVector(ax, ay, az)
end
# --- other perturbations ---

"""
    nbody_perturbation(r_sat_vector::AbstractVector, r_body_vector::AbstractVector, mu_body::Real) -> SVector{3}

Calculates the perturbation of an n-th body on the satellite. Accepts any type
of input vector and converts it to SVector internally for performance.

# Arguments
- `r_sat_vector`: Position vector of the satellite.
- `r_body_vector`: Position vector of the perturbing body.
- `mu_body`: Gravitational parameter of the perturbing body.

# Returns
- `SVector{3}`: Perturbing acceleration vector.
"""
function nbody_perturbation(r_sat_vector::AbstractVector, r_body_vector::AbstractVector, mu_body::Real)
    # ensures that both vectors are static (svectors) for the operations
    r_sat_svector = SVector{3}(r_sat_vector)
    r_body_svector = SVector{3}(r_body_vector)

    # pre-calculates the relative satellite-perturbing body vector only once.
    r_sat_to_body = r_sat_svector - r_body_svector

    norm_r_sat_to_body = norm(r_sat_svector - r_body_svector)
    norm_r_body = norm(r_body_svector)
    
    # calculates the third body acceleration: direct effect minus the indirect effect
    return -mu_body * (r_sat_to_body / norm_r_sat_to_body^3 + r_body_svector / (norm_r_body^3))
end

# --- srp ---
"""
    SolarRadiationPressure(dist_to_sun::Unitful.Length) -> Float64

Calculates the solar radiation pressure (P_SR) in N/m^2.

# Arguments
- `dist_to_sun`: Distance to the sun with units.

# Returns
- `Float64`: Solar radiation pressure without units.
"""
function SolarRadiationPressure(dist_to_sun::Unitful.Length)
    # convert distance to meters
    dist_m = uconvert(u"m", dist_to_sun)

    # solar intensity according to the inverse square law.
    I_effective = Constants.I0_SI * (Constants.AU_IN_M / dist_m)^2   # w/m^2 (quantity)

    # radiation pressure: p = i/c
    P_SR_W_s_m3 = I_effective / Constants.C_SI       # w s/m^3 (quantity) is equal to n/m^2, because w = j/s and j = n m 
    P_SR = uconvert(u"N/m^2", P_SR_W_s_m3) # convert to n/m^2 (just for clarity)
    
    # returns as pure float64
    return ustrip(u"N/m^2", P_SR)
end


"""
    eclipse_factor(r_vector_sat::AbstractVector, R_body::Real, r_vector_sun::AbstractVector) -> Float64

Calculates the eclipse factor (0.0 for shadow, 1.0 for illuminated) using a cylindrical shadow model.

# Arguments
- `r_vector_sat`: Position vector of the satellite relative to the central body.
- `R_body`: Radius of the central body.
- `r_vector_sun`: Position vector of the sun relative to the central body.

# Returns
- `Float64`: 0.0 if in shadow, 1.0 if illuminated.
"""
function eclipse_factor(r_vector_sat::AbstractVector, R_body::Real, r_vector_sun::AbstractVector)
    r_sun_body_norm = norm(r_vector_sun)
    u_body_to_sun = r_vector_sun / r_sun_body_norm

    # projection of satellite position onto the sun direction vector
    h = dot(r_vector_sat, u_body_to_sun)
    # perpendicular distance from the satellite to the sun direction vector
    d = norm(r_vector_sat - h * u_body_to_sun)

    if h < 0 && d < R_body
        return 0.0  # in shadow
    else
        return 1.0  # illuminated
    end
end


"""
    srp_perturbation(r_vector, R, alpha, CR, r_sun_vector; dist_scale=1.0, use_shadow_model=true) -> SVector{3}

Calculates the solar radiation pressure (SRP) perturbation, considering the direction from the sun to the satellite.

# Arguments
- `r_vector`: Position vector of the satellite in km.
- `R`: Radius of the central body in km.
- `alpha`: Area/mass ratio in m^2/kg.
- `CR`: Reflectivity coefficient.
- `r_sun_vector`: Position vector of the sun in km.
- `dist_scale`: Distance scaling factor. Defaults to 1.0.
- `use_shadow_model`: If true, applies a cylindrical shadow model. Defaults to true.

# Returns
- `SVector{3}`: Acceleration vector in km/s^2 as float64.
"""
function srp_perturbation(
    r_vector::AbstractVector,
    R::Real,
    alpha::Real,
    CR::Real,
    r_sun_vector::AbstractVector;
    dist_scale::Real = 1.0,
    use_shadow_model::Bool = true
    )

    # vector of relative position of the sun to the satellite
    r_sun_sat = r_vector - r_sun_vector # km

    # distance - multiply by dist_scale to ensure it's in km
    # if it is physical mode, dist_scale = 1.0. if it is canonical, dist_scale = du
    dist_sat_to_sun_phys = norm(r_sun_sat) * dist_scale

    # pressure - solar radiation pressure always receives physical kilometers
    P_SR = SolarRadiationPressure(dist_sat_to_sun_phys * u"km")

    # lighting factor
    nu = 1.0
    if use_shadow_model
        nu = eclipse_factor(ustrip.(r_vector), ustrip(R), ustrip.(r_sun_vector))
    end

    # if shadow returns null vector
    if iszero(nu)
        return SVector{3}(0.0, 0.0, 0.0)  # pure float64 in km/s^2
    end
    
    # magnitude of acceleration - alpha is already the 'alpha_solver' (scaled or not)
    # the division by 1000 continues to convert m/s^2 to km/s^2 (or equivalent dimensionless number).
    acel_magnitude_m_s2 = nu * P_SR * CR * alpha       # m/s^2
    acel_magnitude_km_s2 = acel_magnitude_m_s2 / 1000  # km/s^2 (float64)
    

    # direction: from the sun to the satellite
    unit_vec_sun_to_sat = r_sun_sat / norm(r_sun_sat)

    # acceleration vector: removed the negative sign because the vector already points from the sun towards the satellite.
    # ref.: o. montenbruck, and e. gill, _satellite orbits: models, methods and applications_, 2012, p.77-79.
    a_srp = acel_magnitude_km_s2 .* unit_vec_sun_to_sat

    return SVector{3}(a_srp)  # float64 in km/s^2
end

# --- atmospheric drag ---
# --- harris-priester atmospheric density model ---
#
# modified harris-priester model with diurnal bulge.
#
# ref.: montenbruck, o. and gill, e., *satellite orbits: models, methods
#       and applications*, springer, 2000/2012, section 3.5, pp. 89-91.
#
# ref.: harris, i. and priester, w., "time-dependent structure of the upper
#       atmosphere", journal of the atmospheric sciences, 19, 1962, pp. 286-301.
#
# the density table below is from montenbruck & gill, table 3.1 (mean solar activity).
# the orekit implementation (apache 2.0 license) was used as cross-reference for the
# table values and the algorithm structure.

"""
    HP_DENSITY_TABLE

harris-priester min/max density table for mean solar activity.
columns: altitude [km], rho_min [kg/m^3], rho_max [kg/m^3].

ref.: montenbruck & gill, *satellite orbits*, table 3.1.
"""
const HP_DENSITY_TABLE = [
#   h [km]    rho_min [kg/m^3]   rho_max [kg/m^3]
    100.0     4.974e-07          4.974e-07
    120.0     2.490e-08          2.490e-08
    130.0     8.377e-09          8.710e-09
    140.0     3.899e-09          4.059e-09
    150.0     2.122e-09          2.215e-09
    160.0     1.263e-09          1.344e-09
    170.0     8.008e-10          8.758e-10
    180.0     5.283e-10          6.010e-10
    190.0     3.617e-10          4.297e-10
    200.0     2.557e-10          3.162e-10
    210.0     1.839e-10          2.396e-10
    220.0     1.341e-10          1.853e-10
    230.0     9.949e-11          1.455e-10
    240.0     7.488e-11          1.157e-10
    250.0     5.709e-11          9.308e-11
    260.0     4.403e-11          7.555e-11
    270.0     3.430e-11          6.182e-11
    280.0     2.697e-11          5.095e-11
    290.0     2.139e-11          4.226e-11
    300.0     1.708e-11          3.526e-11
    320.0     1.099e-11          2.511e-11
    340.0     7.214e-12          1.819e-11
    360.0     4.824e-12          1.337e-11
    380.0     3.274e-12          9.955e-12
    400.0     2.249e-12          7.492e-12
    420.0     1.558e-12          5.684e-12
    440.0     1.091e-12          4.355e-12
    460.0     7.701e-13          3.362e-12
    480.0     5.474e-13          2.612e-12
    500.0     3.916e-13          2.042e-12
    520.0     2.819e-13          1.605e-12
    540.0     2.042e-13          1.267e-12
    560.0     1.488e-13          1.005e-12
    580.0     1.092e-13          7.997e-13
    600.0     8.070e-14          6.390e-13
    620.0     6.012e-14          5.123e-13
    640.0     4.519e-14          4.121e-13
    660.0     3.430e-14          3.325e-13
    680.0     2.632e-14          2.691e-13
    700.0     2.043e-14          2.185e-13
    720.0     1.607e-14          1.779e-13
    740.0     1.281e-14          1.452e-13
    760.0     1.036e-14          1.190e-13
    780.0     8.496e-15          9.776e-14
    800.0     7.069e-15          8.059e-14
    840.0     4.680e-15          5.741e-14
    880.0     3.200e-15          4.210e-14
    920.0     2.210e-15          3.130e-14
    960.0     1.560e-15          2.360e-14
   1000.0     1.150e-15          1.810e-14
]


"""
    harris_priester_density(r_sat_km, r_sun_km, R_body_km; n_exp=4) -> Float64

computes the atmospheric density using the modified harris-priester model.

the model interpolates between minimum (nightside) and maximum (dayside) density
profiles using the cosine of the half-angle between the satellite position and
the diurnal bulge apex. the bulge apex is the sun direction lagged by 30 degrees
in the equatorial plane (local time lag of ~2 hours).

ref.: montenbruck & gill, *satellite orbits*, 2000/2012, section 3.5, pp. 89-91.
ref.: harris & priester, j. atmos. sci., 19, 1962, pp. 286-301.

# arguments
- `r_sat_km::AbstractVector`: satellite position vector in km (inertial frame).
- `r_sun_km::AbstractVector`: sun position vector in km (same inertial frame).
- `R_body_km::Real`: equatorial radius of the central body in km.
- `n_exp::Int`: cosine exponent. recommended range: 2 (equatorial) to 6 (polar).
  default is 4 (mid-inclination). see montenbruck & gill, p. 91.

# returns
- `Float64`: atmospheric density in kg/m^3.
"""
function harris_priester_density(r_sat_km::AbstractVector, r_sun_km::AbstractVector,
                                 R_body_km::Real; n_exp::Int=4)

    # altitude above spherical earth [km]
    r_mag = norm(r_sat_km)
    altitude_km = r_mag - R_body_km

    # table bounds
    h_min = HP_DENSITY_TABLE[1, 1]
    h_max = HP_DENSITY_TABLE[end, 1]

    if altitude_km < h_min
        return HP_DENSITY_TABLE[1, 2]  # return min density at lowest altitude
    end

    if altitude_km > h_max
        return 0.0
    end

    # --- diurnal bulge apex direction ---
    # the bulge apex lags the subsolar point by 30 degrees in right ascension
    # (approximately 2 hours in local time).
    # rotation is applied in the equatorial plane (around z-axis).
    lag_rad = deg2rad(30.0)
    sin_lag = sin(lag_rad)
    cos_lag = cos(lag_rad)

    sun_dir = r_sun_km / norm(r_sun_km)
    # rotate sun direction by +30 deg around z to get bulge apex
    bulge_x = sun_dir[1] * cos_lag + sun_dir[2] * sin_lag
    bulge_y = -sun_dir[1] * sin_lag + sun_dir[2] * cos_lag
    bulge_z = sun_dir[3]
    bulge_mag = sqrt(bulge_x^2 + bulge_y^2 + bulge_z^2)

    # cosine of angle psi between satellite and bulge apex
    sat_dir = r_sat_km / r_mag
    cos_psi = (sat_dir[1] * bulge_x + sat_dir[2] * bulge_y + sat_dir[3] * bulge_z) / bulge_mag

    # cos^n(psi/2) using the half-angle identity: cos^2(psi/2) = (1 + cos(psi))/2
    cos2_half = max(0.0, (1.0 + cos_psi) / 2.0)
    cos_half = sqrt(cos2_half)
    cos_power = cos_half > 1.0e-12 ? cos2_half * cos_half^(n_exp - 2) : 0.0

    # --- altitude interpolation ---
    # find the table bracket
    ia = 1
    n_rows = size(HP_DENSITY_TABLE, 1)
    while ia < n_rows - 1 && altitude_km > HP_DENSITY_TABLE[ia + 1, 1]
        ia += 1
    end

    # fractional height for exponential interpolation
    h_lo = HP_DENSITY_TABLE[ia, 1]
    h_hi = HP_DENSITY_TABLE[ia + 1, 1]
    dH = (h_lo - altitude_km) / (h_lo - h_hi)

    # min density (nightside): exponential interpolation between table rows
    rho_min_lo = HP_DENSITY_TABLE[ia, 2]
    rho_min_hi = HP_DENSITY_TABLE[ia + 1, 2]
    rho_min = rho_min_lo * (rho_min_hi / rho_min_lo)^dH

    if cos_power <= 0.0
        return rho_min
    end

    # max density (dayside): exponential interpolation between table rows
    rho_max_lo = HP_DENSITY_TABLE[ia, 3]
    rho_max_hi = HP_DENSITY_TABLE[ia + 1, 3]
    rho_max = rho_max_lo * (rho_max_hi / rho_max_lo)^dH

    # harris-priester formula: rho = rho_min + (rho_max - rho_min) * cos^n(psi/2)
    return rho_min + (rho_max - rho_min) * cos_power
end


"""
    drag_acceleration(r_vector, v_vector, rho, cd, am, omega_rot) -> SVector{3}

computes the perturbing acceleration due to atmospheric drag given the atmospheric
density `rho` in kg/m^3.

expects position and velocity in km and km/s (physical units), area-to-mass ratio `am`
in m^2/kg, and atmospheric rotation rate `omega_rot` in rad/s.
returns acceleration in km/s^2.
"""
function drag_acceleration(r_vector::AbstractVector, v_vector::AbstractVector,
                           rho::Real, cd::Real, am::Real, omega_rot::Real)
    T = promote_type(eltype(r_vector), eltype(v_vector), typeof(rho))

    if rho <= 0.0
        return SVector{3,T}(0.0, 0.0, 0.0)
    end

    # velocity relative to the co-rotating atmosphere: v_rel = v - omega x r
    v_atm = SVector{3,T}(-omega_rot * r_vector[2],
                          omega_rot * r_vector[1],
                          0.0)
    v_rel = v_vector - v_atm
    v_rel_mag = norm(v_rel)

    # unit conversion factor: 0.5 * 1e3
    # (density in kg/m^3, cd [-], am in m^2/kg, v_rel in km/s -> km/s^2 requires * 1e3)
    factor = -0.5e3 * rho * cd * am * v_rel_mag

    return SVector{3,T}(factor * v_rel[1], factor * v_rel[2], factor * v_rel[3])
end

"""
    drag_perturbation(r_vector, v_vector, r_sun, cd, am, R_body_km, omega_rot; n_exp=4) -> SVector{3}

computes the perturbing acceleration due to atmospheric drag using the
modified harris-priester density model.

the acceleration formula is identical to `drag_perturbation` (vallado, 2013,
eq. 8-41), but the density is computed from the harris-priester model instead
of the static exponential model.

expects position and velocity in km and km/s (physical units).
returns acceleration in km/s^2.

# arguments
- `r_vector::AbstractVector`: satellite position vector in km (inertial frame).
- `v_vector::AbstractVector`: satellite velocity vector in km/s.
- `r_sun::AbstractVector`: sun position vector in km (same inertial frame).
- `cd::Real`: drag coefficient (dimensionless).
- `am::Real`: area-to-mass ratio in m^2/kg.
- `R_body_km::Real`: equatorial radius of the central body in km.
- `omega_rot::Real`: angular rotation rate of the atmosphere in rad/s.
- `n_exp::Int`: cosine exponent for harris-priester (default 4).

# returns
- `SVector{3}`: drag acceleration vector in km/s^2.
"""
function drag_perturbation(r_vector::AbstractVector, v_vector::AbstractVector,
                              r_sun::AbstractVector,
                              cd::Real, am::Real, R_body_km::Real, omega_rot::Real;
                              n_exp::Int=4)
    # atmospheric density [kg/m^3] via modified Harris-Priester
    rho = harris_priester_density(r_vector, r_sun, R_body_km; n_exp=n_exp)
    return drag_acceleration(r_vector, v_vector, rho, cd, am, omega_rot)
end

end # end of module

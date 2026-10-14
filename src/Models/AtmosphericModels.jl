# src/Models/AtmosphericModels.jl

"""
    AtmosphericModels

Atmospheric density models used by the drag perturbation.

The drag acceleration itself lives in `PerturbationEquations.drag_acceleration`; this module only
answers the question *"what is the density at this position and instant?"* through the single
generic function [`atmospheric_density`](@ref), dispatched on the model type:

| model                         | implementation                                   | needs Sun ephemeris |
|-------------------------------|--------------------------------------------------|---------------------|
| `nothing` (default)           | Harris-Priester (native)                         | yes                 |
| `HarrisPriesterAtmosphere`    | Harris-Priester (native)                         | yes                 |
| `NRLMSISE00Atmosphere`        | `SatelliteToolboxAtmosphericModels.jl` extension | no                  |

The NRLMSISE-00 backend is provided by the package extension `CelestialMechanicsNRLMSISE00Ext`,
loaded automatically when the user runs `using SatelliteToolboxAtmosphericModels`.
"""
module AtmosphericModels

using LinearAlgebra
using StaticArrays

using ..Types
using ..PerturbationEquations

export atmospheric_density, eci_to_geodetic, gmst_angle

# --- constants ---
# WGS-84 ellipsoid (NIMA TR8350.2)
const WGS84_A_KM = 6378.137
const WGS84_F    = 1.0 / 298.257223563
const WGS84_E2   = WGS84_F * (2.0 - WGS84_F)

# J2000 mean obliquity of the ecliptic: 84381.448 arcsec (IAU 1976)
const J2000_OBLIQUITY = deg2rad(84381.448 / 3600.0)

const JD_J2000 = 2451545.0

"""
    needs_sun(model) -> Bool

Whether the atmosphere model requires the Sun position (used to decide whether the Sun
ephemeris must be interpolated by `Dynamics.set_perturbation`).
"""
needs_sun(::Nothing) = true
needs_sun(::Types.HarrisPriesterAtmosphere) = true
needs_sun(::Types.AbstractAtmosphereModel) = false

# --- reference frame utilities ---

"""
    gmst_angle(jd_ut1) -> Real

Greenwich Mean Sidereal Time [rad] in the interval [0, 2π), IAU-82 model.

ref.: Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., 2013, eq. 3-47.
"""
function gmst_angle(jd_ut1::Real)
    T = (jd_ut1 - JD_J2000) / 36525.0
    # seconds of time
    θ_s = 67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * T +
          0.093104 * T^2 - 6.2e-6 * T^3
    # 1 s of time = 1/240 deg
    return mod(deg2rad(θ_s / 240.0), 2π)
end

# ecliptic J2000 -> equatorial J2000: rotation about x by -ε
@inline function _ecliptic_to_equatorial(r::AbstractVector)
    s, c = sincos(J2000_OBLIQUITY)
    return SVector(r[1], c * r[2] - s * r[3], s * r[2] + c * r[3])
end

# equatorial inertial -> earth-fixed: rotation about z by +θ (gmst)
@inline function _inertial_to_ecef(r::AbstractVector, θ)
    s, c = sincos(θ)
    return SVector(c * r[1] + s * r[2], -s * r[1] + c * r[2], r[3])
end

"""
    ecef_to_geodetic(r_ecef_km) -> (lat_rad, lon_rad, h_km)

Converts an Earth-fixed position [km] to WGS-84 geodetic latitude, longitude and altitude
using a fixed-point iteration on the latitude (converges to < 1e-12 rad in a few steps).
Generic in the element type (compatible with automatic differentiation).
"""
function ecef_to_geodetic(r::AbstractVector)
    x, y, z = r[1], r[2], r[3]
    lon = atan(y, x)
    p   = sqrt(x^2 + y^2)

    # initial guess (bowring-like) and fixed-point iteration
    lat = atan(z, p * (1.0 - WGS84_E2))
    for _ in 1:6
        sφ = sin(lat)
        N  = WGS84_A_KM / sqrt(1.0 - WGS84_E2 * sφ^2)
        lat = atan(z + WGS84_E2 * N * sφ, p)
    end

    sφ, cφ = sincos(lat)
    N = WGS84_A_KM / sqrt(1.0 - WGS84_E2 * sφ^2)
    # numerically stable altitude formula (valid also near the poles)
    h = p * cφ + z * sφ - WGS84_A_KM * sqrt(1.0 - WGS84_E2 * sφ^2)

    return lat, lon, h
end

"""
    eci_to_geodetic(r_km, jd_utc; frame=:J2000) -> (lat_rad, lon_rad, h_km)

Converts an inertial position [km] at the Julian Day `jd_utc` to WGS-84 geodetic coordinates.
The Earth rotation is modelled by GMST only (UTC ≈ UT1, no precession/nutation/polar motion).
`frame` may be `:J2000` (equatorial) or `:ECLIPJ2000` (ecliptic).
"""
function eci_to_geodetic(r_km::AbstractVector, jd_utc::Real; frame::Symbol = :J2000)
    r_eq = frame === :ECLIPJ2000 ? _ecliptic_to_equatorial(r_km) : SVector(r_km[1], r_km[2], r_km[3])
    r_ecef = _inertial_to_ecef(r_eq, gmst_angle(jd_utc))
    return ecef_to_geodetic(r_ecef)
end

# --- density models ---

"""
    atmospheric_density(model, r_km, t_s, r_sun_km, R_body_km) -> Real

Atmospheric mass density [kg/m^3] at the inertial position `r_km` [km] and time `t_s`
[s, elapsed since the simulation epoch].

- `r_sun_km`: Sun position [km] in the same frame (only used by Harris-Priester; may be `nothing`
  for models that do not need it).
- `R_body_km`: equatorial radius of the central body [km] (only used by Harris-Priester).
"""
function atmospheric_density end

atmospheric_density(::Nothing, r_km, t_s, r_sun_km, R_body_km) =
    atmospheric_density(Types.HarrisPriesterAtmosphere(), r_km, t_s, r_sun_km, R_body_km)

function atmospheric_density(m::Types.HarrisPriesterAtmosphere, r_km, t_s, r_sun_km, R_body_km)
    isnothing(r_sun_km) && error("Harris-Priester density requires the Sun ephemeris (SPICE or AnalyticEphemeris).")
    return PerturbationEquations.harris_priester_density(r_km, r_sun_km, R_body_km; n_exp = m.n_exp)
end

function atmospheric_density(m::Types.NRLMSISE00Atmosphere, r_km, t_s, r_sun_km, R_body_km)
    jd = m.epoch_jd + t_s / 86400.0
    lat, lon, h_km = eci_to_geodetic(r_km, jd; frame = m.frame)
    # NRLMSISE-00 throws for negative altitudes; the reentry callback stops the integration
    # at 100 km anyway, so clamping only protects intermediate integrator stages.
    h_m = max(h_km * 1000.0, zero(h_km))
    return nrlmsise00_density(m, jd, h_m, lat, lon)
end

"""
    nrlmsise00_density(model::NRLMSISE00Atmosphere, jd, h_m, lat_rad, lon_rad) -> Real

Total mass density [kg/m^3] from NRLMSISE-00. The method is implemented by the package
extension activated with `using SatelliteToolboxAtmosphericModels`.
"""
nrlmsise00_density(model, args...) = error(
    "NRLMSISE00Atmosphere requires the optional package SatelliteToolboxAtmosphericModels.jl.\n" *
    "Install it with `] add SatelliteToolboxAtmosphericModels` and load it with " *
    "`using SatelliteToolboxAtmosphericModels` before running the simulation."
)

end # end of module

# src/Models/Constants.jl

"""
    Constants

Module that centralizes the physical parameters of celestial bodies, generic spacecraft attributes, 
and fundamental physical constants used throughout the symbolic and numerical calculations.
"""
module Constants

using Logging, Unitful, UnitfulAstro, Unitful.DefaultSymbols

export BODIES_DATA, I0_SI, C_SI, AU_IN_M, N_MOON 

"""
    BODIES_DATA::Dict{Symbol, NamedTuple}

A global dictionary containing physical parameters and ephemeris identifiers (SPICE IDs) for various celestial bodies.
The data includes gravitational parameters (`mu`), reference radii (`R`), harmonic coefficients (`j2`, `c22`, etc.), 
and rotational velocities (`omega_rot`).

# Supported Bodies
- `:earth`: Earth parameters (J2, J3, J4, C22, S22).
- `:sun`: Sun parameters.
- `:moon`: High-fidelity lunar gravity model including zonal and tesseral harmonics up to degree and order 18.
- `:mercury`, `:jupiter`: Basic planetary parameters.
- `:didymos_system`, `:didymos`, `:dimorphos`: Parameters for the Didymos-Dimorphos binary asteroid system.
- `:vilhena`, `:apophis`: Parameters for specific small bodies.
"""
const BODIES_DATA = Dict(
    :earth => (name=:earth, mu=398600.4418*km^3/s^2, R=6378.137*km, a_sun_AU=1.0u"AU", 
                j2=0.00108263, j3=-2.5327e-6, j4=-1.6196e-6, j5=-2.276e-7, j6=5.4068e-7, 
                c22=1.5744e-6, s22=-0.9038e-6, omega_rot=7.2921151467e-5*rad/s, spice_id="EARTH"),
    :sun   => (name=:sun, mu=1.32712440018e11*km^3/s^2, R=696340.0*km, j2=2.0e-7, spice_id="SUN"),
    :moon  => (name=:moon, mu=4.90280012e3*km^3/s^2, R=1738.1*km, d_AU=1.0u"AU", a_earth_km=384400.0*km, e_earth=0.0549,
                j2=2.032365e-4, j3=8.585355e-6, j4=-9.860033e-6, j5=8.393511e-7, j6=-1.331149e-5, 
                j7=-2.205653e-5, j8=-9.434727e-6, j9=1.511956e-5, j10=3.863970e-6, j11=5.130603e-6, j12=9.103736e-6,

                c22=2.235491e-5, c31=2.849956e-5, c32=4.847437e-6, c33=1.712582e-6, 
                s22=1.535185e-8, s31=5.953103e-6, s32=1.689177e-6, s33=-2.564964e-7,
                c41=-5.0879936e-7, c42=7.84175859e-8, c43=5.92099402e-8, c44=-3.98407411e-9,
                s41=-4.49144872e-7, s42=1.48177868e-7, s43=-1.20077667e-8, s44=6.52571425e-9, 
                omega_rot=2.6617e-6*rad/s, spice_id="MOON"), # source: gagg and sandro, 2025 a semi-analytic theory for preliminary.... 
    :mercury => (name=:mercury, mu=22032.09*km^3/s^2, R=2439.7*km, j2=6.0e-5, j3=1.188e-5, j4=1.95e-5, j5=0.0, j6=0.0, c22=1.24973e-5,
                d_AU=0.387u"AU", e_sun=0.206, spice_id="MERCURY"),
    :jupiter => (name=:jupiter, mu=1.26686534e8*km^3/s^2, R=71492.0*km, j2=0.014736, j3=0.0, j4=-0.000587, j5=0.0, j6=0.0, spice_id="JUPITER"),
    # epoch 2461000.5 (2025-nov-21.0) tdb reference: jpl 219 (heliocentric iau76/j2000 ecliptic) -> https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=didymos
    :didymos_system => (name=:didymos_system, a_sun_AU=1.642564399038672u"AU", e_sun=0.3832284321571334, i_sun=3.414073915788352u"°", spice_id="20065803"),
    :didymos => (name=:didymos, mu=3.489324040e-8*km^3/s^2, R=0.39*km, j2=8.35e-2, j3=-2.27e-2, j4=-5.5e-3, j6=4.0e-5, 
                 c22=1.82e-3, c31=-6.22e-3, s31=-5.94e-3, c32=-7.10e-5, s32=1.01e-3, c33=8.47e-5, s33=-3.07e-4, 
                 c42=-2.06e-5, c44=5.91e-7, omega_rot=0.0007725088531821048*rad/s, spice_id="920065803"), 
    # ref. didymos: robust stability and mission performance of a cubesat orbiting the didymos binary asteroid system, fodde et al. 2023
    # the odd j's in didymos are considered zero when it is modeled as a homogeneous triaxial ellipsoid or a simple flattened sphere, because the mass distribution 
    # above the equator is a "mirror" of the distribution below, mathematically canceling the integrals of the odd terms of the potential.
    :dimorphos => (name=:dimorphos, mu=2.98047e-10*km^3/s^2, R=0.082*km, a_didymos=1.19*km, e_didymos=0.049, spice_id="120065803"),
    :vilhena => (name=:vilhena, mu=6.475626538e-9*km^3/s^2, R=2.1595*km, j2=0.1*sqrt(5), j3=0.0, j4=0.0, j5=0.0, j6=0.0, spice_id="20034604"),
    :apophis => (name=:apophis, mu=2.86e-9*km^3/s^2, R=0.17*km, j2=0.1344360544, j3=-0.0303639272, j4=-0.0463008733, 
                c22=0.0472123118, c31=-0.0003185559, c32=-0.0082529794, c33=-0.0030055538,
                c41=-0.0014941336, c42 = -0.0045027695, c43=0.0001455317, c44=0.0004055808,
                s22=0.0000013070, s31=-0.0052770957, s32=-0.0040155691, s33=0.0005704136,
                s41=-0.0017708778, s42=-0.0005636565, s43=0.0000550923, s44=0.0000371009,
                 omega_rot=5.711155929300816e-5*rad/s, spice_id="20099942"),
)

"""
    I0_SI

Solar constant at 1 Astronomical Unit (AU), expressed in Watts per square meter [W/m^2].
"""
const I0_SI = 1367.0u"W/m^2" # solar constant at 1 au [w/m^2]

"""
    C_SI

Speed of light in vacuum, expressed in meters per second [m/s].
"""
const C_SI = 299792458.0u"m/s" # speed of light [m/s]

"""
    AU_IN_KM

Conversion factor: one Astronomical Unit (AU) expressed in kilometers.
"""
const AU_IN_KM = uconvert(u"km", 1.0u"AU") # astronomical unit in kilometers

"""
    AU_IN_M

Conversion factor: one Astronomical Unit (AU) expressed in meters.
"""
const AU_IN_M = uconvert(u"m", 1.0u"AU") # astronomical unit in meters


# j2=2.032365e-4, j3=8.585355e-6, j4=-9.860033e-6, j5=8.393511e-7, j6=-1.331149e-5, 
# j7=-2.205653e-5, j8=-9.434727e-6, j9=1.511956e-5, j10=3.863970e-6, j11=5.130603e-6, j12=9.103736e-6,

# c22=2.235491e-5, c31=2.849956e-5, c32=4.847437e-6, c33=1.712582e-6, 
# s22=1.535185e-8, s31=5.953103e-6, s32=1.689177e-6, s33=-2.564964e-7,



end # end of module

# src/Models/PerturbationEquations.jl

module PerturbationEquations

using ..Constants
using ..Types
using ..SphericalHarmonics: cunningham_perturbation, spherical_harmonic_perturbation,
                            cunningham_acceleration_body_fixed, cunningham_acceleration_body_fixed!,
                            cunningham_harmonics!, GravityHarmonics
using LinearAlgebra, StaticArrays, Unitful, Unitful.DefaultSymbols, UnitfulAstro

export cunningham_perturbation, spherical_harmonic_perturbation, GravityHarmonics,
       shadow_function_smooth, acc_lorentz, PrecessionPerturbation,
       earth_radiation_pressure, yarkovsky_perturbation,
       relativity_geodesic_perturbation, relativity_lense_thirring_perturbation,
       solid_earth_tides_perturbation, solid_earth_tides_potential,
       coulomb_drag_acceleration, coulomb_effective_area_factor


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

function j3_potential(r_vector, mu, R, j3)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^3*j3*mu*z*(5*z^2-3*r^2))/(2*r^3*(z^2+y^2+x^2)^2)
end

function j3_perturbation(r_vector, mu, R, j3)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (j3*R^3*mu*x*z*(15*z^4+15*y^2*z^2+15*x^2*z^2+17*r^2*z^2-3*r^2*y^2-3*r^2*x^2-12*r^4))/(2*r^11)
    ay = (j3*R^3*mu*y*z*(15*z^4+15*y^2*z^2+15*x^2*z^2+17*r^2*z^2-3*r^2*y^2-3*r^2*x^2-12*r^4))/(2*r^11)
    az = (j3*R^3*mu*(15*z^6+15*y^2*z^4+15*x^2*z^4+2*r^2*z^4-18*r^2*y^2*z^2-18*r^2*x^2*z^2-9*r^4*z^2+3*r^4*y^2+3*r^4*x^2))/(2*r^11)
    return SVector(ax, ay, az)
end

function j4_potential(r_vector, mu, R, j4)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^4*j4*mu*(35*z^4-30*r^2*z^2+3*r^4))/(8*r^9)
end

function j4_perturbation(r_vector, mu, R, j4)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (15*j4*R^4*mu*x*(21*z^4-14*r^2*z^2+r^4))/(8*r^11)
    ay = (15*j4*R^4*mu*y*(21*z^4-14*r^2*z^2+r^4))/(8*r^11)
    az = (5*j4*R^4*mu*z*(63*z^4-70*r^2*z^2+15*r^4))/(8*r^11)
    return SVector(ax, ay, az)
end

function j5_potential(r_vector, mu, R, j5)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^5*j5*mu*z*(63*z^4-70*r^2*z^2+15*r^4))/(8*r^5*(z^2+y^2+x^2)^3)
end


function j5_perturbation(r_vector, mu, R, j5)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (3*j5*R^5*mu*x*z*(105*z^6+105*y^2*z^4+105*x^2*z^4+56*r^2*z^4-70*r^2*y^2*z^2-70*r^2*x^2*z^2-135*r^4*z^2+5*r^4*y^2+5*r^4*x^2+30*r^6))/(8*r^15)
    ay = (3*j5*R^5*mu*y*z*(105*z^6+105*y^2*z^4+105*x^2*z^4+56*r^2*z^4-70*r^2*y^2*z^2-70*r^2*x^2*z^2-135*r^4*z^2+5*r^4*y^2+5*r^4*x^2+30*r^6))/(8*r^15)
    az = (3*j5*R^5*mu*(105*z^8+105*y^2*z^6+105*x^2*z^6-49*r^2*z^6-175*r^2*y^2*z^4-175*r^2*x^2*z^4-65*r^4*z^4+75*r^4*y^2*z^2+75*r^4*x^2*z^2+25*r^6*z^2-5*r^6*y^2-5*r^6*x^2))/(8*r^15)
    return SVector(ax, ay, az)
end

function j6_potential(r_vector, mu, R, j6)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^6*j6*mu*(231*z^6-315*r^2*z^4+105*r^4*z^2-5*r^6))/(16*r^13)
end

function j6_perturbation(r_vector, mu, R, j6)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (7*j6*R^6*mu*x*(429*z^6-495*r^2*z^4+135*r^4*z^2-5*r^6))/(16*r^15)
    ay = (7*j6*R^6*mu*y*(429*z^6-495*r^2*z^4+135*r^4*z^2-5*r^6))/(16*r^15)
    az = (7*j6*R^6*mu*z*(429*z^6-693*r^2*z^4+315*r^4*z^2-35*r^6))/(16*r^15)
    return SVector(ax, ay, az)
end

function j7_potential(r_vector, mu, R, j7)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^7*j7*mu*z*(429*z^6-693*r^2*z^4+315*r^4*z^2-35*r^6))/(16*r^7*(z^2+y^2+x^2)^4)
end

function j7_perturbation(r_vector, mu, R, j7)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (j7*R^7*mu*x*z*(3003*z^8+3003*y^2*z^6+3003*x^2*z^6-33*r^2*z^6-3465*r^2*y^2*z^4-3465*r^2*x^2*z^4-4599*r^4*z^4+945*r^4*y^2*z^2+945*r^4*x^2*z^2+2485*r^6*z^2-35*r^6*y^2-35*r^6*x^2-280*r^8))/(16*r^19)
    ay = (j7*R^7*mu*y*z*(3003*z^8+3003*y^2*z^6+3003*x^2*z^6-33*r^2*z^6-3465*r^2*y^2*z^4-3465*r^2*x^2*z^4-4599*r^4*z^4+945*r^4*y^2*z^2+945*r^4*x^2*z^2+2485*r^6*z^2-35*r^6*y^2-35*r^6*x^2-280*r^8))/(16*r^19)
    az = (j7*R^7*mu*(3003*z^10+3003*y^2*z^8+3003*x^2*z^8-3036*r^2*z^8-6468*r^2*y^2*z^6-6468*r^2*x^2*z^6-1134*r^4*z^6+4410*r^4*y^2*z^4+4410*r^4*x^2*z^4+1540*r^6*z^4-980*r^6*y^2*z^2-980*r^6*x^2*z^2-245*r^8*z^2+35*r^8*y^2+35*r^8*x^2))/(16*r^19)
    return SVector(ax, ay, az)
end

function j8_potential(r_vector, mu, R, j8)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^8*j8*mu*(6435*z^8-12012*r^2*z^6+6930*r^4*z^4-1260*r^6*z^2+35*r^8))/(128*r^17)
end

function j8_perturbation(r_vector, mu, R, j8)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (45*j8*R^8*mu*x*(2431*z^8-4004*r^2*z^6+2002*r^4*z^4-308*r^6*z^2+7*r^8))/(128*r^19)
    ay = (45*j8*R^8*mu*y*(2431*z^8-4004*r^2*z^6+2002*r^4*z^4-308*r^6*z^2+7*r^8))/(128*r^19)
    az = (9*j8*R^8*mu*z*(12155*z^8-25740*r^2*z^6+18018*r^4*z^4-4620*r^6*z^2+315*r^8))/(128*r^19)
    return SVector(ax, ay, az)
end

function j9_potential(r_vector, mu, R, j9)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^9*j9*mu*z*(12155*z^8-25740*r^2*z^6+18018*r^4*z^4-4620*r^6*z^2+315*r^8))/(128*r^9*(z^2+y^2+x^2)^5)
end


function j9_perturbation(r_vector, mu, R, j9)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (5*j9*R^9*mu*x*z*(21879*z^10+21879*y^2*z^8+21879*x^2*z^8-11726*r^2*z^8-36036*r^2*y^2*z^6-36036*r^2*x^2*z^6-33462*r^4*z^6+18018*r^4*y^2*z^4+18018*r^4*x^2*z^4+33264*r^6*z^4-2772*r^6*y^2*z^2-2772*r^6*x^2*z^2-9177*r^8*z^2+63*r^8*y^2+63*r^8*x^2+630*r^10))/(128*r^23)
    ay = (5*j9*R^9*mu*y*z*(21879*z^10+21879*y^2*z^8+21879*x^2*z^8-11726*r^2*z^8-36036*r^2*y^2*z^6-36036*r^2*x^2*z^6-33462*r^4*z^6+18018*r^4*y^2*z^4+18018*r^4*x^2*z^4+33264*r^6*z^4-2772*r^6*y^2*z^2-2772*r^6*x^2*z^2-9177*r^8*z^2+63*r^8*y^2+63*r^8*x^2+630*r^10))/(128*r^23)
    az = (5*j9*R^9*mu*(21879*z^12+21879*y^2*z^10+21879*x^2*z^10-33605*r^2*z^10-57915*r^2*y^2*z^8-57915*r^2*x^2*z^8+2574*r^4*z^8+54054*r^4*y^2*z^6+54054*r^4*x^2*z^6+15246*r^6*z^6-20790*r^6*y^2*z^4-20790*r^6*x^2*z^4-6405*r^8*z^4+2835*r^8*y^2*z^2+2835*r^8*x^2*z^2+567*r^10*z^2-63*r^10*y^2-63*r^10*x^2))/(128*r^23)
    return SVector(ax, ay, az)
end

function j10_potential(r_vector, mu, R, j10)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^10*j10*mu*(46189*z^10-109395*r^2*z^8+90090*r^4*z^6-30030*r^6*z^4+3465*r^8*z^2-63*r^10))/(256*r^21)
end

function j10_perturbation(r_vector, mu, R, j10)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (33*j10*R^10*mu*x*(29393*z^10-62985*r^2*z^8+46410*r^4*z^6-13650*r^6*z^4+1365*r^8*z^2-21*r^10))/(256*r^23)
    ay = (33*j10*R^10*mu*y*(29393*z^10-62985*r^2*z^8+46410*r^4*z^6-13650*r^6*z^4+1365*r^8*z^2-21*r^10))/(256*r^23)
    az = (11*j10*R^10*mu*z*(88179*z^10-230945*r^2*z^8+218790*r^4*z^6-90090*r^6*z^4+15015*r^8*z^2-693*r^10))/(256*r^23)
    return SVector(ax, ay, az)
end

function j11_potential(r_vector, mu, R, j11)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^11*j11*mu*z*(88179*z^10-230945*r^2*z^8+218790*r^4*z^6-90090*r^6*z^4+15015*r^8*z^2-693*r^10))/(256*r^11*(z^2+y^2+x^2)^6)
end


function j11_perturbation(r_vector, mu, R, j11)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (3*j11*R^11*mu*x*z*(323323*z^12+323323*y^2*z^10+323323*x^2*z^10-340119*r^2*z^10-692835*r^2*y^2*z^8-692835*r^2*x^2*z^8-413270*r^4*z^8+510510*r^4*y^2*z^6+510510*r^4*x^2*z^6+
        725010*r^6*z^6-150150*r^6*y^2*z^4-150150*r^6*x^2*z^4-345345*r^8*z^4+15015*r^8*y^2*z^2+15015*r^8*x^2*z^2+59829*r^10*z^2-231*r^10*y^2-231*r^10*x^2-2772*r^12))/
        (256*r^13*(z^2+y^2+x^2)^7)
    ay = (3*j11*R^11*mu*y*z*(323323*z^12+323323*y^2*z^10+323323*x^2*z^10-340119*r^2*z^10-692835*r^2*y^2*z^8-692835*r^2*x^2*z^8-413270*r^4*z^8+510510*r^4*y^2*z^6+510510*r^4*x^2*z^6+
        725010*r^6*z^6-150150*r^6*y^2*z^4-150150*r^6*x^2*z^4-345345*r^8*z^4+15015*r^8*y^2*z^2+15015*r^8*x^2*z^2+59829*r^10*z^2-231*r^10*y^2-231*r^10*x^2-2772*r^12))/
        (256*r^13*(z^2+y^2+x^2)^7)
    az = (3*j11*R^11*mu*(323323*z^14+323323*y^2*z^12+323323*x^2*z^12-663442*r^2*z^12-1016158*r^2*y^2*z^10-1016158*r^2*x^2*z^10+279565*r^4*z^10+1203345*r^4*y^2*z^8+
        1203345*r^4*x^2*z^8+214500*r^6*z^8-660660*r^6*y^2*z^6-660660*r^6*x^2*z^6-195195*r^8*z^6+165165*r^8*y^2*z^4+165165*r^8*x^2*z^4+44814*r^10*z^4-15246*r^10*y^2*z^2-
        15246*r^10*x^2*z^2-2541*r^12*z^2+231*r^12*y^2+231*r^12*x^2))/(256*r^13*(z^2+y^2+x^2)^7)
    return SVector(ax, ay, az)
end

function j12_potential(r_vector, mu, R, j12)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^12*j12*mu*(676039*z^12-1939938*r^2*z^10+2078505*r^4*z^8-1021020*r^6*z^6+225225*r^8*z^4-18018*r^10*z^2+231*r^12))/(1024*r^25)
end

function j12_perturbation(r_vector, mu, R, j12)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (91*j12*R^12*mu*x*(185725*z^12-490314*r^2*z^10+479655*r^4*z^8-213180*r^6*z^6+42075*r^8*z^4-2970*r^10*z^2+33*r^12))/(1024*r^27)
    ay = (91*j12*R^12*mu*y*(185725*z^12-490314*r^2*z^10+479655*r^4*z^8-213180*r^6*z^6+42075*r^8*z^4-2970*r^10*z^2+33*r^12))/(1024*r^27)
    az = (13*j12*R^12*mu*z*(1300075*z^12-4056234*r^2*z^10+4849845*r^4*z^8-2771340*r^6*z^6+765765*r^8*z^4-90090*r^10*z^2+3003*r^12))/(1024*r^27)
    return SVector(ax, ay, az)
end

function j13_potential(r_vector, mu, R, j13)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^13*j13*mu*z*(1300075*z^12-4056234*r^2*z^10+4849845*r^4*z^8-2771340*r^6*z^6+765765*r^8*z^4-90090*r^10*z^2+3003*r^12))/(1024*r^13*(z^2+y^2+x^2)^7)
end

function j13_perturbation(r_vector, mu, R, j13)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (7*j13*R^13*mu*x*z*(2414425*z^14+2414425*y^2*z^12+2414425*x^2*z^12-3773932*r^2*z^12-6374082*r^2*y^2*z^10-6374082*r^2*x^2*z^10-1876953*r^4*z^10+6235515*r^4*y^2*z^8+6235515*r^4*x^2*z^8+6928350*r^6*z^8-2771340*r^6*y^2*z^6-2771340*r^6*x^2*z^6-4995705*r^8*z^6+546975*r^8*y^2*z^4+546975*r^8*x^2*z^4+1492920*r^10*z^4-38610*r^10*y^2*z^2-38610*r^10*x^2*z^2-179751*r^12*z^2+429*r^12*y^2+429*r^12*x^2+6006*r^14))/(1024*r^31)
    ay = (7*j13*R^13*mu*y*z*(2414425*z^14+2414425*y^2*z^12+2414425*x^2*z^12-3773932*r^2*z^12-6374082*r^2*y^2*z^10-6374082*r^2*x^2*z^10-1876953*r^4*z^10+6235515*r^4*y^2*z^8+6235515*r^4*x^2*z^8+6928350*r^6*z^8-2771340*r^6*y^2*z^6-2771340*r^6*x^2*z^6-4995705*r^8*z^6+546975*r^8*y^2*z^4+546975*r^8*x^2*z^4+1492920*r^10*z^4-38610*r^10*y^2*z^2-38610*r^10*x^2*z^2-179751*r^12*z^2+429*r^12*y^2+429*r^12*x^2+6006*r^14))/(1024*r^31)
    az = (7*j13*R^13*mu*(2414425*z^16+2414425*y^2*z^14+2414425*x^2*z^14-6188357*r^2*z^14-8788507*r^2*y^2*z^12-8788507*r^2*x^2*z^12+4497129*r^4*z^12+12609597*r^4*y^2*z^10+12609597*r^4*x^2*z^10+692835*r^6*z^10-9006855*r^6*y^2*z^8-9006855*r^6*x^2*z^8-2224365*r^8*z^8+3318315*r^8*y^2*z^6+3318315*r^8*x^2*z^6+945945*r^10*z^6-585585*r^10*y^2*z^4-585585*r^10*x^2*z^4-141141*r^12*z^4+39039*r^12*y^2*z^2+39039*r^12*x^2*z^2+5577*r^14*z^2-429*r^14*y^2-429*r^14*x^2))/(1024*r^31)
    return SVector(ax, ay, az)
end

function j14_potential(r_vector, mu, R, j14)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^14*j14*mu*(5014575*z^14-16900975*r^2*z^12+22309287*r^4*z^10-14549535*r^6*z^8+4849845*r^8*z^6-765765*r^10*z^4+45045*r^12*z^2-429*r^14))/(2048*r^29)
end

function j14_perturbation(r_vector, mu, R, j14)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (15*j14*R^14*mu*x*(9694845*z^14-30421755*r^2*z^12+37182145*r^4*z^10-22309287*r^6*z^8+6789783*r^8*z^6-969969*r^10*z^4+51051*r^12*z^2-429*r^14))/(2048*r^31)
    ay = (15*j14*R^14*mu*y*(9694845*z^14-30421755*r^2*z^12+37182145*r^4*z^10-22309287*r^6*z^8+6789783*r^8*z^6-969969*r^10*z^4+51051*r^12*z^2-429*r^14))/(2048*r^31)
    az = (15*j14*R^14*mu*z*(9694845*z^14-35102025*r^2*z^12+50702925*r^4*z^10-37182145*r^6*z^8+14549535*r^8*z^6-2909907*r^10*z^4+255255*r^12*z^2-6435*r^14))/(2048*r^31)
    return SVector(ax, ay, az)
end

function j15_potential(r_vector, mu, R, j15)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^15*j15*mu*z*(9694845*z^14-35102025*r^2*z^12+50702925*r^4*z^10-37182145*r^6*z^8+14549535*r^8*z^6-2909907*r^10*z^4+255255*r^12*z^2-6435*r^14))/(2048*r^15*(z^2+y^2+x^2)^8)
end

function j15_perturbation(r_vector, mu, R, j15)
    x, y, z, = r_vector
    r = norm(r_vector)
    ax = (j15*R^15*mu*x*z*(145422675*z^16+145422675*y^2*z^14+145422675*x^2*z^14-301208805*r^2*z^14-456326325*r^2*y^2*z^12-456326325*r^2*x^2*z^12-3900225*r^4*z^12+557732175*r^4*y^2*z^10+557732175*r^4*x^2*z^10+476607495*r^6*z^10-334639305*r^6*y^2*z^8-334639305*r^6*x^2*z^8-493067575*r^8*z^8+101846745*r^8*y^2*z^6+101846745*r^8*x^2*z^6+218243025*r^10*z^6-14549535*r^10*y^2*z^4-14549535*r^10*x^2*z^4-45792747*r^12*z^4+765765*r^12*y^2*z^2+765765*r^12*x^2*z^2+4077645*r^14*z^2-6435*r^14*y^2-6435*r^14*x^2-102960*r^16))/(2048*r^35)
    ay = (j15*R^15*mu*y*z*(145422675*z^16+145422675*y^2*z^14+145422675*x^2*z^14-301208805*r^2*z^14-456326325*r^2*y^2*z^12-456326325*r^2*x^2*z^12-3900225*r^4*z^12+557732175*r^4*y^2*z^10+557732175*r^4*x^2*z^10+476607495*r^6*z^10-334639305*r^6*y^2*z^8-334639305*r^6*x^2*z^8-493067575*r^8*z^8+101846745*r^8*y^2*z^6+101846745*r^8*x^2*z^6+218243025*r^10*z^6-14549535*r^10*y^2*z^4-14549535*r^10*x^2*z^4-45792747*r^12*z^4+765765*r^12*y^2*z^2+765765*r^12*x^2*z^2+4077645*r^14*z^2-6435*r^14*y^2-6435*r^14*x^2-102960*r^16))/(2048*r^35)
    az = (j15*R^15*mu*(145422675*z^18+145422675*y^2*z^16+145422675*x^2*z^16-446631480*r^2*z^16-601749000*r^2*y^2*z^14-601749000*r^2*x^2*z^14+452426100*r^4*z^14+1014058500*r^4*y^2*z^12+1014058500*r^4*x^2*z^12-81124680*r^6*z^12-892371480*r^6*y^2*z^10-892371480*r^6*x^2*z^10-158428270*r^8*z^10+436486050*r^8*y^2*z^8+436486050*r^8*x^2*z^8+116396280*r^10*z^8-116396280*r^10*y^2*z^6-116396280*r^10*x^2*z^6-31243212*r^12*z^6+15315300*r^12*y^2*z^4+15315300*r^12*x^2*z^4+3311880*r^14*z^4-772200*r^14*y^2*z^2-772200*r^14*x^2*z^2-96525*r^16*z^2+6435*r^16*y^2+6435*r^16*x^2))/(2048*r^35)
    return SVector(ax, ay, az)
end

function j16_potential(r_vector, mu, R, j16)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^16*j16*mu*(300540195*z^16-1163381400*r^2*z^14+1825305300*r^4*z^12-1487285800*r^6*z^10+669278610*r^8*z^8-162954792*r^10*z^6+19399380*r^12*z^4-875160*r^14*z^2+6435*r^16))/(32768*r^33)
end

function j16_perturbation(r_vector, mu, R, j16)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (153*j16*R^16*mu*x*(64822395*z^16-235717800*r^2*z^14+345972900*r^4*z^12-262462200*r^6*z^10+109359250*r^8*z^8-24496472*r^10*z^6+2662660*r^12*z^4-108680*r^14*z^2+715*r^16))/(32768*r^35)
    ay = (153*j16*R^16*mu*y*(64822395*z^16-235717800*r^2*z^14+345972900*r^4*z^12-262462200*r^6*z^10+109359250*r^8*z^8-24496472*r^10*z^6+2662660*r^12*z^4-108680*r^14*z^2+715*r^16))/(32768*r^35)
    az = (17*j16*R^16*mu*z*(583401555*z^16-2404321560*r^2*z^14+4071834900*r^4*z^12-3650610600*r^6*z^10+1859107250*r^8*z^8-535422888*r^10*z^6+81477396*r^12*z^4-5542680*r^14*z^2+109395*r^16))/(32768*r^35)
    return SVector(ax, ay, az)
end

function j17_potential(r_vector, mu, R, j17)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^17*j17*mu*z*(583401555*z^16-2404321560*r^2*z^14+4071834900*r^4*z^12-3650610600*r^6*z^10+1859107250*r^8*z^8-535422888*r^10*z^6+81477396*r^12*z^4-5542680*r^14*z^2+109395*r^16))/(32768*r^17*(z^2+y^2+x^2)^9)
end

function j17_perturbation(r_vector, mu, R, j17)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (9*j17*R^17*mu*x*z*(1101980715*z^18+1101980715*y^2*z^16+1101980715*x^2*z^16-2840399490*r^2*z^16-4007202600*r^2*y^2*z^14-4007202600*r^2*x^2*z^14+1072896180*r^4*z^14+5881539300*r^4*y^2*z^12+5881539300*r^4*x^2*z^12+3681812400*r^6*z^12-4461857400*r^6*y^2*z^10-4461857400*r^6*x^2*z^10-5442113950*r^8*z^10+1859107250*r^8*y^2*z^8+1859107250*r^8*x^2*z^8+3301774476*r^10*z^8-416440024*r^10*y^2*z^6-416440024*r^10*x^2*z^6-1025580556*r^12*z^6+45265220*r^12*y^2*z^4+45265220*r^12*x^2*z^4+161107232*r^14*z^4-1847560*r^14*y^2*z^2-1847560*r^14*x^2*z^2-11073205*r^16*z^2+12155*r^16*y^2+12155*r^16*x^2+218790*r^18))/(32768*r^39)
    ay = (9*j17*R^17*mu*y*z*(1101980715*z^18+1101980715*y^2*z^16+1101980715*x^2*z^16-2840399490*r^2*z^16-4007202600*r^2*y^2*z^14-4007202600*r^2*x^2*z^14+1072896180*r^4*z^14+5881539300*r^4*y^2*z^12+5881539300*r^4*x^2*z^12+3681812400*r^6*z^12-4461857400*r^6*y^2*z^10-4461857400*r^6*x^2*z^10-5442113950*r^8*z^10+1859107250*r^8*y^2*z^8+1859107250*r^8*x^2*z^8+3301774476*r^10*z^8-416440024*r^10*y^2*z^6-416440024*r^10*x^2*z^6-1025580556*r^12*z^6+45265220*r^12*y^2*z^4+45265220*r^12*x^2*z^4+161107232*r^14*z^4-1847560*r^14*y^2*z^2-1847560*r^14*x^2*z^2-11073205*r^16*z^2+12155*r^16*y^2+12155*r^16*x^2+218790*r^18))/(32768*r^39)
    az = (9*j17*R^17*mu*(1101980715*z^20+1101980715*y^2*z^18+1101980715*x^2*z^18-3942380205*r^2*z^18-5109183315*r^2*y^2*z^16-5109183315*r^2*x^2*z^16+5080098780*r^4*z^16+9888741900*r^4*y^2*z^14+9888741900*r^4*x^2*z^14-2199726900*r^6*z^14-10343396700*r^6*y^2*z^12-10343396700*r^6*x^2*z^12-980256550*r^8*z^12+6320964650*r^8*y^2*z^10+6320964650*r^8*x^2*z^10+1442667226*r^10*z^10-2275547274*r^10*y^2*z^8-2275547274*r^10*x^2*z^8-609140532*r^12*z^8+461705244*r^12*y^2*z^6+461705244*r^12*x^2*z^6+115842012*r^14*z^6-47112780*r^14*y^2*z^4-47112780*r^14*x^2*z^4-9225645*r^16*z^4+1859715*r^16*y^2*z^2+1859715*r^16*x^2*z^2+206635*r^18*z^2-12155*r^18*y^2-12155*r^18*x^2))/(32768*r^39)
    return SVector(ax, ay, az)
end

function j18_potential(r_vector, mu, R, j18)
    x, y, z = r_vector
    r = norm(r_vector)
    return (R^18*j18*mu*(2268783825*z^18-9917826435*r^2*z^16+18032411700*r^4*z^14-17644617900*r^6*z^12+10039179150*r^8*z^10-3346393050*r^10*z^8+624660036*r^12*z^6-58198140*r^14*z^4+2078505*r^16*z^2-12155*r^18))/(65536*r^37)
end

function j18_perturbation(r_vector, mu, R, j18)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (95*j18*R^18*mu*x*(883631595*z^18-3653936055*r^2*z^16+6263890380*r^4*z^14-5757717420*r^6*z^12+3064591530*r^8*z^10-951080130*r^10*z^8+164384220*r^12*z^6-14090076*r^14*z^4+459459*r^16*z^2-2431*r^18))/(65536*r^39)
    ay = (95*j18*R^18*mu*y*(883631595*z^18-3653936055*r^2*z^16+6263890380*r^4*z^14-5757717420*r^6*z^12+3064591530*r^8*z^10-951080130*r^10*z^8+164384220*r^12*z^6-14090076*r^14*z^4+459459*r^16*z^2-2431*r^18))/(65536*r^39)
    az = (19*j18*R^18*mu*z*(4418157975*z^18-20419054425*r^2*z^16+39671305740*r^4*z^14-42075627300*r^6*z^12+26466926850*r^8*z^10-10039179150*r^10*z^8+2230928700*r^12*z^6-267711444*r^14*z^4+14549535*r^16*z^2-230945*r^18))/(65536*r^39)
    return SVector(ax, ay, az)
end

# ------------ sectoral harmonics  (n = m) ------------ #
# they are in the inertial frame of reference, not the rotating one

function cs22_potential(r_vector, t, mu, R, c22, s22, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)
    return -((3*R^2*mu*(c22*sin(omega_rot*t)^2*y^2+2*s22*cos(omega_rot*t)*sin(omega_rot*t)*y^2-c22*cos(omega_rot*t)^2*y^2-2*s22*sin(omega_rot*t)^2*x*y+4*c22*cos(omega_rot*t)*sin(omega_rot*t)*x*y+2*s22*cos(omega_rot*t)^2*x*y-c22*sin(omega_rot*t)^2*x^2-2*s22*cos(omega_rot*t)*sin(omega_rot*t)*x^2+c22*cos(omega_rot*t)^2*x^2))/r^5)
end

function cs22_perturbation(r_vector, t, mu, R, c22, s22, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)

    ax = -((3*R^2*mu*(c22*(5*sin(omega_rot*t)^2*x*y^2-5*cos(omega_rot*t)^2*x*y^2+20*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y-4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y-5*sin(omega_rot*t)^2*x^3+5*cos(omega_rot*t)^2*x^3+2*r^2*sin(omega_rot*t)^2*x-2*r^2*cos(omega_rot*t)^2*x)+s22*(10*cos(omega_rot*t)*sin(omega_rot*t)*x*y^2-10*sin(omega_rot*t)^2*x^2*y+10*cos(omega_rot*t)^2*x^2*y+2*r^2*sin(omega_rot*t)^2*y-2*r^2*cos(omega_rot*t)^2*y-10*cos(omega_rot*t)*sin(omega_rot*t)*x^3+4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x)))/r^7)
    ay = -((3*R^2*mu*(c22*(5*sin(omega_rot*t)^2*y^3-5*cos(omega_rot*t)^2*y^3+20*cos(omega_rot*t)*sin(omega_rot*t)*x*y^2-5*sin(omega_rot*t)^2*x^2*y+5*cos(omega_rot*t)^2*x^2*y-2*r^2*sin(omega_rot*t)^2*y+2*r^2*cos(omega_rot*t)^2*y-4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x)+s22*(10*cos(omega_rot*t)*sin(omega_rot*t)*y^3-10*sin(omega_rot*t)^2*x*y^2+10*cos(omega_rot*t)^2*x*y^2-10*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y-4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y+2*r^2*sin(omega_rot*t)^2*x-2*r^2*cos(omega_rot*t)^2*x)))/r^7)
    az = -((15*R^2*mu*(c22*(sin(omega_rot*t)^2*y^2-cos(omega_rot*t)^2*y^2+4*cos(omega_rot*t)*sin(omega_rot*t)*x*y-sin(omega_rot*t)^2*x^2+cos(omega_rot*t)^2*x^2)+s22*(2*cos(omega_rot*t)*sin(omega_rot*t)*y^2-2*sin(omega_rot*t)^2*x*y+2*cos(omega_rot*t)^2*x*y-2*cos(omega_rot*t)*sin(omega_rot*t)*x^2))*z)/r^7)
    return SVector(ax, ay, az)
end

function cs33_potential(r_vector, t, mu, R, c33, s33, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)
    return (15*R^3*mu*(c33*sin(omega_rot*t)^3*y^3+3*s33*cos(omega_rot*t)*sin(omega_rot*t)^2*y^3-3*c33*cos(omega_rot*t)^2*sin(omega_rot*t)*y^3-s33*cos(omega_rot*t)^3*y^3-3*s33*sin(omega_rot*t)^3*x*y^2+9*c33*cos(omega_rot*t)*sin(omega_rot*t)^2*x*y^2+9*s33*cos(omega_rot*t)^2*sin(omega_rot*t)*x*y^2-3*c33*cos(omega_rot*t)^3*x*y^2-3*c33*sin(omega_rot*t)^3*x^2*y-9*s33*cos(omega_rot*t)*sin(omega_rot*t)^2*x^2*y+9*c33*cos(omega_rot*t)^2*sin(omega_rot*t)*x^2*y+3*s33*cos(omega_rot*t)^3*x^2*y+s33*sin(omega_rot*t)^3*x^3-3*c33*cos(omega_rot*t)*sin(omega_rot*t)^2*x^3-3*s33*cos(omega_rot*t)^2*sin(omega_rot*t)*x^3+c33*cos(omega_rot*t)^3*x^3))/r^7
end

function cs33_perturbation(r_vector, t, mu, R, c33, s33, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)

    ax = (15*R^3*mu*(c33*(7*sin(omega_rot*t)^3*x*y^3-21*cos(omega_rot*t)^2*sin(omega_rot*t)*x*y^3+63*cos(omega_rot*t)*sin(omega_rot*t)^2*x^2*y^2-21*cos(omega_rot*t)^3*x^2*y^2-9*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*y^2+3*r^2*cos(omega_rot*t)^3*y^2-21*sin(omega_rot*t)^3*x^3*y+63*cos(omega_rot*t)^2*sin(omega_rot*t)*x^3*y+6*r^2*sin(omega_rot*t)^3*x*y-18*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*x*y-21*cos(omega_rot*t)*sin(omega_rot*t)^2*x^4+7*cos(omega_rot*t)^3*x^4+9*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*x^2-3*r^2*cos(omega_rot*t)^3*x^2)+s33*(21*cos(omega_rot*t)*sin(omega_rot*t)^2*x*y^3-7*cos(omega_rot*t)^3*x*y^3-21*sin(omega_rot*t)^3*x^2*y^2+63*cos(omega_rot*t)^2*sin(omega_rot*t)*x^2*y^2+3*r^2*sin(omega_rot*t)^3*y^2-9*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*y^2-63*cos(omega_rot*t)*sin(omega_rot*t)^2*x^3*y+21*cos(omega_rot*t)^3*x^3*y+18*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*x*y-6*r^2*cos(omega_rot*t)^3*x*y+7*sin(omega_rot*t)^3*x^4-21*cos(omega_rot*t)^2*sin(omega_rot*t)*x^4-3*r^2*sin(omega_rot*t)^3*x^2+9*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*x^2)))/r^9
    ay = (15*R^3*mu*(c33*(7*sin(omega_rot*t)^3*y^4-21*cos(omega_rot*t)^2*sin(omega_rot*t)*y^4+63*cos(omega_rot*t)*sin(omega_rot*t)^2*x*y^3-21*cos(omega_rot*t)^3*x*y^3-21*sin(omega_rot*t)^3*x^2*y^2+63*cos(omega_rot*t)^2*sin(omega_rot*t)*x^2*y^2-3*r^2*sin(omega_rot*t)^3*y^2+9*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*y^2-21*cos(omega_rot*t)*sin(omega_rot*t)^2*x^3*y+7*cos(omega_rot*t)^3*x^3*y-18*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*x*y+6*r^2*cos(omega_rot*t)^3*x*y+3*r^2*sin(omega_rot*t)^3*x^2-9*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*x^2)+s33*(21*cos(omega_rot*t)*sin(omega_rot*t)^2*y^4-7*cos(omega_rot*t)^3*y^4-21*sin(omega_rot*t)^3*x*y^3+63*cos(omega_rot*t)^2*sin(omega_rot*t)*x*y^3-63*cos(omega_rot*t)*sin(omega_rot*t)^2*x^2*y^2+21*cos(omega_rot*t)^3*x^2*y^2-9*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*y^2+3*r^2*cos(omega_rot*t)^3*y^2+7*sin(omega_rot*t)^3*x^3*y-21*cos(omega_rot*t)^2*sin(omega_rot*t)*x^3*y+6*r^2*sin(omega_rot*t)^3*x*y-18*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*x*y+9*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*x^2-3*r^2*cos(omega_rot*t)^3*x^2)))/r^9
    az = (105*R^3*mu*(c33*(sin(omega_rot*t)^3*y^3-3*cos(omega_rot*t)^2*sin(omega_rot*t)*y^3+9*cos(omega_rot*t)*sin(omega_rot*t)^2*x*y^2-3*cos(omega_rot*t)^3*x*y^2-3*sin(omega_rot*t)^3*x^2*y+9*cos(omega_rot*t)^2*sin(omega_rot*t)*x^2*y-3*cos(omega_rot*t)*sin(omega_rot*t)^2*x^3+cos(omega_rot*t)^3*x^3)+s33*(3*cos(omega_rot*t)*sin(omega_rot*t)^2*y^3-cos(omega_rot*t)^3*y^3-3*sin(omega_rot*t)^3*x*y^2+9*cos(omega_rot*t)^2*sin(omega_rot*t)*x*y^2-9*cos(omega_rot*t)*sin(omega_rot*t)^2*x^2*y+3*cos(omega_rot*t)^3*x^2*y+sin(omega_rot*t)^3*x^3-3*cos(omega_rot*t)^2*sin(omega_rot*t)*x^3))*z)/r^9
    return SVector(ax, ay, az)
end

function cs44_potential(r_vector, t, mu, R, c44, s44, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)
    return (-(105*R^4*c44*mu*(sin(omega_rot*t)^2*y^2-2*cos(omega_rot*t)*sin(omega_rot*t)*y^2-cos(omega_rot*t)^2*y^2+2*sin(omega_rot*t)^2*x*y+4*cos(omega_rot*t)*sin(omega_rot*t)*x*y-2*cos(omega_rot*t)^2*x*y-sin(omega_rot*t)^2*x^2+2*cos(omega_rot*t)*sin(omega_rot*t)*x^2+cos(omega_rot*t)^2*x^2)*(sin(omega_rot*t)^2*y^2+2*cos(omega_rot*t)*sin(omega_rot*t)*y^2-cos(omega_rot*t)^2*y^2-2*sin(omega_rot*t)^2*x*y+4*cos(omega_rot*t)*sin(omega_rot*t)*x*y+2*cos(omega_rot*t)^2*x*y-sin(omega_rot*t)^2*x^2-2*cos(omega_rot*t)*sin(omega_rot*t)*x^2+cos(omega_rot*t)^2*x^2))-420*R^4*mu*s44*(cos(omega_rot*t)*y-sin(omega_rot*t)*x)*(sin(omega_rot*t)*y+cos(omega_rot*t)*x)*(sin(omega_rot*t)*y-cos(omega_rot*t)*y+sin(omega_rot*t)*x+cos(omega_rot*t)*x)*(sin(omega_rot*t)*y+cos(omega_rot*t)*y-sin(omega_rot*t)*x+cos(omega_rot*t)*x))/r^9
end


function cs44_perturbation(r_vector, t, mu, R, c44, s44, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)

    ax = (-(105*R^4*c44*mu*(9*sin(omega_rot*t)^4*x*y^4-54*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x*y^4+9*cos(omega_rot*t)^4*x*y^4+144*cos(omega_rot*t)*sin(omega_rot*t)^3*x^2*y^3-144*cos(omega_rot*t)^3*sin(omega_rot*t)*x^2*y^3-16*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*y^3+16*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*y^3-54*sin(omega_rot*t)^4*x^3*y^2+324*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^3*y^2-54*cos(omega_rot*t)^4*x^3*y^2+12*r^2*sin(omega_rot*t)^4*x*y^2-72*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x*y^2+12*r^2*cos(omega_rot*t)^4*x*y^2-144*cos(omega_rot*t)*sin(omega_rot*t)^3*x^4*y+144*cos(omega_rot*t)^3*sin(omega_rot*t)*x^4*y+48*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*x^2*y-48*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*x^2*y+9*sin(omega_rot*t)^4*x^5-54*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^5+9*cos(omega_rot*t)^4*x^5-4*r^2*sin(omega_rot*t)^4*x^3+24*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^3-4*r^2*cos(omega_rot*t)^4*x^3))-420*R^4*mu*s44*(9*cos(omega_rot*t)*sin(omega_rot*t)^3*x*y^4-9*cos(omega_rot*t)^3*sin(omega_rot*t)*x*y^4-9*sin(omega_rot*t)^4*x^2*y^3+54*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^2*y^3-9*cos(omega_rot*t)^4*x^2*y^3+r^2*sin(omega_rot*t)^4*y^3-6*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*y^3+r^2*cos(omega_rot*t)^4*y^3-54*cos(omega_rot*t)*sin(omega_rot*t)^3*x^3*y^2+54*cos(omega_rot*t)^3*sin(omega_rot*t)*x^3*y^2+12*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*x*y^2-12*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*x*y^2+9*sin(omega_rot*t)^4*x^4*y-54*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^4*y+9*cos(omega_rot*t)^4*x^4*y-3*r^2*sin(omega_rot*t)^4*x^2*y+18*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^2*y-3*r^2*cos(omega_rot*t)^4*x^2*y+9*cos(omega_rot*t)*sin(omega_rot*t)^3*x^5-9*cos(omega_rot*t)^3*sin(omega_rot*t)*x^5-4*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*x^3+4*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*x^3))/r^11
    ay = (-(105*R^4*c44*mu*(9*sin(omega_rot*t)^4*y^5-54*cos(omega_rot*t)^2*sin(omega_rot*t)^2*y^5+9*cos(omega_rot*t)^4*y^5+144*cos(omega_rot*t)*sin(omega_rot*t)^3*x*y^4-144*cos(omega_rot*t)^3*sin(omega_rot*t)*x*y^4-54*sin(omega_rot*t)^4*x^2*y^3+324*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^2*y^3-54*cos(omega_rot*t)^4*x^2*y^3-4*r^2*sin(omega_rot*t)^4*y^3+24*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*y^3-4*r^2*cos(omega_rot*t)^4*y^3-144*cos(omega_rot*t)*sin(omega_rot*t)^3*x^3*y^2+144*cos(omega_rot*t)^3*sin(omega_rot*t)*x^3*y^2-48*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*x*y^2+48*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*x*y^2+9*sin(omega_rot*t)^4*x^4*y-54*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^4*y+9*cos(omega_rot*t)^4*x^4*y+12*r^2*sin(omega_rot*t)^4*x^2*y-72*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^2*y+12*r^2*cos(omega_rot*t)^4*x^2*y+16*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*x^3-16*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*x^3))-420*R^4*mu*s44*(9*cos(omega_rot*t)*sin(omega_rot*t)^3*y^5-9*cos(omega_rot*t)^3*sin(omega_rot*t)*y^5-9*sin(omega_rot*t)^4*x*y^4+54*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x*y^4-9*cos(omega_rot*t)^4*x*y^4-54*cos(omega_rot*t)*sin(omega_rot*t)^3*x^2*y^3+54*cos(omega_rot*t)^3*sin(omega_rot*t)*x^2*y^3-4*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*y^3+4*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*y^3+9*sin(omega_rot*t)^4*x^3*y^2-54*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^3*y^2+9*cos(omega_rot*t)^4*x^3*y^2+3*r^2*sin(omega_rot*t)^4*x*y^2-18*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x*y^2+3*r^2*cos(omega_rot*t)^4*x*y^2+9*cos(omega_rot*t)*sin(omega_rot*t)^3*x^4*y-9*cos(omega_rot*t)^3*sin(omega_rot*t)*x^4*y+12*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*x^2*y-12*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*x^2*y-r^2*sin(omega_rot*t)^4*x^3+6*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^3-r^2*cos(omega_rot*t)^4*x^3))/r^11
    az = (-(945*R^4*c44*mu*(sin(omega_rot*t)^2*y^2-2*cos(omega_rot*t)*sin(omega_rot*t)*y^2-cos(omega_rot*t)^2*y^2+2*sin(omega_rot*t)^2*x*y+4*cos(omega_rot*t)*sin(omega_rot*t)*x*y-2*cos(omega_rot*t)^2*x*y-sin(omega_rot*t)^2*x^2+2*cos(omega_rot*t)*sin(omega_rot*t)*x^2+cos(omega_rot*t)^2*x^2)*(sin(omega_rot*t)^2*y^2+2*cos(omega_rot*t)*sin(omega_rot*t)*y^2-cos(omega_rot*t)^2*y^2-2*sin(omega_rot*t)^2*x*y+4*cos(omega_rot*t)*sin(omega_rot*t)*x*y+2*cos(omega_rot*t)^2*x*y-sin(omega_rot*t)^2*x^2-2*cos(omega_rot*t)*sin(omega_rot*t)*x^2+cos(omega_rot*t)^2*x^2)*z)-3780*R^4*mu*s44*(cos(omega_rot*t)*y-sin(omega_rot*t)*x)*(sin(omega_rot*t)*y+cos(omega_rot*t)*x)*(sin(omega_rot*t)*y-cos(omega_rot*t)*y+sin(omega_rot*t)*x+cos(omega_rot*t)*x)*(sin(omega_rot*t)*y+cos(omega_rot*t)*y-sin(omega_rot*t)*x+cos(omega_rot*t)*x)*z)/r^11
    return SVector(ax, ay, az)
end

# ------------ tesseral harmonics (m < n) ------------ #
# they are in the inertial frame of reference, not the rotating one

function cs31_potential(r_vector, t, mu, R, c31, s31, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)
    return (3*R^3*c31*mu*(sin(omega_rot*t)*y+cos(omega_rot*t)*x)*(4*z^2-y^2-x^2)+3*R^3*mu*s31*(cos(omega_rot*t)*y-sin(omega_rot*t)*x)*(4*z^2-y^2-x^2))/(2*r^7)
end

function cs31_perturbation(r_vector, t, mu, R, c31, s31, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)

    ax = (3*R^3*mu*(c31*(28*sin(omega_rot*t)*x*y*z^2+28*cos(omega_rot*t)*x^2*z^2-4*r^2*cos(omega_rot*t)*z^2-7*sin(omega_rot*t)*x*y^3-7*cos(omega_rot*t)*x^2*y^2+r^2*cos(omega_rot*t)*y^2-7*sin(omega_rot*t)*x^3*y+2*r^2*sin(omega_rot*t)*x*y-7*cos(omega_rot*t)*x^4+3*r^2*cos(omega_rot*t)*x^2)+s31*(28*cos(omega_rot*t)*x*y*z^2-28*sin(omega_rot*t)*x^2*z^2+4*r^2*sin(omega_rot*t)*z^2-7*cos(omega_rot*t)*x*y^3+7*sin(omega_rot*t)*x^2*y^2-r^2*sin(omega_rot*t)*y^2-7*cos(omega_rot*t)*x^3*y+2*r^2*cos(omega_rot*t)*x*y+7*sin(omega_rot*t)*x^4-3*r^2*sin(omega_rot*t)*x^2)))/(2*r^9)
    ay = (3*R^3*mu*(c31*(28*sin(omega_rot*t)*y^2*z^2+28*cos(omega_rot*t)*x*y*z^2-4*r^2*sin(omega_rot*t)*z^2-7*sin(omega_rot*t)*y^4-7*cos(omega_rot*t)*x*y^3-7*sin(omega_rot*t)*x^2*y^2+3*r^2*sin(omega_rot*t)*y^2-7*cos(omega_rot*t)*x^3*y+2*r^2*cos(omega_rot*t)*x*y+r^2*sin(omega_rot*t)*x^2)+s31*(28*cos(omega_rot*t)*y^2*z^2-28*sin(omega_rot*t)*x*y*z^2-4*r^2*cos(omega_rot*t)*z^2-7*cos(omega_rot*t)*y^4+7*sin(omega_rot*t)*x*y^3-7*cos(omega_rot*t)*x^2*y^2+3*r^2*cos(omega_rot*t)*y^2+7*sin(omega_rot*t)*x^3*y-2*r^2*sin(omega_rot*t)*x*y+r^2*cos(omega_rot*t)*x^2)))/(2*r^9)
    az = (3*R^3*mu*(c31*(sin(omega_rot*t)*y+cos(omega_rot*t)*x)+s31*(cos(omega_rot*t)*y-sin(omega_rot*t)*x))*z*(28*z^2-7*y^2-7*x^2-8*r^2))/(2*r^9)
    return SVector(ax, ay, az)
end

function cs32_potential(r_vector, t, mu, R, c32, s32, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)
    return (-(15*R^3*c32*mu*(sin(omega_rot*t)*y-cos(omega_rot*t)*y+sin(omega_rot*t)*x+cos(omega_rot*t)*x)*(sin(omega_rot*t)*y+cos(omega_rot*t)*y-sin(omega_rot*t)*x+cos(omega_rot*t)*x)*z)-30*R^3*mu*s32*(cos(omega_rot*t)*y-sin(omega_rot*t)*x)*(sin(omega_rot*t)*y+cos(omega_rot*t)*x)*z)/r^7
end

function cs32_perturbation(r_vector, t, mu, R, c32, s32, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)

    ax = -((15*R^3*mu*(c32*(7*sin(omega_rot*t)^2*x*y^2-7*cos(omega_rot*t)^2*x*y^2+28*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y-4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y-7*sin(omega_rot*t)^2*x^3+7*cos(omega_rot*t)^2*x^3+2*r^2*sin(omega_rot*t)^2*x-2*r^2*cos(omega_rot*t)^2*x)+s32*(14*cos(omega_rot*t)*sin(omega_rot*t)*x*y^2-14*sin(omega_rot*t)^2*x^2*y+14*cos(omega_rot*t)^2*x^2*y+2*r^2*sin(omega_rot*t)^2*y-2*r^2*cos(omega_rot*t)^2*y-14*cos(omega_rot*t)*sin(omega_rot*t)*x^3+4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x))*z)/r^9)
    ay = -((15*R^3*mu*(c32*(7*sin(omega_rot*t)^2*y^3-7*cos(omega_rot*t)^2*y^3+28*cos(omega_rot*t)*sin(omega_rot*t)*x*y^2-7*sin(omega_rot*t)^2*x^2*y+7*cos(omega_rot*t)^2*x^2*y-2*r^2*sin(omega_rot*t)^2*y+2*r^2*cos(omega_rot*t)^2*y-4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x)+s32*(14*cos(omega_rot*t)*sin(omega_rot*t)*y^3-14*sin(omega_rot*t)^2*x*y^2+14*cos(omega_rot*t)^2*x*y^2-14*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y-4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y+2*r^2*sin(omega_rot*t)^2*x-2*r^2*cos(omega_rot*t)^2*x))*z)/r^9)
    az = -((15*R^3*mu*(c32*(sin(omega_rot*t)^2*y^2-cos(omega_rot*t)^2*y^2+4*cos(omega_rot*t)*sin(omega_rot*t)*x*y-sin(omega_rot*t)^2*x^2+cos(omega_rot*t)^2*x^2)+s32*(2*cos(omega_rot*t)*sin(omega_rot*t)*y^2-2*sin(omega_rot*t)^2*x*y+2*cos(omega_rot*t)^2*x*y-2*cos(omega_rot*t)*sin(omega_rot*t)*x^2))*(7*z^2-r^2))/r^9)
    return SVector(ax, ay, az)
end

function cs41_potential(r_vector, t, mu, R, c41, s41, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)
    return (5*R^4*c41*mu*(sin(omega_rot*t)*y+cos(omega_rot*t)*x)*z*(4*z^2-3*y^2-3*x^2)+5*R^4*mu*s41*(cos(omega_rot*t)*y-sin(omega_rot*t)*x)*z*(4*z^2-3*y^2-3*x^2))/(2*r^9)
end

function cs41_perturbation(r_vector, t, mu, R, c41, s41, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)

    ax = (5*R^4*c41*mu*z*(36*sin(omega_rot*t)*x*y*z^2+36*cos(omega_rot*t)*x^2*z^2-4*r^2*cos(omega_rot*t)*z^2-27*sin(omega_rot*t)*x*y^3-27*cos(omega_rot*t)*x^2*y^2+3*r^2*cos(omega_rot*t)*y^2-27*sin(omega_rot*t)*x^3*y+6*r^2*sin(omega_rot*t)*x*y-27*cos(omega_rot*t)*x^4+9*r^2*cos(omega_rot*t)*x^2)+5*R^4*mu*s41*z*(36*cos(omega_rot*t)*x*y*z^2-36*sin(omega_rot*t)*x^2*z^2+4*r^2*sin(omega_rot*t)*z^2-27*cos(omega_rot*t)*x*y^3+27*sin(omega_rot*t)*x^2*y^2-3*r^2*sin(omega_rot*t)*y^2-27*cos(omega_rot*t)*x^3*y+6*r^2*cos(omega_rot*t)*x*y+27*sin(omega_rot*t)*x^4-9*r^2*sin(omega_rot*t)*x^2))/(2*r^11)
    ay = (5*R^4*c41*mu*z*(36*sin(omega_rot*t)*y^2*z^2+36*cos(omega_rot*t)*x*y*z^2-4*r^2*sin(omega_rot*t)*z^2-27*sin(omega_rot*t)*y^4-27*cos(omega_rot*t)*x*y^3-27*sin(omega_rot*t)*x^2*y^2+9*r^2*sin(omega_rot*t)*y^2-27*cos(omega_rot*t)*x^3*y+6*r^2*cos(omega_rot*t)*x*y+3*r^2*sin(omega_rot*t)*x^2)+5*R^4*mu*s41*z*(36*cos(omega_rot*t)*y^2*z^2-36*sin(omega_rot*t)*x*y*z^2-4*r^2*cos(omega_rot*t)*z^2-27*cos(omega_rot*t)*y^4+27*sin(omega_rot*t)*x*y^3-27*cos(omega_rot*t)*x^2*y^2+9*r^2*cos(omega_rot*t)*y^2+27*sin(omega_rot*t)*x^3*y-6*r^2*sin(omega_rot*t)*x*y+3*r^2*cos(omega_rot*t)*x^2))/(2*r^11)
    az = (15*R^4*c41*mu*(sin(omega_rot*t)*y+cos(omega_rot*t)*x)*(12*z^4-9*y^2*z^2-9*x^2*z^2-4*r^2*z^2+r^2*y^2+r^2*x^2)+15*R^4*mu*s41*(cos(omega_rot*t)*y-sin(omega_rot*t)*x)*(12*z^4-9*y^2*z^2-9*x^2*z^2-4*r^2*z^2+r^2*y^2+r^2*x^2))/(2*r^11)
    return SVector(ax, ay, az)
end

function cs42_potential(r_vector, t, mu, R, c42, s42, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)
    return (-(15*R^4*c42*mu*(sin(omega_rot*t)*y-cos(omega_rot*t)*y+sin(omega_rot*t)*x+cos(omega_rot*t)*x)*(sin(omega_rot*t)*y+cos(omega_rot*t)*y-sin(omega_rot*t)*x+cos(omega_rot*t)*x)*(6*z^2-y^2-x^2))-30*R^4*mu*s42*(cos(omega_rot*t)*y-sin(omega_rot*t)*x)*(sin(omega_rot*t)*y+cos(omega_rot*t)*x)*(6*z^2-y^2-x^2))/(2*r^9)
end

function cs42_perturbation(r_vector, t, mu, R, c42, s42, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)

    ax = (-(15*R^4*c42*mu*(54*sin(omega_rot*t)^2*x*y^2*z^2-54*cos(omega_rot*t)^2*x*y^2*z^2+216*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y*z^2-24*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y*z^2-54*sin(omega_rot*t)^2*x^3*z^2+54*cos(omega_rot*t)^2*x^3*z^2+12*r^2*sin(omega_rot*t)^2*x*z^2-12*r^2*cos(omega_rot*t)^2*x*z^2-9*sin(omega_rot*t)^2*x*y^4+9*cos(omega_rot*t)^2*x*y^4-36*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y^3+4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y^3-36*cos(omega_rot*t)*sin(omega_rot*t)*x^4*y+12*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y+9*sin(omega_rot*t)^2*x^5-9*cos(omega_rot*t)^2*x^5-4*r^2*sin(omega_rot*t)^2*x^3+4*r^2*cos(omega_rot*t)^2*x^3))-30*R^4*mu*s42*(54*cos(omega_rot*t)*sin(omega_rot*t)*x*y^2*z^2-54*sin(omega_rot*t)^2*x^2*y*z^2+54*cos(omega_rot*t)^2*x^2*y*z^2+6*r^2*sin(omega_rot*t)^2*y*z^2-6*r^2*cos(omega_rot*t)^2*y*z^2-54*cos(omega_rot*t)*sin(omega_rot*t)*x^3*z^2+12*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x*z^2-9*cos(omega_rot*t)*sin(omega_rot*t)*x*y^4+9*sin(omega_rot*t)^2*x^2*y^3-9*cos(omega_rot*t)^2*x^2*y^3-r^2*sin(omega_rot*t)^2*y^3+r^2*cos(omega_rot*t)^2*y^3+9*sin(omega_rot*t)^2*x^4*y-9*cos(omega_rot*t)^2*x^4*y-3*r^2*sin(omega_rot*t)^2*x^2*y+3*r^2*cos(omega_rot*t)^2*x^2*y+9*cos(omega_rot*t)*sin(omega_rot*t)*x^5-4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x^3))/(2*r^11)
    ay = (-(15*R^4*c42*mu*(54*sin(omega_rot*t)^2*y^3*z^2-54*cos(omega_rot*t)^2*y^3*z^2+216*cos(omega_rot*t)*sin(omega_rot*t)*x*y^2*z^2-54*sin(omega_rot*t)^2*x^2*y*z^2+54*cos(omega_rot*t)^2*x^2*y*z^2-12*r^2*sin(omega_rot*t)^2*y*z^2+12*r^2*cos(omega_rot*t)^2*y*z^2-24*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x*z^2-9*sin(omega_rot*t)^2*y^5+9*cos(omega_rot*t)^2*y^5-36*cos(omega_rot*t)*sin(omega_rot*t)*x*y^4+4*r^2*sin(omega_rot*t)^2*y^3-4*r^2*cos(omega_rot*t)^2*y^3-36*cos(omega_rot*t)*sin(omega_rot*t)*x^3*y^2+12*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x*y^2+9*sin(omega_rot*t)^2*x^4*y-9*cos(omega_rot*t)^2*x^4*y+4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x^3))-30*R^4*mu*s42*(54*cos(omega_rot*t)*sin(omega_rot*t)*y^3*z^2-54*sin(omega_rot*t)^2*x*y^2*z^2+54*cos(omega_rot*t)^2*x*y^2*z^2-54*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y*z^2-12*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y*z^2+6*r^2*sin(omega_rot*t)^2*x*z^2-6*r^2*cos(omega_rot*t)^2*x*z^2-9*cos(omega_rot*t)*sin(omega_rot*t)*y^5+9*sin(omega_rot*t)^2*x*y^4-9*cos(omega_rot*t)^2*x*y^4+4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y^3+9*sin(omega_rot*t)^2*x^3*y^2-9*cos(omega_rot*t)^2*x^3*y^2-3*r^2*sin(omega_rot*t)^2*x*y^2+3*r^2*cos(omega_rot*t)^2*x*y^2+9*cos(omega_rot*t)*sin(omega_rot*t)*x^4*y-r^2*sin(omega_rot*t)^2*x^3+r^2*cos(omega_rot*t)^2*x^3))/(2*r^11)
    az = (-(45*R^4*c42*mu*(sin(omega_rot*t)*y-cos(omega_rot*t)*y+sin(omega_rot*t)*x+cos(omega_rot*t)*x)*(sin(omega_rot*t)*y+cos(omega_rot*t)*y-sin(omega_rot*t)*x+cos(omega_rot*t)*x)*z*(18*z^2-3*y^2-3*x^2-4*r^2))-90*R^4*mu*s42*(cos(omega_rot*t)*y-sin(omega_rot*t)*x)*(sin(omega_rot*t)*y+cos(omega_rot*t)*x)*z*(18*z^2-3*y^2-3*x^2-4*r^2))/(2*r^11)
    return SVector(ax, ay, az)
end

function cs43_potential(r_vector, t, mu, R, c43, s43, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)
    return (105*R^4*mu*s43*(cos(omega_rot*t)*y-sin(omega_rot*t)*x)*(3*sin(omega_rot*t)^2*y^2-cos(omega_rot*t)^2*y^2+8*cos(omega_rot*t)*sin(omega_rot*t)*x*y-sin(omega_rot*t)^2*x^2+3*cos(omega_rot*t)^2*x^2)*z+105*R^4*c43*mu*(sin(omega_rot*t)*y+cos(omega_rot*t)*x)*(sin(omega_rot*t)^2*y^2-3*cos(omega_rot*t)^2*y^2+8*cos(omega_rot*t)*sin(omega_rot*t)*x*y-3*sin(omega_rot*t)^2*x^2+cos(omega_rot*t)^2*x^2)*z)/r^9
end

function cs43_perturbation(r_vector, t, mu, R, c43, s43, omega_rot)
    x,y,z = r_vector
    r = norm(r_vector)

    ax = (315*R^4*c43*mu*(3*sin(omega_rot*t)^3*x*y^3-9*cos(omega_rot*t)^2*sin(omega_rot*t)*x*y^3+27*cos(omega_rot*t)*sin(omega_rot*t)^2*x^2*y^2-9*cos(omega_rot*t)^3*x^2*y^2-3*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*y^2+r^2*cos(omega_rot*t)^3*y^2-9*sin(omega_rot*t)^3*x^3*y+27*cos(omega_rot*t)^2*sin(omega_rot*t)*x^3*y+2*r^2*sin(omega_rot*t)^3*x*y-6*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*x*y-9*cos(omega_rot*t)*sin(omega_rot*t)^2*x^4+3*cos(omega_rot*t)^3*x^4+3*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*x^2-r^2*cos(omega_rot*t)^3*x^2)*z+315*R^4*mu*s43*(9*cos(omega_rot*t)*sin(omega_rot*t)^2*x*y^3-3*cos(omega_rot*t)^3*x*y^3-9*sin(omega_rot*t)^3*x^2*y^2+27*cos(omega_rot*t)^2*sin(omega_rot*t)*x^2*y^2+r^2*sin(omega_rot*t)^3*y^2-3*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*y^2-27*cos(omega_rot*t)*sin(omega_rot*t)^2*x^3*y+9*cos(omega_rot*t)^3*x^3*y+6*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*x*y-2*r^2*cos(omega_rot*t)^3*x*y+3*sin(omega_rot*t)^3*x^4-9*cos(omega_rot*t)^2*sin(omega_rot*t)*x^4-r^2*sin(omega_rot*t)^3*x^2+3*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*x^2)*z)/r^11
    ay = (315*R^4*c43*mu*(3*sin(omega_rot*t)^3*y^4-9*cos(omega_rot*t)^2*sin(omega_rot*t)*y^4+27*cos(omega_rot*t)*sin(omega_rot*t)^2*x*y^3-9*cos(omega_rot*t)^3*x*y^3-9*sin(omega_rot*t)^3*x^2*y^2+27*cos(omega_rot*t)^2*sin(omega_rot*t)*x^2*y^2-r^2*sin(omega_rot*t)^3*y^2+3*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*y^2-9*cos(omega_rot*t)*sin(omega_rot*t)^2*x^3*y+3*cos(omega_rot*t)^3*x^3*y-6*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*x*y+2*r^2*cos(omega_rot*t)^3*x*y+r^2*sin(omega_rot*t)^3*x^2-3*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*x^2)*z+315*R^4*mu*s43*(9*cos(omega_rot*t)*sin(omega_rot*t)^2*y^4-3*cos(omega_rot*t)^3*y^4-9*sin(omega_rot*t)^3*x*y^3+27*cos(omega_rot*t)^2*sin(omega_rot*t)*x*y^3-27*cos(omega_rot*t)*sin(omega_rot*t)^2*x^2*y^2+9*cos(omega_rot*t)^3*x^2*y^2-3*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*y^2+r^2*cos(omega_rot*t)^3*y^2+3*sin(omega_rot*t)^3*x^3*y-9*cos(omega_rot*t)^2*sin(omega_rot*t)*x^3*y+2*r^2*sin(omega_rot*t)^3*x*y-6*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*x*y+3*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*x^2-r^2*cos(omega_rot*t)^3*x^2)*z)/r^11
    az = (105*R^4*mu*s43*(cos(omega_rot*t)*y-sin(omega_rot*t)*x)*(3*sin(omega_rot*t)^2*y^2-cos(omega_rot*t)^2*y^2+8*cos(omega_rot*t)*sin(omega_rot*t)*x*y-sin(omega_rot*t)^2*x^2+3*cos(omega_rot*t)^2*x^2)*(3*z-r)*(3*z+r)+105*R^4*c43*mu*(sin(omega_rot*t)*y+cos(omega_rot*t)*x)*(sin(omega_rot*t)^2*y^2-3*cos(omega_rot*t)^2*y^2+8*cos(omega_rot*t)*sin(omega_rot*t)*x*y-3*sin(omega_rot*t)^2*x^2+cos(omega_rot*t)^2*x^2)*(3*z-r)*(3*z+r))/r^11
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
    shadow_function_smooth(r_sat, r_sun, Re) -> Float64

Calculates the smooth illumination factor (approximating penumbra) using a hyperbolic tangent transition.
Returns 0.0 for total umbra, 1.0 for full illumination, and intermediate continuous values across the penumbra.
"""
function shadow_function_smooth(r_sat::AbstractVector, r_sun::AbstractVector, Re::Real)
    rsun_norm = norm(r_sun)
    # s_c determines the shadow state variable using conical geometry.
    s_c = (dot(r_sat, r_sun) / rsun_norm) + sqrt(abs(dot(r_sat, r_sat) - Re^2))

    # Smooth transition using tanh
    return 0.5 * (1.0 + tanh(1e9 * s_c))
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
- `use_shadow_model`: If true or :smooth, applies smooth hyperbolic shadow model. If :cylindrical, applies cylindrical shadow model. Defaults to true.

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
    use_shadow_model::Union{Bool, Symbol} = true
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
    if use_shadow_model === :smooth || use_shadow_model === true
        nu = shadow_function_smooth(ustrip.(r_vector), ustrip.(r_sun_vector), ustrip(R))
    elseif use_shadow_model === :cylindrical
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

# --- lorentz perturbation ---
"""
    acc_lorentz(r, v, qm, omega_rot, B_mag=8.0e15) -> SVector{3}

Calculates the perturbing acceleration due to the Lorentz force for a charged body in a planetary magnetic field.
Decomposes the force into Radial (R_L), Transversal (T_L), and Normal (W_L) components.

# Arguments
- `r::AbstractVector`: Satellite position vector in km (inertial frame).
- `v::AbstractVector`: Satellite velocity vector in km/s (inertial frame).
- `qm::Real`: Charge-to-mass ratio in C/kg.
- `omega_rot::Real`: Central body rotation rate in rad/s.
- `B_mag::Real`: Magnetic dipole field constant (default 8.0e15).

# Returns
- `SVector{3}`: Acceleration vector in km/s^2.
"""
function acc_lorentz(r::AbstractVector, v::AbstractVector, qm::Real, omega_rot::Real, B_mag::Real=8.0e15)
    # convert to SI units (m and m/s)
    r_m = r .* 1000.0
    v_m = v .* 1000.0
    
    r_mag = norm(r_m)

    T = promote_type(eltype(r_m), eltype(v_m), typeof(qm), typeof(omega_rot), typeof(B_mag))

    if qm == 0 || r_mag == 0
        return SVector{3,T}(0.0, 0.0, 0.0)
    end

    # Constants
    omega = T(omega_rot)
    B_mag_val = T(B_mag)

    # Angular momentum vector (h)
    h_vec = cross(r_m, v_m)
    h_mag = norm(h_vec)

    if h_mag == 0
        return SVector{3,T}(0.0, 0.0, 0.0)
    end

    # Extraction of cos(i) and sin(i)
    hz = h_vec[3]
    c = hz / h_mag
    s = sqrt(max(T(0.0), 1.0 - c^2))

    # Trigonometric equivalents
    sin_fg = s > 1e-8 ? r_m[3] / (r_mag * s) : T(0.0)
    
    hx, hy = h_vec[1], h_vec[2]
    n_mag = sqrt(hx^2 + hy^2)
    cos_fg = n_mag > 1e-8 ? (-r_m[1]*hy + r_m[2]*hx) / (r_mag * n_mag) : T(1.0)

    # Radial derivative dr/dt
    r_dot = dot(r_m, v_m) / r_mag
    
    # Magnetic acceleration components (m/s^2)
    R_L = (B_mag_val * qm * omega / r_mag^2) * (1.0 - s^2 * sin_fg^2 - (c * h_mag) / (omega * r_mag^2))
    T_L = (B_mag_val * qm * omega / r_mag^2) * (2.0 * s^2 * sin_fg * cos_fg + (c * r_dot) / (omega * r_mag))
    W_L = (B_mag_val * qm * s / r_mag^2) * (2.0 * (omega * c - h_mag / r_mag^2) * sin_fg - (r_dot / r_mag) * cos_fg)

    # Mapping RTW frame to XYZ (inertial)
    e_R = r_m / r_mag
    e_W = h_vec / h_mag
    e_T = cross(e_W, e_R)

    # Final acceleration vector (m/s^2)
    a_vec_m = R_L * e_R + T_L * e_T + W_L * e_W

    # Return in km/s^2
    return SVector{3,T}(a_vec_m[1]/1000.0, a_vec_m[2]/1000.0, a_vec_m[3]/1000.0)
end

# --- earth precession perturbation ---
"""
    PrecessionPerturbation(v_vector, t) -> SVector{3}

Calculates the Coriolis apparent acceleration generated by Earth's general precession (Lieske et al., IAU 1976).

# Arguments
- `v_vector::AbstractVector`: Satellite velocity vector in km/s.
- `t::Real`: Ephemeris time in seconds since J2000.

# Returns
- `SVector{3}`: Apparent Coriolis acceleration vector in km/s^2.
"""
function PrecessionPerturbation(v_vector::AbstractVector, t::Real)
    # T in Julian centuries from J2000 epoch
    T = t / (36525.0 * 86400.0)
    
    # Precession angles in arcseconds
    zeta_arcsec = 2306.2181 * T + 0.30188 * T^2 + 0.017988 * T^3
    theta_arcsec = 2004.3109 * T - 0.42665 * T^2 - 0.041833 * T^3
    z_arcsec = 2306.2181 * T + 1.09468 * T^2 + 0.018203 * T^3
    
    # Conversion of angles to radians (1 arcsec = pi / 648000 rad)
    arcsec2rad = pi / 648000.0
    zeta = zeta_arcsec * arcsec2rad
    theta = theta_arcsec * arcsec2rad
    z_angle = z_arcsec * arcsec2rad
    
    # Rates in arcseconds per century
    zeta_dot_arcsec_cy = 2306.2181 + 0.60376 * T + 0.053964 * T^2
    theta_dot_arcsec_cy = 2004.3109 - 0.8533 * T - 0.125499 * T^2
    z_dot_arcsec_cy = 2306.2181 + 2.18936 * T + 0.054609 * T^2
    
    # Conversion of rates to radians per second
    arcsec_cy_2_rad_s = arcsec2rad / 3155760000.0
    
    zeta_dot = zeta_dot_arcsec_cy * arcsec_cy_2_rad_s
    theta_dot = theta_dot_arcsec_cy * arcsec_cy_2_rad_s
    z_dot = z_dot_arcsec_cy * arcsec_cy_2_rad_s
    
    # Angular velocity components of the precessing frame
    Px = theta_dot * sin(zeta) - z_dot * cos(zeta) * sin(theta)
    Py = theta_dot * cos(zeta) + z_dot * sin(theta) * sin(zeta)
    Pz = -z_dot * cos(theta) - zeta_dot
    
    P_vec = SVector{3}(Px, Py, Pz)
    
    # Coriolis acceleration: a = -2 * (omega x v)
    a_prec = -2.0 * cross(P_vec, v_vector)
    
    return a_prec
end

# --- earth radiation pressure (albedo + IR) ---
"""
    earth_radiation_pressure(r_vector, R_earth, alpha, cr, r_sun_vector;
                             albedo=0.30, thermal_ir_flux=237.0, dist_scale=1.0) -> SVector{3}

Calculates the perturbing acceleration due to Earth Radiation Pressure (ERP), including:
1. Earth Albedo (diffuse reflection of solar radiation from the illuminated Earth surface)
2. Terrestrial Infrared (IR) thermal emission (emitted continuously day and night)

Based on the analytical model of Knocke, Ries, and Tapley (1988) / Montenbruck & Gill (2000, Sec. 3.4.2).

# Arguments
- `r_vector::AbstractVector`: Satellite position vector in km (inertial frame, Earth-centered).
- `R_earth::Real`: Earth equatorial radius in km.
- `alpha::Real`: Spacecraft area-to-mass ratio [m^2/kg].
- `cr::Real`: Spacecraft radiation pressure coefficient (reflectivity, dimensionless).
- `r_sun_vector::AbstractVector`: Sun position vector in km (inertial frame, Earth-centered).

# Keyword Arguments
- `albedo::Real`: Mean Earth albedo coefficient (default: 0.30).
- `thermal_ir_flux::Real`: Mean Earth infrared flux [W/m^2] (default: 237.0 W/m^2).
- `dist_scale::Real`: Distance scaling factor (default: 1.0).

# Returns
- `SVector{3}`: Perturbing acceleration in km/s^2.
"""
function earth_radiation_pressure(
    r_vector::AbstractVector,
    R_earth::Real,
    alpha::Real,
    cr::Real,
    r_sun_vector::AbstractVector;
    albedo::Real = 0.30,
    thermal_ir_flux::Real = 237.0,
    dist_scale::Real = 1.0
)
    r_mag = norm(r_vector)
    if iszero(r_mag) || iszero(alpha) || iszero(cr)
        return SVector{3}(0.0, 0.0, 0.0)
    end

    # Unit vector from Earth center to satellite (radial outward)
    r_hat = r_vector / r_mag

    # Dilution factor (R_earth / r)^2
    r_mag_phys = r_mag * dist_scale
    ratio = min(1.0, Float64(R_earth) / r_mag_phys)
    dilution = ratio^2

    # 1. Terrestrial Infrared (IR) emission pressure [N/m^2]
    # P_IR = (F_IR / c) * (R_earth / r)^2
    c_val = Float64(ustrip(Constants.C_SI))
    flux_val = Float64(ustrip(thermal_ir_flux))
    P_ir = (flux_val / c_val) * dilution

    # 2. Albedo reflection pressure [N/m^2]
    r_sun_sat = r_vector - r_sun_vector
    dist_sat_to_sun_phys = norm(r_sun_sat) * dist_scale
    P_sun = SolarRadiationPressure(dist_sat_to_sun_phys * u"km")

    r_sun_mag = norm(r_sun_vector)
    s_hat = iszero(r_sun_mag) ? SVector{3}(0.0, 0.0, 0.0) : (r_sun_vector / r_sun_mag)

    # Solar zenith angle at sub-satellite point
    cos_theta_s = dot(r_hat, s_hat)
    illumination = max(0.0, cos_theta_s)

    P_albedo = albedo * P_sun * dilution * illumination

    # Total Earth Radiation Pressure [N/m^2]
    P_total = P_ir + P_albedo

    # Acceleration in km/s^2: a = (P * cr * alpha) / 1000
    acel_magnitude_km_s2 = (P_total * cr * alpha) / 1000.0
    a_erp = acel_magnitude_km_s2 .* r_hat

    return SVector{3}(a_erp)
end

# --- yarkovsky effect (anisotropic thermal re-emission) ---
"""
    yarkovsky_perturbation(r_vector, v_vector, alpha, r_sun_vector;
                           cy=0.15, lag_angle_deg=25.0, spin_axis=nothing,
                           dist_scale=1.0, time_scale=1.0, shadow_factor=1.0) -> SVector{3}

Calculates the perturbing acceleration due to the diurnal Yarkovsky effect
(anisotropic thermal re-emission of solar radiation).

# Arguments
- `r_vector::AbstractVector`: Satellite position vector in km (inertial frame).
- `v_vector::AbstractVector`: Satellite velocity vector in km/s (inertial frame).
- `alpha::Real`: Spacecraft / debris area-to-mass ratio [m^2/kg].
- `r_sun_vector::AbstractVector`: Sun position vector in km (inertial frame).

# Keyword Arguments
- `cy::Real`: Yarkovsky thermal recoil efficiency coefficient (dimensionless, typically 0.10 - 0.25, default 0.15).
- `lag_angle_deg::Real`: Diurnal thermal lag angle in degrees (default: 25.0°).
- `spin_axis::Union{AbstractVector, Nothing}`: Spacecraft spin axis unit vector in inertial frame.
  If `nothing`, defaults to the orbit normal (unit angular momentum vector `h_hat = (r x v) / ||r x v||`),
  representing prograde rotation.
- `dist_scale::Real`: Distance scaling factor (default: 1.0).
- `time_scale::Real`: Time scaling factor (default: 1.0).
- `shadow_factor::Real`: Illumination factor in [0, 1] (0 in eclipse, 1 in full sunlight). Default: 1.0.

# Returns
- `SVector{3}`: Perturbing acceleration in km/s^2.
"""
function yarkovsky_perturbation(
    r_vector::AbstractVector,
    v_vector::AbstractVector,
    alpha::Real,
    r_sun_vector::AbstractVector;
    cy::Real = 0.15,
    lag_angle_deg::Real = 25.0,
    spin_axis::Union{AbstractVector, Nothing} = nothing,
    dist_scale::Real = 1.0,
    time_scale::Real = 1.0,
    shadow_factor::Real = 1.0
)
    if iszero(shadow_factor) || iszero(alpha) || iszero(cy)
        return SVector{3}(0.0, 0.0, 0.0)
    end

    r_sat_to_sun = r_sun_vector - r_vector
    dist_sat_to_sun = norm(r_sat_to_sun)
    if iszero(dist_sat_to_sun)
        return SVector{3}(0.0, 0.0, 0.0)
    end
    s_hat = r_sat_to_sun / dist_sat_to_sun

    # Determine spin axis unit vector w_hat
    w_hat = if isnothing(spin_axis)
        h_vec = cross(r_vector, v_vector)
        h_norm = norm(h_vec)
        if iszero(h_norm)
            return SVector{3}(0.0, 0.0, 0.0)
        end
        h_vec / h_norm
    else
        spin_svec = SVector{3}(spin_axis[1], spin_axis[2], spin_axis[3])
        spin_norm = norm(spin_svec)
        if iszero(spin_norm)
            return SVector{3}(0.0, 0.0, 0.0)
        end
        spin_svec / spin_norm
    end

    # Transverse unit vector in spin equator: e_trans = (w_hat x s_hat) / ||w_hat x s_hat||
    transverse = cross(w_hat, s_hat)
    trans_norm = norm(transverse)
    if iszero(trans_norm)
        return SVector{3}(0.0, 0.0, 0.0)
    end
    e_trans = transverse / trans_norm

    P_sun = SolarRadiationPressure(dist_sat_to_sun * dist_scale * u"km")
    theta_lag = deg2rad(lag_angle_deg)
    a_yark_m_s2 = shadow_factor * P_sun * cy * alpha * sin(theta_lag)
    a_yark_km_s2 = (a_yark_m_s2 / 1000.0) .* e_trans

    return SVector{3}(a_yark_km_s2)
end

# --- general relativity perturbations (post-newtonian) ---

"""
    relativity_geodesic_perturbation(r_vector::AbstractVector, v_vector::AbstractVector, mu::Real;
                                     c_light::Real = ustrip(u"km/s", Constants.C_SI)) -> SVector{3}

Calculates the 1st-order Post-Newtonian (1PN) Schwarzschild/geodesic general relativistic
perturbing acceleration due to spacetime curvature of the central body.

Formula (IERS Conventions 2010, Ch. 10; Montenbruck & Gill, 2000, Sec. 3.6.1):
    a_GR = (mu / (c^2 * r^3)) * [ (4*mu/r - v^2)*r + 4*(r . v)*v ]

This causes a secular advance of the argument of perigee (~3.84 arcsec/year in LEO).

# Arguments
- `r_vector::AbstractVector`: Satellite position vector in km (inertial frame).
- `v_vector::AbstractVector`: Satellite velocity vector in km/s (inertial frame).
- `mu::Real`: Gravitational parameter of the central body in km^3/s^2.

# Keyword Arguments
- `c_light::Real`: Speed of light in km/s (default: ~299792.458 km/s).

# Returns
- `SVector{3}`: Perturbing acceleration in km/s^2.
"""
function relativity_geodesic_perturbation(
    r_vector::AbstractVector,
    v_vector::AbstractVector,
    mu::Real;
    c_light::Real = ustrip(u"km/s", Constants.C_SI)
)
    r2 = dot(r_vector, r_vector)
    if iszero(r2) || iszero(mu)
        return SVector{3}(0.0, 0.0, 0.0)
    end
    r = sqrt(r2)
    v2 = dot(v_vector, v_vector)
    r_dot_v = dot(r_vector, v_vector)

    c_val = Float64(c_light)
    c2 = c_val^2
    factor = Float64(mu) / (c2 * r^3)

    # [(4*mu/r - v^2)*r + 4*(r . v)*v]
    term_r = 4.0 * Float64(mu) / r - v2
    term_v = 4.0 * r_dot_v

    ax = factor * (term_r * r_vector[1] + term_v * v_vector[1])
    ay = factor * (term_r * r_vector[2] + term_v * v_vector[2])
    az = factor * (term_r * r_vector[3] + term_v * v_vector[3])

    return SVector{3}(ax, ay, az)
end

"""
    relativity_lense_thirring_perturbation(r_vector::AbstractVector, v_vector::AbstractVector, mu::Real,
                                           J_spin::Union{AbstractVector, Real};
                                           c_light::Real = ustrip(u"km/s", Constants.C_SI)) -> SVector{3}

Calculates the Lense-Thirring (frame-dragging / gravitomagnetic) general relativistic
perturbing acceleration due to the central body's rotational angular momentum.

Formula (IERS Conventions 2010, Ch. 10; Montenbruck & Gill, 2000, Sec. 3.6.2):
    a_LT = (2*mu / (c^2 * r^3)) * [ (3/r^2) * ((r x v) . J) * r + (v x J) ]

where J is the central body's angular momentum per unit mass in km^2/s.
For Earth: J_vector = [0, 0, J_z] with J_z ≈ 0.3307 * R_earth^2 * omega_rot ≈ 981.0 km^2/s.

# Arguments
- `r_vector::AbstractVector`: Satellite position vector in km (inertial frame).
- `v_vector::AbstractVector`: Satellite velocity vector in km/s (inertial frame).
- `mu::Real`: Gravitational parameter of the central body in km^3/s^2.
- `J_spin::Union{AbstractVector, Real}`: Angular momentum per unit mass in km^2/s.
  If a scalar, assumes aligned with +Z axis: `[0, 0, J_spin]`.

# Keyword Arguments
- `c_light::Real`: Speed of light in km/s (default: ~299792.458 km/s).

# Returns
- `SVector{3}`: Perturbing acceleration in km/s^2.
"""
function relativity_lense_thirring_perturbation(
    r_vector::AbstractVector,
    v_vector::AbstractVector,
    mu::Real,
    J_spin::Union{AbstractVector, Real};
    c_light::Real = ustrip(u"km/s", Constants.C_SI)
)
    r2 = dot(r_vector, r_vector)
    if iszero(r2) || iszero(mu)
        return SVector{3}(0.0, 0.0, 0.0)
    end
    r = sqrt(r2)

    J_vec = if J_spin isa Real
        SVector{3, Float64}(0.0, 0.0, Float64(J_spin))
    else
        SVector{3, Float64}(Float64(J_spin[1]), Float64(J_spin[2]), Float64(J_spin[3]))
    end

    if iszero(norm(J_vec))
        return SVector{3}(0.0, 0.0, 0.0)
    end

    c_val = Float64(c_light)
    c2 = c_val^2
    factor = (2.0 * Float64(mu)) / (c2 * r^3)

    # h = r x v
    h_vec = cross(r_vector, v_vector)

    # (3/r^2) * ((r x v) . J) * r
    h_dot_J = dot(h_vec, J_vec)
    term1 = ((3.0 * h_dot_J) / r2) .* r_vector

    # (v x J)
    term2 = cross(v_vector, J_vec)

    a_lt = factor .* (term1 + term2)
    return SVector{3}(a_lt)
end

# ===============================================================================================
# 7. SOLID EARTH TIDES (Deformações Geofísicas / Marés da Terra Sólida)
# ===============================================================================================

"""
    solid_earth_tides_potential(r_sat_vector::AbstractVector, r_body_vector::AbstractVector, mu_body::Real, R_central::Real, k2::Real) -> Real

Calculates the perturbing gravitational potential energy due to the degree-2 tidal deformation (solid Earth tides)
raised on the central body (Earth) by an external celestial body (Moon or Sun).

The tidal deformation of the elastic body creates an external gravitational potential:
    ΔU = (k2 / 2) * (mu_body * R_central^5) / (r_body^3 * r_sat^3) * [ 3 (r̂_sat · r̂_body)^2 - 1 ]

This potential represents the unified sum of all spherical harmonic corrections (ΔC20, ΔC21, ΔS21, ΔC22, ΔS22)
via the Legendre addition theorem.

# Arguments
- `r_sat_vector`: Geocentric position vector of the satellite [km or DU].
- `r_body_vector`: Geocentric position vector of the tide-generating third body (Moon/Sun) [km or DU].
- `mu_body`: Gravitational parameter of the tide-generating body [km^3/s^2 or canonical].
- `R_central`: Equatorial radius of the central deformed body [km or DU].
- `k2`: Second-degree tidal Love number of the central body (dimensionless, ~0.30 for Earth).

# Returns
- `ΔU`: Perturbing gravitational potential [km^2/s^2 or canonical].
"""
function solid_earth_tides_potential(
    r_sat_vector::AbstractVector,
    r_body_vector::AbstractVector,
    mu_body::Real,
    R_central::Real,
    k2::Real
)
    r2 = dot(r_sat_vector, r_sat_vector)
    rb2 = dot(r_body_vector, r_body_vector)
    if iszero(r2) || iszero(rb2) || iszero(mu_body) || iszero(k2) || iszero(R_central)
        return 0.0
    end

    r_sat = sqrt(r2)
    r_body = sqrt(rb2)

    cos_psi = dot(r_sat_vector, r_body_vector) / (r_sat * r_body)
    P2 = 0.5 * (3.0 * cos_psi^2 - 1.0)

    factor = (Float64(k2) * Float64(mu_body) * Float64(R_central)^5) / (r_body^3 * r_sat^3)
    return factor * P2
end

"""
    solid_earth_tides_perturbation(r_sat_vector::AbstractVector, r_body_vector::AbstractVector, mu_body::Real, R_central::Real, k2::Real) -> SVector{3, Float64}

Calculates the perturbing acceleration vector on a satellite due to the solid body tides (degree-2 tidal deformation)
raised on the central body by an external celestial body (primarily the Moon and the Sun).

The formula corresponds to the exact spatial gradient ∇(ΔU) of the degree-2 tidal potential:
    a_tide = - (3/2) * k2 * (mu_body * R_central^5) / (r_body^3 * r_sat^4) * [ (5 cos²ψ - 1) r̂_sat - 2 cosψ r̂_body ]

where:
    r̂_sat = r_sat / ‖r_sat‖
    r̂_body = r_body / ‖r_body‖
    cosψ = r̂_sat · r̂_body

This formulation accounts simultaneously for all periodic tidal variations in the Earth's gravity field 
(ΔC20, ΔC21, ΔS21, ΔC22, ΔS22), producing periodic perturbations of ~10^-7 m/s^2 (~10^-8 g) essential for 
geodetic and altimetry satellites (GRACE, Sentinel, LAGEOS).

# Arguments
- `r_sat_vector`: Geocentric position vector of the satellite [km or DU].
- `r_body_vector`: Geocentric position vector of the tide-generating third body (Moon/Sun) [km or DU].
- `mu_body`: Gravitational parameter of the tide-generating body [km^3/s^2 or canonical].
- `R_central`: Equatorial radius of the central deformed body [km or DU].
- `k2`: Second-degree tidal Love number of the central body (dimensionless, ~0.30 for Earth).

# Returns
- `a_tide::SVector{3, Float64}`: Perturbing acceleration vector [km/s^2 or canonical].
"""
function solid_earth_tides_perturbation(
    r_sat_vector::AbstractVector,
    r_body_vector::AbstractVector,
    mu_body::Real,
    R_central::Real,
    k2::Real
)
    r2 = dot(r_sat_vector, r_sat_vector)
    rb2 = dot(r_body_vector, r_body_vector)
    if iszero(r2) || iszero(rb2) || iszero(mu_body) || iszero(k2) || iszero(R_central)
        return SVector{3, Float64}(0.0, 0.0, 0.0)
    end

    r_sat = sqrt(r2)
    r_body = sqrt(rb2)

    dot_rb = dot(r_sat_vector, r_body_vector)
    q = dot_rb / rb2

    # C = (3/2) * k2 * mu_body * R_central^5 / (r_body^3 * r_sat^5)
    C = (1.5 * Float64(k2) * Float64(mu_body) * Float64(R_central)^5) / (r_body^3 * r2 * r2 * r_sat)

    term1 = 5.0 * (q^2) * (rb2 / r2) - 1.0
    term2 = 2.0 * q

    ax = C * (term2 * r_body_vector[1] - term1 * r_sat_vector[1])
    ay = C * (term2 * r_body_vector[2] - term1 * r_sat_vector[2])
    az = C * (term2 * r_body_vector[3] - term1 * r_sat_vector[3])

    return SVector{3, Float64}(ax, ay, az)
end


# ===============================================================================================
# 8. SPACE ELECTRODYNAMICS: COULOMB DRAG / PLASMA DRAG
# ===============================================================================================

"""
    coulomb_effective_area_factor(v_rel_ms::Real, potential::Real;
                                  am::Real = 0.01,
                                  debye_length::Real = 0.02,
                                  ion_mass::Real = 2.6567e-26) -> Real

Calculates the effective cross-sectional area enhancement factor (η_cross) for a charged body
moving through an ionospheric plasma (Al'pert et al. 1965; Knechtel & Pitts 1964; Hastings & Garrett 1996).

The effective area accounts for:
1. Orbital-Motion-Limited (OML) electrostatic collection:
   η_coll = 1 + (2 * e * |Φ_s|) / (m_i * v_rel^2)
2. Coulomb scattering of deflected ions through the Debye sheath:
   η_scat = 2 * (λ_D / R_s)^2 * (e * |Φ_s| / E_k)^2 * ln(1 + λ_D / b_min)

# Arguments
- `v_rel_ms`: Relative velocity magnitude in m/s.
- `potential`: Spacecraft surface electric potential [V] relative to ambient plasma (typically -1 to -10 V floating).
- `am`: Area-to-mass ratio [m^2/kg] used to estimate the characteristic physical radius.
- `debye_length`: Plasma Debye shielding length [m] (typically 0.01 to 0.05 m in LEO).
- `ion_mass`: Ambient dominant ion mass in kg (default: 2.6567e-26 kg for O+).

# Returns
- `η_cross`: Dimensionless ratio A_eff / A_geom (>= 1.0).
"""
function coulomb_effective_area_factor(
    v_rel_ms::Real,
    potential::Real;
    am::Real = 0.01,
    debye_length::Real = 0.02,
    ion_mass::Real = 2.6567e-26
)
    if v_rel_ms <= 0.0 || iszero(potential)
        return 1.0
    end

    e_charge = 1.602176634e-19
    E_k = 0.5 * Float64(ion_mass) * (Float64(v_rel_ms)^2)
    V_s = abs(Float64(potential))

    chi = (e_charge * V_s) / max(E_k, 1e-25)

    # Characteristic equivalent body radius R_s [m]
    R_s = max(0.02, sqrt(max(Float64(am), 1e-4) / Float64(pi)))

    # Coulomb scattering logarithm ln(1 + lambda_D / b0)
    ln_lambda = log(1.0 + Float64(debye_length) / (R_s * max(chi, 0.01)))

    eta_cross = 1.0 + chi + 2.0 * ((Float64(debye_length) / R_s)^2) * (chi^2) * ln_lambda
    return max(1.0, eta_cross)
end

"""
    coulomb_drag_acceleration(
        r_vector::AbstractVector,
        v_vector::AbstractVector,
        rho_atm::Real,
        cd::Real,
        am::Real,
        omega_rot::Real;
        potential::Real = -5.0,
        ion_density::Union{Real, Nothing} = nothing,
        debye_length::Real = 0.02,
        ion_mass::Real = 2.6567e-26,
        tether_length::Real = 0.0,
        f_ion::Real = 0.02,
        R_body_km::Real = 6378.137
    ) -> SVector{3, Float64}

Calculates the perturbing acceleration vector due to Coulomb Drag (plasma drag) on a charged satellite
or electrodynamic plasma brake tether in the ionosphere.

In Low Earth Orbit (LEO), interaction of the charged spacecraft surface (or negatively biased tether)
with the ambient ionospheric plasma (primarily O+ ions) transfers forward momentum to the spacecraft,
producing a continuous braking deceleration that supplements neutral aerodynamic drag.

# Arguments
- `r_vector`: Spacecraft position vector [km].
- `v_vector`: Spacecraft velocity vector [km/s].
- `rho_atm`: Neutral atmospheric mass density [kg/m^3] (from atmospheric model).
- `cd`: Aerodynamic drag coefficient (dimensionless, ~2.2).
- `am`: Area-to-mass ratio [m^2/kg].
- `omega_rot`: Planetary rotation rate [rad/s].
- `potential`: Spacecraft surface electric potential in Volts (e.g. -5.0 V natural, or -500 V biased tether).
- `ion_density`: Ambient ion number density in m^-3 (if `nothing`, estimated from `rho_atm` or Chapman layer).
- `debye_length`: Debye shielding length in meters (default 0.02 m).
- `ion_mass`: Ambient ion mass in kg (default 2.6567e-26 kg for O+).
- `tether_length`: Active length of electrodynamic plasma brake tether in meters (0.0 for satellite body).
- `f_ion`: Ionization mass fraction relative to neutral atmosphere (default 0.02).
- `R_body_km`: Central body radius in km.

# Returns
- `a_coulomb::SVector{3, Float64}`: Coulomb drag acceleration vector in km/s^2.
"""
function coulomb_drag_acceleration(
    r_vector::AbstractVector,
    v_vector::AbstractVector,
    rho_atm::Real,
    cd::Real,
    am::Real,
    omega_rot::Real;
    potential::Real = -5.0,
    ion_density::Union{Real, Nothing} = nothing,
    debye_length::Real = 0.02,
    ion_mass::Real = 2.6567e-26,
    tether_length::Real = 0.0,
    f_ion::Real = 0.02,
    R_body_km::Real = 6378.137
)
    r_mag = norm(r_vector)
    h_km = r_mag - Float64(R_body_km)

    # Relative velocity with respect to co-rotating ionosphere: v_rel = v - omega x r
    v_atm = SVector{3, Float64}(-Float64(omega_rot) * r_vector[2],
                                 Float64(omega_rot) * r_vector[1],
                                 0.0)
    v_rel = SVector{3, Float64}(v_vector[1] - v_atm[1],
                                v_vector[2] - v_atm[2],
                                v_vector[3] - v_atm[3])
    v_rel_mag = norm(v_rel)

    if v_rel_mag <= 0.0
        return SVector{3, Float64}(0.0, 0.0, 0.0)
    end

    v_rel_si = v_rel_mag * 1000.0 # m/s

    # Ion mass density [kg/m^3]
    rho_i = if !isnothing(ion_density) && ion_density > 0.0
        Float64(ion_density) * Float64(ion_mass)
    elseif rho_atm > 0.0
        Float64(f_ion) * Float64(rho_atm)
    else
        # Nominal ionospheric Chapman peak profile in LEO (peak ~350 km, n0 ~ 1e11 m^-3)
        n0 = 1.0e11 * exp(-((h_km - 350.0) / 150.0)^2)
        n0 * Float64(ion_mass)
    end

    if rho_i <= 0.0
        return SVector{3, Float64}(0.0, 0.0, 0.0)
    end

    # 1. Body Coulomb drag (enhanced cross section)
    eta = coulomb_effective_area_factor(
        v_rel_si, potential;
        am = am,
        debye_length = debye_length,
        ion_mass = ion_mass
    )

    # factor: -0.5e3 * rho_i * cd * am * eta * v_rel_mag
    # (rho in kg/m^3, am in m^2/kg, v_rel in km/s -> km/s^2 requires * 1e3)
    factor_body = -0.5e3 * rho_i * Float64(cd) * Float64(am) * eta * v_rel_mag
    a_body_x = factor_body * v_rel[1]
    a_body_y = factor_body * v_rel[2]
    a_body_z = factor_body * v_rel[3]

    # 2. Electrodynamic Tether / Plasma Brake contribution (if tether_length > 0)
    a_teth_x = 0.0
    a_teth_y = 0.0
    a_teth_z = 0.0

    if tether_length > 0.0
        eps0 = 8.8541878128e-12
        e_charge = 1.602176634e-19
        V_s = abs(Float64(potential))
        n_i = rho_i / Float64(ion_mass)
        r_sheath = sqrt(eps0 * V_s / (e_charge * max(n_i, 1.0)))

        P_dyn = rho_i * (v_rel_si^2)
        E_k = 0.5 * Float64(ion_mass) * (v_rel_si^2)
        chi = (e_charge * V_s) / max(E_k, 1e-25)

        exp_term = exp(-1.0 / max(chi, 0.01))
        dF_dz = 3.864 * P_dyn * r_sheath * exp_term # N/m

        # Braking acceleration [m/s^2] = (dF/dz * L) / m_sat = (dF/dz * L) * (am / A_ref)
        # Using A_ref ~ 0.04 m^2 default for small satellite platform
        a_brake_ms2 = (dF_dz * Float64(tether_length)) * (Float64(am) / 0.04)
        factor_teth = -(a_brake_ms2 / 1000.0) / v_rel_mag # convert to km/s^2

        a_teth_x = factor_teth * v_rel[1]
        a_teth_y = factor_teth * v_rel[2]
        a_teth_z = factor_teth * v_rel[3]
    end

    return SVector{3, Float64}(a_body_x + a_teth_x, a_body_y + a_teth_y, a_body_z + a_teth_z)
end


end # end of module





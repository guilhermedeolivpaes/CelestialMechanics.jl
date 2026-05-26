# src/Models/PerturbationEquations.jl

module PerturbationEquations

using ..Constants
using LinearAlgebra, StaticArrays, Unitful, Unitful.DefaultSymbols, UnitfulAstro


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

function j3_perturbation(r_vector, mu, R, j3)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (j3*R^3*mu*x*z*(15*z^4+15*y^2*z^2+15*x^2*z^2+17*r^2*z^2-3*r^2*y^2-3*r^2*x^2-12*r^4))/(2*r^11)
    ay = (j3*R^3*mu*y*z*(15*z^4+15*y^2*z^2+15*x^2*z^2+17*r^2*z^2-3*r^2*y^2-3*r^2*x^2-12*r^4))/(2*r^11)
    az = (j3*R^3*mu*(15*z^6+15*y^2*z^4+15*x^2*z^4+2*r^2*z^4-18*r^2*y^2*z^2-18*r^2*x^2*z^2-9*r^4*z^2+3*r^4*y^2+3*r^4*x^2))/(2*r^11)
    return SVector(ax, ay, az)
end

function j4_perturbation(r_vector, mu, R, j4)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (15*j4*R^4*mu*x*(21*z^4-14*r^2*z^2+r^4))/(8*r^11)
    ay = (15*j4*R^4*mu*y*(21*z^4-14*r^2*z^2+r^4))/(8*r^11)
    az = (5*j4*R^4*mu*z*(63*z^4-70*r^2*z^2+15*r^4))/(8*r^11)
    return SVector(ax, ay, az)
end

function j5_perturbation(r_vector, mu, R, j5)
    x, y, z = r_vector
    r = norm(r_vector)
    ax = (3*j5*R^5*mu*x*z*(105*z^6+105*y^2*z^4+105*x^2*z^4+56*r^2*z^4-70*r^2*y^2*z^2-70*r^2*x^2*z^2-135*r^4*z^2+5*r^4*y^2+5*r^4*x^2+30*r^6))/(8*r^15)
    ay = (3*j5*R^5*mu*y*z*(105*z^6+105*y^2*z^4+105*x^2*z^4+56*r^2*z^4-70*r^2*y^2*z^2-70*r^2*x^2*z^2-135*r^4*z^2+5*r^4*y^2+5*r^4*x^2+30*r^6))/(8*r^15)
    az = (3*j5*R^5*mu*(105*z^8+105*y^2*z^6+105*x^2*z^6-49*r^2*z^6-175*r^2*y^2*z^4-175*r^2*x^2*z^4-65*r^4*z^4+75*r^4*y^2*z^2+75*r^4*x^2*z^2+25*r^6*z^2-5*r^6*y^2-5*r^6*x^2))/(8*r^15)
    return SVector(ax, ay, az)
end

function j6_perturbation(r_vector, mu, R, j6)
    x, y, z, = r_vector
    r = norm(r_vector)
    ax = (7*j6*R^6*mu*x*(429*z^6-495*r^2*z^4+135*r^4*z^2-5*r^6))/(16*r^15)
    ay = (7*j6*R^6*mu*y*(429*z^6-495*r^2*z^4+135*r^4*z^2-5*r^6))/(16*r^15)
    az = (7*j6*R^6*mu*z*(429*z^6-693*r^2*z^4+315*r^4*z^2-35*r^6))/(16*r^15)
    return SVector(ax, ay, az)
end

function j7_perturbation(r_vector, mu, R, j7)
    x, y, z, = r_vector
    r = norm(r_vector)
    ax = (j7*R^7*mu*x*z*(3003*z^8+3003*y^2*z^6+3003*x^2*z^6-33*r^2*z^6-3465*r^2*y^2*z^4-3465*r^2*x^2*z^4-4599*r^4*z^4+945*r^4*y^2*z^2+945*r^4*x^2*z^2+2485*r^6*z^2-35*r^6*y^2-35*r^6*x^2-280*r^8))/(16*r^19)
    ay = (j7*R^7*mu*y*z*(3003*z^8+3003*y^2*z^6+3003*x^2*z^6-33*r^2*z^6-3465*r^2*y^2*z^4-3465*r^2*x^2*z^4-4599*r^4*z^4+945*r^4*y^2*z^2+945*r^4*x^2*z^2+2485*r^6*z^2-35*r^6*y^2-35*r^6*x^2-280*r^8))/(16*r^19)
    az = (j7*R^7*mu*(3003*z^10+3003*y^2*z^8+3003*x^2*z^8-3036*r^2*z^8-6468*r^2*y^2*z^6-6468*r^2*x^2*z^6-1134*r^4*z^6+4410*r^4*y^2*z^4+4410*r^4*x^2*z^4+1540*r^6*z^4-980*r^6*y^2*z^2-980*r^6*x^2*z^2-245*r^8*z^2+35*r^8*y^2+35*r^8*x^2))/(16*r^19)
    return SVector(ax, ay, az)
end

function j8_perturbation(r_vector, mu, R, j8)
    x, y, z, = r_vector
    r = norm(r_vector)
    ax = (45*j8*R^8*mu*x*(2431*z^8-4004*r^2*z^6+2002*r^4*z^4-308*r^6*z^2+7*r^8))/(128*r^19)
    ay = (45*j8*R^8*mu*y*(2431*z^8-4004*r^2*z^6+2002*r^4*z^4-308*r^6*z^2+7*r^8))/(128*r^19)
    az = (9*j8*R^8*mu*z*(12155*z^8-25740*r^2*z^6+18018*r^4*z^4-4620*r^6*z^2+315*r^8))/(128*r^19)
    return SVector(ax, ay, az)
end

function j9_perturbation(r_vector, mu, R, j9)
    x, y, z, = r_vector
    r = norm(r_vector)
    ax = (5*j9*R^9*mu*x*z*(21879*z^10+21879*y^2*z^8+21879*x^2*z^8-11726*r^2*z^8-36036*r^2*y^2*z^6-36036*r^2*x^2*z^6-33462*r^4*z^6+18018*r^4*y^2*z^4+18018*r^4*x^2*z^4+33264*r^6*z^4-2772*r^6*y^2*z^2-2772*r^6*x^2*z^2-9177*r^8*z^2+63*r^8*y^2+63*r^8*x^2+630*r^10))/(128*r^23)
    ay = (5*j9*R^9*mu*y*z*(21879*z^10+21879*y^2*z^8+21879*x^2*z^8-11726*r^2*z^8-36036*r^2*y^2*z^6-36036*r^2*x^2*z^6-33462*r^4*z^6+18018*r^4*y^2*z^4+18018*r^4*x^2*z^4+33264*r^6*z^4-2772*r^6*y^2*z^2-2772*r^6*x^2*z^2-9177*r^8*z^2+63*r^8*y^2+63*r^8*x^2+630*r^10))/(128*r^23)
    az = (5*j9*R^9*mu*(21879*z^12+21879*y^2*z^10+21879*x^2*z^10-33605*r^2*z^10-57915*r^2*y^2*z^8-57915*r^2*x^2*z^8+2574*r^4*z^8+54054*r^4*y^2*z^6+54054*r^4*x^2*z^6+15246*r^6*z^6-20790*r^6*y^2*z^4-20790*r^6*x^2*z^4-6405*r^8*z^4+2835*r^8*y^2*z^2+2835*r^8*x^2*z^2+567*r^10*z^2-63*r^10*y^2-63*r^10*x^2))/(128*r^23)
    return SVector(ax, ay, az)
end

function j10_perturbation(r_vector, mu, R, j10)
    x, y, z, = r_vector
    r = norm(r_vector)
    ax = (33*j10*R^10*mu*x*(29393*z^10-62985*r^2*z^8+46410*r^4*z^6-13650*r^6*z^4+1365*r^8*z^2-21*r^10))/(256*r^23)
    ay = (33*j10*R^10*mu*y*(29393*z^10-62985*r^2*z^8+46410*r^4*z^6-13650*r^6*z^4+1365*r^8*z^2-21*r^10))/(256*r^23)
    az = (11*j10*R^10*mu*z*(88179*z^10-230945*r^2*z^8+218790*r^4*z^6-90090*r^6*z^4+15015*r^8*z^2-693*r^10))/(256*r^23)
    return SVector(ax, ay, az)
end


function j11_perturbation(r_vector, mu, R, j11)
    x, y, z, = r_vector
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

function j12_perturbation(r_vector, mu, R, j12)
    x, y, z, = r_vector
    r = norm(r_vector)
    ax = (91*j12*R^12*mu*x*(185725*z^12-490314*r^2*z^10+479655*r^4*z^8-213180*r^6*z^6+42075*r^8*z^4-2970*r^10*z^2+33*r^12))/(1024*r^27)
    ay = (91*j12*R^12*mu*y*(185725*z^12-490314*r^2*z^10+479655*r^4*z^8-213180*r^6*z^6+42075*r^8*z^4-2970*r^10*z^2+33*r^12))/(1024*r^27)
    az = (13*j12*R^12*mu*z*(1300075*z^12-4056234*r^2*z^10+4849845*r^4*z^8-2771340*r^6*z^6+765765*r^8*z^4-90090*r^10*z^2+3003*r^12))/(1024*r^27)
    return SVector(ax, ay, az)
end

function j13_perturbation(r_vector, mu, R, j13)
    x, y, z, = r_vector
    r = norm(r_vector)
    ax = (7*j13*R^13*mu*x*z*(2414425*z^14+2414425*y^2*z^12+2414425*x^2*z^12-3773932*r^2*z^12-6374082*r^2*y^2*z^10-6374082*r^2*x^2*z^10-1876953*r^4*z^10+6235515*r^4*y^2*z^8+6235515*r^4*x^2*z^8+6928350*r^6*z^8-2771340*r^6*y^2*z^6-2771340*r^6*x^2*z^6-4995705*r^8*z^6+546975*r^8*y^2*z^4+546975*r^8*x^2*z^4+1492920*r^10*z^4-38610*r^10*y^2*z^2-38610*r^10*x^2*z^2-179751*r^12*z^2+429*r^12*y^2+429*r^12*x^2+6006*r^14))/(1024*r^31)
    ay = (7*j13*R^13*mu*y*z*(2414425*z^14+2414425*y^2*z^12+2414425*x^2*z^12-3773932*r^2*z^12-6374082*r^2*y^2*z^10-6374082*r^2*x^2*z^10-1876953*r^4*z^10+6235515*r^4*y^2*z^8+6235515*r^4*x^2*z^8+6928350*r^6*z^8-2771340*r^6*y^2*z^6-2771340*r^6*x^2*z^6-4995705*r^8*z^6+546975*r^8*y^2*z^4+546975*r^8*x^2*z^4+1492920*r^10*z^4-38610*r^10*y^2*z^2-38610*r^10*x^2*z^2-179751*r^12*z^2+429*r^12*y^2+429*r^12*x^2+6006*r^14))/(1024*r^31)
    az = (7*j13*R^13*mu*(2414425*z^16+2414425*y^2*z^14+2414425*x^2*z^14-6188357*r^2*z^14-8788507*r^2*y^2*z^12-8788507*r^2*x^2*z^12+4497129*r^4*z^12+12609597*r^4*y^2*z^10+12609597*r^4*x^2*z^10+692835*r^6*z^10-9006855*r^6*y^2*z^8-9006855*r^6*x^2*z^8-2224365*r^8*z^8+3318315*r^8*y^2*z^6+3318315*r^8*x^2*z^6+945945*r^10*z^6-585585*r^10*y^2*z^4-585585*r^10*x^2*z^4-141141*r^12*z^4+39039*r^12*y^2*z^2+39039*r^12*x^2*z^2+5577*r^14*z^2-429*r^14*y^2-429*r^14*x^2))/(1024*r^31)
    return SVector(ax, ay, az)
end

function j14_perturbation(r_vector, mu, R, j14)
    x, y, z, = r_vector
    r = norm(r_vector)
    ax = (15*j14*R^14*mu*x*(9694845*z^14-30421755*r^2*z^12+37182145*r^4*z^10-22309287*r^6*z^8+6789783*r^8*z^6-969969*r^10*z^4+51051*r^12*z^2-429*r^14))/(2048*r^31)
    ay = (15*j14*R^14*mu*y*(9694845*z^14-30421755*r^2*z^12+37182145*r^4*z^10-22309287*r^6*z^8+6789783*r^8*z^6-969969*r^10*z^4+51051*r^12*z^2-429*r^14))/(2048*r^31)
    az = (15*j14*R^14*mu*z*(9694845*z^14-35102025*r^2*z^12+50702925*r^4*z^10-37182145*r^6*z^8+14549535*r^8*z^6-2909907*r^10*z^4+255255*r^12*z^2-6435*r^14))/(2048*r^31)
    return SVector(ax, ay, az)
end

function j15_perturbation(r_vector, mu, R, j15)
    x, y, z, = r_vector
    r = norm(r_vector)
    ax = (j15*R^15*mu*x*z*(145422675*z^16+145422675*y^2*z^14+145422675*x^2*z^14-301208805*r^2*z^14-456326325*r^2*y^2*z^12-456326325*r^2*x^2*z^12-3900225*r^4*z^12+557732175*r^4*y^2*z^10+557732175*r^4*x^2*z^10+476607495*r^6*z^10-334639305*r^6*y^2*z^8-334639305*r^6*x^2*z^8-493067575*r^8*z^8+101846745*r^8*y^2*z^6+101846745*r^8*x^2*z^6+218243025*r^10*z^6-14549535*r^10*y^2*z^4-14549535*r^10*x^2*z^4-45792747*r^12*z^4+765765*r^12*y^2*z^2+765765*r^12*x^2*z^2+4077645*r^14*z^2-6435*r^14*y^2-6435*r^14*x^2-102960*r^16))/(2048*r^35)
    ay = (j15*R^15*mu*y*z*(145422675*z^16+145422675*y^2*z^14+145422675*x^2*z^14-301208805*r^2*z^14-456326325*r^2*y^2*z^12-456326325*r^2*x^2*z^12-3900225*r^4*z^12+557732175*r^4*y^2*z^10+557732175*r^4*x^2*z^10+476607495*r^6*z^10-334639305*r^6*y^2*z^8-334639305*r^6*x^2*z^8-493067575*r^8*z^8+101846745*r^8*y^2*z^6+101846745*r^8*x^2*z^6+218243025*r^10*z^6-14549535*r^10*y^2*z^4-14549535*r^10*x^2*z^4-45792747*r^12*z^4+765765*r^12*y^2*z^2+765765*r^12*x^2*z^2+4077645*r^14*z^2-6435*r^14*y^2-6435*r^14*x^2-102960*r^16))/(2048*r^35)
    az = (j15*R^15*mu*(145422675*z^18+145422675*y^2*z^16+145422675*x^2*z^16-446631480*r^2*z^16-601749000*r^2*y^2*z^14-601749000*r^2*x^2*z^14+452426100*r^4*z^14+1014058500*r^4*y^2*z^12+1014058500*r^4*x^2*z^12-81124680*r^6*z^12-892371480*r^6*y^2*z^10-892371480*r^6*x^2*z^10-158428270*r^8*z^10+436486050*r^8*y^2*z^8+436486050*r^8*x^2*z^8+116396280*r^10*z^8-116396280*r^10*y^2*z^6-116396280*r^10*x^2*z^6-31243212*r^12*z^6+15315300*r^12*y^2*z^4+15315300*r^12*x^2*z^4+3311880*r^14*z^4-772200*r^14*y^2*z^2-772200*r^14*x^2*z^2-96525*r^16*z^2+6435*r^16*y^2+6435*r^16*x^2))/(2048*r^35)
    return SVector(ax, ay, az)
end

function j16_perturbation(r_vector, mu, R, j16)
    x, y, z, = r_vector
    r = norm(r_vector)
    ax = (153*j16*R^16*mu*x*(64822395*z^16-235717800*r^2*z^14+345972900*r^4*z^12-262462200*r^6*z^10+109359250*r^8*z^8-24496472*r^10*z^6+2662660*r^12*z^4-108680*r^14*z^2+715*r^16))/(32768*r^35)
    ay = (153*j16*R^16*mu*y*(64822395*z^16-235717800*r^2*z^14+345972900*r^4*z^12-262462200*r^6*z^10+109359250*r^8*z^8-24496472*r^10*z^6+2662660*r^12*z^4-108680*r^14*z^2+715*r^16))/(32768*r^35)
    az = (17*j16*R^16*mu*z*(583401555*z^16-2404321560*r^2*z^14+4071834900*r^4*z^12-3650610600*r^6*z^10+1859107250*r^8*z^8-535422888*r^10*z^6+81477396*r^12*z^4-5542680*r^14*z^2+109395*r^16))/(32768*r^35)
    return SVector(ax, ay, az)
end

function j17_perturbation(r_vector, mu, R, j17)
    x, y, z, = r_vector
    r = norm(r_vector)
    ax = (9*j17*R^17*mu*x*z*(1101980715*z^18+1101980715*y^2*z^16+1101980715*x^2*z^16-2840399490*r^2*z^16-4007202600*r^2*y^2*z^14-4007202600*r^2*x^2*z^14+1072896180*r^4*z^14+5881539300*r^4*y^2*z^12+5881539300*r^4*x^2*z^12+3681812400*r^6*z^12-4461857400*r^6*y^2*z^10-4461857400*r^6*x^2*z^10-5442113950*r^8*z^10+1859107250*r^8*y^2*z^8+1859107250*r^8*x^2*z^8+3301774476*r^10*z^8-416440024*r^10*y^2*z^6-416440024*r^10*x^2*z^6-1025580556*r^12*z^6+45265220*r^12*y^2*z^4+45265220*r^12*x^2*z^4+161107232*r^14*z^4-1847560*r^14*y^2*z^2-1847560*r^14*x^2*z^2-11073205*r^16*z^2+12155*r^16*y^2+12155*r^16*x^2+218790*r^18))/(32768*r^39)
    ay = (9*j17*R^17*mu*x*z*(1101980715*z^18+1101980715*y^2*z^16+1101980715*x^2*z^16-2840399490*r^2*z^16-4007202600*r^2*y^2*z^14-4007202600*r^2*x^2*z^14+1072896180*r^4*z^14+5881539300*r^4*y^2*z^12+5881539300*r^4*x^2*z^12+3681812400*r^6*z^12-4461857400*r^6*y^2*z^10-4461857400*r^6*x^2*z^10-5442113950*r^8*z^10+1859107250*r^8*y^2*z^8+1859107250*r^8*x^2*z^8+3301774476*r^10*z^8-416440024*r^10*y^2*z^6-416440024*r^10*x^2*z^6-1025580556*r^12*z^6+45265220*r^12*y^2*z^4+45265220*r^12*x^2*z^4+161107232*r^14*z^4-1847560*r^14*y^2*z^2-1847560*r^14*x^2*z^2-11073205*r^16*z^2+12155*r^16*y^2+12155*r^16*x^2+218790*r^18))/(32768*r^39)
    az = (9*j17*R^17*mu*(1101980715*z^20+1101980715*y^2*z^18+1101980715*x^2*z^18-3942380205*r^2*z^18-5109183315*r^2*y^2*z^16-5109183315*r^2*x^2*z^16+5080098780*r^4*z^16+9888741900*r^4*y^2*z^14+9888741900*r^4*x^2*z^14-2199726900*r^6*z^14-10343396700*r^6*y^2*z^12-10343396700*r^6*x^2*z^12-980256550*r^8*z^12+6320964650*r^8*y^2*z^10+6320964650*r^8*x^2*z^10+1442667226*r^10*z^10-2275547274*r^10*y^2*z^8-2275547274*r^10*x^2*z^8-609140532*r^12*z^8+461705244*r^12*y^2*z^6+461705244*r^12*x^2*z^6+115842012*r^14*z^6-47112780*r^14*y^2*z^4-47112780*r^14*x^2*z^4-9225645*r^16*z^4+1859715*r^16*y^2*z^2+1859715*r^16*x^2*z^2+206635*r^18*z^2-12155*r^18*y^2-12155*r^18*x^2))/(32768*r^39)
    return SVector(ax, ay, az)
end

function j18_perturbation(r_vector, mu, R, j18)
    x, y, z, = r_vector
    r = norm(r_vector)
    ax = (95*j18*R^18*mu*x*(883631595*z^18-3653936055*r^2*z^16+6263890380*r^4*z^14-5757717420*r^6*z^12+3064591530*r^8*z^10-951080130*r^10*z^8+164384220*r^12*z^6-14090076*r^14*z^4+459459*r^16*z^2-2431*r^18))/(65536*r^39)
    ay = (95*j18*R^18*mu*y*(883631595*z^18-3653936055*r^2*z^16+6263890380*r^4*z^14-5757717420*r^6*z^12+3064591530*r^8*z^10-951080130*r^10*z^8+164384220*r^12*z^6-14090076*r^14*z^4+459459*r^16*z^2-2431*r^18))/(65536*r^39)
    az = (19*j18*R^18*mu*z*(4418157975*z^18-20419054425*r^2*z^16+39671305740*r^4*z^14-42075627300*r^6*z^12+26466926850*r^8*z^10-10039179150*r^10*z^8+2230928700*r^12*z^6-267711444*r^14*z^4+14549535*r^16*z^2-230945*r^18))/(65536*r^39)
    return SVector(ax, ay, az)
end

# ------------ sectoral harmonics  (n = m) ------------ #
# they are in the inertial frame of reference, not the rotating one

function cs22_perturbation(r_vector, t, mu, R, c22, s22, omega_rot)
    x,y,z, = r_vector
    r = norm(r_vector)

    ax = -((3*R^2*mu*(c22*(5*sin(omega_rot*t)^2*x*y^2-5*cos(omega_rot*t)^2*x*y^2+20*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y-4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y-5*sin(omega_rot*t)^2*x^3+5*cos(omega_rot*t)^2*x^3+2*r^2*sin(omega_rot*t)^2*x-2*r^2*cos(omega_rot*t)^2*x)+s22*(10*cos(omega_rot*t)*sin(omega_rot*t)*x*y^2-10*sin(omega_rot*t)^2*x^2*y+10*cos(omega_rot*t)^2*x^2*y+2*r^2*sin(omega_rot*t)^2*y-2*r^2*cos(omega_rot*t)^2*y-10*cos(omega_rot*t)*sin(omega_rot*t)*x^3+4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x)))/r^7)
    ay = -((3*R^2*mu*(c22*(5*sin(omega_rot*t)^2*y^3-5*cos(omega_rot*t)^2*y^3+20*cos(omega_rot*t)*sin(omega_rot*t)*x*y^2-5*sin(omega_rot*t)^2*x^2*y+5*cos(omega_rot*t)^2*x^2*y-2*r^2*sin(omega_rot*t)^2*y+2*r^2*cos(omega_rot*t)^2*y-4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x)+s22*(10*cos(omega_rot*t)*sin(omega_rot*t)*y^3-10*sin(omega_rot*t)^2*x*y^2+10*cos(omega_rot*t)^2*x*y^2-10*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y-4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y+2*r^2*sin(omega_rot*t)^2*x-2*r^2*cos(omega_rot*t)^2*x)))/r^7)
    az = -((15*R^2*mu*(c22*(sin(omega_rot*t)^2*y^2-cos(omega_rot*t)^2*y^2+4*cos(omega_rot*t)*sin(omega_rot*t)*x*y-sin(omega_rot*t)^2*x^2+cos(omega_rot*t)^2*x^2)+s22*(2*cos(omega_rot*t)*sin(omega_rot*t)*y^2-2*sin(omega_rot*t)^2*x*y+2*cos(omega_rot*t)^2*x*y-2*cos(omega_rot*t)*sin(omega_rot*t)*x^2))*z)/r^7)
    return SVector(ax, ay, az)
end

function cs33_perturbation(r_vector, t, mu, R, c33, s33, omega_rot)
    x,y,z, = r_vector
    r = norm(r_vector)

    ax = (15*R^3*mu*(c33*(7*sin(omega_rot*t)^3*x*y^3-21*cos(omega_rot*t)^2*sin(omega_rot*t)*x*y^3+63*cos(omega_rot*t)*sin(omega_rot*t)^2*x^2*y^2-21*cos(omega_rot*t)^3*x^2*y^2-9*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*y^2+3*r^2*cos(omega_rot*t)^3*y^2-21*sin(omega_rot*t)^3*x^3*y+63*cos(omega_rot*t)^2*sin(omega_rot*t)*x^3*y+6*r^2*sin(omega_rot*t)^3*x*y-18*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*x*y-21*cos(omega_rot*t)*sin(omega_rot*t)^2*x^4+7*cos(omega_rot*t)^3*x^4+9*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*x^2-3*r^2*cos(omega_rot*t)^3*x^2)+s33*(21*cos(omega_rot*t)*sin(omega_rot*t)^2*x*y^3-7*cos(omega_rot*t)^3*x*y^3-21*sin(omega_rot*t)^3*x^2*y^2+63*cos(omega_rot*t)^2*sin(omega_rot*t)*x^2*y^2+3*r^2*sin(omega_rot*t)^3*y^2-9*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*y^2-63*cos(omega_rot*t)*sin(omega_rot*t)^2*x^3*y+21*cos(omega_rot*t)^3*x^3*y+18*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*x*y-6*r^2*cos(omega_rot*t)^3*x*y+7*sin(omega_rot*t)^3*x^4-21*cos(omega_rot*t)^2*sin(omega_rot*t)*x^4-3*r^2*sin(omega_rot*t)^3*x^2+9*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*x^2)))/r^9
    ay = (15*R^3*mu*(c33*(7*sin(omega_rot*t)^3*y^4-21*cos(omega_rot*t)^2*sin(omega_rot*t)*y^4+63*cos(omega_rot*t)*sin(omega_rot*t)^2*x*y^3-21*cos(omega_rot*t)^3*x*y^3-21*sin(omega_rot*t)^3*x^2*y^2+63*cos(omega_rot*t)^2*sin(omega_rot*t)*x^2*y^2-3*r^2*sin(omega_rot*t)^3*y^2+9*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*y^2-21*cos(omega_rot*t)*sin(omega_rot*t)^2*x^3*y+7*cos(omega_rot*t)^3*x^3*y-18*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*x*y+6*r^2*cos(omega_rot*t)^3*x*y+3*r^2*sin(omega_rot*t)^3*x^2-9*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*x^2)+s33*(21*cos(omega_rot*t)*sin(omega_rot*t)^2*y^4-7*cos(omega_rot*t)^3*y^4-21*sin(omega_rot*t)^3*x*y^3+63*cos(omega_rot*t)^2*sin(omega_rot*t)*x*y^3-63*cos(omega_rot*t)*sin(omega_rot*t)^2*x^2*y^2+21*cos(omega_rot*t)^3*x^2*y^2-9*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*y^2+3*r^2*cos(omega_rot*t)^3*y^2+7*sin(omega_rot*t)^3*x^3*y-21*cos(omega_rot*t)^2*sin(omega_rot*t)*x^3*y+6*r^2*sin(omega_rot*t)^3*x*y-18*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)*x*y+9*r^2*cos(omega_rot*t)*sin(omega_rot*t)^2*x^2-3*r^2*cos(omega_rot*t)^3*x^2)))/r^9
    az = (105*R^3*mu*(c33*(sin(omega_rot*t)^3*y^3-3*cos(omega_rot*t)^2*sin(omega_rot*t)*y^3+9*cos(omega_rot*t)*sin(omega_rot*t)^2*x*y^2-3*cos(omega_rot*t)^3*x*y^2-3*sin(omega_rot*t)^3*x^2*y+9*cos(omega_rot*t)^2*sin(omega_rot*t)*x^2*y-3*cos(omega_rot*t)*sin(omega_rot*t)^2*x^3+cos(omega_rot*t)^3*x^3)+s33*(3*cos(omega_rot*t)*sin(omega_rot*t)^2*y^3-cos(omega_rot*t)^3*y^3-3*sin(omega_rot*t)^3*x*y^2+9*cos(omega_rot*t)^2*sin(omega_rot*t)*x*y^2-9*cos(omega_rot*t)*sin(omega_rot*t)^2*x^2*y+3*cos(omega_rot*t)^3*x^2*y+sin(omega_rot*t)^3*x^3-3*cos(omega_rot*t)^2*sin(omega_rot*t)*x^3))*z)/r^9
    return SVector(ax, ay, az)
end

function cs44_perturbation(r_vector, t, mu, R, c44, s44, omega_rot)
    x,y,z, = r_vector
    r = norm(r_vector)

    ax = -((105*R^4*mu*(9*c44*sin(omega_rot*t)^4*x*y^4+36*s44*cos(omega_rot*t)*sin(omega_rot*t)^3*x*y^4-54*c44*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x*y^4-36*s44*cos(omega_rot*t)^3*sin(omega_rot*t)*x*y^4+9*c44*cos(omega_rot*t)^4*x*y^4-36*s44*sin(omega_rot*t)^4*x^2*y^3+144*c44*cos(omega_rot*t)*sin(omega_rot*t)^3*x^2*y^3+216*s44*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^2*y^3-144*c44*cos(omega_rot*t)^3*sin(omega_rot*t)*x^2*y^3-36*s44*cos(omega_rot*t)^4*x^2*y^3+4*s44*r^2*sin(omega_rot*t)^4*y^3-16*c44*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*y^3-24*s44*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*y^3+16*c44*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*y^3+4*s44*r^2*cos(omega_rot*t)^4*y^3-54*c44*sin(omega_rot*t)^4*x^3*y^2-216*s44*cos(omega_rot*t)*sin(omega_rot*t)^3*x^3*y^2+324*c44*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^3*y^2+216*s44*cos(omega_rot*t)^3*sin(omega_rot*t)*x^3*y^2-54*c44*cos(omega_rot*t)^4*x^3*y^2+12*c44*r^2*sin(omega_rot*t)^4*x*y^2+48*s44*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*x*y^2-72*c44*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x*y^2-48*s44*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*x*y^2+12*c44*r^2*cos(omega_rot*t)^4*x*y^2+36*s44*sin(omega_rot*t)^4*x^4*y-144*c44*cos(omega_rot*t)*sin(omega_rot*t)^3*x^4*y-216*s44*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^4*y+144*c44*cos(omega_rot*t)^3*sin(omega_rot*t)*x^4*y+36*s44*cos(omega_rot*t)^4*x^4*y-12*s44*r^2*sin(omega_rot*t)^4*x^2*y+48*c44*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*x^2*y+72*s44*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^2*y-48*c44*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*x^2*y-12*s44*r^2*cos(omega_rot*t)^4*x^2*y+9*c44*sin(omega_rot*t)^4*x^5+36*s44*cos(omega_rot*t)*sin(omega_rot*t)^3*x^5-54*c44*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^5-36*s44*cos(omega_rot*t)^3*sin(omega_rot*t)*x^5+9*c44*cos(omega_rot*t)^4*x^5-4*c44*r^2*sin(omega_rot*t)^4*x^3-16*s44*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*x^3+24*c44*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^3+16*s44*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*x^3-4*c44*r^2*cos(omega_rot*t)^4*x^3))/r^11)
    ay = -((105*R^4*mu*(9*c44*sin(omega_rot*t)^4*y^5+36*s44*cos(omega_rot*t)*sin(omega_rot*t)^3*y^5-54*c44*cos(omega_rot*t)^2*sin(omega_rot*t)^2*y^5-36*s44*cos(omega_rot*t)^3*sin(omega_rot*t)*y^5+9*c44*cos(omega_rot*t)^4*y^5-36*s44*sin(omega_rot*t)^4*x*y^4+144*c44*cos(omega_rot*t)*sin(omega_rot*t)^3*x*y^4+216*s44*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x*y^4-144*c44*cos(omega_rot*t)^3*sin(omega_rot*t)*x*y^4-36*s44*cos(omega_rot*t)^4*x*y^4-54*c44*sin(omega_rot*t)^4*x^2*y^3-216*s44*cos(omega_rot*t)*sin(omega_rot*t)^3*x^2*y^3+324*c44*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^2*y^3+216*s44*cos(omega_rot*t)^3*sin(omega_rot*t)*x^2*y^3-54*c44*cos(omega_rot*t)^4*x^2*y^3-4*c44*r^2*sin(omega_rot*t)^4*y^3-16*s44*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*y^3+24*c44*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*y^3+16*s44*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*y^3-4*c44*r^2*cos(omega_rot*t)^4*y^3+36*s44*sin(omega_rot*t)^4*x^3*y^2-144*c44*cos(omega_rot*t)*sin(omega_rot*t)^3*x^3*y^2-216*s44*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^3*y^2+144*c44*cos(omega_rot*t)^3*sin(omega_rot*t)*x^3*y^2+36*s44*cos(omega_rot*t)^4*x^3*y^2+12*s44*r^2*sin(omega_rot*t)^4*x*y^2-48*c44*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*x*y^2-72*s44*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x*y^2+48*c44*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*x*y^2+12*s44*r^2*cos(omega_rot*t)^4*x*y^2+9*c44*sin(omega_rot*t)^4*x^4*y+36*s44*cos(omega_rot*t)*sin(omega_rot*t)^3*x^4*y-54*c44*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^4*y-36*s44*cos(omega_rot*t)^3*sin(omega_rot*t)*x^4*y+9*c44*cos(omega_rot*t)^4*x^4*y+12*c44*r^2*sin(omega_rot*t)^4*x^2*y+48*s44*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*x^2*y-72*c44*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^2*y-48*s44*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*x^2*y+12*c44*r^2*cos(omega_rot*t)^4*x^2*y-4*s44*r^2*sin(omega_rot*t)^4*x^3+16*c44*r^2*cos(omega_rot*t)*sin(omega_rot*t)^3*x^3+24*s44*r^2*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^3-16*c44*r^2*cos(omega_rot*t)^3*sin(omega_rot*t)*x^3-4*s44*r^2*cos(omega_rot*t)^4*x^3))/r^11)
    az = -((945*R^4*mu*(c44*sin(omega_rot*t)^4*y^4+4*s44*cos(omega_rot*t)*sin(omega_rot*t)^3*y^4-6*c44*cos(omega_rot*t)^2*sin(omega_rot*t)^2*y^4-4*s44*cos(omega_rot*t)^3*sin(omega_rot*t)*y^4+c44*cos(omega_rot*t)^4*y^4-4*s44*sin(omega_rot*t)^4*x*y^3+16*c44*cos(omega_rot*t)*sin(omega_rot*t)^3*x*y^3+24*s44*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x*y^3-16*c44*cos(omega_rot*t)^3*sin(omega_rot*t)*x*y^3-4*s44*cos(omega_rot*t)^4*x*y^3-6*c44*sin(omega_rot*t)^4*x^2*y^2-24*s44*cos(omega_rot*t)*sin(omega_rot*t)^3*x^2*y^2+36*c44*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^2*y^2+24*s44*cos(omega_rot*t)^3*sin(omega_rot*t)*x^2*y^2-6*c44*cos(omega_rot*t)^4*x^2*y^2+4*s44*sin(omega_rot*t)^4*x^3*y-16*c44*cos(omega_rot*t)*sin(omega_rot*t)^3*x^3*y-24*s44*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^3*y+16*c44*cos(omega_rot*t)^3*sin(omega_rot*t)*x^3*y+4*s44*cos(omega_rot*t)^4*x^3*y+c44*sin(omega_rot*t)^4*x^4+4*s44*cos(omega_rot*t)*sin(omega_rot*t)^3*x^4-6*c44*cos(omega_rot*t)^2*sin(omega_rot*t)^2*x^4-4*s44*cos(omega_rot*t)^3*sin(omega_rot*t)*x^4+c44*cos(omega_rot*t)^4*x^4)*z)/r^11)
    return SVector(ax, ay, az)
end

# ------------ tesseral harmonics (m < n) ------------ #
# they are in the inertial frame of reference, not the rotating one

function cs31_perturbation(r_vector, t, mu, R, c31, s31, omega_rot)
    x,y,z, = r_vector
    r = norm(r_vector)

    ax = (3*R^3*mu*(c31*(28*sin(omega_rot*t)*x*y*z^2+28*cos(omega_rot*t)*x^2*z^2-4*r^2*cos(omega_rot*t)*z^2-7*sin(omega_rot*t)*x*y^3-7*cos(omega_rot*t)*x^2*y^2+r^2*cos(omega_rot*t)*y^2-7*sin(omega_rot*t)*x^3*y+2*r^2*sin(omega_rot*t)*x*y-7*cos(omega_rot*t)*x^4+3*r^2*cos(omega_rot*t)*x^2)+s31*(28*cos(omega_rot*t)*x*y*z^2-28*sin(omega_rot*t)*x^2*z^2+4*r^2*sin(omega_rot*t)*z^2-7*cos(omega_rot*t)*x*y^3+7*sin(omega_rot*t)*x^2*y^2-r^2*sin(omega_rot*t)*y^2-7*cos(omega_rot*t)*x^3*y+2*r^2*cos(omega_rot*t)*x*y+7*sin(omega_rot*t)*x^4-3*r^2*sin(omega_rot*t)*x^2)))/(2*r^9)
    ay = (3*R^3*mu*(c31*(28*sin(omega_rot*t)*y^2*z^2+28*cos(omega_rot*t)*x*y*z^2-4*r^2*sin(omega_rot*t)*z^2-7*sin(omega_rot*t)*y^4-7*cos(omega_rot*t)*x*y^3-7*sin(omega_rot*t)*x^2*y^2+3*r^2*sin(omega_rot*t)*y^2-7*cos(omega_rot*t)*x^3*y+2*r^2*cos(omega_rot*t)*x*y+r^2*sin(omega_rot*t)*x^2)+s31*(28*cos(omega_rot*t)*y^2*z^2-28*sin(omega_rot*t)*x*y*z^2-4*r^2*cos(omega_rot*t)*z^2-7*cos(omega_rot*t)*y^4+7*sin(omega_rot*t)*x*y^3-7*cos(omega_rot*t)*x^2*y^2+3*r^2*cos(omega_rot*t)*y^2+7*sin(omega_rot*t)*x^3*y-2*r^2*sin(omega_rot*t)*x*y+r^2*cos(omega_rot*t)*x^2)))/(2*r^9)
    az = (3*R^3*mu*(c31*(sin(omega_rot*t)*y+cos(omega_rot*t)*x)+s31*(cos(omega_rot*t)*y-sin(omega_rot*t)*x))*z*(28*z^2-7*y^2-7*x^2-8*r^2))/(2*r^9)
    return SVector(ax, ay, az)
end


function cs32_perturbation(r_vector, t, mu, R, c32, s32, omega_rot)
    x,y,z, = r_vector
    r = norm(r_vector)

    ax = -((15*R^3*mu*(c32*(7*sin(omega_rot*t)^2*x*y^2-7*cos(omega_rot*t)^2*x*y^2+28*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y-4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y-7*sin(omega_rot*t)^2*x^3+7*cos(omega_rot*t)^2*x^3+2*r^2*sin(omega_rot*t)^2*x-2*r^2*cos(omega_rot*t)^2*x)+s32*(14*cos(omega_rot*t)*sin(omega_rot*t)*x*y^2-14*sin(omega_rot*t)^2*x^2*y+14*cos(omega_rot*t)^2*x^2*y+2*r^2*sin(omega_rot*t)^2*y-2*r^2*cos(omega_rot*t)^2*y-14*cos(omega_rot*t)*sin(omega_rot*t)*x^3+4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x))*z)/r^9)
    ay = -((15*R^3*mu*(c32*(7*sin(omega_rot*t)^2*y^3-7*cos(omega_rot*t)^2*y^3+28*cos(omega_rot*t)*sin(omega_rot*t)*x*y^2-7*sin(omega_rot*t)^2*x^2*y+7*cos(omega_rot*t)^2*x^2*y-2*r^2*sin(omega_rot*t)^2*y+2*r^2*cos(omega_rot*t)^2*y-4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x)+s32*(14*cos(omega_rot*t)*sin(omega_rot*t)*y^3-14*sin(omega_rot*t)^2*x*y^2+14*cos(omega_rot*t)^2*x*y^2-14*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y-4*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y+2*r^2*sin(omega_rot*t)^2*x-2*r^2*cos(omega_rot*t)^2*x))*z)/r^9)
    az = -((15*R^3*mu*(c32*(sin(omega_rot*t)^2*y^2-cos(omega_rot*t)^2*y^2+4*cos(omega_rot*t)*sin(omega_rot*t)*x*y-sin(omega_rot*t)^2*x^2+cos(omega_rot*t)^2*x^2)+s32*(2*cos(omega_rot*t)*sin(omega_rot*t)*y^2-2*sin(omega_rot*t)^2*x*y+2*cos(omega_rot*t)^2*x*y-2*cos(omega_rot*t)*sin(omega_rot*t)*x^2))*(7*z^2-r^2))/r^9)
    return SVector(ax, ay, az)
end

function cs42_perturbation(r_vector, t, mu, R, c42, s42, omega_rot)
    x,y,z, = r_vector
    r = norm(r_vector)

    ax = -((15*R^4*mu*(54*c42*sin(omega_rot*t)^2*x*y^2*z^2+108*s42*cos(omega_rot*t)*sin(omega_rot*t)*x*y^2*z^2-54*c42*cos(omega_rot*t)^2*x*y^2*z^2-108*s42*sin(omega_rot*t)^2*x^2*y*z^2+216*c42*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y*z^2+108*s42*cos(omega_rot*t)^2*x^2*y*z^2+12*s42*r^2*sin(omega_rot*t)^2*y*z^2-24*c42*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y*z^2-12*s42*r^2*cos(omega_rot*t)^2*y*z^2-54*c42*sin(omega_rot*t)^2*x^3*z^2-108*s42*cos(omega_rot*t)*sin(omega_rot*t)*x^3*z^2+54*c42*cos(omega_rot*t)^2*x^3*z^2+12*c42*r^2*sin(omega_rot*t)^2*x*z^2+24*s42*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x*z^2-12*c42*r^2*cos(omega_rot*t)^2*x*z^2-9*c42*sin(omega_rot*t)^2*x*y^4-18*s42*cos(omega_rot*t)*sin(omega_rot*t)*x*y^4+9*c42*cos(omega_rot*t)^2*x*y^4+18*s42*sin(omega_rot*t)^2*x^2*y^3-36*c42*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y^3-18*s42*cos(omega_rot*t)^2*x^2*y^3-2*s42*r^2*sin(omega_rot*t)^2*y^3+4*c42*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y^3+2*s42*r^2*cos(omega_rot*t)^2*y^3+18*s42*sin(omega_rot*t)^2*x^4*y-36*c42*cos(omega_rot*t)*sin(omega_rot*t)*x^4*y-18*s42*cos(omega_rot*t)^2*x^4*y-6*s42*r^2*sin(omega_rot*t)^2*x^2*y+12*c42*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y+6*s42*r^2*cos(omega_rot*t)^2*x^2*y+9*c42*sin(omega_rot*t)^2*x^5+18*s42*cos(omega_rot*t)*sin(omega_rot*t)*x^5-9*c42*cos(omega_rot*t)^2*x^5-4*c42*r^2*sin(omega_rot*t)^2*x^3-8*s42*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x^3+4*c42*r^2*cos(omega_rot*t)^2*x^3))/(2*r^11))
    ay = -((15*R^4*mu*(54*c42*sin(omega_rot*t)^2*y^3*z^2+108*s42*cos(omega_rot*t)*sin(omega_rot*t)*y^3*z^2-54*c42*cos(omega_rot*t)^2*y^3*z^2-108*s42*sin(omega_rot*t)^2*x*y^2*z^2+216*c42*cos(omega_rot*t)*sin(omega_rot*t)*x*y^2*z^2+108*s42*cos(omega_rot*t)^2*x*y^2*z^2-54*c42*sin(omega_rot*t)^2*x^2*y*z^2-108*s42*cos(omega_rot*t)*sin(omega_rot*t)*x^2*y*z^2+54*c42*cos(omega_rot*t)^2*x^2*y*z^2-12*c42*r^2*sin(omega_rot*t)^2*y*z^2-24*s42*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y*z^2+12*c42*r^2*cos(omega_rot*t)^2*y*z^2+12*s42*r^2*sin(omega_rot*t)^2*x*z^2-24*c42*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x*z^2-12*s42*r^2*cos(omega_rot*t)^2*x*z^2-9*c42*sin(omega_rot*t)^2*y^5-18*s42*cos(omega_rot*t)*sin(omega_rot*t)*y^5+9*c42*cos(omega_rot*t)^2*y^5+18*s42*sin(omega_rot*t)^2*x*y^4-36*c42*cos(omega_rot*t)*sin(omega_rot*t)*x*y^4-18*s42*cos(omega_rot*t)^2*x*y^4+4*c42*r^2*sin(omega_rot*t)^2*y^3+8*s42*r^2*cos(omega_rot*t)*sin(omega_rot*t)*y^3-4*c42*r^2*cos(omega_rot*t)^2*y^3+18*s42*sin(omega_rot*t)^2*x^3*y^2-36*c42*cos(omega_rot*t)*sin(omega_rot*t)*x^3*y^2-18*s42*cos(omega_rot*t)^2*x^3*y^2-6*s42*r^2*sin(omega_rot*t)^2*x*y^2+12*c42*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x*y^2+6*s42*r^2*cos(omega_rot*t)^2*x*y^2+9*c42*sin(omega_rot*t)^2*x^4*y+18*s42*cos(omega_rot*t)*sin(omega_rot*t)*x^4*y-9*c42*cos(omega_rot*t)^2*x^4*y-2*s42*r^2*sin(omega_rot*t)^2*x^3+4*c42*r^2*cos(omega_rot*t)*sin(omega_rot*t)*x^3+2*s42*r^2*cos(omega_rot*t)^2*x^3))/(2*r^11))
    az = -((45*R^4*mu*(c42*sin(omega_rot*t)^2*y^2+2*s42*cos(omega_rot*t)*sin(omega_rot*t)*y^2-c42*cos(omega_rot*t)^2*y^2-2*s42*sin(omega_rot*t)^2*x*y+4*c42*cos(omega_rot*t)*sin(omega_rot*t)*x*y+2*s42*cos(omega_rot*t)^2*x*y-c42*sin(omega_rot*t)^2*x^2-2*s42*cos(omega_rot*t)*sin(omega_rot*t)*x^2+c42*cos(omega_rot*t)^2*x^2)*z*(18*z^2-3*y^2-3*x^2-4*r^2))/(2*r^11))
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

end # end of module

# src/Core/Types.jl

module Types

using DataFrames, DifferentialEquations, Unitful, UnitfulAstro
using Unitful: 𝐋, 𝐓, Quantity
using Base: @kwdef  # useful for large structures where the programmer doesn't need to write all the fields when calling them
using GeometryBasics # for point2f (makie)
using DataInterpolations # to cubicspline
using StaticArrays # for svector

# public functions for users
export InitialConditions, InitialPlanetaryConditions, PerturbationParameters, SpiceInformations, GraphicInformation, PlottingOptions, 
    PropagatorOptions, PhysicalParams, GridParams, AbstractPropagator, CowellPropagator, HamiltonianPropagator, LagrangePEPropagator,
    NBodyPropagator, NBodyParticle, NBodySystemIC, NBodyParameters, CR3BPPropagator, CR3BPParameters,
    HamiltonEquations, LagrangeEquations, GaussEquations

"""
    CanonicalUnits(DU::Float64, TU::Float64, VU::Float64)

Structure holding the canonical units used to normalize the state vectors.

# Fields
- `DU::Float64`: Distance unit in kilometers (km).
- `TU::Float64`: Time unit in seconds (s).
- `VU::Float64`: Velocity unit in kilometers per second (km/s).
"""
struct CanonicalUnits
    DU::Float64 # distance unit (km)
    TU::Float64 # time unit (s)
    VU::Float64 # speed unit (km/s)
end


abstract type AbstractPropagator end
struct CowellPropagator <: AbstractPropagator end
struct HamiltonianPropagator <: AbstractPropagator end
struct LagrangePEPropagator <: AbstractPropagator end
struct NBodyPropagator <: AbstractPropagator end
struct CR3BPPropagator <: AbstractPropagator end

abstract type AbstractInitialConditions end

# types of planetary equations
abstract type AbstractEquationType end
struct HamiltonEquations <: AbstractEquationType end
struct LagrangeEquations <: AbstractEquationType end
struct GaussEquations <: AbstractEquationType end

# spice
# context for dispatch
abstract type SpiceContext end
struct StandardContext <: SpiceContext end
struct BinarySystemContext <: SpiceContext end

"""
    PerturbingBody(; name, mu, spice_id)

Defines an external perturbing celestial body for N-body simulations.

# Fields
- `name::Symbol`: Identification symbol of the body (e.g., `:moon`).
- `mu::Quantity`: The gravitational parameter of the body with physical units.
- `spice_id::String`: The SPICE NAIF integer ID used to retrieve ephemeris data.
"""
@kwdef struct PerturbingBody
    name::Symbol
    mu::Quantity{<:Real, 𝐋^3 * 𝐓^-2}
    spice_id::String
end

"""
    PerturbationParameters(; kwargs...)

Configuration structure containing all active physical and perturbing forces in the numerical simulation.

# Fields
- `mu`, `R`, `omega_rot`: Central body gravitational parameter, equatorial radius, and rotation rate.
- `j2` to `j18`: Zonal harmonic coefficients.
- `c22` to `c44`, `s22` to `s44`: Tesseral and sectorial harmonic coefficients.
- `alpha`, `cr`: Area-to-mass ratio and reflectivity coefficient for solar radiation pressure.
- `shadow_in_srp::Bool`: Flag to toggle eclipse/shadow modeling for solar radiation pressure.
- `n_bodies::Vector{PerturbingBody}`: List of external bodies acting as N-body perturbations.
"""
@kwdef struct PerturbationParameters{Tmu, TR, Tomega_rot, Talpha}
    mu::Tmu = nothing; R::TR = nothing; omega_rot::Tomega_rot = nothing;
    j2::Union{Real, Nothing} = nothing; j3::Union{Real, Nothing} = nothing; j4::Union{Real, Nothing} = nothing;
    j5::Union{Real, Nothing} = nothing; j6::Union{Real, Nothing} = nothing; j7::Union{Real, Nothing} = nothing;
    j8::Union{Real, Nothing} = nothing; j9::Union{Real, Nothing} = nothing; j10::Union{Real, Nothing} = nothing; 
    j11::Union{Real, Nothing} = nothing; j12::Union{Real, Nothing} = nothing; j13::Union{Real, Nothing} = nothing;
    j14::Union{Real, Nothing} = nothing; j15::Union{Real, Nothing} = nothing; j16::Union{Real, Nothing} = nothing;
    j17::Union{Real, Nothing} = nothing; j18::Union{Real, Nothing} = nothing;
    c22::Union{Real, Nothing} = nothing; c31::Union{Real, Nothing} = nothing; c32::Union{Real, Nothing} = nothing; c33::Union{Real, Nothing} = nothing; 
    c41::Union{Real, Nothing} = nothing; c42::Union{Real, Nothing} = nothing; c43::Union{Real, Nothing} = nothing; c44::Union{Real, Nothing} = nothing;
    s22::Union{Real, Nothing} = nothing; s31::Union{Real, Nothing} = nothing; s32::Union{Real, Nothing} = nothing; s33::Union{Real, Nothing} = nothing;
    s41::Union{Real, Nothing} = nothing; s42::Union{Real, Nothing} = nothing; s43::Union{Real, Nothing} = nothing; s44::Union{Real, Nothing} = nothing;
    alpha::Talpha = nothing; cr::Union{Real, Nothing} = nothing;
    shadow_in_srp::Bool = false
    n_bodies::Vector{PerturbingBody} = PerturbingBody[]
end

"""
    InitialConditions(; a0, e0, i0, h0, g0, f0)

Structure defining the initial conditions of the orbit using classical Keplerian elements.

# Fields
- `a0`: Semi-major axis.
- `e0`: Eccentricity.
- `i0`: Inclination.
- `h0`: Right ascension of the ascending node (RAAN).
- `g0`: Argument of periapsis.
- `f0`: True anomaly.
"""
@kwdef struct InitialConditions{Ta, Te, Ti, Th, Tg, Tf} <: AbstractInitialConditions
    # t is a type parameter - it can be of any type, and we will call this type tparam for internal organizational purposes
    a0::Ta
    e0::Te
    i0::Ti
    h0::Th
    g0::Tg
    f0::Tf
end

"""
    InitialPlanetaryConditions(; a0, e0, i0, h0, g0, l0)

Structure defining the initial conditions using a mix of Keplerian elements and the mean anomaly, commonly used in planetary equations.

# Fields
- `a0`: Semi-major axis.
- `e0`: Eccentricity.
- `i0`: Inclination.
- `h0`: Right ascension of the ascending node (RAAN).
- `g0`: Argument of periapsis.
- `l0`: Mean anomaly.
"""
@kwdef struct InitialPlanetaryConditions{Ta, Te, Ti, Th, Tg, Tl} <: AbstractInitialConditions
    a0::Ta
    e0::Te
    i0::Ti
    h0::Th
    g0::Tg
    l0::Tl
end


"""
    NBodyParticle(; name, r0, v0, mu, R)

One particle in a self-consistent N-body integration.

# Fields
- `name::Symbol`           : Human-readable label (e.g. `:sun`, `:jupiter`).
- `r0::SVector{3,Float64}` : Initial position [km] in the integration frame.
- `v0::SVector{3,Float64}` : Initial velocity [km s^-1] in the integration frame.
- `mu::Float64`            : Gravitational parameter mu = GM [km^3 s^-2].
- `R::Float64`             : Mean radius [km] - used for collision detection.
"""
struct NBodyParticle
    name::Symbol
    r0::SVector{3, Float64}
    v0::SVector{3, Float64}
    mu::Float64
    R::Float64
end

# convenience constructor with r defaulting to 0 (point mass)
NBodyParticle(; name, r0, v0, mu, R=0.0) =
    NBodyParticle(name, SVector{3,Float64}(r0), SVector{3,Float64}(v0), Float64(mu), Float64(R))

"""
    NBodySystemIC <: AbstractInitialConditions

Initial conditions for the full N-body integrator.

# Fields
- `bodies::Vector{NBodyParticle}` : All bodies to be integrated simultaneously.

# Notes
All positions and velocities must be expressed in the *same* inertial reference
frame (e.g. Solar System barycentre J2000 or a heliocentric frame).
"""
struct NBodySystemIC <: AbstractInitialConditions
    bodies::Vector{NBodyParticle}
end


"""
    PhysicalParams(; kwargs...)

Structure used to map numerical values to the symbolic symbols generated by Maxima.
It aggregates parameters for the central body, third-body perturbations, and solar radiation pressure,
ensuring strict geometric and numerical consistency during expression evaluation.
"""
@kwdef struct PhysicalParams

    # central body
    mu::Union{Float64, Nothing} = nothing
    R::Union{Float64, Nothing} = nothing
    # zonal harmonics
    j2::Union{Float64, Nothing} = nothing
    j3::Union{Float64, Nothing} = nothing
    j4::Union{Float64, Nothing} = nothing
    j5::Union{Float64, Nothing} = nothing
    j6::Union{Float64, Nothing} = nothing
    j7::Union{Float64, Nothing} = nothing
    j8::Union{Float64, Nothing} = nothing
    j9::Union{Float64, Nothing} = nothing
    j10::Union{Float64, Nothing} = nothing
    j11::Union{Float64, Nothing} = nothing
    j12::Union{Float64, Nothing} = nothing
    j13::Union{Float64, Nothing} = nothing
    j14::Union{Float64, Nothing} = nothing
    j15::Union{Float64, Nothing} = nothing
    j16::Union{Float64, Nothing} = nothing
    j17::Union{Float64, Nothing} = nothing
    j18::Union{Float64, Nothing} = nothing
    # tesseral harmonics
    c22::Union{Float64, Nothing} = nothing
    c31::Union{Float64, Nothing} = nothing
    c32::Union{Float64, Nothing} = nothing
    c33::Union{Float64, Nothing} = nothing
    c41::Union{Float64, Nothing} = nothing
    c42::Union{Float64, Nothing} = nothing
    c43::Union{Float64, Nothing} = nothing
    c44::Union{Float64, Nothing} = nothing
    omega_rot::Union{Float64, Nothing} = nothing # central body rotation rate

    # ========================================= #

    # third body
    mu_3::Union{Float64, Nothing} = nothing
    a_3::Union{Float64, Nothing} = nothing
    e_3::Union{Float64, Nothing} = nothing
    i_3::Union{Float64, Nothing} = nothing
    n_3::Union{Float64, Nothing} = nothing 
    l_3::Union{Float64, Nothing} = nothing    
    l_3_0::Union{Float64, Nothing} = nothing   # initial mean anomaly
    g_3::Union{Float64, Nothing} = nothing
    h_3::Union{Float64, Nothing} = nothing

    # ========================================= #

    # solar radiation pressure
    mu_sun::Union{Float64, Nothing} = nothing
    a_sun::Union{Float64, Nothing} = nothing
    e_sun::Union{Float64, Nothing} = nothing
    i_sun::Union{Float64, Nothing} = nothing
    n_sun::Union{Float64, Nothing} = nothing
    l_sun::Union{Float64, Nothing} = nothing   
    l_sun_0::Union{Float64, Nothing} = nothing # initial mean anomaly
    g_sun::Union{Float64, Nothing} = nothing   
    h_sun::Union{Float64, Nothing} = nothing   
    lambda_sun::Union{Float64, Nothing} = nothing
    beta::Union{Float64, Nothing} = nothing
end

"""
    SimulationParameters(perturb_params, perturb_func)

Container structure holding both the numerical parameters and the compiled physical dynamics function.
"""
@kwdef struct SimulationParameters{P <: PerturbationParameters}
    perturb_params::P
    perturb_func::Function
end


"""
    NBodyParameters

Runtime parameter bundle passed to the N-body ODE right-hand side.

# Fields
- `bodies::Vector{NBodyParticle}` : Snapshot of all particles (positions updated via the ODE state).
"""
struct NBodyParameters
    bodies::Vector{NBodyParticle}
end

"""
    CR3BPParameters{T}

Parameters for the Restricted Circular Three-Body Problem.
- `mu::T`: Mass ratio m2 / (m1 + m2).
"""
struct CR3BPParameters{T<:Real}
    mu::T
end

"""
    SpiceInformations(; kwargs...)

Configuration structure for SPICE ephemeris kernels and time domains.

# Fields
- `path_leapseconds_tls`: Path to the leapseconds (.tls) kernel.
- `path_solar_system_bsp`: Path to the main planetary ephemeris (.bsp) kernel.
- `path_another_body`: Optional path for specific celestial bodies (e.g., asteroids downloaded from jpl horizons).
- `path_primary_body`: Path to the central body kernel.
- `path_binary_system`, `path_primary_body_bin_sys`: Paths for barycentric and primary components of binary systems.
- `primary_body_SPICE`: NAIF ID of the main central body.
- `reference_frame`: Coordinate frame for ephemeris extraction (default: "ECLIPJ2000").
- `binary_system_SPICE`, `primary_body_bin_sys_SPICE`: IDs used to handle observer logic in binary systems.
- `initial_date`, `final_date`: ISO 8601 strings defining the simulation time boundaries.
"""
@kwdef struct SpiceInformations
    path_leapseconds_tls::Union{String, Nothing} = nothing
    path_solar_system_bsp::Union{String, Nothing} = nothing
    path_another_body::Union{String, Nothing} = nothing # for bodies not listed in path_solar_system_bsp
    path_primary_body::Union{String, Nothing} = nothing
    path_binary_system::Union{String, Nothing} = nothing
    path_primary_body_bin_sys::Union{String, Nothing} = nothing
    primary_body_SPICE::Union{String, Nothing} = nothing
    reference_frame::String = "ECLIPJ2000" # ecliptic reference frame (xy plane is earth's orbit)
    binary_system_SPICE::Union{String, Nothing} = nothing
    primary_body_bin_sys_SPICE::Union{String, Nothing} = nothing
    initial_date::Union{String, Nothing} = nothing
    final_date::Union{String, Nothing} = nothing # can be nothing if only initial date is needed
end

"""
    GraphicInformation(; kwargs...)

Structure containing the aesthetic properties and labels for generating Makie plots.
"""
@kwdef struct GraphicInformation
    body_color::Symbol
    alpha_opac::Float64 # opacity of the central body color
    orbit_color::Symbol
    color_a::Symbol = :blue
    color_e::Symbol = :blue
    color_i::Symbol = :blue
    color_h::Symbol = :blue
    color_g::Symbol = :blue
    color_alt::Symbol = :blue
    # ex: dict(:a => "semi-major axis (km)", :i => "inclination (rad)")
    custom_labels::Dict{Symbol, String} = Dict{Symbol, String}()
end

"""
    PlottingOptions(; kwargs...)

Defines the type of graphical analysis, window resolutions, and physical units applied during the post-processing pipeline.
"""
@kwdef struct PlottingOptions
    graph_type::Symbol
    separate_element::Symbol = :all_elements
    show_interactive::Bool = false
    use_scatter_plot::Bool = false
    width_fig::Int64 = 1000
    height_fig::Int64 = 800
    XY_projection::Bool = false
    XZ_projection::Bool = false
    YZ_projection::Bool = false
    axis_adjustment::Float64 = 0.0
    time_unit::Unitful.FreeUnits = u"d"
    dist_unit::Unitful.FreeUnits = u"km" # km, au, or m
    plot_in_radii::Bool = false # central body radius
end

"""
    PropagatorOptions(; kwargs...)

Defines the numerical integrator settings, tolerances, and environment flags.

# Fields
- `propagator`: The propagation method interface (e.g., `CowellPropagator()`).
- `canonical_unit_normalization::Bool`: Toggles canonical unit scaling (DU, TU) to ensure numerical stability.
- `second_order::Bool`: Flag indicating if the ODE is formulated as a second-order problem.
- `integrator`: The chosen `DifferentialEquations.jl` algorithm (e.g., `Vern7()`).
- `abstol`, `reltol`: Absolute and relative tolerances for the adaptive step solver.
- `dt`: Fixed time step size. If `nothing`, adaptive stepping is utilized.
- `maxiters::Int`: Maximum number of iterations allowed for the solver.
- `poincare_callback::Bool`: Toggles the generation of Poincare section events.
- `saveat::Bool`: If true, interpolates and saves data exactly at the user-specified time vector points.
"""
@kwdef struct PropagatorOptions{T<:AbstractPropagator, A}
    propagator::T = CowellPropagator()
    canonical_unit_normalization::Bool = false
    second_order::Bool = false
    integrator::A # type of parametric integrator to avoid any
    abstol::Float64 = 1e-8
    reltol::Float64 = 1e-8
    dt::Union{Real, Unitful.Time, Nothing} = nothing
    maxiters::Int = 1_000_000
    poincare_callback::Bool = false
    saveat::Bool=false # saves integrator timings by default
end


"""
    SimulationResult(; kwargs...)

Result structure parameterized by the propagator type. It stores the exact solver output, processed elements, and Poincare sections.

# Fields
- `solution`: The output from `DifferentialEquations.jl` (`ODESolution`).
- `elements`: DataFrame containing the processed orbital elements over time.
- `units`: The canonical units used during integration, if normalized.
- `initial_conditions`: The initial conditions used to start the simulation.
- `parameters`: The physical and perturbation parameters of the system.
- `propagator`: The type of propagator used (e.g., `CowellPropagator`).
- `equation_type`: Identifies the phase space used (e.g., Delaunay or Keplerian variables).
- `poincare_raw`: Raw crossing events generated by the continuous callback.
- `poincare_e_g`, `poincare_i_h`, `poincare_a_e`, `poincare_g_h`: Specific 2D projected points for Poincare maps.
"""
@kwdef struct SimulationResult
    solution::Union{Nothing, ODESolution} = nothing
    elements::Union{Nothing, DataFrame} = nothing # cowell no longer creates this by default
    units::Union{Nothing, CanonicalUnits} = nothing
    initial_conditions::Union{Nothing, AbstractInitialConditions} = nothing 
    parameters::Union{Nothing, PerturbationParameters, PhysicalParams, NamedTuple, Dict, CR3BPParameters} = nothing
    propagator::Union{Nothing, AbstractPropagator} = nothing
    equation_type::Union{Nothing, AbstractEquationType} = nothing    
    poincare_raw::Union{Nothing, Vector{Vector{Float64}}} = nothing
    poincare_e_g::Union{Nothing, Vector{Point2f}} = nothing
    poincare_i_h::Union{Nothing, Vector{Point2f}} = nothing
    poincare_a_e::Union{Nothing, Vector{Point2f}} = nothing
    poincare_g_h::Union{Nothing, Vector{Point2f}} = nothing
end

"""
    BodyInterpolator{T<:Real}(spice_id, mu, itp_x, itp_y, itp_z)

Container holding the evaluated splines (X, Y, Z) for a specific celestial body.
This ensures type stability and fast memory access inside the integration loop.
"""
struct BodyInterpolator{T<:Real}
    spice_id::String
    mu::T              # physical or dimensionless unit
    itp_x::CubicSpline
    itp_y::CubicSpline
    itp_z::CubicSpline
end

"""
    GridParams(; a_min, a_max, e_min, e_max, i_min, i_max, num_points)

Parameters defining the multidimensional grid boundaries used for numerical root-finding and phase space mapping.
"""
@kwdef struct GridParams
    a_min::Float64
    a_max::Float64
    e_min::Float64
    e_max::Float64
    i_min::Union{Float64, Nothing} = nothing
    i_max::Union{Float64, Nothing} = nothing
    num_points::Int
end

"""
    MappedRoots(; x_vals, y_vals, z_matrix, filtered_pairs, solved_variable)

Structure retaining the evaluated results of an analytical function across a parameter grid, 
usually representing energy surfaces or equilibrium manifolds.
"""
@kwdef struct MappedRoots
    x_vals::AbstractVector{<:Real}   # first axis of grid
    y_vals::AbstractVector{<:Real}   # second axis of grid
    z_matrix::AbstractMatrix{<:Real} # results matrix
    filtered_pairs::Vector{Tuple{Float64,Float64,Float64}}
    solved_variable::Symbol 
end


"""
    EquationF(; f, returns_angle)

Encapsulates a dynamically compiled Julia function originally derived from a Maxima symbolic expression.
The `returns_angle` flag indicates whether the output is strictly an angular dimension (radians).
"""
@kwdef struct EquationF{F}
    f::F
    returns_angle::Bool
end



end # end of module

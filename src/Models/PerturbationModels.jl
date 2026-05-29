# src/models/perturbationmodels.jl

"""
    PerturbationModels

Module responsible for constructing the physical perturbation models for celestial bodies.

It acts as a factory, extracting the necessary constants and gravity coefficients from the 
`Constants` module to build the complete `PerturbationParameters` structure required by the numerical solvers.

"""
module PerturbationModels

using ..Types
using  ..Constants
using Logging, Unitful, UnitfulAstro, Unitful.DefaultSymbols

export create_perturbation_model

"""
    create_perturbation_model(body_symbol::Symbol; kwargs...)

Constructs and returns a `PerturbationParameters` object containing the gravitational and 
environmental properties of the specified central body, along with any active perturbing forces.

This function selectively loads harmonic coefficients and N-body parameters based on user 
input. If a specific harmonic or body is not requested, its corresponding value is ignored, 
optimizing the integration process by ensuring the solver only computes active forces.

# Arguments
- `body_symbol::Symbol`: The central celestial body (e.g., `:earth`, `:moon`, `:didymos`). Must exist in `Constants.BODIES_DATA`.

# Keyword Arguments
- `j_harmonics::Vector{Int}`: A list of zonal harmonic degrees to activate (e.g., `[2, 3, 4]` activates J2, J3, and J4). Defaults to `Int[]`.
- `cs_harmonics::Vector{Int}`: A list of tesseral/sectorial harmonic degrees/orders to activate (e.g., `[22]` activates C22 and S22). Defaults to `Int[]`.
- `n_body_symbols::Vector{Symbol}`: A list of symbols representing external perturbing bodies (e.g., `[:sun, :moon]`). Defaults to `Symbol[]`.
- `srp_cr::Union{Real, Nothing}`: The reflectivity coefficient (Cr) for Solar Radiation Pressure (SRP) modeling. Defaults to `nothing`.
- `srp_alpha::Union{Real, Nothing}`: The area-to-mass ratio for SRP modeling, preferably with `Unitful` units. Defaults to `nothing`.
- `shadow_on::Bool`: Flag indicating whether to calculate the eclipse shadow model for SRP. Defaults to `false`.

# Returns
- `Types.PerturbationParameters`: A populated configuration structure containing all the requested physical constants and active perturbations.
"""
function create_perturbation_model(
    body_symbol::Symbol;
    j_harmonics::Vector{Int} = Int[], # ex: j_harmonics=[2, 3, 4]
    cs_harmonics::Vector{Int} = Int[], # ex: cs_harmonics=[22] activates c22 and s22
    n_body_symbols::Vector{Symbol} = Symbol[],
    srp_cr = nothing,
    srp_alpha = nothing,
    shadow_on::Bool = false
    )
    
    if !haskey(Constants.BODIES_DATA, body_symbol); error("Data for '\$body_symbol' not found."); end
    base_data = Constants.BODIES_DATA[body_symbol]


    # builds the list of perturbing bodies
    perturbing_bodies = Types.PerturbingBody[]
    for body_sym in n_body_symbols
        if haskey(Constants.BODIES_DATA, body_sym)
            body_data = Constants.BODIES_DATA[body_sym]
            push!(perturbing_bodies, Types.PerturbingBody(body_sym, body_data.mu, body_data.spice_id))
        else
            @warn "Data for the perturbing body '\$body_sym' not found. Ignoring."
        end
    end

    # conditionally extract active harmonic coefficients based on user input
    return Types.PerturbationParameters(
        mu  = get(base_data, :mu, nothing),
        R  = get(base_data, :R, nothing),
        omega_rot  = get(base_data, :omega_rot, nothing),
        j2 = 2 in j_harmonics ? get(base_data, :j2, nothing) : nothing,
        j3 = 3 in j_harmonics ? get(base_data, :j3, nothing) : nothing,
        j4 = 4 in j_harmonics ? get(base_data, :j4, nothing) : nothing,
        j5 = 5 in j_harmonics ? get(base_data, :j5, nothing) : nothing,
        j6 = 6 in j_harmonics ? get(base_data, :j6, nothing) : nothing,
        j7 = 7 in j_harmonics ? get(base_data, :j7, nothing) : nothing,
        j8 = 8 in j_harmonics ? get(base_data, :j8, nothing) : nothing,
        j9 = 9 in j_harmonics ? get(base_data, :j9, nothing) : nothing,
        j10 = 10 in j_harmonics ? get(base_data, :j10, nothing) : nothing,
        j11 = 11 in j_harmonics ? get(base_data, :j11, nothing) : nothing,
        j12 = 12 in j_harmonics ? get(base_data, :j12, nothing) : nothing,
        j13 = 13 in j_harmonics ? get(base_data, :j13, nothing) : nothing, 
        j14 = 14 in j_harmonics ? get(base_data, :j14, nothing) : nothing,
        j15 = 15 in j_harmonics ? get(base_data, :j15, nothing) : nothing,
        j16 = 16 in j_harmonics ? get(base_data, :j16, nothing) : nothing,
        j17 = 17 in j_harmonics ? get(base_data, :j17, nothing) : nothing,
        j18 = 18 in j_harmonics ? get(base_data, :j18, nothing) : nothing,
        c22 = 22 in cs_harmonics ? get(base_data, :c22, nothing) : nothing,
        c31 = 31 in cs_harmonics ? get(base_data, :c31, nothing) : nothing,
        c32 = 32 in cs_harmonics ? get(base_data, :c32, nothing) : nothing,
        c33 = 33 in cs_harmonics ? get(base_data, :c33, nothing) : nothing,
        c41 = 41 in cs_harmonics ? get(base_data, :c41, nothing) : nothing,
        c42 = 42 in cs_harmonics ? get(base_data, :c42, nothing) : nothing,
        c43 = 43 in cs_harmonics ? get(base_data, :c43, nothing) : nothing,
        c44 = 44 in cs_harmonics ? get(base_data, :c44, nothing) : nothing,
        s22 = 22 in cs_harmonics ? get(base_data, :s22, nothing) : nothing,
        s31 = 31 in cs_harmonics ? get(base_data, :s31, nothing) : nothing,
        s32 = 32 in cs_harmonics ? get(base_data, :s32, nothing) : nothing,
        s33 = 33 in cs_harmonics ? get(base_data, :s33, nothing) : nothing,
        s41 = 41 in cs_harmonics ? get(base_data, :s41, nothing) : nothing,
        s42 = 42 in cs_harmonics ? get(base_data, :s42, nothing) : nothing,
        s43 = 43 in cs_harmonics ? get(base_data, :s43, nothing) : nothing,
        s44 = 44 in cs_harmonics ? get(base_data, :s44, nothing) : nothing,
        cr = srp_cr,
        alpha  = srp_alpha,
        shadow_in_srp = shadow_on,
        n_bodies = perturbing_bodies
    )
end

end # end of module

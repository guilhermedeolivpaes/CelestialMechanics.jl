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

# atmospheric drag
- `drag_cd`: drag coefficient (dimensionless).
- `drag_am`: area-to-mass ratio for drag (m^2/kg).
- `atmosphere`: atmospheric density model (`AbstractAtmosphereModel`, default `nothing` = Harris-Priester).
- `min_altitude`: minimum altitude for reentry termination callback (default `100.0u"km"`, accepts `Unitful.Length` or `Real` in km).

# lorentz force
- `lorentz_qm`: charge-to-mass ratio for Lorentz force modeling (C/kg). Defaults to `nothing`.
- `lorentz_B`: planetary magnetic field dipole constant (default `nothing` = 8.0e15).

# earth precession
- `precession_on`: flag indicating whether to model Earth's general precession (Coriolis acceleration). Defaults to `false`.

# earth radiation pressure (albedo + IR)
- `albedo`: Mean Earth albedo coefficient (e.g. 0.30). Defaults to `nothing`.
- `thermal_ir_flux`: Mean Earth infrared radiation flux in W/m^2 (e.g. 237.0 W/m^2). Defaults to `nothing`.

# yarkovsky effect
- `yarkovsky_cy`: Yarkovsky thermal recoil efficiency coefficient (dimensionless, e.g. 0.15). Defaults to `nothing`.
- `yarkovsky_lag`: Diurnal thermal lag angle in degrees (default 25.0°). Defaults to `nothing`.
- `spin_axis`: Optional spin axis vector for the Yarkovsky effect (default: orbit normal). Defaults to `nothing`.

# general relativity (post-newtonian)
- `relativity_geodesic::Bool`: flag to model 1st-order Post-Newtonian (1PN) Schwarzschild geodesic precession. Defaults to `false`.
- `relativity_lense_thirring::Bool`: flag to model Lense-Thirring frame-dragging (gravitomagnetic) perturbation. Defaults to `false`.
- `spin_angular_momentum`: central body spin angular momentum per unit mass J [km^2/s] for Lense-Thirring. Defaults to `nothing`.

# solid earth tides
- `solid_tides::Bool`: flag to model degree-2 solid Earth tidal deformation (Moon/Sun). Defaults to `false`.
- `love_k2`: second-degree tidal Love number k2 (default `nothing` = 0.30 for Earth).

# space electrodynamics: coulomb drag (plasma drag)
- `coulomb_drag::Bool`: flag to model Coulomb drag / plasma drag on charged body. Defaults to `false`.
- `plasma_potential`: surface electric potential [V] (default `nothing` = -5.0 V).
- `plasma_ion_density`: ambient ion number density [m^-3] (default `nothing` = auto from atmosphere/altitude).
- `plasma_debye_length`: Debye shielding length in meters (default `nothing` = 0.02 m).
- `plasma_tether_length`: length of electrodynamic tether [m] for plasma brake (default `nothing` = 0.0 m).

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
    shadow_on = false,
    drag_cd = nothing,
    drag_am = nothing,
    atmosphere = nothing,
    min_altitude::Union{Real, Unitful.Length} = 100.0u"km",
    lorentz_qm = nothing,
    lorentz_B = nothing,
    precession_on::Bool = false,
    albedo = nothing,
    thermal_ir_flux = nothing,
    yarkovsky_cy = nothing,
    yarkovsky_lag = nothing,
    spin_axis = nothing,
    relativity_geodesic::Bool = false,
    relativity_lense_thirring::Bool = false,
    spin_angular_momentum = nothing,
    solid_tides::Bool = false,
    love_k2 = nothing,
    coulomb_drag::Bool = false,
    plasma_potential = nothing,
    plasma_ion_density = nothing,
    plasma_debye_length = nothing,
    plasma_tether_length = nothing,
    gravity_harmonics::Union{Nothing, Types.GravityHarmonics} = nothing,
    use_cunningham::Bool = false,
    max_degree::Union{Nothing, Int} = nothing,
    max_order::Union{Nothing, Int} = nothing,
    normalized_gravity::Bool = false,
    cs_phase::Bool = false,
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

    # --- input validation: warn if requested coefficients are missing ---
    for n in j_harmonics
        key = Symbol("j$n")
        if !haskey(base_data, key) || isnothing(get(base_data, key, nothing))
            @warn "zonal harmonic j$n requested but not available for '$body_symbol'; it will be ignored."
        end
    end

    for nm in cs_harmonics
        ckey = Symbol("c$nm")
        skey = Symbol("s$nm")
        c_missing = !haskey(base_data, ckey) || isnothing(get(base_data, ckey, nothing))
        s_missing = !haskey(base_data, skey) || isnothing(get(base_data, skey, nothing))
        if c_missing && s_missing
            @warn "tesseral/sectorial harmonics c$nm and s$nm requested but not available for '$body_symbol'; they will be ignored."
        elseif c_missing
            @warn "coefficient c$nm requested but not available for '$body_symbol'; it will be ignored."
        elseif s_missing
            @warn "coefficient s$nm requested but not available for '$body_symbol'; it will be ignored."
        end
    end

    # assemble GravityHarmonics if requested or if max_degree is specified
    gh = gravity_harmonics
    if isnothing(gh) && (use_cunningham || !isnothing(max_degree))
        deg = if !isnothing(max_degree)
            max_degree
        else
            d_j = isempty(j_harmonics) ? 0 : maximum(j_harmonics)
            d_cs = isempty(cs_harmonics) ? 0 : maximum(div(nm, 10) for nm in cs_harmonics)
            max(d_j, d_cs, 2)
        end
        ord = if !isnothing(max_order)
            max_order
        elseif !isempty(cs_harmonics)
            maximum(mod(nm, 10) for nm in cs_harmonics)
        else
            min(deg, 4)
        end

        C_mat = zeros(Float64, deg + 1, ord + 1)
        S_mat = zeros(Float64, deg + 1, ord + 1)

        if !isnothing(max_degree)
            for n in 2:deg
                j_key = Symbol("j$n")
                if haskey(base_data, j_key) && !isnothing(get(base_data, j_key, nothing))
                    C_mat[n + 1, 1] = -Float64(base_data[j_key])
                end
                for m in 1:min(n, ord)
                    c_key = Symbol("c$n$m")
                    s_key = Symbol("s$n$m")
                    if haskey(base_data, c_key) && !isnothing(get(base_data, c_key, nothing))
                        C_mat[n + 1, m + 1] = Float64(base_data[c_key])
                    end
                    if haskey(base_data, s_key) && !isnothing(get(base_data, s_key, nothing))
                        S_mat[n + 1, m + 1] = Float64(base_data[s_key])
                    end
                end
            end
        else
            for n in j_harmonics
                j_key = Symbol("j$n")
                if haskey(base_data, j_key) && !isnothing(get(base_data, j_key, nothing))
                    C_mat[n + 1, 1] = -Float64(base_data[j_key])
                end
            end
            for nm in cs_harmonics
                n = div(nm, 10)
                m = mod(nm, 10)
                if n <= deg && m <= ord
                    c_key = Symbol("c$nm")
                    s_key = Symbol("s$nm")
                    if haskey(base_data, c_key) && !isnothing(get(base_data, c_key, nothing))
                        C_mat[n + 1, m + 1] = Float64(base_data[c_key])
                    end
                    if haskey(base_data, s_key) && !isnothing(get(base_data, s_key, nothing))
                        S_mat[n + 1, m + 1] = Float64(base_data[s_key])
                    end
                end
            end
        end

        is_norm = normalized_gravity || (haskey(base_data, :normalized_gravity) && base_data.normalized_gravity == true)
        gh = Types.GravityHarmonics(C_mat, S_mat; max_degree=deg, max_order=ord, normalized=is_norm, cs_phase=cs_phase)
    end

    # conditionally extract active harmonic coefficients based on user input
    return Types.PerturbationParameters(
        mu  = get(base_data, :mu, nothing),
        R  = get(base_data, :R, nothing),
        omega_rot  = get(base_data, :omega_rot, nothing),
        gravity_harmonics = gh,
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
        cd = drag_cd,
        am_drag = drag_am,
        atmosphere = atmosphere,
        min_altitude = min_altitude,
        qm = lorentz_qm,
        B_mag = lorentz_B,
        precession_on = precession_on,
        albedo = albedo,
        thermal_ir_flux = thermal_ir_flux,
        yarkovsky_cy = yarkovsky_cy,
        yarkovsky_lag = yarkovsky_lag,
        spin_axis = spin_axis,
        relativity_geodesic = relativity_geodesic,
        relativity_lense_thirring = relativity_lense_thirring,
        spin_angular_momentum = spin_angular_momentum,
        solid_tides = solid_tides,
        love_k2 = love_k2,
        coulomb_drag = coulomb_drag,
        plasma_potential = plasma_potential,
        plasma_ion_density = plasma_ion_density,
        plasma_debye_length = plasma_debye_length,
        plasma_tether_length = plasma_tether_length,
        n_bodies = perturbing_bodies
    )
end

end # end of module

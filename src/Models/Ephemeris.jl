# src/Models/Ephemeris.jl

"""
    Ephemeris

Module responsible for obtaining the position vectors of celestial bodies using NASA's SPICE library.

It provides functions to calculate the positions of perturbing bodies (such as the Sun, Moon, or other planets) relative to a central observer over a specified time domain.


"""
module Ephemeris

using SPICE
using ..Types
using ..Coordinates
using Logging
using Unitful
using Unitful.DefaultSymbols
using LinearAlgebra

export get_ics_celestial_bodies, create_particle

"""
    get_body_position_vectors(spice_info, t_vector_seconds, target, observer)

Calculates the time series of position vectors of a `target` body relative to an `observer` body.

This function acts as a wrapper around the `SPICE.spkpos` routine. It converts the initial UTC date to Ephemeris Time (ET) and iterates over the provided time vector to extract the position at each step.

# Arguments
- `spice_info::Types.SpiceInformations`: Structure containing the initial date and the reference frame (e.g., "ECLIPJ2000").
- `t_vector_seconds::AbstractVector`: A vector of time steps in seconds, relative to the initial date.
- `target::Union{String, Nothing}`: The SPICE NAIF integer ID (as a string) of the target body to be observed.
- `observer::Union{String, Nothing}`: The SPICE NAIF integer ID (as a string) of the central observer body.

# Returns
- `Vector{Vector{<:Unitful.Quantity}}`: A vector of 3D position arrays containing the coordinates of the target relative to the observer, with physical units (km). Returns an empty array if the target or observer is not defined.
"""
function get_body_position_vectors(
    spice_info::Types.SpiceInformations, 
    t_vector_seconds::AbstractVector,
    target::Union{String, Nothing}, # target positions
    observer::Union{String, Nothing} # in relation to the observer
    )

    if isnothing(target) || isnothing(observer)
        @warn "Target ($target) ou Observer ($observer) nao definido. Retornando vetor vazio."

        return Vector{Vector{<:Unitful.Quantity}}()
    end

    isnothing(spice_info.initial_date) && error("initial_date is required for SPICE ephemeris evaluation.")
    start_time_et = utc2et(spice_info.initial_date)
    
    # using a "comprehension" for performance and clarity
    return [
        let
            et = start_time_et + t
            # target orbit around the observer
            pos, _ = spkpos(target, et, spice_info.reference_frame, "none", observer) 
            pos * km
        end
        for t in t_vector_seconds
    ]
end

# --- public functions (API) ---

"""
    get_ics_celestial_bodies(; spice_info, mu_barycenter, target, observer)

Extracts the exact initial conditions (Keplerian orbital elements and mean anomaly) of a target celestial body relative to an observer at a specific epoch using SPICE kernels.

This function loads all the necessary kernels (.bsp, .tls) specified in the `spice_info` structure, retrieves the Cartesian state vector (position and velocity) via `SPICE.spkezr`, and finally converts it to classical orbital elements using the `Coordinates.jl` module.

# Keyword Arguments
- `spice_info::Types.SpiceInformations`: Configuration structure containing the kernel paths, the reference frame, and the initial UTC date.
- `mu_barycenter::Float64`: The standard gravitational parameter (μ) of the central body or barycenter, required for the Cartesian-to-Keplerian conversion.
- `target::Union{String, Nothing}`: The SPICE ID of the celestial body whose orbit is being calculated.
- `observer::Union{String, Nothing}`: The SPICE ID of the central body acting as the focal point of the orbit.

# Returns
- `Tuple`: Returns `(a, e, i, g, h, l)`, which correspond to the semi-major axis (km), eccentricity, inclination (rad), argument of periapsis (rad), right ascension of the ascending node (rad), and mean anomaly (rad). Returns a tuple of NaNs if the SPICE evaluation fails.
"""
function get_ics_celestial_bodies(;
    spice_info::Types.SpiceInformations, 
    mu_barycenter::Float64, 
    target::Union{String, Nothing},
    observer::Union{String, Nothing} 
    )

    if isnothing(target) || isnothing(observer) 
        @error "Target or Observer not defined. Returning an empty array."
        return (NaN, NaN, NaN, NaN, NaN, NaN, NaN)
    end

    # define start_date_str here so the catch block can see it
    isnothing(spice_info.initial_date) && error("initial_date is required for SPICE ephemeris evaluation.")
    start_date_str = spice_info.initial_date

    try      
        !isnothing(spice_info.path_leapseconds_tls) && isfile(spice_info.path_leapseconds_tls) && furnsh(spice_info.path_leapseconds_tls)
        !isnothing(spice_info.path_solar_system_bsp) && isfile(spice_info.path_solar_system_bsp) && furnsh(spice_info.path_solar_system_bsp)
        !isnothing(spice_info.path_another_body)     && isfile(spice_info.path_another_body)     && furnsh(spice_info.path_another_body)
        !isnothing(spice_info.path_binary_system)    && isfile(spice_info.path_binary_system)    && furnsh(spice_info.path_binary_system)
        !isnothing(spice_info.path_primary_body_bin_sys) && isfile(spice_info.path_primary_body_bin_sys) && furnsh(spice_info.path_primary_body_bin_sys)
        
        et_start = utc2et(start_date_str)

        # orbit of the target around the observer
        rv_central_body_target, _ = spkezr(target, et_start, spice_info.reference_frame, "NONE", observer)
        r_target_vec = reshape(@views(rv_central_body_target[1:3]), 3, 1) # 3x1 format for the function 
        v_target_vec = reshape(@views(rv_central_body_target[4:6]), 3, 1)

        # converts to orbital elements 
        a_t, e_t, i_t, h_t, g_t, f_t = Coordinates.state_vectors_to_orbital_elements(r_target_vec, v_target_vec, mu_barycenter)
        # extracts scalars (the function returns vectors)
        a_target_km = a_t[1]
        e_target_val = e_t[1]
        i_target_rad = deg2rad(i_t[1]) # function returns rad
        g_target_rad = deg2rad(g_t[1]) # pericenter argument
        h_target_rad = deg2rad(h_t[1]) # raan
        f_target_rad = deg2rad(f_t[1]) # true anomaly
        # retorna angulos em rad (-pi, pi)

        # calculation of the initial mean solar anomaly (nu -> M)
        l_target_rad = Coordinates.true_to_mean_anomaly(f_target_rad, e_target_val)

        return a_target_km, e_target_val, i_target_rad, g_target_rad, h_target_rad, l_target_rad
    catch e
        @error "SPICE Error for $target: $(sprint(showerror, e))"
        return (NaN, NaN, NaN, NaN, NaN, NaN)
    finally
        kclear()
    end
end

"""
create_particle(; body_data::NamedTuple, spice_info::Types.SpiceInformations)

creates an nbodyparticle extracting the initial cartesian state vector from spice ephemerides.
it automatically loads the required kernels, converts the initial date to ephemeris time, 
and fetches the position and velocity relative to the solar system barycenter.

# arguments
- `body_data::NamedTuple`: a named tuple containing the physical parameters of the celestial body (mu, r, spice_id, name).
- `spice_info::Types.SpiceInformations`: structure containing the start date, reference frame, and paths to the spice kernels.

# returns
- `Types.NBodyParticle`: a populated particle struct ready for nbody numerical integration.
"""
function create_particle(; body_data::NamedTuple, spice_info::Types.SpiceInformations,)

    isnothing(spice_info.initial_date) && error("initial_date is required for SPICE ephemeris evaluation.")
    start_date_str = spice_info.initial_date

    et_start = utc2et(start_date_str)
    try
        !isnothing(spice_info.path_leapseconds_tls) && isfile(spice_info.path_leapseconds_tls) && furnsh(spice_info.path_leapseconds_tls)
        !isnothing(spice_info.path_solar_system_bsp) && isfile(spice_info.path_solar_system_bsp) && furnsh(spice_info.path_solar_system_bsp)
        !isnothing(spice_info.path_another_body)     && isfile(spice_info.path_another_body)     && furnsh(spice_info.path_another_body)
        !isnothing(spice_info.path_binary_system)    && isfile(spice_info.path_binary_system)    && furnsh(spice_info.path_binary_system)
        !isnothing(spice_info.path_primary_body_bin_sys) && isfile(spice_info.path_primary_body_bin_sys) && furnsh(spice_info.path_primary_body_bin_sys)

        # extracts position (km) and speed ​​(km/s) via SPICE
        # spkezr returns (state_vector, light_time)
        # observer pattern and the barycenter of the Solar System
        state, _ = spkezr(body_data.spice_id, et_start, spice_info.reference_frame, "NONE", "0")
        r0 = state[1:3]
        v0 = state[4:6]

        return NBodyParticle(
            name = body_data.name,
            r0   = r0,
            v0   = v0,
            mu   = ustrip(body_data.mu), 
            R    = ustrip(body_data.R)
        )
        
    finally
        kclear()
    end # end try

end # end function


end # end module
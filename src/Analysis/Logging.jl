# src/Analysis/Logging.jl

"""
    Logging

Module responsible for formatting and displaying structured telemetry and configuration summaries in the console. 

It provides a transparent, human-readable overview of the physical parameters and active perturbing forces loaded into the numerical simulation environment prior to execution.
"""
module Logging

using ..Types
using Unitful

export log_simulation_setup

"""
    log_simulation_setup(p_params::Types.PerturbationParameters, spice_info::Types.SpiceInformations)

Generates and prints a detailed, formatted log of the numerical simulation setup to the console.

This function inspects the provided perturbation parameters and SPICE configurations, dynamically 
constructing a summary report that lists the central body properties and explicitly enumerates all 
active forces. This includes zonal and tesseral harmonics, N-body gravitational perturbations, and 
Solar Radiation Pressure (SRP) settings (including shadow model status and reflectivity coefficients).

# Arguments
- `p_params::Types.PerturbationParameters`: The structure containing the active physical parameters and perturbations.
- `spice_info::Types.SpiceInformations`: The structure containing SPICE kernel configurations and reference frames.
"""
function log_simulation_setup(p_params::Types.PerturbationParameters, spice_info::Types.SpiceInformations)
    log_message = """
    -------------------------------------------
    ---  SIMULATION CONFIGURATION (NUMERICAL)  ---
    -------------------------------------------

    ** Central Body ($(spice_info.primary_body_SPICE)) **
      - Gravitational Parameter (mu): $(p_params.mu)
      - Equatorial Radius (R):       $(p_params.R)

    ** Activated Perturbations **"""
    perturbations_found = false

    if !isnothing(p_params.gravity_harmonics)
        gh = p_params.gravity_harmonics
        norm_str = gh.normalized ? "normalized (4π)" : "unnormalized"
        log_message *= "\n  - Spherical Harmonics (Cunningham): degree $(gh.max_degree), order $(gh.max_order) [$norm_str]\n -----------------------------------------------------------"
        perturbations_found = true
    elseif !isnothing(p_params.j2) && !iszero(p_params.j2)
        log_message *= "\n  - j2: $(p_params.j2) (analytical fast-path)\n -----------------------------------------------------------"
        perturbations_found = true
    end

    # checks the list of n-bodies with a loop
    if !isempty(p_params.n_bodies)
        perturbations_found = true
        for body in p_params.n_bodies
            log_message *= "\n  - Disturbing Body: $(body.name) (mu = $(body.mu)); Reference Frame: $(spice_info.reference_frame) \n -----------------------------------------------------------"
        end
    end

    # checks solar radiation pressure (srp)
    if !isnothing(p_params.cr) && !iszero(p_params.cr)
        log_message *= "\n  - Solar Radiation Pressure (SRP):"
        log_message *= "\n    - Reflectivity Coefficient (CR): $(p_params.cr)"
        log_message *= "\n    - Area/mass ratio (alpha):           $(p_params.alpha)"
        log_message *= "\n    - Shadow Model Activated:       $(p_params.shadow_in_srp)"
        log_message *= "\n    - Reference Frame:            $(spice_info.reference_frame) \n -----------------------------------------------------------"
        perturbations_found = true
    end

    if !isnothing(p_params.cd) && !iszero(p_params.cd)
       log_message *= "\n  - Atmospheric Drag:"
       log_message *= "\n    - Drag Coefficient (Cd):        $(p_params.cd)"
       log_message *= "\n    - Area/mass ratio (Am_drag):     $(p_params.am_drag)"
       model_str = isnothing(p_params.atmosphere) ? "Harris-Priester (default)" : string(typeof(p_params.atmosphere).name.name)
       log_message *= "\n    - Atmosphere Model:              $model_str"
       log_message *= "\n    - Min Reentry Altitude:          $(p_params.min_altitude)"
       log_message *= "\n -----------------------------------------------------------"
       perturbations_found = true
   end

    if !perturbations_found
        log_message *= "\n  - No additional perturbation activated (Keplerian orbit)"
    end

    log_message *= "\n\n    -------------------------------------------"

    @info log_message

end

end # end module

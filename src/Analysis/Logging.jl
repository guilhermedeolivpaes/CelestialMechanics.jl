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

    # checks zonal harmonics
    jharmonics = [
        (2, p_params.j2), (3, p_params.j3), (4, p_params.j4), (5, p_params.j5), (6, p_params.j6),
        (7, p_params.j7), (8, p_params.j8), (9, p_params.j9), (10, p_params.j10), (11, p_params.j11),
        (12, p_params.j12), (13, p_params.j13), (14, p_params.j14), (15, p_params.j15), (16, p_params.j16),
        (17, p_params.j17),(18, p_params.j18)
    ]
    for (n, j_val) in jharmonics
        if !isnothing(j_val) && !iszero(j_val)
            log_message *= "\n  - j$n: $j_val \n -----------------------------------------------------------"
            perturbations_found = true
        end
    end

    # checks tesseral and sectorial harmonics
    charmonics = [(22, p_params.c22, p_params.s22), (31, p_params.c31, p_params.s31), (32, p_params.c32, p_params.s32), (33, p_params.c33, p_params.s33),
                  (41, p_params.c41, p_params.s41), (42, p_params.c42, p_params.s42), (43, p_params.c43, p_params.s43), (44, p_params.c44, p_params.s44)]
    for (n, c_val, s_val) in charmonics
        if (!isnothing(c_val) && !iszero(c_val)) || (!isnothing(s_val) && !iszero(s_val))
            log_message *= "\n  - c$n: $c_val \n  - s$n: $s_val \n -----------------------------------------------------------" 
            perturbations_found = true
        end
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

    if !perturbations_found
        log_message *= "\n  - No additional perturbation activated (Keplerian orbit)"
    end

    log_message *= "\n\n    -------------------------------------------"

    @info log_message

end

end # end module

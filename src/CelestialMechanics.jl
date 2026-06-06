# src/CelestialMechanics.jl

"""
    CelestialMechanics

A package for high precision orbital dynamics simulation and analysis.

The `CelestialMechanics` module integrates the necessary tools to define simulation scenarios,
propagate trajectories of celestial bodies and spacecraft, and analyze the results.
It uses SPICE ephemerides for planetary data and state-of-the-art differential equation solvers
for numerical propagation.

# Available propagators

| Propagator | Module | State vector | Notes |
|---|---|---|---|
| `CowellPropagator`        | `Cowell`          | 6 (r, v)      | Perturbations J2-J18, C22-C44, SRP, N-bodies |
| `HamiltonianPropagator`   | `Hamiltonian`     | 6 (Delaunay)  | Symbolic Hamiltonian loaded from .txt |
| `LagrangePEPropagator`    | `LagrangePE`      | 6 (orbital)  | Lagrange planetary equations |
| `NBodyPropagator`         | `NBodyIntegrator` | 6N (ri, vi) | Full N-bodies, symplectic and explicit |

"""

module CelestialMechanics

using Reexport
using Printf  # needed for @sprintf in NBodyIntegrator diagnostics

# --- 1. inclusion of modules (order matters) ---
# core - no internal dependencies
include("Core/Types.jl") # does not depend on any
include("Core/Constants.jl") # does not depend on any
include("Core/Coordinates.jl")

# models
include("Models/PerturbationModels.jl")     
include("Models/Ephemeris.jl")
include("Models/PerturbationEquations.jl") 

# analysis / utilities
include("Analysis/Logging.jl") # logging depends on types
include("Analysis/Plotting.jl")    # plotting depends on types
include("Analysis/DataHandling.jl") # datahandling depends on types
include("Analysis/ReadEquations.jl")
include("Analysis/EvaluatetxtEquations.jl")
include("Analysis/PostProcessing.jl")

# propagator utilities (shared across all propagators)
include("Propagators/PropagatorUtils.jl")

# physics equations
include("Dynamics.jl")              # dynamics depends on several

# propagators
include("Propagators/Cowell.jl") # cowell depends on almost everything
include("Propagators/Hamiltonian.jl")
include("Propagators/LagrangePE.jl")
include("Propagators/NBodyIntegrator.jl") 
include("Propagators/CR3BP.jl") 


# --- 2. usings to bring names into the scope of the main module ---
@reexport using .Types
@reexport using .Coordinates
@reexport using .PerturbationModels
@reexport using .Ephemeris
@reexport using .DataHandling
@reexport using .Plotting
@reexport using .Cowell
@reexport using .Hamiltonian
@reexport using .LagrangePE
@reexport using .Logging       
@reexport using .Constants
@reexport using .PostProcessing
@reexport using .EvaluatetxtEquations
@reexport using .ReadEquations
@reexport using .NBodyIntegrator
@reexport using .CR3BP


# --- 3. final exports (the public api of my library) ---

# Types.jl
export InitialConditions, InitialPlanetaryConditions, PerturbationParameters, SpiceInformations, GraphicInformation, PlottingOptions, 
    PropagatorOptions, PhysicalParams, GridParams, AbstractPropagator, CowellPropagator, HamiltonianPropagator, LagrangePEPropagator,
    NBodyPropagator, NBodyParticle, NBodySystemIC, NBodyParameters, CR3BPPropagator, CR3BPParameters,
    HamiltonEquations, LagrangeEquations, GaussEquations

# PerturbationModels.jl
export create_perturbation_model, create_particle

# Ephemeris.jl
export get_ics_celestial_bodies, create_particle

# Coordinates.jl
export mean_to_true_anomaly, true_to_mean_anomaly, delaunay_to_keplerian, keplerian_to_delaunay, delaunay_to_cartesian, lagrange_to_cartesian, unwrap_angle

# Constants.jl
export BODIES_DATA, I0_SI, C_SI, AU_IN_M, N_MOON 

# ReadEquations.jl
export load_equation_function, load_ode_system, build_analytical_function

# PostProcessing.jl
export run_post_analysis, running_average, secular_rate_epoch_average

# Plotting.jl
export plot_orbital_results, plot_poincare_section, plot_phase_contours, plot_dynamic_map, plot_nbody_2d


# ── simulation entry points ───────────────────────────────────────────────────
export run_simulation,                  # cowell
       run_lpe_simulation,              # lagrange
       run_hamiltonian_simulation,      # hamiltonian
       run_nbody_simulation,            # nbodyintegrator 
       run_cr3bp_simulation, 
       jacobi_constant


# Logging.jl
export log_simulation_setup

# EvaluatetxtEquations.jl
export numerical_root_mapper, evaluate_analytical_map, compute_osc_corrections, generate_phase_portrait_data

# DataHandling.jl
export save_filtered_results

end # end of module

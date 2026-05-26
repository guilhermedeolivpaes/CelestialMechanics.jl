"""
    PostProcessing

module responsible for the systematic processing of raw numerical integration data.

it bridges the gap between the ode solver output and the final analysis by recovering physical units, converting phase space variables (e.g., delaunay or cartesian) into structured dataframes of classical orbital elements, projecting poincare maps, and orchestrating collision detection, data persistence, and visualization.

"""
module PostProcessing
using DataFrames, CSV, Statistics, Unitful, UnitfulAstro
using RecursiveArrayTools
using GeometryBasics # for point2f
using StaticArrays
using LinearAlgebra
using ..Types
using ..Types: AbstractEquationType
using ..Coordinates
using ..DataHandling
using ..Plotting
using ..PerturbationEquations

export run_post_analysis


"""
    compute_perturbation_potential(r_vector, params) -> Float64

Calculates the specific potential energy of the conservative perturbations (e.g., J2).
Note: The actual perturbation energy added to the system is -U_pert.
"""
function compute_perturbation_potential(r_vector::AbstractVector, params::Types.PerturbationParameters)
    mu = Float64(ustrip(params.mu))
    R  = Float64(ustrip(params.R))
    
    U_pert = 0.0
    
    # add j2 potential if active
    if !isnothing(params.j2) && !iszero(params.j2)
        U_pert += PerturbationEquations.j2_potential(r_vector, mu, R, params.j2)
    end

    # add j3_potential, j4_potential, etc., here in the future
    
    return U_pert
end

"""
    _check_collisions(res, idx)

internal helper function that analyzes the periapsis altitude array to detect collisions with the central body. 
it issues a warning if the altitude drops below zero, and an info message if it enters a predefined low-altitude safety margin (e.g., 5% of the body's radius).

# arguments
- `res`: the simulation result object containing the elements and parameters.
- `idx::Int`: the index of the orbit being analyzed.
"""
function _check_collisions(res, idx)
    # ignore nans to avoid false warnings during convergence failures
    alts = filter(!isnan, res.elements.alt_peri_km)
    if isempty(alts) return end
    
    min_alt = minimum(alts)
    
    # extracts the body radius from the parameter instance saved in the result
    # use uconvert and ustrip to ensure the radius value is in km (pure number)
    # doesn't matter if the user entered the radius in meters or canonical units
    val_R = res.parameters.R
    R_km = val_R isa Unitful.Quantity ? Float64(ustrip(u"km", val_R)) : Float64(val_R)
    
    # define the dynamic limit (e.g., 5% of the radius)
    low_altitude_limit = R_km * 0.05
    
    if min_alt <= 0
        @warn "Orbit $idx: Collision detected! (Min Alt: $(round(min_alt, digits=4)) km)"
    elseif min_alt < low_altitude_limit
        @info "Orbit $idx: Low altitude warning ($(round(min_alt, digits=4)) km)" safe_limit=round(low_altitude_limit, digits=4)
    end
end



"""
    _extract_states(sol, propagator, mu_phys)

internal multiple-dispatch method that extracts the time array, cartesian position vectors, and cartesian velocity vectors from the raw `odesolution`.
it automatically applies the necessary phase-space transformations (e.g., delaunay-to-cartesian or lagrange-to-cartesian) depending on the active `propagator`.

# arguments
- `sol`: the raw `odesolution` object.
- `propagator::Types.CowellPropagator`: the active propagator type.
- `mu_phys::Float64`: the gravitational parameter of the central body in physical units.

# returns
- a tuple containing the time array, cartesian position matrix, and cartesian velocity matrix.
"""
function _extract_states(sol, ::Types.CowellPropagator, mu_phys)
    t = sol.t
    u = sol.u
    if u[1] isa RecursiveArrayTools.ArrayPartition
        v = [step.x[1] for step in u]
        r = [step.x[2] for step in u]
    else
        r = [@view(step[1:3]) for step in u]
        v = [@view(step[4:6]) for step in u]
    end
    return t, reduce(hcat, r), reduce(hcat, v)
end

"""
    _extract_states(sol, ::Types.HamiltonianPropagator, mu_phys)

internal dispatch for extracting cartesian state vectors from a hamiltonian propagation.

iterates over the solution array of delaunay action-angle variables `[L, G, H, l, g, h]`, sequentially converting them to classical keplerian elements (solving kepler's equation for the true anomaly) and finally to cartesian position and velocity vectors.

# arguments
- `sol`: the raw `odesolution` object.
- `::Types.HamiltonianPropagator`: the active propagator type.
- `mu_phys::Float64`: the gravitational parameter of the central body in physical units.

# returns
- a tuple containing the time array, cartesian position matrix, and cartesian velocity matrix.
"""
function _extract_states(sol, ::Types.HamiltonianPropagator, mu_phys) 
    u_states = sol.u
    r_list = Vector{SVector{3, Float64}}()
    v_list = Vector{SVector{3, Float64}}()
    
    for state in u_states
        L, G, H, l, g, h = state
        a, e, i, h, g, l = Coordinates.delaunay_to_keplerian(L, G, H, l, g, h, mu_phys)
        f = Coordinates.mean_to_true_anomaly(l, e)
        r, v = Coordinates.orbital_elements_to_state_vectors(a, e, i, h, g, f, mu_phys)
        
        push!(r_list, SVector{3}(r))
        push!(v_list, SVector{3}(v))
    end
    return sol.t, reduce(hcat, r_list), reduce(hcat, v_list)
end

"""
    _extract_states(sol, ::Types.LagrangePEPropagator, mu_phys)

internal dispatch for extracting cartesian state vectors from a lagrange planetary equations propagation.

iterates over the solution array of keplerian elements parametrized by the mean anomaly `[a, e, i, h, g, l]`. it resolves kepler's equation to convert the mean anomaly (`l`) into the true anomaly (`f`) before transforming the state into cartesian coordinates.

# arguments
- `sol`: the raw `odesolution` object.
- `::Types.LagrangePEPropagator`: the active propagator type.
- `mu_phys::Float64`: the gravitational parameter of the central body in physical units.

# returns
- a tuple containing the time array, cartesian position matrix, and cartesian velocity matrix.
"""
function _extract_states(sol, ::Types.LagrangePEPropagator, mu_phys)
    t = sol.t
    r_list = Vector{SVector{3, Float64}}()
    v_list = Vector{SVector{3, Float64}}()
    
    for state in sol.u
        a, e, i, h, g, l = state
        # mandatory conversion: l (mean) -> f (true)
        # because orbital_elements_to_state_vectors requires 'f'
        f = Coordinates.mean_to_true_anomaly(l, e)
        r, v = Coordinates.lagrange_to_cartesian(a, e, i, h, g, f, mu_phys)
        push!(r_list, r)
        push!(v_list, v)
    end
    return t, reduce(hcat, r_list), reduce(hcat, v_list)
end

"""
    _process_poincare_data(propagator, p_data_raw, units_used, prop_opts, mu_phys)

internal multiple-dispatch method that extracts and scales raw continuous callback events (root crossings) into specific 2d poincare map projections (e.g., `e` vs `g`, `i` vs `h`).

# arguments
- `propagator::Types.CowellPropagator`: the active propagator type.
- `p_data_raw::Dict`: dictionary containing raw poincare map data.
- `units_used`: units used for normalization.
- `prop_opts::Types.PropagatorOptions`: propagator options.
- `mu_phys::Float64`: the gravitational parameter of the central body in physical units.

# returns
- a tuple of arrays representing points in the poincare sections: `(e_g, i_h, a_e, g_h)`.
"""
function _process_poincare_data(::Types.CowellPropagator, p_data_raw, units_used, prop_opts, mu_phys)
    if prop_opts.canonical_unit_normalization && !isnothing(units_used)
        p_a_e = isnothing(p_data_raw[:a_e]) ? nothing : [Point2f(p[1], p[2] * units_used.DU) for p in p_data_raw[:a_e]]
    else
        p_a_e = p_data_raw[:a_e]
    end
    return p_data_raw[:e_g], p_data_raw[:i_h], p_a_e, p_data_raw[:g_h]
end

"""
    _process_poincare_data(::Types.HamiltonianPropagator, p_data_raw, ::Any, ::Any, mu_phys)

internal dispatch for processing raw poincare section events generated during hamiltonian propagation.

transforms the captured delaunay momenta and coordinates into classical keplerian elements, mapping them into standard 2d point2f projection planes (`e` vs `g`, `i` vs `h`, `a` vs `e`, and `g` vs `h`) with angles appropriately converted to degrees.

# arguments
- `::Types.HamiltonianPropagator`: the active propagator type.
- `p_data_raw::Dict`: dictionary containing raw poincare map data.
- `::Any`: ignored units parameter.
- `::Any`: ignored propagator options parameter.
- `mu_phys::Float64`: the gravitational parameter of the central body in physical units.

# returns
- a tuple of arrays representing points in the poincare sections: `(e_g, i_h, a_e, g_h)`.
"""
function _process_poincare_data(::Types.HamiltonianPropagator, p_data_raw, ::Any, ::Any, mu_phys)
    p_e_g = Point2f[]; p_i_h = Point2f[]; p_a_e = Point2f[]; p_g_h = Point2f[]
    
    raw_snaps = p_data_raw[:raw_snaps]
    if !isnothing(raw_snaps)
        for u in raw_snaps
            L, G, H, l, g, h = u
            # now we use mu_phys passed as argument, not via dict
            a, e, i, h, g, l = Coordinates.delaunay_to_keplerian(L, G, H, l, g, h, mu_phys)
            push!(p_e_g, Point2f(rad2deg(mod2pi(g)), e))
            push!(p_i_h, Point2f(rad2deg(mod2pi(h)), rad2deg(i)))
            push!(p_a_e,   Point2f(e, a))
            push!(p_g_h, Point2f(rad2deg(mod2pi(h)), rad2deg(mod2pi(g))))
        end
    end
    return p_e_g, p_i_h, p_a_e, p_g_h
end

"""
    _process_poincare_data(::Types.LagrangePEPropagator, p_data_raw, ::Any, ::Any, mu_phys)

internal dispatch for processing raw poincare section events generated during lagrange planetary equations propagation.

maps the directly captured keplerian elements into standard 2d point2f projection planes (`e` vs `g`, `i` vs `h`, `a` vs `e`, and `g` vs `h`), converting all angular variables to degrees and binding them within `[0, 360)`.

# arguments
- `::Types.LagrangePEPropagator`: the active propagator type.
- `p_data_raw::Dict`: dictionary containing raw poincare map data.
- `::Any`: ignored units parameter.
- `::Any`: ignored propagator options parameter.
- `mu_phys::Float64`: the gravitational parameter of the central body in physical units.

# returns
- a tuple of arrays representing points in the poincare sections: `(e_g, i_h, a_e, g_h)`.
"""
function _process_poincare_data(::Types.LagrangePEPropagator, p_data_raw, ::Any, ::Any, mu_phys)
    p_e_g = Point2f[]; p_i_h = Point2f[]; p_a_e = Point2f[]; p_g_h = Point2f[]
    
    raw_snaps = p_data_raw[:raw_snaps]
    if !isnothing(raw_snaps)
        for u in raw_snaps
            # direct extraction: the lagrangepe state is already keplerian [a, e, i, h, g, l]
            a, e, i, h, g, l = u
            
            push!(p_e_g, Point2f(rad2deg(mod2pi(g)), e))
            push!(p_i_h, Point2f(rad2deg(mod2pi(h)), rad2deg(i)))
            push!(p_a_e, Point2f(e, a))
            push!(p_g_h, Point2f(rad2deg(mod2pi(h)), rad2deg(mod2pi(g))))
        end
    end
    return p_e_g, p_i_h, p_a_e, p_g_h
end

"""
    _process_simulation_results(sol, ic, params, prop_opts, units_used, p_data_raw, eq_type)

core internal engine that constructs the standardized `simulationresult`.

it handles unit denormalization (if canonical units were used), evaluates the geometric orbital elements over the entire time series, and builds the final `dataframe` containing both physical cartesian states and keplerian elements.

# arguments
- `sol`: the raw `odesolution` object.
- `ic`: the initial conditions object.
- `params`: the physical parameters object.
- `prop_opts`: the propagator options.
- `units_used`: the units used for normalization.
- `p_data_raw::Dict`: the raw poincare data dictionary.
- `eq_type`: the equation type used for propagation.

# returns
- a fully populated `Types.SimulationResult` object.
"""
function _process_simulation_results(sol, ic, params, prop_opts, units_used, p_data_raw, eq_type)
    # universal way to get the pure number, whether it is physicalparams or perturbationparameters
    mu_phys = Float64(ustrip(params.mu))
    R_phys  = Float64(ustrip(params.R))

    # 1. extraction via dispatch
    t_raw, r_raw, v_raw = _extract_states(sol, prop_opts.propagator, mu_phys)

    # 2. denormalization (cowell case)
    if prop_opts.propagator isa CowellPropagator && prop_opts.canonical_unit_normalization && !isnothing(units_used)
        t_phys = t_raw .* units_used.TU
        r_mat  = r_raw .* units_used.DU
        v_mat  = v_raw .* units_used.VU
        p_a_e  = isnothing(p_data_raw[:a_e]) ? nothing : [Point2f(p[1], p[2] * units_used.DU) for p in p_data_raw[:a_e]]
    else
        t_phys = t_raw * u"s"
        r_mat  = r_raw  # already in km
        v_mat  = v_raw  # already in km/s
        p_a_e  = p_data_raw[:a_e]
    end

    # 3. poincare
    p_e_g, p_i_h, _, p_g_h = _process_poincare_data(
        prop_opts.propagator, p_data_raw, units_used, prop_opts, mu_phys
    )

    # if it is lagrange or hamiltonian, we get the elements directly from sol.u
    # this avoids the "collision detected" error due to noise in the r,v -> a,e conversion
    if prop_opts.propagator isa LagrangePEPropagator || prop_opts.propagator isa HamiltonianPropagator
        u_matrix = reduce(hcat, sol.u)
        if prop_opts.propagator isa LagrangePEPropagator
            # order: [a, e, i, h, g, l]
            a_v = u_matrix[1, :]
            e_v = u_matrix[2, :]
            i_v = rad2deg.(u_matrix[3, :])
            h_v = rad2deg.(u_matrix[4, :])
            g_v = rad2deg.(u_matrix[5, :])
            l_v = u_matrix[6, :]
            # true anomaly nu requires conversion of l
            f_v = rad2deg.([Coordinates.mean_to_true_anomaly(l, e) for (l, e) in zip(l_v, e_v)])
        else
            # for hamiltonian, convert delaunay -> keplerian first
            elements = [Coordinates.delaunay_to_keplerian(u..., mu_phys) for u in sol.u]
            a_v = [el[1] for el in elements]
            e_v = [el[2] for el in elements]
            i_v = rad2deg.([el[3] for el in elements])
            h_v = rad2deg.([el[4] for el in elements])
            g_v = rad2deg.([el[5] for el in elements])
            f_v = rad2deg.([Coordinates.mean_to_true_anomaly(el[6], el[2]) for el in elements])
        end

        epsilon_v = -mu_phys ./ (2.0 .* a_v)

    else
        # only cowell uses geometric conversion r,v -> a,e
        a_v, e_v, i_v, h_v, g_v, f_v, epsilon_v = Coordinates.state_vectors_to_orbital_elements(
            ustrip.(r_mat), ustrip.(v_mat), mu_phys
        )
    end

    alt_peri_km = (a_v .* (1 .- e_v) .- R_phys)

    # energy calculations
    E_kep_vec = epsilon_v # Keplerian energy
    E_pert_vec = zeros(length(t_phys))
    E_tot_vec = zeros(length(t_phys))
    
    # check if parameters is of type PerturbationParameters (Cowell)
    if prop_opts.propagator isa CowellPropagator && params isa Types.PerturbationParameters
        for j in 1:length(t_phys)
            r_vec = ustrip.(r_mat[:, j])
            U_pert = compute_perturbation_potential(r_vec, params)
            E_pert_vec[j] = U_pert 
            E_tot_vec[j] = E_kep_vec[j] + E_pert_vec[j]
        end
    else
        E_tot_vec .= E_kep_vec
    end

    # relative errors
    E_kep_rel_error = (E_kep_vec .- E_kep_vec[1]) ./ abs(E_kep_vec[1])
    
    # protect against division by zero if there are no perturbations
    E_pert_rel_error = zeros(length(t_phys))
    if abs(E_pert_vec[1]) > 1e-16
        E_pert_rel_error = (E_pert_vec .- E_pert_vec[1]) ./ abs(E_kep_vec[1])
    end
    
    E_tot_rel_error = (E_tot_vec .- E_tot_vec[1]) ./ abs(E_tot_vec[1])

    # dataframe with the elements now much more precise
    df = DataFrame(
        time = t_phys,
        a_km = a_v, e = e_v, i_deg = i_v,
        h_deg = h_v, g_deg = g_v, f_deg = f_v,
        X_km = ustrip.(r_mat[1, :]), Y_km = ustrip.(r_mat[2, :]), Z_km = ustrip.(r_mat[3, :]),
        Vx_km_s = ustrip.(v_mat[1, :]), Vy_km_s = ustrip.(v_mat[2, :]), Vz_km_s = ustrip.(v_mat[3, :]),
        alt_peri_km = alt_peri_km,
        energy_kep_rel_error = E_kep_rel_error,
        energy_pert_rel_error = E_pert_rel_error,
        energy_tot_rel_error = E_tot_rel_error
    )

    mean_motion = sqrt(mu_phys/df.a_km[1]^3)*u"rad/s"
    period = (2*pi* u"rad"/mean_motion)
    period_hr = uconvert(u"hr", period)

    @info "mean motion = $mean_motion"
    @info "period = $period_hr"  

    return Types.SimulationResult(
        solution = sol, 
        elements = df, 
        units = units_used,
        initial_conditions = ic, 
        parameters = params,
        propagator = prop_opts.propagator,
        equation_type = eq_type, # cowell defined as nothing (does not use symbolic eq from .txt)
        poincare_raw = p_data_raw[:raw_snaps],
        poincare_e_g = p_e_g, 
        poincare_i_h = p_i_h,
        poincare_a_e = p_a_e, 
        poincare_g_h = p_g_h
    )
end


"""
    run_post_analysis(::Types.AbstractPropagator, all_results, prop_opts, plot, plot_opts, graph_info, output_dir, save_csv)

executes the complete post processing pipeline on a batch of raw simulation results for standard propagators.

this function iterates through the output of the numerical propagators converting mathematical phase space states into physical orbital elements. it triggers collision detection dynamically saves the cleaned tabular data to disk and generates visualizations if requested.

# arguments
- `::Types.AbstractPropagator`: dispatch type interceptor for standard single body propagators like cowell or hamiltonian.
- `all_results::Vector{Types.SimulationResult}`: array of raw simulation results generated by the propagation modules.
- `prop_opts::Types.PropagatorOptions`: the propagator configuration used necessary to determine unit denormalization and extraction rules.
- `plot::Bool`: flag to trigger the generation of plots via makie.
- `plot_opts::Union{Types.PlottingOptions, Nothing}`: plotting configuration structure required if plot is true.
- `graph_info::Union{Types.GraphicInformation, Nothing}`: aesthetic configuration for plots required if plot is true.
- `output_dir::String`: directory path where the processed csv files will be saved.
- `save_csv::Bool`: flag to toggle the persistence of the processed dataframes to disk.

# returns
- `Tuple{Vector{Types.SimulationResult}, Vector{Any}}`: a tuple containing the array of fully processed simulation results and a vector of the generated makie figure objects.
"""
function run_post_analysis(
    ::Types.AbstractPropagator, 
    all_results::Vector{Types.SimulationResult}, 
    prop_opts::Types.PropagatorOptions, 
    plot::Bool,
    plot_opts::Union{Types.PlottingOptions, Nothing},
    graph_info::Union{Types.GraphicInformation, Nothing},
    output_dir::String,
    save_csv::Bool
    )

    figs = [] # list for figures
    processed_results = Types.SimulationResult[] # list for results

    for (idx, raw_res) in enumerate(all_results)
        # prepare input dictionary for processor
        # if cowell, use point2f fields. if hamiltonian, use poincare_raw.
        p_data_raw = Dict(
            :e_g => raw_res.poincare_e_g,
            :i_h => raw_res.poincare_i_h,
            :a_e => raw_res.poincare_a_e,
            :g_h => raw_res.poincare_g_h,
            :raw_snaps => raw_res.poincare_raw
        )
        
        res = _process_simulation_results(
            raw_res.solution, 
            raw_res.initial_conditions, 
            raw_res.parameters, 
            prop_opts, raw_res.units,
            p_data_raw, 
            raw_res.equation_type
        )

        push!(processed_results, res) # save processed result here

        _check_collisions(res, idx)
        if save_csv; DataHandling.save_data_frame(true, res, idx, output_dir); end

        if plot && !isnothing(plot_opts) && !isnothing(graph_info)
            fig = Plotting.plot_orbital_results(res, plot_opts, graph_info)
            push!(figs, fig)
            if plot_opts.show_interactive; display(fig); end
        end
    end

    # returns a tuple: (results, figures)
    return processed_results, figs
end


"""
    _process_nbody_results(sol, bodies, units)

processes the raw n-body ode solution, mapping the flat 6n state vector 
into a structured dataframe exactly as output by the solver (canonical or physical).

# arguments
- `sol`: the odesolution object from the solver.
- `bodies`: vector of nbodyparticle structures.
- `units`: canonical units for denormalization (nothing if physical).

# returns
- `dataframe`: a wide-format dataframe containing time and states for all bodies.
"""
function _process_nbody_results(sol, bodies, units)
    n_bodies = length(bodies)
    t_vals = sol.t
    
    # generate column names for all bodies
    col_names = Symbol[:t]
    for b in bodies
        name = string(b.name)
        append!(col_names, [
            Symbol("x_", name), Symbol("y_", name), Symbol("z_", name),
            Symbol("vx_", name), Symbol("vy_", name), Symbol("vz_", name)
        ])
    end

    # initialize the matrix to store results
    n_steps = length(t_vals)
    data_matrix = Matrix{Float64}(undef, n_steps, 1 + 6 * n_bodies)

    # since we keep the solver's native units, we can just dump the flat vector directly
    for i in 1:n_steps
        data_matrix[i, 1] = t_vals[i]
        data_matrix[i, 2:end] = sol.u[i]
    end

    return DataFrames.DataFrame(data_matrix, col_names)
end



# ------------------------------------------------------------------------------
# specific method for nbody
# ------------------------------------------------------------------------------
"""
    run_post_analysis(::Types.NBodyPropagator, all_results, prop_opts, plot, plot_opts, graph_info, output_dir, save_csv)

executes the post processing pipeline specifically tailored for n-body numerical simulations.

this function iterates through the raw n-body integration results. it extracts the flat 6n state vectors, applies coordinate denormalization if canonical units were used, and structures the trajectories into a wide format dataframe. standard keplerian plotting routines are bypassed, as n-body visualizations are handled directly in the simulation scripts. it dynamically saves the cleaned tabular data to disk if requested.

# arguments
- `::Types.NBodyPropagator`: dispatch type interceptor for n-body propagators.
- `all_results::Vector{Types.SimulationResult}`: array of raw simulation results generated by the n-body module.
- `prop_opts::Types.PropagatorOptions`: the propagator configuration used.
- `plot::Bool`: plotting flag (currently ignored for n-body processing).
- `plot_opts::Union{Types.PlottingOptions, Nothing}`: plotting configuration structure (ignored).
- `graph_info::Union{Types.GraphicInformation, Nothing}`: aesthetic configuration for plots (ignored).
- `output_dir::String`: directory path where the processed csv files will be saved.
- `save_csv::Bool`: flag to toggle the persistence of the processed dataframes to disk.

# returns
- `Tuple{Vector{Types.SimulationResult}, Vector{Any}}`: a tuple containing the array of fully processed simulation results and an empty vector for the figures.
"""
function run_post_analysis(
    ::Types.NBodyPropagator, 
    all_results::Vector{Types.SimulationResult}, 
    prop_opts::Types.PropagatorOptions, 
    plot::Bool,
    plot_opts::Union{Types.PlottingOptions, Nothing},
    graph_info::Union{Types.GraphicInformation, Nothing},
    output_dir::String,
    save_csv::Bool
    )

    figs = [] 
    processed_results = Types.SimulationResult[] 

    for (idx, raw_res) in enumerate(all_results)
        
        # uses your helper function to create the denormalized dataframe
        df_nbody = _process_nbody_results(
            raw_res.solution, 
            raw_res.initial_conditions.bodies, 
            raw_res.units
        )

        # builds a new simulationresult containing the generated dataframe in elements
        res = Types.SimulationResult(
            solution           = raw_res.solution,
            elements           = df_nbody,
            units              = raw_res.units,
            initial_conditions = raw_res.initial_conditions,
            parameters         = raw_res.parameters,
            propagator         = raw_res.propagator,
            equation_type      = raw_res.equation_type,
            poincare_raw       = raw_res.poincare_raw
        )

        push!(processed_results, res) 

        # if saving csv
        if save_csv; DataHandling.save_data_frame(true, res, idx, output_dir); end

        # note: the original plot_orbital_results expects keplerian elements
        # for n-body, the plot is being done directly in the simulation script
        # therefore, we do not call plotting.plot_orbital_results here
    end

    return processed_results, figs
end


# ------------------------------------------------------------------------------
# main entry point for post-analysis using multiple dispatch
# ------------------------------------------------------------------------------
"""
    run_post_analysis(; all_results, prop_opts, plot, plot_opts, graph_info, output_dir, save_csv)

main entry point for the post processing pipeline. it uses multiple dispatch based on the propagator type defined in prop_opts to route the raw simulation results to the appropriate specialized processing method.

# arguments
- `all_results::Vector{Types.SimulationResult}`: array of raw simulation results.
- `prop_opts::Types.PropagatorOptions`: the propagator configuration used.
- `plot::Bool`: flag to trigger the generation of plots.
- `plot_opts::Union{Types.PlottingOptions, Nothing}`: plotting configuration structure.
- `graph_info::Union{Types.GraphicInformation, Nothing}`: aesthetic configuration for plots.
- `output_dir::String`: directory path where the processed files will be saved.
- `save_csv::Bool`: flag to toggle the persistence of the processed data to disk.

# returns
- `Tuple{Vector{Types.SimulationResult}, Vector{Any}}`: a tuple containing the array of fully processed simulation results and a vector of the generated figure objects.
"""
function run_post_analysis(;
    all_results::Vector{Types.SimulationResult}, 
    prop_opts::Types.PropagatorOptions, 
    plot::Bool = false,
    plot_opts::Union{Types.PlottingOptions, Nothing} = nothing,
    graph_info::Union{Types.GraphicInformation, Nothing} = nothing,
    output_dir::String,
    save_csv::Bool = true
    )

    # the first argument dictates which method will be called
    return run_post_analysis(
        prop_opts.propagator, 
        all_results, 
        prop_opts, 
        plot, 
        plot_opts, 
        graph_info, 
        output_dir, 
        save_csv
    )
end


"""
    run_post_analysis(::Types.CR3BPPropagator, all_results::Vector{Types.SimulationResult}, prop_opts::Types.PropagatorOptions, plot::Bool, plot_opts::Union{Types.PlottingOptions, Nothing}, graph_info::Union{Types.GraphicInformation, Nothing}, output_dir::String, save_csv::Bool)

executes the post processing pipeline tailored for circular restricted three-body problem (cr3bp) simulations.

this function iterates through raw cr3bp results, extracts the state vectors, calculates the jacobi constant and its relative error over time, and structures the data into a dataframe. it also processes poincare section events, persists the tabular data to disk if requested, and triggers the generation and display of cr3bp specific plots.

# arguments
- `::Types.CR3BPPropagator`: dispatch type interceptor for cr3bp propagators.
- `all_results::Vector{Types.SimulationResult}`: array of raw simulation results generated by the cr3bp module.
- `prop_opts::Types.PropagatorOptions`: the propagator configuration used.
- `plot::Bool`: flag to toggle the generation of plots.
- `plot_opts::Union{Types.PlottingOptions, Nothing}`: plotting configuration structure.
- `graph_info::Union{Types.GraphicInformation, Nothing}`: aesthetic configuration for plots.
- `output_dir::String`: directory path where the processed csv files will be saved.
- `save_csv::Bool`: flag to toggle the persistence of the processed dataframes to disk.

# returns
- `Tuple{Vector{Types.SimulationResult}, Vector{Any}}`: a tuple containing the array of fully processed simulation results and a vector of the generated figures.
"""
function run_post_analysis(
    ::Types.CR3BPPropagator, 
    all_results::Vector{Types.SimulationResult}, 
    prop_opts::Types.PropagatorOptions, 
    plot::Bool,
    plot_opts::Union{Types.PlottingOptions, Nothing},
    graph_info::Union{Types.GraphicInformation, Nothing},
    output_dir::String,
    save_csv::Bool
    )

    figs = [] 
    processed_results = Types.SimulationResult[] 

    for (idx, raw_res) in enumerate(all_results)
        sol = raw_res.solution
        mu = raw_res.parameters.mu

        # 1. State Extraction
        t_vals = sol.t
        x_vals = [u[1] for u in sol.u]
        y_vals = [u[2] for u in sol.u]
        z_vals = [u[3] for u in sol.u]
        vx_vals = [u[4] for u in sol.u]
        vy_vals = [u[5] for u in sol.u]
        vz_vals = [u[6] for u in sol.u]

        # 2. Jacobi Constant Calculation (C_J)
        # C_J = (x^2 + y^2) + 2(1-mu)/r1 + 2mu/r2 - (vx^2 + vy^2 + vz^2)
        C_J = zeros(length(t_vals))
        for i in 1:length(t_vals)
            r1 = sqrt((x_vals[i] + mu)^2 + y_vals[i]^2 + z_vals[i]^2)
            r2 = sqrt((x_vals[i] - 1 + mu)^2 + y_vals[i]^2 + z_vals[i]^2)
            v2 = vx_vals[i]^2 + vy_vals[i]^2 + vz_vals[i]^2
            C_J[i] = (x_vals[i]^2 + y_vals[i]^2) + 2*(1 - mu)/r1 + 2*mu/r2 - v2
        end

        # Relative error of the Jacobi Constant
        C_J_error = (C_J .- C_J[1]) ./ C_J[1]

        # 3. DataFrame creation
        df = DataFrame(
            time = t_vals,
            x = x_vals, y = y_vals, z = z_vals,
            vx = vx_vals, vy = vy_vals, vz = vz_vals,
            jacobi_error = C_J_error
        )

        # 4. Poincare Section Processing (x vs vx)
        # We will use the poincare_e_g field of the struct as a generic container for Point2f
        p_x_vx = Point2f[]
        if !isnothing(raw_res.poincare_raw)
            for state in raw_res.poincare_raw
                # state = [x, y, z, vx, vy, vz]
                push!(p_x_vx, Point2f(state[1], state[4])) # Takes x and vx
            end
        end

        # 5. Reconstruct the Result
        res = Types.SimulationResult(
            solution = sol, 
            elements = df, 
            parameters = raw_res.parameters,
            propagator = raw_res.propagator,
            poincare_raw = raw_res.poincare_raw,
            poincare_e_g = p_x_vx # Using as an alias for x_vx
        )

        push!(processed_results, res) 

        if save_csv; DataHandling.save_data_frame(true, res, idx, output_dir); end

        # 6. Plotting
        if plot && !isnothing(plot_opts) && !isnothing(graph_info)
            fig = Plotting.plot_cr3bp_results(res, plot_opts, graph_info)
            push!(figs, fig)
            if plot_opts.show_interactive; display(fig); end
        end
    end

    return processed_results, figs
end

end

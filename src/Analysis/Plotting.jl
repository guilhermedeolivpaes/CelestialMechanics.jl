# src/Analysis/Plotting.jl

"""
    Plotting

module responsible for generating all graphical visualizations of the simulation results using the `makie.jl` library.

it provides a unified interface to plot 2d and 3d orbital trajectories, classical orbital element variations over time, contour maps for hamiltonian phase portraits, 
and heatmaps for multidimensional equilibrium root-finding.

"""
module Plotting

using Makie, DataFrames, Unitful
# import the structs 
using ..Types

export plot_orbital_results, plot_cr3bp_results, plot_phase_contours, plot_dynamic_map, plot_nbody_2d

"""
    _apply_mod_360(vetor)

helper function that applies a modulo 360 operation to all elements of a vector.
this is used to bound angular variables (like raan and argument of periapsis) strictly within the `[0, 360)` degree interval for clean plotting.

# arguments
- `vetor::AbstractVector`: vector of angular values.

# returns
- vector of bounded angular values.
"""
function _apply_mod_360(vetor)
    return mod.(vetor, 360)
end

"""
    _plot_sphere_old!(ax, R_km, info)

internal helper function to generate and render a 3d sphere representing the central celestial body.

it uses standard spherical coordinate parameterization to construct the surface mesh and applies the color and opacity settings defined in the graphic information structure.

# arguments
- `ax`: the makie 3d axis object where the sphere will be drawn.
- `R_km::Real`: the radius of the central body. the unit scale (e.g., km or canonical radii) must match the plot's current distance factor.
- `info::Types.GraphicInformation`: structure containing aesthetic configurations like `body_color` and `alpha_opac`.
"""
function _plot_sphere_old!(ax, R_km, info)
    Θ = range(0, 2π, length=100)
    ϕ = range(0, π, length=50)
    x_sphere = [R_km * cos(θ) * sin(φ) for θ in Θ, φ in ϕ]
    y_sphere = [R_km * sin(θ) * sin(φ) for θ in Θ, φ in ϕ]
    z_sphere = [R_km * cos(φ) for θ in Θ, φ in ϕ]

    color_matrix = zeros(size(x_sphere))
    color_colormap = [(info.body_color, info.alpha_opac), (info.body_color, info.alpha_opac)]
    
    surface!(ax, x_sphere, y_sphere, z_sphere,
        color = color_matrix,
        colormap = color_colormap,
        shading = true # makie now uses only true/false
    )
end

"""
    _plot_sphere!(ax, R_km, info)

internal helper function to draw a central sphere using makie's mesh and sphere primitives.

# arguments
- `ax`: the makie 3d axis.
- `R_km::Real`: radius.
- `info::Types.GraphicInformation`: graphics info struct.
"""
function _plot_sphere!(ax, R_km, info)
    # 1. extract numeric value and ensure float32 for makie
    r_clean = Float32(ustrip(R_km))
    
    # 2. use mesh! with the sphere primitive
    # this creates a perfectly filled sphere without artifacts at the poles
    mesh!(ax, Sphere(Point3f(0), r_clean), 
        color = (info.body_color, info.alpha_opac),
        shading = true, # improves volume perception
        transparency = true          # ensures opacity works well
    )
end

"""
    _plot_orbital_core!(fig, df, opts, info, R_km)

the central plotting engine that processes the trajectory dataframe and generates the requested subplots on the provided makie figure.

this function is responsible for dynamic scaling (converting between physical units like kilometers or astronomical units, or normalizing to central body radii) 
and routing the data to the correct visualization layout based on the `graph_type` flag. it supports 3d trajectories, classical keplerian elements, 2d projections, 
and equinoctial coordinates.

# arguments
- `fig`: the parent makie `figure` object where the axes and plots will be placed.
- `df::DataFrame`: the tabular data containing the time series of state vectors, keplerian elements, and energy metrics.
- `opts::Types.PlottingOptions`: configuration structure detailing the plot type, unit conversions, and specific layout toggles.
- `info::Types.GraphicInformation`: aesthetic parameters including line colors, marker styles, and custom axis labels.
- `R_km::Float64`: the reference radius of the central body in kilometers, used as a baseline for distance scaling.
"""
function _plot_orbital_core!(fig, df, opts, info, R_km)
    # 0. preparation of common variables
    t_raw = df.time
    # if it already has units, do not multiply. if it is a pure number, add "s".
    t_with_units = eltype(t_raw) <: Unitful.AbstractQuantity ? t_raw : t_raw .* u"s"
    
    t_plot = ustrip.(opts.time_unit, t_with_units)

    # 2. distance scale
    # if plot_in_radii is true, factor is 1/r. if false, convert km to the unit (e.g. au)
    # R_km has already been processed and guaranteed as a pure number (Float64) by the caller
    R_num = R_km 
    dist_factor = opts.plot_in_radii ? (1.0 / R_num) : ustrip(uconvert(opts.dist_unit, 1.0u"km"))
    dist_label  = opts.plot_in_radii ? "Radii" : "$(opts.dist_unit)"   
    
    # helper function to get the label (custom or default)
    get_label(key, default) = get(info.custom_labels, key, default)

    elementos_data = Dict(
    :a   => (get_label(:a, "Semimajor axis"), "a ($dist_label)", info.color_a, ustrip.(df.a_km) .* dist_factor),
    :e   => (get_label(:e, "Eccentricity"), "e", info.color_e, df.e),
    :i   => (get_label(:i, "Inclination"), "i (°)", info.color_i, ustrip.(df.i_deg)),
    :h   => (get_label(:h, "RAAN"), "h (°)", info.color_h, _apply_mod_360(ustrip.(df.h_deg))),
    :g   => (get_label(:g, "Periapsis argument"), "g (°)", info.color_g, _apply_mod_360(ustrip.(df.g_deg))),
    :alt => (get_label(:alt, "Periapsis altitude"), "alt_p ($dist_label)", info.color_alt, ustrip.(df.alt_peri_km) .* dist_factor),
    )

    # 2. element plots 
    if opts.graph_type in [:elem, :delta_elem]
        keys = opts.separate_element == :all_elements ? [:a,:e,:i,:h,:g,:alt] : [opts.separate_element]
        
        for (idx, k) in enumerate(keys)
            title, ylabel, col, data = elementos_data[k]
            if opts.graph_type == :delta_elem
                data = data .- data[1]
                title = "Δ " * title
                ylabel = "Δ " * ylabel
            end

            pos = length(keys) == 1 ? (1,1) : ((idx-1)÷2 + 1, (idx-1)%2 + 1)
            ax = Axis(fig[pos...], title = title, xlabel = "t ($(opts.time_unit))", ylabel = ylabel) 
            
            opts.use_scatter_plot ? scatter!(ax, t_plot, data, color=col, markersize=4) : lines!(ax, t_plot, data, color=col)
        end

    # 3. 3d orbit
    elseif opts.graph_type == :d3
        # applying factor to coordinates
        Xp, Yp, Zp = df.X_km .* dist_factor, df.Y_km .* dist_factor, df.Z_km .* dist_factor
        
        max_dist = maximum(sqrt.(Xp.^2 .+ Yp.^2 .+ Zp.^2))
        # the axis adjustment also needs to be scaled to not disappear in au
        plot_limit = max_dist + (opts.axis_adjustment * dist_factor)

        ax = Axis3(fig[1,1], title="3D Orbit ($dist_label)", aspect=:data)
        lines!(ax, Xp, Yp, Zp, color=info.orbit_color)        

        # projections
        if opts.XY_projection
            lines!(ax, Xp, Yp, fill(-plot_limit, length(Zp)), color=(:green, 0.4))
        end
        if opts.XZ_projection
            lines!(ax, Xp, fill(plot_limit, length(Yp)), Zp, color=(:red, 0.4))
        end
        if opts.YZ_projection
            lines!(ax, fill(plot_limit, length(Xp)), Yp, Zp, color=(:purple, 0.4))
        end

        # central sphere: if radii, radius is 1.0. if physical unit, r_km converted.
        _plot_sphere!(ax, opts.plot_in_radii ? 1.0 : R_km * dist_factor, info)
        limits!(ax, -plot_limit, plot_limit, -plot_limit, plot_limit, -plot_limit, plot_limit)

    # 4. 2d trajectory + energy error
    elseif opts.graph_type == :d2
        # xy, xz, yz with dist_factor and dist_label
        Axis(fig[1,1], title="Orbit XY", xlabel="X ($dist_label)", ylabel="Y ($dist_label)") |> 
            ax -> lines!(ax, df.X_km .* dist_factor, df.Y_km .* dist_factor)
        
        Axis(fig[1,2], title="Orbit XZ", xlabel="X ($dist_label)", ylabel="Z ($dist_label)") |> 
            ax -> lines!(ax, df.X_km .* dist_factor, df.Z_km .* dist_factor)
        
        Axis(fig[1,3], title="Orbit YZ", xlabel="Y ($dist_label)", ylabel="Z ($dist_label)") |> 
            ax -> lines!(ax, df.Y_km .* dist_factor, df.Z_km .* dist_factor)
    
    # 5. energy error
    elseif opts.graph_type == :energy_error
        # energy error: only the xlabel needs to change to the chosen time unit
        ax1 = Axis(fig[1,1], title="Keplerian Energy Relative Error", xlabel="t ($(opts.time_unit))", ylabel="Error")
        lines!(ax1, t_plot, ustrip.(df.energy_kep_rel_error))

        ax2 = Axis(fig[2,1], title="Perturbation Energy Relative Error", xlabel="t ($(opts.time_unit))", ylabel="Error")
        lines!(ax2, t_plot, ustrip.(df.energy_pert_rel_error))

        ax3 = Axis(fig[3,1], title="Total Energy Relative Error", xlabel="t ($(opts.time_unit))", ylabel="Error")
        lines!(ax3, t_plot, ustrip.(df.energy_tot_rel_error))

        ax4 = Axis(fig[4,1], title="Jabobi Constant Relative Error", xlabel="t ($(opts.time_unit))", ylabel="Error")
        lines!(ax4, t_plot, ustrip.(df.jacobi_rel_error))
    

    # 6. equinoctial coordinates
    elseif opts.graph_type == :equinoctial_coord
        ax1 = Axis(fig[1,1], title="e sin(ω) vs e cos(ω)", xlabel="e cos(ω)", ylabel="e sin(ω)")
        lines!(ax1, df.e .* cos.(deg2rad.(df.g_deg)), df.e .* sin.(deg2rad.(df.g_deg)))

        ax2 = Axis(fig[1,2], title="sin(i) cos(Ω) vs sin(i) sin(Ω)", xlabel="sin(i) cos(Ω)", ylabel="sin(i) sin(Ω)")
        lines!(ax2, sin.(deg2rad.(df.i_deg)) .* cos.(deg2rad.(df.h_deg)), sin.(deg2rad.(df.i_deg)) .* sin.(deg2rad.(df.h_deg)))
    end
end

"""
    plot_orbital_results(result::Types.SimulationResult, opts::Types.PlottingOptions, info::Types.GraphicInformation)

primary interface for generating orbital plots directly from a populated `SimulationResult` object.

it dynamically adjusts the visual representation based on the requested `opts.graph_type` (e.g., `:d3` for 3d trajectories, `:elem` for individual orbital elements, or `:delta_elem` for element variations relative to the initial epoch).

# arguments
- `result::Types.SimulationResult`: the object containing the processed simulation data.
- `opts::Types.PlottingOptions`: configuration structure detailing the plot type, unit conversions, and resolution.
- `info::Types.GraphicInformation`: aesthetic parameters including colors, opacities, and custom labels.

# returns
- `Figure`: a `makie.figure` object containing the generated plots.
"""
function plot_orbital_results(result::Types.SimulationResult, opts::Types.PlottingOptions, info::Types.GraphicInformation)
    fig = Figure(size=(opts.width_fig, opts.height_fig))

    # protection: checks explicitly for Unitful.Quantity because Quantities are also Numbers in Julia!
    R_val = result.parameters.R
    R_km = R_val isa Unitful.Quantity ? Float64(ustrip(u"km", R_val)) : Float64(R_val)
    
    # uses the dataframe that contains physical x, y, z coming from postprocessing
    # calls the core with the guaranteed numerical value
    _plot_orbital_core!(fig, result.elements, opts, info, R_km)
    
    return fig
end

"""
    plot_orbital_results(df::DataFrame, opts::Types.PlottingOptions, info::Types.GraphicInformation; R_km::Float64)

secondary interface for generating orbital plots directly from a raw `DataFrame`.
this is particularly useful when loading pre-computed or historically saved trajectories via the `DataHandling` module.

# arguments
- `df::DataFrame`: the tabular data containing the time series of state vectors or orbital elements.
- `opts::Types.PlottingOptions`: configuration structure detailing the plot type, unit conversions, and resolution.
- `info::Types.GraphicInformation`: aesthetic parameters including colors, opacities, and custom labels.

# keyword arguments
- `R_km::Float64`: the reference radius of the central body in kilometers, necessary for scaling 3d projections or normalizing distance axes.

# returns
- `Figure`: a `makie.figure` object containing the generated plots.
"""
function plot_orbital_results(df::DataFrame, opts::Types.PlottingOptions, info::Types.GraphicInformation; R_km::Float64)
   
    fig = Figure(size=(opts.width_fig, opts.height_fig)) 
    _plot_orbital_core!(fig, df, opts, info, R_km)
    return fig
end


"""
    plot_cr3bp_results(result::Types.SimulationResult, opts::Types.PlottingOptions, info::Types.GraphicInformation)

generates a comprehensive dashboard of plots for the circular restricted three-body problem (cr3bp).
it includes visual representations of the synodic and inertial reference frames, zero velocity curves (zvc), jacobi constant error, a poincare section, and distances to the primary bodies over time.

# arguments
- `result::Types.SimulationResult`: the data structure containing the computed trajectory elements, system parameters like the mass ratio mu, and poincare section events.
- `opts::Types.PlottingOptions`: configuration structure detailing the plot type, unit conversions, and resolution.
- `info::Types.GraphicInformation`: aesthetic parameters including colors, opacities, and custom labels.

# returns
- `Figure`: a `makie.figure` object containing the generated plots.
"""
function plot_cr3bp_results(result::Types.SimulationResult, opts::Types.PlottingOptions, info::Types.GraphicInformation)
    fig = Figure(size=(1600, 1000), fontsize=16)
    df = result.elements
    mu = result.parameters.mu

    # --- 1. Synodic Frame (Rotating) ---
    ax_syn = Axis(fig[1, 1], title="Synodic Frame", xlabel="x (dimensionless)", ylabel="y (dimensionless)", aspect=DataAspect())
    
    # Plot trajectory
    lines!(ax_syn, df.x, df.y, color=info.orbit_color, linewidth=1.5)
    
    # Plot primary bodies (m1 at -mu, m2 at 1-mu)
    scatter!(ax_syn, [-mu], [0.0], color=:orange, markersize=15, label="Primary 1")
    scatter!(ax_syn, [1.0 - mu], [0.0], color=:gray, markersize=8, label="Primary 2")
    axislegend(ax_syn, position=:rt)

    # --- 2. Inertial Frame ---
    # Rotation matrix application: X = x*cos(t) - y*sin(t) | Y = x*sin(t) + y*cos(t)
    X_in = df.x .* cos.(df.time) .- df.y .* sin.(df.time)
    Y_in = df.x .* sin.(df.time) .+ df.y .* cos.(df.time)

    ax_in = Axis(fig[1, 2], title="Inertial Frame", xlabel="X", ylabel="Y", aspect=DataAspect())
    lines!(ax_in, X_in, Y_in, color=info.orbit_color, linewidth=1.5)
    
    # Plot Barycenter
    scatter!(ax_in, [0.0], [0.0], color=:black, marker=:cross, markersize=15)

    # --- 3. Zero Velocity Curves (ZVC) ---
    ax_zvc = Axis(fig[1, 3], title="Zero Velocity Curves (ZVC)", xlabel="x", ylabel="y", aspect=DataAspect())
    
    # Calculate initial Jacobi constant to define the energy level boundary
    r1_0 = sqrt((df.x[1] + mu)^2 + df.y[1]^2 + df.z[1]^2)
    r2_0 = sqrt((df.x[1] - 1.0 + mu)^2 + df.y[1]^2 + df.z[1]^2)
    v2_0 = df.vx[1]^2 + df.vy[1]^2 + df.vz[1]^2
    C_J0 = (df.x[1]^2 + df.y[1]^2) + 2.0*(1.0 - mu)/r1_0 + 2.0*mu/r2_0 - v2_0

    # Create a larger grid to prevent the curve from cutting off at the edges
    grid_limit = 2.0
    x_grid = range(-grid_limit, grid_limit, length=500)
    y_grid = range(-grid_limit, grid_limit, length=500)
    
    # Evaluate pseudo-potential over the grid
    Z_pot = [ (x^2 + y^2 + 2.0*(1.0 - mu)/sqrt((x + mu)^2 + y^2) + 2.0*mu/sqrt((x - 1.0 + mu)^2 + y^2)) for x in x_grid, y in y_grid ]
    
    # Plot forbidden regions (where pseudo-potential < C_J0)
    # The classic look: gray for forbidden regions, white for allowed regions
    contourf!(ax_zvc, x_grid, y_grid, Z_pot, levels=[0.0, C_J0], colormap=[:lightgray, :transparent])
    contour!(ax_zvc, x_grid, y_grid, Z_pot, levels=[C_J0], color=:black, linewidth=2.0)
    
    # Overlay only the primary bodies (removed the orbit line as requested)
    scatter!(ax_zvc, [-mu], [0.0], color=:orange, markersize=15)
    scatter!(ax_zvc, [1.0 - mu], [0.0], color=:gray, markersize=8)
    
    # Set limits to match the new grid
    xlims!(ax_zvc, -grid_limit, grid_limit)
    ylims!(ax_zvc, -grid_limit, grid_limit)

    # --- 4. Jacobi Constant Error ---
    ax_jac = Axis(fig[2, 1], title="Jacobi Constant Error", xlabel="Time (t)", ylabel="dC_J / C_J(0)")
    lines!(ax_jac, df.time, df.jacobi_error, color=:red, linewidth=2.0)

    # --- 5. Poincare Section ---
    ax_poin = Axis(fig[2, 2], title="Poincare Section (y=0, v_y>0)", xlabel="x", ylabel="v_x")
    
    if !isnothing(result.poincare_e_g) && !isempty(result.poincare_e_g)
        scatter!(ax_poin, result.poincare_e_g, color=:black, markersize=6)
    else
        text!(ax_poin, 0.5, 0.5, text="No crossing detected", align=(:center, :center), space=:relative)
    end

    # --- 6. Distances to Primaries ---
    ax_dist = Axis(fig[2, 3], title="Distances to Primaries", xlabel="Time (t)", ylabel="Distance")
    
    # Calculate distances over time
    r1_vec = sqrt.((df.x .+ mu).^2 .+ df.y.^2 .+ df.z.^2)
    r2_vec = sqrt.((df.x .- 1.0 .+ mu).^2 .+ df.y.^2 .+ df.z.^2)
    
    lines!(ax_dist, df.time, r1_vec, color=:orange, label="r1 (Primary 1)", linewidth=1.5)
    lines!(ax_dist, df.time, r2_vec, color=:gray, label="r2 (Primary 2)", linewidth=1.5)
    axislegend(ax_dist, position=:rt)

    return fig
end


"""
    plot_phase_contours(e_vals, w_vals, P; kwargs...)

generates contour plots of an evaluated hamiltonian energy surface in the phase space defined by eccentricity and the argument of periapsis.

this function takes the pre-calculated energy matrix (`P`) from the `EvaluatetxtEquations` module and overlays constant energy level curves, essential for identifying 
stable and unstable equilibrium points (e.g., frozen orbits).

# arguments
- `e_vals::AbstractVector`: the grid values representing eccentricity (y-axis).
- `w_vals::AbstractVector`: the grid values representing the argument of periapsis (x-axis).
- `P::AbstractMatrix`: the 2d matrix containing the evaluated hamiltonian energy levels.

# keyword arguments
- `title::AbstractString`: the main title of the plot. defaults to "Retrato de Fase".
- `xlabel::AbstractString`: the label for the x-axis. defaults to "Argumento do Pericentro (ω) [rad]".
- `ylabel::AbstractString`: the label for the y-axis. defaults to "Eccentricity (e)".
- `levels::Union{Int, AbstractVector{<:Real}}`: the number of contour levels or specific values to draw. defaults to 30.
- `colormap::Symbol`: the colormap to use. defaults to `:viridis`.
- `linewidth::Real`: the width of the contour lines. defaults to `1.0`.
- `show_fig::Bool`: toggles whether the figure is displayed immediately. defaults to `true`.
- `save_path::Union{Nothing, AbstractString}`: path to export the figure. defaults to `nothing`.

# returns
- `Tuple`: returns `(fig, ax, cont)`, corresponding to the figure, axis, and contour objects respectively.
"""
function plot_phase_contours(
    e_vals::AbstractVector,      # y-axis
    w_vals::AbstractVector,      # x-axis
    P::AbstractMatrix;           # z matrix (energy) already calculated
    # aesthetics
    title::AbstractString = "Retrato de Fase",
    xlabel::AbstractString = "Argumento do Pericentro (ω) [rad]",
    ylabel::AbstractString = "Eccentricity (e)",
    levels::Union{Int,AbstractVector{<:Real}} = 30,
    colormap::Symbol = :viridis,
    linewidth::Real = 1.0,
    show_fig::Bool = true,
    save_path::Union{Nothing,AbstractString} = nothing
    )

    # check dimensions to avoid makie error
    # makie expects z with dimensions [x, y], that is [nw, ne]
    # if the matrix comes as [ne, nw] (which is the standard row x column), we need to transpose
    if size(P) == (length(e_vals), length(w_vals))
        P = permutedims(P)
    end

    fig = Figure(size=(1200, 1000))
    ax  = Axis(fig[1,1], title=title, xlabel=xlabel, ylabel=ylabel)

    # draw contour lines
    # note that now we pass p directly, without calculating phi() inside here
    cont = contour!(ax, w_vals, e_vals, P;
        levels = levels, 
        color = :black, # or use colormap=colormap if preferred colored
        labels = false, 
        linewidth = linewidth
    )

    # aesthetic visual grid
    hlines!(ax, collect(range(minimum(e_vals), maximum(e_vals); length=6)); color=(:gray, 0.25), linewidth=0.5)
    vlines!(ax, collect(range(minimum(w_vals), maximum(w_vals); length=6)); color=(:gray, 0.25), linewidth=0.5)

    if save_path !== nothing
        mkpath(dirname(save_path))
        save(save_path, fig)
    end
    if show_fig
        display(fig)
    end
    return fig, ax, cont
end

"""
    plot_dynamic_map(res::Types.MappedRoots; kwargs...)

high-level interface for visualizing the output of multidimensional numerical root-finding or analytical map evaluations.

this function automatically extracts the solved variable (`:a`, `:e`, or `:i`) from the `MappedRoots` structure and correctly assigns the axes and colorbar labels 
before passing the data to the generic heatmap generator.
It reconstructs the `z_matrix` by mapping the flattened CSV columns back to a 2D grid before rendering.

# arguments
- `csv_path::String`: The file path to the saved CSV data.
- `solved_variable::Symbol`: The variable that was isolated (`:a`, `:e`, or `:i`). Defaults to `:e`.
- `kwargs...`: optional arguments passed to `plot_generic_heatmap`.

# returns
- `Tuple`: returns `(fig, ax, hm)`, corresponding to the figure, axis, and heatmap objects.
"""
function plot_dynamic_map(df::DataFrame; solved_variable::Symbol=:e, kwargs...)
    
    # defines the axes based on the resolved variable (the z-axis)
    if solved_variable == :i
        x_col, y_col, z_col = :a, :e, :i
        labels = (xlabel="Semimajor axis (km)", ylabel="Eccentricity (e)", clabel="Inclination (deg)")
    elseif solved_variable == :e
        x_col, y_col, z_col = :a, :i, :e
        labels = (xlabel="Semimajor axis (km)", ylabel="Inclination (deg)", clabel="Eccentricity")
    elseif solved_variable == :a
        x_col, y_col, z_col = :e, :i, :a
        labels = (xlabel="Eccentricity (e)", ylabel="Inclination (deg)", clabel="Semimajor axis (km)")
    else
        error("Unsupported variable. Use :a, :e, ou :i.")
    end

    # extracts the unique values ​​to reconstruct the grid (x and y)
    x_vals = sort(unique(df[!, x_col]))
    y_vals = sort(unique(df[!, y_col]))

    # reconstructs the z-matrix (initially fills it with NaN)
    z_matrix = fill(NaN, length(x_vals), length(y_vals))

    # creates quick-search dictionaries for the indexes.
    x_dict = Dict(val => idx for (idx, val) in enumerate(x_vals))
    y_dict = Dict(val => idx for (idx, val) in enumerate(y_vals))

    # fill the matrix with the values ​​from the CSV.
    for row in eachrow(df)
        i = x_dict[row[x_col]]
        j = y_dict[row[y_col]]
        z_matrix[i, j] = row[z_col]
    end

    
    return plot_generic_heatmap(
        x_vals, 
        y_vals, 
        z_matrix;
        xlabel = labels.xlabel,
        ylabel = labels.ylabel,
        colorbar_label = labels.clabel,
        title = "Dynamic Map: $(labels.clabel)",
        kwargs... # passes all aesthetic parameters and save_path
    )
end


"""
    plot_generic_heatmap(x_vals, y_vals, z_matrix; kwargs...)

core function for generating highly customizable 2d heatmaps.

it supports clamping values to specific ranges, transposing axes, and handling `NaN` regions gracefully by rendering them transparent (useful for regions where the numerical root-finder failed to converge).

# arguments
- `x_vals::AbstractVector`: the grid values for the x-axis.
- `y_vals::AbstractVector`: the grid values for the y-axis.
- `z_matrix::AbstractMatrix`: the 2d matrix of values determining the color intensity.

# keyword arguments
- `title`, `xlabel`, `ylabel`, `colorbar_label`: text annotations for the plot.
- `colormap::Symbol`: the color scale to use. defaults to `:viridis`.
- `clamp_min`, `clamp_max`: optional limits to clamp the color range, useful for mitigating extreme outliers.
- `manual_color_range`: tuple defining strict limits for the colorbar (e.g., `(0, 90)`). defaults to `makie.automatic`.
- `transpose::Bool`: toggles matrix transposition to match `[x, y]` dimensions. defaults to `false`.
- `show_fig::Bool`: toggles figure display. defaults to `true`.
- `save_path`: optional file path to export the heatmap.

# returns
- `Tuple`: returns `(fig, ax, hm)`, corresponding to the figure, axis, and heatmap objects.
"""
function plot_generic_heatmap(
    x_vals::AbstractVector,
    y_vals::AbstractVector,
    z_matrix::AbstractMatrix;
    title = "Dynamic Map",
    xlabel = "X axis",
    ylabel = "Y axis",
    colorbar_label = "Z Value",
    colormap = :viridis,
    clamp_min = nothing,
    clamp_max = nothing,
    manual_color_range = nothing,
    transpose = false,
    show_fig = true,
    save_path = nothing
    )

    z_plot = copy(z_matrix)
    if transpose; z_plot = permutedims(z_plot); end

    # clamp automatically ignores nans in julia
    if clamp_min !== nothing
        z_plot .= max.(z_plot, clamp_min)
    end
    if clamp_max !== nothing
        z_plot .= min.(z_plot, clamp_max)
    end

    # defines the color range
    cr = manual_color_range !== nothing ? manual_color_range : Makie.automatic

    fig = Figure(size=(1000, 800), fontsize=20)
    ax = Axis(fig[1,1], xlabel=xlabel, ylabel=ylabel, title=title)

    # nan_color=:transparent ensures that where the numerical_root_mapper failed, the graph remains empty
    hm = heatmap!(ax, x_vals, y_vals, z_plot; 
                  colormap=colormap, colorrange=cr, nan_color=:transparent)

    Colorbar(fig[1,2], hm, label=colorbar_label)

    if !isnothing(save_path)
        save(save_path, fig)
    end
    
    show_fig && display(fig)
    return fig, ax, hm
end


# ------------------------------------------------------------------------------
# n-body specific plotting routines
# ------------------------------------------------------------------------------

"""
    _rotate_to_synodic(x_target, y_target, x_ref, y_ref)

helper function to apply 2d rotation to the synodic frame.

# arguments
- `x_target`, `y_target`: coordinates to be rotated.
- `x_ref`, `y_ref`: reference body coordinates to determine the rotation angle.

# returns
- `Tuple`: returns `(x_syn, y_syn)`, the rotated coordinates in the synodic frame.
"""
function _rotate_to_synodic(x_target, y_target, x_ref, y_ref)
    # calculate the angle of the reference body relative to the center
    theta = atan(y_ref, x_ref)
    
    # apply rotation matrix to the target coordinates
    x_syn = x_target * cos(theta) + y_target * sin(theta)
    y_syn = -x_target * sin(theta) + y_target * cos(theta)
    
    return x_syn, y_syn
end

# ------------------------------------------------------------------------------
# n-body specific plotting routines (normalized)
# ------------------------------------------------------------------------------

"""
    plot_nbody_2d(sol, bodies; center_idx=2, ref_idx=3, target_idx=3, plot_reference=false, zoom_limit=nothing, points=5000)

generates 2d inertial and synodic plots using normalized data (du). 
uses the continuous ode interpolant to guarantee smooth trajectories regardless of solver step size.

# arguments
- `sol`: the raw odesolution object (containing states in du).
- `bodies`: vector of nbodyparticle structures.

# keyword arguments
- `center_idx::Int`: index of the central body. defaults to 2.
- `ref_idx::Int`: index of the reference body for the synodic frame. defaults to 3.
- `target_idx::Int`: index of the target body to plot. defaults to 3.
- `plot_reference::Bool`: whether to plot the reference body trajectory. defaults to false.
- `zoom_limit`: limit in du (e.g., 2.0 or 5.0). if nothing, the plot uses auto-limit. defaults to nothing.
- `points::Int`: number of points to interpolate for smooth plotting. defaults to 5000.

# returns
- `Figure`: a `makie.figure` object containing the plots.
"""
function plot_nbody_2d(sol, bodies; center_idx=2, ref_idx=3, target_idx=3, plot_reference=false, zoom_limit=nothing, points=5000)
    fig = Figure(size = (1600, 700), fontsize = 18)
    
    ax_inertial = Axis(fig[1, 1], 
        title = "inertial frame (centered on body $(center_idx))", 
        xlabel = "x (du)", ylabel = "y (du)",
        aspect = DataAspect()
    )
    
    ax_synodic = Axis(fig[1, 2], 
        title = "synodic frame (rotating with body $(ref_idx))", 
        xlabel = "x_syn (du)", ylabel = "y_syn (du)",
        aspect = DataAspect()
    )

    idx_center = 6 * (center_idx - 1)
    idx_ref    = 6 * (ref_idx - 1)
    idx_target = 6 * (target_idx - 1)

    x_in_target, y_in_target = Float64[], Float64[]
    x_syn_target, y_syn_target = Float64[], Float64[]
    x_in_ref, y_in_ref = Float64[], Float64[]

    # create a dense, smooth time array based on the solver's start and end times
    t_smooth = range(sol.t[1], sol.t[end], length=points)

    for t in t_smooth
        # use julia's native ode interpolant to evaluate the state at exact time t
        u = sol(t) 
        
        xc, yc = u[idx_center + 1], u[idx_center + 2]
        xr, yr = u[idx_ref + 1],    u[idx_ref + 2]
        xt, yt = u[idx_target + 1], u[idx_target + 2]

        x_ref_c, y_ref_c = xr - xc, yr - yc
        x_tgt_c, y_tgt_c = xt - xc, yt - yc
        
        push!(x_in_ref, x_ref_c)
        push!(y_in_ref, y_ref_c)
        push!(x_in_target, x_tgt_c)
        push!(y_in_target, y_tgt_c)

        xs, ys = _rotate_to_synodic(x_tgt_c, y_tgt_c, x_ref_c, y_ref_c)
        push!(x_syn_target, xs)
        push!(y_syn_target, ys)
    end

    # plot trajectories
    if plot_reference
        lines!(ax_inertial, x_in_ref, y_in_ref, color = :blue, label = string(bodies[ref_idx].name))
    end
    lines!(ax_inertial, x_in_target, y_in_target, color = :red, label = string(bodies[target_idx].name))
    scatter!(ax_inertial, [0.0], [0.0], color = :black, markersize = 12, label = string(bodies[center_idx].name))
    
    if plot_reference
        r_mean = sum(sqrt.(x_in_ref.^2 .+ y_in_ref.^2)) / length(x_in_ref)
        scatter!(ax_synodic, [r_mean], [0.0], color = :blue, markersize = 12, label = string(bodies[ref_idx].name))
    end
    lines!(ax_synodic, x_syn_target, y_syn_target, color = :red, label = string(bodies[target_idx].name))
    scatter!(ax_synodic, [0.0], [0.0], color = :black, markersize = 12, label = string(bodies[center_idx].name))

    # apply zoom limits in normalized units
    if !isnothing(zoom_limit)
        xlims!(ax_inertial, -zoom_limit, zoom_limit)
        ylims!(ax_inertial, -zoom_limit, zoom_limit)
        xlims!(ax_synodic, -zoom_limit, zoom_limit)
        ylims!(ax_synodic, -zoom_limit, zoom_limit)
    end

    return fig
end

end # end of module

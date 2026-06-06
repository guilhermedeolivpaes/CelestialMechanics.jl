# src/Analysis/EvaluatetxtEquations.jl

"""
    EvaluatetxtEquations

Module responsible for the numerical evaluation and root-finding of analytical 
equations derived from the symbolic processor.

It provides tools for mapping equilibrium conditions (such as frozen orbits) 
across parameter grids, applying Lie series transformations to recover osculating 
elements from mean variables, and generating energy surfaces for Hamiltonian phase portraits.

"""
module EvaluatetxtEquations

using Roots
using RuntimeGeneratedFunctions
using ..Types
using ..Coordinates

export numerical_root_mapper, evaluate_analytical_map, compute_osc_corrections, generate_phase_portrait_data

"""
    numerical_root_mapper(params, grid, eq_func; solve_for=:i, prograde=false)

Performs a numerical root-finding sweep across a 2D parameter grid to find equilibrium 
conditions for a specific orbital element.

Uses the `Roots.jl` package (specifically the Bisection method via bracketing) to solve 
the provided equation `eq_func = 0`. It filters the results to ensure they represent 
physically valid orbits (e.g., periapsis radius strictly greater than the central body radius).

# Arguments
- `params::Types.PhysicalParams`: Physical constants and perturbation parameters of the system.
- `grid::Types.GridParams`: Definition of the parameter grid boundaries and resolution.
- `eq_func::Function`: The compiled analytical equation whose root is to be found.

# Keyword Arguments
- `solve_for::Symbol`: The orbital element to solve for. Supports `:i` (inclination), `:e` (eccentricity), or `:a` (semi-major axis). Defaults to `:i`.
- `prograde::Bool`: When solving for inclination, restricts the search bracket to prograde orbits (0 to 90 degrees) if true, or retrograde (90 to 180 degrees) if false. Defaults to `false`.

# Returns
- `Types.MappedRoots`: A structure containing the grid axes, the matrix of found roots, and a filtered list of valid `(a, e, i)` pairs.
"""
function numerical_root_mapper(params::Types.PhysicalParams, grid::Types.GridParams, eq_func::Function; 
                            solve_for::Symbol=:i, prograde::Bool=false)
    
    # 1. axis configuration (maintaining grid logic)
    if solve_for == :i
        x_vals, y_vals = range(grid.a_min, grid.a_max, length=grid.num_points), range(grid.e_min, grid.e_max, length=grid.num_points)
        bracket = prograde ? (deg2rad(0.1), deg2rad(89.9)) : (deg2rad(90.1), deg2rad(179.9))
    elseif solve_for == :e
        x_vals, y_vals = range(grid.a_min, grid.a_max, length=grid.num_points), range(grid.i_min, grid.i_max, length=grid.num_points)
        bracket = (grid.e_min + 1e-6, grid.e_max - 1e-6)
    else # :a
        x_vals, y_vals = range(grid.e_min, grid.e_max, length=grid.num_points), range(grid.i_min, grid.i_max, length=grid.num_points)
        bracket = (grid.a_min, grid.a_max)
    end

    z_matrix = Matrix{Float64}(undef, length(x_vals), length(y_vals))
    filtered_pairs = Tuple{Float64, Float64, Float64}[]

    @inbounds for (ix, x) in enumerate(x_vals)
        L_const = (solve_for != :a) ? sqrt(params.mu * x) : 0.0
        
        for (jy, y) in enumerate(y_vals)
            
            function objective(var)
                if solve_for == :i
                    a, e, i = x, y, var
                    L, G = L_const, L_const * sqrt(max(0.0, 1.0 - e^2))
                    H = G * cos(i)
                elseif solve_for == :e
                    a, e, i = x, var, deg2rad(y)
                    L = L_const
                    G = L * sqrt(max(0.0, 1.0 - e^2))
                    H = G * cos(i)
                else # :a
                    a, e, i = var, x, deg2rad(y)
                    L = sqrt(params.mu * a)
                    G = L * sqrt(max(0.0, 1.0 - e^2))
                    H = G * cos(i)
                end
                # calls with 7 arguments as in the old code
                #return eq_func(a, e, i, L, G, H, params)
                return Base.invokelatest(eq_func, a, e, i, L, G, H, params)
                
            end

            try
                if objective(bracket[1]) * objective(bracket[2]) < 0
                    root = find_zero(objective, bracket)
                    z_matrix[ix, jy] = (solve_for == :i) ? rad2deg(root) : root
                    
                    # extraction for the filter (always a, e, i_deg)
                    a_f, e_f, i_f = (solve_for == :i) ? (x, y, rad2deg(root)) : 
                                   (solve_for == :e) ? (x, root, y) : (root, x, y)

                    if a_f * (1.0 - e_f) > params.R
                        push!(filtered_pairs, (Float64(a_f), Float64(e_f), Float64(i_f)))
                    end
                else
                    z_matrix[ix, jy] = NaN
                end
            catch err # error log
                @warn "Solver failed at x=$x, y=$y: $err"
                z_matrix[ix, jy] = NaN
            end
        end
    end
    return Types.MappedRoots(collect(x_vals), collect(y_vals), z_matrix, filtered_pairs, solve_for)
end

"""
    evaluate_analytical_map(params, grid, analytical_func; solve_for=:i)

Evaluates an explicit analytical solution over a 2D parameter grid.

Unlike the root mapper, this function directly computes the value of an explicitly solved 
variable (e.g., evaluating an analytical formula for the cosine of the equilibrium inclination).

# Arguments
- `params::Types.PhysicalParams`: Physical constants and perturbation parameters.
- `grid::Types.GridParams`: Definition of the parameter grid boundaries and resolution.
- `analytical_func::Function`: The compiled explicit analytical function.

# Keyword Arguments
- `solve_for::Symbol`: The variable being evaluated. Currently, only `:i` (inclination) is supported, assuming the function returns `cos(i)`. Defaults to `:i`.

# Returns
- `Types.MappedRoots`: A structure containing the evaluated map matrix and filtered valid parameter pairs.
"""
function evaluate_analytical_map(params::Types.PhysicalParams, grid::Types.GridParams, analytical_func::Function; 
                                solve_for::Symbol=:i)
    
    # define axes based on what we are solving
    if solve_for == :i
        x_vals = range(grid.a_min, grid.a_max, length=grid.num_points)
        y_vals = range(grid.e_min, grid.e_max, length=grid.num_points)
    else
        error("Currently, only analytical evaluation of :i is supported.")
    end

    z_matrix = Matrix{Float64}(undef, length(x_vals), length(y_vals))
    filtered_pairs = Tuple{Float64, Float64, Float64}[]

    println(" [AnalyticalMap] Evaluating explicit solution for $solve_for...")

    for (ix, x) in enumerate(x_vals)
        for (jy, y) in enumerate(y_vals)
            
            # here you call the maxima function. 
            # if it returns the value of cos(i), you apply acos.
            # use (passing 0.0 as placeholder for 'i'):
            res_raw = analytical_func(x, y, 0.0, params)
            
            # example: if solve_for == :i, assume res_raw is cos(i)
            # we apply clamp to avoid numerical errors outside [-1, 1]
            val = rad2deg(acos(res_raw))
            #val = rad2deg(acos(clamp(res_raw, -1.0, 1.0)))
            #val = res_raw # to see results in cos_i [-1,1]
            z_matrix[ix, jy] = val

            # collision filter (same logic as root_mapper)
            if x * (1.0 - y) > params.R
                push!(filtered_pairs, (Float64(x), Float64(y), Float64(val)))
            end
        end
    end

    return Types.MappedRoots(collect(x_vals), collect(y_vals), z_matrix, filtered_pairs, solve_for)
end

"""
    compute_osc_corrections(a_mean, e_mean, i_mean_deg, params, transf_list; kwargs...)

Applies a sequence of analytical transformations to convert mean orbital elements 
into osculating elements.

This function evaluates the transformation equations (usually derived from the Lie 
generating function) and iteratively updates the Delaunay momenta and geometric 
elements. This guarantees that each subsequent transformation step evaluates the 
geometry accurately.

For near-circular orbits (e < e_threshold), the osculating corrections can push 
G > L, making e imaginary. In such cases, the function falls back to the mean 
elements, which is justified since the mean-to-osculating difference scales as 
O(J₂·e) and is negligible for small eccentricities.

# Arguments
- `a_mean::Float64`: Mean semi-major axis.
- `e_mean::Float64`: Mean eccentricity.
- `i_mean_deg::Float64`: Mean inclination in degrees.
- `params::Types.PhysicalParams`: Physical parameters of the system.
- `transf_list::Vector{<:Function}`: An ordered list of compiled transformation functions.

# Keyword Arguments
- `l_mean_deg::Float64`: Mean anomaly in degrees. Defaults to 0.0.
- `g_mean_deg::Float64`: Mean argument of periapsis in degrees. Defaults to 270.0.
- `h_mean_deg::Float64`: Mean RAAN in degrees. Defaults to 90.0.
- `e_threshold::Float64`: Below this eccentricity, fallback to mean elements if 
  the correction is ill-conditioned. Defaults to 0.01.

# Returns
- `NamedTuple`: A tuple `(a, e, i, h, g, l)` containing the fully corrected 
  osculating elements (angles in degrees).
"""
function compute_osc_corrections(
    a_mean::Float64, 
    e_mean::Float64, 
    i_mean_deg::Float64, 
    params::Types.PhysicalParams, 
    transf_list::Vector{<:Function};
    l_mean_deg::Float64 = 0.0,
    g_mean_deg::Float64 = 270.0,
    h_mean_deg::Float64 = 90.0,
    e_threshold::Float64 = 0.01
)
    # --- initial state preparation ---
    i_rad = deg2rad(i_mean_deg)
    l_rad = deg2rad(l_mean_deg)
    g_rad = deg2rad(g_mean_deg)
    h_rad = deg2rad(h_mean_deg)

    l_val = sqrt(params.mu * a_mean)
    g_val = l_val * sqrt(max(0.0, 1.0 - e_mean^2))
    h_val = g_val * cos(i_rad)

    # u_current stores [L, G, H, l, g, h]
    u_current = [l_val, g_val, h_val, l_rad, g_rad, h_rad]

    # current geometric elements
    a_curr = a_mean
    e_curr = e_mean
    i_curr = i_rad

    f_rad = Coordinates.mean_to_true_anomaly(l_rad, e_mean)
    r_val = a_mean * (1.0 - e_mean^2) / (1.0 + e_mean * cos(f_rad))

    # --- transformation application loop ---
    for (idx, func) in enumerate(transf_list)
        # recalculate f and r from the current (updated) geometry
        f_rad = Coordinates.mean_to_true_anomaly(u_current[4], e_curr)
        r_val = a_curr * (1.0 - e_curr^2) / (1.0 + e_curr * cos(f_rad))

        # 1. apply the current transformation
        delta = Base.invokelatest(
            func,
            a_curr, e_curr, i_curr, 
            u_current[1], u_current[2], u_current[3], 
            params; 
            l = u_current[4], 
            g = u_current[5], 
            h = u_current[6],
            f = f_rad,   
            r = r_val
        )

        # 2. update the state (add the delta)
        u_current .+= delta

        # 3. recalculate geometric elements for the next transformation
        a_curr = u_current[1]^2 / params.mu
        
        e_sq = 1.0 - (u_current[2] / u_current[1])^2

        # --- singularity guard for near-circular orbits ---
        if e_sq < 0.0
            if e_mean < e_threshold
                @warn "Osculating correction ill-conditioned (G > L) at " *
                      "a=$(round(a_mean, digits=2)), e=$(round(e_mean, sigdigits=4)), " *
                      "i=$(round(i_mean_deg, digits=2))°. " *
                      "Falling back to mean elements (correction ~ O(J₂·e) ≈ $(round(params.j2 * e_mean, sigdigits=2)))."
                return (
                    a = a_mean, 
                    e = e_mean, 
                    i = i_mean_deg, 
                    h = h_mean_deg, 
                    g = g_mean_deg, 
                    l = l_mean_deg
                )
            else
                @error "Osculating correction produced G > L for non-small eccentricity " *
                       "at a=$(round(a_mean, digits=2)), e=$(round(e_mean, sigdigits=4)), " *
                       "i=$(round(i_mean_deg, digits=2))°. This may indicate a problem " *
                       "in the generating function."
                return (a=NaN, e=NaN, i=NaN, h=NaN, g=NaN, l=NaN)
            end
        end

        e_curr = sqrt(e_sq)
        
        cos_i = clamp(u_current[3] / u_current[2], -1.0, 1.0)
        i_curr = acos(cos_i)
    end

    # --- return final results in keplerian ---
    return (
        a = a_curr, 
        e = e_curr, 
        i = rad2deg(i_curr), 
        h = rad2deg(mod2pi(u_current[6])), 
        g = rad2deg(mod2pi(u_current[5])), 
        l = rad2deg(mod2pi(u_current[4]))
    )
end


"""
    generate_phase_portrait_data(params, a_fixed, i_fixed_deg, grid_e, grid_w_deg, ham_func)

Evaluates a Hamiltonian function over a grid of eccentricity and argument of periapsis to 
generate energy surfaces for phase portraits.

# Arguments
- `params::Types.PhysicalParams`: Physical parameters of the system.
- `a_fixed::Float64`: The constant semi-major axis for the phase space slice.
- `i_fixed_deg::Float64`: The constant inclination in degrees.
- `grid_e::Vector{Float64}`: A vector of eccentricity values defining the grid's Y-axis.
- `grid_w_deg::Vector{Float64}`: A vector of argument of periapsis values (degrees) defining the grid's X-axis.
- `ham_func::Function`: The compiled analytical Hamiltonian function.

# Returns
- `Matrix{Float64}`: A 2D matrix of evaluated energy levels corresponding to the `(e, omega)` grid, suitable for contour plotting.
"""
function generate_phase_portrait_data(
    params::Types.PhysicalParams, 
    a_fixed::Float64, 
    i_fixed_deg::Float64, 
    grid_e::Vector{Float64}, 
    grid_w_deg::Vector{Float64}, 
    ham_func::Function
    )
    
    i_rad = deg2rad(i_fixed_deg)
    
    # z matrix (energy)
    # dimensions: (num_e, num_w) to match the heatmap logic
    Z_val = Matrix{Float64}(undef, length(grid_e), length(grid_w_deg))
    
    println(" [EvaluatetxtEquations] Calculating Hamiltonian (a=$(a_fixed) km, i=$(i_fixed_deg) deg)...")

    @inbounds for (iw, w_deg) in enumerate(grid_w_deg)
        w_rad = deg2rad(w_deg)
        
        for (ie, e) in enumerate(grid_e)
            
            # recalculate delaunay momenta for the current point
            L_val = sqrt(params.mu * a_fixed)
            G_val = L_val * sqrt(max(0.0, 1.0 - e^2))
            H_val = G_val * cos(i_rad)
            
            # call generic function
            # important: we pass w_rad to the 'g' argument (omega)
            val = ham_func(
                a_fixed, e, i_rad, L_val, G_val, H_val, params; 
                g=w_rad,
                h=0.0,
                t=0.0 # g in maxima = omega
            )
            
            Z_val[ie, iw] = val
        end
    end
    
    return Z_val
end

end # end module

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
using NLsolve
using ..Types
using ..Coordinates

export numerical_root_mapper, evaluate_analytical_map, compute_osc_corrections, generate_phase_portrait_data
export numerical_2d_system_solver, numerical_4d_system_solver

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
                return eq_func(a, e, i, L, G, H, params)                
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
                @debug "Solver failed at x=$x, y=$y: $err"
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
        g_mean_deg::Float64 = 90.0,
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
                e_curr = e_mean
                u_current[2] = u_current[1] * sqrt(max(0.0, 1.0 - e_mean^2))
                # restore the mean argument of periapsis to prevent numerical divergence
                # for near circular orbits, the inverse Lie transformation causes the 
                # angle correction to explode due to the zero eccentricity singularity
                # anchoring it back to the secular equilibrium ensures a safe and stable 
                # conversion to cartesian coordinates for the numerical integrator
                u_current[5] = deg2rad(g_mean_deg)
            else
                @error "Osculating correction produced G > L for non-small eccentricity..."
                return (a=NaN, e=NaN, i=NaN, h=NaN, g=NaN, l=NaN)
            end
        else
            # it only calculates the square root if e_sq is a positive real number!
            e_curr = sqrt(e_sq)
        end
        
        cos_i = clamp(u_current[3] / u_current[2], -1.0, 1.0)
        i_curr = acos(cos_i)

        # --- singularity guard for inclination in near-circular orbits ---
        # for near-circular orbits, the inverse Lie transformation can corrupt
        # the H momentum due to division by e in the generating function,
        # causing the osculating inclination to diverge from the mean value.
        # restoring H from the mean inclination is safe since the correction
        # scales as O(J2*e) and is negligible for small eccentricities
        if e_mean < e_threshold && abs(i_curr - i_rad) > deg2rad(5.0)
            # 5 degrees is a conservative safety margin; for eccentricities below
            # e_threshold the legitimate osculating correction is O(J2*e),
            # so any deviation beyond a fraction of a degree is numerical noise
            i_curr = i_rad
            u_current[3] = u_current[2] * cos(i_rad)
        end

        # --- singularity guard for near-equatorial orbits ---
        if abs(i_curr) < deg2rad(1.0) || abs(i_curr - π) < deg2rad(1.0)
            u_current[6] = deg2rad(h_mean_deg)
        end
        
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

"""
    numerical_2d_system_solver(params, grid_a, eq_func; guess_e=0.1, guess_i_deg=90.0)

Solves a system of 2 coupled equations (e.g., freezing conditions for g and Phi) 
to find the equilibrium eccentricity (e) and inclination (i) along a grid of 
semi-major axes (a).

Uses the NLsolve package to perform a multidimensional Newton-Raphson root 
finding. It filters the results to ensure they represent physically valid orbits.

# Arguments
- `params::Types.PhysicalParams`: Physical constants and perturbation parameters.
- `grid_a::AbstractVector`: A vector of semi-major axis values (a) defining the grid.
- `eq_func::Function`: The compiled analytical equation function returning a tuple of residuals.

# Keyword Arguments
- `guess_e::Float64`: Initial guess for the eccentricity. Defaults to 0.1.
- `guess_i_deg::Float64`: Initial guess for the inclination in degrees. Defaults to 90.0.

# Returns
- `Vector{Tuple{Float64, Float64, Float64}}`: A list of valid `(a, e, i_deg)` equilibrium pairs.
"""
function numerical_2d_system_solver(params::Types.PhysicalParams, grid_a::AbstractVector, eq_func::Function; 
                                 guess_e=0.1, guess_i_deg=90.0)
    
    # prepare the results array: [a, e, i_deg]
    results = Tuple{Float64, Float64, Float64}[]
    
    println(" [SystemSolver] Solving 2D system for $(length(grid_a)) values of a...")

    for a in grid_a
        # mutating function required by NLsolve f!(F, x) = 0
        # x[1] = eccentricity (e)
        # x[2] = inclination in radians (i)
        
        function f!(F, x)
            e_val = x[1]
            i_rad = x[2]
            
            # If the solver attempts an absurd eccentricity, we return a huge 
            # residual to force it back into the [0, 1) domain
            if e_val >= 0.999 || e_val < 0.0
                F[1] = 1e10 * (1.0 + abs(e_val))
                F[2] = 1e10 * (1.0 + abs(e_val))
                return
            end
            
            # reconstruct delaunay momenta required by the equation
            L_val = sqrt(params.mu * a)
            G_val = L_val * sqrt(max(0.0, 1.0 - e_val^2))
            H_val = G_val * cos(i_rad)
            
            # call the compiled maxima function
            res = eq_func(a, e_val, i_rad, L_val, G_val, H_val, params)
            
            # assign residuals to the F vector
            F[1] = res[1] # eq_Phi_frozen
            F[2] = res[2] # eq_g_frozen
        end
        
        # initial guess
        u0 = [guess_e, deg2rad(guess_i_deg)]
        
        try
            # solve via newton raphson / trust region
            sol = nlsolve(f!, u0, ftol=1e-10)
            
            if converged(sol)
                e_root = sol.zero[1]
                i_root_deg = rad2deg(sol.zero[2])
                
                # physical collision filter (periapsis > body radius)
                if a * (1.0 - e_root) > params.R && 0.0 <= e_root < 1.0
                    push!(results, (Float64(a), Float64(e_root), Float64(i_root_deg)))
                end
            else
                @debug "NLsolve failed to converge for a = $a"
            end
        catch err
            @debug "Solver error for a = $a: $err"
        end
    end
    
    return results
end

"""
    numerical_4d_system_solver(params, a, eq_func; guesses, ftol=1e-10, dedup_tol=1e-6)

solves the full 4d secular system `[dgdt, dhdt, dGdt, dHdt] = 0` for a fixed
semi-major axis `a`, searching for frozen equilibria whose angles `(g, h)` are
not constrained to the classical freezing condition `g = h = pi/2`.

the solar argument of periapsis `g_sun` is read from `params` through the
compiled function (it is one of the fields in `vars_from_p`), so nothing extra
is passed here. the unknown is the reduced state `x = [g, h, G, H]`; the action
`L = sqrt(mu*a)` is constant in the secular hamiltonian. from each converged
equilibrium the keplerian `(e, i)` are recovered from the delaunay momenta
`(G, H)`. multiple initial guesses are swept so newton can land on distinct
equilibria, including asymmetric ones far from `pi/2`; distinct roots are
deduplicated.

# arguments
- `params::Types.PhysicalParams`: physical constants and perturbation parameters
  (must already carry the desired `g_sun`).
- `a::Real`: fixed semi-major axis for this solve.
- `eq_func::Function`: the compiled 4d frozen system (from `load_equation_function`),
  returning the tuple `(dgdt, dhdt, dGdt, dHdt)`.

# keyword arguments
- `guesses::Vector`: list of `(g0, h0, e0, i0_deg)` initial guesses. `g0, h0` in
  radians define the angle start (put them away from pi/2 to target asymmetric
  equilibria); `e0, i0_deg` seed the momenta `G, H`.
- `ftol::Float64`: newton convergence tolerance. defaults to 1e-10.
- `dedup_tol::Float64`: tolerance to treat two equilibria as identical.
- `resid_tol::Float64`: max residual norm accepted; converged points whose
  residual exceeds this are rejected as false zeros. defaults to 1e-8.

# returns
- `Vector{NamedTuple}`: distinct equilibria, each
  `(a, e, i_deg, g_deg, h_deg, G, H)`.
"""
function numerical_4d_system_solver(
        params::Types.PhysicalParams,
        a::Real,
        eq_func::Function;
        guesses::Vector = [(pi/2, pi/2, 0.1, 90.0)],
        ftol::Float64 = 1e-10,
        dedup_tol::Float64 = 1e-6,
        resid_tol::Float64 = 1e-8
    )

    # --- constant action from the fixed semi-major axis ---
    L_val = sqrt(params.mu * a)

    # --- storage for distinct equilibria ---
    solutions = NamedTuple[]

    for (g0, h0, e0, i0_deg) in guesses

        # mutating residual required by NLsolve: f!(F, x) = 0
        # x = [g, h, G, H]
        function f!(F, x)
            g_val = x[1]
            h_val = x[2]
            G_val = x[3]
            H_val = x[4]

            # domain guard: G must stay in (0, L], H in [-G, G]
            if G_val <= 0.0 || G_val > L_val || abs(H_val) > G_val
                F[1] = 1e10; F[2] = 1e10; F[3] = 1e10; F[4] = 1e10
                return
            end

            # recover geometry from the momenta
            e_val = sqrt(max(0.0, 1.0 - (G_val / L_val)^2))
            cos_i = clamp(H_val / G_val, -1.0, 1.0)
            i_val = acos(cos_i)

            # call the compiled 4d system; angles g, h go through keywords
            res = eq_func(a, e_val, i_val, L_val, G_val, H_val, params;
                          g = g_val, h = h_val)

            F[1] = res[1]  # dgdt
            F[2] = res[2]  # dhdt
            F[3] = res[3]  # dGdt
            F[4] = res[4]  # dHdt
        end

        # initial guess: angles from the sweep, momenta from (e0, i0)
        G0 = L_val * sqrt(max(0.0, 1.0 - e0^2))
        H0 = G0 * cos(deg2rad(i0_deg))
        u0 = [g0, h0, G0, H0]

        try
            sol = nlsolve(f!, u0; ftol=ftol, autodiff=:forward)

            if converged(sol)
                g_r, h_r, G_r, H_r = sol.zero

                # --- residual filter: trust-region convergence does not ---
                # guarantee F ~ 0; reject false zeros where the residual is
                # still large (this removes the spurious scattered points)
                Fcheck = similar(sol.zero)
                f!(Fcheck, sol.zero)
                if sqrt(sum(abs2, Fcheck)) > resid_tol
                    continue
                end

                # recover keplerian from the converged momenta
                e_r = sqrt(max(0.0, 1.0 - (G_r / L_val)^2))
                i_r_deg = rad2deg(acos(clamp(H_r / G_r, -1.0, 1.0)))
                g_r_deg = rad2deg(mod2pi(g_r))
                h_r_deg = rad2deg(mod2pi(h_r))

                # physical collision filter (periapsis > body radius)
                if a * (1.0 - e_r) > params.R && 0.0 <= e_r < 1.0

                    cand = (a=Float64(a), e=Float64(e_r), i_deg=Float64(i_r_deg),
                            g_deg=Float64(g_r_deg), h_deg=Float64(h_r_deg),
                            G=Float64(G_r), H=Float64(H_r))

                    # deduplicate against equilibria already found
                    is_new = all(solutions) do s
                        dg = abs(deg2rad(s.g_deg) - deg2rad(cand.g_deg))
                        dh = abs(deg2rad(s.h_deg) - deg2rad(cand.h_deg))
                        dG = abs(s.G - cand.G)
                        dH = abs(s.H - cand.H)
                        (dg + dh + dG + dH) > dedup_tol
                    end

                    if is_new
                        push!(solutions, cand)
                    end
                end
            else
                @debug "NLsolve did not converge for guess (g0=$g0, h0=$h0)"
            end
        catch err
            @debug "Solver error for guess (g0=$g0, h0=$h0): $err"
        end
    end

    return solutions
end

end # end module

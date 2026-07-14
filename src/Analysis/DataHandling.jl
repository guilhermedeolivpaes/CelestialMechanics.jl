# src/Analysis/DataHandling.jl

"""
    DataHandling

Module responsible for the post-processing and persistence of simulation data. 

It utilizes `DataFrames.jl` and `CSV.jl` to create, manage, and export tabular data files. 
The functions provided handle the removal of physical units for clean export, the loading of 
previously saved trajectories, and the formatting of specific orbital solutions (e.g., frozen orbits) 
into ready-to-use Julia structures.
"""
module DataHandling

# the dependency of simulationresult is imported via types
using ..Types
using DataFrames, CSV, Statistics

# the dependency of odesolution comes from simulationresult
using DifferentialEquations: ODESolution

# imports unitful to handle the removal of units before saving
using Unitful

using SpecialFunctions

export save_filtered_results
export load_gravity_sha, sha_to_physical_params, sha_to_body_data

"""
    save_data_frame(save_flag::Bool, result::Types.SimulationResult, ic_index::Int, directory::String)

Exports the fully processed orbital elements from a `SimulationResult` object to a CSV file.

This function automatically strips `Unitful` units from the DataFrame to ensure clean, standard 
numerical columns in the resulting CSV file. It uses a semicolon (`;`) as the delimiter.

# Arguments
- `save_flag::Bool`: Execution control flag. If `false`, the function returns immediately without saving.
- `result::Types.SimulationResult`: The simulation result object containing the `elements` DataFrame.
- `ic_index::Int`: The index of the initial condition, used to uniquely name the output file (e.g., `orbit_complete_1.csv`).
- `directory::String`: The path to the output directory. The directory is created if it does not exist.
"""
function save_data_frame(save_flag::Bool, result::Types.SimulationResult, ic_index::Int, directory::String)
    if !save_flag
        return
    end

    # ensures that the data consists of pure numbers (without internal unitful)
    # to avoid the "_ustrip" suffix in csv column names
    df_to_save = mapcols(ustrip, result.elements)

    # create the folder and save it with the chosen delimiter.
    mkpath(directory)
    file_path = joinpath(directory, "orbit_complete_$(ic_index).csv")
    
    CSV.write(file_path, df_to_save, delim=';')
    
    println("File saved: $file_path")
end

"""
    load_orbit_data(path::String; mode::Symbol = :all)

Reads a previously saved complete orbit CSV file and returns a filtered `DataFrame`.

# Arguments
- `path::String`: The absolute or relative path to the CSV file.

# Keyword Arguments
- `mode::Symbol`: Determines which columns are returned. Defaults to `:all`.
  - `:all`: Returns the entire DataFrame.
  - `:elements`: Returns only the time and Keplerian elements (`time`, `a_km`, `e`, `i_deg`, `h_deg`, `g_deg`, `f_deg`, `alt_peri_km`).
  - `:vectors`: Returns only the time and Cartesian state vectors (`time`, `X_km`, `Y_km`, `Z_km`, `VX_kms`, `VY_kms`, `VZ_kms`).

# Returns
- `DataFrame`: The filtered tabular data.
"""
function load_orbit_data(path::String; mode::Symbol = :all)
    if !isfile(path)
        error("File not found at: $path")
    end

    # reads the csv respecting the delimiter we used when saving
    df = CSV.read(path, DataFrame, delim=';')

    # name standardization (ensures that 'time_s' or 'time_seconds' always becomes 'time')
    for t_col in [:time_s, :time_seconds]
        if hasproperty(df, t_col)
            rename!(df, t_col => :time)
            break
        end
    end

    # column filter
    if mode == :elements
        return select(df, [:time, :a_km, :e, :i_deg, :h_deg, :g_deg, :f_deg, :alt_peri_km])
    elseif mode == :vectors
        return select(df, [:time, :X_km, :Y_km, :Z_km, :VX_kms, :VY_kms, :VZ_kms])
    end

    return df
end

"""
    save_filtered_results(pairs_or_result; io=nothing, output_file=nothing, format=:initialconditions, h0=90.0, g0=90.0, f0=0.0)

Processes and exports a filtered set of orbital parameters (typically derived from equilibrium conditions like Sun-synchronous or frozen orbits).

It accepts raw tuples, named tuples, or specialized result objects (like grid search outputs) and formats them into either a CSV or a text file containing formatted `InitialConditions` struct calls, ready to be copy-pasted into new simulation scripts.

# Arguments
- `pairs_or_result`: The data structure containing the filtered `(a, e, i)` pairs. Can be a vector of tuples, a vector of named tuples, or an object with a `filtered_pairs` or `pairs` field.

# Keyword Arguments
- `io::Union{Nothing, IO}`: An optional IO stream to write the results to. Defaults to `nothing`.
- `output_file::Union{Nothing, AbstractString}`: Custom path for the output file. Defaults to `data/output_data/filtered_results.csv` if `nothing`.
- `format::Symbol`: Output format. Can be `:csv`, `:delaunay` (uses `l0` for mean anomaly), or `:initialconditions` (uses `f0` for true anomaly). Defaults to `:initialconditions`.
- `h0::Float64`: Default value for the right ascension of the ascending node (degrees). Defaults to 90.0.
- `g0::Float64`: Default value for the argument of periapsis (degrees). Defaults to 90.0.
- `f0::Float64`: Default value for the true anomaly (degrees). Defaults to 0.0.

# Returns
- `DataFrame`: A DataFrame containing the processed `a`, `e`, and `i` columns.
"""
function save_filtered_results(pairs_or_result;
    io::Union{Nothing,IO} = nothing,
    output_file::Union{Nothing,AbstractString} = nothing,
    format::Symbol = :initialconditions,
    h0::Float64 = 90.0,
    g0::Float64 = 90.0,
    f0::Float64 = 0.0
    )

    # data extraction 
    pairs =
        if hasproperty(pairs_or_result, :filtered_pairs)
            getfield(pairs_or_result, :filtered_pairs)
        elseif hasproperty(pairs_or_result, :pairs)
            getfield(pairs_or_result, :pairs)
        elseif isa(pairs_or_result, AbstractVector)
            pairs_or_result
        else
            throw(ArgumentError("Type not supported: $(typeof(pairs_or_result))"))
        end

    # detect if input has full 6-element data
    has_angles = false
    if !isempty(pairs)
        el = first(pairs)
        if isa(el, NamedTuple) && all(haskey(el, k) for k in (:a, :e, :i, :h, :g, :l))
            has_angles = true
        end
    end

    cleaned = if has_angles
        NTuple{6,Float64}[]
    else
        NTuple{3,Float64}[]
    end

    for el in pairs
        if has_angles && isa(el, NamedTuple)
            push!(cleaned, (Float64(el.a), Float64(el.e), Float64(el.i),
                            Float64(el.h), Float64(el.g), Float64(el.l)))
        elseif isa(el, Tuple) && length(el) == 3
            push!(cleaned, (Float64(el[1]), Float64(el[2]), Float64(el[3])))
        elseif isa(el, NamedTuple)
            if all(haskey(el, k) for k in (:a, :e, :i_deg))
                push!(cleaned, (Float64(el.a), Float64(el.e), Float64(el.i_deg)))
            elseif all(haskey(el, k) for k in (:a, :e, :i))
                push!(cleaned, (Float64(el.a), Float64(el.e), Float64(el.i)))
            end
        end
    end

    df = if has_angles
        DataFrame(a = [t[1] for t in cleaned],
                  e = [t[2] for t in cleaned],
                  i = [t[3] for t in cleaned],
                  h = [t[4] for t in cleaned],
                  g = [t[5] for t in cleaned],
                  l = [t[6] for t in cleaned])
    else
        DataFrame(a = [t[1] for t in cleaned],
                  e = [t[2] for t in cleaned],
                  i = [t[3] for t in cleaned])
    end

    # definition of the output string
    function get_output_string(row, fmt)
        # use angles from data if available, otherwise fall back to defaults
        h_val = hasproperty(row, :h) ? row.h : h0
        g_val = hasproperty(row, :g) ? row.g : g0
        angle_val = hasproperty(row, :l) ? row.l : f0

        if fmt == :delaunay
            return "InitialConditions(a0=$(row.a)km, e0=$(row.e), i0=$(row.i)°, h0=$(h_val)°, g0=$(g_val)°, l0=$(angle_val)°)"
        else
            return "InitialConditions(a0=$(row.a)km, e0=$(row.e), i0=$(row.i)°, h0=$(h_val)°, g0=$(g_val)°, f0=$(angle_val)°)"
        end
    end

    # writing (io or file)
    if io !== nothing
        if format == :csv
            CSV.write(io, df)
        else
            for row in eachrow(df)
                println(io, get_output_string(row, format))
            end
        end
        return df
    end

    outfile = output_file === nothing ?
        joinpath(pwd(), "output", "filtered_results.csv") :
        normpath(String(output_file))

    mkpath(dirname(outfile))

    if format == :csv
        CSV.write(outfile, df)
    else
        open(outfile, "w") do fio
            for row in eachrow(df)
                println(fio, get_output_string(row, format))
            end
        end
    
        # separates the filename from the extension to create the raw file
        base_name, _ = splitext(outfile)
        pure_outfile = base_name * "_pure.csv"
        
        # saves the strictly numeric dataframe using the CSV package
        CSV.write(pure_outfile, df)
        println("Pure DataFrame saved to: $pure_outfile")
    end
    
    println("Filtered results saved to: $outfile")
    return df
end

"""
# Normalization convention

The fully normalized (4π) coefficients ``\\bar{C}_{nm}`` relate to the unnormalized 
coefficients ``C_{nm}`` via:

```math
\\bar{C}_{nm} = N_{nm} \\cdot C_{nm}, \\quad
N_{nm} = \\sqrt{(2 - \\delta_{0m})(2n+1) \\frac{(n-m)!}{(n+m)!}}
```

The zonal convention is ``J_n = -C_{n,0}`` (unnormalized).

# References
- Kaula, W.M. (1966). *Theory of Satellite Geodesy*.
- Balmino, G. (1994). Celest. Mech. Dyn. Astron. 60, 331-364.
- Lemoine, F.G. et al. (2014). JGR Planets, 119(8), 1676-1698.
"""

"""
    _normalization_factor(n::Int, m::Int) -> Float64

Computes the fully-normalized (geodesy 4π) normalization factor:

```math
N_{nm} = \\sqrt{(2 - \\delta_{0m})(2n+1) \\frac{(n-m)!}{(n+m)!}}
```

Uses `logfactorial` to avoid overflow for high degrees.
"""
function _normalization_factor(n::Int, m::Int)::Float64
    δ = (m == 0) ? 1 : 0
    log_N = 0.5 * (log(2 - δ) + log(2n + 1) + _logfactorial(n - m) - _logfactorial(n + m))
    return exp(log_N)
end

_logfactorial(n::Int) = n < 0 ? error("_logfactorial: negative argument") : SpecialFunctions.lgamma(n + 1.0)


"""
    _build_fields(coeffs, Jn, max_degree) -> Dict{Symbol, Any}

Builds a flat `Dict{Symbol, Any}` of field names (`:j2`, `:c22`, `:s31`, etc.)
from the parsed coefficient dictionaries up to `max_degree`.

This is the core mechanism that makes the output independent of hardcoded 
field lists: it iterates over all `(n, m)` pairs and generates the 
corresponding symbols dynamically.
"""
function _build_fields(coeffs::Dict{Tuple{Int,Int}, Tuple{Float64,Float64}},
                       Jn::Dict{Int,Float64},
                       max_degree::Int)
    fields = Dict{Symbol, Any}()

    for n in 2:max_degree
        # Zonal: jN
        if haskey(Jn, n)
            fields[Symbol("j$n")] = Jn[n]
        end

        # Tesseral/sectorial: cNM, sNM
        for m in 1:n
            if haskey(coeffs, (n, m))
                C, S = coeffs[(n, m)]
                fields[Symbol("c$n$m")] = C
                fields[Symbol("s$n$m")] = S
            end
        end
    end

    return fields
end


"""
    load_gravity_sha(filepath::String; max_degree::Int = 18) -> NamedTuple

Reads a PDS SHA-format gravity field file and returns denormalized coefficients.

# Arguments
- `filepath::String`: Path to the `.tab` file (e.g., `ggmes_100v08_sha.tab`).

# Keyword Arguments
- `max_degree::Int`: Maximum harmonic degree to read. Coefficients beyond this 
  degree are ignored. Defaults to `18`.

# Returns
A `NamedTuple` with fields:
- `R_km::Float64`       — Reference radius in km.
- `GM::Float64`         — Gravitational parameter (GM) in km³/s².
- `max_degree::Int`     — The degree actually loaded.
- `normalized::Bool`    — Whether the source file was normalized (from header flag).
- `coeffs::Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}` — Dictionary mapping 
  `(n, m) => (Cnm_unnorm, Snm_unnorm)`.
- `Jn::Dict{Int, Float64}` — Zonal harmonics: `n => Jn = -C_{n,0}` (unnormalized).

# SHA Header Format
```
R_km, GM, tide_system, n_max, m_max, norm_flag, ref1, ref2
```
where `norm_flag = 1` means fully normalized.

# SHA Data Format (each subsequent line)
```
n, m, Cnm_bar, Snm_bar, sigma_Cnm, sigma_Snm
```

# Example
```julia
grav = load_gravity_sha("ggmes_100v08_sha.tab", max_degree=8)
println("J2 = ", grav.Jn[2])
println("C22 = ", grav.coeffs[(2,2)])
```
"""
function load_gravity_sha(filepath::String; max_degree::Int = 18)
    filepath = filepath
    if !isfile(filepath)
        error("GravityModels: file not found at $filepath")
    end

    lines = readlines(filepath)
    
    # ── Parse header (line 1) ────────────────────────────────────────────────
    header = replace(lines[1], r"[Dd]" => "e")  # Fortran D exponent → Julia e
    header_fields = split(header, ',')
    
    R_km       = parse(Float64, strip(header_fields[1]))
    GM         = parse(Float64, strip(header_fields[2]))
    n_max_file = parse(Int,     strip(header_fields[4]))
    norm_flag  = parse(Int,     strip(header_fields[6]))
    
    is_normalized = (norm_flag == 1)
    n_load = min(max_degree, n_max_file)

    # ── Parse data lines ─────────────────────────────────────────────────────
    coeffs = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
    Jn     = Dict{Int, Float64}()

    for line in lines[2:end]
        stripped = strip(replace(line, r"[Dd]" => "e"))
        isempty(stripped) && continue

        fields = split(stripped, ',')
        length(fields) < 4 && continue

        n = parse(Int, strip(fields[1]))
        m = parse(Int, strip(fields[2]))

        n > n_load && continue
        n < 2 && continue

        Cnm_bar = parse(Float64, strip(fields[3]))
        Snm_bar = parse(Float64, strip(fields[4]))

        # ── Denormalize ──────────────────────────────────────────────────────
        if is_normalized
            N = _normalization_factor(n, m)
            Cnm = Cnm_bar / N
            Snm = Snm_bar / N
        else
            Cnm = Cnm_bar
            Snm = Snm_bar
        end

        coeffs[(n, m)] = (Cnm, Snm)

        if m == 0
            Jn[n] = -Cnm
        end
    end

    @info "GravityModels: loaded $(length(coeffs)) coefficients up to degree $n_load" source=basename(filepath) R_km GM normalized=is_normalized

    return (
        R_km       = R_km,
        GM         = GM,
        max_degree = n_load,
        normalized = is_normalized,
        coeffs     = coeffs,
        Jn         = Jn
    )
end

"""
    sha_to_physical_params(filepath::String; max_degree::Int = 6, kwargs...) -> PhysicalParams

Reads a SHA file and constructs a `PhysicalParams` struct with denormalized 
coefficients, ready for the analytical (Maxima → Julia) pipeline.

Fields are populated dynamically: only coefficients present in the file up to 
`max_degree` are set; all others remain `nothing`.

# Keyword Arguments
- `max_degree::Int`: Maximum degree to load. Defaults to `6`.
- All additional kwargs are forwarded to `PhysicalParams` (e.g., `mu_3`, `a_3`, `beta`).

# Example
```julia
p = sha_to_physical_params("ggmes_100v08_sha.tab", max_degree=8)
# p.mu ≈ 22031.86, p.j2 ≈ 1.006e-5 (denormalized)
```
"""
function sha_to_physical_params(filepath::String; max_degree::Int = 6, kwargs...)
    grav = load_gravity_sha(filepath; max_degree = max_degree)
    fields = _build_fields(grav.coeffs, grav.Jn, grav.max_degree)

    # Map dynamic fields into PhysicalParams constructor kwargs
    pp_kwargs = Dict{Symbol, Any}(:mu => grav.GM, :R => grav.R_km)
    
    for fname in fieldnames(Types.PhysicalParams)
        if haskey(fields, fname)
            pp_kwargs[fname] = fields[fname]
        end
    end

    # Merge user-provided kwargs (third body, SRP, etc.)
    merge!(pp_kwargs, Dict(kwargs))

    return Types.PhysicalParams(; pp_kwargs...)
end

"""
    sha_to_body_data(filepath::String; max_degree::Int = 6, name::Symbol = :custom, 
                     spice_id::String = "", omega_rot::Float64 = 0.0, 
                     extra_fields...) -> NamedTuple

Reads a SHA file and constructs a `NamedTuple` compatible with `BODIES_DATA` 
entries, suitable for `create_perturbation_model`.

Fields (`j2`, `c22`, `s31`, etc.) are **generated dynamically** from `max_degree`.
No manual updates are needed when extending to higher degrees — the function 
iterates over all `(n, m)` pairs present in the file.

# Keyword Arguments
- `max_degree::Int`: Maximum degree to load. Defaults to `6`.
- `name::Symbol`: Body name symbol. Defaults to `:custom`.
- `spice_id::String`: NAIF SPICE ID. Defaults to `""`.
- `omega_rot::Float64`: Rotation rate in rad/s. Defaults to `0.0`.
- `extra_fields...`: Any additional fields to include (e.g., `d_AU=0.387u"AU"`).

# Example
```julia
mercury_hf = sha_to_body_data("ggmes_100v08_sha.tab"; 
    max_degree = 8, name = :mercury_hgm008,
    spice_id = "MERCURY", omega_rot = 1.2399e-6,
    d_AU = 0.387u"AU", e_sun = 0.206)

# Register dynamically
BODIES_DATA[:mercury_hgm008] = mercury_hf
model = create_perturbation_model(:mercury_hgm008, j_harmonics=[2,3,4])
```
"""
function sha_to_body_data(
        filepath::String; 
        max_degree::Int = 6,
        name::Symbol = :custom, 
        spice_id::String = "",
        omega_rot::Float64 = 0.0,
        extra_fields...
    )
    grav = load_gravity_sha(filepath; max_degree = max_degree)
    fields = _build_fields(grav.coeffs, grav.Jn, grav.max_degree)

    # Start with the fixed metadata
    base = Dict{Symbol, Any}(
        :name      => name,
        :mu        => grav.GM * 1.0u"km^3/s^2",
        :R         => grav.R_km * 1.0u"km",
        :omega_rot => omega_rot * 1.0u"rad/s",
        :spice_id  => spice_id
    )

    # Merge harmonic fields (j2, c22, s31, etc.)
    merge!(base, fields)

    # Merge user-provided extra fields (d_AU, e_sun, etc.)
    for (k, v) in extra_fields
        base[k] = v
    end

    # Convert Dict -> NamedTuple
    keys_sorted = sort(collect(keys(base)))
    nt_keys = tuple(keys_sorted...)
    nt_vals = tuple((base[k] for k in keys_sorted)...)
    return NamedTuple{nt_keys}(nt_vals)
end

end # end of module

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

export save_filtered_results

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

    cleaned = Tuple{Float64,Float64,Float64}[]
    for el in pairs
        if isa(el, Tuple) && length(el) == 3
            push!(cleaned, (Float64(el[1]), Float64(el[2]), Float64(el[3])))
        elseif isa(el, NamedTuple)
            if all(haskey(el, k) for k in (:a, :e, :i_deg))
                push!(cleaned, (Float64(el.a), Float64(el.e), Float64(el.i_deg)))
            elseif all(haskey(el, k) for k in (:a, :e, :i))
                push!(cleaned, (Float64(el.a), Float64(el.e), Float64(el.i)))
            end
        end
    end

    df = DataFrame(a = [t[1] for t in cleaned],
                   e = [t[2] for t in cleaned],
                   i = [t[3] for t in cleaned])

    # 2. definition of the output string
    # local function to generate the formatted row as requested
    function get_output_string(row, fmt)
        if fmt == :delaunay
            # hybrid format: keplerian metrics (a,e,i) + delaunay angles (h,g,l)
            return "InitialConditions(a0=$(row.a)km, e0=$(row.e), i0=$(row.i), h0=$(h0), g0=$(g0), l0=$(l0))"
        else
            # standard keplerian format
            return "InitialConditions(a0=$(row.a)km, e0=$(row.e), i0=$(row.i), h0=$(h0), g0=$(g0), f0=$(f0))"
        end
    end

    # 3. writing (io or file)
    
    # if it is output to console/stream
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

    # if it is output to a file
    outfile = output_file === nothing ?
        normpath(joinpath(@__DIR__, "..", "data", "output_data", "filtered_results.csv")) :
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
    end
    
    println("Filtered results saved to: $outfile")
    return df
end

end # end of module

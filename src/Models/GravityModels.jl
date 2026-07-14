# src/Analysis/GravityModels.jl

"""
    GravityModels
 
Module for reading, denormalizing, and packaging spherical harmonic gravity field 
models from PDS SHA-format files (e.g., HgM008, GRGM1200A, GGM50A01).
 
The SHA (Spherical Harmonic ASCII) format is the standard used by NASA GSFC and JPL 
for archiving gravity solutions on the Planetary Data System (PDS). Coefficients in 
these files are **fully normalized** (geodesy 4π convention).
 
This module provides:
- `load_gravity_sha`: Reads a `.tab` SHA file, denormalizes to the unnormalized 
  convention used in classical celestial mechanics (Kaula, 1966), and returns a 
  `Dict` mapping `(n, m)` pairs to `(Cnm, Snm)` values along with metadata.
- `sha_to_physical_params`: Convenience wrapper that produces a `PhysicalParams` 
  struct ready for the analytical pipeline.
- `sha_to_body_data`: Produces a `NamedTuple` compatible with `BODIES_DATA` entries, 
  ready for `create_perturbation_model`. Fields are generated dynamically from 
  `max_degree`, so no manual updates are needed when extending to higher degrees.
 
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
module GravityModels
 
using ..Types
using SpecialFunctions
using Unitful
 
export load_gravity_sha, sha_to_physical_params, sha_to_body_data

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

end # module
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
using StaticArrays
using LinearAlgebra
using Printf
 
export load_gravity_sha, sha_to_physical_params, sha_to_body_data, load_gravity_icgem, icgem_to_body_data
export load_shape_obj, polyhedron_harmonics, polyhedron_to_body_data

"""
    _normalization_factor(n::Int, m::Int) -> Float64

Computes the fully-normalized (geodesy 4π) normalization factor:

```math
N_{nm} = \\sqrt{(2 - \\delta_{0m})(2n+1) \\frac{(n-m)!}{(n+m)!}}
```

Uses `logfactorial` to avoid overflow for high degrees.
"""
function _normalization_factor(n::Int, m::Int)::Float64
    delta = (m == 0) ? 1 : 0
    log_N = 0.5 * (log(2 - delta) + log(2n + 1) + _logfactorial(n - m) - _logfactorial(n + m))
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
        # denormalize
        if is_normalized
            N = _normalization_factor(n, m)
            Cnm = Cnm_bar * N
            Snm = Snm_bar * N
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

"""
    load_gravity_icgem(filepath::String; max_degree::Int = 18) -> NamedTuple

reads an ICGEM-format (.gfc) gravity field file and returns denormalized coefficients.

the ICGEM format is the standard used by the International Centre for Global Earth
Models (GFZ Potsdam) for distributing spherical harmonic gravity field models. it is
supported for all planetary bodies (Earth, Moon, Mars, etc.) and provides access to
hundreds of models via https://icgem.gfz-potsdam.de/.

the file consists of a header section (key-value pairs ending with `end_of_head`)
and a data section with lines of the form:

    gfc  n  m  Cnm  Snm  [sigma_Cnm  sigma_Snm]

time-variable keywords (`gfct`, `trnd`, `asin`, `acos`, `dot`) are skipped;
only static `gfc` coefficients are loaded.

coefficients are assumed fully normalized (4-pi geodesy convention) unless the
header field `norm` indicates otherwise. denormalization to the unnormalized
(kaula) convention is applied automatically.

this function was entirely generated by claude opus 4.6 based on the existing
`load_gravity_sha` function by the author.

# arguments
- `filepath::String`: path to the `.gfc` file.

# keyword arguments
- `max_degree::Int`: maximum harmonic degree to read. defaults to `18`.

# returns
a `NamedTuple` with fields:
- `R_km::Float64`       -- reference radius in km.
- `GM::Float64`         -- gravitational parameter (GM) in km^3/s^2.
- `max_degree::Int`     -- the degree actually loaded.
- `normalized::Bool`    -- whether the source file was normalized.
- `coeffs::Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}` -- dictionary mapping
  `(n, m) => (Cnm_unnorm, Snm_unnorm)`.
- `Jn::Dict{Int, Float64}` -- zonal harmonics: `n => Jn = -C_{n,0}` (unnormalized).

# references
- ICGEM format specification: https://icgem.gfz.de/docs/ICGEM-Format-2023.pdf
- kaula, w.m. (1966). *theory of satellite geodesy*.

# example
```julia
grav = load_gravity_icgem("EGM96.gfc", max_degree=10)
println("J2 = ", grav.Jn[2])
println("C22 = ", grav.coeffs[(2,2)])
```
"""
function load_gravity_icgem(filepath::String; max_degree::Int = 18)
    if !isfile(filepath)
        error("GravityModels: file not found at $filepath")
    end

    lines = readlines(filepath)

    # --- parse header ---
    GM = 0.0             # will be in m^3/s^2 from the file
    R_m = 0.0            # will be in meters from the file
    n_max_file = 0
    is_normalized = true # icgem default is fully_normalized
    header_end = 0

    for (idx, line) in enumerate(lines)
        stripped = strip(line)

        if startswith(stripped, "end_of_head")
            header_end = idx
            break
        end

        # parse key-value header fields
        tokens = split(stripped)
        isempty(tokens) && continue

        key = lowercase(tokens[1])

        if key == "earth_gravity_constant" || key == "gravity_constant"
            GM = parse(Float64, tokens[2])
        elseif key == "radius"
            R_m = parse(Float64, tokens[2])
        elseif key == "max_degree"
            n_max_file = parse(Int, tokens[2])
        elseif key == "norm"
            norm_str = lowercase(tokens[2])
            is_normalized = (norm_str == "fully_normalized")
        end
    end

    if header_end == 0
        @warn "GravityModels: no 'end_of_head' found in ICGEM file; attempting to parse from line 1."
        header_end = 0
    end

    # convert units: icgem uses SI (m, m^3/s^2) -> we want km, km^3/s^2
    R_km = R_m / 1000.0
    GM_km = GM / 1.0e9   # m^3/s^2 -> km^3/s^2

    n_load = n_max_file > 0 ? min(max_degree, n_max_file) : max_degree

    # --- parse data lines ---
    coeffs = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
    Jn = Dict{Int, Float64}()

    for line in lines[(header_end + 1):end]
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, "#") && continue  # comment lines in data section

        tokens = split(stripped)
        length(tokens) < 5 && continue

        # only read static coefficients (keyword "gfc")
        keyword = lowercase(tokens[1])
        keyword != "gfc" && continue

        n = parse(Int, tokens[2])
        m = parse(Int, tokens[3])

        n > n_load && continue
        n < 2 && continue

        Cnm_bar = parse(Float64, replace(tokens[4], r"[Dd]" => "e"))
        Snm_bar = parse(Float64, replace(tokens[5], r"[Dd]" => "e"))

        # denormalize
        if is_normalized
            N = _normalization_factor(n, m)
            Cnm = Cnm_bar * N
            Snm = Snm_bar * N
        else
            Cnm = Cnm_bar
            Snm = Snm_bar
        end

        coeffs[(n, m)] = (Cnm, Snm)

        if m == 0
            Jn[n] = -Cnm
        end
    end

    @info "GravityModels: loaded $(length(coeffs)) coefficients up to degree $n_load" source=basename(filepath) R_km GM_km3s2=GM_km normalized=is_normalized

    return (
        R_km       = R_km,
        GM         = GM_km,
        max_degree = n_load,
        normalized = is_normalized,
        coeffs     = coeffs,
        Jn         = Jn
    )
end


"""
    icgem_to_body_data(filepath::String; max_degree::Int = 6, name::Symbol = :custom,
                       spice_id::String = "", omega_rot::Float64 = 0.0,
                       extra_fields...) -> NamedTuple

reads an ICGEM (.gfc) file and constructs a `NamedTuple` compatible with
`BODIES_DATA` entries, suitable for `create_perturbation_model`.

fields (`j2`, `c22`, `s31`, etc.) are generated dynamically from `max_degree`.

this function was entirely generated by claude opus 4.6 based on the existing
`sha_to_body_data` function by the author.

# keyword arguments
- `max_degree::Int`: maximum degree to load. defaults to `6`.
- `name::Symbol`: body name symbol. defaults to `:custom`.
- `spice_id::String`: naif spice id. defaults to `""`.
- `omega_rot::Float64`: rotation rate in rad/s. defaults to `0.0`.
- `extra_fields...`: any additional fields to include (e.g., `d_AU=1.0u"AU"`).

# example
```julia
earth_egm96 = icgem_to_body_data("EGM96.gfc";
    max_degree = 10, name = :earth_egm96,
    spice_id = "EARTH", omega_rot = 7.2921150e-5)

BODIES_DATA[:earth_egm96] = earth_egm96
model = create_perturbation_model(:earth_egm96, j_harmonics=[2,3,4,5,6])
```
"""
function icgem_to_body_data(
        filepath::String;
        max_degree::Int = 6,
        name::Symbol = :custom,
        spice_id::String = "",
        omega_rot::Float64 = 0.0,
        extra_fields...
    )
    grav = load_gravity_icgem(filepath; max_degree = max_degree)
    fields = _build_fields(grav.coeffs, grav.Jn, grav.max_degree)

    # start with the fixed metadata
    base = Dict{Symbol, Any}(
        :name      => name,
        :mu        => grav.GM * 1.0u"km^3/s^2",
        :R         => grav.R_km * 1.0u"km",
        :omega_rot => omega_rot * 1.0u"rad/s",
        :spice_id  => spice_id
    )

    # merge harmonic fields (j2, c22, s31, etc.)
    merge!(base, fields)

    # merge user-provided extra fields
    for (k, v) in extra_fields
        base[k] = v
    end

    # convert dict -> namedtuple
    keys_sorted = sort(collect(keys(base)))
    nt_keys = tuple(keys_sorted...)
    nt_vals = tuple((base[k] for k in keys_sorted)...)
    return NamedTuple{nt_keys}(nt_vals)
end

# ==============================================================================
# polyhedron harmonics v3 -- exact, arbitrary degree
# ==============================================================================
#
# computes unnormalized gravitational harmonic coefficients Cnm and Snm
# from a homogeneous constant-density polyhedral shape model.
#
# method:
#   1. decompose polyhedron into tetrahedra (each face + origin)
#   2. compute exact cartesian volume moments via simplex integration
#   3. convert cartesian moments to spherical harmonic coefficients
#      using the solid harmonic expansion (automated for any degree)
#
# references:
#   Werner, R.A. (1997). Computers & Geosciences 23(10), 1071-1077.
#   Mirtich, B. (1996). J. Graphics Tools 1(2), 31-50.
#
# note: the author used Claude Opus 4.6 to validate, optimize, and correct 
# the polyhedron functions.
#
# ==============================================================================

"""
    load_shape_obj(filepath::String) -> (vertices, faces)

reads a Wavefront OBJ shape model from `filepath`.

# arguments
- `filepath::String`: path to the `.obj` file.

# returns
a tuple `(vertices, faces)` where:
- `vertices::Vector{SVector{3,Float64}}` — vertex coordinates.
- `faces::Vector{SVector{3,Int}}` — triangular face vertex indices (1-based).

polygonal faces with more than 3 vertices are triangulated as a fan from the
first vertex. the `v/vt/vn` index format is handled (only vertex index is used).

# example
```julia
verts, faces = load_shape_obj("didymos_radar.obj")
```
"""
function load_shape_obj(filepath::String)
    filepath = abspath(filepath)
    if !isfile(filepath)
        error("GravityModels: shape file not found at $filepath")
    end

    vertices = SVector{3,Float64}[]
    faces    = SVector{3,Int}[]

    open(filepath, "r") do io
        for line in eachline(io)
            stripped = strip(line)
            isempty(stripped) && continue
            stripped[1] == '#' && continue
            tokens = split(stripped)
            isempty(tokens) && continue

            if tokens[1] == "v" && length(tokens) >= 4
                push!(vertices, SVector(
                    parse(Float64, tokens[2]),
                    parse(Float64, tokens[3]),
                    parse(Float64, tokens[4])))
            elseif tokens[1] == "f" && length(tokens) >= 4
                vidx = Int[]
                for i in 2:length(tokens)
                    push!(vidx, parse(Int, split(tokens[i], '/')[1]))
                end
                for i in 2:(length(vidx) - 1)
                    push!(faces, SVector(vidx[1], vidx[i], vidx[i+1]))
                end
            end
        end
    end

    @info "GravityModels: loaded shape model" n_vertices=length(vertices) n_faces=length(faces)
    return vertices, faces
end

# ==============================================================================
# exact cartesian moment integral over a tetrahedron (origin, v1, v2, v3)
#
# parametrization: r = u*v1 + v*v2 + w*v3 with 0 <= u,v,w, u+v+w <= 1.
# jacobian = det(v1, v2, v3).
#
# the monomial x^a * y^b * z^c is expanded via multinomial theorem in
# (u, v, w), then integrated term by term using the simplex formula:
#   int u^I * v^J * w^K = I! * J! * K! / (I+J+K+3)!
# ==============================================================================

function _tet_moment(v1::SVector{3,Float64}, v2::SVector{3,Float64},
                      v3::SVector{3,Float64}, a::Int, b::Int, c::Int)

    det_J = dot(v1, cross(v2, v3))
    powers = (a, b, c)

    # for each axis k (1=x, 2=y, 3=z) with power pk, expand
    # (u*vj[k] for j=1,2,3)^pk into multinomial terms
    axis_terms = Vector{Vector{Tuple{Int,Int,Int,Float64}}}(undef, 3)
    for k in 1:3
        pk = powers[k]
        a1, a2, a3 = v1[k], v2[k], v3[k]
        terms = Tuple{Int,Int,Int,Float64}[]
        for i1 in 0:pk
            for i2 in 0:(pk - i1)
                i3 = pk - i1 - i2
                coeff = _multinomial_coeff(pk, i1, i2, i3) * a1^i1 * a2^i2 * a3^i3
                if coeff != 0.0
                    push!(terms, (i1, i2, i3, coeff))
                end
            end
        end
        axis_terms[k] = terms
    end

    # contract across axes and integrate over simplex
    result = 0.0
    for t1 in axis_terms[1]
        for t2 in axis_terms[2]
            for t3 in axis_terms[3]
                I = t1[1] + t2[1] + t3[1]
                J = t1[2] + t2[2] + t3[2]
                K = t1[3] + t2[3] + t3[3]
                coeff = t1[4] * t2[4] * t3[4]
                result += coeff * _simplex_integral(I, J, K)
            end
        end
    end

    return det_J * result
end

# multinomial coefficient: p! / (i! * j! * k!) where i + j + k = p
function _multinomial_coeff(p::Int, i::Int, j::Int, k::Int)
    return factorial(p) / (factorial(i) * factorial(j) * factorial(k))
end

# integral of u^I * v^J * w^K over the unit simplex (u+v+w <= 1)
# = I! * J! * K! / (I + J + K + 3)!
function _simplex_integral(I::Int, J::Int, K::Int)
    return factorial(I) * factorial(J) * factorial(K) / factorial(I + J + K + 3)
end

# ==============================================================================
# compute all cartesian moments up to a given total degree
# ==============================================================================

function _compute_moments(vertices::Vector{SVector{3,Float64}},
                           faces::Vector{SVector{3,Int}},
                           max_deg::Int)

    M = Dict{Tuple{Int,Int,Int}, Float64}()

    exponents = Tuple{Int,Int,Int}[]
    for deg in 0:max_deg
        for a in 0:deg
            for b in 0:(deg - a)
                c = deg - a - b
                push!(exponents, (a, b, c))
            end
        end
    end

    for f in faces
        v1 = vertices[f[1]]
        v2 = vertices[f[2]]
        v3 = vertices[f[3]]
        for (a, b, c) in exponents
            val = _tet_moment(v1, v2, v3, a, b, c)
            M[(a,b,c)] = get(M, (a,b,c), 0.0) + val
        end
    end

    return M
end

# ==============================================================================
# solid harmonics: cartesian polynomial expansion of r^n * Pnm * cos(m*phi)
# and r^n * Pnm * sin(m*phi), for arbitrary (n, m).
#
# uses the recurrence for unnormalized regular solid harmonics
# (without Condon-Shortley phase):
#
#   sectoral (m = n):
#     Rc_{n,n} = (2n-1) [x Rc_{n-1,n-1} - y Rs_{n-1,n-1}]
#     Rs_{n,n} = (2n-1) [y Rc_{n-1,n-1} + x Rs_{n-1,n-1}]
#
#   sub-sectoral (m = n-1):
#     Rc_{n,n-1} = (2n-1) z Rc_{n-1,n-1}
#     Rs_{n,n-1} = (2n-1) z Rs_{n-1,n-1}
#
#   tesseral/zonal (m <= n-2):
#     Rc_{n,m} = (2n-1) z Rc_{n-1,m} - (n+m-1)(n-m-1) r^2 Rc_{n-2,m}
#     Rs_{n,m} = (2n-1) z Rs_{n-1,m} - (n+m-1)(n-m-1) r^2 Rs_{n-2,m}
# ==============================================================================

# polynomial in (x, y, z) as Dict{(a,b,c) => coefficient}
const CartPoly = Dict{Tuple{Int,Int,Int}, Float64}

function _poly_mul_var(p::CartPoly, axis::Int)
    result = CartPoly()
    for ((a, b, c), coeff) in p
        if axis == 1
            key = (a+1, b, c)
        elseif axis == 2
            key = (a, b+1, c)
        else
            key = (a, b, c+1)
        end
        result[key] = get(result, key, 0.0) + coeff
    end
    return result
end

function _poly_mul_r2(p::CartPoly)
    result = CartPoly()
    for ((a, b, c), coeff) in p
        for (da, db, dc) in [(2,0,0), (0,2,0), (0,0,2)]
            key = (a+da, b+db, c+dc)
            result[key] = get(result, key, 0.0) + coeff
        end
    end
    return result
end

function _poly_scale(p::CartPoly, s::Float64)
    result = CartPoly()
    for (key, coeff) in p
        result[key] = coeff * s
    end
    return result
end

function _poly_add(p1::CartPoly, p2::CartPoly)
    result = copy(p1)
    for (key, coeff) in p2
        result[key] = get(result, key, 0.0) + coeff
    end
    return result
end

function _build_solid_harmonics(n_max::Int)
    Rc = Matrix{CartPoly}(undef, n_max + 1, n_max + 1)
    Rs = Matrix{CartPoly}(undef, n_max + 1, n_max + 1)

    for n in 0:n_max
        for m in 0:n_max
            Rc[n+1, m+1] = CartPoly()
            Rs[n+1, m+1] = CartPoly()
        end
    end

    # seed: R_{0,0}^c = 1
    Rc[1, 1] = CartPoly((0,0,0) => 1.0)

    for n in 1:n_max
        f = Float64(2*n - 1)

        # sectoral: m = n
        Rc[n+1, n+1] = _poly_scale(
            _poly_add(
                _poly_mul_var(Rc[n, n], 1),
                _poly_scale(_poly_mul_var(Rs[n, n], 2), -1.0)),
            f)
        Rs[n+1, n+1] = _poly_scale(
            _poly_add(
                _poly_mul_var(Rc[n, n], 2),
                _poly_mul_var(Rs[n, n], 1)),
            f)

        # sub-sectoral: m = n-1
        Rc[n+1, n] = _poly_scale(_poly_mul_var(Rc[n, n], 3), f)
        Rs[n+1, n] = _poly_scale(_poly_mul_var(Rs[n, n], 3), f)

        # tesseral and zonal: m = n-2 down to 0
        for m in (n-2):-1:0
            f1 = Float64(2*n - 1)
            f2 = Float64((n + m - 1) * (n - m - 1))

            term1 = _poly_scale(_poly_mul_var(Rc[n, m+1], 3), f1)
            term2 = _poly_scale(_poly_mul_r2(Rc[n-1, m+1]), f2)
            Rc[n+1, m+1] = _poly_add(term1, _poly_scale(term2, -1.0))

            term1s = _poly_scale(_poly_mul_var(Rs[n, m+1], 3), f1)
            term2s = _poly_scale(_poly_mul_r2(Rs[n-1, m+1]), f2)
            Rs[n+1, m+1] = _poly_add(term1s, _poly_scale(term2s, -1.0))
        end
    end

    return Rc, Rs
end

# ==============================================================================
# convert cartesian moments to harmonic coefficients
# ==============================================================================

function _moments_to_harmonics(M::Dict{Tuple{Int,Int,Int}, Float64},
                                V::Float64, R::Float64, n_max::Int)

    Rc, Rs = _build_solid_harmonics(n_max)

    coeffs = NamedTuple{(:n, :m, :Cnm, :Snm), Tuple{Int,Int,Float64,Float64}}[]

    for n in 1:n_max
        for m in 0:n
            # dot product of solid harmonic polynomial with cartesian moments
            int_c = 0.0
            for ((a, b, c), coeff) in Rc[n+1, m+1]
                int_c += coeff * get(M, (a, b, c), 0.0)
            end

            int_s = 0.0
            for ((a, b, c), coeff) in Rs[n+1, m+1]
                int_s += coeff * get(M, (a, b, c), 0.0)
            end

            # normalization: Cnm = (2 - delta_0m) (n-m)!/(n+m)! (1/(V R^n)) int
            delta_0m = m == 0 ? 1.0 : 0.0
            #factor = (2.0 - delta_0m) * _factorial_ratio_log(n, m) / (V * R^n)
            factor = (2.0 - delta_0m) / (factorial(n + m) * V * R^n)

            push!(coeffs, (n=n, m=m, Cnm=factor * int_c, Snm=factor * int_s))
        end
    end

    return coeffs
end

# (n-m)! / (n+m)! in log-space for numerical stability
function _factorial_ratio_log(n::Int, m::Int)
    m == 0 && return 1.0
    log_ratio = 0.0
    for k in (n - m + 1):(n + m)
        log_ratio -= log(k)
    end
    return exp(log_ratio)
end

"""
    polyhedron_harmonics(vertices, faces, R, n_max) -> Vector{NamedTuple}

computes unnormalized gravitational harmonic coefficients Cnm and Snm for a
homogeneous constant-density polyhedron, up to degree and order `n_max`.

the method is exact: cartesian volume moments are computed analytically over
each tetrahedron (face + origin), then converted to spherical harmonic
coefficients via the solid harmonic polynomial expansion.

# arguments
- `vertices::Vector{SVector{3,Float64}}`: vertex coordinates (same units as R).
- `faces::Vector{SVector{3,Int}}`: triangular face indices (1-based).
- `R::Float64`: reference radius for the harmonic expansion.
- `n_max::Int`: maximum degree and order.

# returns
vector of named tuples `(n=Int, m=Int, Cnm=Float64, Snm=Float64)`.

# references
- Werner, R.A. (1997). Computers & Geosciences 23(10), 1071-1077.
- Mirtich, B. (1996). J. Graphics Tools 1(2), 31-50.

# example
```julia
verts, faces = load_shape_obj("didymos.obj")
coeffs = polyhedron_harmonics(verts, faces, 390.0, 4)
```
"""
function polyhedron_harmonics(vertices::Vector{SVector{3,Float64}},
                               faces::Vector{SVector{3,Int}},
                               R::Float64, n_max::Int)

    n_max >= 1 || error("GravityModels: n_max must be >= 1")

    # polyhedron volume via divergence theorem
    volume = 0.0
    for f in faces
        volume += dot(vertices[f[1]], cross(vertices[f[2]], vertices[f[3]]))
    end
    volume = abs(volume) / 6.0

    @info "GravityModels: polyhedron" volume=volume n_faces=length(faces)

    # compute all cartesian moments up to degree n_max
    M = _compute_moments(vertices, faces, n_max)

    # volume consistency check
    vol_moments = abs(get(M, (0,0,0), 0.0))
    @info "GravityModels: volume check" divergence=volume moments=vol_moments

    return _moments_to_harmonics(M, volume, R, n_max)
end

"""
    polyhedron_to_body_data(filepath, R, n_max; name, spice_id, omega_rot,
                           scale_factor=1.0, extra_fields...) -> NamedTuple

loads an OBJ shape model and computes its harmonic coefficients, returning a
`NamedTuple` compatible with `BODIES_DATA` (same output format as `sha_to_body_data`).

the harmonic coefficient fields are generated dynamically using the same
convention as `sha_to_body_data`:
- zonal: `:j2`, `:j3`, ...  (Jn = -Cn0)
- tesseral/sectoral: `:c22`, `:s31`, ...

# arguments
- `filepath::String`: path to the `.obj` shape model.
- `R::Float64`: reference radius (same units as vertex coordinates after scaling).
- `n_max::Int`: maximum harmonic degree/order.

# keyword arguments
- `name::Symbol`: body identifier (e.g., `:apophis_poly`).
- `spice_id::String`: SPICE body ID string.
- `omega_rot::Float64`: sidereal rotation rate in rad/s.
- `scale_factor::Float64`: multiplicative factor applied to all vertex coordinates
  before computing the harmonics. defaults to `1.0`. use when the shape model
  needs rescaling to match a target volume (e.g., Lang et al. 2022 use 0.25 for
  Apophis).
- `extra_fields...`: any additional fields to include in the NamedTuple
  (e.g., `GM=2.86`).

# example
```julia
apophis = polyhedron_to_body_data("file.obj", 0.170, 4;
              name=:apophis_poly, spice_id="20099942",
              omega_rot=5.711e-5, scale_factor=0.25, GM=2.98047e-10)

BODIES_DATA[:apophis_poly] = apophis
model = create_perturbation_model(:apophis_poly, j_harmonics=[2,3,4], cs_harmonics=[22])
```
"""
function polyhedron_to_body_data(filepath::String, R::Float64, n_max::Int;
                                  name::Symbol, spice_id::String,
                                  omega_rot::Float64,
                                  scale_factor::Float64=1.0,
                                  extra_fields...)
    vertices, faces = load_shape_obj(filepath)

    if scale_factor != 1.0
        vertices = [scale_factor * v for v in vertices]
    end

    coeffs = polyhedron_harmonics(vertices, faces, R, n_max)

    fields = Dict{Symbol, Any}()
    fields[:name]       = name
    fields[:spice_id]   = spice_id
    fields[:omega_rot]  = omega_rot
    fields[:ref_radius] = R

    for c in coeffs
        if c.m == 0
            fields[Symbol("j$(c.n)")] = -c.Cnm
        else
            fields[Symbol("c$(c.n)$(c.m)")] = c.Cnm
            if abs(c.Snm) > 0.0
                fields[Symbol("s$(c.n)$(c.m)")] = c.Snm
            end
        end
    end

    for (k, v) in extra_fields
        fields[k] = v
    end

    keys_tuple = Tuple(sort(collect(keys(fields))))
    vals_tuple = Tuple(fields[k] for k in sort(collect(keys(fields))))
    return NamedTuple{keys_tuple}(vals_tuple)
end

end # module
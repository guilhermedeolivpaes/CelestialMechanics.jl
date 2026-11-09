# src/Models/PolyhedralGravity.jl

"""
    PolyhedralGravity

Module responsible for gravitational modeling of irregular bodies and asteroids
represented as constant-density polyhedral 3D shape meshes (Wavefront .obj).

Computes exact Cartesian volume moments via divergence theorem on triangular faces,
and converts them into spherical harmonic coefficients (C_nm, S_nm) up to arbitrary degree/order.

# References
- Werner, R. A. (1997). Spherical harmonic coefficients for a polyhedron. Computers & Geosciences, 23(10), 1071-1077.
- Mirtich, B. (1996). Fast and accurate computation of polyhedral mass properties. Journal of Graphics Tools, 1(2), 31-50.
"""
module PolyhedralGravity

using LinearAlgebra
using StaticArrays
using SpecialFunctions
using Logging

export load_shape_obj, polyhedron_harmonics, polyhedron_to_body_data

"""
    load_shape_obj(filepath::String) -> (Vector{SVector{3,Float64}}, Vector{SVector{3,Int}})

Reads a 3D shape model in Wavefront OBJ format (`.obj`) and returns:
- `vertices`: array of 3D coordinates `SVector{3,Float64}`.
- `faces`: array of triangular face indices `SVector{3,Int}` (1-based).

Polygonal faces with more than 3 vertices are triangulated as a fan from the
first vertex. The `v/vt/vn` index format is handled (only vertex index is used).

# Example
```julia
verts, faces = load_shape_obj("didymos_radar.obj")
```
"""
function load_shape_obj(filepath::String)
    filepath = abspath(filepath)
    if !isfile(filepath)
        error("PolyhedralGravity: shape file not found at $filepath")
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

    @info "PolyhedralGravity: loaded shape model" n_vertices=length(vertices) n_faces=length(faces)
    return vertices, faces
end

# ==============================================================================
# Exact cartesian moment integral over a tetrahedron (origin, v1, v2, v3)
#
# Parametrization: r = u*v1 + v*v2 + w*v3 with 0 <= u,v,w, u+v+w <= 1.
# Jacobian = det(v1, v2, v3).
#
# The monomial x^a * y^b * z^c is expanded via multinomial theorem in
# (u, v, w), then integrated term by term using the simplex formula:
#   int u^I * v^J * w^K = I! * J! * K! / (I+J+K+3)!
# ==============================================================================

function _tet_moment(v1::SVector{3,Float64}, v2::SVector{3,Float64},
                      v3::SVector{3,Float64}, a::Int, b::Int, c::Int)

    det_J = dot(v1, cross(v2, v3))
    powers = (a, b, c)

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

function _multinomial_coeff(p::Int, i::Int, j::Int, k::Int)
    return factorial(p) / (factorial(i) * factorial(j) * factorial(k))
end

function _simplex_integral(I::Int, J::Int, K::Int)
    return factorial(I) * factorial(J) * factorial(K) / factorial(I + J + K + 3)
end

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

# Polynomial in (x, y, z) as Dict{(a,b,c) => coefficient}
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

function _moments_to_harmonics(M::Dict{Tuple{Int,Int,Int}, Float64},
                                V::Float64, R::Float64, n_max::Int)

    Rc, Rs = _build_solid_harmonics(n_max)

    coeffs = NamedTuple{(:n, :m, :Cnm, :Snm), Tuple{Int,Int,Float64,Float64}}[]

    for n in 1:n_max
        for m in 0:n
            int_c = 0.0
            for ((a, b, c), coeff) in Rc[n+1, m+1]
                int_c += coeff * get(M, (a, b, c), 0.0)
            end

            int_s = 0.0
            for ((a, b, c), coeff) in Rs[n+1, m+1]
                int_s += coeff * get(M, (a, b, c), 0.0)
            end

            delta_0m = m == 0 ? 1.0 : 0.0
            factor = (2.0 - delta_0m) / (factorial(n + m) * V * R^n)

            push!(coeffs, (n=n, m=m, Cnm=factor * int_c, Snm=factor * int_s))
        end
    end

    return coeffs
end

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

Computes unnormalized gravitational harmonic coefficients Cnm and Snm for a
homogeneous constant-density polyhedron, up to degree and order `n_max`.

The method is exact: Cartesian volume moments are computed analytically over
each tetrahedron (face + origin), then converted to spherical harmonic
coefficients via the solid harmonic polynomial expansion.

# Arguments
- `vertices::Vector{SVector{3,Float64}}`: vertex coordinates (same units as R).
- `faces::Vector{SVector{3,Int}}`: triangular face indices (1-based).
- `R::Float64`: reference radius for the harmonic expansion.
- `n_max::Int`: maximum degree and order.

# Returns
Vector of named tuples `(n=Int, m=Int, Cnm=Float64, Snm=Float64)`.
"""
function polyhedron_harmonics(vertices::Vector{SVector{3,Float64}},
                               faces::Vector{SVector{3,Int}},
                               R::Float64, n_max::Int)

    n_max >= 1 || error("PolyhedralGravity: n_max must be >= 1")

    volume = 0.0
    for f in faces
        volume += dot(vertices[f[1]], cross(vertices[f[2]], vertices[f[3]]))
    end
    volume = abs(volume) / 6.0

    @info "PolyhedralGravity: polyhedron" volume=volume n_faces=length(faces)

    M = _compute_moments(vertices, faces, n_max)

    vol_moments = abs(get(M, (0,0,0), 0.0))
    @info "PolyhedralGravity: volume check" divergence=volume moments=vol_moments

    return _moments_to_harmonics(M, volume, R, n_max)
end

"""
    polyhedron_to_body_data(filepath, R, n_max; name, spice_id, omega_rot,
                           scale_factor=1.0, extra_fields...) -> NamedTuple

Loads an OBJ shape model and computes its harmonic coefficients, returning a
`NamedTuple` compatible with `BODIES_DATA`.

# Arguments
- `filepath::String`: path to the `.obj` shape model.
- `R::Float64`: reference radius (same units as vertex coordinates after scaling).
- `n_max::Int`: maximum harmonic degree/order.

# Keyword Arguments
- `name::Symbol`: body identifier (e.g., `:didymos_poly`).
- `spice_id::String`: SPICE body ID string.
- `omega_rot::Float64`: sidereal rotation rate in rad/s.
- `scale_factor::Float64`: multiplicative factor applied to all vertex coordinates
  before computing the harmonics. Defaults to `1.0`.
- `extra_fields...`: any additional fields to include in the NamedTuple (e.g., `GM=2.86`).
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

    # center of mass for a closed mesh of uniform density
    signed6_volume = 0.0
    centroid_sum = zero(vertices[1])

    for f in faces
        v1, v2, v3 = vertices[f[1]], vertices[f[2]], vertices[f[3]]
        d = dot(v1, cross(v2, v3))
        signed6_volume += d
        centroid_sum += d * (v1 + v2 + v3)
    end

    iszero(signed6_volume) && error("The mesh has zero volume.")
    centroid = centroid_sum / (4 * signed6_volume)
    vertices = [v - centroid for v in vertices]

    @info "PolyhedralGravity: mesh centered" centroid=centroid

    coeffs = polyhedron_harmonics(vertices, faces, R, n_max)

    fields = Dict{Symbol, Any}()
    fields[:name]       = name
    fields[:spice_id]   = spice_id
    fields[:omega_rot]  = omega_rot
    fields[:ref_radius] = R
    fields[:harmonic_degree] = n_max

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

end # module PolyhedralGravity

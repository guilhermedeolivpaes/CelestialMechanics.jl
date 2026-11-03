# src/Models/GravityModels.jl

"""
    GravityModels

Backward-compatibility shim module.
Spherical harmonic gravity loaders have moved to [`SphericalHarmonics`](@ref).
Polyhedral and asteroid shape gravity have moved to [`PolyhedralGravity`](@ref).
"""
module GravityModels

using ..SphericalHarmonics
using ..PolyhedralGravity

export load_gravity_sha, sha_to_physical_params, sha_to_body_data, load_gravity_icgem, icgem_to_body_data,
       load_shape_obj, polyhedron_harmonics, polyhedron_to_body_data

const load_gravity_sha = SphericalHarmonics.load_gravity_sha
const sha_to_physical_params = SphericalHarmonics.sha_to_physical_params
const sha_to_body_data = SphericalHarmonics.sha_to_body_data
const load_gravity_icgem = SphericalHarmonics.load_gravity_icgem
const icgem_to_body_data = SphericalHarmonics.icgem_to_body_data

const load_shape_obj = PolyhedralGravity.load_shape_obj
const polyhedron_harmonics = PolyhedralGravity.polyhedron_harmonics
const polyhedron_to_body_data = PolyhedralGravity.polyhedron_to_body_data

end # module GravityModels
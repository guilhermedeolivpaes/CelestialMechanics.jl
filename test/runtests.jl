# test/runtests.jl

using Test

@testset "CelestialMechanics.jl" begin
    include("test_coordinates.jl")
    include("test_canonical_units.jl")
    include("test_delaunay.jl")
    include("test_kepler_equation.jl")
    include("test_n_body_dynamics.jl")
    include("test_perturbation_models.jl")
    # Requer SPICE kernels — rode localmente com kernels disponíveis
    # include("test_spice_integration.jl")
end
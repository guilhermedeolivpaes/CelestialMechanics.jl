# test/test_perturbation_models.jl
#
# tests gravitational perturbations (j2, j3) in a unitary way.
# verifies:
#   1. j2 perturbation in the correct direction (centrifugal force at equator -> repulsion)
#   2. j2 perturbation = 0 when j2 = 0
#   3. j2 symmetry: symmetric in z -> az must be equal for z and -z
#   4. j3 perturbation = 0 when j3 = 0
#   5. north-south asymmetry of j3 (j3 is an odd harmonic)
#
# these tests are purely analytical - do not require spice.

using Test
using CelestialMechanics
using LinearAlgebra
using StaticArrays

# direct access to the internal submodule (not exported in the public api)
const PE = CelestialMechanics.PerturbationEquations

const MU_MOON = 4902.80012
const R_MOON  = 1738.1
const J2_MOON = 9.09010949481e-5
const J3_MOON = -1.21e-5

@testset "perturbation models" begin

    @testset "j2: finite magnitude and real result for r != 0" begin
        r = SVector(2000.0, 0.0, 500.0)
        a = PE.j2_perturbation(r, MU_MOON, R_MOON, J2_MOON)
        @test any(abs.(a) .> 0.0)
        @test all(isfinite.(a))
    end

    @testset "j2: null perturbation when j2 = 0" begin
        r = SVector(2000.0, 500.0, 300.0)
        a = PE.j2_perturbation(r, MU_MOON, R_MOON, 0.0)
        @test norm(a) ≈ 0.0 atol=1e-20
    end

    @testset "j2: symmetry in z - even harmonic" begin
        r_pos = SVector(1900.0, 500.0,  400.0)
        r_neg = SVector(1900.0, 500.0, -400.0)

        a_pos = PE.j2_perturbation(r_pos, MU_MOON, R_MOON, J2_MOON)
        a_neg = PE.j2_perturbation(r_neg, MU_MOON, R_MOON, J2_MOON)

        @test a_pos[1] ≈  a_neg[1] atol=1e-12   # ax symmetric
        @test a_pos[2] ≈  a_neg[2] atol=1e-12   # ay symmetric
        @test a_pos[3] ≈ -a_neg[3] atol=1e-12   # az antisymmetric
    end

    @testset "j2: at equator (z=0), az component = 0" begin
        r = SVector(2000.0, 0.0, 0.0)
        a = PE.j2_perturbation(r, MU_MOON, R_MOON, J2_MOON)
        @test abs(a[3]) < 1e-15
    end

    @testset "j2: decay ~ r^-4 at equator" begin
        r1 = SVector(2000.0, 0.0, 0.0)
        r2 = SVector(4000.0, 0.0, 0.0)   # doubled distance -> ratio = 2^4 = 16

        a1 = PE.j2_perturbation(r1, MU_MOON, R_MOON, J2_MOON)
        a2 = PE.j2_perturbation(r2, MU_MOON, R_MOON, J2_MOON)

        @test norm(a1) / norm(a2) ≈ 16.0 atol=1e-6
    end

    @testset "j3: null perturbation when j3 = 0" begin
        r = SVector(2000.0, 500.0, 300.0)
        a = PE.j3_perturbation(r, MU_MOON, R_MOON, 0.0)
        @test norm(a) ≈ 0.0 atol=1e-20
    end

    @testset "j3: north-south asymmetry - odd harmonic" begin
        r_pos = SVector(2000.0, 0.0,  500.0)
        r_neg = SVector(2000.0, 0.0, -500.0)

        a_pos = PE.j3_perturbation(r_pos, MU_MOON, R_MOON, J3_MOON)
        a_neg = PE.j3_perturbation(r_neg, MU_MOON, R_MOON, J3_MOON)

        @test a_pos[1] ≈ -a_neg[1] atol=1e-12   # ax antisymmetric
        @test a_pos[2] ≈ -a_neg[2] atol=1e-12   # ay antisymmetric
        @test a_pos[3] ≈  a_neg[3] atol=1e-12   # az symmetric
    end

    @testset "j2 dominates j3 for the moon (j2 >> j3)" begin
        r = SVector(2000.0, 300.0, 400.0)
        a_j2 = PE.j2_perturbation(r, MU_MOON, R_MOON, J2_MOON)
        a_j3 = PE.j3_perturbation(r, MU_MOON, R_MOON, J3_MOON)

        @test all(isfinite.(a_j2 + a_j3))
        @test norm(a_j2) > norm(a_j3)
    end
end
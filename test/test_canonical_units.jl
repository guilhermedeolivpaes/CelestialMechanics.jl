# test/test_canonical_units.jl
#
# tests canonical units (du, tu, vu) and normalization/denormalization functions.
# reference: vallado, "fundamentals of astrodynamics and applications", p. 237.
#
# verified properties:
#   1. du = r, vu = du/tu, tu = sqrt(du^3/mu)
#   2. normalize_state and denormalize_state = identity
#   3. normalized state is dimensionless (norm of r = 1 for r = r)

using Test
using CelestialMechanics
using CelestialMechanics.Coordinates
using LinearAlgebra

const MU_MOON = 4902.80012
const R_MOON  = 1738.1

@testset "canonical units" begin

    @testset "definition: du, tu, vu (vallado)" begin
        mu = MU_MOON
        R  = R_MOON

        units = Coordinates.canonical_units(mu, R)

        @test units.DU ≈ R                        atol=1e-12
        @test units.TU ≈ sqrt(R^3 / mu)           atol=1e-10
        @test units.VU ≈ units.DU / units.TU      atol=1e-10

        # vu is also sqrt(mu/r) - circular velocity on the surface
        @test units.VU ≈ sqrt(mu / R)             atol=1e-10
    end

    @testset "normalization: r on surface -> |r_adim| = 1" begin
        mu = MU_MOON; R = R_MOON
        units = Coordinates.canonical_units(mu, R)

        v_circ = sqrt(mu / R)   # km/s - circular velocity on the surface
        state_dim = [R, 0.0, 0.0, 0.0, v_circ, 0.0]

        state_adim = Coordinates.normalize_state(state_dim, units)

        @test norm(state_adim[1:3]) ≈ 1.0      atol=1e-12
        @test norm(state_adim[4:6]) ≈ 1.0      atol=1e-10
    end

    @testset "inversibility: normalize -> denormalize = identity" begin
        mu = MU_MOON; R = R_MOON
        units = Coordinates.canonical_units(mu, R)

        r_orig = [2000.0, 500.0, -300.0]
        v_orig = [0.5, 1.2, -0.3]
        state_orig = [r_orig; v_orig]

        state_adim = Coordinates.normalize_state(state_orig, units)
        r_rec, v_rec = Coordinates.denormalize_state(state_adim, units)

        @test r_rec ≈ r_orig atol=1e-10
        @test v_rec ≈ v_orig atol=1e-12
    end

    @testset "normalized tspan consistent with dt" begin
        # ensures that dividing tspan by tu and dt by tu gives consistent results
        mu = MU_MOON; R = R_MOON
        units = Coordinates.canonical_units(mu, R)

        dt_s   = 60.0                     # step in seconds
        t_end  = 7.0 * 24 * 3600.0       # 7 days in seconds

        dt_adim   = dt_s  / units.TU
        tend_adim = t_end / units.TU

        # the ratio must be preserved
        @test tend_adim / dt_adim ≈ t_end / dt_s atol=1e-10

        # dimensionless dt must be << 1 (sub-orbital step)
        @test dt_adim < 1.0
    end

    @testset "different bodies -> different units" begin
        mu_earth = 398600.4418
        R_earth  = 6378.137
        mu_moon  = MU_MOON
        R_moon   = R_MOON

        units_e = Coordinates.canonical_units(mu_earth, R_earth)
        units_m = Coordinates.canonical_units(mu_moon,  R_moon)

        # du must be the body radius
        @test units_e.DU ≈ R_earth atol=1e-12
        @test units_m.DU ≈ R_moon  atol=1e-12

        # earth tu is greater than moon tu (greater radius and mass)
        # (tu = sqrt(r^3/mu); for earth sqrt(6378^3/398600) = 806 s)
        @test units_e.TU ≈ sqrt(R_earth^3 / mu_earth) atol=1e-8
        @test units_m.TU ≈ sqrt(R_moon^3  / mu_moon)  atol=1e-8
    end
end
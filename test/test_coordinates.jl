# test/test_coordinates.jl
#
# tests conversions between keplerian elements and cartesian vectors.
# verified properties:
#   1. inversibility: elements -> cartesian -> elements must be identity
#   2. conservation of energy and angular momentum
#   3. singular cases: circular, equatorial, circular-equatorial orbits

using Test
using CelestialMechanics
using CelestialMechanics.Coordinates
using LinearAlgebra
using StaticArrays

# ------------------------------------------------------------------------------
# reference parameters (moon)
# ------------------------------------------------------------------------------
const MU_MOON  = 4902.80012   # km^3/s^2
const R_MOON   = 1738.1       # km

# tolerances
const ATOL_POS = 1e-8   # km  - position
const ATOL_VEL = 1e-8   # km/s - velocity
const ATOL_ELM = 1e-10  # dimensionless elements

@testset "coordinates: keplerian <-> cartesian" begin

    @testset "general orbit (arbitrary a, e, i, h, g, f)" begin
        a  = 2000.0       # km
        e  = 0.3
        i  = deg2rad(45.0)
        h  = deg2rad(30.0)
        g  = deg2rad(60.0)
        f  = deg2rad(120.0)
        mu = MU_MOON

        r, v = Coordinates.orbital_elements_to_state_vectors(a, e, i, h, g, f, mu)

        # --- conservation of specific energy ---
        eps = norm(v)^2 / 2 - mu / norm(r)
        eps_ref = -mu / (2a)
        @test eps ≈ eps_ref atol=1e-6

        # --- conservation of angular momentum (magnitude) ---
        p = a * (1 - e^2)            # semi-latus rectum
        h_mag_ref = sqrt(mu * p)
        h_vec = cross(r, v)
        @test norm(h_vec) ≈ h_mag_ref atol=1e-6

        # --- orbital radius ---
        r_ref = a * (1 - e^2) / (1 + e * cos(f))
        @test norm(r) ≈ r_ref atol=ATOL_POS
    end

    @testset "inversibility: elements -> r,v -> elements" begin
        # set of varied cases
        cases = [
            # (a km,  e,    i deg,   h deg,   g deg,   f deg)
            (2000.0, 0.05, 45.0,  30.0,  60.0, 120.0),
            (5000.0, 0.6,  53.3,   0.0, 270.0, 180.0),
            (1900.0, 0.01, 90.0,  90.0,  45.0,  45.0),
            (3000.0, 0.4,  10.0, 180.0, 120.0, 300.0),
        ]

        for (a, e, i_d, h_d, g_d, f_d) in cases
            i = deg2rad(i_d); h = deg2rad(h_d)
            g = deg2rad(g_d); f = deg2rad(f_d)
            mu = MU_MOON

            r, v = Coordinates.orbital_elements_to_state_vectors(a, e, i, h, g, f, mu)

            # reconstruct elements from r, v
            r_mat = reshape(collect(r), 3, 1)
            v_mat = reshape(collect(v), 3, 1)
            a_r, e_r, i_r, h_r, g_r, f_r, _ =
                Coordinates.state_vectors_to_orbital_elements(r_mat, v_mat, mu)

            @test a_r[1] ≈ a           atol=ATOL_POS
            @test e_r[1] ≈ e           atol=ATOL_ELM
            @test i_r[1] ≈ i_d         atol=1e-8
            @test h_r[1] ≈ h_d         atol=1e-8
            @test g_r[1] ≈ g_d         atol=1e-8
            @test f_r[1] ≈ f_d         atol=1e-8
        end
    end

    @testset "circular orbit (e ~ 0)" begin
        a  = 1837.1    # km  (low lunar orbit, ~100 km altitude)
        e  = 0.0
        i  = deg2rad(90.0)
        h  = deg2rad(0.0)
        g  = deg2rad(0.0)   # undefined, but must work
        f  = deg2rad(45.0)
        mu = MU_MOON

        r, v = Coordinates.orbital_elements_to_state_vectors(a, e, i, h, g, f, mu)

        # r must be constant throughout the orbit -> equal to the semi-axis
        @test norm(r) ≈ a atol=ATOL_POS

        # circular velocity
        v_circ = sqrt(mu / a)
        @test norm(v) ≈ v_circ atol=ATOL_VEL
    end

    @testset "equatorial orbit (i = 0)" begin
        a  = 3000.0
        e  = 0.2
        i  = 0.0
        h  = 0.0
        g  = deg2rad(90.0)
        f  = deg2rad(30.0)
        mu = MU_MOON

        r, v = Coordinates.orbital_elements_to_state_vectors(a, e, i, h, g, f, mu)

        # for i = 0, z must be zero
        @test abs(r[3]) < 1e-10
        @test abs(v[3]) < 1e-10
    end

    @testset "retrograde (i = 180 deg)" begin
        a  = 2500.0
        e  = 0.15
        i  = deg2rad(180.0)
        h  = deg2rad(45.0)
        g  = deg2rad(45.0)
        f  = deg2rad(90.0)
        mu = MU_MOON

        r, v = Coordinates.orbital_elements_to_state_vectors(a, e, i, h, g, f, mu)

        # angular momentum must be antiparallel to the z axis (h_z < 0)
        h_vec = cross(r, v)
        @test h_vec[3] < 0.0
    end
end
# test/test_delaunay.jl
#
# tests transformations between keplerian elements and delaunay variables.
#
# canonical relations:
#   l = sqrt(mu * a)
#   g = l * sqrt(1 - e^2)   = sqrt(mu * a * (1 - e^2))   = |h| (angular momentum)
#   h = g * cos(i)          = z component of angular momentum
#
# verified properties:
#   1. keplerian_to_delaunay and delaunay_to_keplerian = identity
#   2. physical meaning of l, g, h
#   3. delaunay_to_cartesian x keplerian -> cartesian consistent

using Test
using CelestialMechanics
using CelestialMechanics.Coordinates
using LinearAlgebra

const MU_MOON = 4902.80012

@testset "delaunay variables" begin

    @testset "canonical relations l, g, h" begin
        a  = 3000.0
        e  = 0.3
        i  = deg2rad(45.0)
        h_ang = deg2rad(30.0)
        g_ang = deg2rad(60.0)
        l_ang = deg2rad(90.0)
        mu = MU_MOON

        L, G, H, l, g, h = Coordinates.keplerian_to_delaunay(a, e, i, h_ang, g_ang, l_ang, mu)

        @test L ≈ sqrt(mu * a)                    atol=1e-8
        @test G ≈ sqrt(mu * a * (1 - e^2))        atol=1e-8
        @test H ≈ G * cos(i)                      atol=1e-8

        # angles preserved
        @test l ≈ l_ang atol=1e-14
        @test g ≈ g_ang atol=1e-14
        @test h ≈ h_ang atol=1e-14
    end

    @testset "inversibility: keplerian -> delaunay -> keplerian" begin
        cases = [
            (2000.0, 0.05, 45.0,  30.0,  60.0,  90.0),
            (5000.0, 0.6,  53.3,   0.0, 270.0, 180.0),
            (1900.0, 0.01, 89.9,  90.0,  45.0,  45.0),
            (4000.0, 0.4,   5.0, 180.0, 120.0, 300.0),
        ]

        for (a, e, i_d, h_d, g_d, l_d) in cases
            i = deg2rad(i_d); hh = deg2rad(h_d)
            g = deg2rad(g_d); l  = deg2rad(l_d)
            mu = MU_MOON

            L, G, H, l_r, g_r, h_r =
                Coordinates.keplerian_to_delaunay(a, e, i, hh, g, l, mu)

            a2, e2, i2, h2, g2, l2 =
                Coordinates.delaunay_to_keplerian(L, G, H, l_r, g_r, h_r, mu)

            @test a2 ≈ a     atol=1e-8
            @test e2 ≈ e     atol=1e-10
            @test i2 ≈ i     atol=1e-10
            @test l2 ≈ l     atol=1e-14
            @test g2 ≈ g     atol=1e-14
            @test h2 ≈ hh    atol=1e-14
        end
    end

    @testset "g = |specific angular momentum|" begin
        # for any orbit, g must be equal to the magnitude of the angular momentum
        a  = 2500.0
        e  = 0.25
        i  = deg2rad(60.0)
        h_ang = deg2rad(10.0)
        g_ang = deg2rad(200.0)
        f  = deg2rad(45.0)
        mu = MU_MOON

        l_ang = Coordinates.true_to_mean_anomaly(f, e)

        L, G, H, _, _, _ =
            Coordinates.keplerian_to_delaunay(a, e, i, h_ang, g_ang, l_ang, mu)

        r, v = Coordinates.orbital_elements_to_state_vectors(a, e, i, h_ang, g_ang, f, mu)
        h_vec = cross(r, v)

        @test G ≈ norm(h_vec) atol=1e-6
        @test H ≈ h_vec[3]   atol=1e-6
    end

    @testset "delaunay_to_cartesian consistent with orbital_elements_to_state_vectors" begin
        a  = 3000.0
        e  = 0.2
        i  = deg2rad(30.0)
        h_ang = deg2rad(50.0)
        g_ang = deg2rad(100.0)
        f  = deg2rad(75.0)
        mu = MU_MOON

        l_ang = Coordinates.true_to_mean_anomaly(f, e)

        L, G, H, l, g, h =
            Coordinates.keplerian_to_delaunay(a, e, i, h_ang, g_ang, l_ang, mu)

        r_d, v_d = Coordinates.delaunay_to_cartesian(L, G, H, l, g, h, mu)
        r_k, v_k = Coordinates.orbital_elements_to_state_vectors(a, e, i, h_ang, g_ang, f, mu)

        @test r_d ≈ r_k atol=1e-6
        @test v_d ≈ v_k atol=1e-6
    end

    @testset "circular orbit (e -> 0): g -> l" begin
        a  = 1838.0
        e  = 1e-10       # essentially zero
        i  = deg2rad(90.0)
        mu = MU_MOON

        L, G, H, _, _, _ =
            Coordinates.keplerian_to_delaunay(a, e, i, 0.0, 0.0, 0.0, mu)

        @test L ≈ G  atol=1e-4   # g ~ l when e ~ 0
        @test abs(H) < 1e-4      # h ~ 0 when i = 90 deg
    end
end
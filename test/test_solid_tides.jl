# test/test_solid_tides.jl
using Test
using LinearAlgebra
using StaticArrays
using Unitful
using UnitfulAstro
using CelestialMechanics
using DifferentialEquations

@testset "Solid Earth Tides Perturbation Suite" begin
    # Physical constants
    MU_EARTH = CelestialMechanics.Constants.BODIES_DATA[:earth].mu
    R_EARTH  = CelestialMechanics.Constants.BODIES_DATA[:earth].R
    MU_MOON  = CelestialMechanics.Constants.BODIES_DATA[:moon].mu
    MU_SUN   = CelestialMechanics.Constants.BODIES_DATA[:sun].mu

    mu_earth_val = Float64(ustrip(MU_EARTH))
    R_earth_val  = Float64(ustrip(R_EARTH))
    mu_moon_val  = Float64(ustrip(MU_MOON))
    mu_sun_val   = Float64(ustrip(MU_SUN))

    k2_nominal = 0.30
    d_moon = 384400.0 # km

    @testset "1. Tidal Potential Analytical Formulation" begin
        # Moon along X axis
        r_moon = SVector{3, Float64}(d_moon, 0.0, 0.0)

        # Satellite at sub-lunar point (r = [7000, 0, 0]) -> cos(psi) = 1, P2 = 1
        r_sat_sub = SVector{3, Float64}(7000.0, 0.0, 0.0)
        U_sub = solid_earth_tides_potential(r_sat_sub, r_moon, mu_moon_val, R_earth_val, k2_nominal)
        expected_U_sub = (k2_nominal * mu_moon_val * R_earth_val^5) / (d_moon^3 * 7000.0^3) * 1.0
        @test isapprox(U_sub, expected_U_sub, rtol=1e-12)
        @test U_sub > 0.0

        # Satellite at quadrature (r = [0, 7000, 0]) -> cos(psi) = 0, P2 = -0.5
        r_sat_quad = SVector{3, Float64}(0.0, 7000.0, 0.0)
        U_quad = solid_earth_tides_potential(r_sat_quad, r_moon, mu_moon_val, R_earth_val, k2_nominal)
        expected_U_quad = (k2_nominal * mu_moon_val * R_earth_val^5) / (d_moon^3 * 7000.0^3) * (-0.5)
        @test isapprox(U_quad, expected_U_quad, rtol=1e-12)
        @test U_quad < 0.0

        # Zero Love number yields zero potential
        @test solid_earth_tides_potential(r_sat_sub, r_moon, mu_moon_val, R_earth_val, 0.0) == 0.0
        # Zero body mass yields zero potential
        @test solid_earth_tides_potential(r_sat_sub, r_moon, 0.0, R_earth_val, k2_nominal) == 0.0
    end

    @testset "2. Exact Gradient Consistency" begin
        # Test that analytical acceleration vector matches numerical gradient of U_tide to machine precision
        r_moon = SVector{3, Float64}(250000.0, -180000.0, 220000.0)
        r_sat  = SVector{3, Float64}(4500.0, -3200.0, 4800.0)

        a_ana = solid_earth_tides_perturbation(r_sat, r_moon, mu_moon_val, R_earth_val, k2_nominal)

        eps = 1e-4
        grad_num = zeros(3)
        for i in 1:3
            dx = zeros(3)
            dx[i] = eps
            rp = r_sat + SVector{3, Float64}(dx...)
            rm = r_sat - SVector{3, Float64}(dx...)
            grad_num[i] = (solid_earth_tides_potential(rp, r_moon, mu_moon_val, R_earth_val, k2_nominal) -
                           solid_earth_tides_potential(rm, r_moon, mu_moon_val, R_earth_val, k2_nominal)) / (2.0 * eps)
        end

        @test isapprox(a_ana, SVector{3, Float64}(grad_num...), rtol=1e-6)

        # Zero cases
        @test solid_earth_tides_perturbation(r_sat, r_moon, mu_moon_val, R_earth_val, 0.0) == SVector{3, Float64}(0,0,0)
        @test solid_earth_tides_perturbation(r_sat, r_moon, 0.0, R_earth_val, k2_nominal) == SVector{3, Float64}(0,0,0)
        @test solid_earth_tides_perturbation(r_sat, r_moon, mu_moon_val, 0.0, k2_nominal) == SVector{3, Float64}(0,0,0)
    end

    @testset "3. Physical Magnitude and Direction" begin
        # Moon along X axis
        r_moon = SVector{3, Float64}(d_moon, 0.0, 0.0)
        r_sub  = SVector{3, Float64}(7000.0, 0.0, 0.0)
        a_sub  = solid_earth_tides_perturbation(r_sub, r_moon, mu_moon_val, R_earth_val, k2_nominal)

        # Sub-lunar: extra mass bulge below satellite pulls inwards towards Earth (-X)
        @test a_sub[1] < 0.0
        @test isapprox(a_sub[2], 0.0, atol=1e-15)
        @test isapprox(a_sub[3], 0.0, atol=1e-15)

        # Magnitude at LEO ~ 3.4e-7 m/s^2 (~3.4e-8 g)
        a_sub_si = norm(a_sub) * 1000.0 # m/s^2
        @test 1.0e-7 < a_sub_si < 1.0e-6

        # Solar tide comparison (Sun is at 1 AU ~ 1.496e8 km)
        r_sun = SVector{3, Float64}(1.496e8, 0.0, 0.0)
        a_sun = solid_earth_tides_perturbation(r_sub, r_sun, mu_sun_val, R_earth_val, k2_nominal)
        a_sun_si = norm(a_sun) * 1000.0

        # Lunar tide is ~ 2.18x solar tide
        ratio = a_sub_si / a_sun_si
        @test isapprox(ratio, 2.18, atol=0.05)
    end

    @testset "4. Model Construction and Parameter Forwarding" begin
        model = create_perturbation_model(
            :earth;
            solid_tides = true,
            love_k2 = 0.3019
        )
        @test model.solid_tides == true
        @test model.love_k2 == 0.3019

        # Default model has solid_tides = false
        default_model = create_perturbation_model(:earth)
        @test default_model.solid_tides == false
        @test default_model.love_k2 === nothing
    end

    @testset "5. Numerical Cowell Integration with Solid Tides" begin
        ics = [InitialConditions(
            a0 = 7000.0u"km",
            e0 = 0.005,
            i0 = 65.0u"°",
            h0 = 0.0u"°",
            g0 = 0.0u"°",
            f0 = 0.0u"°"
        )]

        tspan = (0.0, 86400.0) # 1 day
        t_vector = collect(range(0.0, 86400.0, length=121))

        # Setup analytic ephemeris with Sun and Moon
        eph = AnalyticEphemeris(mu_central = mu_earth_val)
        add_body!(eph, "SUN"; a = 1.0u"AU", e = 0.0167, mu = mu_sun_val)
        add_body!(eph, "MOON"; a = 384400.0u"km", e = 0.0549, mu = mu_earth_val)

        prop_opts = PropagatorOptions(
            propagator = CowellPropagator(),
            canonical_unit_normalization = true,
            integrator = Vern7(),
            abstol = 1e-9, reltol = 1e-9, maxiters = 10_000_000,
            saveat = true
        )

        # Baseline: J2 only
        model_base = create_perturbation_model(:earth; j_harmonics=[2])
        out_dir_base = mktempdir()
        res_base = run_simulation(
            ics = ics,
            perturbation_params = model_base,
            analytic_ephemeris = eph,
            tspan = tspan,
            t_vector = t_vector,
            propagator_options = prop_opts,
            output_directory = out_dir_base
        )
        sol_base = res_base[1].solution
        @test string(sol_base.retcode) == "Success"
        rf_base = SVector(sol_base.u[end][1:3]...)

        # With Solid Earth Tides
        model_tides = create_perturbation_model(
            :earth;
            j_harmonics = [2],
            n_body_symbols = [:sun, :moon],
            solid_tides = true,
            love_k2 = 0.30
        )
        out_dir_tides = mktempdir()
        res_tides = run_simulation(
            ics = ics,
            perturbation_params = model_tides,
            analytic_ephemeris = eph,
            tspan = tspan,
            t_vector = t_vector,
            propagator_options = prop_opts,
            output_directory = out_dir_tides
        )
        sol_tides = res_tides[1].solution
        @test string(sol_tides.retcode) == "Success"
        rf_tides = SVector(sol_tides.u[end][1:3]...)

        # Verify orbits are stable, finite, and distinct
        diff_rf = norm(rf_tides - rf_base)
        @test diff_rf > 0.0 # Tides perturbed the trajectory
        @test all(isfinite, rf_tides)

        rm(out_dir_base, recursive=true)
        rm(out_dir_tides, recursive=true)
    end
end

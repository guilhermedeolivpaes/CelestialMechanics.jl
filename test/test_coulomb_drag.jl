# test/test_coulomb_drag.jl
using Test
using LinearAlgebra
using StaticArrays
using Unitful
using UnitfulAstro
using CelestialMechanics
using DifferentialEquations

@testset "Space Electrodynamics: Coulomb Drag Suite" begin
    MU_EARTH = Float64(ustrip(CelestialMechanics.Constants.BODIES_DATA[:earth].mu))
    R_EARTH  = Float64(ustrip(CelestialMechanics.Constants.BODIES_DATA[:earth].R))
    omega_earth = Float64(ustrip(CelestialMechanics.Constants.BODIES_DATA[:earth].omega_rot))

    @testset "1. Effective Area Factor (OML & Debye Sheath)" begin
        v_rel = 7500.0 # m/s

        # Neutral spacecraft has unit factor
        @test coulomb_effective_area_factor(v_rel, 0.0) == 1.0

        # Negative floating potential enhances effective cross-section
        eta_5v = coulomb_effective_area_factor(v_rel, -5.0)
        @test eta_5v > 1.0
        @test 1.5 < eta_5v < 4.0 # Physical range for O+ in LEO

        # Greater negative voltage further expands the Debye sheath
        eta_50v = coulomb_effective_area_factor(v_rel, -50.0)
        @test eta_50v > eta_5v

        eta_500v = coulomb_effective_area_factor(v_rel, -500.0)
        @test eta_500v > eta_50v

        # Zero relative velocity returns 1.0
        @test coulomb_effective_area_factor(0.0, -10.0) == 1.0
    end

    @testset "2. Coulomb Drag Acceleration Vector & Physics" begin
        # Position at 400 km altitude along X
        r_vec = SVector{3, Float64}(R_EARTH + 400.0, 0.0, 0.0)
        # Velocity in Y direction ~ 7.67 km/s
        v_vec = SVector{3, Float64}(0.0, 7.67, 0.0)

        rho_atm = 1.0e-12 # kg/m^3
        cd = 2.2
        am = 0.01 / 4.0   # 0.0025 m^2/kg (3U CubeSat)

        a_coulomb = coulomb_drag_acceleration(
            r_vec, v_vec, rho_atm, cd, am, omega_earth;
            potential = -5.0
        )

        # 1. Opposes relative motion (pure deceleration)
        v_atm = SVector{3, Float64}(-omega_earth * r_vec[2], omega_earth * r_vec[1], 0.0)
        v_rel = v_vec - v_atm
        @test dot(a_coulomb, v_rel) < 0.0

        # Y component is dominant braking direction
        @test a_coulomb[2] < 0.0
        @test isapprox(a_coulomb[1], 0.0, atol=1e-15)
        @test isapprox(a_coulomb[3], 0.0, atol=1e-15)

        # 2. Order of magnitude check: ~ 1e-9 to 1e-8 m/s^2 (~ 1e-12 to 1e-11 km/s^2)
        a_si = norm(a_coulomb) * 1000.0 # m/s^2
        @test 1.0e-10 < a_si < 1.0e-7

        # 3. Plasma brake tether mode (100 m biased tether at -500 V)
        a_tether = coulomb_drag_acceleration(
            r_vec, v_vec, rho_atm, cd, am, omega_earth;
            potential = -500.0,
            tether_length = 100.0
        )
        a_tether_si = norm(a_tether) * 1000.0
        # Tether produces significantly higher drag for CubeSat deorbiting (~ micro-m/s^2)
        @test a_tether_si > a_si
        @test 1.0e-7 < a_tether_si < 1.0e-4

        # Zero velocity yields zero acceleration
        @test coulomb_drag_acceleration(r_vec, SVector{3, Float64}(0,0,0), rho_atm, cd, am, 0.0) == SVector{3, Float64}(0,0,0)
    end

    @testset "3. Model Configuration & Parameter Forwarding" begin
        model = create_perturbation_model(
            :earth;
            coulomb_drag = true,
            plasma_potential = -10.0,
            plasma_ion_density = 5.0e11,
            plasma_debye_length = 0.015,
            plasma_tether_length = 200.0
        )
        @test model.coulomb_drag == true
        @test model.plasma_potential == -10.0
        @test model.plasma_ion_density == 5.0e11
        @test model.plasma_debye_length == 0.015
        @test model.plasma_tether_length == 200.0

        # Default model has coulomb_drag = false
        default_model = create_perturbation_model(:earth)
        @test default_model.coulomb_drag == false
        @test default_model.plasma_potential === nothing
    end

    @testset "4. Numerical Cowell Integration with Coulomb Drag" begin
        # 400 km LEO orbit (CubeSat)
        ics = [InitialConditions(
            a0 = (R_EARTH + 400.0)u"km",
            e0 = 0.001,
            i0 = 51.6u"°",
            h0 = 0.0u"°",
            g0 = 0.0u"°",
            f0 = 0.0u"°"
        )]

        tspan = (0.0, 86400.0) # 1 day
        t_vector = collect(range(0.0, 86400.0, length=121))

        eph = AnalyticEphemeris(mu_central = MU_EARTH)
        add_body!(eph, "SUN"; a = 1.0u"AU", e = 0.0167, mu = Float64(ustrip(CelestialMechanics.Constants.BODIES_DATA[:sun].mu)))

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

        # Perturbed: J2 + Coulomb Drag (Plasma Brake Tether)
        model_plasma = create_perturbation_model(
            :earth;
            j_harmonics = [2],
            coulomb_drag = true,
            plasma_potential = -300.0,
            plasma_tether_length = 150.0,
            drag_cd = 2.2,
            drag_am = 0.01u"m^2/kg"
        )
        out_dir_plasma = mktempdir()
        res_plasma = run_simulation(
            ics = ics,
            perturbation_params = model_plasma,
            analytic_ephemeris = eph,
            tspan = tspan,
            t_vector = t_vector,
            propagator_options = prop_opts,
            output_directory = out_dir_plasma
        )
        sol_plasma = res_plasma[1].solution
        @test string(sol_plasma.retcode) == "Success"
        rf_plasma = SVector(sol_plasma.u[end][1:3]...)

        # Plasma brake deceleration measurably alters the trajectory
        diff_rf = norm(rf_plasma - rf_base)
        @test diff_rf > 0.0
        @test all(isfinite, rf_plasma)

        rm(out_dir_base, recursive=true)
        rm(out_dir_plasma, recursive=true)
    end
end

# test/test_relativistic_perturbations.jl

using Test
using LinearAlgebra
using StaticArrays
using Unitful, UnitfulAstro
using Unitful.DefaultSymbols
using DifferentialEquations
using CelestialMechanics
using CelestialMechanics.Coordinates

const MU_EARTH = 398600.4418 # km^3/s^2
const C_LIGHT_KM_S = 299792.458 # km/s

@testset "General Relativity Perturbations" begin

    r_leo = SVector(7000.0, 0.0, 0.0) # km
    v_circ = sqrt(MU_EARTH / 7000.0) # ~7.546 km/s
    v_leo = SVector(0.0, v_circ, 0.0)  # km/s
    J_earth = 981.0 # km^2/s (Earth spin angular momentum per unit mass)

    @testset "1PN Schwarzschild / Geodesic Precession" begin
        # 1. Circular orbit in equatorial plane (r . v = 0)
        a_gr = relativity_geodesic_perturbation(r_leo, v_leo, MU_EARTH)
        @test norm(a_gr) > 0.0
        # In circular orbit, (4*mu/r - v^2) = 3*v^2 > 0, so acceleration is purely radial outward (+X)
        @test a_gr[1] > 0.0
        @test abs(a_gr[2]) < 1e-18
        @test abs(a_gr[3]) < 1e-18

        # 2. Check exact numerical value for circular orbit:
        # a_GR = (mu / (c^2 * r^3)) * (4*mu/r - v^2) * r = (mu / (c^2 * r^2)) * 3 * (mu/r)
        term_expected = (MU_EARTH / (C_LIGHT_KM_S^2 * 7000.0^2)) * 3.0 * (MU_EARTH / 7000.0)
        @test isapprox(a_gr[1], term_expected; rtol=1e-5)

        # 3. Elliptical orbit with non-zero radial velocity (r . v > 0)
        v_ell = SVector(0.5, v_circ, 0.0)
        a_gr_ell = relativity_geodesic_perturbation(r_leo, v_ell, MU_EARTH)
        # Tangential component along v must be non-zero
        @test abs(a_gr_ell[2]) > 0.0

        # 4. Zero conditions
        @test norm(relativity_geodesic_perturbation(r_leo, v_leo, 0.0)) == 0.0

        # 5. Theoretical secular perigee advance:
        # Delta_omega = 6*pi*mu / (c^2 * a * (1 - e^2))
        # At a = 7000 km, e = 0.01: ~13.34 arcsec/yr
        a_leo = 7000.0
        e_leo = 0.01
        d_omega_leo = 6.0 * pi * MU_EARTH / (C_LIGHT_KM_S^2 * a_leo * (1.0 - e_leo^2))
        T_leo = 2.0 * pi * sqrt(a_leo^3 / MU_EARTH)
        orbits_leo_yr = (365.25 * 86400.0) / T_leo
        d_omega_arcsec_yr_leo = rad2deg(d_omega_leo * orbits_leo_yr) * 3600.0
        @test isapprox(d_omega_arcsec_yr_leo, 13.34; atol=0.1)

        # At higher geodetic orbit (a ~ 11400 km): ~3.84 arcsec/yr
        a_geo = 11520.0
        d_omega_geo = 6.0 * pi * MU_EARTH / (C_LIGHT_KM_S^2 * a_geo * (1.0 - e_leo^2))
        T_geo = 2.0 * pi * sqrt(a_geo^3 / MU_EARTH)
        orbits_geo_yr = (365.25 * 86400.0) / T_geo
        d_omega_arcsec_yr_geo = rad2deg(d_omega_geo * orbits_geo_yr) * 3600.0
        @test isapprox(d_omega_arcsec_yr_geo, 3.84; atol=0.1)
    end

    @testset "Lense-Thirring (Frame Dragging)" begin
        # 1. Equatorial orbit: r along +X, v along +Y, h along +Z, J along +Z
        a_lt_eq = relativity_lense_thirring_perturbation(r_leo, v_leo, MU_EARTH, J_earth)
        @test norm(a_lt_eq) > 0.0
        # In equatorial orbit: (r x v) . J = h * J_z > 0.
        # term1 = (3/r^2)*(h*J_z)*r is along +X.
        # term2 = v x J = (v_y * J_z) along +X.
        # Thus, net force is purely along +X (radial).
        @test a_lt_eq[1] > 0.0
        @test abs(a_lt_eq[2]) < 1e-18
        @test abs(a_lt_eq[3]) < 1e-18

        # 2. Polar orbit: r along +Z, v along +Y, h along -X, J along +Z
        r_polar = SVector(0.0, 0.0, 7000.0)
        v_polar = SVector(0.0, v_circ, 0.0)
        a_lt_pol = relativity_lense_thirring_perturbation(r_polar, v_polar, MU_EARTH, J_earth)
        @test norm(a_lt_pol) > 0.0
        # For polar orbit: (r x v) = -7000*v along +X, so (r x v) . J = 0 (term1 = 0).
        # term2 = v x J = (0, v, 0) x (0, 0, J) = (v*J, 0, 0) along +X (out-of-plane!).
        # This causes the famous nodal precession of LAGEOS!
        @test a_lt_pol[1] > 0.0
        @test abs(a_lt_pol[2]) < 1e-18
        @test abs(a_lt_pol[3]) < 1e-18

        # 3. Vector vs Scalar J_spin equivalence
        a_lt_vec = relativity_lense_thirring_perturbation(r_leo, v_leo, MU_EARTH, [0.0, 0.0, J_earth])
        @test a_lt_vec ≈ a_lt_eq

        # 4. Zero angular momentum yields zero
        @test norm(relativity_lense_thirring_perturbation(r_leo, v_leo, MU_EARTH, 0.0)) == 0.0
    end

    @testset "Model Parameters Setup" begin
        # Defaults
        m_def = create_perturbation_model(:earth)
        @test m_def.relativity_geodesic == false
        @test m_def.relativity_lense_thirring == false
        @test isnothing(m_def.spin_angular_momentum)

        # Configured
        m_gr = create_perturbation_model(
            :earth;
            relativity_geodesic = true,
            relativity_lense_thirring = true,
            spin_angular_momentum = 981.0
        )
        @test m_gr.relativity_geodesic == true
        @test m_gr.relativity_lense_thirring == true
        @test m_gr.spin_angular_momentum == 981.0
    end

    @testset "Full Simulation Propagation with Cowell" begin
        outdir = mktempdir()

        a0, e0, i0 = 7000.0u"km", 0.01, 45.0u"°"
        ics = [InitialConditions(a0 = a0, e0 = e0, i0 = i0, h0 = 0.0u"°", g0 = 0.0u"°", f0 = 0.0u"°")]

        t_start, t_end, step = 0.0u"hr", 2.0u"hr", 1.0u"minute"
        t_vec_u  = t_start:step:t_end
        tspan    = (ustrip(u"s", t_start), ustrip(u"s", t_end))
        t_vector = collect(ustrip.(u"s", t_vec_u))

        opts = PropagatorOptions(
            propagator                   = CowellPropagator(),
            canonical_unit_normalization = true,
            integrator                   = Vern7(),
            abstol = 1e-9, reltol = 1e-9, maxiters = 10_000_000,
            saveat = true,
        )

        # Baseline: J2 only
        params_base = create_perturbation_model(:earth; j_harmonics = [2])
        res_base = run_simulation(
            ics = ics, perturbation_params = params_base,
            tspan = tspan, t_vector = t_vector, propagator_options = opts,
            output_directory = outdir,
            analytic_ephemeris = AnalyticEphemeris(mu_central = MU_EARTH),
        )

        # Perturbed: J2 + General Relativity (1PN Geodesic + Lense-Thirring)
        params_gr = create_perturbation_model(
            :earth;
            j_harmonics = [2],
            relativity_geodesic = true,
            relativity_lense_thirring = true
        )
        res_gr = run_simulation(
            ics = ics, perturbation_params = params_gr,
            tspan = tspan, t_vector = t_vector, propagator_options = opts,
            output_directory = outdir,
            analytic_ephemeris = AnalyticEphemeris(mu_central = MU_EARTH),
        )

        sol_base = res_base[1].solution
        sol_gr   = res_gr[1].solution

        @test string(sol_base.retcode) == "Success"
        @test string(sol_gr.retcode)   == "Success"

        rf_base = SVector(sol_base.u[end][1:3]...)
        rf_gr   = SVector(sol_gr.u[end][1:3]...)

        # General relativity physically alters the trajectory
        @test norm(rf_gr - rf_base) > 0.0
        @test all(isfinite, rf_gr)
    end

end

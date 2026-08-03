# test/test_analytic_ephemeris.jl
#
# Tests the SPICE-free analytic (two-body Keplerian) ephemeris used for N-body,
# SRP and drag perturbations. Verifies:
#   1. Pure ephemeris: circular orbit -> constant radius and periodicity.
#   2. Pure ephemeris: elliptical orbit -> radius bounded by a(1±e), perigee at M0=0.
#   3. add_body! converts Unitful quantities correctly.
#   4. Interpolator builder: SUN is included when SRP is active; splines match analytics.
#   5. Full Cowell run with an analytic third body (Moon) — no SPICE — completes and
#      changes the trajectory relative to the unperturbed (J2-only) run.
#   6. Full Cowell run with analytic SRP (Sun) — no SPICE — completes and perturbs.
#
# None of these tests require SPICE kernels; they run in CI.

using Test
using CelestialMechanics
using CelestialMechanics.Coordinates
using DifferentialEquations
using LinearAlgebra
using StaticArrays
using Unitful, UnitfulAstro
using Unitful.DefaultSymbols

const MU_EARTH = 398600.4418          # km^3/s^2
const MU_SUN   = 1.32712440018e11     # km^3/s^2

# direct access to internals not exported in the public API
const EPH = CelestialMechanics.Ephemeris
const DYN = CelestialMechanics.Dynamics

@testset "analytic ephemeris (SPICE-free)" begin

    @testset "circular orbit: constant radius and periodicity" begin
        a = 42164.0
        elements = (a, 0.0, 0.0, 0.0, 0.0, 0.0, MU_EARTH)   # (a,e,i,Ω,ω,M0,μ)
        n = sqrt(MU_EARTH / a^3)
        T = 2π / n
        t_vec = collect(range(0.0, T; length = 64))

        pos = EPH.get_body_position_vectors_kepler(elements, t_vec)
        radii = [norm(ustrip.(p)) for p in pos]

        @test all(r -> isapprox(r, a; rtol = 1e-8), radii)          # radius constant
        @test isapprox(ustrip.(pos[end]), ustrip.(pos[1]); atol = 1e-3)  # back to start after T
    end

    @testset "elliptical orbit: bounded radius, perigee at M0 = 0" begin
        a, e = 384400.0, 0.2
        elements = (a, e, deg2rad(10.0), 0.0, 0.0, 0.0, MU_EARTH)
        n = sqrt(MU_EARTH / a^3)
        t_vec = collect(range(0.0, 2π / n; length = 128))

        radii = [norm(ustrip.(p)) for p in EPH.get_body_position_vectors_kepler(elements, t_vec)]

        @test minimum(radii) ≥ a * (1 - e) - 1e-3
        @test maximum(radii) ≤ a * (1 + e) + 1e-3
        @test isapprox(radii[1], a * (1 - e); rtol = 1e-6)          # M0 = 0 -> perigee
    end

    @testset "add_body! converts Unitful inputs" begin
        eph = AnalyticEphemeris(mu_central = MU_EARTH)
        add_body!(eph, "MOON"; a = 384400.0u"km", e = 0.0549, i = 5.145u"°")
        add_body!(eph, "SUN";  a = 1.0u"AU", e = 0.0167, mu = MU_SUN * u"km^3/s^2")

        moon = eph.elements["MOON"]
        sun  = eph.elements["SUN"]

        @test moon[1] ≈ 384400.0                    # a [km]
        @test moon[3] ≈ deg2rad(5.145)              # i [rad]
        @test moon[7] ≈ MU_EARTH                    # μ defaults to mu_central
        @test sun[1]  ≈ ustrip(u"km", 1.0u"AU")     # AU -> km
        @test sun[7]  ≈ MU_SUN                       # explicit μ_sun
    end

    @testset "interpolator builder includes SUN when SRP active" begin
        params = create_perturbation_model(:earth;
            n_body_symbols = [:moon],
            srp_cr = 1.5, srp_alpha = 0.02u"m^2/kg",
        )
        eph = AnalyticEphemeris(mu_central = MU_EARTH)
        add_body!(eph, "MOON"; a = 384400.0u"km", e = 0.0549, i = 5.145u"°")
        add_body!(eph, "SUN";  a = 1.0u"AU", e = 0.0167, mu = MU_SUN)

        t_vec = collect(range(0.0, 86400.0; length = 128))
        itps  = DYN._build_interpolators_analytic(eph, params, t_vec, t_vec, 1.0)

        ids = Set(bi.spice_id for bi in itps)
        @test "MOON" in ids
        @test "SUN"  in ids

        # spline of the Moon must match the analytic position at a sample time
        bi_moon = itps[findfirst(bi -> bi.spice_id == "MOON", itps)]
        t = t_vec[10]
        r_spline = [bi_moon.itp_x(t), bi_moon.itp_y(t), bi_moon.itp_z(t)]
        r_exact  = ustrip.(EPH.get_body_position_vectors_kepler(eph.elements["MOON"], [t])[1])
        @test isapprox(r_spline, r_exact; rtol = 1e-6)

        # Moon carries a physical μ (used for N-body); the Sun is position-only (μ = 0)
        @test bi_moon.mu > 0.0
        @test itps[findfirst(bi -> bi.spice_id == "SUN", itps)].mu == 0.0
    end

    # ---- full end-to-end Cowell runs, no SPICE ----

    a0, e0, i0 = 42164.0u"km", 0.01, 1.0u"°"
    ics = [InitialConditions(a0 = a0, e0 = e0, i0 = i0, h0 = 0.0u"°", g0 = 0.0u"°", f0 = 0.0u"°")]

    t_start, t_end, step = 0.0u"hr", 12.0u"hr", 1.0u"minute"
    t_vec_u  = t_start:step:t_end
    tspan    = (ustrip(u"s", t_start), ustrip(u"s", t_end))
    t_vector = collect(ustrip.(u"s", t_vec_u))

    opts = PropagatorOptions(
        propagator                   = CowellPropagator(),
        canonical_unit_normalization = true,
        integrator                   = Vern7(),
        abstol = 1e-10, reltol = 1e-10, maxiters = 10_000_000,
        saveat = true,
    )

    @testset "N-body (Moon) without SPICE perturbs the orbit" begin
        eph = AnalyticEphemeris(mu_central = MU_EARTH)
        add_body!(eph, "MOON"; a = 384400.0u"km", e = 0.0549, i = 5.145u"°")

        outdir = mktempdir()

        # baseline: J2 only (empty analytic ephemeris keeps it on the SPICE-free path)
        params_base = create_perturbation_model(:earth; j_harmonics = [2])
        res_base = run_simulation(
            ics = ics, perturbation_params = params_base,
            tspan = tspan, t_vector = t_vector, propagator_options = opts,
            output_directory = outdir,
            analytic_ephemeris = AnalyticEphemeris(mu_central = MU_EARTH),
        )

        # perturbed: J2 + Moon third body via analytic ephemeris
        params_moon = create_perturbation_model(:earth; j_harmonics = [2], n_body_symbols = [:moon])
        res_moon = run_simulation(
            ics = ics, perturbation_params = params_moon,
            tspan = tspan, t_vector = t_vector, propagator_options = opts,
            output_directory = outdir,
            analytic_ephemeris = eph,
        )

        sol_base = res_base[1].solution
        sol_moon = res_moon[1].solution

        @test string(sol_base.retcode) == "Success"
        @test string(sol_moon.retcode) == "Success"
        @test isfile(joinpath(outdir, "orbit_1.csv"))

        rf_base = SVector(sol_base.u[end][1:3]...)
        rf_moon = SVector(sol_moon.u[end][1:3]...)
        @test norm(rf_moon - rf_base) > 0.0        # third body changed the trajectory
        @test all(isfinite, rf_moon)
    end

    @testset "SRP (Sun) without SPICE perturbs the orbit" begin
        eph = AnalyticEphemeris(mu_central = MU_EARTH)
        add_body!(eph, "SUN"; a = 1.0u"AU", e = 0.0167, mu = MU_SUN)

        outdir = mktempdir()

        params_base = create_perturbation_model(:earth; j_harmonics = [2])
        res_base = run_simulation(
            ics = ics, perturbation_params = params_base,
            tspan = tspan, t_vector = t_vector, propagator_options = opts,
            output_directory = outdir,
            analytic_ephemeris = AnalyticEphemeris(mu_central = MU_EARTH),
        )

        params_srp = create_perturbation_model(:earth;
            j_harmonics = [2], srp_cr = 1.5, srp_alpha = 0.02u"m^2/kg",
        )
        res_srp = run_simulation(
            ics = ics, perturbation_params = params_srp,
            tspan = tspan, t_vector = t_vector, propagator_options = opts,
            output_directory = outdir,
            analytic_ephemeris = eph,
        )

        sol_base = res_base[1].solution
        sol_srp  = res_srp[1].solution

        @test string(sol_srp.retcode) == "Success"
        rf_base = SVector(sol_base.u[end][1:3]...)
        rf_srp  = SVector(sol_srp.u[end][1:3]...)
        @test norm(rf_srp - rf_base) > 0.0
        @test all(isfinite, rf_srp)
    end
end
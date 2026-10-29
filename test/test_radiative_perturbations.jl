# test/test_radiative_perturbations.jl

using Test
using LinearAlgebra
using StaticArrays
using Unitful, UnitfulAstro
using Unitful.DefaultSymbols
using DifferentialEquations
using CelestialMechanics
using CelestialMechanics.Coordinates

const MU_EARTH = 398600.4418
const MU_SUN   = 1.32712440018e11

@testset "Radiative & Thermal Perturbations" begin

    R_earth = 6378.137 # km
    r_sat_leo = SVector(7000.0, 0.0, 0.0) # km (~622 km altitude)
    v_sat_leo = SVector(0.0, 7.5, 0.0)    # km/s
    r_sun = SVector(1.496e8, 0.0, 0.0)    # km (Sun on +X axis)
    alpha = 0.02 # m^2/kg
    cr = 1.2

    @testset "Earth Radiation Pressure (Albedo + IR)" begin
        # 1. Day side (Subsolar point: r_sat along +X)
        a_day = earth_radiation_pressure(r_sat_leo, R_earth, alpha, cr, r_sun; albedo=0.30, thermal_ir_flux=237.0)
        @test norm(a_day) > 0.0
        # Should be directed radially outward (+X)
        @test a_day[1] > 0.0
        @test abs(a_day[2]) < 1e-15
        @test abs(a_day[3]) < 1e-15

        # 2. Night side (r_sat along -X)
        r_sat_night = SVector(-7000.0, 0.0, 0.0)
        a_night = earth_radiation_pressure(r_sat_night, R_earth, alpha, cr, r_sun; albedo=0.30, thermal_ir_flux=237.0)
        @test norm(a_night) > 0.0
        # At night, points radially outward (-X)
        @test a_night[1] < 0.0
        
        # Day side must be stronger than night side because day = IR + Albedo, night = IR only
        @test norm(a_day) > norm(a_night)

        # 3. Night side with zero thermal IR flux must yield zero
        a_no_ir = earth_radiation_pressure(r_sat_night, R_earth, alpha, cr, r_sun; albedo=0.30, thermal_ir_flux=0.0)
        @test norm(a_no_ir) == 0.0

        # 4. Zero area/mass or zero Cr yields zero
        @test norm(earth_radiation_pressure(r_sat_leo, R_earth, 0.0, cr, r_sun)) == 0.0
        @test norm(earth_radiation_pressure(r_sat_leo, R_earth, alpha, 0.0, r_sun)) == 0.0

        # 5. Distance attenuation: higher altitude should have lower ERP
        r_sat_geo = SVector(42164.0, 0.0, 0.0)
        a_geo = earth_radiation_pressure(r_sat_geo, R_earth, alpha, cr, r_sun; albedo=0.30, thermal_ir_flux=237.0)
        @test norm(a_day) > norm(a_geo) * 30.0 # (42164/7000)^2 ≈ 36.3
    end

    @testset "Yarkovsky Effect (Diurnal Thermal Recoil)" begin
        # Orbit in XY plane, moving counter-clockwise:
        # r = (7000, 0, 0), v = (0, 7.5, 0) -> h = r x v is along +Z.
        # Sun at +X: r_sun = (1.496e8, 0, 0).
        # Vector sat -> Sun: s_hat ≈ +X.
        # Orbit normal w_hat = +Z.
        # Transverse vector: w_hat x s_hat = +Z x +X = +Y (along velocity v_sat!).
        a_yark_pro = yarkovsky_perturbation(
            r_sat_leo, v_sat_leo, alpha, r_sun;
            cy=0.15, lag_angle_deg=25.0
        )
        @test norm(a_yark_pro) > 0.0
        # Transverse force must point along +Y (in the direction of orbital velocity)
        @test a_yark_pro[2] > 0.0
        @test abs(a_yark_pro[1]) < 1e-15
        @test abs(a_yark_pro[3]) < 1e-15

        # Retrograde spin: spin_axis along -Z
        a_yark_ret = yarkovsky_perturbation(
            r_sat_leo, v_sat_leo, alpha, r_sun;
            cy=0.15, lag_angle_deg=25.0, spin_axis=SVector(0.0, 0.0, -1.0)
        )
        @test norm(a_yark_ret) ≈ norm(a_yark_pro)
        # Retrograde force must oppose velocity (along -Y)
        @test a_yark_ret[2] < 0.0
        @test a_yark_ret[2] ≈ -a_yark_pro[2]

        # In shadow (shadow_factor = 0): force must be zero
        a_yark_shadow = yarkovsky_perturbation(
            r_sat_leo, v_sat_leo, alpha, r_sun;
            cy=0.15, shadow_factor=0.0
        )
        @test norm(a_yark_shadow) == 0.0

        # Scaling with lag angle: sin(50°) > sin(25°)
        a_yark_50 = yarkovsky_perturbation(
            r_sat_leo, v_sat_leo, alpha, r_sun;
            cy=0.15, lag_angle_deg=50.0
        )
        @test norm(a_yark_50) > norm(a_yark_pro)
    end

    @testset "Model Creation & Parameters" begin
        # Default: perturbations are nothing
        model_default = create_perturbation_model(:earth)
        @test isnothing(model_default.albedo)
        @test isnothing(model_default.thermal_ir_flux)
        @test isnothing(model_default.yarkovsky_cy)
        @test isnothing(model_default.yarkovsky_lag)
        @test isnothing(model_default.spin_axis)

        # Configured model
        model_rad = create_perturbation_model(
            :earth;
            albedo = 0.30,
            thermal_ir_flux = 237.0,
            yarkovsky_cy = 0.15,
            yarkovsky_lag = 25.0,
            spin_axis = [0.0, 0.0, 1.0],
            srp_cr = 1.2,
            srp_alpha = 0.02u"m^2/kg"
        )
        @test model_rad.albedo == 0.30
        @test model_rad.thermal_ir_flux == 237.0
        @test model_rad.yarkovsky_cy == 0.15
        @test model_rad.yarkovsky_lag == 25.0
        @test model_rad.spin_axis == [0.0, 0.0, 1.0]
    end

    @testset "Full Simulation Propagation with Cowell" begin
        eph = AnalyticEphemeris(mu_central = MU_EARTH)
        add_body!(eph, "SUN"; a = 1.0u"AU", e = 0.0167, mu = MU_SUN)

        outdir = mktempdir()

        a0, e0, i0 = 7000.0u"km", 0.001, 28.5u"°"
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

        # Perturbed: J2 + Earth Radiation Pressure (Albedo + IR)
        params_erp = create_perturbation_model(
            :earth;
            j_harmonics = [2],
            albedo = 0.30,
            thermal_ir_flux = 237.0,
            srp_cr = 1.2,
            srp_alpha = 0.05u"m^2/kg"
        )
        res_erp = run_simulation(
            ics = ics, perturbation_params = params_erp,
            tspan = tspan, t_vector = t_vector, propagator_options = opts,
            output_directory = outdir,
            analytic_ephemeris = eph,
        )

        # Perturbed: J2 + Yarkovsky effect
        params_yark = create_perturbation_model(
            :earth;
            j_harmonics = [2],
            yarkovsky_cy = 0.20,
            yarkovsky_lag = 30.0,
            srp_alpha = 0.05u"m^2/kg"
        )
        res_yark = run_simulation(
            ics = ics, perturbation_params = params_yark,
            tspan = tspan, t_vector = t_vector, propagator_options = opts,
            output_directory = outdir,
            analytic_ephemeris = eph,
        )

        sol_base = res_base[1].solution
        sol_erp  = res_erp[1].solution
        sol_yark = res_yark[1].solution

        @test string(sol_base.retcode) == "Success"
        @test string(sol_erp.retcode)  == "Success"
        @test string(sol_yark.retcode) == "Success"

        rf_base = SVector(sol_base.u[end][1:3]...)
        rf_erp  = SVector(sol_erp.u[end][1:3]...)
        rf_yark = SVector(sol_yark.u[end][1:3]...)

        # Both perturbations physically alter the trajectory relative to baseline
        @test norm(rf_erp  - rf_base) > 0.0
        @test norm(rf_yark - rf_base) > 0.0
        @test all(isfinite, rf_erp)
        @test all(isfinite, rf_yark)
    end

end

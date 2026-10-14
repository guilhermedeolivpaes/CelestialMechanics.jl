# test/test_atmospheric_drag.jl

using Test
using CelestialMechanics
using LinearAlgebra
using StaticArrays
using Dates

const PE = CelestialMechanics.PerturbationEquations
const AM = CelestialMechanics.AtmosphericModels

@testset "Atmospheric Drag & Models" begin

    @testset "drag_acceleration: basic physical properties" begin
        r = SVector(6700.0, 0.0, 0.0)      # km
        v = SVector(0.0, 7.7, 0.0)         # km/s
        rho = 1e-11                        # kg/m^3
        cd = 2.2
        am = 0.01                          # m^2/kg
        omega_rot = 7.292115e-5            # rad/s

        a_drag = PE.drag_acceleration(r, v, rho, cd, am, omega_rot)
        
        # 1. Acceleration must oppose the relative velocity vector
        v_atm = SVector(-omega_rot * r[2], omega_rot * r[1], 0.0)
        v_rel = v - v_atm
        @test dot(a_drag, v_rel) < 0.0
        
        # 2. Acceleration is zero when density is zero
        a_zero = PE.drag_acceleration(r, v, 0.0, cd, am, omega_rot)
        @test norm(a_zero) == 0.0

        # 3. Linearity with Cd and area-to-mass ratio
        a_double_cd = PE.drag_acceleration(r, v, rho, 2 * cd, am, omega_rot)
        @test norm(a_double_cd) ≈ 2 * norm(a_drag) atol=1e-12

        a_double_am = PE.drag_acceleration(r, v, rho, cd, 2 * am, omega_rot)
        @test norm(a_double_am) ≈ 2 * norm(a_drag) atol=1e-12
    end

    @testset "Harris-Priester atmospheric density" begin
        R_earth = 6378.137
        r_sun = SVector(1.496e8, 0.0, 0.0)

        # 300 km altitude on dayside
        r_day = SVector(R_earth + 300.0, 0.0, 0.0)
        rho_day = atmospheric_density(HarrisPriesterAtmosphere(), r_day, 0.0, r_sun, R_earth)
        @test rho_day > 0.0

        # 300 km altitude on nightside (opposite to Sun)
        r_night = SVector(-(R_earth + 300.0), 0.0, 0.0)
        rho_night = atmospheric_density(HarrisPriesterAtmosphere(), r_night, 0.0, r_sun, R_earth)
        @test rho_night > 0.0

        # Dayside density near diurnal bulge should exceed nightside minimum
        @test rho_day > rho_night

        # Density decreases with altitude
        r_high = SVector(R_earth + 600.0, 0.0, 0.0)
        rho_high = atmospheric_density(HarrisPriesterAtmosphere(), r_high, 0.0, r_sun, R_earth)
        @test rho_day > rho_high > 0.0

        # Above maximum table altitude (1000 km), density is 0
        r_space = SVector(R_earth + 1200.0, 0.0, 0.0)
        rho_space = atmospheric_density(HarrisPriesterAtmosphere(), r_space, 0.0, r_sun, R_earth)
        @test rho_space == 0.0

        # Default model (nothing) behaves identically to HarrisPriesterAtmosphere
        @test atmospheric_density(nothing, r_day, 0.0, r_sun, R_earth) == rho_day
    end

    @testset "Geodetic coordinates & GMST" begin
        # GMST angle must be within [0, 2π)
        theta = gmst_angle(2451545.0)
        @test 0.0 <= theta < 2π

        # Point on the equator at Greenwich meridian when GMST = 0
        r_eq = SVector(6378.137, 0.0, 0.0)
        lat, lon, h = AM.ecef_to_geodetic(r_eq)
        @test abs(lat) < 1e-10
        @test abs(lon) < 1e-10
        @test abs(h) < 1e-6 # altitude ~ 0 km on WGS-84 equator

        # Point along the Z-axis (North Pole)
        r_pole = SVector(0.0, 0.0, 6356.7523) # semi-minor axis
        lat_p, lon_p, h_p = AM.ecef_to_geodetic(r_pole)
        @test lat_p ≈ π/2 atol=1e-5
        @test abs(h_p) < 1e-3
    end

    @testset "NRLMSISE00Atmosphere configuration & extension check" begin
        # Epoch conversions
        model_dt = NRLMSISE00Atmosphere(epoch=DateTime(2026, 1, 1), F10a=150.0, F10=150.0, ap=4.0)
        @test model_dt.epoch_jd ≈ Dates.datetime2julian(DateTime(2026, 1, 1))

        model_str = NRLMSISE00Atmosphere(epoch="2026-01-01T00:00:00", F10a=150.0, F10=150.0, ap=4.0)
        @test model_str.epoch_jd ≈ model_dt.epoch_jd

        # Automatic mode (all indices nothing)
        model_auto = NRLMSISE00Atmosphere(epoch=2460000.5)
        @test isnothing(model_auto.F10a)
        @test isnothing(model_auto.F10)
        @test isnothing(model_auto.ap)

        # Incomplete indices validation
        @test_throws ArgumentError NRLMSISE00Atmosphere(epoch=2460000.5, F10=120.0)
        @test_throws ArgumentError NRLMSISE00Atmosphere(epoch=2460000.5, F10a=120.0, F10=120.0) # missing ap

        # Invalid frame validation
        @test_throws ArgumentError NRLMSISE00Atmosphere(epoch=2460000.5, frame=:UNKNOWN)

        # Calling without loading SatelliteToolboxAtmosphericModels throws helpful error
        r_test = SVector(6700.0, 0.0, 0.0)
        @test_throws ErrorException atmospheric_density(model_dt, r_test, 0.0, nothing, 6378.137)
    end

    @testset "create_perturbation_model atmosphere integration" begin
        p_hp = create_perturbation_model(:earth; drag_cd=2.2, drag_am=0.01)
        @test isnothing(p_hp.atmosphere)

        model_nrl = NRLMSISE00Atmosphere(epoch=2460000.5, F10a=100.0, F10=100.0, ap=3.0)
        p_nrl = create_perturbation_model(:earth; drag_cd=2.2, drag_am=0.01, atmosphere=model_nrl)
        @test p_nrl.atmosphere === model_nrl
    end

end

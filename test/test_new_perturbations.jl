# test/test_new_perturbations.jl
#
# Tests for newly ported perturbations:
# 1. Smooth shadow model (shadow_function_smooth)
# 2. Lorentz force perturbation (acc_lorentz)
# 3. Earth general precession perturbation (PrecessionPerturbation)
# 4. create_perturbation_model integration with new fields

using Test
using CelestialMechanics
using LinearAlgebra
using StaticArrays

@testset "New Perturbations" begin

    @testset "Smooth Shadow Function" begin
        Re = 6378.137
        r_sun = SVector(1.496e8, 0.0, 0.0)

        # Satellite directly between Earth and Sun -> full illumination (nu ≈ 1.0)
        r_illuminated = SVector(7000.0, 0.0, 0.0)
        nu_ill = shadow_function_smooth(r_illuminated, r_sun, Re)
        @test isapprox(nu_ill, 1.0, atol=1e-5)

        # Satellite in Earth's shadow (behind Earth, x < 0 and inside cylinder) -> umbra (nu ≈ 0.0)
        r_shadow = SVector(-7000.0, 0.0, 0.0)
        nu_shad = shadow_function_smooth(r_shadow, r_sun, Re)
        @test isapprox(nu_shad, 0.0, atol=1e-5)

        # Illumination factor must stay within [0.0, 1.0]
        @test 0.0 <= nu_ill <= 1.0
        @test 0.0 <= nu_shad <= 1.0
    end

    @testset "Lorentz Force Perturbation" begin
        r = SVector(7000.0, 0.0, 0.0)      # km
        v = SVector(0.0, 7.5, 0.0)        # km/s
        omega = 7.292115e-5               # rad/s (Earth)

        # Null charge -> zero acceleration
        a_zero = acc_lorentz(r, v, 0.0, omega)
        @test norm(a_zero) == 0.0

        # Positive charge -> non-zero finite acceleration
        a_charged = acc_lorentz(r, v, 1e-4, omega)
        @test norm(a_charged) > 0.0
        @test all(isfinite.(a_charged))

        # Reversing charge reverses acceleration direction
        a_neg = acc_lorentz(r, v, -1e-4, omega)
        @test a_neg ≈ -a_charged atol=1e-15
    end

    @testset "Earth General Precession" begin
        v = SVector(0.0, 7.5, 0.0)  # km/s

        # Zero velocity -> zero Coriolis acceleration
        @test norm(PrecessionPerturbation(SVector(0.0, 0.0, 0.0), 0.0)) == 0.0

        # Non-zero velocity -> finite Coriolis acceleration
        a_prec = PrecessionPerturbation(v, 0.0)
        @test norm(a_prec) > 0.0
        @test all(isfinite.(a_prec))

        # Coriolis acceleration is perpendicular to velocity: dot(a, v) == 0
        @test abs(dot(a_prec, v)) < 1e-14
    end

    @testset "create_perturbation_model Integration" begin
        # Model with all three new perturbations active
        p = create_perturbation_model(:earth,
            srp_cr = 1.0,
            srp_alpha = 0.01,
            shadow_on = true,
            lorentz_qm = 1e-5,
            lorentz_B = 8.0e15,
            precession_on = true
        )

        @test p.shadow_in_srp == true
        @test p.qm == 1e-5
        @test p.B_mag == 8.0e15
        @test p.precession_on == true

        # Model with default inactive values
        p_default = create_perturbation_model(:earth)
        @test p_default.shadow_in_srp == false
        @test isnothing(p_default.qm)
        @test isnothing(p_default.B_mag)
        @test p_default.precession_on == false
    end

end

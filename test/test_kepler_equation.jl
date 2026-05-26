# test/test_kepler_equation.jl

using Test
using CelestialMechanics
using CelestialMechanics.Coordinates

@testset "Kepler Equation: M ↔ f" begin

    @testset "Inversibilidade M → f → M" begin
        for e in [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95]
            for M_deg in [0.0, 30.0, 90.0, 135.0, 179.9, 180.1, 270.0, 350.0]
                M = deg2rad(M_deg)
                f = Coordinates.mean_to_true_anomaly(M, e)
                M2 = Coordinates.true_to_mean_anomaly(f, e)
                # Compara em [0, 2π] — a função pode retornar M ∈ (-π, π]
                diff = mod(M2 - M + π, 2π) - π
                @test abs(diff) < 1e-10
            end
        end
    end

    @testset "Inversibilidade f → M → f" begin
        for e in [0.0, 0.1, 0.5, 0.8]
            for f_deg in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
                f = deg2rad(f_deg)
                M = Coordinates.true_to_mean_anomaly(f, e)
                f2 = Coordinates.mean_to_true_anomaly(M, e)
                # Normaliza ambos para (-π, π] antes de comparar
                diff = mod(f2 - f + π, 2π) - π
                @test abs(diff) < 1e-10
            end
        end
    end

    @testset "Casos limites" begin
        # Periapsis: M = 0 → f = 0
        @test Coordinates.mean_to_true_anomaly(0.0, 0.3) ≈ 0.0 atol=1e-12
        # Apoapsis: M = π → f = π (usa float(π) para evitar MethodError com Irrational)
        @test abs(Coordinates.mean_to_true_anomaly(float(π), 0.3)) ≈ float(π) atol=1e-10
        # Órbita circular: f = M para qualquer M
        for M_deg in [0.0, 45.0, 90.0, 180.0, 270.0]
            M = deg2rad(M_deg)
            f = Coordinates.mean_to_true_anomaly(M, 0.0)
            diff = mod(f - M + π, 2π) - π
            @test abs(diff) < 1e-10
        end
    end

    @testset "Consistência com a equação de Kepler: M = E - e*sin(E)" begin
        for e in [0.1, 0.5, 0.8]
            for M_deg in [30.0, 90.0, 150.0, 210.0, 300.0]
                M = deg2rad(M_deg)
                f = Coordinates.mean_to_true_anomaly(M, e)
                cosE = (e + cos(f)) / (1 + e * cos(f))
                sinE = sqrt(1 - e^2) * sin(f) / (1 + e * cos(f))
                E = atan(sinE, cosE)
                M_check = mod2pi(E - e * sin(E))
                @test M_check ≈ mod2pi(M) atol=1e-10
            end
        end
    end

    @testset "Alta excentricidade (e = 0.95)" begin
        e = 0.95
        for M_deg in [10.0, 60.0, 120.0, 240.0, 300.0]
            M = deg2rad(M_deg)
            f = Coordinates.mean_to_true_anomaly(M, e)
            M2 = Coordinates.true_to_mean_anomaly(f, e)
            @test mod2pi(M2) ≈ mod2pi(M) atol=1e-8
        end
    end

    @testset "true_to_mean_anomaly: resultado em [0, 2π]" begin
        e = 0.4
        for f_deg in [0.0, 90.0, 180.0, 270.0, 359.9]
            M = Coordinates.true_to_mean_anomaly(deg2rad(f_deg), e)
            @test 0.0 ≤ M ≤ 2π + 1e-10
        end
    end
end
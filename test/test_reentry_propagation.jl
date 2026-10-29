# test/test_reentry_propagation.jl

using Test
using Random
using LinearAlgebra
using Statistics
using CelestialMechanics

@testset "Reentry Trajectory and Footprint Propagation" begin

    # --------------------------------------------------------------------------
    # 1. Testes do Modelo Atmosférico US76 (0 a 130 km)
    # --------------------------------------------------------------------------
    @testset "Modelo Atmosférico US76" begin
        # Nível do mar
        rho_sea = us_standard_atmosphere_1976(0.0)
        @test rho_sea ≈ 1.225 atol=1e-6
        @test us_standard_atmosphere_1976(-5.0) ≈ 1.225 atol=1e-6

        # Altitude de quebra (~78 km)
        rho_78 = us_standard_atmosphere_1976(78.0)
        @test 1.5e-5 < rho_78 < 3.5e-5

        # Termosfera / 120 km
        rho_120 = us_standard_atmosphere_1976(120.0)
        @test 1.0e-8 < rho_120 < 5.0e-8

        # Verificação da diminuição estrita com a altitude
        altitudes = collect(0.0:10.0:120.0)
        densities = [us_standard_atmosphere_1976(h) for h in altitudes]
        @test issorted(densities, rev=true)
    end

    # --------------------------------------------------------------------------
    # 2. Teste da Propagação de Fragmento Individual (Segregação por A/m)
    # --------------------------------------------------------------------------
    @testset "Propagação Individual e Segregação Aerodinâmica" begin
        # Estado inicial no ponto de ruptura (h ~ 78 km) com ângulo de reentrada suave (-1.5°)
        r_earth = 6378.137
        h_break = 78.0
        r0 = [r_earth + h_break, 0.0, 0.0]

        gamma = deg2rad(-1.5)
        v_mag = 7.60
        v0 = [v_mag * sin(gamma), v_mag * cos(gamma), 0.0]

        u_rad = r0 ./ norm(r0)
        v_tang = v0 .- dot(v0, u_rad) .* u_rad
        t_hat = normalize(v_tang)
        c_hat = normalize(cross(t_hat, u_rad))

        # Fragmento Denso: baixa razão A/m (ex.: tanque de titânio 50 kg, A = 0.4 m²)
        frag_denso = DebrisFragment(
            1, 50.0, 0.008, 0.40, 2.2, copy(r0), copy(v0), [0.0, 0.0, 0.0], true
        )

        # Fragmento Leve: alta razão A/m (ex.: painel solar 2 kg, A = 0.5 m²)
        frag_leve = DebrisFragment(
            2, 2.0, 0.250, 0.50, 2.2, copy(r0), copy(v0), [0.0, 0.0, 0.0], false
        )

        imp_denso = propagate_fragment(frag_denso, r0, t_hat, c_hat; max_duration=3600.0, save_trajectory=true)
        imp_leve  = propagate_fragment(frag_leve, r0, t_hat, c_hat; max_duration=3600.0, save_trajectory=true)

        # Ambos devem tocar o solo (h <= 0.0 km)
        @test imp_denso.trajectory[end, 8] <= 0.001
        @test imp_leve.trajectory[end, 8]  <= 0.001

        # Velocidade terminal no solo (relativa à superfície da Terra):
        # Denso (50 kg): v_term ~ 30 m/s
        # Leve (2 kg):   v_term ~ 5 m/s (flutua como paraquedas na baixa atmosfera)
        @test 20.0 <= imp_denso.v_term_ms <= 60.0
        @test 3.0  <= imp_leve.v_term_ms  <= 15.0

        # O fragmento denso cai com maior velocidade terminal de impacto
        @test imp_denso.v_term_ms > imp_leve.v_term_ms

        # E o fragmento denso cai muito mais rápido (tempo de voo menor)
        @test imp_denso.t_flight < imp_leve.t_flight

        # Trajetórias salvas devem conter histórico
        @test size(imp_denso.trajectory, 1) > 10
        @test size(imp_leve.trajectory, 1) > 10
    end

    # --------------------------------------------------------------------------
    # 3. Propagação da Nuvem Completa (propagate_debris_cloud) e Elipse de Dispersão
    # --------------------------------------------------------------------------
    @testset "Propagação da Nuvem e Debris Footprint" begin
        r_earth = 6378.137
        h_break = 78.0
        r_sat = [r_earth + h_break, 0.0, 0.0]

        gamma = deg2rad(-1.5)
        v_mag = 7.62
        v_sat = [v_mag * sin(gamma), v_mag * cos(gamma), 0.30]
        N_frag = 30
        M_sat = 750.0

        rng = MersenneTwister(999)
        event = generate_debris_cloud(
            r_sat, v_sat, N_frag;
            total_mass = M_sat,
            epoch_utc  = "2026-10-24T22:00:00Z",
            rng        = rng
        )

        footprint = propagate_debris_cloud(
            event;
            max_duration = 3600.0,
            save_trajectories = false,
            parallel = false
        )

        @test footprint.num_survived == N_frag
        @test footprint.total_surviving_mass ≈ M_sat atol=1e-6
        @test footprint.total_kinetic_energy_mj > 0.0

        # A elipse deve ter comprimento de dezenas a centenas de quilômetros
        @test footprint.footprint_length_km > 50.0
        @test footprint.footprint_width_km >= 0.0

        # Conversão para matriz de dados
        fp_mat = footprint_to_matrix(footprint)
        @test size(fp_mat) == (N_frag, 14)
        @test sum(fp_mat[:, 3]) ≈ M_sat atol=1e-6

        # Relatório textual de footprint sem erros
        io_buf = IOBuffer()
        footprint_summary(footprint; io=io_buf)
        sum_str = String(take!(io_buf))
        @test occursin("RELATÓRIO DA ELIPSE DE DISPERSÃO NO SOLO", sum_str)
        @test occursin("DIMENSÕES GEOMÉTRICAS DA ELIPSE", sum_str)
        @test occursin("SEGREGAÇÃO AERODINÂMICA", sum_str)
    end

end

# test/test_reentry_breakup.jl

using Test
using Random
using LinearAlgebra
using Statistics
using CelestialMechanics

@testset "Reentry Breakup Distributions" begin

    rng = MersenneTwister(1234)
    config = BreakupDistributionConfig()

    # --------------------------------------------------------------------------
    # 1. Configuração e Estruturas de Dados
    # --------------------------------------------------------------------------
    @testset "Configuração e Parâmetros Padrão" begin
        @test config.fraction_dense == 0.20
        @test config.log10_am_dense_mean == -2.00
        @test config.log10_am_struct_mean == -0.85
        @test config.am_min == 0.001
        @test config.am_max == 2.50
        @test config.delta_v_mean == 3.50
        @test config.delta_v_max == 10.0
        @test config.h_break_mean == 78.0
        @test config.cd_default == 2.20

        # Custom config
        custom_cfg = BreakupDistributionConfig(fraction_dense=0.30, delta_v_mean=4.0)
        @test custom_cfg.fraction_dense == 0.30
        @test custom_cfg.delta_v_mean == 4.0
    end

    # --------------------------------------------------------------------------
    # 2. Distribuição A: Razão Área/Massa (A/m) e Alocação de Massas
    # --------------------------------------------------------------------------
    @testset "Distribuição A: Razão Área/Massa (A/m) e Massas" begin
        N = 300
        total_mass = 1200.0  # kg

        am_vals, is_dense = sample_area_to_mass(N, config; rng=rng)

        @test length(am_vals) == N
        @test length(is_dense) == N
        @test all(config.am_min .<= am_vals .<= config.am_max)

        # Proporção aproximada de densos vs estruturais
        dense_ratio = count(is_dense) / N
        @test 0.10 <= dense_ratio <= 0.35

        # O A/m dos componentes densos deve ser estatisticamente menor que o dos leves
        dense_ams = am_vals[is_dense]
        struct_ams = am_vals[.!is_dense]
        @test mean(dense_ams) < mean(struct_ams)

        # Alocação de massa com conservação
        masses = sample_fragment_masses(N, total_mass, is_dense; rng=rng)
        @test length(masses) == N
        @test all(masses .> 0.0)
        @test sum(masses) ≈ total_mass atol=1e-8

        # A maior parte da massa deve estar nos fragmentos densos
        dense_mass = sum(masses[is_dense])
        @test dense_mass > 0.40 * total_mass

        # Teste da geração combinada de aerodinâmica
        am2, m2, a2, d2 = generate_fragment_aerodynamics(N, total_mass, config; rng=rng)
        @test length(a2) == N
        @test all(a2 .≈ (am2 .* m2))
        @test sum(m2) ≈ total_mass atol=1e-8
    end

    # --------------------------------------------------------------------------
    # 3. Distribuição B: Velocidade de Ejeção (Δv) e Orientação 3D
    # --------------------------------------------------------------------------
    @testset "Distribuição B: Velocidade de Ejeção (Δv)" begin
        N = 200
        v_sat = [0.0, 7.4, 1.2]  # km/s

        # Teste de direção isotrópica
        for _ in 1:50
            u = sample_isotropic_direction(rng)
            @test norm(u) ≈ 1.0 atol=1e-12
        end

        # Teste de direção esteira
        for _ in 1:50
            u_wake = sample_wake_aligned_direction(v_sat; rng=rng)
            @test norm(u_wake) ≈ 1.0 atol=1e-12
        end

        # Amostragem de magnitude de Δv
        dv_mag = sample_delta_v_magnitudes(N, config; rng=rng)
        @test length(dv_mag) == N
        @test all(config.delta_v_min .<= dv_mag .<= config.delta_v_max)
        @test 2.5 <= mean(dv_mag) <= 4.5

        # Geração de velocidades vetoriais
        delta_v_vecs, v0_vecs = generate_ejection_velocities(N, v_sat, config; rng=rng)
        @test length(delta_v_vecs) == N
        @test length(v0_vecs) == N

        for i in 1:N
            # Conversão m/s -> km/s: dv_kms ≈ dv_ms * 1e-3
            dv_norm_kms = norm(delta_v_vecs[i])
            @test (config.delta_v_min * 1e-3 - 1e-9) <= dv_norm_kms <= (config.delta_v_max * 1e-3 + 1e-9)
            # v0 = v_sat + Δv
            @test v0_vecs[i] ≈ (v_sat .+ delta_v_vecs[i]) atol=1e-12
        end
    end

    # --------------------------------------------------------------------------
    # 4. Distribuição C: Altitude de Fragmentação (h_break) e Pressão Dinâmica
    # --------------------------------------------------------------------------
    @testset "Distribuição C: Altitude de Fragmentação (h_break)" begin
        # Amostragem de altitudes
        h_samples = [sample_breakup_altitude(config; rng=rng) for _ in 1:100]
        @test all(config.h_break_min .<= h_samples .<= config.h_break_max)
        @test 74.0 <= mean(h_samples) <= 82.0

        # Pressão dinâmica q = 0.5 * ρ * v² * 1e-3 (kPa)
        # Exemplo: ρ = 2.0e-5 kg/m³, v = 7.5 km/s (7500 m/s)
        # q = 0.5 * 2.0e-5 * (7500)^2 * 1e-3 = 0.5 * 2.0e-5 * 5.625e7 * 1e-3 = 562.5 * 1e-3 = 0.5625 kPa * 1e3?
        # 0.5 * 2e-5 * 56250000 = 562.5 Pa = 0.5625 kPa.
        rho_test = 2.0e-5
        v_test = 7.5
        q_calc = compute_dynamic_pressure(rho_test, v_test)
        @test q_calc ≈ 0.5625 atol=1e-6

        # Teste de condição de quebra
        @test is_breakup_condition_reached(77.0, 78.0) == true
        @test is_breakup_condition_reached(79.0, 78.0) == false

        # Teste de superação de pressão dinâmica
        @test is_dynamic_pressure_exceeded(35.0, 30.0) == true
        @test is_dynamic_pressure_exceeded(25.0, 30.0) == false
    end

    # --------------------------------------------------------------------------
    # 5. Integração Completa: Geração da Nuvem de Fragmentos (generate_debris_cloud)
    # --------------------------------------------------------------------------
    @testset "Geração da Nuvem de Detritos e Diagnóstico" begin
        # Posição e velocidade realistas de reentrada (altitude ~78 km)
        r_earth = 6378.137
        h_target = 78.0
        r_mag = r_earth + h_target
        r_sat = [r_mag, 0.0, 0.0]
        v_sat = [0.0, 7.65, 0.0]
        N_frag = 150
        M_sat = 850.0  # kg

        event = generate_debris_cloud(
            r_sat,
            v_sat,
            N_frag;
            total_mass=M_sat,
            config=config,
            epoch_utc="2026-10-24T22:00:00Z",
            t_seconds=1250.0,
            rng=rng
        )

        @test event.num_fragments == N_frag
        @test event.total_mass == M_sat
        @test event.h_break ≈ h_target atol=1e-6
        @test length(event.fragments) == N_frag

        # Conservação rigorosa de massa
        mass_sum = sum(f.mass for f in event.fragments)
        @test mass_sum ≈ M_sat atol=1e-8

        # Todos os fragmentos devem ter velocidade perturbada realisticamente
        for f in event.fragments
            @test f.cd == config.cd_default
            @test f.r0 == r_sat
            @test f.v0 ≈ (v_sat .+ f.delta_v) atol=1e-12
            @test f.area ≈ (f.area_to_mass * f.mass) atol=1e-10
        end

        # Conversão para matriz de dados
        mat = debris_to_matrix(event)
        @test size(mat) == (N_frag, 13)
        @test mat[:, 1] == collect(1.0:Float64(N_frag))
        @test sum(mat[:, 2]) ≈ M_sat atol=1e-8

        # Execução do relatório textual (não deve lançar erro)
        io_buf = IOBuffer()
        debris_summary(event; io=io_buf)
        summary_str = String(take!(io_buf))
        @test occursin("RELATÓRIO DE FRAGMENTAÇÃO ATMOSFÉRICA", summary_str)
        @test occursin("78.00 km", summary_str)
        @test occursin("850.00 kg", summary_str)
    end

end

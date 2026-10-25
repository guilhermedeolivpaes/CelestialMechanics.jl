# src/ReentryBreakup/ReentryBreakup.jl

"""
    ReentryBreakup

Módulo central para modelagem de fragmentação estocástica e propagação com arrasto de satélites
durante reentrada atmosférica em LEO.

Organiza e integra os pilares fundamentais da reentrada:
1. `BreakupTypes`: Tipos de dados, structs e configurações (`DebrisFragment`, `BreakupDistributionConfig`, `BreakupEvent`).
2. `AreaToMassDistribution`: Distribuição bimodal de Razão Área/Massa (A/m) e conservação rigorosa de massa.
3. `EjectionVelocityDistribution`: Distribuição de velocidade de ejeção relativa (Δv) e orientação esférica 3D uniforme.
4. `BreakupAltitudeDistribution`: Distribuição estocástica da altitude de fragmentação (h_break) e critérios de pressão dinâmica.
5. `ReentryAtmosphere`: Atmosfera padrão contínua U.S. Standard 1976 (US76) até a superfície (0 a 130 km).
6. `ImpactFootprint`: Estruturas e diagnósticos da Elipse de Dispersão no Solo (*Debris Footprint*).
7. `TrajectoryPropagator`: Propagador orbital e atmosférico com arrasto, gravidade J2 e detecção de impacto no solo.

Fornece as funções de alto nível:
- `generate_debris_cloud`: Gera a nuvem no instante de ruptura.
- `propagate_debris_cloud`: Propaga todos os fragmentos da nuvem com arrasto até o solo.
- `debris_summary` / `footprint_summary`: Relatórios estatísticos detalhados.
- `debris_to_matrix` / `footprint_to_matrix`: Exportação numérica para matrizes / Python.
"""
module ReentryBreakup

using Reexport
using Random
using LinearAlgebra
using Statistics
using Printf

# Inclusão dos submódulos estruturais
include("BreakupTypes.jl")
include("AreaToMassDistribution.jl")
include("EjectionVelocityDistribution.jl")
include("BreakupAltitudeDistribution.jl")
include("ReentryAtmosphere.jl")
include("ImpactFootprint.jl")
include("TrajectoryPropagator.jl")

# Reexportação dos submódulos para consumo direto
@reexport using .BreakupTypes
@reexport using .AreaToMassDistribution
@reexport using .EjectionVelocityDistribution
@reexport using .BreakupAltitudeDistribution
@reexport using .ReentryAtmosphere
@reexport using .ImpactFootprint
@reexport using .TrajectoryPropagator

export generate_debris_cloud, debris_summary, debris_to_matrix,
       propagate_debris_cloud, propagate_fragment,
       us_standard_atmosphere_1976, US76_LAYER_TABLE,
       ImpactPoint, DebrisFootprint, footprint_summary, footprint_to_matrix

"""
    generate_debris_cloud(
        r_sat::Vector{Float64},
        v_sat::Vector{Float64},
        N::Int = 100;
        total_mass::Float64 = 1000.0,
        config::BreakupDistributionConfig = BreakupDistributionConfig(),
        epoch_utc::String = "",
        t_seconds::Float64 = 0.0,
        rng::AbstractRNG = Random.GLOBAL_RNG
    ) -> BreakupEvent

Gera a nuvem completa de fragmentos (`BreakupEvent`) no ponto exato de ruptura do satélite pai.

Aplica de forma consistente e simultânea:
1. **Distribuição A (A/m)**: Bimodal (densos vs estruturais) com massas em lei de potência conservando `total_mass`.
2. **Distribuição B (Δv)**: Módulo gaussiano suave (0.5 a 10.0 m/s) com direção 3D isotrópica ou esteira.
3. **Distribuição C (h_break)**: Determinação da altitude do evento a partir do estado inercial `r_sat`.
"""
function generate_debris_cloud(
    r_sat::Vector{Float64},
    v_sat::Vector{Float64},
    N::Int = 100;
    total_mass::Float64 = 1000.0,
    config::BreakupDistributionConfig = BreakupDistributionConfig(),
    epoch_utc::String = "",
    t_seconds::Float64 = 0.0,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    # 1. Altitude no momento da quebra (calculada geometricamente a partir de r_sat)
    r_norm = norm(r_sat)
    r_earth = 6378.137  # Raio equatorial WGS84 em km
    h_break = r_norm - r_earth

    # 2. Amostragem da Distribuição A: Aerodinâmica (A/m, massas e áreas com conservação de massa)
    am_values, masses, areas, is_dense = generate_fragment_aerodynamics(
        N, total_mass, config; rng=rng
    )

    # 3. Amostragem da Distribuição B: Velocidade de ejeção (Δv e v0)
    delta_v_vecs, v0_vecs = generate_ejection_velocities(
        N, v_sat, config; rng=rng
    )

    # 4. Construção da lista de fragmentos
    fragments = Vector{DebrisFragment}(undef, N)
    for i in 1:N
        fragments[i] = DebrisFragment(
            i,
            masses[i],
            am_values[i],
            areas[i],
            config.cd_default,
            copy(r_sat),       # Co-localizados inicialmente no ponto de ruptura
            v0_vecs[i],        # Velocidade inercial com o Δv adicionado
            delta_v_vecs[i],   # Velocidade de ejeção relativa
            is_dense[i]
        )
    end

    return BreakupEvent(
        epoch_utc,
        t_seconds,
        copy(r_sat),
        copy(v_sat),
        h_break,
        total_mass,
        N,
        fragments
    )
end

"""
    debris_summary(event::BreakupEvent; io::IO = stdout)

Exibe um relatório detalhado e formatado com métricas estatísticas da nuvem de fragmentos gerada.
"""
function debris_summary(event::BreakupEvent; io::IO = stdout)
    N = event.num_fragments
    masses = [f.mass for f in event.fragments]
    ams = [f.area_to_mass for f in event.fragments]
    dvs_ms = [norm(f.delta_v) * 1e3 for f in event.fragments]
    dense_count = count(f.is_dense_component for f in event.fragments)
    struct_count = N - dense_count

    total_sampled_mass = sum(masses)
    dense_mass = sum(f.mass for f in event.fragments if f.is_dense_component)
    struct_mass = sum(f.mass for f in event.fragments if !f.is_dense_component)

    println(io, "================================================================================")
    println(io, "                RELATÓRIO DE FRAGMENTAÇÃO ATMOSFÉRICA (REENTRY)                ")
    println(io, "================================================================================")
    @printf(io, "• Altitude de Ruptura (h_break) : %.2f km
", event.h_break)
    @printf(io, "• Velocidade do Satélite        : %.3f km/s
", norm(event.v_sat))
    @printf(io, "• Total de Fragmentos Gerados   : %d
", N)
    @printf(io, "• Massa Total Pai / Somatório   : %.2f kg / %.2f kg (Erro: %.1e kg)
",
            event.total_mass, total_sampled_mass, abs(total_sampled_mass - event.total_mass))
    println(io, "--------------------------------------------------------------------------------")
    println(io, "  1. CLASSIFICAÇÃO DOS FRAGMENTOS:")
    @printf(io, "     - Componentes Densos (Tanques/Motores) : %d (%.1f%%) | Massa: %.2f kg (%.1f%%)
",
            dense_count, (dense_count / N) * 100.0, dense_mass, (dense_mass / total_sampled_mass) * 100.0)
    @printf(io, "     - Componentes Estruturais / Leves      : %d (%.1f%%) | Massa: %.2f kg (%.1f%%)
",
            struct_count, (struct_count / N) * 100.0, struct_mass, (struct_mass / total_sampled_mass) * 100.0)
    println(io, "--------------------------------------------------------------------------------")
    println(io, "  2. DISTRIBUIÇÃO RAZÃO ÁREA/MASSA (A/m):")
    @printf(io, "     - Mínimo / Máximo : %.5f m²/kg  /  %.4f m²/kg
", minimum(ams), maximum(ams))
    @printf(io, "     - Média ± Desvio  : %.4f ± %.4f m²/kg
", mean(ams), std(ams))
    @printf(io, "     - Mediana         : %.4f m²/kg
", median(ams))
    println(io, "--------------------------------------------------------------------------------")
    println(io, "  3. DISTRIBUIÇÃO VELOCIDADE DE EJEÇÃO (Δv):")
    @printf(io, "     - Mínimo / Máximo : %.2f m/s  /  %.2f m/s
", minimum(dvs_ms), maximum(dvs_ms))
    @printf(io, "     - Média ± Desvio  : %.2f ± %.2f m/s
", mean(dvs_ms), std(dvs_ms))
    println(io, "================================================================================")
end

"""
    debris_to_matrix(event::BreakupEvent) -> Matrix{Float64}

Converte os dados da nuvem de fragmentos para uma matriz numérica para fácil exportação e análise:
Colunas: [id, mass_kg, am_m2_kg, area_m2, cd, rx_km, ry_km, rz_km, vx_kms, vy_kms, vz_kms, dv_mag_ms, is_dense]
"""
function debris_to_matrix(event::BreakupEvent)
    N = event.num_fragments
    mat = zeros(Float64, N, 13)
    for (i, f) in enumerate(event.fragments)
        mat[i, 1]  = Float64(f.id)
        mat[i, 2]  = f.mass
        mat[i, 3]  = f.area_to_mass
        mat[i, 4]  = f.area
        mat[i, 5]  = f.cd
        mat[i, 6]  = f.r0[1]
        mat[i, 7]  = f.r0[2]
        mat[i, 8]  = f.r0[3]
        mat[i, 9]  = f.v0[1]
        mat[i, 10] = f.v0[2]
        mat[i, 11] = f.v0[3]
        mat[i, 12] = norm(f.delta_v) * 1e3  # m/s
        mat[i, 13] = f.is_dense_component ? 1.0 : 0.0
    end
    return mat
end

end # module ReentryBreakup

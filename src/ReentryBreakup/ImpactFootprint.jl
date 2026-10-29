# src/ReentryBreakup/ImpactFootprint.jl

"""
    ImpactFootprint

Módulo para armazenamento, estatística e caracterização da Elipse de Dispersão de Detritos no Solo
(*Debris Footprint*) decorrente da reentrada atmosférica.
"""
module ImpactFootprint

using LinearAlgebra
using Printf
using Statistics

using ..BreakupTypes: DebrisFragment, BreakupEvent

export ImpactPoint, DebrisFootprint, footprint_summary, footprint_to_matrix

"""
    ImpactPoint

Registra os dados geográficos, temporais e mecânicos de um fragmento no exato momento de impacto com o solo.

Campos:
- `id::Int`: Identificador do fragmento.
- `is_dense::Bool`: Se é componente denso (tanque/motor) ou estrutural leve.
- `mass::Float64`: Massa do fragmento em kg.
- `area_to_mass::Float64`: Razão área/massa em m²/kg.
- `t_flight::Float64`: Tempo total de voo da quebra até o impacto em segundos (s).
- `r_impact_eci::Vector{Float64}`: Posição de impacto no referencial inercial ECI (km).
- `v_impact_eci::Vector{Float64}`: Velocidade terminal no referencial inercial ECI (km/s).
- `v_term_ms::Float64`: Velocidade terminal de queda no impacto em m/s.
- `kinetic_energy_joules::Float64`: Energia cinética de impacto (E_k = 0.5 * m * v² em Joules).
- `latitude_deg::Float64`: Latitude geodésica WGS-84 de impacto (graus).
- `longitude_deg::Float64`: Longitude geodésica WGS-84 de impacto (graus).
- `downrange_km::Float64`: Distância ao longo da trajetória de descida (*along-track*, em km).
- `crossrange_km::Float64`: Desvio lateral perpendicular à trajetória (*cross-track*, em km).
- `trajectory::Matrix{Float64}`: Histórico da trajetória [t_s, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms, h_km, lat_deg, lon_deg].
"""
struct ImpactPoint
    id::Int
    is_dense::Bool
    mass::Float64
    area_to_mass::Float64
    t_flight::Float64
    r_impact_eci::Vector{Float64}
    v_impact_eci::Vector{Float64}
    v_term_ms::Float64
    kinetic_energy_joules::Float64
    latitude_deg::Float64
    longitude_deg::Float64
    downrange_km::Float64
    crossrange_km::Float64
    trajectory::Matrix{Float64}
end

"""
    DebrisFootprint

Estrutura agregadora da elipse de dispersão de detritos gerada pela propagação com arrasto da nuvem.

Campos:
- `event::BreakupEvent`: Evento pai da quebra.
- `impact_points::Vector{ImpactPoint}`: Lista de pontos de impacto no solo.
- `num_survived::Int`: Quantidade de fragmentos sobreviventes.
- `total_surviving_mass::Float64`: Massa total que atingiu o solo (kg).
- `total_kinetic_energy_mj::Float64`: Energia cinética somada no solo (Megajoules, MJ).
- `footprint_length_km::Float64`: Comprimento total da elipse de dispersão (along-track, km).
- `footprint_width_km::Float64`: Largura máxima transversal da elipse (cross-track, km).
- `mean_flight_time_s::Float64`: Tempo médio de queda dos fragmentos (s).
"""
struct DebrisFootprint
    event::BreakupEvent
    impact_points::Vector{ImpactPoint}
    num_survived::Int
    total_surviving_mass::Float64
    total_kinetic_energy_mj::Float64
    footprint_length_km::Float64
    footprint_width_km::Float64
    mean_flight_time_s::Float64
end

"""
    footprint_summary(fp::DebrisFootprint; io::IO = stdout)

Exibe um relatório estruturado e detalhado com as métricas da elipse de dispersão (*footprint*) no solo.
"""
function footprint_summary(fp::DebrisFootprint; io::IO = stdout)
    N = fp.num_survived
    if N == 0
        println(io, "Nenhum fragmento atingiu a superfície (todos foram desintegrados/queimados).")
        return
    end

    lats = [p.latitude_deg for p in fp.impact_points]
    lons = [p.longitude_deg for p in fp.impact_points]
    v_terms = [p.v_term_ms for p in fp.impact_points]
    t_flights = [p.t_flight for p in fp.impact_points]
    energies = [p.kinetic_energy_joules for p in fp.impact_points]
    downranges = [p.downrange_km for p in fp.impact_points]

    dense_survived = count(p.is_dense for p in fp.impact_points)
    struct_survived = N - dense_survived

    println(io, "================================================================================")
    println(io, "         RELATÓRIO DA ELIPSE DE DISPERSÃO NO SOLO (DEBRIS FOOTPRINT)            ")
    println(io, "================================================================================")
    @printf(io, "• Fragmentos Sobreviventes no Solo : %d de %d (%.1f%%)
",
            N, fp.event.num_fragments, (N / fp.event.num_fragments) * 100.0)
    @printf(io, "• Massa Total no Solo              : %.2f kg (de %.2f kg pai)
",
            fp.total_surviving_mass, fp.event.total_mass)
    @printf(io, "• Energia Cinética Total Impacto   : %.2f MJ
", fp.total_kinetic_energy_mj)
    println(io, "--------------------------------------------------------------------------------")
    println(io, "  1. DIMENSÕES GEOMÉTRICAS DA ELIPSE DE DISPERSÃO:")
    @printf(io, "     - Comprimento Longitudinal (Downrange) : %.2f km
", fp.footprint_length_km)
    @printf(io, "     - Largura Transversal (Crossrange)     : %.2f km
", fp.footprint_width_km)
    @printf(io, "     - Intervalo de Latitudes no Solo       : [%.4f°, %.4f°]
", minimum(lats), maximum(lats))
    @printf(io, "     - Intervalo de Longitudes no Solo      : [%.4f°, %.4f°]
", minimum(lons), maximum(lons))
    println(io, "--------------------------------------------------------------------------------")
    println(io, "  2. DINÂMICA DE QUEDA E TEMPO DE VOO:")
    @printf(io, "     - Tempo de Descida (Mín / Méd / Máx)   : %.1f s / %.1f s / %.1f s (%.1f a %.1f min)
",
            minimum(t_flights), fp.mean_flight_time_s, maximum(t_flights),
            minimum(t_flights)/60.0, maximum(t_flights)/60.0)
    @printf(io, "     - Velocidade Terminal de Queda         : Mín %.1f m/s | Média %.1f m/s | Máx %.1f m/s
",
            minimum(v_terms), mean(v_terms), maximum(v_terms))
    println(io, "--------------------------------------------------------------------------------")
    println(io, "  3. SEGREGAÇÃO AERODINÂMICA (Peneira por A/m):")
    dense_pts = filter(p -> p.is_dense, fp.impact_points)
    struct_pts = filter(p -> !p.is_dense, fp.impact_points)
    if !isempty(dense_pts)
        d_down = [p.downrange_km for p in dense_pts]
        @printf(io, "     - Densos (%d peças)      : Alcance médio %.1f km | V_term média: %.1f m/s
",
                length(dense_pts), mean(d_down), mean([p.v_term_ms for p in dense_pts]))
    end
    if !isempty(struct_pts)
        s_down = [p.downrange_km for p in struct_pts]
        @printf(io, "     - Estruturais (%d peças) : Alcance médio %.1f km | V_term média: %.1f m/s
",
                length(struct_pts), mean(s_down), mean([p.v_term_ms for p in struct_pts]))
    end
    println(io, "================================================================================")
end

"""
    footprint_to_matrix(fp::DebrisFootprint) -> Matrix{Float64}

Converte todos os pontos de impacto para matriz numérica pronta para interop ou plotagem em Python.
Colunas:
1: id
2: is_dense (1.0 ou 0.0)
3: mass (kg)
4: area_to_mass (m²/kg)
5: t_flight (s)
6: v_term (m/s)
7: kinetic_energy (J)
8: latitude (deg)
9: longitude (deg)
10: downrange (km)
11: crossrange (km)
12: rx_eci (km)
13: ry_eci (km)
14: rz_eci (km)
"""
function footprint_to_matrix(fp::DebrisFootprint)
    N = fp.num_survived
    mat = zeros(Float64, N, 14)
    for (i, p) in enumerate(fp.impact_points)
        mat[i, 1]  = Float64(p.id)
        mat[i, 2]  = p.is_dense ? 1.0 : 0.0
        mat[i, 3]  = p.mass
        mat[i, 4]  = p.area_to_mass
        mat[i, 5]  = p.t_flight
        mat[i, 6]  = p.v_term_ms
        mat[i, 7]  = p.kinetic_energy_joules
        mat[i, 8]  = p.latitude_deg
        mat[i, 9]  = p.longitude_deg
        mat[i, 10] = p.downrange_km
        mat[i, 11] = p.crossrange_km
        mat[i, 12] = p.r_impact_eci[1]
        mat[i, 13] = p.r_impact_eci[2]
        mat[i, 14] = p.r_impact_eci[3]
    end
    return mat
end

end # module ImpactFootprint

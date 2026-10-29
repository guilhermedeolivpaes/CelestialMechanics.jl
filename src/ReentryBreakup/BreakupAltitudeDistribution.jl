# src/ReentryBreakup/BreakupAltitudeDistribution.jl

"""
    BreakupAltitudeDistribution

Módulo responsável pela amostragem da altitude de fragmentação (h_break) e modelagem dos critérios
aerodinâmicos de ruptura (pressão dinâmica crítica) durante a reentrada atmosférica.

Em reentradas não-controladas ou destrutivas de LEO, o satélite sofre desintegração estrutural
quando as forças de pressão dinâmica q = 0.5 * ρ * v² e o aquecimento térmico superam a rigidez
das juntas e carenagens, o que ocorre tipicamente na faixa de 75 a 85 km (nominal ~78 km).
"""
module BreakupAltitudeDistribution

using Random
using LinearAlgebra

using ..BreakupTypes: BreakupDistributionConfig

export sample_breakup_altitude, compute_dynamic_pressure,
       is_breakup_condition_reached, is_dynamic_pressure_exceeded

"""
    sample_breakup_altitude(
        config::BreakupDistributionConfig;
        rng::AbstractRNG = Random.GLOBAL_RNG
    ) -> Float64

Sorteia a altitude estocástica de fragmentação h_break em quilômetros (km), amostrada a partir
de uma distribuição Gaussiana em torno da cota nominal (config.h_break_mean) com desvio (config.h_break_std),
estritamente truncada nos limites físicos [config.h_break_min, config.h_break_max].
"""
function sample_breakup_altitude(
    config::BreakupDistributionConfig;
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    h_raw = config.h_break_mean + config.h_break_std * randn(rng)
    return clamp(h_raw, config.h_break_min, config.h_break_max)
end

"""
    compute_dynamic_pressure(rho_kg_m3::Float64, v_rel_kms::Float64) -> Float64

Calcula a pressão dinâmica local q em quilopascais (kPa):
q = 0.5 * ρ * v_rel² * 1e-3
onde:
- `rho_kg_m3`: Densidade atmosférica local em kg/m³.
- `v_rel_kms`: Velocidade relativa ao escoamento atmosférico em km/s (convertida internamente para m/s).

Retorna:
- `q_kPa`: Pressão dinâmica em kPa (1 kPa = 1000 N/m²).
"""
function compute_dynamic_pressure(rho_kg_m3::Float64, v_rel_kms::Float64)
    v_ms = v_rel_kms * 1e3
    q_pa = 0.5 * rho_kg_m3 * (v_ms^2)
    return q_pa * 1e-3  # kPa
end

"""
    is_breakup_condition_reached(altitude_km::Float64, h_break_sampled::Float64) -> Bool

Verifica se a altitude atual do satélite cruzou a altitude estocástica de fragmentação estipulada.
"""
function is_breakup_condition_reached(altitude_km::Float64, h_break_sampled::Float64)
    return altitude_km <= h_break_sampled
end

"""
    is_dynamic_pressure_exceeded(q_kPa::Float64, q_crit_kPa::Float64 = 30.0) -> Bool

Critério físico alternativo de ruptura estrutural baseado na superação da pressão dinâmica crítica
q_crit (tipicamente 20 a 45 kPa para estruturas espaciais típicas em LEO).
"""
function is_dynamic_pressure_exceeded(q_kPa::Float64, q_crit_kPa::Float64 = 30.0)
    return q_kPa >= q_crit_kPa
end

end # module BreakupAltitudeDistribution

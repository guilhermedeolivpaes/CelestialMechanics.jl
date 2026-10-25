# src/ReentryBreakup/AreaToMassDistribution.jl

"""
    AreaToMassDistribution

Módulo responsável pela amostragem estatística da Razão Área/Massa (A/m) e alocação de massas
individuais dos fragmentos gerados na quebra atmosférica de satélites em órbita LEO.

Baseado nas formulações bimodais do NASA Standard Breakup Model e ESA DRAMA para fragmentação
estrutural aero-termodinâmica.
"""
module AreaToMassDistribution

using Random
using LinearAlgebra
using Statistics

using ..BreakupTypes: BreakupDistributionConfig

export sample_area_to_mass, sample_fragment_masses, generate_fragment_aerodynamics

"""
    sample_area_to_mass(N::Int, config::BreakupDistributionConfig; rng=Random.GLOBAL_RNG) -> (Vector{Float64}, BitVector)

Sorteia a razão Área/Massa (A/m em m²/kg) para `N` fragmentos utilizando uma distribuição
bimodal log-normal:
1. Componentes Densos (tanques de titânio, baterias, giroscópios/reaction wheels, mancais de motores):
   - Média de log10(A/m) ~ config.log10_am_dense_mean (ex.: -2.0 -> 0.01 m²/kg)
   - Baixo A/m -> alta penetração balística, atingem altitudes mais baixas / solo.
2. Componentes Estruturais / Leves (painéis de favo de mel, mantas MLI, suportes, antenas):
   - Média de log10(A/m) ~ config.log10_am_struct_mean (ex.: -0.85 -> ~0.14 m²/kg)
   - Alto A/m -> rápida desaceleração aerodinâmica e queima / ablação precoce.

Retorna:
- `am_values::Vector{Float64}`: Vetor de A/m (m²/kg).
- `is_dense::BitVector`: Máscara booleana indicando se o fragmento é da classe densa.
"""
function sample_area_to_mass(
    N::Int,
    config::BreakupDistributionConfig;
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    am_values = Vector{Float64}(undef, N)
    is_dense = falses(N)

    for i in 1:N
        # Sorteio da categoria (denso vs estrutural)
        if rand(rng) < config.fraction_dense
            is_dense[i] = true
            mu = config.log10_am_dense_mean
            sigma = config.log10_am_dense_std
        else
            is_dense[i] = false
            mu = config.log10_am_struct_mean
            sigma = config.log10_am_struct_std
        end

        # Amostragem normal para log10(A/m)
        log10_am = mu + sigma * randn(rng)
        am = 10.0^log10_am

        # Truncamento nos limites físicos
        am_clamped = clamp(am, config.am_min, config.am_max)
        am_values[i] = am_clamped
    end

    return am_values, is_dense
end

"""
    sample_fragment_masses(
        N::Int,
        total_mass::Float64,
        is_dense::BitVector;
        alpha::Float64 = 0.80,
        dense_mass_fraction::Float64 = 0.65,
        rng::AbstractRNG = Random.GLOBAL_RNG
    ) -> Vector{Float64}

Gera as massas individuais (kg) dos `N` fragmentos a partir de uma distribuição tipo lei de potência
(power-law) com conservação estrita da massa total do satélite:
sum(masses) == total_mass.

Parâmetros:
- `N`: Número total de fragmentos.
- `total_mass`: Massa total do satélite pai em kg.
- `is_dense`: Máscara identificando fragmentos densos.
- `alpha`: Expoente da lei de potência (padrão 0.80).
- `dense_mass_fraction`: Fração da massa total alocada aos componentes densos (padrão 0.65, ou seja, 65%).
"""
function sample_fragment_masses(
    N::Int,
    total_mass::Float64,
    is_dense::BitVector;
    alpha::Float64 = 0.80,
    dense_mass_fraction::Float64 = 0.65,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    masses = zeros(Float64, N)
    n_dense = count(is_dense)
    n_struct = N - n_dense

    # Função interna para sortear pesos com lei de potência truncada
    function sample_power_law_weights(cnt::Int)
        if cnt == 0
            return Float64[]
        elseif cnt == 1
            return [1.0]
        end
        weights = Vector{Float64}(undef, cnt)
        for j in 1:cnt
            u = rand(rng) * 0.95  # corte superior a 95% para estabilidade numérica
            weights[j] = (1.0 - u)^(-1.0 / alpha)
        end
        return weights
    end

    # Se todos forem densos ou todos forem estruturais
    if n_dense == 0
        w = sample_power_law_weights(n_struct)
        masses .= (w ./ sum(w)) .* total_mass
        return masses
    elseif n_struct == 0
        w = sample_power_law_weights(n_dense)
        masses .= (w ./ sum(w)) .* total_mass
        return masses
    end

    # Divisão realista da massa entre as duas categorias
    m_dense_target = total_mass * dense_mass_fraction
    m_struct_target = total_mass * (1.0 - dense_mass_fraction)

    w_dense = sample_power_law_weights(n_dense)
    w_struct = sample_power_law_weights(n_struct)

    # Normalização de cada grupo
    m_dense_norm = (w_dense ./ sum(w_dense)) .* m_dense_target
    m_struct_norm = (w_struct ./ sum(w_struct)) .* m_struct_target

    idx_dense = 1
    idx_struct = 1
    for i in 1:N
        if is_dense[i]
            masses[i] = m_dense_norm[idx_dense]
            idx_dense += 1
        else
            masses[i] = m_struct_norm[idx_struct]
            idx_struct += 1
        end
    end

    # Ajuste fino para garantia de conservação exata da massa numérica
    masses .*= (total_mass / sum(masses))

    return masses
end

"""
    generate_fragment_aerodynamics(
        N::Int,
        total_mass::Float64,
        config::BreakupDistributionConfig;
        rng::AbstractRNG = Random.GLOBAL_RNG
    ) -> Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, BitVector}

Função de conveniência que calcula em conjunto:
- `am_values`: Razão Área/Massa (m²/kg).
- `masses`: Massa de cada fragmento (kg).
- `areas`: Área aerodinâmica de seção transversal em m² (Area = A/m * mass).
- `is_dense`: Máscara indicativa dos fragmentos densos.
"""
function generate_fragment_aerodynamics(
    N::Int,
    total_mass::Float64,
    config::BreakupDistributionConfig;
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    am_values, is_dense = sample_area_to_mass(N, config; rng=rng)
    masses = sample_fragment_masses(N, total_mass, is_dense; rng=rng)
    areas = am_values .* masses

    return am_values, masses, areas, is_dense
end

end # module AreaToMassDistribution

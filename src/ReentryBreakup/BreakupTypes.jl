# src/ReentryBreakup/BreakupTypes.jl

module BreakupTypes

using LinearAlgebra

export DebrisFragment, BreakupDistributionConfig, BreakupEvent

"""
    DebrisFragment

Estrutura representativa de um único fragmento gerado após a quebra estrutural do satélite.

Campos:
- `id::Int`: Identificador único do fragmento (1 a N).
- `mass::Float64`: Massa do fragmento em quilogramas (kg).
- `area_to_mass::Float64`: Razão área/massa em m²/kg (A/m).
- `area::Float64`: Área aerodinâmica de seção transversal em m² (A = (A/m) * mass).
- `cd::Float64`: Coeficiente de arrasto aerodinâmico (adimensional, tipicamente 2.0 a 2.4).
- `r0::Vector{Float64}`: Vetor posição inicial no referencial inercial ECI (km) [x, y, z].
- `v0::Vector{Float64}`: Vetor velocidade inicial no referencial inercial ECI (km/s) [vx, vy, vz].
- `delta_v::Vector{Float64}`: Vetor de velocidade de ejeção relativa ao satélite (km/s).
- `is_dense_component::Bool`: Booleano identificando se pertence à categoria de blocos densos ou chapas leves.
"""
struct DebrisFragment
    id::Int
    mass::Float64
    area_to_mass::Float64
    area::Float64
    cd::Float64
    r0::Vector{Float64}
    v0::Vector{Float64}
    delta_v::Vector{Float64}
    is_dense_component::Bool
end

"""
    BreakupDistributionConfig

Configuração paramétrica das 3 distribuições estatísticas da fragmentação.

# Razão Área/Massa (A/m):
- `fraction_dense::Float64`: Proporção de fragmentos densos (padrão 0.20, ou seja, 20% tanques/motores, 80% chapas/painéis).
- `log10_am_dense_mean::Float64`: Média de log10(A/m) para blocos densos (padrão -2.0 -> 0.01 m²/kg).
- `log10_am_dense_std::Float64`: Desvio padrão de log10(A/m) para blocos densos (padrão 0.35).
- `log10_am_struct_mean::Float64`: Média de log10(A/m) para componentes estruturais/leves (padrão -0.85 -> ~0.14 m²/kg).
- `log10_am_struct_std::Float64`: Desvio padrão de log10(A/m) para componentes estruturais (padrão 0.40).
- `am_min::Float64`: Limite físico inferior de A/m em m²/kg (padrão 0.001 m²/kg).
- `am_max::Float64`: Limite físico superior de A/m em m²/kg (padrão 2.50 m²/kg).

# Velocidade de Ejeção (Δv):
- `delta_v_mean::Float64`: Média da velocidade de ejeção em m/s (padrão 3.5 m/s).
- `delta_v_std::Float64`: Desvio padrão da velocidade de ejeção em m/s (padrão 1.5 m/s).
- `delta_v_min::Float64`: Limite mínimo de ejeção em m/s (padrão 0.2 m/s).
- `delta_v_max::Float64`: Valor máximo de corte para Δv em m/s (padrão 10.0 m/s).
- `direction_mode::Symbol`: `:isotropic` (distribuição esférica 3D uniforme) ou `:wake_aligned` (preferência na esteira).

# Altitude de Fragmentação (h_break):
- `h_break_mean::Float64`: Altitude nominal média de ruptura em km (padrão 78.0 km).
- `h_break_std::Float64`: Incerteza/desvio na altitude de ruptura em km (padrão 2.5 km).
- `h_break_min::Float64`: Cota mínima de quebra em km (padrão 65.0 km).
- `h_break_max::Float64`: Cota máxima de quebra em km (padrão 92.0 km).
- `cd_default::Float64`: Coeficiente de arrasto padrão dos fragmentos (padrão 2.2).
"""
Base.@kwdef struct BreakupDistributionConfig
    # A/m bimodal
    fraction_dense::Float64       = 0.20
    log10_am_dense_mean::Float64  = -2.00   # 10^(-2.0) = 0.01 m^2/kg
    log10_am_dense_std::Float64   = 0.35
    log10_am_struct_mean::Float64 = -0.85   # 10^(-0.85) ~ 0.14 m^2/kg
    log10_am_struct_std::Float64  = 0.40
    am_min::Float64               = 0.001
    am_max::Float64               = 2.50

    # Δv
    delta_v_mean::Float64         = 3.50    # m/s
    delta_v_std::Float64          = 1.50    # m/s
    delta_v_min::Float64          = 0.20    # m/s
    delta_v_max::Float64          = 10.0    # m/s
    direction_mode::Symbol        = :isotropic

    # Altitude de quebra
    h_break_mean::Float64         = 78.0    # km
    h_break_std::Float64          = 2.50    # km
    h_break_min::Float64          = 65.0    # km
    h_break_max::Float64          = 92.0    # km

    # Drag Cd
    cd_default::Float64           = 2.20
end

"""
    BreakupEvent

Registra as condições do satélite pai e o conjunto de fragmentos no instante exato da quebra.
"""
struct BreakupEvent
    epoch_utc::String
    t_seconds::Float64
    r_sat::Vector{Float64}
    v_sat::Vector{Float64}
    h_break::Float64
    total_mass::Float64
    num_fragments::Int
    fragments::Vector{DebrisFragment}
end

end # module BreakupTypes

# src/ReentryBreakup/ReentryAtmosphere.jl

"""
    ReentryAtmosphere

Módulo de densidade atmosférica contínua dedicado à reentrada e descida terminal de detritos espaciais
(do topo da atmosfera até o nível do mar, 130 km -> 0 km).

Implementa o modelo padrão internacional **U.S. Standard Atmosphere 1976 (US76)** em tabela exponencial
estratificada por camadas barométricas (Vallado, 4ª ed., Tabela 8-4), garantindo densidade estritamente
crescente e fisicamente realista até a superfície terrestre (ρ₀ = 1.225 kg/m³ no nível do mar).
"""
module ReentryAtmosphere

using LinearAlgebra

export us_standard_atmosphere_1976, US76_LAYER_TABLE

"""
    US76_LAYER_TABLE

Tabela de parâmetros da U.S. Standard Atmosphere 1976 (Vallado, 4ª ed., Tabela 8-4):
Colunas:
1. `h_base`: Altitude da base da camada em km.
2. `rho_base`: Densidade na base da camada em kg/m³.
3. `H_scale`: Altura de escala barométrica (Scale Height) em km.
"""
const US76_LAYER_TABLE = [
#   h_base (km)   rho_base (kg/m^3)   H_scale (km)
      0.0         1.225               7.249
     25.0         3.899e-2            6.349
     30.0         1.774e-2            6.682
     40.0         3.972e-3            7.554
     50.0         1.057e-3            8.382
     60.0         3.206e-4            7.714
     70.0         8.770e-5            6.549
     80.0         1.905e-5            5.799
     90.0         3.396e-6            5.382
    100.0         5.297e-7            5.877
    110.0         9.661e-8            7.263
    120.0         2.438e-8            9.473
    130.0         8.484e-9           12.636
]

"""
    us_standard_atmosphere_1976(h_km::Float64) -> Float64

Calcula a densidade atmosférica local ρ em kg/m³ para qualquer altitude `h_km` entre 0 km e 150 km:
ρ(h) = ρ_base * exp(-(h - h_base) / H_scale)

- Para `h <= 0.0 km`: retorna a densidade padrão no nível do mar (1.225 kg/m³).
- Para `h > 130.0 km`: extrapola exponencialmente a última camada ou zera gradualmente.
"""
function us_standard_atmosphere_1976(h_km::Float64)
    if h_km <= 0.0
        return 1.225
    end

    n_layers = size(US76_LAYER_TABLE, 1)

    # Identifica a camada correta
    idx = 1
    for i in 1:(n_layers - 1)
        if h_km >= US76_LAYER_TABLE[i, 1] && h_km < US76_LAYER_TABLE[i + 1, 1]
            idx = i
            break
        end
    end

    if h_km >= US76_LAYER_TABLE[end, 1]
        idx = n_layers
    end

    h_base   = US76_LAYER_TABLE[idx, 1]
    rho_base = US76_LAYER_TABLE[idx, 2]
    H_scale  = US76_LAYER_TABLE[idx, 3]

    delta_h = h_km - h_base
    return rho_base * exp(-delta_h / H_scale)
end

end # module ReentryAtmosphere

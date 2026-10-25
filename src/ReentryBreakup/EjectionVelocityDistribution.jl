# src/ReentryBreakup/EjectionVelocityDistribution.jl

"""
    EjectionVelocityDistribution

Módulo responsável pela amostragem da velocidade de ejeção relativa (Δv) e orientação vetorial 3D
dos fragmentos gerados pela ruptura do satélite durante a reentrada atmosférica.

Diferente de explosões orbitais de alta energia (onde Δv > 100 m/s), a quebra por reentrada
é dominada por falha aero-termo-mecânica sob desaceleração de arrasto, gerando velocidades
relativas suaves (típicas de 0.5 a 10.0 m/s).
"""
module EjectionVelocityDistribution

using Random
using LinearAlgebra

using ..BreakupTypes: BreakupDistributionConfig

export sample_isotropic_direction, sample_wake_aligned_direction,
       sample_delta_v_magnitudes, generate_ejection_velocities

"""
    sample_isotropic_direction(rng::AbstractRNG = Random.GLOBAL_RNG) -> Vector{Float64}

Gera um vetor unitário tridimensional com distribuição perfeitamente uniforme sobre a esfera unitária S²:
- θ ∈ [0, 2π) sorteado uniformemente.
- cos(ϕ) ∈ [-1, 1] sorteado uniformemente para evitar concentração de pontos nos polos.

Retorna vetor unitário [ux, uy, uz].
"""
function sample_isotropic_direction(rng::AbstractRNG = Random.GLOBAL_RNG)
    u1 = rand(rng)
    u2 = rand(rng)

    theta = 2.0 * π * u1
    cos_phi = 2.0 * u2 - 1.0
    sin_phi = sqrt(max(0.0, 1.0 - cos_phi^2))

    ux = sin_phi * cos(theta)
    uy = sin_phi * sin(theta)
    uz = cos_phi

    return [ux, uy, uz]
end

"""
    sample_wake_aligned_direction(v_sat::Vector{Float64}; rng::AbstractRNG = Random.GLOBAL_RNG) -> Vector{Float64}

Gera uma direção preferencialmente alinhada com a esteira aerodinâmica (wake) ou plano transversal
em relação ao vetor velocidade instantâneo do satélite.
"""
function sample_wake_aligned_direction(
    v_sat::Vector{Float64};
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    v_norm = norm(v_sat)
    t_hat = v_norm > 1e-6 ? (v_sat ./ v_norm) : [1.0, 0.0, 0.0]

    # Escolhe um vetor arbitrário para construir base ortogonal
    arb = abs(t_hat[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
    n_hat = normalize(cross(t_hat, arb))
    b_hat = cross(t_hat, n_hat)

    # Componente na esteira (negativa ao longo de v_sat com dispersão)
    # 70% componente transversal e desaceleração aerodinâmica
    alpha = rand(rng) * 2.0 * π
    transverse_mag = sqrt(rand(rng))
    wake_comp = -0.5 * rand(rng)  # leve retardo na esteira

    dir = wake_comp .* t_hat .+ transverse_mag .* (cos(alpha) .* n_hat .+ sin(alpha) .* b_hat)
    return normalize(dir)
end

"""
    sample_delta_v_magnitudes(
        N::Int,
        config::BreakupDistributionConfig;
        rng::AbstractRNG = Random.GLOBAL_RNG
    ) -> Vector{Float64}

Sorteia a magnitude da velocidade de ejeção relativa Δv em m/s utilizando uma distribuição
normal truncada dentro de [config.delta_v_min, config.delta_v_max].
"""
function sample_delta_v_magnitudes(
    N::Int,
    config::BreakupDistributionConfig;
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    dv_mag = Vector{Float64}(undef, N)
    for i in 1:N
        val = config.delta_v_mean + config.delta_v_std * randn(rng)
        dv_mag[i] = clamp(val, config.delta_v_min, config.delta_v_max)
    end
    return dv_mag
end

"""
    generate_ejection_velocities(
        N::Int,
        v_sat::Vector{Float64},
        config::BreakupDistributionConfig;
        rng::AbstractRNG = Random.GLOBAL_RNG
    ) -> Tuple{Vector{Vector{Float64}}, Vector{Vector{Float64}}}

Calcula os vetores de velocidade de ejeção relativa (Δv em km/s) e a velocidade total inicial de cada fragmento
no referencial inercial ECI (v0 em km/s):
v0 = v_sat + Δv.

Retorna:
- `delta_v_vectors`: Vetor de vetores 3D de velocidade de ejeção relativa (km/s).
- `v0_vectors`: Vetor de vetores 3D de velocidade inercial absoluta de cada fragmento (km/s).
"""
function generate_ejection_velocities(
    N::Int,
    v_sat::Vector{Float64},
    config::BreakupDistributionConfig;
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    # Sorteia magnitudes em m/s
    dv_m_s = sample_delta_v_magnitudes(N, config; rng=rng)

    delta_v_vectors = Vector{Vector{Float64}}(undef, N)
    v0_vectors = Vector{Vector{Float64}}(undef, N)

    for i in 1:N
        if config.direction_mode == :wake_aligned
            dir_unit = sample_wake_aligned_direction(v_sat; rng=rng)
        else
            dir_unit = sample_isotropic_direction(rng)
        end

        # Conversão de m/s para km/s
        dv_kms = (dv_m_s[i] * 1e-3) .* dir_unit
        delta_v_vectors[i] = dv_kms
        v0_vectors[i] = v_sat .+ dv_kms
    end

    return delta_v_vectors, v0_vectors
end

end # module EjectionVelocityDistribution

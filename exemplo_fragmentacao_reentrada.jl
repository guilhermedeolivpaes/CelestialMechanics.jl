# exemplo_fragmentacao_reentrada.jl
# ==============================================================================
# Simulação Completa: Fragmentação Estocástica + Propagação com Arrasto em LEO
# Pacote: CelestialMechanics.jl
# ==============================================================================

using Pkg
Pkg.activate(@__DIR__)

using CelestialMechanics
using LinearAlgebra
using Printf
using Random

println(">>> [1/3] Configurando Parâmetros da Quebra do Satélite...")

# 1. Configuração Paramétrica das 3 Distribuições Fundamentais
config = BreakupDistributionConfig(
    fraction_dense       = 0.20,    # 20% componentes densos, 80% estruturais leves
    log10_am_dense_mean  = -2.00,   # log10(A/m) médio dos densos (0.01 m²/kg)
    log10_am_dense_std   = 0.35,
    log10_am_struct_mean = -0.85,   # log10(A/m) médio dos estruturais (~0.14 m²/kg)
    log10_am_struct_std  = 0.40,
    am_min               = 0.001,   # corte mínimo físico (m²/kg)
    am_max               = 2.500,   # corte máximo físico (m²/kg)
    delta_v_mean         = 3.50,    # velocidade média de separação (m/s)
    delta_v_std          = 1.50,
    delta_v_max          = 10.0,    # corte máximo de reentrada suave (m/s)
    direction_mode       = :isotropic,
    h_break_mean         = 78.0,    # altitude nominal média de colapso estrutural (km)
    cd_default           = 2.20     # coeficiente de arrasto representativo
)

# 2. Condições de Entrada do Satélite Pai no Ponto de Ruptura (h ~ 78 km)
r_earth = 6378.137  # Raio equatorial WGS84 em km
altitude_break = 78.0
r_mag = r_earth + altitude_break
r_sat = [r_mag, 0.0, 0.0]        # Vetor posição ECI (km)

# Velocidade de reentrada com ângulo de descida suave gamma = -1.5°
gamma_entry = deg2rad(-1.5)
v_mag = 7.62
v_sat = [v_mag * sin(gamma_entry), v_mag * cos(gamma_entry), 0.20]

total_mass = 1200.0              # Massa total do satélite (kg)
num_fragments = 100              # Quantidade de fragmentos da nuvem
rng = MersenneTwister(42)        # Semente para reprodutibilidade

println(">>> [2/3] Gerando Nuvem de Detritos no Ponto de Ruptura...")
event = generate_debris_cloud(
    r_sat,
    v_sat,
    num_fragments;
    total_mass = total_mass,
    config     = config,
    epoch_utc  = "2026-10-24T22:00:00Z",
    t_seconds  = 3600.0,
    rng        = rng
)

# Relatório da nuvem instantânea (Foto no instante da quebra)
debris_summary(event)

println("\n>>> [3/3] Propagando a Nuvem com Arrasto Atmosférico (US76) até o Solo...")
t_start_prop = time()
footprint = propagate_debris_cloud(
    event;
    max_duration = 3600.0,
    dt_initial   = 0.50,
    dt_save      = 2.0,
    save_trajectories = true,
    parallel     = true
)
t_elapsed = time() - t_start_prop
@printf("Propagação concluída em %.2f segundos (%d fragmentos integrados até h=0).

",
        t_elapsed, footprint.num_survived)

# Relatório da elipse de dispersão de impacto no solo (Filme da descida)
footprint_summary(footprint)

# Amostra Comparativa dos Primeiros Pontos de Impacto
println("\n--- AMOSTRA DOS PRIMEIROS 8 PONTOS DE IMPACTO NO SOLO ---")
@printf("%-4s | %-12s | %-10s | %-10s | %-11s | %-10s | %-12s\n",
        "ID", "Tipo", "Massa (kg)", "V_term (m/s)", "Downrange", "Crossrange", "Energia (kJ)")
println("-"^80)

for i in 1:min(8, footprint.num_survived)
    p = footprint.impact_points[i]
    tipo = p.is_dense ? "Denso (Tanque)" : "Estrutural"
    @printf("%-4d | %-12s | %10.2f | %10.2f   | %8.2f km | %7.2f km | %10.1f\n",
            p.id, tipo, p.mass, p.v_term_ms, p.downrange_km, p.crossrange_km, p.kinetic_energy_joules * 1e-3)
end

# Matriz numérica para exportação
footprint_matrix = footprint_to_matrix(footprint)
println("\nMatriz numérica de dados do Footprint gerada com dimensão: ", size(footprint_matrix))
println(">>> Simulação e propagação completas finalizadas com sucesso!")

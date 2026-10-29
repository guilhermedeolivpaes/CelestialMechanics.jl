# gerar_graficos_reentrada.jl
# ==============================================================================
# Geração Gráfica das 3 Distribuições Fundamentais e da Elipse de Impacto
# Pacote: CelestialMechanics.jl
# ==============================================================================

using Pkg
Pkg.activate(@__DIR__)

using CelestialMechanics
using CelestialMechanics.ReentryBreakup
using Makie
using GLMakie
using LinearAlgebra
using Printf
using Random
using Statistics

GLMakie.activate!()

println(">>> [1/4] Inicializando Simulação da Fragmentação e Propagação...")

# 1. Configuração e Sorteio da Nuvem de Detritos
config = BreakupDistributionConfig(
    fraction_dense       = 0.20,
    log10_am_dense_mean  = -2.00,
    log10_am_dense_std   = 0.35,
    log10_am_struct_mean = -0.85,
    log10_am_struct_std  = 0.40,
    am_min               = 0.001,
    am_max               = 2.500,
    delta_v_mean         = 3.50,
    delta_v_std          = 1.50,
    delta_v_max          = 10.0,
    direction_mode       = :isotropic,
    h_break_mean         = 78.0,
    cd_default           = 2.20
)

r_earth = 6378.137
h_break = 78.0
r_sat = [r_earth + h_break, 0.0, 0.0]

# Velocidade de reentrada com ângulo gamma = -1.5°
gamma_entry = deg2rad(-1.5)
v_mag = 7.62
v_sat = [v_mag * sin(gamma_entry), v_mag * cos(gamma_entry), 0.20]

total_mass = 1200.0  # kg
num_fragments = 150  # 150 fragmentos representativos
rng = MersenneTwister(42)

event = generate_debris_cloud(
    r_sat, v_sat, num_fragments;
    total_mass = total_mass,
    config     = config,
    rng        = rng
)

println(">>> [2/4] Propagando Nuvem com Arrasto até o Solo...")
footprint = propagate_debris_cloud(
    event;
    max_duration      = 3600.0,
    dt_initial        = 0.50,
    dt_save           = 5.0,
    save_trajectories = true,
    parallel          = true
)

output_dir = joinpath(@__DIR__, "output", "reentry_breakup")
mkpath(output_dir)

# ==============================================================================
# FIGURA 1: PAINEL DAS 3 DISTRIBUIÇÕES FUNDAMENTAIS
# ==============================================================================
println(">>> [3/4] Renderizando Figura 1: Painel das Distribuições Fundamentais...")

fig1 = Figure(size = (1500, 950), fontsize = 16)

# Título Geral
Label(fig1[0, 1:2], "Modelagem Estocástica de Fragmentação por Reentrada Atmosférica (LEO)",
      fontsize = 22, font = :bold)

# Subplot 1: Distribuição A - Razão Área/Massa (log10 A/m)
ax1 = Axis(fig1[1, 1],
    title  = "Distribuição A: Razão Área/Massa (A/m Bimodal)",
    xlabel = "log10(A/m)  [m²/kg]",
    ylabel = "Contagem de Fragmentos"
)

log10_ams_dense = [log10(f.area_to_mass) for f in event.fragments if f.is_dense_component]
log10_ams_struct = [log10(f.area_to_mass) for f in event.fragments if !f.is_dense_component]

hist!(ax1, log10_ams_dense, bins = 15, color = (:navy, 0.75), label = "Componentes Densos (Tanques)")
hist!(ax1, log10_ams_struct, bins = 20, color = (:coral, 0.75), label = "Estruturais Leves (Painéis)")
axislegend(ax1, position = :rt)

# Subplot 2: Distribuição B - Velocidade de Ejeção (Δv)
ax2 = Axis(fig1[1, 2],
    title  = "Distribuição B: Velocidade de Ejeção Relativa (Δv)",
    xlabel = "Módulo de Δv (m/s)",
    ylabel = "Contagem de Fragmentos"
)

dvs_ms = [norm(f.delta_v) * 1e3 for f in event.fragments]
hist!(ax2, dvs_ms, bins = 18, color = (:teal, 0.75), label = "Amostragem Δv")
vlines!(ax2, [config.delta_v_mean], color = :red, linestyle = :dash, linewidth = 2.5,
        label = @sprintf("Média Nominal = %.1f m/s", config.delta_v_mean))
axislegend(ax2, position = :rt)

# Subplot 3: Dispersão 3D das Velocidades de Ejeção
ax3 = Axis3(fig1[2, 1],
    title  = "Distribuição Vetorial 3D: Isotropia Esférica de Ejeção",
    xlabel = "Δvx (m/s)",
    ylabel = "Δvy (m/s)",
    zlabel = "Δvz (m/s)"
)

dv_x = [f.delta_v[1] * 1e3 for f in event.fragments]
dv_y = [f.delta_v[2] * 1e3 for f in event.fragments]
dv_z = [f.delta_v[3] * 1e3 for f in event.fragments]
colors_type = [f.is_dense_component ? :navy : :coral for f in event.fragments]

scatter!(ax3, dv_x, dv_y, dv_z, color = colors_type, markersize = 14)

# Subplot 4: Correlação Massa vs. Razão Área/Massa
ax4 = Axis(fig1[2, 2],
    title  = "Massa do Fragmento vs. Razão Área/Massa (A/m)",
    xlabel = "Razão Área/Massa (m²/kg)",
    ylabel = "Massa Individual (kg)",
    xscale = log10,
    yscale = log10
)

ams_all = [f.area_to_mass for f in event.fragments]
masses_all = [f.mass for f in event.fragments]
scatter!(ax4, ams_all, masses_all, color = colors_type, markersize = 13)

# Salvar Figura 1
fig1_path = joinpath(output_dir, "painel_distribuicoes_fundamentais.png")
save(fig1_path, fig1)
println("   -> Salvo: ", fig1_path)

# ==============================================================================
# FIGURA 2: ELIPSE DE IMPACTO NO SOLO (DEBRIS FOOTPRINT) & PERFIS DE DESCIDA
# ==============================================================================
println(">>> [4/4] Renderizando Figura 2: Elipse de Dispersão no Solo e Perfis de Queda...")

fig2 = Figure(size = (1500, 950), fontsize = 16)

Label(fig2[0, 1:2], "Dispersão e Segregação Aerodinâmica na Reentrada (Debris Footprint)",
      fontsize = 22, font = :bold)

# Painel 1: A Elipse de Impacto no Solo (Downrange vs. Crossrange)
ax_fp = Axis(fig2[1, 1:2],
    title  = @sprintf("Elipse de Impacto no Solo (Comprimento: %.1f km | Largura: %.1f km)",
                      footprint.footprint_length_km, footprint.footprint_width_km),
    xlabel = "Downrange ao longo da trajetória (km)",
    ylabel = "Crossrange transversal (km)"
)

fp_downrange = [p.downrange_km for p in footprint.impact_points]
fp_crossrange = [p.crossrange_km for p in footprint.impact_points]
fp_vterm = [p.v_term_ms for p in footprint.impact_points]
fp_dense = [p.is_dense for p in footprint.impact_points]

# Pontos de impacto coloridos por velocidade terminal
sc = scatter!(ax_fp, fp_downrange, fp_crossrange,
    color = fp_vterm, colormap = :plasma, markersize = 16,
    strokewidth = 1.0, strokecolor = :black
)
Colorbar(fig2[1, 3], sc, label = "Velocidade Terminal no Solo (m/s)")

# Linhas de referência da elipse
vlines!(ax_fp, [mean(fp_downrange)], color = :grey, linestyle = :dot)
hlines!(ax_fp, [0.0], color = :grey, linestyle = :dot)

# Painel 2: Perfis de Descida Temporal (Altitude vs. Tempo) - A "Peneira Aerodinâmica"
ax_desc = Axis(fig2[2, 1:2],
    title  = "Perfis de Altitude vs. Tempo de Descida (A Peneira Aerodinâmica por A/m)",
    xlabel = "Tempo decorrido de reentrada (segundos)",
    ylabel = "Altitude Geodésica (km)"
)

# Plotar amostra das trajetórias
for (i, p) in enumerate(footprint.impact_points)
    traj = p.trajectory
    if size(traj, 1) > 1
        t_vals = traj[:, 1]
        h_vals = traj[:, 8]
        cor_linha = p.is_dense ? (:navy, 0.80) : (:coral, 0.40)
        lines!(ax_desc, t_vals, h_vals, color = cor_linha, linewidth = p.is_dense ? 2.0 : 1.2)
    end
end

hlines!(ax_desc, [0.0], color = :black, linewidth = 1.5)

# Legenda customizada
dens_dummy = lines!(ax_desc, [0, 1], [-10, -10], color = :navy, linewidth = 2.5, label = "Densos (Tanques - Queda Rápida ~10 min)")
struct_dummy = lines!(ax_desc, [0, 1], [-10, -10], color = :coral, linewidth = 2.0, label = "Estruturais (Painéis - Flutuação ~30-50 min)")
axislegend(ax_desc, [dens_dummy, struct_dummy], ["Densos (Tanques)", "Estruturais (Painéis)"], position = :rt)
ylims!(ax_desc, -2.0, 82.0)

fig2_path = joinpath(output_dir, "painel_elipse_impacto_footprint.png")
save(fig2_path, fig2)
println("   -> Salvo: ", fig2_path)

println("
================================================================================")
println("                  GRÁFICOS GERADOS E SALVOS COM SUCESSO!                        ")
println("================================================================================")
println("1. ", fig1_path)
println("2. ", fig2_path)
println("================================================================================")

# exemplo_uso.jl
#
# Script de referência e exemplo para simulação orbital numérica com o pacote CelestialMechanics.
# Este script demonstra como ativar, configurar e parametrizar TODAS as forças e perturbações orbitais
# disponíveis no pacote:
#
#   1. Harmônicos Gravitacionais do Corpo Central (Zonais J2-J18 e Tesserais/Setoriais Cnm/Snm)
#   2. Harmônicos Esféricos Recursivos (Cunningham até ordem/grau arbitrários)
#   3. Gravitação de Terceiros Corpos (Perturbações N-Corpos: Sol, Lua, planetas)
#   4. Pressão de Radiação Solar Direta (SRP) com modelo de sombra/eclipse suave
#   5. Radiação Terrestre (Earth Radiation Pressure - ERP: Albedo difuso + Emissão Infravermelha Térmica)
#   6. Efeito Yarkovsky (Reemissão Térmica Anisotrópica Diurna / Recuo Térmico)
#   7. Arrasto Atmosférico (Harris-Priester ou NRLMSISE-00 com fluxo solar e índices geomagnéticos)
#   8. Força de Lorentz (Interação de corpo carregado com dipolo magnético planetário)
#   9. Precessão Geral da Terra (Aceleração aparente de Coriolis IAU 1976)
#  10. Relatividade Geral: Efeito Geodésico / Avanço do Perigeu 1PN (Schwarzschild / De Sitter)
#  11. Relatividade Geral: Arrasto de Referenciais / Efeito Lense-Thirring (Gravitomagnetismo)
#  12. Deformações Geofísicas: Marés Sólidas da Terra (Solid Earth Tides - k2 de Love)
#  13. Eletrodinâmica Espacial: Arrasto de Plasma / Coulomb Drag (Bainha de Debye / Freio de Plasma)

using CelestialMechanics
using SatelliteToolboxAtmosphericModels
using SpaceIndices
using DifferentialEquations
using Unitful
using UnitfulAstro
using Unitful.DefaultSymbols 
using Dates  
using Logging
using LoggingExtras
using CSV
using DataFrames, DataFramesMeta
using LinearAlgebra
using StaticArrays
using Accessors

include("helpers/helper_directories.jl")

const INTERACTIVE_MODE = false # Mude para `true` para abrir gráficos interativos, ou `false` para salvar em lote

if INTERACTIVE_MODE
    @info "Modo Interativo ativado"
else
    @info "Modo Batch (Lote) ativado"
end

using GLMakie
GLMakie.activate!()

# -----------------------------------------------------------------------------------------------
# CONFIGURAÇÕES INICIAIS: LOGGING, EFEMÉRIDES (SPICE) E CONDIÇÕES INICIAIS
# -----------------------------------------------------------------------------------------------

function initial_settings()
    # Inicialização de logging com timestamp
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    log_path = joinpath(output_dir_cowell, "simulation_$timestamp.log")

    terminal_logger = ConsoleLogger()
    file_logger = MinLevelLogger(FileLogger(log_path), Logging.Info)
    global_logger(TeeLogger(terminal_logger, file_logger))

    # -------------------------------------------------------------------------------------------
    # CONFIGURAÇÃO DE EFEMÉRIDES PLANETÁRIAS VIA SPICE (NAIF)
    # -------------------------------------------------------------------------------------------
    spice_info = SpiceInformations(
        path_leapseconds_tls   = joinpath(ephemeris_dir, "naif0012.tls"),
        path_solar_system_bsp  = joinpath(ephemeris_dir, "sistema_solar/de440.bsp"),
        primary_body_SPICE     = BODIES_DATA[:earth].spice_id,
        reference_frame        = "J2000",
        initial_date           = "2020-01-01T00:00:00",
        final_date             = "2035-01-01T00:00:00"
    )

    # -------------------------------------------------------------------------------------------
    # CONDIÇÕES INICIAIS DA ÓRBITA (Elementos Keplerianos Clássicos)
    # Exemplo: Órbita quase-circular e hélio-síncrona (LEO)
    # -------------------------------------------------------------------------------------------
    ics = [
        InitialConditions(
            a0 = 6920.459u"km",        # Semi-eixo maior (~542 km de altitude)
            e0 = 0.00675,              # Excentricidade
            i0 = 98.37u"°",            # Inclinação orbital
            h0 = 72.98u"°",            # RAAN (Longitude do Nó Ascendente)
            g0 = 96.90u"°",            # Argumento do Perigeu
            f0 = 263.75u"°"            # Anomalia Verdadeira
        )
    ]

    # Intervalo de tempo de propagação
    t_start = 0.0u"d"
    t_end   = 5.0u"d"                  # Duração da propagação (ex: 5 dias para teste, ou 4383 dias para 12 anos)
    step    = 1.0u"hr"                 # Passo de amostragem dos dados salvos

    t_vector_with_units = t_start:step:t_end
    tspan = (ustrip(u"s", t_start), ustrip(u"s", t_end))
    t_vector = ustrip.(u"s", t_vector_with_units)

    @info "Tempo total de propagação configurado:" t=t_end passo=step

    global g_spice_info = spice_info

    return spice_info, ics, tspan, t_vector
end

# -----------------------------------------------------------------------------------------------
# CONFIGURAÇÃO DE TODAS AS FORÇAS E PERTURBAÇÕES FÍSICAS
# -----------------------------------------------------------------------------------------------

function simulation_config()
    # ===========================================================================================
    # MODELO ATMOSFÉRICO PARA O ARRASTO
    # ===========================================================================================
    # Opção 1: Modelo NRLMSISE-00 com fluxo solar e índice geomagnético constantes:
    atm = NRLMSISE00Atmosphere(
        epoch = DateTime("2020-01-01T00:00:00"),
        F10a  = 150.0,                 # Fluxo solar F10.7 médio de 81 dias
        F10   = 150.0,                 # Fluxo solar diário F10.7
        ap    = 8.0,                   # Índice geomagnético linear equivalente Ap (nT)
        frame = :J2000
    )

    # Opção 2: Modelo empírico Harris-Priester com bojo diurno (SPICE-free):
    # atm = HarrisPriesterAtmosphere() # ou passe `nothing` em atmosphere para usar Harris-Priester padrão

    @info "Modelo atmosférico configurado" drag_model=atm

    # ===========================================================================================
    # CONSTRUÇÃO DO MODELO DE PERTURBAÇÕES (create_perturbation_model)
    # ===========================================================================================
    perturbation_params = create_perturbation_model(
        :earth;

        # ---------------------------------------------------------------------------------------
        # 1. GRAVIDADE E HARMÔNICOS ESFÉRICOS (Corpo Central: Terra)
        # ---------------------------------------------------------------------------------------
        # Modelo analítico clássico explícito:
        # - Zonais: j_harmonics aceita graus de 2 a 18 (e.g. [2, 3, 4] ativa J2, J3 e J4).
        # - Tesserais/Setoriais: cs_harmonics aceita nm (e.g. [22, 31, 32, 33, 41, 42, 43, 44]).
        j_harmonics        = Int[2, 3, 4],
        cs_harmonics       = Int[22],

        # Modelo recursivo de Cunningham (para graus e ordens arbitrários elevados):
        # Para ativar, defina use_cunningham = true e especifique max_degree e max_order:
        use_cunningham     = false,
        max_degree         = nothing,          # Ex: 10
        max_order          = nothing,          # Ex: 10
        normalized_gravity = false,            # true se coeficientes forem normalizados (e.g. EGM2008)

        # ---------------------------------------------------------------------------------------
        # 2. GRAVITAÇÃO DE TERCEIROS CORPOS (N-Body Perturbations)
        # ---------------------------------------------------------------------------------------
        # Lista de corpos perturbadores presentes em BODIES_DATA (ex: [:sun, :moon, :jupiter])
        n_body_symbols     = Symbol[:sun, :moon],

        # ---------------------------------------------------------------------------------------
        # 3. PRESSÃO DE RADIAÇÃO SOLAR DIRETA (Solar Radiation Pressure - SRP)
        # ---------------------------------------------------------------------------------------
        # Considera os fótons incidentes diretamente do Sol.
        srp_cr             = 1.2,                     # Coeficiente de reflexão (adimensional, ~1.0 - 1.5)
        srp_alpha          = (1.5u"m^2" / 100.0u"kg"),# Razão Área/Massa do satélite (m²/kg)
        shadow_on          = true,                    # Ativa modelo de sombra/eclipse da Terra (função suave)

        # ---------------------------------------------------------------------------------------
        # 4. RADIAÇÃO TERRESTRE (Earth Radiation Pressure - ERP: Albedo + IR Térmico)
        # ---------------------------------------------------------------------------------------
        # O complemento natural do SRP em órbita baixa (LEO):
        # - Albedo: Luz solar refletida difusamente pela calota iluminada da Terra (~30%).
        # - IR Térmico: Calor próprio emitido pela Terra continuamente (~237 W/m²), dia e noite.
        # Em 400 a 1000 km, essa força de repulsão equivale a 10% a 25% do SRP solar direto!
        albedo             = 0.30,                    # Coeficiente médio de albedo da Terra (30%)
        thermal_ir_flux    = 237.0,                   # Fluxo térmico infravermelho médio da Terra em W/m²

        # ---------------------------------------------------------------------------------------
        # 5. EFEITO YARKOVSKY (Reemissão Térmica Anisotrópica Diurna)
        # ---------------------------------------------------------------------------------------
        # Aquecimento diurno com atraso térmico de rotação gerando empuxo contínuo minúsculo.
        # Crucial para satélites leves e detritos espaciais com alto A/m (MLI, fragmentos de painéis).
        yarkovsky_cy       = 0.15,                    # Coeficiente de eficiência térmica (típico 0.10 a 0.25)
        yarkovsky_lag      = 25.0,                    # Ângulo de atraso térmico diurno em graus (default 25.0°)
        spin_axis          = nothing,                 # nothing = normal da órbita (prograde); ou vetor [wx, wy, wz]

        # ---------------------------------------------------------------------------------------
        # 6. ARRASTO ATMOSFÉRICO (Atmospheric Drag)
        # ---------------------------------------------------------------------------------------
        drag_cd            = 2.2,                     # Coeficiente de arrasto aerodinâmico (adimensional)
        drag_am            = (0.505u"m^2" / 94.0u"kg"),# Razão Área/Massa para arrasto aerodinâmico
        min_altitude       = 100.0u"km",              # Altitude limite para término da simulação (reentrada)
        atmosphere         = atm,                     # Instância do modelo de atmosfera (atm configurado acima)

        # ---------------------------------------------------------------------------------------
        # 7. FORÇA DE LORENTZ (Interação Eletrodinâmica com o Campo Magnético Planetário)
        # ---------------------------------------------------------------------------------------
        # Aceleração sobre satélite eletrizado em órbita: a = (q/m) * (v x B)
        lorentz_qm         = nothing,                 # Razão carga/massa em C/kg (ex: 1e-5; nothing = desligado)
        lorentz_B          = 8.0e15,                  # Constante dipolar do campo magnético (default 8.0e15)

        # ---------------------------------------------------------------------------------------
        # 8. PRECESSÃO GERAL DA TERRA (Aceleração Aparente de Coriolis / IAU 1976)
        # ---------------------------------------------------------------------------------------
        # Efeito Coriolis aparente decorrente do referencial em precessão lenta da Terra.
        precession_on      = false,                   # true para ativar a aceleração de Coriolis por precessão

        # ---------------------------------------------------------------------------------------
        # 9. RELATIVIDADE GERAL: EFEITO GEODÉSICO (1PN Schwarzschild / Avanço do Perigeu de Einstein)
        # ---------------------------------------------------------------------------------------
        # Correção pós-Newtoniana de 1ª ordem (1PN) decorrente da curvatura do espaço-tempo:
        # a_GR = (mu / (c^2 * r^3)) * [ (4*mu/r - v^2)*r + 4*(r . v)*v ]
        # Provoca avanço secular do perigeu (~3.84''/ano a 13.34''/ano em LEO).
        # Indispensável para satélites de navegação (GPS, Galileo) e geodésia espacial de alta precisão.
        relativity_geodesic = false,                  # true para ativar o avanço do perigeu de Schwarzschild

        # ---------------------------------------------------------------------------------------
        # 10. RELATIVIDADE GERAL: ARRASTO DE REFERENCIAIS (Efeito Lense-Thirring / Gravitomagnetismo)
        # ---------------------------------------------------------------------------------------
        # A rotação própria da Terra arrasta a métrica do espaço-tempo ao seu redor:
        # a_LT = (2*mu / (c^2 * r^3)) * [ (3/r^2)*((r x v) . J_earth)*r + (v x J_earth) ]
        # Medido experimentalmente pelas missões geodésicas LAGEOS e LARES.
        relativity_lense_thirring = false,            # true para ativar gravito-magnetismo de rotação
        spin_angular_momentum     = nothing,          # nothing = Terra rígida (Jz ≈ 981.0 km²/s); ou vetor [0, 0, Jz]

        # ---------------------------------------------------------------------------------------
        # 11. DEFORMAÇÕES GEOFÍSICAS: MARÉS SÓLIDAS DA TERRA (Solid Earth Tides)
        # ---------------------------------------------------------------------------------------
        # Deformação periódica da crosta e manto terrestre causada pela atração da Lua e do Sol.
        # Altera os harmônicos gravitacionais equivalentes (Delta C20, Delta C22, Delta S22)
        # gerando acelerações periódicas de ~10^-7 m/s^2 (~10^-8 g).
        # Perturbação essencial para missões de alta precisão geodésica e altimetria (GRACE, Sentinel, LAGEOS).
        solid_tides               = false,            # true para ativar marés sólidas da Terra
        love_k2                   = 0.30,             # Número de Love k2 elástico nominal da Terra (default 0.30)

        # ---------------------------------------------------------------------------------------
        # 12. ELETRODINÂMICA ESPACIAL: ARRASTO DE PLASMA (Coulomb Drag / Freio de Plasma)
        # ---------------------------------------------------------------------------------------
        # Interação eletrostática da superfície eletrizada do satélite com os íons da ionosfera.
        # Aumenta a seção de choque aerodinâmica efetiva através da bainha de Debye (Sheath).
        # Crucial para nano-satélites (CubeSats), cabos eletrodinâmicos (Plasma Brakes) e detritos espaciais.
        coulomb_drag              = false,            # true para ativar o arrasto de plasma (Coulomb Drag)
        plasma_potential          = -5.0,             # Potencial elétrico superficial [V] (-1 a -10 V natural, ou até -1000 V ativo)
        plasma_ion_density        = nothing,          # Densidade iônica [m^-3] (nothing = estimada via atmosfera/Chapman LEO)
        plasma_debye_length       = 0.02,             # Comprimento de Debye ionosférico [m] (default ~0.02 m)
        plasma_tether_length      = nothing           # Comprimento de cabo condutor [m] (nothing = satélite; >0 = cabo de freio de plasma)
    )

    # -------------------------------------------------------------------------------------------
    # OPÇÕES DO INTEGRADOR NUMÉRICO (SOLVER OPTIONS)
    # -------------------------------------------------------------------------------------------
    propagator_opts = PropagatorOptions(
        propagator                   = CowellPropagator(),
        canonical_unit_normalization = true,          # Normalização canônica (DU, TU, VU) para estabilidade
        second_order                 = false,
        integrator                   = Vern9(),       # Integrador de alta precisão (Verner 9th-order)
        abstol                       = 1e-8,
        reltol                       = 1e-8,    
        maxiters                     = 50_000_000,
        poincare_callback            = false,
        poincare_sections            = [:periapsis_map],
        T_perturbation               = nothing,
        saveat                       = false
    )

    # -------------------------------------------------------------------------------------------
    # CONFIGURAÇÕES GRÁFICAS E VISUALIZAÇÃO
    # -------------------------------------------------------------------------------------------
    plotting_opts = PlottingOptions(
        graph_type        = :elem,
        separate_element  = :a,
        show_interactive  = INTERACTIVE_MODE,
        use_scatter_plot  = false,
        width_fig         = 1200, 
        height_fig        = 800,
        base_fontsize     = 22,
        XY_projection     = true, 
        XZ_projection     = true, 
        YZ_projection     = true,
        axis_adjustment   = 2000.0,
        time_unit         = u"d",
        dist_unit         = u"km",
        plot_in_radii     = false
    )

    graphs_info = GraphicInformation(
        body_color   = :blue, 
        alpha_opac   = 0.2, 
        orbit_color  = :black
    )

    return perturbation_params, propagator_opts, plotting_opts, graphs_info
end

# -----------------------------------------------------------------------------------------------
# EXECUÇÃO DA SIMULAÇÃO NUMÉRICA
# -----------------------------------------------------------------------------------------------

function run_sim(ics, perturbation_params, spice_info, tspan, t_vector, propagator_opts)
    @info "\n--- Iniciando a execução da simulação orbital (Cowell) ---"
    @info "Integrador selecionado:" integrator=propagator_opts.integrator 

    all_results = run_simulation(
        ics                 = ics,
        perturbation_params = perturbation_params,
        spice_info          = spice_info,
        tspan               = tspan,
        t_vector            = t_vector,
        propagator_options  = propagator_opts, 
        output_directory    = output_dir_cowell,
        # analytic_ephemeris = eph, # Opcional: Ative esta linha e comente spice_info para usar efemérides analíticas sem SPICE
    )

    @info "Simulação concluída com sucesso!"
    return all_results
end

# -----------------------------------------------------------------------------------------------
# PÓS-PROCESSAMENTO, EXPORTAÇÃO CSV E GERAÇÃO DE GRÁFICOS
# -----------------------------------------------------------------------------------------------

function post_process(all_results, propagator_opts, plotting_opts, graphs_info; plot_types = [:elem, :d3])
    global SAVE_FIGURE = true
    @info "\n--- Iniciando Pipeline de Pós-Processamento ---"

    # Exportação de dados brutos e elementos osculadores para CSV
    results, _ = run_post_analysis(
        all_results = all_results,
        prop_opts   = propagator_opts,
        plot        = false,
        plot_opts   = plotting_opts,
        graph_info  = graphs_info,
        output_dir  = output_dir_cowell,
        save_csv    = true
    )

    all_figs = Figure[]

    # Geração dos gráficos solicitados (:elem para elementos orbitais, :d3 para trajetória 3D)
    for gtype in plot_types
        p_opts = PlottingOptions(
            graph_type       = gtype,
            separate_element = plotting_opts.separate_element,
            show_interactive = false,
            use_scatter_plot = plotting_opts.use_scatter_plot,
            width_fig        = plotting_opts.width_fig,
            height_fig       = plotting_opts.height_fig,
            base_fontsize    = plotting_opts.base_fontsize,
            XY_projection    = plotting_opts.XY_projection,
            XZ_projection    = plotting_opts.XZ_projection,
            YZ_projection    = plotting_opts.YZ_projection,
            axis_adjustment  = plotting_opts.axis_adjustment,
            time_unit        = plotting_opts.time_unit,
            dist_unit        = plotting_opts.dist_unit,
            plot_in_radii    = plotting_opts.plot_in_radii
        )

        type_figs = [plot_orbital_results(res, p_opts, graphs_info) for res in results]

        for (i, fig) in enumerate(type_figs)
            if INTERACTIVE_MODE 
                display(GLMakie.Screen(), fig)
            end
            if !INTERACTIVE_MODE || SAVE_FIGURE
                fig_filename = joinpath(output_dir_cowell, "simulation_plot_$(gtype)_$i.png")
                Makie.save(fig_filename, fig)
                @info "Gráfico salvo em:" arquivo=fig_filename
            end
        end

        append!(all_figs, type_figs)
    end

    @info "Pós-processamento concluído com sucesso!"
    return results, all_figs
end

# -----------------------------------------------------------------------------------------------
# PONTO DE ENTRADA PRINCIPAL (MAIN)
# -----------------------------------------------------------------------------------------------

function main_cowell()
    spice_info, ics, tspan, t_vector = initial_settings()
    perturbation_params, propagator_opts, plotting_opts, graphs_info = simulation_config()
    all_results = run_sim(ics, perturbation_params, spice_info, tspan, t_vector, propagator_opts)
    results, figs = post_process(all_results, propagator_opts, plotting_opts, graphs_info; plot_types = [:elem, :d3])
    return results, figs
end

# Executa se chamado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    main_cowell()
end

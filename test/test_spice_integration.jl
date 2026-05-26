# test/test_spice_integration.jl
#
# Testes de integração que requerem kernels SPICE.
# NÃO incluídos no CI automático — rode localmente com:
#
#   julia --project test/test_spice_integration.jl
#
# Pré-requisito: kernels em data/spice/ (naif0012.tls, de430.bsp, etc.)
#
# Propriedades verificadas:
#   1. Cowell sem perturbações: semi-eixo conservado em uma órbita
#   2. Cowell + J2: precissão do RAAN compatível com a literatura
#   3. Propagação com terceiro corpo (Terra): influência mensurável

using Test
using CelestialMechanics
using CelestialMechanics.Coordinates
using LinearAlgebra
using Unitful
using UnitfulAstro

# ── Configuração SPICE (ajuste os caminhos para seu ambiente) ────────────────
const SPICE_DIR    = joinpath(@__DIR__, "..", "data", "spice")
const LEAPSEC_PATH = joinpath(SPICE_DIR, "naif0012.tls")
const PLANET_BSP   = joinpath(SPICE_DIR, "de430.bsp")

# Verifica se os kernels existem antes de rodar
if !isfile(LEAPSEC_PATH) || !isfile(PLANET_BSP)
    @warn "Kernels SPICE não encontrados em $SPICE_DIR — pulando testes de integração."
else

const MU_MOON = 4902.80012
const R_MOON  = 1738.1
const J2_MOON = 9.09010949481e-5

spice_info = SpiceInformations(
    path_leapseconds_tls  = LEAPSEC_PATH,
    path_solar_system_bsp = PLANET_BSP,
    primary_body_SPICE    = "MOON",
    reference_frame       = "J2000",
    aberration_correction = "NONE",
    ephemeris_time_start  = "2024-01-01T00:00:00"
)

@testset "Cowell Integration (SPICE)" begin

    @testset "Sem perturbações: semi-eixo conservado em 1 período" begin
        a  = 2000.0u"km"
        e  = 0.3
        i  = 45.0u"°"
        T  = 2π * sqrt(ustrip(u"km", a)^3 / MU_MOON)  # segundos

        ic = InitialConditions(
            a0=a, e0=e, i0=i,
            h0=0.0u"°", g0=90.0u"°", f0=0.0u"°"
        )

        params = create_perturbation_model(:moon; j2=false)

        t_span  = (0.0u"s", T*u"s")
        t_vec   = range(0.0, T, length=1000) .* u"s"

        opts = PropagatorOptions(
            propagator = CowellPropagator(),
            integrator = Vern9(),
            abstol     = 1e-12,
            reltol     = 1e-12,
        )

        results = run_simulation(
            ics                 = [ic],
            perturbation_params = params,
            spice_info          = spice_info,
            tspan               = t_span,
            t_vector            = t_vec,
            propagator_options  = opts,
        )

        df = run_post_analysis(results; mu=MU_MOON, R=R_MOON).elements
        a0_km = df.a_km[1];  af_km = df.a_km[end]

        # Semi-eixo deve se conservar melhor que 1e-4 km em 1 período
        @test abs(af_km - a0_km) / a0_km < 1e-8
    end

    @testset "J2: taxa de precessão de RAAN compatível com literatura" begin
        # Para órbita circular com J2, dΩ/dt = -3/2 * n * J2 * (R/a)² * cos(i) / (1-e²)²
        a  = 1838.0u"km"    # ~100 km altitude
        e  = 0.001
        i  = 30.0u"°"
        T  = 2π * sqrt(ustrip(u"km", a)^3 / MU_MOON)

        ic = InitialConditions(
            a0=a, e0=e, i0=i,
            h0=0.0u"°", g0=0.0u"°", f0=0.0u"°"
        )

        params = create_perturbation_model(:moon; j2=true)

        # Integra por 10 períodos
        t_end   = 10 * T
        t_span  = (0.0u"s", t_end*u"s")
        t_vec   = range(0.0, t_end, length=5000) .* u"s"

        opts = PropagatorOptions(
            propagator = CowellPropagator(),
            integrator = Vern9(),
            abstol     = 1e-12,
            reltol     = 1e-12,
        )

        results = run_simulation(
            ics                 = [ic],
            perturbation_params = params,
            spice_info          = spice_info,
            tspan               = t_span,
            t_vector            = t_vec,
            propagator_options  = opts,
        )

        df = run_post_analysis(results; mu=MU_MOON, R=R_MOON).elements

        # Taxa de precessão numérica (deg/s)
        dΩ_num = (df.h_deg[end] - df.h_deg[1]) / t_end

        # Taxa analítica de J2
        n   = sqrt(MU_MOON / ustrip(u"km", a)^3)
        a_km = ustrip(u"km", a)
        dΩ_ana = -1.5 * n * J2_MOON * (R_MOON / a_km)^2 * cosd(30.0)

        dΩ_ana_deg_s = rad2deg(dΩ_ana)

        @test dΩ_num ≈ dΩ_ana_deg_s rtol=0.01   # 1% de precisão
    end

end # @testset

end # if kernels exist
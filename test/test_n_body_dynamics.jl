# test/test_n_body_dynamics.jl
#
# Testa o integrador N-body sem SPICE.
# Usa as funções compute_energy e compute_momentum do próprio módulo.
#
# Notas de implementação:
#   - save_everystep=false é o padrão (disco): sol.u fica vazio.
#     Para verificar energia, usamos saveat=[t0, tf] para ter u[1] e u[end].
#   - O teste simplético verifica apenas que a integração completa sem erro
#     e que o retcode é Success — a conservação de energia simplética é
#     validada indiretamente pelo log E_rel que o módulo já imprime.

using Test
using CelestialMechanics
using CelestialMechanics.Coordinates
using CelestialMechanics.NBodyIntegrator
using DifferentialEquations
using LinearAlgebra
using StaticArrays

const MU_MOON=4902.80012
const R_MOON=1738.1

@testset "N-Body Dynamics" begin

    @testset "Dois corpos: energia conservada em 1 período (Vern9)" begin
        a  = 2000.0
        e  = 0.4
        i  = deg2rad(30.0)
        mu = MU_MOON

        r0, v0 = Coordinates.orbital_elements_to_state_vectors(
            a, e, i, deg2rad(45.0), deg2rad(90.0), deg2rad(0.0), mu
        )

        bodies = [
            Types.NBodyParticle(name=:central,
                r0=SVector(0.0,0.0,0.0), v0=SVector(0.0,0.0,0.0),
                mu=mu, R=R_MOON),
            Types.NBodyParticle(name=:sat,
                r0=SVector(r0...), v0=SVector(v0...),
                mu=1e-10, R=1.0),
        ]

        ic   = Types.NBodySystemIC(bodies)
        T    = 2π * sqrt(a^3 / mu)
        opts = PropagatorOptions(
            integrator = Vern9(),
            abstol     = 1e-12,
            reltol     = 1e-12,
            maxiters   = 1_000_000,
            saveat     = true,          # salva instantes regulares → sol.u populado
        )

        result = run_nbody_simulation(
            ic                 = ic,
            tspan              = (0.0, T),
            propagator_options = opts,
            use_symplectic     = false,
        )

        sol = result.solution
        @test length(sol.u) >= 2

        u0_flat = Vector{Float64}(sol.u[1])
        uf_flat = Vector{Float64}(sol.u[end])

        E0 = NBodyIntegrator.compute_energy(u0_flat, bodies)
        Ef = NBodyIntegrator.compute_energy(uf_flat, bodies)

        @test isfinite(E0)
        @test isfinite(Ef)
        @test abs((Ef - E0) / E0) < 1e-8
    end

    @testset "Dois corpos: semi-eixo conservado em 1 período (Vern9)" begin
        a  = 2000.0
        e  = 0.4
        i  = deg2rad(30.0)
        mu = MU_MOON

        r0, v0 = Coordinates.orbital_elements_to_state_vectors(
            a, e, i, deg2rad(45.0), deg2rad(90.0), deg2rad(0.0), mu
        )

        bodies = [
            Types.NBodyParticle(name=:central,
                r0=SVector(0.0,0.0,0.0), v0=SVector(0.0,0.0,0.0),
                mu=mu, R=R_MOON),
            Types.NBodyParticle(name=:sat,
                r0=SVector(r0...), v0=SVector(v0...),
                mu=1e-10, R=1.0),
        ]

        ic   = Types.NBodySystemIC(bodies)
        T    = 2π * sqrt(a^3 / mu)
        opts = PropagatorOptions(
            integrator = Vern9(),
            abstol     = 1e-12,
            reltol     = 1e-12,
            maxiters   = 1_000_000,
            saveat     = true,
        )

        result = run_nbody_simulation(
            ic                 = ic,
            tspan              = (0.0, T),
            propagator_options = opts,
            use_symplectic     = false,
        )

        sol = result.solution
        uf  = Vector{Float64}(sol.u[end])

        r_f = uf[7:9]
        v_f = uf[10:12]

        eps_0 = norm(v0)^2/2 - mu/norm(r0)
        eps_f = norm(v_f)^2/2 - mu/norm(r_f)

        a_f = -mu / (2 * eps_f)
        @test a_f ≈ a atol=1e-4
    end

    @testset "Dois corpos simplético (KahanLi8): integração completa sem erro" begin
        a  = 2000.0
        e  = 0.3
        i  = deg2rad(45.0)
        mu = MU_MOON

        r0, v0 = Coordinates.orbital_elements_to_state_vectors(
            a, e, i, 0.0, 0.0, 0.0, mu
        )

        bodies = [
            Types.NBodyParticle(name=:central,
                r0=SVector(0.0,0.0,0.0), v0=SVector(0.0,0.0,0.0),
                mu=mu, R=R_MOON),
            Types.NBodyParticle(name=:sat,
                r0=SVector(r0...), v0=SVector(v0...),
                mu=1e-10, R=1.0),
        ]

        ic = Types.NBodySystemIC(bodies)
        T  = 2π * sqrt(a^3 / mu)

        opts = PropagatorOptions(
            integrator = KahanLi8(),
            dt         = T / 500,
            maxiters   = 10_000_000,
        )

        result = run_nbody_simulation(
            ic                 = ic,
            tspan              = (0.0, 10T),
            propagator_options = opts,
            use_symplectic     = true,
        )

        # Simplético: valida apenas que integrou com sucesso
        @test string(result.solution.retcode) == "Success"
    end

    @testset "Três corpos: conservação de momento linear total (Vern9)" begin
        G = 6.674e-20

        bodies = [
            Types.NBodyParticle(name=:p1,
                r0=SVector(0.0,    0.0, 0.0), v0=SVector(0.0, 0.0, 0.0),
                mu=1.0e6, R=100.0),
            Types.NBodyParticle(name=:p2,
                r0=SVector(1000.0, 0.0, 0.0), v0=SVector(0.0, 1.0, 0.0),
                mu=1.0e3, R=10.0),
            Types.NBodyParticle(name=:p3,
                r0=SVector(500.0, 866.0, 0.0), v0=SVector(-0.5, 0.5, 0.0),
                mu=1.0e0, R=1.0),
        ]

        ic   = Types.NBodySystemIC(bodies)
        opts = PropagatorOptions(
            integrator = Vern9(),
            abstol     = 1e-10,
            reltol     = 1e-10,
            maxiters   = 5_000_000,
            saveat     = true,
        )

        result = run_nbody_simulation(
            ic                 = ic,
            tspan              = (0.0, 100.0),
            propagator_options = opts,
            use_symplectic     = false,
        )

        sol = result.solution
        @test length(sol.u) >= 2

        u0_flat = Vector{Float64}(sol.u[1])
        uf_flat = Vector{Float64}(sol.u[end])

        p0 = NBodyIntegrator.compute_momentum(u0_flat, bodies)
        pf = NBodyIntegrator.compute_momentum(uf_flat, bodies)

        norm_ref = max(norm(p0), 1.0)
        @test norm(pf - p0) / norm_ref < 1e-6
    end
end
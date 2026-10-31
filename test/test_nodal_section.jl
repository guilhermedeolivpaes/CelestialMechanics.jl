using Test
using CelestialMechanics
using DifferentialEquations
using Unitful

# Autonomous test Hamiltonian: K = G + H^2/2.
function nodal_test_rhs!(du, u, p, t)
    fill!(du, 0)
    du[5] = 1
    du[6] = u[3]
end

@testset "Delaunay collision termination" begin
    function decreasing_action!(du, u, p, t)
        fill!(du, 0.0)
        du[2] = -0.02
    end
    opts = PropagatorOptions(propagator=HamiltonianPropagator(), integrator=Vern9(),
        poincare_sections=Symbol[], terminate_on_collision=true,
        section_dtmax=0.05, saveat=true)
    params = PhysicalParams(mu=1.0, R=3.0)
    ics = [InitialPlanetaryConditions(a0=4.0u"km", e0=0.2,
        i0=60.0u"deg", h0=0.0u"rad", g0=0.0u"rad", l0=0.0u"rad")]
    result = only(run_hamiltonian_simulation(; ics, equation_func=decreasing_action!,
        perturbation_params=params, tspan=(0.0, 10.0), t_vector=collect(0.0:0.1:10.0),
        propagator_options=opts))
    collision = result.poincare_all[:collision]
    @test result.solution.retcode == SciMLBase.ReturnCode.Terminated
    @test collision[:occurred]
    @test 0.0 < collision[:time_s] < 10.0
    @test abs(collision[:periapsis_altitude_km]) < 1e-8
end

@testset "Oscillatory Hamiltonian energy and step refinement" begin
    function oscillator!(du, u, p, t)
        fill!(du, 0)
        du[2] = -u[5]
        du[5] = u[2] - 1.5
        du[6] = u[3]
    end
    energy(u, p) = ((u[2]-1.5)^2 + u[5]^2 + u[3]^2)/2
    points = []
    for step in (0.2, 0.1)
        opts = PropagatorOptions(integrator=Vern9(), poincare_callback=true,
            section_angle=0.3, section_hamiltonian=energy)
        cb, data = CelestialMechanics.PropagatorUtils.setup_nodal_section_callback(opts)
        u0 = [2.0, 1.5, 0.8, 0.0, 0.1, -0.2]
        prob = ODEProblem(oscillator!, u0, (0.0,20.0), (mu=1.0,))
        sol = solve(prob, Vern9(); callback=cb, dtmax=step,
            abstol=1e-11, reltol=1e-11)
        @test maximum(abs(energy(u, prob.p)-energy(u0, prob.p)) for u in sol.u) < 1e-9
        for (t,u) in zip(data[:times], data[:canonical_states])
            @test u[5] ≈ 0.1*cos(t) atol=1e-9
            @test u[2] ≈ 1.5-0.1*sin(t) atol=1e-9
        end
        push!(points, data[:times])
    end
    @test points[1] ≈ points[2] atol=1e-9
end
nodal_test_energy(u, p) = u[2] + u[3]^2 / 2

function nodal_test_run(h0, H; angle=0.3, speed=0.0, energy=true)
    opts = PropagatorOptions(propagator=HamiltonianPropagator(),
        integrator=Vern9(), poincare_callback=true,
        poincare_sections=[:nodal_section], section_angle=angle,
        section_min_speed=speed,
        section_hamiltonian=energy ? nodal_test_energy : nothing)
    cb, data = CelestialMechanics.PropagatorUtils.setup_all_poincare_callbacks(
        opts, HamiltonEquations())
    prob = ODEProblem(nodal_test_rhs!, [2.0, 1.5, H, 0.0, 0.0, h0],
        (0.0, 20.0), (mu=1.0,))
    solve(prob, Vern9(); callback=cb, dtmax=0.2, abstol=1e-12, reltol=1e-12)
    data[:nodal_section]
end

@testset "Nodal section: exact crossings and diagnostics" begin
    data = nodal_test_run(-0.2, 1.0)
    expected = [0.5 + 2*pi*k for k in 0:3]
    @test data[:times] ≈ expected atol=1e-9
    @test length(data[:canonical_states]) == length(expected)
    @test all(diff(data[:times]) .> 0)
    @test all(abs(sin(u[6] - 0.3)) < 1e-9 for u in data[:canonical_states])
    @test all(cos(u[6] - 0.3) > 0 for u in data[:canonical_states])
    @test all(==(1.0), data[:nodal_rates])
    @test maximum(abs.(data[:energies] .- 2.0)) < 1e-11
    @test isempty(nodal_test_run(-0.2, -1.0)[:times])
    @test isempty(nodal_test_run(0.3, 0.0)[:times])
    @test isempty(nodal_test_run(-0.2, 1.0; speed=1.1)[:times])
    shifted = nodal_test_run(-0.2 + 4*pi, 1.0)
    @test shifted[:times] ≈ expected atol=1e-9
    initial = nodal_test_run(0.3, 1.0)
    @test initial[:times] ≈ [2*pi*k for k in 1:3] atol=1e-9
    @test all(isnan, nodal_test_run(-0.2, 1.0; energy=false)[:energies])
end

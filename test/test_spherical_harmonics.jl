# test/test_spherical_harmonics.jl

using Test
using CelestialMechanics
using LinearAlgebra
using StaticArrays
using Unitful

const PE = CelestialMechanics.PerturbationEquations

@testset "SphericalHarmonics (Cunningham recurrence)" begin

    mu_moon = 4902.80012
    R_moon  = 1738.1
    r_pos   = SVector(2000.0, 500.0, 400.0)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Normalization factors and roundtrip
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Normalization factors" begin
        # N_20 = sqrt(5)
        @test normalization_factor(2, 0) ≈ sqrt(5.0) atol=1e-14
        # N_22 = sqrt(10 / 24)
        @test normalization_factor(2, 2) ≈ sqrt(10.0 / 24.0) atol=1e-14

        # Roundtrip normalization / denormalization
        C_orig = zeros(5, 5)
        S_orig = zeros(5, 5)
        for n in 0:4, m in 0:n
            C_orig[n + 1, m + 1] = rand()
            S_orig[n + 1, m + 1] = rand()
        end
        C_bar, S_bar = normalize_coefficients(C_orig, S_orig; degree=4, order=4)
        C_back, S_back = denormalize_coefficients(C_bar, S_bar; degree=4, order=4)

        @test C_back ≈ C_orig atol=1e-14
        @test S_back ≈ S_orig atol=1e-14
    end

    # Helper to calculate the central finite-difference gradient of the potential ∇U
    function _num_grad(r, t, mu, R, C, S; degree, order, omega_rot=0.0, cs_phase=false, h=1e-3)
        g = zeros(3)
        for i in 1:3
            dr = zeros(3); dr[i] = h
            Up = cunningham_potential(r + dr, t, mu, R, C, S; degree=degree, order=order, omega_rot=omega_rot, cs_phase=cs_phase)
            Um = cunningham_potential(r - dr, t, mu, R, C, S; degree=degree, order=order, omega_rot=omega_rot, cs_phase=cs_phase)
            g[i] = (Up - Um) / (2 * h)
        end
        return SVector{3}(g)
    end

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Zonal Harmonics comparison: Cunningham vs Analytical J2 and vs Numerical Potential Gradient
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Zonal harmonics J2 to J18: Cunningham acceleration vs Potential Gradient" begin
        for n in 2:18
            jn_val = 1.0e-4 / n
            C = zeros(n + 1, 1)
            S = zeros(n + 1, 1)
            C[n + 1, 1] = -jn_val

            a_cunningham = cunningham_perturbation(r_pos, 0.0, mu_moon, R_moon, C, S; degree=n, order=0)
            
            # For J2, verify against analytical j2_perturbation to machine precision
            if n == 2
                a_j2 = PE.j2_perturbation(r_pos, mu_moon, R_moon, jn_val)
                diff_rel_j2 = norm(a_cunningham - a_j2) / norm(a_j2)
                @test diff_rel_j2 < 1e-13
            end

            # Verify analytical acceleration equals gradient of potential ∇U
            a_grad = _num_grad(r_pos, 0.0, mu_moon, R_moon, C, S; degree=n, order=0)
            diff_rel = norm(a_cunningham - a_grad) / norm(a_cunningham)
            @test diff_rel < 1e-7
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Sectoral & Tesseral harmonics comparison: Cunningham vs Potential Gradient (t=0)
    # ─────────────────────────────────────────────────────────────────────────
    @testset "CS harmonics at t=0: Cunningham acceleration vs Potential Gradient" begin
        cs_pairs = [
            (2, 2),
            (3, 1), (3, 2), (3, 3),
            (4, 1), (4, 2), (4, 3), (4, 4)
        ]

        for (n, m) in cs_pairs
            c_val = 1.2e-5 / (n * m)
            s_val = -0.8e-5 / (n * m)

            C = zeros(n + 1, m + 1)
            S = zeros(n + 1, m + 1)
            C[n + 1, m + 1] = c_val
            S[n + 1, m + 1] = s_val

            cs_flag = isodd(m)
            a_cunningham = cunningham_perturbation(r_pos, 0.0, mu_moon, R_moon, C, S;
                                                   degree=n, order=m, cs_phase=cs_flag)

            a_grad = _num_grad(r_pos, 0.0, mu_moon, R_moon, C, S;
                               degree=n, order=m, cs_phase=cs_flag)

            diff_rel = norm(a_cunningham - a_grad) / norm(a_cunningham)
            @test diff_rel < 1e-7
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Sectoral & Tesseral in rotating frame (t > 0, omega_rot > 0)
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Rotating body frame: Cunningham acceleration vs Potential Gradient" begin
        t_epoch = 4567.8
        omega = 2.5e-5

        cs_pairs = [(2, 2), (3, 2), (4, 2), (4, 4)]
        for (n, m) in cs_pairs
            c_val = 1.5e-5 / (n * m)
            s_val = -0.7e-5 / (n * m)

            C = zeros(n + 1, m + 1)
            S = zeros(n + 1, m + 1)
            C[n + 1, m + 1] = c_val
            S[n + 1, m + 1] = s_val

            a_cunningham = cunningham_perturbation(r_pos, t_epoch, mu_moon, R_moon, C, S;
                                                   degree=n, order=m, omega_rot=omega)

            a_grad = _num_grad(r_pos, t_epoch, mu_moon, R_moon, C, S;
                               degree=n, order=m, omega_rot=omega)

            diff_rel = norm(a_cunningham - a_grad) / norm(a_cunningham)
            @test diff_rel < 1e-7
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Normalized vs Unnormalized coefficients
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Normalized coefficient handling" begin
        # Unnormalized coefficients
        C_unnorm = zeros(3, 3)
        S_unnorm = zeros(3, 3)
        C_unnorm[3, 1] = -1.08263e-3  # -J2
        C_unnorm[3, 3] = 1.57e-6      # C22
        S_unnorm[3, 3] = -0.90e-6     # S22

        # Convert to normalized
        C_bar, S_bar = normalize_coefficients(C_unnorm, S_unnorm; degree=2, order=2)

        # Evaluate Cunningham with unnormalized
        a1 = cunningham_perturbation(r_pos, 0.0, mu_moon, R_moon, C_unnorm, S_unnorm;
                                     degree=2, order=2, normalized=false)

        # Evaluate Cunningham with normalized
        a2 = cunningham_perturbation(r_pos, 0.0, mu_moon, R_moon, C_bar, S_bar;
                                     degree=2, order=2, normalized=true)

        @test a1 ≈ a2 atol=1e-15
    end

    # ─────────────────────────────────────────────────────────────────────────
    # 6. GravityHarmonics struct, callable syntax and CunninghamWorkspace
    # ─────────────────────────────────────────────────────────────────────────
    @testset "GravityHarmonics struct and callable" begin
        Jn_dict = Dict(2 => 1.0e-4, 3 => -2.0e-5)
        gh = GravityHarmonics(max_degree=3, max_order=0, Jn=Jn_dict)

        a_gh = gh(r_pos, 0.0, mu_moon, R_moon)

        a_j2 = PE.j2_perturbation(r_pos, mu_moon, R_moon, 1.0e-4)
        
        C3 = zeros(4, 1); S3 = zeros(4, 1)
        C3[4, 1] = 2.0e-5
        a_j3 = cunningham_perturbation(r_pos, 0.0, mu_moon, R_moon, C3, S3; degree=3, order=0)

        @test a_gh ≈ (a_j2 + a_j3) atol=1e-15

        # Preallocated workspace in-place check
        ws = CunninghamWorkspace(gh; T=Float64)
        a_ws = cunningham_perturbation!(ws, r_pos, 0.0, mu_moon, R_moon, 0.0)
        @test a_ws ≈ a_gh atol=1e-15
    end

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Orbit propagation comparison (Cowell propagator)
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Cowell orbit propagation: Cunningham vs Symbolic J2" begin
        # Test orbit propagation with use_cunningham=false vs use_cunningham=true
        ic = InitialConditions(
            a0 = 42164.0u"km",
            e0 = 0.01,
            i0 = 1.0u"°",
            h0 = 0.0u"°",
            g0 = 0.0u"°",
            f0 = 0.0u"°"
        )
        tspan = (0.0, 7200.0) # 2 hours in seconds
        t_vec = collect(range(0.0, 7200.0, length=50))

        # 1. Model with classical symbolic J2
        p_symbolic = create_perturbation_model(:earth, j_harmonics=[2], use_cunningham=false)

        # 2. Model with Cunningham J2
        p_cunningham = create_perturbation_model(:earth, j_harmonics=[2], use_cunningham=true)

        opt = PropagatorOptions(
            integrator = DifferentialEquations.Vern7(),
            abstol = 1e-12,
            reltol = 1e-12,
            canonical_unit_normalization = true
        )

        res_sym = run_simulation(
            ics = [ic],
            perturbation_params = p_symbolic,
            tspan = tspan,
            t_vector = t_vec,
            propagator_options = opt,
            output_directory = mktempdir()
        )

        res_cun = run_simulation(
            ics = [ic],
            perturbation_params = p_cunningham,
            tspan = tspan,
            t_vector = t_vec,
            propagator_options = opt,
            output_directory = mktempdir()
        )

        sol_sym = res_sym[1].solution
        sol_cun = res_cun[1].solution

        # Trajectories should agree to very high precision
        diff_state = norm(sol_sym.u[end] - sol_cun.u[end])
        @test diff_state < 1e-7
    end

    @testset "ICGEM & SHA normalized flags" begin
        gfc_data = """
        product_type            gravity_field
        modelname               test_egm
        earth_gravity_constant  398600.4415e9
        radius                  6378136.3
        max_degree              2
        norm                    fully_normalized
        end_of_head ====================================================================
        gfc  2  0 -0.484165143790815e-03  0.000000000000000e+00  0.0  0.0
        gfc  2  2  0.243938357328313e-05 -0.140027370385934e-05  0.0  0.0
        """
        tmp_gfc = tempname() * ".gfc"
        write(tmp_gfc, gfc_data)

        # Default denormalizes
        g_unnorm = load_gravity_icgem(tmp_gfc; max_degree=2)
        @test g_unnorm.normalized == false
        @test isapprox(g_unnorm.Jn[2], 0.0010826261738522225; rtol=1e-12)

        # normalized=true retains normalized coefficients
        g_norm = load_gravity_icgem(tmp_gfc; max_degree=2, normalized=true)
        @test g_norm.normalized == true
        @test isapprox(g_norm.Jn[2], 0.000484165143790815; rtol=1e-12)

        # icgem_to_body_data with normalized=true
        b_norm = icgem_to_body_data(tmp_gfc; max_degree=2, normalized=true)
        @test b_norm.normalized_gravity == true
        @test isapprox(b_norm.j2, 0.000484165143790815; rtol=1e-12)

        # GravityHarmonics from namedtuple
        gh_from_grav = GravityHarmonics(g_norm)
        @test gh_from_grav.normalized == true
        @test isapprox(gh_from_grav.C[3, 1], -0.0010826261738522225; rtol=1e-12)

        rm(tmp_gfc)
    end

    @testset "PolyhedralGravity (mesh to harmonics)" begin
        # 2x2x2 cube centered at origin
        cube_obj = """
        v -1.0 -1.0 -1.0
        v  1.0 -1.0 -1.0
        v  1.0  1.0 -1.0
        v -1.0  1.0 -1.0
        v -1.0 -1.0  1.0
        v  1.0 -1.0  1.0
        v  1.0  1.0  1.0
        v -1.0  1.0  1.0
        f 1 2 3
        f 1 3 4
        f 5 7 6
        f 5 8 7
        f 1 6 2
        f 1 5 6
        f 2 7 3
        f 2 6 7
        f 3 8 4
        f 3 7 8
        f 4 5 1
        f 4 8 5
        """
        tmp_obj = tempname() * ".obj"
        write(tmp_obj, cube_obj)

        verts, faces = load_shape_obj(tmp_obj)
        @test length(verts) == 8
        @test length(faces) == 12

        poly_grav = polyhedron_harmonics(verts, faces, 1.0, 2)
        @test poly_grav isa Vector
        idx_j2 = findfirst(c -> c.n == 2 && c.m == 0, poly_grav)
        @test !isnothing(idx_j2)
        @test abs(poly_grav[idx_j2].Cnm) < 1e-12

        # Convert to body data
        body_data = polyhedron_to_body_data(tmp_obj, 1.0, 2; name=:cube, spice_id="CUBE", omega_rot=0.0)
        @test haskey(body_data, :name)
        @test haskey(body_data, :ref_radius)
        @test haskey(body_data, :j2)

        rm(tmp_obj)
    end
end

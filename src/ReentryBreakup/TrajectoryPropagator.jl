# src/ReentryBreakup/TrajectoryPropagator.jl

"""
    TrajectoryPropagator

Módulo de alta performance para propagação temporal da trajetória de cada fragmento da nuvem de detritos
sob ação simultânea de:
1. Gravidade central da Terra (μ_Earth = 398600.4418 km³/s²).
2. Perturbação de achatamento zonal J2 da Terra (J2 = 0.00108263).
3. Arrasto aerodinâmico atmosférico (US76) com rotação da atmosfera (ω_Earth = 7.292115e-5 rad/s).
4. Detecção contínua e precisa de impacto no solo (altitude geodésica WGS-84 h <= 0.0 km).

Oferece suporte nativo a paralelismo por multithreading (`Threads.@threads`).
"""
module TrajectoryPropagator

using LinearAlgebra
using Printf
using Statistics
using Base.Threads

using ..BreakupTypes: DebrisFragment, BreakupEvent
using ..ReentryAtmosphere: us_standard_atmosphere_1976
using ..ImpactFootprint: ImpactPoint, DebrisFootprint

export propagate_fragment, propagate_debris_cloud

# Constantes geofísicas WGS-84 / IERS
const MU_EARTH_KM3_S2   = 398600.4418
const R_EARTH_EQUAT_KM  = 6378.137
const WGS84_F           = 1.0 / 298.257223563
const WGS84_E2          = WGS84_F * (2.0 - WGS84_F)
const J2_EARTH          = 0.00108263
const OMEGA_EARTH_RAD_S = 7.2921151467e-5

"""
    eci_to_geodetic_approx(r_eci::AbstractVector, t_sec::Float64) -> (lat_deg, lon_deg, h_km)

Converte posição inercial ECI (km) para coordenadas geodésicas aproximadas (WGS-84) no instante `t_sec`:
Considera a rotação da Terra θ_rot = ω_Earth * t_sec.
"""
function eci_to_geodetic_approx(r_eci::AbstractVector, t_sec::Float64)
    x = r_eci[1]
    y = r_eci[2]
    z = r_eci[3]

    # Rotação em torno do eixo Z para referencial ECEF
    theta_rot = OMEGA_EARTH_RAD_S * t_sec
    cos_th = cos(theta_rot)
    sin_th = sin(theta_rot)

    x_ecef =  cos_th * x + sin_th * y
    y_ecef = -sin_th * x + cos_th * y
    z_ecef =  z

    p = sqrt(x_ecef^2 + y_ecef^2)
    lon_rad = atan(y_ecef, x_ecef)

    # Iteração de Bowring para latitude geodésica
    lat_rad = atan(z_ecef, p * (1.0 - WGS84_E2))
    for _ in 1:4
        s_lat = sin(lat_rad)
        N = R_EARTH_EQUAT_KM / sqrt(1.0 - WGS84_E2 * s_lat^2)
        lat_rad = atan(z_ecef + WGS84_E2 * N * s_lat, p)
    end

    s_lat, c_lat = sincos(lat_rad)
    N = R_EARTH_EQUAT_KM / sqrt(1.0 - WGS84_E2 * s_lat^2)
    h_km = p * c_lat + z_ecef * s_lat - R_EARTH_EQUAT_KM * sqrt(1.0 - WGS84_E2 * s_lat^2)

    return rad2deg(lat_rad), rad2deg(lon_rad), h_km
end

"""
    compute_derivatives(r::AbstractVector, v::AbstractVector, cd::Float64, am::Float64, t_sec::Float64)

Calcula as derivadas do estado [rx, ry, rz, vx, vy, vz] incluindo:
- Gravidade Newtoniana
- Perturbação J2
- Arrasto atmosférico US76 com co-rotação
"""
@inline function compute_derivatives(r::AbstractVector, v::AbstractVector, cd::Float64, am::Float64, t_sec::Float64)
    rx, ry, rz = r[1], r[2], r[3]
    vx, vy, vz = v[1], v[2], v[3]

    r_mag = sqrt(rx^2 + ry^2 + rz^2)
    r3 = r_mag^3
    r5 = r_mag^5

    # 1. Gravidade Central
    a_grav_x = -MU_EARTH_KM3_S2 * rx / r3
    a_grav_y = -MU_EARTH_KM3_S2 * ry / r3
    a_grav_z = -MU_EARTH_KM3_S2 * rz / r3

    # 2. Perturbação Zonal J2
    z2_r2 = (rz / r_mag)^2
    factor_j2 = -1.5 * J2_EARTH * MU_EARTH_KM3_S2 * (R_EARTH_EQUAT_KM^2) / r5
    a_j2_x = factor_j2 * rx * (1.0 - 5.0 * z2_r2)
    a_j2_y = factor_j2 * ry * (1.0 - 5.0 * z2_r2)
    a_j2_z = factor_j2 * rz * (3.0 - 5.0 * z2_r2)

    # 3. Arrasto Aerodinâmico
    _, _, h_km = eci_to_geodetic_approx(r, t_sec)
    rho = us_standard_atmosphere_1976(h_km)

    # Velocidade da atmosfera co-rotante: v_atm = [-ω*y, ω*x, 0]
    v_atm_x = -OMEGA_EARTH_RAD_S * ry
    v_atm_y =  OMEGA_EARTH_RAD_S * rx
    v_atm_z =  0.0

    v_rel_x = vx - v_atm_x
    v_rel_y = vy - v_atm_y
    v_rel_z = vz - v_atm_z
    v_rel_mag = sqrt(v_rel_x^2 + v_rel_y^2 + v_rel_z^2)

    # Conversão de unidades: 0.5 * 1e3 * rho * cd * am * v_rel * v_rel_vec (km/s^2)
    factor_drag = -0.5e3 * rho * cd * am * v_rel_mag
    a_drag_x = factor_drag * v_rel_x
    a_drag_y = factor_drag * v_rel_y
    a_drag_z = factor_drag * v_rel_z

    ax = a_grav_x + a_j2_x + a_drag_x
    ay = a_grav_y + a_j2_y + a_drag_y
    az = a_grav_z + a_j2_z + a_drag_z

    return [ax, ay, az], h_km
end

"""
    propagate_fragment(
        fragment::DebrisFragment,
        r_break_ref::Vector{Float64},
        t_hat_track::Vector{Float64},
        c_hat_track::Vector{Float64};
        max_duration::Float64 = 3600.0,
        dt_initial::Float64 = 0.50,
        dt_save::Float64 = 2.0,
        save_trajectory::Bool = true
    ) -> ImpactPoint

Propaga um único fragmento da altitude de ruptura (~78 km) até o solo (h <= 0.0 km) utilizando
integrador adaptativo Runge-Kutta 4ª ordem com passo controlado e interpolação de impacto.
"""
function propagate_fragment(
    fragment::DebrisFragment,
    r_break_ref::Vector{Float64},
    t_hat_track::Vector{Float64},
    c_hat_track::Vector{Float64};
    max_duration::Float64 = 3600.0,
    dt_initial::Float64 = 0.50,
    dt_save::Float64 = 2.0,
    save_trajectory::Bool = true
)
    r = copy(fragment.r0)
    v = copy(fragment.v0)
    cd = fragment.cd
    am = fragment.area_to_mass

    t = 0.0
    dt = dt_initial

    traj_records = Vector{Vector{Float64}}()

    # Ponto inicial
    lat, lon, h = eci_to_geodetic_approx(r, t)
    if save_trajectory
        push!(traj_records, [t, r[1], r[2], r[3], v[1], v[2], v[3], h, lat, lon])
    end

    last_save_t = 0.0

    while t < max_duration && h > 0.0
        # Adaptação do passo conforme a altitude diminui (arrasto cresce exponencialmente)
        if h < 20.0
            dt = min(dt, 0.05)
        elseif h < 50.0
            dt = min(dt, 0.15)
        else
            dt = min(dt, 0.50)
        end

        # Passos Runge-Kutta 4 (RK4 clássico robusto e estável)
        k1_v, _ = compute_derivatives(r, v, cd, am, t)
        k1_r = v

        r_k2 = r .+ 0.5 .* dt .* k1_r
        v_k2 = v .+ 0.5 .* dt .* k1_v
        k2_v, _ = compute_derivatives(r_k2, v_k2, cd, am, t + 0.5 * dt)
        k2_r = v_k2

        r_k3 = r .+ 0.5 .* dt .* k2_r
        v_k3 = v .+ 0.5 .* dt .* k2_v
        k3_v, _ = compute_derivatives(r_k3, v_k3, cd, am, t + 0.5 * dt)
        k3_r = v_k3

        r_k4 = r .+ dt .* k3_r
        v_k4 = v .+ dt .* k3_v
        k4_v, _ = compute_derivatives(r_k4, v_k4, cd, am, t + dt)
        k4_r = v_k4

        r_next = r .+ (dt / 6.0) .* (k1_r .+ 2.0 .* k2_r .+ 2.0 .* k3_r .+ k4_r)
        v_next = v .+ (dt / 6.0) .* (k1_v .+ 2.0 .* k2_v .+ 2.0 .* k3_v .+ k4_v)
        t_next = t + dt

        lat_next, lon_next, h_next = eci_to_geodetic_approx(r_next, t_next)

        # Verificação de cruzamento com o solo (h <= 0.0)
        if h_next <= 0.0
            # Interpolação linear fina para o exato ponto de contato h = 0.0
            fraction = h / (h - h_next)
            fraction = clamp(fraction, 0.0, 1.0)
            t_impact = t + fraction * dt
            r_impact = r .+ fraction .* (r_next .- r)
            v_impact = v .+ fraction .* (v_next .- v)

            lat_imp, lon_imp, _ = eci_to_geodetic_approx(r_impact, t_impact)

            # Velocidade de impacto relativa à superfície da Terra em rotação
            v_atm_imp = [-OMEGA_EARTH_RAD_S * r_impact[2], OMEGA_EARTH_RAD_S * r_impact[1], 0.0]
            v_ground = v_impact .- v_atm_imp
            v_term_ms = norm(v_ground) * 1e3
            ke_joules = 0.5 * fragment.mass * (v_term_ms^2)

            # Cálculo de Downrange e Crossrange relativos ao ponto de ruptura
            delta_r = r_impact .- r_break_ref
            downrange = dot(delta_r, t_hat_track)
            crossrange = dot(delta_r, c_hat_track)

            if save_trajectory
                push!(traj_records, [t_impact, r_impact[1], r_impact[2], r_impact[3],
                                     v_impact[1], v_impact[2], v_impact[3], 0.0, lat_imp, lon_imp])
            end

            traj_matrix = isempty(traj_records) ? zeros(Float64, 0, 10) : reduce(vcat, [rec' for rec in traj_records])

            return ImpactPoint(
                fragment.id,
                fragment.is_dense_component,
                fragment.mass,
                fragment.area_to_mass,
                t_impact,
                r_impact,
                v_impact,
                v_term_ms,
                ke_joules,
                lat_imp,
                lon_imp,
                downrange,
                crossrange,
                traj_matrix
            )
        end

        r = r_next
        v = v_next
        t = t_next
        h = h_next

        if save_trajectory && (t - last_save_t >= dt_save)
            push!(traj_records, [t, r[1], r[2], r[3], v[1], v[2], v[3], h, lat_next, lon_next])
            last_save_t = t
        end
    end

    # Caso atinja o tempo máximo sem tocar o solo
    lat_end, lon_end, _ = eci_to_geodetic_approx(r, t)
    v_atm_end = [-OMEGA_EARTH_RAD_S * r[2], OMEGA_EARTH_RAD_S * r[1], 0.0]
    v_ground = v .- v_atm_end
    v_term_ms = norm(v_ground) * 1e3
    ke_joules = 0.5 * fragment.mass * (v_term_ms^2)
    delta_r = r .- r_break_ref
    downrange = dot(delta_r, t_hat_track)
    crossrange = dot(delta_r, c_hat_track)

    traj_matrix = isempty(traj_records) ? zeros(Float64, 0, 10) : reduce(vcat, [rec' for rec in traj_records])

    return ImpactPoint(
        fragment.id,
        fragment.is_dense_component,
        fragment.mass,
        fragment.area_to_mass,
        t,
        r,
        v,
        v_term_ms,
        ke_joules,
        lat_end,
        lon_end,
        downrange,
        crossrange,
        traj_matrix
    )
end

"""
    propagate_debris_cloud(
        event::BreakupEvent;
        max_duration::Float64 = 3600.0,
        dt_initial::Float64 = 0.50,
        dt_save::Float64 = 2.0,
        save_trajectories::Bool = true,
        parallel::Bool = true
    ) -> DebrisFootprint

Propaga todos os fragmentos da nuvem a partir do ponto de quebra até o solo com arrasto aerodinâmico.
Calcula a elipse de dispersão de detritos (*Debris Footprint*), tempos de queda, segregação balística
e energia cinética de impacto.
"""
function propagate_debris_cloud(
    event::BreakupEvent;
    max_duration::Float64 = 3600.0,
    dt_initial::Float64 = 0.50,
    dt_save::Float64 = 2.0,
    save_trajectories::Bool = true,
    parallel::Bool = true
)
    N = event.num_fragments

    # Construção do sistema local de coordenadas ao longo da trajetória (Ground Track)
    r_break = event.r_sat
    v_break = event.v_sat

    r_norm = norm(r_break)
    u_rad = r_norm > 1e-6 ? (r_break ./ r_norm) : [0.0, 0.0, 1.0]

    # Velocidade tangencial à superfície
    v_tang = v_break .- dot(v_break, u_rad) .* u_rad
    v_tang_norm = norm(v_tang)
    t_hat = v_tang_norm > 1e-6 ? (v_tang ./ v_tang_norm) : [1.0, 0.0, 0.0]

    # Vetor lateral de cross-track
    c_hat = cross(t_hat, u_rad)
    c_hat_norm = norm(c_hat)
    c_hat = c_hat_norm > 1e-6 ? (c_hat ./ c_hat_norm) : [0.0, 1.0, 0.0]

    impact_points = Vector{ImpactPoint}(undef, N)

    if parallel && nthreads() > 1
        @threads for i in 1:N
            impact_points[i] = propagate_fragment(
                event.fragments[i],
                r_break,
                t_hat,
                c_hat;
                max_duration = max_duration,
                dt_initial = dt_initial,
                dt_save = dt_save,
                save_trajectory = save_trajectories
            )
        end
    else
        for i in 1:N
            impact_points[i] = propagate_fragment(
                event.fragments[i],
                r_break,
                t_hat,
                c_hat;
                max_duration = max_duration,
                dt_initial = dt_initial,
                dt_save = dt_save,
                save_trajectory = save_trajectories
            )
        end
    end

    # Estatísticas do Footprint
    downranges = [p.downrange_km for p in impact_points]
    crossranges = [p.crossrange_km for p in impact_points]
    total_mass_ground = sum(p.mass for p in impact_points)
    total_ke_mj = sum(p.kinetic_energy_joules for p in impact_points) * 1e-6
    mean_t_flight = mean([p.t_flight for p in impact_points])

    fp_length = maximum(downranges) - minimum(downranges)
    fp_width  = maximum(crossranges) - minimum(crossranges)

    return DebrisFootprint(
        event,
        impact_points,
        N,
        total_mass_ground,
        total_ke_mj,
        fp_length,
        fp_width,
        mean_t_flight
    )
end

end # module TrajectoryPropagator

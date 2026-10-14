# ext/CelestialMechanicsNRLMSISE00Ext.jl

module CelestialMechanicsNRLMSISE00Ext

using CelestialMechanics
using CelestialMechanics.Types: NRLMSISE00Atmosphere
import CelestialMechanics.AtmosphericModels: nrlmsise00_density
import SatelliteToolboxAtmosphericModels as STAM

function nrlmsise00_density(m::NRLMSISE00Atmosphere, jd::Real, h_m::Real, lat_rad::Real, lon_rad::Real)
    # If space indices (F10a, F10, ap) were explicitly provided, use constant indices:
    out = if !isnothing(m.F10a) && !isnothing(m.F10) && !isnothing(m.ap)
        STAM.AtmosphericModels.nrlmsise00(
            jd, h_m, lat_rad, lon_rad, m.F10a, m.F10, m.ap;
            include_anomalous_oxygen = m.include_anomalous_oxygen
        )
    else
        # Automatic fetching via SpaceIndices (user must have initialized SpaceIndices.init()):
        STAM.AtmosphericModels.nrlmsise00(
            jd, h_m, lat_rad, lon_rad;
            include_anomalous_oxygen = m.include_anomalous_oxygen
        )
    end
    return out.total_density
end

end # module

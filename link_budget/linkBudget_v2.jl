# ============================================================
# MODULAR LINK BUDGET CALCULATOR — CLEAN EDITION
# ============================================================
# All parameters are read from link_budget_config.csv.
#
# Fixes vs. original (modular_link_budget.jl):
#   [F1] RF bandwidth derived from modulation order, FEC code rate,
#        and RRC roll-off — no longer an independent (inconsistent) input.
#   [F2] Cascaded noise figure computed via Friis formula in linear
#        domain; not by adding dB values.
#   [F3] Transmission-line noise uses the standard Pozar/Balanis
#        formula: T_in = T_ant/L + T_phys·(1 − 1/L).
#   [F4] RX antenna gain and 3 dB beamwidth are derived from dish
#        diameter and frequency; not independently specified.
#   [F5] Pointing loss uses the geometry-derived beamwidth [F4],
#        removing the inconsistency between dish size and beamwidth.
#   [F6] A single cable_length parameter drives both the power-path
#        insertion loss and the noise-path thermal contribution.
#   [F7] Ground-station name is an explicit config field, not a comment.
#   [F8] ITU noise reference temperature standardised to 290 K throughout.
#
# Parameters added vs. original:
#   + LNA gain            (for Friis cascade — was hardcoded 20 dB)
#   + Backend NF          (SDR NF — was hardcoded 10 dB)
#   + FEC code rate       (affects symbol rate and occupied bandwidth)
#   + RRC roll-off factor (affects occupied bandwidth)
#   + Target BER          (documents the Eb/N0 threshold)
#   + cable_loss_db_per_100m  (frequency-specific — was hardcoded)
#   + physical_temperature    (cable thermal noise reference)
#   + fading_margin_db    (was hardcoded to 0 dB)
#   + min_elevation_mask  (was hardcoded in multiple places)
#   + Ground station name
# ============================================================

using Printf
using Statistics
using LinearAlgebra
using CSV
using DataFrames
using Dates
using Plots
using ItuRPropagation

# ============================================================================
# SECTION 1 — PHYSICAL CONSTANTS
# ============================================================================
const C_LIGHT   = 299_792_458.0          # Speed of light, m/s  (exact SI)
const K_B       = 1.380_649e-23          # Boltzmann constant, J/K (exact SI)
const K_B_DBM   = 10 * log10(K_B * 1e3) # dBm/Hz/K  = −198.60 dBm/Hz/K
const RE        = 6_371_000.0            # Mean Earth radius, m
const F_WGS84   = 1.0 / 298.257_223_563 # WGS84 flattening
const E_SQ      = 2F_WGS84 - F_WGS84^2 # First eccentricity squared
const T_REF     = 290.0                  # ITU/IEEE noise reference temp, K

# ============================================================================
# SECTION 2 — CONFIG LOADER
# ============================================================================

"""
    load_config(path) → Dict{String, Any}

Reads `path` (CSV with columns: category, parameter, value, unit, description).
Numeric strings → Float64.  All else → String.
"""
function load_config(path::String)::Dict{String, Any}
    df  = CSV.read(path, DataFrame; types=String, silencewarnings=true)
    cfg = Dict{String, Any}()
    for row in eachrow(df)
        key = strip(string(row.parameter))
        raw = strip(string(row.value))
        cfg[key] = something(tryparse(Float64, raw), raw)
    end
    return cfg
end

gf(cfg, k, d=0.0)  = Float64(get(cfg, k, d))   # fetch as Float64
gs(cfg, k, d="")   = string(get(cfg, k, d))     # fetch as String

# ============================================================================
# SECTION 3 — DERIVED LINK PARAMETERS
# ============================================================================

"""Return number of information bits encoded per channel symbol."""
function bits_per_symbol(mod::AbstractString)::Int
    table = Dict("BPSK"=>1, "QPSK"=>2, "8PSK"=>3,
                 "16QAM"=>4, "32QAM"=>5, "64QAM"=>6)
    key   = uppercase(strip(mod))
    haskey(table, key) && return table[key]
    @warn "Unknown modulation '$mod'; defaulting to QPSK (2 bits/symbol)"
    return 2
end

"""
    derive_link_params(cfg) → NamedTuple

Single function that converts raw CSV values into every derived quantity
needed by the link budget.  Physics is applied once, here.
"""
function derive_link_params(cfg::Dict{String,Any})

    # ── Frequency / Wavelength ───────────────────────────────────────────────
    f_hz = gf(cfg, "frequency")
    λ    = C_LIGHT / f_hz

    # ── Waveform / Modulation  [F1] ──────────────────────────────────────────
    mod_str   = gs(cfg, "modulation", "QPSK")
    bps       = bits_per_symbol(mod_str)
    code_rate = gf(cfg, "fec_code_rate", 0.5)
    α_ro      = gf(cfg, "rolloff_factor", 0.35)
    data_rate = gf(cfg, "data_rate")

    # Channel symbol rate includes FEC overhead:
    #   info bits/s ÷ code_rate → channel bits/s ÷ bits_per_symbol → symbols/s
    symbol_rate  = (data_rate / code_rate) / bps
    bandwidth_hz = symbol_rate * (1.0 + α_ro)   # occupied RF bandwidth

    # ── RX Dish Geometry  [F4] ───────────────────────────────────────────────
    D   = gf(cfg, "antenna_diameter")
    η   = gf(cfg, "antenna_efficiency", 0.81)
    rx_gain_dbi = 10 * log10(η * (π * D / λ)^2)
    rx_bw_deg   = 70.0 * λ / D                  # 3 dB beamwidth (degrees)

    # ── Pointing Loss  [F5] ──────────────────────────────────────────────────
    pt_err    = gf(cfg, "pointing_error", 0.0)
    pt_loss   = -12.0 * (pt_err / rx_bw_deg)^2

    # ── Cascaded Noise Figure (Friis)  [F2] ──────────────────────────────────
    NF1_dB = gf(cfg, "lna_noise_figure",     1.5)
    G1_dB  = gf(cfg, "lna_gain",            20.0)
    NF2_dB = gf(cfg, "backend_noise_figure", 10.0)
    F1     = 10^(NF1_dB / 10)
    G1     = 10^(G1_dB  / 10)
    F2     = 10^(NF2_dB / 10)
    NF_cas_dB = 10 * log10(F1 + (F2 - 1.0) / G1)

    # ── Cable — single source of truth  [F6] ─────────────────────────────────
    cable_len       = gf(cfg, "cable_length",           1.0)
    loss_per_100m   = gf(cfg, "cable_loss_db_per_100m", 49.0)
    rx_line_loss_db = loss_per_100m * cable_len / 100.0

    return (
        frequency_hz        = f_hz,
        wavelength_m        = λ,
        bandwidth_hz        = bandwidth_hz,
        symbol_rate_sps     = symbol_rate,
        data_rate_bps       = data_rate,
        bits_per_sym        = bps,
        fec_code_rate       = code_rate,
        modulation          = mod_str,
        rx_antenna_gain_dbi = rx_gain_dbi,
        rx_beamwidth_deg    = rx_bw_deg,
        pointing_loss_db    = pt_loss,
        rx_noise_figure_db  = NF_cas_dB,
        rx_line_loss_db     = rx_line_loss_db,
    )
end

"""Print a human-readable summary of all derived parameters."""
function print_derived_params(dp, cfg)
    println("\n" * "="^70)
    println("DERIVED LINK PARAMETERS")
    println("="^70)
    @printf("  Ground station  : %s\n",   gs(cfg, "name"))
    @printf("  Location        : %.4f° N,  %.4f° E,  %.0f m\n",
            gf(cfg, "latitude"), gf(cfg, "longitude"), gf(cfg, "altitude"))
    println()
    @printf("  Carrier freq    : %.3f GHz\n", dp.frequency_hz / 1e9)
    @printf("  Wavelength      : %.4f m\n",   dp.wavelength_m)
    @printf("  Modulation      : %s  (%d bits/symbol)\n",
            dp.modulation, dp.bits_per_sym)
    @printf("  FEC code rate   : %.2f\n",  dp.fec_code_rate)
    @printf("  RRC roll-off    : %.2f\n",  gf(cfg, "rolloff_factor", 0.35))
    @printf("  Data rate       : %.3f kbps\n",   dp.data_rate_bps   / 1e3)
    @printf("  Symbol rate     : %.3f ksps\n",   dp.symbol_rate_sps / 1e3)
    @printf("  RF bandwidth    : %.3f kHz   [derived: R_s × (1+α)]  [F1]\n",
            dp.bandwidth_hz / 1e3)
    println()
    @printf("  RX dish diam.   : %.2f m\n",   gf(cfg, "antenna_diameter"))
    @printf("  RX aperture eff : %.0f%%\n",   gf(cfg, "antenna_efficiency") * 100)
    @printf("  RX gain         : %.2f dBi  [derived from geometry]  [F4]\n",
            dp.rx_antenna_gain_dbi)
    @printf("  RX 3dB bw       : %.2f°     [derived from geometry]  [F4]\n",
            dp.rx_beamwidth_deg)
    @printf("  Pointing error  : %.1f°\n",  gf(cfg, "pointing_error", 0.0))
    @printf("  Pointing loss   : %.2f dB  [uses geometry beamwidth] [F5]\n",
            dp.pointing_loss_db)
    println()
    @printf("  LNA NF          : %.2f dB\n", gf(cfg, "lna_noise_figure"))
    @printf("  LNA gain        : %.1f dB\n",  gf(cfg, "lna_gain"))
    @printf("  Backend NF      : %.1f dB\n",  gf(cfg, "backend_noise_figure"))
    @printf("  Cascaded NF     : %.3f dB  [Friis, linear domain]    [F2]\n",
            dp.rx_noise_figure_db)
    println()
    @printf("  Cable length    : %.2f m\n",    gf(cfg, "cable_length"))
    @printf("  Cable loss/100m : %.1f dB/100m\n", gf(cfg, "cable_loss_db_per_100m", 49.0))
    @printf("  RX line loss    : %.4f dB  [single source]           [F6]\n",
            dp.rx_line_loss_db)
    println("="^70)
end

# ============================================================================
# SECTION 4 — COORDINATE GEOMETRY
# ============================================================================

"""Geodetic (lat_rad, lon_rad, h_m) → ECEF [x, y, z] in metres."""
function geodetic_to_ecef(lat_rad, lon_rad, h_m)
    N = RE / sqrt(1.0 - E_SQ * sin(lat_rad)^2)
    x = (N + h_m) * cos(lat_rad) * cos(lon_rad)
    y = (N + h_m) * cos(lat_rad) * sin(lon_rad)
    z = (N * (1.0 - E_SQ) + h_m) * sin(lat_rad)
    return [x, y, z]
end

"""Elevation angle (degrees) from ground station ECEF to satellite ECEF."""
function elevation_angle(sat_ecef, gs_ecef)
    los   = sat_ecef .- gs_ecef
    cos_z = clamp(dot(los, gs_ecef) / (norm(los) * norm(gs_ecef)), -1.0, 1.0)
    return 90.0 - acosd(cos_z)
end

"""Off-nadir angle (degrees) at satellite toward the ground station."""
function nadir_off_boresight(sat_ecef, gs_ecef)
    boresight = -sat_ecef          # nadir direction
    to_gs     = gs_ecef .- sat_ecef
    cos_a = clamp(dot(to_gs, boresight) / (norm(to_gs) * norm(boresight)), -1.0, 1.0)
    return acosd(cos_a)
end

# ============================================================================
# SECTION 5 — ANTENNA MODELS
# ============================================================================

"""
Patch antenna gain pattern (cosine^n model).
theta_deg: off-boresight angle; hpbw_deg: half-power beamwidth.
"""
function patch_antenna_gain(theta_deg, hpbw_deg, max_gain_dbi=5.0)
    θ = deg2rad(abs(theta_deg))
    n = log(0.5) / log(cos(deg2rad(hpbw_deg / 2)))
    g_lin = max(cos(θ)^n, 0.0)
    return 10 * log10(g_lin + 1e-10) + max_gain_dbi
end

# ============================================================================
# SECTION 6 — ATMOSPHERIC PROPAGATION
# ============================================================================

"""Free-space path loss (dB)."""
function free_space_loss(distance_m, frequency_hz)
    λ = C_LIGHT / frequency_hz
    return 20 * log10(4π * distance_m / λ)
end

"""
ITU-R atmospheric attenuation via ItuRPropagation.jl.
Returns a named tuple (Ac, Ag, Ar, As, At, Tsky, XPD).
Falls back to elevation = 5° when elevation_deg < 5° (ITU model lower limit).
"""
function itu_attenuation(lat, lon, f_hz, exceedance, elevation_deg,
                          ant_d, ant_eff_pct, alt_km)
    el_clamped = max(elevation_deg, 5.0)
    ll  = LatLon(lat, lon)
    f_GHz = f_hz / 1e9
    return ItuRPropagation.downlinkparameters(
        ll, f_GHz, exceedance, el_clamped, ant_d, ant_eff_pct, alt_km)
end

# ============================================================================
# SECTION 7 — NOISE MODEL  (fixed)
# ============================================================================

"""
Antenna noise temperature as a function of elevation angle.
Piecewise-linear model: horizon (150 K) → high elevation (100 K).
Galactic background (10 K) and spillover (25 K) added.
"""
function antenna_noise_temperature(elevation_deg)
    T_gal      = 10.0
    T_spillover = 70
    if elevation_deg < 5.0
        T_sky = 150.0
    elseif elevation_deg < 30.0
        T_sky = 100.0 + (150.0 - 100.0) * (30.0 - elevation_deg) / 25.0
    else
        T_sky = 100.0
    end
    return T_sky + T_gal + T_spillover
end



"""
Convert noise figure (dB) to equivalent noise temperature (K).
Reference temperature = T_REF = 290 K (ITU/IEEE standard).  [F8]
"""
function receiver_noise_temperature(nf_db)
    return T_REF * (10^(nf_db / 10) - 1.0)
end

"""
System noise temperature via the standard Pozar/Balanis
transmission-line formula.                                 [F3]

  T_sys = T_ant/L + T_phys·(1 − 1/L) + T_receiver

where L = 10^(line_loss_db/10) ≥ 1 is the linear loss factor.

Arguments:
  T_ant         : antenna noise temperature (K)
  T_phys        : physical cable temperature (K)
  T_receiver    : receiver equivalent noise temperature (K)
  line_loss_db  : cable insertion loss (dB, positive number)
"""
function system_noise_temperature(T_ant, T_phys, T_receiver, line_loss_db)
    L    = 10^(line_loss_db / 10)
    T_in = T_ant / L + T_phys * (1.0 - 1.0 / L)
    return T_in + T_receiver
end

# ============================================================================
# SECTION 8 — LINK BUDGET CORE
# ============================================================================

"""
Calculate the complete link budget for one geometry point.
All derived quantities (gain, noise figure, bandwidth, …) come from
`derive_link_params`; raw config scalars are passed directly.
"""
function calculate_link_budget(;
        # Transmitter
        tx_power_dbm,
        tx_antenna_gain_dbi,
        tx_line_loss_db,
        # Receiver (derived)
        rx_antenna_gain_dbi,
        rx_line_loss_db,
        rx_noise_figure_db,
        rx_beamwidth_deg,
        # Geometry
        distance_m,
        elevation_deg,
        gs_lat, gs_lon, gs_alt_km,
        # Waveform
        frequency_hz,
        bandwidth_hz,
        data_rate_bps,
        required_ebn0_db,
        # Environmental
        polarization_loss_db,
        fading_margin_db,
        pointing_loss_db,
        exceedance,
        ant_d,
        ant_eff_pct,
        T_cable_physical = 290.0,
    )

    # ── EIRP ──────────────────────────────────────────────────────────────────
    eirp_dbm = tx_power_dbm + tx_antenna_gain_dbi - tx_line_loss_db

    # ── Path Losses ───────────────────────────────────────────────────────────
    fspl = free_space_loss(distance_m, frequency_hz)

    itu  = itu_attenuation(gs_lat, gs_lon, frequency_hz,
                            exceedance, elevation_deg,
                            ant_d, ant_eff_pct, gs_alt_km)
    atm_total = itu.At      # already includes cloud, rain, gas, scintillation

    total_path_loss = (fspl + atm_total
                      + polarization_loss_db
                      + fading_margin_db
                      + pointing_loss_db)

    # ── Received Power ────────────────────────────────────────────────────────
    rx_power_dbm = eirp_dbm - total_path_loss + rx_antenna_gain_dbi - rx_line_loss_db

    # ── Noise ─────────────────────────────────────────────────────────────────
    T_ant = antenna_noise_temperature(elevation_deg)
    T_rx  = receiver_noise_temperature(rx_noise_figure_db)           # [F2, F8]
    T_sys = system_noise_temperature(T_ant, T_cable_physical,        # [F3, F6]
                                      T_rx, rx_line_loss_db)

    noise_power_dbm = K_B_DBM + 10*log10(T_sys) + 10*log10(bandwidth_hz)

    # ── C/N, C/N0, Eb/N0 ─────────────────────────────────────────────────────
    cn_db     = rx_power_dbm - noise_power_dbm
    cn0_dbhz  = rx_power_dbm - K_B_DBM - 10*log10(T_sys)
    ebn0_db   = cn0_dbhz - 10*log10(data_rate_bps)

    # ── Link Margin & G/T ─────────────────────────────────────────────────────
    link_margin_db = ebn0_db - required_ebn0_db
    fom_dBK        = rx_antenna_gain_dbi - 10*log10(T_sys)   # G/T

    return (
        eirp_dbm       = eirp_dbm,
        fspl_db        = fspl,
        atm_loss_db    = atm_total,
        rain_loss_db   = itu.Ar,
        cloud_loss_db  = itu.Ac,
        total_loss_db  = total_path_loss,
        rx_power_dbm   = rx_power_dbm,
        noise_power_dbm= noise_power_dbm,
        T_sys_K        = T_sys,
        cn_db          = cn_db,
        cn0_dbhz       = cn0_dbhz,
        ebn0_db        = ebn0_db,
        link_margin_db = link_margin_db,
        fom_dBK        = fom_dBK,
        link_closed    = link_margin_db > 0.0,
    )
end

# ============================================================================
# SECTION 9 — ELEVATION SWEEP DIAGNOSTIC
# ============================================================================

function print_elevation_sweep(cfg, dp)
    println("\n" * "="^130)
    println("LINK BUDGET vs ELEVATION  (zenith → horizon, every 5°)")
    println("="^130)
    @printf("%-6s %-8s %-9s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n",
        "El(°)", "OBore(°)", "Dist(km)",
        "EIRP(dBm)", "FSPL(dB)", "AtmL(dB)",
        "TotL(dB)", "RxPwr(dBm)", "Tsys(K)",
        "G/T(dBK)", "BW(kHz)", "Eb/N0(dB)", "Margin")
    println("-"^130)

    gs_lat = gf(cfg, "latitude")
    gs_lon = gf(cfg, "longitude")
    gs_alt = gf(cfg, "altitude")
    h      = gf(cfg, "orbit_altitude")
    gs_ecef = geodetic_to_ecef(deg2rad(gs_lat), deg2rad(gs_lon), gs_alt)

    tx_pwr  = gf(cfg, "tx_power_dbm")
    tx_ll   = gf(cfg, "tx_line_loss_db")
    hpbw    = gf(cfg, "sat_antenna_hpbw")
    pk_gain = gf(cfg, "sat_antenna_peak_gain")
    pol_l   = gf(cfg, "polarization_loss_db")
    fad_m   = gf(cfg, "fading_margin_db")
    exc     = gf(cfg, "exceedance_probability")
    req_eb  = gf(cfg, "required_ebn0_db")
    pt_l    = dp.pointing_loss_db
    T_cab   = gf(cfg, "physical_temperature", 290.0)

    for el in 90.0:-5.0:5.0
        nadir_rad  = asin(RE * cosd(el) / (RE + h))
        eca_rad    = deg2rad(90.0 - el) - nadir_rad
        sat_lat    = gs_lat
        sat_lon    = gs_lon + rad2deg(eca_rad)
        sat_ecef   = geodetic_to_ecef(deg2rad(sat_lat), deg2rad(sat_lon), h)

        elev_check = elevation_angle(sat_ecef, gs_ecef)
        ant_angle  = nadir_off_boresight(sat_ecef, gs_ecef)
        distance   = norm(sat_ecef .- gs_ecef)
        sat_gain   = patch_antenna_gain(ant_angle, hpbw, pk_gain)

        r = calculate_link_budget(
            tx_power_dbm        = tx_pwr,
            tx_antenna_gain_dbi = sat_gain,
            tx_line_loss_db     = tx_ll,
            rx_antenna_gain_dbi = dp.rx_antenna_gain_dbi,
            rx_line_loss_db     = dp.rx_line_loss_db,
            rx_noise_figure_db  = dp.rx_noise_figure_db,
            rx_beamwidth_deg    = dp.rx_beamwidth_deg,
            distance_m          = distance,
            elevation_deg       = elev_check,
            gs_lat              = gs_lat,
            gs_lon              = gs_lon,
            gs_alt_km           = gs_alt / 1000.0,
            frequency_hz        = dp.frequency_hz,
            bandwidth_hz        = dp.bandwidth_hz,
            data_rate_bps       = dp.data_rate_bps,
            required_ebn0_db    = req_eb,
            polarization_loss_db= pol_l,
            fading_margin_db    = fad_m,
            pointing_loss_db    = pt_l,
            exceedance          = exc,
            ant_d               = gf(cfg, "antenna_diameter"),
            ant_eff_pct         = gf(cfg, "antenna_efficiency") * 100,
            T_cable_physical    = T_cab,
        )

        marker = r.link_closed ? "✓ CLOSED" : "✗ OPEN"
        @printf("%-6.1f %-8.2f %-9.1f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.1f %-10.2f %-10.2f %-10.2f %s\n",
            elev_check, ant_angle, distance/1e3,
            r.eirp_dbm, r.fspl_db, r.atm_loss_db,
            r.total_loss_db, r.rx_power_dbm, r.T_sys_K,
            r.fom_dBK, dp.bandwidth_hz/1e3, r.ebn0_db, marker)
    end
    println("="^130)
end

# ============================================================================
# SECTION 10 — GEOGRAPHIC COVERAGE ANALYSIS
# ============================================================================

function analyze_geographic_coverage(cfg, dp)
    println("\n" * "="^70)
    println("GEOGRAPHIC LINK COVERAGE ANALYSIS")
    println("="^70)

    gs_lat  = gf(cfg, "latitude");   gs_lon = gf(cfg, "longitude")
    gs_alt  = gf(cfg, "altitude");   h      = gf(cfg, "orbit_altitude")
    res     = gf(cfg, "grid_resolution", 1.0)
    lat_r   = (gf(cfg, "lat_range_min"), gf(cfg, "lat_range_max"))
    lon_r   = (gf(cfg, "lon_range_min"), gf(cfg, "lon_range_max"))
    el_mask = gf(cfg, "min_elevation_mask", 5.0)

    gs_ecef = geodetic_to_ecef(deg2rad(gs_lat), deg2rad(gs_lon), gs_alt)
    lat_grid = (gs_lat + lat_r[1]):res:(gs_lat + lat_r[2])
    lon_grid = (gs_lon + lon_r[1]):res:(gs_lon + lon_r[2])
    n_lat, n_lon = length(lat_grid), length(lon_grid)

    println("Grid  : $n_lat × $n_lon  ($(round(res, digits=2))° resolution)")
    @printf("GS    : %.4f° N,  %.4f° E,  %.0f m\n", gs_lat, gs_lon, gs_alt)

    el_map  = fill(NaN, n_lat, n_lon)
    aa_map  = fill(NaN, n_lat, n_lon)
    lm_map  = fill(-999.0, n_lat, n_lon)
    rx_map  = fill(-999.0, n_lat, n_lon)
    d_map   = zeros(n_lat, n_lon)
    eb_map  = fill(NaN, n_lat, n_lon)
    at_map  = fill(NaN, n_lat, n_lon)
    gt_map  = fill(NaN, n_lat, n_lon)

    tx_pwr  = gf(cfg, "tx_power_dbm");  tx_ll = gf(cfg, "tx_line_loss_db")
    hpbw    = gf(cfg, "sat_antenna_hpbw"); pk_gain = gf(cfg, "sat_antenna_peak_gain")
    pol_l   = gf(cfg, "polarization_loss_db"); fad_m = gf(cfg, "fading_margin_db")
    exc     = gf(cfg, "exceedance_probability"); req_eb = gf(cfg, "required_ebn0_db")
    T_cab   = gf(cfg, "physical_temperature", 290.0)
    ant_d   = gf(cfg, "antenna_diameter"); ant_eff_pct = gf(cfg, "antenna_efficiency") * 100

    println("\nCalculating coverage…")
    for (i, lat) in enumerate(lat_grid)
        for (j, lon) in enumerate(lon_grid)
            sat_ecef  = geodetic_to_ecef(deg2rad(lat), deg2rad(lon), h)
            elev      = elevation_angle(sat_ecef, gs_ecef)
            ant_angle = nadir_off_boresight(sat_ecef, gs_ecef)
            dist      = norm(sat_ecef .- gs_ecef)

            el_map[i,j] = elev
            aa_map[i,j] = ant_angle
            d_map[i,j]  = dist

            elev < el_mask && continue

            sat_gain = patch_antenna_gain(ant_angle, hpbw, pk_gain)
            r = calculate_link_budget(
                tx_power_dbm        = tx_pwr,
                tx_antenna_gain_dbi = sat_gain,
                tx_line_loss_db     = tx_ll,
                rx_antenna_gain_dbi = dp.rx_antenna_gain_dbi,
                rx_line_loss_db     = dp.rx_line_loss_db,
                rx_noise_figure_db  = dp.rx_noise_figure_db,
                rx_beamwidth_deg    = dp.rx_beamwidth_deg,
                distance_m          = dist,
                elevation_deg       = elev,
                gs_lat              = gs_lat,
                gs_lon              = gs_lon,
                gs_alt_km           = gs_alt / 1000.0,
                frequency_hz        = dp.frequency_hz,
                bandwidth_hz        = dp.bandwidth_hz,
                data_rate_bps       = dp.data_rate_bps,
                required_ebn0_db    = req_eb,
                polarization_loss_db= pol_l,
                fading_margin_db    = fad_m,
                pointing_loss_db    = dp.pointing_loss_db,
                exceedance          = exc,
                ant_d               = ant_d,
                ant_eff_pct         = ant_eff_pct,
                T_cable_physical    = T_cab,
            )

            lm_map[i,j] = r.link_margin_db
            rx_map[i,j] = r.rx_power_dbm
            eb_map[i,j] = r.ebn0_db
            at_map[i,j] = r.atm_loss_db
            gt_map[i,j] = r.fom_dBK
        end
        i % 10 == 0 && @printf("  %.0f%%\n", 100i/n_lat)
    end
    println("✓ Complete")

    valid = lm_map .>= 0
    any(valid) && @printf("Minimum link elevation angle: %.2f°\n",
                           minimum(el_map[valid]))

    return (lat_grid=collect(lat_grid), lon_grid=collect(lon_grid),
            elevation=el_map, antenna_angle=aa_map,
            link_margin=lm_map, rx_power=rx_map,
            distance=d_map, link_ebn0=eb_map,
            atm_loss=at_map, fom=gt_map)
end

# ============================================================================
# SECTION 11 — VISUALISATION
# ============================================================================

function plot_coverage_maps(results, gs_lat, gs_lon)
    lat, lon = results.lat_grid, results.lon_grid
    p = Plots.plot(layout=(4,2), size=(2000, 1600), background_color=:black)

    nan_floor(A, lo) = (B = copy(A); B[B .< lo] .= NaN; B)

    Plots.contourf!(p[1], lon, lat, results.elevation',
        title="Elevation (°)", xlabel="Lon", ylabel="Lat", colorbar=true)
    Plots.scatter!(p[1], [gs_lon], [gs_lat], marker=:star, color=:red,
        markersize=8, label="GS")

    Plots.contourf!(p[2], lon, lat, results.antenna_angle',
        title="Off-Boresight Angle (°)", xlabel="Lon", ylabel="Lat", colorbar=true)

    Plots.contourf!(p[3], lon, lat, results.distance' ./ 1e3,
        title="Range (km)", xlabel="Lon", ylabel="Lat", colorbar=true)

    Plots.contourf!(p[4], lon, lat, nan_floor(results.rx_power, -200)',
        title="RX Power (dBm)", xlabel="Lon", ylabel="Lat", colorbar=true)

    Plots.contourf!(p[5], lon, lat, nan_floor(results.link_margin, -100)',
        title="Link Margin (dB)", xlabel="Lon", ylabel="Lat", colorbar=true)

    avail = Float64.((results.link_margin' .>= 0) .& (results.elevation' .> 0))
    Plots.heatmap!(p[6], lon, lat, avail,
        title="Link Availability", color=:RdYlGn, colorbar=true)
    Plots.scatter!(p[6], [gs_lon], [gs_lat], marker=:star, color=:red,
        markersize=8, label="GS")

    Plots.contourf!(p[7], lon, lat, nan_floor(results.link_ebn0, -50)',
        title="Eb/N0 (dB)", xlabel="Lon", ylabel="Lat", colorbar=true)

    Plots.contourf!(p[8], lon, lat, results.atm_loss',
        title="Atm Attenuation (dB)", xlabel="Lon", ylabel="Lat", colorbar=true)

    display(p)
    Plots.savefig(p, "coverage_maps.png")
    println("✓ Saved coverage_maps.png")
    return p
end

# ============================================================================
# SECTION 12 — MAIN
# ============================================================================

function main()
    config_path = joinpath(dirname(@__FILE__), "link_budget_config.csv")

    println("="^70)
    println("MODULAR LINK BUDGET CALCULATOR — CLEAN EDITION")
    println("="^70)
    println("Config : $config_path")

    # 1. Load raw config
    cfg = load_config(config_path)

    # 2. Derive all computed quantities (single physics pass)
    dp  = derive_link_params(cfg)

    # 3. Print derived parameter summary
    print_derived_params(dp, cfg)

    # 4. Elevation sweep
    print_elevation_sweep(cfg, dp)

    # 5. Geographic coverage map
    results = analyze_geographic_coverage(cfg, dp)

    # 6. Figure of merit (G/T) at zenith
    T_sys_zenith = begin
        T_ant = antenna_noise_temperature(90.0)
        T_rx  = receiver_noise_temperature(dp.rx_noise_figure_db)
        system_noise_temperature(T_ant,
                                  gf(cfg, "physical_temperature", 290.0),
                                  T_rx,
                                  dp.rx_line_loss_db)
    end
    @printf("\nG/T (zenith, 90°): %.2f dB/K\n",
            dp.rx_antenna_gain_dbi - 10*log10(T_sys_zenith))

    # 7. Plot
    plot_coverage_maps(results, gf(cfg, "latitude"), gf(cfg, "longitude"))

    println("\n✓ Analysis complete.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
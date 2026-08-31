# ============================================================================
#  RASE / ROSE two-rephasing-pulse echo protocol, with the SECOND rephasing
#  pulse replaced by a 3-segment ARP (3ARP) composite pi-pulse.
#
#  Sequence (1st-order, :ground):
#     1. Gaussian input signal      (stored)
#     2. First WURST pulse          (silences the echo)   -- unchanged
#     3. 3ARP composite (seg 1/2/3) (revives the echo)    -- REPLACES "Second WURST pulse"
#
#  The 3ARP occupies exactly the [600us, 1000us] window the second WURST held in
#  examples/rose_1st_order.jl (T_budget = 400us -> segment durations 100/200/100us,
#  amplitudes amp1 : amp1/sqrt(2) : amp1, all sweeping bandwidth = 5*FWHM), so the
#  only change from that reference ROSE run is WURST2 -> 3ARP.
#
#  Note: this is structurally the ROSE family (deterministic input signal + two
#  rephasing elements). "RASE" proper (rephased ASE from an inverted ensemble)
#  has no deterministic signal to seed a 1st-order mean-field reference, so an
#  input-signal ROSE-style sequence is what makes RASE3ARP.jld2 usable as a
#  reference for optimise_control_pulse_from_jld2 later (signal identified as
#  background; WURST1 + 3ARP = the one control envelope).
#
#  Writes:  data/data_1st_order/RASE3ARP.jld2   (+ RASE3ARP_pulsemat.csv)
#
#  RECONCILE NOTE: this is a TWO-pi-pulse sequence (WURST1 + 3ARP), so the net
#  population returns to ground and the reference "inversion" is ~0. The stored
#  state (a / Sigma_p / Sigma_z) reconciles to ~1e-5, but reconcile_reference's
#  *relative* inversion check is ill-conditioned near zero (|stored_inv| ~ 1e-6,
#  same order as the default atol_use = 100*abstol). Run the pipeline on this
#  file with atol_check = 1e-3, e.g.
#     optimise_control_pulse_from_jld2("data/data_1st_order/RASE3ARP.jld2";
#         atol_check = 1e-3, param_budget = 60, num_epochs = 20, n_hops = 2,
#         grad_mode = :adjoint, ...)
#  (verified: atol_check=1e-3 -> reconcile PASS, rel_inv 1.8e-4; state checks
#   unaffected. Contrast 3ARP_pi_gstd_1em06Hz.jld2, a single 3ARP = one pi,
#   whose inversion ~1 makes the relative check well-conditioned.)
# ============================================================================

using InhomogeneousSpinCavityDynamics

OUTDIR = joinpath(@__DIR__, "..", "data", "data_1st_order")
mkpath(OUTDIR)

SIM_SETTING = (
    simulation_order = :order1,

    # discretization -- constant coupling (g_std negligible), so M_g = 1.
    M_delta = 3000,
    M_g     = 1,

    initial_condition = :ground,

    Ttotal  = 1100e-6,          # last pulse ends at 1000us; ~100us settle
    Nt_save = 11001,
    reltol  = 1e-8,
    abstol  = 1e-8,

    saved_file_name = joinpath(OUTDIR, "RASE3ARP.jld2"),

    # silenced echo (after WURST1) and revived echo (after the 3ARP)
    peak_detection = (
        labels = [:echo_silenced, :echo_revived],
        times = [550e-6, 1050e-6],
        half_window = 10e-6,
    ),
)

SYSTEM_CONFIG = (
    C_ens   = 0.6,

    delta0  = 0.0,
    kappa_e = 2*pi*1e6,
    kappa_i = 2*pi*0,

    freq_inhomogeneity = (
        kind = :lorentzian,
        FWHM = 2*pi*1e6,
        span_gamma = 2.5,
        renormalize = false,
    ),

    g_inhomogeneity = (
        kind = :constant,
        g_value = 2π * 100,
    ),
)

FWHM   = SYSTEM_CONFIG.freq_inhomogeneity.FWHM
SQRTKE = sqrt(SYSTEM_CONFIG.kappa_e)

# paper conventions (src/datagen/datagen_pulse.jl):
#   signal amp = 0.5*sqrt(kappa_e)*0.332 ,  WURST amp = 0.5*sqrt(kappa_e)*2.0e4
SIGNAL_AMP = 0.5 * SQRTKE * 0.332
WURST_AMP  = 0.5 * SQRTKE * 2.0e4          # == amp1 of the existing 3ARP reference
BW         = 5.0 * FWHM

PULSE_CONFIG = (
    (
        name  = "Gaussian input signal",
        kind  = :gaussian,
        t0    = 50e-6,
        sigma = 3e-6,
        amp   = SIGNAL_AMP,
        omega = 0.0,
        phase = 0.0,
    ),

    (
        name       = "First WURST pulse",
        kind       = :wurst,
        t_center   = 300e-6,
        duration   = 400e-6,           # spans [100us, 500us]
        amp        = WURST_AMP,
        bandwidth  = BW,
        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 1e-4,
    ),

    # --- 3ARP composite: replaces the "Second WURST pulse" in [600us, 1000us] ---
    (
        name       = "3ARP segment 1 (+k)",
        kind       = :wurst,
        t_center   = 650e-6,           # [600us, 700us]
        duration   = 100e-6,
        amp        = WURST_AMP,
        bandwidth  = BW,
        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 1e-4,
    ),
    (
        name       = "3ARP segment 2 (+k/2)",
        kind       = :wurst,
        t_center   = 800e-6,           # [700us, 900us]
        duration   = 200e-6,
        amp        = WURST_AMP / sqrt(2),
        bandwidth  = BW,
        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 1e-4,
    ),
    (
        name       = "3ARP segment 3 (+k)",
        kind       = :wurst,
        t_center   = 950e-6,           # [900us, 1000us]
        duration   = 100e-6,
        amp        = WURST_AMP,
        bandwidth  = BW,
        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 1e-4,
    ),
)

run_simulation(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG)

println("Wrote reference file: ", SIM_SETTING.saved_file_name)
nothing

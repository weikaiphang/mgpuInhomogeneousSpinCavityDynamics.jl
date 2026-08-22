# ============================================================
# 1ST-ORDER PARAMETER SWEEP
#
# Runs N_RUNS independent 1st-order simulations built around the
# same two-WURST photon-echo sequence used in
# examples/rose_1st_order.jl and paper/fig_3_b/ace_under_various_c.jl.
#
# STRICTLY 1ST ORDER:
#   simulation_order is hardcoded to :order1 below and is never
#   overwritten by the random draws. A runtime @assert right
#   before every run_simulation() call additionally refuses to
#   proceed if it is ever anything else.
#
# PHYSICS-CONSISTENT RANGES:
#   Every value below is either held fixed at the exact value
#   used throughout paper/, or drawn from the exact range that
#   paper/ validates as swept. Nothing is drawn from an invented
#   or extrapolated range. See the comment above each field for
#   its source. Where a parameter is fixed in every paper script
#   (e.g. kappa_e, the WURST amp/bandwidth/n/chirp_sign, the
#   Gaussian-input pulse shape), it is fixed here too — the paper
#   gives no evidence that other values are physically valid for
#   this model, so it is not swept.
#
# Each run is saved to its own .jld2 file under
# data/sweep_1st_order/, and a JSON manifest recording the drawn
# parameters and pass/fail status for every run is written to
# data/sweep_1st_order/manifest.json.
#
# Re-running this script skips any run whose output file already
# exists, so an interrupted sweep (e.g. a GPU driver crash) can
# simply be restarted.
# ============================================================

using InhomogeneousSpinCavityDynamics
using Random
using Printf
using JSON3

const N_RUNS = 200
const SEED   = 20260818

const OUTDIR = joinpath(@__DIR__, "..", "data", "sweep_1st_order")
mkpath(OUTDIR)

const MANIFEST_FILE = joinpath(OUTDIR, "manifest.json")

Random.seed!(SEED)

# ============================================================
# RANDOM-DRAW HELPERS
# ============================================================

rand_in(lo, hi)   = lo + rand() * (hi - lo)
rand_choice(opts) = opts[rand(1:length(opts))]

# ============================================================
# VALIDATED SWEEP RANGES (cited to paper/)
# ============================================================

# C_ens: paper/fig_3_b/ace_under_various_c.jl sweeps 0.1-0.6 (and
# its own comment gives range(0.1, 1.0) as a valid alternative
# scan); paper/fig_3_d/*.jl (2nd-order, same C_ens meaning)
# extends the validated lower end to 0.05 and upper end to 0.8.
# Union of both: [0.05, 1.0].
const C_ENS_RANGE = (0.05, 1.0)

# M_delta: the only two values used for 1st-order runs anywhere
# in paper/ are 1000 (fig_3_c/silencing_factor.jl) and 3000
# (fig_3_b/ace_under_various_c.jl). Only these two are drawn from.
const M_DELTA_OPTIONS = [1000, 3000]

# g_inhomogeneity std (Gaussian coupling spread), in Hz (converted
# to rad/s below): paper/fig_3_c/silencing_factor.jl sweeps
# g_std_Hz up to 0.6 Hz at WURST duration = 100 us — the closest
# validated duration to the 400 us WURST used in this sequence
# (paper/fig_3_b, examples/rose_1st_order.jl). Since fig_3_c shows
# the tolerable g_std shrinking as WURST duration grows (2.4 Hz at
# 10 us -> 1.4 Hz at 30 us -> 0.6 Hz at 100 us), 0.6 Hz is used
# here as a conservative upper bound for the even longer 400 us
# pulse, rather than extrapolating past what was measured.
const G_STD_HZ_RANGE = (1e-6, 0.6)

# ============================================================
# CONFIG BUILDER
#
# All pulse timing/shape parameters (Gaussian input sigma/amp,
# WURST t_center/duration/amp/bandwidth/n/chirp_sign/edge_frac),
# Ttotal, the cavity (kappa_e, kappa_i, delta0), the detuning
# distribution (freq_inhomogeneity), initial_condition, and the
# solver tolerances are held fixed at the exact values shared by
# every 1st-order script in paper/ and examples/rose_1st_order.jl.
# Only C_ens, the coupling-distribution kind/spread, and M_delta
# are randomized, using the validated ranges above.
# ============================================================

function build_configs(run_id)
    Ttotal = 1100e-6

    # --- coupling distribution ---
    # kind and g_value/mean are the only forms paper/ validates
    # for 1st-order runs (fig_3_b/fig_3_d use :constant with
    # g_value = 2pi*100; fig_3_c/fig_3_d use :gaussian with
    # mean = 2pi*100). :powerlaw_g and :user_defined are never
    # used anywhere in paper/, so they are not sampled here.
    g_kind = rand_choice([:constant, :gaussian])

    g_inhomogeneity, M_g = if g_kind == :constant
        (
            (
                kind = :constant,
                g_value = 2 * pi * 100.0,
            ),
            1,
        )
    else
        g_std_Hz = rand_in(G_STD_HZ_RANGE...)

        (
            (
                kind = :gaussian,
                mean = 2 * pi * 100.0,
                std = 2 * pi * g_std_Hz,
                span_sigma = 3.0,
                renormalize = true,
            ),
            # M_g = 20 matches the only 1st-order Gaussian-coupling
            # run in paper/ (fig_3_c/silencing_factor.jl). M_g is a
            # discretization resolution for the g-distribution, not
            # itself a physical parameter, so reusing it here for a
            # different pulse topology is a numerical choice, not a
            # physics extrapolation.
            20,
        )
    end

    SYSTEM_CONFIG = (
        C_ens = rand_in(C_ENS_RANGE...),

        delta0 = 0.0,
        kappa_e = 2 * pi * 1e6,
        kappa_i = 2 * pi * 0.0,

        freq_inhomogeneity = (
            kind = :lorentzian,
            FWHM = 2 * pi * 1e6,
            span_gamma = 2.5,
            renormalize = false,
        ),

        g_inhomogeneity = g_inhomogeneity,
    )

    SIM_SETTING = (
        simulation_order = :order1,

        M_delta = rand_choice(M_DELTA_OPTIONS),
        M_g     = M_g,

        initial_condition = :ground,

        Ttotal = Ttotal,

        Nt_save = 5001,
        reltol  = 1e-8,
        abstol  = 1e-8,

        saved_file_name = joinpath(OUTDIR, @sprintf("run_%03d.jld2", run_id)),

        peak_detection = (
            labels = [:echo1, :echo2],
            times = [550e-6, 1050e-6],
            half_window = 10e-6,
        ),
    )

    PULSE_CONFIG = (
        (
            name  = "Gaussian input signal",
            kind  = :gaussian,

            t0    = 50e-6,
            sigma = 3e-6,
            amp   = 0.5 * sqrt(SYSTEM_CONFIG.kappa_e) * 0.332,
            omega = 0.0,
            phase = 0.0,
        ),
        (
            name       = "First WURST pulse",
            kind       = :wurst,

            t_center   = 300e-6,
            duration   = 400e-6,
            amp        = 0.5 * sqrt(SYSTEM_CONFIG.kappa_e) * 2.0e4,
            bandwidth  = 5.0 * SYSTEM_CONFIG.freq_inhomogeneity.FWHM,
            n          = 20.0,
            omega0     = 0.0,
            chirp_sign = +1.0,
            phase0     = 0.0,
            edge_frac  = 1e-4,
        ),
        (
            name       = "Second WURST pulse",
            kind       = :wurst,

            t_center   = 800e-6,
            duration   = 400e-6,
            amp        = 0.5 * sqrt(SYSTEM_CONFIG.kappa_e) * 2.0e4,
            bandwidth  = 5.0 * SYSTEM_CONFIG.freq_inhomogeneity.FWHM,
            n          = 20.0,
            omega0     = 0.0,
            chirp_sign = +1.0,
            phase0     = 0.0,
            edge_frac  = 1e-4,
        ),
    )

    return SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG
end

# ============================================================
# MANIFEST HELPERS
# ============================================================

function load_manifest()
    if isfile(MANIFEST_FILE)
        return Dict{String, Any}(
            String(k) => v
            for (k, v) in JSON3.read(read(MANIFEST_FILE, String))
        )
    else
        return Dict{String, Any}()
    end
end

function save_manifest(manifest)
    open(MANIFEST_FILE, "w") do io
        JSON3.write(io, manifest)
    end
    return nothing
end

# ============================================================
# MAIN SWEEP LOOP
#
# Wrapped in a function (rather than left at top-level) so the
# loop body is a normal function scope: run_id, n_ok, n_skipped,
# and n_failed are then unambiguous locals, avoiding Julia's
# top-level soft-scope warnings for loop-local assignment.
# ============================================================

function run_sweep()
    manifest = load_manifest()

    n_ok = 0
    n_skipped = 0
    n_failed = 0

    for run_id in 1:N_RUNS
        run_key = @sprintf("run_%03d", run_id)

        SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG = build_configs(run_id)

        # Hard safety check: this sweep must only ever run 1st-order
        # simulations, regardless of any future edit to build_configs.
        @assert SIM_SETTING.simulation_order === :order1 (
            "sweep_1st_order.jl must only run :order1 simulations, " *
            "got simulation_order = $(SIM_SETTING.simulation_order)"
        )

        if isfile(SIM_SETTING.saved_file_name)
            println("[$run_key] output already exists, skipping.")
            n_skipped += 1
            continue
        end

        println()
        println("=" ^ 60)
        println("[$run_key] starting  (order=$(SIM_SETTING.simulation_order), " *
                "M_delta=$(SIM_SETTING.M_delta), M_g=$(SIM_SETTING.M_g), " *
                "C_ens=$(SYSTEM_CONFIG.C_ens), " *
                "g_kind=$(SYSTEM_CONFIG.g_inhomogeneity.kind))")
        println("=" ^ 60)

        entry = Dict{String, Any}(
            "run_id" => run_id,
            "saved_file_name" => SIM_SETTING.saved_file_name,
            "simulation_order" => String(SIM_SETTING.simulation_order),
            "SIM_SETTING" => Dict(
                "M_delta" => SIM_SETTING.M_delta,
                "M_g" => SIM_SETTING.M_g,
                "initial_condition" => String(SIM_SETTING.initial_condition),
                "Ttotal" => SIM_SETTING.Ttotal,
            ),
            "SYSTEM_CONFIG" => Dict(
                "C_ens" => SYSTEM_CONFIG.C_ens,
                "kappa_e" => SYSTEM_CONFIG.kappa_e,
                "kappa_i" => SYSTEM_CONFIG.kappa_i,
                "freq_inhomogeneity" => Dict(pairs(SYSTEM_CONFIG.freq_inhomogeneity)),
                "g_inhomogeneity" => Dict(pairs(SYSTEM_CONFIG.g_inhomogeneity)),
            ),
        )

        t0 = time_ns()

        try
            run_simulation(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG; clean_gpu = true)

            entry["status"] = "ok"
            entry["elapsed_seconds"] = (time_ns() - t0) / 1e9
            n_ok += 1

            println("[$run_key] OK")

        catch err
            entry["status"] = "failed"
            entry["error"] = sprint(showerror, err)
            entry["elapsed_seconds"] = (time_ns() - t0) / 1e9
            n_failed += 1

            println("[$run_key] FAILED: $(sprint(showerror, err))")
        end

        manifest[run_key] = entry
        save_manifest(manifest)
    end

    println()
    println("Sweep finished: $n_ok ok, $n_skipped skipped, $n_failed failed " *
            "(out of $N_RUNS total).")
    println("Manifest written to: $MANIFEST_FILE")

    return nothing
end

run_sweep()

nothing

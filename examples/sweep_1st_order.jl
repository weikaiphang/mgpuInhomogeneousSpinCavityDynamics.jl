
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


rand_in(lo, hi)   = lo + rand() * (hi - lo)
rand_choice(opts) = opts[rand(1:length(opts))]


const C_ENS_RANGE = (0.05, 1.0)

const M_DELTA_OPTIONS = [1000, 3000]

const G_STD_HZ_RANGE = (1e-6, 0.6)


function build_configs(run_id)
    Ttotal = 1100e-6







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


function run_sweep()
    manifest = load_manifest()

    n_ok = 0
    n_skipped = 0
    n_failed = 0

    for run_id in 1:N_RUNS
        run_key = @sprintf("run_%03d", run_id)

        SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG = build_configs(run_id)



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
                "C_eff=$(cooperativity_honesty(SYSTEM_CONFIG).C_eff), " *
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
                "C_eff" => cooperativity_honesty(SYSTEM_CONFIG).C_eff,
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

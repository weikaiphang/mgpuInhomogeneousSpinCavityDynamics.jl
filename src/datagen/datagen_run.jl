
module DataGen

using InhomogeneousSpinCavityDynamics
using CUDA
using JLD2
using JSON3
using LinearAlgebra
using Printf

const ISC = InhomogeneousSpinCavityDynamics

include("datagen_io.jl")
include("datagen_system.jl")
include("datagen_pulse.jl")
include("datagen_execute.jl")
include("datagen_catalog.jl")

function _need_value(args, i, flag)
    i < length(args) || error("$flag requires a value.")
    return args[i + 1], i + 2
end

function _parse_int(val, flag)
    n = tryparse(Int, val)
    n === nothing && error("$flag expects an integer, got $(val).")
    return n
end

function parse_args(args)
    phase = nothing
    dry_run = false
    limit = 0
    start_id = 1
    stop_id = 0
    skip_existing = true
    conditions = "cannon"
    M_cap = RULE_M_CAP
    M_g_max = RULE_M_G_MAX
    m_sizing = "default"
    Nt_save = RULE_NT_SAVE

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--help" || a == "-h"
            return (help = true,)
        elseif a == "--phase"
            phase, i = _need_value(args, i, a)
        elseif a == "--dry-run"
            dry_run = true
            i += 1
        elseif a == "--limit"
            val, i = _need_value(args, i, a)
            limit = _parse_int(val, a)
        elseif a == "--start"
            val, i = _need_value(args, i, a)
            start_id = _parse_int(val, a)
        elseif a == "--stop"
            val, i = _need_value(args, i, a)
            stop_id = _parse_int(val, a)
        elseif a == "--no-skip"
            skip_existing = false
            i += 1
        elseif a == "--tracks" || a == "--default-conditions" || a == "--conditions"
            conditions, i = _need_value(args, i, a)
        elseif a == "--M-cap"
            val, i = _need_value(args, i, a)
            M_cap = _parse_int(val, a)
        elseif a == "--M-g-cap"
            val, i = _need_value(args, i, a)
            M_g_max = _parse_int(val, a)
        elseif a == "--M-sizing"
            m_sizing, i = _need_value(args, i, a)
        elseif a == "--NT-save"
            val, i = _need_value(args, i, a)
            Nt_save = _parse_int(val, a)
        else
            error("Unknown argument: $a")
        end
    end

    phase === nothing && error("required: --phase configs|simulate|all  (see --help)")
    phase in ("configs", "simulate", "all") || error(
        "--phase must be configs, simulate, or all, got $(phase)."
    )
    dry_run && phase == "simulate" && error(
        "--dry-run is only for --phase configs."
    )
    limit >= 0 || error("--limit must be non-negative, got $limit.")
    start_id >= 1 || error("--start must be >= 1, got $start_id.")
    stop_id >= 0 || error("--stop must be >= 0 (0 = no stop), got $stop_id.")
    stop_id == 0 || stop_id >= start_id || error(
        "--stop=$stop_id is before --start=$start_id."
    )
    return (
        help = false,
        phase = phase,
        dry_run = dry_run,
        limit = limit,
        start_id = start_id,
        stop_id = stop_id,
        skip_existing = skip_existing,
        run = make_run_params(;
            ics = parse_tracks(conditions),
            M_cap = M_cap,
            M_g_max = M_g_max,
            n_sizes = parse_m_sizing(m_sizing),
            Nt_save = Nt_save,
        ),
    )
end

function print_usage()
    println(
        """
        Datagen — first-order PINN trajectory catalog
        Documentation: src/datagen/README.md

          julia --project=. src/datagen/datagen_run.jl --phase configs [--dry-run] [--limit N]
          julia -t auto --project=. src/datagen/datagen_run.jl --phase simulate [options]
          julia -t auto --project=. src/datagen/datagen_run.jl --phase all [options]

        --phase is required (configs|simulate|all). No arguments is an error.

        configs  replaces data/datagen/configs/ with a new simulconfig catalog.
                 Do not use it to resume a simulate.
        simulate loads that catalog and runs pending (track, split) jobs.
                 Exits 1 if any entry or trajectory failed.
                 Resume with the same flags; skip is filename-based.
        all      configs then simulate. --limit applies to both (it is not
                 “simulate N of the existing catalog”).

        Simulate options:
          --start IDX --stop IDX   1-based window into the sorted configs listing
          --limit N --no-skip
          --tracks T[,T...]   ICs to integrate (default cannon = ground,equator)
                              T: ground, inverted, equator (alias equatorial),
                              weak, weak_inverted.
                              groups: poles, precess, cannon, approx, all
          --default-conditions / --conditions   aliases of --tracks
          --M-cap N --M-g-cap N --M-sizing default|N --NT-save N

        Default --M-cap 60000 and --NT-save 5001 need on the order of 10 GB
        host RAM per job. Smoke with a smaller cap before a full campaign.
        """
    )
    return nothing
end

function main(args)
    opt = parse_args(args)
    if hasproperty(opt, :help) && opt.help
        print_usage()
        return nothing
    end
    println("PINN datagen  run_rules_version=$(RUN_RULES_VERSION)")
    println("Output root: $DATAGEN_ROOT")
    if opt.phase in ("simulate", "all") && !(opt.dry_run && opt.phase == "all")
        r = opt.run
        println(
            "Run params: tracks=$(r.ics)  M-cap=$(r.M_cap)  M-g-cap=$(r.M_g_max)  " *
            "M-sizing=$(r.n_sizes == 1 ? "default" : r.n_sizes)  NT-save=$(r.Nt_save)"
        )
    end

    n_failed = 0
    if opt.phase in ("configs", "all")
        phase_configs(; dry_run = opt.dry_run, limit = opt.limit)
    end
    if opt.phase in ("simulate", "all")
        opt.dry_run && return nothing
        stats = phase_simulate(
            opt.run;
            skip_existing = opt.skip_existing,
            start_id = opt.start_id,
            stop_id = opt.stop_id,
            limit = opt.limit,
        )
        n_failed = stats.n_failed
    end
    if n_failed > 0
        println("datagen: $n_failed failed job(s) or catalog entries.")
        Base.exit(1)
    end
    return nothing
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    DataGen.main(ARGS)
end

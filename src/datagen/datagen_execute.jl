# ============================================================
# Ensemble split, trajectory execution, reduced save.
#
# From a simulconfig (SYSTEM + PULSE), at simulate time:
#   1. Ttotal from the pulse timeline (+ system settle).
#   2. M_delta_min = Ttotal * BW / 2π  (Dirichlet / Fourier bound).
#   3. Default split, under M = M_delta * M_g ≤ M_cap and safety ≥ 3:
#        - constant g: M_g = 1, largest safety with M_delta ≤ M_cap
#        - continuous g: largest M_g ≤ M_g_cap that still allows
#          safety ≥ 3, then largest M_delta for that M_g.
#      If the floor cannot fit under M_cap, the split errors.
#   4. M-sizing n>1 adds extra grids at safeties equally spaced on
#      (3, S], including S; the floor 3 itself is not a sample.
# Caps, ICs, Nt_save, and n_sizes come from simulate-time run params.
# Result files are named {stem}_{ic}_Md{M_delta}_Mg{M_g}_Nt{Nt_save}.jld2
# with ic in DATAGEN_TRACKS (ground, inverted, equator, weak, weak_inverted);
# skip if that path and its pulsemat csv both exist and are non-empty.
#
# Trajectories are dispatched one-per-functional-CUDA-device (CPU if none).
# Concurrent GPU occupancy needs julia -t N with N >= number of GPUs.
# After every job the owning worker synchronizes, GC.gc(false), and
# CUDA.reclaim()s that device; the pool reclaims every device on exit.
# ============================================================

const _DATAGEN_IO_LOCK = ReentrantLock()
const _DATAGEN_MANIFEST_LOCK = ReentrantLock()

function frequency_bandwidth(freq_inhomogeneity)
    kind = freq_inhomogeneity.kind
    FWHM = freq_inhomogeneity.FWHM
    if kind === :gaussian
        sigma = FWHM / (2 * sqrt(2 * log(2)))
        return 2 * freq_inhomogeneity.span_sigma * sigma
    elseif kind === :lorentzian
        return 2 * freq_inhomogeneity.span_gamma * (FWHM / 2)
    else
        error("Unsupported freq_inhomogeneity kind for splitting: $(kind).")
    end
end

function _cannot_meet_safety(safety_min, M_delta_needed, M_delta_min, M_cap)
    error(
        "cannot meet safety ≥ $safety_min: need M_delta ≥ $M_delta_needed " *
        "(M_delta_min=$M_delta_min) but M_cap=$M_cap."
    )
end

"""
    compute_optimal_splitting(freq_inhomogeneity, g_inhomogeneity, T_max;
                              M_cap=RULE_M_CAP, M_g_max=RULE_M_G_MAX,
                              safety_min=RULE_SAFETY_MIN) -> NamedTuple

Largest M_g ≤ M_g_max (or 1 if g is constant) and largest Fourier safety
such that M_delta * M_g ≤ M_cap and M_delta / M_delta_min ≥ safety_min.
"""
function compute_optimal_splitting(
    freq_inhomogeneity,
    g_inhomogeneity,
    T_max::Real;
    M_cap::Integer=RULE_M_CAP,
    M_g_max::Integer=RULE_M_G_MAX,
    safety_min::Real=RULE_SAFETY_MIN,
)
    M_cap >= 1 || error("M_cap must be positive, got $M_cap.")
    M_g_max >= 1 || error("M_g_max must be positive, got $M_g_max.")
    safety_min > 0 || error("safety_min must be positive, got $safety_min.")

    BW = frequency_bandwidth(freq_inhomogeneity)
    M_delta_min = (T_max * BW) / TWO_PI
    isfinite(M_delta_min) && M_delta_min > 0 || error(
        "M_delta_min must be positive and finite (T_max=$T_max, BW=$BW)."
    )

    kind_g = g_inhomogeneity.kind
    M_delta_needed = max(1, ceil(Int, safety_min * M_delta_min))

    if kind_g === :constant
        M_g = 1
        M_delta = Int(M_cap)
        M_delta_needed > M_cap && _cannot_meet_safety(safety_min, M_delta_needed, M_delta_min, M_cap)
    elseif kind_g in (:gaussian, :powerlaw_g, :user_defined)
        M_g = min(Int(M_g_max), div(Int(M_cap), M_delta_needed))
        M_g < 1 && _cannot_meet_safety(safety_min, M_delta_needed, M_delta_min, M_cap)
        M_delta = max(1, div(Int(M_cap), M_g))
    else
        error("Unsupported g_inhomogeneity kind for splitting: $(kind_g).")
    end

    safety = M_delta / M_delta_min
    safety >= safety_min || error(
        "computed safety=$safety < safety_min=$safety_min " *
        "(M_delta=$M_delta, M_delta_min=$M_delta_min, M_g=$M_g)."
    )
    M_total = M_delta * M_g
    M_total <= M_cap || error(
        "computed M_total=$M_total exceeds M_cap=$M_cap (M_delta=$M_delta, M_g=$M_g)."
    )

    return (
        M_delta = M_delta,
        M_g = M_g,
        M_total = M_total,
        M_delta_min = M_delta_min,
        BW = BW,
        safety_factor = safety,
        target_safety = safety,
        safety_min = Float64(safety_min),
        M_cap = Int(M_cap),
        M_g_max = Int(M_g_max),
    )
end

function splitting_for_run(sys, Ttotal::Float64; M_cap::Integer=RULE_M_CAP, M_g_max::Integer=RULE_M_G_MAX)
    return compute_optimal_splitting(
        sys.freq_inhomogeneity,
        sys.g_inhomogeneity,
        Ttotal;
        M_cap = M_cap,
        M_g_max = M_g_max,
        safety_min = RULE_SAFETY_MIN,
    )
end

function m_g_for_kind(kind_g, M_delta::Integer, M_cap::Integer, M_g_max::Integer)
    if kind_g === :constant
        return 1
    elseif kind_g in (:gaussian, :powerlaw_g, :user_defined)
        return min(Int(M_g_max), div(Int(M_cap), Int(M_delta)))
    else
        error("Unsupported g_inhomogeneity kind for splitting: $(kind_g).")
    end
end

function split_at_target_safety(default_split, target_safety, kind_g)
    M_delta_min = default_split.M_delta_min
    M_cap = default_split.M_cap
    M_g_max = default_split.M_g_max
    need = target_safety * M_delta_min
    isfinite(need) && need > 0 || error(
        "target_safety * M_delta_min is not a positive finite number " *
        "(target=$target_safety, M_delta_min=$M_delta_min)."
    )
    M_delta = max(1, min(Int(M_cap), ceil(Int, need)))
    M_g = m_g_for_kind(kind_g, M_delta, M_cap, M_g_max)
    M_g < 1 && error(
        "M_delta=$M_delta does not leave room for M_g under M_cap=$M_cap."
    )
    M_total = M_delta * M_g
    M_total <= M_cap || error(
        "computed M_total=$M_total exceeds M_cap=$M_cap (M_delta=$M_delta, M_g=$M_g)."
    )
    safety = M_delta / M_delta_min
    safety >= RULE_SAFETY_MIN - 1e-12 || error(
        "computed safety=$safety < safety_min=$(RULE_SAFETY_MIN) " *
        "(M_delta=$M_delta, M_delta_min=$M_delta_min, M_g=$M_g)."
    )
    return (
        M_delta = M_delta,
        M_g = M_g,
        M_total = M_total,
        M_delta_min = M_delta_min,
        BW = default_split.BW,
        safety_factor = safety,
        target_safety = Float64(target_safety),
        safety_min = Float64(RULE_SAFETY_MIN),
        M_cap = Int(M_cap),
        M_g_max = Int(M_g_max),
    )
end

function safety_targets(default_safety::Real, n_sizes::Integer)
    n_sizes >= 1 || error("n_sizes must be >= 1, got $n_sizes.")
    n_sizes == 1 && return [Float64(default_safety)]
    S = Float64(default_safety)
    S <= RULE_SAFETY_MIN && return [S]
    # n points on (3, S], equally spaced, including S.
    return collect(range(RULE_SAFETY_MIN, S; length = n_sizes + 1)[2:end])
end

"""
Default split (max safety under the cap, safety ≥ 3), plus n_sizes-1 extra
grids whose target safeties are equally spaced on (3, default_safety].
n_sizes == 1 is the default split only. Duplicate (M_delta, M_g) pairs
are dropped.
"""
function splits_for_run(sys, Ttotal::Float64, run)
    default = splitting_for_run(sys, Ttotal; M_cap = run.M_cap, M_g_max = run.M_g_max)
    targets = safety_targets(default.safety_factor, run.n_sizes)
    kind_g = sys.g_inhomogeneity.kind
    by_key = Dict{Tuple{Int,Int}, NamedTuple}()
    order = Tuple{Int,Int}[]
    for (i, s) in enumerate(targets)
        split = i == length(targets) ? default : split_at_target_safety(default, s, kind_g)
        key = (Int(split.M_delta), Int(split.M_g))
        if !haskey(by_key, key)
            push!(order, key)
        end
        by_key[key] = split
    end
    isempty(order) && error("no ensemble splits produced for Ttotal=$Ttotal.")
    return [by_key[k] for k in order]
end

function build_sim_setting(Ttotal::Float64, ic::Symbol, saved_file_name::AbstractString, split, run)
    return (
        simulation_order = RULE_SIMULATION_ORDER,
        M_delta = split.M_delta,
        M_g = split.M_g,
        initial_condition = ic,
        Ttotal = Ttotal,
        Nt_save = run.Nt_save,
        reltol = RULE_RELTOL,
        abstol = RULE_ABSTOL,
        saved_file_name = saved_file_name,
    )
end

function reduce_trajectory(t, a, Sp, Sz, d, E_of_t)
    M = Int(d.M)
    M_delta = Int(d.M_delta)
    M_g = Int(d.M_g)
    Nt = length(t)
    length(a) == Nt || error("trajectory a length $(length(a)) != Nt=$Nt.")
    if hasproperty(d, :Nt) && Nt != Int(d.Nt)
        error("trajectory length $Nt != derived Nt=$(d.Nt).")
    end
    size(Sp) == (Nt, M) && size(Sz) == (Nt, M) || error(
        "trajectory Sp/Sz shapes $(size(Sp))/$(size(Sz)) != ($((Nt, M)))."
    )

    idelta_res = argmin(abs.(d.delta_b_1d))
    delta_res = d.delta_b_1d[idelta_res]
    keep_range = idelta_res:M_delta:M
    keep_bins = collect(keep_range)
    length(keep_bins) == M_g || error(
        "resonant-delta slice length $(length(keep_bins)) != M_g=$M_g."
    )

    a_sol = collect(ComplexF64.(a))
    Σp_sol = Vector{ComplexF64}(undef, Nt)
    Σz_sol = Vector{ComplexF64}(undef, Nt)
    Sp_keep = Matrix{ComplexF64}(undef, M_g, Nt)
    Sz_keep = Matrix{ComplexF64}(undef, M_g, Nt)

    @inbounds for k in 1:Nt
        Σp_sol[k] = sum(Sp[k, :])
        Σz_sol[k] = sum(Sz[k, :])
        Sp_keep[:, k] .= ComplexF64.(Sp[k, keep_range])
        Sz_keep[:, k] .= ComplexF64.(Sz[k, keep_range])
    end

    t_saved = collect(Float64.(t))
    E_of_t_arr = ComplexF64[E_of_t(tt) for tt in t_saved]

    return (
        t_saved = t_saved,
        a_sol = a_sol,
        Σp_sol = Σp_sol,
        Σz_sol = Σz_sol,
        E_of_t_arr = E_of_t_arr,
        M_delta = M_delta,
        M_g = M_g,
        M_total = M,
        delta_b_1d = collect(d.delta_b_1d),
        g_b_1d = collect(d.g_b_1d),
        Nj_2d = collect(d.Nj_2d),
        idelta_res = idelta_res,
        delta_res = delta_res,
        keep_bins = keep_bins,
        g_keep = collect(d.g_b_1d),  # 1-D g grid, length M_g (not the flat M-vector)
        delta_keep = fill(delta_res, M_g),
        Sp_keep = Sp_keep,
        Sz_keep = Sz_keep,
        peak_detection_config = nothing,
        peak_detection_results = nothing,
        N_total = d.N_total,
    )
end

function save_datagen_result(filename, data, E_of_t)
    dir = dirname(filename)
    isempty(dir) || mkpath(dir)
    csv_path = pulsemat_from_result(filename)
    jld_tmp = filename * ".part"
    csv_tmp = csv_path * ".part"
    committed_jld = false
    committed_csv = false
    try
        isfile(jld_tmp) && rm(jld_tmp; force=true)
        isfile(csv_tmp) && rm(csv_tmp; force=true)
        JLD2.@save jld_tmp data
        ISC.sample_E_of_t(
            E_of_t,
            data.SIM_SETTING.Ttotal,
            data.SIM_SETTING.Nt_save;
            savepath = csv_tmp,
        )
        mv(jld_tmp, filename; force=true)
        committed_jld = true
        mv(csv_tmp, csv_path; force=true)
        committed_csv = true
        result_is_complete(filename) || error(
            "save did not produce a complete result pair at $filename."
        )
    catch
        isfile(jld_tmp) && rm(jld_tmp; force=true)
        isfile(csv_tmp) && rm(csv_tmp; force=true)
        # A half-written dest pair must not look skippable, and must not
        # linger as a jld2 without its pulsemat sibling.
        if committed_jld && !committed_csv
            isfile(filename) && rm(filename; force=true)
        elseif committed_jld && committed_csv
            isfile(filename) && rm(filename; force=true)
            isfile(csv_path) && rm(csv_path; force=true)
        end
        rethrow()
    end
    return filename
end

const _TRACK_TOKEN_ALIASES = Dict{String, Tuple{Vararg{Symbol}}}(
    "ground" => (:ground,),
    "inverted" => (:inverted,),
    "equator" => (:equator,),
    "equatorial" => (:equator,),
    "weak" => (:weak,),
    "weak_inverted" => (:weak_inverted,),
    "weak-inverted" => (:weak_inverted,),
    "poles" => DATAGEN_TRACK_POLES,
    "precess" => DATAGEN_TRACK_PRECESS,
    "cannon" => DATAGEN_TRACK_CANNON,
    "approx" => DATAGEN_TRACK_APPROX,
    "all" => DATAGEN_TRACKS,
)

function _tracks_help_tokens()
    return "ground, inverted, equator (alias equatorial), weak, weak_inverted; " *
        "groups: poles (=ground,inverted), precess (=weak,weak_inverted), " *
        "cannon (=ground,equator), approx (=ground,weak), all"
end

"""
    parse_tracks(s) -> Tuple{Vararg{Symbol}}

Parse `--tracks` / `--default-conditions`. Comma-separated tokens,
order-preserving, duplicates dropped. Tokens: `ground`, `inverted`,
`equator` (`equatorial`), `weak`, `weak_inverted`. Groups: `poles`,
`precess`, `cannon`, `approx`, `all`. These are ICs, not the optimizer
cost-mode `track=:dual|:weak`.
"""
function parse_tracks(s::AbstractString)
    raw = strip(String(s))
    isempty(raw) && error(
        "--tracks must be a non-empty list of $(_tracks_help_tokens())."
    )
    out = Symbol[]
    seen = Set{Symbol}()
    for part in split(raw, ',')
        t = lowercase(strip(part))
        isempty(t) && continue
        aliases = get(_TRACK_TOKEN_ALIASES, t, nothing)
        if aliases === nothing
            error(
                "unknown --tracks token $(repr(part)). Allowed: $(_tracks_help_tokens())."
            )
        end
        for ic in aliases
            is_datagen_track(ic) || error("internal: alias $(t) produced illegal track $ic.")
            if ic ∉ seen
                push!(out, ic)
                push!(seen, ic)
            end
        end
    end
    isempty(out) && error(
        "--tracks must name at least one track ($(_tracks_help_tokens())), got $(repr(s))."
    )
    return Tuple(out)
end

parse_default_conditions(s::AbstractString) = parse_tracks(s)

function parse_m_sizing(s::AbstractString)
    t = lowercase(strip(String(s)))
    t == "default" && return 1
    n = tryparse(Int, t)
    n === nothing && error("--M-sizing must be 'default' or a positive integer, got $(s).")
    n >= 1 || error("--M-sizing must be >= 1, got $n.")
    return n
end

function make_run_params(;
    ics::Tuple = DATAGEN_ICS,
    M_cap::Integer = RULE_M_CAP,
    M_g_max::Integer = RULE_M_G_MAX,
    n_sizes::Integer = 1,
    Nt_save::Integer = RULE_NT_SAVE,
)
    isempty(ics) && error("run ics must be non-empty.")
    for ic in ics
        is_datagen_track(ic) || error("unknown initial condition $ic.")
    end
    M_cap >= 1 || error("M_cap must be positive, got $M_cap.")
    M_g_max >= 1 || error("M_g_max must be positive, got $M_g_max.")
    n_sizes >= 1 || error("n_sizes must be >= 1, got $n_sizes.")
    Nt_save > 1 || error("Nt_save must be > 1, got $Nt_save.")
    return (
        ics = ics,
        M_cap = Int(M_cap),
        M_g_max = Int(M_g_max),
        n_sizes = Int(n_sizes),
        Nt_save = Int(Nt_save),
    )
end

function run_params_fingerprint(run)
    return Dict{String, Any}(
        "ics" => [String(ic) for ic in run.ics],
        "M_cap" => run.M_cap,
        "M_g_max" => run.M_g_max,
        "n_sizes" => run.n_sizes,
        "Nt_save" => run.Nt_save,
    )
end

function run_one_ic(sys, PULSE_SPEC, ic::Symbol, saved_file_name::AbstractString, split, run, Ttotal::Float64; compute::Symbol=:auto)
    SIM_SETTING = build_sim_setting(Ttotal, ic, saved_file_name, split, run)
    CONFIG = merge(SIM_SETTING, sys)
    ISC.validate_config(CONFIG)

    PULSE_CONFIG = materialize_pulse_config(PULSE_SPEC)
    ok, msg = pulse_config_is_valid(PULSE_CONFIG)
    ok || error("validate_pulse_config failed: $msg")

    d = ISC.prepare_derived(CONFIG)
    E_of_t = ISC.build_E_of_t(PULSE_CONFIG)

    t0 = time_ns()
    t, a, Sp, Sz = ISC.run_sim_1st_order_trajectory(
        E_of_t, d;
        initial_condition = ic,
        reltol = RULE_RELTOL,
        abstol = RULE_ABSTOL,
        compute = compute,
    )
    elapsed = (time_ns() - t0) / 1e9

    reduced = reduce_trajectory(t, a, Sp, Sz, d, E_of_t)
    # Host trajectories are Nt × M; drop them before the JLD2 write so the
    # peak RSS is the reduced payload, not reduced + raw.
    t = nothing
    a = nothing
    Sp = nothing
    Sz = nothing
    d = nothing
    data = merge(
        reduced,
        (
            SIM_SETTING = SIM_SETTING,
            SYSTEM_CONFIG = sys,
            PULSE_CONFIG = PULSE_SPEC.segments,
            PULSE_SPEC = PULSE_SPEC,
            # Analytic WURST/Gaussian segments are package-legal PULSE_CONFIG.
            # Composite records are not: rebuild with materialize_pulse_config(PULSE_SPEC)
            # or use the sibling _pulsemat.csv (jld2 loader load_mode=:csv).
            pulse_rebuild = any(seg -> seg.kind === :composite_record, PULSE_SPEC.segments) ?
                "pulse_spec" : "pulse_config",
            elapsed_seconds = elapsed,
            run_rules_version = RUN_RULES_VERSION,
            run_params = run_params_fingerprint(run),
            safety_factor = split.safety_factor,
            target_safety = split.target_safety,
            M_delta_min = split.M_delta_min,
        ),
    )
    save_datagen_result(saved_file_name, data, E_of_t)
    return elapsed
end

function _datagen_println(io_args...)
    lock(_DATAGEN_IO_LOCK) do
        println(io_args...)
    end
    return nothing
end

"""
Functional CUDA devices for datagen. Empty if CUDA is missing or unusable.
Uses every device (no 8-GPU pulse-optimizer cap).
"""
function datagen_cuda_devices()
    try
        CUDA.functional() || return Any[]
        return collect(CUDA.devices())
    catch
        return Any[]
    end
end

function datagen_gpu_count()
    return length(datagen_cuda_devices())
end

function describe_datagen_compute(n_jobs::Integer=0; n_gpu::Union{Nothing,Int}=nothing)
    n_gpu_eff = n_gpu === nothing ? datagen_gpu_count() : Int(n_gpu)
    n_th = max(1, Threads.nthreads())
    if n_gpu_eff < 1
        n_workers = n_jobs > 0 ? min(n_jobs, n_th) : n_th
        return "CPU  julia_threads=$n_th  workers=$(n_workers)"
    end
    n_workers = n_jobs > 0 ? min(n_jobs, n_gpu_eff, n_th) : min(n_gpu_eff, n_th)
    return "$n_gpu_eff CUDA GPU(s)  julia_threads=$n_th  concurrent=$(n_workers)"
end

"""
Idle the current CUDA device, drop unreachable `CuArray`s, and return the
CUDA.jl cached-free pool to the driver. No-op without a functional device.
Never throws. Call only from the worker that owns this device.
"""
function datagen_reclaim_current_gpu!()
    try
        CUDA.functional() || return nothing
        CUDA.synchronize()
        GC.gc(false)
        CUDA.reclaim()
    catch
    end
    return nothing
end

"""
`synchronize` + `reclaim` on every listed device. Restores the caller's
current device. Call only when no datagen ODE is in flight.
"""
function datagen_reclaim_all_gpus!(devices)
    isempty(devices) && return nothing
    try
        CUDA.functional() || return nothing
        prev = CUDA.device()
        try
            GC.gc(false)
            for dev in devices
                CUDA.device!(dev)
                CUDA.synchronize()
                CUDA.reclaim()
            end
        finally
            CUDA.device!(prev)
        end
    catch
    end
    return nothing
end

function plan_catalog_jobs(stem::AbstractString, sys, PULSE_SPEC, run; skip_existing::Bool=true)
    Ttotal = derive_ttotal(sys, PULSE_SPEC)
    splits = splits_for_run(sys, Ttotal, run)
    reports = Dict{String, Any}()
    jobs = Any[]
    n_skipped = 0
    for ic in run.ics
        for split in splits
            outpath, key = result_target(stem, ic, split.M_delta, split.M_g, run.Nt_save)
            if skip_existing && result_is_complete(outpath)
                reports[key] = Dict("status" => "skipped", "path" => outpath)
                n_skipped += 1
                continue
            end
            push!(jobs, (
                stem = String(stem),
                sys = sys,
                PULSE_SPEC = PULSE_SPEC,
                ic = ic,
                split = split,
                run = run,
                Ttotal = Ttotal,
                outpath = outpath,
                key = key,
            ))
        end
    end
    return Ttotal, splits, jobs, reports, n_skipped
end

function _job_outcome_ok(job, worker_tag, elapsed)
    return (
        stem = job.stem,
        key = job.key,
        status = "ok",
        path = job.outpath,
        elapsed_seconds = elapsed,
        M_delta = job.split.M_delta,
        M_g = job.split.M_g,
        safety_factor = job.split.safety_factor,
        worker = String(worker_tag),
    )
end

function _job_outcome_failed(job, worker_tag, msg)
    return (
        stem = job.stem,
        key = job.key,
        status = "failed",
        path = job.outpath,
        error = String(msg),
        worker = String(worker_tag),
    )
end

function _fill_unassigned_outcomes!(outcomes, jobs, msg::AbstractString)
    for i in eachindex(outcomes)
        if !isassigned(outcomes, i)
            outcomes[i] = _job_outcome_failed(jobs[i], "unassigned", msg)
        end
    end
    return outcomes
end

function _notify_datagen_complete(on_complete, job, st)
    on_complete === nothing && return nothing
    try
        on_complete(job, st)
    catch err
        rethrow_interrupt(err)
        @error "datagen on_complete callback failed" exception=err
    end
    return nothing
end

function _execute_datagen_job(job, compute::Symbol, worker_tag::AbstractString)
    try
        elapsed = run_one_ic(
            job.sys, job.PULSE_SPEC, job.ic, job.outpath, job.split, job.run, job.Ttotal;
            compute = compute,
        )
        return _job_outcome_ok(job, worker_tag, elapsed)
    catch err
        rethrow_interrupt(err)
        msg = sprint(showerror, err)
        _datagen_println("[$(worker_tag)] [$(job.stem) $(job.key)] FAILED: ", msg)
        return _job_outcome_failed(job, worker_tag, msg)
    finally
        # Solver already reclaimed primal GPU buffers; this catches host
        # arrays and any leftover cached pool on *this* worker's device.
        datagen_reclaim_current_gpu!()
    end
end

"""
Run independent trajectories concurrently: one in-flight ODE per CUDA
device. Falls back to CPU thread workers when no GPU is available.

Every index of the returned vector is assigned, including jobs that never
started (marked `failed` with an explanatory `error`). `InterruptException`
is rethrown after that fill and after reclaiming every device.

`executor` and `on_complete` are for the self-test and for crash-safe
in-memory report updates; production simulate uses the defaults.
"""
function run_datagen_jobs!(jobs; executor=_execute_datagen_job, on_complete=nothing)
    n = length(jobs)
    n == 0 && return NamedTuple[]
    devices = datagen_cuda_devices()
    n_gpu = length(devices)
    compute = n_gpu > 0 ? :gpu : :cpu
    n_th = max(1, Threads.nthreads())
    n_workers = min(n, n_gpu > 0 ? n_gpu : n_th, n_th)

    if n_gpu > 1 && n_th < n_gpu
        @warn "Julia has $n_th thread(s) but $n_gpu GPUs; only $n_workers GPU(s) will run at once. Start with `julia -t $n_gpu` or `julia -t auto`."
    end

    println("Dispatching $n trajectory job(s) on $(describe_datagen_compute(n; n_gpu=n_gpu))")

    outcomes = Vector{Any}(undef, n)
    stop = Threads.Atomic{Bool}(false)
    jobq = Channel{Int}(n)
    for i in 1:n
        put!(jobq, i)
    end
    close(jobq)

    function worker(wid::Int)
        if n_gpu > 0
            CUDA.device!(devices[wid])
        end
        worker_tag = n_gpu > 0 ? "gpu $(wid - 1)" : "cpu $(wid)"
        try
            for i in jobq
                stop[] && break
                job = jobs[i]
                try
                    _datagen_println("[$(worker_tag)] $(job.stem) $(job.key) start  M=$(job.split.M_total)")
                    st = executor(job, compute, worker_tag)
                    outcomes[i] = st
                    if st.status == "ok"
                        _datagen_println(
                            @sprintf("[%s] %s %s ok  %.1fs", worker_tag, job.stem, job.key, st.elapsed_seconds)
                        )
                    end
                    _notify_datagen_complete(on_complete, job, st)
                catch err
                    if !isassigned(outcomes, i)
                        outcomes[i] = _job_outcome_failed(
                            job, worker_tag, sprint(showerror, unwrap_task_failure(err)),
                        )
                    end
                    _notify_datagen_complete(on_complete, job, outcomes[i])
                    if is_interrupt(err)
                        stop[] = true
                        throw(unwrap_task_failure(err))
                    end
                    @error "datagen worker $worker_tag died on $(job.stem) $(job.key)" exception=err
                end
                datagen_reclaim_current_gpu!()
            end
        finally
            datagen_reclaim_current_gpu!()
        end
        return nothing
    end

    pool_err = nothing
    try
        if n_workers == 1
            worker(1)
        else
            blas_n = BLAS.get_num_threads()
            try
                BLAS.set_num_threads(1)
                @sync for w in 1:n_workers
                    Threads.@spawn worker(w)
                end
            finally
                BLAS.set_num_threads(blas_n)
            end
        end
    catch err
        pool_err = err
        if is_interrupt(err)
            stop[] = true
        else
            @error "datagen worker pool failed" exception=err
        end
    finally
        _fill_unassigned_outcomes!(
            outcomes, jobs, "job was not started (worker pool stopped)",
        )
        datagen_reclaim_all_gpus!(devices)
    end
    pool_err !== nothing && rethrow_interrupt(pool_err)
    return outcomes
end

function merge_job_outcomes!(reports, outcomes)
    n_ok = 0
    n_failed = 0
    for st in outcomes
        if st.status == "ok"
            reports[st.key] = Dict(
                "status" => "ok",
                "path" => st.path,
                "elapsed_seconds" => st.elapsed_seconds,
                "M_delta" => st.M_delta,
                "M_g" => st.M_g,
                "safety_factor" => st.safety_factor,
                "worker" => st.worker,
            )
            n_ok += 1
        else
            reports[st.key] = Dict(
                "status" => "failed",
                "path" => st.path,
                "error" => st.error,
                "worker" => st.worker,
            )
            n_failed += 1
        end
    end
    return n_ok, n_failed
end

function count_job_report_statuses(reports)
    n_ok = 0
    n_failed = 0
    for (_, rep) in reports
        st = string(json_get(rep, "status"))
        if st == "ok"
            n_ok += 1
        elseif st == "failed"
            n_failed += 1
        end
    end
    return n_ok, n_failed
end

function simulate_catalog_entry(stem::AbstractString, sys, PULSE_SPEC, run; skip_existing::Bool=true)
    Ttotal, splits, jobs, reports, n_skipped = plan_catalog_jobs(
        stem, sys, PULSE_SPEC, run; skip_existing = skip_existing,
    )
    println("  Ttotal=$(Ttotal * 1e6) us  n_splits=$(length(splits))  pending=$(length(jobs))")
    for split in splits
        println(
            @sprintf(
                "  split  M_delta=%d  M_g=%d  M=%d  safety=%.3g  target=%.3g",
                split.M_delta, split.M_g, split.M_total, split.safety_factor, split.target_safety,
            )
        )
    end
    outcomes = run_datagen_jobs!(jobs)
    n_ok, n_failed = merge_job_outcomes!(reports, outcomes)
    return n_ok, n_skipped, n_failed, reports
end

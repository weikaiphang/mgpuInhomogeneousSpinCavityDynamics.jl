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
# Result files are named {stem}_{ic}_Md{M_delta}_Mg{M_g}_Nt{Nt_save}.jld2;
# skip if that path and its pulsemat csv both exist and are non-empty.
# ============================================================

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
        g_keep = collect(d.g_b_1d),
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
        mv(csv_tmp, csv_path; force=true)
        result_is_complete(filename) || error(
            "save did not produce a complete result pair at $filename."
        )
    catch
        isfile(jld_tmp) && rm(jld_tmp; force=true)
        isfile(csv_tmp) && rm(csv_tmp; force=true)
        rethrow()
    end
    return filename
end

function parse_default_conditions(s::AbstractString)
    t = lowercase(strip(String(s)))
    if t == "ground"
        return (:ground,)
    elseif t == "equatorial" || t == "equator"
        return (:equator,)
    elseif t == "both"
        return (:ground, :equator)
    else
        error("--default-conditions must be ground, equatorial, or both, got $(s).")
    end
end

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
        ic in (:ground, :equator) || error("unknown initial condition $ic.")
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

function run_one_ic(sys, PULSE_SPEC, ic::Symbol, saved_file_name::AbstractString, split, run, Ttotal::Float64)
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
        compute = :auto,
    )
    elapsed = (time_ns() - t0) / 1e9

    reduced = reduce_trajectory(t, a, Sp, Sz, d, E_of_t)
    data = merge(
        reduced,
        (
            SIM_SETTING = SIM_SETTING,
            SYSTEM_CONFIG = sys,
            PULSE_CONFIG = PULSE_SPEC.segments,
            PULSE_SPEC = PULSE_SPEC,
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

function simulate_catalog_entry(stem::AbstractString, sys, PULSE_SPEC, run; skip_existing::Bool = true)
    n_ok = 0
    n_skipped = 0
    n_failed = 0
    reports = Dict{String, Any}()

    Ttotal = derive_ttotal(sys, PULSE_SPEC)
    splits = splits_for_run(sys, Ttotal, run)
    println("  Ttotal=$(Ttotal * 1e6) us  n_splits=$(length(splits))")
    for split in splits
        println(
            @sprintf(
                "  split  M_delta=%d  M_g=%d  M=%d  safety=%.3g  target=%.3g",
                split.M_delta, split.M_g, split.M_total, split.safety_factor, split.target_safety,
            )
        )
    end

    for ic in run.ics
        for split in splits
            outpath, key = result_target(stem, ic, split.M_delta, split.M_g, run.Nt_save)
            if skip_existing && result_is_complete(outpath)
                reports[key] = Dict("status" => "skipped", "path" => outpath)
                n_skipped += 1
                continue
            end
            try
                elapsed = run_one_ic(sys, PULSE_SPEC, ic, outpath, split, run, Ttotal)
                reports[key] = Dict(
                    "status" => "ok",
                    "path" => outpath,
                    "elapsed_seconds" => elapsed,
                    "M_delta" => split.M_delta,
                    "M_g" => split.M_g,
                    "safety_factor" => split.safety_factor,
                )
                n_ok += 1
            catch err
                rethrow_interrupt(err)
                msg = sprint(showerror, err)
                reports[key] = Dict(
                    "status" => "failed",
                    "path" => outpath,
                    "error" => msg,
                )
                n_failed += 1
                println("[$stem $key] FAILED: ", msg)
            end
            GC.gc()
        end
    end

    return n_ok, n_skipped, n_failed, reports
end

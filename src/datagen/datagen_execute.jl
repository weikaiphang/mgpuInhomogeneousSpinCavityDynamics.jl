# ============================================================
# Run-rules, trajectory execution, reduced save.
# ============================================================

const RULE_SIMULATION_ORDER = :order1
const RULE_NT_SAVE = 5001
const RULE_RELTOL = 1e-8
const RULE_ABSTOL = 1e-8
# Paper 1st-order maxima (Hanamura & Touzard):
#   M_delta = 3000  — fig. 3b ACE / ROSE
#   M_g     = 20    — fig. 3c silencing (Gaussian g)
# Constant g still forces M_g = 1 (package binning).
const RULE_M_DELTA = 3000
const RULE_M_G_INHOM = 20
const RULE_M_PRODUCT_MAX = 3000 * 20

function run_rule_M_g(sys)
    return sys.g_inhomogeneity.kind === :constant ? 1 : RULE_M_G_INHOM
end

function build_sim_setting(sys, Ttotal::Float64, ic::Symbol, saved_file_name::AbstractString)
    M_g = run_rule_M_g(sys)
    M_delta = RULE_M_DELTA
    M_delta * M_g <= RULE_M_PRODUCT_MAX || error(
        "M_delta*M_g = $(M_delta * M_g) exceeds $(RULE_M_PRODUCT_MAX)."
    )
    return (
        simulation_order = RULE_SIMULATION_ORDER,
        M_delta = M_delta,
        M_g = M_g,
        initial_condition = ic,
        Ttotal = Ttotal,
        Nt_save = RULE_NT_SAVE,
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
    JLD2.@save filename data

    pulsemat = filename[1:end-length(".jld2")] * "_pulsemat.csv"
    ISC.sample_E_of_t(
        E_of_t,
        data.SIM_SETTING.Ttotal,
        data.SIM_SETTING.Nt_save;
        savepath = pulsemat,
    )
    return filename
end

function run_one_ic(sys, PULSE_SPEC, ic::Symbol, saved_file_name::AbstractString)
    Ttotal = derive_ttotal(sys, PULSE_SPEC)
    SIM_SETTING = build_sim_setting(sys, Ttotal, ic, saved_file_name)
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
        ),
    )
    save_datagen_result(saved_file_name, data, E_of_t)
    return elapsed
end

function simulate_catalog_entry(run_id::Integer, sys, PULSE_SPEC; skip_existing::Bool = true)
    n_ok = 0
    n_skipped = 0
    n_failed = 0
    reports = Dict{String, Any}()

    for ic in (:ground, :equator)
        outpath = result_path(run_id, ic)
        if skip_existing && isfile(outpath)
            reports[String(ic)] = Dict("status" => "skipped", "path" => outpath)
            n_skipped += 1
            continue
        end
        try
            elapsed = run_one_ic(sys, PULSE_SPEC, ic, outpath)
            reports[String(ic)] = Dict(
                "status" => "ok",
                "path" => outpath,
                "elapsed_seconds" => elapsed,
            )
            n_ok += 1
        catch err
            reports[String(ic)] = Dict(
                "status" => "failed",
                "path" => outpath,
                "error" => sprint(showerror, err),
            )
            n_failed += 1
            println("[run_$(lpad(run_id, 6, '0')) $ic] FAILED: ", sprint(showerror, err))
        end
        GC.gc()
    end

    return n_ok, n_skipped, n_failed, reports
end

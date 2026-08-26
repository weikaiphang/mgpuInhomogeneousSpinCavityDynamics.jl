# ============================================================
# Run-rules, trajectory execution, reduced save.
# ============================================================

const RULE_SIMULATION_ORDER = :order1
const RULE_NT_SAVE = 5001
const RULE_RELTOL = 1e-8
const RULE_ABSTOL = 1e-8
const RULE_SPLIT_SAFETY_FACTOR = 3.0
const RULE_QUADRATURE_BINS = 30

# ============================================================
# Optimal Ensemble Splitting Framework
# Computes M_delta and M_g dynamically based on the Fourier
# limits of the chosen continuous distributions.
# ============================================================

"""
    compute_optimal_splitting(freq_inhomogeneity, g_inhomogeneity, T_max;
                              safety_factor=3.0, quadrature_bins=30) -> NamedTuple

Calculates the optimal allocation for M_delta and M_g to prevent
artificial Dirichlet revivals while minimizing computational waste.

# Arguments
- `freq_inhomogeneity`: The detuning NamedTuple from SYSTEM_CONFIG.
- `g_inhomogeneity`: The coupling NamedTuple from SYSTEM_CONFIG.
- `T_max`: The maximum physical duration of the pulse sequence (in seconds).
- `safety_factor`: Multiplier applied to M_min. Use 3.0 for global search,
                   10.0 for geometric stepping, and higher for asymptotic convergence.
- `quadrature_bins`: The fixed number of bins used for M_g if the coupling
                     distribution is continuous.
"""
function compute_optimal_splitting(
    freq_inhomogeneity,
    g_inhomogeneity,
    T_max::Real;
    safety_factor::Real=3.0,
    quadrature_bins::Integer=30
)
    # --------------------------------------------------------
    # 1. Determine Frequency Bandwidth (BW)
    # --------------------------------------------------------
    kind_f = freq_inhomogeneity.kind
    FWHM = freq_inhomogeneity.FWHM

    if kind_f == :gaussian
        # Matches gaussian_sigma_from_FWHM in frequency_inhomogeneity.jl
        sigma = FWHM / (2 * sqrt(2 * log(2)))
        span_sigma = freq_inhomogeneity.span_sigma
        BW = 2 * span_sigma * sigma

    elseif kind_f == :lorentzian
        # Matches lorentzian_gamma_from_FWHM in frequency_inhomogeneity.jl
        gammaL = FWHM / 2
        span_gamma = freq_inhomogeneity.span_gamma
        BW = 2 * span_gamma * gammaL

    else
        error("Unsupported freq_inhomogeneity kind for splitting: $(kind_f).")
    end

    # --------------------------------------------------------
    # 2. Determine M_delta (Phase Sensitivity / Fourier Bound)
    # --------------------------------------------------------
    M_delta_min = (T_max * BW) / (2 * pi)

    if safety_factor < 1.0
        @warn "safety_factor < 1.0 violates the Fourier limit. Optimization will likely overfit to artificial revivals."
    end

    M_delta = ceil(Int, M_delta_min * safety_factor)

    # --------------------------------------------------------
    # 3. Determine M_g (Amplitude Sensitivity / Quadrature)
    # --------------------------------------------------------
    kind_g = g_inhomogeneity.kind

    if kind_g == :constant
        # A constant coupling uses exactly one effective bin
        M_g = 1
    elseif kind_g in (:gaussian, :powerlaw_g, :user_defined)
        # Continuous distributions converge rapidly via standard quadrature
        M_g = quadrature_bins
    else
        error("Unsupported g_inhomogeneity kind for splitting: $(kind_g).")
    end

    M_total = M_delta * M_g

    return (
        M_delta = M_delta,
        M_g = M_g,
        M_total = M_total,
        M_delta_min = M_delta_min,
        BW = BW
    )
end

function splitting_for_run(sys, Ttotal::Float64)
    split = compute_optimal_splitting(
        sys.freq_inhomogeneity,
        sys.g_inhomogeneity,
        Ttotal;
        safety_factor = RULE_SPLIT_SAFETY_FACTOR,
        quadrature_bins = RULE_QUADRATURE_BINS,
    )
    split.M_delta >= 1 || error("computed M_delta = $(split.M_delta) is not positive.")
    split.M_g >= 1 || error("computed M_g = $(split.M_g) is not positive.")
    return split
end

function build_sim_setting(sys, Ttotal::Float64, ic::Symbol, saved_file_name::AbstractString)
    split = splitting_for_run(sys, Ttotal)
    return (
        simulation_order = RULE_SIMULATION_ORDER,
        M_delta = split.M_delta,
        M_g = split.M_g,
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

function simulate_catalog_entry(stem::AbstractString, sys, PULSE_SPEC; skip_existing::Bool = true)
    n_ok = 0
    n_skipped = 0
    n_failed = 0
    reports = Dict{String, Any}()

    for ic in (:ground, :equator)
        outpath = result_path(stem, ic)
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
            println("[$stem $ic] FAILED: ", sprint(showerror, err))
        end
        GC.gc()
    end

    return n_ok, n_skipped, n_failed, reports
end

# ============================================================
# JLD2-DRIVEN SIGNAL/CONTROL PULSE OPTIMISATION
#
# Loads a saved .jld2 run (SIM_SETTING/SYSTEM_CONFIG/PULSE_CONFIG plus its
# recorded trajectory -- see save_run_data/run_sim_1st_order), splits its
# PULSE_CONFIG into a FIXED signal pulse and the file's own original
# control pulse(s), and reconciles a signal+original-control forward
# solve (via this package's own CPU rhs_1st_order! machinery -- see
# pulse_optimizer.jl) against the file's saved trajectory BEFORE allowing
# any optimisation to proceed. Once reconciled, optimise_control_pulse_
# from_jld2 hands off to optimise_composite_pulse with the control pulse
# replaced by a CompositePulse (B-spline) and the signal pulse layered in
# as a fixed background drive that is never part of the optimised
# parameter vector -- structurally impossible for the optimiser to touch
# (see run_sim_1st_order_pure's own docstring).
# ============================================================

"""
    load_jld2_run(path) -> data

Loads a saved run's `data` NamedTuple (`SIM_SETTING`, `SYSTEM_CONFIG`,
`PULSE_CONFIG`, the saved trajectory, ...) from a `.jld2` file written by
[`save_run_data`](@ref)/`run_sim_1st_order`'s own convention (`@save
filename data`).
"""
function load_jld2_run(path::AbstractString)
    return JLD2.load(path, "data")
end

"""
    split_signal_control(PULSE_CONFIG; n_signal=1) -> (signal_cfg, control_cfg)

Splits a `PULSE_CONFIG` tuple (as saved in a `.jld2` run, or built by
hand) into its leading `n_signal` pulse(s) (the SIGNAL) and the remaining
ones (the CONTROL) -- matching this package's own convention (see
`examples/sweep_1st_order.jl`/`run_demo.jl`) of one fixed "input signal"
pulse (typically Gaussian) followed by one or more WURST "echo"/control
pulses. Both returned pieces are themselves `PULSE_CONFIG`-shaped tuples,
directly usable with the existing [`build_E_of_t`](@ref).
"""
function split_signal_control(PULSE_CONFIG; n_signal::Integer=1)
    n_signal >= 1 || error("n_signal must be >= 1, got $n_signal.")
    length(PULSE_CONFIG) > n_signal || error(
        "PULSE_CONFIG has only $(length(PULSE_CONFIG)) pulse(s); need more than " *
        "n_signal=$n_signal to have anything left over as the control pulse."
    )
    return PULSE_CONFIG[1:n_signal], PULSE_CONFIG[n_signal+1:end]
end

"""
    build_signal_E_of_t(signal_cfg, use_signal::Bool) -> (t -> Complex)

Builds the FIXED signal drive from `signal_cfg` (a `PULSE_CONFIG`-shaped
tuple, via the existing [`build_E_of_t`](@ref)) when `use_signal` is
`true`; otherwise returns [`_zero_drive`](@ref), identically zero at
every `t`, so the signal pulse has NO effect on anything downstream (the
ensemble never sees it at all) while keeping the same `t -> Complex`
calling convention every other drive in this package uses -- this is the
`USE_SIGNAL` mode flag: set `use_signal=false` to zero the signal pulse
out completely for whatever uses the returned closure next.
"""
function build_signal_E_of_t(signal_cfg, use_signal::Bool)
    return use_signal ? build_E_of_t(signal_cfg) : _zero_drive
end

"""
    run_sim_1st_order_trajectory(E_of_t, d; initial_condition=:ground, alg=Tsit5(), reltol=1e-8, abstol=1e-8, tstops=Float64[]) -> (t, a, Sp, Sz)

Forward-only (not `ForwardDiff`-differentiated) CPU analogue of
`run_sim_1st_order` returning the FULL trajectory at `d.t_save`, unlike
[`run_sim_1st_order_pure`](@ref) which only returns the final state --
reuses the SAME `rhs_1st_order!`/ensemble machinery, just saving every
requested time point instead of one. `Sp`/`Sz` are returned as `(Nt, M)`
matrices (one row per saved time point, matching this package's own
`Sp_sol`/`Sz_sol` orientation from `run_sim_1st_order`). Used by
[`reconcile_against_jld2`](@ref) to compare against a saved run's own
recorded trajectory.
"""
function run_sim_1st_order_trajectory(
    E_of_t, d;
    initial_condition::Symbol=:ground, alg=Tsit5(),
    reltol=1e-8, abstol=1e-8, tstops=Float64[],
)
    M = d.M
    u0 = build_u0_1st_order_cpu(M, d.Nj, Float64, initial_condition)
    p = (d.delta0, d.kappa_e, d.kappa_i, d.delta_b, d.g_b, M, E_of_t)
    prob = ODEProblem(rhs_1st_order!, u0, d.timespan, p)
    sol = solve(prob, alg; reltol=reltol, abstol=abstol, saveat=d.t_save, tstops=tstops)

    Nt = length(sol.t)
    a = Vector{ComplexF64}(undef, Nt)
    Sp = Matrix{ComplexF64}(undef, Nt, M)
    Sz = Matrix{ComplexF64}(undef, Nt, M)
    @inbounds for i in 1:Nt
        ai, Spi, Szi = unpack_state_1st_order_u(sol.u[i], M)
        a[i] = ai
        Sp[i, :] .= Spi
        Sz[i, :] .= Szi
    end
    return sol.t, a, Sp, Sz
end

"""
    reconcile_against_jld2(path; n_signal=1, rtol=1e-3, atol=0.0, reltol=nothing, abstol=nothing, verbose=true) -> (ok, report, data, d)

Loads `path`, reconstructs the run's own ORIGINAL signal+control drive
EXACTLY as recorded in `data.PULSE_CONFIG` (signal always included here,
regardless of what `use_signal` a subsequent optimisation run will use --
this function's whole job is to validate against what the file actually
recorded, which was produced WITH the signal pulse present), re-solves
with [`run_sim_1st_order_trajectory`](@ref) (this package's own CPU
1st-order physics -- the SAME equations `run_sim_1st_order` used to
produce `path` in the first place, just without the GPU/callback/file-I/O
machinery), and checks the result against `data`'s own saved
`a_sol`/`Σp_sol`/`Σz_sol` to relative tolerance `rtol` (`atol` is an
absolute floor added to each comparison's scale, guarding against
division by a near-zero saved value). `reltol`/`abstol` default to
`data.SIM_SETTING`'s own values (the tolerances the file was originally
produced with), so the comparison is as apples-to-apples as reasonably
possible; `tstops` is deliberately left empty here too, matching
`run_sim_1st_order`'s own solve call (no `tstops`), so the adaptive
stepper behaves the same way it would have during the original
production run.

Returns `(ok::Bool, report::NamedTuple, data, d)` -- `report` holds
`rel_a`/`rel_p`/`rel_z` (and their absolute counterparts) for inspection
regardless of `ok`. This is a REQUIRED gate before
[`optimise_control_pulse_from_jld2`](@ref) proceeds to optimisation --
any parsing/ensemble/physics mismatch between this port and the file's
own provenance is caught here before it can silently corrupt an
"optimised" pulse built on the wrong physics.
"""
function reconcile_against_jld2(
    path::AbstractString;
    n_signal::Integer=1,
    rtol::Real=1e-3,
    atol::Real=0.0,
    reltol=nothing,
    abstol=nothing,
    verbose::Bool=true,
)
    data = load_jld2_run(path)
    CONFIG = build_full_config(data.SIM_SETTING, data.SYSTEM_CONFIG)
    d = prepare_derived(CONFIG)

    signal_cfg, control_cfg = split_signal_control(data.PULSE_CONFIG; n_signal=n_signal)
    signal_E = build_E_of_t(signal_cfg)
    control_E = build_E_of_t(control_cfg)
    E_of_t(t) = signal_E(t) + control_E(t)

    reltol_solve = reltol === nothing ? data.SIM_SETTING.reltol : reltol
    abstol_solve = abstol === nothing ? data.SIM_SETTING.abstol : abstol

    _, a_check, Sp_check, Sz_check = run_sim_1st_order_trajectory(
        E_of_t, d; reltol=reltol_solve, abstol=abstol_solve,
    )
    Sigma_p_check = vec(sum(Sp_check, dims=2))
    Sigma_z_check = vec(sum(Sz_check, dims=2))

    err_a = maximum(abs.(a_check .- data.a_sol))
    err_p = maximum(abs.(Sigma_p_check .- data.Σp_sol))
    err_z = maximum(abs.(Sigma_z_check .- data.Σz_sol))

    scale_a = maximum(abs.(data.a_sol)) + atol
    scale_p = maximum(abs.(data.Σp_sol)) + atol
    scale_z = maximum(abs.(data.Σz_sol)) + atol

    rel_a, rel_p, rel_z = err_a / scale_a, err_p / scale_p, err_z / scale_z
    ok = rel_a < rtol && rel_p < rtol && rel_z < rtol

    if verbose
        status = ok ? "PASS" : "FAIL"
        println("Reconciliation against $path: $status (rtol=$rtol)")
        println("  a:  max_abs_err=$err_a  rel_err=$rel_a")
        println("  Σp: max_abs_err=$err_p  rel_err=$rel_p")
        println("  Σz: max_abs_err=$err_z  rel_err=$rel_z")
    end

    report = (rel_a=rel_a, rel_p=rel_p, rel_z=rel_z, err_a=err_a, err_p=err_p, err_z=err_z)
    return ok, report, data, d
end

"""
    optrunlog_paths(path; out_dir=nothing) -> (optrunlog_path, pulsemat_path)

Derives this run's `<basename>_optrunlog.jld2`/`<basename>_opt_pulsemat.csv`
output paths from the SOURCE `.jld2` path's own basename -- the same
"same directory, same basename + suffix" convention `save_run_data`
already uses for its own `_pulsemat.csv` sibling (see `pulses.jl`). Pass
`out_dir` to write both files elsewhere instead of alongside `path`;
either way the containing directory is created if it doesn't exist yet.
"""
function optrunlog_paths(path::AbstractString; out_dir=nothing)
    endswith(path, ".jld2") || error("Expected a .jld2 path, got $path.")
    base = path[1:end-length(".jld2")]
    dir = out_dir === nothing ? dirname(base) : out_dir
    isempty(dir) || mkpath(dir)
    base_name = basename(base)
    return joinpath(dir, base_name * "_optrunlog.jld2"), joinpath(dir, base_name * "_opt_pulsemat.csv")
end

"""
    save_optimisation_run_log(path, data, d, pulse, signal_E_of_t,
                               n_signal, use_signal, u0, initial_metrics,
                               best_u, final_metrics, history, optimizer_settings;
                               out_dir=nothing, pulsemat_N=nothing)
        -> (optrunlog_path, pulsemat_path)

Writes this run's full record to `<basename>_optrunlog.jld2` (see
[`optrunlog_paths`](@ref)) and samples the FINAL/optimal CONTROL pulse's
own drive (`build_E_of_t(pulse, best_u)` -- the control pulse alone, NOT
combined with the signal, since the signal is a separate FIXED input that
was never part of what got optimised) to `<basename>_opt_pulsemat.csv`
via the existing [`sample_E_of_t`](@ref)/[`save_E_samples`](@ref) (same
format/read pattern as every other `_pulsemat.csv` in this package).
`pulsemat_N` defaults to the source run's own `SIM_SETTING.Nt_save`, for
comparability with the original file's own sampling density.

The `.jld2` log holds, all under the top-level key `"data"` (same
convention [`load_jld2_run`](@ref) already reads):
  - `source_path`, `n_signal`, `use_signal`
  - `SIM_SETTING`, `SYSTEM_CONFIG` (copied from the source run, for
    provenance/reproducibility -- enough, together with `k`/`n_coeff_A`/
    `n_coeff_f` below, to rebuild `d`/`pulse` deterministically later)
  - `k`, `n_coeff_A`, `n_coeff_f`
  - `optimizer_settings` -- every setting that actually affects
    replication of this run: `USE_SIGNAL`/`n_signal` plus every knob
    [`optimise_composite_pulse`](@ref) exposes (`num_epochs`,
    `learning_rate`, `patience`, `tol`, `n_hops`, `hop_patience`,
    `hop_step_size`, `temperature`, `w_tmax`, `seed`, `degree`,
    `taper_frac`, and any numeric
    `solve_kwargs` override such as `reltol`/`abstol`/`w_inv`/`w_coh`/
    `w_time`) -- see [`optimise_composite_pulse`](@ref)'s own docstring
    for exactly what it captures and why (and what it deliberately
    excludes, e.g. non-serialisable closures)
  - `initial_u` (the candidate pulse's own raw parameterisation, `u0`,
    used only after reconciliation passed)
  - `initial_metrics` (`(cost, inversion, coherence, duration)` at `u0`,
    from [`pulse_cost`](@ref) -- depends only on inversion, coherence,
    and duration, no area/pi-pulse-area term)
  - `initial_output` (`(a, Sigma_p, Sigma_z)` at `t1`, from actually
    simulating `u0` -- the candidate pulse's raw simulated output)
  - `history` -- one row per optimiser epoch, across every hop (see
    [`run_local_adam`](@ref)): `hop, epoch, k, cost, inversion,
    coherence, duration, improved`
  - `final_u` (the optimised control pulse's raw parameterisation,
    `best_u`)
  - `final_metrics`/`final_output` -- same shape as the initial ones,
    for `best_u`
"""
function save_optimisation_run_log(
    path::AbstractString, data, d, pulse::CompositePulse, signal_E_of_t,
    n_signal::Integer, use_signal::Bool,
    u0::AbstractVector, initial_metrics,
    best_u::AbstractVector, final_metrics, history, optimizer_settings;
    out_dir=nothing, pulsemat_N=nothing,
)
    optrunlog_path, pulsemat_path = optrunlog_paths(path; out_dir=out_dir)

    a0, Sp0, Sz0, _ = run_sim_1st_order_pure(u0, pulse, d; signal_E_of_t=signal_E_of_t)
    initial_output = (a=a0, Sigma_p=sum(Sp0), Sigma_z=sum(Sz0))

    a1, Sp1, Sz1, _ = run_sim_1st_order_pure(best_u, pulse, d; signal_E_of_t=signal_E_of_t)
    final_output = (a=a1, Sigma_p=sum(Sp1), Sigma_z=sum(Sz1))

    full_settings = merge((n_signal=n_signal, USE_SIGNAL=use_signal), optimizer_settings)

    run_log = (
        source_path=path, n_signal=n_signal, use_signal=use_signal,
        SIM_SETTING=data.SIM_SETTING, SYSTEM_CONFIG=data.SYSTEM_CONFIG,
        k=pulse.k, n_coeff_A=pulse.n_coeff_A, n_coeff_f=pulse.n_coeff_f,
        optimizer_settings=full_settings,
        initial_u=collect(u0), initial_metrics=initial_metrics, initial_output=initial_output,
        history=history,
        final_u=collect(best_u), final_metrics=final_metrics, final_output=final_output,
    )
    JLD2.save(optrunlog_path, "data", run_log)
    println("Saved optimisation run log to $optrunlog_path")

    N = pulsemat_N === nothing ? data.SIM_SETTING.Nt_save : pulsemat_N
    control_E_of_t = build_E_of_t(pulse, best_u)
    sample_E_of_t(control_E_of_t, pulse.T_max, N; savepath=pulsemat_path)

    return optrunlog_path, pulsemat_path
end

"""
    optimise_control_pulse_from_jld2(path, k, n_coeff_A, n_coeff_f;
        n_signal=1, use_signal=true, reconcile=true, rtol_check=1e-3, atol_check=0.0,
        check_reltol=nothing, check_abstol=nothing, save_log=true, log_out_dir=nothing,
        pulsemat_N=nothing, optimizer_kwargs...)
        -> (best_u, best_cost, pulse::CompositePulse, signal_E_of_t, d, data)

End-to-end workflow tying [`load_jld2_run`](@ref),
[`reconcile_against_jld2`](@ref), [`optimise_composite_pulse`](@ref), and
[`save_optimisation_run_log`](@ref) together:

  1. Loads `path` (a `.jld2` run written by `run_sim_1st_order`/
     `save_run_data`), parses `SIM_SETTING`/`SYSTEM_CONFIG`/
     `PULSE_CONFIG`, and rebuilds the ensemble via `prepare_derived`
     (`build_full_config`).
  2. Splits `PULSE_CONFIG` into the signal pulse (its first `n_signal`
     entries) and the file's own ORIGINAL control pulse(s) (the rest) --
     see [`split_signal_control`](@ref).
  3. Unless `reconcile=false`, runs [`reconcile_against_jld2`](@ref) --
     reconstructs signal+ORIGINAL-control exactly as recorded and
     re-solves with this package's own CPU 1st-order physics, comparing
     against the file's saved trajectory. Throws an error and refuses to
     proceed if this does not match within `rtol_check`/`atol_check`:
     optimisation MUST NOT run against physics this port hasn't first
     verified it can reproduce.
  4. Builds the FIXED signal drive ([`build_signal_E_of_t`](@ref)) --
     `use_signal=true` (default) uses the file's own recorded signal
     pulse exactly; `use_signal=false` (the `USE_SIGNAL` mode flag) zeroes
     it out completely, identically zero at every `t`, so it has no
     effect whatsoever on the ensemble -- then runs
     [`optimise_composite_pulse`](@ref)-style Adam + basin-hopping
     optimisation on a NEW `CompositePulse` (`k` B-spline sub-pulses)
     CONTROL pulse, with the signal passed through as
     `signal_E_of_t`. The signal pulse is NEVER part of the optimised
     parameter vector `u` -- see [`run_sim_1st_order_pure`](@ref)'s own
     docstring for why this is structural (a plain captured closure, not
     a differentiated quantity), not merely a convention: the optimiser
     is physically incapable of touching the signal pulse.
  5. Unless `save_log=false`, writes the full run record via
     [`save_optimisation_run_log`](@ref) (`log_out_dir`/`pulsemat_N`
     forwarded to it as `out_dir`/`pulsemat_N`), including a complete
     `optimizer_settings` record -- everything [`optimise_composite_pulse`](@ref)
     captured (`num_epochs`, `learning_rate`, `patience`, `tol`, `n_hops`,
     `hop_patience`, `hop_step_size`, `temperature`, `w_tmax`, `degree`,
     `taper_frac`, `seed`, any numeric `solve_kwargs` override) merged with
     this function's own
     (`n_signal`, `USE_SIGNAL`, `reconcile`, `rtol_check`, `atol_check`,
     `check_reltol`, `check_abstol`) -- so a
     saved run can be replicated exactly later just by reading its own
     log.

Returns everything a caller would need afterwards: the optimised raw
parameters, its cost, the `CompositePulse` they decode against, the fixed
`signal_E_of_t` actually used, the derived ensemble `d`, and the loaded
`data` (e.g. for further comparison/plotting against the original run).
"""
function optimise_control_pulse_from_jld2(
    path::AbstractString,
    k::Integer, n_coeff_A::Integer, n_coeff_f::Integer;
    n_signal::Integer=1,
    use_signal::Bool=true,
    reconcile::Bool=true,
    rtol_check::Real=1e-3,
    atol_check::Real=0.0,
    check_reltol=nothing,
    check_abstol=nothing,
    save_log::Bool=true,
    log_out_dir=nothing,
    pulsemat_N=nothing,
    optimizer_kwargs...,
)
    data = load_jld2_run(path)
    CONFIG = build_full_config(data.SIM_SETTING, data.SYSTEM_CONFIG)
    d = prepare_derived(CONFIG)
    signal_cfg, _ = split_signal_control(data.PULSE_CONFIG; n_signal=n_signal)

    if reconcile
        ok, report, _, _ = reconcile_against_jld2(
            path; n_signal=n_signal, rtol=rtol_check, atol=atol_check,
            reltol=check_reltol, abstol=check_abstol,
        )
        ok || error(
            "Reconciliation against $path FAILED (rel_a=$(report.rel_a), " *
            "rel_p=$(report.rel_p), rel_z=$(report.rel_z), tolerance rtol=$rtol_check) -- " *
            "refusing to optimise against physics this port hasn't verified it can " *
            "reproduce. Inspect the printed errors above, or pass reconcile=false to " *
            "override at your own risk."
        )
    end

    signal_E_of_t = build_signal_E_of_t(signal_cfg, use_signal)

    best_u, best_cost, pulse, u0, initial_metrics, history, final_metrics, optimizer_settings = optimise_composite_pulse(
        k, n_coeff_A, n_coeff_f, d; signal_E_of_t=signal_E_of_t, optimizer_kwargs...,
    )

    if save_log
        full_settings = merge(
            (reconcile=reconcile, rtol_check=rtol_check, atol_check=atol_check,
             check_reltol=check_reltol, check_abstol=check_abstol),
            optimizer_settings,
        )
        save_optimisation_run_log(
            path, data, d, pulse, signal_E_of_t, n_signal, use_signal,
            u0, initial_metrics, best_u, final_metrics, history, full_settings;
            out_dir=log_out_dir, pulsemat_N=pulsemat_N,
        )
    end

    return best_u, best_cost, pulse, signal_E_of_t, d, data
end

"""
    optimise_control_pulse_from_jld2_over_k(path, n_coeff_A, n_coeff_f;
        kinds=(:hs1, :corpse, :bb1), specs=nothing, threaded=true,
        n_signal=1, use_signal=true, reconcile=true, ..., optimizer_kwargs...)
        -> (result, signal_E_of_t, d, data)

Load/reconcile a `.jld2` run once, then
[`optimise_composite_pulse_over_k`](@ref) with that file's fixed signal
drive. `k` is not passed: each canonical kind has its own `k`.
`save_log` writes the winning `k` only (same paths as
[`save_optimisation_run_log`](@ref)). `result` is the NamedTuple from
[`optimise_composite_pulse_over_k`](@ref).
"""
function optimise_control_pulse_from_jld2_over_k(
    path::AbstractString,
    n_coeff_A::Integer, n_coeff_f::Integer;
    kinds=(:hs1, :corpse, :bb1),
    specs=nothing,
    threaded::Bool=true,
    n_signal::Integer=1,
    use_signal::Bool=true,
    reconcile::Bool=true,
    rtol_check::Real=1e-3,
    atol_check::Real=0.0,
    check_reltol=nothing,
    check_abstol=nothing,
    save_log::Bool=true,
    log_out_dir=nothing,
    pulsemat_N=nothing,
    Omega_max=nothing,
    beta=nothing,
    mu=nothing,
    seed::Integer=42,
    optimizer_kwargs...,
)
    data = load_jld2_run(path)
    CONFIG = build_full_config(data.SIM_SETTING, data.SYSTEM_CONFIG)
    d = prepare_derived(CONFIG)
    signal_cfg, _ = split_signal_control(data.PULSE_CONFIG; n_signal=n_signal)

    if reconcile
        ok, report, _, _ = reconcile_against_jld2(
            path; n_signal=n_signal, rtol=rtol_check, atol=atol_check,
            reltol=check_reltol, abstol=check_abstol,
        )
        ok || error(
            "Reconciliation against $path FAILED (rel_a=$(report.rel_a), " *
            "rel_p=$(report.rel_p), rel_z=$(report.rel_z), tolerance rtol=$rtol_check) -- " *
            "refusing to optimise against physics this port hasn't verified it can " *
            "reproduce. Inspect the printed errors above, or pass reconcile=false to " *
            "override at your own risk."
        )
    end

    signal_E_of_t = build_signal_E_of_t(signal_cfg, use_signal)

    result = optimise_composite_pulse_over_k(
        n_coeff_A, n_coeff_f, d;
        kinds=kinds, specs=specs, threaded=threaded,
        Omega_max=Omega_max, beta=beta, mu=mu, seed=seed,
        signal_E_of_t=signal_E_of_t, optimizer_kwargs...,
    )

    if save_log
        full_settings = merge(
            (reconcile=reconcile, rtol_check=rtol_check, atol_check=atol_check,
             check_reltol=check_reltol, check_abstol=check_abstol,
             n_signal=n_signal, USE_SIGNAL=use_signal, kinds=collect(kinds)),
            result.optimizer_settings,
        )
        save_optimisation_run_log(
            path, data, d, result.pulse, signal_E_of_t, n_signal, use_signal,
            result.u0, result.initial_metrics, result.best_u, result.final_metrics,
            result.history, full_settings;
            out_dir=log_out_dir, pulsemat_N=pulsemat_N,
        )
    end

    return result, signal_E_of_t, d, data
end

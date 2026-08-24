# ============================================================
# JLD2-DRIVEN SIGNAL/CONTROL PULSE OPTIMISATION
#
# Loads a saved .jld2 run (SIM_SETTING/SYSTEM_CONFIG/PULSE_CONFIG plus its
# recorded trajectory -- see save_run_data/run_sim_1st_order), splits its
# PULSE_CONFIG into a FIXED signal pulse and the file's own original
# control pulse(s), and reconciles a signal+original-control forward
# solve (via this package's own CPU rhs_1st_order! machinery -- see
# pulse_optimizer2.jl) against the file's saved trajectory BEFORE allowing
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
    _segment_matched_seed_init(control_cfg, pulse::CompositePulse) -> Union{Vector{Float64},Nothing}

Builds a raw parameter vector for `pulse` by matching each of its `k`
sub-pulses 1:1 against `control_cfg[i]` -- ONLY when `length(control_cfg)
== pulse.k` (returns `nothing` otherwise, so the caller falls back to
[`fit_composite_pulse`](@ref)'s own generic, timing-blind
[`initial_guess`](@ref)). Sub-pulse `i` is placed at the SAME
`t_start`/`t_end` as `control_cfg[i]` (from its own `t_center`/`duration`
fields, encoded back through the same `gap_scale`/`dur_scale`/`dur_floor`
softplus reparameterisation [`decode`](@ref) uses), its amplitude B-spline
started FLAT at `control_cfg[i]`'s own peak `amp` (falling back to
`pulse.amp_scale` if that field is absent), and -- when `control_cfg[i]`
has a `bandwidth` field (i.e. looks like a `:wurst` spec) -- its frequency
B-spline control points started as a LINEAR ramp from `-bandwidth/2` to
`+bandwidth/2` (times `control_cfg[i].chirp_sign`, defaulting to `+1.0`),
a reasonable starting approximation of a linear chirp sweep (`0` chirp
otherwise).

This exists because the generic random `initial_guess` turned out to be a
poor starting point for fitting a MULTI-segment target: verified on this
package's own 3-ARP reference pulse (k=3, 3 WURST segments) that 200
epochs of Adam from `initial_guess` plateaus at `rel_l2~0.998` (i.e.
essentially no better than an all-zero pulse) -- the random init's own
amplitude scale sits ~3x below the target's actual peak, and its sub-pulse
timings have no relation to where the target's segments actually are, so
early gradients are largely uninformative (a sub-pulse that doesn't
overlap the target in time gets no useful amplitude/shape signal from it).
Starting from each segment's own known timing/amplitude/chirp instead
begins descent already in roughly the right basin; [`fit_composite_pulse`](@ref)'s
own Adam descent still does all the actual curve shaping from there.
"""
function _segment_matched_seed_init(control_cfg, pulse::CompositePulse)
    length(control_cfg) == pulse.k || return nothing
    k = pulse.k

    raw_gap = Vector{Float64}(undef, k)
    raw_dur = Vector{Float64}(undef, k)
    raw_cA = Matrix{Float64}(undef, pulse.n_coeff_A, k)
    raw_cf = Matrix{Float64}(undef, pulse.n_coeff_f, k)

    t_prev_end = 0.0
    for i in 1:k
        cfg = control_cfg[i]
        duration = Float64(cfg.duration)
        t_start = Float64(cfg.t_center) - duration / 2
        gap = max(t_start - t_prev_end, pulse.gap_scale * 1e-3)
        dur_arg = max(duration - pulse.dur_floor, pulse.dur_scale * 1e-3)

        raw_gap[i] = _softplus_inv(gap / pulse.gap_scale)
        raw_dur[i] = _softplus_inv(dur_arg / pulse.dur_scale)

        amp = hasproperty(cfg, :amp) ? Float64(cfg.amp) : pulse.amp_scale
        raw_cA[:, i] .= _softplus_inv(max(amp, 1e-30) / pulse.amp_scale)

        if hasproperty(cfg, :bandwidth)
            bw = Float64(cfg.bandwidth)
            chirp_sign = hasproperty(cfg, :chirp_sign) ? Float64(cfg.chirp_sign) : 1.0
            raw_cf[:, i] .= chirp_sign .* range(-bw / 2, bw / 2; length=pulse.n_coeff_f) ./ pulse.freq_scale
        else
            raw_cf[:, i] .= 0.0
        end

        t_prev_end = t_start + duration
    end

    return pack(pulse, raw_gap, raw_dur, raw_cA, raw_cf)
end

"""
    fit_composite_pulse_seed(control_cfg, pulse::CompositePulse; kwargs...) -> (u_fit, fit_report)

Interprets a recorded `PULSE_CONFIG`'s control segments `control_cfg`
(e.g. [`split_signal_control`](@ref)'s second return value -- this file's
own original, non-B-spline control pulse, however it was built, e.g. the
`:wurst` segments [`generate_3arp_pi_pulse`](@ref) produces) as a target
drive (`build_E_of_t(control_cfg)`, `pulses.jl`'s tuple-of-specs method)
and fits `pulse`'s own B-spline raw parameters to approximate it (see
[`fit_composite_pulse`](@ref) in pulse_optimizer2.jl for the actual
ForwardDiff/Adam shape fit -- `kwargs` are forwarded there unchanged, EXCEPT
`u_init`: unless the caller passes one explicitly, this defaults to
[`_segment_matched_seed_init`](@ref)`(control_cfg, pulse)` rather than
`fit_composite_pulse`'s own generic random `initial_guess`, when
`length(control_cfg) == pulse.k` makes a 1:1 segment match possible -- see
that function's own docstring for why this matters in practice). This is
how [`optimise_control_pulse_from_jld2`](@ref) builds its default
optimisation seed FROM the source file's own recorded pulse, rather than a
shape-blind random/canonical guess -- a fit, not an exact reconstruction:
`control_cfg`'s own pulse kind (e.g. WURST's `tanh` gate + `sin^n`
envelope + quadratic-phase chirp) is a different functional family from
`pulse`'s clamped-B-spline-times-Gevrey-taper envelope, so some residual
mismatch is inherent and expected -- see `fit_report.rel_l2` for a
scale-free measure of how much, and confirm the physics separately (this
function makes NO ODE solve at all -- see
[`optimise_control_pulse_from_jld2`](@ref)'s own seed-reconciliation step,
which does).
"""
function fit_composite_pulse_seed(control_cfg, pulse::CompositePulse; kwargs...)
    E_target = build_E_of_t(control_cfg)
    kwargs_nt = NamedTuple(kwargs)
    if !(:u_init in keys(kwargs_nt))
        smart_init = _segment_matched_seed_init(control_cfg, pulse)
        if smart_init !== nothing
            kwargs_nt = merge(kwargs_nt, (u_init=smart_init,))
        end
    end
    return fit_composite_pulse(pulse, E_target; kwargs_nt...)
end

"""
    fit_composite_pulse_seed_auto(control_cfg, d; N_samples=5001, kwargs...)
        -> (pulse::CompositePulse, u_fit, fit_report, segments)

Thin convenience wrapper around
[`fit_composite_pulse_from_samples`](@ref) for a recorded `PULSE_CONFIG`'s
control segments: samples `control_cfg` (via `pulses.jl`'s
[`build_E_of_t`](@ref)/[`sample_E_of_t`](@ref)) at `N_samples` evenly
spaced points over `[0, d.timespan[2]-d.timespan[1]]`, then discovers `k`
and sizes `n_coeff_A`/`n_coeff_f` from that sampled trace ITSELF rather
than from `control_cfg`'s own labelled entry count -- unlike
[`fit_composite_pulse_seed`](@ref) (which assumes `k == length(control_cfg)`
and takes a caller-built `pulse` with caller-chosen coefficient counts),
this makes no such assumption: it would work identically on a raw sampled
trace with no `PULSE_CONFIG` behind it at all. `kwargs` are forwarded to
[`fit_composite_pulse_from_samples`](@ref) (`points_per_segment`,
`degree`, `taper_frac`, `rel_thresh`, `min_active_samples`,
`min_silence_samples`, `num_epochs`, `learning_rate`, `seed`).
"""
function fit_composite_pulse_seed_auto(control_cfg, d; N_samples::Integer=5001, kwargs...)
    E_target = build_E_of_t(control_cfg)
    T_max = d.timespan[2] - d.timespan[1]
    Ex, Ep = sample_E_of_t(E_target, T_max, N_samples)
    t = collect(range(0.0, T_max; length=N_samples))
    return fit_composite_pulse_from_samples(t, Ex, Ep, d; kwargs...)
end

"""
    fit_composite_pulse_seed_auto_linear(control_cfg, d; N_samples=5001, kwargs...)
        -> (pulse::CompositePulse, u_fit, fit_report, segments)

Same as [`fit_composite_pulse_seed_auto`](@ref), but delegates to
[`fit_composite_pulse_from_samples_linear`](@ref) (closed-form linear
least-squares) instead of the `ForwardDiff`/Adam route -- see that
function's own docstring for why this is the practical choice once
`n_coeff` gets large (a dense, real sampled trace at the
20-25-points-per-segment resolution routinely needs `n_coeff` in the
tens to ~80, at which point `ForwardDiff.gradient`+Adam over `n_params =
2k + 2*k*n_coeff` raw parameters becomes impractically slow with this
package's current, non-preallocating `bspline_basis`).
"""
function fit_composite_pulse_seed_auto_linear(control_cfg, d; N_samples::Integer=5001, kwargs...)
    E_target = build_E_of_t(control_cfg)
    T_max = d.timespan[2] - d.timespan[1]
    Ex, Ep = sample_E_of_t(E_target, T_max, N_samples)
    t = collect(range(0.0, T_max; length=N_samples))
    return fit_composite_pulse_from_samples_linear(t, Ex, Ep, d; kwargs...)
end

"""
    smoke_test_fit_from_pulsemat(pulsemat_path;
        linear=true, out_dir=nothing, points_per_segment=nothing,
        degree=3, taper_frac=0.1, rel_thresh=1e-3, min_active_samples=5,
        min_silence_samples=3, cA_floor_frac=_GRAD_SAFE_FRAC, cf_clip_mult=20.0,
        num_epochs=1000, learning_rate=0.002, seed=42)
        -> (pulse, u_fit, fit_report, segments, diff_report, paths)

End-to-end smoke test of the sample-derived seed-fitting pipeline
([`fit_composite_pulse_from_samples_linear`](@ref)/
[`fit_composite_pulse_from_samples`](@ref)) against a REAL recorded pulse,
given nothing but its `_pulsemat.csv`:

  1. Reads `pulsemat_path` via [`load_E_samples`](@ref) for the raw
     `(t, I, Q)` trace (`t` reconstructed as `range(0.0, t_end;
     length=N)`, the same grid [`sample_E_of_t`](@ref) wrote it on).
  2. Loads the SIBLING run `pulsemat_path` was written alongside --
     same directory, same basename, `.jld2` (the convention
     [`save_run_data`](@ref) writes both files under) -- and rebuilds `d`
     via `build_full_config`/`prepare_derived`, exactly like
     [`optimise_control_pulse_from_jld2`](@ref) does. This is where the
     cavity/ensemble parameters (`kappa_e`, `g_mean`, `FWHM`, ...) that
     `CompositePulse`'s own `amp_scale`/`freq_scale` need come from --
     they are NOT recoverable from `_pulsemat.csv` alone. Errors loudly if
     the sibling `.jld2`'s own `SIM_SETTING.Ttotal` disagrees with the
     CSV's own `t_end_us` metadata (the two files not actually belonging
     together).
  3. Fits a `CompositePulse` seed directly from the trace: `linear=true`
     (default) uses [`fit_composite_pulse_from_samples_linear`](@ref)
     (closed-form, practical at a real trace's sample density);
     `linear=false` uses the `ForwardDiff`/Adam
     [`fit_composite_pulse_from_samples`](@ref) instead -- expect this to
     be drastically slower, see that function's own docstring.
  4. Saves the fit's full parameterisation to `<basename>_fitparas.jld2`
     (same shape/convention as
     [`save_optimised_pulse_parameters`](@ref)'s `_opt_pulsepara.jld2`,
     plus this fit's own `fit_report`/`segments`/source paths).
  5. Resamples the FITTED pulse (`build_E_of_t(pulse, u_fit)`) at the same
     `N` evenly-spaced points the original trace was recorded at, saving
     it to `<basename>_fitpulsemat.csv` via the existing
     [`sample_E_of_t`](@ref)/[`save_E_samples`](@ref) (identical format to
     every other `_pulsemat.csv` in this package, so it round-trips
     through [`load_E_samples`](@ref) the same way).
  6. Saves the point-wise `(I - I_fit, Q - Q_fit)` residual to
     `<basename>_fitsamplediff.csv`, same CSV format again.

`out_dir` redirects the three OUTPUT files elsewhere; the SOURCE `.jld2`
run is always read from next to `pulsemat_path` regardless.

Returns `(pulse, u_fit, fit_report, segments, diff_report, paths)`.
`diff_report = (rel_l2, max_abs_diff)`: `rel_l2 =
sqrt(sum(abs2,diff)/sum(abs2,target))` computed on the FULL COMPLEX
difference (unlike `fit_report`'s own `rel_l2_A`/`rel_l2_f`, which score
amplitude/frequency separately) -- this is the smoke test's actual
pass/fail signal, since it round-trips through the real CSV format a
caller would use rather than just checking the in-memory arrays
`fit_composite_pulse_from_samples*` were unit-tested against. `paths =
(source_pulsemat=..., source_jld2=..., fitparas=..., fitpulsemat=...,
fitsamplediff=...)`.

Do not be surprised to see `diff_report.rel_l2` stay large (tens of
percent) even when `fit_report.phi_rms_rad` reports an excellent
per-sub-pulse phase SHAPE fit -- see
[`fit_composite_pulse_from_samples_linear`](@ref)'s own "KNOWN LIMITATION"
paragraph: `CompositePulse` has no parameter for a sub-pulse's ABSOLUTE
phase reference (sub-pulse 1 always reconstructs starting at phase `0`),
so a real trace whose own recorded phase at that instant is not
numerically `0` reconstructs rotated by that difference -- a real,
currently-unfixed gap in `CompositePulse`/`build_E_of_t`, not a bug in
this smoke test or in the fit.
"""
function smoke_test_fit_from_pulsemat(
    pulsemat_path::AbstractString;
    linear::Bool=true,
    out_dir=nothing,
    points_per_segment::Union{Integer,Nothing}=nothing,
    degree::Integer=3, taper_frac::Real=0.1,
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
    cA_floor_frac::Real=_GRAD_SAFE_FRAC, cf_clip_mult::Real=20.0,
    num_epochs::Integer=1000, learning_rate::Real=0.002, seed::Integer=42,
)
    endswith(pulsemat_path, "_pulsemat.csv") || error(
        "Expected a path ending in \"_pulsemat.csv\", got $pulsemat_path."
    )
    base = pulsemat_path[1:end-length("_pulsemat.csv")]
    jld2_path = base * ".jld2"
    isfile(jld2_path) || error(
        "Expected the sibling run $jld2_path (same directory/basename as $pulsemat_path, " *
        "the convention save_run_data writes both files under) -- not found."
    )

    out_base = out_dir === nothing ? base : joinpath(out_dir, basename(base))
    isempty(dirname(out_base)) || mkpath(dirname(out_base))
    fitparas_path = out_base * "_fitparas.jld2"
    fitpulsemat_path = out_base * "_fitpulsemat.csv"
    fitsamplediff_path = out_base * "_fitsamplediff.csv"

    t_end, Ex, Ep = load_E_samples(pulsemat_path)
    N = length(Ex)
    t = collect(range(0.0, t_end; length=N))

    data = load_jld2_run(jld2_path)
    CONFIG = build_full_config(data.SIM_SETTING, data.SYSTEM_CONFIG)
    d = prepare_derived(CONFIG)
    isapprox(d.timespan[2] - d.timespan[1], t_end; rtol=1e-9) || error(
        "T_max mismatch: $jld2_path's own SIM_SETTING.Ttotal implies T_max=" *
        "$(d.timespan[2]-d.timespan[1])s, but $pulsemat_path's own metadata says " *
        "t_end=$(t_end)s -- these two files don't actually belong together."
    )

    fit_kwargs = (
        degree=degree, taper_frac=taper_frac, rel_thresh=rel_thresh,
        min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
    )
    pulse, u_fit, fit_report, segments = if linear
        pps = points_per_segment === nothing ? 6 : points_per_segment
        fit_composite_pulse_from_samples_linear(
            t, Ex, Ep, d; points_per_segment=pps, cA_floor_frac=cA_floor_frac,
            cf_clip_mult=cf_clip_mult, fit_kwargs...,
        )
    else
        pps = points_per_segment === nothing ? 22 : points_per_segment
        fit_composite_pulse_from_samples(
            t, Ex, Ep, d; points_per_segment=pps, cf_clip_mult=cf_clip_mult,
            num_epochs=num_epochs, learning_rate=learning_rate, seed=seed, fit_kwargs...,
        )
    end

    t_start, t_end_decoded, cA, cf = decode(pulse, u_fit)
    fitparas = (
        source_pulsemat=pulsemat_path, source_jld2=jld2_path,
        k=pulse.k, n_coeff_A=pulse.n_coeff_A, n_coeff_f=pulse.n_coeff_f,
        degree=pulse.degree, taper_frac=pulse.taper_frac,
        T_max=pulse.T_max, gap_scale=pulse.gap_scale, dur_scale=pulse.dur_scale,
        dur_floor=pulse.dur_floor, amp_scale=pulse.amp_scale, freq_scale=pulse.freq_scale,
        u_fit=collect(u_fit),
        t_start=collect(t_start), t_end=collect(t_end_decoded), cA=collect(cA), cf=collect(cf),
        fit_report=fit_report, segments=segments, linear=linear,
    )
    JLD2.save(fitparas_path, "data", fitparas)

    E_fit = build_E_of_t(pulse, u_fit)
    Ex_fit, Ep_fit = sample_E_of_t(E_fit, pulse.T_max, N; savepath=fitpulsemat_path)

    dEx = Ex .- Ex_fit
    dEp = Ep .- Ep_fit
    save_E_samples(hcat(dEx, dEp), t_end, fitsamplediff_path)

    target_energy = sum(abs2, Ex) + sum(abs2, Ep) + 1e-30
    diff_energy = sum(abs2, dEx) + sum(abs2, dEp)
    rel_l2 = sqrt(diff_energy / target_energy)
    max_abs_diff = max(maximum(abs, dEx), maximum(abs, dEp))
    diff_report = (rel_l2=rel_l2, max_abs_diff=max_abs_diff)

    println(
        "Smoke test: fit k=$(pulse.k) sub-pulses from $(length(segments)) detected segments, " *
        "N=$N samples. rel_l2 (complex, full trace) = $(round(rel_l2, sigdigits=4)), " *
        "max|diff| = $(round(max_abs_diff, sigdigits=4))."
    )
    println("  fitparas      -> $fitparas_path")
    println("  fitpulsemat   -> $fitpulsemat_path")
    println("  fitsamplediff -> $fitsamplediff_path")

    paths = (
        source_pulsemat=pulsemat_path, source_jld2=jld2_path,
        fitparas=fitparas_path, fitpulsemat=fitpulsemat_path, fitsamplediff=fitsamplediff_path,
    )
    return pulse, u_fit, fit_report, segments, diff_report, paths
end

"""
    _compare_against_saved_trajectory(a_check, Sp_check, Sz_check, data; atol=0.0) -> (rel_a, rel_p, rel_z, err_a, err_p, err_z)

Shared comparison core behind [`reconcile_against_jld2`](@ref) and
[`optimise_control_pulse_from_jld2`](@ref)'s own seed-reconciliation check:
maximum absolute error of `a_check`/`sum(Sp_check,dims=2)`/
`sum(Sz_check,dims=2)` against `data.a_sol`/`data.Σp_sol`/`data.Σz_sol`,
each scaled by that saved trajectory's own peak magnitude (`+ atol`, an
absolute floor guarding against division by a near-zero saved value).
"""
function _compare_against_saved_trajectory(a_check, Sp_check, Sz_check, data; atol::Real=0.0)
    Sigma_p_check = vec(sum(Sp_check, dims=2))
    Sigma_z_check = vec(sum(Sz_check, dims=2))

    err_a = maximum(abs.(a_check .- data.a_sol))
    err_p = maximum(abs.(Sigma_p_check .- data.Σp_sol))
    err_z = maximum(abs.(Sigma_z_check .- data.Σz_sol))

    scale_a = maximum(abs.(data.a_sol)) + atol
    scale_p = maximum(abs.(data.Σp_sol)) + atol
    scale_z = maximum(abs.(data.Σz_sol)) + atol

    return err_a / scale_a, err_p / scale_p, err_z / scale_z, err_a, err_p, err_z
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
    optrunlog_paths(path; out_dir=nothing) -> (optrunlog_path, pulsemat_path, pulsepara_path)

Derives this run's `<basename>_optrunlog.jld2`/`<basename>_opt_pulsemat.csv`/
`<basename>_opt_pulsepara.jld2` output paths from the SOURCE `.jld2` path's
own basename -- the same "same directory, same basename + suffix"
convention `save_run_data` already uses for its own `_pulsemat.csv`
sibling (see `pulses.jl`). Pass `out_dir` to write all three files
elsewhere instead of alongside `path`; either way the containing directory
is created if it doesn't exist yet.
"""
function optrunlog_paths(path::AbstractString; out_dir=nothing)
    endswith(path, ".jld2") || error("Expected a .jld2 path, got $path.")
    base = path[1:end-length(".jld2")]
    dir = out_dir === nothing ? dirname(base) : out_dir
    isempty(dir) || mkpath(dir)
    base_name = basename(base)
    return joinpath(dir, base_name * "_optrunlog.jld2"),
           joinpath(dir, base_name * "_opt_pulsemat.csv"),
           joinpath(dir, base_name * "_opt_pulsepara.jld2")
end

"""
    save_optimised_pulse_parameters(pulsepara_path, source_path, pulse, best_u; final_metrics=nothing) -> pulsepara_path

Writes ONLY the FINAL optimised control pulse's exact parameters to
`pulsepara_path` (`<basename>_opt_pulsepara.jld2`, see
[`optrunlog_paths`](@ref)) -- deliberately separate from, and much
smaller than, `_optrunlog.jld2`'s full per-epoch `history`/settings/
output record, so a caller who only wants the converged pulse can load a
small, self-contained file without pulling in the whole run log. No
per-epoch trail is kept here, just this one converged result.

The `.jld2` file holds, under the top-level key `"data"` (same convention
[`load_jld2_run`](@ref) already reads):
  - `source_path` -- the ORIGINAL `.jld2` run this optimisation started from
  - `k`, `n_coeff_A`, `n_coeff_f`, `degree`, `taper_frac`, `T_max`,
    `amp_scale` -- `pulse`'s own defining fields, everything needed to
    reconstruct an identical `CompositePulse`
  - `final_u` -- the optimised raw parameter vector (`best_u`)
  - `t_start`, `t_end`, `cA`, `cf` -- the DECODED pulse parameters (see
    [`decode`](@ref)): each sub-pulse's start/end time and its amplitude/
    frequency B-spline coefficients, i.e. the actual physical pulse shape
    `best_u` encodes
  - `final_metrics` -- `(cost, inversion, silencing, duration, coherence)`
    from [`pulse_cost`](@ref) at `best_u`, if supplied (`nothing`
    otherwise); context only, not required to reconstruct the pulse
"""
function save_optimised_pulse_parameters(
    pulsepara_path::AbstractString, source_path::AbstractString,
    pulse::CompositePulse, best_u::AbstractVector;
    final_metrics=nothing,
)
    t_start, t_end, cA, cf = decode(pulse, best_u)
    pulsepara = (
        source_path=source_path,
        k=pulse.k, n_coeff_A=pulse.n_coeff_A, n_coeff_f=pulse.n_coeff_f,
        degree=pulse.degree, taper_frac=pulse.taper_frac,
        T_max=pulse.T_max, amp_scale=pulse.amp_scale,
        final_u=collect(best_u),
        t_start=collect(t_start), t_end=collect(t_end), cA=collect(cA), cf=collect(cf),
        final_metrics=final_metrics,
    )
    JLD2.save(pulsepara_path, "data", pulsepara)
    println("Saved optimised pulse parameters to $pulsepara_path")
    return pulsepara_path
end

"""
    save_optimisation_run_log(path, data, d, pulse, signal_E_of_t,
                               n_signal, use_signal, u0, initial_metrics,
                               best_u, final_metrics, history, optimizer_settings;
                               out_dir=nothing, pulsemat_N=nothing)
        -> (optrunlog_path, pulsemat_path, pulsepara_path)

Writes this run's full record to `<basename>_optrunlog.jld2` (see
[`optrunlog_paths`](@ref)), samples the FINAL/optimal CONTROL pulse's own
drive (`build_E_of_t(pulse, best_u)` -- the control pulse alone, NOT
combined with the signal, since the signal is a separate FIXED input that
was never part of what got optimised) to `<basename>_opt_pulsemat.csv`
via the existing [`sample_E_of_t`](@ref)/[`save_E_samples`](@ref) (same
format/read pattern as every other `_pulsemat.csv` in this package), and
writes the same final pulse's exact/decoded parameters to
`<basename>_opt_pulsepara.jld2` via
[`save_optimised_pulse_parameters`](@ref) (a small, standalone file --
see its own docstring for what it holds; `final_u` also still lives
inside `_optrunlog.jld2` below, unchanged, since
[`optimise_composite_pulse`](@ref)'s `warm_start_u` continues to read it
from there).
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
    `solve_kwargs` override such as `reltol`/`abstol`/`w_inv`/`w_sil`/
    `target_F`/`w_time`) -- see [`optimise_composite_pulse`](@ref)'s own
    docstring for exactly what it captures and why (and what it
    deliberately excludes, e.g. non-serialisable closures)
  - `initial_u` (the candidate pulse's own raw parameterisation, `u0`,
    used only after reconciliation passed)
  - `initial_metrics` (`(cost, inversion, silencing, duration, coherence)`
    at `u0`, from [`pulse_cost`](@ref) -- `cost` depends only on
    inversion and the collective silencing factor `|F|` (see
    [`_weighted_silencing_factor`](@ref)) plus the time/power penalties;
    `coherence` rides along in the same tuple but, like `duration`, is
    NOT part of `cost` -- it is the OLDER per-bin `Nj`-weighted mean of
    `|Sp|/(Nj/2)` (see [`_weighted_coherence`](@ref)), DIAGNOSTIC ONLY,
    recorded purely for comparison against the collective `|F|` actually
    being optimised)
  - `initial_coherence` -- same `coherence` value as `initial_metrics`
    above, evaluated once at `u0` from an `:equator` solve, kept as its
    own top-level key purely for convenience so a saved run can be
    compared against that simpler metric without digging into the
    `initial_metrics` tuple
  - `initial_output` (`(a, Sigma_p, Sigma_z)` at `t1`, from actually
    simulating `u0` -- the candidate pulse's raw simulated output)
  - `history` -- one row per optimiser epoch, across every hop (see
    [`run_local_adam`](@ref)): `hop, epoch, k, cost, inversion, silencing,
    duration, coherence, improved`. `coherence` here is the SAME
    diagnostic-only per-bin metric as `initial_coherence`/`final_coherence`
    below, recorded for every epoch (reusing the `:equator` solve already
    run for `silencing`, so it costs nothing extra) -- never fed into
    `cost`, which the optimiser actually descends on. History rows do NOT
    carry the raw pulse parameter vector `u` for that epoch -- only the
    FINAL optimised pulse's exact parameters are saved, and in a separate
    file (see [`save_optimised_pulse_parameters`](@ref)/
    `_opt_pulsepara.jld2` below), not embedded here
  - `final_u` (the optimised control pulse's raw parameterisation,
    `best_u`)
  - `final_metrics`/`final_output`/`final_coherence` -- same shape as
    the initial ones, for `best_u`
"""
function save_optimisation_run_log(
    path::AbstractString, data, d, pulse::CompositePulse, signal_E_of_t,
    n_signal::Integer, use_signal::Bool,
    u0::AbstractVector, initial_metrics,
    best_u::AbstractVector, final_metrics, history, optimizer_settings;
    out_dir=nothing, pulsemat_N=nothing,
)
    optrunlog_path, pulsemat_path, pulsepara_path = optrunlog_paths(path; out_dir=out_dir)

    a0, Sp0, Sz0, _ = run_sim_1st_order_pure(u0, pulse, d; signal_E_of_t=signal_E_of_t)
    initial_output = (a=a0, Sigma_p=sum(Sp0), Sigma_z=sum(Sz0))

    a1, Sp1, Sz1, _ = run_sim_1st_order_pure(best_u, pulse, d; signal_E_of_t=signal_E_of_t)
    final_output = (a=a1, Sigma_p=sum(Sp1), Sigma_z=sum(Sz1))

    # Diagnostic only -- NOT part of the optimised cost (pulse_cost uses
    # the silencing factor |F|, not this per-bin coherence average).
    # Logged purely so a saved run can be compared against the old
    # coherence-based metric without re-running anything: two extra
    # :equator solves at u0/best_u, done once here, not every epoch.
    _, Sp0_eq, _, Nj0_eq = run_sim_1st_order_pure(u0, pulse, d; signal_E_of_t=signal_E_of_t, initial_condition=:equator)
    initial_coherence = _weighted_coherence(Sp0_eq, Nj0_eq, Float64)
    _, Sp1_eq, _, Nj1_eq = run_sim_1st_order_pure(best_u, pulse, d; signal_E_of_t=signal_E_of_t, initial_condition=:equator)
    final_coherence = _weighted_coherence(Sp1_eq, Nj1_eq, Float64)

    full_settings = merge((n_signal=n_signal, USE_SIGNAL=use_signal), optimizer_settings)

    run_log = (
        source_path=path, n_signal=n_signal, use_signal=use_signal,
        SIM_SETTING=data.SIM_SETTING, SYSTEM_CONFIG=data.SYSTEM_CONFIG,
        k=pulse.k, n_coeff_A=pulse.n_coeff_A, n_coeff_f=pulse.n_coeff_f,
        optimizer_settings=full_settings,
        initial_u=collect(u0), initial_metrics=initial_metrics, initial_output=initial_output,
        initial_coherence=initial_coherence,
        history=history,
        final_u=collect(best_u), final_metrics=final_metrics, final_output=final_output,
        final_coherence=final_coherence,
    )
    JLD2.save(optrunlog_path, "data", run_log)
    println("Saved optimisation run log to $optrunlog_path")

    N = pulsemat_N === nothing ? data.SIM_SETTING.Nt_save : pulsemat_N
    control_E_of_t = build_E_of_t(pulse, best_u)
    sample_E_of_t(control_E_of_t, pulse.T_max, N; savepath=pulsemat_path)

    save_optimised_pulse_parameters(pulsepara_path, path, pulse, best_u; final_metrics=final_metrics)

    return optrunlog_path, pulsemat_path, pulsepara_path
end

"""
    optimise_control_pulse_from_jld2(path, k, n_coeff_A, n_coeff_f;
        n_signal=1, use_signal=true, reconcile=true, rtol_check=1e-3, atol_check=0.0,
        check_reltol=nothing, check_abstol=nothing,
        fit_seed_from_file=true, fit_N=4000, fit_num_epochs=1000, fit_learning_rate=0.002,
        reconcile_seed=true, rtol_seed_check=0.1, atol_seed_check=0.0,
        save_log=true, log_out_dir=nothing,
        pulsemat_N=nothing, optimizer_kwargs...)
        -> (best_u, best_cost, pulse::CompositePulse, signal_E_of_t, d, data, seed_fit_report)

End-to-end workflow tying [`load_jld2_run`](@ref),
[`reconcile_against_jld2`](@ref), [`fit_composite_pulse_seed`](@ref),
[`optimise_composite_pulse`](@ref), and
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
     effect whatsoever on the ensemble.
  5. Builds the optimiser's SEED. Unless an explicit `warm_start_u` is
     passed in `optimizer_kwargs` (which always wins, skipping this step
     entirely) or `fit_seed_from_file=false`: interprets `path`'s own
     recorded CONTROL pulse (step 2's `control_cfg`, e.g. an analytic
     WURST/ARP pulse -- NOT a `CompositePulse`) via
     [`fit_composite_pulse_seed`](@ref) -- a ForwardDiff/Adam SHAPE fit
     (`fit_N`/`fit_num_epochs`/`fit_learning_rate` forwarded to it), no ODE
     solve involved -- into a `CompositePulse(k, n_coeff_A, n_coeff_f, d)`
     raw parameter vector `u_fit`. Unless `reconcile_seed=false`,
     `u_fit`'s ACTUAL physics (signal, always on here -- same as step 3 --
     plus `build_E_of_t(pulse, u_fit)`) is then simulated
     ([`run_sim_1st_order_trajectory`](@ref)) and compared against the
     file's own saved trajectory to `rtol_seed_check` (looser than
     `rtol_check` by default: `fit_composite_pulse_seed`'s B-spline family
     cannot exactly reproduce an arbitrary recorded pulse shape, only
     approximate it, so SOME residual mismatch here is expected and
     inherent, unlike step 3's near-exact reconciliation). Throws an error
     and refuses to proceed if this does not match within
     `rtol_seed_check`/`atol_seed_check` -- raise `fit_N`/`fit_num_epochs`,
     adjust `k`/`n_coeff_A`/`n_coeff_f`, loosen `rtol_seed_check`, or pass
     `reconcile_seed=false` to override at your own risk. If `warm_start_u`
     was explicitly given, or `fit_seed_from_file=false`, this whole step
     is skipped and [`optimise_composite_pulse`](@ref) falls back to its
     own default (`warm_start_u` if given, else a fresh random
     `initial_guess`).
  6. Runs [`optimise_composite_pulse`](@ref)-style Adam + basin-hopping
     optimisation on the `CompositePulse` (`k` B-spline sub-pulses)
     CONTROL pulse from that seed, with the signal passed through as
     `signal_E_of_t`. The signal pulse is NEVER part of the optimised
     parameter vector `u` -- see [`run_sim_1st_order_pure`](@ref)'s own
     docstring for why this is structural (a plain captured closure, not
     a differentiated quantity), not merely a convention: the optimiser
     is physically incapable of touching the signal pulse.
  7. Unless `save_log=false`, writes the full run record via
     [`save_optimisation_run_log`](@ref) (`log_out_dir`/`pulsemat_N`
     forwarded to it as `out_dir`/`pulsemat_N`), including a complete
     `optimizer_settings` record -- everything [`optimise_composite_pulse`](@ref)
     captured (`num_epochs`, `learning_rate`, `patience`, `tol`, `n_hops`,
     `hop_patience`, `hop_step_size`, `temperature`, `w_tmax`, `degree`,
     `taper_frac`, `seed`, any numeric `solve_kwargs` override) merged with
     this function's own
     (`n_signal`, `USE_SIGNAL`, `reconcile`, `rtol_check`, `atol_check`,
     `check_reltol`, `check_abstol`, `fit_seed_from_file`, `fit_N`,
     `fit_num_epochs`, `fit_learning_rate`, `reconcile_seed`,
     `rtol_seed_check`, `atol_seed_check`) -- so a
     saved run can be replicated exactly later just by reading its own
     log.

Returns everything a caller would need afterwards: the optimised raw
parameters, its cost, the `CompositePulse` they decode against, the fixed
`signal_E_of_t` actually used, the derived ensemble `d`, the loaded
`data` (e.g. for further comparison/plotting against the original run),
and `seed_fit_report` -- `nothing` if step 5 was skipped, else
`(u_fit, fit_report, seed_reconcile_ok, seed_reconcile_report)` from that
step (`fit_report`/`seed_reconcile_report` as documented in
[`fit_composite_pulse_seed`](@ref)/[`_compare_against_saved_trajectory`](@ref)).
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
    fit_seed_from_file::Bool=true,
    fit_N::Integer=4000,
    fit_num_epochs::Integer=1000,
    fit_learning_rate::Real=0.002,
    reconcile_seed::Bool=true,
    rtol_seed_check::Real=0.1,
    atol_seed_check::Real=0.0,
    save_log::Bool=true,
    log_out_dir=nothing,
    pulsemat_N=nothing,
    optimizer_kwargs...,
)
    data = load_jld2_run(path)
    CONFIG = build_full_config(data.SIM_SETTING, data.SYSTEM_CONFIG)
    d = prepare_derived(CONFIG)
    signal_cfg, control_cfg = split_signal_control(data.PULSE_CONFIG; n_signal=n_signal)

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

    seed_fit_report = nothing
    if fit_seed_from_file && !(:warm_start_u in keys(optimizer_kwargs))
        degree = get(optimizer_kwargs, :degree, 3)
        taper_frac = get(optimizer_kwargs, :taper_frac, 0.1)
        seed_pulse = CompositePulse(k, n_coeff_A, n_coeff_f, d; degree=degree, taper_frac=taper_frac)
        fit_seed = get(optimizer_kwargs, :seed, 42)

        u_fit, fit_report = fit_composite_pulse_seed(
            control_cfg, seed_pulse;
            N_fit=fit_N, num_epochs=fit_num_epochs, learning_rate=fit_learning_rate, seed=fit_seed,
        )
        println(
            "Fitted a k=$k CompositePulse seed from $path's own recorded control pulse: " *
            "mse=$(round(fit_report.mse, sigdigits=4)) rel_l2=$(round(fit_report.rel_l2, sigdigits=4))"
        )

        seed_ok = true
        seed_report = nothing
        if reconcile_seed
            signal_E_always = build_E_of_t(signal_cfg)
            control_E_fit = build_E_of_t(seed_pulse, u_fit)
            E_seed_check(t) = signal_E_always(t) + control_E_fit(t)
            reltol_solve = check_reltol === nothing ? data.SIM_SETTING.reltol : check_reltol
            abstol_solve = check_abstol === nothing ? data.SIM_SETTING.abstol : check_abstol
            _, a_check, Sp_check, Sz_check = run_sim_1st_order_trajectory(
                E_seed_check, d; reltol=reltol_solve, abstol=abstol_solve,
            )
            rel_a, rel_p, rel_z, err_a, err_p, err_z = _compare_against_saved_trajectory(
                a_check, Sp_check, Sz_check, data; atol=atol_seed_check,
            )
            seed_ok = rel_a < rtol_seed_check && rel_p < rtol_seed_check && rel_z < rtol_seed_check
            seed_report = (rel_a=rel_a, rel_p=rel_p, rel_z=rel_z, err_a=err_a, err_p=err_p, err_z=err_z)
            status = seed_ok ? "PASS" : "FAIL"
            println(
                "Seed reconciliation against $path: $status (rtol=$rtol_seed_check)  " *
                "rel_a=$(round(rel_a, sigdigits=4)) rel_p=$(round(rel_p, sigdigits=4)) rel_z=$(round(rel_z, sigdigits=4))"
            )
            seed_ok || error(
                "Fitted seed's own simulated trajectory does not reconcile against $path " *
                "(rel_a=$rel_a, rel_p=$rel_p, rel_z=$rel_z, tolerance rtol=$rtol_seed_check, " *
                "shape-fit rel_l2=$(fit_report.rel_l2)) -- refusing to optimise from a seed whose " *
                "physics doesn't match the source file. Raise fit_N/fit_num_epochs, adjust " *
                "k/n_coeff_A/n_coeff_f, loosen rtol_seed_check, or pass reconcile_seed=false " *
                "to override at your own risk."
            )
        end

        optimizer_kwargs = merge(NamedTuple(optimizer_kwargs), (warm_start_u=u_fit,))
        seed_fit_report = (u_fit=u_fit, fit_report=fit_report, seed_reconcile_ok=seed_ok, seed_reconcile_report=seed_report)
    end

    best_u, best_cost, pulse, u0, initial_metrics, history, final_metrics, optimizer_settings = optimise_composite_pulse(
        k, n_coeff_A, n_coeff_f, d; signal_E_of_t=signal_E_of_t, optimizer_kwargs...,
    )

    if save_log
        full_settings = merge(
            (reconcile=reconcile, rtol_check=rtol_check, atol_check=atol_check,
             check_reltol=check_reltol, check_abstol=check_abstol,
             fit_seed_from_file=fit_seed_from_file, fit_N=fit_N, fit_num_epochs=fit_num_epochs,
             fit_learning_rate=fit_learning_rate, reconcile_seed=reconcile_seed,
             rtol_seed_check=rtol_seed_check, atol_seed_check=atol_seed_check),
            optimizer_settings,
        )
        save_optimisation_run_log(
            path, data, d, pulse, signal_E_of_t, n_signal, use_signal,
            u0, initial_metrics, best_u, final_metrics, history, full_settings;
            out_dir=log_out_dir, pulsemat_N=pulsemat_N,
        )
    end

    return best_u, best_cost, pulse, signal_E_of_t, d, data, seed_fit_report
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

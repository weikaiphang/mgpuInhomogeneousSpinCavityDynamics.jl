# ============================================================
# TRANS-DIMENSIONAL (RJMCMC-FLAVOURED) K-HOPPING EXTENSION
#
# This file is a THIN, ADDITIVE extension of pulse_optimizer2.jl: it does
# NOT redefine any of the shared physics/local-descent machinery
# (`build_u0_1st_order_cpu`, `_zero_drive`, `PulseSolveFailed`,
# `_successful_solve`, `run_sim_1st_order_pure`, `_forbid_initial_condition`,
# `_weighted_inversion`, `_weighted_coherence`, `_weighted_silencing_factor`,
# `pulse_metrics`, `pulse_cost`, `AdamState`, `adam_step!`, `run_local_adam`,
# `_normalise_k_specs`) -- those are used here EXACTLY as pulse_optimizer2.jl
# defines them, by direct reference, not by a local copy. This file must
# therefore be loaded AFTER pulse_optimizer2.jl (and after composite_pulse.jl/
# canon_pulses.jl, same as pulse_optimizer2.jl itself requires) into the SAME
# namespace -- e.g. `include("pulse_optimizer2.jl")` then
# `include("pulse_optimizer2_RJMCMC.jl")`, or (for ad hoc testing outside the
# package's own include chain) `using InhomogeneousSpinCavityDynamics` (which
# already includes pulse_optimizer2.jl) followed by
# `Base.include(InhomogeneousSpinCavityDynamics, "pulse_optimizer2_RJMCMC.jl")`.
#
# This deduplication replaces an earlier version of this file that carried
# its OWN full copy of every function above (a Julia port of
# InhomogeneousSpinCavityDynamics.py/pulse_optimized_spline.py, further
# extended with the dual-trajectory/silencing-factor cost -- see
# pulse_optimizer2.jl's own module docstring for that history). Keeping two
# independent, near-identical copies of ~500 lines of cost-function/local-
# descent code across two files is exactly the kind of thing that silently
# drifts the moment one copy changes and nobody remembers to update the
# other -- which is precisely what happened here (this file's copy fell out
# of sync with pulse_optimizer2.jl's own coherence -> silencing-factor
# rewrite before being caught and re-synced by hand). Deduplicating removes
# that failure mode structurally: there is now exactly ONE definition of the
# physics/cost function/local descent, in pulse_optimizer2.jl, and this file
# can only ever be as stale as pulse_optimizer2.jl itself, not independently
# stale on its own.
#
# What THIS file actually adds, layered strictly on top of that shared
# core, is a trans-dimensional outer hop loop: each hop can propose k+0
# (stay), k+1 (grow), or k-1 (shrink) -- not just a continuous perturbation
# at fixed k -- turning the ordinary basin-hopping loop into a (Metropolis-
# only, not Green's-ratio-corrected) reversible-jump search over the
# sub-pulse count itself. See `_k_move_probabilities`/`_choose_k_move` for
# the move probabilities and `_grow_pulse`/`_shrink_pulse` for how the raw
# parameter vector `u` is re-mapped across a dimension change.
#
# NAMING: pulse_optimizer2.jl's own `optimise_composite_pulse`/
# `optimise_composite_pulse_over_k` are fixed-k -- `pulse.k` is guaranteed to
# equal the input `k` for their entire duration. The trans-dimensional
# versions here do NOT have that guarantee (k can drift via `_grow_pulse`/
# `_shrink_pulse`), so they are deliberately given DISTINCT names --
# `optimise_composite_pulse_rjmcmc`/`optimise_composite_pulse_over_k_rjmcmc`
# -- rather than reusing pulse_optimizer2.jl's names. This is not just a
# style choice: both files define a function with the identical positional
# signature `(k, n_coeff_A, n_coeff_f, d; ...)`, so if they were ever given
# the SAME name and loaded into the same module, Julia would not treat them
# as two dispatchable methods -- the second `include` would simply
# overwrite the first's definition silently. Distinct names let both
# versions coexist safely in the same namespace: existing callers of
# pulse_optimizer2.jl's `optimise_composite_pulse` (e.g.
# jld2_pulse_loader.jl's `optimise_control_pulse_from_jld2`, which assumes
# `pulse.k == k` throughout) keep working completely unaffected, while
# callers that explicitly want trans-dimensional search opt in by calling
# the `_rjmcmc`-suffixed entry points instead.
# ============================================================

# ============================================================
# REVERSIBLE-JUMP (TRANS-DIMENSIONAL) K-HOPPING
#
# Extends the basin-hopping outer loop with probabilistic moves in the
# DISCRETE sub-pulse count k itself, not just a continuous perturbation at
# fixed k: each hop (after hop 0) proposes k+0 (stay), k+1 (grow), or k-1
# (shrink), maps the CURRENT accepted raw parameter vector onto a new one
# valid for the proposed k (_grow_pulse/_shrink_pulse below), runs one
# local Adam descent from there, and feeds the result through the same
# Metropolis test the fixed-k loop already used. This is basin-hopping
# across a trans-dimensional parameter space, not a fully Green's-ratio-
# corrected RJMCMC sampler -- no dimension-matching Jacobian is applied to
# the acceptance probability, consistent with the existing hop loop
# already comparing basins by raw cost, not by a normalised posterior
# density.
# ============================================================

"""
    _k_move_probabilities(k) -> (p_stay, p_up, p_down)

Base (unnormalised) weights for the three k-hopping moves: k+0 (stay,
weight 0.6), k+1 (grow, weight 0.2), k-1 (shrink, weight 0.2). `k-1` is
invalid whenever `k <= 1` ([`CompositePulse`](@ref) requires `k >= 1`),
so its weight is redistributed to the other two IN PROPORTION to their
own weight:

    P(k+0, new) = P(k+0) + [P(k+0) / (P(k+0)+P(k+1))] * P(k-1)
    P(k+1, new) = P(k+1) + [P(k+1) / (P(k+0)+P(k+1))] * P(k-1)

which -- since both remaining weights get scaled by the identical factor
`1 + P(k-1)/(P(k+0)+P(k+1))` -- is algebraically equivalent to simply
renormalising `{P(k+0), P(k+1)}` on their own; that's what this function
does. The returned triple always sums to 1 (the base 0.6/0.2/0.2 weights
already sum to 1 on their own, so this renormalisation step is only
actually a no-op there -- it still matters, and rescales the remaining
two weights, whenever k-1 is invalid and its 0.2 gets redistributed).
"""
function _k_move_probabilities(k::Integer)
    p_stay, p_up, p_down = 0.6, 0.2, 0.2
    if k <= 1
        p_down = 0.0
    end
    total = p_stay + p_up + p_down
    return p_stay / total, p_up / total, p_down / total
end

"""
    _choose_k_move(rng, k) -> :stay | :up | :down

Samples one of the three k-hopping moves using [`_k_move_probabilities`](@ref).
"""
function _choose_k_move(rng, k::Integer)
    p_stay, p_up, _ = _k_move_probabilities(k)
    r = rand(rng)
    r < p_stay && return :stay
    r < p_stay + p_up && return :up
    return :down
end

"""
    _physical_gap_dur(pulse, u) -> (gap, dur)

Per-sub-pulse physical `gap[i]` (silence immediately BEFORE sub-pulse
`i` -- `t_start[1] - 0` for `i=1`, `t_start[i] - t_end[i-1]` otherwise)
and `dur[i]` (`t_end[i] - t_start[i]`), recovered from `decode`'s own
`t_start`/`t_end`. Used by [`_grow_pulse`](@ref)/[`_shrink_pulse`](@ref)
as the physical-units intermediate that survives a k-change unchanged
for every UNTOUCHED sub-pulse (see those functions' docstrings for why
the RAW `u` values themselves cannot simply be copied across a k-change).
"""
function _physical_gap_dur(pulse::CompositePulse, u::AbstractVector)
    t_start, t_end, _, _, _ = decode(pulse, u)
    k = pulse.k
    gap = Vector{Float64}(undef, k)
    dur = Vector{Float64}(undef, k)
    prev_end = 0.0
    @inbounds for i in 1:k
        gap[i] = t_start[i] - prev_end
        dur[i] = t_end[i] - t_start[i]
        prev_end = t_end[i]
    end
    return gap, dur
end

"""
    _encode_gap_dur(new_pulse, gap, dur) -> (raw_gap, raw_dur)

Inverse of `decode`'s gap/duration half: re-encodes PHYSICAL `gap`/`dur`
vectors into raw (pre-softplus) values against `new_pulse`'s OWN
`gap_scale`/`dur_scale`/`dur_floor`. This is why [`_grow_pulse`](@ref)/
[`_shrink_pulse`](@ref) cannot just copy an untouched sub-pulse's raw
`u` entries across a k-change: `CompositePulse`'s own `gap_scale` and
`dur_scale` are both `T_max/(2k)` -- they change with `k` -- so decoding
the SAME raw value under the new pulse's scale would silently rescale
that sub-pulse's physical gap/duration. Re-deriving physical values first
(via [`_physical_gap_dur`](@ref)) and re-encoding them here against the
new scale is what keeps every untouched sub-pulse's absolute
`t_start`/`t_end` fixed across the k-change. (`freq_scale = d.FWHM` has no
k-dependence, so `raw_cf` needs no re-encoding and is copied unchanged by
both callers -- but `amp_scale` is NOT guaranteed k-invariant in general
(see [`CompositePulse`](@ref)'s own constructor), so `raw_cA` gets the
SAME physical-recover/re-encode treatment via
[`_physical_cA`](@ref)/[`_encode_cA`](@ref) below.)
"""
function _encode_gap_dur(new_pulse::CompositePulse, gap::AbstractVector, dur::AbstractVector)
    floor_gap = 1e-6 * new_pulse.gap_scale
    floor_dur = 1e-6 * new_pulse.dur_scale
    raw_gap = _softplus_inv.(max.(gap, floor_gap) ./ new_pulse.gap_scale)
    raw_dur = _softplus_inv.(max.(dur .- new_pulse.dur_floor, floor_dur) ./ new_pulse.dur_scale)
    return raw_gap, raw_dur
end

"""
    _physical_cA(pulse, u) -> Matrix{Float64}

Physical (decoded) amplitude-coefficient matrix `cA` (`n_coeff_A x k`),
recovered from `decode`'s own output -- the amplitude analogue of
[`_physical_gap_dur`](@ref). `CompositePulse`'s own `amp_scale` is NOT
guaranteed to be independent of `k` (see its constructor's own docstring:
it only happens to stay k-invariant while the hard-pulse bandwidth bound
dominates the `max` inside it, not in general), so -- exactly like
`gap`/`dur` -- `raw_cA` cannot simply be copied across a k-change:
decoding the SAME raw value under a different `amp_scale` would silently
rescale that sub-pulse's physical amplitude. [`_grow_pulse`](@ref)/
[`_shrink_pulse`](@ref) recover physical `cA` here first and re-encode it
via [`_encode_cA`](@ref) against the NEW pulse's own `amp_scale`.
"""
function _physical_cA(pulse::CompositePulse, u::AbstractVector)
    _, _, _, cA, _ = decode(pulse, u)
    return collect(Float64, cA)
end

"""
    _encode_cA(new_pulse, cA) -> Matrix{Float64}

Inverse of `decode`'s amplitude half: re-encodes a PHYSICAL `cA` matrix
into raw (pre-softplus) values against `new_pulse`'s OWN `amp_scale`. When
`amp_scale` is unchanged between the old and new pulse (the common case --
see [`_physical_cA`](@ref)), this reproduces the OLD raw values bit-for-
bit; it only actually rescales anything when `amp_scale` itself differs.
"""
function _encode_cA(new_pulse::CompositePulse, cA::AbstractMatrix)
    floor_amp = 1e-9 * new_pulse.amp_scale
    return _softplus_inv.(max.(cA, floor_amp) ./ new_pulse.amp_scale)
end

"""
    _grow_pulse(pulse, u, d) -> (new_pulse::CompositePulse, new_u::Vector{Float64})

Birth move for k -> k+1. Recovers `u`'s per-sub-pulse physical `gap`/`dur`
([`_physical_gap_dur`](@ref)), finds the LONGEST silence gap -- either the
leading gap (before sub-pulse 1) or one of the `k-1` internal gaps -- and
splices a new sub-pulse into its middle third (a third of the original
silence left on each side of the new sub-pulse). Every physical
`gap`/`dur` value (not just the split one) is then RE-ENCODED against the
new, `(k+1)`-sub-pulse `CompositePulse`'s own `gap_scale`/`dur_scale`
([`_encode_gap_dur`](@ref)) -- required because those scales are
`T_max/(2k)` and so change with k; skipping this step would silently
rescale every OTHER sub-pulse's physical timing, not just leave it alone
as intended. Every OLD sub-pulse's own `cA` is likewise recovered in
physical units ([`_physical_cA`](@ref)) and re-encoded against the NEW
pulse's `amp_scale` ([`_encode_cA`](@ref)) -- `amp_scale` is not
guaranteed k-invariant either (see [`CompositePulse`](@ref)'s own
constructor), so this is required for the same reason as gap/dur, even
though it's a no-op in the common case where `amp_scale` doesn't actually
change with k. The new sub-pulse itself is given a target PHYSICAL
amplitude of `0.01 * new_pulse.amp_scale` in every coefficient -- "quiet"
relative to a typical, inversion-capable segment at the NEW k, but NOT
too deep into softplus's saturated region: an earlier version of this
function used a fixed `raw_cA = -20` (`softplus(-20) ~ 2e-9` regardless of
`amp_scale`), and `softplus'(-20) = sigmoid(-20) ~ 2e-9` put a birthed
sub-pulse so deep into that saturated region that its cost gradient was
indistinguishable from float64 noise (`ForwardDiff` measured `~1e-15`, 13
orders of magnitude below an established sub-pulse's own `~1e-2`) --
verified directly: 5 real Adam epochs from `raw_cA=-20` left the decoded
amplitude unchanged past the 9th significant digit, i.e. the sub-pulse was
permanently inert within any realistic epoch budget. Encoding a target of
`0.01*amp_scale` (`softplus'` there is `~0.01`, ~7 orders of magnitude
larger) fixes that while still starting quiet, and now stays "1% of
amp_scale" in the correct, current units even if `amp_scale` itself
differs between the old and new pulse. `raw_cf = 0` (no chirp;
`freq_scale = d.FWHM` has no k-dependence, so this needs no re-encoding).

The new sub-pulse's own duration `d_new` is clamped to `new_pulse.dur_floor`
BEFORE the surrounding gaps are derived from it (see the inline comment at
the split itself) -- this is what actually keeps every OTHER sub-pulse's
absolute timing fixed when the gap being split is small; splitting first
and only discovering the floor during `_encode_gap_dur`'s own internal
clamp (the pre-existing behaviour) would silently widen the birthed
sub-pulse AFTER `g_after` had already been computed from the narrower,
pre-clamp value, drifting every later sub-pulse's timing by the difference.
"""
function _grow_pulse(pulse::CompositePulse, u::AbstractVector, d)
    _, _, raw_phi0, _, raw_cf = unpack(pulse, u)
    gap, dur = _physical_gap_dur(pulse, u)
    cA = _physical_cA(pulse, u)
    k = pulse.k

    new_pulse = CompositePulse(k + 1, pulse.n_coeff_A, pulse.n_coeff_f, d;
                                degree=pulse.degree, taper_frac=pulse.taper_frac)

    # `slot` (in the OLD 1:k numbering) is the sub-pulse whose own LEADING
    # gap is being split: slot=1 is the gap before the very first sub-pulse
    # (t=0 -> t_start[1]); slot=i (i>1) is the internal gap between old
    # sub-pulses i-1 and i. The new sub-pulse is inserted AT this index,
    # shifting old sub-pulse `slot` (and everything after it) one slot later.
    #
    # d_new is clamped to new_pulse.dur_floor (a fixed, k-independent
    # T_max*1e-3, see CompositePulse's own constructor) BEFORE g_after is
    # derived from it -- not after. _encode_gap_dur below cannot represent a
    # duration below dur_floor: encoding an unclamped d_new=G/3 < dur_floor
    # and then decoding it back silently inflates the birthed sub-pulse to
    # ~dur_floor, but g_after would already have been computed from the
    # SMALLER, pre-inflation G/3 -- so the new sub-pulse ends up wider than
    # budgeted and every later sub-pulse's absolute t_start silently drifts
    # later, violating this function's own "only the new sub-pulse's timing
    # changes" contract. Clamping here instead keeps g_after (and hence
    # every downstream sub-pulse's timing) consistent with what will
    # actually be encoded, by construction -- a genuine no-op whenever
    # G/3 >= dur_floor (the common case: g_new/d_new/g_after are then
    # bit-identical to the unclamped G/3 split). The extra max(.,0.0) below
    # guards the (rarer still) case G < dur_floor itself, where d_new alone
    # would otherwise exceed the whole gap being split.
    slot = argmax(gap)
    G = gap[slot]
    d_new = max(G / 3, new_pulse.dur_floor)
    g_new = max((G - d_new) / 2, 0.0)
    g_after = max(G - g_new - d_new, 0.0)

    new_gap = copy(gap)
    new_dur = copy(dur)
    new_gap[slot] = g_after
    insert!(new_gap, slot, g_new)
    insert!(new_dur, slot, d_new)

    new_raw_gap, new_raw_dur = _encode_gap_dur(new_pulse, new_gap, new_dur)

    silent_cA = fill(0.01 * new_pulse.amp_scale, pulse.n_coeff_A)
    zero_cf = zeros(pulse.n_coeff_f)
    new_cA = hcat(cA[:, 1:slot-1], silent_cA, cA[:, slot:end])
    new_raw_cA = _encode_cA(new_pulse, new_cA)
    raw_cf_mat = collect(Float64, raw_cf)
    new_raw_cf = hcat(raw_cf_mat[:, 1:slot-1], zero_cf, raw_cf_mat[:, slot:end])

    # raw_phi0 is UNCONSTRAINED (no softplus, no k-dependent scale -- see
    # decode's own docstring), so -- exactly like raw_cf -- it needs no
    # re-encoding across a k-change, only insertion of the new sub-pulse's
    # own slot. A birthed sub-pulse gets phi0=0 (no discrete jump), matching
    # its own zero_cf/quiet-cA treatment: no accumulated phase change of any
    # kind on top of whatever `running` already was at that point.
    raw_phi0_vec = collect(Float64, raw_phi0)
    new_raw_phi0 = vcat(raw_phi0_vec[1:slot-1], 0.0, raw_phi0_vec[slot:end])

    new_u = pack(new_pulse, new_raw_gap, new_raw_dur, new_raw_phi0, new_raw_cA, new_raw_cf)
    return new_pulse, new_u
end

"""
    _shrink_pulse(pulse, u, d) -> (new_pulse::CompositePulse, new_u::Vector{Float64})

Death move for k -> k-1 (requires `k >= 2`). Finds the sub-pulse with the
smallest AREA -- mean decoded amplitude times physical duration
(`mean(cA[:, i]) * dur[i]`, a coarse rectangular approximation of the
sub-pulse's actual net contribution, same spirit as [`total_area`](@ref)'s
own approximation) -- and removes its own slot from `u` entirely --
equivalent to first setting that sub-pulse's amplitude to `0.0` and then
dropping the now-inert slot, since a truly zero-amplitude sub-pulse
contributes nothing to [`build_E_of_t`](@ref) either way -- and, if it
wasn't the last sub-pulse, folds its own `gap + duration` into the NEXT
sub-pulse's gap. Area, not bare mean amplitude, is what decides removal:
a long, low-AMPLITUDE segment can still contribute substantial net
rotation/area (and composite sequences like CORPSE/BB1, already seeded
by [`seed_canonical`](@ref), deliberately rely on small-flip-angle
segments for error compensation), so ranking by amplitude alone risks
pruning a functionally important sub-pulse over a short, high-amplitude
one that actually contributes less. As in [`_grow_pulse`](@ref), every
SURVIVING sub-pulse's physical `gap`/`dur` is then RE-ENCODED against the
new, `(k-1)`-sub-pulse `CompositePulse`'s own `gap_scale`/`dur_scale`
([`_encode_gap_dur`](@ref)) -- required because those scales change with
k -- which is what keeps every surviving sub-pulse's absolute
`t_start`/`t_end` unchanged; surviving `cA` is likewise recovered in
physical units and re-encoded against the new pulse's `amp_scale`
([`_physical_cA`](@ref)/[`_encode_cA`](@ref)), since `amp_scale` is not
guaranteed k-invariant either (see [`CompositePulse`](@ref)'s own
constructor).
"""
function _shrink_pulse(pulse::CompositePulse, u::AbstractVector, d)
    pulse.k >= 2 || error("_shrink_pulse requires k >= 2, got k=$(pulse.k).")
    _, _, raw_phi0, _, raw_cf = unpack(pulse, u)
    gap, dur = _physical_gap_dur(pulse, u)
    cA = _physical_cA(pulse, u)
    k = pulse.k

    amps = vec(sum(cA; dims=1)) ./ pulse.n_coeff_A
    areas = amps .* dur
    m = argmin(areas)

    new_gap = copy(gap)
    new_dur = copy(dur)
    if m < k
        new_gap[m+1] += new_gap[m] + new_dur[m]
    end
    deleteat!(new_gap, m)
    deleteat!(new_dur, m)

    new_pulse = CompositePulse(k - 1, pulse.n_coeff_A, pulse.n_coeff_f, d;
                                degree=pulse.degree, taper_frac=pulse.taper_frac)
    new_raw_gap, new_raw_dur = _encode_gap_dur(new_pulse, new_gap, new_dur)

    keep = [1:m-1; m+1:k]
    new_cA = cA[:, keep]
    new_raw_cA = _encode_cA(new_pulse, new_cA)
    raw_cf_mat = collect(Float64, raw_cf)
    new_raw_cf = raw_cf_mat[:, keep]
    new_raw_phi0 = collect(Float64, raw_phi0)[keep]

    new_u = pack(new_pulse, new_raw_gap, new_raw_dur, new_raw_phi0, new_raw_cA, new_raw_cf)
    return new_pulse, new_u
end

"""
    _extract_physics_cost(cost, u, pulse, w_power) -> Float64

Strips [`pulse_cost`](@ref)'s L2 power penalty back out of an already-
computed `cost`, returning `physics_cost + w_time*(duration/T_max) +
tmax_penalty` alone, where `physics_cost = (1 -
inversion*silencing_success)^2` is `pulse_cost`'s multiplicative
fidelity term. `power_penalty =
w_power*sum(abs2, cA/amp_scale)/length(cA/amp_scale)` is a MEAN over
`k*n_coeff_A` coefficients, so it is NOT k-invariant: splicing in a
near-zero-amplitude sub-pulse dilutes that mean and lowers `power_penalty`
by itself, with zero change in inversion/silencing/duration -- verified
directly, a k=1->2 [`_grow_pulse`](@ref) call with NO local descent
afterwards dropped `pulse_cost` by ~0.019 (bigger than the entire
pre-growth cost) purely from this dilution. Left uncorrected, the
Metropolis test in [`optimise_composite_pulse_rjmcmc`](@ref) (`accept =
delta<0.0 || ...`, always taking an improving delta) would treat every
"grow" hop as a free, physics-free improvement and every "shrink" hop as
a free, physics-free penalty -- a one-way ratchet toward inflating `k`
that has nothing to do with reaching inversion=1/silencing=target_F.
Recomputing
`cA` via `decode(pulse, u)` here reproduces exactly what `pulse_cost`
itself computed internally (same deterministic call), so `cost -
power_penalty` recovers the k-INVARIANT part of the objective bit-for-bit;
`optimise_composite_pulse_rjmcmc` uses THIS, not the raw `cost`, for every
cross-hop Metropolis/global-best comparison, while still reporting the
ordinary full `pulse_cost` (via `final_metrics`) for the winning point.
Note this also removes `power_penalty` from same-k ("stay") hop
comparisons, not only cross-k ones -- harmless there (the mean is over
the SAME coefficient count on both sides, so no dilution artifact exists
to remove), but it does mean the outer hop-acceptance loop no longer uses
power efficiency as a tie-breaker at all; the per-epoch Adam descent
inside each hop still directly minimises the full `pulse_cost` (power
penalty included), so within-hop amplitude regularisation is unaffected.
"""
function _extract_physics_cost(cost, u::AbstractVector, pulse::CompositePulse, w_power)
    _, _, _, cA, _ = decode(pulse, u)
    normalized_cA = cA ./ pulse.amp_scale
    power_penalty = w_power * (sum(abs2, normalized_cA) / length(normalized_cA))
    return cost - power_penalty
end

"""
    optimise_composite_pulse_rjmcmc(k, n_coeff_A, n_coeff_f, d;
        num_epochs=30, learning_rate=0.05, patience=5, tol=1e-3,
        n_hops=3, hop_patience=2, hop_step_size=0.5, temperature=1.0,
        degree=3, taper_frac=0.1, w_tmax=1.0, w_power=0.05,
        target_F=1.0, w_time=0.15, seed=42,
        warm_start_u=nothing, label_prefix="",
        anneal_direct_weights=true, x_tune_alpha=_DEFAULT_X_TUNE_ALPHA,
        recalibrate_optima_x=true,
        I_min=_DEFAULT_PENALTY_MIN, kappa_I=_DEFAULT_PENALTY_KAPPA,
        S_min=_DEFAULT_PENALTY_MIN, kappa_S=_DEFAULT_PENALTY_KAPPA, solve_kwargs...)
        -> (best_u, best_cost, pulse::CompositePulse, u0, initial_metrics, history, final_metrics, optimizer_settings)

Trans-dimensional (reversible-jump-flavoured) basin-hopping global search
over composite-pulse solutions for THIS package's own 1st-order physics,
each basin explored by [`run_local_adam`](@ref)'s early-stopped Adam
descent:

  1. Run a local Adam descent from a random initial guess at the input
     `k` (hop 0).
  2. For each further hop: draw a move from [`_choose_k_move`](@ref) --
     k+0/"stay" (weight 0.6), k+1/"grow" (weight 0.2), or k-1/"shrink"
     (weight 0.2), renormalised, with the k-1 weight redistributed to the
     other two whenever `k <= 1` (see [`_k_move_probabilities`](@ref)).
     "stay" perturbs the CURRENT accepted point with isotropic Gaussian
     noise (`hop_step_size`, in raw/pre-softplus units), exactly as
     before. "grow"/"shrink" instead re-map the current point onto a
     `CompositePulse` with one more/fewer sub-pulse via
     [`_grow_pulse`](@ref)/[`_shrink_pulse`](@ref) -- splicing a new,
     near-zero-amplitude sub-pulse into the longest silence gap, or
     dropping the sub-pulse with the smallest AREA -- with NO added noise
     on top of that deterministic re-map. Either way, a local Adam descent
     runs from the resulting point (now possibly at a different `k`, hence
     a different `CompositePulse`/parameter-vector length than the
     previous hop; the descent itself still minimises the FULL
     [`pulse_cost`](@ref), power penalty included), then the standard
     basin-hopping Metropolis test applies to the [`_extract_physics_cost`](@ref)
     -- NOT the raw cost -- of the result: always accept an improving
     basin, accept a worse one with probability
     `exp(-(physcost_new-physcost_old)/temperature)` -- to decide whether
     the NEXT hop proposes from this new point (and its own `k`) or falls
     back to the previous one. The globally best `(u, cost, pulse)` seen
     over ALL hops (by physics cost) is tracked separately and is what
     gets returned; its own `k` need not match the input `k` at all.
  3. Stop hopping early once `hop_patience` consecutive hops produce no
     global improvement `> tol` (also measured in physics cost).

This is basin-hopping across a trans-dimensional space, not a fully
Green's-ratio-corrected RJMCMC sampler: no dimension-matching Jacobian
enters the acceptance probability above, consistent with every basin
being compared by cost, not by a normalised posterior density.
Comparisons use [`_extract_physics_cost`](@ref) rather than raw
[`pulse_cost`](@ref) specifically because the latter's L2 power penalty is
a MEAN over `k*n_coeff_A` coefficients -- not k-invariant -- so comparing
raw cost across a k-change would let a birthed near-zero sub-pulse's mere
presence (diluting that mean) masquerade as a physics improvement; see
[`_extract_physics_cost`](@ref)'s own docstring for the measured size of
that artifact (larger than the entire pre-growth cost, in one direct
test). `final_metrics` below still reports the ordinary full `pulse_cost`
for the winning point -- only the internal search comparisons are
physics-only.

`d` is `prepare_derived(CONFIG)`'s own return value (build it once via
this package's existing `build_full_config`/`prepare_derived`, e.g. from
`SIM_SETTING`/`SYSTEM_CONFIG` NamedTuples the same way `run_sim_1st_order`
does). Each epoch differentiates through one full forward solve of
`rhs_1st_order!` via `ForwardDiff.gradient` -- genuinely expensive for a
fine ensemble/tight tolerance, same caveat the Python port's own
docstrings carry: `num_epochs`/`n_hops` default to modest smoke-test
budgets, not a converged global optimum.

`degree` / `taper_frac` are forwarded to [`CompositePulse`](@ref) (defaults
3 and 0.1, same as constructing the pulse by hand). `w_tmax`, `w_power`,
`target_F`, and `w_time` are forwarded to [`pulse_cost`](@ref), targeting
`target_F=1.0` (RASE-style revival; pass `target_F=0.0` for ROSE-style
silencing instead). `track` (`:dual` default / `:weak`, see
[`_assert_track`](@ref)) chooses one or two ODE solves per cost
evaluation and rides through every `run_local_adam` hop (any k). These
are explicit keywords so they are NOT passed through to the ODE solver.
Do not pass `initial_condition` — the cost fixes its own ICs.

Besides the optimised `(best_u, best_cost, pulse)` -- `pulse` here is the
`CompositePulse` matching `best_u`'s OWN `k`, which (unlike the fixed-k
basin-hopping in pulse_optimizer2.jl) need not equal the input `k`, since
hops may have grown or shrunk it; `best_cost` is the ordinary full
`pulse_cost` (power penalty included) AT the point selected by physics-
cost comparison -- it is therefore the lowest RAW cost reachable via that
selection, not necessarily the single lowest raw cost value evaluated
anywhere during the run, since a higher-k point with a more diluted power
penalty could report a lower raw cost without actually being a better
pulse (see [`_extract_physics_cost`](@ref)) -- also returns: `u0` (the
initial/candidate parameterisation hop 0 actually started from -- either
a fresh random guess or `warm_start_u`, see below, always at the INPUT
`k`),
`initial_metrics` (`(cost, inversion, silencing, duration)` at `u0`, from
[`pulse_cost`](@ref)), `history` -- the [`run_local_adam`](@ref) per-epoch
log from EVERY hop, concatenated in run order (each row tagged with its
own `hop`/`epoch`) -- `final_metrics` (the same `(cost, inversion,
silencing, duration)` shape, for `best_u` specifically -- NOT necessarily
`history[end]`, since `best_u` can come from an earlier epoch than the
last one run), and `optimizer_settings`, a `NamedTuple` of every setting
that actually affected this run: `k`/`n_coeff_A`/`n_coeff_f`/`degree`/
`taper_frac` plus every one of this function's own explicit keyword
arguments (`num_epochs`, `learning_rate`, `patience`, `tol`, `n_hops`,
`hop_patience`, `hop_step_size`, `temperature`, `w_tmax`, `w_power`,
`target_F`, `w_time`, `seed`), plus
any of `solve_kwargs` whose value isn't a `Function` (so e.g. a numeric
`reltol`/`abstol`/`target_F`/`w_time` override is captured, while a
non-serialisable closure like `signal_E_of_t` is deliberately excluded --
that one is captured separately, as `use_signal`/`n_signal`, by
[`optimise_control_pulse_from_jld2`](@ref), since those two scalars are
enough to rebuild the exact same closure deterministically). It also
carries `final_inversion_ground`/`final_inv_gap` from the automatic winner
re-check (canonical `:ground` inversion of `best_u` and the O(ε)
single-track bias; `0.0` gap for `track=:dual`) -- see [`_assert_track`](@ref).
All of these are exactly what [`optimise_control_pulse_from_jld2`](@ref)
needs to write a full, replicable run log; ordinary callers that only want
the optimised pulse can simply ignore the extra return values.

`warm_start_u`: if given (e.g. a previous run's saved `final_u`, from a
loaded `_optrunlog.jld2` -- see [`load_jld2_run`](@ref)), hop 0 starts
from THIS point instead of a fresh `initial_guess(pulse; seed=seed)`, so
a later call can pick up and continue optimising from where an earlier
one left off. Must have length `n_params(pulse)`, i.e. be a raw parameter
vector for a `CompositePulse` with the SAME `(k, n_coeff_A, n_coeff_f)`
as this call's -- an error is raised otherwise, since a length mismatch
would silently decode into a nonsensical pulse rather than fail loudly.

`anneal_direct_weights` (default `true`, forwarded unchanged to every
`run_local_adam` call, hop 0 and every subsequent hop -- INCLUDING across
a `_grow_pulse`/`_shrink_pulse` dimension change, since it's a
`run_local_adam`-level schedule, not a `pulse`-level one) anneals each
hop's own `w_time` gradient (`w_tmax`/`w_power` are NEVER annealed --
always the caller's own base weight) from near-zero as that hop's physics
fidelity improves -- see [`run_local_adam`](@ref)'s own docstring and
[`_curriculum_fidelity_weight`](@ref) for the schedule. As with the
fixed-`k` `optimise_composite_pulse`, this is an explicit keyword here
specifically so it is NEVER part of `solve_kwargs` -- `initial_metrics`/
`final_metrics` (`pulse_cost` calls, which know nothing about it) and
every hop's `_extract_physics_cost` comparison therefore still see only
the STATIC, caller-configured `w_time` `run_local_adam` already
re-evaluates `best_u` under before returning. Pass
`anneal_direct_weights=false` to disable annealing entirely and recover
the original fixed-`w_time` cost.

**`hop==0` always has `w_time` suppressed to `0.0`** -- exactly mirroring
[`run_local_adam`](@ref)'s own unconditional `hop==0` rule (physics-only
optimisation for the entire first hop, REGARDLESS of
`anneal_direct_weights`'s own value -- confirmed deliberate, not an
oversight; see that function's own docstring for why this is a genuinely
different `factor` default than the ordinary `anneal_direct_weights=false`
one). No calibration `pulse_cost` evaluation spent on hop 0 either way.
**Hop 1 is the first hop ORDINARY annealing (the `anneal_direct_weights`-
gated kind) ever applies to** -- INCLUDING when hop 1 itself performs a
`_grow_pulse`/`_shrink_pulse` k-change, since which hop number this is is
orthogonal to `k` -- and is therefore this function's own "seed" hop:
`x_tune_alpha` (default [`_DEFAULT_X_TUNE_ALPHA`](@ref)) picks the
schedule's curvature via [`solve_optimal_x_start`](@ref) -- there is no
raw, manually-set curvature keyword; it is always either calibrated or
the plain linear sentinel (see [`run_local_adam`](@ref)'s own docstring).
Under the defaults, hop 1's own `run_local_adam` call performs a SINGLE
MANDATORY calibration from ITS OWN starting point, exactly as
[`run_local_adam`](@ref) already does internally for any nonzero `hop` --
no separate calibration code is needed in this function for hop 1, unlike
hop 0's now-moot upfront calibration (removed entirely, since hop 0 never
anneals and therefore never needs a calibrated `x_tune` at all).
`recalibrate_optima_x` (default `true`) controls every hop AFTER hop 1
(hop 2 onwards) -- INCLUDING across a `_grow_pulse`/`_shrink_pulse`
dimension change: `true` re-runs the SAME per-hop calibration
`run_local_adam` already supports internally, against THAT hop's own
starting point (whose fidelity can differ a lot after a k-change); `false`
instead reuses hop 1's own single calibrated value (captured from its
returned `history`) for every subsequent hop unchanged, via
`run_local_adam`'s internal `_precalibrated_x_tune`, never recalibrating
again. Passing `x_tune_alpha=nothing` EXPLICITLY bypasses calibration
entirely from hop 1 onwards -- the only way to run the annealed schedule
uncalibrated (plain linear) -- and is a silent no-op whenever
`anneal_direct_weights=false`. If `n_hops == 1` (no hop ever reaches hop
1), the entire run never anneals at all.

`I_min`/`kappa_I`/`S_min`/`kappa_S` (defaults [`_DEFAULT_PENALTY_MIN`](@ref)/
[`_DEFAULT_PENALTY_KAPPA`](@ref) each, forwarded unchanged to every
`run_local_adam` call, hop 0 and every subsequent hop) are
[`run_local_adam`](@ref)'s own squared-hinge penalty feature, forwarded
through unchanged -- see that function's own docstring. Unlike annealing,
it is NOT gated by `hop==0`; it applies identically on every hop,
including hop 0.
"""
function optimise_composite_pulse_rjmcmc(
    k::Integer, n_coeff_A::Integer, n_coeff_f::Integer, d;
    num_epochs::Integer=30, learning_rate::Real=0.05, patience::Integer=5, tol::Real=1e-3,
    n_hops::Integer=3, hop_patience::Integer=2, hop_step_size::Real=0.5, temperature::Real=1.0,
    degree::Integer=3, taper_frac::Real=0.1, w_tmax::Real=1.0, w_power::Real=0.05,
    target_F::Real=1.0, w_time::Real=0.15,
    seed::Integer=42, warm_start_u=nothing, label_prefix::AbstractString="",
    track::Symbol=:dual,
    anneal_direct_weights::Bool=true,
    x_tune_alpha::Union{Nothing,Real}=_DEFAULT_X_TUNE_ALPHA, recalibrate_optima_x::Bool=true,
    I_min::Real=_DEFAULT_PENALTY_MIN, kappa_I::Real=_DEFAULT_PENALTY_KAPPA,
    S_min::Real=_DEFAULT_PENALTY_MIN, kappa_S::Real=_DEFAULT_PENALTY_KAPPA,
    solve_kwargs...,
)
    _forbid_initial_condition(solve_kwargs)
    _assert_track(track)
    pulse = CompositePulse(k, n_coeff_A, n_coeff_f, d; degree=degree, taper_frac=taper_frac)
    cost_kwargs = (w_tmax=w_tmax, w_power=w_power, target_F=target_F, w_time=w_time,
                   I_min=I_min, kappa_I=kappa_I, S_min=S_min, kappa_S=kappa_S, track=track)
    rng = Random.Xoshiro(seed)

    solve_settings = NamedTuple(kv for kv in pairs(solve_kwargs) if !(kv[2] isa Function))
    optimizer_settings = merge(
        (k=k, n_coeff_A=n_coeff_A, n_coeff_f=n_coeff_f, degree=degree, taper_frac=taper_frac,
         num_epochs=num_epochs, learning_rate=learning_rate, patience=patience, tol=tol,
         n_hops=n_hops, hop_patience=hop_patience, hop_step_size=hop_step_size, temperature=temperature,
         w_tmax=w_tmax, w_power=w_power, target_F=target_F, w_time=w_time, seed=seed,
         track=track,
         anneal_direct_weights=anneal_direct_weights, x_tune_alpha=x_tune_alpha,
         recalibrate_optima_x=recalibrate_optima_x,
         I_min=I_min, kappa_I=kappa_I, S_min=S_min, kappa_S=kappa_S),
        solve_settings,
    )

    println(
        "$(label_prefix)Optimising k=$k pulses, $(n_params(pulse)) raw parameters (ForwardDiff/Adam + " *
        "basin-hopping, physics: InhomogeneousSpinCavityDynamics.jl rhs_1st_order!) ..."
    )

    if warm_start_u === nothing
        u0 = initial_guess(pulse; seed=seed)
    else
        length(warm_start_u) == n_params(pulse) || error(
            "warm_start_u has length $(length(warm_start_u)), but this CompositePulse " *
            "(k=$k, n_coeff_A=$n_coeff_A, n_coeff_f=$n_coeff_f) needs $(n_params(pulse))."
        )
        u0 = collect(Float64, warm_start_u)
        println("$(label_prefix)Warm-starting hop 0 from a supplied raw vector.")
    end
    initial_metrics = pulse_cost(u0, pulse, d; cost_kwargs..., solve_kwargs...)
    history = NamedTuple[]

    # Hop 0 NEVER anneals (see run_local_adam's own hop==0 rule), so it has
    # nothing to calibrate x_tune for -- no seed-calibration block needed
    # here any more; x_tune_alpha/_precalibrated_x_tune are simply not
    # forwarded to hop 0's own call.
    current_u, current_cost, _, _, _, hop0_history = run_local_adam(
        u0, pulse, d, cost_kwargs; hop=0, num_epochs, patience, tol, learning_rate,
        label="$(label_prefix)[hop 0]", anneal_direct_weights=anneal_direct_weights,
        solve_kwargs...
    )
    append!(history, hop0_history)
    x_tune_seed = 0.0   # populated once hop 1 completes; used only if recalibrate_optima_x=false
    current_pulse = pulse
    # Physics-only cost (pulse_cost minus its k-dependent power-penalty
    # MEAN, see _extract_physics_cost) is what every hop's Metropolis test
    # and global-best comparison below actually uses, NOT the raw cost --
    # the raw power penalty is not k-invariant, so comparing it across a
    # k-change would let a birthed sub-pulse's mere presence (diluting the
    # mean) masquerade as a physics improvement.
    current_phys_cost = _extract_physics_cost(current_cost, current_u, current_pulse, w_power)
    global_best_pulse, global_best_u, global_best_cost = current_pulse, current_u, current_cost
    global_best_phys_cost = current_phys_cost
    hops_since_improve = 0

    for hop in 1:(n_hops-1)
        move = _choose_k_move(rng, current_pulse.k)
        if move === :stay
            candidate_pulse = current_pulse
            perturbation = hop_step_size .* randn(rng, length(current_u))
            candidate_u0 = current_u .+ perturbation
        elseif move === :up
            candidate_pulse, candidate_u0 = _grow_pulse(current_pulse, current_u, d)
        else
            candidate_pulse, candidate_u0 = _shrink_pulse(current_pulse, current_u, d)
        end

        # hop==1 is the first hop annealing ever applies to (see
        # run_local_adam's own hop==0 rule) -- it ALWAYS calibrates fresh
        # from its own starting point (possibly at a different k after a
        # grow/shrink move), exactly as run_local_adam's own mandatory
        # calibration already does for any nonzero hop; there is no earlier
        # annealed hop to reuse a value from yet. From hop 2 onwards,
        # recalibrate_optima_x=true (default) keeps recalibrating fresh
        # each time; false reuses hop 1's own calibrated value (captured
        # below) for every remaining hop unchanged, via run_local_adam's
        # internal _precalibrated_x_tune.
        hop_x_tune_alpha, hop_precal = if hop == 1 || recalibrate_optima_x
            (x_tune_alpha, nothing)
        else
            (nothing, x_tune_seed)
        end

        cand_u, cand_cost, _, _, _, hop_history = run_local_adam(
            candidate_u0, candidate_pulse, d, cost_kwargs;
            hop, num_epochs, patience, tol, learning_rate,
            label="$(label_prefix)[hop $hop move=$move k=$(candidate_pulse.k)]",
            anneal_direct_weights=anneal_direct_weights,
            x_tune_alpha=hop_x_tune_alpha, _precalibrated_x_tune=hop_precal,
            solve_kwargs...,
        )
        append!(history, hop_history)
        if hop == 1
            x_tune_seed = isempty(hop_history) ? 0.0 : hop_history[1].x_tune
        end
        cand_phys_cost = _extract_physics_cost(cand_cost, cand_u, candidate_pulse, w_power)

        if cand_phys_cost < global_best_phys_cost - tol
            global_best_pulse, global_best_u, global_best_cost = candidate_pulse, cand_u, cand_cost
            global_best_phys_cost = cand_phys_cost
            hops_since_improve = 0
        else
            hops_since_improve += 1
        end

        delta = cand_phys_cost - current_phys_cost
        accept = delta < 0.0 || rand(rng) < exp(-delta / max(temperature, 1e-12))
        if accept
            current_pulse, current_u, current_cost = candidate_pulse, cand_u, cand_cost
            current_phys_cost = cand_phys_cost
        end

        accept_str = accept ? "accepted" : "rejected"
        println(
            "$(label_prefix)hop $hop (move=$move, k=$(candidate_pulse.k)): local best raw_cost=$(round(cand_cost, digits=4)) " *
            "phys_cost=$(round(cand_phys_cost, digits=4)) " *
            "($accept_str as new basin, phys_delta=$(round(delta, digits=4))) " *
            "global best raw_cost=$(round(global_best_cost, digits=4)), global best k=$(global_best_pulse.k)"
        )

        if hops_since_improve >= hop_patience
            println("$(label_prefix)Basin-hopping stopped early after $hop hops (no global improvement > $tol for $hop_patience hops).")
            break
        end
    end

    final_metrics = pulse_cost(global_best_u, global_best_pulse, d; cost_kwargs..., solve_kwargs...)

    # Winner re-check under `track=:weak` -- see the identical block in
    # `optimise_composite_pulse`. One `:ground` solve of the winner so the saved
    # run log carries the canonical `:ground` inversion and the O(ε) bias.
    if track === :weak
        sk_final = _solver_kwargs(solve_kwargs)
        _, _, Sz_gf, Nj_gf = run_sim_1st_order_pure(
            global_best_u, global_best_pulse, d; sk_final..., initial_condition=:ground,
        )
        final_inversion_ground = Float64(_weighted_inversion(Sz_gf, d.g_b, Nj_gf, Float64))
        final_inv_gap = Float64(final_metrics[2]) - final_inversion_ground
        println(
            "$(label_prefix)track=:weak winner re-check: inversion(:weak)=" *
            "$(round(Float64(final_metrics[2]), sigdigits=6))  inversion(:ground)=" *
            "$(round(final_inversion_ground, sigdigits=6))  inv_gap=$(round(final_inv_gap, sigdigits=3))"
        )
    else
        final_inversion_ground = Float64(final_metrics[2])
        final_inv_gap = 0.0
    end
    optimizer_settings = merge(
        optimizer_settings,
        (final_inversion_ground=final_inversion_ground, final_inv_gap=final_inv_gap),
    )

    println("$(label_prefix)Optimisation complete. Global best cost: $(round(global_best_cost, digits=4)), global best k=$(global_best_pulse.k).")
    return global_best_u, global_best_cost, global_best_pulse, u0, initial_metrics, history, final_metrics, optimizer_settings
end

"""
    optimise_composite_pulse_over_k_rjmcmc(n_coeff_A, n_coeff_f, d;
        kinds=(:hs1, :corpse, :bb1), specs=nothing, threaded=true,
        Omega_max=nothing, beta=nothing, mu=nothing, seed=42,
        optimizer_kwargs...)
        -> NamedTuple

Discrete search over sub-pulse count `k`. `k` is not a continuous
decision variable: each `k` gets its own `CompositePulse`, a canonical
warm-start ([`seed_canonical`](@ref)), and an independent
[`optimise_composite_pulse_rjmcmc`](@ref) run on that pulse's continuous raw
parameters. The runs are independent and are threaded when
`threaded=true` and `Threads.nthreads() > 1` (start Julia with
`JULIA_NUM_THREADS=N`; for ODE-heavy work consider
`BLAS.set_num_threads(1)` so threads do not oversubscribe).

Each spec's `k` only fixes the STARTING sub-pulse count (hop 0):
[`optimise_composite_pulse_rjmcmc`](@ref) here is the trans-dimensional
k-hopping version, so a given spec's own run can drift to a different
final `k` via its `_grow_pulse`/`_shrink_pulse` moves. `per_k[i].k` (and
`best_k` below) therefore report the WINNING run's own actual `pulse.k`,
not the spec's nominal starting `k` -- so they always agree with the
`pulse`/`best_u` returned alongside them, even though this makes it
possible for two specs with different starting `k` to report the same
final `k` (the duplicate-`k` check above only rejects duplicate STARTING
`k`, which stays meaningful since each spec's own hop-0 seed is still
distinct).

Default `kinds` is HS1 (`k=1`), CORPSE (`k=5`), BB1 (`k=7`). Pass
`specs=((k, kind), ...)` to choose `k` and seed explicitly, including
`:random` for `initial_guess` (e.g. `specs=((3, :random), (5, :corpse))`).

`Omega_max` defaults to each pulse's own `amp_scale` (cavity-input
units). HS1 `beta`/`mu` default as in [`seed_canonical`](@ref).
`optimizer_kwargs` are forwarded to [`optimise_composite_pulse_rjmcmc`](@ref)
(`num_epochs`, `signal_E_of_t`, `w_tmax`, ...). Do not pass
`warm_start_u` -- the seed is built per `k`.

Returns a NamedTuple: `best_kind`, `best_k`, `best_u`, `best_cost`,
`pulse`, `u0`, `initial_metrics`, `history`, `final_metrics`,
`optimizer_settings` (the winning run, same payload as
[`optimise_composite_pulse_rjmcmc`](@ref) plus kind), and `per_k` (one
NamedTuple per spec, in input order).
"""
function optimise_composite_pulse_over_k_rjmcmc(
    n_coeff_A::Integer, n_coeff_f::Integer, d;
    kinds=(:hs1, :corpse, :bb1),
    specs=nothing,
    threaded::Bool=true,
    Omega_max=nothing,
    beta=nothing,
    mu=nothing,
    seed::Integer=42,
    optimizer_kwargs...,
)
    :warm_start_u in keys(optimizer_kwargs) && error(
        "optimise_composite_pulse_over_k_rjmcmc builds a per-k canonical seed; do not pass warm_start_u."
    )
    _forbid_initial_condition(optimizer_kwargs)

    job_specs = _normalise_k_specs(kinds, specs)
    isempty(job_specs) && error("No (k, kind) specs to optimise.")
    seen = Dict{Int,Symbol}()
    for (k, kind) in job_specs
        k isa Integer && k >= 1 || error("k must be a positive integer, got $k.")
        kind isa Symbol || error("kind must be a Symbol, got $kind.")
        haskey(seen, k) && error("Duplicate k=$k (kinds $(seen[k]) and $kind). Each k can run once.")
        seen[k] = kind
        if kind !== :random
            k_of_seed_kind(kind) == k || error(
                "kind $kind requires k=$(k_of_seed_kind(kind)), got k=$k."
            )
        end
    end

    n = length(job_specs)
    nthreads = Threads.nthreads()
    use_threads = threaded && nthreads > 1 && n > 1
    println(
        "Discrete-k search: $n independent continuous optimisations " *
        (use_threads ? "on $nthreads threads" : "serially") *
        ". kinds=$(collect(spec[2] for spec in job_specs))."
    )

    function run_spec(spec)
        k, kind = spec
        prefix = "[$kind k=$k] "
        deg = get(optimizer_kwargs, :degree, 3)
        tfrac = get(optimizer_kwargs, :taper_frac, 0.1)
        pulse_seed = CompositePulse(k, n_coeff_A, n_coeff_f, d; degree=deg, taper_frac=tfrac)
        Ω = Omega_max === nothing ? pulse_seed.amp_scale : Omega_max
        u0 = seed_canonical(pulse_seed, kind; Omega_max=Ω, beta=beta, mu=mu, seed=seed)
        best_u, best_cost, pulse_out, u0_out, initial_metrics, history, final_metrics, optimizer_settings =
            optimise_composite_pulse_rjmcmc(
                k, n_coeff_A, n_coeff_f, d;
                optimizer_kwargs...,
                seed=seed + 1000 * Int(k),
                warm_start_u=u0,
                label_prefix=prefix,
            )
        optimizer_settings = merge(optimizer_settings, (seed_kind=kind,))
        return (
            kind=kind, k=pulse_out.k, start_k=Int(k),
            best_u=best_u, best_cost=best_cost, pulse=pulse_out,
            u0=u0_out, initial_metrics=initial_metrics, history=history,
            final_metrics=final_metrics, optimizer_settings=optimizer_settings,
        )
    end

    per_k = Vector{NamedTuple}(undef, n)
    if use_threads
        Threads.@threads for i in 1:n
            per_k[i] = run_spec(job_specs[i])
        end
    else
        for i in 1:n
            per_k[i] = run_spec(job_specs[i])
        end
    end

    best = per_k[1]
    for r in per_k
        if r.best_cost < best.best_cost
            best = r
        end
    end

    println("Discrete-k search complete.")
    for r in per_k
        # Compared by `start_k`, not `k`: `start_k` is guaranteed unique
        # across specs (the upfront dedup check enforces distinct STARTING
        # k's), but `k` -- the winning run's own final, possibly hopped-to,
        # sub-pulse count -- is not, so two different specs' runs could
        # legitimately land on the same final k and both get marked "*"
        # if compared by `k` instead.
        mark = r.start_k == best.start_k ? "*" : " "
        k_str = r.k == r.start_k ? "k=$(r.k)" : "k=$(r.start_k)->$(r.k)"
        println(
            "  $mark $k_str $(r.kind): cost=$(round(r.best_cost, digits=4))"
        )
    end
    println("  winner: k=$(best.k) (started k=$(best.start_k)) $(best.kind)  cost=$(round(best.best_cost, digits=4))")

    return (
        best_kind=best.kind, best_k=best.k,
        best_u=best.best_u, best_cost=best.best_cost, pulse=best.pulse,
        u0=best.u0, initial_metrics=best.initial_metrics, history=best.history,
        final_metrics=best.final_metrics, optimizer_settings=best.optimizer_settings,
        per_k=per_k,
    )
end

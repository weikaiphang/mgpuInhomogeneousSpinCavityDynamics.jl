# ============================================================
# DIFFERENTIABLE PULSE OPTIMISATION (dual-trajectory)
#
# Julia port of InhomogeneousSpinCavityDynamics.py/pulse_optimized_spline.py,
# extended with a DUAL-TRAJECTORY cost (see pulse_metrics/pulse_cost below):
# where pulse_optimizer.jl scores a candidate pulse from a single :ground
# solve, this file runs the SAME pulse u from two independent initial
# conditions -- :ground (bright-mode / cooperativity-weighted inversion I,
# paper App. H) and :weak, a WEAK-EXCITATION seed (Sp = eps*Nj/2,
# eps = _WEAK_SEED << 1, so each bin evolves as an independent
# driven TLS -- paper App. B) scored by the paper silencing factor
# |F|_* = <|F(omega)|>, built PER FREQUENCY SLICE (Eq. 5 / A.132), NOT one
# sum over all bins and NOT a per-bin coherence average -- see
# _weighted_silencing_factor -- and optimises both simultaneously via a
# target_F-driven penalty (target_F=1 for RASE-style revival / pure chirp,
# target_F=0 for ROSE-style g-space silencing), plus an L2 power penalty
# on the decoded amplitude coefficients (w_power). This
# is the finalised, actively-maintained pulse-optimisation entry point
# for this package (see InhomogeneousSpinCavityDynamics.jl's own include
# list); pulse_optimizer.jl predates the dual-trajectory cost and is no
# longer included in the module.
#
# Ports the algorithmic structure (B-spline composite pulse, Adam descent
# with early stopping, basin-hopping outer loop) while driving THIS
# package's own real physics (rhs_1st_order!, prepare_derived) rather
# than the Python side's simplified toy model. Gradients are computed via
# ForwardDiff (forward-mode) -- the only AD backend available in this
# package's dependency tree, and one well-suited here: rhs_1st_order!'s
# ComplexF64 state differentiates natively and correctly through
# ForwardDiff.Dual (verified against finite differences; ordinary Julia
# generic-complex-arithmetic code has none of diffrax's own documented
# "complex dtype backward-mode may not yet produce correct results"
# caveat, since forward-mode propagates dual numbers through ordinary
# arithmetic rather than needing a custom reverse-mode/checkpointing rule
# for complex state).
#
# Cost is expensive: each epoch differentiates through TWO full ODE
# solves (one per initial condition) via a single combined
# ForwardDiff.gradient call. On the M=20000 real ensemble
# (data/data_1st_order/duration_100us_gstd_1em06Hz.jld2), a 15-epoch,
# single-hop run took ~6 hours and produced a genuinely good result under
# this file's PREDECESSOR cost (a per-bin |Sp| coherence average, not
# the silencing factor below): inversion 0.91, coherence 0.93, up from
# 0.47/0.88 at init -- verified end-to-end, not just gradient-checked on
# the small toy config. That result is evidence the dual-trajectory
# (:ground + :weak) approach itself works end-to-end; the silencing
# factor replacing coherence has been gradient-checked on the small toy
# config only (see _weighted_silencing_factor's own docstring for why a
# naive per-bin phasor version was rejected) -- a full real-ensemble run
# under the new cost has not been repeated yet.
# ============================================================

"""
    _WEAK_SEED

Weak-excitation seed amplitude `ε` for the `:weak` / `:weak_inverted`
initial conditions: each bin starts at `Sp = ε·Nj/2` with a polar angle
`≈ ε` off its pole. The paper (App. B) defines the silencing factor `F`
as a property of the single-TLS propagator, so every bin must stay in the
weak-excitation (Holstein–Primakoff) regime where the collective cavity
source `Σ_j g_j Sp_j` it feeds back is negligible next to the control
drive. `ε = 1e-3` is inside the `1e-3 … 1e-4` HP window the paper uses;
`ε = 1` (the macroscopic Dicke state `Sp = Nj/2`, i.e. the true
`:equator`) is explicitly rejected for this track.
[`_weighted_silencing_factor`](@ref) divides the initial per-slice
overlap `ε·Σ g_j² Nj_j/2` back out, so `|F| ∈ [0, 1]` and the
`target_F ∈ {0, 1}` (RASE/ROSE) semantics are unchanged by the rescaling.
"""
const _WEAK_SEED = 1.0e-3

"""
    build_u0_1st_order_cpu(M, Nj, ::Type{T}, initial_condition=:ground) -> Vector{Complex{T}}

Plain-`Vector` (not `CuArray`), element-type-generic analogue of
`build_u0_gpu_1st_order` -- `T` should be `eltype` of whatever pulse
parameter vector `u` is being solved/differentiated with (`Float64` for
an ordinary forward solve, `ForwardDiff.Dual` when differentiating),
so the initial state promotes correctly alongside the ODE's `E_of_t`-
driven trajectory. The same IC set is accepted by
`build_u0_gpu_1st_order`, `build_u0_gpu_2nd_order`, and
`set_initial_condition!`.

`initial_condition`:
  * `:ground`         -- south pole, `Sz = -Nj/2`, `Sp = 0`
  * `:inverted`       -- north pole, `Sz = +Nj/2`, `Sp = 0`
  * `:equator`        -- true equator, `Sz = 0`, `Sp = Nj/2` (macroscopic
    Dicke state; radiates, outside Holstein–Primakoff)
  * `:weak`          -- south pole + tiny `+x` seed, `Sz = -Nj/2`,
    `Sp = ε·Nj/2` (`ε = _WEAK_SEED`). The paper App. B silencing / `|F|`
    track: near-`|g⟩`, weak-excitation, each bin an independent driven TLS.
  * `:weak_inverted` -- north pole + tiny `+x` seed, `Sz = +Nj/2`,
    `Sp = ε·Nj/2`. The `:inverted` analogue of `:weak` (weak-excitation
    seed near the excited pole); not consumed by the dual-trajectory cost.
  * `:custom`         -- all zero (caller fills it in)
"""
function build_u0_1st_order_cpu(M::Integer, Nj::AbstractVector, ::Type{T},
    initial_condition::Symbol=:ground) where {T}
    u0 = zeros(Complex{T}, state_length_1st_order(M))
    sp = IDX1_Sp_start:idx1_Sz_start(M)-1
    sz = idx1_Sz_start(M):state_length_1st_order(M)
    if initial_condition == :ground
        # South pole: Sz = -Nj/2, Sp = 0
        u0[sz] .= .-Nj ./ 2
    elseif initial_condition == :inverted
        # North pole: Sz = +Nj/2, Sp = 0
        u0[sz] .= Nj ./ 2
    elseif initial_condition == :equator
        # True equator: Sz = 0, Sp = Nj/2 (real). Macroscopic Dicke state --
        # it radiates and sits outside Holstein–Primakoff, so it is NOT the
        # state the paper defines the silencing factor F from (use :weak).
        u0[sp] .= Nj ./ 2
    elseif initial_condition == :weak
        # South pole + tiny +x seed. Polar angle ≈ ε, ε = _WEAK_SEED ≪ 1.
        # Paper App. B "acceptable fallback inside the existing ODE": a
        # near-|g⟩ trajectory that keeps every bin in the weak-excitation
        # (Holstein–Primakoff) regime, evolving as an independent driven TLS
        # -- the collective cavity source Σ_j g_j Sp_j it feeds back is then
        # negligible next to the control. `_weighted_silencing_factor`
        # divides the initial per-slice overlap ε·Σ g² Nj/2 back out, so |F|
        # keeps its [0, 1] scale. Kept as a coherent-spin state to O(ε²):
        # the Bloch radius is (Nj/2)·√(1+ε²), off-sphere by ~ε²/2 ≈ 5e-7
        # for ε = 1e-3 -- Sp(0) is held at EXACTLY ε·Nj/2 so it matches the
        # silencing metric's initial-overlap denominator term-for-term.
        u0[sz] .= .-Nj ./ 2
        u0[sp] .= T(_WEAK_SEED) .* Nj ./ 2
    elseif initial_condition == :weak_inverted
        # North pole + the same tiny +x seed: the :inverted analogue of
        # :weak. Symmetric completion; not used by the dual-trajectory cost.
        u0[sz] .= Nj ./ 2
        u0[sp] .= T(_WEAK_SEED) .* Nj ./ 2
    elseif initial_condition == :custom
        # already zero
    else
        error("Unknown initial_condition = $(initial_condition). Use :ground, :inverted, " *
              ":equator, :weak, :weak_inverted, or :custom.")
    end
    return u0
end

"""
    _zero_drive(t) -> ComplexF64

The always-off signal drive: identically zero at every `t`, regardless
of type of `t` (real or `ForwardDiff.Dual`, since `t` itself is the ODE
integrator's own time variable, never a differentiated quantity). Default
`signal_E_of_t` for [`run_sim_1st_order_pure`](@ref) when no fixed signal
pulse is being layered under the (optimised) control pulse.
"""
_zero_drive(t) = zero(ComplexF64)

struct PulseSolveFailed <: Exception
    retcode
end

function Base.showerror(io::IO, e::PulseSolveFailed)
    print(io, "1st-order pulse ODE solve failed (retcode=$(e.retcode))")
end

function _successful_solve(sol)
    rc = sol.retcode
    name = rc isa Symbol ? rc : Symbol(string(rc))
    return name === :Success || name === :Terminated
end

# ============================================================
# GPU DEVICE POOL + MEMORY GUARDS (pulse ODE / gradient)
#
# Auto-detects functional CUDA devices (0 if CUDA is not loaded or not
# functional -- the test suite includes this file without CUDA, so every
# helper below must no-op in that case). Caps at 8 GPUs. The CPU Dual
# ODE path is unchanged when no device is available; GPU execution is a
# drop-in array-backend swap of the SAME `rhs_1st_order!` / Tsit5 solve
# already used on the host, never a different discretisation.
# ============================================================

const _PULSE_MAX_GPUS = 8
const _PULSE_GPU_MIN_M = 256
const _GPU_DUAL_OK = Ref{Union{Nothing,Bool}}(nothing)
const _PULSE_GPU_LOGGED = Threads.Atomic{Bool}(false)

function _cuda_mod()
    return isdefined(@__MODULE__, :CUDA) ? getfield(@__MODULE__, :CUDA) : nothing
end

"""
    pulse_gpu_count() -> Int

Number of functional CUDA devices this pulse stack will use: `0` if CUDA
is missing/unusable, otherwise `min(8, ndevices)`.
"""
function pulse_gpu_count()
    C = _cuda_mod()
    C === nothing && return 0
    try
        C.functional() || return 0
        n = length(C.devices())
        return n < 1 ? 0 : min(_PULSE_MAX_GPUS, n)
    catch
        return 0
    end
end

function _pulse_gpu_devices()
    n = pulse_gpu_count()
    n == 0 && return Any[]
    C = _cuda_mod()
    return collect(C.devices())[1:n]
end

function _resolve_compute(compute::Symbol, M::Integer)
    compute === :cpu && return :cpu
    (compute === :gpu || compute === :auto) || error(
        "compute must be :auto, :cpu, or :gpu, got $(repr(compute))."
    )
    n = pulse_gpu_count()
    if n == 0
        compute === :gpu && error(
            "compute=:gpu requested but no functional CUDA device was found."
        )
        return :cpu
    end
    compute === :auto && M < _PULSE_GPU_MIN_M && return :cpu
    return :gpu
end

function _ode_workspace_bytes(M::Integer, ::Type{T}) where {T}
    return state_length_1st_order(M) * sizeof(T) * 12
end

function _gpu_free_bytes()
    C = _cuda_mod()
    C === nothing && return 0
    try
        return Int(C.free_memory())
    catch
        return 0
    end
end

function _reclaim_current_gpu_memory()
    C = _cuda_mod()
    C === nothing && return nothing
    try
        C.functional() || return nothing
        GC.gc(false)
        C.reclaim()
    catch
    end
    return nothing
end

"""
    _reclaim_gpu_memory()

`GC.gc(false)` then `CUDA.reclaim()` on every pulse device. Call only
when no pulse ODE is in flight on those devices (e.g. after
`_run_pulse_jobs!` returns). Mid-job per-device cleanup uses
[`_reclaim_current_gpu_memory`](@ref) so a sibling GPU is not touched.
"""
function _reclaim_gpu_memory()
    C = _cuda_mod()
    C === nothing && return nothing
    try
        C.functional() || return nothing
        GC.gc(false)
        devices = _pulse_gpu_devices()
        if isempty(devices)
            C.reclaim()
            return nothing
        end
        prev = C.device()
        try
            for dev in devices
                C.device!(dev)
                C.reclaim()
            end
        finally
            C.device!(prev)
        end
    catch
    end
    return nothing
end

function _log_pulse_compute_once(msg::AbstractString)
    if !Threads.atomic_cas!(_PULSE_GPU_LOGGED, false, true)
        println(msg)
    end
    return nothing
end

function _maybe_cuarray(x, compute::Symbol)
    compute === :gpu || return x
    C = _cuda_mod()
    C === nothing && error("internal: compute=:gpu but CUDA is not loaded.")
    return C.CuArray(x)
end

function _allowscalar_solve(compute::Symbol, f)
    compute === :gpu || return f()
    C = _cuda_mod()
    return C.allowscalar() do
        f()
    end
end

function _assert_ensemble_shapes(d)
    hasproperty(d, :M) || error("derived ensemble `d` is missing field M.")
    M = Int(d.M)
    M >= 1 || error("ensemble size M must be >= 1, got $M.")
    hasproperty(d, :Nj) && hasproperty(d, :delta_b) && hasproperty(d, :g_b) || error(
        "derived ensemble `d` must provide Nj, delta_b, and g_b."
    )
    length(d.Nj) == M || error("Nj length $(length(d.Nj)) != M=$M.")
    length(d.delta_b) == M || error("delta_b length $(length(d.delta_b)) != M=$M.")
    length(d.g_b) == M || error("g_b length $(length(d.g_b)) != M=$M.")
    hasproperty(d, :timespan) || error("derived ensemble `d` is missing timespan.")
    length(d.timespan) == 2 || error("timespan must have length 2, got $(length(d.timespan)).")
    return M
end

function _assert_state_shapes(Sp, Sz, Nj, M::Integer, fname::AbstractString)
    length(Sp) == M || error("$fname: Sp length $(length(Sp)) != M=$M.")
    length(Sz) == M || error("$fname: Sz length $(length(Sz)) != M=$M.")
    length(Nj) == M || error("$fname: Nj length $(length(Nj)) != M=$M.")
    return nothing
end

"""
    _solve_1st_order_ode(u0, p, timespan; save_mode=:final, ...)

Shared Tsit5 (or caller `alg`) integration of `rhs_1st_order!`.
`save_mode=:final` keeps only the last state; `save_mode=:trajectory`
saves at `t_save`. `compute=:gpu` uploads `u0` already as a `CuArray`
(the caller is responsible for that) and wraps the solve in
`CUDA.allowscalar` -- the production 1st-order solver's own pattern.
"""
function _solve_1st_order_ode(
    u0, p, timespan;
    alg=Tsit5(), reltol=1e-8, abstol=1e-8, tstops=Float64[],
    save_mode::Symbol=:final, t_save=nothing, compute::Symbol=:cpu,
)
    (save_mode === :final || save_mode === :trajectory) || error(
        "save_mode must be :final or :trajectory, got $(repr(save_mode))."
    )
    prob = ODEProblem(rhs_1st_order!, u0, timespan, p)
    sol = _allowscalar_solve(compute, () -> begin
        if save_mode === :final
            solve(prob, alg; reltol=reltol, abstol=abstol, tstops=tstops,
                  save_everystep=false, save_start=false)
        else
            t_save === nothing && error("save_mode=:trajectory requires t_save.")
            solve(prob, alg; reltol=reltol, abstol=abstol, tstops=tstops, saveat=t_save)
        end
    end)
    _successful_solve(sol) || throw(PulseSolveFailed(sol.retcode))
    return sol
end

"""
    _run_sim_1st_order_from_u0(u0_host, E_of_t, d; save_mode=:final, compute=:cpu, ...)

Host `u0` → optional GPU upload → [`_solve_1st_order_ode`](@ref) → CPU
arrays. Ensemble vectors (`delta_b`, `g_b`) are uploaded with `u0` so the
RHS broadcasts stay on one device. GPU buffers are dropped on the way
out; the caller decides when to [`_reclaim_gpu_memory`](@ref).
"""
function _run_sim_1st_order_from_u0(
    u0_host::AbstractVector, E_of_t, d;
    alg=Tsit5(), reltol=1e-8, abstol=1e-8, tstops=Float64[],
    save_mode::Symbol=:final, t_save=nothing, compute::Symbol=:cpu,
)
    M = _assert_ensemble_shapes(d)
    length(u0_host) == state_length_1st_order(M) || error(
        "u0 has length $(length(u0_host)), expected state_length_1st_order($M) = $(state_length_1st_order(M))."
    )
    u0 = _maybe_cuarray(u0_host, compute)
    delta_b = _maybe_cuarray(collect(d.delta_b), compute)
    g_b = _maybe_cuarray(collect(d.g_b), compute)
    p = (d.delta0, d.kappa_e, d.kappa_i, delta_b, g_b, M, E_of_t)
    sol = nothing
    try
        sol = _solve_1st_order_ode(
            u0, p, d.timespan;
            alg=alg, reltol=reltol, abstol=abstol, tstops=tstops,
            save_mode=save_mode, t_save=t_save, compute=compute,
        )
        if save_mode === :final
            u_end = Array(sol.u[end])
            a, Sp, Sz = unpack_state_1st_order_u(u_end, M)
            _assert_state_shapes(Sp, Sz, d.Nj, M, "_run_sim_1st_order_from_u0")
            return a, collect(Sp), collect(Sz)
        end
        Nt = length(sol.t)
        a = Vector{eltype(sol.u[1])}(undef, Nt)
        Sp = Matrix{eltype(sol.u[1])}(undef, Nt, M)
        Sz = Matrix{eltype(sol.u[1])}(undef, Nt, M)
        @inbounds for i in 1:Nt
            ui = Array(sol.u[i])
            ai, Spi, Szi = unpack_state_1st_order_u(ui, M)
            a[i] = ai
            Sp[i, :] .= Spi
            Sz[i, :] .= Szi
        end
        return sol.t, a, Sp, Sz
    finally
        u0 = nothing
        delta_b = nothing
        g_b = nothing
        p = nothing
        sol = nothing
        # Dual GPU keeps the CUDA pool for the next Dual chunk/IC on this
        # device. Primal GPU solves (jld2, Float64 pulse_cost) have already
        # copied results to the host.
        if compute === :gpu && !(eltype(u0_host) <: ForwardDiff.Dual)
            _reclaim_current_gpu_memory()
        end
    end
end

function _split_index_ranges(n::Integer, nparts::Integer)
    n >= 1 || error("n must be >= 1, got $n.")
    nparts = max(1, min(Int(nparts), n))
    base, rem = divrem(n, nparts)
    ranges = UnitRange{Int}[]
    start = 1
    for p in 1:nparts
        len = base + (p <= rem ? 1 : 0)
        push!(ranges, start:(start + len - 1))
        start += len
    end
    start == n + 1 || error("internal: index split of 1:$n into $nparts parts did not cover the range.")
    return ranges
end

# Run independent jobs with at most one Dual-ODE in flight per GPU.
# `f(job, dev)` must bind the device itself if it launches CUDA work;
# `dev` is `nothing` on the CPU path.
function _run_pulse_jobs!(jobs, f)
    njob = length(jobs)
    njob == 0 && return nothing
    devices = _pulse_gpu_devices()
    if isempty(devices)
        Threads.@threads for i in 1:njob
            f(jobs[i], nothing)
        end
        return nothing
    end
    q = Channel{Any}(length(devices))
    for dev in devices
        put!(q, dev)
    end
    @sync for job in jobs
        Threads.@spawn begin
            dev = take!(q)
            try
                C = _cuda_mod()
                C.device!(dev)
                f(job, dev)
            finally
                put!(q, dev)
            end
        end
    end
    return nothing
end

"""
    run_sim_1st_order_pure(u, pulse::CompositePulse, d; signal_E_of_t=_zero_drive, initial_condition=:ground, alg=Tsit5(), reltol=1e-8, abstol=1e-8, compute=:auto) -> (a, Sp, Sz, Nj)

Pure, ForwardDiff-differentiable analogue of `run_sim_1st_order`:
builds the CONTROL drive from the composite pulse `u` ([`build_E_of_t`](@ref)),
adds a FIXED background `signal_E_of_t(t)` on top of it (any `t -> Complex`
callable -- e.g. [`build_signal_E_of_t`](@ref)'s output when driving this
from a loaded .jld2 run; defaults to [`_zero_drive`](@ref), i.e. no
signal at all), and integrates the SAME `rhs_1st_order!` the GPU
production solver uses (completely unmodified) over `d.timespan`,
returning the final `(a, Sp, Sz)` state at `t1` plus `d.Nj` (handed back
so callers can build population-weighted ensemble metrics without
re-deriving the ensemble). `signal_E_of_t` is a plain closure captured by
value here, NOT part of `u` -- it is therefore structurally impossible
for `ForwardDiff.gradient(uu -> ..., u)` to ever differentiate through it,
which is what guarantees an optimisation built on top of this only ever
touches the control pulse (see `optimise_control_pulse_from_jld2` in
jld2_pulse_loader.jl). No callback, no file I/O. `d` is
`prepare_derived(CONFIG)`'s own return value, exactly what
`run_sim_1st_order` itself builds internally, just constructed once by
the caller and reused across many calls (e.g. across an outer
optimisation loop) instead of rebuilt from `CONFIG` on every call.

`compute=:auto` (default) runs the Dual/primal ODE on a CUDA device when
one is functional AND the ensemble is large enough (`M >= 256`) that
kernel launch overhead is not the dominant cost; otherwise it stays on
the host `Vector` path this file originally used (and that
`test/runtests.jl` still exercises). Pass `compute=:cpu` to force the
host path, `compute=:gpu` to require a device. The equations, stepper
(`Tsit5` unless `alg` is overridden), and tolerances are identical
either way -- only the array backend changes.
"""
function run_sim_1st_order_pure(
    u::AbstractVector,
    pulse::CompositePulse,
    d;
    signal_E_of_t = _zero_drive,
    initial_condition::Symbol=:ground,
    alg=Tsit5(),
    reltol=1e-8,
    abstol=1e-8,
    compute::Symbol=:auto,
)
    M = _assert_ensemble_shapes(d)
    length(u) == n_params(pulse) || error(
        "run_sim_1st_order_pure: u has length $(length(u)), but this CompositePulse " *
        "(k=$(pulse.k), n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)) needs $(n_params(pulse))."
    )
    T = eltype(u)
    compute_eff = _resolve_compute(compute, M)
    if compute_eff === :gpu && T <: ForwardDiff.Dual && _GPU_DUAL_OK[] === false
        compute_eff = :cpu
    end
    if compute_eff === :gpu
        need = _ode_workspace_bytes(M, Complex{T})
        free = _gpu_free_bytes()
        if free > 0 && need > free ÷ 2
            compute === :gpu && error(
                "GPU 1st-order solve needs ~$need bytes of Dual/state workspace " *
                "but only $free bytes are free on the current device."
            )
            compute_eff = :cpu
        end
    end
    if compute_eff === :gpu
        _log_pulse_compute_once(
            "pulse ODE: GPU backend on $(pulse_gpu_count()) device(s) " *
            "(M=$M, eltype=$T, rhs_1st_order!/Tsit5 unchanged)"
        )
    end

    control_E_of_t = build_E_of_t(pulse, u)
    E_of_t(t) = control_E_of_t(t) + signal_E_of_t(t)
    u0 = build_u0_1st_order_cpu(M, d.Nj, T, initial_condition)

    # Historical note (the underlying issue is now fixed a different way,
    # see below, but this is why `tstops` are still here): each sub-pulse
    # used to have a HARD (exact-silence) edge at its own t_start/t_end --
    # a genuine discontinuity in the RHS at a time that MOVES as the
    # differentiated parameters (raw_gap/raw_dur) move it, which pinning
    # every known edge as an explicit `tstop` (standard SciML practice for
    # parameter-dependent switching times) mostly, but not entirely, fixed
    # -- raw_gap (which translates BOTH t_start and t_end together) still
    # disagreed with a finite-difference cross-check even with `tstops`
    # set, because `tstops` only forces the SOLVER to land on the switch;
    # it does nothing about the missing jump/transversality term a
    # parameter-dependent VALUE discontinuity requires in the trajectory's
    # OWN sensitivity (see CompositePulse's module docstring in
    # composite_pulse.jl for the full explanation). That's fixed properly
    # now: CompositePulse.build_E_of_t multiplies the amplitude by a C^∞
    # taper window that matches every derivative of the identically-zero
    # silence region at the edges, removing the discontinuity itself
    # (verified: raw_gap now agrees with finite differences to <0.0001%,
    # and in fact `tstops` are no longer even required for correctness at
    # all once the window is in place -- also verified directly). They're
    # kept here anyway as a harmless hint for the adaptive stepper near
    # each taper region, not because correctness depends on them anymore.
    t_start, t_end, _, _, _ = decode(pulse, u)
    tstops = ForwardDiff.value.(vcat(t_start, t_end))

    try
        a, Sp, Sz = _run_sim_1st_order_from_u0(
            u0, E_of_t, d;
            alg=alg, reltol=reltol, abstol=abstol, tstops=tstops,
            save_mode=:final, compute=compute_eff,
        )
        if compute_eff === :gpu && T <: ForwardDiff.Dual
            _GPU_DUAL_OK[] = true
        end
        return a, Sp, Sz, d.Nj
    catch e
        e isa PulseSolveFailed && rethrow()
        if compute_eff === :gpu && T <: ForwardDiff.Dual && _GPU_DUAL_OK[] !== true
            _GPU_DUAL_OK[] = false
            _reclaim_current_gpu_memory()
            @warn "GPU Dual 1st-order ODE failed; falling back to the host Dual path" exception = e
            a, Sp, Sz = _run_sim_1st_order_from_u0(
                u0, E_of_t, d;
                alg=alg, reltol=reltol, abstol=abstol, tstops=tstops,
                save_mode=:final, compute=:cpu,
            )
            return a, Sp, Sz, d.Nj
        end
        rethrow()
    end
end

function _forbid_initial_condition(kwargs)
    :initial_condition in keys(kwargs) && error(
        "the cost fixes its own initial conditions (:ground and/or :weak, see `track`). " *
        "Do not pass initial_condition into pulse_cost / pulse_metrics / optimise_composite_pulse."
    )
    return nothing
end

"""
    _assert_track(track::Symbol) -> Symbol

`track` selects how `inversion` and `silencing` are obtained:

  * `:weak` (default) -- ONE `:weak` solve; `inversion` is read from THAT
    solve's own `Sz` too. Halves the ODE cost. Justified because the
    `:weak` seed perturbs `Sz` only at O(ε) (ε = `_WEAK_SEED` = 1e-3), so
    `inversion` on `:weak` ≈ `inversion` on `:ground` to ~1e-3 (measured
    4.2e-5 on the toy `hs1` config). This is the default because the O(ε)
    bias is negligible for realistic fidelity targets and the winner
    re-check below restores the canonical `:ground` inversion anyway.
  * `:dual` -- OPT-IN ONLY (pass `track=:dual` explicitly): two ODE solves
    per cost evaluation: `:ground` (`Sp=0`) gives `inversion`, the
    weak-excitation `:weak` seed (`Sp=ε·Nj/2`) gives `silencing`/
    `coherence`/`field_amp`/`weak_seed_retention`. Nothing in this package
    selects `:dual` on its own -- it is never the fallback for a missing
    or unrecognised keyword.

The single-track gradient (ForwardDiff, threaded-Jacobian, AND the
1-forward/2-reverse adjoint via [`_adjoint_track_multi`](@ref)) is the
EXACT gradient of `pulse_cost(u; track=:weak)` -- verified roundoff-
identical across all three backends. The only approximation is that
`pulse_cost(u; track=:weak)` itself ≈ `pulse_cost(u; track=:dual)` to
O(ε); a `track=:weak` optimum (the default) is therefore an
O(ε)-approximate stationary point of the `:dual` objective.
[`optimise_composite_pulse`](@ref) /
[`optimise_composite_pulse_rjmcmc`](@ref) do this re-check automatically:
after the search they spend ONE `:ground` solve of the winner and record
`final_inversion_ground` (the canonical value) and `final_inv_gap`
(`inversion:weak - inversion:ground`) in `optimizer_settings`, so any
saved run log carries the true `:ground` inversion. For an ad-hoc `u`,
`pulse_metrics(u, pulse, d; track=:dual)[1]` gives the same number.
"""
function _assert_track(track::Symbol)
    track === :weak || track === :dual || error(
        "track must be :weak (the default -- a single :weak solve, inversion " *
        "read from it too) or :dual (opt-in -- separate :ground + :weak " *
        "solves), got $(repr(track))."
    )
    return track
end

function _solver_kwargs(kwargs)
    nt = NamedTuple(kwargs)
    for k in (:compute, :threaded_grad, :grad_mode, :checkpoint_stride, :use_checkpoints)
        nt = haskey(nt, k) ? Base.structdiff(nt, NamedTuple{(k,)}) : nt
    end
    return nt
end

"""
    _weighted_inversion(Sz, g_b, Nj, ::Type{T}) -> T

Paper-aligned ensemble inversion `I ∈ [0, 1]` from the `:ground` track's
end-state `Sz = <S^z>` (App. H):

    ι_j = clamp(real(Sz_j) / (Nj_j/2), -1, 1),   I_j = (1 + ι_j)/2 ∈ [0, 1]
    I   = Σ_j Nj_j g_j² I_j  /  Σ_j Nj_j g_j²

Each bin carries the bright-mode / cooperativity weight `|P(g)|² ∝ N g²`
(the same `g2_avg`-style weighting used elsewhere in this package -- see
`ensemble.jl`), NOT plain `Nj`: only the part of `ρ(g)` that overlaps the
bright mode `P(g)` counts, so weak-`g` Landau–Zener failures that do not
sit in the bright mode no longer pollute `I`. Same `1e-30` denominator
epsilon as the silencing metric (`abs`/division through `Dual(0)`).
"""
function _weighted_inversion(Sz, g_b, Nj, ::Type{T}) where {T}
    length(Sz) == length(Nj) == length(g_b) || error(
        "_weighted_inversion: Sz/g_b/Nj lengths $(length(Sz))/$(length(g_b))/$(length(Nj)) must match."
    )
    w = Nj .* abs2.(g_b)                            # paper |P(g)|² ∝ N g² (App. H)
    weight = w ./ (sum(w) + T(1e-30))              # normalise first (well-conditioned; no ~Σ(Nj g²) intermediate)
    Sz_fraction = real.(Sz) ./ (Nj ./ 2 .+ 1e-30)
    Ij = clamp.((Sz_fraction .+ 1) ./ 2, zero(T), one(T))
    return sum(weight .* Ij)
end

"""
    _weak_seed_retention(Sp, g_b, Nj, delta_b, ::Type{T}; eps_seed=_WEAK_SEED) -> T

UNCLAMPED cooperativity-weighted, per-frequency-slice, ε-normalised
MAGNITUDE ratio of the `:weak` track's end-state -- the same construction
as [`_weighted_silencing_factor`](@ref) but with `|Sp_j|` taken INSIDE the
per-slice sum (no phase cancellation), so it is the triangle-inequality
upper bound on that factor's aggregate:

    B(ω)  = {j : ω_j = ω}                             (see _frequency_slice_indices)
    C(ω)  = ( Σ_{j∈B(ω)} g_j² |Sp_j| ) / ( ε · Σ_{j∈B(ω)} g_j² Nj_j/2 )
    n(ω)  = Σ_{j∈B(ω)} Nj_j g_j²
    ret   = Σ_ω n(ω) C(ω) / Σ_ω n(ω)                  (NOT clamped)

Interpretation -- a weak-excitation VALIDITY check for the paper's App. B
single-TLS-propagator picture:

  * `ret ≈ 1`  -- every bin stayed at its `Sp_j(0) = ε·Nj_j/2` seed; the
    pulse acts as a (near) phase-only operation, so the
    `_weighted_silencing_factor` / `_weighted_coherence` initial-overlap
    normalisation is exact and `|F|_⋆` is fully trustworthy.
  * `ret ≫ 1`  (up to ~`1/ε`) -- the pulse has PUMPED transverse coherence
    far above the seed (it is not operating in the Holstein–Primakoff
    regime). `|F|_⋆` and `coherence` are still bounded in `[0, 1]` by their
    own clamps, but their normalisation is being stretched by this factor
    -- read them with care.
  * `ret < 1`  -- net transverse decay below the seed.

Diagnostic only; never part of `pulse_cost`'s objective. Same `1e-30`
epsilons as the silencing factor.
"""
function _weak_seed_retention(Sp::AbstractVector, g_b::AbstractVector, Nj::AbstractVector,
        delta_b::AbstractVector, ::Type{T}; eps_seed::Real=_WEAK_SEED) where {T}
    length(Sp) == length(g_b) == length(Nj) == length(delta_b) || error(
        "_weak_seed_retention: Sp/g_b/Nj/delta_b lengths " *
        "$(length(Sp))/$(length(g_b))/$(length(Nj))/$(length(delta_b)) must match."
    )
    eps_seed > 0 || error("_weak_seed_retention: eps_seed must be > 0, got $eps_seed.")
    slices = _frequency_slice_indices(delta_b)
    num_acc = zero(T)     # Σ_ω n(ω) C(ω)
    den_acc = zero(T)     # Σ_ω n(ω)
    for idx in slices
        wg = abs2.(g_b[idx])                                             # g_j²
        C_num = sum(wg .* sqrt.(abs2.(Sp[idx]) .+ 1e-30))                # Σ g² |Sp|  (magnitudes)
        C_den = sum(wg .* (convert(T, eps_seed) .* (Nj[idx] ./ 2)))      # ε Σ g² Nj/2
        n_omega = sum(Nj[idx] .* wg)                                     # n(ω) = Σ N g²
        num_acc += n_omega * (C_num / (C_den + 1e-30))
        den_acc += n_omega
    end
    return num_acc / (den_acc + 1e-30)
end

"""
    _weighted_coherence(Sp, g_b, Nj, delta_b, ::Type{T}; eps_seed=_WEAK_SEED) -> T

`clamp([`_weak_seed_retention`](@ref), 0, 1)` -- the per-frequency-slice
MAGNITUDE companion to [`_weighted_silencing_factor`](@ref), on the SAME
`:weak` solve, with the SAME `Nj g²` slice weighting and the SAME `ε`
initial-overlap normalisation. By the triangle inequality `|Σ z| ≤ Σ|z|`
applied slice by slice, `coherence ≥ silencing` always, so

    coherence − silencing  ∈  [0, coherence]

is exactly the fraction of the surviving transverse magnitude that was
lost to `g`-space PHASE SPREAD rather than to decoherence:

  * `silencing ≈ coherence ≈ 1`  -- RASE / ACE (pure `ω²/κ` chirp).
  * `silencing ≈ 0`, `coherence ≈ 1`  -- ROSE ideal: magnitude intact,
    the collective sum killed purely by destructive interference.
  * `silencing ≈ 0`, `coherence ≈ 0`  -- decoherence (`|Sp_j| → 0`).

Diagnostic only; never fed into [`pulse_cost`](@ref)'s objective. See
[`_weak_seed_retention`](@ref) for the un-clamped value (a validity check
on the weak-excitation assumption itself).
"""
function _weighted_coherence(Sp::AbstractVector, g_b::AbstractVector, Nj::AbstractVector,
        delta_b::AbstractVector, ::Type{T}; eps_seed::Real=_WEAK_SEED) where {T}
    return clamp(_weak_seed_retention(Sp, g_b, Nj, delta_b, T; eps_seed=eps_seed),
                 zero(T), one(T))
end

"""
    _frequency_slice_indices(delta_b) -> Vector{Vector{Int}}

Groups bin indices into frequency slices `B(ω) = {j : ω_j = ω}` by shared
detuning `delta_b[j]`. On this package's product `(M_delta × M_g)` mesh
(`build_2d_bins` in ensemble.jl) each distinct `delta_b_1d` value collects
its whole `g`-column into one slice; with `M_g == 1` every slice is a
single bin. `delta_b` is always a plain `Float64` ensemble vector (never a
differentiated quantity), so this grouping is a constant of the ensemble,
recomputed cheaply per call rather than cached.

Used by [`_weighted_silencing_factor`](@ref) so `F` is built INSIDE
frequency slices and never as one complex sum over the whole `(g, ω)` mesh
-- a pure `ω²/κ` chirp then leaves every per-slice `|F(ω)| = 1` (ACE), and
only a genuinely `g`-dependent phase spread drives `|F| → 0` (paper Eq. 5 /
A.132; the single-global-sum form mixes the ACE chirp into `|F|` and is
explicitly NOT what the paper computes).
"""
function _frequency_slice_indices(delta_b::AbstractVector)
    slices = Vector{Int}[]
    slot = Dict{Float64,Int}()
    @inbounds for j in eachindex(delta_b)
        key = Float64(delta_b[j])
        s = get(slot, key, 0)
        if s == 0
            push!(slices, Int[j])
            slot[key] = length(slices)
        else
            push!(slices[s], j)
        end
    end
    return slices
end

"""
    _weighted_silencing_factor(Sp, g_b, Nj, delta_b, ::Type{T}; eps_seed=_WEAK_SEED) -> T

Paper silencing factor `|F|_⋆ ∈ [0, 1]` (Eq. 5 / A.132 / Eq. 6), built
per frequency slice from the `:weak` track's end-state
`Sp = <S^+>` and reduced to a scalar by the `⟨|F(ω)|⟩` average:

    B(ω)     = {j : ω_j = ω}                                   (see _frequency_slice_indices)
    F(ω)     = ( Σ_{j∈B(ω)} g_j² Sp_j )  /  ( Σ_{j∈B(ω)} g_j² Sp_j(0) )
    n(ω)     = Σ_{j∈B(ω)} Nj_j g_j²                            (bright-mode density)
    |F|_⋆    = Σ_ω n(ω) |F(ω)|  /  Σ_ω n(ω)

Key points, all from "inversion and silencing":

  * **Per frequency slice, never one sum over all bins.** A pure `ω²/κ`
    chirp is a common phase to every bin in a slice, so `|F(ω)| = 1`
    (ACE energy conserved); only a `g`-dependent phase spread WITHIN a
    slice drives `|F(ω)| → 0`. One global `(g, ω)` sum would fold the
    chirp into `|F|`.
  * **Weight on the raw `Sp_j` is `g_j²`, not `Nj_j g_j²`.** `Sp_j` is
    already extensive (`∝ Nj_j`); the extra `Nj_j` the old global-sum
    form carried in both numerator and denominator is the one the paper
    says "must go". After the ratio, the `Σ g_j² Sp_j(0) = ε Σ g_j² Nj_j/2`
    denominator makes this a `Nj_j g_j²`-weighted phasor average of the
    per-bin coherence, matching the paper's `Σ Nj g² e^{iφ}` form.
  * **Denominator is the INITIAL per-slice overlap** `Σ g_j² Sp_j(0)`,
    with `Sp_j(0) = ε·Nj_j/2` the [`_WEAK_SEED`](@ref) seed
    ([`build_u0_1st_order_cpu`](@ref)). The literal Eq.-6 denominator
    `Σ g_j² (Nj_j/2)` assumes `|Sp_j| = Nj_j/2` (a fully coherent packet);
    our seed is `ε ×` that, so an un-rescaled `F(ω)` would be `O(ε)` even
    when every bin stays perfectly aligned. Dividing by the actual initial
    overlap restores `|F|_⋆ → 1` for an ideal ACE / RASE phase-only pulse
    and `→ 0` for full ROSE silencing, so `target_F ∈ {0, 1}` keeps its
    meaning. Robust to a non-uniform seed (pass a matching `Sp0` implicitly
    by changing `eps_seed`); it is a fixed, `u`-independent constant.
  * **`clamp` is applied to `|F|_⋆` AFTER the ratio**, on the magnitude --
    never to `Sp_j` components before summing, and NOT as a per-bin unit
    phasor `Sp_j/|Sp_j|` (whose derivative blows up like `1/√eps` for any
    bin passing near `Sp_j = 0`, routine across a wide ensemble).

The `g`-linear, field-amplitude-proportional companion diagnostic is
[`_weighted_field_amplitude`](@ref) (never fed into `pulse_cost`).
"""
function _weighted_silencing_factor(Sp::AbstractVector, g_b::AbstractVector, Nj::AbstractVector,
        delta_b::AbstractVector, ::Type{T}; eps_seed::Real=_WEAK_SEED) where {T}
    length(Sp) == length(g_b) == length(Nj) == length(delta_b) || error(
        "_weighted_silencing_factor: Sp/g_b/Nj/delta_b lengths " *
        "$(length(Sp))/$(length(g_b))/$(length(Nj))/$(length(delta_b)) must match."
    )
    eps_seed > 0 || error("_weighted_silencing_factor: eps_seed must be > 0, got $eps_seed.")
    slices = _frequency_slice_indices(delta_b)
    num_acc = zero(T)     # Σ_ω n(ω) |F(ω)|
    den_acc = zero(T)     # Σ_ω n(ω)
    for idx in slices
        wg = abs2.(g_b[idx])                                   # g_j²  (raw S₊ weight)
        F_num = sum(wg .* Sp[idx])                             # complex, Σ g² S₊(final)
        F_den = sum(wg .* (convert(T, eps_seed) .* (Nj[idx] ./ 2)))  # Σ g² S₊(0) = ε Σ g² Nj/2
        F_omega = F_num / (F_den + 1e-30)
        abs_F = sqrt(abs2(F_omega) + 1e-30)
        n_omega = sum(Nj[idx] .* wg)                           # bright-mode density n(ω) = Σ N g²
        num_acc += n_omega * abs_F
        den_acc += n_omega
    end
    return clamp(num_acc / (den_acc + 1e-30), zero(T), one(T))
end

"""
    _weighted_field_amplitude(Sp, g_b, Nj, ::Type{T}) -> T

DIAGNOSTIC-ONLY companion to [`_weighted_silencing_factor`](@ref): weights
the collective sum by `g_j` ALONE (not `Nj_j g_j`) -- `Sp_j` is already
EXTENSIVE (bounded by `Nj_j/2`, not by `1`), so `Σ_j g_j Sp_j` (this
function's numerator) is exactly, term-for-term, the Maxwell-Bloch cavity
source `Σ_j g_j Sp_j` in `rhs_1st_order!` (`ȧ ⊃ -i Σ_j g_j conj(Sp_j)`) --
no extra `Nj_j` factor belongs in the weight, since `Sp_j` itself already
carries its bin's own population scaling. (An earlier version of this
function used `weight = Nj_j g_j`, i.e. `Σ_j Nj_j g_j Sp_j = Σ_j Nj_j² g_j
s_j` for intensive per-bin coherence `s_j = Sp_j/Nj_j` -- quadratic in
`Nj_j`, not linear, so it did NOT actually mirror the source term despite
the docstring's claim at the time: verified directly, e.g. two bins with
equal `g` and `Nj=[10,1000]` give a source-term ratio of exactly `100`
between "only the small bin coherent" and "only the large bin coherent",
but the old `Nj_j g_j`-weighted metric gave `10000` for the same
comparison -- corrected here.)

Normalised to `[0, 1]` by the triangle-inequality bound `|Σ_j g_j Sp_j| <=
Σ_j |g_j| |Sp_j| <= Σ_j |g_j| (ε·Nj_j/2)` (the denominator, using `|g_j|`
since a signed/complex coupling's SIGN cannot help the bound, and the
weak-excitation seed scale `ε = eps_seed` = [`_WEAK_SEED`](@ref)
so an undisturbed seed reads `1`), same `1e-30` epsilon pattern for
`Sp==0`/`g==0` degenerate cases. This is a SINGLE global sum by design --
unlike `|F|`, the radiated field `Σ_j g_j Sp_j` genuinely is one sum over
the whole ensemble (it IS the cavity source term), so no per-frequency
slicing applies here.

Never fed into [`pulse_cost`](@ref)'s optimised objective or its gradient
backends -- recorded purely so callers/logs can compare it against the
collective `|F|` actually being optimised, exactly the same role
`_weighted_coherence` already plays relative to `_weighted_silencing_factor`.
"""
function _weighted_field_amplitude(Sp::AbstractVector, g_b::AbstractVector, Nj::AbstractVector, ::Type{T};
        eps_seed::Real=_WEAK_SEED) where {T}
    length(Sp) == length(g_b) == length(Nj) || error(
        "_weighted_field_amplitude: Sp/g_b/Nj lengths $(length(Sp))/$(length(g_b))/$(length(Nj)) must match."
    )
    max_field_sum = convert(T, eps_seed) * sum(abs.(g_b) .* Nj) / 2
    E_complex = sum(g_b .* Sp) / (max_field_sum + 1e-30)
    abs_E = sqrt(abs2(E_complex) + 1e-30)
    return clamp(abs_E, zero(T), one(T))
end

"""
    pulse_metrics(u, pulse, d; kwargs...)
        -> (inversion, silencing, coherence, field_amp, weak_seed_retention)

Dual-trajectory metrics used by [`pulse_cost`](@ref). `inversion` and
`silencing` are both in `[0, 1]`, higher = better, and are NOT two
coordinates of one Bloch vector:

- `inversion`: from `:ground` (`Sz = -Nj/2`, `Sp = 0`). Bright-mode /
  cooperativity (`Nj g²`) weighted mean of `real(Sz)/(Nj/2)` mapped from
  `[-1, 1]` to `[0, 1]` (paper App. H -- see [`_weighted_inversion`](@ref)).
  A π pulse scores near 1.
- `silencing`: from the `:weak` track (`Sz = -Nj/2`,
  `Sp = ε·Nj/2`, `ε = _WEAK_SEED`). Paper silencing factor
  `|F|_⋆ = ⟨|F(ω)|⟩` built PER FREQUENCY SLICE (see
  [`_weighted_silencing_factor`](@ref)), NOT one sum over all bins and NOT
  a per-bin magnitude average. `F=1`: every bin phase-aligned within its
  slice, a pure `ω²/κ` chirp (RASE-style revival / ACE). `F=0`: `g`-space
  phase spread cancels the collective sum (ROSE-style silencing) -- these
  are NOT distinguishable from `|F|` alone, by design.
- `coherence`: the per-frequency-slice MAGNITUDE companion to `silencing`
  (see [`_weighted_coherence`](@ref)) -- SAME `:weak` solve, SAME `Nj g²`
  slice weighting, SAME `ε` initial-overlap normalisation, but with
  `|Sp_j|` inside the per-slice sum. `coherence ≥ silencing` always, and
  `coherence − silencing` is the fraction of surviving magnitude lost to
  `g`-space PHASE SPREAD (vs. decoherence). DIAGNOSTIC ONLY -- never fed
  into [`pulse_cost`](@ref)'s objective.
- `field_amp`: the linearly-`g`-weighted counterpart of `silencing` (see
  [`_weighted_field_amplitude`](@ref)), from the SAME `:weak` solve.
  DIAGNOSTIC ONLY -- proportional to the actual emitted cavity field
  amplitude, unlike `silencing`'s cooperativity weighting.
- `weak_seed_retention`: the UN-clamped value `coherence` clamps (see
  [`_weak_seed_retention`](@ref)) -- a validity check on the App. B
  weak-excitation assumption. `≈ 1`: pulse is (near) phase-only, `|F|_⋆`
  normalisation exact. `≫ 1`: pulse pumped coherence above the `ε` seed,
  `|F|_⋆`/`coherence` normalisation stretched -- interpret with care.
  DIAGNOSTIC ONLY.

Do not pass `initial_condition` — the ICs are fixed here. `track`
(`:weak` default / `:dual` opt-in only, see [`_assert_track`](@ref))
chooses one or two solves. Solver kwargs (`signal_E_of_t`, `reltol`, ...)
are forwarded.
"""
function pulse_metrics(u::AbstractVector, pulse::CompositePulse, d;
                        compute::Symbol=:auto, track::Symbol=:weak, kwargs...)
    _forbid_initial_condition(kwargs)
    _assert_track(track)
    T = eltype(u)
    sk = _solver_kwargs(kwargs)
    if track === :dual
        _, _, Sz, Nj = run_sim_1st_order_pure(u, pulse, d; compute=compute, sk..., initial_condition=:ground)
        inversion = _weighted_inversion(Sz, d.g_b, Nj, T)
        _, Sp, _, Nj_eq = run_sim_1st_order_pure(u, pulse, d; compute=compute, sk..., initial_condition=:weak)
    else
        _, Sp, Sz_w, Nj_eq = run_sim_1st_order_pure(u, pulse, d; compute=compute, sk..., initial_condition=:weak)
        inversion = _weighted_inversion(Sz_w, d.g_b, Nj_eq, T)
    end
    silencing = _weighted_silencing_factor(Sp, d.g_b, Nj_eq, d.delta_b, T)
    coherence = _weighted_coherence(Sp, d.g_b, Nj_eq, d.delta_b, T)
    field_amp = _weighted_field_amplitude(Sp, d.g_b, Nj_eq, T)
    weak_seed_retention = _weak_seed_retention(Sp, d.g_b, Nj_eq, d.delta_b, T)
    return inversion, silencing, coherence, field_amp, weak_seed_retention
end

# ============================================================
# Squared-hinge exterior penalty (I_min/kappa_I, S_min/kappa_S): an
# AD-integrated exact-penalty method replacing the earlier per-epoch
# p_exp/q_exp exponent barrier. That exponent barrier had two structural
# problems, independently audited and confirmed against the code (not
# just suspected):
#   (1) gradient starvation: its restoring gradient term was
#       p*inversion^(p-1) (ordinary power rule), which -> 0 as
#       inversion -> 0 for ANY finite p>1 -- so the "barrier" went SLACK
#       exactly where inversion had collapsed hardest, the opposite of
#       what a barrier should do. No choice of the old max_exp/lambda
#       could fix this: x^(p-1) -> 0 as x -> 0 is unavoidable for finite
#       p, not a tuning artifact.
#   (2) phase lag: p_exp/q_exp were computed once per epoch from
#       `last_good_aux` -- the PREVIOUS accepted point's detached
#       Float64 metrics, not the live AD state being differentiated THIS
#       epoch -- so the barrier's shape always reacted one epoch late to
#       inversion/silencing_success changes.
# This penalty fixes both by construction: it is evaluated directly on
# the CURRENT epoch's own inversion/silencing_success (whatever
# Dual/Float64 type they arrive as -- inside pulse_cost's ForwardDiff
# tape for the primal, and via the SAME live inversion/silencing values
# already available in both gradient backends), so there is no
# detach-and-lag step; and its restoring gradient is LINEAR in the
# violation (kappa*(I_min-inversion)), a nonzero, non-vanishing
# coefficient as inversion -> 0 (unlike the old power-law term), tunable
# via kappa_I/kappa_S rather than fixed by a saturating exponent. Both
# tracks (inversion AND silencing_success) are barriered symmetrically,
# each with its own floor/strength pair, matching the ORIGINAL barrier
# feature's own "barrier both tracks" decision -- this is a like-for-like
# replacement of the barrier MECHANISM, not a scope reduction.
#
# Because kappa_I/kappa_S are STATIC (caller-supplied constants, not an
# epoch-varying schedule the way p_exp/q_exp were), physics_cost is the
# SAME formula every epoch regardless of hop/epoch -- there is no longer
# any "reconstitute the barrier's own contribution back to p=q=1" step
# needed in run_local_adam's epoch loop: the penalty is a permanent,
# reported part of the objective, not a gradient-only reshaping that has
# to be corrected back out for best_cost/Metropolis comparisons to stay
# apples-to-apples.
#
# This formula/gradient pair is shared by FOUR independent consumers
# (pulse_cost, _pulse_cost_grad_threaded, pulse_cost_grad_adjoint,
# pulse_cost_on_frozen_mesh) that would otherwise each hand-copy it --
# exactly the kind of drift risk _schedule_shape was factored out to
# avoid for the x_tune schedule.
# ============================================================

"""
    _DEFAULT_PENALTY_MIN, _DEFAULT_PENALTY_KAPPA

Default floor (`0.85`, the same numeric value the earlier exponent
barrier used for its own `barrier_min`) and default penalty strength
(`50.0`) for [`run_local_adam`](@ref)'s `I_min`/`kappa_I`/`S_min`/
`kappa_S` squared-hinge penalty (see [`_fidelity_physics_cost`](@ref)).
Unlike the old barrier's exponents (self-normalising: `x^p ∈ [0,1]`
regardless of `p`), `kappa_I`/`kappa_S` are NOT self-normalising -- the
penalty term is unbounded and its magnitude relative to the `(1-fid)^2`
base term depends entirely on this choice, so `50.0` is a starting
default, not a derived constant; retune per-problem if the penalty
under- or over-dominates the base term near the floor.
"""
const _DEFAULT_PENALTY_MIN = 0.85
const _DEFAULT_PENALTY_KAPPA = 50.0

"""
    _assert_penalty_params(I_min, kappa_I, S_min, kappa_S)

Validates the squared-hinge penalty's own parameters -- NOT validated by
either consumer function itself (mirrors the old barrier's own
`_assert_barrier_exponents`, now this feature's analogous defensive
contract). Two DISTINCT foot-guns this guards against, each confirmed
directly reachable through any public entry point (`pulse_cost`,
`run_local_adam`, `optimise_composite_pulse`/`_rjmcmc`) since `I_min`/
`kappa_I`/`S_min`/`kappa_S` are ordinary caller-facing keywords with no
prior validation anywhere in the call graph:

1. `kappa_I`/`kappa_S < 0`: the penalty's own gradient contribution is
   `-kappa*max(0,floor-x)`, so a NEGATIVE `kappa` flips its sign --
   instead of restoring `x` back toward `floor`, it actively REWARDS `x`
   for being further below `floor` (confirmed: `_fidelity_physics_cost`
   with `kappa_I=-50` gives a cost that gets MORE NEGATIVE as `inversion`
   drops, and a gradient that pushes `inversion` DOWN, not up) -- silent
   misoptimization with no error, not a hypothetical.
2. `I_min`/`S_min` outside `[0,1]`: `inversion`/`silencing_success` are
   both provably in `[0,1]` (see `_weighted_inversion`/
   `_weighted_silencing_factor`'s own clamps; `silencing_success`'s own
   bound holds for any documented `target_F ∈ [0,1]`), so a FLOOR above
   `1` can never be satisfied -- the penalty stays permanently active,
   with a permanently nonzero restoring gradient, even at PERFECT
   fidelity (confirmed: `I_min=1.5` at `inversion=1.0` still gives a
   nonzero penalty and gradient) -- a plausible unit/typo foot-gun (e.g.
   passing a percentage `85` instead of `0.85`) that would otherwise
   silently prevent convergence to the true optimum with no error to
   flag it.
"""
function _assert_penalty_params(I_min, kappa_I, S_min, kappa_S)
    (isfinite(kappa_I) && kappa_I >= 0) || error(
        "_assert_penalty_params: kappa_I must be finite and >= 0, got $kappa_I " *
        "(negative kappa_I would REWARD low inversion instead of penalising it)."
    )
    (isfinite(kappa_S) && kappa_S >= 0) || error(
        "_assert_penalty_params: kappa_S must be finite and >= 0, got $kappa_S " *
        "(negative kappa_S would REWARD low silencing_success instead of penalising it)."
    )
    (isfinite(I_min) && 0 <= I_min <= 1) || error(
        "_assert_penalty_params: I_min must be finite and in [0,1] (inversion's own " *
        "provable range), got $I_min -- a value > 1 can never be satisfied, leaving " *
        "the penalty permanently active even at inversion=1.0."
    )
    (isfinite(S_min) && 0 <= S_min <= 1) || error(
        "_assert_penalty_params: S_min must be finite and in [0,1] (silencing_success's " *
        "own provable range for target_F ∈ [0,1]), got $S_min -- a value > 1 can never " *
        "be satisfied, leaving the penalty permanently active even at silencing_success=1.0."
    )
end

"""
    _fidelity_physics_cost(inversion::T, silencing::T, target_F, I_min, kappa_I, S_min, kappa_S) where {T}
        -> (physics_cost::T, fidelity_phys::T, silencing_success::T)

    silencing_success = 1 - (silencing - target_F)^2
    fidelity_phys      = inversion * silencing_success                       # always the plain product now
    J_base             = (1 - fidelity_phys)^2
    J_pen_I            = 0.5 * kappa_I * max(0, I_min - inversion)^2
    J_pen_S            = 0.5 * kappa_S * max(0, S_min - silencing_success)^2
    physics_cost       = J_base + J_pen_I + J_pen_S

`kappa_I == kappa_S == 0` reproduces the legacy, unbarriered
`(1-inversion*silencing_success)^2` formula BIT-FOR-BIT (`0.5*0*x^2 ==
0.0` exactly in IEEE754 for any finite `x`, no floating-point
cancellation risk). `max(0, ...)` makes each penalty a genuine ONE-SIDED
hinge: zero contribution (and zero gradient) whenever the corresponding
track is already at/above its floor, so this is inert overhead-free
outside the barrier's own activation region. Generic in `T` so the SAME
code serves [`pulse_cost`](@ref)'s `ForwardDiff.Dual` primal (ForwardDiff
differentiates `max`/`^2` through the ordinary chain rule automatically,
so the primal's gradient stays exact without any manual chain rule) AND
the two gradient backends' (`_pulse_cost_grad_threaded`/
`pulse_cost_grad_adjoint`) plain-`Float64` analytical chain rule via
[`_fidelity_gradient_coefficients`](@ref) below.
"""
function _fidelity_physics_cost(inversion::T, silencing::T, target_F, I_min, kappa_I, S_min, kappa_S) where {T}
    _assert_penalty_params(I_min, kappa_I, S_min, kappa_S)
    ss = one(T) - (silencing - convert(T, target_F))^2
    fid = inversion * ss
    J_base = (one(T) - fid)^2

    pen_I_val = inversion < convert(T, I_min) ? (convert(T, I_min) - inversion) : zero(T)
    pen_S_val = ss < convert(T, S_min) ? (convert(T, S_min) - ss) : zero(T)
    J_pen = convert(T, 0.5 * kappa_I) * pen_I_val^2 + convert(T, 0.5 * kappa_S) * pen_S_val^2

    return J_base + J_pen, fid, ss
end

"""
    _fidelity_gradient_coefficients(inversion, silencing_success, fidelity_phys, I_min, kappa_I, S_min, kappa_S)
        -> (coeff_I, coeff_S)

`∂physics_cost/∂inversion` and `∂physics_cost/∂silencing_success` for
[`_fidelity_physics_cost`](@ref), each the sum of the base multiplicative
term (ordinary product-rule/chain-rule derivative of `(1-inversion*
silencing_success)^2`, distributed rather than factored since the penalty
term below does NOT share the base term's own `-2*(1-fid)` factor) and
the squared-hinge penalty's own restoring gradient
(`d/dx[0.5*kappa*max(0,floor-x)^2] = -kappa*max(0,floor-x)`, continuous
at `x==floor`, i.e. `0` exactly there -- a genuine C1 kink, not a
discontinuity). At `kappa_I == kappa_S == 0` these reduce EXACTLY to the
legacy chain rule's own coefficients `(silencing_success, inversion)`.
"""
function _fidelity_gradient_coefficients(inversion, silencing_success, fidelity_phys, I_min, kappa_I, S_min, kappa_S)
    _assert_penalty_params(I_min, kappa_I, S_min, kappa_S)
    base_coeff_I = -2.0 * (1.0 - fidelity_phys) * silencing_success
    base_coeff_S = -2.0 * (1.0 - fidelity_phys) * inversion

    pen_coeff_I = inversion < I_min ? -Float64(kappa_I) * (Float64(I_min) - inversion) : 0.0
    pen_coeff_S = silencing_success < S_min ? -Float64(kappa_S) * (Float64(S_min) - silencing_success) : 0.0

    return base_coeff_I + pen_coeff_I, base_coeff_S + pen_coeff_S
end

"""
    pulse_cost(u, pulse, d; target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0, kwargs...)
        -> (cost, inversion, silencing, duration, coherence, field_amp, weak_seed_retention)

Scalar cost to be minimised. With `track=:weak` (default) there is ONE
`:weak` solve (`Sp = ε·Nj/2`): the paper per-frequency-slice silencing
factor `|F|_⋆ = ⟨|F(ω)|⟩` (Eq. 5 / A.132) comes from it, and `inversion`
is read from its own `Sz` (halves the ODE cost; O(ε)≈1e-3 bias vs the
`:ground` inversion -- see [`_assert_track`](@ref)). Pass `track=:dual`
explicitly to instead take inversion and silencing from **two
independent ODE solves** of the same pulse `u` (see [`pulse_metrics`](@ref)):
`:ground` → bright-mode (`Nj g²`) weighted inversion `I` (paper App. H),
the `:weak` track → the silencing factor. `:dual` is never selected
automatically. `target_F` picks which
cavity-QED protocol this pulse is being
optimised for: `target_F=1.0` (default) rewards preserving collective
coherence -- a pure `ω²/κ` chirp, `|F|→1` (RASE-style revival / ACE);
`target_F=0.0` rewards `g`-space cancellation, `|F|→0` (ROSE-style echo
silencing).

    silencing_success = 1 - (silencing - target_F)²
    fidelity_phys = inversion * silencing_success
    physics_cost = (1 - fidelity_phys)² + 0.5*kappa_I*max(0, I_min-inversion)²
                                         + 0.5*kappa_S*max(0, S_min-silencing_success)²

    J = physics_cost + w_time*(duration/T_max) + w_tmax*max(t_end[end]-T_max, 0)²/T_max²
        + w_power*mean(|cA/amp_scale|²)

`fidelity_phys` is the MULTIPLICATIVE combination of the two tracks (not
the additive `-w_inv*inversion + w_sil*(...)²` an earlier version of this
cost used): both a well-inverted `:ground` track AND a well-silenced/
well-revived `:weak` track are required for `physics_cost`'s base term
to reach its minimum of 0 -- either track alone, however good, cannot
drive the cost down on its own, since `fidelity_phys` is their PRODUCT,
not their weighted sum. There is accordingly no `w_inv`/`w_sil` knob any
more: both `inversion` and `silencing` always enter the objective,
unconditionally. What `track` changes is only how many ODE solves supply
them -- one `:weak` solve (default) or a separate `:ground` + `:weak`
pair (`track=:dual`, opt-in) -- never whether a metric is in the cost.

`I_min`/`kappa_I`/`S_min`/`kappa_S` (defaults [`_DEFAULT_PENALTY_MIN`](@ref)/
[`_DEFAULT_PENALTY_KAPPA`](@ref) each) are a squared-hinge exterior
penalty -- see [`_fidelity_physics_cost`](@ref) -- added directly to the
SAME cost every call, unconditionally (unlike the earlier `p_exp`/`q_exp`
exponent barrier this replaces, there is no per-epoch schedule here: the
formula is identical for `initial_metrics`/`final_metrics` and every
`run_local_adam` epoch alike). Pass `kappa_I=0, kappa_S=0` to disable the
penalty entirely and recover the plain `(1-inversion*silencing_success)^2`
formula bit-for-bit.

`w_power` is an L2 penalty on the decoded, scale-normalised amplitude
coefficients (i.e. `softplus.(raw_cA)`). Failed solves return `Inf`.
Do not pass `initial_condition`.

`coherence`, `field_amp` and `weak_seed_retention` are all DIAGNOSTIC
ONLY -- computed from the SAME `:weak` solve as `silencing` (no extra ODE
solve), never part of `cost`, which depends only on `inversion` and
`silencing`:

  * `coherence` (see [`_weighted_coherence`](@ref)/[`pulse_metrics`](@ref))
    -- the per-frequency-slice MAGNITUDE companion to `silencing`, same
    weighting/normalisation; `coherence ≥ silencing` and the gap is the
    magnitude lost to `g`-space phase spread.
  * `field_amp` (see [`_weighted_field_amplitude`](@ref)) -- the
    linearly-`g`-weighted, radiated-field-proportional counterpart.
  * `weak_seed_retention` (see [`_weak_seed_retention`](@ref)) -- the
    un-clamped magnitude ratio `coherence` clamps; a validity check on the
    App. B weak-excitation assumption (`≈ 1` good, `≫ 1` the pulse pumped
    the ensemble and the `|F|_⋆` normalisation is stretched).

All are appended at the END of the return tuple so existing
positional-unpacking callers (`cost, inv, sil, dur = pulse_cost(...)`)
keep working unchanged (Julia's tuple destructuring ignores extra
trailing values).
"""
function pulse_cost(u::AbstractVector, pulse::CompositePulse, d;
                     target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0,
                     I_min::Real=_DEFAULT_PENALTY_MIN, kappa_I::Real=_DEFAULT_PENALTY_KAPPA,
                     S_min::Real=_DEFAULT_PENALTY_MIN, kappa_S::Real=_DEFAULT_PENALTY_KAPPA,
                     compute::Symbol=:auto, track::Symbol=:weak, kwargs...)
    _forbid_initial_condition(kwargs)
    _assert_track(track)
    T = eltype(u)
    sk = _solver_kwargs(kwargs)
    duration = pulse_duration(pulse, u)
    _, t_end, _, cA, _ = decode(pulse, u)

    tmax_excess = max(t_end[end] - pulse.T_max, zero(T))
    tmax_penalty = w_tmax * (tmax_excess / pulse.T_max)^2

    n_cA = length(cA)
    n_cA > 0 || error("pulse_cost: decoded cA is empty.")
    normalized_cA = cA ./ pulse.amp_scale
    power_penalty = w_power * (sum(abs2, normalized_cA) / n_cA)

    inversion = zero(T)
    silencing = zero(T)
    coherence = zero(T)
    field_amp = zero(T)
    weak_seed_retention = zero(T)

    try
        if track === :dual
            # Two solves: :ground -> inversion, :weak -> the rest.
            _, _, Sz, Nj = run_sim_1st_order_pure(u, pulse, d; compute=compute, sk..., initial_condition=:ground)
            inversion = _weighted_inversion(Sz, d.g_b, Nj, T)
            _, Sp, _, Nj_eq = run_sim_1st_order_pure(u, pulse, d; compute=compute, sk..., initial_condition=:weak)
        else
            # One :weak solve; inversion read from its own Sz (O(ε) bias, see _assert_track).
            _, Sp, Sz_w, Nj_eq = run_sim_1st_order_pure(u, pulse, d; compute=compute, sk..., initial_condition=:weak)
            inversion = _weighted_inversion(Sz_w, d.g_b, Nj_eq, T)
        end
        silencing = _weighted_silencing_factor(Sp, d.g_b, Nj_eq, d.delta_b, T)
        coherence = _weighted_coherence(Sp, d.g_b, Nj_eq, d.delta_b, T)
        field_amp = _weighted_field_amplitude(Sp, d.g_b, Nj_eq, T)
        weak_seed_retention = _weak_seed_retention(Sp, d.g_b, Nj_eq, d.delta_b, T)
    catch e
        e isa PulseSolveFailed || rethrow()
        infT = convert(T, Inf)
        nanT = convert(T, NaN)
        return infT, nanT, nanT, duration, nanT, nanT, nanT
    end

    physics_cost, _, _ = _fidelity_physics_cost(inversion, silencing, target_F, I_min, kappa_I, S_min, kappa_S)

    cost = physics_cost + w_time * (duration / pulse.T_max) + tmax_penalty + power_penalty
    return cost, inversion, silencing, duration, coherence, field_amp, weak_seed_retention
end

# ============================================================
# ADAM (hand-rolled -- Optimisers.jl/Optim.jl are not dependencies of
# this package; Adam's update rule is short enough that adding either
# purely for this would be more overhead than benefit)
# ============================================================

mutable struct AdamState
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end

AdamState(n::Integer) = AdamState(zeros(n), zeros(n), 0)

"""
    adam_step!(u, grad, state::AdamState; lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, lr_scale=nothing)

One in-place Adam update of `u` given `grad = ∇cost(u)`, mutating `state`.
Standard bias-corrected Adam (Kingma & Ba 2015). `lr_scale`, if given, is a
per-parameter multiplier (same length as `u`) applied to the FINAL step
only (`u[i] -= lr*lr_scale[i]*m_hat/(sqrt(v_hat)+eps)`) -- the moment
estimates `state.m`/`state.v` themselves still accumulate the raw
`grad`, unscaled, so `lr_scale` reshapes the effective per-parameter step
size without distorting Adam's own gradient-magnitude bookkeeping. Default
`nothing` reproduces the original uniform-`lr` update exactly (skips the
per-element multiply entirely, not just multiplies by an implicit `1.0`).
See [`run_local_adam`](@ref)'s `cf_lr_scale` for why a composite-pulse
optimisation in particular benefits from decoupling the chirp
coefficients' own step size from the rest of `u`.
"""
function adam_step!(u::AbstractVector, grad::AbstractVector, state::AdamState;
                     lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, lr_scale::Union{AbstractVector,Nothing}=nothing)
    state.t += 1
    @inbounds for i in eachindex(u)
        state.m[i] = beta1 * state.m[i] + (1 - beta1) * grad[i]
        state.v[i] = beta2 * state.v[i] + (1 - beta2) * grad[i]^2
        m_hat = state.m[i] / (1 - beta1^state.t)
        v_hat = state.v[i] / (1 - beta2^state.t)
        step = lr * m_hat / (sqrt(v_hat) + eps)
        u[i] -= lr_scale === nothing ? step : lr_scale[i] * step
    end
    return u
end

# ============================================================
# SHAPE FITTING: warm-starting a CompositePulse from a FIXED target drive
#
# Separate from pulse_cost/run_local_adam below (which fit against this
# package's own PHYSICS via a full ODE solve): this fits purely against a
# target WAVEFORM, no ODE solve at all -- minimising mean squared error
# between build_E_of_t(pulse, u) and a target t -> Complex drive at many
# sample points, via the same ForwardDiff.gradient + hand-rolled Adam this
# file already uses elsewhere. Orders of magnitude cheaper than a physics
# fit (no ODE solve per epoch), which is the point: it turns an arbitrary
# recorded pulse (e.g. a jld2 run's own analytic control pulse -- see
# jld2_pulse_loader.jl's fit_linear_seed) into a REASONED
# CompositePulse seed for the physics optimisation below, rather than a
# shape-blind random/canonical guess.
# ============================================================

"""
    fit_composite_pulse(pulse::CompositePulse, E_target; N_fit=4000, num_epochs=1000,
                         learning_rate=0.002, seed=42, u_init=nothing) -> (u_fit, fit_report)

Fits `pulse`'s raw parameters `u` so that `build_E_of_t(pulse, u)`
approximates a target drive `E_target(t)` (any `t -> Complex` callable --
e.g. `pulses.jl`'s `build_E_of_t(PULSE_CONFIG)` applied to an existing
recorded pulse) over `[0, pulse.T_max]`, by minimising mean squared error
at `N_fit` evenly spaced sample points -- sampled via `pulses.jl`'s own
[`sample_E_of_t`](@ref)`(E_target, pulse.T_max, N_fit)`, the SAME sampling
function every other reconstructed-curve consumer in this package uses
(`plot_E_of_t`, `save_run_data`'s `_pulsemat.csv`), rather than a second,
independent sampling loop. Purely a SHAPE fit -- it says nothing about the
resulting physics (inversion/silencing); follow up with a real 1st-order
solve (e.g. [`run_sim_1st_order_trajectory`](@ref)) to check that
separately.

`learning_rate` defaults to `0.002`, NOT `run_local_adam`/`pulse_cost`'s
own `0.05` -- that value is tuned for `pulse_cost`'s dimensionless,
already-normalised cost gradient, whereas `mse_only` here is raw squared
error in `E_target`'s own physical units (order `amp_scale^2 ~ 1e14` for
this package's typical cavity-input-flux scale), and Adam's per-parameter
step size, while adaptive to gradient MAGNITUDE, is still set in absolute
raw-parameter units by `lr` itself. Verified on this package's own 3-ARP
reference pulse (k=3): `learning_rate=0.05` (and even `0.005`) overshoots
on the very first step and never recovers within hundreds of epochs
(`rel_l2` stays pinned near its EPOCH-1 value or worse), while `0.002`
converges steadily to `rel_l2~0.09-0.12` over ~1000 epochs -- if fitting a
very differently-scaled target/ensemble, re-check this the same way (watch
`fit_report.history` for a non-monotonic first few epochs, a sign `lr` is
too large for that target's own scale).

`u_init` defaults to [`initial_guess`](@ref)`(pulse; seed=seed)` (same
random-but-physically-sensible starting point the physics optimiser
itself uses) rather than all-zeros, since an all-zero `raw_gap`/`raw_dur`
still decodes to a valid (if arbitrary) placement via `decode`'s
softplus + cumulative-sum reparameterisation, but starting from a point
already spread out over `[0, T_max]` gives every sub-pulse's gradient a
chance to find its own share of the target from the first epoch, rather
than all `k` sub-pulses initially overlapping near `t=0`. This is still a
GENERIC, timing-blind default -- verified (again on the 3-ARP reference)
to plateau at `rel_l2~0.998` (no better than an all-zero pulse) after 200
epochs, because its amplitude scale and sub-pulse timings have no relation
to the target's own; [`fit_linear_seed`](@ref) in
jld2_pulse_loader.jl supplies a closed-form linear least-squares
`warm_start_u` of the control pulse instead of this Adam waveform fit.

Returns `(u_fit, fit_report)`: `u_fit` is the LOWEST-mse `u` seen across
all epochs (not necessarily the last, same "track the best, don't just
return wherever descent stopped" pattern [`run_local_adam`](@ref) uses).
`fit_report = (mse, rel_l2, history)`: `rel_l2 = sqrt(mse*N_fit /
sum(abs2, target))` is a scale-free fit-quality number (`0` = perfect,
`~1` = no better than an all-zero pulse) that stays meaningful across
different targets/ensembles, unlike raw `mse` (whose scale depends on
`E_target`'s own amplitude). `history` is a `Vector{<:NamedTuple}` with
one `(epoch, mse)` row per epoch actually run.
"""
function fit_composite_pulse(
    pulse::CompositePulse, E_target;
    N_fit::Integer=4000, num_epochs::Integer=1000,
    learning_rate::Real=0.002, seed::Integer=42,
    u_init::Union{AbstractVector,Nothing}=nothing,
)
    t_grid = range(0.0, pulse.T_max; length=N_fit)
    Ex, Ep = sample_E_of_t(E_target, pulse.T_max, N_fit)
    target = complex.(Ex, Ep)
    target_energy = sum(abs2, target) + 1e-30

    u = u_init === nothing ? initial_guess(pulse; seed=seed) : collect(Float64, u_init)
    length(u) == n_params(pulse) || error(
        "u_init has length $(length(u)), but this CompositePulse (k=$(pulse.k), " *
        "n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)) needs $(n_params(pulse))."
    )
    n = length(u)
    adam = AdamState(n)
    history = NamedTuple[]

    function mse_only(uu)
        E_of_t = build_E_of_t(pulse, uu)
        s = zero(eltype(uu))
        @inbounds for i in eachindex(t_grid)
            s += abs2(E_of_t(t_grid[i]) - target[i])
        end
        return s / N_fit
    end

    best_u, best_mse = copy(u), Inf
    for epoch in 1:num_epochs
        grad = ForwardDiff.gradient(mse_only, u)
        mse = Float64(mse_only(u))
        if mse < best_mse
            best_mse, best_u = mse, copy(u)
        end
        push!(history, (epoch=epoch, mse=mse))
        adam_step!(u, grad, adam; lr=learning_rate)
    end

    rel_l2 = sqrt(best_mse * N_fit / target_energy)
    return best_u, (mse=best_mse, rel_l2=rel_l2, history=history)
end

# ============================================================
# AMPLITUDE/FREQUENCY-SPACE FITTING FROM A RAW SAMPLED (t, I, Q) TRACE
#
# A second, more general route to a CompositePulse seed than
# fit_composite_pulse above: given ONLY a sampled I/Q waveform (no
# PULSE_CONFIG, no known sub-pulse count), discover how many sub-pulses it
# contains (silence-thresholding), extract its own amplitude/frequency
# decomposition, size ONE shared n_coeff_A/n_coeff_f from how many raw
# samples the densest detected sub-pulse actually has (~20-25 raw points
# per cubic B-spline piece), and fit build_A_f_of_t's amplitude/frequency
# curves against that decomposition directly -- rather than
# fit_composite_pulse's complex-valued MSE, which entangles amplitude and
# phase error in a way that is especially unstable exactly where the
# target is near-silent (phase becomes numerically meaningless as
# amplitude -> 0).
# ============================================================

"""
    _instantaneous_frequency(t, I, Q) -> (phi, f)

Unwrapped phase `phi(t_j) = atan(Q,I)` (continuous, no `2π` jumps at
consecutive samples) and its instantaneous angular frequency `f(t_j) =
dphi/dt` from a sampled I/Q trace: central difference on the unwrapped
`phi` (one-sided at the two endpoints). Both are returned at EVERY sample,
including where `I=Q=0` (phase, and hence `f`, is numerically meaningless
there) -- this function does not special-case or mask those points;
callers (see [`fit_composite_pulse_af`](@ref)/
[`fit_composite_pulse_from_samples`](@ref) with `fit_mode=:linear`) are expected to
down-weight them via an amplitude-based weight instead, since the
physically meaningful thing ("this region carries no drive") is already
fully captured by `A~0` there, not by discarding samples.

`phi` is returned (not just discarded internally the way an earlier
version of this function did) because it is itself a valid, DIRECT fit
target: `phi`'s own scale-invariant reference point is arbitrary (whatever
`atan` happens to return at the first sample, unrelated to
[`build_E_of_t`](@ref)'s own `phase_offset` convention), but its SHAPE over
one active sub-pulse is exactly the quantity a `cf`-fit that targets
[`build_E_of_t`](@ref)'s own EXACT phase integral `Φ(t) = ∫f dτ` should
match -- fitting `cf` against `f` alone (the OLDER approach) can leave
`Φ`'s accumulated integral drifting even when the per-point frequency
residual is tiny, since integration does not average away a small but
STRUCTURED (not i.i.d.) residual the way an RMS frequency comparison
would; see [`_fit_composite_pulse_from_samples_linear`](@ref)'s own
docstring for a real, measured case of exactly this (a `rel_l2_f~1e-13`
per-point frequency fit still producing a ~43% full-complex-trace
reconstruction error, traced to phase drift concentrated at the trace's
own peak amplitude).
"""
function _instantaneous_frequency(t::AbstractVector, I::AbstractVector, Q::AbstractVector)
    n = length(t)
    n == length(I) == length(Q) || error(
        "t/I/Q must have the same length, got $(n)/$(length(I))/$(length(Q))."
    )
    phi = atan.(Q, I)
    @inbounds for j in 2:n
        d = phi[j] - phi[j-1]
        while d > pi
            phi[j] -= 2 * pi
            d = phi[j] - phi[j-1]
        end
        while d < -pi
            phi[j] += 2 * pi
            d = phi[j] - phi[j-1]
        end
    end
    f = Vector{Float64}(undef, n)
    f[1] = (phi[2] - phi[1]) / (t[2] - t[1])
    f[n] = (phi[n] - phi[n-1]) / (t[n] - t[n-1])
    @inbounds for j in 2:n-1
        f[j] = (phi[j+1] - phi[j-1]) / (t[j+1] - t[j-1])
    end
    return phi, f
end

"""
    _detect_subpulse_segments(t, A; rel_thresh=1e-3, min_active_samples=5, min_silence_samples=3)
        -> Vector{Tuple{Int,Int}}

Detects contiguous "active" (non-silent) runs in a sampled amplitude trace
`A(t_j)`, returning `(i_start, i_end)` SAMPLE-INDEX ranges (inclusive
`t` indices), one per detected sub-pulse -- found purely from the sampled
waveform, with no assumed segment count or `PULSE_CONFIG` structure.

A sample counts as silent when `A[j] < rel_thresh * maximum(A)`. Two
robustness guards against noise: a candidate active run shorter than
`min_active_samples` is discarded (not a real sub-pulse, just a blip
poking above threshold in what's otherwise silence); a silent gap shorter
than `min_silence_samples` is merged back into "active" rather than
splitting one true sub-pulse into two (guards against a single noisy dip
near a target's own smooth near-zero region being mistaken for a genuine
separator).
"""
function _detect_subpulse_segments(
    t::AbstractVector, A::AbstractVector;
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
)
    n = length(A)
    thresh = rel_thresh * maximum(A)
    active = A .>= thresh

    j = 1
    while j <= n
        if !active[j]
            j0 = j
            while j <= n && !active[j]
                j += 1
            end
            gap_len = j - j0
            if gap_len < min_silence_samples && j0 > 1 && j <= n
                active[j0:j-1] .= true
            end
        else
            j += 1
        end
    end

    segments = Tuple{Int,Int}[]
    j = 1
    while j <= n
        if active[j]
            j0 = j
            while j <= n && active[j]
                j += 1
            end
            if j - j0 >= min_active_samples
                push!(segments, (j0, j - 1))
            end
        else
            j += 1
        end
    end
    return segments
end

"""
    _spline_coeff_count(n_samples; points_per_segment=22, degree=3) -> Int

Number of B-spline coefficients so each piecewise-cubic segment spans
roughly `points_per_segment` raw sample points: `n_pieces =
ceil(n_samples/points_per_segment)`, and (since a degree-`degree` clamped
B-spline with `n_coeff` coefficients has `n_coeff-degree` pieces --
[`make_clamped_knots`](@ref)) `n_coeff = n_pieces + degree`, floored at
`degree+1` (the minimum [`CompositePulse`](@ref) accepts).
"""
function _spline_coeff_count(n_samples::Integer; points_per_segment::Integer=22, degree::Integer=3)
    n_pieces = max(cld(n_samples, points_per_segment), 1)
    return max(n_pieces + degree, degree + 1)
end

"""
    points_per_segment_for_budget(n_samples_max, k; degree=3, param_budget=60) -> Int

Largest-detail `points_per_segment` (in [`_spline_coeff_count`](@ref)'s own
sense) that still keeps a `CompositePulse`'s total raw parameter count
(`n_params = 3*k + 2*k*n_coeff`, since `n_coeff_A = n_coeff_f = n_coeff` --
see [`n_params`](@ref)) at or under `param_budget`, given `k` sub-pulses and
`n_samples_max` (the LONGEST detected segment's own sample count --
`_spline_coeff_count` sizes the single shared `n_coeff` off this one value).

Inverts `_spline_coeff_count`'s own `n_pieces = ceil(n_samples/pps)`,
`n_coeff = n_pieces + degree`: first finds the largest feasible `n_coeff`
(`n_coeff_max = (param_budget - 3*k) ÷ (2*k)`, floor division), then the
SMALLEST `pps` that achieves at most `n_coeff_max - degree` pieces (`pps =
ceil(n_samples_max / n_pieces_max)`) -- ceiling division only ever pushes
the resulting `n_pieces` DOWN relative to `n_pieces_max`, never up, so the
ACTUAL `n_coeff` `_spline_coeff_count` returns for this `pps` is guaranteed
`<= n_coeff_max`, hence `n_params <= param_budget` exactly (checked below,
not just argued), not merely approximately.

Throws if `param_budget` cannot be met even at `CompositePulse`'s own
minimum coefficient count (`degree+1` per sub-pulse -- `_spline_coeff_count`'s
own floor), i.e. `param_budget < 3*k + 2*k*(degree+1)`: no `points_per_segment`,
however large, can go lower than that floor, so reduce `k` (fewer detected
sub-pulses) or `degree`, or raise `param_budget`, instead.
"""
function points_per_segment_for_budget(n_samples_max::Integer, k::Integer; degree::Integer=3, param_budget::Integer=60)
    k >= 1 || error("k must be a positive integer, got $k.")
    n_samples_max >= 1 || error("n_samples_max must be a positive integer, got $n_samples_max.")

    # n_params_for MUST match CompositePulse's own n_params(pulse) =
    # 3*pulse.k + pulse.k*pulse.n_coeff_A + pulse.k*pulse.n_coeff_f
    # (composite_pulse.jl) for the case n_coeff_A=n_coeff_f=n_coeff this
    # function always builds -- kept as ONE local closure (rather than
    # re-typed at each use below) specifically so there is only one place
    # to update if that formula ever changes again.
    n_params_for(n_coeff) = 3 * k + 2 * k * n_coeff

    n_coeff_floor = degree + 1
    min_budget = n_params_for(n_coeff_floor)
    param_budget >= min_budget || error(
        "param_budget=$param_budget cannot be met for k=$k sub-pulses at degree=$degree: " *
        "even the minimum coefficient count per sub-pulse ($n_coeff_floor) already needs " *
        "$min_budget total parameters (3*k + 2*k*(degree+1)). Reduce k/degree, or raise param_budget."
    )

    n_coeff_max = (param_budget - 3 * k) ÷ (2 * k)
    n_pieces_max = n_coeff_max - degree
    pps = cld(n_samples_max, n_pieces_max)

    n_coeff_actual = _spline_coeff_count(n_samples_max; points_per_segment=pps, degree=degree)
    n_params_actual = n_params_for(n_coeff_actual)
    n_params_actual <= param_budget || error(
        "internal inconsistency: computed points_per_segment=$pps still gives " *
        "n_params=$n_params_actual > param_budget=$param_budget (n_coeff=$n_coeff_actual)."
    )

    return pps
end

"""
    points_per_segment_for_budget(t, I, Q; degree=3, param_budget=60,
        rel_thresh=1e-3, min_active_samples=5, min_silence_samples=3) -> (pps::Int, segments)

Convenience wrapper over the `(n_samples_max, k)` method above: runs the
SAME segment detection ([`_detect_subpulse_segments`](@ref)) that
[`fit_composite_pulse_from_samples`](@ref)'s `fit_mode=:learned`/`:linear`
implementations use internally, then sizes
`points_per_segment` against the resulting `k` and longest-segment sample
count. Returns `(pps, segments)` so a caller can pass `pps` straight into
either fitter's own `points_per_segment` keyword -- or just pass
`param_budget` directly to those fitters, which do exactly this internally
(see their own docstrings).
"""
function points_per_segment_for_budget(
    t::AbstractVector, I::AbstractVector, Q::AbstractVector;
    degree::Integer=3, param_budget::Integer=60,
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
)
    A = sqrt.(I .^ 2 .+ Q .^ 2)
    segments = _detect_subpulse_segments(
        t, A; rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
    )
    isempty(segments) && error(
        "No active sub-pulses detected (rel_thresh=$rel_thresh too high relative to the " *
        "trace's own peak, or the trace really is all silence)."
    )
    k = length(segments)
    n_samples_max = maximum(i_end - i_start + 1 for (i_start, i_end) in segments)
    pps = points_per_segment_for_budget(n_samples_max, k; degree=degree, param_budget=param_budget)
    return pps, segments
end

"""
    _resolve_points_per_segment(points_per_segment, param_budget, n_samples_each, k, degree, caller_name) -> Int

Shared `param_budget` OVERRIDE step behind [`fit_composite_pulse_from_samples`](@ref)'s
two `fit_mode` implementations ([`_fit_composite_pulse_from_samples_learned`](@ref)
and [`_fit_composite_pulse_from_samples_linear`](@ref)): when `param_budget`
is given, replaces the caller's own `points_per_segment` with
[`points_per_segment_for_budget`](@ref)'s own sizing (against the ALREADY-
DETECTED `k`/`n_samples_each`, never a caller guess), printing what it
picked (`caller_name` names which of the two callers is logging, so the
message stays traceable back to its own call site); otherwise returns
`points_per_segment` unchanged.
"""
function _resolve_points_per_segment(
    points_per_segment::Integer, param_budget::Union{Nothing,Integer},
    n_samples_each, k::Integer, degree::Integer, caller_name::AbstractString,
)
    param_budget === nothing && return points_per_segment
    pps = points_per_segment_for_budget(maximum(n_samples_each), k; degree=degree, param_budget=param_budget)
    println("$caller_name: param_budget=$param_budget -> points_per_segment=$pps (k=$k)")
    return pps
end

"""
    fit_composite_pulse_af(pulse::CompositePulse, t_samples, A_target, f_target;
                            weight=A_target.^2, num_epochs=1000, learning_rate=0.002,
                            seed=42, u_init=nothing) -> (u_fit, fit_report)

Fits `pulse`'s raw parameters so its OWN amplitude/frequency curves
([`build_A_f_of_t`](@ref)) match `A_target`/`f_target` at `t_samples`,
instead of [`fit_composite_pulse`](@ref)'s complex-valued MSE. Minimises

    mean((A_of_t(t_j) - A_target[j])^2) +
    mean(weight[j] * (f_of_t(t_j) - f_target[j])^2) / mean(weight)

`weight` defaults to `A_target.^2`: instantaneous frequency extracted from
a sampled I/Q trace ([`_instantaneous_frequency`](@ref)) is numerically
meaningless wherever the amplitude is near 0 (phase is undefined at the
origin), so the frequency term is amplitude-weighted rather than given
equal weight everywhere; the amplitude term needs no such weighting since
`A_target` stays well-defined, and physically meaningful, all the way
down to 0. Dividing the frequency term by `mean(weight)` keeps the two
terms on comparable absolute scale regardless of `A_target`'s own units,
rather than letting whichever term happens to have larger raw magnitude
dominate the gradient purely because of a units mismatch.

Returns `(u_fit, fit_report)`: `u_fit` is the LOWEST-loss `u` seen across
all epochs. `fit_report = (loss, rel_l2_A, history)`: `rel_l2_A` is the
amplitude term's OWN scale-free residual (`0`=perfect), reported
separately from the combined `loss` since that's the more interpretable
single number for comparing fits (frequency error has no natural `[0,1]`
scale the way `fit_composite_pulse`'s `rel_l2` does).
"""
function fit_composite_pulse_af(
    pulse::CompositePulse, t_samples::AbstractVector, A_target::AbstractVector, f_target::AbstractVector;
    weight::AbstractVector=A_target .^ 2,
    num_epochs::Integer=1000, learning_rate::Real=0.002, seed::Integer=42,
    u_init::Union{AbstractVector,Nothing}=nothing,
)
    N = length(t_samples)
    (length(A_target) == N && length(f_target) == N && length(weight) == N) || error(
        "t_samples/A_target/f_target/weight must all have the same length, got " *
        "$(N)/$(length(A_target))/$(length(f_target))/$(length(weight))."
    )
    weight_mean = sum(weight) / N + 1e-30
    A_energy = sum(abs2, A_target) + 1e-30

    u = u_init === nothing ? initial_guess(pulse; seed=seed) : collect(Float64, u_init)
    length(u) == n_params(pulse) || error(
        "u_init has length $(length(u)), but this CompositePulse (k=$(pulse.k), " *
        "n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)) needs $(n_params(pulse))."
    )
    n = length(u)
    adam = AdamState(n)
    history = NamedTuple[]

    function loss_only(uu)
        A_of_t, f_of_t = build_A_f_of_t(pulse, uu)
        sA = zero(eltype(uu))
        sf = zero(eltype(uu))
        @inbounds for j in 1:N
            sA += abs2(A_of_t(t_samples[j]) - A_target[j])
            sf += weight[j] * abs2(f_of_t(t_samples[j]) - f_target[j])
        end
        return sA / N + (sf / N) / weight_mean
    end

    best_u, best_loss = copy(u), Inf
    for epoch in 1:num_epochs
        grad = ForwardDiff.gradient(loss_only, u)
        loss = Float64(loss_only(u))
        if loss < best_loss
            best_loss, best_u = loss, copy(u)
        end
        push!(history, (epoch=epoch, loss=loss))
        adam_step!(u, grad, adam; lr=learning_rate)
    end

    A_of_t_best, _ = build_A_f_of_t(pulse, best_u)
    A_resid = sum(j -> abs2(A_of_t_best(t_samples[j]) - A_target[j]), 1:N)
    rel_l2_A = sqrt(A_resid / A_energy)

    return best_u, (loss=best_loss, rel_l2_A=rel_l2_A, history=history)
end

# ============================================================
# GRADIENT-SAFETY FLOOR/CLIP FOR SAMPLE-DERIVED SEEDS
#
# `decode`'s `gap`/`duration`/`cA` all go through `_softplus_inv` to reach
# RAW space, and `d(softplus)/d(raw) = sigmoid(raw)` -- for
# `raw = _softplus_inv(y)` with small `y = physical/scale`,
# `sigmoid(raw) ≈ y` (verified: `y=1e-6` -> `sigmoid(raw)≈1e-6`, `y=1e-30`
# -> `sigmoid(raw)≈1e-30`). Encoding a fitted quantity at a tiny fraction
# of its own scale -- this file previously did so via `max(peak_amp,
# 1e-30)` for `cA` in `fit_composite_pulse_from_samples`'s (`fit_mode=:learned`)
# seed, and `cA_floor_frac=1e-6` in its `fit_mode=:linear` implementation --
# therefore lands that raw parameter at a point where `decode`'s own
# gradient is attenuated by that same tiny factor from the very first
# optimisation epoch onward: `run_local_adam`'s Adam step can move such a
# parameter only as fast as this near-zero local sensitivity allows,
# regardless of how many epochs run, i.e. a practically dead parameter at
# the seed. `_GRAD_SAFE_FRAC=1e-2` keeps that attenuation to about 100x
# instead of 1e6-1e30x -- a deliberate seed-generation trade-off (a truly
# near-zero target coefficient gets encoded slightly too high) in exchange
# for guaranteeing every seeded parameter starts somewhere gradient
# descent can actually move it from.
# ============================================================

const _GRAD_SAFE_FRAC = 1e-2

"""
    _encode_scaled_softplus(physical, scale) -> Float64

Gradient-safe RAW encoding of a `decode`-softplus-reparameterised quantity
(`gap`, `duration`, or `cA`) fitted/derived directly from sampled data:
`_softplus_inv(max(physical, _GRAD_SAFE_FRAC*scale) / scale)`. See
[`_GRAD_SAFE_FRAC`](@ref) for why the floor is relative to `scale` rather
than an absolute numerical epsilon.
"""
_encode_scaled_softplus(physical::Real, scale::Real) = _softplus_inv(max(physical, _GRAD_SAFE_FRAC * scale) / scale)

"""
    _clip_cf_raw(cf_raw, cf_clip_mult) -> raw value(s), clamped

Clamps a fitted, already-scale-normalised chirp coefficient (`cf/freq_scale`)
to `± cf_clip_mult`. Unlike `gap`/`duration`/`cA`, `cf` is UNCONSTRAINED in
RAW space (no softplus, see [`decode`](@ref)), so it carries no vanishing-
gradient tail -- but a weighted least-squares (or MSE) fit against a noisy
instantaneous-frequency estimate can still occasionally return an outlier
coefficient far beyond any physically sensible chirp rate (the `A²`
weighting in [`fit_composite_pulse_af`](@ref)/[`_fit_composite_pulse_from_samples_linear`](@ref)
suppresses, but does not eliminate, noise from low-but-above-threshold-
amplitude samples). An extreme `cf` makes the very first physics-cost ODE
solve unnecessarily stiff or prone to outright failure -- and a failed
solve makes `pulse_cost` return a constant `Inf`/`NaN` with a ZERO
gradient (see `run_local_adam`), stalling the optimiser at the seed with
no way to move at all. `cf_clip_mult=20` (this section's default) is
generous relative to `pulse.freq_scale` (the ensemble's own `FWHM`, already
the natural chirp scale for this package's physics -- see
[`CompositePulse`](@ref)'s own `Omega_adiabatic` derivation) so it only
ever engages on genuine noise-driven outliers, not ordinary chirped
sub-pulses.
"""
_clip_cf_raw(cf_raw, cf_clip_mult::Real) = clamp.(cf_raw, -cf_clip_mult, cf_clip_mult)

"""
    _fit_composite_pulse_from_samples_learned(t, I, Q, d;
        points_per_segment=22, degree=3, taper_frac=0.1,
        rel_thresh=1e-3, min_active_samples=5, min_silence_samples=3,
        cf_clip_mult=20.0, num_epochs=1000, learning_rate=0.002, seed=42)
        -> (pulse::CompositePulse, u_fit, fit_report, segments)

Private `fit_mode=:learned` implementation behind
[`fit_composite_pulse_from_samples`](@ref) -- see that function for the
public entry point. Builds a [`CompositePulse`](@ref) seed DIRECTLY from a sampled I/Q trace,
with NO prior knowledge of how many sub-pulses it contains or where they
are:

  1. Detects sub-pulses via [`_detect_subpulse_segments`](@ref) (silence
     thresholding on `A=sqrt(I^2+Q^2)`) -- `k` is however many segments
     that finds, not a caller-supplied count.
  2. Extracts the target amplitude/frequency decomposition for the WHOLE
     trace ([`_instantaneous_frequency`](@ref)).
  3. Sizes ONE shared `n_coeff_A`/`n_coeff_f` (used for every sub-pulse --
     `CompositePulse` has no per-sub-pulse coefficient count) via
     [`_spline_coeff_count`](@ref), from whichever detected segment has
     the MOST samples -- every segment gets AT LEAST ~`points_per_segment`
     raw points per cubic piece this way, though a shorter segment ends up
     with more spline resolution than its own sample count would strictly
     need.
  4. Builds a segment-matched `u_init`: each sub-pulse placed at its
     detected segment's own `t`-span, amplitude flat at that segment's own
     peak `A`, frequency ramped linearly across that segment's own
     `extrema(f)` -- the same idea as the linear control-seed path in
     jld2_pulse_loader.jl ([`fit_linear_seed`](@ref)), but
     derived from the DETECTED segments' own sample data rather than a
     labelled `PULSE_CONFIG`.
  5. Fits via [`fit_composite_pulse_af`](@ref) (amplitude/frequency-space).

`d` is `prepare_derived(CONFIG)`'s own return value (only used for
`CompositePulse`'s own `T_max`/scale fields -- `t`'s own span need not
equal `d.timespan`, though for a physically meaningful seed it should).

Seed encoding is GRADIENT-SAFE by construction (see
[`_encode_scaled_softplus`](@ref)/[`_clip_cf_raw`](@ref)): `gap`/`duration`/
`cA` are floored at `_GRAD_SAFE_FRAC` (`1e-2`) of their own scale rather
than at an arbitrarily small numerical epsilon, and `cf` is clipped to
`± cf_clip_mult * pulse.freq_scale` -- both guard against handing
[`optimise_composite_pulse`](@ref)/`run_local_adam` a seed with a
practically-dead (vanishing-decode-gradient) parameter or a noise-driven
chirp outlier that makes the very first physics ODE solve fail (a failed
solve makes `pulse_cost` return `Inf`/`NaN` with a ZERO gradient, stalling
the optimiser at the seed with no way to move at all -- see
[`_clip_cf_raw`](@ref)'s own docstring).

Returns `(pulse, u_fit, fit_report, segments)` -- `segments` is the raw
`(i_start, i_end)` sample-index list from step 1, for inspection/plotting.

Pass `param_budget` (e.g. `60`) instead of hand-picking `points_per_segment`
to instead cap the resulting `n_params = 3*k + 2*k*n_coeff` directly -- see
[`points_per_segment_for_budget`](@ref), which this calls internally (after
step 1 determines `k`) to override `points_per_segment` when `param_budget`
is given. Useful because `ForwardDiff.gradient`+Adam (this function's own
descent) becomes impractically slow well before `n_coeff` reaches the tens,
per this docstring's own opening paragraph -- capping `n_params` up front is
the practical way to keep this route usable on a densely-sampled real trace.
"""
function _fit_composite_pulse_from_samples_learned(
    t::AbstractVector, I::AbstractVector, Q::AbstractVector, d;
    points_per_segment::Integer=22, degree::Integer=3, taper_frac::Real=0.1,
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
    cf_clip_mult::Real=20.0, num_epochs::Integer=1000, learning_rate::Real=0.002, seed::Integer=42,
    param_budget::Union{Nothing,Integer}=nothing,
)
    A = sqrt.(I .^ 2 .+ Q .^ 2)
    phi, f = _instantaneous_frequency(t, I, Q)

    segments = _detect_subpulse_segments(
        t, A; rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
    )
    isempty(segments) && error(
        "No active sub-pulses detected (rel_thresh=$rel_thresh too high relative to the " *
        "trace's own peak, or the trace really is all silence)."
    )
    k = length(segments)

    n_samples_each = [i_end - i_start + 1 for (i_start, i_end) in segments]
    points_per_segment = _resolve_points_per_segment(
        points_per_segment, param_budget, n_samples_each, k, degree, "fit_composite_pulse_from_samples (fit_mode=:learned)",
    )
    n_coeff = _spline_coeff_count(maximum(n_samples_each); points_per_segment=points_per_segment, degree=degree)

    pulse = CompositePulse(k, n_coeff, n_coeff, d; degree=degree, taper_frac=taper_frac)

    raw_gap = Vector{Float64}(undef, k)
    raw_dur = Vector{Float64}(undef, k)
    raw_phi0 = Vector{Float64}(undef, k)
    raw_cA = Matrix{Float64}(undef, n_coeff, k)
    raw_cf = Matrix{Float64}(undef, n_coeff, k)
    t_prev_end = 0.0
    running_seed = 0.0
    for (idx, (i_start, i_end)) in enumerate(segments)
        t_s, t_e = t[i_start], t[i_end]
        duration = t_e - t_s
        gap = max(t_s - t_prev_end, 0.0)
        dur_arg = max(duration - pulse.dur_floor, 0.0)
        raw_gap[idx] = _encode_scaled_softplus(gap, pulse.gap_scale)
        raw_dur[idx] = _encode_scaled_softplus(dur_arg, pulse.dur_scale)

        # build_A_f_of_t's own loss (fit_composite_pulse_af, below) never
        # references phi0 at all, so ForwardDiff/Adam leaves this component
        # untouched (zero gradient) -- this seed IS the final fitted value,
        # not just a starting point. Approximate `running` (build_E_of_t's
        # own accumulator) with the TARGET trace's own raw phase at each
        # segment's boundary, since this path's `raw_cf` seed is only a
        # crude linear ramp (not an exact antiderivative fit) and so has no
        # equally cheap EXACT `d_f[end]` to accumulate instead.
        raw_phi0[idx] = phi[i_start] - running_seed
        running_seed = phi[i_end]

        peak_amp = maximum(view(A, i_start:i_end))
        raw_cA[:, idx] .= _encode_scaled_softplus(peak_amp, pulse.amp_scale)

        f_lo, f_hi = extrema(view(f, i_start:i_end))
        raw_cf[:, idx] .= _clip_cf_raw(range(f_lo, f_hi; length=n_coeff) ./ pulse.freq_scale, cf_clip_mult)

        t_prev_end = t_e
    end
    u_init = pack(pulse, raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf)

    u_fit, fit_report = fit_composite_pulse_af(
        pulse, t, A, f; num_epochs=num_epochs, learning_rate=learning_rate, seed=seed, u_init=u_init,
    )
    return pulse, u_fit, fit_report, segments
end

"""
    _fit_composite_pulse_from_samples_linear(t, I, Q, d;
        points_per_segment=6, degree=3, taper_frac=0.1,
        rel_thresh=1e-3, min_active_samples=5, min_silence_samples=3,
        cA_floor_frac=_GRAD_SAFE_FRAC, cf_clip_mult=20.0,
        param_budget=nothing, segments=nothing)
        -> (pulse::CompositePulse, u_fit, fit_report, segments)

Private `fit_mode=:linear` implementation behind
[`fit_composite_pulse_from_samples`](@ref) -- see that function for the
public entry point. Closed-form alternative to
[`_fit_composite_pulse_from_samples_learned`](@ref) (`fit_mode=:learned`):
same segment detection / `n_coeff` sizing (steps 1-3 of that function's own
docstring), but NO `ForwardDiff`/Adam descent at all. This matters at the
resolution the 20-25-points-per-segment rule implies for a real, densely
sampled trace: a single ~200us sub-pulse sampled at ~5000 points over
~600us needs `n_coeff~80`, and for `k=3` that is `n_params~480` --
verified impractical for `ForwardDiff.gradient`+Adam as currently
implemented (`bspline_basis` allocates fresh temporaries on every call,
with no caching; a real attempt at this scale did not finish a single
epoch in 10 minutes, ~9x10^8 allocations in). This function instead
exploits that MOST of what's being fit is actually linear, given the
segmentation this file already computes independently of any pulse
parameter:

  - `t_start`/`t_end` per sub-pulse are taken DIRECTLY from the detected
    segment's own sample boundaries -- no fitting needed, since
    segmentation already locates them exactly (to sample resolution).
  - Given those (hence given the B-spline's knot vector), `A_spline(t) =
    Σ cA_i B_i(t)` is LINEAR in `cA` -- the exact solution of a
    taper-weighted (folding the KNOWN, FIXED `_taper_window` multiplier
    into the design matrix so the amplitude term being solved for is
    `pulse`'s own actual physical envelope, not the bare untapered spline)
    linear least-squares problem, via `\\`.
  - `cf` is fit against the ACCUMULATED PHASE `phi` (the raw unwrapped
    `atan(Q,I)` from [`_instantaneous_frequency`](@ref)), NOT the
    pointwise frequency curve `f_spline(t) = Σ cf_i B_i(t)` an earlier
    version of this function fit directly. `Φ(t) = ∫f dτ` is itself LINEAR
    in `cf` (de Boor's antiderivative construction -- the same one
    [`bspline_antiderivative`](@ref) uses, expressed here as an explicit
    `(n_coeff+1, n_coeff)` linear map so the fit stays a single weighted
    least-squares solve, verified bit-for-bit equivalent to that
    function's own cumulative-sum recursion), so this is still one
    `n_samples x (n_coeff+1)` solve per sub-pulse, exact and just as cheap
    -- the `+1` column is an unconstrained additive constant absorbing
    `phi`'s own arbitrary reference point (unrelated to
    [`build_E_of_t`](@ref)'s own `phase_offset` bookkeeping), discarded
    after the solve. This matters because a per-point frequency fit can
    have an excellent RMS residual while its INTEGRAL still drifts
    (integration doesn't average away a small but structured, rather than
    i.i.d., residual) -- measured directly on this package's own 3-ARP
    reference: fitting `cf` against `f` gave `rel_l2_f~1e-13` (an
    essentially perfect per-point frequency fit) yet the reconstructed
    COMPLEX pulse (`build_E_of_t(pulse,u_fit)`, resampled and compared
    point-by-point against the original trace) was off by `rel_l2~0.43`,
    with >99.9999% of that error concentrated at the trace's own peak
    amplitude sample -- exactly where a modest phase error costs the most
    in a squared-error sum. Fitting `cf` against `phi` directly targets the
    quantity that actually enters `E(t)=A(t)*exp(iΦ(t))`, closing that gap.
    Both `cf`'s solve and the (retained, diagnostic-only) frequency-domain
    comparison are WEIGHTED by `A_target.^2` (same rationale as
    [`fit_composite_pulse_af`](@ref)'s `weight` default: phase/frequency
    are numerically meaningless wherever the target amplitude is near 0).

`points_per_segment` defaults to `6` (not the `fit_mode=:learned`
implementation's 20-25 spec) -- verified on this package's own 3-ARP reference pulse to
still improve `rel_l2_A` noticeably over the 20-25 range (0.0013 at 22
points/segment down to 0.00055 at 6), with sharply diminishing returns
below that (a further halving to 4 points/segment only reached 0.00046)
and negligible runtime cost either way (all of 22/12/8/6/4 fit in under
1.5s combined on that reference case, since this is a handful of small
linear solves, not an iterative descent).

`cA_floor_frac`: ordinary least squares has no non-negativity constraint,
but `CompositePulse`'s own parameterisation requires `cA >= 0`
(`decode`'s `cA = amp_scale*softplus(raw_cA)`, always non-negative, so
`_softplus_inv` needs a positive argument) -- any solved coefficient below
`cA_floor_frac * pulse.amp_scale` is clamped up to that floor before
encoding. This doubles as this seed's GRADIENT-SAFETY floor (see
[`_GRAD_SAFE_FRAC`](@ref)): `decode`'s own softplus-reparameterisation
gradient at the seed is attenuated by roughly `cA_floor_frac` itself, so
the previous default (`1e-6`, chosen only to keep `_softplus_inv`'s
argument positive) left any floored coefficient with an effectively DEAD
decode-gradient for the whole physics optimisation that follows --
`_GRAD_SAFE_FRAC=1e-2` keeps that attenuation to about 100x instead of
1e6x. This is still a pragmatic guard against small negative undershoots
near sharp features, NOT a proper non-negative least-squares solve; on the
package's own 3-ARP reference case this floor starts triggering (a small
handful of coefficients, out of hundreds) right around `points_per_segment
= 6`, so `fit_report.n_cA_floored` is worth checking at this default --
a persistently nonzero count is a sign a true NNLS solve would do better
than this clamp. `cf_clip_mult` guards the frequency side the same way
[`_clip_cf_raw`](@ref) does for [`_fit_composite_pulse_from_samples_learned`](@ref):
the weighted per-segment linear solve for `cf` has no such floor issue
(unconstrained, no softplus) but can still return a noise-driven outlier
coefficient that makes the very first physics ODE solve stiff or prone to
failure -- clipped to `± cf_clip_mult * pulse.freq_scale`.

Returns `(pulse, u_fit, fit_report, segments)` -- `fit_report =
(rel_l2_A, rel_l2_f, phi_rms_rad, rel_l2_complex, n_cA_floored,
n_cf_clipped)`: `rel_l2_A` is the same scale-free `[0,1]`-ish residual
`fit_composite_pulse`/`_af` report (`0` = perfect); `rel_l2_f` is now a
DIAGNOSTIC-ONLY pointwise frequency-curve residual (`Basis*cf_seg` vs
`f_seg`, evaluated at the `cf_seg` the PHASE fit produced) -- kept for
comparison, no longer what `cf` is actually fit against; `phi_rms_rad` is
the `A²`-weighted RMS phase residual in RADIANS (`Φ(t)` vs `phi`, the
quantity `cf`'s solve DOES target) -- unlike `rel_l2_A`/`rel_l2_f` this
has no natural `[0,1]` scale (it's an absolute angle), so judge it against
how many radians of phase error would actually matter for your own physics
(a fraction of a radian is generally fine; an O(1) value at high amplitude
is not). `rel_l2_complex` is the full-trace COMPLEX reconstruction error
(`build_E_of_t(pulse,u_fit)` resampled at `t`, compared against
`complex.(I,Q)` directly, `0`=perfect) -- unlike the other three, which
each check one decoupled piece, this is what a caller of the fitted pulse
actually experiences; see the "RESOLVED" note below for why this can (and
used to) disagree sharply with `phi_rms_rad` alone. `n_cA_floored`/
`n_cf_clipped` are the total counts of amplitude/frequency coefficients
(across all sub-pulses) that hit the non-negativity floor / clip bound
above.

RESOLVED (previously a known limitation of this function) -- a
near-perfect `phi_rms_rad` does NOT by itself guarantee a near-perfect
reconstructed COMPLEX pulse: each sub-pulse's phase fit here includes a
free additive constant (`phase_const` in the loop below) that absorbs
`phi`'s own arbitrary reference point. An earlier version of this function
discarded that constant after the solve, and `CompositePulse` itself had
no parameter to receive it back -- `build_E_of_t` hardcoded sub-pulse 1's
own phase to start at exactly `0` and accumulated every later sub-pulse's
own `phase_offset[i]` purely from the FITTED sub-pulses' internal `∫f`,
never from anything about the target's own true absolute phase reference.
Measured directly on this package's own 3-ARP reference before the fix:
sub-pulse 1's own raw phase at its detected start was `0.4466` rad,
predicting a `2*sin(0.4466/2) = 0.443` relative full-trace error --
matching the then-measured `rel_l2_complex = 0.434` almost exactly, despite
`phi_rms_rad ~ 1e-13`. `CompositePulse` now has an explicit per-sub-pulse
`raw_phi0` (see [`decode`](@ref)/[`build_E_of_t`](@ref)), and `phase_const`
is recovered into it EXACTLY here (not re-fit, not approximated -- see
`raw_phi0[idx] = phase_const - running_phase` in the loop below): on the
same 3-ARP reference, `rel_l2_complex` now measures `0.0004`, matching
`rel_l2_A` rather than sitting two orders of magnitude worse.

RESOLVED (a second, independent gap the above fix did not close) -- even
with `raw_phi0` recovered exactly, `rel_l2_complex` could still sit at
`~1.4 (~sqrt(2))` regardless of fit resolution (`phi_rms_rad` from `~0.04`
down to `~1e-13`), while the pointwise reconstructed phase
(`build_E_of_t(pulse,u_fit)` vs the target, sample by sample) swept a full
extra `2*pi` across each sub-pulse rather than sitting at one constant
offset -- so no single per-segment phase realignment could fix it either
(measured directly: the closed-form global-phase-alignment correction
that minimises `‖e^{i*delta}*E_pred - E_tar‖^2` per segment came out at a
fraction of a degree and left `rel_l2_complex` completely unchanged).
Root cause: this loop built each sub-pulse's knot vector / taper window /
antiderivative from the raw SAMPLE bounds `(t_s, t_e) = (t[i_start],
t[i_end])`, but [`_encode_scaled_softplus`](@ref) floors `gap`/`dur_arg`
up to `_GRAD_SAFE_FRAC*scale` before encoding -- so whenever a sub-pulse's
true leading gap is smaller than that floor (the common case for a
sub-pulse starting at/near `t=0`), [`decode`](@ref)'s cumulative-sum
`t_start`/`t_end` for THAT sub-pulse, and (because the sum is cumulative)
every LATER sub-pulse too, is shifted from `(t_s, t_e)` by the floored
amount. [`build_E_of_t`](@ref) reconstructs on `decode`'s shifted knots
but is sampled at the trace's own unshifted physical `t` -- for a chirped
drive a pure time shift is `offset*f(t)`, not a phase constant, which is
exactly the sweeping-2*pi symptom above. Fixed by recomputing `(t_s, t_e)`
right after `raw_gap[idx]`/`raw_dur[idx]` are encoded, from that SAME
encoding (`t_prev_end + scale*softplus(raw)[+floor]` -- `decode`'s own
round trip, no second pass), and carrying the decoded end forward as
`t_prev_end` for the next sub-pulse -- so every downstream design matrix
in this loop is built on exactly the domain `build_E_of_t` will actually
reconstruct on, for any gap/dur (floored or not). Verified on the 3-ARP
`run_105_3ARP_M20000` reference: `rel_l2_complex` drops from `~1.4` to
match `rel_l2_A`'s own scale at the same fit resolution.

Pass `param_budget` (e.g. `60`) instead of hand-picking `points_per_segment`
to instead cap the resulting `n_params = 3*k + 2*k*n_coeff` directly -- see
[`points_per_segment_for_budget`](@ref), which this calls internally (after
step 1 determines `k`) to override `points_per_segment` when `param_budget`
is given. Since this route is a closed-form linear solve (not iterative),
raising `param_budget` costs only a bigger (still cheap) linear system, not
a slower descent -- unlike [`_fit_composite_pulse_from_samples_learned`](@ref), where
`param_budget` exists mainly to keep `ForwardDiff`/Adam tractable at all.

Pass `segments` (the same `Vector{Tuple{Int,Int}}` [`_detect_subpulse_segments`](@ref)
itself returns) to SKIP step 1's own detection and use the given segments
directly -- for a caller (e.g. [`fit_composite_pulse_seed_linear_exact`](@ref))
that already ran detection itself against this SAME `(t, A)` to validate
`k`/size `points_per_segment` before calling this function, avoiding a
second identical `O(N)` amplitude-threshold scan. `rel_thresh`/
`min_active_samples`/`min_silence_samples` are ignored when `segments` is
given (nothing left for them to control).
"""
function _fit_composite_pulse_from_samples_linear(
    t::AbstractVector, I::AbstractVector, Q::AbstractVector, d;
    points_per_segment::Integer=6, degree::Integer=3, taper_frac::Real=0.1,
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
    cA_floor_frac::Real=_GRAD_SAFE_FRAC, cf_clip_mult::Real=20.0,
    param_budget::Union{Nothing,Integer}=nothing,
    segments::Union{Nothing,Vector{Tuple{Int,Int}}}=nothing,
)
    A = sqrt.(I .^ 2 .+ Q .^ 2)
    phi, f = _instantaneous_frequency(t, I, Q)

    segments = segments === nothing ? _detect_subpulse_segments(
        t, A; rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
    ) : segments
    isempty(segments) && error(
        "No active sub-pulses detected (rel_thresh=$rel_thresh too high relative to the " *
        "trace's own peak, or the trace really is all silence)."
    )
    k = length(segments)

    n_samples_each = [i_end - i_start + 1 for (i_start, i_end) in segments]
    points_per_segment = _resolve_points_per_segment(
        points_per_segment, param_budget, n_samples_each, k, degree, "fit_composite_pulse_from_samples (fit_mode=:linear)",
    )
    n_coeff = _spline_coeff_count(maximum(n_samples_each); points_per_segment=points_per_segment, degree=degree)

    pulse = CompositePulse(k, n_coeff, n_coeff, d; degree=degree, taper_frac=taper_frac)
    cA_floor = cA_floor_frac * pulse.amp_scale

    raw_gap = Vector{Float64}(undef, k)
    raw_dur = Vector{Float64}(undef, k)
    raw_phi0 = Vector{Float64}(undef, k)
    raw_cA = Matrix{Float64}(undef, n_coeff, k)
    raw_cf = Matrix{Float64}(undef, n_coeff, k)
    n_cA_floored = 0
    n_cf_clipped = 0
    A_resid = 0.0
    f_resid = 0.0
    phi_resid = 0.0
    f_weight_sum = 0.0
    t_prev_end = 0.0
    running_phase = 0.0

    for (idx, (i_start, i_end)) in enumerate(segments)
        t_s, t_e = t[i_start], t[i_end]
        duration = t_e - t_s
        gap = max(t_s - t_prev_end, 0.0)
        dur_arg = max(duration - pulse.dur_floor, 0.0)
        raw_gap[idx] = _encode_scaled_softplus(gap, pulse.gap_scale)
        raw_dur[idx] = _encode_scaled_softplus(dur_arg, pulse.dur_scale)

        # decode()'s OWN round trip of the encoding just above -- NOT
        # necessarily (t_s, t_e) themselves. _encode_scaled_softplus floors
        # its physical argument up to _GRAD_SAFE_FRAC*scale before encoding
        # (a deliberate decode-gradient-safety floor, see that function's
        # own docstring); whenever gap/dur_arg is smaller than that floor --
        # the common case for a sub-pulse whose true leading gap is only a
        # sample or two, e.g. sub-pulse 1 starting at/near t=0 -- decode(u)
        # reconstructs a t_start/t_end for THIS sub-pulse shifted from
        # (t_s, t_e) by the floored amount. Because decode's t_start is a
        # CUMULATIVE sum, that shift then carries forward UNCHANGED into
        # every later sub-pulse's own t_start/t_end too (their own gap/dur
        # encode exactly; they just inherit the previous sub-pulse's
        # shifted t_end as their own base). build_E_of_t evaluates each
        # sub-pulse's B-spline/antiderivative on knots built from those
        # DECODED times, sampled at the trace's own UNSHIFTED physical t --
        # so a fit built on knots spanning the raw sample bounds (t_s, t_e)
        # is silently evaluated by build_E_of_t on knots spanning
        # (t_s+shift, t_e+shift) instead: the same spline SHAPE, offset in
        # time. For a chirped drive (dPhi/dt not constant) a pure time
        # offset is NOT a constant phase error -- it is offset*f(t), which
        # sweeps across the sub-pulse exactly as fast as the chirp itself.
        # Verified directly on this package's own 3-ARP WURST reference
        # (run_105_3ARP_M20000): an unflagged ~1.17us leading-gap floor on
        # sub-pulse 1 -- invisible to rel_l2_A/rel_l2_f/phi_rms_rad, which
        # are computed from THIS loop's own (self-consistent but
        # wrongly-anchored) Phi_pred, never from build_E_of_t itself --
        # propagated onto every sub-pulse and produced a reconstructed
        # phase that swept a full extra 2*pi across each one
        # (rel_l2_complex~1.4 regardless of fit resolution). A GLOBAL
        # per-segment phase realignment cannot fix this: the residual
        # isn't a constant (confirmed: the optimal such correction came
        # out at a fraction of a degree and left rel_l2_complex unchanged)
        # -- only matching the fit's own domain to decode's actual domain
        # does. Recomputing (t_s, t_e) here from the SAME encode this loop
        # just performed (not from decode(pulse,u_fit) after the fact, so
        # no second pass / no extra CompositePulse-shaped allocation) makes
        # every downstream knot vector, taper window and antiderivative in
        # THIS sub-pulse's own fit exactly the domain build_E_of_t will
        # actually reconstruct on, for ANY gap/dur (floored or not) --
        # closing the gap at its root instead of its symptom.
        t_s = t_prev_end + pulse.gap_scale * _softplus(raw_gap[idx])
        t_e = t_s + pulse.dur_scale * _softplus(raw_dur[idx]) + pulse.dur_floor

        t_seg = view(t, i_start:i_end)
        A_seg = view(A, i_start:i_end)
        f_seg = view(f, i_start:i_end)
        phi_seg = view(phi, i_start:i_end)
        n_seg = length(t_seg)

        knots = make_clamped_knots(n_coeff, t_s, t_e, degree)
        Basis = Matrix{Float64}(undef, n_seg, n_coeff)
        @inbounds for j in 1:n_seg
            Basis[j, :] .= bspline_basis(t_seg[j], knots, degree)
        end

        taper_w = [_taper_window(t_seg[j], t_s, t_e, taper_frac) for j in 1:n_seg]
        M_A = Basis .* taper_w
        cA_seg = M_A \ collect(A_seg)
        n_cA_floored += count(<(cA_floor), cA_seg)
        cA_seg = max.(cA_seg, cA_floor)
        raw_cA[:, idx] .= _softplus_inv.(cA_seg ./ pulse.amp_scale)

        # Fit cf against the ACCUMULATED PHASE (the exact quantity
        # build_E_of_t's Φ(t)=∫f dτ reconstructs), not the pointwise
        # frequency curve -- see this function's own docstring for why a
        # pointwise-f fit can leave Φ drifting even when its own per-point
        # RMS looks excellent. `bspline_antiderivative`'s own degree-(p+1)
        # antiderivative construction, expressed here as an explicit
        # (n_coeff+1, n_coeff) linear map `L` (`d = L*cf`) so the whole
        # fit stays a single weighted linear least-squares solve; verified
        # bit-for-bit equivalent to `bspline_antiderivative`'s own
        # cumulative-sum recursion. The augmented constant column absorbs
        # `phi`'s own arbitrary reference point (whatever `atan` returned
        # at the trace's first sample) -- unrelated to and decoupled from
        # `build_E_of_t`'s own `phase_offset` bookkeeping, exactly the way
        # an intercept term in ordinary linear regression decouples a
        # slope fit from an unknown additive offset.
        knots_p1 = vcat(knots[1:1], knots, knots[end:end])
        Basis_p1 = Matrix{Float64}(undef, n_seg, n_coeff + 1)
        @inbounds for j in 1:n_seg
            Basis_p1[j, :] .= bspline_basis(t_seg[j], knots_p1, degree + 1)
        end
        L = zeros(Float64, n_coeff + 1, n_coeff)
        for j in 1:n_coeff
            width = (knots[j+degree+1] - knots[j]) / (degree + 1)
            L[j+1:end, j] .= width
        end
        M_Phi_base = Basis_p1 * L
        M_Phi = hcat(M_Phi_base, ones(n_seg))

        f_weight = A_seg .^ 2 .+ 1e-30
        sw = sqrt.(f_weight)
        M_Phi_weighted = M_Phi .* sw
        phi_target_weighted = collect(phi_seg) .* sw
        sol = M_Phi_weighted \ phi_target_weighted
        cf_seg = sol[1:n_coeff]
        phase_const = sol[end]

        cf_seg_raw = cf_seg ./ pulse.freq_scale
        n_cf_clipped += count(x -> abs(x) > cf_clip_mult, cf_seg_raw)
        raw_cf[:, idx] .= _clip_cf_raw(cf_seg_raw, cf_clip_mult)

        A_pred = M_A * cA_seg
        f_pred = Basis * cf_seg
        Phi_pred = M_Phi_base * cf_seg .+ phase_const
        A_resid += sum(abs2, A_pred .- A_seg)
        f_resid += sum(f_weight .* abs2.(f_pred .- f_seg))
        phi_resid += sum(f_weight .* abs2.(Phi_pred .- phi_seg))
        f_weight_sum += sum(f_weight)

        # `phase_const` IS the fitted value of build_E_of_t's own
        # `phase_offset[i]` (bspline_antiderivative references `d_f[1]=0`
        # at each sub-pulse's own t_start, so `Phi_pred` at t_start is
        # exactly `phase_const`) -- recover the DISCRETE JUMP `raw_phi0`
        # build_E_of_t actually adds (`phase_offset[i] = running+phi0[i]`)
        # by subtracting off what `running` will be at this point, tracked
        # here with the SAME recipe (`running = phase_offset[i]+d_f[end]`).
        # `d_f[end]` is [`bspline_antiderivative`](@ref)'s own last
        # coefficient -- by that function's cumulative-sum construction,
        # EXACTLY `bspline_area(cf_seg, knots, degree)` (the degree-`degree`
        # spline's full-domain integral, `Σ cf_seg[j]*width_j/(degree+1)`),
        # regardless of where any SAMPLE falls. Computed that way here too
        # (not as `Phi_pred[end]-phase_const`, a SAMPLE-point evaluation
        # that silently assumed `t_seg[end] == t_e` -- true only when this
        # sub-pulse's own `t_s` was NOT shifted by `_encode_scaled_softplus`'s
        # floor above; when it was, `t_e` moves but `t_seg[end]` (the raw
        # last sample) does not, so that sample no longer sits at the
        # spline's own right knot, and `Phi_pred[end]` silently stopped
        # being `d_f[end]` -- this broke `running_phase`, and hence every
        # LATER sub-pulse's own `raw_phi0`, exactly the same way the
        # `(t_s, t_e)` domain fix above addresses the WITHIN-sub-pulse
        # reconstruction: both are instances of the same root cause,
        # closed the same way -- match `build_E_of_t`'s own construction
        # exactly rather than a sample-grid coincidence).
        # `bspline_area` must be fed the ACTUAL coefficients `decode` will
        # hand back to `build_E_of_t` -- i.e. `raw_cf[:, idx]` (just clipped
        # above by `_clip_cf_raw`, for the rare coefficient that lands
        # outside `±cf_clip_mult*freq_scale`, typically a poorly-constrained
        # edge coefficient where `f_weight~0`), not the raw unclipped
        # `cf_seg` the LS solve returned. Using `cf_seg` here would silently
        # reintroduce this same function's own root cause one level up:
        # `running_phase` -- and hence every LATER sub-pulse's own
        # `raw_phi0` -- would accumulate a chirp integral `build_E_of_t`
        # never actually reconstructs, whenever this sub-pulse's own fit
        # triggered clipping.
        raw_phi0[idx] = phase_const - running_phase
        cf_seg_used = raw_cf[:, idx] .* pulse.freq_scale
        d_f_end = bspline_area(cf_seg_used, knots, degree)
        running_phase = phase_const + d_f_end

        t_prev_end = t_e
    end

    u_fit = pack(pulse, raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf)

    A_energy = sum(abs2, A) + 1e-30
    rel_l2_A = sqrt(A_resid / A_energy)
    rel_l2_f = sqrt(f_resid / (f_weight_sum + 1e-30)) / pulse.freq_scale
    phi_rms_rad = sqrt(phi_resid / (f_weight_sum + 1e-30))

    # Full-trace COMPLEX reconstruction error -- the one number that
    # actually reflects what a caller of build_E_of_t(pulse, u_fit) will
    # see, as opposed to rel_l2_A/rel_l2_f/phi_rms_rad, each of which only
    # checks one decoupled piece. Deliberately built from build_E_of_t
    # itself (not from any of this loop's own per-segment intermediates,
    # e.g. A_pred/Basis_p1/cf_seg, which are LOCAL to a single sub-pulse
    # and don't include phi0/taper/silence) -- this is the same quantity
    # [`fit_linear_seed`](@ref) reports as `fit_report.rel_l2_complex`;
    # CSV round-trip; computing it here too means a direct caller of this
    # function (no file I/O involved) gets it for free.
    E_fit = build_E_of_t(pulse, u_fit)
    E_pred = ComplexF64[E_fit(tt) for tt in t]
    E_tar = complex.(I, Q)
    complex_energy = sum(abs2, E_tar) + 1e-30
    rel_l2_complex = sqrt(sum(abs2, E_pred .- E_tar) / complex_energy)

    fit_report = (
        rel_l2_A=rel_l2_A, rel_l2_f=rel_l2_f, phi_rms_rad=phi_rms_rad,
        rel_l2_complex=rel_l2_complex, n_cA_floored=n_cA_floored, n_cf_clipped=n_cf_clipped,
    )
    return pulse, u_fit, fit_report, segments
end

"""
    fit_composite_pulse_from_samples(t, I, Q, d;
        fit_mode=:linear, points_per_segment=nothing, degree=3, taper_frac=0.1,
        rel_thresh=1e-3, min_active_samples=5, min_silence_samples=3,
        cf_clip_mult=20.0, param_budget=nothing, kwargs...)
        -> (pulse::CompositePulse, u_fit, fit_report, segments)

Builds a [`CompositePulse`](@ref) seed DIRECTLY from a sampled I/Q trace,
with NO prior knowledge of how many sub-pulses it contains or where they
are (segment detection via [`_detect_subpulse_segments`](@ref) -- `k` is
however many segments that finds, not a caller-supplied count). Dispatches
to one of two implementations, selected by `fit_mode`:

  - `fit_mode=:linear` (default): [`_fit_composite_pulse_from_samples_linear`](@ref)
    -- a closed-form, per-segment weighted least-squares solve (`cA` from the
    tapered amplitude trace, `cf` from the accumulated phase). No iterative
    descent, so it stays cheap even at hundreds of coefficients; this is the
    route this package's own optimisation pipeline
    ([`optimise_control_pulse_from_jld2`](@ref)) uses exclusively. Accepts
    `cA_floor_frac`/`segments` via `kwargs...`.
  - `fit_mode=:learned`: [`_fit_composite_pulse_from_samples_learned`](@ref)
    -- `ForwardDiff.gradient`+Adam descent (via [`fit_composite_pulse_af`](@ref)).
    Verified impractically slow once `n_coeff` reaches the tens (see that
    function's own docstring) -- kept for small/low-resolution fits or
    comparison against the closed-form route, not for a densely sampled real
    trace. Accepts `num_epochs`/`learning_rate`/`seed` via `kwargs...`.

`points_per_segment=nothing` resolves to each mode's own prior default (`6`
for `:linear`, `22` for `:learned`) -- pass an explicit value to override
either. `degree`/`taper_frac`/`rel_thresh`/`min_active_samples`/
`min_silence_samples`/`cf_clip_mult`/`param_budget` are shared by both modes
and forwarded as-is; see either implementation's own docstring for the full
mathematical detail and rationale (segment detection, spline construction,
gradient-safety floors/clips, `param_budget` sizing) -- none of that changed
by this dispatcher, which only selects which implementation runs.

Returns `(pulse, u_fit, fit_report, segments)` in both modes, though
`fit_report`'s own field set DIFFERS between them (see each implementation's
docstring) since the two fits report genuinely different diagnostics.
"""
function fit_composite_pulse_from_samples(
    t::AbstractVector, I::AbstractVector, Q::AbstractVector, d;
    fit_mode::Symbol=:linear,
    points_per_segment::Union{Nothing,Integer}=nothing,
    degree::Integer=3, taper_frac::Real=0.1,
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
    cf_clip_mult::Real=20.0,
    param_budget::Union{Nothing,Integer}=nothing,
    kwargs...,
)
    pps = points_per_segment === nothing ? (fit_mode === :linear ? 6 : 22) : points_per_segment
    if fit_mode === :linear
        return _fit_composite_pulse_from_samples_linear(
            t, I, Q, d; points_per_segment=pps, degree=degree, taper_frac=taper_frac,
            rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
            cf_clip_mult=cf_clip_mult, param_budget=param_budget, kwargs...,
        )
    elseif fit_mode === :learned
        return _fit_composite_pulse_from_samples_learned(
            t, I, Q, d; points_per_segment=pps, degree=degree, taper_frac=taper_frac,
            rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
            cf_clip_mult=cf_clip_mult, param_budget=param_budget, kwargs...,
        )
    else
        error("fit_mode must be :linear or :learned, got $(repr(fit_mode)).")
    end
end

# ============================================================
# INTERIOR SEED (map a fitted control seed toward inversion≈silencing≈0.5)
#
# A closed-form, non-optimising rewrite of an already-fitted `u_fit`: sample
# the physical envelope, scale its area onto a π/2-equivalent pulse, replace
# the (possibly chaotic) phase with a single linear chirp, then re-fit with
# the same gradient-safe linear solver as `fit_linear_seed`. Wired into
# `optimise_control_pulse_from_jld2` when `use_interior=true` (off by
# default); the linear seed is used unchanged when `use_interior=false`.
# ============================================================

"""
    _interior_amp_scale_factor(inversion, target_inversion) -> Float64

Amplitude multiplier that maps a resonant-pulse inversion `I = sin²(θ/2)`
onto `target_inversion` by scaling pulse area (`θ → θ_target`). For a π
pulse (`inversion = 1`) and `target_inversion = 0.5` this is exactly
`0.5` -- a π/2-area equivalent. `inversion = 0.5` already is that area,
so the factor is `1`.
"""
function _interior_amp_scale_factor(inversion::Real, target_inversion::Real)
    I = Float64(inversion)
    tgt = Float64(target_inversion)
    isfinite(I) || error("_interior_amp_scale_factor: inversion must be finite, got $inversion.")
    isfinite(tgt) || error(
        "_interior_amp_scale_factor: target_inversion must be finite, got $target_inversion.",
    )
    (0.0 <= I <= 1.0) || error(
        "_interior_amp_scale_factor: inversion must be in [0, 1], got $inversion.",
    )
    (0.0 < tgt <= 1.0) || error(
        "_interior_amp_scale_factor: target_inversion must be in (0, 1], got $target_inversion.",
    )
    I_floor = 1e-12
    I < I_floor && error(
        "_interior_amp_scale_factor: inversion=$inversion is too small to map onto a " *
        "π/2-area seed (need inversion > $I_floor).",
    )
    return asin(sqrt(tgt)) / asin(sqrt(clamp(I, I_floor, 1.0)))
end

"""
    generate_interior_seed(u_fit, inversion, silencing, pulse, d;
        target_inversion=0.5, amp_scale_factor=nothing,
        chirp_bandwidth=2π*1e6, N_samples=4000, param_budget=60,
        preserve_shape=false, degree=nothing, taper_frac=nothing, ...)
        -> (pulse_new, u_new, fit_report, segments)

Rewrite an already-fitted control seed `(u_fit, inversion=I, silencing=S)`
into a new seed whose physics sits near `(inversion, silencing) ≈
(target_inversion, 0.5)` -- typically `(0.5, 0.5)` -- without running
Adam / `pulse_cost`. `u_fit` is the raw parameter vector from
[`fit_composite_pulse_from_samples`](@ref) or
[`fit_composite_pulse_seed_linear_exact`](@ref); `pulse` is the
[`CompositePulse`](@ref) that vector belongs to; `d` is
`prepare_derived`'s ensemble (same object the original fit used).
`inversion` / `silencing` are that seed's current
[`pulse_metrics`](@ref) (`[0, 1]`). [`optimise_control_pulse_from_jld2`](@ref)
calls this after `fit_linear_seed` when `use_interior=true` (off by default),
using step 5's reference `inversion`/`silencing`.

Construction (linear least-squares re-fit, not a physics solve):

  1. Sample `build_E_of_t(pulse, u_fit)` on `N_samples` points over
     `[0, pulse.T_max]`.
  2. Scale the envelope by `amp_scale_factor`. When that keyword is
     omitted it is [`_interior_amp_scale_factor`](@ref)`(inversion,
     target_inversion)`: a π pulse (`I=1`) maps to a π/2-area pulse
     (factor `0.5`); a seed already at `I=0.5` is left at unit scale.
  3. Replace the sampled phase with a monotonic linear chirp of
     angular bandwidth `chirp_bandwidth` (default `2π * 1e6` rad/s)
     spanning the full window. `Φ(t) = ∫ ω(s) ds` with `ω` ramping
     from `-chirp_bandwidth/2` to `+chirp_bandwidth/2` -- the
     gradient-friendly regularizer that lands silencing near `0.5`
     instead of a chaotic echo phase. `silencing` is the current
     metric (required, validated, recorded); the chirp itself does
     not depend on its value. `chirp_bandwidth=0` keeps the original
     I/Q phase and only scales amplitude.
  4. Reconstruct Cartesian `(I, Q)` from the scaled envelope and
     chirped phase.
  5. Re-fit with the gradient-safe linear solver. Default
     (`preserve_shape=false`) is AUTO
     [`fit_composite_pulse_from_samples`](@ref) (`fit_mode=:linear`,
     `param_budget=60`) as in the original interior-seed construction.
     `preserve_shape=true` keeps `pulse`'s own
     `(k, n_coeff_A, n_coeff_f)` via the same linear fitter so `u_new`
     is a drop-in `warm_start_u` for the same `CompositePulse`.

Returns `(pulse_new, u_new, fit_report, segments)`. `fit_report` is the
linear fitter's own NamedTuple plus `inversion_in`, `silencing_in`,
`target_inversion`, `amp_scale_factor`, `chirp_bandwidth`, `N_samples`.
"""
function generate_interior_seed(
    u_fit::AbstractVector,
    inversion::Real,
    silencing::Real,
    pulse::CompositePulse,
    d;
    target_inversion::Real=0.5,
    amp_scale_factor::Union{Nothing,Real}=nothing,
    chirp_bandwidth::Real=2 * pi * 1e6,
    N_samples::Integer=4000,
    param_budget::Integer=60,
    preserve_shape::Bool=false,
    degree::Union{Nothing,Integer}=nothing,
    taper_frac::Union{Nothing,Real}=nothing,
    rel_thresh::Real=1e-3,
    min_active_samples::Integer=5,
    min_silence_samples::Integer=3,
    cf_clip_mult::Real=20.0,
)
    length(u_fit) == n_params(pulse) || error(
        "generate_interior_seed: u_fit has length $(length(u_fit)), but this CompositePulse " *
        "(k=$(pulse.k), n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)) needs " *
        "$(n_params(pulse)).",
    )
    N = Int(N_samples)
    N >= 2 || error("generate_interior_seed: N_samples must be >= 2, got $N_samples.")
    pulse.T_max > 0 || error(
        "generate_interior_seed: pulse.T_max must be positive, got $(pulse.T_max).",
    )
    S = Float64(silencing)
    isfinite(S) || error("generate_interior_seed: silencing must be finite, got $silencing.")
    (0.0 <= S <= 1.0) || error(
        "generate_interior_seed: silencing must be in [0, 1], got $silencing.",
    )
    bw = Float64(chirp_bandwidth)
    isfinite(bw) || error(
        "generate_interior_seed: chirp_bandwidth must be finite, got $chirp_bandwidth.",
    )
    bw >= 0 || error(
        "generate_interior_seed: chirp_bandwidth must be >= 0, got $chirp_bandwidth.",
    )

    scale = if amp_scale_factor === nothing
        _interior_amp_scale_factor(inversion, target_inversion)
    else
        s = Float64(amp_scale_factor)
        isfinite(s) && s > 0 || error(
            "generate_interior_seed: amp_scale_factor must be finite and > 0, got $amp_scale_factor.",
        )
        s
    end

    deg = degree === nothing ? pulse.degree : Int(degree)
    tap = taper_frac === nothing ? pulse.taper_frac : Float64(taper_frac)

    u_ref = collect(Float64, u_fit)
    t_grid = collect(range(0.0, pulse.T_max; length=N))
    E_ref = build_E_of_t(pulse, u_ref).(t_grid)
    A_mod = abs.(E_ref) .* scale

    if bw == 0.0
        I_mod = real.(E_ref) .* scale
        Q_mod = imag.(E_ref) .* scale
    else
        f_ramp = collect(range(-bw / 2, bw / 2; length=N))
        dt = pulse.T_max / (N - 1)
        Phi_mod = cumsum(f_ramp) .* dt
        I_mod = A_mod .* cos.(Phi_mod)
        Q_mod = A_mod .* sin.(Phi_mod)
    end

    if preserve_shape
        pulse.n_coeff_A == pulse.n_coeff_f || error(
            "generate_interior_seed: preserve_shape=true requires n_coeff_A == n_coeff_f " *
            "(got n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)); pass " *
            "preserve_shape=false to AUTO-size a new shape, or rebuild `pulse` with equal " *
            "coefficient counts.",
        )
        A_det = sqrt.(I_mod .^ 2 .+ Q_mod .^ 2)
        segments_det = _detect_subpulse_segments(
            t_grid, A_det; rel_thresh=rel_thresh, min_active_samples=min_active_samples,
            min_silence_samples=min_silence_samples,
        )
        length(segments_det) == pulse.k || error(
            "generate_interior_seed: preserve_shape=true detected $(length(segments_det)) " *
            "sub-pulse(s) but pulse.k=$(pulse.k) — pass preserve_shape=false, or adjust " *
            "rel_thresh/min_active_samples/min_silence_samples so detection matches k.",
        )
        n_pieces_target = pulse.n_coeff_A - deg
        n_pieces_target >= 1 || error(
            "generate_interior_seed: n_coeff_A=$(pulse.n_coeff_A) is too small for " *
            "degree=$deg (need n_coeff_A >= degree+1 = $(deg + 1)).",
        )
        n_samples_max = maximum(i_end - i_start + 1 for (i_start, i_end) in segments_det)
        pps = cld(n_samples_max, n_pieces_target)
        n_pieces_lo = cld(n_samples_max, pps)
        n_pieces_lo == n_pieces_target || error(
            "generate_interior_seed: n_coeff_A=$(pulse.n_coeff_A) is not achievable for a " *
            "segment of $n_samples_max samples at degree=$deg (points_per_segment-based " *
            "sizing has a gap here). Pass preserve_shape=false to AUTO-size, or change " *
            "N_samples.",
        )
        pulse_new, u_new, fit_report, segments = _fit_composite_pulse_from_samples_linear(
            t_grid, I_mod, Q_mod, d; points_per_segment=pps, degree=deg, taper_frac=tap,
            segments=segments_det, cf_clip_mult=cf_clip_mult,
        )
        (pulse_new.k == pulse.k && pulse_new.n_coeff_A == pulse.n_coeff_A &&
         pulse_new.n_coeff_f == pulse.n_coeff_f) || error(
            "generate_interior_seed: preserve_shape=true expected (k=$(pulse.k), " *
            "n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)) but the fit " *
            "produced (k=$(pulse_new.k), n_coeff_A=$(pulse_new.n_coeff_A), " *
            "n_coeff_f=$(pulse_new.n_coeff_f)).",
        )
    else
        pulse_new, u_new, fit_report, segments = fit_composite_pulse_from_samples(
            t_grid, I_mod, Q_mod, d;
            fit_mode=:linear, degree=deg, taper_frac=tap, param_budget=param_budget,
            rel_thresh=rel_thresh, min_active_samples=min_active_samples,
            min_silence_samples=min_silence_samples, cf_clip_mult=cf_clip_mult,
        )
    end

    report = merge(fit_report, (
        inversion_in=Float64(inversion),
        silencing_in=S,
        target_inversion=Float64(target_inversion),
        amp_scale_factor=scale,
        chirp_bandwidth=bw,
        N_samples=N,
        preserve_shape=preserve_shape,
    ))
    return pulse_new, u_new, report, segments
end

# ============================================================
# TASK-PARALLEL GRADIENT (2 ICs, optional multi-GPU param chunks)
#
# pulse_cost's own scalar `cost` is LITERALLY additively separable into a
# :ground-only term, a :weak-only term, and a direct (no-ODE-solve)
# term. ∇cost is therefore the sum of three independent ForwardDiff
# gradients. With no CUDA devices this is the original 2-way
# Threads.@threads split (CPU Dual ODE, bit-identical to the serial
# `ForwardDiff.gradient(pulse_cost)` path -- see runtests.jl). With 1-8
# GPUs, the same two IC terms are dispatched onto devices (one in-flight
# Dual ODE per GPU); leftover devices beyond 2 split the parameter index
# set so each GPU differentiates a disjoint block. Linearity of
# differentiation makes the assembled gradient exact for that Dual
# discretisation. The bspline scratch pool is keyed by thread id, so
# concurrent Dual RHS evaluations on different Julia threads do not
# alias buffers.
# ============================================================

"""
    _tmax_power_components(uu, pulse) -> (tmax_frac_sq, power_mean)

The two `pulse_cost`/[`_direct_cost_term`](@ref) pieces that don't already
have a dedicated getter (`pulse_duration` covers the third): `tmax_frac_sq
= (max(t_end[end]-T_max, 0)/T_max)^2` (the RAW `tmax_penalty`, before its
own `w_tmax` weight) and `power_mean = mean(|cA/amp_scale|^2)` (the RAW
`power_penalty`, before its own `w_power` weight). Both are cheap, pure
functions of `(uu, pulse)` alone -- no ODE solve, just `decode` and
arithmetic -- shared by `_direct_cost_term` and by
[`_reconstitute_static_direct_cost`](@ref)'s exact-annealing correction
(which needs them independent of whatever `w_tmax`/`w_power` the caller
happened to evaluate `pulse_cost` under).
"""
function _tmax_power_components(uu, pulse::CompositePulse)
    _, t_end, _, cA, _ = decode(pulse, uu)
    Tu = eltype(uu)
    n_cA = length(cA)
    n_cA > 0 || error("_tmax_power_components: decoded cA is empty.")
    tmax_excess = max(t_end[end] - pulse.T_max, zero(Tu))
    tmax_frac_sq = (tmax_excess / pulse.T_max)^2
    normalized_cA = cA ./ pulse.amp_scale
    power_mean = sum(abs2, normalized_cA) / n_cA
    return tmax_frac_sq, power_mean
end

function _direct_cost_term(uu, pulse::CompositePulse, w_time, w_power, w_tmax)
    dur = pulse_duration(pulse, uu)
    tmax_frac_sq, power_mean = _tmax_power_components(uu, pulse)
    return w_time * (dur / pulse.T_max) + w_tmax * tmax_frac_sq + w_power * power_mean
end

# ============================================================
# CURRICULUM ANNEALING OF w_time (the direct, non-ODE cost weight)
# (run_local_adam(anneal_direct_weights=true))
#
# Optional curriculum schedule: suppress w_time ONLY (w_tmax/w_power are
# NEVER annealed -- always the caller's own base weight) while the pulse
# is still physically bad, so early epochs' gradient descends mostly on
# inversion/silencing rather than being pulled toward "just make it
# short" before it even inverts anything, then let w_time ramp back up
# toward the caller's own configured weight as physics_cost improves.
# NEVER applies to hop==0 (factor pinned at 0.0 there; annealing only
# starts from hop 1 onwards -- see run_local_adam's own docstring).
# Detached from the epoch's own AD tape (built from the PREVIOUS accepted
# point's plain-Float64 metrics, never from the Dual/adjoint state being
# differentiated THIS epoch). Inside `run_local_adam`, Adam's own
# `best_cost`/`improved`/early-stopping comparisons use the RAW annealed
# cost (the same sandbox the gradient was computed under); when `w_time`
# changes between epochs, previously-recorded costs are shifted by the
# exact linear `w_time*(duration/T_max)` difference so those comparisons
# stay apples-to-apples under the CURRENT epoch's weight. The value
# returned one level up (basin-hopping's Metropolis test / `final_metrics`
# in `optimise_composite_pulse`/`optimise_composite_pulse_rjmcmc`) is a
# separate final evaluation of `best_u` under the caller's own static
# `w_time`, so hops are still compared on the SAME fixed objective.
# ============================================================

"""
    _schedule_shape(x_tune, fidelity_phys) -> Float64

The pure normalised-exponential curve [`_curriculum_fidelity_weight`](@ref)
applies to whatever `fidelity_phys` it computes from `(inv, sil,
target_F)`: `(exp(x_tune*fidelity_phys) - 1) / (exp(x_tune) - 1)`, linear
(`= fidelity_phys`) in the removable-singularity limit `x_tune -> 0`
(`abs(x_tune) < 1e-4`; see `_curriculum_fidelity_weight`'s own docstring
for why that threshold). Factored out so [`solve_optimal_x_start`](@ref)
-- which inverts this SAME shape for an already-known starting fidelity,
not the full `(inv, sil, target_F)` computation -- can reuse it directly,
rather than risk a second, independently-drifting copy of the formula.
Does NOT validate `x_tune` itself (`_curriculum_fidelity_weight` and
`solve_optimal_x_start` each do that at their own entry point, with their
own appropriate error messages).
"""
function _schedule_shape(x_tune::Real, fidelity_phys::Real)
    abs(x_tune) < 1e-4 && return Float64(fidelity_phys)
    return (exp(x_tune * fidelity_phys) - 1.0) / (exp(x_tune) - 1.0)
end

"""
    _curriculum_fidelity_weight(inv, sil, target_F, x_tune) -> Float64

Normalised `[0, 1]` annealing FACTOR for [`run_local_adam`](@ref)'s
`anneal_direct_weights=true` schedule -- multiplies `w_time` ONLY
(`w_tmax`/`w_power` are never annealed; this function itself is a general
`[0,1]` schedule curve, agnostic to what its one caller applies it to --
see `run_local_adam`'s own docstring for the current w_time-only scope):
`f(fidelity_phys) = (exp(x_tune*fidelity_phys) - 1) /
(exp(x_tune) - 1)`, where `fidelity_phys = inv*(1-(sil-target_F)^2)` is
[`pulse_cost`](@ref)'s own `fidelity_phys` formula evaluated at the
supplied `(inv, sil)` (the LAST ACCEPTED point's metrics, plain `Float64`
-- see this section's own comment for why that detachment matters).
`inv`/`sil` are clamped to `[0, 1]` (defensive against float overshoot;
`_weighted_inversion`/`_weighted_silencing_factor` already clamp their own
outputs there) and `NaN` (no accepted point yet, e.g. epoch 1) is treated
as `0` for each, i.e. worst-case `fidelity_phys=0`, the correct "haven't
measured any fidelity yet" starting point.

`f(0)=0`, `f(1)=1` for ANY `x_tune != 0`, so this always maps
`fidelity_phys ∈ [0,1]` onto `[0,1]` exactly like the caller's own weights
get scaled down at worst-case fidelity and fully restored at perfect
fidelity. `x_tune` shapes the curve in between: `x_tune>0` is convex
(back-loaded -- stays low until fidelity is already good), `x_tune<0` is
concave (front-loaded -- meaningful weight even at low fidelity, which is
what actually helps when a run gets stuck with fidelity pinned near 0:
`f'(0) = x_tune/(exp(x_tune)-1)`, which GROWS as `x_tune` becomes more
negative, unlike a `physics_loss`-gated schedule whose floor is
insensitive to any tuning once fidelity is stuck). `abs(x_tune) < 1e-4`
uses the removable-singularity limit `f(t) = t` directly (plain linear
interpolation) instead of evaluating the `0/0` form -- `1e-4` is far above
where `exp(x)-1` would lose precision to cancellation, so this is a
correctness special-case (avoiding an actual `0/0`), not a numerical
work-around. `abs(x_tune) >= 700` is rejected outright: `exp(700)` is
already within a factor of 5 of `Float64`'s overflow threshold, so larger
magnitudes risk silently returning `Inf`/`NaN`.
"""
function _curriculum_fidelity_weight(inv::Real, sil::Real, target_F::Real, x_tune::Real)
    isfinite(x_tune) || error("_curriculum_fidelity_weight: x_tune must be finite, got $x_tune.")
    abs(x_tune) < 700 || error(
        "_curriculum_fidelity_weight: |x_tune| must be < 700 (Float64 exp overflow), got $x_tune."
    )
    inv_c = isnan(inv) ? 0.0 : clamp(Float64(inv), 0.0, 1.0)
    sil_c = isnan(sil) ? 0.0 : clamp(Float64(sil), 0.0, 1.0)
    silencing_success = 1.0 - (sil_c - Float64(target_F))^2
    fidelity_phys = inv_c * silencing_success
    return _schedule_shape(x_tune, fidelity_phys)
end

"""
    _DEFAULT_X_TUNE_ALPHA

Default calibration target for [`run_local_adam`](@ref)'s mandatory
`x_tune_alpha` auto-calibration (used whenever `anneal_direct_weights=true`
and no explicit `x_tune_alpha` is passed, from hop 1 onwards -- annealing
never applies to hop 0, see `run_local_adam`'s own docstring): a small
guaranteed floor of `w_time` weight even from a near-zero starting
fidelity, protecting the physics-fidelity objective early without letting
the schedule collapse `w_time` to zero -- see
[`solve_optimal_x_start`](@ref)'s own docstring for the pathological
near-zero-fidelity case this guards against.
"""
const _DEFAULT_X_TUNE_ALPHA = 0.025

"""
    solve_optimal_x_start(F_0, alpha; x_max=100.0, tol=1e-6, max_iter=200) -> Float64

Finds the curvature `x_tune` such that [`_schedule_shape`](@ref) (the same
normalised exponential schedule [`_curriculum_fidelity_weight`](@ref)
applies) equals `alpha` at starting fidelity `F_0`. Meant to be called
ONCE, before a `run_local_adam` basin's own epoch loop starts (see
`run_local_adam(anneal_direct_weights=true, x_tune_alpha=alpha)`), against
`F_0` measured directly at that basin's own starting point -- so `x_tune`
is calibrated to what the run is ACTUALLY starting from, rather than
guessed blind.

For fixed `F_0 ∈ (0,1)`, `_schedule_shape(x, F_0)` is a strictly
monotonically DECREASING, continuous bijection from `x ∈ (-∞,∞)` onto
`factor ∈ (0,1)`: `factor -> 1` as `x -> -∞`, `factor == F_0` at `x=0`,
`factor -> 0` as `x -> +∞`. So for ANY `alpha ∈ (0,1)` there is exactly
one root: `alpha < F_0` needs POSITIVE `x` (back-loaded -- LESS relief
than a linear schedule); `alpha > F_0` needs NEGATIVE `x` (front-loaded
-- MORE relief than linear, i.e. real weight even at a low starting
fidelity -- the actually-useful regime for a seed that starts near-zero
fidelity, see [`_curriculum_fidelity_weight`](@ref)'s own docstring for
why that matters). An earlier version of this function only ever
bisected `x ∈ [1e-4, x_max]` -- the `alpha < F_0` half -- and silently
fell back to `x=1e-4` (near-linear) whenever `alpha >= F_0`, which is NOT
a solution in that regime, just the closest value reachable in the half
it happened to search. This version searches whichever half of the real
line actually contains the root (the comparison direction in the
bisection loop is identical on both halves, since `_schedule_shape` is
monotonic across the WHOLE real line, not just piecewise).

Degenerate `F_0`: at `F_0<=0` or `F_0>=1`, `_schedule_shape(x,F_0)` is
IDENTICALLY `0` or `1` respectively for every `x` -- no curvature can
reach any other `alpha`, so this returns the linear sentinel `1e-4`
rather than let a bisection run pointlessly. `alpha` within `tol` of
`F_0` similarly short-circuits to `1e-4` (`x=0`, the linear point,
already achieves it by definition). If `alpha` itself sits at or beyond
the OPEN endpoint on its side (e.g. `alpha=1` with `F_0<1`, which
strictly needs `x -> -∞`), no finite `x` reaches it exactly; the
bisection still runs and returns the nearest boundary (`±x_max`) reached
within `max_iter` -- the closest achievable curvature, not an exact
root, since none exists at finite `x`.

`x_max` must be `< 700` (`Float64`'s `exp` overflow threshold, the SAME
bound [`_curriculum_fidelity_weight`](@ref) enforces on its own `x_tune`)
-- above that, `_schedule_shape(x_max, F_0)` can silently evaluate
`Inf/Inf = NaN` at the search boundary, which the bisection's `val >
alpha` comparison would then treat as `false` (Julia's `NaN > x` is
always `false`) rather than erroring, letting the search silently return
a meaningless value instead of the intended boundary.

For ANY `F_0, alpha ∈ (0,1)` a finite root exists SOMEWHERE on the real
line (the strict-monotone-bijection argument above) -- so failing to
converge within the CALLER-supplied `x_max` (the default `100.0` is a
performance/typical-case choice, not a mathematical bound) means only
that `x_max` itself was too small for this particular `(F_0, alpha)`
pair, not that no root exists. An earlier version of this function
returned that unconverged boundary value SILENTLY, with no way for a
caller to distinguish "genuinely converged" from "clipped because x_max
was too small" -- confirmed reachable in normal use: `F_0=0.99`,
`alpha=_DEFAULT_X_TUNE_ALPHA=0.025` (the package's own default) needs
`x≈369`, so the default `x_max=100.0` silently returned `x=100.0` with
`_schedule_shape(100.0,0.99)≈0.37`, a `0.34`-off `factor` at that hop's
FIRST epoch -- nowhere near the intended `alpha=0.025` floor, and no
error or warning to flag it. Fixed here by auto-widening the search
bracket (geometric doubling, capped just under the hard `700` overflow
limit) whenever the bisection fails to converge within `tol` at the
current bound -- so the CALLER's `x_max` is only ever a performance
hint (skip searching wider than typically needed), never a silent
accuracy ceiling. Only a genuinely UNREACHABLE `alpha` (at/beyond the
open endpoint `0`/`1` on its side, needing `x -> ±∞` exactly) still
returns the nearest boundary after exhausting the full `(-700,700)`
range -- that case has no finite root to converge to, unlike the
too-small-`x_max` case this fixes.
"""
function solve_optimal_x_start(F_0::Real, alpha::Real; x_max::Real=100.0, tol::Real=1e-6, max_iter::Integer=200)
    0.0 <= F_0 <= 1.0 || error("solve_optimal_x_start: F_0 must be in [0,1], got $F_0.")
    0.0 <= alpha <= 1.0 || error("solve_optimal_x_start: alpha must be in [0,1], got $alpha.")
    x_max > 0 || error("solve_optimal_x_start: x_max must be > 0, got $x_max.")
    x_max < 700 || error(
        "solve_optimal_x_start: x_max must be < 700 (Float64 exp overflow, matching " *
        "_curriculum_fidelity_weight's own |x_tune|<700 contract), got $x_max. Above " *
        "this, _schedule_shape(x_max, F_0) can silently evaluate Inf/Inf=NaN inside " *
        "the bisection loop -- caught here rather than left to a stuck/NaN search."
    )
    tol > 0 || error("solve_optimal_x_start: tol must be > 0, got $tol.")

    F_0 <= 0.0 && return 1e-4   # factor ≡ 0 for every x -- nothing to solve
    F_0 >= 1.0 && return 1e-4   # factor ≡ 1 for every x -- nothing to solve
    abs(alpha - F_0) < tol && return 1e-4   # linear point (x≈0) already achieves this alpha

    hard_cap = 699.999   # strictly < 700, the exp-overflow contract shared with _curriculum_fidelity_weight
    bound = min(Float64(x_max), hard_cap)
    while true
        left, right = alpha < F_0 ? (1e-4, bound) : (-bound, -1e-4)
        for _ in 1:max_iter
            mid = (left + right) / 2.0
            val = _schedule_shape(mid, F_0)
            abs(val - alpha) < tol && return mid
            # _schedule_shape(., F_0) decreases monotonically in x across the
            # WHOLE real line (see docstring), so the comparison direction is
            # identical on both halves: too much relief (val > alpha) means x
            # needs to go UP; too little (val < alpha) means x needs to come
            # DOWN -- regardless of which half we started in.
            if val > alpha
                left = mid
            else
                right = mid
            end
        end
        # Did not converge within `bound` -- if we haven't yet searched out
        # to the hard cap, this pair's true root simply lies further out
        # than `bound` (see docstring: guaranteed to exist somewhere in
        # (-700,700) for any interior alpha), so widen and retry rather
        # than silently returning this bracket's midpoint. Once `bound`
        # itself has reached the hard cap, no further widening is possible
        # under the shared exp-overflow contract -- return the boundary as
        # the closest achievable value, exactly like the true "alpha at an
        # unreachable open endpoint" case already documented above.
        if bound >= hard_cap
            result = (left + right) / 2.0
            achieved = _schedule_shape(result, F_0)
            if abs(achieved - alpha) >= tol
                @warn "solve_optimal_x_start: no finite x_tune within the (-700,700) exp-overflow " *
                      "range reaches alpha=$alpha at F_0=$F_0 (this pair's true root lies beyond the " *
                      "representable domain, not merely beyond x_max) -- returning the closest " *
                      "achievable boundary x=$result (achieved factor=$(round(achieved, sigdigits=4)), " *
                      "target=$alpha). Only occurs for F_0/alpha extremely close to 0 or 1."
            end
            return result
        end
        bound = min(bound * 4.0, hard_cap)
    end
end

"""
    _reconstitute_static_direct_cost(dyn_cost, base_w_time, dyn_w_time, base_w_tmax, dyn_w_tmax,
                                      base_w_power, dyn_w_power, duration, tmax_frac_sq, power_mean, T_max) -> Float64

Exact (not approximate) conversion of a [`pulse_cost`](@ref) value computed
under `(w_time, w_tmax, w_power) = (dyn_w_time, dyn_w_tmax, dyn_w_power)`
back to what it would have been under the caller's own
`(base_w_time, base_w_tmax, base_w_power)`, WITHOUT a second ODE solve.
Valid because all three weights enter `pulse_cost` LINEARLY, each through
its own independent term (`w_time*(duration/T_max)`,
`w_tmax*tmax_frac_sq`, `w_power*power_mean` -- see
[`_tmax_power_components`](@ref)), and `physics_cost` depends on none of
them -- so for the SAME `u` (hence the same `duration`/`tmax_frac_sq`/
`power_mean`/`physics_cost`), the cost at any other triple of weights
differs from `dyn_cost` by exactly the sum of each term's own linear
correction. `dyn_cost=Inf` (a caught `PulseSolveFailed`, see
[`pulse_cost`](@ref)'s own failure contract) passes through unchanged
(`Inf` plus finite corrections is still `Inf`), correctly preserving
[`run_local_adam`](@ref)'s `!isfinite(cost)` failure check.
"""
function _reconstitute_static_direct_cost(dyn_cost::Real,
                                           base_w_time::Real, dyn_w_time::Real,
                                           base_w_tmax::Real, dyn_w_tmax::Real,
                                           base_w_power::Real, dyn_w_power::Real,
                                           duration::Real, tmax_frac_sq::Real, power_mean::Real, T_max::Real)
    return dyn_cost +
           (Float64(base_w_time) - Float64(dyn_w_time)) * (Float64(duration) / Float64(T_max)) +
           (Float64(base_w_tmax) - Float64(dyn_w_tmax)) * Float64(tmax_frac_sq) +
           (Float64(base_w_power) - Float64(dyn_w_power)) * Float64(power_mean)
end

function _gradient_on_indices(f, u::AbstractVector, idxs::Vector{Int})
    isempty(idxs) && return zeros(Float64, length(u))
    return _gradient_on_indices_val(f, u, idxs, Val{length(idxs)}())
end

function _gradient_on_indices_val(f, u::AbstractVector, idxs::Vector{Int}, ::Val{C}) where {C}
    n = length(u)
    u_host = collect(Float64, u)
    function f_reduced(u_chunk)
        T = eltype(u_chunk)
        uu = Vector{T}(undef, n)
        @inbounds for i in 1:n
            uu[i] = T(u_host[i])
        end
        @inbounds for j in 1:C
            uu[idxs[j]] = u_chunk[j]
        end
        return f(uu)
    end
    u_chunk0 = u_host[idxs]
    cfg = ForwardDiff.GradientConfig(f_reduced, u_chunk0, ForwardDiff.Chunk{C}())
    g_chunk = ForwardDiff.gradient(f_reduced, u_chunk0, cfg)
    g = zeros(Float64, n)
    @inbounds for j in 1:C
        g[idxs[j]] = g_chunk[j]
    end
    return g
end

"""
    _pulse_cost_grad_threaded(u, pulse, d; target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0, compute=:auto, kwargs...)
        -> (grad::Vector{Float64}, cost, inversion, silencing, duration, coherence, field_amp, weak_seed_retention)

Task-parallel drop-in for `ForwardDiff.gradient(uu -> pulse_cost(uu, pulse,
d; kwargs...)[1], u)` plus `pulse_cost`'s own aux outputs -- mathematically
EXACT on the host (not an approximation), and never touches `ForwardDiff`'s
own internal seeding/extraction machinery.

Unlike an earlier, additive version of this cost, [`pulse_cost`](@ref)'s
`physics_cost = (1 - inversion*silencing_success)^2` is NOT separable into
independent per-track pieces -- it couples `inversion` and `silencing`
multiplicatively. What IS still independent is the underlying ODE-solve
Jacobians `∇inversion(u)` and `∇silencing(u)` themselves (one per track,
each from its own `run_sim_1st_order_pure` solve with no shared state), so
this function still dispatches those two `ForwardDiff.gradient` calls
task-parallel exactly as before, then applies the closed-form chain rule
for `physics_cost` ANALYTICALLY afterward, on the host, to combine
`grad_I = ∇inversion` and `grad_F = ∇silencing` into `∇physics_cost`:

    silencing_success = 1 - (silencing - target_F)^2
    fidelity_phys     = inversion * silencing_success
    (coeff_I, coeff_S) = _fidelity_gradient_coefficients(inversion, silencing_success,
                                                          fidelity_phys, I_min, kappa_I, S_min, kappa_S)
    grad_Ssucc        = -2*(silencing - target_F) * grad_F
    ∇physics_cost     = coeff_I*grad_I + coeff_S*grad_Ssucc

`I_min`/`kappa_I`/`S_min`/`kappa_S` (defaults [`_DEFAULT_PENALTY_MIN`](@ref)/
[`_DEFAULT_PENALTY_KAPPA`](@ref) each) are the squared-hinge exterior
penalty -- see [`_fidelity_physics_cost`](@ref)/
[`_fidelity_gradient_coefficients`](@ref), the SAME helpers [`pulse_cost`](@ref)
and [`pulse_cost_grad_adjoint`](@ref) use, so this stays the analytically
exact gradient of `pulse_cost`'s own formula for any `I_min`/`kappa_I`/
`S_min`/`kappa_S`.

`∇cost = ∇physics_cost + ∇[direct term]` (the `w_time`/`tmax`/`power`
pieces, which never touch either ODE solve and are still handed to their
own ordinary `ForwardDiff.gradient` call). Uses ForwardDiff's own PUBLIC
`GradientConfig`/`Chunk` API to force a single chunk of width
`min(60, length(u))` instead of `ForwardDiff.pickchunksize`'s own default
(capped at 12 regardless of `n`). With 1-8 functional CUDA devices the
two ODE terms are pinned onto those devices (at most one Dual ODE in
flight per GPU); extra devices beyond the two ICs split the parameter
indices into disjoint blocks whose gradients sum to the full per-track
gradient. Host-only runs keep the original 2-way `Threads.@threads`
schedule and are the path `test/runtests.jl` checks against serial
ForwardDiff.

`track=:weak` (default) solves ONLY `:weak` and takes `grad_I`/`grad_F`
as the two rows of a single `ForwardDiff.jacobian` of
`[inversion, silencing]` (`inversion` read from that solve's own `Sz`,
O(ε) bias -- see [`_assert_track`](@ref)). Pass `track=:dual` explicitly
to solve both ICs as above; the analytic chain rule below is identical
either way, only the solve count changes. `:dual` is never the implicit
choice.
Neither track can be dropped from the OBJECTIVE (the multiplicative
`fidelity_phys` collapses to 0 without both), only the second SOLVE.
A `PulseSolveFailed` reproduces `pulse_cost`'s failure contract:
`(fill(NaN, n), Inf, NaN, NaN, duration, NaN, NaN, NaN)`.
"""
function _pulse_cost_grad_threaded(u::AbstractVector, pulse::CompositePulse, d;
                                    target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0,
                                    I_min::Real=_DEFAULT_PENALTY_MIN, kappa_I::Real=_DEFAULT_PENALTY_KAPPA,
                                    S_min::Real=_DEFAULT_PENALTY_MIN, kappa_S::Real=_DEFAULT_PENALTY_KAPPA,
                                    compute::Symbol=:auto, track::Symbol=:weak, kwargs...)
    _forbid_initial_condition(kwargs)
    _assert_track(track)
    sk = _solver_kwargs(kwargs)
    n = length(u)
    length(u) == n_params(pulse) || error(
        "_pulse_cost_grad_threaded: u has length $(length(u)), expected n_params=$(n_params(pulse))."
    )
    duration = pulse_duration(pulse, u)
    M = _assert_ensemble_shapes(d)
    compute_eff = _resolve_compute(compute, M)
    use_gpu_pool = compute_eff === :gpu && pulse_gpu_count() >= 1

    # Force a single ForwardDiff chunk covering ALL n_params whenever
    # n_params <= 60 -- this package's own established param_budget cap
    # (see fit_linear_seed / points_per_segment_for_budget),
    # so this covers this function's own intended caller
    # (run_local_adam(threaded_grad=true)) by construction in the common
    # AUTO-mode case. ForwardDiff.pickchunksize's own DEFAULT caps chunk
    # width at 12 regardless of n, splitting e.g. a 57-parameter gradient
    # into 5 chunks -- 5 FULLY REDUNDANT re-solves of the SAME adaptive
    # ODE integration (only which directions carry non-zero Dual partials
    # differs between chunks), each paying the solver's own step-size-
    # control/primal-recomputation overhead again. A single width-60 (or
    # width-n when n<60) chunk pays that ODE-solver control-flow overhead
    # ONCE per initial condition instead. `min(60, n)` (never a bare `60`)
    # is required for CORRECTNESS, not just tidiness -- ForwardDiff errors
    # outright if the requested chunk size exceeds `length(x)`.
    chunk = ForwardDiff.Chunk{min(60, n)}()

    function direct_only(uu)
        return _direct_cost_term(uu, pulse, w_time, w_power, w_tmax)
    end
    grad_direct = ForwardDiff.gradient(direct_only, u, ForwardDiff.GradientConfig(direct_only, u, chunk))
    direct_val = direct_only(u)

    aux_ground = Ref{Float64}(0.0)
    aux_weak = Ref{NTuple{4,Float64}}((0.0, 0.0, 0.0, 0.0))  # (silencing, coherence, field_amp, weak_seed_retention)

    if track === :weak
        # SINGLE-TRACK: one :weak solve. `inversion` is read from its own
        # `Sz`; `grad_I` and `grad_F` are BOTH rows of one Jacobian of
        # `[inversion, silencing]` w.r.t. `u`, so the analytic physics_cost
        # chain rule below is byte-for-byte the same as the :dual path --
        # only the number of ODE solves changes (1, not 2). No GPU-pool
        # param-chunk splitting here (nothing to split across ICs); a
        # compute=:gpu request still runs the single solve on-device.
        function weak_pair(uu)
            _, Sp, Sz_w, Nj_eq = run_sim_1st_order_pure(
                uu, pulse, d; compute=compute, sk..., initial_condition=:weak,
            )
            Tu = eltype(uu)
            inv_ = _weighted_inversion(Sz_w, d.g_b, Nj_eq, Tu)
            sil_ = _weighted_silencing_factor(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
            coh_ = _weighted_coherence(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
            famp_ = _weighted_field_amplitude(Sp, d.g_b, Nj_eq, Tu)
            ret_ = _weak_seed_retention(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
            aux_ground[] = Float64(ForwardDiff.value(inv_))
            aux_weak[] = (Float64(ForwardDiff.value(sil_)), Float64(ForwardDiff.value(coh_)),
                          Float64(ForwardDiff.value(famp_)), Float64(ForwardDiff.value(ret_)))
            return [inv_, sil_]
        end
        local J
        try
            J = ForwardDiff.jacobian(weak_pair, u, ForwardDiff.JacobianConfig(weak_pair, u, chunk))
        catch e
            e isa PulseSolveFailed || rethrow()
            return fill(NaN, n), Inf, NaN, NaN, duration, NaN, NaN, NaN
        end
        inversion = aux_ground[]
        silencing, coherence, field_amp, weak_seed_retention = aux_weak[]
        grad_I = J[1, :]
        grad_F = J[2, :]

    elseif !use_gpu_pool
        grads = Vector{Vector{Float64}}(undef, 2)
        failed = fill(false, 2)
        Threads.@threads for i in 1:2
            try
                if i == 1
                    # Extract ∇inversion independently
                    function ground_only(uu)
                        _, _, Sz, Nj = run_sim_1st_order_pure(
                            uu, pulse, d; compute=compute, sk..., initial_condition=:ground,
                        )
                        inv_ = _weighted_inversion(Sz, d.g_b, Nj, eltype(uu))
                        aux_ground[] = Float64(ForwardDiff.value(inv_))
                        return inv_
                    end
                    grads[1] = ForwardDiff.gradient(ground_only, u, ForwardDiff.GradientConfig(ground_only, u, chunk))
                else
                    # Extract ∇|F| independently
                    function weak_only(uu)
                        _, Sp, _, Nj_eq = run_sim_1st_order_pure(
                            uu, pulse, d; compute=compute, sk..., initial_condition=:weak,
                        )
                        Tu = eltype(uu)
                        sil_ = _weighted_silencing_factor(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
                        coh_ = _weighted_coherence(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
                        famp_ = _weighted_field_amplitude(Sp, d.g_b, Nj_eq, Tu)
                        ret_ = _weak_seed_retention(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
                        aux_weak[] = (Float64(ForwardDiff.value(sil_)), Float64(ForwardDiff.value(coh_)),
                                      Float64(ForwardDiff.value(famp_)), Float64(ForwardDiff.value(ret_)))
                        return sil_
                    end
                    grads[2] = ForwardDiff.gradient(weak_only, u, ForwardDiff.GradientConfig(weak_only, u, chunk))
                end
            catch e
                e isa PulseSolveFailed || rethrow()
                failed[i] = true
            end
        end
        any(failed) && return fill(NaN, n), Inf, NaN, NaN, duration, NaN, NaN, NaN

        inversion = aux_ground[]
        silencing, coherence, field_amp, weak_seed_retention = aux_weak[]
        grad_I = grads[1]
        grad_F = grads[2]

    else
        n_ic = 2
        n_gpu = pulse_gpu_count()
        n_chunks = max(1, n_gpu ÷ n_ic)
        ranges = _split_index_ranges(n, n_chunks)
        _log_pulse_compute_once(
            "pulse gradient: $n_gpu GPU(s), $(n_ic) IC track(s) × $(length(ranges)) param-chunk(s) " *
            "(M=$M, n_params=$n; start Julia with -t $n_gpu or more so device jobs overlap)"
        )

        jobs = NamedTuple[]
        for r in ranges
            push!(jobs, (kind=:ground, idxs=collect(r)))
            push!(jobs, (kind=:weak, idxs=collect(r)))
        end

        grad_I = zeros(n)
        grad_F = zeros(n)
        failed = Threads.Atomic{Bool}(false)
        ground_lock = ReentrantLock()
        weak_lock = ReentrantLock()

        _run_pulse_jobs!(jobs, (job, _dev) -> begin
            failed[] && return nothing
            try
                if job.kind === :ground
                    function ground_gpu(uu)
                        _, _, Sz, Nj = run_sim_1st_order_pure(
                            uu, pulse, d; compute=:gpu, sk..., initial_condition=:ground,
                        )
                        inv_ = _weighted_inversion(Sz, d.g_b, Nj, eltype(uu))
                        if first(job.idxs) == 1
                            aux_ground[] = Float64(ForwardDiff.value(inv_))
                        end
                        return inv_
                    end
                    g = length(job.idxs) == n ?
                        ForwardDiff.gradient(ground_gpu, u, ForwardDiff.GradientConfig(ground_gpu, u, chunk)) :
                        _gradient_on_indices(ground_gpu, u, job.idxs)
                    lock(ground_lock) do
                        grad_I .+= g
                    end
                else
                    function weak_gpu(uu)
                        _, Sp, _, Nj_eq = run_sim_1st_order_pure(
                            uu, pulse, d; compute=:gpu, sk..., initial_condition=:weak,
                        )
                        Tu = eltype(uu)
                        sil_ = _weighted_silencing_factor(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
                        coh_ = _weighted_coherence(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
                        famp_ = _weighted_field_amplitude(Sp, d.g_b, Nj_eq, Tu)
                        ret_ = _weak_seed_retention(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
                        if first(job.idxs) == 1
                            aux_weak[] = (Float64(ForwardDiff.value(sil_)), Float64(ForwardDiff.value(coh_)),
                                          Float64(ForwardDiff.value(famp_)), Float64(ForwardDiff.value(ret_)))
                        end
                        return sil_
                    end
                    g = length(job.idxs) == n ?
                        ForwardDiff.gradient(weak_gpu, u, ForwardDiff.GradientConfig(weak_gpu, u, chunk)) :
                        _gradient_on_indices(weak_gpu, u, job.idxs)
                    lock(weak_lock) do
                        grad_F .+= g
                    end
                end
            catch e
                e isa PulseSolveFailed || rethrow()
                failed[] = true
            end
            return nothing
        end)
        _reclaim_gpu_memory()

        failed[] && return fill(NaN, n), Inf, NaN, NaN, duration, NaN, NaN, NaN
        inversion = aux_ground[]
        silencing, coherence, field_amp, weak_seed_retention = aux_weak[]
    end

    # Explicit analytical application of the non-linear chain rule coupling the two tracks
    physics_cost, fidelity_phys, silencing_success =
        _fidelity_physics_cost(inversion, silencing, Float64(target_F), I_min, kappa_I, S_min, kappa_S)
    coeff_I, coeff_S = _fidelity_gradient_coefficients(inversion, silencing_success,
                                                        fidelity_phys, I_min, kappa_I, S_min, kappa_S)

    grad_S = -2.0 * (silencing - Float64(target_F)) .* grad_F
    grad_physics = coeff_I .* grad_I .+ coeff_S .* grad_S

    grad = grad_physics .+ grad_direct
    cost = physics_cost + direct_val
    return grad, cost, inversion, silencing, duration, coherence, field_amp, weak_seed_retention
end

# ============================================================
# LOCAL DESCENT WITH EARLY STOPPING + BASIN-HOPPING OUTER LOOP
#
# Same two-level structure as the Python port's _run_local_adam/
# optimise_composite_pulse: an early-stopped local Adam descent inside
# each "hop", wrapped in a random-restart-plus-Metropolis-acceptance
# outer loop (standard basin-hopping) with its own early stopping.
# ============================================================

"""
    run_local_adam(u_start, pulse, d, cost_kwargs; hop=0, num_epochs=30, patience=5, tol=1e-3,
        learning_rate=0.05, cf_lr_scale=1.0, label="", threaded_grad=false, compute=:auto,
        grad_mode=:forwarddiff, anneal_direct_weights=true,
        x_tune_alpha=_DEFAULT_X_TUNE_ALPHA, kwargs...)
        -> (best_u, best_cost, best_inversion, best_silencing, best_duration, history)

One basin's local descent: Adam from `u_start`, stopped either after
`num_epochs` or after `patience` consecutive epochs without a cost
improvement of at least `tol` (whichever comes first). Each epoch takes
cost, metrics, and gradient from a single `ForwardDiff.gradient` sweep
(metrics via `ForwardDiff.value`) -- or, when `threaded_grad=true`, from
[`_pulse_cost_grad_threaded`](@ref) instead: a mathematically EXACT,
task-parallel substitute (see that function's own docstring) that
dispatches the epoch's two independent ODE solves (`:ground`, `:weak`)
across `Threads.@threads`, for roughly a 2x wall-clock speedup on the
gradient step -- the dominant cost of an epoch at a large ensemble --
when Julia is started with `-t N`/`JULIA_NUM_THREADS=N>=2` (a no-op,
still-correct fallback to serial execution otherwise). Default `false`:
zero behaviour change unless a caller opts in. A failed ODE (`PulseSolveFailed`,
reported as `Inf` cost by [`pulse_cost`](@ref)) skips the Adam step,
reverts `u` to the last point that DID solve successfully, and halves the
step size for the next attempt -- a standard backtracking-line-search
response. Reverting alone (without also shrinking the step) would not
help: `Adam`'s own momentum (`state.m`/`state.v`) is untouched while a
failing step is skipped, so recomputing the gradient at the same reverted
point with that same frozen momentum would just reproduce the identical
failing step deterministically, looping until `patience` runs out for no
benefit. The step size regrows only GRADUALLY on success (`*1.5`, capped
at `learning_rate`), not snapped straight back -- snapping back
immediately has the identical looping failure mode, since the very next
epoch after a revert is just a re-confirmation of the same already-known-
good point: an instant full-strength reset there would hand `adam_step!`
the same gradient and the same frozen momentum that caused the failure in
the first place, stepping right back into it. Gradual regrowth lets the
step size settle at whatever scale actually stays inside the feasible
region near a hard boundary (e.g. repeatedly overshooting `pulse.T_max`),
so the basin can keep making progress up to that boundary instead of
stalling.

Retrying is NOT re-evaluated from scratch: `last_good_u` is the exact
point already solved (one ODE solve + AD gradient) the moment it was
first accepted, so every backoff retry from it reuses that cached
`(grad, cost, inversion, silencing, duration)` instead of re-running an
identical, deterministic, and potentially expensive (`MaxIters`-bound
before it even reports failure) computation for a point already known.
`Adam`'s own `(m, v, t)` are similarly snapshotted the moment `last_good_u`
is accepted and restored before every retry from it, so a retry blends
that one cached gradient into momentum exactly once, the same as an
ordinary step would -- not once per retry attempt (repeatedly blending an
identical gradient into the EMA on every backoff would let momentum drift
toward that single gradient direction, discarding the history a real
sequence of steps would have preserved). Returns this basin's own
best point -- the caller is responsible for tracking the GLOBAL best across
basins (a basin's local best is not necessarily better than a previous
basin's) -- plus `history`, a `Vector{<:NamedTuple}` with one entry per
epoch actually run
(`hop, epoch, k, cost, inversion, silencing, duration, coherence,
field_amp, weak_seed_retention, improved, x_tune, schedule_factor`),
tagged with the
caller-supplied `hop` index so a caller accumulating history across many
basins can tell which hop each row came from. `k` (the sub-pulse count,
from `pulse.k`) is recorded on every row too, even though it's constant
within a single call, so history rows stay self-describing if ever
concatenated across runs with a different `k` (e.g. a later warm-started
continuation using a different `CompositePulse` shape). `coherence` is
[`pulse_cost`](@ref)'s DIAGNOSTIC-ONLY per-bin `|Sp|/(Nj/2)` average from
the same `:weak` solve as `silencing` (see [`_weighted_coherence`](@ref))
-- recorded for comparison alongside the collective `|F|` actually being
optimised, never part of `cost` itself. `x_tune` is `x_tune_eff` (the
possibly-calibrated value actually driving THIS epoch's schedule -- see
`x_tune_alpha` below; always `0.0` on hop 0, see there too) and
`schedule_factor` is `f(x_tune, fidelity_phys)` itself, i.e. the
[`_curriculum_fidelity_weight`](@ref) value computed from the LAST
ACCEPTED point (`0.0` whenever `anneal_direct_weights=false` OR `hop==0`,
since the factor is computed unconditionally every epoch regardless -- see
the comment at its computation site) -- both recorded on every row so a
caller can see exactly what shaped that epoch's gradient without
recomputing it.

`grad_mode` (default `:forwarddiff`) selects the ODE gradient. `:forwarddiff`
is the production Dual-Tsit5 path (`threaded_grad` applies here only).
`:adjoint` uses [`pulse_cost_grad_adjoint`](@ref)'s frozen-mesh discrete
Tsit5 VJP instead — a different derivative object than Dual-Tsit5;
`threaded_grad` is ignored for that mode. Do not pass `:adjoint` unless
you explicitly want that object.

`cf_lr_scale` (default `1.0`, i.e. no change from the original uniform-`lr`
behaviour) multiplies the effective step size for the CHIRP/frequency
coefficients (`raw_cf`) only -- `gap`/`duration`/`phi0`/`cA` always step at
the full `learning_rate` (`phi0` is a one-shot discrete phase jump, not an
integrated periodic quantity like `cf`, so the periodic-runaway motivation
below doesn't apply to it either). Motivation: `raw_cf` enters the physical drive
through an EXACT phase integral (`build_E_of_t`'s `Φ(t) = ∫f dτ`, via
`bspline_antiderivative`), and the cost depends on that phase only through
`exp(iΦ(t))` -- a periodic, non-convex function of `raw_cf`. `adam_step!`'s
own per-parameter second-moment normalisation already keeps every raw
parameter's step size close to `lr` in magnitude regardless of its raw
gradient scale, so this is NOT compensating for `cf`'s gradient being
larger or smaller than `gap`/`dur`/`cA`'s -- it is deliberately slowing
descent along the periodic sub-manifold specifically, so a single epoch's
step is less likely to carry `Φ(t)` across a `2π` boundary and land in an
entirely different (and possibly worse) local phase-alignment than the one
the rest of `u` was descending toward. Pass e.g. `cf_lr_scale=0.1` to
soften this; the right value is problem-dependent (how large `pulse.
freq_scale*duration` is relative to `2π` for your own config), so no
non-`1.0` value is asserted as a universal default here -- watch
`history`'s `cost`/`silencing` columns for erratic (non-monotone,
large-swing) epoch-to-epoch behaviour as the signal that a smaller
`cf_lr_scale` is worth trying.

`anneal_direct_weights` (default `true`) anneals `w_time` (`w_tmax`/
`w_power` are NEVER annealed -- they always run at the caller's own base
weight, every epoch) via [`_curriculum_fidelity_weight`](@ref): each
epoch's GRADIENT is computed under `w_time = factor * base_w_time`, where
`factor = (exp(x_tune*fidelity_phys)-1)/(exp(x_tune)-1) ∈ [0,1]` and
`fidelity_phys` is [`pulse_cost`](@ref)'s own fidelity from the LAST
ACCEPTED point (detached from this epoch's own AD tape -- see that
function's docstring). So early epochs -- where `fidelity_phys` starts
near its worst case (`0`) -- descend mostly on inversion/silencing, only
picking up the full `w_time` duration penalty once fidelity is actually
good (`factor=1` exactly at `fidelity_phys=1`). Pass
`anneal_direct_weights=false` to disable this entirely and recover the
original fixed-`w_time` cost. `x_tune` (the curvature shaping that curve --
negative front-loads it, positive back-loads it; see
[`_curriculum_fidelity_weight`](@ref) for the exact shape) is not a
keyword here -- it is ALWAYS derived via mandatory calibration, see
`x_tune_alpha` below. Critically, THIS basin's own `best_cost`/`improved`/
early-stopping comparisons use the RAW annealed cost off the tape -- the
same sandbox the gradient was computed under -- not a reconstituted
static value. When `dyn_w_time` changes between epochs, previously-recorded
`best_cost` and `last_good_aux` are shifted by the exact linear
`w_time*(duration/T_max)` difference so those comparisons stay
apples-to-apples under the CURRENT epoch's `w_time`. That lets hop 0
(where `dyn_w_time=0`) accept duration-expanding steps that improve
physics, while later hops naturally tighten duration as `w_time` grows.
The value RETURNED to
[`optimise_composite_pulse`](@ref)/[`optimise_composite_pulse_rjmcmc`](@ref)
is a separate final evaluation of `best_u` under the caller's own static
`cost_kwargs` (full `w_time`), so basin-hopping's Metropolis test and
`final_metrics` still compare hops on the SAME fixed objective.

**With `hop0_phyonly=true` (the default), `hop==0` ALWAYS runs at
`dyn_w_time == 0.0` -- a HARD INVARIANT for that mode, not something
`anneal_direct_weights=false` can override.** `factor` is pinned at `0.0`,
AND `dyn_w_time` is computed as an explicit `(hop==0 && hop0_phyonly) ? 0.0
: base_w_time*factor` short-circuit (not left to emerge from `factor==0`)
and `@assert`ed, so a future change to `factor` cannot silently reintroduce
a `w_time` term on a suppressed hop 0. hop 0's `w_time` contribution to the
gradient AND to `best_cost`/`improved` is fully suppressed (genuinely
physics-only optimisation -- duration ignored -- for the entire first hop)
REGARDLESS of `anneal_direct_weights`'s own value -- unlike every other
hop, where `anneal_direct_weights=false` means the ORIGINAL "disable
annealing, `factor=1.0`, recover the full fixed `w_time`" contract exactly
as before this feature existed. These are two genuinely different `factor`
defaults (`0.0` vs `1.0`) for two different reasons -- do not conflate
them. No calibration `pulse_cost` evaluation is spent on a suppressed hop 0
either way (nothing to calibrate `x_tune` for when `factor` is pinned
regardless of `x_tune_alpha`). Annealing (in the ordinary,
`anneal_direct_weights`-gated sense) then only starts from **hop 1
onwards**.

**`hop0_phyonly=false`** removes this special case: hop 0 becomes a normal
scheduled hop -- `dyn_w_time = base_w_time*factor` with `factor` from the
curriculum anneal (`anneal_direct_weights=true`, one calibration
`pulse_cost` eval at hop 0 to solve `x_tune`, like hop 1) or `1.0`
(`anneal_direct_weights=false`). Use it when hop 0's search should already
weigh pulse duration, not just fidelity. `hop` is this function's own
existing parameter (defaulting to `0`), so a **standalone call with no
explicit `hop`** (e.g. `run_local_adam(u0, pulse, d, cost_kwargs)`) always
has `w_time` suppressed to `0.0` this way -- pass `hop=1` (or any nonzero
`hop`) explicitly to exercise ordinary annealing outside a basin-hopping
pipeline. See
[`optimise_composite_pulse`](@ref)/[`optimise_composite_pulse_rjmcmc`](@ref)
for how hop 1 becomes their own calibration seed.

`x_tune_alpha` (default [`_DEFAULT_X_TUNE_ALPHA`](@ref) -- so calibration
is MANDATORY BY DEFAULT whenever `anneal_direct_weights=true` AND `hop !=
0`) picks `x_tune` via [`solve_optimal_x_start`](@ref): before this basin's
epoch loop starts, `u_start` is evaluated ONCE via [`pulse_cost`](@ref)
(one extra ODE-solve pair, the same cost as evaluating any other epoch) to
get this basin's OWN starting `fidelity_phys = F_0`, then
`solve_optimal_x_start(F_0, x_tune_alpha)` picks the `x_tune` that makes
the annealing factor equal `x_tune_alpha` AT THAT STARTING POINT
specifically -- so the schedule is calibrated to what this basin actually
starts from (which can differ hop to hop, e.g. across a
`_grow_pulse`/`_shrink_pulse` k-change in
[`optimise_composite_pulse_rjmcmc`](@ref)) rather than a single blind
guess reused everywhere. This is logged via `label` before the epoch loop
starts. That same calibration `pulse_cost` evaluation's `(inversion,
silencing)` are planted into `last_good_aux`'s corresponding slots, so
epoch 1 already sees `fidelity_phys = F_0` and its `schedule_factor`
equals `x_tune_alpha` (the calibrated non-zero floor) -- the schedule is
active from the FIRST epoch of every calibrating hop, not first applied on
epoch 2. Passing `x_tune_alpha=nothing` EXPLICITLY skips calibration and
uses the plain linear schedule (`x_tune=0`, i.e. `factor = fidelity_phys`
exactly) instead -- the only way to run the annealed schedule
uncalibrated; not something that happens by leaving arguments at their
defaults. `x_tune_alpha` is a silent no-op whenever
`anneal_direct_weights=false` or `hop==0` (nothing to calibrate `x_tune`
FOR in either case).

`_precalibrated_x_tune` (default `nothing`, internal -- do not pass this
directly) lets [`optimise_composite_pulse`](@ref)/
[`optimise_composite_pulse_rjmcmc`](@ref) inject an already-calibrated
`x_tune` (computed by them from hop 1's own starting point) for their own
`recalibrate_optima_x=false` hops (hop 2 onwards), bypassing calibration
here entirely (takes priority over `x_tune_alpha`, which is ignored when
this is set). Because no calibration `pulse_cost` is spent on these hops,
there is no `(inversion, silencing)` seed for `last_good_aux`, so epoch 1
of a `_precalibrated_x_tune` hop keeps `schedule_factor == 0.0` (i.e.
`dyn_w_time == 0` for that one epoch) -- a deliberately accepted asymmetry
vs the calibrating hops above.

`I_min`/`kappa_I`/`S_min`/`kappa_S` (the squared-hinge exterior penalty on
`inversion`/`silencing_success` respectively, see
[`_fidelity_physics_cost`](@ref)) are NOT separate keywords here -- like
`target_F`/`w_tmax`/`w_power`/`w_time`'s own base values, they travel
through inside the caller's own `cost_kwargs` (defaulting to
[`_DEFAULT_PENALTY_MIN`](@ref)/[`_DEFAULT_PENALTY_KAPPA`](@ref) on
[`pulse_cost`](@ref) itself if the caller's `cost_kwargs` doesn't include
them). This is a SEPARATE mechanism from `anneal_direct_weights`/
`x_tune`, and NOT gated by `hop==0`; it applies identically on every hop,
including hop 0, and identically every epoch (unlike the earlier
`p_exp`/`q_exp` exponent barrier this replaces, there is no per-epoch
schedule or `last_good_aux` detachment here: the penalty is evaluated
directly on the CURRENT epoch's own live inversion/silencing_success,
inside whichever `grad_mode` backend that epoch uses). Because the
penalty is static, `cost` needs no barrier-specific reconstitution step
any more -- it is the SAME formula every epoch, so the sandbox
`best_cost` comparisons (and the returned static cost's Metropolis/
`_extract_physics_cost` comparisons one level up) include it directly.
Include `kappa_I=0, kappa_S=0` in `cost_kwargs` to disable the penalty
entirely and recover `pulse_cost`'s plain `inversion*silencing_success`
formula exactly.
"""
function run_local_adam(u_start::AbstractVector, pulse::CompositePulse, d, cost_kwargs::NamedTuple;
                         hop::Integer=0, num_epochs::Integer=30, patience::Integer=5, tol::Real=1e-3,
                         learning_rate::Real=0.05, cf_lr_scale::Real=1.0, label::AbstractString="",
                         threaded_grad::Bool=false, compute::Symbol=:auto, grad_mode::Symbol=:forwarddiff,
                         anneal_direct_weights::Bool=true, hop0_phyonly::Bool=true,
                         x_tune_alpha::Union{Nothing,Real}=_DEFAULT_X_TUNE_ALPHA,
                         _precalibrated_x_tune::Union{Nothing,Real}=nothing,
                         solve_kwargs...)
    _forbid_initial_condition(solve_kwargs)
    (grad_mode === :forwarddiff || grad_mode === :adjoint) || error(
        "grad_mode must be :forwarddiff or :adjoint, got $(repr(grad_mode))."
    )
    u = copy(u_start)
    n = length(u)
    adam = AdamState(n)
    lr_scale = cf_lr_scale == 1.0 ? nothing : pack(
        pulse, ones(pulse.k), ones(pulse.k), ones(pulse.k), ones(pulse.n_coeff_A, pulse.k),
        fill(cf_lr_scale, pulse.n_coeff_f, pulse.k),
    )
    aux = Ref{NTuple{7,Float64}}((NaN, NaN, NaN, NaN, NaN, NaN, NaN))  # cost, inv, sil, dur, coh, field_amp, weak_seed_retention
    base_w_time = haskey(cost_kwargs, :w_time) ? Float64(cost_kwargs.w_time) : 0.15
    target_F_val = haskey(cost_kwargs, :target_F) ? Float64(cost_kwargs.target_F) : 1.0

    # x_tune is never a caller-supplied keyword -- it is ALWAYS one of:
    # (0) forced to the linear sentinel 0.0 on a `hop==0 && hop0_phyonly` hop
    # -- the physics-only sandbox suppresses annealing entirely, so no
    # calibration pulse_cost evaluation is spent on it at all. (`hop0_phyonly`
    # defaults true; pass `hop0_phyonly=false` to make hop 0 a normal
    # scheduled hop that calibrates x_tune from its own u_start like hop 1.)
    # Otherwise: (1) an
    # already-calibrated value injected by a pipeline caller
    # (_precalibrated_x_tune, for optimise_composite_pulse/
    # optimise_composite_pulse_rjmcmc's own recalibrate_optima_x=false
    # hops, hop>=2); (2) freshly calibrated here, from THIS basin's own
    # starting fidelity, before the epoch loop starts (x_tune_alpha default
    # _DEFAULT_X_TUNE_ALPHA, so this is MANDATORY whenever
    # anneal_direct_weights=true && hop!=0); or (3) the plain linear
    # sentinel 0.0, when x_tune_alpha is explicitly nothing or
    # anneal_direct_weights is false. One extra pulse_cost evaluation at
    # u_start for case (2), same cost as any other epoch; if u_start itself
    # fails to solve, solve_optimal_x_start's own F_0∈[0,1] validation
    # rejects the resulting NaN loudly rather than silently calibrating
    # against garbage.
    # Seed for last_good_aux's (inversion, silencing) slots, so epoch 1 of a
    # calibrating hop already sees a real fidelity_phys (=F_0) instead of the
    # NaN->0 sentinel -- i.e. the calibrated x_tune is actually EXERCISED on
    # epoch 1 (schedule_factor = x_tune_alpha, the deliberate non-zero floor),
    # not first applied on epoch 2. Only set in the branch that already spends a
    # calibration pulse_cost eval (no extra ODE solve). Left NaN for hop==0 (no
    # annealing), for _precalibrated_x_tune hops (recalibrate_optima_x=false --
    # no eval spent, so no seed available; epoch 1 keeps w_time=0 there), and
    # for the uncalibrated linear fallback.
    seed_inv, seed_sil = NaN, NaN
    x_tune_eff = if hop == 0 && hop0_phyonly
        0.0
    elseif _precalibrated_x_tune !== nothing
        Float64(_precalibrated_x_tune)
    elseif x_tune_alpha !== nothing && anneal_direct_weights
        _, inv0, sil0, _, _, _ = pulse_cost(u_start, pulse, d; cost_kwargs..., compute=compute, solve_kwargs...)
        silencing_success0 = 1.0 - (Float64(sil0) - target_F_val)^2
        F_0 = Float64(inv0) * silencing_success0
        val = solve_optimal_x_start(F_0, Float64(x_tune_alpha))
        if isfinite(inv0) && isfinite(sil0)
            seed_inv, seed_sil = Float64(inv0), Float64(sil0)
        end
        println(
            "$label x_tune_alpha=$(x_tune_alpha): calibrated x_tune=$(round(val, sigdigits=4)) " *
            "from F_0=$(round(F_0, sigdigits=4)) at u_start"
        )
        val
    else
        0.0
    end

    best_u = copy(u_start)
    best_cost, best_inv, best_sil, best_dur = Inf, 0.0, 0.0, 0.0
    epochs_since_improve = 0
    history = NamedTuple[]
    last_good_u = copy(u_start)
    last_good_grad = zeros(n)
    # (cost, inversion, silencing, duration, coherence, field_amp, weak_seed_retention). cost stays
    # NaN so the moving-target isfinite(last_good_aux[1]) guard skips at epoch 1
    # and a failed epoch-1 solve still degrades to the revert loop. inversion/
    # silencing carry the calibration seed (or NaN when none was taken) -- these
    # are the only two slots _curriculum_fidelity_weight reads.
    last_good_aux = (NaN, seed_inv, seed_sil, NaN, NaN, NaN, NaN)
    adam_m0 = zeros(n)
    adam_v0 = zeros(n)
    adam_t0 = 0
    lr = learning_rate
    just_reverted = false
    prev_dyn_w_time = base_w_time

    for epoch in 1:num_epochs
        t_wall = time()

        # anneal_direct_weights: shape THIS epoch's gradient by annealing
        # w_time ONLY (w_tmax/w_power are never annealed -- always base_w_*)
        # via a factor computed from the LAST ACCEPTED point's own
        # (plain-Float64, AD-detached) metrics -- see
        # _curriculum_fidelity_weight's own docstring. TWO DIFFERENT
        # defaults, not one: a `hop==0 && hop0_phyonly` hop pins factor at 0.0
        # (w_time fully suppressed for the entire first hop, regardless of
        # anneal_direct_weights -- the physics-only sandbox; annealing only
        # starts from hop 1), whereas anneal_direct_weights=false at any
        # scheduled hop is the ORIGINAL "disable annealing, recover the full
        # fixed w_time" contract, i.e. factor=1.0 (NOT 0.0 -- these two "off"
        # cases are NOT the same value and must not be conflated). With
        # `hop0_phyonly=false`, hop 0 is a normal scheduled hop and takes the
        # anneal/1.0 branch like any other. Computed every epoch (including
        # just_reverted retries): the moving-target shift below needs the
        # current dyn_w_time even when the gradient itself is reused.
        factor = if hop == 0 && hop0_phyonly
            0.0
        elseif anneal_direct_weights
            _curriculum_fidelity_weight(last_good_aux[2], last_good_aux[3], target_F_val, x_tune_eff)
        else
            1.0
        end
        # INVARIANT (when hop0_phyonly): hop 0 is the pure physics sandbox --
        # dyn_w_time is identically 0 for the WHOLE hop (gradient AND
        # accept/reject), no schedule, no calibration, regardless of
        # anneal_direct_weights. Written as an explicit short-circuit (not left
        # to emerge from factor==0) so a future change to `factor` cannot
        # silently break it; the assert locks it. The epoch_cost_kwargs merge
        # below is unconditional, so cost_kwargs.w_time can never leak into a
        # suppressed hop-0 epoch either; and the moving-target block is inert
        # there (dyn_w_time is 0 every epoch and prev_dyn_w_time becomes 0
        # after epoch 1, so it never fires). `hop0_phyonly=false` opts hop 0
        # out of all of this -- it then behaves exactly like hop 1+.
        dyn_w_time = (hop == 0 && hop0_phyonly) ? 0.0 : base_w_time * factor
        @assert !(hop == 0 && hop0_phyonly) || dyn_w_time == 0.0 "hop0_phyonly hop must run at w_time=0 (got dyn_w_time=$dyn_w_time)"

        # I_min/kappa_I/S_min/kappa_S (the squared-hinge penalty on
        # inversion/silencing_success, see _fidelity_physics_cost) are NOT
        # separate run_local_adam keywords -- like target_F/w_tmax/
        # w_power/w_time's own BASE values, they travel through inside the
        # caller's own `cost_kwargs`, so whatever the caller put there
        # (or pulse_cost's own defaults, if they put nothing) is exactly
        # what every epoch -- and initial_metrics/final_metrics one level
        # up, which call pulse_cost directly with the SAME cost_kwargs --
        # actually uses. Unlike the earlier p_exp/q_exp exponent barrier
        # this replaces, there is nothing to compute here per epoch and no
        # separate merge needed: the penalty is evaluated on THIS epoch's
        # own live inversion/silencing_success (via whichever grad_mode
        # backend), not a detached previous point, and is NOT gated by
        # hop==0 -- it applies identically on every hop, including hop 0.
        #
        # Always override w_time with this epoch's dyn_w_time (the merge is
        # a no-op when anneal_direct_weights=false at hop!=0, where
        # factor=1.0 already makes dyn_w_time==base_w_time).
        epoch_cost_kwargs = merge(cost_kwargs, (w_time=dyn_w_time,))

        # Shift historical sandbox costs onto the CURRENT epoch's w_time.
        # Adam then compares apples-to-apples under the live weight: hop 0
        # (dyn_w_time=0) can accept duration-expanding physics-improving
        # steps, while later hops still tighten duration as w_time grows.
        # w_tmax/w_power are never annealed, so they need no analogous shift.
        if epoch > 1 && dyn_w_time != prev_dyn_w_time
            w_diff = dyn_w_time - prev_dyn_w_time
            best_cost += w_diff * (best_dur / pulse.T_max)
            if isfinite(last_good_aux[1])
                shifted_last_cost = last_good_aux[1] + w_diff * (last_good_aux[4] / pulse.T_max)
                last_good_aux = (shifted_last_cost, last_good_aux[2], last_good_aux[3],
                                 last_good_aux[4], last_good_aux[5], last_good_aux[6], last_good_aux[7])
            end
        end
        prev_dyn_w_time = dyn_w_time

        # After a revert, `u` is exactly `last_good_u` -- a point already
        # fully evaluated (one ODE solve + AD gradient) the first time it
        # was accepted. Re-running that identical, expensive, deterministic
        # computation again on every backoff retry would waste exactly the
        # work the retry loop is trying to make productive; reuse the
        # cached result instead. `adam.m`/`adam.v`/`adam.t` are also reset
        # to their snapshot from right when `last_good_u` was accepted, so
        # each retry blends the SAME cached gradient into momentum exactly
        # once (as a normal step would), rather than accumulating it again
        # on top of whatever the previous failed attempt already blended in.
        # `last_good_aux` already holds the sandbox cost, shifted to the
        # current w_time by the moving-target block above if the weight
        # changed this epoch -- read it back directly, do not re-evaluate.
        if just_reverted
            grad = last_good_grad
            cost, inv_, sil_, dur_, coh_, famp_, ret_ = last_good_aux
            adam.m .= adam_m0
            adam.v .= adam_v0
            adam.t = adam_t0
        elseif grad_mode === :adjoint
            grad, cost, inv_, sil_, dur_, coh_, famp_, ret_ = pulse_cost_grad_adjoint(
                u, pulse, d; epoch_cost_kwargs..., compute=compute, solve_kwargs...,
            )
        elseif threaded_grad
            grad, cost, inv_, sil_, dur_, coh_, famp_, ret_ = _pulse_cost_grad_threaded(
                u, pulse, d; epoch_cost_kwargs..., compute=compute, solve_kwargs...,
            )
        else
            function cost_only(uu)
                c, inv_2, sil_2, dur_2, coh_2, famp_2, ret_2 = pulse_cost(uu, pulse, d; epoch_cost_kwargs..., compute=compute, solve_kwargs...)
                aux[] = (
                    Float64(ForwardDiff.value(c)),
                    Float64(ForwardDiff.value(inv_2)),
                    Float64(ForwardDiff.value(sil_2)),
                    Float64(ForwardDiff.value(dur_2)),
                    Float64(ForwardDiff.value(coh_2)),
                    Float64(ForwardDiff.value(famp_2)),
                    Float64(ForwardDiff.value(ret_2)),
                )
                return c
            end
            grad = ForwardDiff.gradient(cost_only, u)
            cost, inv_, sil_, dur_, coh_, famp_, ret_ = aux[]
        end
        if !isfinite(cost)
            epochs_since_improve += 1
            push!(history, (hop=hop, epoch=epoch, k=pulse.k, cost=cost, inversion=inv_,
                             silencing=sil_, duration=dur_, coherence=coh_, field_amp=famp_,
                             weak_seed_retention=ret_, improved=false,
                             x_tune=x_tune_eff, schedule_factor=factor))
            elapsed = time() - t_wall
            u .= last_good_u
            lr /= 2
            just_reverted = true
            println(
                "$label epoch $(lpad(epoch, 3)): $(round(elapsed, digits=1))s  " *
                "cost=Inf   ODE solve failed -- reverted to last valid point, " *
                "halved step size to $(round(lr, sigdigits=3))"
            )
            if epochs_since_improve >= patience
                println("$label early stop at epoch $epoch (no improvement > $tol for $patience epochs)")
                break
            end
            continue
        end

        lr = min(lr * 1.5, learning_rate)
        last_good_u .= u
        last_good_grad .= grad
        last_good_aux = (cost, inv_, sil_, dur_, coh_, famp_, ret_)
        adam_m0 .= adam.m
        adam_v0 .= adam.v
        adam_t0 = adam.t
        just_reverted = false

        improved = cost < best_cost - tol
        if improved
            best_cost, best_u = cost, copy(u)
            best_inv, best_sil, best_dur = inv_, sil_, dur_
            epochs_since_improve = 0
        else
            epochs_since_improve += 1
        end

        push!(history, (hop=hop, epoch=epoch, k=pulse.k, cost=cost, inversion=inv_,
                         silencing=sil_, duration=dur_, coherence=coh_, field_amp=famp_,
                         weak_seed_retention=ret_, improved=improved,
                         x_tune=x_tune_eff, schedule_factor=factor))

        adam_step!(u, grad, adam; lr=lr, lr_scale=lr_scale)

        elapsed = time() - t_wall
        mark = improved ? "*" : " "
        println(
            "$label epoch $(lpad(epoch, 3)): $(round(elapsed, digits=1))s  " *
            "cost=$(round(cost, digits=4)) $mark inversion=$(round(inv_, digits=4)) " *
            "silencing=$(round(sil_, digits=4)) duration=$(round(dur_, sigdigits=4))s"
        )

        if epochs_since_improve >= patience
            println("$label early stop at epoch $epoch (no improvement > $tol for $patience epochs)")
            break
        end
    end

    # Adam optimized the annealed sandbox; basin-hopping Metropolis tests
    # need the strict static cost under the caller's own cost_kwargs.
    # Skip the extra solve when no epoch ever produced a finite cost
    # (best_u is still u_start, which already failed).
    final_static_cost = if isfinite(best_cost)
        pulse_cost(best_u, pulse, d; cost_kwargs..., compute=compute, solve_kwargs...)[1]
    else
        Inf
    end

    return best_u, final_static_cost, best_inv, best_sil, best_dur, history
end

"""
    optimise_composite_pulse(k, n_coeff_A, n_coeff_f, d;
        num_epochs=30, learning_rate=0.05, patience=5, tol=1e-3,
        n_hops=3, hop_patience=2, hop_step_size=0.5, temperature=1.0,
        degree=3, taper_frac=0.1, w_tmax=1.0, w_power=0.05,
        target_F=1.0, w_time=0.15, seed=42,
        warm_start_u=nothing, label_prefix="", threaded_grad=true, compute=:auto,
        grad_mode=:forwarddiff, track=:weak, anneal_direct_weights=true,
        x_tune_alpha=_DEFAULT_X_TUNE_ALPHA, recalibrate_optima_x=true,
        I_min=_DEFAULT_PENALTY_MIN, kappa_I=_DEFAULT_PENALTY_KAPPA,
        S_min=_DEFAULT_PENALTY_MIN, kappa_S=_DEFAULT_PENALTY_KAPPA, solve_kwargs...)
        -> (best_u, best_cost, pulse::CompositePulse, u0, initial_metrics, history, final_metrics, optimizer_settings)

Basin-hopping global search over composite-pulse solutions for THIS
package's own 1st-order physics, each basin explored by
[`run_local_adam`](@ref)'s early-stopped Adam descent (or its
[`_pulse_cost_grad_threaded`](@ref) task-parallel substitute when
`threaded_grad=true` -- forwarded to every `run_local_adam` call, hop 0
and every subsequent hop; default `true` so the two independent Dual ODE
solves run concurrently, and on 1-8 CUDA devices those jobs are pinned
to GPUs; pass `threaded_grad=false` for the original serial
`ForwardDiff.gradient` sweep. `compute=:auto` uses a GPU array backend
for the SAME `rhs_1st_order!`/Tsit5 solve when a device is present and
`M` is large; `:cpu`/`:gpu` force a backend. See
[`_pulse_cost_grad_threaded`](@ref) for why the parallel gradient is
mathematically exact, not an approximation):

  1. Run a local Adam descent from a random initial guess (hop 0).
  2. For each further hop: perturb the CURRENT accepted point with
     isotropic Gaussian noise (`hop_step_size`, in raw/pre-softplus
     units) to get a new basin's starting point, run a local Adam
     descent from there, then apply the standard basin-hopping
     Metropolis test -- always accept an improving basin, accept a worse
     one with probability `exp(-(cost_new-cost_old)/temperature)` -- to
     decide whether the NEXT hop perturbs from this new point or falls
     back to the previous one. The globally best `(u, cost)` seen over
     ALL hops is tracked separately and is what gets returned.
  3. Stop hopping early once `hop_patience` consecutive hops produce no
     global improvement `> tol`.

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
`target_F`, and `w_time` are forwarded to [`pulse_cost`](@ref). Both
inversion and silencing always enter the objective (there is no
`w_inv`/`w_sil` weight any more -- `pulse_cost`'s multiplicative
`fidelity_phys` has no single-track reduction), targeting `target_F=1.0`
(RASE-style revival; pass `target_F=0.0` for ROSE-style silencing
instead). `track` (default `:weak`, forwarded to [`pulse_cost`](@ref) /
[`_pulse_cost_grad_threaded`](@ref) / [`run_local_adam`](@ref)) selects
ONE `:weak` solve per cost evaluation, with `inversion` read from that
solve's own `Sz` (O(ε) bias, corrected by the automatic winner re-check
below); pass `track=:dual` explicitly for the two-solve `:ground` +
`:weak` objective. `:dual` is never chosen automatically. These are
explicit keywords so they are NOT passed through to the ODE solver. Do
not pass `initial_condition` — the cost fixes `:ground`/`:weak` itself.

Besides the optimised `(best_u, best_cost, pulse)`, also returns: `u0`
(the initial/candidate parameterisation hop 0 actually started from --
either a fresh random guess or `warm_start_u`, see below),
`initial_metrics` (the full [`pulse_cost`](@ref) return at `u0`:
`(cost, inversion, silencing, duration, coherence, field_amp,
weak_seed_retention)` -- everything after `silencing` is diagnostic only,
never part of `cost`), `history` -- the [`run_local_adam`](@ref) per-epoch
log from EVERY hop, concatenated in run order (each row tagged with its
own `hop`/`epoch`) -- `final_metrics` (the same `pulse_cost`-tuple
shape, for `best_u` specifically -- NOT necessarily
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
carries `final_inversion_ground` and `final_inv_gap` from the automatic
winner re-check (see [`_assert_track`](@ref)): for `track=:weak` (the
default) `final_inversion_ground` is a fresh canonical `:ground` solve of
`best_u` and `final_inv_gap = inversion:weak - inversion:ground` is the
O(ε) single-track bias; for an explicit `track=:dual` these are just
`final_metrics[2]` and `0.0`. All of
these are exactly what [`optimise_control_pulse_from_jld2`](@ref) needs
to write a full, replicable run log; ordinary callers that only want the
optimised pulse can simply ignore the extra return values.

`warm_start_u`: if given (e.g. a previous run's saved `final_u`, from a
loaded `_optrunlog.jld2` -- see [`load_jld2_run`](@ref)), hop 0 starts
from THIS point instead of a fresh `initial_guess(pulse; seed=seed)`, so
a later call can pick up and continue optimising from where an earlier
one left off. Must have length `n_params(pulse)`, i.e. be a raw parameter
vector for a `CompositePulse` with the SAME `(k, n_coeff_A, n_coeff_f)`
as this call's -- an error is raised otherwise, since a length mismatch
would silently decode into a nonsensical pulse rather than fail loudly.

`cf_lr_scale` (default `1.0`) is forwarded unchanged to every
[`run_local_adam`](@ref) call (hop 0 and every subsequent hop) -- see that
function's own docstring for what it does and why.

`anneal_direct_weights` (default `true`, forwarded unchanged to every
`run_local_adam` call, hop 0 and every subsequent hop) anneals each hop's
own `w_time` gradient (`w_tmax`/`w_power` are NEVER annealed -- always the
caller's own base weight) from near-zero as that hop's physics fidelity
improves -- see [`run_local_adam`](@ref)'s own docstring and
[`_curriculum_fidelity_weight`](@ref) for the schedule and why the `cost`
this function itself compares (basin-hopping's Metropolis test,
`final_metrics`) is unaffected: `run_local_adam` re-evaluates its
returned `best_u` under the static, caller-configured `w_time` before
returning it. Pass `anneal_direct_weights=false` to disable annealing
entirely and recover the original fixed-`w_time` cost.

**`hop0_phyonly` (default `true`) suppresses hop 0's `w_time` to `0.0`** --
exactly mirroring [`run_local_adam`](@ref)'s own `hop==0 && hop0_phyonly`
rule (physics-only optimisation for the entire first hop, REGARDLESS of
`anneal_direct_weights`'s own value -- a genuinely different `factor`
default than the ordinary `anneal_direct_weights=false` one; see that
function's docstring). No calibration `pulse_cost` evaluation spent on a
suppressed hop 0. Pass **`hop0_phyonly=false`** to make hop 0 a normal
scheduled hop instead -- it then anneals `w_time` (and calibrates
`x_tune` from `u0`, one extra `pulse_cost` eval) exactly like hop 1, so
duration is weighed from the very first hop.
**Hop 1 is the first hop ORDINARY annealing (the `anneal_direct_weights`-
gated kind) ever applies to**, and is therefore this function's own "seed"
hop:
`x_tune_alpha` (default [`_DEFAULT_X_TUNE_ALPHA`](@ref)) picks the
schedule's curvature via [`solve_optimal_x_start`](@ref) -- there is no
raw, manually-set curvature keyword; it is always either calibrated or
the plain linear sentinel (see [`run_local_adam`](@ref)'s own docstring).
Under the defaults, hop 1's own `run_local_adam` call performs a SINGLE
MANDATORY calibration from ITS OWN starting point (a perturbation of hop
0's result), exactly as [`run_local_adam`](@ref) already does internally
for any nonzero `hop` -- no separate calibration code is needed in this
function for hop 1, unlike hop 0's now-moot upfront calibration (removed
entirely, since hop 0 never anneals and therefore never needs a
calibrated `x_tune` at all). `recalibrate_optima_x` (default `true`)
controls every hop AFTER hop 1 (hop 2 onwards): `true` re-runs the SAME
per-hop calibration `run_local_adam` already supports internally (against
THAT hop's own starting point, e.g. after a perturbation moved it
elsewhere); `false` instead reuses hop 1's own single calibrated value
(captured from its returned `history`) for every subsequent hop unchanged
(injected via `run_local_adam`'s internal `_precalibrated_x_tune`), never
recalibrating again. Passing `x_tune_alpha=nothing` EXPLICITLY bypasses
calibration entirely from hop 1 onwards -- the only way to run the
annealed schedule uncalibrated (plain linear) -- and is a silent no-op
whenever `anneal_direct_weights=false`. If `n_hops == 1` (no hop ever
reaches hop 1), the entire run never anneals at all.

`I_min`/`kappa_I`/`S_min`/`kappa_S` (defaults [`_DEFAULT_PENALTY_MIN`](@ref)/
[`_DEFAULT_PENALTY_KAPPA`](@ref) each, forwarded unchanged to every
`run_local_adam` call, hop 0 and every subsequent hop) are
[`run_local_adam`](@ref)'s own squared-hinge penalty feature, forwarded
through unchanged -- see that function's own docstring. Unlike annealing,
it is NOT gated by `hop==0`; it applies identically on every hop,
including hop 0.
"""
function optimise_composite_pulse(
    k::Integer, n_coeff_A::Integer, n_coeff_f::Integer, d;
    num_epochs::Integer=30, learning_rate::Real=0.05, cf_lr_scale::Real=1.0, patience::Integer=5, tol::Real=1e-3,
    n_hops::Integer=3, hop_patience::Integer=2, hop_step_size::Real=0.5, temperature::Real=1.0,
    degree::Integer=3, taper_frac::Real=0.1, w_tmax::Real=1.0, w_power::Real=0.05,
    target_F::Real=1.0, w_time::Real=0.15,
    seed::Integer=42, warm_start_u=nothing, label_prefix::AbstractString="",
    threaded_grad::Bool=true, compute::Symbol=:auto, grad_mode::Symbol=:forwarddiff,
    track::Symbol=:weak,
    anneal_direct_weights::Bool=true, hop0_phyonly::Bool=true,
    x_tune_alpha::Union{Nothing,Real}=_DEFAULT_X_TUNE_ALPHA, recalibrate_optima_x::Bool=true,
    I_min::Real=_DEFAULT_PENALTY_MIN, kappa_I::Real=_DEFAULT_PENALTY_KAPPA,
    S_min::Real=_DEFAULT_PENALTY_MIN, kappa_S::Real=_DEFAULT_PENALTY_KAPPA,
    solve_kwargs...,
)
    _forbid_initial_condition(solve_kwargs)
    _assert_track(track)
    (grad_mode === :forwarddiff || grad_mode === :adjoint) || error(
        "grad_mode must be :forwarddiff or :adjoint, got $(repr(grad_mode))."
    )
    pulse = CompositePulse(k, n_coeff_A, n_coeff_f, d; degree=degree, taper_frac=taper_frac)
    cost_kwargs = (w_tmax=w_tmax, w_power=w_power, target_F=target_F, w_time=w_time,
                   I_min=I_min, kappa_I=kappa_I, S_min=S_min, kappa_S=kappa_S, track=track)
    rng = Random.Xoshiro(seed)

    # threaded_grad/compute are deliberately kept OUT of solve_kwargs
    # (unlike every other extra keyword here): solve_kwargs also flows
    # into pulse_cost's initial_metrics/final_metrics calls below. Only
    # run_local_adam / pulse_cost (which know these names) should see them.
    solve_settings = NamedTuple(kv for kv in pairs(solve_kwargs) if !(kv[2] isa Function))
    optimizer_settings = merge(
        (k=k, n_coeff_A=n_coeff_A, n_coeff_f=n_coeff_f, degree=degree, taper_frac=taper_frac,
         num_epochs=num_epochs, learning_rate=learning_rate, cf_lr_scale=cf_lr_scale, patience=patience, tol=tol,
         n_hops=n_hops, hop_patience=hop_patience, hop_step_size=hop_step_size, temperature=temperature,
         w_tmax=w_tmax, w_power=w_power, target_F=target_F, w_time=w_time, seed=seed,
         threaded_grad=threaded_grad, compute=compute, grad_mode=grad_mode, track=track, n_gpus=pulse_gpu_count(),
         anneal_direct_weights=anneal_direct_weights, hop0_phyonly=hop0_phyonly, x_tune_alpha=x_tune_alpha,
         recalibrate_optima_x=recalibrate_optima_x,
         I_min=I_min, kappa_I=kappa_I, S_min=S_min, kappa_S=kappa_S),
        solve_settings,
    )

    println(
        "$(label_prefix)Optimising k=$k pulses, $(n_params(pulse)) raw parameters (ForwardDiff/Adam + " *
        "basin-hopping, physics: InhomogeneousSpinCavityDynamics.jl rhs_1st_order!, " *
        "compute=$(compute), GPUs=$(pulse_gpu_count())) ..."
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
    initial_metrics = pulse_cost(u0, pulse, d; cost_kwargs..., compute=compute, solve_kwargs...)
    history = NamedTuple[]

    # Hop 0: with the default hop0_phyonly=true it never anneals (see
    # run_local_adam's own hop==0 rule), so nothing to calibrate x_tune for.
    # With hop0_phyonly=false it IS a normal scheduled hop and calibrates
    # x_tune from its own u_start -- so x_tune_alpha must be forwarded (it is
    # inert under hop0_phyonly=true). `_precalibrated_x_tune` is still not
    # forwarded: hop 0 is the first hop, there is nothing precalibrated yet.
    current_u, current_cost, _, _, _, hop0_history = run_local_adam(
        u0, pulse, d, cost_kwargs; hop=0, num_epochs, patience, tol, learning_rate, cf_lr_scale,
        label="$(label_prefix)[hop 0]", threaded_grad=threaded_grad, compute=compute,
        grad_mode=grad_mode, anneal_direct_weights=anneal_direct_weights,
        hop0_phyonly=hop0_phyonly, x_tune_alpha=x_tune_alpha,
        solve_kwargs...
    )
    append!(history, hop0_history)
    global_best_u, global_best_cost = current_u, current_cost
    hops_since_improve = 0
    x_tune_seed = 0.0   # populated once hop 1 completes; used only if recalibrate_optima_x=false

    for hop in 1:(n_hops-1)
        perturbation = hop_step_size .* randn(rng, length(current_u))
        candidate_u0 = current_u .+ perturbation

        # hop==1 is the first hop annealing ever applies to (see
        # run_local_adam's own hop==0 rule) -- it ALWAYS calibrates fresh
        # from its own starting point, exactly as run_local_adam's own
        # mandatory calibration already does for any nonzero hop; there is
        # no earlier annealed hop to reuse a value from yet. From hop 2
        # onwards, recalibrate_optima_x=true (default) keeps recalibrating
        # fresh each time; false reuses hop 1's own calibrated value
        # (captured below) for every remaining hop unchanged, via
        # run_local_adam's internal _precalibrated_x_tune.
        hop_x_tune_alpha, hop_precal = if hop == 1 || recalibrate_optima_x
            (x_tune_alpha, nothing)
        else
            (nothing, x_tune_seed)
        end

        cand_u, cand_cost, _, _, _, hop_history = run_local_adam(
            candidate_u0, pulse, d, cost_kwargs;
            hop, num_epochs, patience, tol, learning_rate, cf_lr_scale,
            label="$(label_prefix)[hop $hop]", threaded_grad=threaded_grad, compute=compute,
            grad_mode=grad_mode, anneal_direct_weights=anneal_direct_weights,
            x_tune_alpha=hop_x_tune_alpha, _precalibrated_x_tune=hop_precal,
            solve_kwargs...,
        )
        append!(history, hop_history)
        if hop == 1
            x_tune_seed = isempty(hop_history) ? 0.0 : hop_history[1].x_tune
        end

        if cand_cost < global_best_cost - tol
            global_best_u, global_best_cost = cand_u, cand_cost
            hops_since_improve = 0
        else
            hops_since_improve += 1
        end

        delta = cand_cost - current_cost
        accept = delta < 0.0 || rand(rng) < exp(-delta / max(temperature, 1e-12))
        if accept
            current_u, current_cost = cand_u, cand_cost
        end

        accept_str = accept ? "accepted" : "rejected"
        println(
            "$(label_prefix)hop $hop: local best cost=$(round(cand_cost, digits=4)) " *
            "($accept_str as new basin, delta=$(round(delta, digits=4))) " *
            "global best cost=$(round(global_best_cost, digits=4))"
        )

        if hops_since_improve >= hop_patience
            println("$(label_prefix)Basin-hopping stopped early after $hop hops (no global improvement > $tol for $hop_patience hops).")
            break
        end
    end

    final_metrics = pulse_cost(global_best_u, pulse, d; cost_kwargs..., compute=compute, solve_kwargs...)

    # Winner re-check: a `track=:weak` run scores `inversion` on the `:weak`
    # solve, which biases it by O(ε) vs the canonical `:ground` value (see
    # `_assert_track`). Spend ONE `:ground` solve of the winner so
    # `optimizer_settings` (and therefore any saved run log) always carries the
    # true `:ground` inversion and the measured bias. `:dual` runs already
    # scored inversion on `:ground`, so there `final_inv_gap == 0` and no extra
    # solve runs. Uses the SAME `_solver_kwargs`/`run_sim_1st_order_pure` path
    # `pulse_cost`'s own `:dual` branch uses, so this is the identical solve.
    if track === :weak
        sk_final = _solver_kwargs(solve_kwargs)
        _, _, Sz_gf, Nj_gf = run_sim_1st_order_pure(
            global_best_u, pulse, d; compute=compute, sk_final..., initial_condition=:ground,
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

    println("$(label_prefix)Optimisation complete. Global best cost: $(round(global_best_cost, digits=4)).")
    return global_best_u, global_best_cost, pulse, u0, initial_metrics, history, final_metrics, optimizer_settings
end

function _normalise_k_specs(kinds, specs)
    if specs !== nothing
        return collect(specs)
    end
    return [(k_of_seed_kind(kind), kind) for kind in kinds]
end

"""
    optimise_composite_pulse_over_k(n_coeff_A, n_coeff_f, d;
        kinds=(:hs1, :corpse, :bb1), specs=nothing, threaded=true,
        Omega_max=nothing, beta=nothing, mu=nothing, seed=42,
        optimizer_kwargs...)
        -> NamedTuple

Discrete search over sub-pulse count `k`. `k` is not a continuous
decision variable: each `k` gets its own `CompositePulse`, a canonical
warm-start ([`seed_canonical`](@ref)), and an independent
[`optimise_composite_pulse`](@ref) run on that pulse's continuous raw
parameters. The runs are independent and are threaded when
`threaded=true` and `Threads.nthreads() > 1` (start Julia with
`JULIA_NUM_THREADS=N`; for ODE-heavy work consider
`BLAS.set_num_threads(1)` so threads do not oversubscribe).

Default `kinds` is HS1 (`k=1`), CORPSE (`k=5`), BB1 (`k=7`). Pass
`specs=((k, kind), ...)` to choose `k` and seed explicitly, including
`:random` for `initial_guess` (e.g. `specs=((3, :random), (5, :corpse))`).

`Omega_max` defaults to each pulse's own `amp_scale` (cavity-input
units). HS1 `beta`/`mu` default as in [`seed_canonical`](@ref).
`optimizer_kwargs` are forwarded to [`optimise_composite_pulse`](@ref)
(`num_epochs`, `signal_E_of_t`, `w_tmax`, ...). Do not pass
`warm_start_u` -- the seed is built per `k`.

Returns a NamedTuple: `best_kind`, `best_k`, `best_u`, `best_cost`,
`pulse`, `u0`, `initial_metrics`, `history`, `final_metrics`,
`optimizer_settings` (the winning run, same payload as
[`optimise_composite_pulse`](@ref) plus kind), and `per_k` (one
NamedTuple per spec, in input order).
"""
function optimise_composite_pulse_over_k(
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
        "optimise_composite_pulse_over_k builds a per-k canonical seed; do not pass warm_start_u."
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
            optimise_composite_pulse(
                k, n_coeff_A, n_coeff_f, d;
                optimizer_kwargs...,
                seed=seed + 1000 * Int(k),
                warm_start_u=u0,
                label_prefix=prefix,
            )
        optimizer_settings = merge(optimizer_settings, (seed_kind=kind,))
        return (
            kind=kind, k=Int(k),
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
        mark = r.k == best.k ? "*" : " "
        println(
            "  $mark k=$(r.k) $(r.kind): cost=$(round(r.best_cost, digits=4))"
        )
    end
    println("  winner: k=$(best.k) $(best.kind)  cost=$(round(best.best_cost, digits=4))")

    return (
        best_kind=best.kind, best_k=best.k,
        best_u=best.best_u, best_cost=best.best_cost, pulse=best.pulse,
        u0=best.u0, initial_metrics=best.initial_metrics, history=best.history,
        final_metrics=best.final_metrics, optimizer_settings=best.optimizer_settings,
        per_k=per_k,
    )
end


# ============================================================
# 1st-order RHS
# ============================================================

# --- backend trait ------------------------------------------------------
# `false` for every plain-host array (`Vector`, `SubArray` of `Vector`,
# `Complex{Dual}` ...). A `true` method for `CUDA.AnyCuArray` is added in
# `solver_1st_order.jl` (which is only loaded as part of the full module,
# with CUDA in scope -- the test harness includes this file WITHOUT CUDA
# and gets only the host fallback). `_is_gpu(u)` is a compile-time
# constant once `rhs_1st_order!` is specialised on `typeof(u)`, so the
# CPU/GPU branch in the RHS is folded and only one path is compiled per
# backend (no runtime dispatch, no `Union`-typed `s`/`a`).
_is_gpu(::AbstractArray) = false

# --- cavity collective source  Σ_j g_j conj(S⁺_j) ---------------------
#
# History:
#   v0  `sum(g_b .* conj.(Sp))`      -- length-M temp per RK stage + host sync
#   v1  reuse `work`, `sum(work)`    -- no alloc; still host sync on GPU
#   v2  `sum!(out, work)`            -- result kept on-device (GPU)
#   v3  host: `sum(work)` (scalar); GPU: fused `mapreduce` over the LAZY
#       product, no length-M materialisation at all
#
# Host path (`_cavity_source_host!`): fills the reused `work` buffer with
# EXACTLY `g_b .* conj.(Sp)` and returns `sum(work)` -- bit-for-bit the
# historical expression, so the discrete-adjoint / ForwardDiff results are
# unchanged to the last bit. `work` is cached per (array type, M) in
# task-local storage (NOT `threadid()` -- a Task can migrate OS threads at
# any yield under `:dynamic` `@threads`; task-local storage is bound to the
# Task and needs no lock), matching `_tsit5_adj_workspace`.
#
# GPU path (`_cavity_source_dev`): reduces the lazy `Broadcasted`
# `g_b .* conj.(Sp)` directly. `mapreduce(identity, +, bc; dims=1)` is the
# same reduction `sum` performs (`sum(x;dims) == mapreduce(identity,
# add_sum, x; dims)`, and `add_sum === (+)` for `Complex`); verified
# bit-identical to the host `sum(g_b .* conj.(Sp))` on ComplexF64 and
# Complex{Dual}, and ~2.6x faster than materialise-then-`sum!` at M=1e4.
# Returns a fresh 1-element device array (pooled alloc), consumed as a
# 1-element operand in the cavity broadcast.
function _cavity_work_buffer(Sp::AbstractVector)
    store = get!(
        () -> Dict{Tuple{DataType,Int},Any}(),
        task_local_storage(),
        :InhomogeneousSpinCavityDynamics_cavity_work_buffer,
    )::Dict{Tuple{DataType,Int},Any}
    # Key on BOTH array type and length: one Task may reduce ensembles of
    # different M (adjoint tests use small toy M) and `typeof(Sp)` alone
    # does not distinguish them.
    return get!(() -> similar(Sp), store, (typeof(Sp), length(Sp)))
end

@noinline function _cavity_source_host!(work::AbstractVector, g_b, Sp)
    work .= g_b .* conj.(Sp)
    return sum(work)
end

@inline function _cavity_source_dev(g_b, Sp)
    bc = Base.Broadcast.instantiate(
        Base.broadcasted(*, g_b, Base.broadcasted(conj, Sp))
    )
    return Base.mapreduce(identity, +, bc; dims = 1)
end

function rhs_1st_order!(du, u, p, t)
    # Interaction-picture opt-in: an 8-tuple `p` whose 8th element is `:ip`
    # routes to the co-rotating-frame RHS (rhs_1st_order_ip.jl), which stores
    # `S̃⁺ = S⁺ e^{-iδt}` and drops the stiff `iδ·S⁺` free-precession term.
    # For the 7-tuple `p` every other caller builds, `length(p) >= 8` folds
    # to `false` at compile time and this line is dead-code-eliminated -- the
    # lab-frame body below is byte-for-byte unchanged.
    length(p) >= 8 && p[8] === :ip && return _rhs_1st_order_ip!(du, u, p, t)
    delta0, kappa_e, kappa_i, delta_b_gpu, g_b_gpu, M, E_of_t = p

    # `Sp`/`Sz` views (same slices as `unpack_state_1st_order_u`); the
    # cavity amplitude is handled per-branch below (scalar on CPU, view on
    # GPU), so it is not unpacked here.
    Sp = @view u[IDX1_Sp_start:IDX1_Sp_start + M - 1]
    Sz = @view u[idx1_Sz_start(M):idx1_Sz_start(M) + M - 1]
    dSp, dSz = unpack_state_1st_order_du(du, M)

    κe = kappa_e
    κt = kappa_e + kappa_i
    E_t = E_of_t(t)

    # `_is_gpu(u)` is const-folded once `rhs_1st_order!` is specialised on
    # `typeof(u)`, so exactly ONE of the two branches below is compiled per
    # backend -- no runtime dispatch, no `Union` in `s`/`a`.
    if _is_gpu(u)
        # --- GPU: never scalar-index `u[1]`/`du[1]` (host<->device round
        # trips on every RK stage), keep the collective source on-device.
        a = @view u[IDX1_a:IDX1_a]
        da = @view du[IDX1_a:IDX1_a]
        s = _cavity_source_dev(g_b_gpu, Sp)          # 1-element device array

        da .=
            (sqrt(κe) * E_t) .-
            (1im * delta0) .* a .-
            (1im .* s) .-
            (0.5 * κt) .* a

        dSp .=
            1im .* delta_b_gpu .* Sp .-
            2im .* g_b_gpu .* conj.(a) .* Sz

        dSz .=
            -1im .* g_b_gpu .* a .* Sp .+
             1im .* g_b_gpu .* conj.(a) .* conj.(Sp)
    else
        # --- CPU `Vector` / `Complex{Dual}` (forward solve, discrete
        # adjoint replay, ForwardDiff): the ORIGINAL scalar-`a` vector
        # field, byte-for-byte. No GPU sync to avoid here, and the scalar
        # form allocates nothing per call (a 1-element-view form boxes a
        # `SubArray` operand into every broadcast). The only change from
        # the pre-optimisation code is the reused reduction buffer, which
        # holds EXACTLY `g_b .* conj.(Sp)` so `sum(buf) == sum(g_b .*
        # conj.(Sp))` to the last bit.
        a = u[IDX1_a]
        s = _cavity_source_host!(_cavity_work_buffer(Sp), g_b_gpu, Sp)

        du[IDX1_a] =
            sqrt(κe) * E_t -
            1im * delta0 * a -
            1im * s -
            0.5 * κt * a

        dSp .=
            1im .* delta_b_gpu .* Sp .-
            2im .* g_b_gpu .* conj(a) .* Sz

        dSz .=
            -1im .* g_b_gpu .* a .* Sp .+
             1im .* g_b_gpu .* conj(a) .* conj.(Sp)
    end

    return nothing
end

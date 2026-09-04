# ============================================================
# 1st-order RHS
# ============================================================

# --- backend trait ------------------------------------------------------
# `false` for every plain-host array (`Vector`, `SubArray` of `Vector`,
# `Complex{Dual}` ...). A `true` method for `CUDA.AnyCuArray` is added in
# `solver_1st_order.jl` (which is only loaded as part of the full module,
# with CUDA in scope -- the test harness includes this file WITHOUT CUDA
# and gets only the host fallback). `_is_gpu(Sp)` is a compile-time
# constant once `rhs_1st_order!` is specialised on `typeof(u)`, so the
# branch in the RHS is folded and only one reduction path is compiled per
# backend.
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
    delta0, kappa_e, kappa_i, delta_b_gpu, g_b_gpu, M, E_of_t = p

    # 1-element views for the cavity amplitude / its derivative (never the
    # scalars `u[1]` / `du[1]`): on a CuArray backend those are host<->device
    # round-trips on every RK stage. Arithmetically transparent on Vector/Dual.
    a, Sp, Sz = unpack_state_1st_order_u_views(u, M)
    dSp, dSz = unpack_state_1st_order_du(du, M)
    da = @view du[IDX1_a:IDX1_a]

    κe = kappa_e
    κt = kappa_e + kappa_i
    E_t = E_of_t(t)

    # Collective source Σ_j g_j conj(S⁺_j). `_is_gpu(Sp)` is const-folded
    # per specialisation, so exactly one branch is compiled per backend.
    # `s` is a host scalar on the CPU/Dual path, a 1-element device array
    # on the GPU path; either broadcasts into the length-1 `da` below.
    s = _is_gpu(Sp) ?
        _cavity_source_dev(g_b_gpu, Sp) :
        _cavity_source_host!(_cavity_work_buffer(Sp), g_b_gpu, Sp)

    # --------------------------------------------------------
    # Cavity equation
    #
    #   ȧ = √κe E(t) - i δ0 a - i Σ_j g_j conj(S⁺_j) - ½ κt a
    #
    # Length-1 broadcast into `da` (was the scalar store `du[IDX1_a] = ...`).
    # The scalar sub-terms (√κe·E_t, i·δ0, ½·κt) are host scalars exactly as
    # before; `a` is a 1-element on-device operand. The `.-` chain and the
    # `.*` sub-chains associate left-to-right just like the original scalar
    # expression, so on the CPU/Dual path `da[1]` is bit-identical to the
    # old `du[IDX1_a]`.
    # --------------------------------------------------------
    da .=
        (sqrt(κe) * E_t) .-
        (1im * delta0) .* a .-
        (1im .* s) .-
        (0.5 * κt) .* a

    # --------------------------------------------------------
    # Spin equations
    #
    # Mean-field replacement:
    #     ⟨a† Sz⟩ ≈ conj(a) Sz
    #     ⟨a† S-⟩ ≈ conj(a) conj(S+)
    #
    # `a` is a 1-element view, so `conj(a)` becomes `conj.(a)`; it still
    # broadcasts against the length-M ensemble exactly as the old scalar
    # `conj(a)` did. These length-M broadcasts are byte-for-byte unchanged.
    # --------------------------------------------------------

    dSp .=
        1im .* delta_b_gpu .* Sp .-
        2im .* g_b_gpu .* conj.(a) .* Sz

    dSz .=
        -1im .* g_b_gpu .* a .* Sp .+
         1im .* g_b_gpu .* conj.(a) .* conj.(Sp)

    return nothing
end

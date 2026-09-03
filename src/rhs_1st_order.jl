# ============================================================
# 1st-order RHS
# ============================================================

# Task-local scratch for the cavity collective source Σ_j g_j conj(S⁺_j).
#
# `rhs_1st_order!` previously evaluated that reduction as
# `sum(g_b .* conj.(Sp))`, which allocates a length-M temporary on EVERY
# RK stage (~6.6e5 stages on the production rose case, ~10^2 GiB of churn
# per run). This reuses one buffer per (array type, M) within the running
# Task. The buffer is filled with EXACTLY `g_b .* conj.(Sp)` and then
# `sum`-reduced, so `s` is bit-for-bit the value the old expression
# produced -- only the allocation is removed, the arithmetic (elementwise
# products, reduction order) is unchanged on every backend.
#
# Keyed per-Task via `task_local_storage` (NOT `threadid()`), matching
# `_tsit5_adj_workspace`'s reasoning: under `:dynamic` `@threads` a Task
# can migrate OS threads at any yield point, so `threadid()` is not a
# stable per-task identity; task-local storage is bound to the Task object
# and needs no lock. Keyed by `typeof(Sp)` so the `Vector` (CPU / discrete
# adjoint), `CuArray` (GPU), and `Complex{Dual}` (ForwardDiff) backends --
# any of which may run within a single optimiser Task -- never alias one
# buffer.
function _cavity_source_buffer(Sp::AbstractVector)
    store = get!(
        () -> Dict{Tuple{DataType,Int},Any}(),
        task_local_storage(),
        :InhomogeneousSpinCavityDynamics_cavity_source_buffer,
    )::Dict{Tuple{DataType,Int},Any}
    # Key on BOTH the array type and the length: within one process the
    # optimiser sweeps ensembles of different M (and the adjoint tests use
    # small toy M), and `typeof(Sp)` alone does not distinguish them.
    return get!(() -> similar(Sp), store, (typeof(Sp), length(Sp)))
end

# Function barrier: `buf` arrives typed `Any` from the task-local cache;
# specialise on its concrete type here so the broadcast and `sum` compile.
# `buf .= g_b .* conj.(Sp)` is the SAME fused elementwise expression the
# old `sum(g_b .* conj.(Sp))` built into a fresh array, so `sum(buf)` is
# the identical reduction.
@noinline function _cavity_source!(buf::AbstractVector, g_b, Sp)
    buf .= g_b .* conj.(Sp)
    return sum(buf)
end

function rhs_1st_order!(du, u, p, t)
    delta0, kappa_e, kappa_i, delta_b_gpu, g_b_gpu, M, E_of_t = p

    # 1-element view for the cavity amplitude (never the scalar `u[1]`):
    # keeps the value on-device on a CuArray backend so no RK stage does a
    # host<->device round-trip. Arithmetically transparent on Vector/Dual.
    a, Sp, Sz = unpack_state_1st_order_u_views(u, M)
    dSp, dSz = unpack_state_1st_order_du(du, M)
    da = @view du[IDX1_a:IDX1_a]

    κe = kappa_e
    κt = kappa_e + kappa_i
    E_t = E_of_t(t)

    # Collective source Σ_j g_j conj(S⁺_j), reduced through a reused
    # buffer instead of a fresh `sum(g_b .* conj.(Sp))` temporary.
    s = _cavity_source!(_cavity_source_buffer(Sp), g_b_gpu, Sp)

    # --------------------------------------------------------
    # Cavity equation
    #
    #   ȧ = √κe E(t) - i δ0 a - i Σ_j g_j conj(S⁺_j) - ½ κt a
    #
    # Written as a length-1 broadcast into `da` (was the scalar store
    # `du[IDX1_a] = ...`). Every scalar sub-term (√κe·E_t, i·δ0, i·s,
    # ½·κt) is computed once on the host exactly as before; the `.-`
    # chain and the `.*` sub-chains associate left-to-right just like the
    # original scalar expression, so `da[1]` is bit-identical to the old
    # `du[IDX1_a]`.
    # --------------------------------------------------------
    da .=
        (sqrt(κe) * E_t) .-
        (1im * delta0) .* a .-
        (1im * s) .-
        (0.5 * κt) .* a

    # --------------------------------------------------------
    # Spin equations
    #
    # Mean-field replacement:
    #     ⟨a† Sz⟩ ≈ conj(a) Sz
    #     ⟨a† S-⟩ ≈ conj(a) conj(S+)
    #
    # `a` is now a 1-element view, so `conj(a)` becomes `conj.(a)`; it
    # still broadcasts against the length-M ensemble exactly as the old
    # scalar `conj(a)` did.
    # --------------------------------------------------------

    dSp .=
        1im .* delta_b_gpu .* Sp .-
        2im .* g_b_gpu .* conj.(a) .* Sz

    dSz .=
        -1im .* g_b_gpu .* a .* Sp .+
         1im .* g_b_gpu .* conj.(a) .* conj.(Sp)

    return nothing
end

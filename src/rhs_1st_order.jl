_is_gpu(::AbstractArray) = false

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

    Sp = @view u[IDX1_Sp_start:IDX1_Sp_start + M - 1]
    Sz = @view u[idx1_Sz_start(M):idx1_Sz_start(M) + M - 1]
    dSp, dSz = unpack_state_1st_order_du(du, M)

    κe = kappa_e
    κt = kappa_e + kappa_i
    E_t = E_of_t(t)

    if _is_gpu(u)
        a = @view u[IDX1_a:IDX1_a]
        da = @view du[IDX1_a:IDX1_a]
        s = _cavity_source_dev(g_b_gpu, Sp)

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

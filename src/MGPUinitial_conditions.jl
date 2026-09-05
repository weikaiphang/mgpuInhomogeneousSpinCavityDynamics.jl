

function small_block_initial(M::Int, Nj::AbstractVector, kind::Symbol, ::Type{T}) where {T}
    u = zeros(Complex{T}, small_length(M))

    if kind === :ground
        Sz0 = -Nj ./ 2
        u[small_range(M, F_Sz)]     .= Complex{T}.(Sz0)
        u[small_range(M, F_SzSz_s)] .= Complex{T}.((Nj .^ 2) ./ 4)
    elseif kind === :inverted
        Sz0 = Nj ./ 2
        u[small_range(M, F_Sz)]     .= Complex{T}.(Sz0)
        u[small_range(M, F_SzSz_s)] .= Complex{T}.((Nj .^ 2) ./ 4)
    elseif kind === :custom

    elseif kind === :equator || kind === :weak || kind === :weak_inverted
        Sp0, Sz0 = _mgpu_spin_means(Nj, kind)
        u[small_range(M, F_Sp)]     .= Complex{T}.(Sp0)
        u[small_range(M, F_Sz)]     .= Complex{T}.(Sz0)
        u[small_range(M, F_SpSp_s)] .= Complex{T}.(Sp0 .* Sp0)
        u[small_range(M, F_SzSp_s)] .= Complex{T}.(Sz0 .* Sp0)
        u[small_range(M, F_SmSp_s)] .= Complex{T}.(abs2.(Sp0))
        u[small_range(M, F_SzSz_s)] .= Complex{T}.(Sz0 .* Sz0)
    else
        _unknown_initial_condition(kind)
    end

    return u
end

function _mgpu_spin_means(Nj, kind::Symbol)
    if kind === :equator
        return Nj ./ 2, zero.(Nj)
    elseif kind === :weak
        return _WEAK_SEED .* Nj ./ 2, -Nj ./ 2
    elseif kind === :weak_inverted
        return _WEAK_SEED .* Nj ./ 2, Nj ./ 2
    else
        _unknown_initial_condition(kind)
    end
end


function set_initial_condition!(prob::MGPUProblem{T}, Nj::AbstractVector,
                                kind::Symbol) where {T}
    M = prob.M
    small = small_block_initial(M, Nj, kind, T)
    fill_szsz = kind === :ground || kind === :inverted ||
                kind === :weak || kind === :weak_inverted
    fill_sp_cross = kind === :equator || kind === :weak || kind === :weak_inverted

    each_shard(prob.shards, prob.exec) do s
        u = s.regs[1]





        CUDA.stream!(s.stream) do
            CUDA.fill!(u, zero(Complex{T}))
            copyto!(u, 1, small, 1, length(small))
        end

        n = M * s.mloc
        nblocks = min(cld(n, 256), 4096)
        if fill_szsz
            @cuda threads=256 blocks=nblocks stream=s.stream init_szsz_cross_kernel!(
                u, s.Nj, M, s.mloc, s.joff, s.lo)
        end
        if fill_sp_cross
            sp_scale = kind === :equator ? T(0.5) : T(_WEAK_SEED) / T(2)
            sz_scale = if kind === :equator
                T(0)
            elseif kind === :weak
                T(-0.5)
            else
                T(0.5)
            end
            @cuda threads=256 blocks=nblocks stream=s.stream init_nj_scale_cross_kernel!(
                u, s.Nj, sp_scale, sp_scale, B_SpSp, M, s.mloc, s.joff, s.lo)
            @cuda threads=256 blocks=nblocks stream=s.stream init_nj_scale_cross_kernel!(
                u, s.Nj, sp_scale, sp_scale, B_SmSp, M, s.mloc, s.joff, s.lo)
            if sz_scale != 0
                @cuda threads=256 blocks=nblocks stream=s.stream init_nj_scale_cross_kernel!(
                    u, s.Nj, sz_scale, sp_scale, B_SzSp, M, s.mloc, s.joff, s.lo)
                @cuda threads=256 blocks=nblocks stream=s.stream init_nj_scale_cross_kernel!(
                    u, s.Nj, sp_scale, sz_scale, B_SzSpT, M, s.mloc, s.joff, s.lo)
            end
        end
        CUDA.synchronize(s.stream)
    end

    return nothing
end

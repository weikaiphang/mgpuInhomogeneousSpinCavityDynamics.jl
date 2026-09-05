

struct ObservableStore{T}
    M::Int
    Nt::Int
    save_spins::Bool

    t::Vector{Float64}
    a::Vector{ComplexF64}
    adad::Vector{ComplexF64}
    n::Vector{Float64}

    Sp::Matrix{ComplexF64}
    Sz::Matrix{ComplexF64}
    adSp::Matrix{ComplexF64}
    adSm::Matrix{ComplexF64}
    adSz::Matrix{ComplexF64}

    host::Vector{Complex{T}}
    saved::Base.RefValue{Int}
end

function ObservableStore(::Type{T}, M::Int, Nt::Int; save_spins::Bool = true) where {T}
    mat() = save_spins ? Matrix{ComplexF64}(undef, M, Nt) : Matrix{ComplexF64}(undef, 0, 0)
    return ObservableStore{T}(
        M, Nt, save_spins,
        Vector{Float64}(undef, Nt),
        Vector{ComplexF64}(undef, Nt),
        Vector{ComplexF64}(undef, Nt),
        Vector{Float64}(undef, Nt),
        mat(), mat(), mat(), mat(), mat(),
        Vector{Complex{T}}(undef, save_prefix_length(M)),
        Ref(0),
    )
end


function record!(store::ObservableStore{T}, prob::MGPUProblem{T}, ireg::Int,
                 k::Int, t::Real) where {T}
    s = prob.shards[1]
    CUDA.device!(s.dev)
    CUDA.synchronize(s.stream)

    npref = length(store.host)
    copyto!(store.host, 1, s.regs[ireg], 1, npref)

    M = store.M
    h = store.host

    store.t[k]    = Float64(t)
    store.a[k]    = ComplexF64(h[IDX_a])
    store.adad[k] = ComplexF64(h[IDX_ad_ad])
    store.n[k]    = Float64(real(h[IDX_ad_a]))

    if store.save_spins
        @inbounds for f in (F_Sp, F_Sz, F_adSp, F_adSm, F_adSz)
            dest = f === F_Sp   ? store.Sp :
                   f === F_Sz   ? store.Sz :
                   f === F_adSp ? store.adSp :
                   f === F_adSm ? store.adSm : store.adSz
            off = small_offset(M, f)
            for j in 1:M
                dest[j, k] = ComplexF64(h[off + j])
            end
        end
    end

    store.saved[] += 1
    return nothing
end


# Standalone multi-GPU driver (not part of the package module).
# The supported public 2nd-order multi-GPU API is
# `mgpu_run` / `mgpu_run_simulation` / `run_simulation` (MGPUsolver.jl).
# This file must stay in lockstep with rhs_2nd_order! / rhs_cpu! / kernels
# (κt = κe + κi; same 2nd-order moments).
using CUDA
CUDA.set_runtime_version!(v"12.4")

t0_wall = time_ns()

using Distributions
using QuadGK
using JLD2
using LinearAlgebra
using NCCL
using Base.Threads

ENV["GKSwstype"] = "100"
using Plots
using Measures
using OrdinaryDiffEq
using OrdinaryDiffEqLowStorageRK
using DiffEqCallbacks

println("Threads.nthreads() = ", Threads.nthreads())


const USER = (

    saved_file_name = "demo_multi.jld2",


    C_ens   = 0.6,
    M_delta = 500,
    M_g     = 100,


    FWHM = 2*pi*1e6,


    g_mean = 2*pi*100,
    g_std  = 2*pi*0.00001,
    g_span_sigma = 3.0,


    Ttotal = 150e-6,
    Tpi    = 75e-6,
    Tw     = 10e-6,


    ke0 = 2*pi*1e6,
    ki0 = 0.0,


    alpha0    = 2.0e4,
    n_wurst   = 20.0,
    ω0        = 0.0,
    BW_factor = 5.0,


    edge_frac = 0.0001,


    w_sil = 10e-6,
    pw_g  = 90e-6,


    Nt_save = 5001,
    reltol  = 1e-8,
    abstol  = 1e-8,


    delta0    = 0.0,
    gamma_    = 0.0,
    gamma_phi = 0.0,
)


C_ens   = USER.C_ens
M_delta = USER.M_delta
M_g     = USER.M_g
const M = M_delta * M_g

FWHM = USER.FWHM
γL   = FWHM / 2
dist_delta = Cauchy(0, γL)
edges_delta = range(-2.5*γL, 2.5*γL, length=M_delta+1)

g_mean       = USER.g_mean
g_std        = USER.g_std
g_span_sigma = USER.g_span_sigma
g_low        = max(0.0, g_mean - g_span_sigma*g_std)
g_high       = g_mean + g_span_sigma*g_std
dist_g       = truncated(Normal(g_mean, g_std), g_low, g_high)
edges_g      = range(g_low, g_high, length=M_g+1)

Ttotal = USER.Ttotal
Tpi    = USER.Tpi
Tw     = USER.Tw

ke0     = USER.ke0
ki0     = USER.ki0
kappa_t = ke0 + ki0

g2_avg = g_mean^2 + g_std^2
N_spin = (C_ens) * (kappa_t * FWHM) / (4 * g2_avg)
# Full-line C_ens inverts N; Lorentzian edges are ±2.5 γ (renormalize=false).
# The ODE sees C_eff = C_ens × ∑p_δ. Do not silently label the run as C_ens.
p_delta_mass = 2 * atan(2.5) / π
C_eff = C_ens * p_delta_mass
println("N_spin = $N_spin")
println("C_ens = $C_ens  (full-line; N is built from this)")
println("∑p_δ  = $p_delta_mass  (Lorentzian span_gamma=2.5, renormalize=false)")
println("C_eff = $C_eff  (ODE optical depth = C_ens × ∑p_δ)")

alpha0  = USER.alpha0
A0      = 0.5 * sqrt(ke0) * alpha0
n_wurst = USER.n_wurst
ω0      = USER.ω0
BW      = USER.BW_factor * FWHM
ω_start = ω0 - BW/2

t0w  = Tpi - Tw/2
edge = Tw * USER.edge_frac

pw_g      = USER.pw_g
edge_g    = pw_g * 0.00001
gate_time = Tpi

timespan = (0.0, Ttotal)
t_save   = collect(range(0, Ttotal, length=USER.Nt_save))
Nt       = length(t_save)

delta0    = USER.delta0
gamma_    = USER.gamma_
gamma_phi = USER.gamma_phi


const N_GPU = length(CUDA.devices())
println("Detected $N_GPU GPUs")

@assert M % N_GPU == 0

const M_LOCAL = M ÷ N_GPU
println("M=$M, M_LOCAL=$M_LOCAL per GPU")

if Threads.nthreads() < N_GPU
    @warn "JULIA_NUM_THREADS=$(Threads.nthreads()) < N_GPU=$N_GPU; per-GPU phases will not fully overlap. Set JULIA_NUM_THREADS >= $N_GPU."
end

for i in 0:N_GPU-1, j in 0:N_GPU-1
    i == j && continue
    can_access = try
        val = Ref{Cint}()
        CUDA.cuDeviceCanAccessPeer(val, CuDevice(i), CuDevice(j))
        val[] == 1
    catch
        false
    end
    if !can_access
        @warn "GPU $i → GPU $j: P2P not accessible. Performance will suffer."
    end
end

const COMMS = NCCL.Communicators([CuDevice(i) for i in 0:N_GPU-1])
println("NCCL communicators created")

CUDA.allowscalar(true)


function bin_means_and_probs(dist, edges)
    Mloc = length(edges) - 1
    probs = zeros(Float64, Mloc)
    means = zeros(Float64, Mloc)
    for j in 1:Mloc
        low, high = edges[j], edges[j+1]
        pj = cdf(dist, high) - cdf(dist, low)
        probs[j] = pj
        if pj > 0
            num, _ = quadgk(x -> x * pdf(dist, x), low, high)
            means[j] = num / pj
        end
    end
    return means, probs
end

function build_2d_bins(N_spin, dist_delta, edges_delta, dist_g, edges_g)
    delta_b_1d, p_delta = bin_means_and_probs(dist_delta, edges_delta)
    g_b_1d,     p_g     = bin_means_and_probs(dist_g, edges_g)
    p_g ./= sum(p_g)

    Md = length(delta_b_1d)
    Mg = length(g_b_1d)

    Nj_2d    = zeros(Float64, Md, Mg)
    delta_2d = zeros(Float64, Md, Mg)
    g_2d     = zeros(Float64, Md, Mg)

    for i in 1:Md, k in 1:Mg
        pij = p_delta[i] * p_g[k]
        Nj_2d[i, k]    = N_spin * pij
        delta_2d[i, k] = delta_b_1d[i]
        g_2d[i, k]     = g_b_1d[k]
    end

    return vec(Nj_2d), vec(delta_2d), vec(g_2d), sum(Nj_2d),
           delta_b_1d, g_b_1d, Nj_2d
end

Nj, delta_b, g_b, N_total, delta_b_1d, g_b_1d, Nj_2d =
    build_2d_bins(N_spin, dist_delta, edges_delta, dist_g, edges_g)

const N_LOCAL = 3 + 9*M_LOCAL + 4*M_LOCAL*M

const L_a     = 1
const L_adad  = 2
const L_ada   = 3
const L_VEC_START = 4

const L_Sp   = L_VEC_START : L_VEC_START + M_LOCAL - 1
const L_Sz   = L_Sp[end]+1 : L_Sp[end] + M_LOCAL
const L_adSp = L_Sz[end]+1 : L_Sz[end] + M_LOCAL
const L_adSm = L_adSp[end]+1 : L_adSp[end] + M_LOCAL
const L_adSz = L_adSm[end]+1 : L_adSm[end] + M_LOCAL
const L_SpSp_s = L_adSz[end]+1 : L_adSz[end] + M_LOCAL
const L_SzSp_s = L_SpSp_s[end]+1 : L_SpSp_s[end] + M_LOCAL
const L_SmSp_s = L_SzSp_s[end]+1 : L_SzSp_s[end] + M_LOCAL
const L_SzSz_s = L_SmSp_s[end]+1 : L_SmSp_s[end] + M_LOCAL

const L_SpSp_x = L_SzSz_s[end]+1 : L_SzSz_s[end] + M_LOCAL*M
const L_SzSp_x = L_SpSp_x[end]+1 : L_SpSp_x[end] + M_LOCAL*M
const L_SmSp_x = L_SzSp_x[end]+1 : L_SzSp_x[end] + M_LOCAL*M
const L_SzSz_x = L_SmSp_x[end]+1 : L_SmSp_x[end] + M_LOCAL*M

@assert L_SzSz_x[end] == N_LOCAL


kappa_e_of_t(t) = ke0
kappa_i_of_t(t) = ki0
g_gate_of_t(t)  = 1.0

function E_of_t_num(t)
    τ = t - t0w
    A = A0 * (1 - abs(sin(pi * (τ - Tw/2) / Tw))^n_wurst)
    ϕ = ω_start * τ + 0.5 * (BW / Tw) * τ^2
    gate = 0.5 * (tanh((t - t0w)/edge) - tanh((t - (t0w + Tw))/edge))
    return gate * A * exp(im * ϕ)
end


struct GPUData
    gpu_id     :: Int
    offset     :: Int

    delta_b_local :: CuVector{Float64}
    Nj_local      :: CuVector{Float64}
    g_b_local     :: CuVector{Float64}
    delta_b_full  :: CuVector{Float64}
    g_b_full      :: CuVector{Float64}

    Sp_full    :: CuVector{ComplexF64}
    Sz_full    :: CuVector{ComplexF64}
    adSp_full  :: CuVector{ComplexF64}
    adSm_full  :: CuVector{ComplexF64}
    adSz_full  :: CuVector{ComplexF64}

    SzSp_T_loc :: CuMatrix{ComplexF64}
end

function build_gpu_data(gpu_id::Int)
    CUDA.device!(gpu_id)
    offset = gpu_id * M_LOCAL


    R = (offset+1):(offset+M_LOCAL)
    delta_b_local = CuArray(Float64.(delta_b[R]))
    Nj_local      = CuArray(Float64.(Nj[R]))
    g_b_local     = CuArray(Float64.(g_b[R]))
    delta_b_full  = CuArray(Float64.(delta_b))
    g_b_full      = CuArray(Float64.(g_b))


    Sp_full   = CUDA.zeros(ComplexF64, M)
    Sz_full   = CUDA.zeros(ComplexF64, M)
    adSp_full = CUDA.zeros(ComplexF64, M)
    adSm_full = CUDA.zeros(ComplexF64, M)
    adSz_full = CUDA.zeros(ComplexF64, M)


    SzSp_T_loc = CUDA.zeros(ComplexF64, M_LOCAL, M)

    return GPUData(
        gpu_id, offset,
        delta_b_local, Nj_local, g_b_local,
        delta_b_full, g_b_full,
        Sp_full, Sz_full, adSp_full, adSm_full, adSz_full,
        SzSp_T_loc
    )
end



function initial_shard(gd::GPUData)
    CUDA.device!(gd.gpu_id)
    offset = gd.offset
    R = (offset+1):(offset+M_LOCAL)

    u_cpu = zeros(ComplexF64, N_LOCAL)

    u_cpu[L_a]    = 0.0
    u_cpu[L_adad] = 0.0
    u_cpu[L_ada]  = 0.0

    Nj_loc = Nj[R]
    u_cpu[L_Sz] .= Nj_loc ./ 2
    u_cpu[L_SzSz_s] .= Nj_loc.^2 ./ 4

    SzSz_x_cpu = zeros(ComplexF64, M_LOCAL, M)
    for j_loc in 1:M_LOCAL
        j_glob = offset + j_loc
        for k in 1:M
            if j_glob != k
                SzSz_x_cpu[j_loc, k] = Nj[j_glob] * Nj[k] / 4
            end
        end
    end
    u_cpu[L_SzSz_x] .= vec(SzSz_x_cpu)

    return CuArray(u_cpu)
end


@inline function nccl_send!(buf, peer::Integer, comm)
    NCCL.LibNCCL.ncclSend(buf, length(buf),
        NCCL.LibNCCL.ncclDataType_t(eltype(buf)), Cint(peer),
        comm, CUDA.stream())
    return nothing
end

@inline function nccl_recv!(buf, peer::Integer, comm)
    NCCL.LibNCCL.ncclRecv(buf, length(buf),
        NCCL.LibNCCL.ncclDataType_t(eltype(buf)), Cint(peer),
        comm, CUDA.stream())
    return nothing
end

function communicate!(gpus::Vector{GPUData}, u, recv)
    BLK = M_LOCAL * M_LOCAL










    NCCL.group() do
        for r in 1:N_GPU
            CUDA.device!(r-1)
            src = u.shards[r]::CuVector{ComplexF64}

            NCCL.Allgather!(
                reinterpret(Float64, @view(src[L_Sp])),
                reinterpret(Float64, gpus[r].Sp_full),
                COMMS[r])
            NCCL.Allgather!(
                reinterpret(Float64, @view(src[L_Sz])),
                reinterpret(Float64, gpus[r].Sz_full),
                COMMS[r])
            NCCL.Allgather!(
                reinterpret(Float64, @view(src[L_adSp])),
                reinterpret(Float64, gpus[r].adSp_full),
                COMMS[r])
            NCCL.Allgather!(
                reinterpret(Float64, @view(src[L_adSm])),
                reinterpret(Float64, gpus[r].adSm_full),
                COMMS[r])
            NCCL.Allgather!(
                reinterpret(Float64, @view(src[L_adSz])),
                reinterpret(Float64, gpus[r].adSz_full),
                COMMS[r])



            rcv = recv.shards[r]::CuVector{ComplexF64}
            self = r - 1
            for s in 0:N_GPU-1
                s == self && continue
                send_block = reinterpret(Float64,
                    @view(src[L_SzSp_x[s*BLK + 1 : (s+1)*BLK]]))
                recv_block = reinterpret(Float64,
                    @view(rcv[L_SzSp_x[s*BLK + 1 : (s+1)*BLK]]))
                nccl_send!(send_block, s, COMMS[r])
                nccl_recv!(recv_block, s, COMMS[r])
            end
        end
    end




    for r in 1:N_GPU
        CUDA.device!(r-1)
        src = u.shards[r]::CuVector{ComplexF64}
        rcv = recv.shards[r]::CuVector{ComplexF64}
        self = r - 1
        selfloc = L_SzSp_x[self*BLK + 1 : (self+1)*BLK]
        copyto!(@view(rcv[selfloc]), @view(src[selfloc]))
    end




    for r in 1:N_GPU
        CUDA.device!(r-1)
        gd = gpus[r]
        src_3d = reshape(@view(recv.shards[r][L_SzSp_x]), M_LOCAL, M_LOCAL, N_GPU)
        dst_3d = reshape(gd.SzSp_T_loc,                   M_LOCAL, M_LOCAL, N_GPU)
        permutedims!(dst_3d, src_3d, (2, 1, 3))
    end


    for r in 1:N_GPU
        CUDA.device!(r-1)
        CUDA.synchronize()
    end
end

@inline function zero_diag!(du::CuVector{ComplexF64}, Lstart::Int, offset::Int)
    base = Lstart + offset * M_LOCAL
    last = base + (M_LOCAL - 1) * (M_LOCAL + 1)
    @view(du[base:(M_LOCAL + 1):last]) .= zero(ComplexF64)
    return nothing
end

function local_rhs!(du_dst::CuVector{ComplexF64},
                    u_src::CuVector{ComplexF64},
                    gd::GPUData,
                    t::Float64)



    cavity_cpu = Array(@view u_src[1:3])
    a_val    = cavity_cpu[1]
    adad_val = cavity_cpu[2]
    ada_val  = cavity_cpu[3]

    Sp_loc    = @view u_src[L_Sp]
    Sz_loc    = @view u_src[L_Sz]
    adSp_loc  = @view u_src[L_adSp]
    adSm_loc  = @view u_src[L_adSm]
    adSz_loc  = @view u_src[L_adSz]

    SpSp_s_loc = @view u_src[L_SpSp_s]
    SzSp_s_loc = @view u_src[L_SzSp_s]
    SmSp_s_loc = @view u_src[L_SmSp_s]
    SzSz_s_loc = @view u_src[L_SzSz_s]

    SpSp_x_loc = reshape(@view(u_src[L_SpSp_x]), M_LOCAL, M)
    SzSp_x_loc = reshape(@view(u_src[L_SzSp_x]), M_LOCAL, M)
    SmSp_x_loc = reshape(@view(u_src[L_SmSp_x]), M_LOCAL, M)
    SzSz_x_loc = reshape(@view(u_src[L_SzSz_x]), M_LOCAL, M)


    dSp_loc    = @view du_dst[L_Sp]
    dSz_loc    = @view du_dst[L_Sz]
    dadSp_loc  = @view du_dst[L_adSp]
    dadSm_loc  = @view du_dst[L_adSm]
    dadSz_loc  = @view du_dst[L_adSz]

    dSpSp_s_loc = @view du_dst[L_SpSp_s]
    dSzSp_s_loc = @view du_dst[L_SzSp_s]
    dSmSp_s_loc = @view du_dst[L_SmSp_s]
    dSzSz_s_loc = @view du_dst[L_SzSz_s]

    dSpSp_x_loc = reshape(@view(du_dst[L_SpSp_x]), M_LOCAL, M)
    dSzSp_x_loc = reshape(@view(du_dst[L_SzSp_x]), M_LOCAL, M)
    dSmSp_x_loc = reshape(@view(du_dst[L_SmSp_x]), M_LOCAL, M)
    dSzSz_x_loc = reshape(@view(du_dst[L_SzSz_x]), M_LOCAL, M)


    Sp_full   = gd.Sp_full
    Sz_full   = gd.Sz_full
    adSp_full = gd.adSp_full
    adSm_full = gd.adSm_full
    adSz_full = gd.adSz_full


    SzSp_T_loc = gd.SzSp_T_loc


    κe_t = kappa_e_of_t(t)
    κt_t = κe_t + kappa_i_of_t(t)
    g_t_loc  = gd.g_b_local .* g_gate_of_t(t)
    g_t_full = gd.g_b_full  .* g_gate_of_t(t)
    E_t  = E_of_t_num(t)
    δ_loc = gd.delta_b_local






    d_a = sqrt(κe_t) * E_t -
          1im * delta0 * a_val -
          1im * sum(g_t_full .* conj.(Sp_full)) -
          0.5 * κt_t * a_val

    d_adad = 2im * delta0 * adad_val +
             2im * sum(g_t_full .* adSp_full) -
             κt_t * adad_val +
             2 * sqrt(κe_t) * conj(a_val) * conj(E_t)

    d_ada = 1im * sum(g_t_full .* conj.(adSm_full)) -
            1im * sum(g_t_full .* adSm_full) -
            κt_t * ada_val +
            sqrt(κe_t) * E_t * conj(a_val) +
            sqrt(κe_t) * conj(E_t) * a_val

    copyto!(@view(du_dst[1:3]), ComplexF64[d_a, d_adad, d_ada])




    dSp_loc .= 1im .* δ_loc .* Sp_loc .- 2im .* g_t_loc .* adSz_loc
    dSz_loc .= .-1im .* g_t_loc .* conj.(adSm_loc) .+ 1im .* g_t_loc .* adSm_loc




    # Same-bin + off-diagonal cross only (kernels skip k==j). The unused
    # cross diagonal is not a chimera merge — just the same diag_mask idea.
    sumgSpSp_jk = SpSp_s_loc .* g_t_loc .+ SpSp_x_loc * g_t_full
    sumgSmSp_jk = SmSp_s_loc .* g_t_loc .+ SmSp_x_loc * g_t_full
    sumgSzSp_jk = SzSp_s_loc .* g_t_loc .+ SzSp_x_loc * g_t_full
    @inbounds for jl in 1:M_LOCAL
        j = gd.offset + jl
        gj = g_t_full[j]
        sumgSpSp_jk[jl] -= SpSp_x_loc[jl, j] * gj
        sumgSmSp_jk[jl] -= SmSp_x_loc[jl, j] * gj
        sumgSzSp_jk[jl] -= SzSp_x_loc[jl, j] * gj
    end

    dadSp_loc .= (
        1im * delta0 .* adSp_loc .+ 1im .* δ_loc .* adSp_loc
        .+ 1im .* sumgSpSp_jk
        .- 0.5 .* κt_t .* adSp_loc .+ sqrt(κe_t) .* conj(E_t) .* Sp_loc
        .- 2im .* g_t_loc .* (2 .* conj(a_val) .* adSz_loc .+ adad_val .* Sz_loc .- 2 .* conj(a_val)^2 .* Sz_loc)
    )

    dadSm_loc .= (
        1im * delta0 .* adSm_loc .- 1im .* δ_loc .* adSm_loc
        .+ 2im .* g_t_loc .* Sz_loc .+ 1im .* sumgSmSp_jk
        .- 0.5 .* κt_t .* adSm_loc .+ sqrt(κe_t) .* conj(E_t) .* conj.(Sp_loc)
        .+ 2im .* g_t_loc .* (conj.(adSz_loc) .* conj(a_val) .+ a_val .* adSz_loc .+ Sz_loc .* ada_val .- 2 .* conj(a_val) .* a_val .* Sz_loc)
    )

    dadSz_loc .= (
        1im * delta0 .* adSz_loc .+ 1im .* sumgSzSp_jk
        .- 0.5 .* κt_t .* adSz_loc .+ sqrt(κe_t) .* conj(E_t) .* Sz_loc
        .- 1im .* g_t_loc .* (Sp_loc .+ Sp_loc .* ada_val .+ conj(a_val) .* conj.(adSm_loc) .+ a_val .* adSp_loc .- 2 .* Sp_loc .* conj(a_val) .* a_val)
        .+ 1im .* g_t_loc .* (2 .* conj(a_val) .* adSm_loc .+ adad_val .* conj.(Sp_loc) .- 2 .* conj(a_val)^2 .* conj.(Sp_loc))
    )




    dSpSp_s_loc .= (
        2im .* δ_loc .* SpSp_s_loc .+ 2im .* g_t_loc .* adSp_loc
        .- 4im .* g_t_loc .* (Sp_loc .* adSz_loc .+ SzSp_s_loc .* conj(a_val) .+ adSp_loc .* Sz_loc .- 2 .* Sp_loc .* conj(a_val) .* Sz_loc)
    )

    dSzSp_s_loc .= (
        1im .* δ_loc .* SzSp_s_loc
        .- 1im .* g_t_loc .* (2 .* Sp_loc .* conj.(adSm_loc) .+ a_val .* SpSp_s_loc .- 2 .* Sp_loc.^2 .* a_val)
        .+ 1im .* g_t_loc .* (Sp_loc .* adSm_loc .+ conj(a_val) .* SmSp_s_loc .+ adSp_loc .* conj.(Sp_loc) .- 2 .* Sp_loc .* conj(a_val) .* conj.(Sp_loc))
        .- 2im .* g_t_loc .* (SzSz_s_loc .* conj(a_val) .+ 2 .* adSz_loc .* Sz_loc .- 2 .* conj(a_val) .* Sz_loc.^2)
    )

    dSmSp_s_loc .= (
        2im .* g_t_loc .* (conj.(adSz_loc) .* Sp_loc .+ SzSp_s_loc .* a_val .+ conj.(adSm_loc) .* Sz_loc .- 2 .* Sp_loc .* a_val .* Sz_loc)
        .- 2im .* g_t_loc .* (conj(a_val) .* conj.(SzSp_s_loc) .+ conj.(Sp_loc) .* adSz_loc .+ adSm_loc .* Sz_loc .- 2 .* conj(a_val) .* conj.(Sp_loc) .* Sz_loc)
    )

    dSzSz_s_loc .= (
        1im .* g_t_loc .* conj.(adSm_loc) .- 1im .* g_t_loc .* adSm_loc
        .- 2im .* g_t_loc .* (conj.(adSz_loc) .* Sp_loc .+ SzSp_s_loc .* a_val .+ conj.(adSm_loc) .* Sz_loc .- 2 .* Sp_loc .* a_val .* Sz_loc)
        .+ 2im .* g_t_loc .* (conj(a_val) .* conj.(SzSp_s_loc) .+ conj.(Sp_loc) .* adSz_loc .+ adSm_loc .* Sz_loc .- 2 .* conj(a_val) .* conj.(Sp_loc) .* Sz_loc)
    )




    Δ_col = reshape(δ_loc, M_LOCAL, 1)
    Δ_row = reshape(gd.delta_b_full, 1, M)

    G_col = reshape(g_t_loc, M_LOCAL, 1)
    G_row = reshape(g_t_full, 1, M)

    Sp_col  = reshape(Sp_loc, M_LOCAL, 1)
    Sp_row  = reshape(Sp_full, 1, M)
    Sz_col  = reshape(Sz_loc, M_LOCAL, 1)
    Sz_row  = reshape(Sz_full, 1, M)

    adSp_col = reshape(adSp_loc, M_LOCAL, 1)
    adSp_row = reshape(adSp_full, 1, M)
    adSm_col = reshape(adSm_loc, M_LOCAL, 1)
    adSm_row = reshape(adSm_full, 1, M)
    adSz_col = reshape(adSz_loc, M_LOCAL, 1)
    adSz_row = reshape(adSz_full, 1, M)






    dSpSp_x_loc .= (
        1im .* (Δ_col .+ Δ_row) .* SpSp_x_loc
        .- 2im .* G_col .* (Sp_row .* adSz_col .+ conj(a_val) .* SzSp_x_loc .+ adSp_row .* Sz_col .- 2 .* Sp_row .* conj(a_val) .* Sz_col)
        .- 2im .* G_row .* (Sp_col .* adSz_row .+ conj(a_val) .* SzSp_T_loc .+ Sz_row .* adSp_col .- 2 .* Sp_col .* conj(a_val) .* Sz_row)
    )
    zero_diag!(du_dst, first(L_SpSp_x), gd.offset)






    dSzSp_x_loc .= (
        1im .* Δ_row .* SzSp_x_loc
        .- 1im .* G_col .* (Sp_row .* conj.(adSm_col) .+ Sp_col .* conj.(adSm_row) .+ a_val .* SpSp_x_loc .- 2 .* Sp_row .* Sp_col .* a_val)
        .+ 1im .* G_col .* (Sp_row .* adSm_col .+ conj(a_val) .* SmSp_x_loc .+ adSp_row .* conj.(Sp_col) .- 2 .* Sp_row .* conj(a_val) .* conj.(Sp_col))
        .- 2im .* G_row .* (SzSz_x_loc .* conj(a_val) .+ Sz_row .* adSz_col .+ Sz_col .* adSz_row .- 2 .* conj(a_val) .* Sz_row .* Sz_col)
    )
    zero_diag!(du_dst, first(L_SzSp_x), gd.offset)




    dSmSp_x_loc .= (
        .- 1im .* Δ_col .* SmSp_x_loc .+ 1im .* Δ_row .* SmSp_x_loc
        .+ 2im .* G_col .* (conj.(adSz_col) .* Sp_row .+ conj.(adSm_row) .* Sz_col .+ a_val .* SzSp_x_loc .- 2 .* Sp_row .* a_val .* Sz_col)
        .- 2im .* G_row .* (conj(a_val) .* conj.(SzSp_T_loc) .+ Sz_row .* adSm_col .+ conj.(Sp_col) .* adSz_row .- 2 .* conj(a_val) .* Sz_row .* conj.(Sp_col))
    )
    zero_diag!(du_dst, first(L_SmSp_x), gd.offset)


    dSzSz_x_loc .= (
        .- 1im .* G_col .* (Sp_col .* conj.(adSz_row) .+ conj.(adSm_col) .* Sz_row .+ a_val .* SzSp_T_loc .- 2 .* Sp_col .* Sz_row .* a_val)
        .+ 1im .* G_col .* (conj(a_val) .* conj.(SzSp_T_loc) .+ Sz_row .* adSm_col .+ conj.(Sp_col) .* adSz_row .- 2 .* conj(a_val) .* Sz_row .* conj.(Sp_col))
        .- 1im .* G_row .* (conj.(adSz_col) .* Sp_row .+ conj.(adSm_row) .* Sz_col .+ a_val .* SzSp_x_loc .- 2 .* Sp_row .* a_val .* Sz_col)
        .+ 1im .* G_row .* (conj.(SzSp_x_loc) .* conj(a_val) .+ adSm_row .* Sz_col .+ adSz_col .* conj.(Sp_row) .- 2 .* conj(a_val) .* Sz_col .* conj.(Sp_row))
    )
    zero_diag!(du_dst, first(L_SzSz_x), gd.offset)

    return nothing
end


struct Sharded{T,A<:CuVector{T}} <: AbstractVector{T}
    shards::Vector{A}
end
Sharded(shards::Vector{A}) where {T, A<:CuVector{T}} = Sharded{T,A}(shards)

Base.size(s::Sharded)   = (sum(length, s.shards),)
Base.length(s::Sharded) = sum(length, s.shards)
Base.axes(s::Sharded)   = (Base.OneTo(length(s)),)
Base.eltype(::Sharded{T}) where {T} = T
nshards(s::Sharded)     = length(s.shards)

function Base.similar(s::Sharded{T}) where {T}
    Sharded([ (CUDA.device!(i-1); similar(s.shards[i])) for i in 1:nshards(s) ])
end
Base.similar(s::Sharded{T}, ::Type{S}) where {T,S} =
    Sharded([ (CUDA.device!(i-1); similar(s.shards[i], S)) for i in 1:nshards(s) ])

function Base.copy(s::Sharded{T}) where {T}
    Sharded([ (CUDA.device!(i-1); copy(s.shards[i])) for i in 1:nshards(s) ])
end

function Base.copyto!(dst::Sharded, src::Sharded)
    for i in 1:nshards(dst)
        CUDA.device!(i-1); copyto!(dst.shards[i], src.shards[i])
    end
    dst
end

function Base.fill!(s::Sharded, x)
    for i in 1:nshards(s)
        CUDA.device!(i-1); fill!(s.shards[i], x)
    end
    s
end
Base.zero(s::Sharded) = (z = similar(s); fill!(z, zero(eltype(s))); z)

Base.any(f::Function, s::Sharded) =
    any(i -> (CUDA.device!(i-1); any(f, s.shards[i])), 1:nshards(s))

function Base.getindex(s::Sharded, idx::Integer)
    nl = length(s.shards[1]); g = (idx-1) ÷ nl + 1; loc = (idx-1) % nl + 1
    CUDA.device!(g-1); return CUDA.@allowscalar s.shards[g][loc]
end
function Base.setindex!(s::Sharded, v, idx::Integer)
    nl = length(s.shards[1]); g = (idx-1) ÷ nl + 1; loc = (idx-1) % nl + 1
    CUDA.device!(g-1); CUDA.@allowscalar s.shards[g][loc] = v; v
end

struct ShardedStyle <: Broadcast.AbstractArrayStyle{1} end
Base.BroadcastStyle(::Type{<:Sharded}) = ShardedStyle()
(::Type{ShardedStyle})(::Val{N}) where {N} = ShardedStyle()
Base.BroadcastStyle(::ShardedStyle, ::Broadcast.DefaultArrayStyle) = ShardedStyle()
Base.BroadcastStyle(::Broadcast.DefaultArrayStyle, ::ShardedStyle) = ShardedStyle()

@inline _shard(x, i) = x
@inline _shard(s::Sharded, i) = s.shards[i]
@inline _shard(bc::Broadcast.Broadcasted, i) =
    Broadcast.broadcasted(bc.f, map(a -> _shard(a, i), bc.args)...)

@inline _find(bc::Broadcast.Broadcasted) = _find(bc.args)
@inline _find(args::Tuple) = _find(_find(args[1]), Base.tail(args))
@inline _find(x) = x
@inline _find(a::Sharded, _rest) = a
@inline _find(::Any, rest)        = _find(rest)
@inline _find(::Tuple{})          = error("no Sharded in broadcast")

function Base.copyto!(dst::Sharded, bc::Broadcast.Broadcasted{ShardedStyle})
    for i in 1:nshards(dst)
        CUDA.device!(i-1); copyto!(dst.shards[i], _shard(bc, i))
    end
    dst
end
function Base.copy(bc::Broadcast.Broadcasted{ShardedStyle})
    proto = _find(bc)
    Sharded([
        (CUDA.device!(i-1); copy(_shard(bc, i))) for i in 1:nshards(proto)
    ])
end

function _sumabs2(s::Sharded)
    total = 0.0
    for i in 1:nshards(s)
        CUDA.device!(i-1); total += sum(abs2, s.shards[i])
    end
    return total
end
sharded_norm(u::Sharded, t) = sqrt(_sumabs2(u) / length(u))
sharded_norm(x::Number,  t) = abs(x)
sharded_norm(x, t)          = sqrt(sum(abs2, x) / length(x))

struct RHSParams
    gpus::Vector{GPUData}
end

function f!(du::Sharded, u::Sharded, p::RHSParams, t)
    gpus = p.gpus





    communicate!(gpus, u, du)





    for r in 1:N_GPU
        CUDA.device!(r-1)
        local_rhs!(du.shards[r], u.shards[r], gpus[r], t)
    end
    for r in 1:N_GPU
        CUDA.device!(r-1); CUDA.synchronize()
    end
    return nothing
end



println("Allocating GPU parameter/comm buffers...")
gpu_data = [build_gpu_data(i) for i in 0:N_GPU-1]

println("Building initial state shards...")
u0 = Sharded([initial_shard(gpu_data[r]) for r in 1:N_GPU])



a_save    = Vector{ComplexF64}(undef, Nt)
adad_save = Vector{ComplexF64}(undef, Nt)
n_save    = Vector{Float64}(undef, Nt)

Sp_save   = Matrix{ComplexF64}(undef, M, Nt)
Sz_save   = Matrix{ComplexF64}(undef, M, Nt)

adSp_save = Matrix{ComplexF64}(undef, M, Nt)
adSm_save = Matrix{ComplexF64}(undef, M, Nt)
adSz_save = Matrix{ComplexF64}(undef, M, Nt)

Σp_save = Vector{ComplexF64}(undef, Nt)
Σz_save = Vector{ComplexF64}(undef, Nt)

const progress_next = Ref(0)

function save_func(u::Sharded, t, integrator)

    idx = argmin(abs.(t_save .- t))


    CUDA.device!(0)
    cav0 = Array(@view u.shards[1][1:3])
    a_save[idx]    = cav0[1]
    adad_save[idx] = cav0[2]
    n_save[idx]    = real(cav0[3])

    Sp_full  = zeros(ComplexF64, M)
    Sz_full  = zeros(ComplexF64, M)
    adSp_f   = zeros(ComplexF64, M)
    adSm_f   = zeros(ComplexF64, M)
    adSz_f   = zeros(ComplexF64, M)

    for r in 1:N_GPU
        CUDA.device!(r-1)
        gd = gpu_data[r]; offset = gd.offset
        R = (offset+1):(offset+M_LOCAL)
        sh = u.shards[r]
        Sp_full[R] .= Array(@view sh[L_Sp])
        Sz_full[R] .= Array(@view sh[L_Sz])
        adSp_f[R]  .= Array(@view sh[L_adSp])
        adSm_f[R]  .= Array(@view sh[L_adSm])
        adSz_f[R]  .= Array(@view sh[L_adSz])
    end

    Sp_save[:, idx]   .= Sp_full
    Sz_save[:, idx]   .= Sz_full
    adSp_save[:, idx] .= adSp_f
    adSm_save[:, idx] .= adSm_f
    adSz_save[:, idx] .= adSz_f

    Σp_save[idx] = sum(Sp_full)
    Σz_save[idx] = sum(Sz_full)

    if idx % 500 == 0 || idx == 1
        println("  Saved point $idx / $Nt at t=$(t*1e6) μs")
    end





    pct = floor(Int, 100 * idx / Nt)
    if pct >= progress_next[]
        elapsed = (time_ns() - t0_wall) / 1e9
        println("  [progress] $(progress_next[])%  (save $idx/$Nt, t=$(round(t*1e6, digits=1)) μs, ",
                "elapsed $(round(elapsed, digits=1)) s)")
        flush(stdout)

        while progress_next[] <= pct
            progress_next[] += 5
        end
    end













    if idx == 1 || idx % 500 == 0
        for r in 1:N_GPU
            CUDA.device!(r-1)
            free_b, total_b = CUDA.memory_info()
            used_b = total_b - free_b
            println("  [mem] GPU $(r-1): used $(round(used_b/2^30, digits=1)) / ",
                    "$(round(total_b/2^30, digits=1)) GiB  (free $(round(free_b/2^30, digits=1)) GiB)")
        end

        CUDA.device!(0)
        print("  [mem] pool (GPU 0): "); CUDA.pool_status(); println()
        flush(stdout)
    end




    if idx % 200 == 0
        GC.gc(false)
        for r in 1:N_GPU
            CUDA.device!(r-1); CUDA.reclaim()
        end
    end
    return nothing
end

save_cb = FunctionCallingCallback(save_func;
                                  funcat = t_save,
                                  func_everystep = false,
                                  func_start = true)



const INTEGRATOR = RDPK3SpFSAL510()

println("Starting solver: ", typeof(INTEGRATOR))

prob = ODEProblem(f!, u0, (0.0, Ttotal), RHSParams(gpu_data))


dtmax_resolve = Ttotal / USER.Nt_save

dt_init = (Ttotal / USER.Nt_save) * 0.01

GC.gc()
for r in 1:N_GPU
    CUDA.device!(r-1); CUDA.reclaim()
end
warm = similar(u0)
communicate!(gpu_data, u0, warm)
warm = nothing
GC.gc()
for r in 1:N_GPU
    CUDA.device!(r-1); CUDA.reclaim()
end

println("NCCL warmup done; entering solve() ...")
flush(stdout)

sol = solve(prob, INTEGRATOR;
            dt = dt_init,
            abstol = USER.abstol,
            reltol = USER.reltol,
            internalnorm = sharded_norm,
            callback = save_cb,
            tstops = t_save,
            save_everystep = false,
            save_start = false,
            save_end = false,









            alias = OrdinaryDiffEq.SciMLBase.ODEAliasSpecifier(alias_u0 = true),
            dtmax = dtmax_resolve,
            maxiters = 10_000_000)

println("Solver returncode: ", sol.retcode)

elapsed_seconds = (time_ns() - t0_wall) / 1e9
println("Time taken: $elapsed_seconds seconds")


println("Post-processing...")

if !isdir("fig")
    mkdir("fig")
end

times = t_save .* 1e6
times_cpu = t_save

a_sol    = a_save
n_sol    = n_save
adad_sol = adad_save

kappa_e_arr = [kappa_e_of_t(t) for t in times_cpu]
sqrt_ke     = sqrt.(kappa_e_arr)
E_of_t_arr  = [E_of_t_num(t) for t in times_cpu]
E_in_array  = real.(E_of_t_arr)

a_out_x = real.(E_of_t_arr) .- sqrt_ke .* real.(a_sol)
a_out_p = imag.(E_of_t_arr) .- sqrt_ke .* imag.(a_sol)

data = (
    times = times,
    times_cpu = times_cpu,

    a_sol = a_sol,
    n_sol = n_sol,
    adad_sol = adad_sol,

    Sp = Sp_save,
    Sz = Sz_save,

    Σp = Σp_save,
    Σz = Σz_save,

    adSp = adSp_save,
    adSm = adSm_save,
    adSz = adSz_save,

    kappa_e_arr = kappa_e_arr,
    sqrt_ke = sqrt_ke,
    E_of_t_arr = E_of_t_arr,
    E_in_array = E_in_array,
    a_out_x = a_out_x,
    a_out_p = a_out_p,

    M_delta = M_delta,
    M_g = M_g,
    M_total = M,

    delta_b = delta_b,
    g_b = g_b,
    Nj = Nj,

    delta_b_1d = delta_b_1d,
    g_b_1d = g_b_1d,
    Nj_2d = Nj_2d
)

@save USER.saved_file_name data
println("Saved to: ", USER.saved_file_name)

println("Generating figures into fig/ ...")
try
    if !isdir("fig")
        mkdir("fig")
    end

    default(
        size = (900, 600),
        dpi = 300,
        lw = 2.5,
        guidefont = font(16),
        tickfont = font(13),
        legendfont = font(13),
        titlefont = font(18),
        grid = true,
        left_margin = 3mm,
    )

    N_total_fig = sum(Nj)

    Σx = real.(Σp_save)
    Σy = imag.(Σp_save)
    Σz_real = real.(Σz_save)

    sx_avg = Σx ./ (N_total_fig / 2)
    sy_avg = Σy ./ (N_total_fig / 2)
    sz_avg = Σz_real ./ (N_total_fig / 2)


    plt = plot(times, sx_avg, lw=2, label="⟨σx⟩ = Re⟨S+⟩/(N/2)")
    plot!(plt, times, sy_avg, lw=2, label="⟨σy⟩ = Im⟨S+⟩/(N/2)")
    plot!(plt, times, sz_avg, lw=2, label="⟨σz⟩")
    xlabel!(plt, "time (μs)"); ylabel!(plt, "mean value")
    title!(plt, "Spin dynamics (N_total=$(round(N_total_fig)), Mδ=$(M_delta), Mg=$(M_g))")
    savefig(plt, "fig/plot_spins_pm.png")


    plt = plot(times, E_in_array, lw=2, label="a_in_real(t)")
    plot!(plt, times, a_out_x, lw=2, label="a_out_real(t)")
    plot!(plt, times, a_out_p, lw=2, label="a_out_imag(t)")
    xlabel!(plt, "time (μs)"); ylabel!(plt, "Amplitude (s^{-1/2})")
    title!(plt, "Input / output field")
    savefig(plt, "fig/plot_io_pm.png")


    mid_idx = div(M, 2)
    keep_bins = unique([1, mid_idx, mid_idx + 1, M])
    plt = plot()
    for b in keep_bins
        plot!(plt, times, real.(Sz_save[b, :]) ./ (Nj[b] / 2), lw=2, label="⟨σz⟩(bin=$b)")
    end
    xlabel!(plt, "time (μs)"); ylabel!(plt, "mean value")
    title!(plt, "Selected-bin Sz dynamics (M_total=$M)")
    savefig(plt, "fig/plot_selected_bins_pm.png")


    plt = plot()
    for b in keep_bins
        plot!(plt, times, real.(Sp_save[b, :]) ./ (Nj[b] / 2), lw=2, label="Re⟨S+⟩ bin=$b")
    end
    xlabel!(plt, "time (μs)"); ylabel!(plt, "mean value")
    title!(plt, "Selected-bin Re⟨S+⟩ dynamics")
    savefig(plt, "fig/plot_selected_bins_Sp_real_pm.png")


    plt = plot(times, n_sol, lw=2, label="⟨a†a⟩")
    xlabel!(plt, "time (μs)"); ylabel!(plt, "photon number")
    title!(plt, "Photon Number (M_total=$M)")
    savefig(plt, "fig/photon_number_pm.png")


    Dx = 0.25 .* (adad_sol .+ conj.(adad_sol) .+ 2 .* n_sol .+ 1 .- (a_sol .+ conj.(a_sol)).^2)
    Dy = 0.25 .* (.-adad_sol .- conj.(adad_sol) .+ 2 .* n_sol .+ 1 .+ (a_sol .- conj.(a_sol)).^2)
    Dx_plot = real.(Dx); Dy_plot = real.(Dy); Dt = Dx_plot .+ Dy_plot

    plt = plot(times, Dx_plot, lw=2, label="Var(X)")
    plot!(plt, times, Dy_plot, lw=2, label="Var(Y)")
    xlabel!(plt, "time (μs)"); ylabel!(plt, "Quadrature Variance")
    title!(plt, "Quadrature Variance (M_total=$M)")
    savefig(plt, "fig/quadrature_variance_pm.png")


    plt = plot(times, Dt, lw=2, label="Var(X)+Var(Y)")
    xlabel!(plt, "time (μs)"); ylabel!(plt, "Total Variance")
    title!(plt, "Total Variance (M_total=$M)")
    savefig(plt, "fig/total_variance_pm.png")

    println("Figures saved to fig/")
catch err
    @warn "Figure generation failed (the .jld2 data file is still saved)." exception=(err, catch_backtrace())
end
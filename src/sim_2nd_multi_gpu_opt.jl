# ============================================================
# ROSE — 2nd-order cumulant MULTI-GPU simulation
# ============================================================
#
# Evolves the second-order cumulants of M spin-cavity systems (ROSE spin-echo
# experiment), distributed across the available GPUs. Each GPU owns a row slab
# of the state; the per-GPU shards are exchanged on every RHS evaluation via
# NCCL — AllGather for the spin vectors, AllToAll for the SzSp cross-correlation
# transpose. The distributed state is a custom `Sharded` vector (one
# CuVector{ComplexF64} per GPU): its broadcast fans the integrator's axpy
# updates (u = u + dt*k) across devices via CUDA.device!, and its norm reduces
# the embedded error estimate across shards. OrdinaryDiffEq owns the integrator
# registers; this file owns the persistent per-GPU parameter/comm buffers
# (GPUData) and the RHS.
#
# Time integration uses RDPK3SpFSAL510 — a 5th-order, adaptive, low-storage
# embedded scheme (Ranocha-Dalcin-Parsani-Ketcheson 3S*+, FSAL, 10 stages). A
# low-storage scheme is essential here: the state is dominated by four M_LOCAL×M
# cross-correlation slabs and the integrator replicates the whole state once per
# register, so the register count directly multiplies the largest buffers.
# RDPK3SpFSAL510 needs only ~4 full-state registers (3 working + 1 embedded
# estimate; extra stages add none), keeping the cross slabs replicated few
# enough times that large M still fits in Float64.
#
# ACCURACY: dtmax = Ttotal/Nt_save together with tstops = t_save (see SOLVE)
# pin the stepper to the save grid, so dt never exceeds the save spacing. This
# keeps the fast WURST/chirp carrier resolved through the pulse; letting dt grow
# there aliases the carrier and produces spurious negative variance.
#
# Compatibility: Julia 1.12.2, CUDA 12.4, NCCL. Requires OrdinaryDiffEqLowStorageRK
# (RDPK3SpFSAL510) and DiffEqCallbacks (FunctionCallingCallback).
#
# ENVIRONMENT: export NCCL_P2P_LEVEL=NVL ; JULIA_NUM_THREADS >= N_GPU
# ============================================================

using CUDA
CUDA.set_runtime_version!(v"12.4")

t0_wall = time_ns()

using Distributions
using QuadGK
using JLD2
using LinearAlgebra
using NCCL
using Base.Threads          # only for nthreads() introspection (hot loop is sequential; see f!)

# --- Plotting (figures saved to fig/). Headless GR for HPC compute nodes:
#     GKSwstype=100 = offscreen, so no X display is needed. Must be set
#     before the first savefig (GR reads it lazily). ---
ENV["GKSwstype"] = "100"
using Plots
using Measures
using OrdinaryDiffEq            # ODEProblem, solve, Tsit5
using OrdinaryDiffEqLowStorageRK   # RDPK3SpFSAL510 (low-storage; loaded directly)
using DiffEqCallbacks       # FunctionCallingCallback — confirmed installed

println("Threads.nthreads() = ", Threads.nthreads())

# ============================================================
# 0) USER PARAMETERS
# ============================================================

const USER = (
    # --- output ---
    saved_file_name = "demo_multi.jld2",

    # --- ensemble ---
    C_ens   = 0.6,
    M_delta = 500,
    M_g     = 100,

    # --- inhomogeneous broadening (Lorentzian) ---
    FWHM = 2*pi*1e6,

    # --- g-inhomogeneity ---
    g_mean = 2*pi*100,
    g_std  = 2*pi*0.00001,
    g_span_sigma = 3.0,

    # --- new time sequence ---
    Ttotal = 150e-6,      # total simulation time
    Tpi    = 75e-6,      # center time of the WURST pulse
    Tw     = 10e-6,       # WURST pulse duration

    # --- cavity/coupling ---
    ke0 = 2*pi*1e6,

    # --- WURST parameters ---
    alpha0    = 2.0e4,
    n_wurst   = 20.0,
    ω0        = 0.0,
    BW_factor = 5.0,

    # --- WURST pulse edge ---
    edge_frac = 0.0001,

    # --- echo gating knobs (currently disabled) ---
    w_sil = 10e-6,
    pw_g  = 90e-6,

    # --- solver/save ---
    Nt_save = 5001,
    reltol  = 1e-8,
    abstol  = 1e-8,

    # --- decoherence ---
    delta0    = 0.0,
    gamma_    = 0.0,
    gamma_phi = 0.0,
)

# ============================================================
# 1) DERIVED PARAMETERS
# ============================================================

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
kappa_t = ke0

g2_avg = g_mean^2 + g_std^2
N_spin = (C_ens) * (kappa_t * FWHM) / (4 * g2_avg)
println("N_spin = $N_spin")

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

# ============================================================
# 2) MULTI-GPU SETUP
# ============================================================

const N_GPU = length(CUDA.devices())
println("Detected $N_GPU GPUs")

@assert M % N_GPU == 0 "M=$M must be divisible by N_GPU=$N_GPU. Adjust M_delta/M_g."

const M_LOCAL = M ÷ N_GPU
println("M=$M, M_LOCAL=$M_LOCAL per GPU")

if Threads.nthreads() < N_GPU
    @warn "JULIA_NUM_THREADS=$(Threads.nthreads()) < N_GPU=$N_GPU; per-GPU phases will not fully overlap. Set JULIA_NUM_THREADS >= $N_GPU."
end

# Verify P2P access
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

# Create NCCL communicators (one per GPU)
const COMMS = NCCL.Communicators([CuDevice(i) for i in 0:N_GPU-1])
println("NCCL communicators created")

# Allow scalar GPU indexing as a safety net. The hot solver path does NOT use
# it — cavity reads/writes are batched D2H/H2D copies and the error norm is an
# on-device reduction — so this only guards rarely-hit paths during a long run.
CUDA.allowscalar(true)

# ============================================================
# 3) 2D ENSEMBLE DISCRETIZATION (runs on CPU)
# ============================================================

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

# ============================================================
# 4) LOCAL STATE LAYOUT
# ============================================================
#
# Per-GPU packed CuVector layout:
#   [0]  a, ad_ad, ad_a                        3 scalars (replicated)
#   [1]  Sp_local                               M_LOCAL
#   [2]  Sz_local                               M_LOCAL
#   [3]  adSp_local                             M_LOCAL
#   [4]  adSm_local                             M_LOCAL
#   [5]  adSz_local                             M_LOCAL
#   [6]  SpSp_same_local                        M_LOCAL
#   [7]  SzSp_same_local                        M_LOCAL
#   [8]  SmSp_same_local                        M_LOCAL
#   [9]  SzSz_same_local                        M_LOCAL
#   [10] SpSp_cross_slab   (M_LOCAL × M flat)   M_LOCAL*M
#   [11] SzSp_cross_slab   (M_LOCAL × M flat)   M_LOCAL*M
#   [12] SmSp_cross_slab   (M_LOCAL × M flat)   M_LOCAL*M
#   [13] SzSz_cross_slab   (M_LOCAL × M flat)   M_LOCAL*M
#
const N_LOCAL = 3 + 9*M_LOCAL + 4*M_LOCAL*M

# Index ranges into the local packed vector
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

@assert L_SzSz_x[end] == N_LOCAL "State layout mismatch"

# ============================================================
# 5) PULSES (same physics, evaluated on CPU or broadcast)
# ============================================================

kappa_e_of_t(t) = ke0
g_gate_of_t(t)  = 1.0

function E_of_t_num(t)
    τ = t - t0w
    A = A0 * (1 - abs(sin(pi * (τ - Tw/2) / Tw))^n_wurst)
    ϕ = ω_start * τ + 0.5 * (BW / Tw) * τ^2
    gate = 0.5 * (tanh((t - t0w)/edge) - tanh((t - (t0w + Tw))/edge))
    return gate * A * exp(im * ϕ)
end

# ============================================================
# 6) PER-GPU PERSISTENT DATA  (parameters + comm buffers only)
#
# Solver STATE vectors live in OrdinaryDiffEq's `Sharded` registers, not here.
# GPUData holds only per-GPU parameters and NCCL scratch. To keep the persistent
# footprint small it stores just ONE full M_LOCAL×M cross buffer (SzSp_T_loc);
# three buffers a naive layout would also need are avoided:
#   • No diagonal mask. Each cross slab's excluded diagonal is zeroed in place
#     after the (unmasked) assignment via a small strided kernel (zero_diag!),
#     which is exact because the unmasked RHS is finite on the diagonal.
#   • No transpose scratch. permutedims! writes the SzSp transpose directly into
#     SzSp_T_loc (distinct src/dst).
#   • No separate AllToAll recv buffer. The recv lands in du[L_SzSp_x], which is
#     free scratch during communication (see communicate! / f!), so communicate!
#     takes a `recv` argument.
# ============================================================

struct GPUData
    gpu_id     :: Int          # 0-based
    offset     :: Int          # global bin offset (0-based)
    # Parameters on this GPU
    delta_b_local :: CuVector{Float64}   # M_LOCAL
    Nj_local      :: CuVector{Float64}   # M_LOCAL
    g_b_local     :: CuVector{Float64}   # M_LOCAL
    delta_b_full  :: CuVector{Float64}   # M  (replicated parameter)
    g_b_full      :: CuVector{Float64}   # M  (replicated parameter)
    # Communication buffers — AllGather for 5 spin vectors
    Sp_full    :: CuVector{ComplexF64}    # M (recv AllGather)
    Sz_full    :: CuVector{ComplexF64}    # M
    adSp_full  :: CuVector{ComplexF64}    # M
    adSm_full  :: CuVector{ComplexF64}    # M
    adSz_full  :: CuVector{ComplexF64}    # M
    # Communication buffer — AllToAll for SzSp_cross transpose (the only cross buffer)
    SzSp_T_loc :: CuMatrix{ComplexF64}    # M_LOCAL × M (pre-assembled transpose block)
end

function build_gpu_data(gpu_id::Int)
    CUDA.device!(gpu_id)
    offset = gpu_id * M_LOCAL

    # Local parameter slices
    R = (offset+1):(offset+M_LOCAL)
    delta_b_local = CuArray(Float64.(delta_b[R]))
    Nj_local      = CuArray(Float64.(Nj[R]))
    g_b_local     = CuArray(Float64.(g_b[R]))
    delta_b_full  = CuArray(Float64.(delta_b))
    g_b_full      = CuArray(Float64.(g_b))

    # Communication buffers — spin vector AllGather
    Sp_full   = CUDA.zeros(ComplexF64, M)
    Sz_full   = CUDA.zeros(ComplexF64, M)
    adSp_full = CUDA.zeros(ComplexF64, M)
    adSm_full = CUDA.zeros(ComplexF64, M)
    adSz_full = CUDA.zeros(ComplexF64, M)

    # SzSp_cross transpose target — the only persistent full M_LOCAL×M cross buffer.
    SzSp_T_loc = CUDA.zeros(ComplexF64, M_LOCAL, M)

    return GPUData(
        gpu_id, offset,
        delta_b_local, Nj_local, g_b_local,
        delta_b_full, g_b_full,
        Sp_full, Sz_full, adSp_full, adSm_full, adSz_full,
        SzSp_T_loc
    )
end


# ============================================================
# 7) INITIAL CONDITIONS — build the per-GPU initial state shard
#    Returns the per-GPU initial state as a CuVector shard.
# ============================================================

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

# ============================================================
# 8) NCCL COMMUNICATION  (single-threaded on the main task)
# ============================================================
#
# NOTE: NCCL group start/end is thread-local. ALL grouped Send/Recv and
# Allgather calls below MUST be issued from one host thread, so this
# function is intentionally NOT parallelized with Threads.@spawn.
# The on-device work it triggers is asynchronous and overlaps across GPUs
# anyway; the function synchronizes every device before returning so the
# subsequent (possibly multi-threaded) local-compute phase is safe.

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
    BLK = M_LOCAL * M_LOCAL    # ComplexF64 elements exchanged per peer

    # `recv` is a Sharded vector whose [L_SzSp_x] slab stages the AllToAll recv.
    # In the hot loop recv === du, which is free scratch here (see f!).
    # AllGather targets stay in the persistent gd.*_full buffers.

    # --- Single NCCL group: AllGather (5 spin vectors) + SzSp_cross AllToAll.
    #     Both collectives share ONE group (one barrier, and NCCL can pipeline
    #     them); cross Send/Recv blocks are issued straight from the contiguous
    #     state slab src[L_SzSp_x], with no staging copy. Peer s's block is the
    #     flat slice s*BLK+1:(s+1)*BLK of the column-major M_LOCAL×M slab. ---
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

            # SzSp_cross transpose: send each peer's block directly from the
            # state slab; receive into recv's SzSp_x slab (= du[L_SzSp_x]).
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

    # Self-blocks: pure local copy from the state slab (no fabric). Issued on
    # the main task after the group, so per device they are stream-ordered
    # after the cross Recv that writes the other blocks of recv[L_SzSp_x].
    for r in 1:N_GPU
        CUDA.device!(r-1)
        src = u.shards[r]::CuVector{ComplexF64}
        rcv = recv.shards[r]::CuVector{ComplexF64}
        self = r - 1
        selfloc = L_SzSp_x[self*BLK + 1 : (self+1)*BLK]
        copyto!(@view(rcv[selfloc]), @view(src[selfloc]))
    end

    # Reassemble SzSp_T_loc directly from the recv slab: permutedims! transposes
    # the per-peer M_LOCAL×M_LOCAL blocks from recv[L_SzSp_x] (src) into
    # SzSp_T_loc (dst). Distinct arrays => well-defined permute with no scratch.
    for r in 1:N_GPU
        CUDA.device!(r-1)
        gd = gpus[r]
        src_3d = reshape(@view(recv.shards[r][L_SzSp_x]), M_LOCAL, M_LOCAL, N_GPU)
        dst_3d = reshape(gd.SzSp_T_loc,                   M_LOCAL, M_LOCAL, N_GPU)
        permutedims!(dst_3d, src_3d, (2, 1, 3))
    end

    # Sync all GPUs so the subsequent local-compute phase is safe.
    for r in 1:N_GPU
        CUDA.device!(r-1)
        CUDA.synchronize()
    end
end
# ============================================================
# 9) LOCAL RHS
# ============================================================

# Zero a cross slab's excluded diagonal in place (avoids storing a full
# M_LOCAL×M 0/1 mask). The diagonal entries are (j_loc, offset+j_loc) for
# j_loc = 1..M_LOCAL; in the column-major M_LOCAL×M slab their within-slab flat
# indices form an arithmetic sequence of M_LOCAL terms, stride M_LOCAL+1,
# starting at offset*M_LOCAL+1. Adding (Lstart-1) gives absolute indices into
# du. Zeroing equals masking exactly, since the unmasked RHS is finite on the
# diagonal (0×EXPR == 0).
@inline function zero_diag!(du::CuVector{ComplexF64}, Lstart::Int, offset::Int)
    base = Lstart + offset * M_LOCAL          # absolute index of (1, offset+1)
    last = base + (M_LOCAL - 1) * (M_LOCAL + 1)
    @view(du[base:(M_LOCAL + 1):last]) .= zero(ComplexF64)
    return nothing
end

function local_rhs!(du_dst::CuVector{ComplexF64},
                    u_src::CuVector{ComplexF64},
                    gd::GPUData,
                    t::Float64)

    # --- unpack local views ---
    # Batched D2H read of the 3 cavity scalars (one transfer, no scalar indexing).
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

    # --- unpack derivative views ---
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

    # --- full gathered vectors (from AllGather) ---
    Sp_full   = gd.Sp_full
    Sz_full   = gd.Sz_full
    adSp_full = gd.adSp_full
    adSm_full = gd.adSm_full
    adSz_full = gd.adSz_full

    # SzSp_cross transpose block — pre-assembled by AllToAll in communicate!
    SzSp_T_loc = gd.SzSp_T_loc   # M_LOCAL × M

    # --- physics ---
    κe_t = kappa_e_of_t(t)
    κt_t = κe_t
    g_t_loc  = gd.g_b_local .* g_gate_of_t(t)     # M_LOCAL
    g_t_full = gd.g_b_full  .* g_gate_of_t(t)      # M (for sums)
    E_t  = E_of_t_num(t)
    δ_loc = gd.delta_b_local   # M_LOCAL

    # ----------------------------
    # Cavity equations (scalars).
    # Computed as host scalars from on-device reductions, then written in ONE
    # batched H2D copy (no per-scalar scalar-indexed writes).
    # ----------------------------
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

    # ----------------------------
    # First-order spin (local bins)
    # ----------------------------
    dSp_loc .= 1im .* δ_loc .* Sp_loc .- 2im .* g_t_loc .* adSz_loc
    dSz_loc .= .-1im .* g_t_loc .* conj.(adSm_loc) .+ 1im .* g_t_loc .* adSm_loc

    # ----------------------------
    # Cavity-spin correlations (local bins)
    # ----------------------------
    sumgSpSp_jk = SpSp_s_loc .* g_t_loc .+ SpSp_x_loc * g_t_full
    sumgSmSp_jk = SmSp_s_loc .* g_t_loc .+ SmSp_x_loc * g_t_full
    sumgSzSp_jk = SzSp_s_loc .* g_t_loc .+ SzSp_x_loc * g_t_full

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

    # ----------------------------
    # Same-bin second-order (local bins)
    # ----------------------------
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

    # ----------------------------
    # Cross-bin second-order (row slab: M_LOCAL × M)
    # ----------------------------
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

    # Each cross assignment is written UNMASKED, then its excluded diagonal is
    # zeroed in place with a tiny M_LOCAL-element strided kernel (zero_diag!),
    # avoiding a full M_LOCAL×M diagonal mask.

    # --- SpSp_cross ---
    dSpSp_x_loc .= (
        1im .* (Δ_col .+ Δ_row) .* SpSp_x_loc
        .- 2im .* G_col .* (Sp_row .* adSz_col .+ conj(a_val) .* SzSp_x_loc .+ adSp_row .* Sz_col .- 2 .* Sp_row .* conj(a_val) .* Sz_col)
        .- 2im .* G_row .* (Sp_col .* adSz_row .+ conj(a_val) .* SzSp_T_loc .+ Sz_row .* adSp_col .- 2 .* Sp_col .* conj(a_val) .* Sz_row)
    )
    zero_diag!(du_dst, first(L_SpSp_x), gd.offset)

    # --- SzSp_cross ---
    # NOTE: du_dst[L_SzSp_x] currently holds the AllToAll recv data, already
    # consumed to build SzSp_T_loc in communicate!. The source SzSp_x_loc reads
    # from u_src, so overwriting du_dst[L_SzSp_x] here is safe and nothing later
    # reads it.
    dSzSp_x_loc .= (
        1im .* Δ_row .* SzSp_x_loc
        .- 1im .* G_col .* (Sp_row .* conj.(adSm_col) .+ Sp_col .* conj.(adSm_row) .+ a_val .* SpSp_x_loc .- 2 .* Sp_row .* Sp_col .* a_val)
        .+ 1im .* G_col .* (Sp_row .* adSm_col .+ conj(a_val) .* SmSp_x_loc .+ adSp_row .* conj.(Sp_col) .- 2 .* Sp_row .* conj(a_val) .* conj.(Sp_col))
        .- 2im .* G_row .* (SzSz_x_loc .* conj(a_val) .+ Sz_row .* adSz_col .+ Sz_col .* adSz_row .- 2 .* conj(a_val) .* Sz_row .* Sz_col)
    )
    zero_diag!(du_dst, first(L_SzSp_x), gd.offset)

    # --- SmSp_cross ---
    # conj.(SzSp_T_loc) is applied INLINE inside the fused broadcasts below
    # instead of being materialised into a separate M_LOCAL×M temporary.
    dSmSp_x_loc .= (
        .- 1im .* Δ_col .* SmSp_x_loc .+ 1im .* Δ_row .* SmSp_x_loc
        .+ 2im .* G_col .* (conj.(adSz_col) .* Sp_row .+ conj.(adSm_row) .* Sz_col .+ a_val .* SzSp_x_loc .- 2 .* Sp_row .* a_val .* Sz_col)
        .- 2im .* G_row .* (conj(a_val) .* conj.(SzSp_T_loc) .+ Sz_row .* adSm_col .+ conj.(Sp_col) .* adSz_row .- 2 .* conj(a_val) .* Sz_row .* conj.(Sp_col))
    )
    zero_diag!(du_dst, first(L_SmSp_x), gd.offset)

    # --- SzSz_cross ---
    dSzSz_x_loc .= (
        .- 1im .* G_col .* (Sp_col .* conj.(adSz_row) .+ conj.(adSm_col) .* Sz_row .+ a_val .* SzSp_T_loc .- 2 .* Sp_col .* Sz_row .* a_val)
        .+ 1im .* G_col .* (conj(a_val) .* conj.(SzSp_T_loc) .+ Sz_row .* adSm_col .+ conj.(Sp_col) .* adSz_row .- 2 .* conj(a_val) .* Sz_row .* conj.(Sp_col))
        .- 1im .* G_row .* (conj.(adSz_col) .* Sp_row .+ conj.(adSm_row) .* Sz_col .+ a_val .* SzSp_x_loc .- 2 .* Sp_row .* a_val .* Sz_col)
        .+ 1im .* G_row .* (conj.(SzSp_x_loc) .* conj(a_val) .+ adSm_row .* Sz_col .+ adSz_col .* conj.(Sp_row) .- 2 .* conj(a_val) .* Sz_col .* conj.(Sp_row))
    )
    zero_diag!(du_dst, first(L_SzSz_x), gd.offset)

    return nothing
end

# ============================================================
# 10) SHARDED STATE TYPE + RHS  (drives OrdinaryDiffEq across GPUs)
#
# `Sharded` holds one CuVector{ComplexF64} per GPU (shard r on device r-1).
# The custom broadcast applies the stepper's axpy updates per shard on the
# correct device; `sharded_norm` reduces the error estimate across shards.
# ============================================================

struct Sharded{T,A<:CuVector{T}} <: AbstractVector{T}
    shards::Vector{A}               # shards[i] resides on CUDA device (i-1)
end
# Infer T (and the concrete CuArray memory type A) from the shard vector, so
# `Sharded([cuvec0, cuvec1, ...])` works for CuArray{T,1,CUDA.DeviceMemory}.
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

# NaN/Inf instability check — per-shard device-side reduction (NOT scalar iter).
Base.any(f::Function, s::Sharded) =
    any(i -> (CUDA.device!(i-1); any(f, s.shards[i])), 1:nshards(s))

# Rare slow-path scalar indexing (display / safety only — not the hot loop).
function Base.getindex(s::Sharded, idx::Integer)
    nl = length(s.shards[1]); g = (idx-1) ÷ nl + 1; loc = (idx-1) % nl + 1
    CUDA.device!(g-1); return CUDA.@allowscalar s.shards[g][loc]
end
function Base.setindex!(s::Sharded, v, idx::Integer)
    nl = length(s.shards[1]); g = (idx-1) ÷ nl + 1; loc = (idx-1) % nl + 1
    CUDA.device!(g-1); CUDA.@allowscalar s.shards[g][loc] = v; v
end

# --- Custom broadcast: apply the expression to every shard on its device ---
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

# --- Cross-GPU RMS norm for adaptive error control (passed as internalnorm) ---
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

# --- RHS: NCCL communicate! (across u's shards) then per-GPU local_rhs! ---
struct RHSParams
    gpus::Vector{GPUData}
end

function f!(du::Sharded, u::Sharded, p::RHSParams, t)
    gpus = p.gpus
    # Pass `du` to communicate! as the AllToAll recv scratch. du[L_SzSp_x] is
    # free here: local_rhs! writes it only AFTER assembling SzSp_T_loc from it,
    # and builds dSzSp_x from u + SzSp_T_loc, never du. du and u are distinct
    # registers (in-place OrdinaryDiffEq contract), so send (reads u) and recv
    # (writes du) cannot alias.
    communicate!(gpus, u, du)                  # fills Sp_full.. + SzSp_T_loc
    # NOTE: sequential issue on the main task — NO Threads.@spawn. Spawning a task
    # per shard each RHS call created a fresh task-local CUDA stream every time;
    # over thousands of steps these accumulated until cuStreamCreate failed
    # ("out of memory"). Issuing all devices' kernels here async then syncing
    # still overlaps the GPUs, with one persistent stream per device and no churn.
    for r in 1:N_GPU
        CUDA.device!(r-1)
        local_rhs!(du.shards[r], u.shards[r], gpus[r], t)
    end
    for r in 1:N_GPU
        CUDA.device!(r-1); CUDA.synchronize()
    end
    return nothing
end


# ============================================================
# 11) BUILD GPUs AND INITIAL CONDITIONS
# ============================================================

println("Allocating GPU parameter/comm buffers...")
gpu_data = [build_gpu_data(i) for i in 0:N_GPU-1]

println("Building initial state shards...")
u0 = Sharded([initial_shard(gpu_data[r]) for r in 1:N_GPU])


# ============================================================
# 12) SAVE CALLBACK  (FunctionCallingCallback at t_save points)
#     Writes the small per-bin slices into preallocated host arrays; never
#     stores the full Sharded state.
# ============================================================

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

# Live-progress milestone: next integer percent (0,5,10,…) to report. A Ref so
# save_func can mutate it across calls. Milestone test (not idx % N) so it still
# fires correctly under the adaptive stepper's irregular save cadence.
const progress_next = Ref(0)

function save_func(u::Sharded, t, integrator)
    # exact index: tstops are placed at t_save, so t matches one of them.
    idx = argmin(abs.(t_save .- t))

    # Cavity scalars from GPU 0 (replicated): batched 3-element read.
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

    # --- live progress: print + flush at each 5% milestone ---
    # PBS log stdout is block-buffered (not a TTY), so without an explicit
    # flush nothing appears until the buffer fills or the job exits. Flushing
    # only on milestones (not every save) keeps the NFS log-file cost low.
    pct = floor(Int, 100 * idx / Nt)
    if pct >= progress_next[]
        elapsed = (time_ns() - t0_wall) / 1e9
        println("  [progress] $(progress_next[])%  (save $idx/$Nt, t=$(round(t*1e6, digits=1)) μs, ",
                "elapsed $(round(elapsed, digits=1)) s)")
        flush(stdout)
        # advance to the next unmet 5% mark (handles jumps >5% in one call)
        while progress_next[] <= pct
            progress_next[] += 5
        end
    end

    # Instantaneous footprint (CUDA.jl keeps no high-water mark). By idx==1 the
    # solver's full-state registers (alg_cache), the comm buffers, and NCCL's
    # warmup scratch all exist, so this captures the full PERSISTENT footprint —
    # the part that decides whether you fit. Per-step transients are only
    # O(M_LOCAL)/O(M) vectors (negligible vs the cross slabs), so idx==1 is within
    # a fraction of a percent of the true peak. Re-logging every 500 points
    # surfaces drift: a used figure climbing above the idx==1 value flags a leak
    # or pool growth.
    #   memory_info()   -> driver-level (free,total); decides if the next
    #                      cudaMalloc (incl. NCCL's) succeeds.
    #   pool_status()   -> CUDA.jl pool view (effective usage + reserved); shows
    #                      reserved-but-free blocks the driver still counts as used.
    if idx == 1 || idx % 500 == 0
        for r in 1:N_GPU
            CUDA.device!(r-1)
            free_b, total_b = CUDA.memory_info()
            used_b = total_b - free_b
            println("  [mem] GPU $(r-1): used $(round(used_b/2^30, digits=1)) / ",
                    "$(round(total_b/2^30, digits=1)) GiB  (free $(round(free_b/2^30, digits=1)) GiB)")
        end
        # Pool view (reserved vs in-use) for device 0 as a representative.
        CUDA.device!(0)
        print("  [mem] pool (GPU 0): "); CUDA.pool_status(); println()
        flush(stdout)
    end

    # Long-run hygiene: periodically run finalizers (which destroy any dead
    # CUDA streams) and hand fragmented free pool blocks back to the driver.
    # Cheap relative to the step cost; prevents slow pool/handle growth.
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


# ============================================================
# 13) SOLVE  (OrdinaryDiffEq low-storage adaptive stepper)
# ============================================================

# RDPK3SpFSAL510: 5th-order, adaptive, low-storage 3S*+ scheme (Ranocha-Dalcin-
# Parsani-Ketcheson; 10 stages, FSAL, embedded estimator). 3S*+ uses 3 working
# registers + 1 for the embedded estimate, and extra stages add none, so the
# state is replicated only ~4 times nominally regardless of order — the low
# footprint that lets large M fit in Float64. (Measured steady-state footprint
# is ~6 effective full-state registers; allocator/transient overhead runs above
# the nominal count, so size against the measured value.) 5th order resolves the
# fast pulse region cleanly; a lower-order scheme leaves it rough.
const INTEGRATOR = RDPK3SpFSAL510()
# Test-only alternatives (neither gives the low-storage footprint):
#   Tsit5()         — adaptive, same tol, ~7 registers; useful to validate the
#                     wiring before OrdinaryDiffEqLowStorageRK is installed.
#   RDPK3SpFSAL49() — 4th-order variant; leaves the pulse region rough.
# const INTEGRATOR = Tsit5()
# const INTEGRATOR = RDPK3SpFSAL49()

println("Starting solver: ", typeof(INTEGRATOR))

prob = ODEProblem(f!, u0, (0.0, Ttotal), RHSParams(gpu_data))


# Cap dt at the save spacing. The length-averaged error norm is blind to the few
# fast cavity/spin components, so without a cap the adaptive stepper coarsens
# through the pulse and aliases the fast WURST/chirp carrier (period ≈0.4 µs),
# producing negative variance. Capping dtmax at the save spacing (≈0.03 µs),
# together with a tstop at every save point (see tstops below), pins the stepper
# to the grid and keeps the carrier resolved.
dtmax_resolve = Ttotal / USER.Nt_save     # ≈ 3e-8 s = 0.03 µs

# Seed dt so OrdinaryDiffEq's initial-step estimator (_ode_initdt_iip) never
# runs. It allocates three full-state Sharded temporaries (a zero(u) and a
# broadcast u/u) just to guess dt0; left to run, these linger in the pool and
# inflate the footprint from ~6 to ~9 effective full-state registers. Seeding
# has zero accuracy cost — adaptivity takes over after the first step.
dt_init = (Ttotal / USER.Nt_save) * 0.01

# Pre-allocate NCCL's internal scratch (channel + send/recv staging) BEFORE
# solve() allocates its full-state registers. NCCL grabs that scratch lazily
# from the CUDA driver on the first grouped collective (inside the first
# f!/communicate!). At large M that allocation can fail ("NCCL WARN Cuda failure
# 2 'out of memory'" in ncclGroupEnd) if the integrator registers are already in
# place and the free pool is too small or fragmented for NCCL's cudaMalloc.
#   (1) GC.gc() + CUDA.reclaim() return CUDA.jl's cached-free pool to the driver,
#       so NCCL's driver-level allocation can use it.
#   (2) One warmup communicate!(gpu_data, u0, warm) forces NCCL to allocate its
#       workspace now, while free memory is at its max (no registers yet) and
#       least fragmented. NCCL reuses it every step; the comm buffers it fills
#       are overwritten by the first real RHS, so the result is unaffected.
GC.gc()
for r in 1:N_GPU
    CUDA.device!(r-1); CUDA.reclaim()
end
# Warmup needs a recv slab (communicate! routes the AllToAll into `recv`). Don't
# pass u0 as its own recv — that would clobber its initial SzSp_x slab — so use a
# throwaway scratch and free it immediately, returning the memory BEFORE solve()
# allocates the registers. With JULIA_CUDA_MEMORY_POOL=none the reclaim is real,
# so this scratch doesn't count against the steady-state footprint.
warm = similar(u0)
communicate!(gpu_data, u0, warm)  # NCCL workspace warmup (filled buffers discarded)
warm = nothing
GC.gc()
for r in 1:N_GPU
    CUDA.device!(r-1); CUDA.reclaim()
end

# Live confirmation that setup finished and solve() is about to run. NCCL's
# lazy workspace allocation happens just after this point, so flushing here
# ensures the log records that setup completed even if that allocation fails.
println("NCCL warmup done; entering solve() ...")
flush(stdout)

sol = solve(prob, INTEGRATOR;
            dt = dt_init,                        # seed dt → skip _ode_initdt_iip's transient registers
            abstol = USER.abstol,
            reltol = USER.reltol,
            internalnorm = sharded_norm,        # cross-GPU RMS error norm
            callback = save_cb,                  # saving handled by callback
            tstops = t_save,                     # stop at every save point (with dtmax, pins dt to the grid)
            save_everystep = false,              # do NOT retain Sharded states
            save_start = false,
            save_end = false,
            # Alias u0 as the integrator's working `u` (no defensive copy),
            # saving one full state replication (the four M_LOCAL×M cross slabs
            # dominate it). SAFE: u0 is never read after solve() (post-processing
            # uses the callback's a_save/Sp_save arrays) and solve is never
            # re-run from u0. But u0 IS mutated to the final state — rebuild it
            # each iteration if you add a parameter sweep. MUST use the
            # ODEAliasSpecifier object; the bare `alias_u0 = true` keyword is
            # silently ignored by this SciMLBase (confirmed via cache.u === u0),
            # losing the saving.
            alias = OrdinaryDiffEq.SciMLBase.ODEAliasSpecifier(alias_u0 = true),
            dtmax = dtmax_resolve,               # cap dt at the save spacing (see dtmax_resolve)
            maxiters = 10_000_000)

println("Solver returncode: ", sol.retcode)

elapsed_seconds = (time_ns() - t0_wall) / 1e9
println("Time taken: $elapsed_seconds seconds")

# ============================================================
# 14) POST-PROCESSING + SAVE
# ============================================================

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

# ============================================================
# 15) FIGURES  (saved to fig/ in addition to the .jld2 above)
#     Mirrors retrieve_2nd_order_pm.jl. Wrapped in try/catch so a
#     plotting/backend hiccup cannot discard the already-saved data.
# ============================================================
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

    # (1) total spin dynamics
    plt = plot(times, sx_avg, lw=2, label="⟨σx⟩ = Re⟨S+⟩/(N/2)")
    plot!(plt, times, sy_avg, lw=2, label="⟨σy⟩ = Im⟨S+⟩/(N/2)")
    plot!(plt, times, sz_avg, lw=2, label="⟨σz⟩")
    xlabel!(plt, "time (μs)"); ylabel!(plt, "mean value")
    title!(plt, "Spin dynamics (N_total=$(round(N_total_fig)), Mδ=$(M_delta), Mg=$(M_g))")
    savefig(plt, "fig/plot_spins_pm.png")

    # (2) input / output field
    plt = plot(times, E_in_array, lw=2, label="a_in_real(t)")
    plot!(plt, times, a_out_x, lw=2, label="a_out_real(t)")
    plot!(plt, times, a_out_p, lw=2, label="a_out_imag(t)")
    xlabel!(plt, "time (μs)"); ylabel!(plt, "Amplitude (s^{-1/2})")
    title!(plt, "Input / output field")
    savefig(plt, "fig/plot_io_pm.png")

    # (3) selected-bin Sz
    mid_idx = div(M, 2)
    keep_bins = unique([1, mid_idx, mid_idx + 1, M])
    plt = plot()
    for b in keep_bins
        plot!(plt, times, real.(Sz_save[b, :]) ./ (Nj[b] / 2), lw=2, label="⟨σz⟩(bin=$b)")
    end
    xlabel!(plt, "time (μs)"); ylabel!(plt, "mean value")
    title!(plt, "Selected-bin Sz dynamics (M_total=$M)")
    savefig(plt, "fig/plot_selected_bins_pm.png")

    # (4) selected-bin Re⟨S+⟩
    plt = plot()
    for b in keep_bins
        plot!(plt, times, real.(Sp_save[b, :]) ./ (Nj[b] / 2), lw=2, label="Re⟨S+⟩ bin=$b")
    end
    xlabel!(plt, "time (μs)"); ylabel!(plt, "mean value")
    title!(plt, "Selected-bin Re⟨S+⟩ dynamics")
    savefig(plt, "fig/plot_selected_bins_Sp_real_pm.png")

    # (5) photon number
    plt = plot(times, n_sol, lw=2, label="⟨a†a⟩")
    xlabel!(plt, "time (μs)"); ylabel!(plt, "photon number")
    title!(plt, "Photon Number (M_total=$M)")
    savefig(plt, "fig/photon_number_pm.png")

    # (6) quadrature variances
    Dx = 0.25 .* (adad_sol .+ conj.(adad_sol) .+ 2 .* n_sol .+ 1 .- (a_sol .+ conj.(a_sol)).^2)
    Dy = 0.25 .* (.-adad_sol .- conj.(adad_sol) .+ 2 .* n_sol .+ 1 .+ (a_sol .- conj.(a_sol)).^2)
    Dx_plot = real.(Dx); Dy_plot = real.(Dy); Dt = Dx_plot .+ Dy_plot

    plt = plot(times, Dx_plot, lw=2, label="Var(X)")
    plot!(plt, times, Dy_plot, lw=2, label="Var(Y)")
    xlabel!(plt, "time (μs)"); ylabel!(plt, "Quadrature Variance")
    title!(plt, "Quadrature Variance (M_total=$M)")
    savefig(plt, "fig/quadrature_variance_pm.png")

    # (7) total variance
    plt = plot(times, Dt, lw=2, label="Var(X)+Var(Y)")
    xlabel!(plt, "time (μs)"); ylabel!(plt, "Total Variance")
    title!(plt, "Total Variance (M_total=$M)")
    savefig(plt, "fig/total_variance_pm.png")

    println("Figures saved to fig/")
catch err
    @warn "Figure generation failed (the .jld2 data file is still saved)." exception=(err, catch_backtrace())
end
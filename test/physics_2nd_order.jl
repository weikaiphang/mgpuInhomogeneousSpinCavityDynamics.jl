using LinearAlgebra

const _PHYS_SRC = joinpath(@__DIR__, "..", "src")
for (pred, rel) in (
    (!isdefined(@__MODULE__, :state_length_2nd_order), "state_layout_2nd_order.jl"),
    (!isdefined(@__MODULE__, :small_length), "MGPUlayout.jl"),
    (!isdefined(@__MODULE__, :rhs_cpu!), "MGPUrhs_cpu.jl"),
    (!isdefined(@__MODULE__, :rhs_2nd_order!), "rhs_2nd_order.jl"),
    (!isdefined(@__MODULE__, :Tsit5Tableau), "MGPUtableaus.jl"),
    (!isdefined(@__MODULE__, :Solver2Workspace), "solver_2nd_workspace.jl"),
    (!isdefined(@__MODULE__, :build_u0_cpu_2nd_order), "initial_conditions_2nd_order.jl"),
    (!isdefined(@__MODULE__, :qrt_product_apply!), "qrt_jacobian.jl"),
    (!isdefined(@__MODULE__, :_with_default_ensemble_method), "simulation_api.jl"),
    (!isdefined(@__MODULE__, :build_constant_coupling_bins), "coupling_inhomogeneity.jl"),
    (!isdefined(@__MODULE__, :total_spin_number_from_cooperativity), "frequency_inhomogeneity.jl"),
    (!isdefined(@__MODULE__, :prepare_derived), "ensemble.jl"),
    (!isdefined(@__MODULE__, :_quad_frequency_nodes), "ensemble_quadrature.jl"),
)
    pred && include(joinpath(_PHYS_SRC, rel))
end

if !isdefined(@__MODULE__, :_WEAK_SEED)
    const _WEAK_SEED = 1.0e-3
end

# Authoritative product-state same-bin algebra (all four; no +Sp/2 on SzSp).
@inline function _spsp_product(Sp, Nj)
    invN = Nj > 0 ? (1 - 1 / Nj) : 0.0
    return Sp * Sp * invN
end
@inline function _szsp_product(Sz, Sp, Nj)
    invN = Nj > 0 ? (1 - 1 / Nj) : 0.0
    return Sz * Sp * invN
end
@inline function _smsp_product(Sp, Sz, Nj)
    invN = Nj > 0 ? (1 - 1 / Nj) : 0.0
    return abs2(Sp) * invN + Nj / 2 - Sz
end
@inline function _szsz_product(Sz, Nj)
    invN = Nj > 0 ? (1 - 1 / Nj) : 0.0
    return Sz * Sz * invN + Nj / 4
end

function _check_same_bin_algebra(st, Nj)
    @test st.SpSp_same ≈ _spsp_product.(st.Sp, Nj)
    @test st.SzSp_same ≈ _szsp_product.(st.Sz, st.Sp, Nj)
    @test real.(st.SmSp_same) ≈ real.(_smsp_product.(st.Sp, st.Sz, Nj))
    @test real.(st.SzSz_same) ≈ _szsz_product.(real.(st.Sz), Nj)
    if !all(iszero, st.Sp)
        old_szsp = st.Sz .* st.Sp .* (1 .- 1 ./ Nj) .+ st.Sp ./ 2
        @test st.SzSp_same ≉ old_szsp
    end
end

function _dirty_random_u(M; seed=2, dirty_diag=true)
    Random.seed!(seed)
    u = randn(ComplexF64, state_length_2nd_order(M))
    st = unpack_state_2nd_order_u(u, M)
    if dirty_diag
        for j in 1:M
            st.SpSp_cross[j, j] = 0.41 + 0.13im
            st.SmSp_cross[j, j] = 0.29 - 0.21im
            st.SzSp_cross[j, j] = -0.33 + 0.07im
            st.SzSz_cross[j, j] = 0.17
        end
    else
        for j in 1:M
            st.SpSp_cross[j, j] = 0
            st.SmSp_cross[j, j] = 0
            st.SzSp_cross[j, j] = 0
            st.SzSz_cross[j, j] = 0
        end
    end
    return u
end

function _eval_three_rhs(u, delta_b, g_b, Et)
    M = length(delta_b)
    mask = make_diag_mask_cpu(M)
    du_cpu = zero(u)
    du_mul = zero(u)
    du_ker = zero(u)
    rhs_cpu!(du_cpu, u, 0.1, 1.5, 0.25, delta_b, g_b, Et)
    ws = _rhs2_workspace(u, M)
    p = (0.1, 1.5, 0.25, delta_b, g_b, M, mask, _ -> Et, ws)
    _rhs_2nd_order_mulpath!(du_mul, u, p, 0.0)
    rhs_kernel_replica!(du_ker, u, 0.1, 1.5, 0.25, delta_b, g_b, Et)
    return du_cpu, du_mul, du_ker
end

function _assert_full_du_parity(du_a, du_b, M; atol=1e-12, label="")
    @test du_a ≈ du_b atol = atol
    sa = unpack_state_2nd_order_u(du_a, M)
    sb = unpack_state_2nd_order_u(du_b, M)
    for n in propertynames(sa)
        da = getfield(sa, n)
        db = getfield(sb, n)
        @test da ≈ db atol = atol
    end
    return nothing
end

function _check_cross_mean_products(st, Nj)
    M = length(Nj)
    for k in 1:M, j in 1:M
        if j == k
            @test st.SpSp_cross[j, k] == 0
            @test st.SzSp_cross[j, k] == 0
            @test st.SmSp_cross[j, k] == 0
            @test st.SzSz_cross[j, k] == 0
        else
            @test st.SpSp_cross[j, k] ≈ st.Sp[j] * st.Sp[k]
            @test st.SzSp_cross[j, k] ≈ st.Sz[j] * st.Sp[k]
            @test st.SmSp_cross[j, k] ≈ conj(st.Sp[j]) * st.Sp[k]
            @test st.SzSz_cross[j, k] ≈ st.Sz[j] * st.Sz[k]
        end
    end
end

@testset "C1/H1: 2nd-order product initial conditions" begin
    M = 3
    Nj = [2.0, 4.0, 6.0]

    u_g = build_u0_cpu_2nd_order(M, Nj, :ground)
    st_g = unpack_state_2nd_order_u(u_g, M)
    @test all(iszero, st_g.Sp)
    @test real.(st_g.Sz) ≈ .-Nj ./ 2
    @test real.(st_g.SmSp_same) ≈ Nj
    @test real.(st_g.SmSp_same) ≈ Nj ./ 2 .- real.(st_g.Sz)
    @test real.(st_g.SzSz_same) ≈ (Nj .^ 2) ./ 4
    @test all(iszero, st_g.SpSp_same)
    @test all(iszero, st_g.SzSp_same)
    @test all(iszero, st_g.a) && all(iszero, st_g.ad_a) && all(iszero, st_g.ad_ad)
    _check_same_bin_algebra(st_g, Nj)
    _check_cross_mean_products(st_g, Nj)

    u_i = build_u0_cpu_2nd_order(M, Nj, :inverted)
    st_i = unpack_state_2nd_order_u(u_i, M)
    @test real.(st_i.Sz) ≈ Nj ./ 2
    @test real.(st_i.SmSp_same) ≈ zeros(M) atol = 1e-14
    @test real.(st_i.SzSz_same) ≈ (Nj .^ 2) ./ 4
    @test all(iszero, st_i.SpSp_same)
    @test all(iszero, st_i.SzSp_same)
    _check_same_bin_algebra(st_i, Nj)
    _check_cross_mean_products(st_i, Nj)

    u_e = build_u0_cpu_2nd_order(M, Nj, :equator)
    st_e = unpack_state_2nd_order_u(u_e, M)
    @test real.(st_e.Sp) ≈ Nj ./ 2
    @test all(iszero, st_e.Sz)
    @test all(iszero, st_e.SzSp_same)  # Sz=0 ⇒ Sz*Sp*(1-1/Nj)=0; old +Sp/2 would be Nj/4
    @test real.(st_e.SpSp_same) ≈ real.(_spsp_product.(st_e.Sp, Nj))
    @test real.(st_e.SmSp_same) ≈ _smsp_product.(st_e.Sp, st_e.Sz, Nj)
    @test real.(st_e.SzSz_same) ≈ _szsz_product.(real.(st_e.Sz), Nj)
    @test maximum(abs.(real.(st_e.SmSp_same) .- abs2.(st_e.Sp))) > 0.1
    _check_same_bin_algebra(st_e, Nj)
    _check_cross_mean_products(st_e, Nj)

    @test _WEAK_SEED == 1.0e-3
    u_w = build_u0_cpu_2nd_order(M, Nj, :weak)
    st_w = unpack_state_2nd_order_u(u_w, M)
    _check_same_bin_algebra(st_w, Nj)
    _check_cross_mean_products(st_w, Nj)
    bloch2 = (Nj ./ 2) .^ 2
    obs2 = abs2.(st_w.Sp) .+ abs2.(real.(st_w.Sz))
    @test all(obs2 .> bloch2)  # seed sits outside |⟨S⟩|=Nj/2 by design

    u_wi = build_u0_cpu_2nd_order(M, Nj, :weak_inverted)
    st_wi = unpack_state_2nd_order_u(u_wi, M)
    _check_same_bin_algebra(st_wi, Nj)
    _check_cross_mean_products(st_wi, Nj)
    bloch2i = (Nj ./ 2) .^ 2
    obs2i = abs2.(st_wi.Sp) .+ abs2.(real.(st_wi.Sz))
    @test all(obs2i .> bloch2i)

end

@testset "C1: vacuum ⊗ ground is a 2nd-order fixed point" begin
    M = 4
    Nj = [1.0, 2.0, 3.5, 8.0]
    u = build_u0_cpu_2nd_order(M, Nj, :ground)
    du = zero(u)
    delta_b = [0.0, 1.2e6, -8.0e5, 3.0e4]
    g_b = [1.0e3, 1.1e3, 9.0e2, 1.05e3]
    p = (0.0, 2π * 1e6, 2π * 2e5, delta_b, g_b, M, nothing, t -> 0.0 + 0im)
    rhs_2nd_order!(du, u, p, 0.0)
    @test maximum(abs.(du)) < 1e-10

    u_bad = copy(u)
    _, _, _, _, _, _, _, _, _, _, SmSp, _ = unpack_state_2nd_order_u(u_bad, M)
    SmSp .= 0
    du_bad = zero(u_bad)
    rhs_2nd_order!(du_bad, u_bad, p, 0.0)
    @test maximum(abs.(du_bad)) > 1e-6
end

@testset "H5: monolith rhs_2nd_order! ↔ rhs_cpu! parity" begin
    M = 3
    Nj = [2.0, 3.0, 5.0]
    u = build_u0_cpu_2nd_order(M, Nj, :weak)
    u[1] = 0.02 + 0.01im
    u[2] = 0.001
    u[3] = 0.03
    Random.seed!(1)
    u[4:end] .+= 1e-4 .* randn.(ComplexF64)
    delta_b = Float64[0.4, -0.2, 0.1]
    g_b = Float64[1.1, 0.9, 1.0]
    Et = 0.3 + 0.2im
    du_a = zero(u)
    du_b = zero(u)
    rhs_cpu!(du_a, u, 0.1, 1.5, 0.25, delta_b, g_b, Et)
    rhs_2nd_order!(du_b, u, (0.1, 1.5, 0.25, delta_b, g_b, M, nothing, _ -> Et), 0.0)
    @test du_a ≈ du_b atol = 1e-14
end

@testset "dirty-diag rowsum: monolith mulpath ↔ rhs_cpu!" begin
    # Product ICs zero cross[j,j], so the old unmasked mul! never showed the leak.
    M = 3
    Random.seed!(2)
    u = randn(ComplexF64, state_length_2nd_order(M))
    st = unpack_state_2nd_order_u(u, M)
    for j in 1:M
        st.SpSp_cross[j, j] = 0.41 + 0.13im
        st.SmSp_cross[j, j] = 0.29 - 0.21im
        st.SzSp_cross[j, j] = -0.33 + 0.07im
        st.SzSz_cross[j, j] = 0.17
    end
    delta_b = Float64[0.4, -0.2, 0.1]
    g_b = Float64[1.1, 0.9, 1.0]
    Et = 0.3 + 0.2im
    mask = make_diag_mask_cpu(M)

    leak = st.SpSp_same .* g_b .+ st.SpSp_cross * g_b
    masked = st.SpSp_same .* g_b .+ (st.SpSp_cross .* mask) * g_b
    @test leak ≉ masked
    @test maximum(abs.(leak .- masked)) > 1e-6

    cpu_sum = zeros(ComplexF64, M)
    for j in 1:M
        s = st.SpSp_same[j] * g_b[j]
        for k in 1:M
            k == j && continue
            s += st.SpSp_cross[j, k] * g_b[k]
        end
        cpu_sum[j] = s
    end
    @test masked ≈ cpu_sum
    ws_sum = zeros(ComplexF64, M)
    _rowsum_same_plus_cross!(ws_sum, st.SpSp_same, st.SpSp_cross, g_b, mask)
    @test ws_sum ≈ cpu_sum

    du_cpu, du_mul, du_ker = _eval_three_rhs(u, delta_b, g_b, Et)
    sc = unpack_state_2nd_order_u(du_cpu, M)
    _assert_full_du_parity(du_cpu, du_mul, M)
    _assert_full_du_parity(du_cpu, du_ker, M)

    # Unmasked mul! (the pre-fix formula) must disagree on dad* — this is
    # the test that 240/240 product-IC parity could not catch.
    leak_sm = st.SmSp_same .* g_b .+ st.SmSp_cross * g_b
    leak_sz = st.SzSp_same .* g_b .+ st.SzSp_cross * g_b
    @test sc.adSp ≉ sc.adSp .+ 1im .* (leak .- masked)
    @test sc.adSm ≉ sc.adSm .+ 1im .* (leak_sm .- (st.SmSp_same .* g_b .+ (st.SmSp_cross .* mask) * g_b))
    @test sc.adSz ≉ sc.adSz .+ 1im .* (leak_sz .- (st.SzSp_same .* g_b .+ (st.SzSp_cross .* mask) * g_b))

    gpu_ok = false
    if isdefined(@__MODULE__, :CUDA)
        gpu_ok = try
            CUDA.functional()
        catch
            false
        end
    end
    if gpu_ok
        u_gpu = CuArray(u)
        du_gpu = similar(u_gpu)
        mask_gpu = CuArray(mask)
        δ_gpu = CuArray(delta_b)
        g_gpu = CuArray(g_b)
        ws_gpu = _rhs2_workspace(u_gpu, M)
        p_gpu = (0.1, 1.5, 0.25, δ_gpu, g_gpu, M, mask_gpu, _ -> Et, ws_gpu)
        rhs_2nd_order!(du_gpu, u_gpu, p_gpu, 0.0)
        _assert_full_du_parity(du_cpu, Array(du_gpu), M)
    else
        @info "dirty-diag CuArray / multi-GPU integrate skipped: no NVIDIA GPU on this host"
    end
end

@testset "full-du parity: mulpath ↔ rhs_cpu! ↔ kernel replica" begin
    # Hard-fail: every named field, not only dad* / small block.
    delta_b = Float64[0.4, -0.2, 0.1]
    g_b = Float64[1.1, 0.9, 1.0]
    Et = 0.3 + 0.2im
    M = 3
    Nj = [2.0, 3.0, 5.0]

    u_dirty = _dirty_random_u(M; seed=2, dirty_diag=true)
    du_cpu, du_mul, du_ker = _eval_three_rhs(u_dirty, delta_b, g_b, Et)
    _assert_full_du_parity(du_cpu, du_mul, M)
    _assert_full_du_parity(du_cpu, du_ker, M)

    u_clean = _dirty_random_u(M; seed=3, dirty_diag=false)
    du_cpu, du_mul, du_ker = _eval_three_rhs(u_clean, delta_b, g_b, Et)
    _assert_full_du_parity(du_cpu, du_mul, M)
    _assert_full_du_parity(du_cpu, du_ker, M)

    for ic in (:ground, :inverted, :equator, :weak, :weak_inverted)
        u = build_u0_cpu_2nd_order(M, Nj, ic)
        du_cpu, du_mul, du_ker = _eval_three_rhs(u, delta_b, g_b, Et)
        _assert_full_du_parity(du_cpu, du_mul, M)
        _assert_full_du_parity(du_cpu, du_ker, M)
    end
end

@testset "H4: :constant coupling rejects M_g ≠ 1" begin
    gcfg = (kind = :constant, g_value = 2π * 100)
    @test_throws ErrorException build_constant_coupling_bins(gcfg, 4)
    edges, g, p, gm, gs, g2, info = build_constant_coupling_bins(gcfg, 1)
    @test length(g) == 1
    @test p == [1.0]
    @test g2 ≈ abs2(gcfg.g_value)
end

@testset "H5: quadrature Lorentzian mass ∑p_δ ≈ 2 atan(span)/π" begin
    span = 2.5
    freq = (kind = :lorentzian, FWHM = 2π * 1e6, span_gamma = span, renormalize = false)
    δ, p = _quad_frequency_nodes(freq, 17)
    @test sum(p) ≈ 2 * atan(span) / π rtol = 1e-12
    maybe_renormalize_frequency_probs!(copy(p), freq)
    @test sum(p) < 1
end

@testset "C_eff honesty: truncated Lorentzian is not silent" begin
    C = 0.6
    span = 2.5
    freq = (kind = :lorentzian, FWHM = 2π * 1e6, span_gamma = span, renormalize = false)
    pδ = frequency_truncation_mass(freq)
    @test pδ ≈ 2 * atan(span) / π rtol = 1e-14
    @test 0.70 < pδ < 0.80
    C_eff = effective_cooperativity(C, freq)
    @test C_eff ≈ C * pδ
    @test C_eff != C
    freq_r = merge(freq, (renormalize = true,))
    @test frequency_truncation_mass(freq_r) == 1.0
    @test effective_cooperativity(C, freq_r) == C

    g = (kind = :constant, g_value = 2π * 100)
    cfg = (
        C_ens = C,
        M_delta = 17,
        M_g = 1,
        kappa_e = 2π * 1e6,
        kappa_i = 0.0,
        delta0 = 0.0,
        Ttotal = 1e-6,
        Nt_save = 3,
        freq_inhomogeneity = freq,
        g_inhomogeneity = g,
        ensemble_method = :quadrature,
    )
    d = prepare_derived(cfg)
    @test hasproperty(d, :C_eff) && hasproperty(d, :p_delta_sum)
    @test d.C_ens == C
    @test d.p_delta_sum ≈ pδ rtol = 1e-12
    @test d.C_eff ≈ C * d.p_delta_sum * d.p_g_sum
    @test d.C_eff != d.C_ens
    @test d.N == total_spin_number_from_cooperativity(C, d.kappa_t, d.g2_avg, freq)
    h = cooperativity_honesty(cfg)
    @test h.C_eff ≈ d.C_eff rtol = 1e-12
end

@testset "paper/demos print C_eff when freq truncation is un-renormalized" begin
    roots = (
        joinpath(@__DIR__, "..", "paper"),
        joinpath(@__DIR__, "..", "examples"),
        joinpath(@__DIR__, "..", "scripts"),
    )
    offenders = String[]
    repo = joinpath(@__DIR__, "..")
    for root in roots
        isdir(root) || continue
        for (dir, _, files) in walkdir(root)
            for f in files
                endswith(f, ".jl") || continue
                path = joinpath(dir, f)
                txt = read(path, String)
                occursin(r"renormalize\s*=\s*false", txt) || continue
                (occursin("span_gamma", txt) || occursin("span_sigma", txt)) || continue
                if !occursin("C_eff", txt) && !occursin("print_cooperativity_honesty", txt)
                    push!(offenders, relpath(path, repo))
                end
            end
        end
    end
    @test isempty(offenders)
end

@testset "order-2 ensemble_method defaults to :auto" begin
    sim = (simulation_order = :second_order, Ttotal = 1e-6)
    out = _with_default_ensemble_method(sim, :second_order)
    @test out.ensemble_method === :auto
    sim2 = merge(sim, (ensemble_method = :histogram,))
    @test _with_default_ensemble_method(sim2, :second_order).ensemble_method === :histogram
end

@testset "N from cooperativity: Lorentzian / peak-matched Gaussian" begin
    C, κ, FWHM, g2 = 0.6, 2π * 1e6, 2π * 1e6, abs2(2π * 100)
    freq_l = (kind = :lorentzian, FWHM = FWHM, span_gamma = 2.5)
    freq_g = (kind = :gaussian, FWHM = FWHM, span_sigma = 3.0)
    @test total_spin_number_from_cooperativity(C, κ, g2, freq_l) ≈ C * κ * FWHM / (4 * g2)
    @test total_spin_number_from_cooperativity(C, κ, g2, freq_g) ≈
          C * κ * FWHM / (4 * sqrt(π * log(2)) * g2)
end

@testset "H3: κt = κe + κi in CPU / monolith 2nd-order RHS" begin
    M = 1
    Nj = [4.0]
    u = build_u0_cpu_2nd_order(M, Nj, :inverted)
    u[1] = 0.1
    du_e = zero(u)
    du_t = zero(u)
    g_b = [1.0]
    δ = [0.0]
    rhs_cpu!(du_e, u, 0.0, 2.0, 0.0, δ, g_b, 0.0im)
    rhs_cpu!(du_t, u, 0.0, 2.0, 3.0, δ, g_b, 0.0im)
    @test real(du_t[1] - du_e[1]) ≈ -0.5 * 3.0 * real(u[1]) atol = 1e-14
end

@testset "rhs_cpu! serial ↔ threaded; full-du vs mulpath; no-alloc" begin
    M = 20  # below auto-thread cutoff; force both modes
    @test _RHS_CPU_THREAD_MIN_M == 256
    u = _dirty_random_u(M; seed=4, dirty_diag=true)
    delta_b = Float64[0.1 * (j - 10) for j in 1:M]
    g_b = Float64[0.8 + 0.02 * j for j in 1:M]
    Et = 0.2 + 0.1im
    du_s = zero(u)
    du_t = zero(u)
    rhs_cpu!(du_s, u, 0.1, 1.5, 0.25, delta_b, g_b, Et; threaded=false)
    rhs_cpu!(du_t, u, 0.1, 1.5, 0.25, delta_b, g_b, Et; threaded=true)
    _assert_full_du_parity(du_s, du_t, M)

    du_mul = zero(u)
    du_ker = zero(u)
    mask = make_diag_mask_cpu(M)
    ws = _rhs2_workspace(u, M)
    p = (0.1, 1.5, 0.25, delta_b, g_b, M, mask, _ -> Et, ws)
    _rhs_2nd_order_mulpath!(du_mul, u, p, 0.0)
    rhs_kernel_replica!(du_ker, u, 0.1, 1.5, 0.25, delta_b, g_b, Et)
    _assert_full_du_parity(du_s, du_mul, M)
    _assert_full_du_parity(du_s, du_ker, M)

    rhs_cpu!(du_s, u, 0.1, 1.5, 0.25, delta_b, g_b, Et; threaded=false)
    alloc = @allocated rhs_cpu!(du_s, u, 0.1, 1.5, 0.25, delta_b, g_b, Et; threaded=false)
    @test alloc == 0

    p_prod = (0.1, 1.5, 0.25, delta_b, g_b, M, nothing, Returns(Et), nothing)
    rhs_2nd_order!(du_s, u, p_prod, 0.0)
    alloc2 = @allocated rhs_2nd_order!(du_s, u, p_prod, 0.0)
    @test alloc2 == 0
end

@testset "solver/integrator persistent workspace" begin
    M = 8
    Nt = 6
    Nj = fill(12.0, M)
    u0 = build_u0_cpu_2nd_order(M, Nj, :ground)
    ws = Solver2Workspace(Float64, M, Nt; stages = true)
    attach_u0!(ws, u0)
    @test length(ws.u) == state_length_2nd_order(M)
    @test length(ws.host) == 3 + 5M

    record_save2!(ws, ws.u, 1)
    record_save2!(ws, ws.u, 1)
    alloc_rec = @allocated record_save2!(ws, ws.u, 1)
    @test alloc_rec == 0
    @test ws.a[1] ≈ 0 atol = 1e-15
    @test ws.n[1] ≈ 0 atol = 1e-15
    @test real.(ws.Sz[:, 1]) ≈ -Nj ./ 2 atol = 1e-14

    delta_b = Float64[0.05 * (j - 4) for j in 1:M]
    g_b = fill(0.4, M)
    p = (0.0, 1.5, 0.25, delta_b, g_b, M, nothing, Returns(0.0im), ws.rhs)
    atol = 1e-8
    rtol = 1e-8
    _cpu_rhs!(ws, ws.k1, ws.u, p, 0.0)
    tsit5_cpu_step!(ws, p, 0.0, 1e-4, atol, rtol)
    _cpu_rhs!(ws, ws.k1, ws.u, p, 0.0)
    alloc_step = @allocated tsit5_cpu_step!(ws, p, 0.0, 1e-4, atol, rtol)
    @test alloc_step == 0

    attach_u0!(ws, u0)
    tsave = collect(range(0.0, 0.05; length = Nt))
    stats = solve_cpu_2nd!(ws, p, 0.0, 0.05, tsave; reltol = 1e-8, abstol = 1e-8)
    @test ws.saved[] == Nt
    @test stats.naccept >= Nt - 1
    @test maximum(abs, ws.a) < 1e-12
    @test maximum(abs, ws.n) < 1e-12
    for k in 1:Nt
        @test real.(ws.Sz[:, k]) ≈ -Nj ./ 2 atol = 1e-12
    end

    # Two workspaces, same IC / drive → bit-identical trajectory.
    ws_b = Solver2Workspace(Float64, M, Nt; stages = true)
    attach_u0!(ws_b, u0)
    solve_cpu_2nd!(ws_b, p, 0.0, 0.05, tsave; reltol = 1e-8, abstol = 1e-8)
    @test ws.a == ws_b.a
    @test ws.n == ws_b.n
    @test ws.Sz == ws_b.Sz
    @test ws.u == ws_b.u
end

@testset "CK45 persistent workspace; tol-aware vs Tsit5" begin
    M = 8
    Nt = 11
    Nj = fill(12.0, M)
    u0 = build_u0_cpu_2nd_order(M, Nj, :ground)
    tsave = collect(range(0.0, 0.05; length = Nt))
    delta_b = Float64[0.05 * (j - 4) for j in 1:M]
    g_b = fill(0.4, M)

    ws5 = Solver2Workspace(Float64, M, Nt; integrator = :ck45)
    @test ws5.integrator === :ck45
    @test ws5.tab isa CK45Tableau
    @test length(ws5.u) == state_length_2nd_order(M)
    @test length(ws5.k3) == length(ws5.u)
    @test isempty(ws5.k4) && isempty(ws5.k7)
    @test _cpu_stage_count(:ck45) == 5
    @test _cpu_stage_count(:tsit5) == 9

    p5 = (0.0, 1.5, 0.25, delta_b, g_b, M, nothing, Returns(0.0im), ws5.rhs)
    attach_u0!(ws5, u0)
    @test_throws ErrorException solve_cpu_2nd!(ws5, p5, 0.0, 0.05, tsave; integrator = :tsit5)

    atol = 1e-8
    rtol = 1e-8
    _cpu_rhs!(ws5, ws5.k2, ws5.u, p5, 0.0)
    ck45_cpu_step!(ws5, p5, 0.0, 1e-4, atol, rtol)
    _cpu_rhs!(ws5, ws5.k2, ws5.u, p5, 0.0)
    alloc_ck = @allocated ck45_cpu_step!(ws5, p5, 0.0, 1e-4, atol, rtol)
    @test alloc_ck == 0

    attach_u0!(ws5, u0)
    stats5 = solve_cpu_2nd!(ws5, p5, 0.0, 0.05, tsave; reltol = 1e-8, abstol = 1e-8)
    @test ws5.saved[] == Nt
    @test stats5.naccept >= Nt - 1
    @test maximum(abs, ws5.a) < 1e-12
    @test maximum(abs, ws5.n) < 1e-12
    for k in 1:Nt
        @test real.(ws5.Sz[:, k]) ≈ -Nj ./ 2 atol = 1e-12
    end

    ws5b = Solver2Workspace(Float64, M, Nt; integrator = :ck45)
    attach_u0!(ws5b, u0)
    p5b = (0.0, 1.5, 0.25, delta_b, g_b, M, nothing, Returns(0.0im), ws5b.rhs)
    solve_cpu_2nd!(ws5b, p5b, 0.0, 0.05, tsave; reltol = 1e-8, abstol = 1e-8)
    @test ws5.a == ws5b.a
    @test ws5.u == ws5b.u

    # Driven: same RHS, orders 5 vs 4 — tol-aware, not bit-identical.
    Et = 0.15 + 0.05im
    ws_t = Solver2Workspace(Float64, M, Nt; integrator = :tsit5)
    ws_c = Solver2Workspace(Float64, M, Nt; integrator = :ck45)
    attach_u0!(ws_t, u0)
    attach_u0!(ws_c, u0)
    pt = (0.1, 1.5, 0.25, delta_b, g_b, M, nothing, Returns(Et), ws_t.rhs)
    pc = (0.1, 1.5, 0.25, delta_b, g_b, M, nothing, Returns(Et), ws_c.rhs)
    solve_cpu_2nd!(ws_t, pt, 0.0, 0.05, tsave; reltol = 1e-8, abstol = 1e-8)
    solve_cpu_2nd!(ws_c, pc, 0.0, 0.05, tsave; reltol = 1e-8, abstol = 1e-8)
    @test ws_t.a ≈ ws_c.a atol = 1e-6 rtol = 1e-5
    @test ws_t.n ≈ ws_c.n atol = 1e-6 rtol = 1e-5
    @test ws_t.Sz ≈ ws_c.Sz atol = 1e-6 rtol = 1e-5
    @test ws_t.Sp ≈ ws_c.Sp atol = 1e-6 rtol = 1e-5

    # End-to-end warm solve: no per-step allocator traffic.
    attach_u0!(ws_c, u0)
    solve_cpu_2nd!(ws_c, pc, 0.0, 0.05, tsave; reltol = 1e-8, abstol = 1e-8)
    attach_u0!(ws_c, u0)
    alloc_e2e = @allocated solve_cpu_2nd!(ws_c, pc, 0.0, 0.05, tsave; reltol = 1e-8, abstol = 1e-8)
    @test alloc_e2e == 0
    attach_u0!(ws_t, u0)
    solve_cpu_2nd!(ws_t, pt, 0.0, 0.05, tsave; reltol = 1e-8, abstol = 1e-8)
    attach_u0!(ws_t, u0)
    alloc_e2e5 = @allocated solve_cpu_2nd!(ws_t, pt, 0.0, 0.05, tsave; reltol = 1e-8, abstol = 1e-8)
    @test alloc_e2e5 == 0
end

# ---------------------------------------------------------------------------
# QRT Jacobian (Niels): product = same-bin + cavity (O(M)); full-J is
# a small-M oracle only. Factorized 1st-order is not the product path.
# ---------------------------------------------------------------------------
@testset "QRT star RHS matches rhs_cpu! when cross = 0" begin
    Random.seed!(11)
    M = 3
    Nj = fill(8.0, M)
    delta_b = [-0.4, 0.1, 0.35]
    g_b = [0.12, 0.18, 0.09]
    Et = 0.2 + 0.05im
    u = build_u0_cpu_2nd_order(M, Nj, :equator)
    n = small_length(M)
    u[n+1:end] .= 0
    u[IDX_a] = 0.3 - 0.1im
    u[IDX_ad_ad] = 0.05 + 0.02im
    u[IDX_ad_a] = 0.4
    for j in 1:M
        u[small_offset(M, F_adSp) + j] = 0.02 * j + 0.01im
        u[small_offset(M, F_adSm) + j] = 0.015 * j - 0.02im
        u[small_offset(M, F_adSz) + j] = -0.03 * j
    end
    du_s = zeros(ComplexF64, n)
    du_f = zeros(ComplexF64, global_state_length(M))
    qrt_star_rhs!(du_s, view(u, 1:n), 0.1, 1.5, 0.25, delta_b, g_b, Et)
    rhs_cpu!(du_f, u, 0.1, 1.5, 0.25, delta_b, g_b, Et; threaded=false)
    err = qrt_relabs_err(du_s, view(du_f, 1:n))
    @info "QRT star vs rhs_cpu! (cross=0)" abs = err.abs rel = err.rel
    @test err.abs < 1e-12
    @test err.rel < 1e-12
end

@testset "QRT star RHS differs from rhs_cpu! when cross ≠ 0" begin
    M = 3
    Nj = fill(8.0, M)
    delta_b = [-0.4, 0.1, 0.35]
    g_b = [0.12, 0.18, 0.09]
    Et = 0.2 + 0.05im
    u = build_u0_cpu_2nd_order(M, Nj, :equator)
    n = small_length(M)
    @test any(!iszero, u[n+1:end])
    du_s = zeros(ComplexF64, n)
    du_f = zeros(ComplexF64, global_state_length(M))
    qrt_star_rhs!(du_s, view(u, 1:n), 0.1, 1.5, 0.25, delta_b, g_b, Et)
    rhs_cpu!(du_f, u, 0.1, 1.5, 0.25, delta_b, g_b, Et; threaded=false)
    err = qrt_relabs_err(du_s, view(du_f, 1:n))
    @info "QRT star vs rhs_cpu! (cross≠0, expected differ)" abs = err.abs rel = err.rel
    @test err.abs > 1e-8
end

@testset "QRT product J vs full-J oracle (v_cross = 0)" begin
    Random.seed!(12)
    for (M, ic, Et) in ((2, :equator, 0.0im), (3, :weak, 0.15 + 0.05im))
        Nj = fill(6.0, M)
        delta_b = collect(range(-0.3, 0.3; length=M))
        g_b = fill(0.14, M)
        u = build_u0_cpu_2nd_order(M, Nj, ic)
        n = small_length(M)
        u[IDX_a] = 0.25 + 0.1im
        u[IDX_ad_ad] = 0.04
        u[IDX_ad_a] = 0.3
        g_s = randn(ComplexF64, n)
        g_f = qrt_pad_product_tangent(g_s, M)
        dg_s = similar(g_s)
        dg_f = similar(g_f)
        qrt_product_apply!(dg_s, g_s, u, 0.08, 1.2, 0.3, delta_b, g_b, Et; ε=1e-7)
        qrt_oracle_apply!(dg_f, g_f, u, 0.08, 1.2, 0.3, delta_b, g_b, Et; ε=1e-7)
        err = qrt_relabs_err(dg_s, view(dg_f, 1:n))
        @info "QRT product vs oracle (v_cross=0)" M ic abs = err.abs rel = err.rel
        @test err.abs < 1e-8
        @test err.rel < 1e-6
        @test QRT_PRODUCT === :star_samebin_cavity
        @test QRT_ORACLE === :full_dense
        @test qrt_product_length(M) == n
        @test qrt_oracle_length(M) == global_state_length(M)
    end
end

@testset "QRT product ≠ oracle when tangent has cross" begin
    Random.seed!(13)
    M = 2
    Nj = fill(6.0, M)
    delta_b = [-0.2, 0.25]
    g_b = [0.11, 0.16]
    u = build_u0_cpu_2nd_order(M, Nj, :equator)
    n = small_length(M)
    g_f = randn(ComplexF64, global_state_length(M))
    g_s = g_f[1:n]
    dg_s = similar(g_s)
    dg_f = similar(g_f)
    qrt_product_apply!(dg_s, g_s, u, 0.0, 1.5, 0.25, delta_b, g_b, 0.0im; ε=1e-7)
    qrt_oracle_apply!(dg_f, g_f, u, 0.0, 1.5, 0.25, delta_b, g_b, 0.0im; ε=1e-7)
    err = qrt_relabs_err(dg_s, view(dg_f, 1:n))
    @info "QRT product vs oracle (v_cross≠0, expected differ)" abs = err.abs rel = err.rel
    @test err.abs > 1e-8
end

@testset "QRT oracle dense J*v vs apply" begin
    Random.seed!(14)
    M = 2
    Nj = fill(5.0, M)
    delta_b = [-0.15, 0.2]
    g_b = [0.1, 0.13]
    u = build_u0_cpu_2nd_order(M, Nj, :weak)
    u[IDX_a] = 0.2
    u[IDX_ad_a] = 0.25
    J = qrt_oracle_dense(u, 0.05, 1.0, 0.2, delta_b, g_b, 0.1im; ε=1e-7)
    @test size(J) == (global_state_length(M), global_state_length(M))
    v = randn(ComplexF64, length(u))
    dg = similar(v)
    qrt_oracle_apply!(dg, v, u, 0.05, 1.0, 0.2, delta_b, g_b, 0.1im; ε=1e-7)
    err = qrt_relabs_err(J * v, dg)
    @info "QRT oracle dense J*v vs apply" abs = err.abs rel = err.rel n = size(J, 1)
    @test err.abs < 1e-8
    @test err.rel < 1e-6
end

@testset "QRT adag seed has bosonic +1" begin
    M = 2
    Nj = fill(4.0, M)
    u = build_u0_cpu_2nd_order(M, Nj, :ground)
    u[IDX_a] = 0.4 + 0.1im
    u[IDX_ad_a] = 0.3
    g = zeros(ComplexF64, small_length(M))
    qrt_seed_adag_column!(g, view(u, 1:small_length(M)), M)
    @test g[IDX_a] ≈ u[IDX_ad_a] - abs2(u[IDX_a]) + 1
    ga = zeros(ComplexF64, small_length(M))
    qrt_seed_a_column!(ga, view(u, 1:small_length(M)), M)
    @test ga[IDX_a] ≈ conj(u[IDX_ad_ad]) - u[IDX_a]^2
end



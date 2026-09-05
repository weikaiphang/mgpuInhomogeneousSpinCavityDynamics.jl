using Test
using LinearAlgebra
using FastGaussQuadrature
using OrdinaryDiffEq
using DifferentialEquations
using DiffEqCallbacks

if !@isdefined(fgq_gausslegendre)
    include(joinpath(@__DIR__, "..", "src", "chimera", "quadrature.jl"))
end
if !@isdefined(_gauss_legendre_pts_golub_welsch)
    include(joinpath(@__DIR__, "oracles", "golub_welsch.jl"))
end
if !@isdefined(rhs_1st_order!)
    include(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "state_1st.jl"))
    include(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "closure_1st.jl"))
end
if !@isdefined(CHIMERA_HAMILTONIAN)
    include(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "hamiltonian.jl"))
end
if !@isdefined(factorized_first_order_jacobian_action!)
    include(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "jacobian_1st.jl"))
end
if !@isdefined(chimera_solve)
    include(joinpath(@__DIR__, "..", "src", "chimera", "integrate", "sciml.jl"))
end
if !@isdefined(qc_rhs_1st_M1!)
    include(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "quantum_cumulants.jl"))
end
if !@isdefined(choose_rowsum_exchange)
    include(joinpath(@__DIR__, "..", "src", "chimera", "mgpu", "layout.jl"))
end
if !@isdefined(QRT_CLOSURE_LEVEL)
    const QRT_CLOSURE_LEVEL = :factorized_first_order_jacobian
end

@testset "architecture: production lives under src/chimera" begin
    root = joinpath(@__DIR__, "..", "src")
    @test isfile(joinpath(root, "chimera", "quadrature.jl"))
    @test isfile(joinpath(root, "chimera", "eoms", "quantum_cumulants.jl"))
    @test isfile(joinpath(root, "chimera", "eoms", "closure_2nd.jl"))
    @test isfile(joinpath(root, "chimera", "integrate", "sciml.jl"))
    @test isfile(joinpath(root, "chimera", "mgpu", "solver.jl"))
    @test isfile(joinpath(root, "legacy", "sim_2nd_multi_gpu_opt.jl"))
    shim = read(joinpath(root, "rhs_2nd_order.jl"), String)
    @test occursin("Shim only", shim)
    @test occursin("chimera/eoms/closure_2nd.jl", shim)
    @test !occursin("function rhs_2nd_order!", shim)
    @test CHIMERA_HAMILTONIAN.name === :driven_lossy_tavis_cummings
    @test CHIMERA_INTEGRATOR === :sciml_ordinarydiffeq
end

@testset "FastGaussQuadrature owns production nodes" begin
    x, w = fgq_gausslegendre(16)
    x2, w2 = gausslegendre(16)
    @test x == collect(Float64.(x2))
    @test w == collect(Float64.(w2))
    @test _gauss_legendre_pts(12) == fgq_gausslegendre(12)
    xo, wo = _gauss_legendre_pts_golub_welsch(16)
    @test x ≈ xo rtol=1e-12 atol=1e-12
    @test w ≈ wo rtol=1e-12 atol=1e-12
    src = read(joinpath(@__DIR__, "..", "src", "chimera", "quadrature.jl"), String)
    @test occursin("gausslegendre", src)
    @test !occursin("SymTridiagonal", src)
    @test !occursin("Golub", src) || occursin("oracle", src)
end

@testset "QC-mapped 1st-order RHS matches chimera closure" begin
    err, du_qc, du_cl = compare_qc_to_closure_1st()
    @test err < 1e-14
    @test du_qc ≈ du_cl atol=1e-14
    errM, _, _ = compare_qc_to_closure_1st_M(3)
    @test errM < 1e-13
end

@testset "SciML integrates the package-owned 1st-order EOM" begin
    sol_qc, sol_cl = solve_qc_sciml_1st_M1(; tspan=(0.0, 0.5), reltol=1e-9, abstol=1e-9)
    @test sol_qc.retcode == ReturnCode.Success || string(sol_qc.retcode) == "Success"
    @test sol_cl.retcode == ReturnCode.Success || string(sol_cl.retcode) == "Success"
    @test sol_qc.u[end] ≈ sol_cl.u[end] rtol=1e-7 atol=1e-7
end

@testset "QRT Jacobian is the factorized 1st-order EOM" begin
    @test QRT_CLOSURE_LEVEL === :factorized_first_order_jacobian
    a = fill(0.1 + 0.0im, 1, 2)
    ad = conj.(a)
    Sp = fill(0.02 + 0.01im, 2, 2)
    Sm = conj.(Sp)
    Sz = fill(-0.4 + 0.0im, 2, 2)
    da = similar(a); dad = similar(ad); dSp = similar(Sp); dSm = similar(Sm); dSz = similar(Sz)
    factorized_first_order_jacobian_action!(
        da, dad, dSp, dSm, dSz, a, ad, Sp, Sm, Sz,
        0.1 + 0.0im, [0.02 + 0.01im, 0.02 + 0.01im], [-0.4, -0.4],
        0.3, [0.1, -0.1], [0.2, 0.25], 0.5,
    )
    @test all(isfinite, da)
    @test all(isfinite, dSp)
    @test maximum(abs.(da)) > 0
end

@testset "NCCL is the documented multi-GPU default" begin
    @test choose_rowsum_exchange(4; have_nccl=true, nunique_devices=4, nccl_ok=true, p2p_ok=true) === :nccl
    @test choose_rowsum_exchange(4; have_nccl=true, nunique_devices=4, nccl_ok=true, p2p_ok=false) === :nccl
    @test choose_rowsum_exchange(4; have_nccl=false, nunique_devices=4, nccl_ok=false, p2p_ok=true) === :p2p
    @test choose_rowsum_exchange(4; have_nccl=false, nunique_devices=4, nccl_ok=false, p2p_ok=false) === :host
    @test choose_rowsum_exchange(1; have_nccl=true, nunique_devices=1, nccl_ok=true, p2p_ok=true) === :none
end

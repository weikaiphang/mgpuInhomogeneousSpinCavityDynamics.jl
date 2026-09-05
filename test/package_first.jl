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
    @test isfile(joinpath(root, "chimera", "api.jl"))
    @test !isdir(joinpath(root, "legacy"))
    for gone in (
        "rhs_1st_order.jl", "rhs_1st_order_real.jl", "rhs_1st_order_ip.jl",
        "rhs_2nd_order.jl", "solver_1st_order.jl", "solver_2nd_order.jl",
        "state_layout_1st_order.jl", "state_layout_2nd_order.jl",
        "initial_conditions_1st_order.jl", "initial_conditions_2nd_order.jl",
        "frequency_inhomogeneity.jl", "coupling_inhomogeneity.jl",
        "ensemble.jl", "ensemble_quadrature.jl", "simulation_api.jl",
        "noise.jl", "MGPUlayout.jl", "MGPUrhs_cpu.jl", "MGPUproblem.jl",
        "MGPUkernels.jl", "MGPUintegrator.jl", "MGPUtableaus.jl",
        "MGPUdevices.jl", "MGPUobservables.jl", "MGPUinitial_conditions.jl",
        "MGPUstate_io.jl", "MGPUsolver.jl",
        "accel_solver_1st_order.jl", "sim_2nd_multi_gpu_opt.jl",
    )
        @test !isfile(joinpath(root, gone))
    end
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
    M, B = 2, 3
    a = fill(0.1 + 0.0im, B)
    ad = conj.(a)
    Sp = fill(0.02 + 0.01im, M, B)
    Sm = conj.(Sp)
    Sz = fill(-0.4 + 0.0im, M, B)
    da = similar(a); dad = similar(ad); dSp = similar(Sp); dSm = similar(Sm); dSz = similar(Sz)
    factorized_first_order_jacobian_action!(
        da, dad, dSp, dSm, dSz, a, ad, Sp, Sm, Sz,
        0.1 + 0.0im, fill(0.02 + 0.01im, M), fill(-0.4, M),
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

@testset "homemade MGPU stepper is gone; SciML owns time" begin
    mgpu = joinpath(@__DIR__, "..", "src", "chimera", "mgpu")
    @test !isfile(joinpath(mgpu, "integrator.jl"))
    @test !isfile(joinpath(mgpu, "tableaus.jl"))
    blobs = Dict(f => read(joinpath(mgpu, f), String) for f in readdir(mgpu) if endswith(f, ".jl"))
    allsrc = join(values(blobs), "\n")
    @test !occursin("Tsit5Tableau", allsrc)
    @test !occursin("CK45Tableau", allsrc)
    @test !occursin("function solve_mgpu!", allsrc)
    @test !occursin("function tsit5_step!", allsrc)
    @test !occursin("function ck45_step!", allsrc)
    @test !occursin("function cross_rhs_kernel!", allsrc)
    @test !occursin("function small_rhs_kernel!", allsrc)
    @test !occursin("function combine_kernel!", allsrc)
    @test !occursin("function lowstorage_kernel!", allsrc)
    @test occursin("NCCL.Allreduce!", blobs["problem.jl"])
    @test occursin("exchange_rowsums_p2p!", blobs["problem.jl"])
    @test occursin("rhs_2nd_order_mgpu!", blobs["problem.jl"])
    @test occursin("rowsum_partial_kernel!", blobs["kernels.jl"])
    @test occursin("chimera_solve", blobs["solver.jl"])
    @test occursin("chimera_ode_problem", blobs["solver.jl"])
    @test occursin("return run_sim_1st_order", blobs["solver.jl"])
    @test count("NCCL.Allreduce!", blobs["problem.jl"]) == 1
    @test CHIMERA_INTEGRATOR === :sciml_ordinarydiffeq
    sciml = read(joinpath(@__DIR__, "..", "src", "chimera", "integrate", "sciml.jl"), String)
    @test occursin("OrdinaryDiffEq", sciml)
    @test !occursin("custom MGPU Tsit5/CK45", sciml)
    eoms = read(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "closure_2nd.jl"), String)
    @test occursin("injected_cross_rowsums", eoms)
    @test !occursin("fused sharded replica", eoms)
end

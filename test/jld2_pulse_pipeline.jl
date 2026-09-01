# Signal/control identification and jld2 optimisation pipeline.
# Signal is a fixed background; only the control envelope is in `u`.

using Test
using Random
using ForwardDiff
using DifferentialEquations
using Printf
using LinearAlgebra
using JLD2
using Distributions
using QuadGK

const _SRC = joinpath(@__DIR__, "..", "src")

if !isdefined(@__MODULE__, :CompositePulse)
    include(joinpath(_SRC, "pulses.jl"))
    include(joinpath(_SRC, "bspline.jl"))
    include(joinpath(_SRC, "composite_pulse.jl"))
    include(joinpath(_SRC, "state_layout_1st_order.jl"))
    include(joinpath(_SRC, "rhs_1st_order.jl"))
    include(joinpath(_SRC, "pulse_optimizer2.jl"))
end
if !isdefined(@__MODULE__, :prepare_derived)
    include(joinpath(_SRC, "frequency_inhomogeneity.jl"))
    include(joinpath(_SRC, "coupling_inhomogeneity.jl"))
    include(joinpath(_SRC, "ensemble.jl"))
end
if !isdefined(@__MODULE__, :load_jld2_reference)
    include(joinpath(_SRC, "jld2_pulse_loader.jl"))
end
if !isdefined(@__MODULE__, :build_full_config)
    build_full_config(a, b) = merge(a, b)
end

const D = (
    timespan=(0.0, 1100e-6),
    FWHM=2 * pi * 1e6,
    kappa_t=2 * pi * 1e6,
    g_mean=2 * pi * 100,
    sqrt_kappa_e=sqrt(2 * pi * 1e6),
)

sqrt_ke = D.sqrt_kappa_e
FWHM = D.FWHM

function gaussian_cfg(; t0=50e-6, sigma=3e-6, amp=0.5 * sqrt_ke * 0.332)
    return (
        name="Gaussian input signal", kind=:gaussian,
        t0=t0, sigma=sigma, amp=amp, omega=0.0, phase=0.0,
    )
end

function wurst_cfg(; name="WURST", t_center=300e-6, duration=400e-6, amp=0.5 * sqrt_ke * 2.0e4)
    return (
        name=name, kind=:wurst,
        t_center=t_center, duration=duration, amp=amp,
        bandwidth=5.0 * FWHM, n=20.0, omega0=0.0, chirp_sign=+1.0,
        phase0=0.0, edge_frac=1e-4,
    )
end

@testset "ROSE: gaussian signal + two WURST controls" begin
    pc = (
        gaussian_cfg(),
        wurst_cfg(name="First WURST", t_center=300e-6, duration=400e-6),
        wurst_cfg(name="Second WURST", t_center=800e-6, duration=400e-6),
    )
    r = segment_signal_control(pc; d=D)
    @test r.ok
    @test r.signal_idx == [1]
    @test r.control_idx == [2, 3]
    @test r.features[2].is_wurst
    @test r.features[3].is_wurst
    @test !r.features[1].is_wurst
    @test !r.features[1].is_pi
    @test r.features[1].peak <= 0.1 * r.min_control_mean_A
    sig, ctrl = identified_signal_control(pc; d=D)
    @test length(sig) == 1
    @test length(ctrl) == 2
end

@testset "3ARP: three WURSTs, no signal" begin
    pc = (
        wurst_cfg(name="s1", t_center=55e-6, duration=10e-6),
        wurst_cfg(name="s2", t_center=75e-6, duration=20e-6, amp=0.5 * sqrt_ke * 2.0e4 / sqrt(2)),
        wurst_cfg(name="s3", t_center=95e-6, duration=10e-6),
    )
    r = segment_signal_control(pc; d=D)
    @test r.ok
    @test isempty(r.signal_idx)
    @test r.control_idx == [1, 2, 3]
    @test all(f.is_wurst for f in r.features)
end

@testset "RASE: single WURST is control only" begin
    pc = (wurst_cfg(t_center=75e-6, duration=10e-6),)
    r = segment_signal_control(pc; d=D)
    @test r.ok
    @test isempty(r.signal_idx)
    @test r.control_idx == [1]
end

@testset "forbidden: signal after control" begin
    pc = (
        wurst_cfg(t_center=50e-6, duration=10e-6),
        gaussian_cfg(t0=80e-6),
    )
    r = segment_signal_control(pc; d=D)
    @test !r.ok
    @test occursin("follows a control", r.reason) || occursin("overlap", r.reason)
    @test_throws SignalControlRejected identified_signal_control(pc; d=D)
end

@testset "two signals then control is allowed" begin
    pc = (
        gaussian_cfg(t0=20e-6),
        gaussian_cfg(t0=40e-6),
        wurst_cfg(t_center=200e-6, duration=40e-6),
    )
    r = segment_signal_control(pc; d=D)
    @test r.ok
    @test r.signal_idx == [1, 2]
    @test r.control_idx == [3]
end

@testset "hard π (area) is control; weak gaussian is signal" begin
    ΩE = 4 * D.g_mean * D.sqrt_kappa_e / D.kappa_t
    T = 10e-6
    t_on = 50e-6
    Eπ = π / (ΩE * T)
    hard_pi = (
        name="hard pi", kind=:custom,
        t_start=t_on, t_end=t_on + T,
        f=t -> (t_on <= t <= t_on + T ? Eπ : 0.0 + 0.0im),
    )
    pc = (gaussian_cfg(t0=20e-6), hard_pi)
    dπ = merge(D, (timespan=(0.0, 80e-6),))
    r = segment_signal_control(pc; d=dπ)
    @test r.ok
    @test r.features[2].is_pi_area
    @test r.control_idx == [2]
    @test r.signal_idx == [1]
end

@testset "trace path uses _detect_subpulse_segments" begin
    pc = (
        gaussian_cfg(t0=20e-6, sigma=2e-6),
        wurst_cfg(t_center=80e-6, duration=20e-6),
    )
    E = build_E_of_t(pc)
    t_end = 120e-6
    N = 8001
    t = collect(range(0.0, t_end; length=N))
    I = Vector{Float64}(undef, N)
    Q = Vector{Float64}(undef, N)
    for i in 1:N
        z = E(t[i])
        I[i] = real(z)
        Q[i] = imag(z)
    end
    A = hypot.(I, Q)
    segs_once = _detect_subpulse_segments(t, A)
    # A single global-max pass misses the Gaussian (peak ≪ 1e-3 × WURST).
    @test length(segs_once) == 1
    r = segment_signal_control_from_trace(t, I, Q; d=D)
    @test r.ok
    @test length(r.signal_idx) == 1
    @test length(r.control_idx) >= 1
    @test r.features[r.control_idx[1]].is_wurst || r.features[r.control_idx[1]].is_pi
end

@testset "positional first-WURST is not a signal" begin
    pc = (
        wurst_cfg(name="first", t_center=55e-6, duration=10e-6),
        wurst_cfg(name="second", t_center=75e-6, duration=20e-6),
        wurst_cfg(name="third", t_center=95e-6, duration=10e-6),
    )
    r = segment_signal_control(pc; d=D)
    @test r.ok
    @test isempty(r.signal_idx)
end

@testset "one control envelope excludes signal" begin
    pc = (
        gaussian_cfg(),
        wurst_cfg(name="First WURST", t_center=300e-6, duration=400e-6),
        wurst_cfg(name="Second WURST", t_center=800e-6, duration=400e-6),
    )
    r = segment_signal_control(pc; d=D)
    @test r.ok
    E_c = control_envelope_E_of_t(r)
    E_s = signal_envelope_E_of_t(r)
    @test abs(E_c(50e-6)) < 1e-3 * abs(E_s(50e-6))
    @test abs(E_s(300e-6)) < 1e-3 * abs(E_c(300e-6))
    @test abs(E_s(800e-6)) < 1e-3 * abs(E_c(800e-6))
    E_off = signal_envelope_E_of_t(r; use_signal=false)
    @test E_off(50e-6) == 0
    t = collect(range(0.0, 1100e-6; length=2001))
    I = [real(E_s(ti) + E_c(ti)) for ti in t]
    Q = [imag(E_s(ti) + E_c(ti)) for ti in t]
    Ex, Ep = mask_control_envelope_samples(t, I, Q, r)
    i_sig = argmin(abs.(t .- 50e-6))
    i_w1 = argmin(abs.(t .- 300e-6))
    @test hypot(Ex[i_sig], Ep[i_sig]) < 1e-3 * hypot(Ex[i_w1], Ep[i_w1])
end

const E2E_SQRT_KE = sqrt(2 * pi * 1e6)
const E2E_FWHM = 2 * pi * 1e6
const E2E_D_ID = (
    timespan=(0.0, 1100e-6),
    FWHM=E2E_FWHM,
    kappa_t=2 * pi * 1e6,
    g_mean=2 * pi * 100,
    sqrt_kappa_e=E2E_SQRT_KE,
)
const E2E_D_ODE = (
    timespan=(0.0, 200e-6),
    FWHM=E2E_FWHM,
    kappa_t=2 * pi * 1e6,
    g_mean=2 * pi * 100,
    sqrt_kappa_e=E2E_SQRT_KE,
    M=3,
    Nj=[2.0, 4.0, 6.0],
    delta0=0.0,
    kappa_e=2 * pi * 1e6,
    kappa_i=0.0,
    delta_b=[0.0, 2 * pi * 5e4, -2 * pi * 5e4],
    g_b=[2 * pi * 100, 2 * pi * 90, 2 * pi * 110],
)

e2e_gaussian(; t0=50e-6, sigma=3e-6, amp=0.5 * E2E_SQRT_KE * 0.332) = (
    name="Gaussian input signal", kind=:gaussian,
    t0=t0, sigma=sigma, amp=amp, omega=0.0, phase=0.0,
)
e2e_wurst(; name="WURST", t_center=300e-6, duration=400e-6, amp=0.5 * E2E_SQRT_KE * 2.0e4) = (
    name=name, kind=:wurst,
    t_center=t_center, duration=duration, amp=amp,
    bandwidth=5.0 * E2E_FWHM, n=20.0, omega0=0.0, chirp_sign=+1.0,
    phase0=0.0, edge_frac=1e-4,
)

# Production ROSE (examples/rose_1st_order.jl) and 3ARP (paper/fig_4_d).
const E2E_ROSE_PC = (
    e2e_gaussian(),
    e2e_wurst(name="First WURST", t_center=300e-6, duration=400e-6),
    e2e_wurst(name="Second WURST", t_center=800e-6, duration=400e-6),
)
const E2E_3ARP_PC = (
    e2e_wurst(name="WURST pulse 1", t_center=55e-6, duration=10e-6),
    e2e_wurst(name="WURST pulse 2", t_center=75e-6, duration=20e-6, amp=0.5 * E2E_SQRT_KE * 2.0e4 / sqrt(2)),
    e2e_wurst(name="WURST pulse 3", t_center=95e-6, duration=10e-6),
)
# Compact ROSE: same kinds/order, short enough for a cheap Dual ODE.
const E2E_ROSE_COMPACT_PC = (
    e2e_gaussian(t0=15e-6, sigma=2e-6),
    e2e_wurst(name="First WURST", t_center=80e-6, duration=50e-6),
    e2e_wurst(name="Second WURST", t_center=155e-6, duration=50e-6),
)

function e2e_tiny_sim_system(; Ttotal=200e-6, Nt_save=51, M_delta=4)
    SIM_SETTING = (
        simulation_order=:order1,
        M_delta=M_delta,
        M_g=1,
        initial_condition=:ground,
        Ttotal=Ttotal,
        Nt_save=Nt_save,
        reltol=1e-6,
        abstol=1e-6,
    )
    SYSTEM_CONFIG = (
        C_ens=0.6,
        delta0=0.0,
        kappa_e=2 * pi * 1e6,
        kappa_i=0.0,
        freq_inhomogeneity=(
            kind=:lorentzian, FWHM=2 * pi * 1e6, span_gamma=2.5, renormalize=false,
        ),
        g_inhomogeneity=(kind=:constant, g_value=2 * pi * 100),
    )
    return SIM_SETTING, SYSTEM_CONFIG
end

function e2e_write_jld2(path, SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG)
    data = (SIM_SETTING=SIM_SETTING, SYSTEM_CONFIG=SYSTEM_CONFIG, PULSE_CONFIG=PULSE_CONFIG)
    JLD2.save(path, "data", data)
    return data
end

function e2e_iq_at(E, t)
    z = E(t)
    return hypot(real(z), imag(z))
end

function e2e_opt_kwargs()
    return (
        num_epochs=1,
        n_hops=1,
        patience=1,
        compute=:cpu,
        threaded_grad=false,
        anneal_direct_weights=false,
        reltol=1e-6,
        abstol=1e-6,
        verbose=false,
        save_log=true,
        fit_N=1001,
        param_budget=40,
        use_interior=false,
    )
end

@testset "pipeline invocation signatures" begin
    @test jld2_pipeline_defaults().n_signal === nothing
    @test jld2_pipeline_defaults().use_signal === false
    @test jld2_pipeline_defaults().use_interior === false
    @test hasmethod(try_parse_pulse_config, Tuple{NamedTuple})
    @test hasmethod(load_jld2_reference, Tuple{String})
    @test hasmethod(run_reference_forward, Tuple{Any})
    @test hasmethod(reconcile_reference, Tuple{Any,Any})
    @test hasmethod(fit_linear_seed, Tuple{Any})
    @test hasmethod(optimise_control_pulse_from_jld2, Tuple{String})
    @test hasmethod(segment_signal_control, Tuple{Tuple})
    @test hasmethod(identified_signal_control, Tuple{Tuple})
    @test hasmethod(control_envelope_E_of_t, Tuple{NamedTuple})
    @test hasmethod(signal_envelope_E_of_t, Tuple{NamedTuple})
    @test hasmethod(build_signal_E_of_t, Tuple{Any,Bool})
    @test hasmethod(run_sim_1st_order_pure, Tuple{AbstractVector,CompositePulse,Any})
    @test hasmethod(pulse_cost, Tuple{AbstractVector,CompositePulse,Any})
    @test hasmethod(optimise_composite_pulse, Tuple{Integer,Integer,Integer,Any})
    # n_signal=nothing must dispatch (was Integer-only and threw MethodError).
    mlog = methods(save_optimisation_run_log)
    @test length(mlog) >= 1
    pipe, opt, verbose = _jld2_split_kwargs((;))
    @test pipe.n_signal === nothing
    @test pipe.use_interior === false
    @test pipe.use_signal === false
    @test verbose === true
    @test haskey(opt, :num_epochs)
    pipe_on, _, _ = _jld2_split_kwargs((; use_interior=true, use_signal=true))
    @test pipe_on.use_interior === true
    @test pipe_on.use_signal === true
    @test_throws ErrorException _jld2_split_kwargs((; not_a_real_knob=1))

    # hop0_phyonly is a canonical optimiser knob: present in the defaults,
    # auto-allowlisted, and forwarded into the `opt` split.
    @test haskey(jld2_optimizer_defaults(), :hop0_phyonly)
    @test jld2_optimizer_defaults().hop0_phyonly === true
    _, opt_h0, _ = _jld2_split_kwargs((; hop0_phyonly=false))
    @test opt_h0.hop0_phyonly === false
end

@testset "try_parse ROSE vs 3ARP (not positional)" begin
    rose = try_parse_pulse_config((PULSE_CONFIG=E2E_ROSE_PC,); d=E2E_D_ID)
    @test rose.ok
    @test length(rose.signal_cfg) == 1
    @test length(rose.control_cfg) == 2
    @test rose.signal_cfg[1].kind === :gaussian
    @test all(c.kind === :wurst for c in rose.control_cfg)
    @test rose.identification.signal_idx == [1]
    @test rose.identification.control_idx == [2, 3]

    rose1 = try_parse_pulse_config((PULSE_CONFIG=E2E_ROSE_PC,); n_signal=1, d=E2E_D_ID)
    @test rose1.ok

    rose_bad_n = try_parse_pulse_config((PULSE_CONFIG=E2E_ROSE_PC,); n_signal=0, d=E2E_D_ID)
    @test !rose_bad_n.ok

    arp = try_parse_pulse_config((PULSE_CONFIG=E2E_3ARP_PC,); d=E2E_D_ID)
    @test arp.ok
    @test arp.signal_cfg === () || arp.signal_cfg === nothing || isempty(arp.signal_cfg)
    @test length(arp.control_cfg) == 3
    @test all(c.kind === :wurst for c in arp.control_cfg)

    # Old positional n_signal=1 would steal the first WURST as "signal".
    sig_pos, ctrl_pos = split_signal_control(E2E_3ARP_PC; n_signal=1)
    @test length(sig_pos) == 1 && sig_pos[1].kind === :wurst
    arp_n1 = try_parse_pulse_config((PULSE_CONFIG=E2E_3ARP_PC,); n_signal=1, d=E2E_D_ID)
    @test !arp_n1.ok
    @test occursin("n_signal=1", arp_n1.message)

    @test !try_parse_pulse_config((;)).ok
    @test !try_parse_pulse_config((PULSE_CONFIG=nothing,)).ok
end

@testset "control envelope objects exclude signal" begin
    ident = segment_signal_control(E2E_ROSE_COMPACT_PC; d=E2E_D_ODE)
    @test ident.ok
    E_c = control_envelope_E_of_t(ident)
    E_s = signal_envelope_E_of_t(ident)
    E_s_off = signal_envelope_E_of_t(ident; use_signal=false)
    @test E_s_off === _zero_drive || E_s_off(15e-6) == 0
    @test e2e_iq_at(E_c, 15e-6) < 1e-3 * e2e_iq_at(E_s, 15e-6)
    @test e2e_iq_at(E_s, 80e-6) < 1e-3 * e2e_iq_at(E_c, 80e-6)
    @test e2e_iq_at(E_s, 155e-6) < 1e-3 * e2e_iq_at(E_c, 155e-6)
    @test e2e_iq_at(E_c, 80e-6) > 0
    @test e2e_iq_at(E_c, 155e-6) > 0

    E_cfg = build_E_of_t(ident.control_cfg)
    @test e2e_iq_at(E_c, 80e-6) ≈ e2e_iq_at(E_cfg, 80e-6) rtol=1e-12
    @test build_signal_E_of_t(ident.signal_cfg, true)(15e-6) ≈ E_s(15e-6)
    @test build_signal_E_of_t(ident.signal_cfg, false)(15e-6) == 0
    @test build_signal_E_of_t((), true)(15e-6) == 0
    @test build_signal_E_of_t(nothing, true)(15e-6) == 0

    t, Ex, Ep = _sample_control_cfg(ident.control_cfg, E2E_D_ODE, 2001)
    @test length(t) == length(Ex) == length(Ep) == 2001
    i_sig = argmin(abs.(t .- 15e-6))
    i_w1 = argmin(abs.(t .- 80e-6))
    @test hypot(Ex[i_sig], Ep[i_sig]) < 1e-3 * hypot(Ex[i_w1], Ep[i_w1])

    ident3 = segment_signal_control(E2E_3ARP_PC; d=merge(E2E_D_ID, (timespan=(0.0, 120e-6),)))
    @test ident3.ok
    @test isempty(ident3.signal_idx)
    @test signal_envelope_E_of_t(ident3)(75e-6) == 0
    @test e2e_iq_at(control_envelope_E_of_t(ident3), 75e-6) > 0
end

@testset "seed fit is control-only; ODE sees control(u)+signal" begin
    ident = segment_signal_control(E2E_ROSE_COMPACT_PC; d=E2E_D_ODE)
    t, Ex, Ep = _sample_control_cfg(ident.control_cfg, E2E_D_ODE, 2001)
    fake_ref = (
        control_t=t, control_Ex=Ex, control_Ep=Ep, d=E2E_D_ODE,
        pulse_source=:pulse_config,
    )
    pulse, u_fit, fit_report, segments = fit_linear_seed(
        fake_ref; param_budget=40, degree=3, verbose=false,
    )
    @test pulse isa CompositePulse
    @test pulse.k == 2
    @test length(u_fit) == n_params(pulse)
    @test isfinite(fit_report.rel_l2_complex)
    @test length(segments) == 2

    E_u = build_E_of_t(pulse, u_fit)
    @test e2e_iq_at(E_u, 15e-6) < 0.05 * e2e_iq_at(control_envelope_E_of_t(ident), 80e-6)
    @test e2e_iq_at(E_u, 80e-6) > 0
    @test e2e_iq_at(E_u, 155e-6) > 0

    E_s = signal_envelope_E_of_t(ident)
    a_on, Sp_on, Sz_on, Nj_on = run_sim_1st_order_pure(
        u_fit, pulse, E2E_D_ODE; signal_E_of_t=E_s, compute=:cpu, reltol=1e-6, abstol=1e-6,
    )
    a_off, _, Sz_off, _ = run_sim_1st_order_pure(
        u_fit, pulse, E2E_D_ODE; signal_E_of_t=_zero_drive, compute=:cpu, reltol=1e-6, abstol=1e-6,
    )
    @test Nj_on == E2E_D_ODE.Nj
    @test length(Sp_on) == E2E_D_ODE.M
    @test isfinite(a_on) && isfinite(a_off)
    @test a_on != a_off || sum(Sz_on) != sum(Sz_off)

    cost_on = pulse_cost(u_fit, pulse, E2E_D_ODE; signal_E_of_t=E_s, compute=:cpu, reltol=1e-6, abstol=1e-6)
    cost_off = pulse_cost(u_fit, pulse, E2E_D_ODE; signal_E_of_t=_zero_drive, compute=:cpu, reltol=1e-6, abstol=1e-6)
    @test isfinite(cost_on[1]) && isfinite(cost_off[1])
    @test cost_on[1] != cost_off[1]

    g = ForwardDiff.gradient(
        uu -> pulse_cost(uu, pulse, E2E_D_ODE; signal_E_of_t=E_s, compute=:cpu, reltol=1e-6, abstol=1e-6)[1],
        u_fit,
    )
    @test length(g) == length(u_fit)
    @test all(isfinite, g)

    best_u, best_cost, pulse2, u0, initial_metrics, history, final_metrics, settings =
        optimise_composite_pulse(
            pulse.k, pulse.n_coeff_A, pulse.n_coeff_f, E2E_D_ODE;
            num_epochs=1, n_hops=1, patience=1, compute=:cpu, threaded_grad=false,
            anneal_direct_weights=false, warm_start_u=u_fit, signal_E_of_t=E_s,
            reltol=1e-6, abstol=1e-6, label_prefix="[e2e] ",
        )
    @test length(best_u) == n_params(pulse2)
    @test pulse2.k == pulse.k
    @test isfinite(best_cost)
    @test length(history) >= 1
    @test !haskey(settings, :signal_E_of_t)  # closure is captured, not serialised
    @test isfinite(initial_metrics[1]) && isfinite(final_metrics[1])
end

@testset "load_jld2_reference + reconcile + full jld2 entry (ROSE)" begin
    mktempdir() do tmp
        SIM, SYS = e2e_tiny_sim_system(; Ttotal=200e-6, Nt_save=51, M_delta=4)
        path = joinpath(tmp, "rose_compact.jld2")
        e2e_write_jld2(path, SIM, SYS, E2E_ROSE_COMPACT_PC)

        ref = load_jld2_reference(path; verbose=false, fit_N=1001)
        @test ref.parse_ok
        @test ref.pulse_source === :pulse_config
        @test ref.use_interior === false
        @test ref.n_signal == 1
        @test ref.n_signal_check === nothing
        @test ref.identification !== nothing
        @test ref.use_signal === false
        @test length(ref.signal_cfg) == 1
        @test length(ref.control_cfg) == 2
        @test ref.signal_cfg[1].kind === :gaussian
        @test all(c.kind === :wurst for c in ref.control_cfg)
        @test ref.d.M == 4
        i_sig = argmin(abs.(ref.control_t .- 15e-6))
        i_w1 = argmin(abs.(ref.control_t .- 80e-6))
        @test hypot(ref.control_Ex[i_sig], ref.control_Ep[i_sig]) <
              1e-3 * hypot(ref.control_Ex[i_w1], ref.control_Ep[i_w1])
        @test ref.signal_E_of_t(15e-6) == 0
        @test e2e_iq_at(ref.signal_E_always, 15e-6) > 0
        @test e2e_iq_at(ref.signal_E_always, 80e-6) < 1e-3 * e2e_iq_at(ref.signal_E_always, 15e-6)
        ident = segment_signal_control(E2E_ROSE_COMPACT_PC; d=ref.d)
        @test e2e_iq_at(ref.recorded_E_of_t, 15e-6) ≈ e2e_iq_at(signal_envelope_E_of_t(ident), 15e-6) rtol=1e-8
        @test e2e_iq_at(ref.recorded_E_of_t, 80e-6) ≈ e2e_iq_at(control_envelope_E_of_t(ident), 80e-6) rtol=1e-8

        ref_on = load_jld2_reference(path; use_signal=true, verbose=false, fit_N=501)
        @test e2e_iq_at(ref_on.signal_E_of_t, 15e-6) > 0
        @test e2e_iq_at(ref_on.signal_E_always, 15e-6) > 0
        ref_yes_int = load_jld2_reference(path; use_interior=true, verbose=false, fit_N=501)
        @test ref_yes_int.use_interior === true

        forward = run_reference_forward(ref; compute=:cpu, verbose=false)
        @test 0 <= forward.metrics.inversion <= 1
        @test 0 <= forward.metrics.silencing <= 1
        @test isfinite(forward.metrics.duration)
        ok, report = reconcile_reference(ref, forward; verbose=false)
        @test ok
        @test report.auto_pass

        pulse, u_fit, _, segs = fit_linear_seed(ref; param_budget=40, verbose=false)
        @test pulse.k == 2
        @test length(segs) == 2
        @test length(u_fit) == n_params(pulse)

        pulse_int, u_int, irep, segs_int = generate_interior_seed(
            u_fit, forward.metrics.inversion, forward.metrics.silencing, pulse, ref.d;
            preserve_shape=true, N_samples=max(length(ref.control_t), 2), param_budget=40,
        )
        @test pulse_int.k == pulse.k
        @test length(u_int) == n_params(pulse)
        @test length(segs_int) == 2
        @test isfinite(irep.amp_scale_factor)
        @test irep.inversion_in == forward.metrics.inversion
        @test irep.silencing_in == forward.metrics.silencing

        # Count-check still works for ROSE; 3ARP-style n_signal=1 must not load.
        ref_n1 = load_jld2_reference(path; n_signal=1, verbose=false, fit_N=501)
        @test ref_n1.parse_ok
        @test_throws ErrorException load_jld2_reference(path; n_signal=2, verbose=false)

        kw = e2e_opt_kwargs()
        best_u, best_cost, pulse_opt, signal_E, d_opt, data, seed_rep, ref_met =
            optimise_control_pulse_from_jld2(path; kw..., log_out_dir=tmp)
        @test length(best_u) == n_params(pulse_opt)
        @test pulse_opt.k == 2
        @test isfinite(best_cost)
        @test signal_E(15e-6) == 0
        @test d_opt.M == 4
        @test data.PULSE_CONFIG[1].kind === :gaussian
        @test seed_rep !== nothing
        @test length(seed_rep.u_fit) == n_params(pulse_opt)
        @test !haskey(seed_rep, :interior_report)
        @test isfinite(ref_met.inversion)
        logp, pmat, ppara = optrunlog_paths(path; out_dir=tmp)
        @test isfile(logp) && isfile(pmat) && isfile(ppara)
        log = JLD2.load(logp, "data")
        @test log.n_signal == 1
        @test log.use_signal === false
        @test log.optimizer_settings.use_interior === false
        @test length(log.final_u) == length(best_u)
        # Saved pulsemat is CONTROL only (no Gaussian lobe).
        t_end_csv, Exs, Eps = load_E_samples(pmat)
        tcsv = collect(range(0.0, t_end_csv; length=length(Exs)))
        i_g = argmin(abs.(tcsv .- 15e-6))
        i_c = argmin(abs.(tcsv .- 80e-6))
        @test hypot(Exs[i_g], Eps[i_g]) < 0.05 * hypot(Exs[i_c], Eps[i_c])
    end
end

@testset "load_jld2_reference 3ARP is all control" begin
    mktempdir() do tmp
        SIM, SYS = e2e_tiny_sim_system(; Ttotal=120e-6, Nt_save=41, M_delta=3)
        path = joinpath(tmp, "three_arp.jld2")
        e2e_write_jld2(path, SIM, SYS, E2E_3ARP_PC)
        ref = load_jld2_reference(path; verbose=false, fit_N=801)
        @test ref.parse_ok
        @test (ref.signal_cfg === () || isempty(ref.signal_cfg))
        @test ref.n_signal == 0
        @test length(ref.control_cfg) == 3
        @test ref.signal_E_of_t(75e-6) == 0
        i_w2 = argmin(abs.(ref.control_t .- 75e-6))
        @test hypot(ref.control_Ex[i_w2], ref.control_Ep[i_w2]) > 0
        @test_throws ErrorException load_jld2_reference(path; n_signal=1, verbose=false)
        pulse, u_fit, _, segs = fit_linear_seed(ref; param_budget=40, verbose=false)
        @test pulse.k == 3
        @test length(segs) == 3
        cost0 = pulse_cost(
            u_fit, pulse, ref.d; signal_E_of_t=ref.signal_E_of_t,
            compute=:cpu, reltol=1e-6, abstol=1e-6,
        )
        @test isfinite(cost0[1])
    end
end

@testset "rejected PULSE_CONFIG is a hard error (no mixed CSV fit)" begin
    mktempdir() do tmp
        SIM, SYS = e2e_tiny_sim_system(; Ttotal=120e-6, Nt_save=41, M_delta=3)
        bad_pc = (
            e2e_wurst(t_center=40e-6, duration=10e-6),
            e2e_gaussian(t0=90e-6),
        )
        path = joinpath(tmp, "bad_order.jld2")
        e2e_write_jld2(path, SIM, SYS, bad_pc)
        mixed = build_E_of_t(bad_pc)
        sample_E_of_t(mixed, 120e-6, 501; savepath=joinpath(tmp, "bad_order_pulsemat.csv"))
        @test_throws ErrorException load_jld2_reference(path; verbose=false)
    end
end

@testset "CSV fallback identifies control envelope from mixed I/Q" begin
    mktempdir() do tmp
        SIM, SYS = e2e_tiny_sim_system(; Ttotal=200e-6, Nt_save=1001, M_delta=3)
        path = joinpath(tmp, "csv_only.jld2")
        data = (SIM_SETTING=SIM, SYSTEM_CONFIG=SYS)
        JLD2.save(path, "data", data)
        mixed = build_E_of_t(E2E_ROSE_COMPACT_PC)
        csvp = joinpath(tmp, "csv_only_pulsemat.csv")
        sample_E_of_t(mixed, 200e-6, 1001; savepath=csvp)
        ref = load_jld2_reference(path; verbose=false)
        @test ref.pulse_source === :pulsemat_csv
        @test !ref.parse_ok
        i_sig = argmin(abs.(ref.control_t .- 15e-6))
        i_w1 = argmin(abs.(ref.control_t .- 80e-6))
        @test hypot(ref.control_Ex[i_sig], ref.control_Ep[i_sig]) <
              1e-3 * hypot(ref.control_Ex[i_w1], ref.control_Ep[i_w1])
        @test hypot(ref.control_Ex[i_w1], ref.control_Ep[i_w1]) > 0
        @test e2e_iq_at(ref.recorded_E_of_t, 15e-6) > 0  # mixed recorded drive still has signal
        @test ref.signal_E_of_t(15e-6) == 0  # default use_signal=false zeros the opt background
        @test e2e_iq_at(ref.signal_E_always, 15e-6) > 0
        pulse, u_fit, _, segs = fit_linear_seed(ref; param_budget=40, verbose=false)
        @test pulse.k >= 1
        @test length(u_fit) == n_params(pulse)
        @test !isempty(segs)
    end
end

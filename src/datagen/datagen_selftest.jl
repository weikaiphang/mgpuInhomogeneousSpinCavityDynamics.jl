
include(joinpath(@__DIR__, "datagen_run.jl"))

using Test
using Random
using JLD2
using CUDA
using .DataGen

const DG = DataGen
const QUICK = "--quick" in ARGS
const SKIP_ODE = QUICK || "--skip-ode" in ARGS

function fake_job(i::Int; key::AbstractString="k$i", stem::AbstractString="stem$i")
    return (
        stem = String(stem),
        key = String(key),
        outpath = "/tmp/datagen_selftest_$i.jld2",
        split = (
            M_delta = 8,
            M_g = 1,
            M_total = 8,
            safety_factor = 4.0,
        ),
    )
end

function canonical_rase_spec()
    sys = DG.canonical_system()
    design = first(d for d in DG.rase_designs() if d.canonical)
    return sys, DG.bind_pulse(design, sys)
end

ts = @testset verbose = true begin

    @testset "CLI parse" begin
        @test_throws ErrorException DG.parse_args(String[])
        opt = DG.parse_args(["--phase", "configs"])
        @test opt.help == false
        @test opt.phase == "configs"
        @test opt.dry_run == false
        @test opt.skip_existing == true
        @test opt.run.ics == (:ground, :equator)
        @test opt.run.M_cap == DG.RULE_M_CAP
        @test opt.run.M_g_max == DG.RULE_M_G_MAX
        @test opt.run.n_sizes == 1
        @test opt.run.Nt_save == DG.RULE_NT_SAVE

        @test DG.parse_args(["--help"]).help == true
        @test DG.parse_args(["-h"]).help == true

        opt = DG.parse_args([
            "--phase", "simulate",
            "--start", "2",
            "--stop", "9",
            "--limit", "3",
            "--no-skip",
            "--default-conditions", "equatorial",
            "--M-cap", "800",
            "--M-g-cap", "4",
            "--M-sizing", "4",
            "--NT-save", "11",
        ])
        @test opt.phase == "simulate"
        @test opt.start_id == 2
        @test opt.stop_id == 9
        @test opt.limit == 3
        @test opt.skip_existing == false
        @test opt.run.ics == (:equator,)
        @test opt.run.M_cap == 800
        @test opt.run.M_g_max == 4
        @test opt.run.n_sizes == 4
        @test opt.run.Nt_save == 11

        opt = DG.parse_args(["--phase", "configs", "--conditions", "ground", "--M-sizing", "default"])
        @test opt.run.ics == (:ground,)
        @test opt.run.n_sizes == 1

        opt = DG.parse_args(["--phase", "simulate", "--tracks", "weak,weak_inverted"])
        @test opt.run.ics == (:weak, :weak_inverted)

        opt = DG.parse_args(["--phase", "simulate", "--tracks", "all"])
        @test opt.run.ics == DG.DATAGEN_TRACKS

        opt = DG.parse_args(["--phase", "simulate", "--tracks", "cannon,weak"])
        @test opt.run.ics == (:ground, :equator, :weak)

        opt = DG.parse_args(["--phase", "simulate", "--tracks", "poles"])
        @test opt.run.ics == (:ground, :inverted)

        opt = DG.parse_args(["--phase", "simulate", "--tracks", "precess"])
        @test opt.run.ics == (:weak, :weak_inverted)

        opt = DG.parse_args(["--phase", "simulate", "--tracks", "approx"])
        @test opt.run.ics == (:ground, :weak)

        opt = DG.parse_args(["--phase", "simulate", "--tracks", "inverted"])
        @test opt.run.ics == (:inverted,)

        opt = DG.parse_args(["--phase", "simulate", "--tracks", "weak-inverted"])
        @test opt.run.ics == (:weak_inverted,)

        @test DG.parse_tracks("weak,weak_inverted") == (:weak, :weak_inverted)
        @test DG.parse_tracks("weak, weak") == (:weak,)
        @test DG.parse_tracks("all") == DG.DATAGEN_TRACKS
        @test DG.parse_tracks("cannon") == DG.DATAGEN_ICS
        @test DG.parse_default_conditions("cannon") == DG.DATAGEN_ICS
        @test DG.DATAGEN_TRACKS == (:ground, :inverted, :equator, :weak, :weak_inverted)
        runi = DG.make_run_params(; ics = (:inverted,), M_cap = 8, Nt_save = 11)
        @test runi.ics == (:inverted,)
        runw = DG.make_run_params(; ics = (:weak, :weak_inverted), M_cap = 8, Nt_save = 11)
        @test runw.ics == (:weak, :weak_inverted)

        @test_throws ErrorException DG.parse_args(["--nope"])
        @test_throws ErrorException DG.parse_args(["--phase", "nope"])
        @test_throws ErrorException DG.parse_args(["--phase", "simulate", "--dry-run"])
        @test_throws ErrorException DG.parse_args(["--phase", "simulate", "--stop", "1", "--start", "5"])
        @test_throws ErrorException DG.parse_args(["--phase", "configs", "--limit", "-1"])
        @test_throws ErrorException DG.parse_args(["--phase"])
        @test_throws ErrorException DG.parse_args(["--M-sizing", "0"])
        @test_throws ErrorException DG.parse_args(["--phase", "simulate", "--tracks", "both"])
        @test_throws ErrorException DG.parse_args(["--phase", "simulate", "--tracks", ""])
        @test_throws ErrorException DG.parse_args(["--phase", "simulate", "--NT-save", "1"])
    end

    @testset "fmt / stems / signatures" begin
        @test DG.fmt_plain(0) == "0"
        @test occursin("p", DG.fmt_plain(0.05))
        @test_throws ErrorException DG.fmt_plain(Inf)
        @test_throws ErrorException DG.fmt_plain(NaN)
        @test endswith(DG.fmt_us(10e-6), "us")
        @test endswith(DG.fmt_hz(1e6), "MHz")
        @test endswith(DG.fmt_hz(5e3), "kHz")
        @test endswith(DG.fmt_hz(12.0), "Hz")

        @test DG.sanitize_stem("A B!!c") == "A_B_c"
        used = Set{String}()
        @test DG.uniquify_stem("x", used) == "x"
        @test DG.uniquify_stem("x", used) == "x_2"
        @test DG.uniquify_stem("x", used) == "x_3"

        @test DG.result_signature(:ground, 60000, 1, 5001) == "ground_Md60000_Mg1_Nt5001"
        @test DG.result_signature(:inverted, 8, 1, 11) == "inverted_Md8_Mg1_Nt11"
        @test DG.result_signature(:equator, 8, 2, 11) == "equator_Md8_Mg2_Nt11"
        @test DG.result_signature(:weak, 8, 1, 11) == "weak_Md8_Mg1_Nt11"
        @test DG.result_signature(:weak_inverted, 8, 2, 11) == "weak_inverted_Md8_Mg2_Nt11"
        @test_throws ErrorException DG.result_signature(:custom, 8, 1, 11)
        @test_throws ErrorException DG.result_signature(:ground, 0, 1, 11)

        path, key = DG.result_target("stem", :ground, 8, 1, 11)
        @test key == "ground_Md8_Mg1_Nt11"
        @test endswith(path, "stem_ground_Md8_Mg1_Nt11.jld2")
        pathi, keyi = DG.result_target("stem", :inverted, 8, 1, 11)
        @test keyi == "inverted_Md8_Mg1_Nt11"
        @test endswith(pathi, "stem_inverted_Md8_Mg1_Nt11.jld2")
        pathw, keyw = DG.result_target("stem", :weak, 8, 1, 11)
        @test keyw == "weak_Md8_Mg1_Nt11"
        @test endswith(pathw, "stem_weak_Md8_Mg1_Nt11.jld2")
        pathwi, keywi = DG.result_target("stem", :weak_inverted, 8, 1, 11)
        @test keywi == "weak_inverted_Md8_Mg1_Nt11"
        @test endswith(pathwi, "stem_weak_inverted_Md8_Mg1_Nt11.jld2")
        @test DG.pulsemat_from_result(path) == replace(path, ".jld2" => "_pulsemat.csv")
        @test DG.stem_from_simulconfig_path("/tmp/abc_simulconfig.jld2") == "abc"
        @test_throws ErrorException DG.stem_from_simulconfig_path("nope.jld2")

        sys, spec = canonical_rase_spec()
        stem = DG.datagen_stem(sys, spec)
        @test !isempty(stem)
        @test occursin(r"^[A-Za-z0-9_]+$", stem)
        @test ncodeunits(stem) <= DG.DATAGEN_STEM_MAXLEN
        sig = DG.result_signature(:ground, DG.RULE_M_CAP, 1, DG.RULE_NT_SAVE)
        @test ncodeunits("$(stem)_$(sig).jld2") < 255
        @test ncodeunits("$(stem)_$(sig)_pulsemat.csv") < 255
        sigw = DG.result_signature(:weak_inverted, DG.RULE_M_CAP, 1, DG.RULE_NT_SAVE)
        @test ncodeunits("$(stem)_$(sigw).jld2") < 255
        @test ncodeunits("$(stem)_$(sigw)_pulsemat.csv") < 255

        sys = DG.canonical_system()
        sig_amps = String[]
        for d in DG.arp3_designs()
            d.with_signal || continue
            spec = DG.bind_pulse(d, sys)
            push!(sig_amps, DG.pulse_slug(spec))
        end
        @test length(unique(sig_amps)) == length(sig_amps)
        paper = first(d for d in DG.rose_designs() if hasproperty(d, :t_center1))
        @test occursin("c1", DG.pulse_slug(DG.bind_pulse(paper, sys)))
    end

    @testset "skip completeness" begin
        mktempdir() do dir
            jld = joinpath(dir, "a.jld2")
            csv = DG.pulsemat_from_result(jld)
            @test !DG.result_is_complete(jld)
            write(jld, "x")
            @test !DG.result_is_complete(jld)
            write(csv, "")
            @test !DG.result_is_complete(jld)
            write(csv, "Re,Im\n")
            @test DG.result_is_complete(jld)
        end
    end

    @testset "atomic save / manifest" begin
        mktempdir() do dir
            jld = joinpath(dir, "out.jld2")
            data = (
                SIM_SETTING = (Ttotal = 1e-6, Nt_save = 5),
                payload = 42,
            )
            E_of_t = t -> 0.0 + 0.0im
            DG.save_datagen_result(jld, data, E_of_t)
            @test DG.result_is_complete(jld)
            @test !isfile(jld * ".part")
            @test !isfile(DG.pulsemat_from_result(jld) * ".part")
            raw = JLD2.load(jld)
            @test raw["data"].payload == 42

            manpath = joinpath(dir, "manifest.json")
            @test isempty(DG.load_manifest(manpath))
            DG.save_manifest(Dict("s" => Dict("status" => "ok")), manpath)
            man = DG.load_manifest(manpath)
            @test man["s"]["status"] == "ok"
            @test !isfile(manpath * ".part")

            write(manpath, "{not json")
            man2 = DG.load_manifest(manpath)
            @test man2 isa Dict
            @test isempty(man2)

            cfg = joinpath(dir, "stem_simulconfig.jld2")
            sys, spec = canonical_rase_spec()
            DG.save_simulconfig(cfg, sys, spec)
            @test !isfile(cfg * ".part")
            loaded = DG.load_simulconfig(cfg)
            @test loaded.SYSTEM_CONFIG.C_ens == sys.C_ens
            @test loaded.PULSE_SPEC.family === spec.family
        end
    end

    @testset "interrupt unwrap" begin
        @test DG.is_interrupt(InterruptException())
        @test !DG.is_interrupt(ErrorException("x"))
        @test DG.unwrap_task_failure(ErrorException("x")) isa ErrorException

        ce = CompositeException([ErrorException("a"), InterruptException()])
        @test DG.is_interrupt(ce)
        @test DG.unwrap_task_failure(ce) isa InterruptException

        tsk = @task throw(InterruptException())
        schedule(tsk)
        caught = try
            wait(tsk)
            nothing
        catch err
            err
        end
        @test caught !== nothing
        @test DG.is_interrupt(caught)
    end

    @testset "safety splits" begin
        sys = DG.canonical_system()
        @test sys.g_inhomogeneity.kind === :constant
        T = 80e-6
        default = DG.splitting_for_run(sys, T; M_cap = 60000, M_g_max = 30)
        @test default.M_g == 1
        @test default.M_delta == 60000
        @test default.M_total <= 60000
        @test default.safety_factor >= DG.RULE_SAFETY_MIN
        @test default.safety_factor == default.M_delta / default.M_delta_min

        tgts = DG.safety_targets(default.safety_factor, 1)
        @test tgts == [default.safety_factor]
        tgts4 = DG.safety_targets(default.safety_factor, 4)
        @test length(tgts4) == 4
        @test tgts4[end] == default.safety_factor
        @test all(>(DG.RULE_SAFETY_MIN) , tgts4)
        @test issorted(tgts4)
        step = tgts4[2] - tgts4[1]
        @test all(isapprox(tgts4[i + 1] - tgts4[i], step; rtol = 1e-12) for i in 1:3)

        run4 = DG.make_run_params(; ics = (:ground,), M_cap = 60000, n_sizes = 4, Nt_save = 11)
        splits = DG.splits_for_run(sys, T, run4)
        @test !isempty(splits)
        @test splits[end].M_delta == default.M_delta
        seen = Set{Tuple{Int,Int}}()
        for s in splits
            @test s.safety_factor >= DG.RULE_SAFETY_MIN - 1e-12
            @test s.M_total <= 60000
            @test s.M_g == 1
            key = (s.M_delta, s.M_g)
            @test key ∉ seen
            push!(seen, key)
        end

        sysg = DG.system_from_physical(;
            kappa_t_hz = DG.CANONICAL_KAPPA_T_HZ,
            r = DG.CANONICAL_R,
            gamma = DG.CANONICAL_GAMMA,
            C_ens = DG.CANONICAL_C_ENS,
            g_hz = DG.CANONICAL_G_HZ,
            delta0_over_kt = 0.0,
            freq_kind = :lorentzian,
            g_kind = :gaussian,
            eta = 0.05,
        )
        @test DG.admit_system(sysg)
        gs = DG.splitting_for_run(sysg, T; M_cap = 60000, M_g_max = 30)
        @test 1 <= gs.M_g <= 30
        @test gs.M_total <= 60000
        @test gs.safety_factor >= DG.RULE_SAFETY_MIN
        @test gs.M_delta * gs.M_g == gs.M_total

        @test_throws ErrorException DG.compute_optimal_splitting(
            sys.freq_inhomogeneity, sys.g_inhomogeneity, T;
            M_cap = 1, M_g_max = 1, safety_min = 3.0,
        )
        @test_throws ErrorException DG.safety_targets(10.0, 0)
        @test DG.safety_targets(DG.RULE_SAFETY_MIN, 4) == [Float64(DG.RULE_SAFETY_MIN)]

        rng = MersenneTwister(1)
        for _ in 1:200
            Ttry = 10e-6 + 200e-6 * rand(rng)
            cap = rand(rng, (1000, 5000, 20000, 60000))
            mgmax = rand(rng, (1, 4, 15, 30))
            try
                sp = DG.compute_optimal_splitting(
                    sys.freq_inhomogeneity, sys.g_inhomogeneity, Ttry;
                    M_cap = cap, M_g_max = mgmax,
                )
                @test sp.M_g == 1
                @test sp.M_total <= cap
                @test sp.safety_factor >= DG.RULE_SAFETY_MIN - 1e-12
                @test isfinite(sp.M_delta_min) && sp.M_delta_min > 0
            catch err
                DG.rethrow_interrupt(err)
                @test occursin("cannot meet safety", sprint(showerror, err)) ||
                    occursin("M_delta_min", sprint(showerror, err))
            end
            try
                spg = DG.compute_optimal_splitting(
                    sysg.freq_inhomogeneity, sysg.g_inhomogeneity, Ttry;
                    M_cap = cap, M_g_max = mgmax,
                )
                @test 1 <= spg.M_g <= mgmax
                @test spg.M_total <= cap
                @test spg.safety_factor >= DG.RULE_SAFETY_MIN - 1e-12
            catch err
                DG.rethrow_interrupt(err)
                @test occursin("cannot meet safety", sprint(showerror, err)) ||
                    occursin("M_delta_min", sprint(showerror, err))
            end
        end
    end

    @testset "Ttotal / pulse bind" begin
        sys, spec = canonical_rase_spec()
        Ttotal = DG.derive_ttotal(sys, spec)
        @test isfinite(Ttotal) && Ttotal > 0
        t0, t1 = DG.pulse_drive_span(spec.segments)
        @test t0 >= -1e-15
        @test Ttotal > t1

        pc = DG.materialize_pulse_config(spec)
        ok, msg = DG.pulse_config_is_valid(pc)
        @test ok
        @test isempty(msg)
        @test_throws ErrorException DG.pulse_drive_span(())

        designs = DG.all_pulse_designs()
        n_canon = 0
        families = Set{Symbol}()
        for d in designs
            fam = DG.pulse_family_of(d)
            push!(families, fam)
            d.canonical || continue
            n_canon += 1
            specd = DG.bind_pulse(d, sys)
            Td = DG.derive_ttotal(sys, specd)
            @test isfinite(Td) && Td > 0
            okd, msgd = DG.pulse_config_is_valid(DG.materialize_pulse_config(specd))
            @test okd
            @test isempty(msgd)
        end
        @test n_canon >= 8
        @test :rase_wurst in families
        @test :rose in families
        @test :arp3 in families
    end

    @testset "job pool / recovery" begin
        empty_out = DG.run_datagen_jobs!(Any[])
        @test isempty(empty_out)

        outcomes = Vector{Any}(undef, 3)
        jobs = [fake_job(i) for i in 1:3]
        DG._fill_unassigned_outcomes!(outcomes, jobs, "never started")
        @test all(i -> isassigned(outcomes, i), 1:3)
        @test all(st -> st.status == "failed", outcomes)
        @test occursin("never started", outcomes[2].error)

        n = 8
        jobs = [fake_job(i; key = i in (3, 7) ? "fail$i" : "k$i") for i in 1:n]
        function fake_exec(job, compute, tag)
            startswith(job.key, "fail") && return DG._job_outcome_failed(job, tag, "boom $(job.key)")
            return DG._job_outcome_ok(job, tag, 0.01)
        end
        done = Ref(0)
        function on_done(job, st)
            done[] += 1
            @test st.stem == job.stem
        end
        out = DG.run_datagen_jobs!(jobs; executor = fake_exec, on_complete = on_done)
        @test length(out) == n
        @test all(i -> isassigned(out, i), 1:n)
        @test count(st -> st.status == "ok", out) == 6
        @test count(st -> st.status == "failed", out) == 2
        @test done[] == n

        reports = Dict{String, Any}()
        n_ok, n_failed = DG.merge_job_outcomes!(reports, out)
        @test n_ok == 6
        @test n_failed == 2
        cok, cfailed = DG.count_job_report_statuses(reports)
        @test cok == 6
        @test cfailed == 2

        jobs_int = [fake_job(i) for i in 1:5]
        function exec_int(job, compute, tag)
            job.key == "k2" && throw(InterruptException())
            return DG._job_outcome_ok(job, tag, 0.0)
        end
        caught = try
            DG.run_datagen_jobs!(jobs_int; executor = exec_int)
            nothing
        catch err
            err
        end
        @test caught !== nothing
        @test DG.is_interrupt(caught)

        jobs_boom = [fake_job(i; key = i == 4 ? "throw4" : "k$i") for i in 1:6]
        function exec_throw(job, compute, tag)
            job.key == "throw4" && error("worker-visible boom")
            return DG._job_outcome_ok(job, tag, 0.0)
        end
        out_boom = DG.run_datagen_jobs!(jobs_boom; executor = exec_throw)
        @test length(out_boom) == 6
        @test all(i -> isassigned(out_boom, i), 1:6)
        @test count(st -> st.status == "ok", out_boom) == 5
        @test count(st -> st.status == "failed", out_boom) == 1
        @test occursin("boom", out_boom[4].error)
    end

    @testset "GPU reclaim" begin
        @test DG.datagen_gpu_count() == length(DG.datagen_cuda_devices())
        DG.datagen_reclaim_current_gpu!()
        DG.datagen_reclaim_all_gpus!(Any[])
        devices = DG.datagen_cuda_devices()
        DG.datagen_reclaim_all_gpus!(devices)
        desc = DG.describe_datagen_compute(3; n_gpu = length(devices))
        @test !isempty(desc)
        if isempty(devices)
            @test occursin("CPU", desc)
        else
            @test occursin("GPU", desc)
            @test CUDA.functional()
        end
    end

    @testset "system catalog gate" begin
        systems = DG.enumerate_system_catalog()
        @test length(systems) >= 100
        seen = Set{Any}()
        for sys in systems
            @test DG.admit_system(sys)
            k = DG.system_key(sys)
            @test k ∉ seen
            push!(seen, k)
            @test sys.kappa_e > 0
            @test sys.C_ens > 0
        end
        @test DG.canonical_system() in systems || any(
            s -> DG.system_key(s) == DG.system_key(DG.canonical_system()), systems
        )
        bad = DG.system_from_physical(;
            kappa_t_hz = 1e6, r = 1.0, gamma = 1.0, C_ens = 0.6, g_hz = 1e6,
            delta0_over_kt = 0.0, freq_kind = :lorentzian, g_kind = :constant, eta = 0.0,
        )
        @test !DG.admit_system(bad)
    end

    if !QUICK
        @testset "full pairing stress" begin
            systems = DG.enumerate_system_catalog()
            designs = DG.all_pulse_designs()
            pairs, n_reject = DG.enumerate_pairs(systems, designs)
            @test n_reject == 0
            @test length(pairs) >= 1000
            used = Set{String}()
            maxlen = 0
            for p in pairs
                stem = DG.uniquify_stem(DG.datagen_stem(p.SYSTEM_CONFIG, p.PULSE_SPEC), used)
                maxlen = max(maxlen, ncodeunits(stem))
                Ttotal = DG.derive_ttotal(p.SYSTEM_CONFIG, p.PULSE_SPEC)
                @test isfinite(Ttotal) && Ttotal > 0
            end
            @test maxlen < 180
            @test length(used) == length(pairs)
            println("  pairing: $(length(pairs)) pairs, 0 bind rejects, max stem $maxlen bytes")
        end
    end

    if !SKIP_ODE
        @testset "tiny live trajectory" begin
            mktempdir() do dir
                sys, spec = canonical_rase_spec()
                Ttotal = DG.derive_ttotal(sys, spec)
                run = DG.make_run_params(;
                    ics = (:ground, :inverted, :weak, :weak_inverted),
                    M_cap = 700,
                    M_g_max = 1,
                    n_sizes = 1,
                    Nt_save = 11,
                )
                splits = DG.splits_for_run(sys, Ttotal, run)
                @test length(splits) == 1
                @test splits[1].safety_factor >= DG.RULE_SAFETY_MIN
                compute = DG.datagen_gpu_count() > 0 ? :gpu : :cpu
                println("  tiny ODE compute=$compute  M=$(splits[1].M_total)  Nt=11")
                for ic in run.ics
                    out = joinpath(dir, "tiny_$(ic)_Md$(splits[1].M_delta)_Mg1_Nt11.jld2")
                    elapsed = DG.run_one_ic(
                        sys, spec, ic, out, splits[1], run, Ttotal;
                        compute = compute,
                    )
                    @test elapsed > 0
                    @test DG.result_is_complete(out)
                    data = JLD2.load(out, "data")
                    @test data.SIM_SETTING.initial_condition === ic
                    DG.datagen_reclaim_current_gpu!()
                end
                DG.datagen_reclaim_all_gpus!(DG.datagen_cuda_devices())
            end
        end
    end
end

function _nongreen(ts)
    n = 0
    for r in ts.results
        if r isa Union{Test.Fail, Test.Error}
            n += 1
        elseif r isa Test.DefaultTestSet
            n += _nongreen(r)
        end
    end
    return n
end

n_fail = _nongreen(ts)
println()
if n_fail == 0
    println("datagen selftest: ALL PASSED")
else
    println("datagen selftest: $n_fail failure(s)")
end
exit(n_fail == 0 ? 0 : 1)

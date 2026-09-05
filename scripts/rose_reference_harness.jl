
using InhomogeneousSpinCavityDynamics
using JLD2
using Printf


_env_int(k, d)   = haskey(ENV, k) ? parse(Int, ENV[k])     : d
_env_float(k, d) = haskey(ENV, k) ? parse(Float64, ENV[k]) : d

function frozen_case(; outdir)
    M_delta   = _env_int("ROSE_REF_M_DELTA", 101)
    M_g       = _env_int("ROSE_REF_M_G", 101)
    Nt_save   = _env_int("ROSE_REF_NT_SAVE", 2001)
    Ttotal_us = _env_float("ROSE_REF_TTOTAL_US", 1100.0)

    SYSTEM_CONFIG = (
        C_ens   = 0.6,
        delta0  = 0.0,
        kappa_e = 2π * 1e6,
        kappa_i = 2π * 0.0,
        freq_inhomogeneity = (
            kind        = :lorentzian,
            FWHM        = 2π * 1e6,
            span_gamma  = 2.5,
            renormalize = false,
        ),
        g_inhomogeneity = (
            kind       = :gaussian,
            mean       = 2π * 100.0,
            std        = 2π * 12.0,
            span_sigma = 2.5,
            renormalize = false,
        ),
    )

    SIM_SETTING = (
        simulation_order  = :order1,
        M_delta           = M_delta,
        M_g               = M_g,



        ensemble_method   = :histogram,
        initial_condition = :ground,
        Ttotal            = Ttotal_us * 1e-6,
        Nt_save           = Nt_save,
        reltol            = 1e-8,
        abstol            = 1e-8,
        saved_file_name   = joinpath(outdir, "rose_reference_run.jld2"),
        peak_detection = (
            labels      = [:echo1, :echo2],
            times       = [550e-6, 1050e-6],
            half_window = 10e-6,
        ),
    )

    PULSE_CONFIG = (
        (
            name  = "Gaussian input signal",
            kind  = :gaussian,
            t0    = 50e-6,
            sigma = 3e-6,
            amp   = 0.5 * sqrt(SYSTEM_CONFIG.kappa_e) * 0.332,
            omega = 0.0,
            phase = 0.0,
        ),
        (
            name       = "First WURST pulse",
            kind       = :wurst,
            t_center   = 300e-6,
            duration   = 400e-6,
            amp        = 0.5 * sqrt(SYSTEM_CONFIG.kappa_e) * 2.0e4,
            bandwidth  = 5.0 * SYSTEM_CONFIG.freq_inhomogeneity.FWHM,
            n          = 20.0,
            omega0     = 0.0,
            chirp_sign = +1.0,
            phase0     = 0.0,
            edge_frac  = 1e-4,
        ),
        (
            name       = "Second WURST pulse",
            kind       = :wurst,
            t_center   = 800e-6,
            duration   = 400e-6,
            amp        = 0.5 * sqrt(SYSTEM_CONFIG.kappa_e) * 2.0e4,
            bandwidth  = 5.0 * SYSTEM_CONFIG.freq_inhomogeneity.FWHM,
            n          = 20.0,
            omega0     = 0.0,
            chirp_sign = +1.0,
            phase0     = 0.0,
            edge_frac  = 1e-4,
        ),
    )

    return SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG
end


function run_case()
    outdir = mktempdir()
    SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG = frozen_case(; outdir = outdir)
    print_cooperativity_honesty(SYSTEM_CONFIG)
    data = run_sim_1st_order(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG; clean_gpu = true)
    return data
end

function extract_record(data)
    peaks = NamedTuple[]
    if data.peak_detection_results !== nothing
        for r in data.peak_detection_results
            push!(peaks, (
                label              = String(r.label),
                global_peak_index  = Int(r.global_peak_index),
                local_peak_index   = Int(r.local_peak_index),
                detected_time      = Float64(r.detected_time),
                detected_amplitude = Float64(r.detected_amplitude),
                a_out_at_peak      = ComplexF64(r.a_out_at_peak),
                Sp_at_peak_2d      = ComplexF64.(r.Sp_at_peak_2d),
                Sz_at_peak_2d      = ComplexF64.(r.Sz_at_peak_2d),
            ))
        end
    end
    return (
        meta = (
            M_delta = Int(data.M_delta),
            M_g     = Int(data.M_g),
            M_total = Int(data.M_total),
            Nt      = length(data.t_saved),
            julia   = string(VERSION),
        ),
        t_saved    = collect(Float64, data.t_saved),
        a_sol      = collect(ComplexF64, data.a_sol),
        Sigma_p    = collect(ComplexF64, data.Σp_sol),
        Sigma_z    = collect(ComplexF64, data.Σz_sol),
        E_of_t_arr = collect(ComplexF64, data.E_of_t_arr),
        Sp_keep    = ComplexF64.(data.Sp_keep),
        Sz_keep    = ComplexF64.(data.Sz_keep),
        peaks      = peaks,
    )
end


function ulp_distance(a::Float64, b::Float64)
    (isnan(a) || isnan(b)) && return Inf
    a === b && return 0.0
    (signbit(a) != signbit(b)) && return Inf
    ia = reinterpret(Int64, a)
    ib = reinterpret(Int64, b)
    return Float64(abs(ia - ib))
end

_parts(x::Complex) = (real(x), imag(x))
_parts(x::Real)    = (Float64(x),)

function array_stats(new::AbstractArray, ref::AbstractArray)
    size(new) == size(ref) || return (ok = false, shape = true,
        maxabs = Inf, maxrel = Inf, maxulp = Inf, n = length(new))
    maxabs = 0.0; maxrel = 0.0; maxulp = 0.0
    @inbounds for i in eachindex(new, ref)
        d = abs(new[i] - ref[i])
        maxabs = max(maxabs, d)
        denom = max(abs(new[i]), abs(ref[i]))
        denom > 0 && (maxrel = max(maxrel, d / denom))
        pn = _parts(new[i]); pr = _parts(ref[i])
        for k in eachindex(pn, pr)
            maxulp = max(maxulp, ulp_distance(pn[k], pr[k]))
        end
    end
    return (ok = true, shape = false, maxabs = maxabs, maxrel = maxrel,
            maxulp = maxulp, n = length(new))
end

struct FieldResult
    name::String
    pass::Bool
    detail::String
end

function check_array(name, new, ref, tol_ulp)
    s = array_stats(new, ref)
    if s.shape
        return FieldResult(name, false, "SHAPE MISMATCH new=$(size(new)) ref=$(size(ref))")
    end
    pass = s.maxulp <= tol_ulp
    d = @sprintf("maxabs=%.3e  maxrel=%.3e  maxulp=%g  (n=%d)",
                 s.maxabs, s.maxrel, s.maxulp, s.n)
    return FieldResult(name, pass, d)
end

function check_scalar_exact(name, new, ref)
    pass = new == ref
    return FieldResult(name, pass, pass ? "exact ($new)" : "MISMATCH new=$new ref=$ref")
end

function check_reference(path::AbstractString; tol_ulp::Real = 0, quiet::Bool = false)
    isfile(path) || error("reference file not found: $path  (run `save` mode first)")
    ref = JLD2.load(path, "record")
    @info "re-running frozen ROSE case for comparison ..."
    new = extract_record(run_case())

    results = FieldResult[]


    push!(results, check_scalar_exact("meta.M_delta", new.meta.M_delta, ref.meta.M_delta))
    push!(results, check_scalar_exact("meta.M_g",     new.meta.M_g,     ref.meta.M_g))
    push!(results, check_scalar_exact("meta.Nt",      new.meta.Nt,      ref.meta.Nt))
    push!(results, check_array("t_saved",    new.t_saved,    ref.t_saved,    0))
    push!(results, check_array("E_of_t_arr", new.E_of_t_arr, ref.E_of_t_arr, 0))


    push!(results, check_array("a_sol",   new.a_sol,   ref.a_sol,   tol_ulp))
    push!(results, check_array("Sigma_p", new.Sigma_p, ref.Sigma_p, tol_ulp))
    push!(results, check_array("Sigma_z", new.Sigma_z, ref.Sigma_z, tol_ulp))
    push!(results, check_array("Sp_keep", new.Sp_keep, ref.Sp_keep, tol_ulp))
    push!(results, check_array("Sz_keep", new.Sz_keep, ref.Sz_keep, tol_ulp))


    if length(new.peaks) != length(ref.peaks)
        push!(results, FieldResult("peaks.count", false,
            "MISMATCH new=$(length(new.peaks)) ref=$(length(ref.peaks))"))
    else
        for (pn, pr) in zip(new.peaks, ref.peaks)
            tag = pn.label
            push!(results, check_scalar_exact("peak[$tag].global_index",
                pn.global_peak_index, pr.global_peak_index))
            push!(results, check_scalar_exact("peak[$tag].local_index",
                pn.local_peak_index, pr.local_peak_index))
            push!(results, check_array("peak[$tag].detected_time",
                [pn.detected_time], [pr.detected_time], 0))
            push!(results, check_array("peak[$tag].detected_amplitude",
                [pn.detected_amplitude], [pr.detected_amplitude], tol_ulp))
            push!(results, check_array("peak[$tag].a_out_at_peak",
                [pn.a_out_at_peak], [pr.a_out_at_peak], tol_ulp))
            push!(results, check_array("peak[$tag].Sp_at_peak_2d",
                pn.Sp_at_peak_2d, pr.Sp_at_peak_2d, tol_ulp))
            push!(results, check_array("peak[$tag].Sz_at_peak_2d",
                pn.Sz_at_peak_2d, pr.Sz_at_peak_2d, tol_ulp))
        end
    end

    allpass = all(r -> r.pass, results)

    if !quiet
        println()
        println("="^78)
        @printf("%-34s %-6s  %s\n", "FIELD", "STATUS", "DETAIL")
        println("-"^78)
        for r in results
            @printf("%-34s %-6s  %s\n", r.name, r.pass ? "PASS" : "FAIL", r.detail)
        end
        println("-"^78)
        println(allpass ? "RESULT: PASS  (tol_ulp = $tol_ulp)" :
                          "RESULT: FAIL  (tol_ulp = $tol_ulp)")
        println("="^78)
    end

    return allpass
end

function save_reference(path::AbstractString)
    @info "running frozen ROSE case to build reference ..."
    record = extract_record(run_case())
    mkpath(dirname(path))
    jldsave(path; record = record,
            created = string(now_iso()),
            julia = string(VERSION))
    @info "reference written" path record.meta
    return path
end

now_iso() = Base.Libc.strftime("%Y-%m-%dT%H:%M:%S", time())


function main(args)
    mode = isempty(args) ? "" : args[1]
    rest = length(args) >= 2 ? args[2:end] : String[]


    path = "data/rose_reference.jld2"
    tol_ulp = 0.0
    quiet = false
    i = 1
    while i <= length(rest)
        a = rest[i]
        if a == "--tol-ulp"
            tol_ulp = parse(Float64, rest[i + 1]); i += 2
        elseif a == "--quiet"
            quiet = true; i += 1
        elseif startswith(a, "--")
            error("unknown flag $a")
        else
            path = a; i += 1
        end
    end

    if mode == "save"
        save_reference(path)
    elseif mode == "check"
        ok = check_reference(path; tol_ulp = tol_ulp, quiet = quiet)
        exit(ok ? 0 : 1)
    else
        println("""
        usage:
          julia --project=. scripts/rose_reference_harness.jl save  [path.jld2]
          julia --project=. scripts/rose_reference_harness.jl check [path.jld2] [--tol-ulp N] [--quiet]
        """)
        exit(2)
    end
end

main(ARGS)

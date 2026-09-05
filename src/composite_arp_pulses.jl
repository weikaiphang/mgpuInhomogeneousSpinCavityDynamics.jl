# ARP drive scale: Ω_rms from √⟨g²⟩, not bare ⟨g⟩ / g_value / g.mean.
# See coupling_rms / arp_amp_scale. Mean-g-only is not the paper path.

function _arp_system_scales(d)
    fi = d.freq_inhomogeneity
    hasproperty(fi, :FWHM) || error(
        "_arp_system_scales: d.freq_inhomogeneity has no `FWHM` field " *
        "(expected per validate_frequency_inhomogeneity)."
    )
    FWHM = Float64(fi.FWHM)

    kappa_e = Float64(d.kappa_e)
    kappa_i = Float64(d.kappa_i)
    kappa_t = kappa_e + kappa_i

    g_rms = coupling_rms(d)

    (FWHM > 0 && kappa_e > 0 && kappa_t > 0 && g_rms > 0) || error(
        "_arp_system_scales: non-positive scale (FWHM=$FWHM, kappa_e=$kappa_e, " *
        "kappa_t=$kappa_t, g_rms=$g_rms)."
    )
    return FWHM, kappa_e, kappa_t, g_rms
end


function _arp_segment_plan(target_silencing::Integer, n_pairs::Integer, Tb::Real)
    Tbf = Float64(Tb)

    if target_silencing == 0
        return [Tbf], ["1ARP (single slow chirp, target F ≈ 0)"], Tbf, Tbf, 0
    end

    n = Int(n_pairs)
    n_seg = 2n + 1

    dur_odd = Tbf / (2 * (n + 1))
    dur_even = dur_odd * (n + 1) / n

    durs = [isodd(i) ? dur_odd : dur_even for i in 1:n_seg]
    even_lbl = n == 1 ? "+k/2" : "+($n/$(n + 1))*k"
    labels = ["$(n_seg)ARP segment $i ($(isodd(i) ? "+k" : even_lbl))" for i in 1:n_seg]

    return durs, labels, dur_odd, dur_even, n
end


function generate_2n1_arp_pi_pulse(
    d;
    n_pairs::Integer=1,
    target_silencing::Integer=1,
    bandwidth_fwhm_mult::Real=5.0,
    T_budget::Union{Real,Nothing}=nothing,
    Omega_max::Union{Real,Nothing}=nothing,
    t_start::Real=0.0,
    wurst_n::Real=20.0,
    edge_frac::Real=1e-4,
    signal_E_of_t=_zero_drive,
    compute::Symbol=:cpu,
    reltol::Real=1e-8,
    abstol::Real=1e-8,
    keep_final_state::Bool=false,
)
    target_silencing in (0, 1) || error(
        "target_silencing must be 0 (single ARP, F ≈ 0) or 1 " *
        "((2n+1) composite, F ≈ 1), got $target_silencing."
    )
    n_pairs >= 1 || error("n_pairs must be a positive integer (>= 1), got $n_pairs.")


    T_window = d.timespan[2] - d.timespan[1]
    Tb = T_budget === nothing ? 0.6 * T_window : Float64(T_budget)
    Tb > 0 || error(
        T_budget === nothing ?
        "derived T_budget = 0.6*(d.timespan[2]-d.timespan[1]) = $Tb is not positive; check d.timespan." :
        "T_budget must be positive, got $Tb."
    )


    FWHM, kappa_e, kappa_t, g_rms = _arp_system_scales(d)

    bw = bandwidth_fwhm_mult * FWHM
    bw > 0 || error(
        "bandwidth_fwhm_mult*FWHM must be positive (got bandwidth_fwhm_mult=$bandwidth_fwhm_mult, FWHM=$FWHM)."
    )

    Omega_target = Omega_max === nothing ? pi * FWHM : Float64(Omega_max)
    Omega_target > 0 || error(
        Omega_max === nothing ?
        "default Rabi target pi*FWHM = $Omega_target is not positive; check d.freq_inhomogeneity.FWHM." :
        "Omega_max must be positive, got $Omega_target."
    )
    amp = arp_drive_amplitude(kappa_e, kappa_t, d.g2_avg, Omega_target)
    amp > 0 || error(
        "derived segment amplitude from Ω_rms(√⟨g²⟩) = $amp is not positive; " *
        "check d.kappa_e / d.kappa_i / d.g2_avg."
    )
    println(
        "ARP Ω convention: Ω_rms = 4√⟨g²⟩ √κ_e / κ_t · |E|  ",
        "(g_rms=$(g_rms), g_mean=$(d.g_mean), ⟨g²⟩=$(d.g2_avg); not mean-g)"
    )


    durs, seg_labels, dur_odd, dur_even, n = _arp_segment_plan(target_silencing, n_pairs, Tb)
    n_seg = length(durs)

    t1_start = Float64(t_start)
    starts = t1_start .+ cumsum(durs) .- durs
    t_end = t1_start + sum(durs)

    common = (n=wurst_n, omega0=0.0, chirp_sign=1.0, phase0=0.0, edge_frac=edge_frac)
    PULSE_CONFIG = Tuple(
        merge(
            (name=seg_labels[i], kind=:wurst,
             t_center=starts[i] + durs[i] / 2, duration=durs[i],
             amp=amp, bandwidth=bw),
            common,
        )
        for i in 1:n_seg
    )

    t_end <= d.timespan[2] || @warn(
        "$(n_seg)-segment ARP pulse ends at $(t_end*1e6)us, past d's own simulated window " *
        "end ($(d.timespan[2]*1e6)us) -- the solver will silently truncate it and the " *
        "reported metrics will reflect that truncated pulse. Increase d's own Ttotal, or " *
        "shrink T_budget."
    )


    composite_E_of_t = build_E_of_t(PULSE_CONFIG)
    E_of_t(t) = composite_E_of_t(t) + signal_E_of_t(t)

    T = Float64
    a_final, Sp_dev, Sz_dev = run_sim_1st_order_final(
        E_of_t, d; initial_condition=:weak, compute=compute, reltol=reltol, abstol=abstol,
    )
    Sp_final = collect(Sp_dev)
    Sz_final = collect(Sz_dev)
    inversion = _weighted_inversion(Sz_final, d.g_b, d.Nj, T)
    coherence = _weighted_coherence(Sp_final, d.g_b, d.Nj, d.delta_b, T)
    silencing = _weighted_silencing_factor(Sp_final, d.g_b, d.Nj, d.delta_b, T)
    weak_seed_retention = _weak_seed_retention(Sp_final, d.g_b, d.Nj, d.delta_b, T)

    report = (
        inversion=inversion, coherence=coherence, silencing=silencing,
        weak_seed_retention=weak_seed_retention,
        target_silencing=Int(target_silencing),
        total_duration=t_end - t1_start,
        t_start=t1_start, t_end=t_end,
        bandwidth=bw, duration_odd=dur_odd, duration_even=dur_even,
        amp_odd=amp, amp_even=amp,
        n_pairs=n, total_segments=n_seg,
        g_rms=g_rms, g2_avg=Float64(d.g2_avg), g_mean=Float64(d.g_mean),
        Omega_target=Omega_target, omega_convention=:rms_g2,
    )

    if keep_final_state
        return PULSE_CONFIG, report,
               (t_final=d.timespan[2], a=a_final, Sp=Sp_final, Sz=Sz_final)
    end
    return PULSE_CONFIG, report
end


function generate_3arp_pi_pulse(d; n::Real=20.0, kwargs...)
    Base.depwarn(
        "generate_3arp_pi_pulse is deprecated; call " *
        "generate_2n1_arp_pi_pulse(d; n_pairs=1, target_silencing=1, ...) instead. " *
        "The `n` (WURST exponent) keyword is now `wurst_n`; `report` field names " *
        "changed (duration1/amp1 -> duration_odd/amp_odd, ...); default amplitudes " *
        "are now equal across segments (paper F ≈ 1); metrics come from a single " *
        "`:weak` solve; and an explicit `Omega_max` is a Rabi-frequency target " *
        "rather than a raw drive amplitude.",
        :generate_3arp_pi_pulse,
    )
    return generate_2n1_arp_pi_pulse(d; n_pairs=1, target_silencing=1, wurst_n=n, kwargs...)
end



function _load_run_payload(path::AbstractString)
    isfile(path) || error("generate_2n1_arp_from_jld2: no such file: $path")
    raw = JLD2.load(path)
    if haskey(raw, "data")
        return raw["data"]
    elseif haskey(raw, "SYSTEM_CONFIG")
        return NamedTuple(Symbol(k) => v for (k, v) in raw)
    else
        error(
            "generate_2n1_arp_from_jld2: $path has neither a top-level `data` NamedTuple " *
            "nor a `SYSTEM_CONFIG` key (keys: $(collect(keys(raw)))). Point at a package run file."
        )
    end
end


function _arp_out_path(src::AbstractString, n_seg::Integer, target_silencing::Integer;
                       out_dir::Union{AbstractString,Nothing}=nothing)
    endswith(src, ".jld2") || error("generate_2n1_arp_from_jld2: source must be a .jld2 path, got $src.")
    stem = src[1:end - length(".jld2")]
    name = "$(n_seg)arpcomp$(target_silencing).jld2"
    return out_dir === nothing ? "$(stem)__$(name)" : joinpath(out_dir, "$(basename(stem))__$(name)")
end


function generate_2n1_arp_from_jld2(
    jld2_path::AbstractString;
    target_silencing::Integer=1,
    n_pairs::Integer=1,
    M_delta::Union{Integer,Nothing}=nothing,
    M_g::Union{Integer,Nothing}=nothing,
    out_dir::Union{AbstractString,Nothing}=nothing,
    compute::Symbol=:cpu,
    Ttotal::Union{Real,Nothing}=nothing,
    Nt_save::Union{Integer,Nothing}=nothing,
    bandwidth_fwhm_mult::Real=5.0,
    T_budget::Union{Real,Nothing}=nothing,
    Omega_max::Union{Real,Nothing}=nothing,
    t_start::Real=0.0,
    wurst_n::Real=20.0,
    edge_frac::Real=1e-4,
    reltol::Real=1e-8,
    abstol::Real=1e-8,
)
    payload = _load_run_payload(jld2_path)
    hasproperty(payload, :SYSTEM_CONFIG) || error(
        "generate_2n1_arp_from_jld2: $jld2_path payload has no SYSTEM_CONFIG."
    )
    sys = payload.SYSTEM_CONFIG
    src_sim = hasproperty(payload, :SIM_SETTING) ? payload.SIM_SETTING : nothing

    _from_src(key, override, default) =
        override !== nothing ? override :
        (src_sim !== nothing && hasproperty(src_sim, key)) ? getproperty(src_sim, key) :
        default === nothing ?
            error("generate_2n1_arp_from_jld2: $jld2_path has no SIM_SETTING.$key; pass $key=...") :
            default

    Ttot = Float64(_from_src(:Ttotal, Ttotal, nothing))
    Ttot > 0 || error("generate_2n1_arp_from_jld2: Ttotal must be positive, got $Ttot.")
    ntsave = Int(_from_src(:Nt_save, Nt_save, 5001))
    Md = Int(_from_src(:M_delta, M_delta, nothing))
    Mg = Int(_from_src(:M_g, M_g, nothing))
    (Md > 0 && Mg > 0) || error(
        "generate_2n1_arp_from_jld2: M_delta=$Md and M_g=$Mg must both be positive."
    )


    n_seg = target_silencing == 0 ? 1 : (2 * Int(n_pairs) + 1)
    outpath = _arp_out_path(String(jld2_path), n_seg, Int(target_silencing); out_dir=out_dir)
    SIM_SETTING = (
        simulation_order = :order1,
        M_delta = Md,
        M_g = Mg,
        initial_condition = :weak,
        Ttotal = Ttot,
        Nt_save = ntsave,
        reltol = reltol,
        abstol = abstol,
        saved_file_name = outpath,
    )

    CONFIG = build_full_config(SIM_SETTING, sys)
    validate_config(CONFIG)
    d = prepare_derived(CONFIG)
    d.M == Md * Mg || error(
        "generate_2n1_arp_from_jld2: derived M=$(d.M) != M_delta*M_g=$(Md * Mg)."
    )


    PULSE_CONFIG, report, fs = generate_2n1_arp_pi_pulse(
        d;
        n_pairs=n_pairs, target_silencing=target_silencing,
        bandwidth_fwhm_mult=bandwidth_fwhm_mult, T_budget=T_budget, Omega_max=Omega_max,
        t_start=t_start, wurst_n=wurst_n, edge_frac=edge_frac,
        compute=compute, reltol=reltol, abstol=abstol,
        keep_final_state=true,
    )
    validate_pulse_config(PULSE_CONFIG)
    report.total_segments == n_seg || error(
        "generate_2n1_arp_from_jld2: n_seg mismatch (planned $n_seg, got $(report.total_segments))."
    )

    data = (
        SIM_SETTING = SIM_SETTING,
        SYSTEM_CONFIG = sys,
        PULSE_CONFIG = PULSE_CONFIG,
        initial_condition = :weak,


        t_final = fs.t_final,
        a_final = fs.a,
        Sp_final = fs.Sp,
        Sz_final = fs.Sz,
        Sigma_p_final = sum(fs.Sp),
        Sigma_z_final = sum(fs.Sz),


        inversion = report.inversion,
        coherence = report.coherence,
        silencing = report.silencing,
        weak_seed_retention = report.weak_seed_retention,
        arp_report = report,


        target_silencing = Int(target_silencing),
        n_pairs = report.n_pairs,
        total_segments = report.total_segments,
        M_total = d.M,
        source_jld2 = abspath(String(jld2_path)),
    )

    dir = dirname(outpath)
    isempty(dir) || mkpath(dir)
    JLD2.jldsave(outpath; data = data)
    return outpath
end

# ============================================================
# Paths, run-rule constants, manifest, and filename stems.
# ============================================================

const DATAGEN_ROOT = normpath(joinpath(@__DIR__, "..", "..", "data", "datagen"))
const DATAGEN_CONFIG_DIR = joinpath(DATAGEN_ROOT, "configs")
const DATAGEN_RESULT_DIR = joinpath(DATAGEN_ROOT, "results")
const DATAGEN_MANIFEST = joinpath(DATAGEN_ROOT, "manifest.json")

const RUN_RULES_VERSION = "6"
const DATAGEN_STEM_MAXLEN = 180
const TWO_PI = 2 * pi

const RULE_SIMULATION_ORDER = :order1
const RULE_NT_SAVE = 5001
const RULE_RELTOL = 1e-8
const RULE_ABSTOL = 1e-8
const RULE_M_CAP = 60000
const RULE_M_G_MAX = 30
const RULE_SAFETY_MIN = 3.0

# Allowed simulate-time ICs. These are pulse_optimizer2.jl's named
# initial conditions (`build_u0_1st_order_cpu`), not the cost-mode
# `track` (`:weak` default / `:dual` opt-in). Default is cannon = ground+equator.
const DATAGEN_TRACKS = (:ground, :inverted, :equator, :weak, :weak_inverted)
const DATAGEN_ICS = (:ground, :equator)
const DATAGEN_TRACK_POLES = (:ground, :inverted)
const DATAGEN_TRACK_PRECESS = (:weak, :weak_inverted)
const DATAGEN_TRACK_CANNON = DATAGEN_ICS
const DATAGEN_TRACK_APPROX = (:ground, :weak)

function is_datagen_track(ic::Symbol)
    return ic in DATAGEN_TRACKS
end

function ensure_datagen_dirs()
    mkpath(DATAGEN_CONFIG_DIR)
    mkpath(DATAGEN_RESULT_DIR)
    return nothing
end

"""
Unwrap `TaskFailedException` / `CompositeException` so Ctrl-C from a
`Threads.@spawn` worker is visible as `InterruptException`. Prefers an
interrupt if several tasks failed. Cycle-safe.
"""
function unwrap_task_failure(err)
    seen = IdDict{Any, Nothing}()
    while true
        haskey(seen, err) && return err
        seen[err] = nothing
        if err isa TaskFailedException
            ex = try
                err.task.exception
            catch
                return err
            end
            ex === nothing && return err
            err = ex
        elseif err isa CompositeException && !isempty(err.exceptions)
            for e in err.exceptions
                u = unwrap_task_failure(e)
                u isa InterruptException && return u
            end
            err = err.exceptions[1]
        else
            return err
        end
    end
end

function is_interrupt(err)
    return unwrap_task_failure(err) isa InterruptException
end

function rethrow_interrupt(err)
    u = unwrap_task_failure(err)
    u isa InterruptException && throw(u)
    return err
end

function json_get(obj, key::AbstractString)
    obj isa AbstractDict || return nothing
    v = get(obj, key, nothing)
    v === nothing || return v
    return get(obj, Symbol(key), nothing)
end

# FNV-1a 64: process-stable (unlike hash()), no extra dependency.
function fnv1a64(s::AbstractString)
    h = UInt64(0xcbf29ce484222325)
    for b in codeunits(s)
        h ⊻= UInt64(b)
        h *= UInt64(0x100000001b3)
    end
    return h
end

function stable_stem_tag(s::AbstractString)
    return lpad(string(fnv1a64(s); base=16), 16, '0')[1:8]
end

# Filename tokens match data/data_1st_order: 0p05, 1em06, units glued on.
function fmt_plain(x::Real)
    x = Float64(x)
    isfinite(x) || error("fmt_plain requires a finite number, got $x.")
    iszero(x) && return "0"
    ax = abs(x)
    signc = x < 0 ? "m" : ""
    if ax < 1e-3 || ax >= 1e4
        e = floor(Int, log10(ax))
        m = ax / 10.0^e
        ms = replace(@sprintf("%g", m), "." => "p")
        esign = e < 0 ? "m" : ""
        return signc * ms * "e" * esign * lpad(string(abs(e)), 2, '0')
    end
    return signc * replace(@sprintf("%g", ax), "." => "p")
end

function fmt_us(t::Real)
    return fmt_plain(t * 1e6) * "us"
end

function fmt_hz(hz::Real)
    a = abs(Float64(hz))
    if a >= 1e5
        return fmt_plain(hz / 1e6) * "MHz"
    elseif a >= 1e3
        return fmt_plain(hz / 1e3) * "kHz"
    else
        return fmt_plain(hz) * "Hz"
    end
end

function fmt_hz_from_rad(omega::Real)
    return fmt_hz(omega / TWO_PI)
end

function sanitize_stem(s::AbstractString)
    s = replace(s, r"[^A-Za-z0-9_]+" => "_")
    s = replace(s, r"_+" => "_")
    return string(strip(s, '_'))
end

function pulse_slug(spec)
    fam = spec.family
    d = spec.design
    if fam === :rase_wurst
        s = "RASE_t$(fmt_us(d.t_start))_dur$(fmt_us(d.duration))_amp$(fmt_plain(d.amp_mult))_bw$(fmt_plain(d.bw_fwhm_mult))"
        d.n == 20.0 || (s *= "_n$(fmt_plain(d.n))")
        d.chirp_sign < 0 && (s *= "_chM")
        d.omega0_over_fwhm != 0 && (s *= "_om$(fmt_plain(d.omega0_over_fwhm))")
        d.phase0 != 0 && (s *= "_ph$(fmt_plain(d.phase0))")
        return s
    elseif fam === :rose
        if hasproperty(d, :t_center1)
            return "ROSE_t0$(fmt_us(d.t0))_sig$(fmt_us(d.sigma))_w$(fmt_us(d.wurst_duration))_c1$(fmt_us(d.t_center1))_c2$(fmt_us(d.t_center2))"
        end
        s = "ROSE_t0$(fmt_us(d.t0))_sig$(fmt_us(d.sigma))_w$(fmt_us(d.wurst_duration))_g1$(fmt_us(d.gap_after_signal))_g2$(fmt_us(d.gap_between))"
        d.signal_amp_mult != 1.0 && (s *= "_samp$(fmt_plain(d.signal_amp_mult))")
        d.wurst_amp_mult != 1.0 && (s *= "_wamp$(fmt_plain(d.wurst_amp_mult))")
        d.bw_fwhm_mult != 5.0 && (s *= "_bw$(fmt_plain(d.bw_fwhm_mult))")
        d.n != 20.0 && (s *= "_n$(fmt_plain(d.n))")
        d.chirp_sign < 0 && (s *= "_chM")
        return s
    elseif fam === :arp3 || fam === :arp3_signal
        prefix = fam === :arp3_signal ? "3ARPsig" : "3ARP"
        s = "$(prefix)_T$(fmt_us(d.T_budget))_t$(fmt_us(d.t_start))_bw$(fmt_plain(d.bw_fwhm_mult))_om$(fmt_plain(d.omega_mult))"
        if fam === :arp3_signal
            d.signal_amp_mult != 1.0 && (s *= "_samp$(fmt_plain(d.signal_amp_mult))")
            hasproperty(d, :signal_t0) && d.signal_t0 != 15e-6 && (s *= "_st$(fmt_us(d.signal_t0))")
            hasproperty(d, :signal_sigma) && d.signal_sigma != 3e-6 && (s *= "_ss$(fmt_us(d.signal_sigma))")
        end
        return s
    elseif fam in (:hs1, :corpse, :bb1)
        tag = fam === :hs1 ? "HS1" : fam === :corpse ? "CORPSE" : "BB1"
        s = "$(tag)_T$(fmt_us(d.T_max))_om$(fmt_plain(d.omega_mult))"
        hasproperty(d, :beta) && d.beta !== nothing && (s *= "_beta$(fmt_plain(d.beta))")
        hasproperty(d, :mu) && d.mu !== nothing && (s *= "_mu$(fmt_plain(d.mu))")
        hasproperty(d, :taper_frac) && d.taper_frac != 0.1 && (s *= "_tap$(fmt_plain(d.taper_frac))")
        return s
    elseif fam === :block_pi
        s = "blockpi_t$(fmt_us(d.t_start))_dur$(fmt_us(d.duration))_om$(fmt_plain(d.omega_mult))"
        hasproperty(d, :taper_frac) && d.taper_frac != 0.1 && (s *= "_tap$(fmt_plain(d.taper_frac))")
        return s
    elseif fam === :random_composite
        return "rndcomp_k$(d.k)_n$(d.n_coeff)_seed$(d.seed)_T$(fmt_us(d.T_max))"
    else
        return String(fam)
    end
end

function system_slug(sys)
    fi = sys.freq_inhomogeneity
    gi = sys.g_inhomogeneity
    kappa_t = kappa_t_of(sys)
    freq = fi.kind === :lorentzian ? "lor" : "gau"
    s = "C$(fmt_plain(sys.C_ens))_$(freq)_FWHM$(fmt_hz_from_rad(fi.FWHM))"
    if gi.kind === :constant
        s *= "_g$(fmt_hz_from_rad(gi.g_value))"
    else
        s *= "_g$(fmt_hz_from_rad(gi.mean))_gstd_$(fmt_hz_from_rad(gi.std))"
    end
    s *= "_ke$(fmt_hz_from_rad(sys.kappa_e))"
    sys.kappa_i > 0 && (s *= "_ki$(fmt_hz_from_rad(sys.kappa_i))")
    abs(sys.delta0) > 1e-12 && (s *= "_d0$(fmt_plain(sys.delta0 / kappa_t))kt")
    return s
end

function datagen_stem(sys, spec)
    stem = sanitize_stem(system_slug(sys) * "_" * pulse_slug(spec))
    if ncodeunits(stem) > DATAGEN_STEM_MAXLEN
        stem = sanitize_stem(stem[1:DATAGEN_STEM_MAXLEN] * "_" * stable_stem_tag(stem))
    end
    return stem
end

function uniquify_stem(base::AbstractString, used::Set{String})
    stem = base
    k = 2
    while stem in used
        stem = base * "_" * string(k)
        k += 1
    end
    push!(used, stem)
    return stem
end

function simulconfig_path(stem::AbstractString, dir::AbstractString=DATAGEN_CONFIG_DIR)
    return joinpath(dir, stem * "_simulconfig.jld2")
end

function result_signature(ic::Symbol, M_delta, M_g, Nt_save)
    is_datagen_track(ic) || error("unknown initial condition $ic.")
    md = Int(M_delta)
    mg = Int(M_g)
    nt = Int(Nt_save)
    md >= 1 && mg >= 1 && nt > 1 || error(
        "invalid result signature ic=$ic Md=$md Mg=$mg Nt=$nt."
    )
    return "$(ic)_Md$(md)_Mg$(mg)_Nt$(nt)"
end

function result_target(stem::AbstractString, ic::Symbol, M_delta, M_g, Nt_save)
    sig = result_signature(ic, M_delta, M_g, Nt_save)
    return joinpath(DATAGEN_RESULT_DIR, "$(stem)_$(sig).jld2"), sig
end

function pulsemat_from_result(filename::AbstractString)
    endswith(filename, ".jld2") || error("result filename must end with .jld2, got $filename")
    return filename[1:end-length(".jld2")] * "_pulsemat.csv"
end

function _nonzero_file(path::AbstractString)
    return isfile(path) && filesize(path) > 0
end

function result_is_complete(filename::AbstractString)
    return _nonzero_file(filename) && _nonzero_file(pulsemat_from_result(filename))
end

function stem_from_simulconfig_path(path)
    fname = basename(path)
    endswith(fname, "_simulconfig.jld2") || error("Not a simulconfig filename: $fname")
    return fname[1:end-length("_simulconfig.jld2")]
end

function load_manifest(path::AbstractString=DATAGEN_MANIFEST)
    isfile(path) || return Dict{String, Any}()
    try
        return Dict{String, Any}(
            String(k) => v
            for (k, v) in JSON3.read(read(path, String))
        )
    catch err
        rethrow_interrupt(err)
        @warn "manifest unreadable; continuing with empty manifest" path=path exception=err
        return Dict{String, Any}()
    end
end

function manifest_run_rules_version(entry)
    v = json_get(entry, "run_rules_version")
    return v === nothing ? nothing : string(v)
end

function save_manifest(manifest, path::AbstractString=DATAGEN_MANIFEST)
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    tmp = path * ".part"
    try
        open(tmp, "w") do io
            JSON3.write(io, manifest)
        end
        mv(tmp, path; force=true)
    catch
        isfile(tmp) && rm(tmp; force=true)
        rethrow()
    end
    return nothing
end

function save_simulconfig(path, SYSTEM_CONFIG, PULSE_SPEC)
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    tmp = path * ".part"
    try
        isfile(tmp) && rm(tmp; force=true)
        JLD2.jldsave(tmp; SYSTEM_CONFIG = SYSTEM_CONFIG, PULSE_SPEC = PULSE_SPEC)
        mv(tmp, path; force=true)
    catch
        isfile(tmp) && rm(tmp; force=true)
        rethrow()
    end
    return path
end

function _jld_get(raw, name::AbstractString)
    haskey(raw, name) && return raw[name]
    s = Symbol(name)
    haskey(raw, s) && return raw[s]
    return nothing
end

function load_simulconfig(path)
    raw = JLD2.load(path)
    sys = _jld_get(raw, "SYSTEM_CONFIG")
    spec = _jld_get(raw, "PULSE_SPEC")
    (sys === nothing || spec === nothing) && error(
        "$path must contain SYSTEM_CONFIG and PULSE_SPEC."
    )
    return (
        SYSTEM_CONFIG = sys,
        PULSE_SPEC = spec,
    )
end

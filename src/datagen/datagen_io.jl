# ============================================================
# Datagen paths, manifest, and JLD2 helpers.
# ============================================================

const DATAGEN_ROOT = normpath(joinpath(@__DIR__, "..", "..", "data", "datagen"))
const DATAGEN_CONFIG_DIR = joinpath(DATAGEN_ROOT, "configs")
const DATAGEN_RESULT_DIR = joinpath(DATAGEN_ROOT, "results")
const DATAGEN_MANIFEST = joinpath(DATAGEN_ROOT, "manifest.json")

const RUN_RULES_VERSION = "3"
const DATAGEN_STEM_MAXLEN = 180

function ensure_datagen_dirs()
    mkpath(DATAGEN_CONFIG_DIR)
    mkpath(DATAGEN_RESULT_DIR)
    return nothing
end

# Filename tokens match data/data_1st_order: 0p05, 1em06, units glued on.
function fmt_plain(x::Real)
    x = Float64(x)
    if x == 0
        return "0"
    end
    ax = abs(x)
    if ax < 1e-3 || ax >= 1e4
        e = floor(Int, log10(ax))
        m = x / 10.0^e
        ms = replace(@sprintf("%g", m), "." => "p")
        esign = e < 0 ? "m" : ""
        return ms * "e" * esign * lpad(string(abs(e)), 2, '0')
    end
    s = replace(@sprintf("%g", x), "-" => "m")
    return replace(s, "." => "p")
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
    return fmt_hz(omega / (2 * pi))
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
        d.chirp_sign >= 0 || (s *= "_chM")
        d.omega0_over_fwhm == 0 || (s *= "_om$(fmt_plain(d.omega0_over_fwhm))")
        d.phase0 == 0 || (s *= "_ph$(fmt_plain(d.phase0))")
        return s
    elseif fam === :rose
        if hasproperty(d, :t_center1)
            return "ROSE_t0$(fmt_us(d.t0))_sig$(fmt_us(d.sigma))_w$(fmt_us(d.wurst_duration))"
        end
        s = "ROSE_t0$(fmt_us(d.t0))_sig$(fmt_us(d.sigma))_w$(fmt_us(d.wurst_duration))_g1$(fmt_us(d.gap_after_signal))_g2$(fmt_us(d.gap_between))"
        d.signal_amp_mult == 1.0 || (s *= "_samp$(fmt_plain(d.signal_amp_mult))")
        d.wurst_amp_mult == 1.0 || (s *= "_wamp$(fmt_plain(d.wurst_amp_mult))")
        d.bw_fwhm_mult == 5.0 || (s *= "_bw$(fmt_plain(d.bw_fwhm_mult))")
        d.n == 20.0 || (s *= "_n$(fmt_plain(d.n))")
        d.chirp_sign >= 0 || (s *= "_chM")
        return s
    elseif fam === :arp3 || fam === :arp3_signal
        prefix = fam === :arp3_signal ? "3ARPsig" : "3ARP"
        return "$(prefix)_T$(fmt_us(d.T_budget))_t$(fmt_us(d.t_start))_bw$(fmt_plain(d.bw_fwhm_mult))_om$(fmt_plain(d.omega_mult))"
    elseif fam in (:hs1, :corpse, :bb1)
        tag = fam === :hs1 ? "HS1" : fam === :corpse ? "CORPSE" : "BB1"
        return "$(tag)_T$(fmt_us(d.T_max))_om$(fmt_plain(d.omega_mult))"
    elseif fam === :block_pi
        return "blockpi_t$(fmt_us(d.t_start))_dur$(fmt_us(d.duration))_om$(fmt_plain(d.omega_mult))"
    elseif fam === :random_composite
        return "rndcomp_k$(d.k)_n$(d.n_coeff)_seed$(d.seed)_T$(fmt_us(d.T_max))"
    else
        return String(fam)
    end
end

function system_slug(sys)
    C = sys.C_ens
    fi = sys.freq_inhomogeneity
    gi = sys.g_inhomogeneity
    kappa_t = sys.kappa_e + sys.kappa_i
    freq = fi.kind === :lorentzian ? "lor" : "gau"
    s = "C$(fmt_plain(C))_$(freq)_FWHM$(fmt_hz_from_rad(fi.FWHM))"
    if gi.kind === :constant
        s *= "_g$(fmt_hz_from_rad(gi.g_value))"
    else
        s *= "_g$(fmt_hz_from_rad(gi.mean))_gstd_$(fmt_hz_from_rad(gi.std))"
    end
    s *= "_ke$(fmt_hz_from_rad(sys.kappa_e))"
    if sys.kappa_i > 0
        s *= "_ki$(fmt_hz_from_rad(sys.kappa_i))"
    end
    if abs(sys.delta0) > 1e-12
        s *= "_d0$(fmt_plain(sys.delta0 / kappa_t))kt"
    end
    return s
end

function datagen_stem(sys, spec)
    # [system]_[pulse]_simulconfig.jld2  (tokens like data/data_1st_order: 0p05, 1em06Hz)
    stem = sanitize_stem(system_slug(sys) * "_" * pulse_slug(spec))
    if length(stem) > DATAGEN_STEM_MAXLEN
        h = string(hash(stem); base=16)
        stem = stem[1:DATAGEN_STEM_MAXLEN] * "_" * h[1:8]
        stem = sanitize_stem(stem)
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

function simulconfig_path(stem::AbstractString)
    return joinpath(DATAGEN_CONFIG_DIR, stem * "_simulconfig.jld2")
end

function result_path(stem::AbstractString, ic::Symbol)
    return joinpath(DATAGEN_RESULT_DIR, "$(stem)_$(ic).jld2")
end

function pulsemat_path(stem::AbstractString, ic::Symbol)
    return joinpath(DATAGEN_RESULT_DIR, "$(stem)_$(ic)_pulsemat.csv")
end

function stem_from_simulconfig_path(path)
    fname = basename(path)
    endswith(fname, "_simulconfig.jld2") || error(
        "Not a simulconfig filename: $fname"
    )
    return fname[1:end-length("_simulconfig.jld2")]
end

function load_manifest()
    if isfile(DATAGEN_MANIFEST)
        return Dict{String, Any}(
            String(k) => v
            for (k, v) in JSON3.read(read(DATAGEN_MANIFEST, String))
        )
    else
        return Dict{String, Any}()
    end
end

function save_manifest(manifest)
    ensure_datagen_dirs()
    open(DATAGEN_MANIFEST, "w") do io
        JSON3.write(io, manifest)
    end
    return nothing
end

function save_simulconfig(path, SYSTEM_CONFIG, PULSE_SPEC)
    ensure_datagen_dirs()
    JLD2.jldsave(
        path;
        SYSTEM_CONFIG = SYSTEM_CONFIG,
        PULSE_SPEC = PULSE_SPEC,
    )
    return path
end

function load_simulconfig(path)
    raw = JLD2.load(path)
    haskey(raw, "SYSTEM_CONFIG") && haskey(raw, "PULSE_SPEC") || error(
        "$path must contain SYSTEM_CONFIG and PULSE_SPEC."
    )
    return (
        SYSTEM_CONFIG = raw["SYSTEM_CONFIG"],
        PULSE_SPEC = raw["PULSE_SPEC"],
    )
end

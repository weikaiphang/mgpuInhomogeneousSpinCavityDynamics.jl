# ============================================================
# PULSE CONSTRUCTORS
# ============================================================

function gaussian_drive(; t0, sigma, amp, omega=0.0, phase=0.0)
    t0_f    = Float64(t0)
    sigma_f = Float64(sigma)
    amp_c   = ComplexF64(amp)
    omega_f = Float64(omega)
    phase_f = Float64(phase)

    return function (t)
        τ = t - t0_f

        return amp_c *
               exp(-(τ^2) / (2*sigma_f^2)) *
               exp(1im * (omega_f * τ + phase_f))
    end
end

function wurst_drive(;
    t_center,
    duration,
    amp,
    bandwidth,
    n=20.0,
    omega0=0.0,
    chirp_sign=+1.0,
    phase0=0.0,
    edge_frac=1e-4,
)
    t_center_f  = Float64(t_center)
    duration_f  = Float64(duration)
    amp_c       = ComplexF64(amp)
    bandwidth_f = Float64(bandwidth)
    n_f         = Float64(n)
    omega0_f    = Float64(omega0)
    chirp_s     = Float64(chirp_sign)
    phase0_f    = Float64(phase0)
    edge_frac_f = Float64(edge_frac)

    t_start = t_center_f - duration_f/2
    edge    = max(duration_f * edge_frac_f, eps(Float64))

    return function (t)
        τ = t - t_start

        gate = 0.5 * (
            tanh((t - t_start) / edge) -
            tanh((t - (t_start + duration_f)) / edge)
        )

        envelope = amp_c * (
            1 - abs(sin(pi * (τ - duration_f/2) / duration_f))^n_f
        )

        phase = phase0_f +
                (omega0_f - chirp_s * bandwidth_f/2) * τ +
                0.5 * chirp_s * (bandwidth_f / duration_f) * τ^2

        return gate * envelope * exp(1im * phase)
    end
end

function custom_drive(f)
    return t -> ComplexF64(f(t))
end

function build_drive_pulse(cfg)
    if cfg.kind == :gaussian
        return gaussian_drive(
            t0    = cfg.t0,
            sigma = cfg.sigma,
            amp   = cfg.amp,
            omega = cfg.omega,
            phase = cfg.phase,
        )

    elseif cfg.kind == :wurst
        return wurst_drive(
            t_center   = cfg.t_center,
            duration   = cfg.duration,
            amp        = cfg.amp,
            bandwidth  = cfg.bandwidth,
            n          = cfg.n,
            omega0     = cfg.omega0,
            chirp_sign = cfg.chirp_sign,
            phase0     = cfg.phase0,
            edge_frac  = cfg.edge_frac,
        )

    elseif cfg.kind == :custom
        return custom_drive(cfg.f)

    else
        error("Unknown pulse kind: $(cfg.kind)")
    end
end

function build_E_of_t(PULSE_CONFIG)
    drive_pulses = tuple((build_drive_pulse(cfg) for cfg in PULSE_CONFIG)...)

    return function E_of_t(t)
        Et = 0.0 + 0.0im

        @inbounds for pulse in drive_pulses
            Et += pulse(t)
        end

        return Et
    end
end

function sample_E_of_t(E_of_t, t_end, N)
    t_points = range(0.0, t_end; length=N)
    Et = ComplexF64[E_of_t(t) for t in t_points]
    return real.(Et), imag.(Et)
end
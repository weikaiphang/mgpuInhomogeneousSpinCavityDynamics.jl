function validate_config(CONFIG)
    @assert CONFIG.M_delta > 0 "M_delta must be positive."
    @assert CONFIG.M_g > 0 "M_g must be positive."
    @assert CONFIG.Ttotal > 0 "Ttotal must be positive."
    @assert CONFIG.kappa_e >= 0 "kappa_e must be non-negative."
    @assert CONFIG.kappa_i >= 0 "kappa_i must be non-negative."
    @assert CONFIG.Nt_save > 1 "Nt_save must be larger than 1."
    @assert CONFIG.reltol > 0 "reltol must be positive."
    @assert CONFIG.abstol > 0 "abstol must be positive."
    validate_simulation_order(CONFIG)
    validate_frequency_inhomogeneity(CONFIG.freq_inhomogeneity)
    validate_coupling_inhomogeneity(CONFIG.g_inhomogeneity)

    return nothing
end

function validate_pulse_config(PULSE_CONFIG)
    for cfg in PULSE_CONFIG
        @assert hasproperty(cfg, :kind) "Each pulse must have a kind."

        if cfg.kind == :gaussian
            @assert cfg.sigma > 0 "Gaussian sigma must be positive."

        elseif cfg.kind == :wurst
            @assert cfg.duration > 0 "WURST duration must be positive."
            @assert cfg.bandwidth > 0 "WURST bandwidth must be positive."
            @assert cfg.n > 0 "WURST n must be positive."

        elseif cfg.kind == :custom
            @assert hasproperty(cfg, :f) "Custom pulse must contain function f."

        else
            error("Unknown pulse kind: $(cfg.kind)")
        end
    end

    return nothing
end

function validate_simulation_order(CONFIG)
    order = hasproperty(CONFIG, :simulation_order) ? CONFIG.simulation_order : :second_order

    allowed_orders = (:first_order, :order1, :first, 1,
                      :second_order, :order2, :second, 2)

    if !(order in allowed_orders)
        error("Unknown simulation_order = $(order). Use :first_order or :second_order.")
    end

    return nothing
end

function get_initial_condition(CONFIG)
    if hasproperty(CONFIG, :initial_condition)
        return CONFIG.initial_condition
    else
        error("Unknown initial_condition. Use :ground or :inverted.")
        return nothing
    end
end

function get_simulation_order(SIM_SETTING)
    if hasproperty(SIM_SETTING, :simulation_order)
        return SIM_SETTING.simulation_order
    else
        error("Unknown simulation_order. Use :first_order or :second_order.")
        return nothing
    end
end
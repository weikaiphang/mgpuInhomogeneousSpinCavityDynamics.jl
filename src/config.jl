function validate_config(CONFIG)
    @assert CONFIG.M_delta > 0
    @assert CONFIG.M_g > 0
    @assert CONFIG.Ttotal > 0
    @assert CONFIG.kappa_e >= 0
    @assert CONFIG.kappa_i >= 0
    @assert CONFIG.Nt_save > 1
    @assert CONFIG.reltol > 0
    @assert CONFIG.abstol > 0
    validate_simulation_order(CONFIG)
    validate_frequency_inhomogeneity(CONFIG.freq_inhomogeneity)
    validate_coupling_inhomogeneity(CONFIG.g_inhomogeneity)

    return nothing
end

function validate_pulse_config(PULSE_CONFIG)
    for cfg in PULSE_CONFIG
        @assert hasproperty(cfg, :kind)

        if cfg.kind == :gaussian
            @assert cfg.sigma > 0 "Gaussian sigma must be positive."

        elseif cfg.kind == :wurst
            @assert cfg.duration > 0
            @assert cfg.bandwidth > 0
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
        error("Unknown initial_condition. Use :ground, :inverted, :equator, :weak, :weak_inverted, or :custom.")
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
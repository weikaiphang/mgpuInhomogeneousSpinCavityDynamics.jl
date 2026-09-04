# For a FIRST-ORDER run with no `ensemble_method` pinned, default it to :auto
# so the quadrature selector runs (segregated in ensemble_quadrature.jl). Pin
# `ensemble_method = :histogram` in SIM_SETTING to force the original bins.
# Second-order is left untouched (quadrature is validated for 1st-order only).
function _with_default_ensemble_method(SIM_SETTING, order)
    hasproperty(SIM_SETTING, :ensemble_method) && return SIM_SETTING
    order in (:first_order, :order1, :first, 1) || return SIM_SETTING
    return merge(SIM_SETTING, (ensemble_method = :auto,))
end

function run_simulation(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG; clean_gpu=true)
    order = get_simulation_order(SIM_SETTING)
    SIM_SETTING = _with_default_ensemble_method(SIM_SETTING, order)

    if order in (:first_order, :order1, :first, 1)
        println("Start running 1st-order cumulant spin-cavity simulation...")
        return run_sim_1st_order(
            SIM_SETTING,
            SYSTEM_CONFIG,
            PULSE_CONFIG;
            clean_gpu = clean_gpu,
        )

    elseif order in (:second_order, :order2, :second, 2)
        println("Start running 2nd-order cumulant spin-cavity simulation...")
        return run_sim_2nd_order(
            SIM_SETTING,
            SYSTEM_CONFIG,
            PULSE_CONFIG;
            clean_gpu = clean_gpu,
        )

    else
        error("Unknown simulation_order = $(order). Use :order1 or :order2.")
    end
end

function build_full_config(SIM_SETTING, SYSTEM_CONFIG)
    return merge(SIM_SETTING, SYSTEM_CONFIG)
end
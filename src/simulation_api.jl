function _with_default_ensemble_method(SIM_SETTING, order)
    hasproperty(SIM_SETTING, :ensemble_method) && return SIM_SETTING
    order in (:first_order, :order1, :first, 1,
              :second_order, :order2, :second, 2) || return SIM_SETTING
    return merge(SIM_SETTING, (ensemble_method = :auto,))
end

# One public order-2 path: the MGPU stack (`mgpu_run` / `mgpu_run_simulation`
# / `run_simulation`). Row-sum exchange is NCCL/P2P in exchange_rowsums!.
# `run_sim_2nd_order` remains the DiffEq single-device backend.
# `src/sim_2nd_multi_gpu_opt.jl` is a standalone driver, not this API.

function run_simulation(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG; clean_gpu=true, kwargs...)
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
        println("Start running 2nd-order cumulant spin-cavity simulation (MGPU)...")
        return mgpu_run_sim_2nd_order(
            SIM_SETTING,
            SYSTEM_CONFIG,
            PULSE_CONFIG;
            clean_gpu = clean_gpu,
            kwargs...,
        )

    else
        error("Unknown simulation_order = $(order). Use :order1 or :order2.")
    end
end

function build_full_config(SIM_SETTING, SYSTEM_CONFIG)
    return merge(SIM_SETTING, SYSTEM_CONFIG)
end
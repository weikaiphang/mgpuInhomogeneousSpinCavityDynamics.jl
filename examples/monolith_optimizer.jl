# Mode: optimizer — Adam on B-spline parameters only (n_params = 3k + k nA + k nf).
# Loss = pulse_cost from src/pulse_optimizer2.jl (fidelity physics + time/tmax/power).
include(joinpath(@__DIR__, "monolith_common.jl"))

MODE = :optimizer

SIM_SETTING = (
    simulation_order = :first_order,
    M_delta = 8,
    M_g = 1,
    initial_condition = :ground,
    Ttotal = 100e-6,
    Nt_save = 21,
    reltol = 1e-7,
    abstol = 1e-7,
    ensemble_method = :auto,
    saved_file_name = "data/monolith_optimizer.jld2",
)

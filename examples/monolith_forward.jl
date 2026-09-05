# Mode: forward — 1st-order cumulant dynamics with the raw input pulse.
include(joinpath(@__DIR__, "monolith_common.jl"))

MODE = :forward

SIM_SETTING = (
    simulation_order = :first_order,
    M_delta = 16,
    M_g = 1,
    initial_condition = :ground,
    Ttotal = 100e-6,
    Nt_save = 51,
    reltol = 1e-8,
    abstol = 1e-8,
    ensemble_method = :auto,
    saved_file_name = "data/monolith_forward.jld2",
)

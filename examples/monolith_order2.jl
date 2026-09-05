# Mode: order2 — 1st+2nd-order results with the raw pulse.
# Quadrature is still used when the ensemble is quad-friendly (do not fall back to histogram).
include(joinpath(@__DIR__, "monolith_common.jl"))

MODE = :order2

SIM_SETTING = (
    simulation_order = :second_order,
    M_delta = 8,
    M_g = 1,
    initial_condition = :ground,
    Ttotal = 40e-6,
    Nt_save = 21,
    reltol = 1e-8,
    abstol = 1e-8,
    ensemble_method = :auto,
    saved_file_name = "data/monolith_order2.jld2",
)

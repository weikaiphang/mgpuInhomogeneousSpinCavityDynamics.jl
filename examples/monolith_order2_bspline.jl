# Mode: order2_bspline — B-spline parameterization, then 2nd-order.
# :auto still selects quadrature when Δ/g are quad-friendly.
include(joinpath(@__DIR__, "monolith_common.jl"))

MODE = :order2_bspline

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
    saved_file_name = "data/monolith_order2_bspline.jld2",
)

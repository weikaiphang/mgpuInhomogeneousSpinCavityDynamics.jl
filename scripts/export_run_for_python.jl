# ============================================================
# EXPORT A SAVED RUN FOR CROSS-CHECKING FROM PYTHON
#
# JLD2 packs the saved `data` NamedTuple using Julia-specific
# compound HDF5 types (see sim_io.py's own HONEST CAVEAT docstring
# in InhomogeneousSpinCavityDynamics.py for the same caveat from
# the other direction) -- so it is not readable via plain h5py.
# This script re-exports the one Julia run under comparison into
# two plain, dependency-free formats any Python environment can
# read directly:
#
#   trajectory.csv -- t_saved, a_sol, Sigma_p_sol, Sigma_z_sol,
#                      E_of_t_arr (complex columns split into
#                      _re/_im), one row per saved time point.
#   meta.json      -- N_total, elapsed_seconds, ensemble sizes,
#                      and peak_detection_results.
#
# Usage:
#   julia --project=. scripts/export_run_for_python.jl <path/to/run.jld2> <output_dir>
# ============================================================

using JLD2
using JSON3
using Printf

length(ARGS) >= 2 || error(
    "Usage: julia --project=. scripts/export_run_for_python.jl " *
    "<path/to/run.jld2> <output_dir>"
)

jld2_path = ARGS[1]
outdir    = ARGS[2]

isfile(jld2_path) || error("No such file: $jld2_path")
mkpath(outdir)

@load jld2_path data

# ============================================================
# TRAJECTORY CSV
# ============================================================

csv_path = joinpath(outdir, "trajectory.csv")

open(csv_path, "w") do io
    println(
        io,
        "t_saved,a_sol_re,a_sol_im,Sigma_p_re,Sigma_p_im," *
        "Sigma_z_re,Sigma_z_im,E_of_t_re,E_of_t_im",
    )

    for i in eachindex(data.t_saved)
        @printf(
            io,
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            data.t_saved[i],
            real(data.a_sol[i]), imag(data.a_sol[i]),
            real(data.Σp_sol[i]), imag(data.Σp_sol[i]),
            real(data.Σz_sol[i]), imag(data.Σz_sol[i]),
            real(data.E_of_t_arr[i]), imag(data.E_of_t_arr[i]),
        )
    end
end

# ============================================================
# METADATA JSON
# ============================================================

peaks = [
    Dict(
        "label" => String(r.label),
        "expected_time" => r.expected_time,
        "detected_time" => r.detected_time,
        "detected_amplitude" => r.detected_amplitude,
    )
    for r in data.peak_detection_results
]

meta = Dict(
    "source_file" => abspath(jld2_path),
    "N_total" => data.N_total,
    "elapsed_seconds" => data.elapsed_seconds,
    "M_delta" => data.M_delta,
    "M_g" => data.M_g,
    "M_total" => data.M_total,
    "peak_detection_results" => peaks,
)

meta_path = joinpath(outdir, "meta.json")

open(meta_path, "w") do io
    JSON3.write(io, meta)
end

println("Exported trajectory to: $csv_path")
println("Exported metadata to:   $meta_path")

nothing

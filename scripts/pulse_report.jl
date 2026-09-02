# ============================================================================
#  scripts/pulse_report.jl  --  CLI wrapper around `write_pulse_report`
#
#     julia --project scripts/pulse_report.jl <stem> [options]
#
#  <stem> is the basename (no extension) of a reference run, e.g.
#     run_105_3ARP_M20000__3arpcomp1     3ARP_pi_gstd_1em06Hz
#
#  Reads  <data-dir>/<stem>.jld2  (+ _optrunlog.jld2 + _opt_pulsepara.jld2 if
#  present) and writes a standalone page to  <out-dir>/<stem>.html  with the
#  |E(t)| + A(t) amplitude charts, instantaneous frequency, before/after tiles
#  (duration / inversion / silencing / avg power), a per-segment table, the
#  SYSTEM_CONFIG / SIM_SETTING / derived-ensemble block, the B-spline params
#  (cA/cf), a cost-vs-epoch curve, and the total optimiser run-time.
#
#  The optimiser already calls this automatically at the end of every
#  `optimise_control_pulse_from_jld2` run (pipeline kwarg `write_report=true`);
#  use this script to (re)generate a report by hand.
#
#  OPTIONS
#     --data-dir DIR     default  data/data_1st_order
#     --out-dir  DIR     default  a sibling  html/  of the data dir's parent
#     --log-dir  DIR     where the _optrunlog/_opt_pulsepara siblings live
#                        (default: the data dir) -- pass it when the optimiser
#                        wrote its log to a separate directory
#     --param-budget N   default  120   (only when there is no run log)
#     --runtime SECONDS  total optimiser wall-time (falls back to the run log's
#                        stored `elapsed_seconds`, else "not recorded")
# ============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using InhomogeneousSpinCavityDynamics

function main(argv)
    isempty(argv) && error("usage: julia --project scripts/pulse_report.jl <stem> " *
                           "[--data-dir DIR] [--out-dir DIR] [--log-dir DIR] [--param-budget N] [--runtime SECONDS]")
    stem = argv[1]
    data_dir = nothing; out_dir = nothing; log_dir = nothing
    param_budget = 120; runtime_s = nothing
    i = 2
    while i <= length(argv)
        a = argv[i]
        if     a == "--data-dir";     data_dir     = argv[i+1]; i += 2
        elseif a == "--out-dir";      out_dir      = argv[i+1]; i += 2
        elseif a == "--log-dir";      log_dir      = argv[i+1]; i += 2
        elseif a == "--param-budget"; param_budget = parse(Int, argv[i+1]); i += 2
        elseif a == "--runtime";      runtime_s    = parse(Float64, argv[i+1]); i += 2
        else   error("unknown option $a")
        end
    end
    write_pulse_report(stem; data_dir=data_dir, out_dir=out_dir, log_dir=log_dir,
                       param_budget=param_budget, runtime_s=runtime_s, verbose=true)
end

main(ARGS)

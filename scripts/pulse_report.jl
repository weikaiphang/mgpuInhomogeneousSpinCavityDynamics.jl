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

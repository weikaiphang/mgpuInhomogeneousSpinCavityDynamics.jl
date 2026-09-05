# Chimera production include graph. Public dynamics API lives here.

include("frequency.jl")
include("coupling.jl")
include("quadrature.jl")
include("ensemble.jl")

include("eoms/hamiltonian.jl")
include("eoms/state_1st.jl")
include("eoms/ics_1st.jl")
include("eoms/closure_1st.jl")
include("eoms/closure_1st_real.jl")
include("eoms/closure_1st_ip.jl")
include("eoms/jacobian_1st.jl")

include(joinpath(@__DIR__, "..", "peak_detection_helpers.jl"))
include("integrate/sciml.jl")
include("integrate/sciml_1st.jl")

include("eoms/state_2nd.jl")
include("eoms/ics_2nd.jl")
include("eoms/closure_2nd.jl")
include("integrate/sciml_2nd.jl")

include("mgpu/layout.jl")
include("mgpu/devices.jl")
include("mgpu/kernels.jl")
include("mgpu/problem.jl")
include("mgpu/ics.jl")
include("mgpu/observables.jl")
include("mgpu/state_io.jl")
include("mgpu/rhs_cpu.jl")
include("mgpu/solver.jl")

include("noise/qrt.jl")
include("eoms/quantum_cumulants.jl")
if HAVE_QUANTUMCUMULANTS && !isdefined(@__MODULE__, :_derive_tc_meanfield_impl)
    include("eoms/quantum_cumulants_impl.jl")
end
include("oracle/quantumtoolbox.jl")
include("api.jl")

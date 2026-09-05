# Package-owner load flags. QuantumCumulants + SecondQuantizedAlgebra are
# hard deps of the parent module. QuantumToolbox is required in Project.toml
# and used by the oracle layer.

const HAVE_QUANTUMCUMULANTS = isdefined(@__MODULE__, :QuantumCumulants) ||
    isdefined(@__MODULE__, :meanfield)

const HAVE_SECONDQUANTIZEDALGEBRA = isdefined(@__MODULE__, :SecondQuantizedAlgebra) ||
    HAVE_QUANTUMCUMULANTS

const HAVE_MODELINGTOOLKIT = try
    @eval using ModelingToolkit
    true
catch
    false
end

const HAVE_QUANTUMTOOLBOX = try
    @eval using QuantumToolbox
    true
catch
    false
end

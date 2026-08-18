# ============================================================
# generate_jld2_from_python.jl
#
# Batch converter: InhomogeneousSpinCavityDynamics.py's SAVESIMUL output
# (a "<basename>.h5" + "<basename>___config.json" pair, written by
# sim_io.save_full_trajectory/save_sim_config -- plain Python/h5py, NOT a
# Julia JLD2 file, see that module's own HONEST CAVEAT) into a real, directly
# InhomogeneousSpinCavityDynamics.jl-loadable "<basename>.jld2" sitting
# alongside them in the same directory.
#
# Scans SIM_DATA_DIR (non-recursive) for every "<basename>.h5" that has a
# matching "<basename>___config.json" but NOT yet a matching
# "<basename>.jld2", and generates exactly that -- already-converted or
# incomplete (missing one half of the pair) basenames are left untouched,
# so re-running this script after new SAVESIMUL runs only does the
# incremental work.
#
# WHY THIS EXISTS (see also sim_io.py's own module docstring): Python
# cannot write genuine JLD2 -- it's a Julia-specific format layered on
# HDF5 (custom type tags/reference tables), not reproducible without
# calling into Julia or reimplementing JLD2's own writer. Empirically
# confirmed while designing this script that a plain-HDF5 file (like the
# ".h5" companion here) is NOT transparently readable via `JLD2.load` --
# FileIO's own format-sniffing dispatches it to a generic `:HDF5` handler
# that needs the separate HDF5.jl package (not a dependency of this
# project), so `JLD2.load(path)` / `JLD2.load(path, "data")` both fail
# outright, regardless of top-level structure. The LOWER-level
# `JLD2.jldopen`/`JLD2.load_attributes` API (used below), which bypasses
# FileIO's dispatch entirely, CAN open it directly -- confirmed against a
# real Python-written file -- so no new HDF5-reading dependency is needed;
# only a JSON parser (JSON3, added to this project's Project.toml) for the
# JSON-encoded SIM_SETTING/SYSTEM_CONFIG/PULSE_CONFIG attributes.
#
# Two byte-layout details, both confirmed empirically against a real
# Python-written file before writing this script (not assumed):
#   - h5py's native complex128 encoding is a compound datatype with fields
#     named "r"/"i" (NOT JLD2's own "re"/"im" convention) -- JLD2.jldopen
#     reads such a dataset back as an Array{@NamedTuple{r::Float64,
#     i::Float64}}; reconstructed into ComplexF64 below.
#   - numpy/h5py write arrays row-major; Julia is column-major. The SAME
#     bytes read back via JLD2.jldopen therefore come back with REVERSED
#     shape (e.g. Python's Sp_sol (M, Nt) reads back as (Nt, M) in Julia)
#     -- every 2D array below is permutedims'd back to the Python/Julia-
#     shared (M, Nt) / (M_delta, M_g) convention.
# ============================================================

using JLD2
using JSON3

# ============================================================
# 0) USER SETTINGS
# ============================================================

# Sibling repo layout: .../InhomogeneousSpinCavityDynamics.jl/scripts/../..
# /InhomogeneousSpinCavityDynamics.py/sim_data.
SIM_DATA_DIR = joinpath(@__DIR__, "..", "..", "InhomogeneousSpinCavityDynamics.py", "sim_data")

# ============================================================
# 1) JSON -> JULIA CONVERSION
#
# JSON object -> NamedTuple, JSON array -> Tuple (matching this package's
# own SIM_SETTING/SYSTEM_CONFIG/PULSE_CONFIG shapes -- e.g. PULSE_CONFIG is
# a Tuple of pulse NamedTuples, not a Vector). The VALUE of any
# kind/simulation_order/initial_condition/binning key is converted
# String -> Symbol wherever it appears, since Julia code compares these
# via == :symbol equality (e.g. cfg.kind == :gaussian in src/config.jl,
# src/pulses.jl, src/coupling_inhomogeneity.jl) -- a bare JSON String
# would never compare equal to those Symbols.
# ============================================================

const _SYMBOL_KEYS = (:kind, :simulation_order, :initial_condition, :binning)

_to_julia(value) = value

function _to_julia(obj::JSON3.Object)
    ks = Symbol[]
    vs = Any[]
    for (k, v) in pairs(obj)
        sym_k = Symbol(k)
        push!(ks, sym_k)
        push!(vs, (sym_k in _SYMBOL_KEYS && v isa AbstractString) ? Symbol(v) : _to_julia(v))
    end
    return NamedTuple{Tuple(ks)}(Tuple(vs))
end

_to_julia(arr::JSON3.Array) = Tuple(_to_julia(x) for x in arr)

# ============================================================
# 2) DATASET RECONSTRUCTION (complex compound -> ComplexF64, row-major ->
#    column-major shape fix -- see module docstring above)
# ============================================================

function _reconstruct_dataset(value)
    if value isa AbstractArray && eltype(value) <: NamedTuple && fieldnames(eltype(value)) == (:r, :i)
        value = getproperty.(value, :r) .+ im .* getproperty.(value, :i)
    end
    if value isa AbstractArray && ndims(value) == 2
        value = permutedims(value)
    end
    return value
end

# ============================================================
# 3) LOAD ONE "<basename>.h5" INTO A JULIA NamedTuple `data`
# ============================================================

function load_python_h5_as_data(h5_path::AbstractString)
    f = JLD2.jldopen(h5_path, "r")
    dataset_names = keys(f)
    dataset_values = Dict{Symbol,Any}(
        Symbol(name) => _reconstruct_dataset(f[name]) for name in dataset_names
    )
    close(f)

    # Re-open read-only just for load_attributes -- JLD2.jldopen's file
    # object doesn't expose an attrs()-style accessor; load_attributes is
    # JLD2's own (undocumented but stable across the installed 0.6.x
    # series -- verified directly against this project's Manifest.toml)
    # internal API for reading root-level HDF5 attributes.
    f2 = JLD2.jldopen(h5_path, "r")
    attrs = JLD2.load_attributes(f2, "/")
    close(f2)

    config_values = Dict{Symbol,Any}()
    for (attr_name, attr_value) in attrs
        endswith(attr_name, "__json") || continue
        field_name = Symbol(attr_name[1:end-length("__json")])
        config_values[field_name] = _to_julia(JSON3.read(attr_value))
    end

    merged = merge(dataset_values, config_values)

    # SIM_SETTING.M_delta/M_g must match the ACTUAL ensemble axes, not
    # whatever scalar M_delta/M_g happened to be saved alongside them --
    # noise.jl rebuilds M = M_delta * M_g from exactly these two fields,
    # and a constant coupling distribution collapsing M_g -> 1 (or any
    # other bin-count mismatch) would otherwise silently break its QRT
    # reconstruction of delta_b/g_b (see src/noise.jl's own "Flattening
    # convention" comment).
    if haskey(merged, :SIM_SETTING) && haskey(merged, :delta_b_1d) && haskey(merged, :g_b_1d)
        merged[:SIM_SETTING] = merge(
            merged[:SIM_SETTING],
            (M_delta = length(merged[:delta_b_1d]), M_g = length(merged[:g_b_1d])),
        )
    end

    ks = Tuple(keys(merged))
    vs = Tuple(merged[k] for k in ks)
    return NamedTuple{ks}(vs)
end

# ============================================================
# 4) FIND (.h5, ___config.json) PAIRS MISSING THEIR .jld2
# ============================================================

function find_pending_basenames(dir::AbstractString)
    isdir(dir) || error("sim_data directory does not exist: $dir")

    entries = readdir(dir)
    h5_basenames = Set(
        splitext(name)[1] for name in entries if endswith(name, ".h5")
    )
    config_basenames = Set(
        name[1:end-length("___config.json")]
        for name in entries if endswith(name, "___config.json")
    )
    jld2_basenames = Set(
        splitext(name)[1] for name in entries if endswith(name, ".jld2")
    )

    complete_pairs = intersect(h5_basenames, config_basenames)
    incomplete = symdiff(h5_basenames, config_basenames)
    for basename in incomplete
        @warn "Skipping incomplete pair (missing .h5 or ___config.json)" basename
    end

    return sort(collect(setdiff(complete_pairs, jld2_basenames)))
end

# ============================================================
# 5) RUN
# ============================================================

function generate_all(dir::AbstractString = SIM_DATA_DIR)
    pending = find_pending_basenames(dir)

    if isempty(pending)
        println("Nothing to do -- every (.h5, ___config.json) pair in $dir already has a .jld2.")
        return nothing
    end

    println("Found $(length(pending)) file(s) to convert in $dir:")
    for basename in pending
        println("  ", basename)
    end
    println()

    for basename in pending
        h5_path = joinpath(dir, basename * ".h5")
        jld2_path = joinpath(dir, basename * ".jld2")

        println("Converting: ", basename)
        try
            data = load_python_h5_as_data(h5_path)
            JLD2.jldsave(jld2_path; data = data)
            println("  Saved: ", jld2_path)
        catch e
            @error "  FAILED to convert $basename" exception = (e, catch_backtrace())
        end
    end

    println("\nDone.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_all()
end

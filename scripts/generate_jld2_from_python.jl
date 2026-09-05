
using JLD2
using JSON3


SIM_DATA_DIR = joinpath(@__DIR__, "..", "..", "InhomogeneousSpinCavityDynamics.py", "sim_data")


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


function _reconstruct_dataset(value)
    if value isa AbstractArray && eltype(value) <: NamedTuple && fieldnames(eltype(value)) == (:r, :i)
        value = getproperty.(value, :r) .+ im .* getproperty.(value, :i)
    end
    if value isa AbstractArray && ndims(value) == 2
        value = permutedims(value)
    end
    return value
end


function load_python_h5_as_data(h5_path::AbstractString)
    f = JLD2.jldopen(h5_path, "r")
    dataset_names = keys(f)
    dataset_values = Dict{Symbol,Any}(
        Symbol(name) => _reconstruct_dataset(f[name]) for name in dataset_names
    )
    close(f)






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

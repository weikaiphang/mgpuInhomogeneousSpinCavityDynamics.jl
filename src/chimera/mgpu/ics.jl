# Product-state ICs are the same C1/H1 constructors as the SciML path.
# Multi-GPU does not keep a second IC kernel.

function set_initial_condition!(prob::MGPUProblem, Nj::AbstractVector, kind::Symbol)
    u0 = build_u0_2nd_order(prob.M, Nj, kind)
    T = eltype(prob.delta_b)
    if T !== Float64
        return Complex{T}.(u0)
    end
    return u0
end

small_block_initial(M::Int, Nj::AbstractVector, kind::Symbol, ::Type{T}) where {T} =
    Complex{T}.(build_u0_2nd_order(M, Nj, kind)[1:small_length(M)])

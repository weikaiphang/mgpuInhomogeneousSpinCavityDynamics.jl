# Loaded only when QuantumCumulants is in scope (macros require it at parse time).
# API: QuantumCumulants 0.7 / SecondQuantizedAlgebra (`@variables`, `@qnumbers`).

function _derive_tc_meanfield_impl(M::Int, order::Int)
    M >= 1 || error("derive_tc_meanfield: M must be >= 1")
    order in (1, 2) || error("derive_tc_meanfield: order must be 1 or 2")
    return M == 1 ? _derive_tc_M1(order) : _derive_tc_indexed(M, order)
end

function _derive_tc_M1(order::Int)
    hc = FockSpace(:cavity)
    ha = NLevelSpace(:atom, 2)
    h = hc ⊗ ha
    @variables Δ0::Real δ1::Real g1::Real κt::Real η::Real
    @qnumbers a::Destroy(h)
    σ(α, β) = Transition(h, :σ, α, β)
    H = Δ0 * a' * a + δ1 * (σ(2, 2) - 0.5) +
        g1 * (a * σ(2, 1) + a' * σ(1, 2)) +
        1im * η * (a' - a)
    return complete(meanfield([a, σ(2, 1), σ(2, 2)], H, [a]; rates=[κt], order=order))
end

function _derive_tc_indexed(M::Int, order::Int)
    hc = FockSpace(:cavity)
    ha = NLevelSpace(:atom, 2)
    h = hc ⊗ ha
    i = Index(h, :i, M, ha)
    @variables Δ0::Real κt::Real η::Real
    @qnumbers a::Destroy(h)
    σi(α, β, k) = IndexedOperator(Transition(h, :σ, α, β), k)
    gj = IndexedVariable(:g, i)
    δj = IndexedVariable(:δ, i)
    H = Δ0 * a' * a +
        Σ(δj * (σi(2, 2, i) - 0.5) + gj * (a * σi(2, 1, i) + a' * σi(1, 2, i)), i) +
        1im * η * (a' - a)
    return complete(meanfield([a, σi(2, 1, i), σi(2, 2, i)], H, [a]; rates=[κt], order=order))
end

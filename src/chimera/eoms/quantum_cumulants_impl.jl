# Loaded only when QuantumCumulants is in scope (macros require it at parse time).

function _derive_tc_meanfield_impl(M::Int, order::Int)
    M >= 1 || error("derive_tc_meanfield: M must be >= 1")
    order in (1, 2) || error("derive_tc_meanfield: order must be 1 or 2")

    hc = FockSpace(:cavity)
    if M == 1
        ha = NLevelSpace(:atom, 2)
        h = hc ⊗ ha
        @cnumbers Δ0 δ1 g1 κt η
        @qnumbers a::Destroy(h)
        σ(α, β) = Transition(h, :σ, α, β)
        H = Δ0 * a' * a + δ1 * (σ(2, 2) - 0.5) +
            g1 * (a * σ(2, 1) + a' * σ(1, 2)) +
            1im * η * (a' - a)
        eqs = meanfield([a, σ(2, 1), σ(2, 2)], H, [a]; rates=[κt], order=order)
        return complete(eqs)
    end

    ha = NLevelSpace(:atom, 2)
    h = hc ⊗ ha
    i = Index(h, :i, M, ha)
    @cnumbers Δ0 κt η
    @qnumbers a::Destroy(h)
    σ(α, β, k) = IndexedOperator(Transition(h, :σ, α, β), k)
    gj = IndexedVariable(:g, i)
    δj = IndexedVariable(:δ, i)
    H = Δ0 * a' * a + Σ(δj * (σ(2, 2, i) - 0.5) + gj * (a * σ(2, 1, i) + a' * σ(1, 2, i)), i) +
        1im * η * (a' - a)
    eqs = meanfield([a, σ(2, 1, i), σ(2, 2, i)], H, [a]; rates=[κt], order=order)
    return complete(eqs)
end

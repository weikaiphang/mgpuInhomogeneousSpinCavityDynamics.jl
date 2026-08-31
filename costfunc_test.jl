function _weighted_inversion(Sz, g_b, Nj, ::Type{T}) where {T}
    w = Nj .* abs2.(g_b)                          # paper |P(g)|² ∝ N g²
    ι = real.(Sz) ./ (Nj ./ 2 .+ T(1e-30))
    Ij = clamp.((ι .+ 1) ./ 2, zero(T), one(T))
    return sum(w .* Ij) / (sum(w) + T(1e-30))
end

# F(ω) on a product mesh: `groups` = Dict(ω => indices)
function _paper_F_of_omega(phi_or_Sp, g_b, Nj, groups; phases::Bool)
    Fω = Dict{eltype(keys(groups)),Complex{eltype(g_b)}}()
    for (ω, idx) in groups
        w = @view(Nj[idx]) .* abs2.(@view(g_b[idx]))          # N g² on e^{iφ}
        if phases
            num = sum(w .* cis.(phi_or_Sp[idx]))
            den = sum(w)
        else
            # raw S₊: weight is g², denom g² N/2
            wg = abs2.(@view(g_b[idx]))
            num = sum(wg .* phi_or_Sp[idx])
            den = sum(wg .* @view(Nj[idx])) / 2
        end
        Fω[ω] = num / (den + 1e-30)
    end
    return Fω
end

_paper_silencing(Fω, nω) = sum(nω[ω] * abs(F) for (ω, F) in Fω) / sum(values(nω))
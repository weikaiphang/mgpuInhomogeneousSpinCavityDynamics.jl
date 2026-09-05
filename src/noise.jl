
CUDA.allowscalar(false)


struct NoiseSimulationData
    file::String

    times_us::Vector{Float64}

    a_sol::Vector{ComplexF64}
    n_sol::Vector{Float64}
    adad_sol::Vector{ComplexF64}

    Sp_sol::Matrix{ComplexF64}
    Sz_sol::Matrix{Float64}

    adSp_sol::Matrix{ComplexF64}
    adSm_sol::Matrix{ComplexF64}
    adSz_sol::Matrix{ComplexF64}

    M_delta::Int
    M_g::Int
    M::Int

    delta0_us::Float64
    delta_b_us::Vector{Float64}
    g_b_us::Vector{Float64}

    kappa_e_us::Float64
    kappa_i_us::Float64
    kappa_t_us::Float64
end

function _require_noise_fields(data)
    required = (
        :SIM_SETTING,
        :SYSTEM_CONFIG,
        :t_saved,
        :a_sol,
        :n_sol,
        :adad_sol,
        :Sp_sol,
        :Sz_sol,
        :adSp_sol,
        :adSm_sol,
        :adSz_sol,
        :delta_b_1d,
        :g_b_1d,
    )

    missing = Symbol[]

    for name in required
        hasproperty(data, name) || push!(missing, name)
    end

    isempty(missing) || error(
        "The saved data is missing fields required for QRT noise: " *
        join(string.(missing), ", ")
    )

    return nothing
end

function _validate_noise_data(d::NoiseSimulationData)
    Nt = length(d.times_us)

    d.M == d.M_delta * d.M_g || error(
        "M = $(d.M) is inconsistent with " *
        "M_delta × M_g = $(d.M_delta * d.M_g)."
    )

    length(d.delta_b_us) == d.M || error(
        "length(delta_b_us) must equal M."
    )

    length(d.g_b_us) == d.M || error(
        "length(g_b_us) must equal M."
    )

    length(d.a_sol) == Nt || error("length(a_sol) must equal length(times_us).")
    length(d.n_sol) == Nt || error("length(n_sol) must equal length(times_us).")
    length(d.adad_sol) == Nt || error("length(adad_sol) must equal length(times_us).")

    size(d.Sp_sol) == (d.M, Nt) || error(
        "size(Sp_sol) must be (M, Nt) = ($(d.M), $Nt), " *
        "but received $(size(d.Sp_sol))."
    )

    size(d.Sz_sol) == (d.M, Nt) || error(
        "size(Sz_sol) must be (M, Nt) = ($(d.M), $Nt), " *
        "but received $(size(d.Sz_sol))."
    )

    size(d.adSp_sol) == (d.M, Nt) || error(
        "size(adSp_sol) must be (M, Nt)."
    )

    size(d.adSm_sol) == (d.M, Nt) || error(
        "size(adSm_sol) must be (M, Nt)."
    )

    size(d.adSz_sol) == (d.M, Nt) || error(
        "size(adSz_sol) must be (M, Nt)."
    )

    issorted(d.times_us) || error("times_us must be sorted in increasing order.")
    d.kappa_e_us >= 0 || error("kappa_e must be non-negative.")
    d.kappa_i_us >= 0 || error("kappa_i must be non-negative.")

    return nothing
end


function load_noise_data(
    file::AbstractString;
    verbose::Bool = true,
)
    isfile(file) || error("Noise data file does not exist: $file")

    data = JLD2.load(file, "data")
    _require_noise_fields(data)

    times_us = 1e6 .* Float64.(data.t_saved)

    a_sol    = ComplexF64.(data.a_sol)
    n_sol    = Float64.(real.(data.n_sol))
    adad_sol = ComplexF64.(data.adad_sol)

    Sp_sol = Matrix{ComplexF64}(data.Sp_sol)
    Sz_sol = Matrix{Float64}(real.(data.Sz_sol))

    adSp_sol = Matrix{ComplexF64}(data.adSp_sol)
    adSm_sol = Matrix{ComplexF64}(data.adSm_sol)
    adSz_sol = Matrix{ComplexF64}(data.adSz_sol)

    M_delta = Int(data.SIM_SETTING.M_delta)
    M_g     = Int(data.SIM_SETTING.M_g)



    delta_b = repeat(
        Float64.(data.delta_b_1d),
        M_g,
    )

    g_b = repeat(
        Float64.(data.g_b_1d);
        inner = M_delta,
    )

    M = length(delta_b)

    kappa_e = Float64(data.SYSTEM_CONFIG.kappa_e)
    kappa_i = Float64(data.SYSTEM_CONFIG.kappa_i)
    delta0  = Float64(data.SYSTEM_CONFIG.delta0)

    d = NoiseSimulationData(
        String(file),
        times_us,
        a_sol,
        n_sol,
        adad_sol,
        Sp_sol,
        Sz_sol,
        adSp_sol,
        adSm_sol,
        adSz_sol,
        M_delta,
        M_g,
        M,
        delta0 * 1e-6,
        delta_b .* 1e-6,
        g_b .* 1e-6,
        kappa_e * 1e-6,
        kappa_i * 1e-6,
        (kappa_e + kappa_i) * 1e-6,
    )

    _validate_noise_data(d)

    verbose && print_noise_data_summary(d)

    return d
end


function print_noise_data_summary(d::NoiseSimulationData)
    println("Loaded file: ", d.file)
    println("M_delta = ", d.M_delta)
    println("M_g     = ", d.M_g)
    println("M       = ", d.M)
    println("length(times_us) = ", length(d.times_us))

    println()
    println("Constant cavity rates:")
    println("  kappa_e / 2π = ", d.kappa_e_us * 1e6 / (2π), " Hz")
    println("  kappa_i / 2π = ", d.kappa_i_us * 1e6 / (2π), " Hz")
    println("  kappa_t / 2π = ", d.kappa_t_us * 1e6 / (2π), " Hz")
    println("  delta0 / 2π  = ", d.delta0_us  * 1e6 / (2π), " Hz")

    println()
    println("Coupling bins:")
    println("  minimum g / 2π = ", minimum(d.g_b_us) * 1e6 / (2π), " Hz")
    println("  maximum g / 2π = ", maximum(d.g_b_us) * 1e6 / (2π), " Hz")
    println("  number of g bins = ", d.M_g)

    return nothing
end


_noise_trapz(t, y) = sum(
    0.5 .* (y[1:end-1] .+ y[2:end]) .* diff(t)
)

function _noise_trapz_weights(t::AbstractVector{<:Real})
    n = length(t)

    w = zeros(Float64, n)

    n <= 1 && return w

    w[1] = 0.5 * (t[2] - t[1])

    w[end] = 0.5 * (t[end] - t[end-1])

    @inbounds for i in 2:n-1
        w[i] = 0.5 * (t[i+1] - t[i-1])
    end

    return w
end

function _noise_slice_window(
    times_us::AbstractVector{<:Real},
    center_us::Real,
    Tc_us::Real,
)
    center_us = Float64(center_us)
    Tc_us     = Float64(Tc_us)

    tmin = center_us - Tc_us / 2
    tmax = center_us + Tc_us / 2

    return findall(
        t -> t >= tmin && t <= tmax,
        times_us,
    )
end

function _noise_downsample_indices(
    idxs::Vector{Int},
    Nkeep::Int,
)
    n = length(idxs)

    Nkeep >= n && return idxs

    selected_positions = round.(
        Int,
        range(1, n; length = Nkeep),
    )

    return idxs[selected_positions]
end

function _noise_make_mode_window(
    times_us::Vector{Float64},
    center_us::Real,
    Tc_us::Real,
    idxs::Vector{Int},
)
    center_us = Float64(center_us)
    Tc_us     = Float64(Tc_us)

    t = times_us[idxs]

    tmin = center_us - Tc_us / 2
    tmax = center_us + Tc_us / 2

    wbox = zeros(Float64, length(t))

    @inbounds for i in eachindex(t)
        ti = t[i]

        wbox[i] = (
            ti >= tmin && ti <= tmax
        ) ? 1.0 : 0.0
    end

    norm2 = _noise_trapz(t, wbox.^2)

    f = if norm2 > 0
        wbox ./ sqrt(norm2)
    else
        wbox
    end

    return t, f
end


# Linearized quantum regression theorem (QRT).
#
# Seeds are connected 2nd-order moments from the saved trajectory
# (n-|⟨a⟩|², ⟨a†S⁺⟩-⟨a†⟩⟨S⁺⟩, …). Their two-time drift is the Jacobian of
# the 1st-order Tavis–Cummings mean-field equations evaluated on that
# trajectory — the standard linearized QRT / quantum regression
# approximation (Gardiner & Zoller; Plankensteiner et al., QuantumCumulants).
# A Jacobian of the full 2nd-order cumulant RHS would be a different,
# much larger postprocessor and is intentionally not used here.
function _noise_qrt_rhs_gpu!(
    da,
    dadag,
    dSp,
    dSm,
    dSz,
    a_col,
    adag_col,
    Sp_col,
    Sm_col,
    Sz_col,
    active_row,
    active_vec,
    delta0_us,
    delta_b_gpu,
    g_b_gpu,
    kappa_t_us,
    a_mean,
    Sp_mean,
    Sz_mean,
)
    B = length(a_col)



    delta_col = reshape(delta_b_gpu, :, 1)
    g_col     = reshape(g_b_gpu, :, 1)





    sum_g_Sm = vec(
        sum(
            g_col .* Sm_col;
            dims = 1,
        )
    )

    sum_g_Sp = vec(
        sum(
            g_col .* Sp_col;
            dims = 1,
        )
    )

    da .= (
        (
            -kappa_t_us / 2 -
            1im * delta0_us
        ) .* a_col .-
        1im .* sum_g_Sm
    ) .* active_vec

    dadag .= (
        (
            -kappa_t_us / 2 +
            1im * delta0_us
        ) .* adag_col .+
        1im .* sum_g_Sp
    ) .* active_vec





    adag_mean = conj(a_mean)

    a_row = reshape(
        a_col,
        1,
        B,
    )

    adag_row = reshape(
        adag_col,
        1,
        B,
    )

    dSp .= (
        1im .* delta_col .* Sp_col .-
        2im .* g_col .* (
            adag_mean .* Sz_col .+
            Sz_mean .* adag_row
        )
    ) .* active_row

    dSm .= (
        -1im .* delta_col .* Sm_col .+
        2im .* g_col .* (
            a_mean .* Sz_col .+
            Sz_mean .* a_row
        )
    ) .* active_row

    dSz .= (
        -1im .* g_col .* (
            a_mean .* Sp_col .+
            Sp_mean .* a_row
        ) .+
        1im .* g_col .* (
            adag_mean .* Sm_col .+
            conj.(Sp_mean) .* adag_row
        )
    ) .* active_row

    return nothing
end


function _noise_rk4_step_gpu!(
    a_col,
    adag_col,
    Sp_col,
    Sm_col,
    Sz_col,
    h,
    active_row,
    active_vec,
    delta0_us,
    delta_b_gpu,
    g_b_gpu,
    kappa_t_us,
    a_mean1,
    a_mean_mid,
    a_mean2,
    Sp_mean1,
    Sp_mean_mid,
    Sp_mean2,
    Sz_mean1,
    Sz_mean_mid,
    Sz_mean2,
    tmp,
)
    k1a, k1ad, k1Sp, k1Sm, k1Sz,
    k2a, k2ad, k2Sp, k2Sm, k2Sz,
    k3a, k3ad, k3Sp, k3Sm, k3Sz,
    k4a, k4ad, k4Sp, k4Sm, k4Sz,
    ta, tad, tSp, tSm, tSz = tmp





    _noise_qrt_rhs_gpu!(
        k1a,
        k1ad,
        k1Sp,
        k1Sm,
        k1Sz,
        a_col,
        adag_col,
        Sp_col,
        Sm_col,
        Sz_col,
        active_row,
        active_vec,
        delta0_us,
        delta_b_gpu,
        g_b_gpu,
        kappa_t_us,
        a_mean1,
        Sp_mean1,
        Sz_mean1,
    )

    ta    .= a_col    .+ 0.5h .* k1a
    tad   .= adag_col .+ 0.5h .* k1ad
    tSp   .= Sp_col   .+ 0.5h .* k1Sp
    tSm   .= Sm_col   .+ 0.5h .* k1Sm
    tSz   .= Sz_col   .+ 0.5h .* k1Sz





    _noise_qrt_rhs_gpu!(
        k2a,
        k2ad,
        k2Sp,
        k2Sm,
        k2Sz,
        ta,
        tad,
        tSp,
        tSm,
        tSz,
        active_row,
        active_vec,
        delta0_us,
        delta_b_gpu,
        g_b_gpu,
        kappa_t_us,
        a_mean_mid,
        Sp_mean_mid,
        Sz_mean_mid,
    )

    ta    .= a_col    .+ 0.5h .* k2a
    tad   .= adag_col .+ 0.5h .* k2ad
    tSp   .= Sp_col   .+ 0.5h .* k2Sp
    tSm   .= Sm_col   .+ 0.5h .* k2Sm
    tSz   .= Sz_col   .+ 0.5h .* k2Sz





    _noise_qrt_rhs_gpu!(
        k3a,
        k3ad,
        k3Sp,
        k3Sm,
        k3Sz,
        ta,
        tad,
        tSp,
        tSm,
        tSz,
        active_row,
        active_vec,
        delta0_us,
        delta_b_gpu,
        g_b_gpu,
        kappa_t_us,
        a_mean_mid,
        Sp_mean_mid,
        Sz_mean_mid,
    )

    ta    .= a_col    .+ h .* k3a
    tad   .= adag_col .+ h .* k3ad
    tSp   .= Sp_col   .+ h .* k3Sp
    tSm   .= Sm_col   .+ h .* k3Sm
    tSz   .= Sz_col   .+ h .* k3Sz





    _noise_qrt_rhs_gpu!(
        k4a,
        k4ad,
        k4Sp,
        k4Sm,
        k4Sz,
        ta,
        tad,
        tSp,
        tSm,
        tSz,
        active_row,
        active_vec,
        delta0_us,
        delta_b_gpu,
        g_b_gpu,
        kappa_t_us,
        a_mean2,
        Sp_mean2,
        Sz_mean2,
    )





    a_col .+= h / 6 .* (
        k1a .+
        2 .* k2a .+
        2 .* k3a .+
        k4a
    )

    adag_col .+= h / 6 .* (
        k1ad .+
        2 .* k2ad .+
        2 .* k3ad .+
        k4ad
    )

    Sp_col .+= h / 6 .* (
        k1Sp .+
        2 .* k2Sp .+
        2 .* k3Sp .+
        k4Sp
    )

    Sm_col .+= h / 6 .* (
        k1Sm .+
        2 .* k2Sm .+
        2 .* k3Sm .+
        k4Sm
    )

    Sz_col .+= h / 6 .* (
        k1Sz .+
        2 .* k2Sz .+
        2 .* k3Sz .+
        k4Sz
    )

    return nothing
end


function _noise_EdE_QRT_streaming_gpu(
    tgrid_us::Vector{Float64},
    v::Vector{Float64},
    M::Int,
    delta0_us::Float64,
    delta_b_us::Vector{Float64},
    g_b_us::Vector{Float64},
    kappa_t_us::Float64,
    a_grid::Vector{ComplexF64},
    n_grid::Vector{Float64},
    adad_grid::Vector{ComplexF64},
    Sp_grid_cpu::Matrix{ComplexF64},
    Sz_grid_cpu::Matrix{Float64},
    adSp_grid_cpu::Matrix{ComplexF64},
    adSm_grid_cpu::Matrix{ComplexF64},
    adSz_grid_cpu::Matrix{ComplexF64};
    batch_size::Int = 64,
)
    Ng = length(tgrid_us)

    @assert length(delta_b_us) == M
    @assert length(g_b_us) == M
    @assert size(Sp_grid_cpu, 1) == M
    @assert size(Sz_grid_cpu, 1) == M

    println("Moving QRT window data to GPU...")

    delta_b_gpu = CuArray(delta_b_us)
    g_b_gpu     = CuArray(g_b_us)

    Sp_grid = CuArray(Sp_grid_cpu)
    Sz_grid = CuArray(Sz_grid_cpu)

    try




    total = sum(
        (v .* v) .* n_grid
    )

    println(
        "Initial diagonal EdE contribution = ",
        total,
    )





    for jb in 1:batch_size:Ng
        je = min(
            jb + batch_size - 1,
            Ng,
        )

        js = collect(jb:je)

        B = length(js)

        println(
            "GPU QRT batch: ",
            jb,
            " to ",
            je,
            " / ",
            Ng,
        )





        a0_cpu = zeros(
            ComplexF64,
            B,
        )

        adag0_cpu = zeros(
            ComplexF64,
            B,
        )

        Sp0_cpu = zeros(
            ComplexF64,
            M,
            B,
        )

        Sm0_cpu = zeros(
            ComplexF64,
            M,
            B,
        )

        Sz0_cpu = zeros(
            ComplexF64,
            M,
            B,
        )

        @inbounds for c in 1:B
            j = js[c]

            a_mean    = a_grid[j]
            adag_mean = conj(a_mean)

            a0_cpu[c] =
                n_grid[j] -
                abs2(a_mean)

            adag0_cpu[c] =
                adad_grid[j] -
                adag_mean^2

            Sp0_cpu[:, c] .=
                adSp_grid_cpu[:, j] .-
                adag_mean .* Sp_grid_cpu[:, j]

            Sm0_cpu[:, c] .=
                adSm_grid_cpu[:, j] .-
                adag_mean .* conj.(Sp_grid_cpu[:, j])

            Sz0_cpu[:, c] .=
                adSz_grid_cpu[:, j] .-
                adag_mean .* Sz_grid_cpu[:, j]
        end





        a_col    = CuArray(a0_cpu)
        adag_col = CuArray(adag0_cpu)

        Sp_col = CuArray(Sp0_cpu)
        Sm_col = CuArray(Sm0_cpu)
        Sz_col = CuArray(Sz0_cpu)





        tmp = (
            similar(a_col),
            similar(adag_col),
            similar(Sp_col),
            similar(Sm_col),
            similar(Sz_col),

            similar(a_col),
            similar(adag_col),
            similar(Sp_col),
            similar(Sm_col),
            similar(Sz_col),

            similar(a_col),
            similar(adag_col),
            similar(Sp_col),
            similar(Sm_col),
            similar(Sz_col),

            similar(a_col),
            similar(adag_col),
            similar(Sp_col),
            similar(Sm_col),
            similar(Sz_col),

            similar(a_col),
            similar(adag_col),
            similar(Sp_col),
            similar(Sm_col),
            similar(Sz_col),
        )

        vj_gpu = CuArray(
            v[js],
        )

        adag_start_gpu = CuArray(
            conj.(a_grid[js]),
        )





        for i in jb:(Ng - 1)
            h = tgrid_us[i+1] - tgrid_us[i]

            active_cpu = Float64.(
                js .<= i
            )

            active_vec = CuArray(
                active_cpu,
            )

            active_row = reshape(
                active_vec,
                1,
                B,
            )





            a1 = a_grid[i]
            a2 = a_grid[i+1]
            am = 0.5 * (a1 + a2)

            Sp1 = view(
                Sp_grid,
                :,
                i,
            )

            Sp2 = view(
                Sp_grid,
                :,
                i+1,
            )

            Spm = 0.5 .* (
                Sp1 .+
                Sp2
            )

            Sz1 = view(
                Sz_grid,
                :,
                i,
            )

            Sz2 = view(
                Sz_grid,
                :,
                i+1,
            )

            Szm = 0.5 .* (
                Sz1 .+
                Sz2
            )





            _noise_rk4_step_gpu!(
                a_col,
                adag_col,
                Sp_col,
                Sm_col,
                Sz_col,
                h,
                active_row,
                active_vec,
                delta0_us,
                delta_b_gpu,
                g_b_gpu,
                kappa_t_us,
                a1,
                am,
                a2,
                Sp1,
                Spm,
                Sp2,
                Sz1,
                Szm,
                Sz2,
                tmp,
            )










            active_pair_cpu = Float64.(
                js .< (i + 1)
            )

            active_pair = CuArray(
                active_pair_cpu,
            )

            corr =
                a_col .+
                adag_start_gpu .* a_grid[i+1]

            contrib_gpu =
                2.0 .*
                v[i+1] .*
                vj_gpu .*
                real.(corr) .*
                active_pair

            total += sum(contrib_gpu)
        end

        a_col = nothing
        adag_col = nothing
        Sp_col = nothing
        Sm_col = nothing
        Sz_col = nothing
        tmp = nothing
        vj_gpu = nothing
        adag_start_gpu = nothing
        GC.gc(false)
        CUDA.reclaim()
    end

        return total
    finally
        delta_b_gpu = nothing
        g_b_gpu = nothing
        Sp_grid = nothing
        Sz_grid = nothing
        GC.gc(false)
        CUDA.reclaim()
    end
end


function _noise_mode_QRT_gpu_complex(
    times_us::AbstractVector,
    center_us::Real,
    Tc_us::Real;
    Nkeep::Int,
    M::Int,
    delta0_us::Float64,
    delta_b_us::Vector{Float64},
    g_b_us::Vector{Float64},
    kappa_e_us::Float64,
    kappa_i_us::Float64,
    a_sol::Vector{ComplexF64},
    n_sol::Vector{Float64},
    adad_sol::Vector{ComplexF64},
    Sp_sol::AbstractMatrix{ComplexF64},
    Sz_sol::AbstractMatrix{Float64},
    adSp_sol::AbstractMatrix{ComplexF64},
    adSm_sol::AbstractMatrix{ComplexF64},
    adSz_sol::AbstractMatrix{ComplexF64},
    batch_size::Int = 64,
)
    times_us = collect(
        Float64.(times_us)
    )

    center_us = Float64(center_us)
    Tc_us     = Float64(Tc_us)


    kappa_t_us = kappa_e_us + kappa_i_us





    idx_win = _noise_slice_window(
        times_us,
        center_us,
        Tc_us,
    )

    isempty(idx_win) && error(
        "No points found in analysis window around " *
        "center_us = $center_us"
    )

    idx = _noise_downsample_indices(
        idx_win,
        Nkeep,
    )

    tgrid = times_us[idx]

    _, f = _noise_make_mode_window(
        times_us,
        center_us,
        Tc_us,
        idx,
    )





    w = _noise_trapz_weights(tgrid)


    v = w .* f .* sqrt(kappa_e_us)





    a_grid = ComplexF64.(
        a_sol[idx]
    )

    n_grid = Float64.(
        n_sol[idx]
    )

    adad_grid = ComplexF64.(
        adad_sol[idx]
    )

    Sp_grid_cpu = Matrix{ComplexF64}(
        @view Sp_sol[:, idx]
    )

    Sz_grid_cpu = Matrix{Float64}(
        @view Sz_sol[:, idx]
    )

    adSp_grid_cpu = Matrix{ComplexF64}(
        @view adSp_sol[:, idx]
    )

    adSm_grid_cpu = Matrix{ComplexF64}(
        @view adSm_sol[:, idx]
    )

    adSz_grid_cpu = Matrix{ComplexF64}(
        @view adSz_sol[:, idx]
    )





    EdE = _noise_EdE_QRT_streaming_gpu(
        tgrid,
        v,
        M,
        delta0_us,
        delta_b_us,
        g_b_us,
        kappa_t_us,
        a_grid,
        n_grid,
        adad_grid,
        Sp_grid_cpu,
        Sz_grid_cpu,
        adSp_grid_cpu,
        adSm_grid_cpu,
        adSz_grid_cpu;
        batch_size = batch_size,
    )





    Emean = sum(
        w .*
        f .*
        (
            -sqrt(kappa_e_us) .*
            a_grid
        )
    )

    mean_term = abs2(Emean)





    output_total_variance =
        (
            EdE -
            mean_term +
            0.5
        ) / 0.5

    norm_f = sum(
        w .* f.^2
    )




    kappa_e_eff =
        kappa_e_us *
        norm_f

    return (
        output_total_variance = output_total_variance,
        EdE          = EdE,
        mean_term    = mean_term,
        Emean        = Emean,
        norm_f       = norm_f,
        kappa_e_eff  = kappa_e_eff,
        kappa_e_us   = kappa_e_us,
        kappa_i_us   = kappa_i_us,
        kappa_t_us   = kappa_t_us,
        Ng           = length(tgrid),
        tgrid_us     = tgrid,
        mode_f       = f,
    )
end




function compute_output_mode_noise(
    d::NoiseSimulationData,
    center_us::Real,
    Tc_us::Real;
    Nkeep::Int = 5000,
    batch_size::Int = 64,
)
    Nkeep >= 1 || error("Nkeep must be at least 1.")
    batch_size >= 1 || error("batch_size must be at least 1.")
    Tc_us > 0 || error("Tc_us must be positive.")

    result = _noise_mode_QRT_gpu_complex(
        d.times_us,
        center_us,
        Tc_us;
        Nkeep = Nkeep,
        M = d.M,
        delta0_us = d.delta0_us,
        delta_b_us = d.delta_b_us,
        g_b_us = d.g_b_us,
        kappa_e_us = d.kappa_e_us,
        kappa_i_us = d.kappa_i_us,
        a_sol = d.a_sol,
        n_sol = d.n_sol,
        adad_sol = d.adad_sol,
        Sp_sol = d.Sp_sol,
        Sz_sol = d.Sz_sol,
        adSp_sol = d.adSp_sol,
        adSm_sol = d.adSm_sol,
        adSz_sol = d.adSz_sol,
        batch_size = batch_size,
    )

    noise_photon_number = result.EdE - result.mean_term

    return merge(
        result,
        (
            center_us = Float64(center_us),
            Tc_us = Float64(Tc_us),
            noise_photon_number = noise_photon_number,
        ),
    )
end


function compute_noise_windows(
    d::NoiseSimulationData,
    centers_us,
    Tc_us::Real;
    Nkeep::Int = 5000,
    batch_size::Int = 500,
)
    return [
        compute_output_mode_noise(
            d,
            center_us,
            Tc_us;
            Nkeep = Nkeep,
            batch_size = batch_size,
        )
        for center_us in centers_us
    ]
end


function compute_noise_from_file(
    file::AbstractString,
    centers_us,
    Tc_us::Real;
    Nkeep::Int = 5000,
    batch_size::Int = 64,
    verbose::Bool = true,
)
    d = load_noise_data(file; verbose = verbose)

    return compute_noise_windows(
        d,
        centers_us,
        Tc_us;
        Nkeep = Nkeep,
        batch_size = batch_size,
    )
end


function print_noise_result(
    label,
    result,
)
    println()
    println(label, ":")
    println("  center_us             = ", result.center_us)
    println("  Tc_us                 = ", result.Tc_us)
    println("  output_total_variance = ", result.output_total_variance)
    println("  noise photon number   = ", result.noise_photon_number)
    println("  EdE                   = ", result.EdE)
    println("  |Emean|²              = ", result.mean_term)
    println("  norm_f                = ", result.norm_f)
    println("  Ng                    = ", result.Ng)

    return nothing
end

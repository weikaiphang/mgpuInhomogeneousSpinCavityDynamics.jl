############################################################
# GPU correlation calculations
#
# Solver choice:
#   ASE-RASE   -> adaptive Tsit5
#   ASE-ASE    -> fixed-step RK4
#   RASE-RASE  -> fixed-step RK4
############################################################

# ============================================================
# 1) PUBLIC SETTINGS
# ============================================================

Base.@kwdef struct CorrelationSettings
    tpi_us::Float64

    ASE_window_rel_us::Tuple{Float64, Float64}
    RASE_window_rel_us::Tuple{Float64, Float64}

    Nkeep_ASE::Int = 5000
    Nkeep_RASE::Int = 5000

    tA_choices_rel_us::Vector{Float64} = [-50.0]
    tR_choices_rel_us::Vector{Float64} = [50.0]

    tau_min_us::Float64 = -20.0
    tau_max_us::Float64 = 20.0
    dtau_us::Float64 = 0.01

    # Used only by the adaptive ASE-RASE calculation.
    reltol::Float64 = 1e-8
    abstol::Float64 = 1e-8
end

struct CorrelationWorkspace
    source_file::String

    times_us::Vector{Float64}

    a_sol::Vector{ComplexF64}
    n_sol::Vector{ComplexF64}
    adad_sol::Vector{ComplexF64}

    Sp_sol::Matrix{ComplexF64}
    Sz_sol::Matrix{Float64}

    adSp_sol::Matrix{ComplexF64}
    adSm_sol::Matrix{ComplexF64}
    adSz_sol::Matrix{ComplexF64}

    kappa_e_full_us::Vector{Float64}
    kappa_t_full_us::Vector{Float64}

    M::Int

    delta_b_us::Vector{Float64}
    g_b_us::Vector{Float64}

    delta0_us::Float64
end

# ============================================================
# 2) DATA-LOADING HELPERS
# ============================================================

function _has_value(container, name::Symbol)
    if container isa AbstractDict
        return haskey(container, name) || haskey(container, String(name))
    end

    return hasproperty(container, name)
end

function _get_value(container, name::Symbol)
    if container isa AbstractDict
        haskey(container, name) && return container[name]
        haskey(container, String(name)) && return container[String(name)]
    elseif hasproperty(container, name)
        return getproperty(container, name)
    end

    error("Missing required data field: $name")
end

function _get_first(container, names::Tuple)
    for name in names
        _has_value(container, name) && return _get_value(container, name)
    end

    error(
        "Missing required data field. Expected one of: " *
        join(string.(names), ", "),
    )
end

function _get_optional(container, names::Tuple, default)
    for name in names
        _has_value(container, name) && return _get_value(container, name)
    end

    return default
end

function _get_system_config(container)
    return _get_optional(
        container,
        (:SYSTEM_CONFIG, :system_config),
        nothing,
    )
end

function _get_config_value(config, names::Tuple, default)
    config === nothing && return default
    return _get_optional(config, names, default)
end

function _load_time_axis_us(source)
    if _has_value(source, :times_us)
        return Float64.(_get_value(source, :times_us))
    elseif _has_value(source, :times)
        # The original standalone correlation files store this axis in μs.
        return Float64.(_get_value(source, :times))
    elseif _has_value(source, :t_saved)
        # InhomogeneousSpinCavityDynamics simulation output stores t_saved in seconds.
        return Float64.(_get_value(source, :t_saved)) .* 1e6
    end

    error("Missing time axis. Expected times_us, times, or t_saved.")
end

function _load_kappa_arrays_us(source, Nt::Int)
    system_config = _get_system_config(source)

    kappa_e_raw = _get_optional(
        source,
        (:kappa_e_arr, :kappa_e_full, :kappa_e),
        nothing,
    )

    if kappa_e_raw === nothing
        kappa_e_raw = _get_config_value(
            system_config,
            (:kappa_e,),
            nothing,
        )
    end

    kappa_e_raw === nothing && error(
        "Missing external cavity decay rate. Expected kappa_e_arr, " *
        "kappa_e, or SYSTEM_CONFIG.kappa_e.",
    )

    kappa_e_full_us = if kappa_e_raw isa Number
        fill(Float64(kappa_e_raw) * 1e-6, Nt)
    else
        values = Float64.(kappa_e_raw) .* 1e-6
        length(values) == Nt || error(
            "kappa_e array has length $(length(values)); expected $Nt.",
        )
        values
    end

    kappa_t_raw = _get_optional(
        source,
        (:kappa_t_arr, :kappa_total_arr, :kappa_t_full),
        nothing,
    )

    kappa_t_full_us = if kappa_t_raw === nothing
        # Preserve the behavior of the original correlation implementation.
        copy(kappa_e_full_us)
    elseif kappa_t_raw isa Number
        fill(Float64(kappa_t_raw) * 1e-6, Nt)
    else
        values = Float64.(kappa_t_raw) .* 1e-6
        length(values) == Nt || error(
            "kappa_t array has length $(length(values)); expected $Nt.",
        )
        values
    end

    return kappa_e_full_us, kappa_t_full_us
end

function _validate_workspace(workspace::CorrelationWorkspace)
    Nt = length(workspace.times_us)
    M = workspace.M

    Nt >= 2 || error("At least two saved time points are required.")
    issorted(workspace.times_us) || error("The saved time axis must be sorted.")

    length(workspace.a_sol) == Nt || error("a_sol has the wrong length.")
    length(workspace.n_sol) == Nt || error("n_sol has the wrong length.")
    length(workspace.adad_sol) == Nt || error("adad_sol has the wrong length.")

    size(workspace.Sp_sol) == (M, Nt) || error(
        "Sp_sol has size $(size(workspace.Sp_sol)); expected ($M, $Nt).",
    )
    size(workspace.Sz_sol) == (M, Nt) || error(
        "Sz_sol has size $(size(workspace.Sz_sol)); expected ($M, $Nt).",
    )
    size(workspace.adSp_sol) == (M, Nt) || error(
        "adSp_sol has size $(size(workspace.adSp_sol)); expected ($M, $Nt).",
    )
    size(workspace.adSm_sol) == (M, Nt) || error(
        "adSm_sol has size $(size(workspace.adSm_sol)); expected ($M, $Nt).",
    )
    size(workspace.adSz_sol) == (M, Nt) || error(
        "adSz_sol has size $(size(workspace.adSz_sol)); expected ($M, $Nt).",
    )

    length(workspace.delta_b_us) == M || error(
        "delta_b has length $(length(workspace.delta_b_us)); expected $M.",
    )
    length(workspace.g_b_us) == M || error(
        "g_b has length $(length(workspace.g_b_us)); expected $M.",
    )

    length(workspace.kappa_e_full_us) == Nt || error(
        "kappa_e array has the wrong length.",
    )
    length(workspace.kappa_t_full_us) == Nt || error(
        "kappa_t array has the wrong length.",
    )

    return nothing
end

"""
    load_correlation_workspace(input_file)

Load one second-order simulation file and convert all frequencies to rad/μs
and the saved time axis to μs.

Supported field aliases include both the older standalone layout
(`times`, `Sp`, `Sz`, ...) and the InhomogeneousSpinCavityDynamics layout
(`t_saved`, `Sp_sol`, `Sz_sol`, ...).
"""
function load_correlation_workspace(input_file::AbstractString)
    isfile(input_file) || error("Input file does not exist: $input_file")

    loaded = JLD2.load(input_file)
    source = _has_value(loaded, :data) ? _get_value(loaded, :data) : loaded

    times_us = _load_time_axis_us(source)
    Nt = length(times_us)

    a_sol = ComplexF64.(_get_first(source, (:a_sol, :a)))
    n_sol = ComplexF64.(_get_first(source, (:n_sol, :n, :ad_a)))
    adad_sol = ComplexF64.(_get_first(source, (:adad_sol, :adad, :ad_ad)))

    Sp_sol = ComplexF64.(_get_first(source, (:Sp_sol, :Sp)))
    Sz_sol = Float64.(real.(_get_first(source, (:Sz_sol, :Sz))))

    adSp_sol = ComplexF64.(_get_first(source, (:adSp_sol, :adSp)))
    adSm_sol = ComplexF64.(_get_first(source, (:adSm_sol, :adSm)))
    adSz_sol = ComplexF64.(_get_first(source, (:adSz_sol, :adSz)))

    # The saved second-order data contains separate one-dimensional
    # detuning and coupling axes:
    #
    #   delta_b_1d: length M_delta
    #   g_b_1d:     length M_g
    #   Nj_2d:      size (M_delta, M_g)
    #
    # The spin arrays use one flattened ensemble index with
    # M = M_delta * M_g rows. Reconstruct the flattened axes using
    # Julia column-major ordering, where detuning changes fastest.
    if _has_value(source, :delta_b_1d) || _has_value(source, :g_b_1d)
        _has_value(source, :delta_b_1d) || error(
            "Saved data contains g_b_1d but is missing delta_b_1d.",
        )
        _has_value(source, :g_b_1d) || error(
            "Saved data contains delta_b_1d but is missing g_b_1d.",
        )

        delta_b_1d_us =
            vec(Float64.(_get_value(source, :delta_b_1d))) .* 1e-6

        g_b_1d_us =
            vec(Float64.(_get_value(source, :g_b_1d))) .* 1e-6

        M_delta = length(delta_b_1d_us)
        M_g = length(g_b_1d_us)
        M = M_delta * M_g
        M_spin = size(Sp_sol, 1)

        M == M_spin || error(
            "The ensemble axes contain M_delta × M_g = " *
            "$M_delta × $M_g = $M bins, but Sp_sol contains " *
            "$M_spin ensemble rows.",
        )

        if _has_value(source, :Nj_2d)
            Nj_2d = _get_value(source, :Nj_2d)

            size(Nj_2d) == (M_delta, M_g) || error(
                "Nj_2d has size $(size(Nj_2d)); expected " *
                "($M_delta, $M_g).",
            )
        end

        # Flattening convention:
        #
        #   j = i_delta + (i_g - 1) * M_delta
        #
        # Therefore delta changes fastest and g changes slowest.
        delta_b_us = repeat(
            delta_b_1d_us;
            outer = M_g,
        )

        g_b_us = repeat(
            g_b_1d_us;
            inner = M_delta,
        )
    else
        # Compatibility with older files that already contain flattened
        # delta_b and g_b arrays.
        delta_b_us =
            vec(Float64.(_get_first(source, (:delta_b,)))) .* 1e-6

        g_b_us =
            vec(Float64.(_get_first(source, (:g_b,)))) .* 1e-6

        M = size(Sp_sol, 1)

        length(delta_b_us) == M || error(
            "Flattened delta_b has length $(length(delta_b_us)); " *
            "expected $M.",
        )

        length(g_b_us) == M || error(
            "Flattened g_b has length $(length(g_b_us)); " *
            "expected $M.",
        )
    end

    println(
        "Loaded ensemble grid: ",
        "M = $M, ",
        "length(delta_b) = $(length(delta_b_us)), ",
        "length(g_b) = $(length(g_b_us))",
    )

    kappa_e_full_us, kappa_t_full_us =
        _load_kappa_arrays_us(source, Nt)

    system_config = _get_system_config(source)
    delta0_raw = _get_optional(
        source,
        (:delta0, :delta0_rad_s),
        _get_config_value(system_config, (:delta0,), 0.0),
    )
    delta0_us = Float64(delta0_raw) * 1e-6

    workspace = CorrelationWorkspace(
        abspath(input_file),
        times_us,
        a_sol,
        n_sol,
        adad_sol,
        Sp_sol,
        Sz_sol,
        adSp_sol,
        adSm_sol,
        adSz_sol,
        kappa_e_full_us,
        kappa_t_full_us,
        M,
        delta_b_us,
        g_b_us,
        delta0_us,
    )

    _validate_workspace(workspace)
    return workspace
end

# ============================================================
# 3) INDEX HELPERS
# ============================================================

function _downsample_indices(idxs::Vector{Int}, Nkeep::Int)
    Nkeep > 0 || error("Nkeep must be positive.")

    n = length(idxs)
    Nkeep >= n && return idxs

    selected = round.(Int, range(1, n; length = Nkeep))
    return idxs[selected]
end

function _nearest_index(tgrid::Vector{Float64}, target::Float64)
    k = searchsortedfirst(tgrid, target)

    if k <= 1
        return 1
    elseif k > length(tgrid)
        return length(tgrid)
    end

    return abs(tgrid[k] - target) < abs(tgrid[k - 1] - target) ? k : k - 1
end

function _validate_settings(settings::CorrelationSettings)
    settings.ASE_window_rel_us[1] < settings.ASE_window_rel_us[2] ||
        error("ASE_window_rel_us must be increasing.")

    settings.RASE_window_rel_us[1] < settings.RASE_window_rel_us[2] ||
        error("RASE_window_rel_us must be increasing.")

    settings.tau_min_us <= settings.tau_max_us ||
        error("tau_min_us must not exceed tau_max_us.")

    settings.dtau_us > 0 || error("dtau_us must be positive.")
    settings.reltol > 0 || error("reltol must be positive.")
    settings.abstol > 0 || error("abstol must be positive.")

    return nothing
end

function _build_correlation_indices(
    workspace::CorrelationWorkspace,
    settings::CorrelationSettings;
    verbose::Bool,
)
    t_rel = workspace.times_us .- settings.tpi_us

    idx_A_full = findall(
        t -> settings.ASE_window_rel_us[1] <= t <= settings.ASE_window_rel_us[2],
        t_rel,
    )
    idx_R_full = findall(
        t -> settings.RASE_window_rel_us[1] <= t <= settings.RASE_window_rel_us[2],
        t_rel,
    )

    isempty(idx_A_full) && error("No saved time points fall inside the ASE window.")
    isempty(idx_R_full) && error("No saved time points fall inside the RASE window.")

    IA_scan = _downsample_indices(idx_A_full, settings.Nkeep_ASE)
    IR_scan = _downsample_indices(idx_R_full, settings.Nkeep_RASE)

    tA_scan = t_rel[IA_scan]
    tR_scan = t_rel[IR_scan]

    IA_fixed = Int[]
    tA_fixed = Float64[]

    for requested_time in settings.tA_choices_rel_us
        j = _nearest_index(t_rel, requested_time)

        if settings.ASE_window_rel_us[1] <= t_rel[j] <= settings.ASE_window_rel_us[2]
            push!(IA_fixed, j)
            push!(tA_fixed, t_rel[j])
        else
            @warn "ASE fixed point $requested_time μs is outside the ASE window after snapping."
        end
    end

    IR_fixed = Int[]
    tR_fixed = Float64[]

    for requested_time in settings.tR_choices_rel_us
        j = _nearest_index(t_rel, requested_time)

        if settings.RASE_window_rel_us[1] <= t_rel[j] <= settings.RASE_window_rel_us[2]
            push!(IR_fixed, j)
            push!(tR_fixed, t_rel[j])
        else
            @warn "RASE fixed point $requested_time μs is outside the RASE window after snapping."
        end
    end

    isempty(IA_fixed) && error("No valid fixed ASE time points were selected.")
    isempty(IR_fixed) && error("No valid fixed RASE time points were selected.")

    if verbose
        println(
            "ASE points kept : $(length(IA_scan)) in " *
            "[$(tA_scan[1]), $(tA_scan[end])] μs",
        )
        println(
            "RASE points kept: $(length(IR_scan)) in " *
            "[$(tR_scan[1]), $(tR_scan[end])] μs",
        )
        println("ASE fixed points : ", tA_fixed)
        println("RASE fixed points: ", tR_fixed)
    end

    return (
        t_rel = t_rel,
        IA_scan = IA_scan,
        IR_scan = IR_scan,
        tA_scan = tA_scan,
        tR_scan = tR_scan,
        IA_fixed = IA_fixed,
        IR_fixed = IR_fixed,
        tA_fixed = tA_fixed,
        tR_fixed = tR_fixed,
    )
end

# ============================================================
# 4) INITIAL QRT COLUMN
# ============================================================

function _build_g0_adag_column_cpu(
    j0::Int,
    workspace::CorrelationWorkspace,
)
    M = workspace.M

    a0 = workspace.a_sol[j0]
    adag0 = conj(a0)

    g_a = workspace.n_sol[j0] - abs2(a0)
    g_adag = workspace.adad_sol[j0] - adag0^2

    gSp = Vector{ComplexF64}(undef, M)
    gSm = Vector{ComplexF64}(undef, M)
    gSz = Vector{ComplexF64}(undef, M)

    @inbounds for j in 1:M
        Sp0 = workspace.Sp_sol[j, j0]
        Sm0 = conj(Sp0)
        Sz0 = workspace.Sz_sol[j, j0]

        gSp[j] = workspace.adSp_sol[j, j0] - adag0 * Sp0
        gSm[j] = workspace.adSm_sol[j, j0] - adag0 * Sm0
        gSz[j] = workspace.adSz_sol[j, j0] - adag0 * Sz0
    end

    return ComplexF64(g_a), ComplexF64(g_adag), gSp, gSm, gSz
end

# ============================================================
# 5) RK4 QRT EQUATIONS
# ============================================================

function _qrt_rhs_one_gpu!(
    dSp,
    dSm,
    dSz,
    a_col::ComplexF64,
    adag_col::ComplexF64,
    Sp_col,
    Sm_col,
    Sz_col,
    k::Int,
    workspace::CorrelationWorkspace,
    delta_gpu,
    g_gpu,
    Sp_gpu,
    Sz_gpu,
)
    kappa_t = workspace.kappa_t_full_us[k]

    sum_gSm = CUDA.@allowscalar sum(g_gpu .* Sm_col)
    sum_gSp = CUDA.@allowscalar sum(g_gpu .* Sp_col)

    da =
        (-0.5 * kappa_t - 1im * workspace.delta0_us) * a_col -
        1im * sum_gSm

    dadag =
        (-0.5 * kappa_t + 1im * workspace.delta0_us) * adag_col +
        1im * sum_gSp

    a_mean = workspace.a_sol[k]
    adag_mean = conj(a_mean)

    Sp_mean = view(Sp_gpu, :, k)
    Sz_mean = view(Sz_gpu, :, k)

    dSp .=
        1im .* delta_gpu .* Sp_col .-
        2im .* g_gpu .* (adag_mean .* Sz_col .+ Sz_mean .* adag_col)

    dSm .=
        -1im .* delta_gpu .* Sm_col .+
        2im .* g_gpu .* (a_mean .* Sz_col .+ Sz_mean .* a_col)

    dSz .=
        -1im .* g_gpu .* (a_mean .* Sp_col .+ Sp_mean .* a_col) .+
        1im .* g_gpu .* (
            adag_mean .* Sm_col .+
            conj.(Sp_mean) .* adag_col
        )

    return ComplexF64(da), ComplexF64(dadag)
end

function _rk4_step_one_gpu!(
    a_col::ComplexF64,
    adag_col::ComplexF64,
    Sp_col,
    Sm_col,
    Sz_col,
    k::Int,
    dt::Float64,
    workspace::CorrelationWorkspace,
    delta_gpu,
    g_gpu,
    Sp_gpu,
    Sz_gpu;
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-8,
)
    M = length(Sp_col)

    k1Sp = CUDA.zeros(ComplexF64, M)
    k1Sm = CUDA.zeros(ComplexF64, M)
    k1Sz = CUDA.zeros(ComplexF64, M)

    k2Sp = similar(k1Sp)
    k2Sm = similar(k1Sm)
    k2Sz = similar(k1Sz)

    k3Sp = similar(k1Sp)
    k3Sm = similar(k1Sm)
    k3Sz = similar(k1Sz)

    k4Sp = similar(k1Sp)
    k4Sm = similar(k1Sm)
    k4Sz = similar(k1Sz)

    tmpSp = similar(k1Sp)
    tmpSm = similar(k1Sm)
    tmpSz = similar(k1Sz)

    k1a, k1adag = _qrt_rhs_one_gpu!(
        k1Sp,
        k1Sm,
        k1Sz,
        a_col,
        adag_col,
        Sp_col,
        Sm_col,
        Sz_col,
        k,
        workspace,
        delta_gpu,
        g_gpu,
        Sp_gpu,
        Sz_gpu,
    )

    tmpa = a_col + 0.5 * dt * k1a
    tmpadag = adag_col + 0.5 * dt * k1adag
    tmpSp .= Sp_col .+ 0.5 .* dt .* k1Sp
    tmpSm .= Sm_col .+ 0.5 .* dt .* k1Sm
    tmpSz .= Sz_col .+ 0.5 .* dt .* k1Sz

    k2a, k2adag = _qrt_rhs_one_gpu!(
        k2Sp,
        k2Sm,
        k2Sz,
        tmpa,
        tmpadag,
        tmpSp,
        tmpSm,
        tmpSz,
        k,
        workspace,
        delta_gpu,
        g_gpu,
        Sp_gpu,
        Sz_gpu,
    )

    tmpa = a_col + 0.5 * dt * k2a
    tmpadag = adag_col + 0.5 * dt * k2adag
    tmpSp .= Sp_col .+ 0.5 .* dt .* k2Sp
    tmpSm .= Sm_col .+ 0.5 .* dt .* k2Sm
    tmpSz .= Sz_col .+ 0.5 .* dt .* k2Sz

    k3a, k3adag = _qrt_rhs_one_gpu!(
        k3Sp,
        k3Sm,
        k3Sz,
        tmpa,
        tmpadag,
        tmpSp,
        tmpSm,
        tmpSz,
        k,
        workspace,
        delta_gpu,
        g_gpu,
        Sp_gpu,
        Sz_gpu,
    )

    tmpa = a_col + dt * k3a
    tmpadag = adag_col + dt * k3adag
    tmpSp .= Sp_col .+ dt .* k3Sp
    tmpSm .= Sm_col .+ dt .* k3Sm
    tmpSz .= Sz_col .+ dt .* k3Sz

    k4a, k4adag = _qrt_rhs_one_gpu!(
        k4Sp,
        k4Sm,
        k4Sz,
        tmpa,
        tmpadag,
        tmpSp,
        tmpSm,
        tmpSz,
        k,
        workspace,
        delta_gpu,
        g_gpu,
        Sp_gpu,
        Sz_gpu,
    )

    a_new = a_col + dt / 6 * (k1a + 2k2a + 2k3a + k4a)
    adag_new = adag_col + dt / 6 * (
        k1adag + 2k2adag + 2k3adag + k4adag
    )

    Sp_col .+= dt / 6 .* (k1Sp .+ 2 .* k2Sp .+ 2 .* k3Sp .+ k4Sp)
    Sm_col .+= dt / 6 .* (k1Sm .+ 2 .* k2Sm .+ 2 .* k3Sm .+ k4Sm)
    Sz_col .+= dt / 6 .* (k1Sz .+ 2 .* k2Sz .+ 2 .* k3Sz .+ k4Sz)

    return ComplexF64(a_new), ComplexF64(adag_new)
end

# ============================================================
# 6) TSIT5 QRT EQUATIONS
# ============================================================

function _tsit5_step_one_gpu!(
    a_col::ComplexF64,
    adag_col::ComplexF64,
    Sp_col,
    Sm_col,
    Sz_col,
    k::Int,
    dt::Float64,
    workspace::CorrelationWorkspace,
    delta_gpu,
    g_gpu,
    Sp_gpu,
    Sz_gpu;
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-8,
)
    M = length(Sp_col)

    idx_a = 1:1
    idx_adag = 2:2
    idx_Sp = 3:(2 + M)
    idx_Sm = (3 + M):(2 + 2M)
    idx_Sz = (3 + 2M):(2 + 3M)

    u0 = CUDA.zeros(ComplexF64, 2 + 3M)

    copyto!(@view(u0[1:2]), CuArray(ComplexF64[a_col, adag_col]))
    copyto!(@view(u0[idx_Sp]), Sp_col)
    copyto!(@view(u0[idx_Sm]), Sm_col)
    copyto!(@view(u0[idx_Sz]), Sz_col)

    kappa_t = workspace.kappa_t_full_us[k]

    a_mean = workspace.a_sol[k]
    adag_mean = conj(a_mean)

    Sp_mean = @view Sp_gpu[:, k]
    Sz_mean = @view Sz_gpu[:, k]

    function rhs!(du, u, p, t)
        a = @view u[idx_a]
        adag = @view u[idx_adag]

        Sp = @view u[idx_Sp]
        Sm = @view u[idx_Sm]
        Sz = @view u[idx_Sz]

        da = @view du[idx_a]
        dadag = @view du[idx_adag]

        dSp = @view du[idx_Sp]
        dSm = @view du[idx_Sm]
        dSz = @view du[idx_Sz]

        sum_gSm = sum(g_gpu .* Sm)
        sum_gSp = sum(g_gpu .* Sp)

        da .=
            (-0.5 * kappa_t - 1im * workspace.delta0_us) .* a .-
            1im .* sum_gSm

        dadag .=
            (-0.5 * kappa_t + 1im * workspace.delta0_us) .* adag .+
            1im .* sum_gSp

        dSp .=
            1im .* delta_gpu .* Sp .-
            2im .* g_gpu .* (adag_mean .* Sz .+ Sz_mean .* adag)

        dSm .=
            -1im .* delta_gpu .* Sm .+
            2im .* g_gpu .* (a_mean .* Sz .+ Sz_mean .* a)

        dSz .=
            -1im .* g_gpu .* (a_mean .* Sp .+ Sp_mean .* a) .+
            1im .* g_gpu .* (
                adag_mean .* Sm .+
                conj.(Sp_mean) .* adag
            )

        return nothing
    end

    problem = ODEProblem(rhs!, u0, (0.0, dt))
    solution = solve(
        problem,
        Tsit5();
        reltol = reltol,
        abstol = abstol,
        save_everystep = false,
    )

    u_end = solution.u[end]
    cavity_end = Array(@view u_end[1:2])

    Sp_col .= @view u_end[idx_Sp]
    Sm_col .= @view u_end[idx_Sm]
    Sz_col .= @view u_end[idx_Sz]

    return ComplexF64(cavity_end[1]), ComplexF64(cavity_end[2])
end

# ============================================================
# 7) FORWARD CURVE FROM ONE FIXED TIME
# ============================================================

function _direct_gpu_curve(
    jfix::Int,
    scan_indices::Vector{Int},
    workspace::CorrelationWorkspace,
    delta_gpu,
    g_gpu,
    Sp_gpu,
    Sz_gpu;
    stepper!::Function,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-8,
)
    nscan = length(scan_indices)

    Cadaga = zeros(ComplexF64, nscan)
    Cadagadag = zeros(ComplexF64, nscan)
    Cxx = zeros(ComplexF64, nscan)
    Cpp = zeros(ComplexF64, nscan)

    later_positions = findall(
        position -> scan_indices[position] >= jfix,
        eachindex(scan_indices),
    )

    isempty(later_positions) && return Cxx, Cpp, Cadaga, Cadagadag

    g_a0, g_adag0, gSp0, gSm0, gSz0 =
        _build_g0_adag_column_cpu(jfix, workspace)

    a_col = g_a0
    adag_col = g_adag0

    Sp_col = CuArray(gSp0)
    Sm_col = CuArray(gSm0)
    Sz_col = CuArray(gSz0)

    save_positions = later_positions
    save_indices = scan_indices[save_positions]

    pointer = 1
    last_needed = save_indices[end]

    sqrt_kappa_fixed = sqrt(workspace.kappa_e_full_us[jfix])
    adag_fixed = conj(workspace.a_sol[jfix])

    while pointer <= length(save_indices) && save_indices[pointer] == jfix
        output_position = save_positions[pointer]
        jscan = scan_indices[output_position]

        prefactor =
            sqrt_kappa_fixed * sqrt(workspace.kappa_e_full_us[jscan])

        Cadaga[output_position] = prefactor * (
            a_col + adag_fixed * workspace.a_sol[jscan]
        )

        Cadagadag[output_position] = prefactor * (
            adag_col + adag_fixed * conj(workspace.a_sol[jscan])
        )

        aa = Cadaga[output_position]
        aaqq = Cadagadag[output_position]

        Cxx[output_position] =
            (aaqq + conj(aaqq) + aa + conj(aa)) / 4

        Cpp[output_position] =
            (-aaqq - conj(aaqq) + aa + conj(aa)) / 4

        pointer += 1
    end

    for k in jfix:(last_needed - 1)
        dt = workspace.times_us[k + 1] - workspace.times_us[k]

        a_col, adag_col = stepper!(
            a_col,
            adag_col,
            Sp_col,
            Sm_col,
            Sz_col,
            k,
            dt,
            workspace,
            delta_gpu,
            g_gpu,
            Sp_gpu,
            Sz_gpu;
            reltol = reltol,
            abstol = abstol,
        )

        while pointer <= length(save_indices) && save_indices[pointer] == k + 1
            output_position = save_positions[pointer]
            jscan = scan_indices[output_position]

            prefactor =
                sqrt_kappa_fixed * sqrt(workspace.kappa_e_full_us[jscan])

            Cadaga[output_position] = prefactor * (
                a_col + adag_fixed * workspace.a_sol[jscan]
            )

            Cadagadag[output_position] = prefactor * (
                adag_col + adag_fixed * conj(workspace.a_sol[jscan])
            )

            aa = Cadaga[output_position]
            aaqq = Cadagadag[output_position]

            Cxx[output_position] =
                (aaqq + conj(aaqq) + aa + conj(aa)) / 4

            Cpp[output_position] =
                (-aaqq - conj(aaqq) + aa + conj(aa)) / 4

            pointer += 1
        end
    end

    return Cxx, Cpp, Cadaga, Cadagadag
end

# ============================================================
# 8) FORWARD AND SYMMETRIC CORRELATION MATRICES
# ============================================================

function _fixed_point_scan_curves_gpu(
    fixed_indices::Vector{Int},
    scan_indices::Vector{Int},
    workspace::CorrelationWorkspace,
    delta_gpu,
    g_gpu,
    Sp_gpu,
    Sz_gpu;
    stepper!::Function,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-8,
)
    nscan = length(scan_indices)
    nfix = length(fixed_indices)

    Cadaga_matrix = zeros(ComplexF64, nscan, nfix)
    Cadagadag_matrix = zeros(ComplexF64, nscan, nfix)
    Cxx_matrix = zeros(ComplexF64, nscan, nfix)
    Cpp_matrix = zeros(ComplexF64, nscan, nfix)

    for (column, jfix) in enumerate(fixed_indices)
        Cxx, Cpp, Cadaga, Cadagadag = _direct_gpu_curve(
            jfix,
            scan_indices,
            workspace,
            delta_gpu,
            g_gpu,
            Sp_gpu,
            Sz_gpu;
            stepper! = stepper!,
            reltol = reltol,
            abstol = abstol,
        )

        Cxx_matrix[:, column] .= Cxx
        Cpp_matrix[:, column] .= Cpp
        Cadaga_matrix[:, column] .= Cadaga
        Cadagadag_matrix[:, column] .= Cadagadag
    end

    return Cxx_matrix, Cpp_matrix, Cadaga_matrix, Cadagadag_matrix
end

function _fixed_point_scan_curves_symmetric_gpu(
    fixed_indices::Vector{Int},
    scan_indices::Vector{Int},
    workspace::CorrelationWorkspace,
    delta_gpu,
    g_gpu,
    Sp_gpu,
    Sz_gpu;
    stepper!::Function,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-8,
)
    nscan = length(scan_indices)
    nfix = length(fixed_indices)

    Cadaga_matrix = zeros(ComplexF64, nscan, nfix)
    Cadagadag_matrix = zeros(ComplexF64, nscan, nfix)
    Cxx_matrix = zeros(ComplexF64, nscan, nfix)
    Cpp_matrix = zeros(ComplexF64, nscan, nfix)

    for (column, jfix) in enumerate(fixed_indices)
        Cxx, Cpp, Cadaga, Cadagadag = _direct_gpu_curve(
            jfix,
            scan_indices,
            workspace,
            delta_gpu,
            g_gpu,
            Sp_gpu,
            Sz_gpu;
            stepper! = stepper!,
            reltol = reltol,
            abstol = abstol,
        )

        Cxx_matrix[:, column] .= Cxx
        Cpp_matrix[:, column] .= Cpp
        Cadaga_matrix[:, column] .= Cadaga
        Cadagadag_matrix[:, column] .= Cadagadag

        earlier_positions = findall(
            position -> scan_indices[position] < jfix,
            eachindex(scan_indices),
        )

        for output_position in earlier_positions
            jscan = scan_indices[output_position]

            _, _, Cadaga_forward, Cadagadag_forward = _direct_gpu_curve(
                jscan,
                [jfix],
                workspace,
                delta_gpu,
                g_gpu,
                Sp_gpu,
                Sz_gpu;
                stepper! = stepper!,
                reltol = reltol,
                abstol = abstol,
            )

            prefactor = sqrt(
                workspace.kappa_e_full_us[jscan] *
                workspace.kappa_e_full_us[jfix],
            )

            corr_forward_adag_a = Cadaga_forward[1] / prefactor
            corr_forward_adag_adag = Cadagadag_forward[1] / prefactor

            Cadaga_matrix[output_position, column] =
                prefactor * conj(corr_forward_adag_a)

            Cadagadag_matrix[output_position, column] =
                prefactor * corr_forward_adag_adag

            aa = Cadaga_matrix[output_position, column]
            aaqq = Cadagadag_matrix[output_position, column]

            Cxx_matrix[output_position, column] =
                (aaqq + conj(aaqq) + aa + conj(aa)) / 4

            Cpp_matrix[output_position, column] =
                (-aaqq - conj(aaqq) + aa + conj(aa)) / 4
        end
    end

    return Cxx_matrix, Cpp_matrix, Cadaga_matrix, Cadagadag_matrix
end

# ============================================================
# 9) PUBLIC CORRELATION CALCULATION
# ============================================================

"""
    compute_ase_rase_correlations_gpu(workspace, settings; verbose=true)

Compute all three correlation groups on the GPU:

- ASE-RASE with adaptive `Tsit5()`
- ASE-ASE with fixed-step RK4
- RASE-RASE with fixed-step RK4
"""
function compute_ase_rase_correlations_gpu(
    workspace::CorrelationWorkspace,
    settings::CorrelationSettings;
    verbose::Bool = true,
)
    _validate_settings(settings)
    CUDA.functional() || error("CUDA is not functional on this system.")
    CUDA.allowscalar(false)

    indices = _build_correlation_indices(
        workspace,
        settings;
        verbose = verbose,
    )

    Sp_gpu = nothing
    Sz_gpu = nothing
    delta_gpu = nothing
    g_gpu = nothing

    try
        Sp_gpu = CuArray(workspace.Sp_sol)
        Sz_gpu = CuArray(workspace.Sz_sol)
        delta_gpu = CuArray(workspace.delta_b_us)
        g_gpu = CuArray(workspace.g_b_us)

    verbose && println(
        "Running ASE-RASE correlations with adaptive Tsit5 ...",
    )

    CUDA.synchronize()
    start_AR = time_ns()

    Cxx_AR, Cpp_AR, Cadaga_AR, Cadagadag_AR =
        _fixed_point_scan_curves_gpu(
            indices.IA_fixed,
            indices.IR_scan,
            workspace,
            delta_gpu,
            g_gpu,
            Sp_gpu,
            Sz_gpu;
            stepper! = _tsit5_step_one_gpu!,
            reltol = settings.reltol,
            abstol = settings.abstol,
        )

    CUDA.synchronize()
    time_AR = (time_ns() - start_AR) / 1e9

    verbose && println(
        "Running ASE-ASE correlations with fixed-step RK4 ...",
    )

    start_AA = time_ns()

    Cxx_AA, Cpp_AA, Cadaga_AA, Cadagadag_AA =
        _fixed_point_scan_curves_symmetric_gpu(
            indices.IA_fixed,
            indices.IA_scan,
            workspace,
            delta_gpu,
            g_gpu,
            Sp_gpu,
            Sz_gpu;
            stepper! = _rk4_step_one_gpu!,
        )

    CUDA.synchronize()
    time_AA = (time_ns() - start_AA) / 1e9

    verbose && println(
        "Running RASE-RASE correlations with fixed-step RK4 ...",
    )

    start_RR = time_ns()

    Cxx_RR, Cpp_RR, Cadaga_RR, Cadagadag_RR =
        _fixed_point_scan_curves_symmetric_gpu(
            indices.IR_fixed,
            indices.IR_scan,
            workspace,
            delta_gpu,
            g_gpu,
            Sp_gpu,
            Sz_gpu;
            stepper! = _rk4_step_one_gpu!,
        )

    CUDA.synchronize()
    time_RR = (time_ns() - start_RR) / 1e9

    total_time = time_AR + time_AA + time_RR

    verbose && println("Finished all GPU correlation calculations.")
    verbose && println("ASE-RASE time : $time_AR seconds")
    verbose && println("ASE-ASE time  : $time_AA seconds")
    verbose && println("RASE-RASE time: $time_RR seconds")
    verbose && println("Total time    : $total_time seconds")

    tau_grid_us = collect(
        settings.tau_min_us:settings.dtau_us:settings.tau_max_us,
    )

    result = (
        source_file = workspace.source_file,

        settings = (
            tpi_us = settings.tpi_us,
            ASE_window_rel_us = settings.ASE_window_rel_us,
            RASE_window_rel_us = settings.RASE_window_rel_us,
            Nkeep_ASE = settings.Nkeep_ASE,
            Nkeep_RASE = settings.Nkeep_RASE,
            tA_choices_rel_us = settings.tA_choices_rel_us,
            tR_choices_rel_us = settings.tR_choices_rel_us,
            tau_min_us = settings.tau_min_us,
            tau_max_us = settings.tau_max_us,
            dtau_us = settings.dtau_us,
            reltol = settings.reltol,
            abstol = settings.abstol,
        ),

        solvers = (
            ASE_RASE = :Tsit5,
            ASE_ASE = :RK4,
            RASE_RASE = :RK4,
        ),

        timings = (
            ASE_RASE_seconds = time_AR,
            ASE_ASE_seconds = time_AA,
            RASE_RASE_seconds = time_RR,
            total_seconds = total_time,
        ),

        tau_grid_us = tau_grid_us,

        tA_scan_rel_us = indices.tA_scan,
        tR_scan_rel_us = indices.tR_scan,
        tA_fixed_rel_us = indices.tA_fixed,
        tR_fixed_rel_us = indices.tR_fixed,

        IA_scan_indices = indices.IA_scan,
        IR_scan_indices = indices.IR_scan,
        IA_fixed_indices = indices.IA_fixed,
        IR_fixed_indices = indices.IR_fixed,

        ASE_RASE = (
            Cxx = Cxx_AR,
            Cpp = Cpp_AR,
            Cadaga = Cadaga_AR,
            Cadagadag = Cadagadag_AR,
        ),

        ASE_ASE = (
            Cxx = Cxx_AA,
            Cpp = Cpp_AA,
            Cadaga = Cadaga_AA,
            Cadagadag = Cadagadag_AA,
        ),

        RASE_RASE = (
            Cxx = Cxx_RR,
            Cpp = Cpp_RR,
            Cadaga = Cadaga_RR,
            Cadagadag = Cadagadag_RR,
        ),
    )

        return result
    finally
        Sp_gpu = nothing
        Sz_gpu = nothing
        delta_gpu = nothing
        g_gpu = nothing
        GC.gc(false)
        CUDA.reclaim()
    end
end

# ============================================================
# 10) SAVE RESULTS
# ============================================================

"""
    save_correlation_results(output_file, correlation_data)

Save the returned correlation named tuple under the JLD2 key `data`.
"""
function save_correlation_results(
    output_file::AbstractString,
    correlation_data,
)
    output_path = abspath(output_file)
    mkpath(dirname(output_path))

    data = correlation_data
    JLD2.jldsave(output_path; data = data)

    return output_path
end

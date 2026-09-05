
optional_property(x, name::Symbol, default=nothing) =
    hasproperty(x, name) ? getproperty(x, name) : default


function prepare_peak_detection(SIM_SETTING, t_saved)
    peak_detection = optional_property(
        SIM_SETTING,
        :peak_detection,
        nothing,
    )

    peak_detection === nothing && return nothing

    peak_detection isa NamedTuple || throw(
        ArgumentError(
            "SIM_SETTING.peak_detection must be a NamedTuple or nothing."
        )
    )

    times_raw = optional_property(
        peak_detection,
        :times,
        nothing,
    )

    times_raw === nothing && throw(
        ArgumentError(
            "peak_detection must contain `times`."
        )
    )

    peak_times = if times_raw isa Number
        Float64[Float64(times_raw)]
    else
        Float64.(collect(times_raw))
    end

    isempty(peak_times) && return nothing

    Npeaks = length(peak_times)

    labels_raw = optional_property(
        peak_detection,
        :labels,
        nothing,
    )

    peak_labels = if labels_raw === nothing
        [Symbol("peak$(i)") for i in 1:Npeaks]
    elseif labels_raw isa Symbol
        Symbol[labels_raw]
    elseif labels_raw isa AbstractString
        Symbol[Symbol(labels_raw)]
    else
        Symbol.(collect(labels_raw))
    end

    length(peak_labels) == Npeaks || throw(
        ArgumentError(
            "The number of labels must equal the number of peak times."
        )
    )

    length(unique(peak_labels)) == Npeaks || throw(
        ArgumentError(
            "Every peak label must be unique."
        )
    )

    half_window_raw = optional_property(
        peak_detection,
        :half_window,
        nothing,
    )

    half_window_raw === nothing && throw(
        ArgumentError(
            "peak_detection must contain `half_window`."
        )
    )

    peak_half_windows = if half_window_raw isa Number
        fill(Float64(half_window_raw), Npeaks)
    else
        Float64.(collect(half_window_raw))
    end

    length(peak_half_windows) == Npeaks || throw(
        ArgumentError(
            "`half_window` must be one number or one value per peak."
        )
    )

    all(x -> x > 0, peak_half_windows) || throw(
        ArgumentError(
            "Every peak-search half-window must be positive."
        )
    )

    peak_window_indices = Vector{Vector{Int}}(undef, Npeaks)

    for ip in 1:Npeaks
        expected_time = peak_times[ip]
        half_window = peak_half_windows[ip]

        indices = findall(
            t -> abs(t - expected_time) <= half_window,
            t_saved,
        )

        isempty(indices) && throw(
            ArgumentError(
                "Peak window $(peak_labels[ip]) contains no saved points. " *
                "Center = $expected_time s, " *
                "half-window = $half_window s."
            )
        )

        if length(indices) < 3
            @warn(
                "Peak-search window contains fewer than three saved points",
                label = peak_labels[ip],
                number_of_points = length(indices),
            )
        end

        peak_window_indices[ip] = indices
    end

    return (
        labels = peak_labels,
        times = peak_times,
        half_windows = peak_half_windows,
        window_indices = peak_window_indices,
    )
end
# ============================================================
# Nested pairing, validation, and catalog replace.
#
# Layers:
#   1. Core: every gated system × one canonical pulse per family
#   2. Pulse depth: canonical system × every valid pulse design
#   3. Physics depth: every gated system × canonical RASE / ROSE / 3-ARP
#      (already contained in layer 1; kept as an explicit tag)
#
# --phase configs writes to data/datagen/configs.staging, then replaces
# data/datagen/configs/. See src/datagen/README.md.
# ============================================================

function try_bound_pulse(design, sys)
    try
        spec = bind_pulse(design, sys)
        derive_ttotal(sys, spec)
        pc = materialize_pulse_config(spec)
        ok, msg = pulse_config_is_valid(pc)
        ok || return nothing, "pulse invalid: $msg"
        return spec, ""
    catch err
        rethrow_interrupt(err)
        return nothing, sprint(showerror, err)
    end
end

function pair_key(sys, design)
    return (system_key(sys), design)
end

function canonical_designs_by_family(designs)
    best = Dict{Symbol, Any}()
    for d in designs
        fam = pulse_family_of(d)
        if d.canonical && !haskey(best, fam)
            best[fam] = d
        end
    end
    return best
end

function enumerate_pairs(systems, designs)
    canon_sys = canonical_system()
    canon_pulses = canonical_designs_by_family(designs)
    isempty(canon_pulses) && error("No canonical pulse designs found.")

    seen = Set{Any}()
    pairs = NamedTuple[]
    n_reject = 0

    function consider!(sys, design, layer::String)
        k = pair_key(sys, design)
        k in seen && return
        push!(seen, k)
        spec, _ = try_bound_pulse(design, sys)
        if spec === nothing
            n_reject += 1
            return
        end
        push!(pairs, (
            SYSTEM_CONFIG = sys,
            PULSE_SPEC = spec,
            design = design,
            layer = layer,
        ))
        return nothing
    end

    # Layer 1 / 3: all systems × canonical pulse of each family.
    # Sort families so catalog order (and --limit) is reproducible;
    # Dict iteration order is not.
    for sys in systems
        for fam in sort!(collect(keys(canon_pulses)))
            design = canon_pulses[fam]
            layer = fam in (:rase_wurst, :rose, :arp3) ? "core+physics" : "core"
            consider!(sys, design, layer)
        end
    end

    # Layer 2: canonical system × all pulse designs.
    for design in designs
        consider!(canon_sys, design, "pulse_depth")
    end

    return pairs, n_reject
end

function write_catalog!(pairs; dest_dir::AbstractString = DATAGEN_CONFIG_DIR, dry_run::Bool = false, limit::Int = 0)
    dry_run || mkpath(dest_dir)
    n = 0
    n_write = 0
    used = Set{String}()
    stems = String[]
    for (i, pair) in enumerate(pairs)
        limit > 0 && i > limit && break
        n += 1
        stem = uniquify_stem(datagen_stem(pair.SYSTEM_CONFIG, pair.PULSE_SPEC), used)
        push!(stems, stem)
        if !dry_run
            save_simulconfig(
                simulconfig_path(stem, dest_dir),
                pair.SYSTEM_CONFIG,
                pair.PULSE_SPEC,
            )
            n_write += 1
        end
    end
    return n, n_write, stems
end

const DATAGEN_CONFIG_STAGING = DATAGEN_CONFIG_DIR * ".staging"
const DATAGEN_STAGING_COMPLETE = "COMPLETE"

function _promote_staging_catalog!()
    isdir(DATAGEN_CONFIG_STAGING) || return nothing
    complete = isfile(joinpath(DATAGEN_CONFIG_STAGING, DATAGEN_STAGING_COMPLETE))
    staged = filter(f -> endswith(f, "_simulconfig.jld2"), readdir(DATAGEN_CONFIG_STAGING))
    if complete && !isempty(staged)
        ensure_datagen_dirs()
        @warn "promoting leftover staging catalog into $(DATAGEN_CONFIG_DIR)"
        for fname in staged
            mv(
                joinpath(DATAGEN_CONFIG_STAGING, fname),
                joinpath(DATAGEN_CONFIG_DIR, fname);
                force = true,
            )
        end
    end
    rm(DATAGEN_CONFIG_STAGING; recursive = true)
    return nothing
end

function replace_catalog!(pairs; limit::Int = 0)
    _promote_staging_catalog!()
    mkpath(DATAGEN_CONFIG_STAGING)
    n = 0
    n_write = 0
    stems = String[]
    try
        n, n_write, stems = write_catalog!(pairs; dest_dir = DATAGEN_CONFIG_STAGING, limit = limit)
        open(joinpath(DATAGEN_CONFIG_STAGING, DATAGEN_STAGING_COMPLETE), "w") do io
            write(io, "ok\n")
        end
        ensure_datagen_dirs()
        clear_existing_configs!()
        for fname in readdir(DATAGEN_CONFIG_STAGING)
            endswith(fname, "_simulconfig.jld2") || continue
            mv(
                joinpath(DATAGEN_CONFIG_STAGING, fname),
                joinpath(DATAGEN_CONFIG_DIR, fname);
                force = true,
            )
        end
    catch
        rethrow()
    end
    isdir(DATAGEN_CONFIG_STAGING) && rm(DATAGEN_CONFIG_STAGING; recursive = true)
    return n, n_write, stems
end

function clear_existing_configs!()
    isdir(DATAGEN_CONFIG_DIR) || return nothing
    for fname in readdir(DATAGEN_CONFIG_DIR)
        endswith(fname, "_simulconfig.jld2") || continue
        rm(joinpath(DATAGEN_CONFIG_DIR, fname); force=true)
    end
    return nothing
end

function phase_configs(; dry_run::Bool = false, limit::Int = 0)
    println("Enumerating SYSTEM_CONFIG catalog...")
    systems = enumerate_system_catalog()
    println("  admitted systems: $(length(systems))")

    println("Enumerating PULSE designs...")
    designs = all_pulse_designs()
    println("  pulse designs: $(length(designs))")
    canon = canonical_designs_by_family(designs)
    println("  canonical families: $(sort(collect(keys(canon))))")

    println("Pairing (core + pulse-depth)...")
    pairs, n_reject = enumerate_pairs(systems, designs)
    println("  unique validated (system, pulse) pairs: $(length(pairs))")
    println("  rejected at bind/Ttotal/pulse-validate: $n_reject")
    println("  ICs at simulate time: --default-conditions ground|equatorial|both")

    if dry_run
        n, n_write, stems = write_catalog!(pairs; dry_run = true, limit = limit)
        println("Dry run: would write $n simulconfig files under $(DATAGEN_CONFIG_DIR).")
    else
        n, n_write, stems = replace_catalog!(pairs; limit = limit)
        println("Wrote $n_write simulconfig files under $(DATAGEN_CONFIG_DIR).")
    end
    isempty(stems) || println("  example stem: $(stems[1])")
    return n
end

function phase_simulate(run; skip_existing::Bool = true, start_id::Int = 1, stop_id::Int = 0, limit::Int = 0)
    _promote_staging_catalog!()
    ensure_datagen_dirs()
    files = sort(filter(f -> endswith(f, "_simulconfig.jld2"), readdir(DATAGEN_CONFIG_DIR)))
    isempty(files) && error(
        "No simulconfig files in $(DATAGEN_CONFIG_DIR). Run --phase configs first."
    )

    println("Compute: $(describe_datagen_compute())")

    manifest = load_manifest()
    n_skipped = 0
    n_failed = 0
    n_done = 0

    planned = Any[]
    planned_by_stem = Dict{String, Any}()
    all_jobs = Any[]

    for (idx, fname) in enumerate(files)
        idx < start_id && continue
        stop_id > 0 && idx > stop_id && break
        limit > 0 && n_done >= limit && break

        stem = stem_from_simulconfig_path(fname)
        try
            entry = load_simulconfig(joinpath(DATAGEN_CONFIG_DIR, fname))

            entry_skip = skip_existing
            if skip_existing && haskey(manifest, stem)
                ver = manifest_run_rules_version(manifest[stem])
                if ver !== nothing && ver != RUN_RULES_VERSION
                    entry_skip = false
                    println("[$stem] re-running: run_rules_version changed")
                end
            end

            println()
            println("=" ^ 60)
            println("[$stem]")
            println(
                @sprintf(
                    "  family=%s  C_ens=%.3g  g=%s  freq=%s",
                    entry.PULSE_SPEC.family,
                    entry.SYSTEM_CONFIG.C_ens,
                    entry.SYSTEM_CONFIG.g_inhomogeneity.kind,
                    entry.SYSTEM_CONFIG.freq_inhomogeneity.kind,
                )
            )
            println("=" ^ 60)

            Ttotal, splits, jobs, reports, n_skip = plan_catalog_jobs(
                stem,
                entry.SYSTEM_CONFIG,
                entry.PULSE_SPEC,
                run;
                skip_existing = entry_skip,
            )
            println("  Ttotal=$(Ttotal * 1e6) us  n_splits=$(length(splits))  pending=$(length(jobs))  skipped=$n_skip")
            if run.n_sizes > 1 && length(splits) < run.n_sizes
                println("  note: M-sizing=$(run.n_sizes) collapsed to $(length(splits)) unique (M_delta, M_g) grid(s)")
            end
            for split in splits
                println(
                    @sprintf(
                        "  split  M_delta=%d  M_g=%d  M=%d  safety=%.3g  target=%.3g",
                        split.M_delta, split.M_g, split.M_total, split.safety_factor, split.target_safety,
                    )
                )
            end

            append!(all_jobs, jobs)
            item = (
                stem = stem,
                family = String(entry.PULSE_SPEC.family),
                reports = reports,
                n_skipped = n_skip,
            )
            push!(planned, item)
            planned_by_stem[stem] = item
            n_skipped += n_skip
        catch err
            rethrow_interrupt(err)
            msg = sprint(showerror, err)
            println("[$stem] FAILED: ", msg)
            n_failed += 1
        end
        n_done += 1
    end

    function on_job_complete(job, st)
        lock(_DATAGEN_MANIFEST_LOCK) do
            item = get(planned_by_stem, job.stem, nothing)
            item === nothing && return nothing
            merge_job_outcomes!(item.reports, [st])
            return nothing
        end
    end

    outcomes = Any[]
    pool_err = nothing
    try
        outcomes = run_datagen_jobs!(all_jobs; on_complete = on_job_complete)
    catch err
        pool_err = err
        if is_interrupt(err)
            println()
            println(
                "Interrupted. Writing manifest for completed jobs. " *
                "Resume with --phase simulate (do not re-run --phase configs)."
            )
        else
            rethrow_interrupt(err)
            @error "datagen job pool failed" exception=err
        end
    end

    by_stem = Dict{String, Vector{Any}}()
    for st in outcomes
        push!(get!(Vector{Any}, by_stem, st.stem), st)
    end

    n_ok = 0
    n_job_failed = 0
    for item in planned
        stem_out = get(by_stem, item.stem, Any[])
        lock(_DATAGEN_MANIFEST_LOCK) do
            merge_job_outcomes!(item.reports, stem_out)
        end
        ok, failed = count_job_report_statuses(item.reports)
        n_ok += ok
        n_job_failed += failed
        try
            lock(_DATAGEN_MANIFEST_LOCK) do
                manifest[item.stem] = Dict{String, Any}(
                    "stem" => item.stem,
                    "family" => item.family,
                    "run_rules_version" => RUN_RULES_VERSION,
                    "run_params" => run_params_fingerprint(run),
                    "ics" => item.reports,
                )
                save_manifest(manifest)
            end
        catch err
            rethrow_interrupt(err)
            println("[$(item.stem)] FAILED to write manifest: ", sprint(showerror, err))
        end
    end
    n_failed += n_job_failed

    println()
    if pool_err !== nothing && is_interrupt(pool_err)
        println(
            "Simulate interrupted: $n_ok ok, $n_skipped skipped, $n_failed failed " *
            "($n_done catalog entries visited). Resume with --phase simulate."
        )
    elseif n_done == 0
        println(
            "Simulate finished: no catalog entries in range " *
            "(start=$start_id stop=$(stop_id == 0 ? "end" : stop_id) " *
            "limit=$(limit == 0 ? "none" : limit), $(length(files)) files on disk)."
        )
    else
        println(
            "Simulate finished: $n_ok ok, $n_skipped skipped, $n_failed failed " *
            "($n_done catalog entries visited)."
        )
    end
    println("Manifest: $DATAGEN_MANIFEST")
    pool_err !== nothing && rethrow_interrupt(pool_err)
    return (n_ok = n_ok, n_skipped = n_skipped, n_failed = n_failed, n_done = n_done)
end

# ============================================================
# Nested pairing, validation, and simulconfig writing.
#
# Layers:
#   1. Core: every gated system × one canonical pulse per family
#   2. Pulse depth: canonical system × every valid pulse design
#   3. Physics depth: every gated system × canonical RASE / ROSE / 3-ARP
#      (already contained in layer 1; kept as an explicit tag)
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

function write_catalog!(pairs; dry_run::Bool = false, limit::Int = 0)
    dry_run || ensure_datagen_dirs()
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
                simulconfig_path(stem),
                pair.SYSTEM_CONFIG,
                pair.PULSE_SPEC,
            )
            n_write += 1
        end
    end
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

    if !dry_run
        clear_existing_configs!()
    end
    n, n_write, stems = write_catalog!(pairs; dry_run = dry_run, limit = limit)
    if dry_run
        println("Dry run: would write $n simulconfig files under $(DATAGEN_CONFIG_DIR).")
    else
        println("Wrote $n_write simulconfig files under $(DATAGEN_CONFIG_DIR).")
    end
    isempty(stems) || println("  example stem: $(stems[1])")
    return n
end

function phase_simulate(run; skip_existing::Bool = true, start_id::Int = 1, stop_id::Int = 0, limit::Int = 0)
    ensure_datagen_dirs()
    files = sort(filter(f -> endswith(f, "_simulconfig.jld2"), readdir(DATAGEN_CONFIG_DIR)))
    isempty(files) && error(
        "No simulconfig files in $(DATAGEN_CONFIG_DIR). Run --phase configs first."
    )

    manifest = load_manifest()
    n_ok = 0
    n_skipped = 0
    n_failed = 0
    n_done = 0

    for (idx, fname) in enumerate(files)
        idx < start_id && continue
        stop_id > 0 && idx > stop_id && break
        limit > 0 && n_done >= limit && break

        entry = load_simulconfig(joinpath(DATAGEN_CONFIG_DIR, fname))
        stem = stem_from_simulconfig_path(fname)

        entry_skip = skip_existing
        if skip_existing && haskey(manifest, stem)
            prev = manifest[stem]
            ver = manifest_run_rules_version(prev)
            params_ok = run_params_match_manifest(prev, run)
            if (ver !== nothing && ver != RUN_RULES_VERSION) || !params_ok
                entry_skip = false
                println("[$stem] re-running: run_rules/params changed")
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

        ok, skipped, failed, reports = simulate_catalog_entry(
            stem,
            entry.SYSTEM_CONFIG,
            entry.PULSE_SPEC,
            run;
            skip_existing = entry_skip,
        )
        n_ok += ok
        n_skipped += skipped
        n_failed += failed
        n_done += 1

        manifest[stem] = Dict{String, Any}(
            "stem" => stem,
            "family" => String(entry.PULSE_SPEC.family),
            "run_rules_version" => RUN_RULES_VERSION,
            "run_params" => run_params_fingerprint(run),
            "ics" => reports,
        )
        save_manifest(manifest)
    end

    println()
    if n_done == 0
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
    return nothing
end

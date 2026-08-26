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
        Ttotal = derive_ttotal(sys, spec)
        pc = materialize_pulse_config(spec)
        ok, msg = pulse_config_is_valid(pc)
        ok || return nothing, "pulse invalid: $msg"
        return spec, ""
    catch err
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

    seen = Dict{Any,Bool}()
    pairs = NamedTuple[]

    function consider!(sys, design, layer::String)
        k = pair_key(sys, design)
        haskey(seen, k) && return
        spec, err = try_bound_pulse(design, sys)
        spec === nothing && return
        seen[k] = true
        push!(pairs, (
            SYSTEM_CONFIG = sys,
            PULSE_SPEC = spec,
            design = design,
            layer = layer,
        ))
        return nothing
    end

    # Layer 1 / 3: all systems × canonical pulse of each family.
    for sys in systems
        for (fam, design) in canon_pulses
            layer = fam in (:rase_wurst, :rose, :arp3) ? "core+physics" : "core"
            consider!(sys, design, layer)
        end
    end

    # Layer 2: canonical system × all pulse designs.
    for design in designs
        consider!(canon_sys, design, "pulse_depth")
    end

    return pairs
end

function write_catalog!(pairs; dry_run::Bool = false, limit::Int = 0)
    ensure_datagen_dirs()
    stamp = git_commit_stamp()
    n = 0
    n_write = 0
    for (i, pair) in enumerate(pairs)
        limit > 0 && i > limit && break
        n += 1
        Ttotal = derive_ttotal(pair.SYSTEM_CONFIG, pair.PULSE_SPEC)
        metadata = (
            run_id = i,
            run_rules_version = RUN_RULES_VERSION,
            git_commit = stamp,
            layer = pair.layer,
            family = pair.PULSE_SPEC.family,
            derived_Ttotal = Ttotal,
            M_delta = RULE_M_DELTA,
            M_g = run_rule_M_g(pair.SYSTEM_CONFIG),
        )
        if !dry_run
            save_simulconfig(
                simulconfig_path(i),
                pair.SYSTEM_CONFIG,
                pair.PULSE_SPEC,
                metadata,
            )
            n_write += 1
        end
    end
    return n, n_write
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
    pairs = enumerate_pairs(systems, designs)
    println("  unique validated (system, pulse) pairs: $(length(pairs))")
    println("  simulations if executed: $(2 * length(pairs))  (ground + equator)")

    n, n_write = write_catalog!(pairs; dry_run = dry_run, limit = limit)
    if dry_run
        println("Dry run: would write $n simulconfig files under $(DATAGEN_CONFIG_DIR).")
    else
        println("Wrote $n_write simulconfig files under $(DATAGEN_CONFIG_DIR).")
    end
    return n
end

function phase_simulate(; skip_existing::Bool = true, start_id::Int = 1, stop_id::Int = 0)
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

    for fname in files
        entry = load_simulconfig(joinpath(DATAGEN_CONFIG_DIR, fname))
        run_id = Int(entry.metadata.run_id)
        run_id < start_id && continue
        stop_id > 0 && run_id > stop_id && continue

        println()
        println("=" ^ 60)
        println(
            @sprintf(
                "[run_%06d] family=%s  C_ens=%.3g  g=%s  freq=%s",
                run_id,
                entry.PULSE_SPEC.family,
                entry.SYSTEM_CONFIG.C_ens,
                entry.SYSTEM_CONFIG.g_inhomogeneity.kind,
                entry.SYSTEM_CONFIG.freq_inhomogeneity.kind,
            )
        )
        println("=" ^ 60)

        ok, skipped, failed, reports = simulate_catalog_entry(
            run_id,
            entry.SYSTEM_CONFIG,
            entry.PULSE_SPEC;
            skip_existing = skip_existing,
        )
        n_ok += ok
        n_skipped += skipped
        n_failed += failed
        n_done += 1

        manifest[@sprintf("run_%06d", run_id)] = Dict{String, Any}(
            "run_id" => run_id,
            "family" => String(entry.PULSE_SPEC.family),
            "run_rules_version" => RUN_RULES_VERSION,
            "ics" => reports,
        )
        save_manifest(manifest)
    end

    println()
    println(
        "Simulate finished: $n_ok ok, $n_skipped skipped, $n_failed failed " *
        "($n_done catalog entries visited)."
    )
    println("Manifest: $DATAGEN_MANIFEST")
    return nothing
end

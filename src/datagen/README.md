# Datagen

`src/datagen/` is a **script package**, not part of the
`InhomogeneousSpinCavityDynamics` module. It builds a first-order
trajectory catalog for PINN training. It does not change solver physics:
it enumerates `(SYSTEM_CONFIG, PULSE_SPEC)` pairs, chooses the ensemble
split at simulate time, and calls `run_sim_1st_order_trajectory`.

| File | Role |
| --- | --- |
| `datagen_run.jl` | CLI module `DataGen` (`--help`, `main`) |
| `datagen_io.jl` | paths, run-rule constants, stems, manifest, skip |
| `datagen_system.jl` | system catalog and microwave-cavity gate |
| `datagen_pulse.jl` | pulse design tables, binding, `Ttotal` |
| `datagen_execute.jl` | split, job pool, trajectory, reduced save, GPU reclaim |
| `datagen_catalog.jl` | pairing, `--phase configs`, `--phase simulate` |
| `datagen_selftest.jl` | correctness and stress tests (does not write `data/datagen/`) |

On disk the catalog lives at `data/datagen/` (resolved from this folder as
`../../data/datagen`). `--phase configs --dry-run` prints the current
admitted-system / design / pair counts; those numbers are not frozen in
this file.

---

## User guide

### What you run

`--phase` is **required**. No arguments, or a missing `--phase`, is an
error. That is deliberate: configs **replaces** the live simulconfig
catalog.

From the repository root, after `Pkg.instantiate()`:

```bash
julia --project=. src/datagen/datagen_run.jl --help

# Count pairs; write nothing.
julia --project=. src/datagen/datagen_run.jl --phase configs --dry-run

# Replace data/datagen/configs/ with a new catalog.
julia --project=. src/datagen/datagen_run.jl --phase configs

# Integrate pending jobs. Threads ≥ number of CUDA GPUs.
julia -t auto --project=. src/datagen/datagen_run.jl --phase simulate
```

`--phase all` is configs **then** simulate. `--limit` on `all` applies to
**both**: configs writes at most *N* simulconfigs (after replacing the
catalog), then simulate visits at most *N* files. That is not “simulate
*N* of the existing catalog.”

Self-test (writes only under `/tmp`; not hooked to `Pkg.test()`):

```bash
julia --project=. src/datagen/datagen_selftest.jl
julia --project=. src/datagen/datagen_selftest.jl --quick      # no full pairing, no live ODE
julia --project=. src/datagen/datagen_selftest.jl --skip-ode   # pairing yes, live ODE no
```

### Two phases

**`configs`** enumerates every admitted `(system, pulse)` pair and writes
one `{stem}_simulconfig.jld2`. The file contains **only**
`SYSTEM_CONFIG` and `PULSE_SPEC`. Caps, initial conditions, `Nt_save`,
and Fourier safety are **not** stored.

**`simulate`** loads those files, derives `Ttotal`, chooses
`(M_delta, M_g)` from the **current** CLI flags, and runs each pending
`(IC, split)` job. If any catalog entry failed to plan or any trajectory
failed, the process **exits 1**.

Resume a killed simulate with `--phase simulate` and the **same** flags
(`--M-cap`, `--M-g-cap`, `--M-sizing`, `--NT-save`,
`--default-conditions`). Do **not** re-run `--phase configs` to resume:
that replaces the catalog. Result JLD2s are not deleted by configs;
stems that still match will skip on the next simulate.

### CLI

| Flag | Phase | Default | Meaning |
| --- | --- | --- | --- |
| `--phase configs\|simulate\|all` | both | none (required) | which work to do |
| `--help`, `-h` | both | | print usage and exit 0 |
| `--dry-run` | configs | off | enumerate and print; write nothing. Illegal with `--phase simulate` |
| `--limit N` | both | `0` (none) | configs: first *N* pairs written. simulate: first *N* catalog **files** in the window |
| `--start IDX` | simulate | `1` | 1-based index into the **sorted** `configs/` listing |
| `--stop IDX` | simulate | `0` | inclusive end index; `0` means last file |
| `--no-skip` | simulate | skip on | overwrite complete result pairs |
| `--default-conditions ground\|equatorial\|both` | simulate | `both` | ICs. CLI `equatorial` is solver `:equator` |
| `--conditions` | simulate | same | alias of `--default-conditions` |
| `--M-cap N` | simulate | `60000` | `M_delta * M_g ≤ N` |
| `--M-g-cap N` | simulate | `30` | `M_g ≤ N` when *g* is not constant |
| `--M-sizing default\|N` | simulate | `default` | `default` or `1`: one max-safety split. integer `N>1`: *N* target safeties on `(3, S]` |
| `--NT-save N` | simulate | `5001` | saved time samples; must be `> 1` |

`--start` / `--stop` are ignored during configs. Rel/abs ODE tolerances
and simulation order are **not** flags (`1e-8`, `:order1`).

### Recommended first runs

Smoke one catalog file after configs exists (cheap IC, one stem):

```bash
julia -t auto --project=. src/datagen/datagen_run.jl --phase simulate \
  --start 1 --stop 1 --default-conditions ground
```

Default `--M-cap 60000` and `--NT-save 5001` make host `Sp`/`Sz` on the
order of **10 GB** per job (`Nt × M` complex, twice), plus the solver,
plus the reduced copy. A 16 GB node will OOM. Prove the machine with a
smaller cap before a full campaign, for example `--M-cap 8000 --NT-save 501`.

Change discretization later by keeping the catalog and passing new
`--M-cap` / `--M-sizing` / `--NT-save`. New signatures are new files; old
files stay. Rebuild the physics list with `--phase configs` only when
systems or pulses in source changed.

### GPU and threads

If `CUDA.functional()` is true, every pending ODE is `compute=:gpu`.
There is no silent GPU→CPU fallback on OOM: that job is `failed`, the
worker reclaims the device, and the pool continues. There is no VRAM
preflight.

Occupancy is **one in-flight trajectory per functional CUDA device**.
*K* concurrent GPUs need `julia -t K` or `-t auto`. With 1 Julia thread
and *K* GPUs, jobs run sequentially on GPU 0; the other devices stay idle
(a warning is printed when `nthreads() < n_gpu`).

If no functional CUDA device exists, workers are CPU threads:
`min(n_jobs, nthreads())`. When more than one worker runs, BLAS is
pinned to 1 thread for the pool.

### Output layout

```text
data/datagen/
  configs/{stem}_simulconfig.jld2
  configs.staging/          # ephemeral; sibling of configs/, not inside it
  results/{stem}_{ground|equator}_Md{M_delta}_Mg{M_g}_Nt{Nt_save}.jld2
  results/{stem}_{ground|equator}_Md{M_delta}_Mg{M_g}_Nt{Nt_save}_pulsemat.csv
  manifest.json
```

Linux `NAME_MAX` is 255 bytes. Catalog stems in the current tables are
well under that (self-test max was 97 bytes). The result basename adds
`_{ic}_Md…_Mg…_Nt….jld2` and `_pulsemat.csv`.

---

## Loading a result

Each successful job writes a JLD2/CSV pair. The JLD2 `data` named tuple
contains:

| Field | Meaning |
| --- | --- |
| `t_saved`, `a_sol`, `Σp_sol`, `Σz_sol`, `E_of_t_arr` | trajectory and drive samples on `t_saved` |
| `delta_b_1d`, `g_b_1d`, `Nj_2d`, `N_total` | 1-D grids and bin occupations |
| `idelta_res`, `delta_res`, `keep_bins`, `g_keep`, `delta_keep` | resonant-δ slice metadata (`g_keep` is length `M_g`) |
| `Sp_keep`, `Sz_keep` | resonant-δ trajectories, shape `(M_g, Nt)` — same convention as `run_sim_1st_order` |
| `SIM_SETTING`, `SYSTEM_CONFIG` | merge these for `prepare_derived` |
| `PULSE_SPEC` | serializable pulse (always the rebuild source for composites) |
| `PULSE_CONFIG` | `PULSE_SPEC.segments` (see below) |
| `pulse_rebuild` | `"pulse_config"` or `"pulse_spec"` |
| `elapsed_seconds`, `run_rules_version`, `run_params` | bookkeeping |
| `safety_factor`, `target_safety`, `M_delta_min` | split metadata |

`PULSE_CONFIG` is **not** always legal for the package
`build_E_of_t` / `validate_pulse_config`:

- WURST and Gaussian segments (`pulse_rebuild == "pulse_config"`): pass
  `data.PULSE_CONFIG` to `build_E_of_t` as in a normal run.
- HS1, CORPSE, BB1, block-π, random composite (`pulse_rebuild == "pulse_spec"`):
  segments are `:composite_record`. Closures are not stored. Rebuild with
  datagen’s `materialize_pulse_config(data.PULSE_SPEC)` then
  `build_E_of_t`, or ignore the JLD2 drive and use the sibling
  `_pulsemat.csv` (`load_mode=:csv` in `jld2_pulse_loader.jl`).

The pulsemat CSV is `sample_E_of_t` on `[0, Ttotal]` with `Nt_save`
points: a `# t_end_us,…` line, then `Re,Im`.

Peak detection fields are stored as `nothing` (datagen does not run the
package peak detector).

---

## How configs is built

### Systems

Physical coordinates (`κ_t` in Hz, overcoupling *r*, `FWHM/κ_t`,
`C_ens`, single-spin *g*, …) are mapped to the package `NamedTuple`, then
admitted only if `physics_gate_reason` returns `nothing`:

- `g/κ_t ≤ 10⁻²`, `g/FWHM ≤ 10⁻²`
- `FWHM/κ_t ∈ [0.2, 5]`, `r ∈ [0.5, 1]`, `C_ens ∈ [0.05, 1]`
- implied spin number `N ∈ [10⁴, 10¹⁶]`
- Gaussian *g* truncated onto `g > 0` (`mean - span_sigma·std > 0`)
- `validate_config` on a dummy first-order setting (`M_delta=8`, …)

Only `:lorentzian` / `:gaussian` frequency and `:constant` / `:gaussian`
coupling are catalogued. The product is three overlapping grids at or
around the canonical cavity (`κ_t/2π = 1 MHz`, `r = 1`, `FWHM/κ_t = 1`,
`C_ens = 0.6`, `g/2π = 100 Hz`, Lorentzian, constant *g*), plus
`C_ens`, line-shape, *g*-spec, `κ_t`, overcoupling, detuning, and
single-spin *g* sweeps defined in `datagen_system.jl`. Duplicates are
dropped by `system_key`.

### Pulses

Design tables are pulse-intrinsic (times, dimensionless multipliers).
Binding to a system sets amplitudes and bandwidths from `κ_e` and FWHM.

Families: RASE WURST, ROSE (paper two-WURST and gap-parameterized),
3-ARP ± Gaussian signal, HS1 / CORPSE / BB1 seeds, block-π, random
composites.

`Ttotal` is **derived**, not stored:

```text
Ttotal = max(drive support end, echo window end) + t_settle
t_settle = max(10/κ_t, 3/FWHM, 1 µs)
```

Drive support is 5σ for Gaussians and the WURST duration about
`t_center`. ROSE and 3-ARP-with-signal add a 10 µs half-window past the
predicted echo. The Gaussian left tail must not start before `t = 0`.

`:custom` closures are built only in `materialize_pulse_config` at
simulate time so simulconfig JLD2s stay serializable. Composite pulses
are stored as `:composite_record` plus the packed `u`.

### Pairing

`enumerate_pairs` keeps unique `(system_key, design)`:

1. Every gated system × the canonical design of each family. RASE, ROSE,
   and 3-ARP (no signal) are tagged `core+physics`; other families
   `core`.
2. The canonical system × every design (`pulse_depth`).

A pair is written only if `bind_pulse`, `derive_ttotal`, and
`validate_pulse_config` on the materialized segments all succeed.
Rejects are counted, not written.

### Catalog replace (configs)

Unless `--dry-run`:

1. Recover any leftover **complete** staging directory
   (`data/datagen/configs.staging` with a `COMPLETE` file) into
   `configs/`.
2. Write the new catalog into `configs.staging`.
3. Write the `COMPLETE` marker.
4. Delete live `configs/*_simulconfig.jld2`.
5. Move staging files into `configs/`.
6. Remove the staging directory.

A crash **before** `COMPLETE` leaves the old catalog; incomplete staging
is discarded on the next run. A crash **after** `COMPLETE` but during
the swap is repaired by promoting remaining staging files the next time
configs or simulate starts. A crash **during** step 4 can leave extra
old stems mixed with the new set; that is the residual hole. Do not use
configs as a resume mechanism.

`--phase configs --limit N` still **replaces** the live catalog; the new
catalog has at most *N* files.

Stem collisions after sanitizing get `_2`, `_3`, ….

---

## How simulate chooses the grid

Let `BW` be the frequency-inhomogeneity support used by the splitter
(Gaussian: `2 * span_sigma * σ` with `σ` from FWHM;
Lorentzian: `2 * span_gamma * (FWHM/2)`).

```text
M_delta_min = Ttotal * BW / 2π
safety      = M_delta / M_delta_min
```

Hard floor: `safety ≥ 3` (`RULE_SAFETY_MIN`). If that cannot fit under
`M_cap`, the split **errors**. That catalog entry is `failed`; the rest
of the window continues. There is no clamp below 3.

**Default** (`--M-sizing default` or `1`):

- constant *g*: `M_g = 1`, `M_delta = M_cap`
- inhomogeneous *g*: largest `M_g ≤ M_g-cap` that still allows safety ≥ 3,
  then largest `M_delta` with `M_delta * M_g ≤ M_cap`

**`--M-sizing n` with `n > 1`:** *n* target safeties equally spaced on
`(3, S]`, including `S` (the default split’s safety). The floor `3` is
not a sample. Duplicate `(M_delta, M_g)` pairs collapse; the last
(highest-safety / default) metadata wins. If collapse happens, simulate
prints how many unique grids remain.

Solver ICs are `:ground` and/or `:equator`.

---

## Skip, resume, and recovery

A job is skipped when **both** paths exist and `filesize > 0`:

```text
{stem}_{ground|equator}_Md{M_delta}_Mg{M_g}_Nt{Nt_save}.jld2
{stem}_{ground|equator}_Md{M_delta}_Mg{M_g}_Nt{Nt_save}_pulsemat.csv
```

Skip does not open the JLD2. A non-empty truncated file counts as
complete. `--no-skip` overwrites.

If `manifest.json` has that stem with `run_rules_version` other than the
current `"6"`, the stem is forced to re-run even if files look complete.
Bump `RUN_RULES_VERSION` only when simulation semantics change (not for
docs or the GPU pool).

Results, simulconfigs, and the manifest are written as `*.part` then
`mv`. If the CSV `mv` fails after the JLD2 `mv`, the dest JLD2 is
removed so skip cannot accept a half pair.

`phase_simulate` plans the whole window, then `run_datagen_jobs!`: one
worker per GPU (or per CPU thread), a `Channel` of job indices, each GPU
worker pinning `CUDA.device!(devices[wid])` once.

| Event | What happens |
| --- | --- |
| ODE / IO error | job `failed`; worker continues; that device reclaimed |
| GPU OOM | same; **not** retried on CPU |
| Worker death | siblings drain the queue; never-started indices become `failed` (“job was not started”) |
| Ctrl-C / `InterruptException` (including from `@spawn`) | stop flag; in-flight job `failed`; unstarted filled; all GPUs reclaimed; manifest written for completed work; exception thrown. Resume with `--phase simulate` |
| Kill during `*.part` write | parts deleted on the next save attempt; dest incomplete → not skipped |
| Kill after both `mv`s | next simulate skips that signature |
| Unreadable `manifest.json` | warning; empty manifest; skip still uses files |
| Planning error (bad simulconfig, safety floor) | that stem `failed`; window continues |

Every outcome slot is assigned before the pool returns or the interrupt
is thrown. In-memory per-stem reports update as jobs finish. After the
pool (and on interrupt) `manifest.json` is rewritten. **Files are the
source of truth for skip.** The manifest is bookkeeping (`run_rules_version`,
status text). Do not treat a missing or stale manifest entry as “the
file does not exist.”

After every job the owning worker does `CUDA.synchronize()`,
`GC.gc(false)`, `CUDA.reclaim()` on **that** device. When the pool
exits it does the same on every listed device and restores the caller’s
device. Host `t, a, Sp, Sz` and `d` are dropped after the reduce,
before the JLD2 write. Reclaim helpers never throw.

---

## `manifest.json`

One object per stem, keys are stems:

```json
{
  "stem": "...",
  "family": "rase_wurst",
  "run_rules_version": "6",
  "run_params": {
    "ics": ["ground", "equator"],
    "M_cap": 60000,
    "M_g_max": 30,
    "n_sizes": 1,
    "Nt_save": 5001
  },
  "ics": {
    "ground_Md60000_Mg1_Nt5001": {
      "status": "ok",
      "path": "...",
      "elapsed_seconds": 12.3,
      "M_delta": 60000,
      "M_g": 1,
      "safety_factor": 294.1,
      "worker": "gpu 0"
    }
  }
}
```

`status` is `ok`, `skipped`, or `failed`. Failed entries store `error`
instead of the timing fields.

---

## Stems

`{system_slug}_{pulse_slug}`, then `[^A-Za-z0-9_]+` → `_`. Tokens follow
`data/data_1st_order` style (`0p05`, `1em06`, units glued on).

Pulse slugs include the fields that actually differ in the tables:
RASE amplitude / duration / bandwidth / optional `n`, chirp, `omega0`,
phase; ROSE gaps or, for the paper two-center form, `c1`/`c2`; 3-ARP
budget / start / bandwidth / `omega`, and for signal-on designs
`signal_amp_mult` when it is not `1`; composite `k`, `n_coeff`, seed,
`T_max`, optional taper / HS1 `beta`.

If the sanitized stem exceeds `DATAGEN_STEM_MAXLEN` (180) **bytes**, it
is truncated and tagged with a process-stable FNV-1a 8-hex digest
(`hash()` is not used; it is per-process randomized).

---

## Run-rule constants

`datagen_io.jl`. Do not fork these in callers.

| Name | Value | Role |
| --- | --- | --- |
| `RUN_RULES_VERSION` | `"6"` | stem-wide re-run if the manifest disagrees |
| `RULE_SIMULATION_ORDER` | `:order1` | first-order only |
| `RULE_NT_SAVE` | `5001` | default `--NT-save` |
| `RULE_RELTOL` / `RULE_ABSTOL` | `1e-8` | ODE tolerances |
| `RULE_M_CAP` | `60000` | default `--M-cap` |
| `RULE_M_G_MAX` | `30` | default `--M-g-cap` |
| `RULE_SAFETY_MIN` | `3.0` | Fourier safety floor |
| `DATAGEN_ICS` | `(:ground, :equator)` | default IC pair |
| `DATAGEN_STEM_MAXLEN` | `180` | stem truncation threshold (bytes) |

---

## What this does not do

- Second-order or multi-GPU cumulant simulations
- Pulse optimisation (`pulse_optimizer2.jl`)
- Store `M_delta` / `M_g` / safety in simulconfigs
- Clamp safety below 3
- Fall back GPU→CPU when `compute=:gpu` OOMs
- Treat a lone JLD2 (no pulsemat) as complete
- Open the JLD2 to decide skip (non-empty bytes are enough)
- Preflight GPU memory
- Hook into `Pkg.test()` (`datagen_selftest.jl` is a separate script)
- Use `--phase configs` as resume
- Guarantee a mixed catalog cannot appear if configs is killed in the
  middle of deleting old simulconfigs
)

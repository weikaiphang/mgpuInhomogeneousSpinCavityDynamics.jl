# ============================================================================
#  pulse_report.jl  --  seed-vs-optimised pulse report (self-contained HTML)
#
#  write_pulse_report(<stem-or-.jld2-path>; data_dir=<auto>, out_dir=<auto>,
#                     log_dir=<data_dir>, param_budget=120, runtime_s=nothing,
#                     verbose=true) -> outpath
#
#  Reads  <data_dir>/<stem>.jld2 (+ <log_dir>/_optrunlog.jld2 + _opt_pulsepara.jld2 if present)
#  and writes a standalone page to  <out_dir>/<stem>.html  (see module docs).
#  Called automatically at the end of optimise_control_pulse_from_jld2 (and by
#  scripts/pulse_report.jl on the CLI). Uses only names already in the module.
# ============================================================================

# minimal JSON encoder for the exact payload shapes this file produces
_jsonval(::Nothing) = "null"
_jsonval(x::Bool) = x ? "true" : "false"
_jsonval(x::Integer) = string(x)
_jsonval(x::Real) = isfinite(x) ? string(x) : "null"
function _jsonval(x::AbstractString)
    s = replace(String(x), '\\' => "\\\\", '"' => "\\\"")
    return string('"', s, '"')
end
_jsonval(x::Tuple) = _jsonval(collect(x))
_jsonval(x::AbstractVector) = string("[", join((_jsonval(v) for v in x), ","), "]")
_jsonval(x::AbstractDict) =
    string("{", join((string(_jsonval(String(k)), ":", _jsonval(v)) for (k, v) in x), ","), "}")

# seconds -> "6h 48m 12s" / "48m 12s" / "12.3s"
function fmt_dur(s)
    s === nothing && return "—"
    s = Float64(s)
    h = floor(Int, s / 3600); m = floor(Int, (s % 3600) / 60); sec = s - 3600h - 60m
    h > 0   ? @sprintf("%dh %dm %ds", h, m, round(Int, sec)) :
    m > 0   ? @sprintf("%dm %ds", m, round(Int, sec)) :
              @sprintf("%.1fs", sec)
end

# ------------------------------------------------------------- helpers --------
sig(x, n=5) = x === nothing ? "—" : (isfinite(x) ? string(round(x; sigdigits=n)) : string(x))
us(x)       = @sprintf("%.2f", x * 1e6)               # seconds -> microseconds string
twopi_mhz(x)= @sprintf("2π × %.4g MHz", x / (2π * 1e6))
twopi_hz(x) = @sprintf("2π × %.4g Hz",  x / (2π))

# unwrap a phase vector in place-ish, return new
function _unwrap(ph)
    out = copy(ph)
    @inbounds for k in 2:length(out)
        d = out[k] - out[k-1]
        d >  π && (out[k:end] .-= 2π)
        d < -π && (out[k:end] .+= 2π)
    end
    return out
end

"""sample a t->Complex drive: returns (A, re, im, f) with f masked (NaN) where |E|<1e-3·peak."""
function sample_drive(E, tgrid)
    z  = ComplexF64.(E.(tgrid))
    A  = abs.(z); re = real.(z); im = imag.(z)
    ph = _unwrap(angle.(z))
    f  = similar(A); f[1] = 0.0
    @inbounds for k in 2:length(A)
        f[k] = (ph[k] - ph[k-1]) / (tgrid[k] - tgrid[k-1])
    end
    amax = maximum(A)
    @inbounds for k in eachindex(A)
        A[k] < 1e-3 * amax && (f[k] = NaN)
    end
    return A, re, im, f
end

"""raw per-segment B-spline amplitude A_spline(t) -- the physical cA envelope
   BEFORE the C∞ edge taper (build_E_of_t applies |E| = A_spline · taper).
   Returns NaN in the inter-segment silence so the plotted trace BREAKS there
   instead of diving to zero and back (which reads as a spike)."""
function amp_bspline_of_t(cp, u)
    ts, te, _, cA, _ = decode(cp, u)
    kn = [make_clamped_knots(cp.n_coeff_A, ts[i], te[i], cp.degree) for i in 1:cp.k]
    return function (t)
        @inbounds for i in 1:cp.k
            (ts[i] <= t <= te[i]) && return max(0.0, bspline_eval(t, cA[:, i], kn[i], cp.degree))
        end
        return NaN
    end
end

"""exact instantaneous-frequency spline f(t) (build_A_f_of_t's own f_of_t), NaN
   outside every sub-pulse -- the honest curve, vs a numerical d/dt of angle(E)
   which blows up wherever the taper drives |E| toward zero."""
function freq_spline_of_t(cp, u)
    ts, te, _, _, _ = decode(cp, u)
    ffun = build_A_f_of_t(cp, u)[2]
    return function (t)
        @inbounds for i in 1:cp.k
            (ts[i] <= t <= te[i]) && return ffun(t)
        end
        return NaN
    end
end

"""per-segment (duration, peak|E|, chirp span, chirp rate) from a sampled drive, edges trimmed."""
function seg_stats(E, segwins)
    rows = NTuple{4,Float64}[]
    for (a, b) in segwins
        span = b - a
        lo, hi = a + 0.06span, b - 0.06span
        ts = collect(range(lo, hi; length=800)); dt = ts[2] - ts[1]
        z  = ComplexF64.(E.(ts)); Amag = abs.(z)
        ph = _unwrap(angle.(z)); inst = diff(ph) ./ dt
        chirp_span = isempty(inst) ? 0.0 : (maximum(inst) - minimum(inst))
        push!(rows, (b - a, maximum(Amag), chirp_span, chirp_span / span))
    end
    return rows
end

# --------------------------------------------------- load run + build pulses --
struct Bundle
    stem::String
    ref                # load_jld2_reference output
    d                  # prepare_derived
    raw                # raw `data` NamedTuple from <stem>.jld2
    k::Int; nA::Int; nf::Int; degree::Int; taper::Float64
    T_max::Float64; amp_scale::Float64; freq_scale::Float64
    seed_pulse; seed_u::Vector{Float64}
    opt_pulse;  opt_u::Union{Nothing,Vector{Float64}}
    init_metrics::Union{Nothing,NTuple}
    final_metrics::Union{Nothing,NTuple}
    opt_settings::Union{Nothing,NamedTuple}
    history::Union{Nothing,Vector}
    runtime_s::Union{Nothing,Float64}
    seed_kind::String
    have_opt::Bool
end

function load_bundle(stem, data_dir, param_budget, runtime_cli, log_dir=nothing)
    dd   = data_dir
    ld   = log_dir === nothing ? dd : log_dir
    refp = joinpath(dd, "$(stem).jld2")
    isfile(refp) || error("reference not found: $refp")
    logp = joinpath(ld, "$(stem)_optrunlog.jld2")
    parp = joinpath(ld, "$(stem)_opt_pulsepara.jld2")

    ref = load_jld2_reference(refp; verbose=false)
    d   = ref.d
    raw = load_jld2_run(refp)

    runlog = isfile(logp) ? JLD2.load(logp, "data") : nothing
    para   = isfile(parp) ? JLD2.load(parp, "data") : nothing

    # ---- pulse shape params ----
    if para !== nothing
        k, nA, nf = para.k, para.n_coeff_A, para.n_coeff_f
        degree, taper = para.degree, para.taper_frac
    elseif runlog !== nothing
        k, nA, nf = runlog.k, runlog.n_coeff_A, runlog.n_coeff_f
        os = runlog.optimizer_settings
        degree = get(os, :degree, 3); taper = get(os, :taper_frac, 0.1)
    else
        pb = param_budget
        sp, _u, _fr, _seg = fit_linear_seed(ref; param_budget=pb, verbose=false)
        k, nA, nf, degree, taper = sp.k, sp.n_coeff_A, sp.n_coeff_f, sp.degree, sp.taper_frac
    end

    cp    = CompositePulse(k, nA, nf, d; degree=degree, taper_frac=taper)
    Tmax  = cp.T_max
    ascl  = cp.amp_scale
    fscl  = cp.freq_scale

    # ---- seed u (warm start) ----
    seed_kind = "linear seed (fresh fit)"
    if runlog !== nothing && hasproperty(runlog, :initial_u)
        seed_u = collect(Float64, runlog.initial_u)
        seed_kind = "run-log warm start (initial_u)"
    else
        pb = param_budget
        _sp, seed_u, _fr, _seg = fit_linear_seed(ref; param_budget=pb, verbose=false)
        seed_u = collect(Float64, seed_u)
    end
    seed_pulse = cp

    # ---- optimised u ----
    opt_u = para !== nothing ? collect(Float64, para.final_u) :
            (runlog !== nothing && hasproperty(runlog, :final_u) ? collect(Float64, runlog.final_u) : nothing)
    opt_pulse = cp
    have_opt  = opt_u !== nothing

    init_metrics  = runlog !== nothing ? Tuple(runlog.initial_metrics) : nothing
    final_metrics = runlog !== nothing ? Tuple(runlog.final_metrics) :
                    (para !== nothing && hasproperty(para, :final_metrics) ? Tuple(para.final_metrics) : nothing)
    opt_settings  = runlog !== nothing ? runlog.optimizer_settings : nothing
    history       = runlog !== nothing && hasproperty(runlog, :history) ? collect(runlog.history) : nothing

    # ---- total run-time: --runtime, then a stored field, else nothing ----
    # (a field may be PRESENT but nothing -- save_optimisation_run_log's own
    # default -- so test the value, not just hasproperty/haskey.)
    runtime_s = runtime_cli === nothing ? nothing : Float64(runtime_cli)
    if runtime_s === nothing && runlog !== nothing
        for f in (:elapsed_seconds, :runtime_seconds, :elapsed)
            if hasproperty(runlog, f) && getproperty(runlog, f) !== nothing
                runtime_s = Float64(getproperty(runlog, f)); break
            end
        end
        if runtime_s === nothing && opt_settings !== nothing
            for f in (:elapsed_seconds, :runtime_seconds, :wall_seconds)
                if haskey(opt_settings, f) && opt_settings[f] !== nothing
                    runtime_s = Float64(opt_settings[f]); break
                end
            end
        end
    end

    return Bundle(stem, ref, d, raw, k, nA, nf, degree, taper, Tmax, ascl, fscl,
                  seed_pulse, seed_u, opt_pulse, opt_u,
                  init_metrics, final_metrics, opt_settings, history, runtime_s,
                  seed_kind, have_opt)
end

# ------------------------------------------------------------- HTML bits ------
const CSS = raw"""
  :root{
    --bg:#f0f0ee; --surface:#fbfbfa; --surface-2:#f5f5f2;
    --ink:#17181c; --ink-2:#585b62; --ink-3:#8a8d94;
    --line:rgba(20,21,28,.09); --line-strong:rgba(20,21,28,.22);
    --seed:#2a78d6; --opt:#eb6834;
    --band:rgba(42,120,214,.055); --band-line:rgba(42,120,214,.18);
    --good:#1c7d54;
    --shadow:0 1px 2px rgba(20,21,28,.05),0 8px 26px -14px rgba(20,21,28,.22);
    color-scheme:light;
  }
  @media (prefers-color-scheme:dark){
    :root:not([data-theme="light"]){
      --bg:#0d0e11; --surface:#16171b; --surface-2:#1b1d22;
      --ink:#edeef2; --ink-2:#9aa0a8; --ink-3:#6a6e77;
      --line:rgba(237,238,242,.10); --line-strong:rgba(237,238,242,.26);
      --seed:#3f8ee8; --opt:#e46a3c;
      --band:rgba(63,142,232,.10); --band-line:rgba(63,142,232,.28);
      --good:#4bbf8a;
      --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 34px -16px rgba(0,0,0,.6);
      color-scheme:dark;
    }
  }
  :root[data-theme="dark"]{
    --bg:#0d0e11; --surface:#16171b; --surface-2:#1b1d22;
    --ink:#edeef2; --ink-2:#9aa0a8; --ink-3:#6a6e77;
    --line:rgba(237,238,242,.10); --line-strong:rgba(237,238,242,.26);
    --seed:#3f8ee8; --opt:#e46a3c;
    --band:rgba(63,142,232,.10); --band-line:rgba(63,142,232,.28);
    --good:#4bbf8a;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 34px -16px rgba(0,0,0,.6);
    color-scheme:dark;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font-family:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1040px;margin:0 auto;padding:56px 24px 96px}
  .mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
  header{margin-bottom:34px}
  .eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.13em;
    text-transform:uppercase;color:var(--ink-3);margin:0 0 14px}
  h1{font-family:"Zilla Slab",Georgia,serif;font-weight:700;font-size:clamp(28px,4.4vw,40px);
    line-height:1.08;letter-spacing:-.01em;margin:0 0 12px;text-wrap:balance}
  .dek{font-size:16px;color:var(--ink-2);max-width:66ch;margin:0}
  .dek b{color:var(--ink);font-weight:600}
  .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:30px 0 8px}
  @media (max-width:900px){.stats{grid-template-columns:repeat(2,1fr)}}
  .stat{background:var(--surface);border:1px solid var(--line);border-radius:12px;
    padding:16px 18px 15px;box-shadow:var(--shadow)}
  .stat .k{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.11em;
    text-transform:uppercase;color:var(--ink-3);margin-bottom:9px}
  .stat .v{font-family:"Zilla Slab",serif;font-weight:600;font-size:21px;line-height:1.15;
    display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}
  .stat .v .from{color:var(--ink-3);font-weight:500;font-size:14px}
  .stat .v .arrow{color:var(--ink-3);font-size:13px}
  .stat .tag{font-family:"IBM Plex Mono",monospace;font-size:11.5px;margin-top:7px;color:var(--ink-2)}
  .tag.drop{color:var(--opt)} .tag.held{color:var(--good)} .tag.rise{color:var(--good)}
  section.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;
    padding:22px 22px 16px;margin-top:22px;box-shadow:var(--shadow)}
  .panel h2{font-family:"Zilla Slab",serif;font-weight:600;font-size:19px;margin:0 0 3px;letter-spacing:-.005em}
  .panel .sub{font-size:13px;color:var(--ink-2);margin:0 0 8px}
  .legend{display:flex;gap:20px;align-items:center;margin:8px 0 2px;flex-wrap:wrap}
  .legend .item{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--ink-2)}
  .legend .swatch{width:26px;height:0;border-top-width:2.5px;border-top-style:solid;display:inline-block}
  .legend .swatch.seed{border-top-color:var(--seed);border-top-style:dashed}
  .legend .swatch.opt{border-top-color:var(--opt)}
  .legend b{color:var(--ink);font-weight:600}
  .chartbox{position:relative;margin-top:6px}
  svg.chart{display:block;width:100%;height:auto;overflow:visible}
  .grid line{stroke:var(--line);stroke-width:1}
  .axis text{fill:var(--ink-3);font-family:"IBM Plex Mono",monospace;font-size:10.5px}
  .zero line{stroke:var(--line-strong);stroke-width:1}
  .segband{fill:var(--band)}
  .segband-line{stroke:var(--band-line);stroke-width:1;stroke-dasharray:2 3}
  .seglabel{fill:var(--seed);font-family:"IBM Plex Mono",monospace;font-size:9.5px;letter-spacing:.04em;opacity:.7}
  .trace{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
  .trace.seed{stroke:var(--seed);stroke-width:1.7;stroke-dasharray:5 3.5}
  .trace.opt{stroke:var(--opt)}
  .trace.cost{stroke:var(--opt);stroke-width:2}
  .trace.best{stroke:var(--seed);stroke-width:1.4;stroke-dasharray:4 3}
  .hopline{stroke:var(--line-strong);stroke-width:1;stroke-dasharray:2 3}
  .hoptext{fill:var(--ink-3);font-family:"IBM Plex Mono",monospace;font-size:9px;letter-spacing:.04em}
  .cdot{fill:var(--opt);stroke:var(--surface);stroke-width:1}
  .endlabel{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:500}
  .endlabel.seed{fill:var(--seed)} .endlabel.opt{fill:var(--opt)}
  .cross{stroke:var(--line-strong);stroke-width:1;stroke-dasharray:3 3}
  .dot{stroke:var(--surface);stroke-width:1.5}
  .dot.seed{fill:var(--seed)} .dot.opt{fill:var(--opt)}
  .tip{position:absolute;pointer-events:none;z-index:5;opacity:0;transition:opacity .12s;
    background:var(--surface);border:1px solid var(--line-strong);border-radius:9px;
    box-shadow:var(--shadow);padding:9px 11px;font-size:12px;min-width:140px;
    font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
  .tip .tt{color:var(--ink-3);font-size:10px;letter-spacing:.09em;text-transform:uppercase;margin-bottom:5px}
  .tip .row{display:flex;justify-content:space-between;gap:16px;line-height:1.75}
  .tip .row .nm{display:flex;align-items:center;gap:6px;color:var(--ink-2)}
  .tip .row .nm i{width:9px;height:9px;border-radius:2px;display:inline-block}
  .tip .row .nm i.seed{background:var(--seed)} .tip .row .nm i.opt{background:var(--opt)}
  .tip .row b{color:var(--ink);font-weight:500}
  table{border-collapse:collapse;width:100%;margin-top:6px;font-size:13px}
  .tablewrap{overflow-x:auto}
  th,td{padding:9px 12px;text-align:right;white-space:nowrap}
  th:first-child,td:first-child{text-align:left}
  thead th{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.09em;
    text-transform:uppercase;color:var(--ink-3);font-weight:500;border-bottom:1px solid var(--line-strong)}
  tbody td{border-bottom:1px solid var(--line);font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
  tbody tr:last-child td{border-bottom:none}
  tbody .ratio td{color:var(--ink-2);font-size:12px}
  tbody .ratio td:first-child{color:var(--ink-3)}
  .s{color:var(--seed)} .o{color:var(--opt)}
  td .delta{color:var(--ink-3);font-size:11px;margin-left:6px}
  .cfg{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px 26px;margin-top:4px}
  .cfg .grp{border-top:1px solid var(--line-strong);padding-top:9px}
  .cfg .grp h3{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.1em;
    text-transform:uppercase;color:var(--ink-3);margin:0 0 7px;font-weight:500}
  .cfg dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:3px 14px}
  .cfg dt{color:var(--ink-3)}
  .cfg dd{margin:0;text-align:right;font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
  details{margin-top:12px;border-top:1px solid var(--line);padding-top:10px}
  summary{cursor:pointer;font-family:"IBM Plex Mono",monospace;font-size:11.5px;
    letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}
  summary:focus-visible{outline:2px solid var(--seed);outline-offset:3px;border-radius:3px}
  .splinegrid{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:10px}
  @media (max-width:760px){.splinegrid{grid-template-columns:1fr}}
  .splinegrid h3{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.1em;
    text-transform:uppercase;color:var(--ink-3);margin:0 0 4px;font-weight:500}
  footer{margin-top:40px;padding-top:20px;border-top:1px solid var(--line);
    font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--ink-3);line-height:1.9}
  footer .kv b{color:var(--ink-2);font-weight:500}
  @media (max-width:680px){.stats{grid-template-columns:1fr}.wrap{padding:40px 16px 72px}}
  @media (prefers-reduced-motion:reduce){
    .trace{stroke-dasharray:none!important;stroke-dashoffset:0!important;animation:none!important}
    .trace.seed{stroke-dasharray:5 3.5!important}
  }
"""

const JS = raw"""
(function(){
  "use strict";
  var DATA = JSON.parse(document.getElementById("pulsedata").textContent);
  var reduce = window.matchMedia("(prefers-reduced-motion:reduce)").matches;
  var XMIN = DATA.xmin, XMAX = DATA.xmax;

  function niceMax(v){
    if(v<=0) return 1;
    var e = Math.pow(10, Math.floor(Math.log10(v)));
    var m = v/e;
    var nm = m<=1?1:m<=2?2:m<=2.5?2.5:m<=5?5:10;
    return nm*e;
  }
  function ticks(lo,hi,n){
    var out=[],step=(hi-lo)/n;
    for(var i=0;i<=n;i++){ var t=lo+i*step; out.push(Math.abs(t)<1e-9?0:parseFloat(t.toPrecision(4))); }
    return out;
  }

  function build(cfg){
    var box=document.getElementById(cfg.box), tip=document.getElementById(cfg.tip);
    var W=940,H=286,m={l:52,r:58,t:14,b:30};
    var ix0=m.l,ix1=W-m.r,iy0=H-m.b,iy1=m.t;
    var allv=[]; for(var i=0;i<DATA.t.length;i++){ if(cfg.seed[i]!=null)allv.push(cfg.seed[i]); if(cfg.opt[i]!=null)allv.push(cfg.opt[i]); }
    var vmax = niceMax(Math.max.apply(null, allv.map(Math.abs)));
    var ymin = cfg.symmetric ? -vmax : 0, ymax = vmax;
    var yt = cfg.symmetric ? ticks(-vmax,vmax,4) : ticks(0,vmax,4);
    var xs=function(v){return ix0+(v-XMIN)/(XMAX-XMIN)*(ix1-ix0);};
    var ys=function(v){return iy0+(v-ymin)/(ymax-ymin)*(iy1-iy0);};
    var NS="http://www.w3.org/2000/svg";
    function el(n,a){var e=document.createElementNS(NS,n);for(var k in a)e.setAttribute(k,a[k]);return e;}
    var svg=el("svg",{class:"chart",viewBox:"0 0 "+W+" "+H,role:"img","aria-label":cfg.aria});
    DATA.seg_seed.forEach(function(s,i){
      svg.appendChild(el("rect",{class:"segband",x:xs(s[0]),y:iy1,width:Math.max(0,xs(s[1])-xs(s[0])),height:iy0-iy1}));
      if(i>0) svg.appendChild(el("line",{class:"segband-line",x1:xs(s[0]),x2:xs(s[0]),y1:iy1,y2:iy0}));
    });
    var g=el("g",{class:"grid"}),ax=el("g",{class:"axis"});
    yt.forEach(function(t){
      g.appendChild(el("line",{x1:ix0,x2:ix1,y1:ys(t),y2:ys(t)}));
      var tx=el("text",{x:ix0-9,y:ys(t)+3.5,"text-anchor":"end"});tx.textContent=t;ax.appendChild(tx);
    });
    var xstep = (XMAX-XMIN)>800?200:(XMAX-XMIN)>400?100:50;
    for(var xt=Math.ceil(XMIN/xstep)*xstep; xt<=XMAX; xt+=xstep){
      g.appendChild(el("line",{x1:xs(xt),x2:xs(xt),y1:iy1,y2:iy0,opacity:.5}));
      var txx=el("text",{x:xs(xt),y:iy0+18,"text-anchor":"middle"});txx.textContent=xt;ax.appendChild(txx);
    }
    var xl=el("text",{x:(ix0+ix1)/2,y:H-3,"text-anchor":"middle"});xl.textContent="time  (µs)";ax.appendChild(xl);
    svg.appendChild(g);
    if(cfg.symmetric){var z=el("g",{class:"zero"});z.appendChild(el("line",{x1:ix0,x2:ix1,y1:ys(0),y2:ys(0)}));svg.appendChild(z);}
    svg.appendChild(ax);
    DATA.seg_seed.forEach(function(s,i){
      var t=el("text",{class:"seglabel",x:(xs(s[0])+xs(s[1]))/2,y:iy1+11,"text-anchor":"middle"});
      t.textContent=DATA.seg_labels[i]||("seg "+(i+1));svg.appendChild(t);
    });
    function pathFor(arr){
      var d="",pen=false;
      for(var i=0;i<DATA.t.length;i++){
        var x=DATA.t[i],y=arr[i];
        if(x<XMIN-2||x>XMAX+2){pen=false;continue;}
        if(y==null){pen=false;continue;}
        d+=(pen?"L":"M")+xs(x).toFixed(1)+" "+ys(y).toFixed(1)+" ";pen=true;
      }
      return d.trim();
    }
    var pSeed=el("path",{class:"trace seed",d:pathFor(cfg.seed)});
    var pOpt =el("path",{class:"trace opt", d:pathFor(cfg.opt)});
    svg.appendChild(pSeed);svg.appendChild(pOpt);
    function lastPt(arr){for(var i=DATA.t.length-1;i>=0;i--){if(DATA.t[i]<=XMAX&&arr[i]!=null)return{x:DATA.t[i],y:arr[i]};}return null;}
    var ls=lastPt(cfg.seed),lo=lastPt(cfg.opt),dy=cfg.symmetric?-4:3.5;
    if(ls){var t1=el("text",{class:"endlabel seed",x:xs(ls.x)+7,y:ys(ls.y)+dy});t1.textContent="seed";svg.appendChild(t1);}
    if(lo){var t2=el("text",{class:"endlabel opt",x:xs(lo.x)+7,y:ys(lo.y)+(cfg.symmetric?12:3.5)});t2.textContent="opt";svg.appendChild(t2);}
    var cg=el("g",{opacity:0});
    var cl=el("line",{class:"cross",y1:iy1,y2:iy0});
    var dS=el("circle",{class:"dot seed",r:3.6}),dO=el("circle",{class:"dot opt",r:3.6});
    cg.appendChild(cl);cg.appendChild(dS);cg.appendChild(dO);svg.appendChild(cg);
    var ov=el("rect",{x:ix0,y:iy1,width:ix1-ix0,height:iy0-iy1,fill:"transparent",style:"cursor:crosshair"});
    svg.appendChild(ov);
    box.insertBefore(svg,box.firstChild);
    function fmt(v){return v==null?"—":v.toFixed(3);}
    function move(ev){
      var r=svg.getBoundingClientRect();
      var px=(ev.clientX-r.left)/r.width*W;
      var tv=XMIN+(px-ix0)/(ix1-ix0)*(XMAX-XMIN);
      if(tv<XMIN||tv>XMAX){hide();return;}
      var lo2=0,hi2=DATA.t.length-1;
      while(hi2-lo2>1){var mid=(lo2+hi2)>>1;if(DATA.t[mid]<tv)lo2=mid;else hi2=mid;}
      var i=(tv-DATA.t[lo2]<DATA.t[hi2]-tv)?lo2:hi2;
      var x=DATA.t[i],sy=cfg.seed[i],oy=cfg.opt[i];
      cg.setAttribute("opacity",1);
      cl.setAttribute("x1",xs(x));cl.setAttribute("x2",xs(x));
      if(sy!=null){dS.setAttribute("opacity",1);dS.setAttribute("cx",xs(x));dS.setAttribute("cy",ys(sy));}else dS.setAttribute("opacity",0);
      if(oy!=null){dO.setAttribute("opacity",1);dO.setAttribute("cx",xs(x));dO.setAttribute("cy",ys(oy));}else dO.setAttribute("opacity",0);
      tip.innerHTML='<div class="tt">t = '+x.toFixed(1)+' µs</div>'+
        '<div class="row"><span class="nm"><i class="seed"></i>seed</span><b>'+fmt(sy)+'</b></div>'+
        '<div class="row"><span class="nm"><i class="opt"></i>opt</span><b>'+fmt(oy)+'</b></div>';
      tip.style.opacity=1;
      var bx=box.getBoundingClientRect();
      var lx=(ev.clientX-bx.left)+16;
      if(lx+156>bx.width)lx=(ev.clientX-bx.left)-156;
      tip.style.left=Math.max(4,lx)+"px";
      tip.style.top=Math.max(2,(ev.clientY-bx.top)-10)+"px";
    }
    function hide(){cg.setAttribute("opacity",0);tip.style.opacity=0;}
    ov.addEventListener("pointermove",move);
    ov.addEventListener("pointerleave",hide);
    if(!reduce){
      [pSeed,pOpt].forEach(function(p,k){
        var len=p.getTotalLength();p.style.transition="none";
        p.style.strokeDasharray=(k===0?(len+" "+len):len);p.style.strokeDashoffset=len;
        requestAnimationFrame(function(){requestAnimationFrame(function(){
          p.style.transition="stroke-dashoffset 900ms cubic-bezier(.4,0,.2,1) "+(k*120)+"ms";
          p.style.strokeDashoffset=0;});});
      });
      pSeed.addEventListener("transitionend",function(){pSeed.style.strokeDasharray="5 3.5";pSeed.style.transition="";});
      pOpt.addEventListener("transitionend",function(){pOpt.style.strokeDasharray="none";pOpt.style.transition="";});
    }
  }
  build({box:"boxA",tip:"tipA",seed:DATA.seed_A,opt:DATA.opt_A,symmetric:false,
    aria:"Delivered amplitude envelope of the seed and optimised pulses over time."});
  if(DATA.seed_Araw) build({box:"boxD",tip:"tipD",seed:DATA.seed_Araw,opt:DATA.opt_Araw,symmetric:false,
    aria:"Raw B-spline amplitude of the seed and optimised pulses over time."});
  build({box:"boxB",tip:"tipB",seed:DATA.seed_f,opt:DATA.opt_f,symmetric:true,
    aria:"Instantaneous frequency of the seed and optimised pulses over time."});

  // ---- cost vs epoch ----
  var HC = DATA.hist;
  if(HC && HC.cost && HC.cost.length > 1){
    var box=document.getElementById("boxC"), tip=document.getElementById("tipC");
    var W=940,H=270,m={l:58,r:20,t:14,b:30};
    var ix0=m.l,ix1=W-m.r,iy0=H-m.b,iy1=m.t;
    var n=HC.cost.length;
    var cmin=Infinity,cmax=-Infinity;
    for(var i=0;i<n;i++){ var c=HC.cost[i]; if(c<cmin)cmin=c; if(c>cmax)cmax=c; }
    var useLog = cmin>0 && cmax/cmin > 12;
    var yof = useLog ? Math.log10 : function(v){return v;};
    var ylo = useLog ? Math.floor(yof(cmin)) : 0;
    var yhi = useLog ? Math.ceil(yof(cmax)) : niceMax(cmax);
    var xs=function(i){return ix0+(i)/(n-1)*(ix1-ix0);};
    var ys=function(v){return iy0+(yof(v)-ylo)/(yhi-ylo)*(iy1-iy0);};
    var NS="http://www.w3.org/2000/svg";
    function el(t,a){var e=document.createElementNS(NS,t);for(var k in a)e.setAttribute(k,a[k]);return e;}
    var svg=el("svg",{class:"chart",viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Optimiser cost per epoch."});
    var g=el("g",{class:"grid"}),ax=el("g",{class:"axis"});
    var yticks = useLog ? (function(){var a=[];for(var e=ylo;e<=yhi;e++)a.push(e);return a;})()
                        : ticks(0,yhi,4);
    yticks.forEach(function(e){
      var yy = useLog ? ys(Math.pow(10,e)) : ys(e);
      g.appendChild(el("line",{x1:ix0,x2:ix1,y1:yy,y2:yy}));
      var tx=el("text",{x:ix0-9,y:yy+3.5,"text-anchor":"end"});
      tx.textContent = useLog ? ("1e"+e) : e;
      ax.appendChild(tx);
    });
    // hop boundaries + labels
    var lastHop=null;
    for(var i=0;i<n;i++){
      if(HC.hop[i]!==lastHop){
        if(lastHop!==null){
          g.appendChild(el("line",{class:"hopline",x1:xs(i-0.5),x2:xs(i-0.5),y1:iy1,y2:iy0}));
        }
        var ht=el("text",{class:"hoptext",x:xs(i)+2,y:iy1+9});
        ht.textContent="hop "+HC.hop[i];
        ax.appendChild(ht);
        lastHop=HC.hop[i];
      }
    }
    var xl=el("text",{x:(ix0+ix1)/2,y:H-3,"text-anchor":"middle"});xl.textContent="epoch (cumulative across hops)";ax.appendChild(xl);
    svg.appendChild(g);svg.appendChild(ax);
    // running best
    var best=Infinity, bpath="";
    for(var i=0;i<n;i++){ best=Math.min(best,HC.cost[i]); bpath+=(i?"L":"M")+xs(i).toFixed(1)+" "+ys(best).toFixed(1)+" "; }
    svg.appendChild(el("path",{class:"trace best",d:bpath.trim()}));
    // cost line
    var cpath="";
    for(var i=0;i<n;i++){ cpath+=(i?"L":"M")+xs(i).toFixed(1)+" "+ys(HC.cost[i]).toFixed(1)+" "; }
    var pc=el("path",{class:"trace cost",d:cpath.trim()});
    svg.appendChild(pc);
    // improved-epoch dots
    for(var i=0;i<n;i++){ if(HC.improved[i]) svg.appendChild(el("circle",{class:"cdot",cx:xs(i),cy:ys(HC.cost[i]),r:2.6})); }
    // legend
    var lg=el("text",{x:ix1,y:iy1+2,"text-anchor":"end",class:"hoptext"});
    lg.textContent = "— cost   ·  – – running best   ●  improved";
    ax.appendChild(lg);
    var cg=el("g",{opacity:0});
    var cl=el("line",{class:"cross",y1:iy1,y2:iy0});
    var dd=el("circle",{class:"cdot",r:3.8});
    cg.appendChild(cl);cg.appendChild(dd);svg.appendChild(cg);
    var ov=el("rect",{x:ix0,y:iy1,width:ix1-ix0,height:iy0-iy1,fill:"transparent",style:"cursor:crosshair"});
    svg.appendChild(ov);
    box.insertBefore(svg,box.firstChild);
    function cmove(ev){
      var r=svg.getBoundingClientRect();
      var frac=(ev.clientX-r.left)/r.width*W;
      var i=Math.round((frac-ix0)/(ix1-ix0)*(n-1));
      if(i<0||i>n-1){cg.setAttribute("opacity",0);tip.style.opacity=0;return;}
      cg.setAttribute("opacity",1);
      cl.setAttribute("x1",xs(i));cl.setAttribute("x2",xs(i));
      dd.setAttribute("cx",xs(i));dd.setAttribute("cy",ys(HC.cost[i]));
      tip.innerHTML='<div class="tt">hop '+HC.hop[i]+' · ep '+HC.epoch[i]+'</div>'+
        '<div class="row"><span class="nm">cost</span><b>'+HC.cost[i].toPrecision(5)+'</b></div>'+
        (HC.improved[i]?'<div class="row"><span class="nm">improved</span><b>✓</b></div>':'');
      tip.style.opacity=1;
      var bx=box.getBoundingClientRect();
      var lx=(ev.clientX-bx.left)+16; if(lx+156>bx.width)lx=(ev.clientX-bx.left)-156;
      tip.style.left=Math.max(4,lx)+"px"; tip.style.top=Math.max(2,(ev.clientY-bx.top)-10)+"px";
    }
    ov.addEventListener("pointermove",cmove);
    ov.addEventListener("pointerleave",function(){cg.setAttribute("opacity",0);tip.style.opacity=0;});
    if(!reduce){
      var L=pc.getTotalLength();pc.style.strokeDasharray=L;pc.style.strokeDashoffset=L;
      requestAnimationFrame(function(){requestAnimationFrame(function(){
        pc.style.transition="stroke-dashoffset 1000ms cubic-bezier(.4,0,.2,1)";pc.style.strokeDashoffset=0;});});
      pc.addEventListener("transitionend",function(){pc.style.strokeDasharray="none";pc.style.transition="";});
    }
  }
})();
"""

esc(s) = replace(string(s), "&"=>"&amp;", "<"=>"&lt;", ">"=>"&gt;")

# key/value <dl> rows from pairs
kv(pairs) = join(("      <dt>$(esc(k))</dt><dd>$(esc(v))</dd>" for (k,v) in pairs), "\n")

function cfg_section(b::Bundle)
    sys = b.raw.SYSTEM_CONFIG
    sim = b.raw.SIM_SETTING
    d   = b.d
    fi  = sys.freq_inhomogeneity; gi = sys.g_inhomogeneity
    cav = [("κ_e", twopi_mhz(sys.kappa_e)), ("κ_i", twopi_mhz(sys.kappa_i)),
           ("κ_t", twopi_mhz(sys.kappa_e + sys.kappa_i)), ("δ₀", sig(sys.delta0))]
    ens = [("C_ens", sig(sys.C_ens)),
           ("freq inhom.", "$(fi.kind), FWHM " * twopi_mhz(fi.FWHM)),
           ("freq span", string(get(fi, :span_gamma, get(fi, :span_sigma, "—"))) *
                         (haskey(fi, :renormalize) ? ", renorm=$(fi.renormalize)" : "")),
           ("g inhom.", string(gi.kind) *
                        (haskey(gi, :mean) ? ", mean " * twopi_hz(gi.mean) : "") *
                        (haskey(gi, :g_value) ? " " * twopi_hz(gi.g_value) : "")),
           ("g std", haskey(gi, :std) ? twopi_hz(gi.std) : "—"),
           ("FWHM (derived)", @sprintf("%.5g rad/s", d.FWHM)),
           ("g_mean / g_std", @sprintf("%.5g / %.4g  (cv %.3g)", d.g_mean, d.g_std, d.g_std/d.g_mean)),
           ("N total spins", @sprintf("%.5g", d.N)),
           ("mesh  M_δ × M_g", @sprintf("%d × %d = %d", sim.M_delta, sim.M_g, d.M))]
    smd = [("order", string(sim.simulation_order)),
           ("initial condition", string(get(sim, :initial_condition, "—"))),
           ("Ttotal", @sprintf("%.4g µs", sim.Ttotal * 1e6)),
           ("Nt_save", string(get(sim, :Nt_save, "—"))),
           ("reltol / abstol", @sprintf("%.0e / %.0e", sim.reltol, sim.abstol))]
    return """
  <section class="panel">
    <h2>System configuration</h2>
    <p class="sub">SYSTEM_CONFIG and SIM_SETTING from <span class="mono">$(esc(b.stem)).jld2</span>, plus the derived ensemble.</p>
    <div class="cfg">
      <div class="grp"><h3>Cavity</h3><dl>
$(kv(cav))
      </dl></div>
      <div class="grp"><h3>Ensemble</h3><dl>
$(kv(ens))
      </dl></div>
      <div class="grp"><h3>Simulation</h3><dl>
$(kv(smd))
      </dl></div>
    </div>
  </section>"""
end

function spline_table(label, cp, u, fscl)
    ts, te, phi0, cA, cf = decode(cp, u)
    n = cp.k
    seg_rows = join((@sprintf("        <tr><td>%d</td><td>%.4f</td><td>%.4f</td><td>%.4f</td><td>%.5f</td></tr>",
                              i, ts[i]*1e6, te[i]*1e6, (te[i]-ts[i])*1e6, phi0[i]) for i in 1:n), "\n")
    function mat_rows(M, pre)
        join((@sprintf("        <tr><td>%s%d</td>%s</tr>", pre, r,
                       join((@sprintf("<td>% .6e</td>", M[r,j]) for j in 1:n))) for r in 1:size(M,1)), "\n")
    end
    hdr = "<tr><th>coeff</th>" * join(("<th>seg $j</th>" for j in 1:n)) * "</tr>"
    return """
      <h3>$(esc(label)) &mdash; segment timing</h3>
      <div class="tablewrap"><table><thead>
        <tr><th>seg</th><th>t_start µs</th><th>t_end µs</th><th>dur µs</th><th>phi0 rad</th></tr>
      </thead><tbody>
$(seg_rows)
      </tbody></table></div>
      <h3 style="margin-top:14px">cA &mdash; amplitude coefficients (physical, amp_scale·softplus(raw))</h3>
      <div class="tablewrap"><table><thead>$(hdr)</thead><tbody>
$(mat_rows(cA, "a"))
      </tbody></table></div>
      <h3 style="margin-top:14px">cf &mdash; instantaneous-frequency coefficients (rad/s)</h3>
      <div class="tablewrap"><table><thead>$(hdr)</thead><tbody>
$(mat_rows(cf, "f"))
      </tbody></table></div>"""
end

function spline_section(b::Bundle)
    npar = 3*b.k + b.k*b.nA + b.k*b.nf
    shape = [("k", string(b.k)), ("degree", string(b.degree)),
             ("n_coeff_A", string(b.nA)), ("n_coeff_f", string(b.nf)),
             ("n_params", string(npar)), ("taper_frac", sig(b.taper)),
             ("T_max", @sprintf("%.6g s", b.T_max)),
             ("amp_scale", @sprintf("%.6g", b.amp_scale)),
             ("freq_scale", @sprintf("%.6g rad/s", b.freq_scale))]
    opt_block = b.have_opt ? spline_table("Optimised pulse", b.opt_pulse, b.opt_u, b.freq_scale) :
        "<p class=\"sub\">No optimised pulse found &mdash; showing the seed only.</p>"
    seed_block = """
    <details>
      <summary>Seed / warm-start spline parameters &nbsp;($(esc(b.seed_kind)))</summary>
      <div style="margin-top:12px">
$(spline_table("Seed", b.seed_pulse, b.seed_u, b.freq_scale))
      </div>
    </details>"""
    return """
  <section class="panel">
    <h2>B-spline parameters</h2>
    <p class="sub">CompositePulse parameterisation. Each segment's amplitude and instantaneous
      frequency are degree-$(b.degree) clamped B-splines on normalised segment time; the drive is
      <span class="mono">E(t) = A(t)·exp(i·Φ(t))</span> with <span class="mono">Φ = phi0 + ∫f</span>.</p>
    <div class="cfg"><div class="grp"><h3>Shape</h3><dl>
$(kv(shape))
    </dl></div></div>
    <div style="margin-top:16px">
$(opt_block)
    </div>
$(seed_block)
  </section>"""
end

# ------------------------------------------------------------- assemble -------
function build_report(b::Bundle)
    d = b.d
    Eseed = build_E_of_t(b.seed_pulse, b.seed_u)
    Eopt  = b.have_opt ? build_E_of_t(b.opt_pulse, b.opt_u) : Eseed

    ts_s, te_s, _, _, _ = decode(b.seed_pulse, b.seed_u)
    segwin_s = [(ts_s[i], te_s[i]) for i in 1:b.k]
    if b.have_opt
        ts_o, te_o, _, _, _ = decode(b.opt_pulse, b.opt_u)
        segwin_o = [(ts_o[i], te_o[i]) for i in 1:b.k]
    else
        segwin_o = segwin_s
    end

    lo = min(minimum(first.(segwin_s)), minimum(first.(segwin_o)))
    hi = max(maximum(last.(segwin_s)),  maximum(last.(segwin_o)))
    pad = 0.04 * (hi - lo)
    x0 = max(0.0, lo - pad); x1 = hi + pad

    N = 3001
    tg = collect(range(x0, x1; length=N))
    As, _, _, _ = sample_drive(Eseed, tg)
    Ao, _, _, _ = b.have_opt ? sample_drive(Eopt, tg) : (As, nothing, nothing, nothing)

    # instantaneous frequency = the EXACT f-spline (build_A_f_of_t), NaN in the
    # gaps. Not a numerical d/dt of angle(E): that explodes at every tapered
    # edge (|E| -> 0, phase undefined) and across each sub-pulse phi0 jump.
    fseed = freq_spline_of_t(b.seed_pulse, b.seed_u)
    fs = fseed.(tg)
    fo = b.have_opt ? freq_spline_of_t(b.opt_pulse, b.opt_u).(tg) : fs

    # raw B-spline amplitude A(t) (pre-taper); clip display at 3× the |E| peak
    Aspl_s = amp_bspline_of_t(b.seed_pulse, b.seed_u).(tg)
    Aspl_o = b.have_opt ? amp_bspline_of_t(b.opt_pulse, b.opt_u).(tg) : Aspl_s
    aclip  = 3 * max(maximum(As), maximum(Ao))
    clipped = any(>(aclip), Aspl_s) || any(>(aclip), Aspl_o)
    As_raw = min.(Aspl_s, aclip); Ao_raw = min.(Aspl_o, aclip)

    # ---- drive power: avg over EACH pulse's own segment span; energy over all ----
    dts = tg[2] - tg[1]
    lo_s = minimum(first.(segwin_s)); hi_s = maximum(last.(segwin_s))
    lo_o = minimum(first.(segwin_o)); hi_o = maximum(last.(segwin_o))
    ms = (tg .>= lo_s) .& (tg .<= hi_s)
    mo = (tg .>= lo_o) .& (tg .<= hi_o)
    pow_s = sum((As[ms] ./ b.amp_scale) .^ 2) / max(count(ms), 1)   # ⟨|E/amp_scale|²⟩ while on
    pow_o = b.have_opt ? sum((Ao[mo] ./ b.amp_scale) .^ 2) / max(count(mo), 1) : pow_s
    en_s  = sum(As .^ 2) * dts                                       # ∫|E|² dt over the whole trace
    en_o  = b.have_opt ? sum(Ao .^ 2) * dts : en_s

    step = 2
    keep = 1:step:N
    j(v) = [isnan(x) ? nothing : round(x/1e7; sigdigits=5) for x in v[keep]]
    payload = Dict{String,Any}(
        "t"    => [round(t*1e6; digits=2) for t in tg[keep]],
        "xmin" => round(x0*1e6; digits=1), "xmax" => round(x1*1e6; digits=1),
        "seed_A" => j(As), "opt_A" => j(Ao),
        "seed_Araw" => j(As_raw), "opt_Araw" => j(Ao_raw),
        "seed_f" => j(fs), "opt_f" => j(fo),
        "seg_seed" => [[round(a*1e6;digits=2), round(bb*1e6;digits=2)] for (a,bb) in segwin_s],
        "seg_opt"  => [[round(a*1e6;digits=2), round(bb*1e6;digits=2)] for (a,bb) in segwin_o],
        "seg_labels" => b.k == 3 ? ["+k","+k/2","+k"] : ["seg $i" for i in 1:b.k],
    )
    if b.history !== nothing && !isempty(b.history)
        payload["hist"] = Dict(
            "epoch"    => [Int(r.epoch) for r in b.history],
            "hop"      => [Int(r.hop) for r in b.history],
            "cost"     => [Float64(r.cost) for r in b.history],
            "improved" => [r.improved === true for r in b.history],
        )
    end
    json = _jsonval(payload)

    # ---- stat tiles ----
    im = b.init_metrics; fm = b.final_metrics
    function tile(name, i0, i1; pct=true, fmtf=x->sig(x,5))
        v0 = im === nothing ? nothing : im[i0]
        v1 = fm === nothing ? nothing : fm[i1]
        chg = ""
        if v0 !== nothing && v1 !== nothing
            dp = 100*(v1/v0 - 1)
            cls = abs(dp) < 0.5 ? "held" : (dp < 0 ? "drop" : "rise")
            chg = "<div class=\"tag $cls\">" * (abs(dp) < 0.05 ? "held" : @sprintf("%+.1f%%", dp)) * "</div>"
        end
        vv = v1 === nothing ? "—" : fmtf(v1)
        ff = v0 === nothing ? "" : "<span class=\"from\">$(fmtf(v0))</span><span class=\"arrow\">&rarr;</span>"
        """
    <div class="stat"><div class="k">$(esc(name))</div>
      <div class="v">$ff<span>$vv</span></div>$chg</div>"""
    end
    # metrics tuple order: (cost, inversion, silencing, duration, coherence, ...)
    dur_tiles = tile("Duration", 4, 4; fmtf=x->x===nothing ? "—" : @sprintf("%.1f µs", x*1e6))
    inv_tiles = tile("Inversion", 2, 2)
    sil_tiles = tile("Silencing |F|⋆", 3, 3)

    # power tile (not in the metrics tuple -- computed from the sampled drive)
    function tile2(name, v0, v1, ff)
        dp  = 100*(v1/v0 - 1)
        cls = abs(dp) < 0.5 ? "held" : (dp < 0 ? "drop" : "rise")
        chg = "<div class=\"tag $cls\">" * (abs(dp) < 0.05 ? "held" : @sprintf("%+.1f%%", dp)) * "</div>"
        """
    <div class="stat"><div class="k">$(esc(name))</div>
      <div class="v"><span class="from">$(ff(v0))</span><span class="arrow">&rarr;</span><span>$(ff(v1))</span></div>$chg</div>"""
    end
    pow_tiles = tile2("Avg power ⟨|E/A₀|²⟩", pow_s, pow_o, x->@sprintf("%.3g", x))

    seg_s = seg_stats(Eseed, segwin_s)
    seg_o = b.have_opt ? seg_stats(Eopt, segwin_o) : seg_s
    function segrow(i)
        ds,ps,rs = seg_s[i][1]*1e6, seg_s[i][2], seg_s[i][4]
        doo,po,ro = seg_o[i][1]*1e6, seg_o[i][2], seg_o[i][4]
        pc(a,b2) = @sprintf("%+.0f%%", 100*(b2/a-1))
        @sprintf("""        <tr><td>%d</td>
          <td><span class="s">%.1f</span> / <span class="o">%.1f</span><span class="delta">%s</span></td>
          <td><span class="s">%.3g</span> / <span class="o">%.3g</span><span class="delta">%s</span></td>
          <td><span class="s">%.3g</span> / <span class="o">%.3g</span><span class="delta">%s</span></td></tr>""",
          i, ds, doo, pc(ds,doo), ps/1e7, po/1e7, pc(ps,po), rs/1e11, ro/1e11, pc(rs,ro))
    end
    d1s = seg_s[1][1]; d1o = seg_o[1][1]
    r1s = seg_s[1][4]; r1o = seg_o[1][4]
    p1s = seg_s[1][2]; p1o = seg_o[1][2]
    ratio_row = @sprintf("""        <tr class="ratio"><td>ratio (seg1=1)</td>
          <td><span class="s">1 : %.2f : %.2f</span> / <span class="o">1 : %.2f : %.2f</span></td>
          <td><span class="s">1 : %.2f : %.2f</span> / <span class="o">1 : %.2f : %.2f</span></td>
          <td><span class="s">1 : %.2f : %.2f</span> / <span class="o">1 : %.2f : %.2f</span></td></tr>""",
        seg_s[2][1]/d1s, seg_s[3][1]/d1s, seg_o[2][1]/d1o, seg_o[3][1]/d1o,
        seg_s[2][2]/p1s, seg_s[3][2]/p1s, seg_o[2][2]/p1o, seg_o[3][2]/p1o,
        seg_s[2][4]/r1s, seg_s[3][4]/r1s, seg_o[2][4]/r1o, seg_o[3][4]/r1o)

    os = b.opt_settings
    foot_run = os === nothing ? "no run log" :
        "param_budget=$(get(os,:param_budget,"?"))  k=$(b.k)  n_coeff_A=n_coeff_f=$(b.nA)  " *
        "grad_mode=$(get(os,:grad_mode,"?"))  track=$(get(os,:track,"?"))  " *
        "lr=$(get(os,:learning_rate,"?"))  cf_lr_scale=$(get(os,:cf_lr_scale,"?"))  " *
        "kappa_I=$(get(os,:kappa_I,"?"))  num_epochs=$(get(os,:num_epochs,"?"))  n_hops=$(get(os,:n_hops,"?"))"

    dur_from = im === nothing ? nothing : im[4]
    dur_to   = fm === nothing ? nothing : fm[4]
    rt_txt   = b.runtime_s === nothing ? "" : @sprintf(" Optimised in <b>%s</b>.", fmt_dur(b.runtime_s))
    dek = if b.have_opt && dur_from !== nothing && dur_to !== nothing
        @sprintf("Warm start &rarr; optimised control pulse for <b>%s</b>. Duration <b>%.1f &micro;s &rarr; %.1f &micro;s</b> (%+.0f%%) at held fidelity.%s",
                 esc(b.stem), dur_from*1e6, dur_to*1e6, 100*(dur_to/dur_from-1), rt_txt)
    else
        "Seed / control-pulse report for <b>$(esc(b.stem))</b>.$rt_txt"
    end

    cost_panel = (b.history !== nothing && !isempty(b.history)) ? """

  <section class="panel">
    <h2>Cost vs epoch</h2>
    <p class="sub">Optimiser cost across every hop (cumulative epoch axis). Orange = cost, blue dashed = running best,
      dots mark epochs that improved the best. Dashed verticals are hop boundaries$(any(r->r.cost>0, b.history) && maximum(r.cost for r in b.history)/max(minimum(r.cost for r in b.history),1e-9) > 12 ? "; y-axis is log" : "").</p>
    <div class="chartbox" id="boxC"><div class="tip" id="tipC"></div></div>
  </section>""" : ""

    body = """
<div class="wrap">
  <header>
    <p class="eyebrow">Pulse report &nbsp;·&nbsp; $(esc(b.stem))</p>
    <h1>Seed vs Optimised pulse</h1>
    <p class="dek">$dek</p>
  </header>
  <div class="stats">
$dur_tiles
$inv_tiles
$sil_tiles
$pow_tiles
  </div>

  <section class="panel">
    <h2>Amplitude envelope &nbsp; |E(t)|</h2>
    <p class="sub">Delivered drive amplitude (B-spline &times; C<sup>&infin;</sup> edge taper), units of 10<sup>7</sup> rad s<sup>&minus;1</sup>. Bands mark the seed's segments.</p>
    <div class="legend">
      <span class="item"><span class="swatch seed"></span><b>Seed</b></span>
      <span class="item"><span class="swatch opt"></span><b>Optimised</b></span>
    </div>
    <div class="chartbox" id="boxA"><div class="tip" id="tipA"></div></div>
  </section>

  <section class="panel">
    <h2>B-spline amplitude &nbsp; A(t)</h2>
    <p class="sub">The raw <span class="mono">cA</span> envelope <em>before</em> the taper &mdash; what the amplitude coefficients encode. The clamped-spline endpoints sit high; the taper crushes each segment's edges to zero (see |E(t)| above).$(clipped ? " Display clipped at 3&times; the |E| peak." : "") Units of 10<sup>7</sup> rad s<sup>&minus;1</sup>.</p>
    <div class="legend">
      <span class="item"><span class="swatch seed"></span><b>Seed</b></span>
      <span class="item"><span class="swatch opt"></span><b>Optimised</b></span>
    </div>
    <div class="chartbox" id="boxD"><div class="tip" id="tipD"></div></div>
  </section>

  <section class="panel">
    <h2>Instantaneous frequency</h2>
    <p class="sub">Chirp d&phi;/dt, units of 10<sup>7</sup> rad s<sup>&minus;1</sup>.</p>
    <div class="legend">
      <span class="item"><span class="swatch seed"></span><b>Seed</b></span>
      <span class="item"><span class="swatch opt"></span><b>Optimised</b></span>
    </div>
    <div class="chartbox" id="boxB"><div class="tip" id="tipB"></div></div>
  </section>

  <section class="panel">
    <h2>Per-segment structure</h2>
    <p class="sub">Seed / optimised, with change. Chirp rate = frequency span / segment duration.</p>
    <div class="tablewrap"><table><thead>
      <tr><th>seg</th><th>duration µs <span class="s">seed</span>/<span class="o">opt</span></th>
          <th>peak |E| (×10<sup>7</sup>)</th><th>chirp rate (×10<sup>11</sup> rad s<sup>&minus;2</sup>)</th></tr>
    </thead><tbody>
$(join((segrow(i) for i in 1:b.k), "\n"))
$(b.k == 3 ? ratio_row : "")
    </tbody></table></div>
  </section>

$(cfg_section(b))

$(spline_section(b))
$cost_panel

  <footer>
    <div class="kv"><b>run</b> &nbsp; $foot_run</div>
    <div class="kv"><b>runtime</b> &nbsp; $(b.runtime_s === nothing ? "not recorded (pass --runtime SECONDS)" : fmt_dur(b.runtime_s))</div>
    <div class="kv"><b>power</b> &nbsp; $(@sprintf("avg ⟨|E/A₀|²⟩ %.3g → %.3g (%+.1f%%)  ·  energy ∫|E|²dt %.3g → %.3g (%+.1f%%)", pow_s, pow_o, 100*(pow_o/pow_s-1), en_s, en_o, 100*(en_o/en_s-1)))</div>
    <div class="kv"><b>seed</b> &nbsp; $(esc(b.seed_kind))</div>
    <div class="kv"><b>generated</b> &nbsp; write_pulse_report</div>
  </footer>
</div>

<script type="application/json" id="pulsedata">$json</script>
<script>$JS</script>
"""

    return """<title>$(esc(b.stem))</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zilla+Slab:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>$CSS</style>
$body
"""
end

"""
    write_pulse_report(stem_or_path; data_dir=nothing, out_dir=nothing, log_dir=nothing,
                       param_budget=120, runtime_s=nothing, verbose=true) -> outpath

Generate `<out_dir>/<stem>.html`. `stem_or_path` may be a bare stem or a
`<stem>.jld2` path; when it is a path, `data_dir`/`out_dir` default from it
(`out_dir` -> a sibling `html/` of the data dir's parent). `log_dir` is where
the `_optrunlog.jld2`/`_opt_pulsepara.jld2` siblings live (defaults to the
data dir) — pass it when the optimiser wrote its log to a separate directory.
"""
function write_pulse_report(stem_or_path::AbstractString;
                            data_dir::Union{Nothing,AbstractString}=nothing,
                            out_dir::Union{Nothing,AbstractString}=nothing,
                            log_dir::Union{Nothing,AbstractString}=nothing,
                            param_budget::Integer=120,
                            runtime_s::Union{Nothing,Real}=nothing,
                            verbose::Bool=true)
    if endswith(stem_or_path, ".jld2")
        dd0  = dirname(stem_or_path)
        stem = basename(stem_or_path)[1:end-length(".jld2")]
    else
        dd0  = data_dir === nothing ? joinpath("data", "data_1st_order") : String(data_dir)
        stem = stem_or_path
    end
    dd = data_dir === nothing ? dd0 : String(data_dir)
    ld = log_dir === nothing ? nothing : String(log_dir)
    od = out_dir === nothing ? joinpath(dirname(rstrip(dd, ('/', '\\'))), "html") : String(out_dir)
    rt = runtime_s === nothing ? nothing : Float64(runtime_s)

    verbose && println("pulse_report: stem=$stem  data-dir=$dd  " *
                       (ld === nothing ? "" : "log-dir=$ld  ") * "out-dir=$od")
    b = load_bundle(stem, dd, Int(param_budget), rt, ld)
    verbose && println("  k=$(b.k)  n_coeff_A=$(b.nA)  n_coeff_f=$(b.nf)  " *
                       "have_opt=$(b.have_opt)  seed=$(b.seed_kind)")
    html = build_report(b)
    mkpath(od)
    outp = joinpath(od, "$(stem).html")
    write(outp, html)
    verbose && println("  wrote $outp  ($(filesize(outp)) bytes)")
    return outp
end

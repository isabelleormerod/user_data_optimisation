#!/usr/bin/env python3
r"""
quality_viz.py - Raw Tracking Data-Quality Dashboard (lazy-loaded, single UI)

Stage-1 sense-checking for the pen / body / hand streams. Reads the RAW per-frame
trial CSVs directly -- NOT the fPCA cache, which only keeps curves resampled
within Place events and has therefore discarded the between-event gaps and
idle-period dropout this tool exists to reveal.

OUTPUT (a small folder, not one giant file -- see WHY below):
    <out>/quality_dashboard.html   the UI (participant -> trial -> marker -> zoom)
    <out>/data/<PID>__<stem>.js    one lightweight data file per trial
The dashboard loads only the trial you select, so it stays fast no matter how
many participants/trials exist.

WHY LAZY: embedding every trial's full raw coordinates in one HTML is hundreds
of MB at cohort scale (75 trials x ~40 markers x thousands of frames) -- slow to
build and painful to open. Here the HTML is small; each trial's coordinates live
in their own data/<PID>__<stem>.js loaded on demand via a <script> tag (which
works offline under file://, unlike fetch/XHR).

ON EACH SELECTION it redraws:
  - x/y/z over time; GAPS as line breaks + an orange rug (NaN/zero-fill frames
    AND frame-interval spikes); JUMPS as red x's (robust MAD speed outliers),
  - green PLACE-event shading,
  - a "Zoom to height" control that snaps x AND y to the High/Medium/Low window.
A worst-first summary table (embedded, all trials) sits on top; click a row to
jump straight to that trial + marker.

USAGE:
    python quality_viz.py --landmarks-root A:\Automated_chain_BETA\Participant_Landmarks
    python quality_viz.py --landmarks-root ... --out .\quality_output   # write locally instead of the A: drive
    python quality_viz.py --landmarks-root ... --participants P002,P003
    python quality_viz.py --landmarks-root ... --trials Short_Large --max-trials 8
    python quality_viz.py --landmarks-root ... --stride 2 --round 4

  Optional:
    --out PATH          Output dir (DEFAULT <landmarks-root>/metrics/quality_viz -- i.e. ON THE A: DRIVE;
                        pass --out .\somewhere to write next to where you run the script)
    --participants ...  Comma-separated PIDs
    --trials SUBSTR     Only trials whose stem contains any given substring (comma-separated)
    --max-trials N      Cap number of trials
    --gap-factor F      dt > F * median(dt) counts as a time-gap (default 2.5)
    --jump-mad K        speed > median + K*MAD counts as a jump (default 6.0)
    --stride N          Plot every Nth frame of the coordinate LINES (default 1). Gaps/jumps are
                        detected on FULL resolution regardless, so nothing is hidden; this only
                        thins the drawn line to shrink the data files.
    --round N           Round embedded coords to N decimals (default 4)

NOTES / ASSUMPTIONS TO VERIFY:
  - A frame is "dropout" if its coords are NaN OR all-zero (Quest/MediaPipe often
    zero-fill a lost marker). If loss is flagged via a confidence/visibility/
    data_quality column instead, tell me the column and I'll use it.
  - Jumps: speed = |dp|/dt, per-marker median+K*MAD threshold. Quaternion markers
    are shown but not jump-tested.
  - Reads siblings *_body(_labelled).csv / *_hand(_labelled).csv next to the
    labelled pen file. Place/height shading needs the pen Place + High/Medium/Low
    columns; absent -> no shading, everything else still works.

Self-contained: numpy + pandas + plotly only, no project/utils imports.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

HEIGHTS = ["High", "Medium", "Low"]


# ------------------------------------------------------------------ discovery
def find_labelled_pen(trial_dir):
    for pat in ("*_pen_flattened_labelled.csv", "*_pen_labelled.csv"):
        hits = sorted(trial_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def pen_stem(pen_path):
    stem = pen_path.stem
    for suffix in ("_pen_flattened_labelled", "_pen_labelled"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def find_sibling(trial_dir, stem, stream):
    for name in (f"{stem}_{stream}_labelled.csv", f"{stem}_{stream}.csv"):
        p = trial_dir / name
        if p.is_file():
            return p
    return None


def iter_trials(root, pfilter, trial_substrs):
    for pid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pid = pid_dir.name
        if pfilter and pid not in pfilter:
            continue
        for trial_dir in sorted(t for t in pid_dir.iterdir() if t.is_dir()):
            pen = find_labelled_pen(trial_dir)
            if pen is None:
                continue
            stem = pen_stem(pen)
            if trial_substrs and not any(s in stem for s in trial_substrs):
                continue
            yield stem, pid, trial_dir, pen


# ------------------------------------------------------------ marker detection
def detect_markers(df, stream):
    cols = set(df.columns)
    markers = {}
    if stream == "pen":
        for tag, trip in (("tip (flattened)", ("x_flat", "y_flat", "z_flat")),
                          ("tip (raw)", ("x", "y", "z"))):
            if all(c in cols for c in trip):
                markers[tag] = {"kind": "xyz", "cols": list(trip)}
    for c in df.columns:
        if c.endswith("_x"):
            base = c[:-2]
            trip = (f"{base}_x", f"{base}_y", f"{base}_z")
            if all(t in cols for t in trip):
                markers[base] = {"kind": "xyz", "cols": list(trip)}
    for c in df.columns:
        if c.endswith("_qw"):
            base = c[:-3]
            quad = (f"{base}_qw", f"{base}_qx", f"{base}_qy", f"{base}_qz")
            if all(q in cols for q in quad):
                markers[f"{base} (quat)"] = {"kind": "quat", "cols": list(quad)}
    if stream == "pen" and all(q in cols for q in ("qw", "qx", "qy", "qz")):
        markers["orientation (quat)"] = {"kind": "quat", "cols": ["qw", "qx", "qy", "qz"]}
    return markers


# ---------------------------------------------------------------- run helpers
def contiguous_runs(mask):
    mask = np.asarray(mask)
    if not mask.any():
        return
    idx = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks], idx[-1]]
    for s, e in zip(starts, stops):
        yield int(s), int(e)


def col_spans(df, col):
    if col not in df.columns or "t_s" not in df.columns:
        return []
    t = df["t_s"].to_numpy(float)
    flag = df[col].astype(str).str.strip().isin(["1", "1.0", "True", "true"]).to_numpy()
    return [[float(t[s]), float(t[e])] for s, e in contiguous_runs(flag)]


# --------------------------------------------------------- per-marker analysis
def analyse_marker(t, XYZ, gap_factor, jump_mad):
    T = len(t)
    finite = np.isfinite(XYZ).all(axis=1)
    nonzero = ~(np.abs(XYZ) < 1e-9).all(axis=1)
    valid = finite & nonzero
    dropout = ~valid

    dt = np.full(T, np.nan)
    if T > 1:
        dt[1:] = np.diff(t)
    med_dt = np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else np.nan
    gap_frame = (dt > gap_factor * med_dt) if np.isfinite(med_dt) else np.zeros(T, bool)
    gap_any = dropout | np.nan_to_num(gap_frame, nan=False).astype(bool)

    jump = np.zeros(T, bool)
    vi = np.flatnonzero(valid)
    if len(vi) > 2:
        dp = np.linalg.norm(np.diff(XYZ[vi], axis=0), axis=1)
        ddt = np.diff(t[vi]); ddt[ddt <= 0] = np.nan
        sp = dp / ddt
        med = np.nanmedian(sp)
        mad = np.nanmedian(np.abs(sp - med))
        thr = med + jump_mad * 1.4826 * mad if (mad and np.isfinite(mad)) else np.inf
        jump[vi[1:]] = sp > thr

    return {"dropout": dropout, "gap_any": gap_any, "dt": dt, "med_dt": med_dt, "jump": jump,
            "dropout_frac": float(dropout.mean()) if T else 1.0,
            "max_gap_s": float(np.nanmax(dt)) if np.any(np.isfinite(dt)) else 0.0,
            "n_jumps": int(jump.sum())}


def in_place_dropout(t, dropout, spans):
    if not spans:
        return None
    ip = np.zeros(len(t), bool)
    for s, e in spans:
        ip |= (t >= s) & (t <= e)
    return float(dropout[ip].mean()) if ip.any() else None


def in_place_dropout_by_height(t, dropout, place_spans, height_spans):
    """Dropout fraction within Place frames that fall in each working-height window."""
    place = np.zeros(len(t), bool)
    for s, e in place_spans:
        place |= (t >= s) & (t <= e)
    out = {}
    for h, spans in (height_spans or {}).items():
        hm = np.zeros(len(t), bool)
        for s, e in spans:
            hm |= (t >= s) & (t <= e)
        m = place & hm
        out[h] = float(dropout[m].mean()) if m.any() else None
    return out


def jsonable(a, rnd):
    out = np.round(np.asarray(a, float), rnd)
    return [None if not np.isfinite(v) else float(v) for v in out]


# ---------------------------------------------------------------- build a trial
def build_trial_record(pen_df, trial_dir, stem, args):
    spans = col_spans(pen_df, "Place")
    hspans = {h: col_spans(pen_df, h) for h in HEIGHTS if h in pen_df.columns}
    streams_out, summary = {}, []
    for sname, src in (("pen", pen_df),
                       ("body", find_sibling(trial_dir, stem, "body")),
                       ("hand", find_sibling(trial_dir, stem, "hand"))):
        df = src if isinstance(src, pd.DataFrame) else (pd.read_csv(src) if src else None)
        if df is None or df.empty or "t_s" not in df.columns:
            continue
        markers = detect_markers(df, sname)
        if not markers:
            continue
        t = df["t_s"].to_numpy(float)
        stride = max(1, args.stride)
        t_disp = t[::stride]
        mout, med_dt, worst = {}, None, None
        for mname, info in markers.items():
            arr = df[info["cols"]].to_numpy(float)
            if info["kind"] == "xyz":
                a = analyse_marker(t, arr, args.gap_factor, args.jump_mad)
                med_dt = a["med_dt"]
                ipd_h = in_place_dropout_by_height(t, a["dropout"], spans, hspans)
                ip_parts = [f"{h[0]} {v*100:.1f}%" for h, v in
                            (("High", ipd_h.get("High")), ("Medium", ipd_h.get("Medium")), ("Low", ipd_h.get("Low")))
                            if v is not None]
                ip_txt = f" (in-Place {' '.join(ip_parts)})" if ip_parts else ""
                meta = (f"{sname}: {mname} — dropout {a['dropout_frac']*100:.1f}%{ip_txt}"
                        f" | max gap {a['max_gap_s']*1000:.0f} ms | jumps {a['n_jumps']}")
                # store display (possibly strided) coords, but jumps/gaps as (t,value) from FULL res
                mout[mname] = {"kind": "xyz",
                               "x": jsonable(arr[::stride, 0], args.round),
                               "y": jsonable(arr[::stride, 1], args.round),
                               "z": jsonable(arr[::stride, 2], args.round),
                               "jt": jsonable(t[a["jump"]], 4),
                               "jy": jsonable(arr[a["jump"], 0], args.round),
                               "gt": jsonable(t[a["gap_any"]], 4),
                               "meta": meta}
                rec = {"pid": trial_dir.parent.name, "trial": stem, "stream": sname,
                       "dropout_pct": a["dropout_frac"] * 100,
                       "in_place_H": (None if ipd_h.get("High") is None else ipd_h["High"] * 100),
                       "in_place_M": (None if ipd_h.get("Medium") is None else ipd_h["Medium"] * 100),
                       "in_place_L": (None if ipd_h.get("Low") is None else ipd_h["Low"] * 100),
                       "max_gap_ms": a["max_gap_s"] * 1000, "n_jumps": a["n_jumps"],
                       "worst_marker": mname}
                if worst is None or rec["dropout_pct"] > worst["dropout_pct"]:
                    worst = rec
            else:
                mout[mname] = {"kind": "quat",
                               "qw": jsonable(arr[::stride, 0], args.round),
                               "qx": jsonable(arr[::stride, 1], args.round),
                               "qy": jsonable(arr[::stride, 2], args.round),
                               "qz": jsonable(arr[::stride, 3], args.round),
                               "meta": f"{sname}: {mname} — orientation (quaternion), not jump-tested"}
        streams_out[sname] = {"t": jsonable(t_disp, 4),
                              "dt": jsonable(np.r_[np.nan, np.diff(t_disp)] if len(t_disp) > 1 else [np.nan], 4),
                              "gap_thresh": (None if not (med_dt and np.isfinite(med_dt))
                                             else round(args.gap_factor * med_dt, 4)),
                              "markers": mout}
        if worst:
            summary.append(worst)
    rec = {"place_spans": spans, "height_spans": hspans, "streams": streams_out}
    return rec, summary


# ---------------------------------------------------------------- HTML template
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Data-quality dashboard</title>
<script>__PLOTLY__</script>
<style>
 html,body{margin:0} body{font-family:"Linux Libertine","Linux Libertine O","Libertinus Serif",Libertine,"Times New Roman",Georgia,serif;margin:18px;color:#2c3e50;overflow-y:auto}
 h1{font-size:18px;margin:0 0 2px} .note{color:#666;font-size:12px;margin:4px 0}
 .controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:10px 0}
 label{font-size:12px;color:#555;display:block;margin-bottom:2px}
 select{font-size:13px;padding:4px 6px;min-width:210px}
 #meta{font-size:13px;color:#333;margin:6px 0;min-height:18px}
 #plot{width:100%;height:640px}
 details{margin:8px 0 14px} summary{cursor:pointer;font-size:13px}
 table{border-collapse:collapse;font-size:12px;margin-top:6px}
 th,td{border:1px solid #ddd;padding:4px 8px;text-align:left}
 th{background:#f5f6f7} tr.click{cursor:pointer} tr.click:hover{background:#eef6ff}
</style></head><body>
<h1>Raw tracking data-quality dashboard</h1>
<div class="note" id="scope">__SCOPE__</div>
<div class="note">Dropout = NaN/zero frames. <b>in-Place H/M/L</b> = dropout inside Place windows at each working height.
 Orange rug / line breaks = gaps; red &times; = jumps; green = Place events; blue = selected height window.</div>

<details open><summary>Summary (worst first) — click a row to open it</summary>
<table id="summary"><thead><tr><th>Participant</th><th>Trial</th><th>Stream</th>
 <th>dropout %</th><th>in-Place H %</th><th>in-Place M %</th><th>in-Place L %</th><th>max gap (ms)</th><th>jumps</th><th>worst marker</th></tr></thead>
<tbody></tbody></table></details>

<div class="controls">
 <div><label>Participant</label><select id="pid"></select></div>
 <div><label>Trial</label><select id="trial"></select></div>
 <div><label>Marker (coordinates)</label><select id="marker"></select></div>
 <div><label>Zoom to height</label><select id="zoom">
   <option value="Full">Full trial</option><option value="High">High</option>
   <option value="Medium">Medium</option><option value="Low">Low</option></select></div>
</div>
<div id="meta"></div>
<div id="plot"></div>

<script>
const INDEX = __INDEX__;          // {participants:[], trials:{pid:[stem,...]}}
const SUMMARY = __SUMMARY__;      // [{pid,trial,stream,...}]
const CACHE = {};                 // key "pid/stem" -> trial record (lazy)
let _cb = null;
window.__REG__ = function(pid, stem, rec){ CACHE[pid+'/'+stem] = rec; if(_cb){const f=_cb;_cb=null;f();} };

const pidSel=document.getElementById('pid'), trialSel=document.getElementById('trial'),
      markerSel=document.getElementById('marker'), zoomSel=document.getElementById('zoom'),
      plot=document.getElementById('plot'), metaDiv=document.getElementById('meta');

function opt(sel, vals){ sel.innerHTML=''; vals.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;sel.appendChild(o);}); }

function ensureTrial(pid, stem, cb){ const key=pid+'/'+stem;
  if(CACHE[key]){ cb(); return; }
  metaDiv.textContent='loading '+stem+' …'; _cb=cb;
  const s=document.createElement('script'); s.src='data/'+pid+'__'+stem+'.js';
  s.onerror=()=>{ _cb=null; metaDiv.textContent='could not load data/'+pid+'__'+stem+'.js'; };
  document.head.appendChild(s); }

function fillTrials(){ opt(trialSel, INDEX.trials[pidSel.value]||[]); onTrial(); }
function onTrial(){ const pid=pidSel.value, stem=trialSel.value; if(!stem)return;
  ensureTrial(pid, stem, fillMarkers); }
function fillMarkers(){ const rec=CACHE[pidSel.value+'/'+trialSel.value];
  markerSel.innerHTML='';
  ['pen','body','hand'].forEach(s=>{ if(!rec.streams[s])return;
    const og=document.createElement('optgroup'); og.label=s;
    Object.keys(rec.streams[s].markers).forEach(m=>{ const o=document.createElement('option');
      o.value=s+'||'+m; o.textContent=m; og.appendChild(o); });
    markerSel.appendChild(og); });
  draw(); }

function rangeOver(t,series,x0,x1){ let lo=Infinity,hi=-Infinity;
  for(let i=0;i<t.length;i++){ if(t[i]>=x0&&t[i]<=x1){ series.forEach(a=>{const v=a[i];
    if(v!=null){ if(v<lo)lo=v; if(v>hi)hi=v; } }); } }
  return (isFinite(lo)&&isFinite(hi))?[lo,hi]:null; }

function draw(){ const pid=pidSel.value, stem=trialSel.value; if(!markerSel.value)return;
  const rec=CACHE[pid+'/'+stem]; const [sname,mname]=markerSel.value.split('||');
  const st=rec.streams[sname], m=st.markers[mname], t=st.t; const traces=[];
  const zoom=zoomSel.value; let xr=null, yr=null;
  if(zoom!=='Full'){ const hs=((rec.height_spans||{})[zoom])||[];
    if(hs.length){ const lo=Math.min(...hs.map(s=>s[0])), hi=Math.max(...hs.map(s=>s[1]));
      const pad=((hi-lo)*0.05)||0.1; xr=[lo-pad,hi+pad];
      if(m.kind==='xyz'){ const yy=rangeOver(t,[m.x,m.y,m.z],xr[0],xr[1]);
        if(yy){ const yp=((yy[1]-yy[0])*0.08)||0.01; yr=[yy[0]-yp,yy[1]+yp]; } } } }
  if(m.kind==='xyz'){
    ['x','y','z'].forEach(ax=>traces.push({x:t,y:m[ax],name:ax,mode:'lines',line:{width:1},connectgaps:false}));
    traces.push({x:m.jt,y:m.jy,name:'jump',mode:'markers',marker:{color:'red',symbol:'x',size:8}});
    const ylo=(yr?yr[0]:(rangeOver(t,[m.x,m.y,m.z],t[0],t[t.length-1])||[0])[0]);
    traces.push({x:m.gt,y:m.gt.map(()=>ylo),name:'gap',mode:'markers',marker:{color:'orange',symbol:'line-ns-open',size:10}});
    metaDiv.textContent=m.meta;
  } else {
    ['qw','qx','qy','qz'].forEach(ax=>traces.push({x:t,y:m[ax],name:ax,mode:'lines',line:{width:1},connectgaps:false}));
    metaDiv.textContent=m.meta;
  }
  const shapes=[];
  if(zoom!=='Full'){ (((rec.height_spans||{})[zoom])||[]).forEach(([s,e])=>{
    shapes.push({type:'rect',xref:'x',yref:'paper',x0:s,x1:e,y0:0,y1:1,fillcolor:'#3498db',opacity:0.10,line:{width:0},layer:'below'}); }); }
  (rec.place_spans||[]).forEach(([s,e])=>{
    shapes.push({type:'rect',xref:'x',yref:'paper',x0:s,x1:e,y0:0,y1:1,fillcolor:'green',opacity:0.07,line:{width:0},layer:'below'}); });
  const yTitle=(m.kind==='xyz')?'position (m)':'quaternion component';
  const layout={margin:{t:10,r:16,l:64,b:52},showlegend:true,
    font:{family:'"Linux Libertine","Linux Libertine O","Libertinus Serif",Libertine,"Times New Roman",Georgia,serif'},
    legend:{orientation:'h',y:1.06,x:1,xanchor:'right'},shapes:shapes,
    xaxis:{title:'time (s)',range:xr,autorange:xr?false:true},
    yaxis:{title:yTitle,range:yr,autorange:yr?false:true}};
  Plotly.react(plot,traces,layout,{responsive:true,displayModeBar:true,scrollZoom:false});
}

(function(){ const tb=document.querySelector('#summary tbody');
  SUMMARY.forEach(r=>{ const tr=document.createElement('tr'); tr.className='click';
    const f1=v=>v==null?'n/a':v.toFixed(1);
    const cells=[r.pid,r.trial,r.stream,r.dropout_pct.toFixed(1),
      f1(r.in_place_H),f1(r.in_place_M),f1(r.in_place_L),r.max_gap_ms.toFixed(0),r.n_jumps,r.worst_marker];
    const ipv={4:r.in_place_H,5:r.in_place_M,6:r.in_place_L};
    cells.forEach((c,i)=>{const td=document.createElement('td');td.textContent=c;
      if(i===3&&r.dropout_pct>=20)td.style.color='#c0392b'; else if(i===3&&r.dropout_pct>=5)td.style.color='#e67e22';
      if(i in ipv && ipv[i]!=null){ if(ipv[i]>=10)td.style.color='#c0392b'; else if(ipv[i]>=1)td.style.color='#e67e22'; }
      tr.appendChild(td);});
    tr.onclick=()=>{ pidSel.value=r.pid; opt(trialSel,INDEX.trials[r.pid]||[]); trialSel.value=r.trial;
      zoomSel.value='Full';
      ensureTrial(r.pid,r.trial,()=>{ fillMarkers(); markerSel.value=r.stream+'||'+r.worst_marker; draw();
        document.getElementById('plot').scrollIntoView({behavior:'smooth'}); }); };
    tb.appendChild(tr); }); })();

opt(pidSel, INDEX.participants); fillTrials();
pidSel.onchange=fillTrials; trialSel.onchange=onTrial; markerSel.onchange=draw; zoomSel.onchange=draw;
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks-root", type=Path, required=True)
    ap.add_argument("--participants", type=str, default=None)
    ap.add_argument("--trials", type=str, default=None)
    ap.add_argument("--max-trials", type=int, default=None)
    ap.add_argument("--gap-factor", type=float, default=2.5)
    ap.add_argument("--jump-mad", type=float, default=6.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--round", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.landmarks_root.is_dir():
        sys.exit(f"ERROR: {args.landmarks_root} is not a directory")

    out_dir = args.out or (args.landmarks_root / "metrics" / "quality_viz")
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    pfilter = {p.strip() for p in args.participants.split(",")} if args.participants else None
    trial_substrs = [s.strip() for s in args.trials.split(",")] if args.trials else None
    trials = list(iter_trials(args.landmarks_root, pfilter, trial_substrs))
    if args.max_trials:
        trials = trials[:args.max_trials]
    if not trials:
        sys.exit("No matching trials found.")
    print(f"Found {len(trials)} trial(s).")

    index = {"participants": [], "trials": {}}
    summary = []
    total_bytes = 0
    for stem, pid, trial_dir, pen_path in trials:
        pen_df = pd.read_csv(pen_path)
        rec, tsum = build_trial_record(pen_df, trial_dir, stem, args)
        if not rec["streams"]:
            print(f"  [skip] {pid}/{stem}: no usable streams")
            continue
        js = f"window.__REG__({json.dumps(pid)},{json.dumps(stem)},{json.dumps(rec, separators=(',', ':'))});"
        fpath = data_dir / f"{pid}__{stem}.js"
        fpath.write_text(js, encoding="utf-8")
        total_bytes += fpath.stat().st_size
        summary.extend(tsum)
        index["trials"].setdefault(pid, []).append(stem)
        if pid not in index["participants"]:
            index["participants"].append(pid)
        print(f"  [ok]  {pid}/{stem}  ({len(rec['streams'])} stream(s), {len(rec['place_spans'])} Place event(s))")

    def _ipmax(r):
        vals = [v for v in (r["in_place_H"], r["in_place_M"], r["in_place_L"]) if v is not None]
        return max(vals) if vals else -1
    summary.sort(key=lambda r: (-_ipmax(r), -r["dropout_pct"]))
    scope = (f"Source: {args.landmarks_root}  —  {len(index['participants'])} participant(s), "
             f"{sum(len(v) for v in index['trials'].values())} trial(s)  ·  data {total_bytes/1e6:.1f} MB across "
             f"{sum(len(v) for v in index['trials'].values())} file(s)")
    html = (PAGE.replace("__PLOTLY__", get_plotlyjs())
                .replace("__SCOPE__", scope)
                .replace("__INDEX__", json.dumps(index, separators=(",", ":")))
                .replace("__SUMMARY__", json.dumps(summary, separators=(",", ":"))))
    out_path = out_dir / "quality_dashboard.html"
    out_path.write_text(html, encoding="utf-8")

    print("\n" + "=" * 66)
    print("OUTPUT WRITTEN TO (open this file):")
    print(f"   {out_path.resolve()}")
    print("   NOTE: by default this is under your --landmarks-root (the A: drive),")
    print("   NOT the folder you ran the script from. Use --out .\\somewhere to change it.")
    print("=" * 66)


if __name__ == "__main__":
    main()

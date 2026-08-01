#!/usr/bin/env python3
"""
Fit a Gompertz growth model to dog weight data and generate a GitHub Pages site.

Usage:
    python fit_model.py
Outputs:
    docs/index.html   — interactive GitHub Pages site
"""

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("weights.csv")
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
df = df.sort_values("date").reset_index(drop=True)

ref_date = df["date"].min()
df["days"] = (df["date"] - ref_date).dt.days.astype(float)
t_data = df["days"].values
w_data = df["weight"].values.astype(float)

# ── Gompertz growth model: W(t) = L * exp(-exp(-k*(t - t_mid))) ──────────────
# Inflection at W = L/e ≈ 37% of adult weight (vs 50% for logistic).
# Better suited for animal growth: rapid early phase, slow late plateau.
def gompertz(t, L, k, t_mid):
    return L * np.exp(-np.exp(-k * (t - t_mid)))

popt, pcov = curve_fit(
    gompertz, t_data, w_data,
    p0=[25.0, 0.04, 65.0],
    bounds=([10, 0.001, 0], [100, 1, 500]),
    maxfev=20_000,
)
L_fit, k_fit, t_mid_fit = popt

w_pred = gompertz(t_data, *popt)
ss_res = np.sum((w_data - w_pred) ** 2)
ss_tot = np.sum((w_data - np.mean(w_data)) ** 2)
r2 = float(1 - ss_res / ss_tot)
rmse = float(np.sqrt(ss_res / len(w_data)))

# ── Prediction curve: from 10 days before start to 2 years out ────────────────
t_curve = np.linspace(-10, 730, 2000)
w_curve = gompertz(t_curve, *popt)

# 95 % confidence band via Monte Carlo
rng = np.random.default_rng(42)
param_samples = rng.multivariate_normal(popt, pcov, 3000)
# Clip samples to physical bounds (positive L and k)
param_samples = param_samples[
    (param_samples[:, 0] > 0) & (param_samples[:, 1] > 0)
]
w_mc = np.array([gompertz(t_curve, *s) for s in param_samples])
w_lo = np.percentile(w_mc, 2.5, axis=0)
w_hi = np.percentile(w_mc, 97.5, axis=0)

# ── Date helpers ───────────────────────────────────────────────────────────────
def to_iso(days_arr):
    return [
        (ref_date + timedelta(days=float(d))).strftime("%Y-%m-%d")
        for d in days_arr
    ]

today = datetime.now()
today_days = (today - ref_date.to_pydatetime()).days
split_idx = int(np.searchsorted(t_curve, today_days))

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"Gompertz fit results:")
print(f"  L    = {L_fit:.3f} kg  (estimated adult weight)")
print(f"  k    = {k_fit:.6f}  (growth rate)")
print(f"  t₀   = {t_mid_fit:.1f} days  (inflection point)")
print(f"  R²   = {r2:.6f}")
print(f"  RMSE = {rmse:.4f} kg")
print(f"  Inflection date: {(ref_date + timedelta(days=t_mid_fit)).strftime('%d %b %Y')}")

# ── JSON payload for the website ───────────────────────────────────────────────
payload = {
    "ref_date":    ref_date.strftime("%Y-%m-%d"),
    "today_days":  int(today_days),
    "today_iso":   today.strftime("%Y-%m-%d"),
    "L":           float(L_fit),
    "k":           float(k_fit),
    "t_mid":       float(t_mid_fit),
    "r2":          r2,
    "rmse":        rmse,
    "data_x":      to_iso(t_data),
    "data_y":      [float(v) for v in w_data],
    "curve_x":     to_iso(t_curve),
    "curve_y":     [float(v) for v in w_curve],
    "ci_lo":       [float(v) for v in w_lo],
    "ci_hi":       [float(v) for v in w_hi],
    "split_idx":   split_idx,
}
payload_json = json.dumps(payload)

# ── HTML template ──────────────────────────────────────────────────────────────
# Uses __PAYLOAD__ placeholder to avoid f-string curly-brace escaping.

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Inu Growth Tracker</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', system-ui, sans-serif;
      background: #F7F4F0;
      color: #1C1917;
      min-height: 100vh;
    }

    /* ── layout ── */
    .container { max-width: 1080px; margin: 0 auto; padding: 0 1.5rem; }

    /* ── header ── */
    header {
      background: #1C1917;
      padding: 2rem 0 2.5rem;
      color: #FAFAF8;
    }
    header .inner { display: flex; align-items: flex-end; gap: 1.5rem; }
    header h1 { font-size: 1.85rem; font-weight: 700; letter-spacing: -0.4px; line-height: 1.1; }
    header p  { font-size: .875rem; color: #A8A29E; margin-top: .35rem; }
    .badge {
      display: inline-block;
      background: #292524;
      color: #D97706;
      font-size: .7rem;
      font-weight: 600;
      letter-spacing: .07em;
      text-transform: uppercase;
      padding: .25rem .65rem;
      border-radius: 99px;
      border: 1px solid #3F3833;
      margin-bottom: .5rem;
    }

    /* ── stats strip ── */
    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
      gap: .875rem;
      margin-top: -1.25rem;
      margin-bottom: 1.75rem;
    }
    .stat-card {
      background: white;
      border-radius: .75rem;
      padding: 1.1rem 1.35rem 1rem;
      border: 1px solid #E7E5E4;
    }
    .stat-label { font-size: .7rem; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; color: #A8A29E; }
    .stat-val   { font-size: 1.75rem; font-weight: 700; margin-top: .2rem; color: #1C1917; line-height: 1.1; }
    .stat-sub   { font-size: .75rem; color: #A8A29E; margin-top: .2rem; }
    .stat-val span { font-size: 1rem; font-weight: 400; color: #78716C; }

    /* ── card ── */
    .card {
      background: white;
      border-radius: .75rem;
      padding: 1.5rem;
      border: 1px solid #E7E5E4;
      margin-bottom: 1.25rem;
    }
    .card-title {
      font-size: .7rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .07em;
      color: #A8A29E;
      margin-bottom: 1.1rem;
    }

    /* ── chart ── */
    #chart { width: 100%; height: 420px; }

    /* ── estimators grid ── */
    .est-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.25rem;
      margin-bottom: 1.25rem;
    }
    @media (max-width: 620px) { .est-grid { grid-template-columns: 1fr; } }

    /* ── form ── */
    label {
      display: block;
      font-size: .8rem;
      font-weight: 500;
      color: #57534E;
      margin-bottom: .4rem;
    }
    input[type="date"], input[type="number"] {
      width: 100%;
      padding: .55rem .85rem;
      border: 1.5px solid #E7E5E4;
      border-radius: .5rem;
      font-size: .9rem;
      font-family: inherit;
      color: #1C1917;
      background: white;
      transition: border-color .15s;
      margin-bottom: .85rem;
    }
    input:focus { outline: none; border-color: #D97706; box-shadow: 0 0 0 3px rgba(217,119,6,.12); }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: .4rem;
      background: #1C1917;
      color: #FAFAF8;
      border: none;
      padding: .58rem 1.25rem;
      border-radius: .5rem;
      font-size: .875rem;
      font-weight: 500;
      cursor: pointer;
      font-family: inherit;
      transition: background .15s;
    }
    .btn:hover { background: #292524; }

    .result {
      margin-top: .9rem;
      padding: .9rem 1.1rem;
      background: #FFFBEB;
      border: 1px solid #FDE68A;
      border-radius: .5rem;
      display: none;
    }
    .result-val { font-size: 1.3rem; font-weight: 700; color: #92400E; }
    .result-sub { font-size: .78rem; color: #B45309; margin-top: .15rem; }

    .warn {
      font-size: .78rem;
      color: #DC2626;
      margin-top: .4rem;
      display: none;
    }

    /* ── table ── */
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: .85rem; }
    th { text-align: left; font-size: .7rem; font-weight: 600; text-transform: uppercase;
         letter-spacing: .06em; color: #A8A29E; padding: .5rem .75rem; border-bottom: 1px solid #E7E5E4; }
    td { padding: .55rem .75rem; border-bottom: 1px solid #F5F5F4; color: #1C1917; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #FAFAFA; }
    .diff-pos { color: #16A34A; font-weight: 500; }
    .diff-neg { color: #DC2626; font-weight: 500; }

    /* ── legend ── */
    .legend { display: flex; flex-wrap: wrap; gap: .75rem 1.25rem; margin-bottom: 1rem; }
    .leg-item { display: flex; align-items: center; gap: .4rem; font-size: .78rem; color: #78716C; }
    .leg-dot  { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
    .leg-line { width: 20px; height: 2.5px; flex-shrink: 0; }
    .leg-dash { width: 20px; height: 0; border-top: 2.5px dashed; flex-shrink: 0; }

    footer { text-align: center; padding: 2rem 1rem; color: #A8A29E; font-size: .78rem; }
    footer a { color: #D97706; text-decoration: none; }
    .param-strip { margin-top: .4rem; font-size: .75rem; }
  </style>
</head>
<body>

<header>
  <div class="container">
    <div class="inner">
      <div>
        <div class="badge">Growth Tracker</div>
        <h1>Inu Weight Growth</h1>
        <p>Gompertz growth model &middot; last updated __TODAY__</p>
      </div>
    </div>
  </div>
</header>

<main class="container" style="padding-top:2rem; padding-bottom:1rem;">

  <div class="stats" id="stats-strip"></div>

  <div class="card">
    <div class="card-title">Growth Curve</div>
    <div class="legend">
      <span class="leg-item"><span class="leg-dot" style="background:#1D4ED8;"></span> Measurements</span>
      <span class="leg-item"><span class="leg-line" style="background:#D97706;"></span> Model fit (past)</span>
      <span class="leg-item"><span class="leg-dash" style="border-color:#D97706;"></span> Forecast</span>
      <span class="leg-item"><span class="leg-line" style="background:rgba(217,119,6,.25);height:10px;border-radius:2px;"></span> 95 % CI</span>
    </div>
    <div id="chart"></div>
  </div>

  <div class="est-grid">
    <div class="card" style="margin-bottom:0">
      <div class="card-title">Date &rarr; Weight</div>
      <label for="inp-date">Select a date</label>
      <input type="date" id="inp-date" />
      <button class="btn" onclick="estimateWeight()">Estimate weight</button>
      <p class="warn" id="warn-d"></p>
      <div class="result" id="res-w">
        <div class="result-val" id="res-w-val"></div>
        <div class="result-sub" id="res-w-sub"></div>
      </div>
    </div>

    <div class="card" style="margin-bottom:0">
      <div class="card-title">Weight &rarr; Date</div>
      <label for="inp-weight">Target weight (kg)</label>
      <input type="number" id="inp-weight" placeholder="e.g. 22" step="0.1" min="0.1" />
      <button class="btn" onclick="estimateDate()">Estimate date</button>
      <p class="warn" id="warn-w"></p>
      <div class="result" id="res-d">
        <div class="result-val" id="res-d-val"></div>
        <div class="result-sub" id="res-d-sub"></div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Measurement log</div>
    <div class="table-wrap">
      <table id="data-table">
        <thead>
          <tr>
            <th>Date</th>
            <th>Weight (kg)</th>
            <th>Model (kg)</th>
            <th>Residual</th>
            <th>Week gain</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

</main>

<footer>
  <div>Fitted with a <strong>Gompertz growth model</strong> &middot; built with Python &amp; Plotly.js</div>
  <div class="param-strip" id="params"></div>
</footer>

<script>
const D = __PAYLOAD__;

// ── helpers ──────────────────────────────────────────────────────────────────
function daysFromRef(isoStr) {
  const ref = new Date(D.ref_date + 'T00:00:00');
  const d   = new Date(isoStr + 'T00:00:00');
  return (d - ref) / 86400000;
}
function gompertz(days) {
  return D.L * Math.exp(-Math.exp(-D.k * (days - D.t_mid)));
}
function invertGompertz(w) {
  // t = t_mid - ln(-ln(w / L)) / k
  if (w <= 0 || w >= D.L) return null;
  return D.t_mid - Math.log(-Math.log(w / D.L)) / D.k;
}
function daysToISO(days) {
  const ref = new Date(D.ref_date + 'T00:00:00');
  return new Date(ref.getTime() + days * 86400000).toISOString().split('T')[0];
}
function fmtDate(iso) {
  const [y, m, dd] = iso.split('-');
  return `${dd} ${['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][+m-1]} ${y}`;
}

// Interpolate the precomputed CI band at a given days value
function ciAt(days) {
  const idx = D.curve_x.findIndex((_, i) =>
    daysFromRef(D.curve_x[i]) >= days
  );
  if (idx < 0) return { lo: D.ci_lo[D.ci_lo.length-1], hi: D.ci_hi[D.ci_hi.length-1] };
  return { lo: D.ci_lo[idx], hi: D.ci_hi[idx] };
}

// ── stats strip ───────────────────────────────────────────────────────────────
const curW   = D.data_y[D.data_y.length - 1];
const curD   = D.data_x[D.data_x.length - 1];
const pct    = (curW / D.L * 100).toFixed(1);
const infl   = fmtDate(daysToISO(D.t_mid));

document.getElementById('stats-strip').innerHTML = `
  <div class="stat-card">
    <div class="stat-label">Latest weight</div>
    <div class="stat-val">${curW.toFixed(2)} <span>kg</span></div>
    <div class="stat-sub">${fmtDate(curD)}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Est. adult weight</div>
    <div class="stat-val">${D.L.toFixed(1)} <span>kg</span></div>
    <div class="stat-sub">${pct} % reached</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Fastest growth</div>
    <div class="stat-val" style="font-size:1.2rem">${infl}</div>
    <div class="stat-sub">inflection point</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Model fit (R²)</div>
    <div class="stat-val">${D.r2.toFixed(4)}</div>
    <div class="stat-sub">RMSE ${D.rmse.toFixed(3)} kg</div>
  </div>
`;

// ── chart ─────────────────────────────────────────────────────────────────────
const si   = D.split_idx;
const pX   = D.curve_x.slice(0, si + 1);
const pY   = D.curve_y.slice(0, si + 1);
const fX   = D.curve_x.slice(si);
const fY   = D.curve_y.slice(si);
const ciX  = [...D.curve_x, ...[...D.curve_x].reverse()];
const ciY  = [...D.ci_hi,   ...[...D.ci_lo].reverse()];

const traces = [
  // CI band
  { x: ciX, y: ciY, fill: 'toself', fillcolor: 'rgba(217,119,6,.12)',
    line: { color: 'transparent' }, hoverinfo: 'skip', showlegend: false, type: 'scatter' },
  // Past fit
  { x: pX, y: pY, mode: 'lines', line: { color: '#D97706', width: 2.5 },
    name: 'Model fit', hovertemplate: '%{x}: <b>%{y:.2f} kg</b><extra></extra>' },
  // Future forecast
  { x: fX, y: fY, mode: 'lines', line: { color: '#D97706', width: 2.5, dash: 'dash' },
    name: 'Forecast', hovertemplate: '%{x}: <b>%{y:.2f} kg</b><extra></extra>' },
  // Measurements
  { x: D.data_x, y: D.data_y, mode: 'markers',
    marker: { color: '#1D4ED8', size: 8, symbol: 'circle', line: { color: 'white', width: 1.5 } },
    name: 'Measurements', hovertemplate: '%{x}: <b>%{y:.2f} kg</b><extra></extra>' },
];

Plotly.newPlot('chart', traces, {
  paper_bgcolor: 'white', plot_bgcolor: '#FAFAFA',
  margin: { t: 10, r: 20, b: 48, l: 55 },
  xaxis: {
    title: { text: 'Date', font: { size: 11 } },
    showgrid: true, gridcolor: '#F0EEE8', linecolor: '#E7E5E4', tickfont: { size: 11 },
  },
  yaxis: {
    title: { text: 'Weight (kg)', font: { size: 11 } },
    showgrid: true, gridcolor: '#F0EEE8', linecolor: '#E7E5E4',
    tickfont: { size: 11 }, rangemode: 'tozero',
  },
  showlegend: false,
  hovermode: 'x unified',
  shapes: [
    {
      type: 'line', x0: D.today_iso, x1: D.today_iso, y0: 0, y1: 1,
      xref: 'x', yref: 'paper', line: { color: '#10B981', width: 1.5, dash: 'dot' },
    },
    {
      type: 'line', x0: D.curve_x[0], x1: D.curve_x[D.curve_x.length-1], y0: D.L, y1: D.L,
      xref: 'x', yref: 'y', line: { color: '#D1D5DB', width: 1, dash: 'dot' },
    },
  ],
  annotations: [
    {
      x: D.today_iso, y: 0.97, xref: 'x', yref: 'paper',
      text: 'Today', showarrow: false, font: { size: 11, color: '#10B981' },
      xanchor: 'left', yanchor: 'top', xshift: 5,
    },
    {
      x: D.curve_x[D.curve_x.length-1], y: D.L, xref: 'x', yref: 'y',
      text: `~${D.L.toFixed(1)} kg adult`, showarrow: false,
      font: { size: 10, color: '#9CA3AF' }, xanchor: 'right', yanchor: 'bottom',
    },
  ],
}, { responsive: true, displayModeBar: true,
     modeBarButtonsToRemove: ['lasso2d', 'select2d'], displaylogo: false });

// ── estimators ────────────────────────────────────────────────────────────────
function estimateWeight() {
  const v = document.getElementById('inp-date').value;
  const warnEl = document.getElementById('warn-d');
  const resEl  = document.getElementById('res-w');
  warnEl.style.display = 'none';
  if (!v) { warnEl.textContent = 'Please select a date.'; warnEl.style.display = 'block'; return; }
  const days = daysFromRef(v);
  const w    = gompertz(days);
  const ci   = ciAt(days);
  document.getElementById('res-w-val').textContent = `${w.toFixed(2)} kg`;
  const isPast = days <= D.today_days;
  document.getElementById('res-w-sub').textContent = isPast
    ? `Historical model estimate for ${fmtDate(v)}`
    : `Forecast for ${fmtDate(v)} · 95 % CI: ${ci.lo.toFixed(1)} – ${ci.hi.toFixed(1)} kg`;
  resEl.style.display = 'block';
}

function estimateDate() {
  const wVal   = parseFloat(document.getElementById('inp-weight').value);
  const warnEl = document.getElementById('warn-w');
  const resEl  = document.getElementById('res-d');
  warnEl.style.display = 'none';
  resEl.style.display  = 'none';
  if (isNaN(wVal) || wVal <= 0) {
    warnEl.textContent = 'Please enter a positive weight.'; warnEl.style.display = 'block'; return;
  }
  if (wVal >= D.L * 0.99) {
    warnEl.textContent = `Weight must be below the estimated adult weight (${D.L.toFixed(1)} kg).`;
    warnEl.style.display = 'block'; return;
  }
  const days = invertGompertz(wVal);
  const iso  = daysToISO(days);
  const isPast = days <= D.today_days;
  document.getElementById('res-d-val').textContent = fmtDate(iso);
  document.getElementById('res-d-sub').textContent = isPast
    ? `${wVal} kg was reached around ${fmtDate(iso)}`
    : `${wVal} kg is expected around ${fmtDate(iso)}`;
  resEl.style.display = 'block';
}

document.getElementById('inp-date').addEventListener('keydown', e => { if (e.key === 'Enter') estimateWeight(); });
document.getElementById('inp-weight').addEventListener('keydown', e => { if (e.key === 'Enter') estimateDate(); });

// Pre-fill today
document.getElementById('inp-date').value = D.today_iso;
estimateWeight();

// ── measurement log table ─────────────────────────────────────────────────────
const tbody = document.querySelector('#data-table tbody');
D.data_x.forEach((dateStr, i) => {
  const actual  = D.data_y[i];
  const model   = gompertz(daysFromRef(dateStr));
  const resid   = (actual - model).toFixed(3);
  const prevW   = i > 0 ? D.data_y[i-1] : null;
  const prevD   = i > 0 ? daysFromRef(D.data_x[i-1]) : null;
  const days_el = i > 0 ? daysFromRef(dateStr) - prevD : null;
  const gain    = prevW !== null ? ((actual - prevW) / days_el * 7).toFixed(2) : '—';
  const diffCls = +resid >= 0 ? 'diff-pos' : 'diff-neg';
  tbody.innerHTML += `
    <tr>
      <td>${fmtDate(dateStr)}</td>
      <td>${actual.toFixed(2)}</td>
      <td>${model.toFixed(2)}</td>
      <td class="${diffCls}">${resid >= 0 ? '+' : ''}${resid}</td>
      <td>${gain !== '—' ? gain + ' kg/wk' : '—'}</td>
    </tr>
  `;
});

// ── footer params ─────────────────────────────────────────────────────────────
document.getElementById('params').innerHTML =
  `Model: L = ${D.L.toFixed(2)} kg &nbsp;|&nbsp; k = ${D.k.toFixed(5)} &nbsp;|&nbsp; ` +
  `t<sub>0</sub> = ${D.t_mid.toFixed(1)} days &nbsp;|&nbsp; R² = ${D.r2.toFixed(4)}`;
</script>
</body>
</html>
"""

# Substitute today date and payload
HTML = HTML.replace("__TODAY__", today.strftime("%B %d, %Y"))
HTML = HTML.replace("__PAYLOAD__", payload_json)

# ── Write output ───────────────────────────────────────────────────────────────
os.makedirs("docs", exist_ok=True)
with open("docs/index.html", "w") as fh:
    fh.write(HTML)

print(f"\nGenerated docs/index.html")
print(f"To host on GitHub Pages: push to GitHub and enable Pages from the docs/ folder.")

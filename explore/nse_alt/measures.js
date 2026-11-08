// JS port of measures.py (single-vector form) for the artefact's synthetic lab.
// Checked against the Python implementation by check_measures_js.py.
const TINY = 1e-12;
const F32_EPS = 1.1920929e-7;

function sum(v) { let s = 0; for (const x of v) s += x; return s; }
function sortedAsc(v) { return Float64Array.from(v).sort(); }
function median(v) {
  const s = sortedAsc(v), n = s.length;
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

function nse(v) {
  const n = v.length, s = sum(v);
  if (!(Math.abs(s) >= 1.1754944e-38) || !isFinite(s)) return 0;
  let h = 0;
  for (const x of v) { const p = x / s; if (p >= F32_EPS) h -= p * Math.log2(p); }
  return Math.min(1, Math.max(0, h / Math.log2(n)));
}
function gini(v) {
  const s = sortedAsc(v), n = s.length, tot = sum(s);
  if (tot <= TINY) return 0;
  let acc = 0;
  for (let i = 0; i < n; i++) acc += (i + 1) * s[i];
  return 2 * acc / (n * tot) - (n + 1) / n;
}
function hoyer(v) {
  const n = v.length;
  let l1 = 0, l2 = 0;
  for (const x of v) { l1 += Math.abs(x); l2 += x * x; }
  l2 = Math.sqrt(l2);
  return l2 > TINY ? (Math.sqrt(n) - l1 / l2) / (Math.sqrt(n) - 1) : 0;
}
function renyi2(v) {
  const s = Math.max(sum(v), TINY);
  let q = 0;
  for (const x of v) { const p = x / s; q += p * p; }
  return Math.log(1 / Math.max(q, TINY)) / Math.log(v.length);
}
function moments(v) {
  const n = v.length, m = sum(v) / n;
  let m2 = 0, m4 = 0;
  for (const x of v) { const d = x - m; m2 += d * d; m4 += d ** 4; }
  return [m, m2 / n, m4 / n];
}
function kurtosis(v) { const [, var_, m4] = moments(v); return var_ > TINY ? m4 / var_ ** 2 - 3 : 0; }
function cv(v) { const [m, var_] = moments(v); return Math.sqrt(var_) / Math.max(m, TINY); }
function peakToMean(v) { return Math.max(...v) / Math.max(sum(v) / v.length, TINY); }
function top10Mass(v) {
  const k = Math.max(1, Math.round(v.length * 0.1));
  const s = sortedAsc(v);
  let top = 0;
  for (let i = s.length - k; i < s.length; i++) top += s[i];
  return top / Math.max(sum(v), TINY);
}
function margin(v) {
  const s = sortedAsc(v), a = s[s.length - 1], b = s[s.length - 2];
  return (a - b) / Math.max(a, TINY);
}
function robustZ(v) {
  const med = median(v);
  const mad = median(Array.from(v, (x) => Math.abs(x - med)));
  return (Math.max(...v) - med) / Math.max(1.4826 * mad, TINY);
}
function nseMinsub(v) { const mn = Math.min(...v); return nse(Array.from(v, (x) => x - mn)); }
function nseMedsub(v) { const md = median(v); return nse(Array.from(v, (x) => Math.max(x - md, 0))); }
function mean(v) { return sum(v) / v.length; }
function max(v) { return Math.max(...v); }

const MEASURE_FNS = {
  nse, nse_minsub: nseMinsub, nse_medsub: nseMedsub, renyi2, gini, hoyer, kurtosis,
  peak_to_mean: peakToMean, cv, top10_mass: top10Mass, margin, robust_z: robustZ, mean, max,
};

if (typeof module !== "undefined") module.exports = { MEASURE_FNS };

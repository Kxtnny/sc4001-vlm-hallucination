// Result tables, built directly from results/report/numbers.json (no hand-copied numbers).
const { Tbl, Caption } = require("./lib");

const pct = (x, d = 1) => (x === undefined || x === null ? "-" : (100 * x).toFixed(d));
const f3 = (x) => (x === undefined || x === null ? "-" : x.toFixed(3));
const pval = (p) => (p === undefined || p === null ? "-" : p < 0.001 ? "<0.001" : p.toFixed(3));
const NAMES = { baseline: "Baseline", cautious: "Cautious prompt", intervention: "Attention boost", pai: "PAI-style boost" };

function popeTable(N, caption) {
  const rows = [["Split", "Method", "Acc", "Prec", "Rec", "F1", "Yes %", "McNemar p"]];
  for (const split of ["random", "popular", "adversarial"]) {
    const s = (N.pope || {})[split] || {};
    for (const tag of ["baseline", "cautious", "intervention", "pai"]) {
      const m = s[tag];
      if (!m) continue;
      rows.push([tag === "baseline" ? split : "", NAMES[tag], pct(m.accuracy), pct(m.precision), pct(m.recall),
        pct(m.f1), pct(m.yes_ratio), pval(m.mcnemar_p)]);
    }
  }
  return [Tbl(rows, [1250, 1700, 900, 900, 900, 900, 900, 1200]), Caption(caption)];
}

function chairTable(N, caption) {
  const rows = [["Method", "CHAIR-s", "CHAIR-i", "Recall", "Words", "Objects/caption", "Truncated %"]];
  for (const tag of ["baseline", "intervention", "pai"]) {
    const m = (N.chair || {})[tag];
    if (!m) continue;
    rows.push([NAMES[tag], pct(m.chair_s), pct(m.chair_i), pct(m.recall), m.avg_len_words.toFixed(1),
      m.avg_mentions.toFixed(2), pct(m.frac_hit_max_tokens)]);
  }
  const d = (N.chair || {}).intervention_minus_baseline;
  if (d) {
    const c = (k) => `${(100 * d[k].diff).toFixed(1)} [${(100 * d[k].ci[0]).toFixed(1)}, ${(100 * d[k].ci[1]).toFixed(1)}]`;
    rows.push(["Boost - baseline (95% CI)", c("chair_s"), c("chair_i"), c("recall"), "", "", ""]);
  }
  const dp = (N.chair || {}).pai_minus_baseline;
  if (dp) {
    const c = (k) => `${(100 * dp[k].diff).toFixed(1)} [${(100 * dp[k].ci[0]).toFixed(1)}, ${(100 * dp[k].ci[1]).toFixed(1)}]`;
    rows.push(["PAI-style - baseline (95% CI)", c("chair_s"), c("chair_i"), c("recall"), "", "", ""]);
  }
  const lm = (N.chair || {}).length_matched;
  if (lm) {
    const e = lm.intervention_minus_baseline;
    const c = (k) => `${(100 * e[k].diff).toFixed(1)} [${(100 * e[k].ci[0]).toFixed(1)}, ${(100 * e[k].ci[1]).toFixed(1)}]`;
    rows.push(["Matched length: baseline", pct(lm.baseline.chair_s), pct(lm.baseline.chair_i), pct(lm.baseline.recall), "", lm.baseline.avg_mentions.toFixed(2), ""]);
    rows.push(["Matched length: boost", pct(lm.intervention.chair_s), pct(lm.intervention.chair_i), pct(lm.intervention.recall), "", lm.intervention.avg_mentions.toFixed(2), ""]);
    rows.push(["Matched length: difference", c("chair_s"), c("chair_i"), c("recall"), "", "", ""]);
  }
  return [Tbl(rows, [2500, 1450, 1450, 1450, 800, 1200, 1000], { boldRows: [] }), Caption(caption)];
}

function sweepTable(N, caption, maxRows = 99, key = null) {
  const sw = key ? (N[key] || {}).table : N.sweep;
  if (!sw) return [];
  const sel = (key ? (N[key] || {}).selected : N.selected) || {};
  const rows = [["alpha", "Layers", "POPE acc", "POPE F1", "Yes %", "CHAIR-s", "CHAIR-i", "Recall", "Words"]];
  for (const r of sw.rows.slice(0, maxRows)) {
    rows.push([r.alpha === 0 ? "0 (base)" : String(r.alpha), r.layers + (r.tag === sel.tag ? " *" : ""),
      pct(r.pope_acc), pct(r.pope_f1), pct(r.pope_yes_ratio), pct(r.chair_s), pct(r.chair_i), pct(r.recall),
      r.avg_len_words.toFixed(0)]);
  }
  return [Tbl(rows, [1000, 1100, 1050, 1050, 950, 1050, 1050, 1050, 900], { size: 16 }), Caption(caption)];
}

function detectorTable(N, caption) {
  const A = (N.attention || {});
  const names = {
    position: "Position in caption", logprob: "Token log-probability", attention: "Image attention (band)",
    "position+logprob": "Position + log-prob", "position+logprob+attention": "Position + log-prob + attention",
  };
  const rows = [["Features (logistic regression, 5-fold CV by image)", "AUROC dev", "AUROC test"]];
  for (const k of Object.keys(names)) {
    rows.push([names[k], f3(A.dev && A.dev.detectors_cv_auroc[k]), f3(A.chair_test && A.chair_test.detectors_cv_auroc[k])]);
  }
  return [Tbl(rows, [5000, 1600, 1600]), Caption(caption)];
}

function abstentionTable(N, caption) {
  const ab = N.abstention;
  if (!ab) return [];
  const rows = [["Split", "Acc (all)", "Score used to rank questions", "AUROC (wrong)", "Area acc-cov", "Coverage @ dev thr.", "Acc @ dev thr."]];
  for (const split of ["random", "popular", "adversarial"]) {
    const r = ab.pope[split];
    if (!r) continue;
    const ex = r.exploratory_high_attention;
    const sigs = [["image attention", `low image attention, layers ${ab.band}`], ["answer confidence", "answer confidence"]];
    sigs.forEach(([sig, label], i) => {
      const e = r[sig];
      rows.push([i === 0 ? split : "", i === 0 ? pct(r.accuracy_full) : "", label, f3(e.auroc_wrong),
        pct(e.area_under_acc_cov), pct(e.coverage_at_dev_threshold), pct(e.accuracy_at_dev_threshold)]);
    });
    if (ex) rows.push(["", "", `*high* attention, layer ${ex.layer} (exploratory)`, f3(ex.auroc_wrong), pct(ex.area_under_acc_cov), "-", "-"]);
    rows.push(["", "", "random order", f3(r.random.auroc_wrong), pct(r.random.area_under_acc_cov), "-", "-"]);
  }
  return [Tbl(rows, [1100, 850, 3250, 1100, 1050, 1200, 1050], { size: 16 }), Caption(caption)];
}

module.exports = { popeTable, chairTable, sweepTable, detectorTable, abstentionTable, pct, f3, pval };

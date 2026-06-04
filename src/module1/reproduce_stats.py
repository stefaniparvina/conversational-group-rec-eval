"""
reproduce_stats.py
==================
Produces descriptive statistics from `data/results/results.csv` (and the
per-agent personality data in `data/full_dataset/`).

Output:
  1. Prints all sections to stdout (so `> stats_output.txt` still works).
  2. Writes a multi-sheet workbook to `data/results/stats_output.xlsx`,
     with one sheet per section for easy copy/paste into the thesis.
"""

import sys
import json
import glob
from pathlib import Path

import pandas as pd
import numpy as np

# Force UTF-8 stdout so special characters write correctly when redirected on Windows.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RESULTS_CSV  = Path("data/results/results.csv")
FULL_DATASET = Path("data/full_dataset")
OUT_XLSX     = Path("data/results/stats_output.xlsx")

df = pd.read_csv(RESULTS_CSV)
dc = df[df.consensus_reached]
pd.set_option("display.max_columns", None)

# Collected DataFrames written to the Excel workbook at the end.
frames = {}

# ── Section 2 — Consensus ─────────────────────────────────────────────────────
print("=== Section 2: Consensus ===")
overall_cons = df.consensus_reached.mean()
by_cfg_cons  = df.groupby("group_config").consensus_reached.mean()
print("Consensus rate:", overall_cons)
print(by_cfg_cons)

consensus_df = by_cfg_cons.to_frame("consensus_rate").round(4)
consensus_df.loc["OVERALL"] = round(overall_cons, 4)
consensus_df.index.name = "group_config"
frames["Consensus"] = consensus_df

# ── Section 3 — Outcome quality ───────────────────────────────────────────────
print("\n=== Section 3: Outcome quality ===")
metrics_cfg  = ["gss", "min_sat", "sat_variance", "stability_rate", "maj_min_gap"]
metrics_size = ["gss", "min_sat", "sat_variance", "n_strategies_matched"]

oq_by_config = dc.groupby("group_config")[metrics_cfg].mean().round(4)
oq_by_size   = dc.groupby("n_agents")[metrics_size].mean().round(4)
print(oq_by_config)
print(oq_by_size)

frames["Outcome Quality by Config"] = oq_by_config
frames["Outcome Quality by Size"]   = oq_by_size

# ── Section 4 — ISS distribution ──────────────────────────────────────────────
print("\n=== Section 4: ISS distribution ===")
iss_cols = [c for c in df.columns if c.startswith("ISS_")]
m = (df.melt(id_vars=["group_id", "group_config", "consensus_reached"],
             value_vars=iss_cols, var_name="agent", value_name="iss")
       .dropna()
       .query("consensus_reached"))

iss_describe = m["iss"].describe().round(4)
iss_by_cfg = (m.groupby("group_config")["iss"]
               .agg(mean="mean", median="median", std="std",
                    pct_lt_02=lambda x: (x < 0.20).mean())
               .round(4))
print(iss_describe)
print(iss_by_cfg)

frames["ISS Overall"]   = iss_describe.to_frame(name="value")
frames["ISS by Config"] = iss_by_cfg

# ── Section 5 — Strategy comparison ───────────────────────────────────────────
print("\n=== Section 5: Strategy comparison ===")
strats = ["ADD", "LMS", "MPL", "MAJ", "APP", "FAI"]
strat_match = dc.groupby("group_config")[[f"matches_{s}" for s in strats]].mean().round(4)

no_strat_overall = (dc["n_strategies_matched"] == 0).mean()
no_strat_by_cfg  = dc.groupby("group_config").apply(
    lambda g: (g["n_strategies_matched"] == 0).mean(), include_groups=False
).round(4)

welfare_by_cfg = dc.groupby("group_config")["conversation_vs_best_strategy_gss"].mean().round(4)

print(strat_match)
print("No-strategy rate:", no_strat_overall)
print(no_strat_by_cfg)
print(welfare_by_cfg)

frames["Strategy Match Rates"] = strat_match
no_strat_df = no_strat_by_cfg.to_frame("no_strategy_match_rate")
no_strat_df.loc["OVERALL"] = round(no_strat_overall, 4)
frames["No-Strategy Match Rate"] = no_strat_df
frames["Welfare Gap by Config"] = welfare_by_cfg.to_frame("conv_vs_best_strategy_gss")

# ── Section 6 — Personality correlations ──────────────────────────────────────
print("\n=== Section 6: Personality correlations ===")
rows = []
for fp in sorted(FULL_DATASET.glob("group_simulation_*.json")):
    d = json.load(open(fp, encoding="utf-8"))
    f = d.get("final_rec", "")
    if f in ("NO CONSENSUS REACHED", "No preference yet"):
        continue
    for ag in d["agents"]:
        h = ag.get("history", {})
        if not h or f not in h:
            continue
        iss = h[f] / max(h.values())
        p = ag.get("personality", {})
        rows.append({"config": d["group_config"], "iss": iss,
                     "tone": ag.get("tone", ""), **p})

agdf = pd.DataFrame(rows)
trait_results = []
for t in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
    r = agdf[[t, "iss"]].corr().iloc[0, 1]
    print(t, r)
    trait_results.append({
        "trait": t,
        "pearson_r": round(r, 4),
        "abs_r": round(abs(r), 4),
        "within_negligible_bound": bool(abs(r) < 0.05),
    })

personality_df = pd.DataFrame(trait_results).set_index("trait")
frames["Personality Correlations"] = personality_df

# ── Section 7 — Report-traceability additions ────────────────────────────────
# Every figure quoted in the thesis but not previously written to a result file
# is computed here so the report is fully traceable to stats_output.xlsx.
print("\n=== Section 7: Report-traceability additions ===")

# 7a. Per-config standard deviations for Table I (population SD, ddof=0, to
#     match the values rounded to 2 dp in the report).
oq_sd_by_config = dc.groupby("group_config")[metrics_cfg].std(ddof=0).round(4)
frames["Outcome Quality SD by Config"] = oq_sd_by_config
print(oq_sd_by_config)

# 7b. Mean number of rounds to consensus, per config and overall.
rounds_df = dc.groupby("group_config")["turn_counter"].mean().round(2).to_frame("mean_rounds")
rounds_df.loc["OVERALL"] = round(float(dc["turn_counter"].mean()), 2)
frames["Rounds to Consensus"] = rounds_df
print(rounds_df)

# 7c. Single-round fraction (consensus groups and all groups), so the report's
#     "single round" figure is traceable.
single_round_df = pd.DataFrame(
    {"value": [round(float(dc["single_round_group"].mean()), 4),
               round(float(df["single_round_group"].mean()), 4)]},
    index=["single_round_frac_consensus", "single_round_frac_all_groups"])
frames["Single-Round Fraction"] = single_round_df
print(single_round_df)

# 7d. Agent-level ISS melt that ALSO carries n_agents, for ISS<0.20 by size and
#     by config x size (the existing `m` only carries group_config).
m_sz = (df.melt(id_vars=["group_id", "group_config", "n_agents", "consensus_reached"],
                value_vars=iss_cols, var_name="agent", value_name="iss")
          .dropna()
          .query("consensus_reached"))

# Overall ISS<0.20 share appended to the ISS Overall sheet.
frames["ISS Overall"].loc["pct_lt_0.20"] = round(float((m_sz["iss"] < 0.20).mean()), 4)

# ISS distribution by group size.
frames["ISS by Size"] = (m_sz.groupby("n_agents")["iss"]
    .agg(mean="mean", median="median", std="std",
         pct_lt_02=lambda x: (x < 0.20).mean())
    .round(4))
print(frames["ISS by Size"])

# ISS<0.20 by config x size (shows the n=2 divergent case the report rounds over).
frames["ISS by Config x Size"] = (m_sz.groupby(["group_config", "n_agents"])["iss"]
    .agg(mean="mean", pct_lt_02=lambda x: (x < 0.20).mean())
    .round(4))

# 7e. Overall strategy-match row appended to the per-config match-rate sheet
#     (this is the "Overall" row of Table II).
frames["Strategy Match Rates"].loc["OVERALL"] = (
    dc[[f"matches_{s}" for s in strats]].mean().round(4).values)
print(frames["Strategy Match Rates"])

# 7f. H3-subset threshold justification (median MinSat, 75th-percentile
#     MajMinGap, the two threshold shares, and the resulting subset size).
thr_min_sat = dc["min_sat"]
thr_maj_gap = dc["maj_min_gap"]
frames["H3 Threshold Justification"] = pd.DataFrame(
    {"value": [round(float(thr_min_sat.median()), 4),
               round(float(thr_maj_gap.quantile(0.75)), 4),
               round(float((thr_min_sat < 0.20).mean()), 4),
               round(float((thr_maj_gap > 0.30).mean()), 4),
               int(((dc["min_sat"] < 0.20) & (dc["maj_min_gap"] > 0.30)).sum())]},
    index=["median_MinSat", "MajMinGap_75th_pct", "frac_MinSat_lt_0.20",
           "frac_MajMinGap_gt_0.30", "n_H3_subset"])
print(frames["H3 Threshold Justification"])

# ── Excel output ──────────────────────────────────────────────────────────────
OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
try:
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31])
    print(f"\nExcel workbook saved -> {OUT_XLSX}")
except ImportError:
    print("\n[WARN] openpyxl not installed -- Excel output skipped.")
    print("       Run: pip install openpyxl")

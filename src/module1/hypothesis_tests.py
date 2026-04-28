#!/usr/bin/env python3
"""
hypothesis_tests.py — Formal Statistical Tests for the Thesis
==============================================================
Runs the pre-specified statistical tests for hypotheses H1, H2, H4 on the
output of evaluation_framework.py (results.csv, results.json) and reports
the H3 subset size (H3 itself is blocked on Module 2 output).

Tests performed
---------------
H1 — Conversations underperform formal aggregation
     Wilcoxon signed-rank test of welfare gap vs. zero (one-sided 'less'),
     overall and per configuration. Effect size: rank-biserial correlation.

H2 — Configuration determines fairness
     (a) Kruskal-Wallis on MinSat across the four configurations.
     (b) Pairwise Mann-Whitney U with Holm-Bonferroni correction.
     (c) Wilcoxon signed-rank on MajMinGap vs. zero (one-sided 'greater'),
         per configuration.

H3 — Structural metrics insufficient (status report only)
     Reports the size of the H3 subset (MinSat < 0.20 AND MajMinGap > 0.30).
     Full test requires Module 2 output.

H4 — Personality and tone do NOT predict ISS (null hypothesis)
     (a) Pearson r between each Big Five trait and ISS at agent level.
     (b) Kruskal-Wallis across tone categories.
     Note: a null hypothesis cannot be 'proven' — we report effect sizes and
     argue the null is supported when |r| < 0.05 and epsilon-squared < 0.04.
"""

import json
import sys
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
from scipy import stats


# ── EFFECT SIZE HELPERS ──────────────────────────────────────────────────────

def rank_biserial_wilcoxon(x):
    """Rank-biserial correlation for one-sample Wilcoxon (vs 0)."""
    x = np.asarray(x, dtype=float)
    x = x[x != 0]
    n = len(x)
    if n == 0:
        return float('nan')
    ranks = stats.rankdata(np.abs(x))
    w_plus  = ranks[x > 0].sum()
    w_minus = ranks[x < 0].sum()
    total = w_plus + w_minus
    return 0.0 if total == 0 else (w_plus - w_minus) / total


def rank_biserial_mwu(u, n1, n2):
    """Rank-biserial correlation for Mann-Whitney U."""
    if n1 == 0 or n2 == 0:
        return float('nan')
    return 1.0 - (2.0 * u) / (n1 * n2)


def epsilon_squared_kw(h, n):
    """Epsilon-squared effect size for Kruskal-Wallis."""
    return float('nan') if n <= 1 else h / (n - 1)


def interpret_r(r):
    a = abs(r)
    if a < 0.10: return "negligible"
    if a < 0.30: return "small"
    if a < 0.50: return "medium"
    return "large"


def holm_correction(pvals):
    """Holm-Bonferroni adjusted p-values (preserves original order)."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running_max = 0.0
    for i, idx in enumerate(order):
        adj_p = pvals[idx] * (n - i)
        running_max = max(running_max, adj_p)
        adj[idx] = min(running_max, 1.0)
    return adj


# ── FORMATTING ───────────────────────────────────────────────────────────────

CONFIGS = ["uniform", "divergent", "coalitional", "minority"]
HEAD = "=" * 78
SUB  = "-" * 78


def section(title):    print(f"\n{HEAD}\n  {title}\n{HEAD}")
def subsection(title): print(f"\n{SUB}\n  {title}\n{SUB}")
def fmt_p(p):          return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


# ── H1 ───────────────────────────────────────────────────────────────────────

def test_h1(df):
    section("H1 — Conversations underperform formal aggregation")
    print("Claim   : Welfare gap (conversation GSS - best-strategy GSS) is significantly")
    print("          below zero, overall and within each configuration.")
    print("Test    : Wilcoxon signed-rank against zero (one-sided alternative='less').")
    print("Effect  : Rank-biserial correlation (negative = gap is below zero).")
    print("Sample  : Consensus groups only.")

    consensus = df[df["consensus_reached"]].copy()

    subsection("Overall (all consensus groups)")
    gaps = consensus["conversation_vs_best_strategy_gss"].values
    n = len(gaps)
    stat, p = stats.wilcoxon(gaps, alternative="less")
    rb = rank_biserial_wilcoxon(gaps)
    print(f"  n            : {n}")
    print(f"  Mean gap     : {gaps.mean():+.4f}  ({gaps.mean()*100:+.2f} pp)")
    print(f"  Median gap   : {np.median(gaps):+.4f}")
    print(f"  Wilcoxon W   : {stat:.1f}")
    print(f"  p-value      : {fmt_p(p)}")
    print(f"  Rank-biserial: {rb:+.3f}  ({interpret_r(rb)})")

    subsection("By configuration")
    print(f"  {'config':<12} {'n':>5} {'mean':>10} {'median':>10} "
          f"{'W':>12} {'p':>10} {'r_rb':>7}")
    for cfg in CONFIGS:
        sub = consensus[consensus["group_config"] == cfg]
        gaps_c = sub["conversation_vs_best_strategy_gss"].values
        if len(gaps_c) == 0:
            continue
        stat, p = stats.wilcoxon(gaps_c, alternative="less")
        rb = rank_biserial_wilcoxon(gaps_c)
        print(f"  {cfg:<12} {len(gaps_c):>5} "
              f"{gaps_c.mean():+10.4f} {np.median(gaps_c):+10.4f} "
              f"{stat:>12.1f} {fmt_p(p):>10} {rb:>+7.3f}")

    print("\n  Decision: H1 supported wherever p < 0.05 and effect direction is negative.")


# ── H2 ───────────────────────────────────────────────────────────────────────

def test_h2(df):
    section("H2 — Configuration determines fairness")
    print("Claim   : MinSat differs across configurations; MajMinGap > 0 per config.")
    print("Tests   : (a) Kruskal-Wallis on MinSat across 4 configs.")
    print("          (b) Pairwise Mann-Whitney U on MinSat with Holm correction.")
    print("          (c) Wilcoxon signed-rank on MajMinGap > 0, per config.")
    print("Sample  : Consensus groups only.")

    consensus = df[df["consensus_reached"]].copy()

    # (a) Kruskal-Wallis on MinSat
    subsection("(a) Kruskal-Wallis on MinSat across 4 configurations")
    samples = [consensus[consensus["group_config"] == c]["min_sat"].values for c in CONFIGS]
    sizes   = [len(s) for s in samples]
    h, p = stats.kruskal(*samples)
    n_total = sum(sizes)
    eps2 = epsilon_squared_kw(h, n_total)
    for cfg, n_, s in zip(CONFIGS, sizes, samples):
        print(f"  {cfg:<12}  n = {n_:>5}   mean MinSat = {s.mean():.4f}   "
              f"median = {np.median(s):.4f}")
    print(f"\n  H statistic : {h:.3f}")
    print(f"  p-value     : {fmt_p(p)}")
    print(f"  epsilon²    : {eps2:.4f}  (small <0.04, medium <0.16, large ≥0.16)")

    # (b) Pairwise Mann-Whitney U
    subsection("(b) Pairwise Mann-Whitney U on MinSat (Holm-corrected)")
    pairs = list(combinations(range(4), 2))
    raw_p, rows = [], []
    for i, j in pairs:
        x, y = samples[i], samples[j]
        u, p_raw = stats.mannwhitneyu(x, y, alternative="two-sided")
        rb = rank_biserial_mwu(u, len(x), len(y))
        raw_p.append(p_raw)
        rows.append((CONFIGS[i], CONFIGS[j], len(x), len(y), u, p_raw, rb))
    adj_p = holm_correction(raw_p)
    print(f"  {'A':<12} {'B':<12} {'nA':>5} {'nB':>5} "
          f"{'p_raw':>10} {'p_holm':>10} {'r_rb':>7}")
    for (ci, cj, na, nb, u, pr, rb), pa in zip(rows, adj_p):
        print(f"  {ci:<12} {cj:<12} {na:>5} {nb:>5} "
              f"{fmt_p(pr):>10} {fmt_p(pa):>10} {rb:>+7.3f}")

    # (c) Wilcoxon on MajMinGap per config
    subsection("(c) Wilcoxon signed-rank on MajMinGap > 0, per config")
    print(f"  {'config':<12} {'n':>5} {'mean':>10} {'median':>10} "
          f"{'W':>12} {'p':>10} {'r_rb':>7}")
    for cfg in CONFIGS:
        gaps = consensus.loc[consensus["group_config"] == cfg, "maj_min_gap"].values
        nonzero = gaps[gaps != 0]
        if len(nonzero) == 0:
            print(f"  {cfg:<12} {0:>5}   no nonzero MajMinGap")
            continue
        stat, p = stats.wilcoxon(nonzero, alternative="greater")
        rb = rank_biserial_wilcoxon(nonzero)
        print(f"  {cfg:<12} {len(nonzero):>5} "
              f"{nonzero.mean():+10.4f} {np.median(nonzero):+10.4f} "
              f"{stat:>12.1f} {fmt_p(p):>10} {rb:>+7.3f}")


# ── H3 ───────────────────────────────────────────────────────────────────────

def test_h3(df):
    section("H3 — Structural metrics insufficient (status report)")
    print("Claim  : Process Quality Score (Module 2) adds variance not captured by")
    print("         structural metrics.")
    print("Test   : Spearman r between Process Quality Score and structural metrics on")
    print("         the H3 subset; supplemented by ISS ~ structural + LLM regression.")
    print("Status : BLOCKED on Module 2 run. Reporting H3 subset size only.")

    consensus = df[df["consensus_reached"]].copy()
    h3_mask = (consensus["min_sat"] < 0.20) & (consensus["maj_min_gap"] > 0.30)
    h3_sub = consensus[h3_mask]
    print(f"\n  H3 subset (MinSat < 0.20 AND MajMinGap > 0.30): {len(h3_sub)} groups")
    print(f"\n  By configuration:")
    for cfg in CONFIGS:
        n_cfg = (h3_sub["group_config"] == cfg).sum()
        n_total = (consensus["group_config"] == cfg).sum()
        share = 100 * n_cfg / n_total if n_total else 0
        print(f"    {cfg:<12} {n_cfg:>5} / {n_total:<5} ({share:.1f}%)")


# ── H4 ───────────────────────────────────────────────────────────────────────

def collect_agent_level(full_dataset_folder, results_json_path):
    """Long-format DataFrame: one row per consensus agent with personality + ISS."""
    iss_lookup = {}
    with open(results_json_path) as f:
        for r in json.load(f):
            iss_lookup[r["group_id"]] = r["iss_per_agent"]

    rows = []
    for jf in sorted(Path(full_dataset_folder).glob("group_simulation_*.json")):
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        gid = data["group_id"]
        if data["final_rec"] == "NO CONSENSUS REACHED":
            continue
        per_agent_iss = iss_lookup.get(gid, {})
        for a in data["agents"]:
            name = a["name"]
            if name not in per_agent_iss:
                continue
            p = a.get("personality", {})
            rows.append({
                "group_id":          gid,
                "name":              name,
                "openness":          p.get("openness"),
                "conscientiousness": p.get("conscientiousness"),
                "extraversion":      p.get("extraversion"),
                "agreeableness":     p.get("agreeableness"),
                "neuroticism":       p.get("neuroticism"),
                "tone":              a.get("tone"),
                "iss":               per_agent_iss[name],
            })
    return pd.DataFrame(rows)


def test_h4(full_dataset_folder, results_json_path):
    section("H4 — Personality and tone do NOT predict ISS (null hypothesis)")
    print("Claim   : Big Five traits and tone categories show no meaningful relationship")
    print("          with individual satisfaction.")
    print("Tests   : (a) Pearson r between each Big Five trait and ISS at agent level.")
    print("          (b) Kruskal-Wallis across tone categories.")
    print("Sample  : Consensus agents only.")
    print("Caveat  : A null hypothesis cannot be 'proven'. We treat the null as")
    print("          supported when |r| < 0.05 (and epsilon² < 0.04 for tone).")

    print(f"\n  Loading agent-level data from {full_dataset_folder} ...")
    agents = collect_agent_level(full_dataset_folder, results_json_path)
    print(f"  Loaded {len(agents)} consensus-agent observations.")

    subsection("(a) Pearson r — each Big Five trait vs. ISS")
    print(f"  {'trait':<20} {'n':>6} {'r':>9} {'p':>10}  interpretation")
    for trait in ["openness", "conscientiousness", "extraversion",
                  "agreeableness", "neuroticism"]:
        sub = agents.dropna(subset=[trait])
        if len(sub) < 3:
            continue
        r, p = stats.pearsonr(sub[trait].values, sub["iss"].values)
        print(f"  {trait:<20} {len(sub):>6} {r:>+9.4f} {fmt_p(p):>10}  {interpret_r(r)}")

    subsection("(b) Kruskal-Wallis — ISS across tone categories")
    by_tone = agents.dropna(subset=["tone"]).groupby("tone")["iss"].apply(list)
    samples = list(by_tone.values)
    sizes   = [len(s) for s in samples]
    n_tones = len(samples)
    if n_tones < 2:
        print("  Not enough tone categories for KW.")
    else:
        h, p = stats.kruskal(*samples)
        eps2 = epsilon_squared_kw(h, sum(sizes))
        means = [np.mean(s) for s in samples]
        print(f"  Number of tone categories : {n_tones}")
        print(f"  Total observations        : {sum(sizes)}")
        print(f"  Mean ISS by tone (range)  : [{min(means):.3f}, {max(means):.3f}]")
        print(f"  H statistic               : {h:.3f}")
        print(f"  p-value                   : {fmt_p(p)}")
        print(f"  epsilon²                  : {eps2:.4f}  ({interpret_r(eps2**0.5)})")

    print("\n  Decision: H4 retained if all |r| < 0.05 and epsilon² < 0.04.")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print("Usage: python hypothesis_tests.py <results_csv> <full_dataset_folder>")
        sys.exit(1)

    results_csv  = Path(sys.argv[1])
    full_dataset = Path(sys.argv[2])
    results_json = results_csv.parent / "results.json"

    if not results_csv.exists():
        sys.exit(f"results.csv not found: {results_csv}")
    if not full_dataset.is_dir():
        sys.exit(f"full_dataset folder not found: {full_dataset}")
    if not results_json.exists():
        sys.exit(f"results.json not found: {results_json}")

    df = pd.read_csv(results_csv)

    print(HEAD)
    print(f"  HYPOTHESIS TESTS — {len(df)} groups loaded")
    print(f"  results.csv  : {results_csv}")
    print(f"  full_dataset : {full_dataset}")
    print(HEAD)

    test_h1(df)
    test_h2(df)
    test_h3(df)
    test_h4(full_dataset, results_json)

    print(f"\n{HEAD}\n  All tests completed.\n{HEAD}\n")


if __name__ == "__main__":
    main()

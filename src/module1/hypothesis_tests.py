#!/usr/bin/env python3
"""
hypothesis_tests.py — Formal Statistical Tests for the Thesis
==============================================================

Runs the pre-specified statistical tests for hypotheses H1, H2, and H4 on
the output of `evaluation_framework.py` (`results.csv`),
and reports the H3 subset size (the full H3 regression is blocked on
Module 2 output).

Output
------
A single multi-sheet Excel workbook. Every test produces one or more
result tables; no terminal output, no text file.

Sheets produced
---------------
  H1 Distribution Diagnostic  Distribution shape of the welfare gap
                              (skewness, kurtosis, % zero/below/above,
                              min, max). Justifies the choice of
                              Wilcoxon over a t-test.
  H1 Welfare Gap              One-sample Wilcoxon (one-sided 'less') vs
                              zero, overall + per configuration. An
                              assumption-free sign test is reported
                              alongside as a robustness backup, and Holm-
                              Bonferroni is applied across the per-config
                              tests. Effect size: rank-biserial.
  H2a Kruskal-Wallis          Per-configuration MinSat means + omnibus
                              H statistic, p-value, epsilon-squared.
  H2a Pairwise MWU            Six pairwise Mann-Whitney U comparisons
                              on MinSat with Holm-Bonferroni correction.
  H2b MajMinGap               One-sample Wilcoxon (one-sided 'greater') on
                              MajMinGap, per configuration, Holm-corrected.
                              Non-zero gaps only: groups with a realized
                              majority-minority satisfaction difference.
                              Zero gaps are excluded because they provide
                              no signed evidence for the direction of the gap.
  H3 Subset                   H3 subset size by configuration (status only;
                              the full H3 regression requires Module 2
                              process metrics, which are not yet available).
  H4 Big Five                 Pearson r between each Big Five trait and
                              ISS at the agent level, each with a 95%
                              confidence interval. Equivalence decision
                              rule: H4 holds iff every CI lies entirely
                              within +/-0.05 (a negligible effect).
  H4 Tone (Supplementary)     Kruskal-Wallis across the 20 tone categories.
                              Tone was randomly assigned and is NOT part
                              of H4; this is reported only as a
                              manipulation check.

Usage
-----
    python src/module1/hypothesis_tests.py <results_csv> <full_dataset_folder> [output_xlsx]

If `output_xlsx` is omitted, defaults to
`<results_csv parent>/hypothesis_tests_output.xlsx`.

Requirements: scipy, pandas, numpy, openpyxl.
"""

import sys
import json
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
from scipy import stats


# ── EFFECT-SIZE HELPERS ──────────────────────────────────────────────────────

def rank_biserial_wilcoxon(x):
    """Rank-biserial correlation for one-sample Wilcoxon (vs 0)."""
    x = np.asarray(x, dtype=float)
    x = x[x != 0]
    if len(x) == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(x))
    w_plus  = ranks[x > 0].sum()
    w_minus = ranks[x < 0].sum()
    total = w_plus + w_minus
    return 0.0 if total == 0 else (w_plus - w_minus) / total


def rank_biserial_mwu(u, n1, n2):
    """Rank-biserial correlation for Mann-Whitney U.

    SciPy's mannwhitneyu returns U for the FIRST sample (config_A). With
    the convention r = 2U/(n1*n2) - 1, a POSITIVE r means config_A tends
    to rank HIGHER than config_B (negative means lower).
    """
    if n1 == 0 or n2 == 0:
        return float("nan")
    return (2.0 * u) / (n1 * n2) - 1.0


def epsilon_squared_kw(h, n):
    """Epsilon-squared effect size for Kruskal-Wallis."""
    return float("nan") if n <= 1 else h / (n - 1)


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


def sign_test(x, alternative):
    """Assumption-free sign test that the median differs from 0.

    Unlike the Wilcoxon signed-rank test it makes NO symmetry assumption,
    so it is a robust backup when the data is strongly skewed. Zeros are
    ignored.
      alternative='less'    -> evidence the median is below 0 (few positives)
      alternative='greater' -> evidence the median is above 0 (few negatives)
    Returns (n_positive, n_negative, p_value).
    """
    x = np.asarray(x, dtype=float)
    n_pos = int((x > 0).sum())
    n_neg = int((x < 0).sum())
    n = n_pos + n_neg
    if n == 0:
        return n_pos, n_neg, float("nan")
    k = n_pos if alternative == "less" else n_neg
    p = stats.binomtest(k, n, 0.5, alternative="less").pvalue
    return n_pos, n_neg, float(p)


def pearson_ci(r, n, conf=0.95):
    """Fisher z-transform confidence interval for a Pearson correlation r.

    Version-independent: does not rely on SciPy returning a CI object.
    Returns (low, high), or (nan, nan) if n is too small.
    """
    if n < 4 or abs(r) >= 1.0:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = stats.norm.ppf(1.0 - (1.0 - conf) / 2.0)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


# ── CONSTANTS / STATE ────────────────────────────────────────────────────────

CONFIGS = ["uniform", "divergent", "coalitional", "minority"]

# Each test function populates this dict with one or more DataFrames.
# main() writes them all to the Excel workbook at the end.
FRAMES: dict[str, pd.DataFrame] = {}


# ── H1 ───────────────────────────────────────────────────────────────────────

def test_h1(df):
    consensus = df[df["consensus_reached"]].copy()
    gaps = consensus["conversation_vs_best_strategy_gss"].values

    # Distribution diagnostic — justifies Wilcoxon over a t-test.
    skew = pd.Series(gaps).skew()
    kurt = pd.Series(gaps).kurt()
    FRAMES["H1 Distribution Diagnostic"] = pd.DataFrame({
        "value": [
            round(float(skew), 3),
            round(float(kurt), 3),
            round(float(100 * (gaps == 0).mean()), 1),
            round(float(100 * (gaps <  0).mean()), 1),
            round(float(100 * (gaps >  0).mean()), 1),
            round(float(gaps.min()), 4),
            round(float(gaps.max()), 4),
        ],
    }, index=[
        "skewness", "excess_kurtosis",
        "pct_gaps_zero", "pct_gaps_below_zero", "pct_gaps_above_zero",
        "min_gap", "max_gap",
    ])

    # Welfare-gap test: overall + per configuration.
    #   Primary test : one-sample Wilcoxon (one-sided 'less').
    #   Backup test  : sign test -- makes no symmetry assumption, so it
    #                  confirms the result is not an artefact of the gap's
    #                  strong skew.
    #   The four per-configuration Wilcoxon p-values are Holm-corrected;
    #   the OVERALL row is the single primary test and is left uncorrected.
    rows = []
    stat, p = stats.wilcoxon(gaps, alternative="less")
    rb = rank_biserial_wilcoxon(gaps)
    s_pos, s_neg, s_p = sign_test(gaps, "less")
    rows.append({
        "scope": "OVERALL", "n": len(gaps),
        "mean": round(float(gaps.mean()), 4),
        "median": round(float(np.median(gaps)), 4),
        "std": round(float(gaps.std()), 4),
        "wilcoxon_W": round(float(stat), 1),
        "p_value": p,
        "p_holm": float("nan"),
        "r_rank_biserial": round(float(rb), 3),
        "sign_pos": s_pos, "sign_neg": s_neg, "sign_test_p": s_p,
        "supported": bool(p < 0.05 and rb < 0),
    })
    cfg_rows = []
    for cfg in CONFIGS:
        gaps_c = consensus.loc[consensus["group_config"] == cfg,
                               "conversation_vs_best_strategy_gss"].values
        if len(gaps_c) == 0:
            continue
        stat, p = stats.wilcoxon(gaps_c, alternative="less")
        rb = rank_biserial_wilcoxon(gaps_c)
        s_pos, s_neg, s_p = sign_test(gaps_c, "less")
        cfg_rows.append({
            "scope": cfg, "n": len(gaps_c),
            "mean": round(float(gaps_c.mean()), 4),
            "median": round(float(np.median(gaps_c)), 4),
            "std": round(float(gaps_c.std()), 4),
            "wilcoxon_W": round(float(stat), 1),
            "p_value": p,
            "r_rank_biserial": round(float(rb), 3),
            "sign_pos": s_pos, "sign_neg": s_neg, "sign_test_p": s_p,
        })
    if cfg_rows:
        adj = holm_correction([row["p_value"] for row in cfg_rows])
        for row, pa in zip(cfg_rows, adj):
            row["p_holm"] = pa
            row["supported"] = bool(pa < 0.05 and row["r_rank_biserial"] < 0)
    rows.extend(cfg_rows)
    FRAMES["H1 Welfare Gap"] = pd.DataFrame(rows).set_index("scope")


# ── H2 ───────────────────────────────────────────────────────────────────────

def test_h2(df):
    consensus = df[df["consensus_reached"]].copy()
    samples = [consensus[consensus["group_config"] == c]["min_sat"].values
               for c in CONFIGS]
    sizes = [len(s) for s in samples]
    n_total = sum(sizes)

    # (a) Kruskal-Wallis on MinSat
    h, p = stats.kruskal(*samples)
    eps2 = epsilon_squared_kw(h, n_total)
    kw_rows = []
    for cfg, n_, s in zip(CONFIGS, sizes, samples):
        kw_rows.append({
            "config": cfg, "n": n_,
            "mean_min_sat": round(float(s.mean()), 4),
            "median_min_sat": round(float(np.median(s)), 4),
        })
    kw_df = pd.DataFrame(kw_rows).set_index("config")
    kw_df.loc["OMNIBUS_N"]   = [n_total, float("nan"), float("nan")]
    kw_df.loc["H_statistic"] = [round(float(h), 3), float("nan"), float("nan")]
    kw_df.loc["p_value"]     = [p, float("nan"), float("nan")]
    kw_df.loc["epsilon_sq"]  = [round(float(eps2), 4), float("nan"), float("nan")]
    FRAMES["H2a Kruskal-Wallis"] = kw_df

    # (b) Pairwise Mann-Whitney U with Holm correction
    raw_p, pair_rows = [], []
    for i, j in combinations(range(4), 2):
        x, y = samples[i], samples[j]
        u, pr = stats.mannwhitneyu(x, y, alternative="two-sided")
        rb = rank_biserial_mwu(u, len(x), len(y))
        raw_p.append(pr)
        pair_rows.append({
            "config_A": CONFIGS[i], "config_B": CONFIGS[j],
            "nA": len(x), "nB": len(y),
            "U_statistic": round(float(u), 1),
            "p_raw": pr,
            "r_rank_biserial": round(float(rb), 3),
            "effect_size": interpret_r(rb),
        })
    adj_p = holm_correction(raw_p)
    for row, pa in zip(pair_rows, adj_p):
        row["p_holm"] = pa
        row["significant_holm"] = bool(pa < 0.05)
    # Tidy column order
    pairwise_df = pd.DataFrame(pair_rows)[[
        "config_A", "config_B", "nA", "nB",
        "U_statistic", "p_raw", "p_holm",
        "r_rank_biserial", "effect_size", "significant_holm",
    ]]
    FRAMES["H2a Pairwise MWU"] = pairwise_df

    # (c) Wilcoxon on MajMinGap per configuration.
    #     NON-ZERO gaps only: zero gaps can come from unanimous groups OR
    #     from split-vote groups where majority and minority voters have equal
    #     ISS. They provide no signed evidence for whether the realized
    #     majority-minority satisfaction gap is positive, so the tested
    #     population is consensus groups with a non-zero realized gap.
    #     The four per-configuration p-values are Holm-Bonferroni corrected.
    majmin_rows = []
    for cfg in CONFIGS:
        gaps = consensus.loc[consensus["group_config"] == cfg, "maj_min_gap"].values
        nonzero = gaps[gaps != 0]
        if len(nonzero) == 0:
            continue
        stat, p = stats.wilcoxon(nonzero, alternative="greater")
        rb = rank_biserial_wilcoxon(nonzero)
        majmin_rows.append({
            "config": cfg, "n_nonzero": len(nonzero),
            "mean": round(float(nonzero.mean()), 4),
            "median": round(float(np.median(nonzero)), 4),
            "wilcoxon_W": round(float(stat), 1),
            "p_value": p,
            "r_rank_biserial": round(float(rb), 3),
        })
    if majmin_rows:
        adj = holm_correction([row["p_value"] for row in majmin_rows])
        for row, pa in zip(majmin_rows, adj):
            row["p_holm"] = pa
            row["supported"] = bool(pa < 0.05 and row["r_rank_biserial"] > 0)
    FRAMES["H2b MajMinGap"] = pd.DataFrame(majmin_rows).set_index("config")


# ── H3 ───────────────────────────────────────────────────────────────────────

def test_h3(df):
    consensus = df[df["consensus_reached"]].copy()
    h3_mask = (consensus["min_sat"] < 0.20) & (consensus["maj_min_gap"] > 0.30)
    h3_sub = consensus[h3_mask]
    rows = []
    for cfg in CONFIGS:
        n_cfg = int((h3_sub["group_config"] == cfg).sum())
        n_total = int((consensus["group_config"] == cfg).sum())
        share = 100 * n_cfg / n_total if n_total else 0
        rows.append({
            "config": cfg,
            "n_in_h3_subset": n_cfg,
            "n_total_consensus": n_total,
            "pct_of_config_in_h3": round(share, 1),
        })
    rows.append({
        "config": "TOTAL",
        "n_in_h3_subset": len(h3_sub),
        "n_total_consensus": len(consensus),
        "pct_of_config_in_h3": round(100 * len(h3_sub) / len(consensus), 1)
                                if len(consensus) else 0,
    })
    FRAMES["H3 Subset"] = pd.DataFrame(rows).set_index("config")


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
        if data["final_rec"] in ("NO CONSENSUS REACHED", "No preference yet"):
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
    agents = collect_agent_level(full_dataset_folder, results_json_path)

    # Big Five: highlighted traits listed first.
    ordered_traits = [
        ("agreeableness",     "primary focus (Barile 2024)"),
        ("neuroticism",       "primary focus (= reversed Emotional Stability)"),
        ("openness",          ""),
        ("conscientiousness", ""),
        ("extraversion",      ""),
    ]
    # Each correlation gets a 95% confidence interval (Fisher z-transform).
    # H4 claims the effect is NEGLIGIBLE, so the decision rule is an
    # equivalence-style test: H4 holds only if the WHOLE confidence
    # interval of every trait lies within +/-0.05. Checking the point
    # estimate alone would ignore the uncertainty around it.
    rows = []
    all_point_within = True
    all_ci_within = True
    for trait, note in ordered_traits:
        sub = agents.dropna(subset=[trait])
        if len(sub) < 4:
            continue
        r, p = stats.pearsonr(sub[trait].values, sub["iss"].values)
        ci_lo, ci_hi = pearson_ci(float(r), len(sub), conf=0.95)
        point_within = abs(r) < 0.05
        ci_within = (ci_lo > -0.05) and (ci_hi < 0.05)
        all_point_within = all_point_within and point_within
        all_ci_within = all_ci_within and ci_within
        rows.append({
            "trait": trait, "n": len(sub),
            "pearson_r": round(float(r), 4),
            "abs_r": round(float(abs(r)), 4),
            "ci95_low": round(float(ci_lo), 4),
            "ci95_high": round(float(ci_hi), 4),
            "p_value": p,
            "point_within_005": bool(point_within),
            "ci_within_005": bool(ci_within),
            "focus_note": note,
        })
    h4_df = pd.DataFrame(rows).set_index("trait")
    h4_df.loc["RESULT"] = [
        len(agents), float("nan"), float("nan"), float("nan"),
        float("nan"), float("nan"),
        bool(all_point_within), bool(all_ci_within),
        "H4 SUPPORTED" if all_ci_within else "H4 NOT SUPPORTED",
    ]
    FRAMES["H4 Big Five"] = h4_df

    # Supplementary: tone manipulation check (NOT part of H4).
    by_tone = agents.dropna(subset=["tone"]).groupby("tone")["iss"].apply(list)
    samples = list(by_tone.values)
    sizes   = [len(s) for s in samples]
    n_tones = len(samples)
    if n_tones >= 2:
        h, p = stats.kruskal(*samples)
        eps2 = epsilon_squared_kw(h, sum(sizes))
        means = [np.mean(s) for s in samples]
        FRAMES["H4 Tone (Supplementary)"] = pd.DataFrame({
            "value": [
                n_tones, sum(sizes),
                round(min(means), 3), round(max(means), 3),
                round(float(h), 3), p, round(float(eps2), 4),
            ],
        }, index=[
            "n_tone_categories", "n_observations",
            "mean_iss_tone_min", "mean_iss_tone_max",
            "H_statistic", "p_value", "epsilon_sq",
        ])


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) not in (3, 4):
        sys.exit("Usage: python hypothesis_tests.py "
                 "<results_csv> <full_dataset_folder> [output_xlsx]")

    results_csv  = Path(sys.argv[1])
    full_dataset = Path(sys.argv[2])
    xlsx_path = (Path(sys.argv[3]) if len(sys.argv) == 4
                 else results_csv.parent / "hypothesis_tests_output.xlsx")
    results_json = results_csv.parent / "results.json"

    if not results_csv.exists():
        sys.exit(f"results.csv not found: {results_csv}")
    if not full_dataset.is_dir():
        sys.exit(f"full_dataset folder not found: {full_dataset}")
    if not results_json.exists():
        sys.exit(f"results.json not found: {results_json}")

    df = pd.read_csv(results_csv)
    test_h1(df)
    test_h2(df)
    test_h3(df)
    test_h4(full_dataset, results_json)

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for sheet_name, frame in FRAMES.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31])
    print(f"Saved: {xlsx_path}")


if __name__ == "__main__":
    main()

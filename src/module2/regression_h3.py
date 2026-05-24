#!/usr/bin/env python3
"""
regression_h3.py  --  Module 2 / RQ3-H3 nested OLS regression
=============================================================
Tests H3: within structurally-unfair groups (the H3 subset), do
process-level metrics add diagnostic information BEYOND structural
metrics?

Input : data/results/h3_agent_level.csv     (built by build_h3_dataset.py)
Output: data/results/regression_h3_output.xlsx
        data/results/regression_h3_output.txt

Design (see more/module2_analysis_plan.md)
------------------------------------------
Dependent variable : iss  (individual satisfaction, 0-1, agent level)

Model 1 (baseline) : structural controls only
    iss ~ min_sat + maj_min_gap + n_agents + majority_voter
          + C(group_config)              [uniform = reference]

Model 2 (full)     : Model 1 + the four process predictors
    + mention_rate + repetition_index + social_shift + process_quality

Both models are OLS at the agent level with cluster-robust (Huber-White)
standard errors clustered by group_id, and -- crucially -- are fit on the
EXACT SAME ROWS, so the nested comparison is valid.

Decision rule (H3 supported iff all three hold)
-----------------------------------------------
 1. the cluster-robust joint F-test that the 4 process coefficients are
    all zero is significant at alpha = 0.05  (this IS the Model 1 vs
    Model 2 comparison);
 2. >= 2 of the 4 process predictors have Holm-corrected p < 0.05 with
    the expected sign (mention_rate +, repetition_index -,
    social_shift -, process_quality +);
 3. the R-squared increment from Model 1 to Model 2 is >= 0.03.
Exactly one significant process predictor -> "partially supported".
None -> "rejected".

Robustness checks (secondary, reported in the workbook / appendix)
 1. worst-agent ISS as DV, group level
 2. Model 2 on the full judged set (not only the H3 subset)
 3. Model 2 fit separately within each group configuration

Requires: pandas, numpy, statsmodels, openpyxl     (pip install statsmodels)

Run from the repo root:
    python src/module2/regression_h3.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:
    sys.exit("ERROR: statsmodels is required.  Install it with:\n"
             "    pip install statsmodels")

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEF_CSV = REPO_ROOT / "data" / "results" / "h3_agent_level.csv"
DEF_XLSX = REPO_ROOT / "data" / "results" / "regression_h3_output.xlsx"
DEF_TXT = REPO_ROOT / "data" / "results" / "regression_h3_output.txt"

DV = "iss"
STRUCT_TERMS = ["min_sat", "maj_min_gap", "n_agents", "majority_voter"]
PROCESS_TERMS = ["mention_rate", "repetition_index", "social_shift",
                 "process_quality"]
CONFIG_TERM = "C(group_config, Treatment(reference='uniform'))"

# expected sign of each process predictor under the diagnostic interpretation
EXPECTED_SIGN = {"mention_rate": +1, "repetition_index": -1,
                 "social_shift": -1, "process_quality": +1}

ALPHA = 0.05
R2_INCREMENT_MIN = 0.03
MIN_CLUSTERS_WARN = 30           # below this, cluster-robust SEs are shaky

LOG: list = []


def out(s: str = "") -> None:
    """Print to stdout and capture the line for the .txt report."""
    print(s)
    LOG.append(str(s))


def rule(char: str = "-", n: int = 70) -> None:
    out(char * n)


# --------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------
def fit_clustered(formula: str, data: pd.DataFrame):
    """Agent-level OLS with cluster-robust SE clustered by group_id."""
    model = smf.ols(formula, data=data)
    return model.fit(cov_type="cluster",
                     cov_kwds={"groups": data["group_id"]})


def coef_table(res, holm_p: dict | None = None) -> pd.DataFrame:
    """Tidy coefficient table: coef, robust SE, t, p, 95% CI (+ Holm p)."""
    ci = res.conf_int()
    tab = pd.DataFrame({
        "coef": res.params,
        "std_err": res.bse,
        "t": res.tvalues,
        "p_raw": res.pvalues,
        "ci_2.5%": ci.iloc[:, 0],
        "ci_97.5%": ci.iloc[:, 1],
    })
    if holm_p:
        tab["p_holm"] = [holm_p.get(t, np.nan) for t in tab.index]
    return tab.reset_index().rename(columns={"index": "term"})


def drop_incomplete(df: pd.DataFrame, cols: list):
    """Drop rows missing ANY model variable, so Model 1 and Model 2 are
    fit on the identical sample (required for a valid nested F-test)."""
    before = len(df)
    clean = df.dropna(subset=cols).copy()
    return clean, before - len(clean)


def joint_F(res_full, terms: list):
    """Cluster-robust joint Wald F-test that every coefficient in `terms`
    is zero.  Applied to the process block this is the Model 1 vs Model 2
    comparison."""
    ft = res_full.f_test([f"{t} = 0" for t in terms])
    return (float(np.asarray(ft.fvalue).squeeze()),
            float(np.asarray(ft.pvalue).squeeze()),
            float(np.asarray(ft.df_num).squeeze()),
            float(np.asarray(ft.df_denom).squeeze()))


# --------------------------------------------------------------------
# PRIMARY H3 TEST
# --------------------------------------------------------------------
def run_primary(df_h3: pd.DataFrame) -> dict:
    rule("=")
    out("PRIMARY H3 TEST  --  agent-level nested OLS, cluster-robust by group")
    rule("=")

    model_vars = [DV, "group_id", "group_config"] + STRUCT_TERMS + PROCESS_TERMS
    data, n_drop = drop_incomplete(df_h3, model_vars)
    n_obs = len(data)
    n_grp = data["group_id"].nunique()
    out(f"agent observations : {n_obs}")
    out(f"groups (clusters)  : {n_grp}")
    if n_drop:
        out(f"rows dropped (missing a model variable): {n_drop}")
    if n_grp < MIN_CLUSTERS_WARN:
        out("")
        out(f"WARNING: only {n_grp} clusters -- cluster-robust SEs and the")
        out(f"         joint F-test are unreliable below ~{MIN_CLUSTERS_WARN}")
        out("         clusters.  Treat this run as a PIPELINE TEST, not as")
        out("         valid statistical inference.")
    out("")

    f_struct = " + ".join(STRUCT_TERMS) + " + " + CONFIG_TERM
    f_m1 = f"{DV} ~ {f_struct}"
    f_m2 = f"{f_m1} + " + " + ".join(PROCESS_TERMS)

    m1 = fit_clustered(f_m1, data)
    m2 = fit_clustered(f_m2, data)

    # --- model comparison -------------------------------------------
    fF, fp, df_num, df_den = joint_F(m2, PROCESS_TERMS)
    r2_1, r2_2 = m1.rsquared, m2.rsquared
    r2a_1, r2a_2 = m1.rsquared_adj, m2.rsquared_adj
    r2_inc = r2_2 - r2_1

    # classical homoscedastic nested F -- reported for reference only
    classical_F = classical_p = float("nan")
    try:
        m1o = smf.ols(f_m1, data=data).fit()
        m2o = smf.ols(f_m2, data=data).fit()
        av = anova_lm(m1o, m2o)
        classical_F = float(av["F"].iloc[-1])
        classical_p = float(av["Pr(>F)"].iloc[-1])
    except Exception as exc:                       # noqa: BLE001
        out(f"(classical anova_lm unavailable: {exc})")

    # --- Holm correction across the 4 process p-values --------------
    raw_p = {t: float(m2.pvalues[t]) for t in PROCESS_TERMS}
    _, p_holm_arr, _, _ = multipletests([raw_p[t] for t in PROCESS_TERMS],
                                        alpha=ALPHA, method="holm")
    holm_p = dict(zip(PROCESS_TERMS, [float(x) for x in p_holm_arr]))

    # --- sign check --------------------------------------------------
    sig_correct, sig_wrong = [], []
    for t in PROCESS_TERMS:
        sign_ok = np.sign(m2.params[t]) == EXPECTED_SIGN[t]
        if holm_p[t] < ALPHA:
            (sig_correct if sign_ok else sig_wrong).append(t)

    # --- VIF on the Model 2 design matrix ---------------------------
    exog = np.asarray(m2.model.exog)
    names = list(m2.model.exog_names)
    vif = {names[i]: float(variance_inflation_factor(exog, i))
           for i in range(len(names))}
    vif_no_int = {k: v for k, v in vif.items() if k != "Intercept"}
    max_vif = max(vif_no_int.values()) if vif_no_int else float("nan")

    # --- report ------------------------------------------------------
    out(f"Model 1 (baseline): R2 = {r2_1:.4f}   adj-R2 = {r2a_1:.4f}"
        f"   overall F p = {m1.f_pvalue:.4g}")
    out(f"Model 2 (full)    : R2 = {r2_2:.4f}   adj-R2 = {r2a_2:.4f}"
        f"   overall F p = {m2.f_pvalue:.4g}")
    out(f"R2 increment      : {r2_inc:+.4f}   (threshold >= {R2_INCREMENT_MIN})")
    out("")
    out("Model 1 vs Model 2 -- cluster-robust joint F-test on the 4 "
        "process terms:")
    out(f"   F({df_num:.0f}, {df_den:.0f}) = {fF:.3f}   p = {fp:.4g}")
    if not np.isnan(classical_F):
        out(f"   (classical homoscedastic F = {classical_F:.3f}, "
            f"p = {classical_p:.4g} -- reference only; the clustered test")
        out("    above is the one the decision rule uses)")
    out("")
    out("Process predictors in Model 2:")
    out(f"   {'term':18s} {'coef':>10s}  {'sign':>11s}  {'p_raw':>10s}"
        f"  {'p_holm':>10s}")
    for t in PROCESS_TERMS:
        coef = m2.params[t]
        want = "+" if EXPECTED_SIGN[t] > 0 else "-"
        got = "+" if coef >= 0 else "-"
        flag = "ok" if got == want else "WRONG"
        out(f"   {t:18s} {coef:>+10.4f}  exp {want} got {got} {flag:>3s}"
            f"  {raw_p[t]:>10.4g}  {holm_p[t]:>10.4g}")
    out("")
    out(f"VIF (max non-intercept) = {max_vif:.2f}"
        + ("  -- EXCEEDS 5: inspect collinearity (consider dropping the "
           "more redundant of mention_rate / process_quality)"
           if max_vif > 5 else "  -- OK (<= 5)"))
    out("")

    # --- decision rule ----------------------------------------------
    cond1 = fp < ALPHA
    cond2 = len(sig_correct) >= 2
    cond3 = r2_inc >= R2_INCREMENT_MIN
    rule()
    out("DECISION RULE")
    out(f"  [{'PASS' if cond1 else 'FAIL'}] (1) joint F-test significant "
        f"(p = {fp:.4g} {'<' if cond1 else '>='} {ALPHA})")
    out(f"  [{'PASS' if cond2 else 'FAIL'}] (2) >= 2 process predictors "
        f"Holm-significant with the expected sign")
    out(f"           -> have {len(sig_correct)}: "
        f"{', '.join(sig_correct) if sig_correct else 'none'}")
    out(f"  [{'PASS' if cond3 else 'FAIL'}] (3) R2 increment >= "
        f"{R2_INCREMENT_MIN}  (have {r2_inc:+.4f})")
    if sig_wrong:
        out(f"  note: {', '.join(sig_wrong)} reached significance but with "
            f"the WRONG sign -- not counted toward support")

    if cond1 and cond2 and cond3:
        verdict = "H3 SUPPORTED"
    elif len(sig_correct) == 1:
        verdict = ("H3 PARTIALLY SUPPORTED -- one process channel carries "
                   "distinct signal")
    elif len(sig_correct) == 0:
        verdict = ("H3 REJECTED -- within structurally unfair groups, process "
                   "metrics are redundant with structural metrics")
    else:
        verdict = ("H3 NOT FULLY SUPPORTED -- >= 2 process predictors carry "
                   "signal, but not every decision-rule condition is met "
                   "(see the three conditions above)")
    out("")
    out(f"  VERDICT: {verdict}")
    rule()
    out("")

    # --- assemble workbook tables -----------------------------------
    cmp_tbl = pd.DataFrame({
        "metric": ["n_obs", "n_groups", "rows_dropped",
                   "R2_model1", "R2_model2", "R2_increment",
                   "adjR2_model1", "adjR2_model2",
                   "model1_overall_F", "model1_overall_F_p",
                   "model2_overall_F", "model2_overall_F_p",
                   "joint_process_F", "joint_F_df_num", "joint_F_df_denom",
                   "joint_process_F_p",
                   "classical_F_reference", "classical_F_p_reference"],
        "value": [n_obs, n_grp, n_drop, r2_1, r2_2, r2_inc, r2a_1, r2a_2,
                  float(m1.fvalue), float(m1.f_pvalue),
                  float(m2.fvalue), float(m2.f_pvalue),
                  fF, df_num, df_den, fp, classical_F, classical_p],
    })
    vif_tbl = pd.DataFrame({"term": list(vif.keys()),
                            "VIF": list(vif.values())})
    summary_tbl = pd.DataFrame({
        "item": ["condition_1_jointF_significant",
                 "condition_2_two_process_predictors_correct_sign",
                 "condition_3_R2_increment_ge_0.03",
                 "n_process_significant_correct_sign",
                 "n_process_significant_wrong_sign",
                 "VERDICT"],
        "result": [cond1, cond2, cond3,
                   len(sig_correct), len(sig_wrong), verdict],
    })
    return {
        "Summary": summary_tbl,
        "Model1_coef": coef_table(m1),
        "Model2_coef": coef_table(m2, holm_p=holm_p),
        "Model_comparison": cmp_tbl,
        "VIF": vif_tbl,
    }


# --------------------------------------------------------------------
# ROBUSTNESS 1 -- worst-agent ISS as DV (group level)
# --------------------------------------------------------------------
def run_worst_iss(df_h3: pd.DataFrame) -> pd.DataFrame:
    rule("=")
    out("ROBUSTNESS 1  --  worst-agent ISS as dependent variable (group level)")
    rule("=")
    model_vars = [DV, "group_id", "group_config"] + STRUCT_TERMS + PROCESS_TERMS
    data, _ = drop_incomplete(df_h3, model_vars)
    g = data.groupby("group_id")
    grp = pd.DataFrame({
        "worst_iss": g["iss"].min(),
        "min_sat": g["min_sat"].first(),
        "maj_min_gap": g["maj_min_gap"].first(),
        "n_agents": g["n_agents"].first(),
        "group_config": g["group_config"].first(),
        "mention_rate": g["mention_rate"].mean(),      # group mean
        "repetition_index": g["repetition_index"].mean(),
        "social_shift": g["social_shift"].mean(),      # = proportion shifted
        "process_quality": g["process_quality"].first(),
    }).reset_index()
    out(f"groups: {len(grp)}  (one row per group; majority_voter is agent-")
    out("level and is therefore omitted; agent-level process predictors are")
    out("entered as group means. One row per group -> HC3 robust SE, no clustering.")
    n_params_m2 = 1 + 3 + len(STRUCT_TERMS) - 1 + len(PROCESS_TERMS)
    if len(grp) <= n_params_m2 + 1:
        out(f"SKIPPED: only {len(grp)} groups for ~{n_params_m2} parameters.")
        out("")
        return pd.DataFrame()

    f_struct = "min_sat + maj_min_gap + n_agents + " + CONFIG_TERM
    f1 = f"worst_iss ~ {f_struct}"
    f2 = f"{f1} + " + " + ".join(PROCESS_TERMS)
    m1 = smf.ols(f1, data=grp).fit(cov_type="HC3")
    m2 = smf.ols(f2, data=grp).fit(cov_type="HC3")
    ft = m2.f_test([f"{t} = 0" for t in PROCESS_TERMS])
    fF = float(np.asarray(ft.fvalue).squeeze())
    fp = float(np.asarray(ft.pvalue).squeeze())
    out(f"Model 1 R2 = {m1.rsquared:.4f}   Model 2 R2 = {m2.rsquared:.4f}"
        f"   increment = {m2.rsquared - m1.rsquared:+.4f}")
    out(f"joint process F = {fF:.3f}   p = {fp:.4g}")
    out("")
    return coef_table(m2)


# --------------------------------------------------------------------
# ROBUSTNESS 2 -- Model 2 on the full judged set
# --------------------------------------------------------------------
def run_full_judged(df_all: pd.DataFrame) -> pd.DataFrame:
    rule("=")
    out("ROBUSTNESS 2  --  Model 2 on the full judged set (not only H3)")
    rule("=")
    model_vars = [DV, "group_id", "group_config"] + STRUCT_TERMS + PROCESS_TERMS
    data, _ = drop_incomplete(df_all, model_vars)
    n_h3 = int(data["is_h3"].sum())
    n_non = len(data) - n_h3
    if n_non == 0:
        out("All judged groups are inside the H3 subset, so this check is")
        out("identical to the primary test.  To run it as the plan intends,")
        out("the LLM judge must also be run on consensus groups OUTSIDE the")
        out("H3 subset (a larger / costlier judging run than the H3-only one).")
        out("")
        return pd.DataFrame()
    out(f"agents: {len(data)}   (H3 = {n_h3}, non-H3 = {n_non})")
    f_struct = " + ".join(STRUCT_TERMS) + " + " + CONFIG_TERM
    f2 = f"{DV} ~ {f_struct} + " + " + ".join(PROCESS_TERMS)
    m2 = fit_clustered(f2, data)
    ft = m2.f_test([f"{t} = 0" for t in PROCESS_TERMS])
    fp = float(np.asarray(ft.pvalue).squeeze())
    out(f"Model 2 R2 = {m2.rsquared:.4f}   joint process F p = {fp:.4g}")
    out("")
    return coef_table(m2)


# --------------------------------------------------------------------
# ROBUSTNESS 3 -- Model 2 within each group configuration
# --------------------------------------------------------------------
def run_per_config(df_h3: pd.DataFrame) -> pd.DataFrame:
    rule("=")
    out("ROBUSTNESS 3  --  Model 2 fit separately within each configuration")
    rule("=")
    model_vars = [DV, "group_id", "group_config"] + STRUCT_TERMS + PROCESS_TERMS
    f_struct = " + ".join(STRUCT_TERMS)          # no config dummy within a config
    f2 = f"{DV} ~ {f_struct} + " + " + ".join(PROCESS_TERMS)
    min_groups = 10
    rows = []
    for cfg in ["uniform", "divergent", "coalitional", "minority"]:
        sub = df_h3[df_h3["group_config"] == cfg]
        sub, _ = drop_incomplete(sub, model_vars)
        n_obs, n_grp = len(sub), sub["group_id"].nunique()
        if n_grp < min_groups or n_obs <= len(STRUCT_TERMS) + len(PROCESS_TERMS) + 2:
            out(f"  {cfg:12s}: SKIPPED ({n_grp} groups / {n_obs} agents "
                f"-- too few for a stable fit)")
            continue
        try:
            m2 = fit_clustered(f2, sub)
            ft = m2.f_test([f"{t} = 0" for t in PROCESS_TERMS])
            fp = float(np.asarray(ft.pvalue).squeeze())
            out(f"  {cfg:12s}: {n_grp} groups / {n_obs} agents   "
                f"R2 = {m2.rsquared:.4f}   joint process F p = {fp:.4g}")
            for t in PROCESS_TERMS:
                rows.append({"config": cfg, "term": t,
                             "coef": float(m2.params[t]),
                             "std_err": float(m2.bse[t]),
                             "p_raw": float(m2.pvalues[t]),
                             "n_groups": n_grp, "n_agents": n_obs})
        except Exception as exc:                  # noqa: BLE001
            out(f"  {cfg:12s}: FAILED ({exc})")
    out("")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="RQ3-H3 nested OLS regression (Module 2)")
    ap.add_argument("--csv", default=str(DEF_CSV),
                    help="agent-level CSV from build_h3_dataset.py")
    ap.add_argument("--xlsx", default=str(DEF_XLSX))
    ap.add_argument("--txt", default=str(DEF_TXT))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out("regression_h3.py  --  Module 2 / RQ3-H3 nested OLS regression")
    out(f"input: {args.csv}")
    out(f"total agent rows: {len(df)}   "
        f"(H3 = {int(df['is_h3'].sum())}, "
        f"non-H3 = {int((df['is_h3'] == 0).sum())})")
    out("")

    df_h3 = df[df["is_h3"] == 1].copy()
    if df_h3.empty:
        sys.exit("No H3 rows in the input CSV -- nothing to test.")

    sheets: dict = {}
    sheets.update(run_primary(df_h3))

    try:
        rob1 = run_worst_iss(df_h3)
        if not rob1.empty:
            sheets["Rob1_worstISS"] = rob1
    except Exception as exc:                       # noqa: BLE001
        out(f"ROBUSTNESS 1 failed: {exc}")
        out("")

    try:
        rob2 = run_full_judged(df)
        if not rob2.empty:
            sheets["Rob2_fulljudged"] = rob2
    except Exception as exc:                       # noqa: BLE001
        out(f"ROBUSTNESS 2 failed: {exc}")
        out("")

    try:
        rob3 = run_per_config(df_h3)
        if not rob3.empty:
            sheets["Rob3_perconfig"] = rob3
    except Exception as exc:                       # noqa: BLE001
        out(f"ROBUSTNESS 3 failed: {exc}")
        out("")

    # --- write workbook ---------------------------------------------
    xlsx_path = Path(args.xlsx)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        for name, tbl in sheets.items():
            if tbl is None or tbl.empty:
                continue
            tbl.to_excel(xw, sheet_name=name[:31], index=False)
    out(f"workbook written : {xlsx_path}")

    # --- write text report ------------------------------------------
    txt_path = Path(args.txt)
    txt_path.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"text report written : {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

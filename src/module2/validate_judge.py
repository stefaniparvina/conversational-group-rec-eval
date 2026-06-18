"""Module 2: validate the GPT-4o judge against the human annotations on the 15
validation transcripts, reporting inter-rater agreement per metric (Cohen's kappa,
within-1, PABAK, and Landis & Koch labels). Reads validation_set.csv, the
human-filled annotation_workbook.xlsx, and llm_results_validation.jsonl; writes
judge_validation_report.txt/.csv."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# -- Configuration -------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

VAL_DIR    = PROJECT_ROOT / "data" / "validation"
VAL_CSV    = VAL_DIR / "validation_set.csv"
WB         = VAL_DIR / "annotation_workbook.xlsx"
LLM_JSONL  = VAL_DIR / "llm_results_validation.jsonl"
OUT_TXT    = VAL_DIR / "judge_validation_report.txt"
OUT_CSV    = VAL_DIR / "judge_validation_report.csv"

SEED    = 42
N_BOOT  = 2000

PQ_DIMS = ["preferences_heard", "shifts_justified",
           "mutual_respect", "logical_support"]


# -- Agreement statistics ------------------------------------------------------

def landis_koch(k) -> str:
    """Verbal label for a kappa value (Landis & Koch, 1977)."""
    if k is None:
        return "undefined (no label variance)"
    if k < 0.00:
        return "poor (worse than chance)"
    if k <= 0.20:
        return "slight"
    if k <= 0.40:
        return "fair"
    if k <= 0.60:
        return "moderate"
    if k <= 0.80:
        return "substantial"
    return "almost perfect"


def cohen_kappa(y1, y2, labels=None, weights=None):
    """Cohen's kappa between two label sequences (weights None / 'linear' /
    'quadratic'). Pure numpy/pandas, equivalent to sklearn's cohen_kappa_score.
    Returns None when kappa is undefined (no expected disagreement)."""
    if labels is None:
        labels = sorted(set(y1) | set(y2), key=str)
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    conf = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2):
        if a in idx and b in idx:
            conf[idx[a], idx[b]] += 1.0
    n = conf.sum()
    if n == 0:
        return None
    row, col = conf.sum(axis=1), conf.sum(axis=0)
    expected = np.outer(row, col) / n
    if weights is None:
        w = np.ones((k, k)); np.fill_diagonal(w, 0.0)
    elif weights == "linear":
        w = np.array([[abs(i - j) for j in range(k)] for i in range(k)], float)
    elif weights == "quadratic":
        w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float)
    else:
        raise ValueError(f"unknown weights option: {weights}")
    denom = float((w * expected).sum())
    if denom == 0.0:
        return None
    return 1.0 - float((w * conf).sum()) / denom


def kappa(human, judge, weights=None, labels=None):
    """Cohen's kappa, or None when it is undefined (fewer than two labels)."""
    if len(human) < 2 or len(set(human) | set(judge)) < 2:
        return None
    k = cohen_kappa(human, judge, labels=labels, weights=weights)
    return None if (k is None or np.isnan(k)) else float(k)


def bootstrap_ci(human, judge, weights=None, labels=None):
    """Percentile bootstrap 95% CI for Cohen's kappa."""
    human = np.asarray(human, dtype=object)
    judge = np.asarray(judge, dtype=object)
    n = len(human)
    if n < 3:
        return (None, None)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        hb, jb = human[idx], judge[idx]
        if len(set(hb) | set(jb)) < 2:
            continue
        try:
            k = cohen_kappa(hb, jb, labels=labels, weights=weights)
        except Exception:
            continue
        if k is not None and not np.isnan(k):
            vals.append(k)
    if len(vals) < 50:
        return (None, None)
    return (round(float(np.percentile(vals, 2.5)), 3),
            round(float(np.percentile(vals, 97.5)), 3))


def exact_pct(human, judge) -> float:
    if not human:
        return 0.0
    return round(sum(1 for h, j in zip(human, judge) if h == j) / len(human), 3)


def within1_pct(human, judge):
    """Adjacent agreement: fraction of ordinal pairs within one point.
    Returns None when the labels are not numeric (non-ordinal metrics)."""
    if not human:
        return None
    try:
        diffs = [abs(int(h) - int(j)) for h, j in zip(human, judge)]
    except (TypeError, ValueError):
        return None
    return round(sum(1 for d in diffs if d <= 1) / len(diffs), 3)


def pabak(human, judge):
    """Prevalence-adjusted bias-adjusted kappa = 2*p_observed - 1: a chance-corrected index not deflated when one category dominates (the
    'prevalence paradox'). Reported for the binary metrics."""
    if not human:
        return None
    p_o = sum(1 for h, j in zip(human, judge) if h == j) / len(human)
    return round(2.0 * p_o - 1.0, 3)


def make_result(name, human, judge, labels=None, ordinal=False, note=""):
    """Assemble one report row from paired human/judge label lists."""
    n = len(human)
    if n == 0:
        return {"metric": name, "n": 0, "exact": None, "within1": None, "kappa": None,
                "ci_low": None, "ci_high": None, "weighted": None, "pabak": None,
                "reading": "no paired data", "note": note}

    k_unw = kappa(human, judge, weights=None, labels=labels)
    k_wt  = kappa(human, judge, weights="quadratic", labels=labels) if ordinal else None
    headline = k_wt if ordinal else k_unw
    lo, hi = bootstrap_ci(human, judge,
                          weights="quadratic" if ordinal else None,
                          labels=labels)
    return {
        "metric":  name,
        "n":       n,
        "exact":   exact_pct(human, judge),
        "within1": within1_pct(human, judge) if ordinal else None,
        "kappa":   None if k_unw is None else round(k_unw, 3),
        "ci_low":  lo,
        "ci_high": hi,
        "weighted": None if k_wt is None else round(k_wt, 3),
        "pabak":   pabak(human, judge) if (labels is not None and len(labels) == 2) else None,
        "reading": landis_koch(headline),
        "note":    note,
    }


# -- Data loading --------------------------------------------------------------

def load_llm() -> dict:
    """Map group_id -> llm_output, reading the judge results file."""
    out = {}
    if not LLM_JSONL.exists():
        return out
    for line in LLM_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            out[int(rec["group_id"])] = rec.get("llm_output", {})
        except Exception:
            pass
    return out


def read_sheet(name: str):
    try:
        return pd.read_excel(WB, sheet_name=name)
    except Exception:
        return None


def llm_dominant_sentiment(entry: dict):
    """The most frequent sentiment across an agent's mention list."""
    sents = [m.get("sentiment") for m in entry.get("mentions", []) if m.get("sentiment")]
    if not sents:
        return None
    return Counter(sents).most_common(1)[0][0]


def norm_yesno(v):
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("yes", "y", "true"):
            return True
        if s in ("no", "n", "false"):
            return False
    if isinstance(v, bool):
        return v
    return None


# -- Metric pairing ------------------------------------------------------------

def pair_process_quality(llm: dict, results: list):
    """Four ordinal 0-3 dimensions, plus all dimensions pooled."""
    df = read_sheet("ProcessQuality")
    if df is None:
        return
    pooled_h, pooled_j = [], []
    for dim in PQ_DIMS:
        h, j = [], []
        for _, row in df.iterrows():
            gid = int(row["group_id"])
            hv = row.get(dim)
            if pd.isna(hv) or gid not in llm:
                continue
            jv = llm[gid].get("process_quality", {}).get(f"{dim}_score")
            if jv is None:
                continue
            h.append(int(hv)); j.append(int(jv))
        pooled_h += h; pooled_j += j
        results.append(make_result(f"Process Quality - {dim}", h, j,
                                   labels=[0, 1, 2, 3], ordinal=True))
    results.append(make_result("Process Quality - ALL DIMENSIONS POOLED",
                               pooled_h, pooled_j,
                               labels=[0, 1, 2, 3], ordinal=True,
                               note="headline kappa for the Process Quality Score"))


def pair_mention_rate(llm: dict, results: list):
    df = read_sheet("MentionRate")
    if df is None:
        return
    h_ack, j_ack, h_sen, j_sen = [], [], [], []
    for _, row in df.iterrows():
        gid = int(row["group_id"])
        agent = str(row["agent"])
        if gid not in llm:
            continue
        entry = next((e for e in llm[gid].get("mention_rate", [])
                      if str(e.get("agent")) == agent), None)
        if entry is None:
            continue
        hv = norm_yesno(row.get("acknowledged"))
        if hv is not None and entry.get("acknowledged") is not None:
            h_ack.append(bool(hv)); j_ack.append(bool(entry["acknowledged"]))
        hs = row.get("dominant_sentiment")
        js = llm_dominant_sentiment(entry)
        if isinstance(hs, str) and hs.strip() and js is not None:
            h_sen.append(hs.strip().lower()); j_sen.append(js)
    results.append(make_result("Mention Rate - acknowledged", h_ack, j_ack,
                               labels=[False, True]))
    results.append(make_result("Mention Rate - dominant_sentiment", h_sen, j_sen,
                               labels=["positive", "neutral", "dismissive"]))


def pair_justified_shifts(llm: dict, results: list):
    df = read_sheet("JustifiedShifts")
    if df is None:
        return
    h, j = [], []
    unmatched = 0
    for _, row in df.iterrows():
        if pd.isna(row.get("group_id")) or pd.isna(row.get("round")):
            continue
        gid = int(row["group_id"])
        agent = str(row["agent"])
        rnd = int(row["round"])
        hv = row.get("shift_type")
        if not (isinstance(hv, str) and hv.strip()) or gid not in llm:
            continue
        frm = str(row.get("from_preference"))
        to  = str(row.get("to_preference"))
        jshifts = llm[gid].get("justified_shifts", [])
        match = next((s for s in jshifts
                      if str(s.get("agent")) == agent
                      and int(s.get("round", -1)) == rnd), None)
        if match is None:        # fallback: match on agent + from/to preference
            match = next((s for s in jshifts
                          if str(s.get("agent")) == agent
                          and str(s.get("from_preference")) == frm
                          and str(s.get("to_preference")) == to), None)
        if match is None:
            unmatched += 1
            continue
        h.append(hv.strip().lower()); j.append(str(match.get("shift_type")))
    note = f"{unmatched} human-listed shift(s) had no matching judge entry" if unmatched else ""
    results.append(make_result("Justified Shifts - shift_type", h, j,
                               labels=["quality-based", "social", "unexplained"],
                               note=note))


def pair_repetition_index(llm: dict, results: list):
    df = read_sheet("RepetitionIndex")
    if df is None:
        return
    h_raw, j_raw = [], []
    for _, row in df.iterrows():
        gid = int(row["group_id"])
        agent = str(row["agent"])
        hv = row.get("times_repeated_before_ack")
        if pd.isna(hv) or gid not in llm:
            continue
        entry = next((e for e in llm[gid].get("repetition_index", [])
                      if str(e.get("agent")) == agent), None)
        if entry is None or entry.get("times_repeated_before_ack") is None:
            continue
        h_raw.append(int(hv)); j_raw.append(int(entry["times_repeated_before_ack"]))
    if not h_raw:
        results.append(make_result("Repetition Index - any repetition", [], []))
        return
    mad = round(float(np.mean([abs(a - b) for a, b in zip(h_raw, j_raw)])), 3)
    raw_exact = exact_pct(h_raw, j_raw)
    h_bin = [v >= 1 for v in h_raw]
    j_bin = [v >= 1 for v in j_raw]
    results.append(make_result(
        "Repetition Index - any repetition", h_bin, j_bin, labels=[False, True],
        note=f"raw-count exact agreement {raw_exact}, mean abs. difference {mad}"))


# -- Reporting -----------------------------------------------------------------

def fmt_ci(lo, hi):
    if lo is None or hi is None:
        return "      n/a      "
    return f"[{lo:+.2f}, {hi:+.2f}]"


def write_report(results: list, n_llm: int, n_human_sheets: int):
    lines = []
    lines.append("=" * 78)
    lines.append("MODULE 2 -- JUDGE VALIDATION REPORT  (GPT-4o judge vs. human annotator)")
    lines.append("=" * 78)
    lines.append(f"Judge results found for {n_llm}/15 validation transcripts.")
    lines.append(f"Human annotation sheets with data: {n_human_sheets}/4.")
    lines.append("")
    header = f"{'Metric':<42}{'n':>4}{'exact':>8}{'kappa':>8}{'95% CI':>17}{'  reading'}"
    lines.append(header)
    lines.append("-" * 78)
    for r in results:
        exact = "   n/a" if r["exact"] is None else f"{r['exact']*100:5.0f}%"
        kp = "  n/a" if r["kappa"] is None else f"{r['kappa']:+.2f}"
        if r["weighted"] is not None:
            kp = f"{r['weighted']:+.2f}w"
        lines.append(f"{r['metric']:<42}{r['n']:>4}{exact:>8}{kp:>8}"
                     f"{fmt_ci(r['ci_low'], r['ci_high']):>17}  {r['reading']}")
        if r["note"]:
            lines.append(f"{'':<42}  -> {r['note']}")
    lines.append("-" * 78)
    lines.append("")
    lines.append("Supplementary agreement statistics")
    lines.append("(robust to the kappa prevalence paradox -- see notes below):")
    lines.append(f"{'Metric':<42}{'exact':>8}{'within1':>9}{'PABAK':>8}")
    for r in results:
        ex = "   n/a" if r.get("exact") is None else f"{r['exact']*100:5.0f}%"
        w1 = "      n/a" if r.get("within1") is None else f"{r['within1']*100:7.0f}%"
        pb = "     n/a" if r.get("pabak") is None else f"{r['pabak']:+.2f}"
        lines.append(f"{r['metric']:<42}{ex:>8}{w1:>9}{pb:>8}")
    lines.append("")
    lines.append("Notes:")
    lines.append("- 'kappa' shows quadratic-weighted kappa (suffix w) for the ordinal")
    lines.append("  0-3 Process Quality dimensions, and unweighted Cohen's kappa")
    lines.append("  otherwise. Verbal labels:")
    lines.append("  <=.20 slight, <=.40 fair, <=.60 moderate, <=.80 substantial,")
    lines.append("  >.80 almost perfect.")
    lines.append("- The validation set is 15 transcripts. Per-dimension n is small (15),")
    lines.append("  so the bootstrap CI is wide -- always report kappa WITH its CI, and")
    lines.append("  lean on the pooled Process Quality row (n=60) as the headline.")
    lines.append("- A judge that does not reach 'substantial' agreement on a metric is a")
    lines.append("  finding to report and discuss, not a failure to hide.")
    lines.append("- 'within1' = adjacent agreement: share of ordinal 0-3 score pairs")
    lines.append("  agreeing within one point. 'PABAK' is a")
    lines.append("  prevalence-adjusted agreement index for the binary metrics, where")
    lines.append("  Cohen's kappa is distorted when one category dominates.")
    lines.append("=" * 78)

    report = "\n".join(lines)
    OUT_TXT.write_text(report, encoding="utf-8")
    pd.DataFrame(results)[
        ["metric", "n", "exact", "within1", "kappa", "ci_low", "ci_high",
         "weighted", "pabak", "reading", "note"]
    ].to_csv(OUT_CSV, index=False)
    print(report)
    print(f"\nSaved -> {OUT_TXT}")
    print(f"Saved -> {OUT_CSV}")


# -- Main ----------------------------------------------------------------------

def main():
    if not VAL_CSV.exists():
        sys.exit("validation_set.csv not found -- run select_validation_set.py first.")
    if not WB.exists():
        sys.exit("annotation_workbook.xlsx not found -- run "
                 "build_annotation_workbook.py first.")

    val_ids = set(pd.read_csv(VAL_CSV)["group_id"].astype(int))
    llm = {g: o for g, o in load_llm().items() if g in val_ids}
    if not llm:
        sys.exit("No judge results for the validation transcripts yet -- run:\n"
                 "    python src/module2/llm_evaluator.py --mode validate")

    n_filled = 0
    for sheet, col in [("ProcessQuality", "preferences_heard"),
                       ("MentionRate", "acknowledged"),
                       ("JustifiedShifts", "shift_type"),
                       ("RepetitionIndex", "times_repeated_before_ack")]:
        df = read_sheet(sheet)
        if df is not None and col in df.columns and df[col].notna().any():
            n_filled += 1
    if n_filled == 0:
        sys.exit("The annotation workbook has no human scores yet.\n"
                 "Fill in data/validation/annotation_workbook.xlsx, then rerun.")

    results = []
    pair_process_quality(llm, results)
    pair_mention_rate(llm, results)
    pair_justified_shifts(llm, results)
    pair_repetition_index(llm, results)

    write_report(results, n_llm=len(llm), n_human_sheets=n_filled)


if __name__ == "__main__":
    main()

# =============================================================================
# Module 2: Validation-set selector
# =============================================================================
# Picks a small, reproducible set of H3 transcripts to be hand-annotated by a
# human. Those human annotations are later compared against the GPT-4o judge
# (see validate_judge.py) to measure agreement (Cohen's kappa) -- i.e. to check
# whether the LLM judge can be trusted to stand in for a human annotator.
#
# Sampling design: STRATIFIED PROPORTIONAL sampling by group_config.
#   The H3 subset is roughly 38.5% divergent / 34.3% coalitional /
#   22.7% minority / 4.5% uniform. A 15-transcript sample is therefore
#   allocated 6 / 5 / 3 / 1, so the validation set mirrors the composition of
#   the data the judge will actually score. Proportional stratification keeps
#   the agreement estimate representative rather than skewed toward one
#   conversation type.
#
# Reproducibility: a single fixed RNG seed. Rerunning this script always
# returns exactly the same 15 transcripts.
#
# Paths are resolved relative to this script's location, so it runs from anywhere:
#   src/module2/select_validation_set.py    <- this script
#   data/results/results.csv                <- input  (Module 1 structural results)
#   data/validation/validation_set.csv      <- output (the 15 chosen transcripts)
#
# Usage:  python src/module2/select_validation_set.py
# Requirements:  pip install pandas
# =============================================================================

from pathlib import Path

import pandas as pd

# -- Configuration -------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

RESULTS_CSV = PROJECT_ROOT / "data" / "results" / "results.csv"
OUT_DIR     = PROJECT_ROOT / "data" / "validation"
OUT_CSV     = OUT_DIR / "validation_set.csv"

# H3 subset definition -- must match llm_evaluator.py.
H3_MINSAT = 0.20     # MinSat threshold for the H3 subset
H3_MMGAP  = 0.30     # MajMinGap threshold for the H3 subset

SEED = 42            # fixed seed -> the same 15 transcripts every run

# Per-stratum quotas. Sum = 15. Derived from the H3 subset proportions
# (divergent 38.5%, coalitional 34.3%, minority 22.7%, uniform 4.5%).
QUOTAS = {
    "divergent":   6,
    "coalitional": 5,
    "minority":    3,
    "uniform":     1,
}

# Columns carried into the output file (kept small and human-readable).
KEEP_COLS = [
    "group_id", "group_config", "n_agents", "turn_counter",
    "total_messages", "min_sat", "maj_min_gap", "final_rec",
]


def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"results.csv not found at {RESULTS_CSV}")

    df = pd.read_csv(RESULTS_CSV)

    # H3 subset: consensus reached, low minority satisfaction, large maj/min gap.
    h3 = df[
        df["consensus_reached"]
        & (df["min_sat"] < H3_MINSAT)
        & (df["maj_min_gap"] > H3_MMGAP)
    ]

    total_target = sum(QUOTAS.values())
    print(f"H3 subset size: {len(h3)} groups")
    print(f"Target validation set: {total_target} transcripts\n")
    print(f"{'config':<13}{'in H3':>8}{'share':>9}{'quota':>8}{'drawn':>8}")
    print("-" * 46)

    picks = []
    for config, quota in QUOTAS.items():
        stratum = h3[h3["group_config"] == config]
        share   = len(stratum) / len(h3) if len(h3) else 0.0
        n_draw  = min(quota, len(stratum))
        if n_draw < quota:
            print(f"  WARNING: stratum '{config}' has only {len(stratum)} groups "
                  f"(< quota {quota})")
        drawn = stratum.sample(n=n_draw, random_state=SEED)
        picks.append(drawn)
        print(f"{config:<13}{len(stratum):>8}{share:>8.1%}{quota:>8}{n_draw:>8}")

    selected = pd.concat(picks, ignore_index=True)
    selected = selected[KEEP_COLS].sort_values("group_id").reset_index(drop=True)

    print("-" * 46)
    print(f"{'TOTAL':<13}{len(h3):>8}{'':>9}{total_target:>8}{len(selected):>8}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(selected)} transcript IDs -> {OUT_CSV}")
    print("\nSelected group IDs:", selected["group_id"].tolist())


if __name__ == "__main__":
    main()

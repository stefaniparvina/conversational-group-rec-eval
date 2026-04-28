# A Framework for Evaluating Preference Fairness in Conversational Group Recommendation

---

## Overview

This repository contains the code and data for a two-module evaluation framework
that studies how different group configurations and aggregation strategies affect
fairness outcomes in conversational group recommendation systems.

The simulation data is based on Barile et al. (2024), covering 4 group
configurations (uniform, divergent, coalitional, minority) across 8,000 simulated
group conversations.

---

## Repository Structure

```
conversational-group-rec-eval/
├── data/
│   ├── raw/                  # Raw simulation outputs per LLM (gemma, llama, mistral, olmo)
│   ├── full_dataset/         # Merged dataset (8,000 group_simulation_*.json)
│   └── results/              # Computed metrics and hypothesis test outputs
│       ├── results.csv
│       ├── results.json
│       ├── stats_output.txt
│       └── hypothesis_test_results.txt
├── src/
│   ├── module1/
│   │   ├── merge_dataset.py        # Merges raw subfolders into full_dataset/
│   │   ├── evaluation_framework.py # Computes 7 fairness metrics per group
│   │   ├── reproduce_stats.py      # Reproduces descriptive statistics
│   │   └── hypothesis_tests.py     # Runs all statistical hypothesis tests
│   └── module2/                    # Process-level LLM evaluation
└── README.md
```

---

## How to Run (Module 1)

All commands should be run from the repo root (`conversational-group-rec-eval/`).

**Step 1 — Merge raw data into full dataset:**
```bash
python src/module1/merge_dataset.py
```

**Step 2 — Compute metrics:**
```bash
python src/module1/evaluation_framework.py data/full_dataset/
```

**Step 3 — Reproduce descriptive statistics:**
```bash
python src/module1/reproduce_stats.py
```

**Step 4 — Run hypothesis tests:**
```bash
python src/module1/hypothesis_tests.py data/results/results.csv data/full_dataset/
```

---

## Requirements

```bash
pip install pandas numpy scipy
```

---

## Notes

- `data/full_dataset/` and `data/raw/` are excluded from version control (large files).
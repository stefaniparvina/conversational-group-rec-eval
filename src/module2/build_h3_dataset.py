#!/usr/bin/env python3
"""
build_h3_dataset.py  --  Module 2 / RQ3-H3 dataset assembler
============================================================
Joins the structural results (results.json) with the LLM-judge process
metrics (llm_results.jsonl) and expands everything to ONE ROW PER AGENT.

Output: data/results/h3_agent_level.csv  -- the input to regression_h3.py.

Why two source files
--------------------
* results.json       : group-level structural metrics, per-agent ISS, and
                       the majority / minority voter lists.
* llm_results.jsonl  : the LLM judge's raw output.  The *per-agent* process
                       metrics live ONLY here -- the companion file
                       llm_results_summary.csv holds group-level averages
                       and therefore cannot drive an agent-level regression.

Columns written
---------------
group_id            int    group identifier
agent               str    agent (first) name
group_config        str    uniform / divergent / coalitional / minority
n_agents            int    group size
is_h3               int    1 if the group is in the H3 subset
                           (consensus & min_sat < 0.20 & maj_min_gap > 0.30)
audit_ok            int    1 if the judge's structural_audit passed, 0 if the
                           judge output was flagged (group-level; drives the
                           Robustness-4 sensitivity check in regression_h3.py)
iss                 float  DEPENDENT VARIABLE: individual satisfaction (0-1)
mention_rate        float  process predictor 1 (sentiment-weighted, agent)
repetition_index    int    process predictor 2 (agent)
social_shift        int    process predictor 3 (agent, binary)
process_quality     float  process predictor 4 (group, broadcast, 0-1)
min_sat             float  structural control (group)
maj_min_gap         float  structural control (group)
majority_voter      int    structural control (agent, 1 = voted the winner)

Run from the repo root:
    python src/module2/build_h3_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEF_RESULTS_JSON = REPO_ROOT / "data" / "results" / "results.json"
DEF_JUDGE_JSONL = REPO_ROOT / "data" / "results" / "llm_results.jsonl"
DEF_OUT_CSV = REPO_ROOT / "data" / "results" / "h3_agent_level.csv"

# H3 subset definition -- must match more/module2_analysis_plan.md
H3_MIN_SAT_MAX = 0.20
H3_GAP_MIN = 0.30

# Mention Rate: +1 positive / 0 neutral / -1 dismissive
SENTIMENT_WEIGHT = {"positive": 1, "neutral": 0, "dismissive": -1}


def is_h3_group(r: dict) -> bool:
    """A group is in the H3 subset iff it reached consensus AND is
    structurally unfair (low minimum satisfaction, large majority gap)."""
    if not r.get("consensus_reached"):
        return False
    if r.get("min_sat") is None or r.get("maj_min_gap") is None:
        return False
    return r["min_sat"] < H3_MIN_SAT_MAX and r["maj_min_gap"] > H3_GAP_MIN


def agent_mention_rate(mr_entry: dict, agent: str, n_agents: int) -> float:
    """Sentiment-weighted Mention Rate for one agent.

    +1 per positive, 0 per neutral, -1 per dismissive mention received
    from OTHER agents, normalised by (n_agents - 1).
    """
    if n_agents < 2:
        return 0.0
    total = 0
    for m in mr_entry.get("mentions", []):
        if m.get("by_agent") == agent:          # ignore any self-mention
            continue
        total += SENTIMENT_WEIGHT.get(m.get("sentiment"), 0)
    return total / (n_agents - 1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Assemble the agent-level H3 dataset for regression_h3.py")
    ap.add_argument("--results", default=str(DEF_RESULTS_JSON),
                    help="path to results.json")
    ap.add_argument("--judge", default=str(DEF_JUDGE_JSONL),
                    help="path to llm_results.jsonl")
    ap.add_argument("--out", default=str(DEF_OUT_CSV),
                    help="path to write the agent-level CSV")
    args = ap.parse_args()

    # ---- load structural results -----------------------------------
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    by_gid = {int(r["group_id"]): r for r in results}
    print(f"results.json        : {len(by_gid)} groups")

    # ---- load judge output -----------------------------------------
    judge: dict[int, dict] = {}
    with open(args.judge, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            judge[int(rec["group_id"])] = rec
    print(f"llm_results.jsonl   : {len(judge)} judged groups")

    rows: list[dict] = []
    name_mismatches: list[tuple] = []
    missing_metric: list[tuple] = []
    missing_audit: list[int] = []

    for gid, jrec in sorted(judge.items()):
        if gid not in by_gid:
            print(f"  !! judged group {gid} absent from results.json -- skipped")
            continue
        r = by_gid[gid]
        out = jrec["llm_output"]
        n_agents = int(r["n_agents"])
        h3 = is_h3_group(r)
        audit = jrec.get("structural_audit")
        if audit is None:
            missing_audit.append(gid)
        # audit_ok = 1 unless the judge's structural audit explicitly failed
        audit_ok = 0 if (audit is not None
                         and audit.get("structural_ok") is False) else 1

        # index the judge's per-agent lists by agent name
        mr_by_agent = {e["agent"]: e for e in out.get("mention_rate", [])}
        ri_by_agent = {e["agent"]: e for e in out.get("repetition_index", [])}
        social_agents = {s["agent"] for s in out.get("justified_shifts", [])
                         if s.get("shift_type") == "social"}
        pq = out.get("process_quality", {}).get("score")
        maj_voters = set(r.get("majority_voters", []))

        for agent, iss in r["iss_per_agent"].items():
            mr_entry = mr_by_agent.get(agent)
            ri_entry = ri_by_agent.get(agent)
            if mr_entry is None or ri_entry is None:
                name_mismatches.append(
                    (gid, agent, mr_entry is not None, ri_entry is not None))

            mention_rate = (agent_mention_rate(mr_entry, agent, n_agents)
                            if mr_entry is not None else None)
            repetition_index = (ri_entry.get("times_repeated_before_ack")
                                if ri_entry is not None else None)
            social_shift = 1 if agent in social_agents else 0

            if mention_rate is None or repetition_index is None or pq is None:
                missing_metric.append((gid, agent))

            rows.append({
                "group_id": gid,
                "agent": agent,
                "group_config": r["group_config"],
                "n_agents": n_agents,
                "is_h3": int(h3),
                "audit_ok": audit_ok,
                "iss": iss,
                "mention_rate": mention_rate,
                "repetition_index": repetition_index,
                "social_shift": social_shift,
                "process_quality": pq,
                "min_sat": r["min_sat"],
                "maj_min_gap": r["maj_min_gap"],
                "majority_voter": 1 if agent in maj_voters else 0,
            })

    # ---- write CSV --------------------------------------------------
    fields = ["group_id", "agent", "group_config", "n_agents", "is_h3",
              "audit_ok", "iss", "mention_rate", "repetition_index",
              "social_shift", "process_quality", "min_sat", "maj_min_gap",
              "majority_voter"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- report -----------------------------------------------------
    n_h3 = sum(1 for x in rows if x["is_h3"])
    n_audit_flag = sum(1 for x in rows if x["audit_ok"] == 0)
    flagged_groups = {x["group_id"] for x in rows if x["audit_ok"] == 0}
    print("-" * 62)
    print(f"agent rows written  : {len(rows)}  ->  {out_path}")
    print(f"  in H3 subset      : {n_h3}")
    print(f"  outside H3 subset : {len(rows) - n_h3}")
    print(f"  audit-flagged     : {n_audit_flag} agent rows in "
          f"{len(flagged_groups)} groups (structural_ok=False)")
    if missing_audit:
        print(f"  !! {len(missing_audit)} group(s) had NO structural_audit "
              f"record -- treated as audit_ok=1")
    if name_mismatches:
        print(f"  !! {len(name_mismatches)} agent(s) had no judge entry "
              f"(possible name mismatch) -- first few:")
        for gid, agent, hm, hr in name_mismatches[:10]:
            print(f"       group {gid}: {agent!r} "
                  f"(mention_rate={'ok' if hm else 'MISSING'}, "
                  f"repetition_index={'ok' if hr else 'MISSING'})")
    else:
        print("  agent-name match  : OK (every agent matched the judge output)")
    if missing_metric:
        print(f"  !! {len(missing_metric)} row(s) carry a missing process "
              f"metric; regression_h3.py will drop these from BOTH models")
    return 0


if __name__ == "__main__":
    sys.exit(main())

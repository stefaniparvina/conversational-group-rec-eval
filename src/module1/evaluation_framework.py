#!/usr/bin/env python3
"""Module 1 structural analysis: compute the per-group fairness metrics and the
six aggregation-strategy baselines, then export them to CSV/JSON."""

import csv
import json
import statistics
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

# A rating >= this counts as an "approval" for APP; 5 is the 1-10 scale midpoint.
APP_APPROVAL_THRESHOLD = 5


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_iss(agent_history: dict, final_rec: str) -> float:
    """ISS = agent's rating of final_rec / its max rating; 0.0 if no consensus."""
    if final_rec in ("NO CONSENSUS REACHED", "No preference yet"):
        return 0.0  # a failed conversation benefits no one
    if final_rec not in agent_history:
        return 0.0
    max_rating = max(agent_history.values(), default=0)
    if max_rating == 0:
        return 0.0
    return agent_history[final_rec] / max_rating


def hypothetical_iss(agent_history: dict, restaurant: str) -> float:
    """ISS the agent would get for an arbitrary restaurant (for strategy comparison)."""
    max_rating = max(agent_history.values(), default=0)
    if max_rating == 0:
        return 0.0
    return agent_history.get(restaurant, 0) / max_rating


def hypothetical_gss(agents: list, restaurant: str) -> float:
    """Mean ISS across all agents if the given restaurant were chosen."""
    vals = [hypothetical_iss(a["history"], restaurant) for a in agents]
    return statistics.mean(vals) if vals else 0.0


# ── Aggregation strategies ───────────────────────────────
# Each returns (recommendation, optimal_set): optimal_set is every restaurant tied
# for the best score on the strategy's criterion (used for the tie-independent
# "match" test); recommendation is the single reported choice after tie-breaking.

def _optimal_set(scores: dict) -> list[str]:
    """Every restaurant tied for the highest score, sorted by restaurant ID."""
    if not scores:
        return []
    best = max(scores.values())
    return sorted(r for r, v in scores.items() if v == best)


def _resolve(candidates: list[str], *tiebreaks: dict) -> str | None:
    """Reduce tied candidates to one: apply each tiebreak map in turn, then fall
    back to the lexicographically smallest ID (deterministic, order-independent)."""
    if not candidates:
        return None
    pool = list(candidates)
    for scores in tiebreaks:
        if len(pool) == 1:
            break
        best = max(scores.get(r, 0) for r in pool)
        pool = [r for r in pool if scores.get(r, 0) == best]
    return min(pool)


def _add_scores(agents: list, restaurants: list) -> dict:
    """Total of every agent's rating for each restaurant (the ADD criterion)."""
    return {r: sum(a["history"].get(r, 0) for a in agents) for r in restaurants}


def strategy_add(agents: list, restaurants: list) -> tuple[str | None, list[str]]:
    """ADD (Additive Utilitarian): highest total rating; lexicographic tiebreak."""
    scores = _add_scores(agents, restaurants)
    opt = _optimal_set(scores)
    return _resolve(opt), opt


def strategy_lms(agents: list, restaurants: list) -> tuple[str | None, list[str]]:
    """LMS (Least Misery): highest minimum individual rating; lexicographic tiebreak."""
    scores = {
        r: min(a["history"].get(r, 0) for a in agents)
        for r in restaurants
    }
    opt = _optimal_set(scores)
    return _resolve(opt), opt


def strategy_mpl(agents: list, restaurants: list) -> tuple[str | None, list[str]]:
    """MPL (Most Pleasure): highest single rating from any agent; lexicographic tiebreak."""
    scores = {
        r: max(a["history"].get(r, 0) for a in agents)
        for r in restaurants
    }
    opt = _optimal_set(scores)
    return _resolve(opt), opt


def strategy_maj(agents: list, restaurants: list) -> tuple[str | None, list[str]]:
    """MAJ (Majority): top preference of the most agents; ADD then lexicographic tiebreak."""
    votes: dict = defaultdict(Fraction)
    for a in agents:
        hist = a["history"]
        if not hist:
            continue
        # Split an agent's vote equally among joint favourites (exact Fraction),
        # so the result never depends on rating-history insertion order.
        best = max(hist.values())
        tops = [r for r, v in hist.items() if v == best]
        share = Fraction(1, len(tops))
        for r in tops:
            votes[r] += share
    opt = _optimal_set(dict(votes))
    return _resolve(opt, _add_scores(agents, restaurants)), opt


def strategy_app(agents: list, restaurants: list,
                 threshold: int = APP_APPROVAL_THRESHOLD
                 ) -> tuple[str | None, list[str]]:
    """APP (Approval Voting): most ratings >= threshold; ADD then lexicographic tiebreak."""
    scores = {
        r: sum(1 for a in agents if a["history"].get(r, 0) >= threshold)
        for r in restaurants
    }
    opt = _optimal_set(scores)
    return _resolve(opt, _add_scores(agents, restaurants)), opt


def strategy_fai(agents: list, restaurants: list) -> tuple[str | None, list[str]]:
    """FAI (Fairness / Round-Robin): agents take turns claiming their top unclaimed
    item; most-claimed wins (ADD tiebreak). Differs from MAJ when agents share a top
    pick, since later agents are forced onto their next-best items."""
    if not agents or not restaurants:
        return None, []

    agents_snapshot = list(agents)          # original order = stable tiebreak
    agents_remaining = list(agents_snapshot)
    cumulative_sat: dict[str, float] = {a["name"]: 0.0 for a in agents_snapshot}
    pick_counts: dict[str, int] = defaultdict(int)
    remaining = list(restaurants)

    # Each agent, in ascending-satisfaction (then original) order, claims its top
    # remaining item.
    while agents_remaining and remaining:
        ordered = sorted(
            agents_remaining,
            key=lambda a: (cumulative_sat[a["name"]], agents_snapshot.index(a))
        )
        picker = ordered[0]
        agents_remaining.remove(picker)

        best = max(remaining, key=lambda r: picker["history"].get(r, 0))
        pick_counts[best] += 1
        cumulative_sat[picker["name"]] += picker["history"].get(best, 0)
        remaining.remove(best)

    if not pick_counts:
        return None, []

    max_picks = max(pick_counts.values())
    candidates = [r for r, cnt in pick_counts.items() if cnt == max_picks]

    # Break the most-claimed set by ADD score (then lexicographic ID via _resolve).
    add_scores = {
        r: sum(a["history"].get(r, 0) for a in agents_snapshot)
        for r in candidates
    }
    opt = _optimal_set(add_scores)
    return _resolve(opt), opt


# ── Group evaluation ──────────────────────────────────────────────────────────

def evaluate_group(data: dict) -> dict:
    """Compute all structural metrics for one group-simulation JSON."""
    group_id    = data["group_id"]
    config      = data["group_config"]
    agents      = data["agents"]
    final_rec   = data["final_rec"]
    conversation = data.get("conversation", [])
    turn_prefs  = data.get("turn_preferences", {})
    vote_details = data.get("vote_details", [])
    restaurants = data.get("restaurant_options", [])
    turn_counter = data.get("turn_counter", 0)

    N = len(agents)
    consensus = (final_rec not in ("NO CONSENSUS REACHED", "No preference yet"))

    # ISS per agent, then GSS / MinSat over them.
    iss_map: dict[str, float] = {
        a["name"]: compute_iss(a["history"], final_rec)
        for a in agents
    }
    iss_vals = list(iss_map.values())
    gss = statistics.mean(iss_vals)
    min_sat = min(iss_vals)
    total_messages = len(conversation)

    # StabilityRate [descriptive]: agents whose round-1 preference == their final
    # one. Trivially 1.0 for single-round groups (flagged below).
    stable_count = 0
    for a in agents:
        prefs = turn_prefs.get(a["name"], [])
        if len(prefs) >= 1 and prefs[0] == prefs[-1]:
            stable_count += 1
    stability_rate = stable_count / N if N > 0 else 0.0
    single_round_group = (turn_counter <= 1)

    # SatVariance [descriptive]: within-group inequality of ISS.
    sat_variance = statistics.pvariance(iss_vals) if N > 1 else 0.0

    # MajMinGap: mean ISS of winning-vote agents minus that of the rest. Stays 0.0
    # when there is no minority side (unanimous) or no vote_details.
    maj_min_gap    = 0.0
    majority_voters: list[str] = []
    minority_voters: list[str] = []

    if vote_details and consensus:
        vote_buckets: dict[str, list[str]] = defaultdict(list)
        for vd in vote_details:
            vote_buckets[vd["vote"]].append(vd["agent"])

        winning_rec = max(vote_buckets, key=lambda r: len(vote_buckets[r]))
        majority_voters = vote_buckets[winning_rec]
        minority_voters = [
            vd["agent"] for vd in vote_details if vd["vote"] != winning_rec
        ]

        if minority_voters:
            maj_iss_vals = [iss_map[a] for a in majority_voters if a in iss_map]
            min_iss_vals = [iss_map[a] for a in minority_voters if a in iss_map]
            maj_mean = statistics.mean(maj_iss_vals) if maj_iss_vals else 0.0
            min_mean = statistics.mean(min_iss_vals) if min_iss_vals else 0.0
            maj_min_gap = maj_mean - min_mean

    # Strategy comparison. A strategy "matches" when the agreed restaurant is in its
    # optimal_set, keeping the match independent of arbitrary tie-breaks.
    strat = {
        "ADD": strategy_add(agents, restaurants),
        "LMS": strategy_lms(agents, restaurants),
        "MPL": strategy_mpl(agents, restaurants),
        "MAJ": strategy_maj(agents, restaurants),
        "APP": strategy_app(agents, restaurants),
        "FAI": strategy_fai(agents, restaurants),
    }
    strat_recs = {s: rec for s, (rec, _opt) in strat.items()}
    strat_sets = {s: opt for s, (_rec, opt) in strat.items()}
    strat_match = {
        s: bool(consensus and final_rec in strat_sets[s])
        for s in strat
    }
    strategies_matched = [s for s, matched in strat_match.items() if matched]

    # Hypothetical GSS each strategy's single recommendation would have produced.
    strat_gss = {
        s: (hypothetical_gss(agents, rec) if rec else 0.0)
        for s, rec in strat_recs.items()
    }

    best_strat_gss = max(strat_gss.values()) if strat_gss else 0.0
    conversation_vs_best = gss - best_strat_gss  # positive = conversation beat best strategy

    return {
        "group_id":           group_id,
        "group_config":       config,
        "n_agents":           N,
        "consensus_reached":  consensus,
        "final_rec":          final_rec,
        "turn_counter":       turn_counter,
        "total_messages":     total_messages,
        "single_round_group": single_round_group,
        "iss_per_agent": {k: round(v, 4) for k, v in iss_map.items()},
        "gss":          round(gss, 4),
        "min_sat":      round(min_sat, 4),
        "maj_min_gap":  round(maj_min_gap, 4),
        "sat_variance":    round(sat_variance, 4),
        "stability_rate":  round(stability_rate, 4),
        "majority_voters": majority_voters,
        "minority_voters": minority_voters,
        "strategy_recommendations":          strat_recs,
        "strategy_optimal_sets":             strat_sets,
        "strategy_match":                    strat_match,
        "strategies_matched":                strategies_matched,
        "n_strategies_matched":              len(strategies_matched),
        "strategy_gss":                      {s: round(v, 4) for s, v in strat_gss.items()},
        "conversation_vs_best_strategy_gss": round(conversation_vs_best, 4),
    }


# ── Output: JSON ──────────────────────────────────────────────────────────────

def save_json(results: list[dict], path: Path) -> None:
    """Write full per-group nested JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


# ── Output: CSV ───────────────────────────────────────────────────────────────

def save_csv(results: list[dict], path: Path) -> None:
    """Write one flat row per group; per-agent ISS becomes dynamic ISS_<name> columns."""
    scalar_cols = [
        "group_id", "group_config", "n_agents", "consensus_reached",
        "final_rec", "turn_counter", "total_messages", "single_round_group",
        "gss", "min_sat", "maj_min_gap",
        "sat_variance", "stability_rate",
        "n_strategies_matched", "conversation_vs_best_strategy_gss",
    ]
    strategy_cols = [
        "ADD_rec", "LMS_rec", "MPL_rec", "MAJ_rec", "APP_rec", "FAI_rec",
        "matches_ADD", "matches_LMS", "matches_MPL", "matches_MAJ",
        "matches_APP", "matches_FAI",
        "ADD_gss", "LMS_gss", "MPL_gss", "MAJ_gss", "APP_gss", "FAI_gss",
    ]

    # One ISS_<name> column per agent name seen anywhere in the dataset.
    all_agents = sorted({
        name
        for r in results
        for name in r["iss_per_agent"]
    })
    iss_cols = [f"ISS_{name}" for name in all_agents]

    fieldnames = scalar_cols + strategy_cols + iss_cols

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for r in results:
            row: dict = {col: r.get(col, "") for col in scalar_cols}

            for s in ("ADD", "LMS", "MPL", "MAJ", "APP", "FAI"):
                row[f"{s}_rec"]     = r["strategy_recommendations"][s]
                row[f"matches_{s}"] = r["strategy_match"][s]
                row[f"{s}_gss"]     = r["strategy_gss"][s]

            # Blank for agents not present in this group.
            for name in all_agents:
                row[f"ISS_{name}"] = r["iss_per_agent"].get(name, "")

            writer.writerow(row)


# ── Output: printed summary ───────────────────────────────────────────────────

_LINE = "=" * 76
_DASH = "-" * 76

def _section(title: str) -> None:
    print(f"\n{_LINE}")
    print(f"  {title}")
    print(_LINE)


def print_summary(results: list[dict]) -> None:
    """Print a human-readable report grouped by configuration type."""
    config_order = ["uniform", "divergent", "coalitional", "minority"]

    _section(f"STRUCTURAL ANALYSIS RESULTS  -  {len(results)} groups")

    # Per-group detail.
    for r in sorted(results, key=lambda x: x["group_id"]):
        cfg   = r["group_config"].upper()
        grp   = r["group_id"]
        n     = r["n_agents"]
        turns = r["turn_counter"]
        msgs  = r["total_messages"]

        if r["consensus_reached"]:
            outcome = f"✓  {r['final_rec']}"
        else:
            outcome = "✗  NO CONSENSUS"

        print(f"\n  Group {grp:>3}  [{cfg:<12}]  {n} agents  {outcome}")
        print(f"  {'':5}Rounds: {turns:>2}  Messages: {msgs:>3}"
              + ("  [single-round: stability trivially 1.0]" if r["single_round_group"] else ""))

        print(f"  {'':5}GSS={r['gss']:.3f}  MinSat={r['min_sat']:.3f}  "
              f"MajMinGap={r['maj_min_gap']:.3f}")
        print(f"  {'':5}SatVariance={r['sat_variance']:.4f}  "
              f"Stability={r['stability_rate']:.2f}")

        iss_str = "  ".join(
            f"{name}={val:.2f}" for name, val in r["iss_per_agent"].items()
        )
        print(f"  {'':5}ISS: {iss_str}")

        matched = r["strategies_matched"] if r["strategies_matched"] else ["(none)"]
        strat_gss = r["strategy_gss"]
        delta = r["conversation_vs_best_strategy_gss"]
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        print(f"  {'':5}Strategy match: {', '.join(matched)}  "
              f"[ADD={strat_gss['ADD']:.3f} LMS={strat_gss['LMS']:.3f} "
              f"MPL={strat_gss['MPL']:.3f} MAJ={strat_gss['MAJ']:.3f} "
              f"APP={strat_gss['APP']:.3f} FAI={strat_gss['FAI']:.3f}]  "
              f"conv. vs. best: {delta_str}")

    # Aggregate by configuration.
    for cfg in config_order:
        group = [r for r in results if r["group_config"] == cfg]
        if not group:
            continue
        n_groups    = len(group)
        n_consensus = sum(1 for r in group if r["consensus_reached"])
        consensus_g = [r for r in group if r["consensus_reached"]]

        def avg(key: str) -> float:
            return statistics.mean(r[key] for r in group)

        print(f"\n  {cfg.upper():<12}  {n_groups} groups  "
              f"({n_consensus}/{n_groups} reached consensus)")
        print(f"    Avg GSS:          {avg('gss'):.3f}")
        print(f"    Avg MinSat:       {avg('min_sat'):.3f}")
        print(f"    Avg MajMinGap:    {avg('maj_min_gap'):.3f}")
        print(f"    Avg SatVariance:  {avg('sat_variance'):.4f}  [descriptive]")
        print(f"    Avg Stability:    {avg('stability_rate'):.3f}  [descriptive]")
        print(f"    Avg Turns:        {statistics.mean(r['turn_counter'] for r in group):.1f}")

        if consensus_g:
            print(f"    Strategy alignment (consensus groups only):")
            for s in ("ADD", "LMS", "MPL", "MAJ", "APP", "FAI"):
                n_match = sum(1 for r in consensus_g if r["strategy_match"][s])
                hyp_gss = statistics.mean(r["strategy_gss"][s] for r in consensus_g)
                print(f"      {s}: matched {n_match}/{len(consensus_g)}  "
                      f"(avg hypothetical GSS: {hyp_gss:.3f})")
            no_match = sum(1 for r in consensus_g if not r["strategies_matched"])
            print(f"    Groups matching NO strategy: {no_match}/{len(consensus_g)}")
            avg_delta = statistics.mean(
                r["conversation_vs_best_strategy_gss"] for r in consensus_g
            )
            sign = "+" if avg_delta >= 0 else ""
            print(f"    Avg conv. GSS vs. best strategy: {sign}{avg_delta:.3f}")

    # Overall summary.
    _section("OVERALL SUMMARY")

    consensus_all = [r for r in results if r["consensus_reached"]]
    all_iss = [v for r in results for v in r["iss_per_agent"].values()]

    print(f"  Total groups:               {len(results)}")
    print(f"  Consensus reached:          {len(consensus_all)}/{len(results)}")
    print(f"  Overall avg GSS:         {statistics.mean(r['gss'] for r in results):.3f}")
    print(f"  Overall avg MinSat:      {statistics.mean(r['min_sat'] for r in results):.3f}")
    print(f"  Overall avg MajMinGap:   {statistics.mean(r['maj_min_gap'] for r in results):.3f}")
    print(f"  Overall avg SatVariance: {statistics.mean(r['sat_variance'] for r in results):.4f}  [descriptive]")
    print(f"  Agents with ISS < 0.20:  {sum(1 for v in all_iss if v < 0.20)}")
    print(f"  Agents with ISS >= 0.80: {sum(1 for v in all_iss if v >= 0.80)}")

    if consensus_all:
        no_match_all = sum(1 for r in consensus_all if not r["strategies_matched"])
        print(f"  Consensus groups matching no formal strategy: "
              f"{no_match_all}/{len(consensus_all)}")
        avg_delta_all = statistics.mean(
            r["conversation_vs_best_strategy_gss"] for r in consensus_all
        )
        sign = "+" if avg_delta_all >= 0 else ""
        print(f"  Avg conv. GSS vs. best strategy (all consensus): {sign}{avg_delta_all:.3f}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_folder: str, out_folder: str | None = None) -> None:
    folder = Path(data_folder)
    if not folder.is_dir():
        print(f"Error: '{data_folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(folder.glob("group_simulation_*.json"))
    if not json_files:
        print(f"No JSON files found in '{data_folder}'.", file=sys.stderr)
        sys.exit(1)

    results = []
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        results.append(evaluate_group(data))

    # Write to out_folder if given (created if missing), else next to the input.
    if out_folder is not None:
        out_dir = Path(out_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = folder.parent
    csv_path  = out_dir / "results.csv"
    json_path = out_dir / "results.json"

    save_csv(results, csv_path)
    save_json(results, json_path)

    print(f"CSV  saved → {csv_path}")
    print(f"JSON saved → {json_path}")



if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python evaluation_framework.py <folder_with_json_files> [output_folder]")
        print("Example: python evaluation_framework.py data/full_dataset/ data/results/")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) == 3 else None
    main(sys.argv[1], out)

#!/usr/bin/env python3
"""
evaluation_framework.py - Structural Analysis Module
================
Computes 7 deterministic metrics per group:
  1.  ISS          - Individual Satisfaction Score  (per agent)
  2.  GSS          - Group Satisfaction Score       (mean ISS)
  3.  MinSat       - Minimum Satisfaction           (worst-off agent)
  4.  StabilityRate - fraction of agents who kept initial preference  [descriptive]
  5.  SatVariance  - inequality in outcomes (population variance of ISS)  [descriptive]
  6.  MajMinGap    - ISS gap between majority and minority voters
  7.  StrategyComparison - ADD / LMS / MPL / MAJ / APP / FAI alignment with actual outcome

Aggregation strategies implemented (Barile et al. 2024):
  ADD - Additive Utilitarian: highest sum of ratings
  LMS - Least Misery: highest minimum rating
  MPL - Most Pleasure: highest individual rating
  MAJ - Majority: top choice of most agents (fractional vote on ties); ADD tiebreak
  APP - Approval Voting: most ratings >= APP_APPROVAL_THRESHOLD (5/10)
  FAI - Fairness (round-robin): each agent picks top item in turn; most-picked wins.
        Agents start at equal satisfaction (no prior history). FAI and MAJ diverge
        when agents share the same top preference: FAI forces later agents to claim
        their next-best unclaimed item, potentially producing a different winner.

ISS convention for no-consensus groups
---------------------------------------
When final_rec == "NO CONSENSUS REACHED" or "No preference yet", ISS is set to 0.0 for every agent.

This is a deliberate design choice: a failed conversation produces no benefit for
any participant. These zero values are computed and stored for every group, so a
no-consensus group CAN be carried into an analysis as a worst-case outcome when
that is wanted; the `consensus_reached` flag lets each downstream script decide.
What the downstream scripts actually do: the formal hypothesis tests
(hypothesis_tests.py) and the descriptive statistics (reproduce_stats.py) are run
on consensus groups ONLY -- metrics such as the welfare gap and the
majority-minority gap are only defined for a conversation that produced an
outcome. The thesis therefore reports H1, H2 and H4 as conditional on consensus,
and analyses the consensus rate itself as a separate result.
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

# Approval Voting threshold: a rating >= this value counts as an "approval."
# Set to 5 (the midpoint of the 1–10 scale), matching the observed mean
# rating across the dataset (~4.78) and representing a natural "acceptable" cutoff.
APP_APPROVAL_THRESHOLD = 5


# ──────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def compute_iss(agent_history: dict, final_rec: str) -> float:
    """
    Individual Satisfaction Score.
    ISS = agent's rating of final_rec / agent's maximum rating.
    Returns 0.0 if no consensus was reached or restaurant is unknown.
    """
    if final_rec in ("NO CONSENSUS REACHED", "No preference yet"):
        return 0.0
    if final_rec not in agent_history:
        return 0.0
    max_rating = max(agent_history.values(), default=0)
    if max_rating == 0:
        return 0.0
    return agent_history[final_rec] / max_rating


def hypothetical_iss(agent_history: dict, restaurant: str) -> float:
    """ISS for a hypothetical restaurant choice (used in strategy comparison)."""
    max_rating = max(agent_history.values(), default=0)
    if max_rating == 0:
        return 0.0
    return agent_history.get(restaurant, 0) / max_rating


def hypothetical_gss(agents: list, restaurant: str) -> float:
    """Mean ISS across all agents if the given restaurant were chosen."""
    vals = [hypothetical_iss(a["history"], restaurant) for a in agents]
    return statistics.mean(vals) if vals else 0.0


# ── Aggregation strategy recommendations ──────────────────────────────────────
#
# Each strategy scores the restaurants on its own criterion and returns a
# (recommendation, optimal_set) pair:
#   * optimal_set     every restaurant tied for the best score on that
#                     criterion. A strategy "matches" the conversation when the
#                     agreed restaurant lies in this set.
#   * recommendation  the single restaurant reported for the strategy (the
#                     *_rec columns) - the strategy's declared tie-break applied
#                     to the optimal set. The final, universal tie-break is
#                     lexicographic restaurant ID: an arbitrary but
#                     deterministic rule that, unlike list-insertion order, is
#                     independent of the restaurant ordering and therefore does
#                     not bias the strategy-match statistics.


def _optimal_set(scores: dict) -> list[str]:
    """Every restaurant tied for the highest score, sorted by restaurant ID."""
    if not scores:
        return []
    best = max(scores.values())
    return sorted(r for r, v in scores.items() if v == best)


def _resolve(candidates: list[str], *tiebreaks: dict) -> str | None:
    """
    Reduce tied candidates to a single restaurant.

    Each map in `tiebreaks` is applied in order, narrowing the pool to the
    candidates that maximise it. Anything still tied afterwards is resolved by
    lexicographic restaurant ID (declared, deterministic, order-independent).
    """
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
    """
    ADD (Additive Utilitarian): restaurant with the highest sum of all agents'
    ratings. Ties broken by lexicographic restaurant ID.
    Returns (recommendation, optimal_set).
    """
    scores = _add_scores(agents, restaurants)
    opt = _optimal_set(scores)
    return _resolve(opt), opt


def strategy_lms(agents: list, restaurants: list) -> tuple[str | None, list[str]]:
    """
    LMS (Least Misery): restaurant where the minimum individual rating is
    highest. Ties broken by lexicographic restaurant ID.
    Returns (recommendation, optimal_set).
    """
    scores = {
        r: min(a["history"].get(r, 0) for a in agents)
        for r in restaurants
    }
    opt = _optimal_set(scores)
    return _resolve(opt), opt


def strategy_mpl(agents: list, restaurants: list) -> tuple[str | None, list[str]]:
    """
    MPL (Most Pleasure): restaurant with the highest single rating from any
    agent. Ties broken by lexicographic restaurant ID.
    Returns (recommendation, optimal_set).
    """
    scores = {
        r: max(a["history"].get(r, 0) for a in agents)
        for r in restaurants
    }
    opt = _optimal_set(scores)
    return _resolve(opt), opt


def strategy_maj(agents: list, restaurants: list) -> tuple[str | None, list[str]]:
    """
    MAJ (Majority): restaurant that is the top preference of the most agents.

    Each agent casts one vote for its top-rated restaurant. An agent with
    several joint-favourite restaurants splits its vote equally among them
    (fractional voting, kept exact with Fraction), so the outcome never depends
    on the insertion order of an agent's rating history. Ties on the vote total
    are broken first by ADD score, then by lexicographic restaurant ID.
    Returns (recommendation, optimal_set).
    """
    votes: dict = defaultdict(Fraction)
    for a in agents:
        hist = a["history"]
        if not hist:
            continue
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
    """
    APP (Approval Voting): restaurant with the most ratings >= threshold.
    Ties broken first by ADD score, then by lexicographic restaurant ID.
    Threshold defaults to APP_APPROVAL_THRESHOLD (5 / 10-point scale).
    Returns (recommendation, optimal_set).
    """
    scores = {
        r: sum(1 for a in agents if a["history"].get(r, 0) >= threshold)
        for r in restaurants
    }
    opt = _optimal_set(scores)
    return _resolve(opt, _add_scores(agents, restaurants)), opt


def strategy_fai(agents: list, restaurants: list) -> tuple[str | None, list[str]]:
    """
    FAI (Fairness / Round-Robin): items are ranked by how individuals choose
    them in turn.

    Each agent (ordered by ascending cumulative satisfaction, ties broken by
    original JSON position) claims their top-rated unclaimed item. The item
    claimed by the most agents wins; ADD tiebreak on ties.

    FAI differs from MAJ when agents share the same top preference: MAJ gives
    that item a plurality win immediately, while FAI assigns it to the first
    agent in the rotation and forces the remaining agents to claim their
    next-best unclaimed items. If the ADD score of one of those next-best items
    exceeds the ADD score of the jointly preferred item, FAI and MAJ will
    recommend different restaurants.

    Among the most-claimed items, FAI's optimal_set is those tied for the
    highest ADD score; the recommendation is its lexicographically smallest
    member. Returns (recommendation, optimal_set).
    """
    if not agents or not restaurants:
        return None, []

    agents_snapshot = list(agents)          # preserve original order for stable tiebreak
    agents_remaining = list(agents_snapshot)
    cumulative_sat: dict[str, float] = {a["name"]: 0.0 for a in agents_snapshot}
    pick_counts: dict[str, int] = defaultdict(int)
    remaining = list(restaurants)

    # One pass: each agent (in ascending-satisfaction, original-index order) picks
    # their top-rated unclaimed item.
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

    # FAI resolves the most-claimed items by ADD score: the optimal set is
    # every such item tied for the highest ADD score, and the recommendation
    # is its lexicographically smallest member.
    add_scores = {
        r: sum(a["history"].get(r, 0) for a in agents_snapshot)
        for r in candidates
    }
    opt = _optimal_set(add_scores)
    return _resolve(opt), opt


# ──────────────────────────────────────────────────────────────────────────────
# GROUP EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_group(data: dict) -> dict:
    """
    Compute all structural metrics for one group simulation.

    Parameters
    ----------
    data : dict
        Parsed JSON for one group simulation file.

    Returns
    -------
    dict
        All computed metrics, ready for CSV / JSON export.
    """
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

    # ── 1. ISS per agent ──────────────────────────────────────────────────────
    iss_map: dict[str, float] = {
        a["name"]: compute_iss(a["history"], final_rec)
        for a in agents
    }
    iss_vals = list(iss_map.values())

    # ── 2. GSS ────────────────────────────────────────────────────────────────
    gss = statistics.mean(iss_vals)

    # ── 3. MinSat ─────────────────────────────────────────────────────────────
    min_sat = min(iss_vals)

    # ── 4. Total messages (for output / descriptive use) ─────────────────────
    total_messages = len(conversation)

    # ── 5. StabilityRate [descriptive] ───────────────────────────────────────
    # An agent is "stable" if their round-1 preference equals their final preference.
    # For single-round groups the value is trivially 1.0; this is noted in the output.
    stable_count = 0
    for a in agents:
        prefs = turn_prefs.get(a["name"], [])
        if len(prefs) >= 1 and prefs[0] == prefs[-1]:
            stable_count += 1
    stability_rate = stable_count / N if N > 0 else 0.0
    single_round_group = (turn_counter <= 1)

    # ── 6. SatVariance [descriptive] ─────────────────────────────────────────
    sat_variance = statistics.pvariance(iss_vals) if N > 1 else 0.0

    # ── 7. MajMinGap ──────────────────────────────────────────────────────────
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

    # ── 9. Strategy Comparison ────────────────────────────────────────────────
    # Each strategy yields a single recommendation (the *_rec columns) and an
    # optimal set - every restaurant tied for the best score on that strategy's
    # criterion. A strategy "matches" the conversation when the agreed
    # restaurant lies in that optimal set, i.e. the outcome is something the
    # strategy regards as optimal; this keeps the match independent of how an
    # arbitrary tie among equally-optimal restaurants is broken.
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

    # Hypothetical GSS under each strategy's single recommendation
    strat_gss = {
        s: (hypothetical_gss(agents, rec) if rec else 0.0)
        for s, rec in strat_recs.items()
    }

    # How does the actual outcome compare to the best available strategy?
    best_strat_gss = max(strat_gss.values()) if strat_gss else 0.0
    conversation_vs_best = gss - best_strat_gss  # positive = conversation beat best strategy

    # ── Package results ───────────────────────────────────────────────────────
    return {
        # Identification
        "group_id":           group_id,
        "group_config":       config,
        "n_agents":           N,
        "consensus_reached":  consensus,
        "final_rec":          final_rec,
        "turn_counter":       turn_counter,
        "total_messages":     total_messages,
        "single_round_group": single_round_group,
        # Per-agent ISS (not flattened here; handled separately in CSV export)
        "iss_per_agent": {k: round(v, 4) for k, v in iss_map.items()},
        # Primary metrics
        "gss":          round(gss, 4),
        "min_sat":      round(min_sat, 4),
        "maj_min_gap":  round(maj_min_gap, 4),
        # Descriptive metrics
        "sat_variance":    round(sat_variance, 4),
        "stability_rate":  round(stability_rate, 4),
        # Voting breakdown
        "majority_voters": majority_voters,
        "minority_voters": minority_voters,
        # Strategy comparison
        "strategy_recommendations":          strat_recs,
        "strategy_optimal_sets":             strat_sets,
        "strategy_match":                    strat_match,
        "strategies_matched":                strategies_matched,
        "n_strategies_matched":              len(strategies_matched),
        "strategy_gss":                      {s: round(v, 4) for s, v in strat_gss.items()},
        "conversation_vs_best_strategy_gss": round(conversation_vs_best, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT: JSON
# ──────────────────────────────────────────────────────────────────────────────

def save_json(results: list[dict], path: Path) -> None:
    """Write full per-group nested JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT: CSV
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(results: list[dict], path: Path) -> None:
    """
    Write flat CSV - one row per group.
    Per-agent ISS values are added as dynamic columns: ISS_<AgentName>.
    """
    # Scalar columns in desired order
    scalar_cols = [
        "group_id", "group_config", "n_agents", "consensus_reached",
        "final_rec", "turn_counter", "total_messages", "single_round_group",
        # Primary metrics
        "gss", "min_sat", "maj_min_gap",
        # Descriptive metrics
        "sat_variance", "stability_rate",
        # Strategy comparison
        "n_strategies_matched", "conversation_vs_best_strategy_gss",
    ]
    strategy_cols = [
        "ADD_rec", "LMS_rec", "MPL_rec", "MAJ_rec", "APP_rec", "FAI_rec",
        "matches_ADD", "matches_LMS", "matches_MPL", "matches_MAJ",
        "matches_APP", "matches_FAI",
        "ADD_gss", "LMS_gss", "MPL_gss", "MAJ_gss", "APP_gss", "FAI_gss",
    ]

    # Collect all agent names that appear across all groups (for ISS columns)
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

            # Strategy columns
            for s in ("ADD", "LMS", "MPL", "MAJ", "APP", "FAI"):
                row[f"{s}_rec"]     = r["strategy_recommendations"][s]
                row[f"matches_{s}"] = r["strategy_match"][s]
                row[f"{s}_gss"]     = r["strategy_gss"][s]

            # Per-agent ISS columns (blank for agents not in this group)
            for name in all_agents:
                row[f"ISS_{name}"] = r["iss_per_agent"].get(name, "")

            writer.writerow(row)


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT: PRINTED SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

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

    # ── Per-group detail ──────────────────────────────────────────────────────
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

        # ISS breakdown per agent
        iss_str = "  ".join(
            f"{name}={val:.2f}" for name, val in r["iss_per_agent"].items()
        )
        print(f"  {'':5}ISS: {iss_str}")

        # Strategy comparison
        matched = r["strategies_matched"] if r["strategies_matched"] else ["(none)"]
        strat_gss = r["strategy_gss"]
        delta = r["conversation_vs_best_strategy_gss"]
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        print(f"  {'':5}Strategy match: {', '.join(matched)}  "
              f"[ADD={strat_gss['ADD']:.3f} LMS={strat_gss['LMS']:.3f} "
              f"MPL={strat_gss['MPL']:.3f} MAJ={strat_gss['MAJ']:.3f} "
              f"APP={strat_gss['APP']:.3f} FAI={strat_gss['FAI']:.3f}]  "
              f"conv. vs. best: {delta_str}")

    # ── Aggregate by configuration ────────────────────────────────────────

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

    # ── Overall summary ────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(data_folder: str, out_folder: str | None = None) -> None:
    folder = Path(data_folder)
    if not folder.is_dir():
        print(f"Error: '{data_folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(folder.glob("group_simulation_*.json"))
    if not json_files:
        print(f"No JSON files found in '{data_folder}'.", file=sys.stderr)
        sys.exit(1)

    # Load and evaluate all groups
    results = []
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        results.append(evaluate_group(data))

    # Determine output directory:
    #   - if out_folder is given, write there (created if missing)
    #   - else default to the parent of the input folder (legacy behaviour)
    if out_folder is not None:
        out_dir = Path(out_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = folder.parent
    csv_path  = out_dir / "results.csv"
    json_path = out_dir / "results.json"

    # Write outputs
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
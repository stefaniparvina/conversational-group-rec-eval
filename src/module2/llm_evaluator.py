# =============================================================================
# Module 2: LLM Evaluator — Process-Level Evaluation of Group Recommendation Dialogs
# =============================================================================
# Evaluates the PROCESS of each conversation using an LLM judge.
# Four metrics:
#   1. Mention Rate        - Was each agent preference acknowledged? With what sentiment?
#   2. Justified Shifts    - When agents changed preference, was a reason given?
#   3. Repetition Index    - How many times did an agent repeat before being heard?
#   4. Process Quality Score - Rubric-based overall score (0-12, normalised 0-1)
#
# Usage:
#   python llm_evaluator.py --api anthropic --mode test    # 20 random H3 groups
#   python llm_evaluator.py --api openai    --mode test
#   python llm_evaluator.py --api anthropic --mode h3      # full H3 subset (~1376 groups)
#   python llm_evaluator.py --api anthropic --mode all     # all 8,000 groups
#
# Output:
#   llm_results.jsonl          - one JSON object per group, appended incrementally
#   llm_results_summary.csv    - flat summary per group (aggregated metrics)
#
# Requirements:
#   pip install anthropic openai pandas tqdm
#
# API keys: set environment variables ANTHROPIC_API_KEY or OPENAI_API_KEY
# =============================================================================

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR    = Path("full_dataset")
RESULTS_CSV = Path("results.csv")
OUT_JSONL   = Path("llm_results.jsonl")
OUT_CSV     = Path("llm_results_summary.csv")

TEST_N      = 20      # groups to evaluate in test mode
H3_MINSAT   = 0.20   # MinSat threshold for H3 subset
H3_MMGAP    = 0.30   # MajMinGap threshold for H3 subset
MAX_RETRIES = 3
RETRY_DELAY = 10     # seconds between retries


# ── Conversation formatters ───────────────────────────────────────────────────

def format_transcript(data: dict) -> str:
    lines = []
    for i, msg in enumerate(data["conversation"], 1):
        lines.append(f"[{i}] {msg['agent']}: {msg['message']}")
    return "\n".join(lines)


def format_preference_history(data: dict) -> str:
    tp = data.get("turn_preferences", {})
    lines = []
    for agent, prefs in tp.items():
        lines.append(f"  {agent}: " + " -> ".join(prefs))
    return "\n".join(lines)


def format_agent_profiles(data: dict) -> str:
    lines = []
    for ag in data["agents"]:
        h = ag.get("history", {})
        if h:
            top3 = sorted(h.items(), key=lambda x: -x[1])[:3]
            top3_str = ", ".join(f"{r}({v})" for r, v in top3)
            lines.append(f"  {ag['name']}: top choices = {top3_str}")
    return "\n".join(lines)


# ── Prompt construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert evaluator of group decision-making conversations. "
    "You analyse multi-turn dialogs where agents negotiate a joint restaurant choice "
    "and assess whether the process was fair — specifically whether each agent's "
    "preferences were genuinely heard, whether preference changes were explained, "
    "and whether the negotiation respected all participants.\n\n"
    "You always respond with valid JSON matching the exact schema provided. "
    "Do not add any commentary or text outside the JSON object."
)

OUTPUT_SCHEMA = """{
  "mention_rate": [
    {
      "agent": "<agent name>",
      "initial_preference": "<restaurant they started with>",
      "mentions": [
        {"by_agent": "<name of agent who mentioned it>", "sentiment": "positive|neutral|dismissive"}
      ],
      "total_mentions": 0,
      "acknowledged": true
    }
  ],
  "justified_shifts": [
    {
      "agent": "<agent name>",
      "from_preference": "<restaurant>",
      "to_preference": "<restaurant>",
      "round": 1,
      "shift_type": "quality-based|social|unexplained",
      "evidence": "<short quote from the conversation supporting the classification>"
    }
  ],
  "repetition_index": [
    {
      "agent": "<agent name>",
      "times_repeated_before_ack": 0
    }
  ],
  "process_quality": {
    "preferences_heard": 0,
    "shifts_justified": 0,
    "mutual_respect": 0,
    "logical_support": 0,
    "total_raw": 0,
    "score": 0.0,
    "reasoning": "<one sentence per dimension, separated by | >"
  }
}"""


def build_prompt(data: dict) -> str:
    transcript     = format_transcript(data)
    pref_history   = format_preference_history(data)
    agent_profiles = format_agent_profiles(data)
    final_rec      = data.get("final_rec", "UNKNOWN")
    group_config   = data.get("group_config", "unknown")
    n_agents       = len(data.get("agents", []))

    return f"""Evaluate the following group recommendation conversation.

## Group context
- Configuration: {group_config} ({n_agents} agents)
- Final recommendation: {final_rec}

## Agent preference histories (top choices, rating 1-10)
{agent_profiles}

## Preference evolution per round (each agent's suggestion per round)
{pref_history}

## Full conversation transcript
{transcript}

---

## Your task

Analyse this conversation and return a single JSON object with exactly these four keys:

### 1. mention_rate
For EACH agent, identify every time another agent explicitly mentioned or responded to that
agent's preferred restaurant (their round-1 suggestion). One entry per agent.
Classify each mention's sentiment:
- "positive": supportive, considering it seriously, or in agreement
- "neutral": acknowledged without positive/negative evaluation
- "dismissive": rejected, mocked, minimised, or actively argued against
Set "acknowledged" to true if the preference received at least one non-dismissive mention.

### 2. justified_shifts
For EACH agent who changed their preference between rounds (visible in the preference
evolution table above), create one entry per shift. Classify the reason:
- "quality-based": agent cited specific merits/drawbacks of the restaurants
- "social": agent shifted due to group pressure, solidarity, or to avoid conflict
- "unexplained": no reason given; agent simply changed position
Include a short supporting quote from the conversation as evidence.
If no agent shifted preference, return an empty list.

### 3. repetition_index
For EACH agent, count how many times they had to restate their initial preference
before another agent substantively acknowledged it.
0 = acknowledged on first mention. If never acknowledged, count total restatements.

### 4. process_quality
Score the overall conversation on four dimensions. Each dimension is scored 0-3:
  0 = absent/very poor  1 = weak  2 = adequate  3 = strong

- preferences_heard (0-3): Were ALL agents' preferred options discussed by others —
  not just mentioned but actually considered with reasons?
- shifts_justified (0-3): Were preference changes accompanied by reasons?
  (3 = all shifts explained; 0 = no shifts explained or no shifts at all and no discussion)
- mutual_respect (0-3): Was the tone respectful throughout, without dismissiveness
  or personal attacks?
- logical_support (0-3): Were recommendations backed by specific arguments about
  the restaurants (ratings, features, comparisons)?

Compute:
  total_raw = sum of four scores (max 12)
  score = round(total_raw / 12, 2)

In "reasoning", write one sentence explaining each dimension score, separated by " | ".

## Return ONLY the following JSON structure filled with your analysis:

{OUTPUT_SCHEMA}"""


# ── API callers ───────────────────────────────────────────────────────────────

# def call_anthropic(prompt: str) -> str:
#     import anthropic
#     client = anthropic.Anthropic()
#     resp = client.messages.create(
#         model="claude-sonnet-4-6",
#         max_tokens=2048,
#         temperature=0,
#         system=SYSTEM_PROMPT,
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return resp.content[0].text


def call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ]
    )
    return resp.choices[0].message.content


def call_llm(prompt: str, api: str) -> dict:
    """Call the chosen API with retries. Returns parsed JSON dict."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # raw = call_anthropic(prompt) if api == "anthropic" else call_openai(prompt)
            raw = call_openai(prompt)

            raw = raw.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            return json.loads(raw)
        except Exception as e:
            print(f"    Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"All {MAX_RETRIES} attempts failed")


# ── Results flattener ─────────────────────────────────────────────────────────

def flatten_result(group_id: int, data: dict, llm_out: dict) -> dict:
    """Produce one flat row per group for the summary CSV."""
    row = {
        "group_id":    group_id,
        "group_config": data["group_config"],
        "n_agents":    len(data["agents"]),
        "final_rec":   data.get("final_rec"),
    }

    # Mention Rate
    mr = llm_out.get("mention_rate", [])
    if mr:
        all_sentiments = [m["sentiment"] for x in mr for m in x.get("mentions", [])]
        row["mr_mean_mentions"]  = round(sum(x.get("total_mentions", 0) for x in mr) / len(mr), 3)
        row["mr_pct_acknowledged"] = round(sum(1 for x in mr if x.get("acknowledged")) / len(mr), 3)
        row["mr_pct_positive"]   = round(all_sentiments.count("positive")   / len(all_sentiments), 3) if all_sentiments else 0
        row["mr_pct_neutral"]    = round(all_sentiments.count("neutral")    / len(all_sentiments), 3) if all_sentiments else 0
        row["mr_pct_dismissive"] = round(all_sentiments.count("dismissive") / len(all_sentiments), 3) if all_sentiments else 0
    else:
        row["mr_mean_mentions"] = row["mr_pct_acknowledged"] = None
        row["mr_pct_positive"]  = row["mr_pct_neutral"] = row["mr_pct_dismissive"] = None

    # Justified Shifts
    js = llm_out.get("justified_shifts", [])
    row["js_n_shifts"] = len(js)
    if js:
        row["js_pct_quality"] = round(sum(1 for x in js if x.get("shift_type") == "quality-based") / len(js), 3)
        row["js_pct_social"]  = round(sum(1 for x in js if x.get("shift_type") == "social")        / len(js), 3)
        row["js_pct_unexplained"] = round(sum(1 for x in js if x.get("shift_type") == "unexplained") / len(js), 3)
    else:
        row["js_pct_quality"] = row["js_pct_social"] = row["js_pct_unexplained"] = None

    # Repetition Index
    ri = llm_out.get("repetition_index", [])
    row["ri_mean"] = round(sum(x.get("times_repeated_before_ack", 0) for x in ri) / len(ri), 3) if ri else None

    # Process Quality Score
    pq = llm_out.get("process_quality", {})
    row["pq_preferences_heard"] = pq.get("preferences_heard")
    row["pq_shifts_justified"]  = pq.get("shifts_justified")
    row["pq_mutual_respect"]    = pq.get("mutual_respect")
    row["pq_logical_support"]   = pq.get("logical_support")
    row["pq_total_raw"]         = pq.get("total_raw")
    row["pq_score"]             = pq.get("score")

    return row


# ── Group selection ───────────────────────────────────────────────────────────

def get_group_ids(mode: str) -> list:
    df = pd.read_csv(RESULTS_CSV)
    dc = df[df["consensus_reached"]]
    if mode in ("h3", "test"):
        subset = dc[(dc["min_sat"] < H3_MINSAT) & (dc["maj_min_gap"] > H3_MMGAP)]
        if mode == "test":
            subset = subset.sample(TEST_N, random_state=42)
    else:
        subset = dc
    return subset["group_id"].tolist()


def already_done() -> set:
    done = set()
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["group_id"])
            except Exception:
                pass
    return done


# ── Main ──────────────────────────────────────────────────────────────────────

def run(api: str, mode: str):
    group_ids = get_group_ids(mode)
    done      = already_done()
    todo      = [g for g in group_ids if g not in done]

    print(f"Mode: {mode} | API: {api}")
    print(f"Total in scope: {len(group_ids)} | Already done: {len(done)} | To process: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    rows = []
    for i, gid in enumerate(todo, 1):
        path = DATA_DIR / f"group_simulation_{gid}.json"
        if not path.exists():
            print(f"[{i}/{len(todo)}] SKIP {gid} — file not found")
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        cfg  = data["group_config"]
        n    = len(data["agents"])
        print(f"[{i}/{len(todo)}] group {gid} ({cfg}, {n} agents) ... ", end="", flush=True)

        try:
            prompt  = build_prompt(data)
            llm_out = call_llm(prompt, api)

            # Save raw output immediately (safe against crashes)
            record = {"group_id": gid, "group_config": cfg, "llm_output": llm_out}
            with open(OUT_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            flat = flatten_result(gid, data, llm_out)
            rows.append(flat)
            pq = llm_out.get("process_quality", {}).get("score", "?")
            print(f"OK  pq_score={pq}")

        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()

    # Write / append summary CSV
    if rows:
        new_df = pd.DataFrame(rows)
        if OUT_CSV.exists():
            existing = pd.read_csv(OUT_CSV)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset="group_id", keep="last").to_csv(OUT_CSV, index=False)
        else:
            new_df.to_csv(OUT_CSV, index=False)
        print(f"\nSaved {len(rows)} new rows -> {OUT_CSV}")
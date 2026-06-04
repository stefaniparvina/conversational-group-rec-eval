# =============================================================================
# Module 2: LLM Evaluator -- Process-Level Evaluation of Group Recommendation Dialogs
# =============================================================================
# Evaluates the PROCESS of each conversation with an LLM judge (OpenAI GPT-4o).
#
# Four metrics:
#   1. Mention Rate          - Was each agent's preference acknowledged, and with
#                              what sentiment?
#   2. Justified Shifts      - When agents changed preference, was a reason given?
#   3. Repetition Index      - How many times did an agent repeat before being heard?
#   4. Process Quality Score - Rubric score over four dimensions (0-3 each). The
#                              total (0-12) and normalised score (0-1) are computed
#                              in Python, NOT by the model.
#
# Judge model: gpt-4o-2024-11-20 -- a pinned, dated snapshot (not the moving
# "gpt-4o" alias), called with temperature=0 and a fixed seed for reproducibility.
# The model returns its answer via OpenAI Structured Outputs (a strict JSON
# schema), so the response shape is guaranteed.
#
# Paths are resolved relative to this script's location, so it runs from anywhere:
#   src/module2/llm_evaluator.py          <- this script
#   data/full_dataset/                    <- input  (8,000 group_simulation_*.json)
#   data/results/results.csv              <- input  (used to select the H3 subset)
#   data/results/llm_results.jsonl        <- output (one raw record per group)
#   data/results/llm_results_summary.csv  <- output (flat metrics per group)
#   NOTE: --mode validate writes instead to data/validation/ (the files
#   llm_results_validation.jsonl and llm_results_summary_validation.csv).
#
# Usage (run from anywhere inside the project):
#   python src/module2/llm_evaluator.py --mode test      # 20 random H3 groups (smoke test)
#   python src/module2/llm_evaluator.py --mode validate  # the 15 human-annotated transcripts
#   python src/module2/llm_evaluator.py --mode h3        # full H3 subset (the analysis run)
#
# Requirements:  pip install openai pandas  (no python-dotenv needed)
# API key:       put OPENAI_API_KEY=sk-... in a .env file at the project root
#                (or set the environment variable directly).
# =============================================================================

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# -- Configuration -------------------------------------------------------------
# Paths are anchored to this script's location (src/module2/), two levels below
# the project root, so the script works regardless of the current directory.
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_DIR       = PROJECT_ROOT / "data" / "full_dataset"
RESULTS_CSV    = PROJECT_ROOT / "data" / "results" / "results.csv"
VALIDATION_CSV = PROJECT_ROOT / "data" / "validation" / "validation_set.csv"
OUT_JSONL      = PROJECT_ROOT / "data" / "results" / "llm_results.jsonl"
OUT_CSV        = PROJECT_ROOT / "data" / "results" / "llm_results_summary.csv"

# Judge model -- pinned dated snapshot for reproducibility.
MODEL = "gpt-4o-2024-11-20"
SEED  = 42           # fixed seed: with temperature=0, maximises run-to-run determinism

TEST_N      = 20     # groups to evaluate in test mode
H3_MINSAT   = 0.20   # MinSat threshold for the H3 subset
H3_MMGAP    = 0.30   # MajMinGap threshold for the H3 subset
MAX_RETRIES = 3
RETRY_DELAY = 10     # seconds between retries


# -- API key loading -----------------------------------------------------------
# The OpenAI client reads the key from the OPENAI_API_KEY environment variable.
# To make a .env file work too, this loads KEY=VALUE lines from .env (checked at
# the project root, then src/module2/) into the environment. A real environment
# variable always takes precedence. No python-dotenv dependency needed.

def _load_env_file():
    for env_path in (PROJECT_ROOT / ".env", SCRIPT_DIR / ".env"):
        if not env_path.exists():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
        break


_load_env_file()


# -- Conversation formatters ---------------------------------------------------

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


def detect_shifts(data: dict) -> list:
    """Find every preference change from the turn_preferences table.

    turn_preferences[agent] is the agent's suggested option per round; a change
    between consecutive rounds is a shift. This is the SAME detection rule the
    human annotation workbook uses (build_annotation_workbook.py), so the judge
    classifies exactly the same set of shifts the human annotator scored."""
    shifts = []
    for agent, prefs in data.get("turn_preferences", {}).items():
        for i in range(1, len(prefs)):
            if prefs[i] != prefs[i - 1]:
                shifts.append({
                    "agent": agent,
                    "from_preference": prefs[i - 1],
                    "to_preference": prefs[i],
                    "round": i + 1,          # prefs[0] is round 1
                })
    return shifts


def format_detected_shifts(shifts: list) -> str:
    """Render the detected shift list for the prompt, one numbered line each."""
    if not shifts:
        return "(no preference shifts occurred in this conversation)"
    items = []
    for i, s in enumerate(shifts, 1):
        items.append(f"  {i}. {s['agent']}: {s['from_preference']} -> "
                     f"{s['to_preference']} (round {s['round']})")
    return chr(10).join(items)


# -- Output schema (OpenAI Structured Outputs) ---------------------------------
# A strict JSON schema. The API guarantees the model's response matches this
# exactly, so no markdown-stripping or shape-guessing is needed.
# Note: in process_quality, each dimension's *reasoning* field is placed before
# its *score* field -- the model reasons first, then scores (chain-of-thought).

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["mention_rate", "justified_shifts", "repetition_index", "process_quality"],
    "properties": {
        "mention_rate": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["agent", "initial_preference", "mentions",
                             "total_mentions", "acknowledged"],
                "properties": {
                    "agent": {"type": "string"},
                    "initial_preference": {"type": "string"},
                    "mentions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["by_agent", "sentiment"],
                            "properties": {
                                "by_agent": {"type": "string"},
                                "sentiment": {
                                    "type": "string",
                                    "enum": ["positive", "neutral", "dismissive"],
                                },
                            },
                        },
                    },
                    "total_mentions": {"type": "integer"},
                    "acknowledged": {"type": "boolean"},
                },
            },
        },
        "justified_shifts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["agent", "from_preference", "to_preference",
                             "round", "shift_type", "evidence"],
                "properties": {
                    "agent": {"type": "string"},
                    "from_preference": {"type": "string"},
                    "to_preference": {"type": "string"},
                    "round": {"type": "integer"},
                    "shift_type": {
                        "type": "string",
                        "enum": ["quality-based", "social", "unexplained"],
                    },
                    "evidence": {"type": "string"},
                },
            },
        },
        "repetition_index": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["agent", "times_repeated_before_ack"],
                "properties": {
                    "agent": {"type": "string"},
                    "times_repeated_before_ack": {"type": "integer"},
                },
            },
        },
        "process_quality": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "preferences_heard_reasoning", "preferences_heard_score",
                "shifts_justified_reasoning", "shifts_justified_score",
                "mutual_respect_reasoning", "mutual_respect_score",
                "logical_support_reasoning", "logical_support_score",
            ],
            "properties": {
                "preferences_heard_reasoning": {"type": "string"},
                "preferences_heard_score": {"type": "integer"},
                "shifts_justified_reasoning": {"type": "string"},
                "shifts_justified_score": {"type": "integer"},
                "mutual_respect_reasoning": {"type": "string"},
                "mutual_respect_score": {"type": "integer"},
                "logical_support_reasoning": {"type": "string"},
                "logical_support_score": {"type": "integer"},
            },
        },
    },
}

# The four process-quality dimension scores, summed in Python (not by the model).
PQ_DIMENSIONS = [
    "preferences_heard_score",
    "shifts_justified_score",
    "mutual_respect_score",
    "logical_support_score",
]


# -- Prompt construction -------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert evaluator of group decision-making conversations. "
    "You analyse multi-turn dialogs where agents negotiate a joint restaurant choice "
    "and assess whether the process was fair -- specifically whether each agent's "
    "preferences were genuinely heard, whether preference changes were explained, "
    "and whether the negotiation respected all participants. "
    "Base every judgment only on what is present in the transcript."
)


def build_prompt(data: dict) -> str:
    transcript     = format_transcript(data)
    pref_history   = format_preference_history(data)
    agent_profiles = format_agent_profiles(data)
    detected_shifts = detect_shifts(data)
    shift_block    = format_detected_shifts(detected_shifts)
    n_shifts       = len(detected_shifts)
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

Analyse this conversation and fill in the four required fields.

### 1. mention_rate
For EACH agent, identify every time another agent explicitly mentioned or responded
to that agent's preferred restaurant (their round-1 suggestion). One entry per agent.
Classify each mention's sentiment:
- "positive": supportive, considering it seriously, or in agreement
- "neutral": acknowledged without positive/negative evaluation
- "dismissive": rejected, mocked, minimised, or actively argued against
Set "acknowledged" to true if the preference received at least one non-dismissive mention.

### 2. justified_shifts
The preference shifts in this conversation have ALREADY been detected from the
preference evolution table and are listed below. Do NOT search for additional
shifts, and do NOT add, drop, split, or merge any. Your ONLY task is to classify
WHY each listed shift happened.

Return EXACTLY one entry per listed shift ({n_shifts} in total). For each entry,
copy the agent, from_preference, to_preference, and round EXACTLY as given in the
list below, then add two fields:
- "shift_type":
   - "quality-based": the agent cited a specific merit or drawback of a
     restaurant (food, atmosphere, rating, price, location) as the reason.
   - "social": the agent changed because of the group -- majority momentum,
     solidarity, peer pressure, or to avoid or end conflict -- not the venue
     itself.
   - "unexplained": the agent gave no genuine reason, only vented, or the reason
     cannot be determined from the transcript.
- "evidence": a short supporting quote from the turn where the agent makes or
  explains the change.

Detected shifts to classify:
{shift_block}

If the list shows no shifts occurred, return an empty justified_shifts list.

### 3. repetition_index
For EACH agent, count how many times they had to restate their initial preference
before another agent substantively acknowledged it.
0 = acknowledged on first mention. If never acknowledged, count total restatements.

### 4. process_quality
Score the overall conversation on four INDEPENDENT dimensions, each on a 0-3 scale
with the concrete anchors below.

SCORING DISCIPLINE -- read this before scoring:
- A score is EARNED by specific evidence; it is not given by default. Do not park
  scores at 2 or 3 just because nothing obviously terrible happened.
- Most real group conversations are flawed in at least one respect. A 3 is reserved
  for a dimension whose top anchor is met with NO exceptions: a single agent left
  out, a single unexplained shift, or a single dismissive remark is enough to
  prevent a 3.
- If you cannot point to specific transcript evidence for a level, you have not
  observed it. Absence of evidence is scored at the LOWER anchor, never assumed.
- The four dimensions are independent. A conversation can be perfectly respectful
  (high mutual_respect) yet still ignore an agent's preference (low
  preferences_heard). Score each dimension only on its own evidence.

HOW TO SCORE each dimension:
1. In the "<dimension>_reasoning" field, FIRST cite the specific things you
   observed (name the agents and turns involved).
2. Then start from 3 and work DOWNWARD: check whether the 3-anchor holds
   completely. At the first specific failure, drop to the anchor that matches what
   you actually saw.
3. Finish the reasoning by stating explicitly why the score is NOT one level
   higher than the one you chose.
4. Then put the integer in the matching "<dimension>_score" field.
Always reason before you score. If a conversation falls between two anchors, choose
the lower one.

**preferences_heard** -- Were all agents' preferred options taken up and discussed
by the other agents?
- 0 = One or more agents' preferences were never engaged with -- ignored entirely,
      or named in passing with no response.
- 1 = Preferences were named but barely discussed; at least one agent's preference
      received no substantive engagement from anyone else.
- 2 = Every agent's preference was acknowledged and most were discussed with at
      least a brief reason, but engagement was uneven across agents.
- 3 = Every agent's preference was explicitly discussed by other agents with
      substantive reasons (merits or drawbacks weighed); no agent was left out.

**shifts_justified** -- When agents changed their stated preference, was the change
explained?
- 0 = At least one agent changed preference with no stated reason at all.
- 1 = Shifts happened but were only weakly justified (e.g. "fine, okay" or bare
      capitulation with no content).
- 2 = Most shifts came with a clear, content-based reason; a minority were
      unexplained or purely social.
- 3 = Every preference change was accompanied by an explicit, content-based reason
      (a concrete merit or drawback of an option).
  (If NO agent changed preference, see the Edge cases section below -- do not
  default to 0 or 3.)

**mutual_respect** -- Was the tone respectful toward agents and their preferences
throughout?
- 0 = Personal attacks, mocking, or repeated dismissiveness toward agents or their
      preferences.
- 1 = Generally civil but with clear dismissiveness -- a preference brushed off
      without consideration, or a curt rejection.
- 2 = Respectful throughout; disagreement was expressed politely; minor curtness
      at most.
- 3 = Consistently respectful; disagreement was paired with acknowledgement of the
      other agent's view.

**logical_support** -- Were positions backed by specific arguments about the
restaurants?
- 0 = Choices asserted with no reasons (just "I want X" / "let's do Y").
- 1 = Reasons given but vague or generic ("it's nice", "sounds good"), with no
      restaurant-specific content.
- 2 = Most claims backed by at least one concrete restaurant feature (rating,
      cuisine, price, location).
- 3 = Claims consistently backed by specific, comparative arguments that weigh
      features of more than one option.

### 4b. Worked calibration example (illustrative -- NOT one of the transcripts above)
Mini-conversation: Ana suggests Spot A ("I just like it"). Ben suggests Spot B
("B has great reviews and is cheaper than A"). Cleo suggests Spot C. Ana and Ben
debate A versus B for several turns; nobody ever responds to Cleo's Spot C. Ben
then says "okay, fine, A works" with no further reason. The tone stays polite the
whole time.
Correct scoring:
- preferences_heard = 1: Ana's and Ben's options were discussed, but Cleo's Spot C
  got no engagement from anyone -- one agent left out, so not 2 and not 3.
- shifts_justified = 1: Ben's shift to A was a bare capitulation ("okay, fine")
  with no content.
- mutual_respect = 3: the tone was polite throughout; no preference was mocked or
  rudely brushed off.
- logical_support = 1: only Ben gave a concrete reason (reviews, price); the rest
  was vague ("I just like it") -- mostly generic, not option-specific.
This shows the dimensions are scored independently on the evidence, and that 1 (and
0) are valid scores when the evidence supports them -- not only 2 and 3.

### 5. Edge cases & ambiguity
Apply these rules so that ambiguous transcripts are scored consistently:
- **No preference shifts:** if no agent changed preference, return an empty
  justified_shifts list. For the shifts_justified dimension, do not score 0 or 3
  by default -- instead judge how the group settled: score 2 if the agents
  genuinely weighed the options and converged on reasoned grounds, lower if they
  converged with little or no discussion. State this in the reasoning.
- **A preference never acknowledged:** if no other agent ever engages with an
  agent's preference, set acknowledged=false and total_mentions=0 for that agent;
  treat this as "not heard" and let it lower the preferences_heard score.
- **Ambiguous sentiment:** if a mention is neither clearly supportive nor clearly
  dismissive, classify it as "neutral".
- **Ambiguous shift reason:** if a shift could be quality-based or social and the
  transcript does not make the reason clear, classify it as "unexplained" and note
  the ambiguity in the evidence field.
- **Assigned personality is not disrespect:** agents are simulated with assigned
  personalities and tones (e.g. blunt, assertive). A blunt style is not by itself
  a respect violation -- judge mutual_respect by how agents treat each other's
  preferences, not by personality.
- **Short conversations:** score only what is actually present. Do not infer
  unobserved respect, reasoning, or engagement. Absence of evidence is scored at
  the lower anchor, never assumed positive.
- Base every judgment only on the transcript text above.

Do NOT compute any totals -- provide only the four reasoning fields and the four
0-3 scores. The total and normalised score are calculated separately."""


# -- API caller ----------------------------------------------------------------

def call_openai(client, prompt: str):
    """Single call to the OpenAI judge. Returns the raw response object."""
    return client.chat.completions.create(
        model=MODEL,
        temperature=0,
        seed=SEED,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "process_evaluation",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )


def call_llm(client, prompt: str):
    """Call the judge with retries. Returns (parsed_output, metadata)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp   = call_openai(client, prompt)
            choice = resp.choices[0]

            if choice.finish_reason == "length":
                raise RuntimeError("response truncated (finish_reason='length')")
            if getattr(choice.message, "refusal", None):
                raise RuntimeError(f"model refused: {choice.message.refusal}")

            parsed = json.loads(choice.message.content)

            usage = resp.usage
            meta = {
                "model": resp.model,
                "system_fingerprint": getattr(resp, "system_fingerprint", None),
                "finish_reason": choice.finish_reason,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                } if usage else None,
            }
            return parsed, meta
        except Exception as e:
            print(f"    Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"All {MAX_RETRIES} attempts failed")


# -- Score finalisation --------------------------------------------------------

def finalize_scores(llm_out: dict) -> dict:
    """Compute total_raw and the normalised score in Python.

    The model supplies only the four 0-3 dimension scores; the arithmetic is done
    here so a model mistake cannot corrupt the results. Scores are clamped to 0-3.
    """
    pq = llm_out.get("process_quality", {})
    scores = []
    for dim in PQ_DIMENSIONS:
        try:
            v = int(pq.get(dim, 0))
        except (TypeError, ValueError):
            v = 0
        v = max(0, min(3, v))
        pq[dim] = v
        scores.append(v)
    pq["total_raw"] = sum(scores)                  # 0-12
    pq["score"]     = round(sum(scores) / 12, 4)   # 0-1
    llm_out["process_quality"] = pq
    return llm_out


def sanitize_counts(llm_out: dict) -> dict:
    """Force the model's integer count fields to be non-negative integers.

    Structured Outputs guarantee these fields are integers but cannot bound
    them -- the strict-schema format supports no minimum/maximum -- so a stray
    negative or non-integer value is corrected here, the same defensive step
    finalize_scores applies to the 0-3 process-quality scores.
    """
    def nonneg_int(v):
        try:
            return max(0, int(v))
        except (TypeError, ValueError):
            return 0
    for e in llm_out.get("mention_rate", []):
        if "total_mentions" in e:
            e["total_mentions"] = nonneg_int(e.get("total_mentions"))
    for e in llm_out.get("repetition_index", []):
        if "times_repeated_before_ack" in e:
            e["times_repeated_before_ack"] = nonneg_int(
                e.get("times_repeated_before_ack"))
    return llm_out


# -- Structural audit ----------------------------------------------------------

def audit_structure(data: dict, llm_out: dict) -> dict:
    """Verify the judge returned complete, correctly-identified output.

    Structured Outputs guarantee the SHAPE of the response but not its
    CARDINALITY or IDENTITY: the model could omit an agent, misspell a name,
    or return a shift that was not in the detected list. This audit checks,
    per group, that mention_rate and repetition_index each carry exactly one
    entry per real agent, and that justified_shifts matches the pre-detected
    shift set exactly. It never raises -- it returns a flag plus a list of
    human-readable issues, which are stored in the JSONL record so flagged
    groups can be reviewed after the run.
    """
    real_set = {a["name"] for a in data.get("agents", [])}
    issues = []

    for field in ("mention_rate", "repetition_index"):
        names = [e.get("agent") for e in llm_out.get(field, [])]
        got = set(names)
        missing = real_set - got
        extra = got - real_set
        if missing:
            issues.append(f"{field}: missing agent(s) {sorted(missing)}")
        if extra:
            issues.append(f"{field}: unknown agent name(s) {sorted(extra)}")
        if len(names) != len(got):
            dups = sorted({n for n in names if names.count(n) > 1})
            issues.append(f"{field}: duplicate entr(ies) for {dups}")

    # Per-entry consistency checks on the mention_rate records.
    turn_prefs = data.get("turn_preferences", {})
    for e in llm_out.get("mention_rate", []):
        ag = e.get("agent")
        mentions = e.get("mentions", [])
        init_pref = e.get("initial_preference")
        total = e.get("total_mentions")
        rounds = turn_prefs.get(ag) or []
        real_rounds = [r for r in rounds if r != "No preference yet"]
        if real_rounds and init_pref != real_rounds[0]:
            issues.append(f"mention_rate: {ag} initial_preference "
                          f"{init_pref!r} != first stated preference "
                          f"{real_rounds[0]!r}")
        if total != len(mentions):
            issues.append(f"mention_rate: {ag} total_mentions {total} != "
                          f"mentions list length {len(mentions)}")
        unknown = sorted({m.get("by_agent") for m in mentions
                          if m.get("by_agent") not in real_set})
        if unknown:
            issues.append(f"mention_rate: {ag} has mention(s) by unknown "
                          f"agent(s) {unknown}")
        # "acknowledged" = engaged with by ANOTHER agent. Self-mentions (the
        # judge sometimes logs the opening declaration) are excluded here.
        other = [m for m in mentions if m.get("by_agent") != ag]
        expected_ack = any(m.get("sentiment") != "dismissive" for m in other)
        if bool(e.get("acknowledged")) != expected_ack:
            issues.append(f"mention_rate: {ag} acknowledged="
                          f"{bool(e.get('acknowledged'))} but its mentions imply "
                          f"{expected_ack}")

    detected_list = detect_shifts(data)
    judged_list = llm_out.get("justified_shifts", [])
    if len(judged_list) != len(detected_list):
        issues.append(f"justified_shifts: returned {len(judged_list)} entr(ies) "
                      f"for {len(detected_list)} detected shift(s)")
    detected = {(s["agent"], s["from_preference"], s["to_preference"],
                 s["round"]) for s in detected_list}
    judged = {(s.get("agent"), s.get("from_preference"),
               s.get("to_preference"), s.get("round"))
              for s in judged_list}
    if detected - judged:
        issues.append(f"justified_shifts: {len(detected - judged)} detected "
                      f"shift(s) missing or altered")
    if judged - detected:
        issues.append(f"justified_shifts: {len(judged - detected)} extra "
                      f"shift(s) not in the detected list")

    # Cross-field rubric check: an agent with zero mentions cannot be reconciled
    # with a preferences-heard score of 2 or 3 (those anchors require every
    # preference to have been engaged with).
    zero_mention = sorted(
        e.get("agent") for e in llm_out.get("mention_rate", [])
        if not any(m.get("by_agent") != e.get("agent")
                   for m in e.get("mentions", [])))
    ph = llm_out.get("process_quality", {}).get("preferences_heard_score")
    if zero_mention and isinstance(ph, int) and ph >= 2:
        issues.append(f"process_quality: preferences_heard_score={ph} but "
                      f"agent(s) {zero_mention} received zero mentions")

    return {"structural_ok": not issues, "issues": issues}


# -- Results flattener ---------------------------------------------------------

def flatten_result(group_id: int, data: dict, llm_out: dict) -> dict:
    """Produce one flat row per group for the summary CSV."""
    row = {
        "group_id":     group_id,
        "group_config": data["group_config"],
        "n_agents":     len(data["agents"]),
        "final_rec":    data.get("final_rec"),
    }

    # Mention Rate
    mr = llm_out.get("mention_rate", [])
    if mr:
        all_sentiments = [m["sentiment"] for x in mr for m in x.get("mentions", [])]
        row["mr_mean_mentions"]    = round(sum(x.get("total_mentions", 0) for x in mr) / len(mr), 3)
        row["mr_pct_acknowledged"] = round(sum(1 for x in mr if x.get("acknowledged")) / len(mr), 3)
        row["mr_pct_positive"]     = round(all_sentiments.count("positive")   / len(all_sentiments), 3) if all_sentiments else 0
        row["mr_pct_neutral"]      = round(all_sentiments.count("neutral")    / len(all_sentiments), 3) if all_sentiments else 0
        row["mr_pct_dismissive"]   = round(all_sentiments.count("dismissive") / len(all_sentiments), 3) if all_sentiments else 0
    else:
        row["mr_mean_mentions"] = row["mr_pct_acknowledged"] = None
        row["mr_pct_positive"]  = row["mr_pct_neutral"] = row["mr_pct_dismissive"] = None

    # Justified Shifts
    js = llm_out.get("justified_shifts", [])
    row["js_n_shifts"] = len(js)
    if js:
        row["js_pct_quality"]     = round(sum(1 for x in js if x.get("shift_type") == "quality-based") / len(js), 3)
        row["js_pct_social"]      = round(sum(1 for x in js if x.get("shift_type") == "social")        / len(js), 3)
        row["js_pct_unexplained"] = round(sum(1 for x in js if x.get("shift_type") == "unexplained")   / len(js), 3)
    else:
        row["js_pct_quality"] = row["js_pct_social"] = row["js_pct_unexplained"] = None

    # Repetition Index
    ri = llm_out.get("repetition_index", [])
    row["ri_mean"] = round(sum(x.get("times_repeated_before_ack", 0) for x in ri) / len(ri), 3) if ri else None

    # Process Quality (scores finalised in Python by finalize_scores)
    pq = llm_out.get("process_quality", {})
    row["pq_preferences_heard"] = pq.get("preferences_heard_score")
    row["pq_shifts_justified"]  = pq.get("shifts_justified_score")
    row["pq_mutual_respect"]    = pq.get("mutual_respect_score")
    row["pq_logical_support"]   = pq.get("logical_support_score")
    row["pq_total_raw"]         = pq.get("total_raw")
    row["pq_score"]             = pq.get("score")

    return row


# -- Group selection -----------------------------------------------------------

def get_group_ids(mode: str) -> list:
    """Return the group IDs to evaluate.

    validate -> exactly the 15 hand-annotated transcripts (validation_set.csv),
                used to measure judge-vs-human agreement before the main run.
    test     -> a fixed random sample of TEST_N groups from the H3 subset.
    h3       -> the full H3 subset (consensus reached, min_sat < threshold,
                maj_min_gap > threshold) -- the analysis run.
    """
    if mode == "validate":
        if not VALIDATION_CSV.exists():
            raise FileNotFoundError(
                f"{VALIDATION_CSV} not found -- run select_validation_set.py first.")
        return pd.read_csv(VALIDATION_CSV)["group_id"].tolist()

    df = pd.read_csv(RESULTS_CSV)
    dc = df[df["consensus_reached"]]
    subset = dc[(dc["min_sat"] < H3_MINSAT) & (dc["maj_min_gap"] > H3_MMGAP)]
    if mode == "test":
        subset = subset.sample(min(TEST_N, len(subset)), random_state=42)
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


def rebuild_summary_csv() -> None:
    """Rebuild the summary CSV from scratch from the full JSONL.

    The JSONL is the crash-safe source of truth -- one line per group, written
    the moment that group is judged. Rebuilding the summary CSV from it at the
    end of every run, rather than appending newly-judged rows to the old CSV,
    keeps the CSV complete and correct even when an earlier run was interrupted
    and resumed, and means a missing or corrupt previous CSV can never abort
    the run.
    """
    if not OUT_JSONL.exists():
        return
    summary_rows = []
    for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            gid = int(rec["group_id"])
        except Exception:
            continue
        path = DATA_DIR / f"group_simulation_{gid}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        summary_rows.append(flatten_result(gid, data, rec.get("llm_output", {})))
    if summary_rows:
        df = pd.DataFrame(summary_rows).drop_duplicates(
            subset="group_id", keep="last").sort_values("group_id")
        try:
            df.to_csv(OUT_CSV, index=False)
            print(f"Summary CSV rebuilt from JSONL: {len(df)} rows -> {OUT_CSV}")
        except OSError as exc:
            print(f"WARNING: could not write the summary CSV ({exc}). The raw "
                  f"results in {OUT_JSONL} are complete and safe -- close the "
                  f"file if it is open, then re-run to rebuild the CSV.")


# -- Main ----------------------------------------------------------------------

def run(mode: str):
    # Validation output is kept separate from the main H3 results so the two
    # runs never collide: validate -> data/validation/, test/h3 -> data/results/.
    global OUT_JSONL, OUT_CSV
    if mode == "validate":
        OUT_JSONL = PROJECT_ROOT / "data" / "validation" / "llm_results_validation.jsonl"
        OUT_CSV   = PROJECT_ROOT / "data" / "validation" / "llm_results_summary_validation.csv"
    else:
        OUT_JSONL = PROJECT_ROOT / "data" / "results" / "llm_results.jsonl"
        OUT_CSV   = PROJECT_ROOT / "data" / "results" / "llm_results_summary.csv"

    group_ids = get_group_ids(mode)
    done      = already_done()
    todo      = [g for g in group_ids if g not in done]

    print(f"Mode: {mode} | Judge model: {MODEL}")
    print(f"Total in scope: {len(group_ids)} | Already done: {len(done)} | To process: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        rebuild_summary_csv()   # keep the summary CSV in sync even with no new work
        return

    # Work remains -- only now are the OpenAI client and an API key
    # needed, so a no-work run (e.g. a CSV-only rebuild) needs no key.
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Put it in a .env file at the project "
            "root as a line:  OPENAI_API_KEY=sk-...  (or set it as a real "
            "environment variable).")
    client = OpenAI()      # reads OPENAI_API_KEY from the environment (or .env)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    flagged = []
    total_tokens = 0

    for i, gid in enumerate(todo, 1):
        path = DATA_DIR / f"group_simulation_{gid}.json"
        if not path.exists():
            print(f"[{i}/{len(todo)}] SKIP {gid} -- file not found")
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cfg  = data["group_config"]
            n    = len(data["agents"])
            print(f"[{i}/{len(todo)}] group {gid} ({cfg}, {n} agents) ... ", end="", flush=True)
            prompt        = build_prompt(data)
            llm_out, meta = call_llm(client, prompt)
            llm_out       = finalize_scores(llm_out)
            llm_out       = sanitize_counts(llm_out)
            audit         = audit_structure(data, llm_out)

            # Save the raw record immediately (safe against crashes)
            record = {
                "group_id":           gid,
                "group_config":       cfg,
                "timestamp":          datetime.now(timezone.utc).isoformat(),
                "model":              meta["model"],
                "seed":               SEED,
                "system_fingerprint": meta["system_fingerprint"],
                "finish_reason":      meta["finish_reason"],
                "usage":              meta["usage"],
                "structural_audit":   audit,
                "llm_output":         llm_out,
            }
            with open(OUT_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            if meta["usage"]:
                total_tokens += meta["usage"]["total_tokens"]

            if audit["structural_ok"]:
                print(f"OK  pq_score={llm_out['process_quality']['score']}")
            else:
                flagged.append((gid, audit["issues"]))
                print(f"OK  pq_score={llm_out['process_quality']['score']}"
                      f"  [!] STRUCTURAL FLAG: {'; '.join(audit['issues'])}")

        except Exception as e:
            print(f"ERROR group {gid}: {e}")
            traceback.print_exc()

    # Rebuild the summary CSV from the full JSONL: it is always complete, and
    # it never reads the previous CSV, so an interrupted or corrupt earlier
    # CSV from a prior run can never abort this one.
    rebuild_summary_csv()

    # Structural audit summary
    print("")
    if flagged:
        print(f"[!] STRUCTURAL AUDIT: {len(flagged)} of {len(todo)} group(s) "
              f"flagged for review:")
        for gid, issues in flagged:
            print(f"    group {gid}: {'; '.join(issues)}")
        print("    (recorded under 'structural_audit' in the JSONL; the run "
              "itself completed normally)")
    else:
        print("Structural audit: all processed groups complete and "
              "consistent -- no flags.")

    print(f"Total tokens used this run: {total_tokens:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 2 -- LLM judge (OpenAI GPT-4o) for group-recommendation dialogs."
    )
    parser.add_argument(
        "--mode",
        choices=["test", "validate", "h3"],
        default="test",
        help="test = 20 random H3 groups (smoke test); "
             "validate = the 15 hand-annotated transcripts; "
             "h3 = full H3 subset (the analysis run).",
    )
    args = parser.parse_args()
    run(args.mode)

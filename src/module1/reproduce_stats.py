import pandas as pd, numpy as np, json, glob

df = pd.read_csv("data/results/results.csv")
dc = df[df.consensus_reached]

# Section 2 -- Consensus and process
print("=== Section 2: Consensus ===")
print("Consensus rate:", df.consensus_reached.mean())
print(df.groupby("group_config").consensus_reached.mean())

# Section 3 -- Outcome quality
print("\n=== Section 3: Outcome quality ===")
metrics = ["gss","min_sat","sat_variance","stability_rate","maj_min_gap"]
print(dc.groupby("group_config")[metrics].mean().round(4))
print(dc.groupby("n_agents")[["gss","min_sat","sat_variance",
                              "n_strategies_matched"]].mean().round(3))

# Section 4 -- ISS distribution
print("\n=== Section 4: ISS distribution ===")
iss_cols = [c for c in df.columns if c.startswith("ISS_")]
m = (df.melt(id_vars=["group_id","group_config","consensus_reached"],
             value_vars=iss_cols, var_name="agent", value_name="iss")
       .dropna().query("consensus_reached"))
print(m["iss"].describe())
print(m.groupby("group_config")["iss"]
       .agg(["mean","median","std", lambda x:(x<0.2).mean()]))

# Section 5 -- Strategy comparison
print("\n=== Section 5: Strategy comparison ===")
strats = ["ADD","LMS","MPL","MAJ","APP","FAI"]
print(dc.groupby("group_config")[[f"matches_{s}" for s in strats]].mean())
print("No-strategy rate:", (dc["n_strategies_matched"]==0).mean())
print(dc.groupby("group_config").apply(
        lambda g:(g["n_strategies_matched"]==0).mean(), include_groups=False))
print(dc.groupby("group_config")["conversation_vs_best_strategy_gss"].mean())

# Section 6 -- Personality correlations (requires JSONs in full_dataset/)
print("\n=== Section 6: Personality correlations ===")
rows=[]
for fp in sorted(glob.glob("data/full_dataset/group_simulation_*.json")):
    d = json.load(open(fp, encoding="utf-8"))
    f = d.get("final_rec","")
    if f == "NO CONSENSUS REACHED": continue
    for ag in d["agents"]:
        h = ag.get("history",{})
        if not h or f not in h: continue
        iss = h[f]/max(h.values())
        p = ag.get("personality",{})
        rows.append({"config":d["group_config"],"iss":iss,
                     "tone":ag.get("tone",""), **p})
agdf = pd.DataFrame(rows)
for t in ["openness","conscientiousness","extraversion",
          "agreeableness","neuroticism"]:
    print(t, agdf[[t,"iss"]].corr().iloc[0,1])

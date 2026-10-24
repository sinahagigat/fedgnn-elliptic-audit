"""
Every number, table and figure in the manuscript is produced here from the
1,300 v7 run records (plus the earlier GPU runs, used only in the
reproduction section). Nothing in the manuscript is typed by hand.

Outputs: numbers.json (scalar values referenced as {{key}} in the text),
tables.json (table rows), figures/*.png
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCR))
from load_v7 import load_v7, load_v6a  # noqa: E402
from stats_v7 import wide, holm, wil, pairwise, vs_base, null_range, SEEDS  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"; OUT.mkdir(exist_ok=True)
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
N, T = {}, {}

df = load_v7(); v6 = load_v6a()
GCN8 = ['Centralized_GCN', 'FedAvg_GCN', 'FedProx_GCN', 'FedProto_GCN', 'FedGNN_Hard_GCN',
        'FedGNN_Soft_GCN', 'FedGNN_Soft_Virtual_GCN', 'FedGNN_Soft_Virtual_GAT']
CONV7 = GCN8[:-1]
FED6 = ['FedAvg_GCN', 'FedProto_GCN', 'FedGNN_Hard_GCN', 'FedGNN_Soft_GCN',
        'FedGNN_Soft_Virtual_GCN', 'FedGNN_Soft_Virtual_GAT']
SHORT = {'Centralized_GCN': 'Central-GCN', 'FedAvg_GCN': 'FedAvg', 'FedProx_GCN': 'FedProx',
         'FedProto_GCN': 'FedProto', 'FedGNN_Hard_GCN': 'Hard', 'FedGNN_Soft_GCN': 'Soft',
         'FedGNN_Soft_Virtual_GCN': 'Soft-VN', 'FedGNN_Soft_Virtual_GAT': 'Soft-VN-GAT',
         'Centralized_MLP': 'Central-MLP', 'FedAvg_MLP': 'FedAvg-MLP'}


def f4(x): return f"{x:.4f}"
def f3(x): return f"{x:.3f}"
def sgn4(x): return f"{x:+.4f}"
def pfmt(p):
    if p < 1e-4: return "<0.0001"
    if p < 0.001 or 0.04 <= p < 0.06: return f"{p:.4f}"
    return f"{p:.3f}"
def ci95(x):
    x = np.asarray(x); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h

N["n_runs_total"] = f"{len(df):,}"
N["n_runs_v6a"] = f"{len(v6):,}"
for e in ["A", "B", "D", "E"]:
    N[f"n_runs_{e}"] = str((df.exp == e).sum())
N["n_runs_C"] = str(df.exp.str.startswith("C").sum())
assert df.code_hash.nunique() == 1
N["code_hash"] = df.code_hash.iloc[0]
for e in sorted(df.exp.unique()):
    N[f"hash_{e}"] = df[df.exp == e].config_hash.iloc[0]

# ------------------------------------------------------------- reproduction
rep_rows = []; per_run = []
for n, o in [("A", "v6a_A"), ("B", "v6a_B"), ("C0", "v6a_C")]:
    a = df[df.exp == n].set_index(["variant", "seed"]).ap
    b = v6[v6.exp == o].set_index(["variant", "seed"]).ap
    j = pd.concat([a.rename("v7"), b.rename("v6a")], axis=1).dropna()
    d = j.v7 - j.v6a; per_run.append(d)
    m = j.groupby(level=0).mean()
    rep_rows.append([n, str(len(j)), f4(d.abs().median()), f4(d.abs().max()),
                     f4((m.v7 - m.v6a).abs().max()), f3(j.corr().iloc[0, 1])])
T["reproduction"] = {"header": ["Protocol", "Paired runs", "Median |ΔAP|", "Max |ΔAP|",
                                "Max |Δ mean|", "Pearson r"], "rows": rep_rows}
dall = pd.concat(per_run)
N["rep_median_absdiff"] = f4(dall.abs().median())
N["rep_n_pairs"] = str(len(dall))
v6A = wide(v6.assign(exp=v6.exp.str.replace("v6a_", "")), "A")
d_avg_prox = (v6A.FedAvg_GCN - v6A.FedProx_GCN).abs()
N["v6a_avgprox_max"] = f4(d_avg_prox.max()); N["v6a_avgprox_median"] = f"{d_avg_prox.median():.1e}"
sameround = sum(v6[(v6.exp == "v6a_A") & (v6.variant == "FedAvg_GCN")].set_index("seed").best_round.loc[SEEDS].values
                == v6[(v6.exp == "v6a_A") & (v6.variant == "FedProx_GCN")].set_index("seed").best_round.loc[SEEDS].values)
N["v6a_avgprox_diffround"] = str(20 - sameround)
wA = wide(df, "A")
N["v7_avgprox_max"] = f"{(wA.FedAvg_GCN - wA.FedProx_GCN).abs().max():.1g}"
# top-3 order in v6a vs v7
v6m = v6A[GCN8].mean().sort_values(ascending=False); v7m = wA[GCN8].mean().sort_values(ascending=False)
N["v6a_top"] = SHORT[v6m.index[0]]; N["v7_top"] = SHORT[v7m.index[0]]

# ------------------------------------------------------------- run A
A8 = wA[GCN8]
rowsA = []
for v in list(A8.mean().sort_values(ascending=False).index):
    x = A8[v]; lo, hi = ci95(x)
    conv = df[(df.exp == "A") & (df.variant == v)].converged.mean()
    rowsA.append([SHORT[v], f4(x.mean()), f4(x.std(ddof=1)), f"[{lo:.4f}, {hi:.4f}]", f4(x.min()), f4(x.max()), f"{int(conv*20)}/20"])
for v in ["Centralized_MLP", "FedAvg_MLP"]:
    x = wA[v]; lo, hi = ci95(x); conv = df[(df.exp == "A") & (df.variant == v)].converged.mean()
    rowsA.append([SHORT[v] + " (control)", f4(x.mean()), f4(x.std(ddof=1)), f"[{lo:.4f}, {hi:.4f}]", f4(x.min()), f4(x.max()), f"{int(conv*20)}/20"])
T["runA"] = {"header": ["Design", "Mean AP", "SD", "95% CI of mean", "Min", "Max", "Converged"], "rows": rowsA}
m7 = A8[CONV7].mean(); m8 = A8.mean()
N["A_spread7"] = f4(m7.max() - m7.min()); N["A_spread8"] = f4(m8.max() - m8.min())
sd7 = A8[CONV7].std(ddof=1)
N["A_sd7_min"] = f4(sd7.min()); N["A_sd7_max"] = f4(sd7.max())
N["A_se7_typ"] = f4((sd7 / np.sqrt(20)).mean())
N["A_gat_sd"] = f4(A8.FedGNN_Soft_Virtual_GAT.std(ddof=1)); N["A_gat_min"] = f4(A8.FedGNN_Soft_Virtual_GAT.min())
N["A_gat_max"] = f4(A8.FedGNN_Soft_Virtual_GAT.max())
N["A_gat_mean"] = f4(A8.FedGNN_Soft_Virtual_GAT.mean())
others_min = A8.drop(columns="FedGNN_Soft_Virtual_GAT").min().min(); N["A_others_min"] = f4(others_min)
N["A_gat_below_others"] = str(int((A8.FedGNN_Soft_Virtual_GAT < others_min).sum()))
ranks = A8.rank(axis=1, ascending=False)
N["A_gat_first"] = str(int((ranks.FedGNN_Soft_Virtual_GAT == 1).sum())); N["A_gat_last"] = str(int((ranks.FedGNN_Soft_Virtual_GAT == 8).sum()))
obs7, null7, p7 = null_range(A8[CONV7]); obs8, null8, p8 = null_range(A8)
N.update(A_null7_obs=f4(obs7), A_null7_exp=f4(null7), A_null7_p=f3(p7), A_null8_obs=f4(obs8), A_null8_exp=f4(null8), A_null8_p=f3(p8))
pwA = pairwise(A8)
N["A_pairs_raw"] = str(int((pwA.p < .05).sum())); N["A_pairs_holm"] = str(int((pwA.p_holm < .05).sum()))
sig = pwA[pwA.p_holm < .05]
N["A_sig_pairs"] = "; ".join(f"{SHORT[r.a]} vs {SHORT[r.b]} (ΔAP {r['diff']:+.4f}, Holm p={pfmt(r.p_holm)})" for _, r in sig.iterrows())
vbA = vs_base(A8).set_index("variant")
N["A_proto_diff"] = sgn4(vbA.loc["FedProto_GCN", "diff"]); N["A_proto_wins"] = str(int(vbA.loc["FedProto_GCN", "wins"]))
N["A_proto_pholm"] = pfmt(vbA.loc["FedProto_GCN", "p_holm"])
N["A_cent_minus_avg"] = sgn4(A8.Centralized_GCN.mean() - A8.FedAvg_GCN.mean())
N["A_cent_avg_p"] = pfmt(wil(A8.Centralized_GCN, A8.FedAvg_GCN))
N["A_cent_avg_wins"] = str(int((A8.Centralized_GCN > A8.FedAvg_GCN).sum()))
wins = []
for a, b in itertools.combinations(CONV7, 2):
    if {a, b} == {"FedAvg_GCN", "FedProx_GCN"}:
        continue  # numerically identical (see Section 4.1)
    hi, lo = (a, b) if A8[a].mean() > A8[b].mean() else (b, a); wins.append(int((A8[lo] > A8[hi]).sum()))
N["A_worse_wins_min"] = str(min(wins)); N["A_worse_wins_max"] = str(max(wins)); N["A_worse_wins_mean"] = f"{np.mean(wins):.1f}"

# seed subsampling
full = A8.mean().values; fullrank = stats.rankdata(-full); M = A8.values.T
sub_rows = []
for k in (3, 5, 6):
    first = np.zeros(8); exact = 0; rhos = []; tot = 0
    for c in itertools.combinations(range(20), k):
        mu = M[:, c].mean(1); first[np.argmax(mu)] += 1
        r = stats.rankdata(-mu); exact += int((r == fullrank).all()); rhos.append(np.corrcoef(r, fullrank)[0, 1]); tot += 1
    order = np.argsort(-first)
    sub_rows.append([f"{k} ({tot:,} subsets)",
                     ", ".join(f"{SHORT[GCN8[i]]} " + (f"{100*first[i]/tot:.1f}" if 100*first[i]/tot >= 0.05 else "<0.1") for i in order if first[i] > 0),
                     f"{np.mean(rhos):+.3f}", f"{100*exact/tot:.2f}%"])
    if k == 6:
        N["sub6_n_first"] = str(int((first > 0).sum())); N["sub6_leader"] = SHORT[GCN8[order[0]]]
        N["sub6_leader_pct"] = f"{100*first[order[0]]/tot:.1f}"; N["sub6_rho"] = f"{np.mean(rhos):+.3f}"
        N["sub6_exact"] = f"{100*exact/tot:.2f}"
T["subsample"] = {"header": ["Seeds per study", "Share of subsets in which each design ranks first (%)",
                             "Mean ρ with 20-seed order", "Exact order reproduced"], "rows": sub_rows}

# power / equivalence
sd_d = []; rho_p = []
for a, b in itertools.combinations(GCN8, 2):
    sd_d.append((A8[a] - A8[b]).std(ddof=1)); rho_p.append(np.corrcoef(A8[a], A8[b])[0, 1])
sdd = float(np.mean(sd_d))
N["sigma_d"] = f4(sdd); N["rho_pair"] = f"{np.mean(rho_p):+.3f}"
N["rho_pair_min"] = f"{min(rho_p):+.3f}"; N["rho_pair_max"] = f"{max(rho_p):+.3f}"
N["n_equiv_005"] = str(int(np.ceil((1.645 + 1.282) ** 2 * sdd ** 2 / 0.005 ** 2)))
N["n_equiv_010"] = str(int(np.ceil((1.645 + 1.282) ** 2 * sdd ** 2 / 0.010 ** 2)))
N["n_detect_003"] = str(int(np.ceil((1.96 + 0.8416) ** 2 * sdd ** 2 / 0.003 ** 2)))
N["n_detect_005"] = str(int(np.ceil((1.96 + 0.8416) ** 2 * sdd ** 2 / 0.005 ** 2)))

# ------------------------------------------------------------- partitions
prot = [("A", "Louvain, redrawn per seed"), ("B", "Random node, redrawn per seed")] + \
       [(f"C{k}", f"Louvain, fixed (seed {k})") for k in range(5)]
part_rows = []
for e, desc in prot:
    s = df[(df.exp == e) & df.retention.notna()]
    part_rows.append([e, desc, f"{s.retention.mean():.5f}", f"{1-s.retention.mean():.2%}", f"{s.label_skew.mean():.4f}"])
T["partitions"] = {"header": ["Protocol", "Partition", "Edge retention", "Severed arcs", "Label skew (SD of local positive rate)"],
                   "rows": part_rows}
sA = df[(df.exp == "A") & df.retention.notna()]; sB = df[(df.exp == "B") & df.retention.notna()]
N["ret_A"] = f"{sA.retention.mean():.5f}"; N["sev_A"] = f"{1-sA.retention.mean():.2%}"
N["sev_B"] = f"{1-sB.retention.mean():.2%}"; N["skew_A"] = f"{sA.label_skew.mean():.4f}"; N["skew_B"] = f"{sB.label_skew.mean():.4f}"

# ------------------------------------------------------------- fixed partitions
fp_rows = []; fp = {}
for k in range(5):
    w = wide(df, f"C{k}")[FED6 + ["FedProx_GCN"]]
    vb = vs_base(w).set_index("variant")
    fp[k] = vb
for v in FED6[1:]:
    row = [SHORT[v]]
    for k in range(5):
        r = fp[k].loc[v]; star = "*" if r.p_holm < 0.05 else ""
        row.append(f"{r['diff']:+.4f}{star} ({int(r.wins)}/20)")
    wa = wA[v] - wA.FedAvg_GCN; pa = holm([wil(wA[x], wA.FedAvg_GCN) for x in FED6[1:]])[FED6[1:].index(v)]
    row.append(f"{wa.mean():+.4f}{'*' if pa < 0.05 else ''} ({int((wA[v] > wA.FedAvg_GCN).sum())}/20)")
    fp_rows.append(row)
T["fixed"] = {"header": ["Design vs FedAvg", "C0", "C1", "C2", "C3", "C4", "A (redrawn)"], "rows": fp_rows}
N["C0_soft"] = sgn4(fp[0].loc["FedGNN_Soft_GCN", "diff"]); N["C1_soft"] = sgn4(fp[1].loc["FedGNN_Soft_GCN", "diff"])
N["C0_soft_wins"] = str(int(fp[0].loc["FedGNN_Soft_GCN", "wins"])); N["C1_soft_wins"] = str(int(fp[1].loc["FedGNN_Soft_GCN", "wins"]))
nsig = {k: int((fp[k].drop("FedProx_GCN").p_holm < .05).sum()) for k in range(5)}
N["C_nsig"] = ", ".join(f"C{k}: {nsig[k]}" for k in range(5))
pos_sig = sum(int(((fp[k].drop("FedProx_GCN").p_holm < .05) & (fp[k].drop("FedProx_GCN")["diff"] > 0)).sum()) for k in range(5))
neg_sig = sum(int(((fp[k].drop("FedProx_GCN").p_holm < .05) & (fp[k].drop("FedProx_GCN")["diff"] < 0)).sum()) for k in range(5))
N["C_pos_sig"] = str(pos_sig); N["C_neg_sig"] = str(neg_sig)
MC = pd.DataFrame({k: wide(df, f"C{k}")[FED6].mean() for k in range(5)})
rc = MC.corr(method="spearman"); N["C_rank_rho01"] = f"{rc.loc[0, 1]:+.2f}"
N["C_rank_rho_min"] = f"{rc.values[np.triu_indices(5, 1)].min():+.2f}"; N["C_rank_rho_max"] = f"{rc.values[np.triu_indices(5, 1)].max():+.2f}"
N["C_winners"] = ", ".join(f"C{k}: {SHORT[MC[k].idxmax()]}" for k in range(5))
proto_sig = sum(int(fp[k].loc["FedProto_GCN", "p_holm"] < .05) for k in range(5))
N["C_proto_sig"] = str(proto_sig)
N["C_proto_range"] = f"{min(fp[k].loc['FedProto_GCN','diff'] for k in range(5)):+.4f} to {max(fp[k].loc['FedProto_GCN','diff'] for k in range(5)):+.4f}"
vc = []
for v in FED6:
    W = np.array([wide(df, f"C{k}")[v].values for k in range(5)])
    within = W.var(axis=1, ddof=1).mean(); means = W.mean(1); between = max(means.var(ddof=1) - within / 20, 0)
    vc.append([SHORT[v], f4(np.sqrt(within)), f4(np.sqrt(between)), f"{means.min():.4f}–{means.max():.4f}", f4(wA[v].std(ddof=1))])
T["varcomp"] = {"header": ["Design", "Within-partition SD", "Between-partition SD", "Range of partition means", "SD under redrawn partitions (A)"], "rows": vc}
N["between_sd_avg"] = vc[0][2]; N["between_sd_soft"] = vc[3][2]
N["between_sd_struct_min"] = f4(min(float(r[2]) for r in vc[2:])); N["between_sd_struct_max"] = f4(max(float(r[2]) for r in vc[2:]))
N["between_sd_proto"] = vc[1][2]

# ------------------------------------------------------------- graph-free control
wE = wide(df, "E"); wD = wide(df, "D")
def cmp(x, y): return np.mean(x - y), wil(x, y), int((x > y).sum())
d, p, wn = cmp(wA.Centralized_MLP, wA.Centralized_GCN); N.update(mlpA_c_d=sgn4(d), mlpA_c_p=pfmt(p), mlpA_c_w=str(wn))
d, p, wn = cmp(wA.FedAvg_MLP, wA.FedAvg_GCN); N.update(mlpA_f_d=sgn4(d), mlpA_f_p=pfmt(p), mlpA_f_w=str(wn))
d, p, wn = cmp(wE.Centralized_MLP, wD.Centralized_GCN); N.update(mlpD_c_d=sgn4(d), mlpD_c_p=pfmt(p), mlpD_c_w=str(wn))
d, p, wn = cmp(wE.FedAvg_MLP, wD.FedAvg_GCN); N.update(mlpD_f_d=sgn4(d), mlpD_f_p=pfmt(p), mlpD_f_w=str(wn))
d, p, wn = cmp(wD.FedAvg_GCN, wD.Centralized_GCN); N.update(gapD_gcn_d=sgn4(d), gapD_gcn_p=pfmt(p), gapD_gcn_w=str(wn))
d, p, wn = cmp(wE.FedAvg_MLP, wE.Centralized_MLP); N.update(gapD_mlp_d=sgn4(d), gapD_mlp_p=pfmt(p), gapD_mlp_w=str(wn))
N["A_fedavg_mlp_mean"] = f4(wA.FedAvg_MLP.mean()); N["A_cent_mlp_mean"] = f4(wA.Centralized_MLP.mean())
N["A_cent_gcn_mean"] = f4(wA.Centralized_GCN.mean()); N["A_fedavg_gcn_mean"] = f4(wA.FedAvg_GCN.mean())

# ------------------------------------------------------------- B: evaluation protocol
wB = wide(df, "B"); lB = wide(df, "B", "loc_ap"); nvB = wide(df, "B", "locnv_ap")
lA = wide(df, "A", "loc_ap"); nvA = wide(df, "A", "locnv_ap")
b_rows = []
for v in list(lB[GCN8].mean().sort_values(ascending=False).index) + ["FedAvg_MLP", "Centralized_MLP"]:
    vn = f"{(lB[v]-nvB[v]).mean():+.4f}" if v in ("FedGNN_Soft_Virtual_GCN", "FedGNN_Soft_Virtual_GAT") else "–"
    b_rows.append([SHORT[v] + (" (control)" if "MLP" in v else ""), f4(wA[v].mean()), f4(wB[v].mean()), f4(lB[v].mean()),
                   f"{(lB[v]-wB[v]).mean():+.4f}", vn])
T["runB"] = {"header": ["Design", "A: global", "B: global inference", "B: per-client inference", "Per-client − global", "Virtual nodes on − off"],
             "rows": b_rows}
N["B_avg_global"] = f4(wB.FedAvg_GCN.mean()); N["B_avg_local"] = f4(lB.FedAvg_GCN.mean())
N["B_cent"] = f4(wB.Centralized_GCN.mean())
N["B_gap_global"] = f4(wB.Centralized_GCN.mean() - wB.FedAvg_GCN.mean())
N["B_gap_local"] = f4(lB.Centralized_GCN.mean() - lB.FedAvg_GCN.mean())
pv = holm([wil(lB[v], lB.FedAvg_GCN) for v in ['Centralized_GCN', 'FedProto_GCN', 'FedGNN_Hard_GCN', 'FedGNN_Soft_GCN', 'FedGNN_Soft_Virtual_GCN', 'FedGNN_Soft_Virtual_GAT', 'FedProx_GCN']])
N["B_gap_local_pholm"] = pfmt(pv[0]); N["B_gap_local_wins"] = str(int((lB.Centralized_GCN > lB.FedAvg_GCN).sum()))
N["B_gap_global_wins"] = str(int((wB.Centralized_GCN > wB.FedAvg_GCN).sum()))
N["B_gap_global_p"] = pfmt(wil(wB.Centralized_GCN, wB.FedAvg_GCN))
N["B_mlp_fed"] = f4(wB.FedAvg_MLP.mean())
N["B_mlp_vs_cent"] = sgn4(wB.FedAvg_MLP.mean() - wB.Centralized_GCN.mean())
N["B_fed_cost_global"] = f4(np.mean([wA[v].mean() - wB[v].mean() for v in FED6 + ["FedProx_GCN"]]))
N["B_fed_cost_local"] = f4(np.mean([wA[v].mean() - lB[v].mean() for v in FED6 + ["FedProx_GCN"]]))
for v, key in [("FedGNN_Soft_Virtual_GCN", "gcn"), ("FedGNN_Soft_Virtual_GAT", "gat")]:
    dd = lB[v] - nvB[v]; N[f"B_vn_{key}"] = sgn4(dd.mean()); N[f"B_vn_{key}_p"] = pfmt(wil(lB[v], nvB[v])); N[f"B_vn_{key}_w"] = str(int((dd > 0).sum()))
    N[f"A_vn_{key}_max"] = f"{(lA[v]-nvA[v]).abs().max():.1e}"
dd = lB.FedGNN_Soft_Virtual_GCN - lB.FedGNN_Soft_GCN; N["B_abl_local"] = sgn4(dd.mean()); N["B_abl_local_p"] = pfmt(wil(lB.FedGNN_Soft_Virtual_GCN, lB.FedGNN_Soft_GCN))
dd = wB.FedGNN_Soft_Virtual_GCN - wB.FedGNN_Soft_GCN; N["B_abl_global"] = sgn4(dd.mean()); N["B_abl_global_p"] = pfmt(wil(wB.FedGNN_Soft_Virtual_GCN, wB.FedGNN_Soft_GCN))
N["B_gat_local"] = f4(lB.FedGNN_Soft_Virtual_GAT.mean())
dd = lB.FedGNN_Soft_Virtual_GAT - lB.FedAvg_GCN; N["B_gat_vs_avg_local"] = sgn4(dd.mean()); N["B_gat_vs_avg_local_pholm"] = pfmt(pv[5])
loc6 = lB[['FedAvg_GCN', 'FedProto_GCN', 'FedGNN_Hard_GCN', 'FedGNN_Soft_GCN', 'FedGNN_Soft_Virtual_GCN']]
o, nm, pp = null_range(loc6); N.update(B_loc_null_obs=f4(o), B_loc_null_exp=f4(nm), B_loc_null_p=f3(pp))
N["A_loc_glob_maxdiff"] = f"{(lA[GCN8]-wA[GCN8]).abs().mean().max():.4f}"

# ------------------------------------------------------------- budget
bud_rows = []
for v in GCN8 + ["Centralized_MLP", "FedAvg_MLP"]:
    a = wA[v]; b = wD[v] if v in wD else wE[v]
    sa = df[(df.exp == "A") & (df.variant == v)]; sb = df[(df.exp.isin(["D", "E"])) & (df.variant == v)]
    bud_rows.append([SHORT[v] + (" (control)" if "MLP" in v else ""), f4(a.mean()), f4(b.mean()), sgn4(b.mean() - a.mean()), pfmt(wil(b, a)),
                     f"{sa.best_round.mean():.0f}", f"{sb.best_round.mean():.0f}", f"{int(sb.converged.sum())}/20"])
T["budget"] = {"header": ["Design", "AP, budget 1", "AP, budget 2", "Change", "p", "Best round/step, budget 1", "Best round/step, budget 2", "Converged, budget 2"],
               "rows": bud_rows}
N["D_cent"] = f4(wD.Centralized_GCN.mean()); N["D_cent_gain"] = sgn4(wD.Centralized_GCN.mean() - wA.Centralized_GCN.mean())
N["D_cent_beststep"] = f"{df[(df.exp=='D')&(df.variant=='Centralized_GCN')].best_round.mean():.0f}"
fg = [wD[v].mean() - wA[v].mean() for v in ['FedAvg_GCN', 'FedProto_GCN', 'FedGNN_Hard_GCN', 'FedGNN_Soft_GCN', 'FedGNN_Soft_Virtual_GCN']]
N["D_fed_gain_min"] = sgn4(min(fg)); N["D_fed_gain_max"] = sgn4(max(fg))
D6 = wD[['FedAvg_GCN', 'FedProx_GCN', 'FedProto_GCN', 'FedGNN_Hard_GCN', 'FedGNN_Soft_GCN', 'FedGNN_Soft_Virtual_GCN']]
o, nm, pp = null_range(D6); N.update(D_null_obs=f4(o), D_null_exp=f4(nm), D_null_p=f3(pp))
vbD = vs_base(wD[GCN8]).set_index("variant").drop(["Centralized_GCN"])
N["D_fed_sig_holm"] = str(int((vbD.p_holm < 0.05).sum()))
N["D_prox_avg"] = sgn4((wD.FedProx_GCN - wD.FedAvg_GCN).mean()); N["D_prox_avg_p"] = pfmt(wil(wD.FedProx_GCN, wD.FedAvg_GCN))
pwD = pairwise(wD[GCN8]); N["D_cent_pairs_sig"] = str(int(((pwD.p_holm < .05) & ((pwD.a == "Centralized_GCN") | (pwD.b == "Centralized_GCN"))).sum()))
N["D_other_pairs_sig"] = str(int(((pwD.p_holm < .05) & ~((pwD.a == "Centralized_GCN") | (pwD.b == "Centralized_GCN"))).sum()))

# ------------------------------------------------------------- metrics
Am = df[(df.exp == "A") & df.variant.isin(GCN8)]
mm = Am.groupby("variant")[["ap", "auc", "f1", "mcc", "tun_f1", "tun_mcc"]].mean()
N["rho_ap_f1_design"] = f"{stats.spearmanr(mm.ap, mm.f1)[0]:+.2f}"
N["rho_ap_f1_run"] = f"{stats.spearmanr(Am.ap, Am.f1)[0]:+.2f}"; N["rho_ap_tf1_run"] = f"{stats.spearmanr(Am.ap, Am.tun_f1)[0]:+.2f}"
N["rho_ap_mcc_run"] = f"{stats.spearmanr(Am.ap, Am.mcc)[0]:+.2f}"; N["rho_ap_tmcc_run"] = f"{stats.spearmanr(Am.ap, Am.tun_mcc)[0]:+.2f}"
N["f1_default"] = f3(Am.f1.mean()); N["f1_tuned"] = f3(Am.tun_f1.mean()); N["f1_gain"] = f"{(Am.tun_f1-Am.f1).mean():+.3f}"
N["thr_mean"] = f"{Am.tun_threshold.mean():.2f}"
met_rows = []
for v in list(mm.ap.sort_values(ascending=False).index):
    r = mm.loc[v]; met_rows.append([SHORT[v], f4(r.ap), f4(r.auc), f3(r.f1), f3(r.tun_f1), f3(r.mcc), f3(r.tun_mcc)])
T["metrics"] = {"header": ["Design", "AP", "ROC-AUC", "F1 @0.5", "F1 @tuned", "MCC @0.5", "MCC @tuned"], "rows": met_rows}
winners = {k: SHORT[mm[k].idxmax()] for k in ["ap", "auc", "f1", "mcc", "tun_f1", "tun_mcc"]}
N["metric_winners_default"] = f"AP: {winners['ap']}, ROC-AUC: {winners['auc']}, F1: {winners['f1']}, MCC: {winners['mcc']}"
N["metric_winners_tuned"] = f"F1: {winners['tun_f1']}, MCC: {winners['tun_mcc']}"

# ------------------------------------------------------------- temporal shift
cA = df[(df.exp == "A") & (df.variant == "Centralized_GCN")]; cD = df[(df.exp == "D") & (df.variant == "Centralized_GCN")]
N["shift_cent_A"] = f4((cA.best_val_ap - cA.ap).mean()); N["shift_cent_D"] = f4((cD.best_val_ap - cD.ap).mean())
N["shift_vs_spread"] = f"{(cA.best_val_ap - cA.ap).mean() / (m7.max() - m7.min()):.0f}"
N["valap_cent_A"] = f4(cA.best_val_ap.mean()); N["valap_cent_D"] = f4(cD.best_val_ap.mean())

# ------------------------------------------------------------- cost
FLOATS = {'FedAvg_GCN': 15300, 'FedProx_GCN': 15300, 'FedProto_GCN': 15300 + 128, 'FedGNN_Hard_GCN': 15300 + 128,
          'FedGNN_Soft_GCN': 15300 + 128, 'FedGNN_Soft_Virtual_GCN': 42884 + 128 + 166, 'FedGNN_Soft_Virtual_GAT': 55812 + 128 + 166}
cost_rows = []
for v in GCN8:
    s = df[(df.exp == "A") & (df.variant == v)]
    tt = s.train_time.mean()
    if v in FLOATS:
        per_round = 2 * 10 * FLOATS[v] * 4 / 1e6
        tot = per_round * s.rounds.mean()
        cost_rows.append([SHORT[v], f"{FLOATS[v]:,}", f"{per_round:.2f}", f"{s.rounds.mean():.0f}", f"{tot:.0f}", f"{tt:.1f}"])
    else:
        cost_rows.append([SHORT[v], "–", "–", f"{s.rounds.mean():.0f}", "–", f"{tt:.1f}"])
T["cost"] = {"header": ["Design", "Floats exchanged per client per direction", "MB per round (10 clients, up+down)",
                        "Rounds run (mean)", "MB per run", "Training time (s, CPU)"], "rows": cost_rows}
N["mb_round_base"] = f"{2*10*15300*4/1e6:.2f}"; N["mb_round_vn"] = f"{2*10*(42884+294)*4/1e6:.2f}"
N["mb_round_gat"] = f"{2*10*(55812+294)*4/1e6:.2f}"
ttA = df[(df.exp == "A")].groupby("variant").train_time.mean()
N["t_cent"] = f"{ttA.Centralized_GCN:.1f}"; N["t_avg"] = f"{ttA.FedAvg_GCN:.1f}"; N["t_soft"] = f"{ttA.FedGNN_Soft_GCN:.1f}"; N["t_gat"] = f"{ttA.FedGNN_Soft_Virtual_GAT:.1f}"
N["t_ratio_avg"] = f"{ttA.FedAvg_GCN/ttA.Centralized_GCN:.1f}"; N["t_ratio_soft"] = f"{ttA.FedGNN_Soft_GCN/ttA.Centralized_GCN:.1f}"

# ------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
INK = "#1f3b57"; ACC = "#b4432b"; GREY = "#8c8c8c"

# Fig 1: run A strip plot
order = list(A8.mean().sort_values(ascending=False).index) + ["FedAvg_MLP", "Centralized_MLP"]
fig, ax = plt.subplots(figsize=(7.0, 3.4))
rng = np.random.default_rng(1)
for i, v in enumerate(order):
    y = wA[v].values; ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=9, color=GREY if "MLP" in v else INK, alpha=0.75, lw=0)
    lo, hi = ci95(y); ax.errorbar(i + 0.25, y.mean(), yerr=[[y.mean() - lo], [hi - y.mean()]], fmt="o", color=ACC, ms=4, capsize=2.5, lw=1.2)
ax.axhspan(m7.min(), m7.max(), color="#d9d9d9", alpha=0.7, lw=0, label=f"spread of seven convolutional designs ({obs7:.4f} AP)")
ax.set_xticks(range(len(order))); ax.set_xticklabels([SHORT[v] for v in order], rotation=30, ha="right")
ax.set_ylabel("Test average precision"); ax.legend(loc="lower left", frameon=False, fontsize=8)
ax.axvline(len(order) - 2.5, color=GREY, lw=0.6, ls=":")
ax.text(len(order) - 1.5, ax.get_ylim()[1], "graph-free\ncontrols", ha="center", va="top", fontsize=7.5, color=GREY)
fig.tight_layout(); fig.savefig(FIG / "fig1_runA.png", dpi=300); fig.savefig(FIG / "fig1_runA.pdf"); plt.close(fig)

# Fig 2: fixed partitions forest plot
fig, ax = plt.subplots(figsize=(7.0, 3.6))
des = FED6[1:]; cols = ["#1f3b57", "#4f7ca8", "#8fb0d0", "#c47a5a", "#b4432b"]
for k in range(5):
    w = wide(df, f"C{k}")
    for j, v in enumerate(des):
        d = w[v] - w.FedAvg_GCN; lo, hi = ci95(d); x = j + (k - 2) * 0.14
        sigk = fp[k].loc[v, "p_holm"] < 0.05
        ax.errorbar(x, d.mean(), yerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o" if sigk else "o", mfc=cols[k] if sigk else "white",
                    color=cols[k], ms=4.5, capsize=2, lw=1, label=f"C{k} (fixed partition, seed {k})" if j == 0 else None)
dA = [wA[v] - wA.FedAvg_GCN for v in des]
for j, d in enumerate(dA):
    lo, hi = ci95(d); ax.errorbar(j + 0.42, d.mean(), yerr=[[d.mean() - lo], [hi - d.mean()]], fmt="s", color="black", mfc="black", ms=4, capsize=2, lw=1,
                                  label="A (partition redrawn per seed)" if j == 0 else None)
ax.axhline(0, color=GREY, lw=0.8)
ax.set_xticks(range(len(des))); ax.set_xticklabels([SHORT[v] for v in des])
ax.set_ylabel("AP difference to FedAvg\n(mean, 95% CI over 20 seeds)")
ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
ax.text(0.99, 0.98, "filled marker: significant after Holm correction", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=GREY)
fig.tight_layout(); fig.savefig(FIG / "fig2_fixed_partitions.png", dpi=300); fig.savefig(FIG / "fig2_fixed_partitions.pdf"); plt.close(fig)

# Fig 3: run B global vs per-client
ordB = ['Centralized_GCN', 'FedGNN_Soft_Virtual_GAT', 'FedGNN_Soft_Virtual_GCN', 'FedAvg_GCN', 'FedGNN_Hard_GCN', 'FedGNN_Soft_GCN', 'FedProto_GCN', 'FedAvg_MLP']
fig, ax = plt.subplots(figsize=(7.0, 3.4)); x = np.arange(len(ordB)); wdt = 0.26
ax.bar(x - wdt, [wA[v].mean() for v in ordB], wdt, color="#c9c9c9", label="A: Louvain, global inference")
ax.bar(x, [wB[v].mean() for v in ordB], wdt, color=INK, label="B: random partition, global inference")
ax.bar(x + wdt, [lB[v].mean() for v in ordB], wdt, color=ACC, label="B: random partition, per-client inference")
ax.set_ylim(0.40, 0.54); ax.set_xticks(x); ax.set_xticklabels([SHORT[v] for v in ordB], rotation=25, ha="right")
ax.set_ylabel("Mean test AP (20 seeds)"); ax.legend(fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.tight_layout(); fig.savefig(FIG / "fig3_runB.png", dpi=300); fig.savefig(FIG / "fig3_runB.pdf"); plt.close(fig)

# Fig 4: budget
pairs = [("Centralized_GCN", "Central-GCN"), ("Centralized_MLP", "Central-MLP"), ("FedAvg_GCN", "FedAvg-GCN"), ("FedAvg_MLP", "FedAvg-MLP")]
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for i, (v, lab) in enumerate(pairs):
    a = wA[v]; b = wD[v] if v in wD else wE[v]
    for xx, s, c in [(i - 0.15, a, "#9a9a9a"), (i + 0.15, b, ACC)]:
        lo, hi = ci95(s); ax.errorbar(xx, s.mean(), yerr=[[s.mean() - lo], [hi - s.mean()]], fmt="o", color=c, ms=5, capsize=2.5)
    ax.plot([i - 0.15, i + 0.15], [a.mean(), b.mean()], color=GREY, lw=0.8)
ax.scatter([], [], color="#9a9a9a", label="budget 1 (100 rounds × 1 local step)"); ax.scatter([], [], color=ACC, label="budget 2 (300 rounds × 5 local steps)")
ax.set_xticks(range(4)); ax.set_xticklabels([p[1] for p in pairs]); ax.set_ylabel("Mean test AP, 95% CI"); ax.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "fig4_budget.png", dpi=300); fig.savefig(FIG / "fig4_budget.pdf"); plt.close(fig)

# Fig 5: null-range distribution for A (7 designs)
X = A8[CONV7].values; r = np.random.default_rng(0); null = []
for _ in range(20000):
    P = np.array([row[r.permutation(len(row))] for row in X]); mm_ = P.mean(0); null.append(mm_.max() - mm_.min())
null = np.array(null)
fig, ax = plt.subplots(figsize=(5.2, 2.8))
ax.hist(null, bins=60, color="#c9c9c9", edgecolor="white")
ax.axvline(obs7, color=ACC, lw=1.6, label=f"observed spread {obs7:.4f} (p = {p7:.2f})")
ax.set_xlabel("Spread of the seven design means under exchangeability (AP)"); ax.set_ylabel("Permutations")
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "fig5_null_range.png", dpi=300); fig.savefig(FIG / "fig5_null_range.pdf"); plt.close(fig)

(OUT / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False), encoding="utf-8")
(OUT / "tables.json").write_text(json.dumps(T, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(N, indent=0, ensure_ascii=False))

# ---- additional values referenced in the text
N["A_avg_beats_proto"] = str(int((A8.FedAvg_GCN > A8.FedProto_GCN).sum()))
sa = ["FedGNN_Hard_GCN", "FedGNN_Soft_GCN", "FedGNN_Soft_Virtual_GCN", "FedGNN_Soft_Virtual_GAT"]
N["C_pos_sig_sa"] = str(sum(int(((fp[k].loc[sa].p_holm < .05) & (fp[k].loc[sa]["diff"] > 0)).sum()) for k in range(5)))
N["C_neg_sig_sa"] = str(sum(int(((fp[k].loc[sa].p_holm < .05) & (fp[k].loc[sa]["diff"] < 0)).sum()) for k in range(5)))
N["gapD_gcn_abs"] = f4(abs(wD.FedAvg_GCN.mean() - wD.Centralized_GCN.mean()))
N["gapD_mlp_abs"] = f4(abs(wE.FedAvg_MLP.mean() - wE.Centralized_MLP.mean()))
v6t = v6[v6.exp == "v6a_A"].groupby("variant").runtime.mean()
N["v6a_t_ratio"] = f"{v6t.FedAvg_GCN / v6t.Centralized_GCN:.1f}"
N["B_mlp_A"] = f4(wA.FedAvg_MLP.mean())
(OUT / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False), encoding="utf-8")
print({k: N[k] for k in ["A_avg_beats_proto", "C_pos_sig_sa", "C_neg_sig_sa", "gapD_gcn_abs", "gapD_mlp_abs", "v6a_t_ratio", "B_mlp_A"]})
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
N["C0_soft_abs"] = f4(abs(fp[0].loc["FedGNN_Soft_GCN", "diff"])); N["C1_soft_abs"] = f4(abs(fp[1].loc["FedGNN_Soft_GCN", "diff"]))
N["v6a_avgprox_median"] = f"{d_avg_prox.median():.5f}"
N["sub6_n_first_w"] = WORDS[int(N["sub6_n_first"])]; N["A_pairs_holm_w"] = WORDS[int(N["A_pairs_holm"])]
N["C_pos_sig_sa_w"] = WORDS[int(N["C_pos_sig_sa"])]; N["C_neg_sig_sa_w"] = WORDS[int(N["C_neg_sig_sa"])]
assert float(N["A_vn_gcn_max"]) < 1e-6 and float(N["A_vn_gat_max"]) < 1e-6
assert N["D_other_pairs_sig"] == "0"
(OUT / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False), encoding="utf-8")

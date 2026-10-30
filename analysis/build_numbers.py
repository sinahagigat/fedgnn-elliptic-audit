"""
Every number, table and figure in the manuscript is produced here from the
run records in results/v7, the diagnostics in results/diagnostics and, for
the reproduction section only, the earlier GPU runs in results/v6a.
Nothing in the manuscript is typed by hand.

Outputs: numbers.json (scalar values referenced as {{key}} in the text),
tables.json (table rows), figures/*.png
"""
import gzip
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
RESULTS_V7 = Path(__file__).resolve().parents[1] / "results" / "v7"
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
# three code versions produced runs (tools/verify_release.py): the first runs,
# the version with fed.val_split_weighting, and the version with fed.select_on_aggregate
VALSPLIT = df.exp.str.match(r"^(P\d+|V\d+|Q\d+|N|R|L|M|W|S\d+)$")
AGGSEL = df.exp.str.match(r"^Z")
OLD = ~(VALSPLIT | AGGSEL)
assert all(df[m].code_hash.nunique() == 1 for m in (OLD, VALSPLIT, AGGSEL))
N["code_hash"] = df[OLD].code_hash.iloc[0]
N["code_hash_rev3"] = df[VALSPLIT].code_hash.iloc[0]
N["code_hash_z"] = df[AGGSEL].code_hash.iloc[0]
N["n_runs_z"] = f"{int(AGGSEL.sum()):,}"
sys.path.insert(0, str(SCR.parent / "tools"))
from code_versions import RELEASED as CODE_RELEASED  # noqa: E402
N["code_hash_release"] = CODE_RELEASED
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
       [(f"C{k}", f"Louvain, fixed (partition seed {k if k < 5 else 37 + k})") for k in range(10)]
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
Am = df[(df.exp == "A") & df.variant.isin([v for v in GCN8 if v != "FedProx_GCN"])]  # FedProx = FedAvg under budget 1
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
ax.axhspan(m7.min(), m7.max(), color="#d9d9d9", alpha=0.7, lw=0, label=f"spread of the convolutional designs ({obs7:.4f} AP)")
ax.set_xticks(range(len(order))); ax.set_xticklabels([SHORT[v] for v in order], rotation=30, ha="right")
ax.set_ylabel("Test average precision"); ax.legend(loc="lower left", frameon=False, fontsize=8)
ax.axvline(len(order) - 2.5, color=GREY, lw=0.6, ls=":")
ax.text(len(order) - 1.5, ax.get_ylim()[1], "graph-free\ncontrols", ha="center", va="top", fontsize=7.5, color=GREY)
fig.tight_layout(); fig.savefig(FIG / "fig1_runA.png", dpi=300); fig.savefig(FIG / "fig1_runA.pdf"); plt.close(fig)

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
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
N["C0_soft_abs"] = f4(abs(fp[0].loc["FedGNN_Soft_GCN", "diff"])); N["C1_soft_abs"] = f4(abs(fp[1].loc["FedGNN_Soft_GCN", "diff"]))
N["v6a_avgprox_median"] = f"{d_avg_prox.median():.5f}"
N["sub6_n_first_w"] = WORDS[int(N["sub6_n_first"])]; N["A_pairs_holm_w"] = WORDS[int(N["A_pairs_holm"])]
N["C_pos_sig_sa_w"] = WORDS[int(N["C_pos_sig_sa"])]; N["C_neg_sig_sa_w"] = WORDS[int(N["C_neg_sig_sa"])]
assert float(N["A_vn_gcn_max"]) < 1e-6 and float(N["A_vn_gat_max"]) < 1e-6
assert N["D_other_pairs_sig"] == "0"

# ------------------------------------------------------------- fixed partitions C0-C9,
# local features, tabular references, partition and aggregation diagnostics,
# temporal breakdown
DIAG = Path(__file__).resolve().parents[1] / "results" / "diagnostics"
SA4 = ["FedGNN_Hard_GCN", "FedGNN_Soft_GCN", "FedGNN_Soft_Virtual_GCN", "FedGNN_Soft_Virtual_GAT"]
DES5 = ["FedProto_GCN"] + SA4
KS = list(range(10))
PSEED = {k: (k if k < 5 else 37 + k) for k in KS}          # C5..C9 use partition seeds 42..46

# ---- run accounting
acc = [("A", "Louvain, redrawn per seed", "1", "165", 10), ("B", "random node, redrawn per seed", "1", "165", 10)]
acc += [(f"C{k}", f"Louvain, fixed (partition seed {PSEED[k]})", "1", "165", 7) for k in KS]
acc += [("D", "Louvain, redrawn per seed", "2", "165", 8), ("E", "Louvain, redrawn per seed", "2", "165", 2),
        ("F", "random node, redrawn per seed", "2", "165", 2), ("G", "Louvain, redrawn per seed", "2", "93 (local only)", 4)]
T["accounting"] = None  # built at the end of the script, once every protocol is listed in acc
N["n_runs_main"] = f"{int(df.exp.isin(['A', 'B', 'D', 'E'] + [f'C{k}' for k in range(5)]).sum()):,}"
N["n_runs_new"] = f"{int(df.exp.isin(['F', 'G'] + [f'C{k}' for k in range(5, 10)]).sum()):,}"
N["n_v6a_each"] = str(int((v6.exp == "v6a_C").sum()))

# ---- pipeline identity: run A with seed s is the cell (partition s, training seed s) of C(s-37)
ident = []
for k in range(5, 10):
    s = PSEED[k]
    a = df[(df.exp == "A") & (df.seed == s)].set_index("variant")[["ap", "auc", "f1", "best_round"]]
    c = df[(df.exp == f"C{k}") & (df.seed == s)].set_index("variant")[["ap", "auc", "f1", "best_round"]]
    j = a.index.intersection(c.index); ident.append(float((a.loc[j] - c.loc[j]).abs().max().max()))
assert max(ident) == 0.0
N["ident_n"] = str(5 * 7)

# ---- A without the FedProx duplicate
CONV6 = [v for v in CONV7 if v != "FedProx_GCN"]
o, nm, pp = null_range(A8[CONV6]); N.update(A_null6_obs=f4(o), A_null6_exp=f4(nm), A_null6_p=f3(pp))

# ---- paired SD per pair (identical pair excluded)
sdp = {}
for a, b in itertools.combinations(GCN8, 2):
    if {a, b} == {"FedAvg_GCN", "FedProx_GCN"}:
        continue
    sdp[(a, b)] = (A8[a] - A8[b]).std(ddof=1)
sv = np.array(list(sdp.values()))
N["sigma_d"] = f4(sv.mean()); N["sigma_d_med"] = f4(np.median(sv)); N["sigma_d_min"] = f4(sv.min()); N["sigma_d_max"] = f4(sv.max())
N["n_pairs_sd"] = str(len(sv))
sdd = float(np.median(sv))
N["n_equiv_005"] = str(int(np.ceil((1.645 + 1.282) ** 2 * sdd ** 2 / 0.005 ** 2)))
N["n_equiv_010"] = str(int(np.ceil((1.645 + 1.282) ** 2 * sdd ** 2 / 0.010 ** 2)))
N["n_detect_003"] = str(int(np.ceil((1.96 + 0.8416) ** 2 * sdd ** 2 / 0.003 ** 2)))
N["n_detect_003_lo"] = str(int(np.ceil((1.96 + 0.8416) ** 2 * sv.min() ** 2 / 0.003 ** 2)))
N["n_detect_003_hi"] = str(int(np.ceil((1.96 + 0.8416) ** 2 * sv.max() ** 2 / 0.003 ** 2)))

# ---- ten fixed partitions
fp = {}; Wk = {}
for k in KS:
    w = wide(df, f"C{k}"); Wk[k] = w
    fp[k] = vs_base(w[["FedAvg_GCN", "FedProx_GCN"] + DES5]).set_index("variant")
rows = []
for k in KS:
    r = [f"C{k} (seed {PSEED[k]})"]
    for v in DES5:
        x = fp[k].loc[v]; r.append(f"{x['diff']:+.4f}{'*' if x.p_holm < 0.05 else ''} ({int(x.wins)})")
    rows.append(r)
pA = dict(zip(FED6[1:], holm([wil(wA[x], wA.FedAvg_GCN) for x in FED6[1:]])))
rows.append(["A (redrawn)"] + [f"{(wA[v]-wA.FedAvg_GCN).mean():+.4f}{'*' if pA[v] < 0.05 else ''} ({int((wA[v] > wA.FedAvg_GCN).sum())})" for v in DES5])
pooled = {}
for v in DES5:
    d = np.array([fp[k].loc[v, "diff"] for k in KS]); t_ = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    pooled[v] = dict(mean=d.mean(), lo=d.mean() - h, hi=d.mean() + h, p=2 * stats.t.sf(abs(t_), len(d) - 1), sd=d.std(ddof=1),
                     npos=int((d > 0).sum()))
rows.append(["Mean over C0–C9 [95% CI]"] + [f"{pooled[v]['mean']:+.4f} [{pooled[v]['lo']:+.4f}, {pooled[v]['hi']:+.4f}]" for v in DES5])
T["fixed"] = {"header": ["Partition", "FedProto", "Hard", "Soft", "Soft-VN", "Soft-VN-GAT"], "rows": rows}
sig_pos = sum(int(((fp[k].loc[SA4].p_holm < .05) & (fp[k].loc[SA4]["diff"] > 0)).sum()) for k in KS)
sig_neg = sum(int(((fp[k].loc[SA4].p_holm < .05) & (fp[k].loc[SA4]["diff"] < 0)).sum()) for k in KS)
N["C_pos_sig_sa"] = str(sig_pos); N["C_neg_sig_sa"] = str(sig_neg)
N["C_part_any_sig"] = str(sum(int((fp[k].loc[SA4].p_holm < .05).any()) for k in KS))
N["C_part_soft_sig"] = str(sum(int(fp[k].loc["FedGNN_Soft_GCN", "p_holm"] < .05) for k in KS))
N["C_part_soft_pos"] = str(sum(int(fp[k].loc["FedGNN_Soft_GCN", "p_holm"] < .05 and fp[k].loc["FedGNN_Soft_GCN", "diff"] > 0) for k in KS))
N["C_part_soft_neg"] = str(sum(int(fp[k].loc["FedGNN_Soft_GCN", "p_holm"] < .05 and fp[k].loc["FedGNN_Soft_GCN", "diff"] < 0) for k in KS))
N["C0_soft"] = sgn4(fp[0].loc["FedGNN_Soft_GCN", "diff"]); N["C1_soft"] = sgn4(fp[1].loc["FedGNN_Soft_GCN", "diff"])
N["C0_soft_abs"] = f4(abs(fp[0].loc["FedGNN_Soft_GCN", "diff"])); N["C1_soft_abs"] = f4(abs(fp[1].loc["FedGNN_Soft_GCN", "diff"]))
N["C0_soft_wins"] = str(int(fp[0].loc["FedGNN_Soft_GCN", "wins"])); N["C1_soft_wins"] = str(int(fp[1].loc["FedGNN_Soft_GCN", "wins"]))
N["C_soft_max"] = sgn4(max(fp[k].loc["FedGNN_Soft_GCN", "diff"] for k in KS)); N["C_soft_min"] = sgn4(min(fp[k].loc["FedGNN_Soft_GCN", "diff"] for k in KS))
for v, key in [("FedGNN_Soft_GCN", "soft"), ("FedGNN_Hard_GCN", "hard"), ("FedGNN_Soft_Virtual_GCN", "softvn"),
               ("FedGNN_Soft_Virtual_GAT", "gat"), ("FedProto_GCN", "proto")]:
    q = pooled[v]; N[f"pool_{key}"] = sgn4(q["mean"]); N[f"pool_{key}_ci"] = f"[{q['lo']:+.4f}, {q['hi']:+.4f}]"; N[f"pool_{key}_p"] = pfmt(q["p"])
    N[f"pool_{key}_npos"] = str(q["npos"])
N["C_proto_sig"] = str(sum(int(fp[k].loc["FedProto_GCN", "p_holm"] < .05 and fp[k].loc["FedProto_GCN", "diff"] < 0) for k in KS))
N["C_proto_neg"] = str(sum(int(fp[k].loc["FedProto_GCN", "diff"] < 0) for k in KS))
N["C_proto_range"] = f"{min(fp[k].loc['FedProto_GCN','diff'] for k in KS):+.4f} to {max(fp[k].loc['FedProto_GCN','diff'] for k in KS):+.4f}"
MC = pd.DataFrame({k: Wk[k][FED6].mean() for k in KS})
N["C_rank_rho01"] = f"{MC.corr(method='spearman').loc[0, 1]:+.2f}"
N["C_winners"] = ", ".join(f"C{k}: {SHORT[MC[k].idxmax()]}" for k in KS)
N["C_n_winners"] = str(MC.idxmax().nunique())
WORDS.update({9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"})

# variance components, including the design-by-partition interaction of the difference to FedAvg
vc = []
for v in FED6:
    W = np.array([Wk[k][v].values for k in KS]); within = W.var(axis=1, ddof=1).mean(); means = W.mean(1)
    between = max(means.var(ddof=1) - within / 20, 0)
    if v != "FedAvg_GCN":
        Dm = np.array([(Wk[k][v] - Wk[k].FedAvg_GCN).values for k in KS]); wd = Dm.var(axis=1, ddof=1).mean()
        inter = f4(np.sqrt(max(Dm.mean(1).var(ddof=1) - wd / 20, 0)))
    else:
        inter = "–"
    vc.append([SHORT[v], f4(np.sqrt(within)), f4(np.sqrt(between)), inter, f"{means.min():.4f}–{means.max():.4f}", f4(wA[v].std(ddof=1))])
T["varcomp"] = {"header": ["Design", "Within-partition SD", "Between-partition SD", "Interaction SD (difference to FedAvg)",
                           "Range of partition means", "SD under redrawn partitions (A)"], "rows": vc}
VC = {r[0]: r for r in vc}
N["between_sd_avg"] = VC["FedAvg"][2]; N["between_sd_proto"] = VC["FedProto"][2]
N["between_sd_struct_min"] = f4(min(float(VC[SHORT[v]][2]) for v in SA4)); N["between_sd_struct_max"] = f4(max(float(VC[SHORT[v]][2]) for v in SA4))
N["inter_sd_proto"] = VC["FedProto"][3]
N["inter_sd_struct_min"] = f4(min(float(VC[SHORT[v]][3]) for v in SA4)); N["inter_sd_struct_max"] = f4(max(float(VC[SHORT[v]][3]) for v in SA4))
N["within_sd_soft"] = VC["Soft"][1]; N["between_sd_soft"] = VC["Soft"][2]
N["between_sd_soft_old"] = f4(np.sqrt(max(np.array([Wk[k].FedGNN_Soft_GCN.mean() for k in range(5)]).var(ddof=1)
                                          - np.array([Wk[k].FedGNN_Soft_GCN.var(ddof=1) for k in range(5)]).mean() / 20, 0)))
N["between_sd_soft_new"] = f4(np.sqrt(max(np.array([Wk[k].FedGNN_Soft_GCN.mean() for k in range(5, 10)]).var(ddof=1)
                                          - np.array([Wk[k].FedGNN_Soft_GCN.var(ddof=1) for k in range(5, 10)]).mean() / 20, 0)))

# dispersion of A against the ten-partition matrix (resampling)
rng_ = np.random.default_rng(0); disp = {}
for v in FED6:
    Mx = np.array([Wk[k][v].values for k in KS]); sims = np.empty(20000)
    for i in range(20000):
        rows_ = rng_.choice(10, 10, replace=True); kk = rows_[rng_.integers(0, 10, 20)]
        sims[i] = Mx[kk, np.arange(20)].std(ddof=1)
    disp[v] = (float(np.median(sims)), float((sims <= wA[v].std(ddof=1)).mean()))
N["disp_soft_exp"] = f4(disp["FedGNN_Soft_GCN"][0]); N["disp_soft_p"] = f3(disp["FedGNN_Soft_GCN"][1])
N["disp_softvn_p"] = f3(disp["FedGNN_Soft_Virtual_GCN"][1])
N["disp_other_pmin"] = f3(min(disp[v][1] for v in ["FedAvg_GCN", "FedProto_GCN", "FedGNN_Hard_GCN", "FedGNN_Soft_Virtual_GAT"]))
N["A_soft_sd"] = f4(wA.FedGNN_Soft_GCN.std(ddof=1))
N["C_low_runs_soft"] = str(int(sum((Wk[k].FedGNN_Soft_GCN < 0.45).sum() for k in KS))); N["A_soft_min"] = f4(wA.FedGNN_Soft_GCN.min())

# ---- client composition of every partition
PJ = json.loads((DIAG / "partitions.json").read_text())
comp = {}
for p in PJ["partitions"]:
    if p["strategy"] != "louvain":
        continue
    cl = p["clients"]
    comp[p["seed"]] = dict(minvalpos=min(c["pos_val"] for c in cl), mintrpos=min(c["pos_train"] for c in cl),
                           sdval=float(np.std([c["pos_val"] / c["n_val"] for c in cl])), steps=min(c["steps_present"] for c in cl),
                           ent=float(np.mean([c["step_entropy"] for c in cl])), ntrain={str(c["client"]): c["n_train"] for c in cl})
assert all(comp[s]["steps"] == 49 for s in comp)
N["minvalpos_lo"] = str(min(comp[s]["minvalpos"] for s in comp)); N["minvalpos_hi"] = str(max(comp[s]["minvalpos"] for s in comp))
softd = [fp[k].loc["FedGNN_Soft_GCN", "diff"] for k in KS]
rhos = {f: stats.spearmanr([comp[PSEED[k]][f] for k in KS], softd) for f in ["minvalpos", "mintrpos", "sdval", "ent"]}
N["comp_rho_maxabs"] = f"{max(abs(r[0]) for r in rhos.values()):.2f}"; N["comp_rho_pmin"] = f"{min(r[1] for r in rhos.values()):.2f}"

# ---- effective number of clients under Soft aggregation (per-round history)
H = json.loads(gzip.open(DIAG / "history.json.gz").read())
def soft_weights(run, part):
    out = []; top = []
    for e in run["rounds"][: max(run["best_round_idx"] or 1, 1) + 1]:
        sc = e["score"]
        if not isinstance(sc, dict) or not sc:
            continue
        ks = sorted(sc, key=int); s = np.array([sc[q] for q in ks]) / 0.5; w = np.exp(s - s.max()); w /= w.sum()
        dw = np.array([comp[part]["ntrain"][q] for q in ks], float); w = w * dw / dw.sum(); w /= w.sum()
        out.append(1 / np.sum(w ** 2)); top.append(ks[int(np.argmax(w))])
    return out, top
enc_rows = []; enc_all = []; topshare = []
for k in range(5):
    e_k, t_k = [], []
    for key, run in H.items():
        if key.startswith(f"C{k}_") and "/FedGNN_Soft_GCN__" in key:
            o_, t_ = soft_weights(run, PSEED[k]); e_k.append(np.mean(o_)); t_k += t_
    vals, cnt = np.unique(t_k, return_counts=True); share = cnt.max() / cnt.sum()
    enc_all.append(np.mean(e_k)); topshare.append(share)
    enc_rows.append([f"C{k}", f"{fp[k].loc['FedGNN_Soft_GCN','diff']:+.4f}", f"{np.mean(e_k):.2f}", f"client {vals[np.argmax(cnt)]} ({100*share:.0f}%)"])
eA = []
for key, run in H.items():
    if key[0] == "A" and "/FedGNN_Soft_GCN__" in key:
        s = int(key.split("seed")[1]); o_, _ = soft_weights(run, s); eA.append(np.mean(o_))
N["enc_lo"] = f"{min(enc_all):.1f}"; N["enc_hi"] = f"{max(enc_all):.1f}"; N["enc_A"] = f"{np.mean(eA):.1f}"
N["top_lo"] = f"{100*min(topshare):.0f}"; N["top_hi"] = f"{100*max(topshare):.0f}"

# ---- graph-free control: CIs and equivalence at +-0.01 AP
def ci_str(d, lev=0.95):
    d = np.asarray(d); h = stats.t.ppf(1 - (1 - lev) / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)); return d.mean() - h, d.mean() + h
EQ = 0.01; N["eq_margin"] = f"{EQ:.2f}"
for key, x, y in [("A_c", wA.Centralized_MLP, wA.Centralized_GCN), ("A_f", wA.FedAvg_MLP, wA.FedAvg_GCN),
                  ("D_c", wE.Centralized_MLP, wD.Centralized_GCN), ("D_f", wE.FedAvg_MLP, wD.FedAvg_GCN)]:
    lo, hi = ci_str(x - y); l9, h9 = ci_str(x - y, 0.90)
    N[f"mlp{key}_ci"] = f"[{lo:+.4f}, {hi:+.4f}]"; N[f"mlp{key}_ci90"] = f"[{l9:+.4f}, {h9:+.4f}]"; N[f"mlp{key}_eq"] = "yes" if (l9 > -EQ and h9 < EQ) else "no"
N["mlp_eq_any"] = "yes" if any(N[f"mlp{k}_eq"] == "yes" for k in ["A_c", "A_f", "D_c", "D_f"]) else "no"

# ---- protocol G (93 local features, budget 2) and tabular references
wH = wide(df, "G"); wG = wide(df, "F")
h_rows = []
for v, ref in [("Centralized_GCN", wD), ("FedAvg_GCN", wD), ("Centralized_MLP", wE), ("FedAvg_MLP", wE)]:
    lo, hi = ci95(wH[v]); d = wH[v] - ref[v]
    h_rows.append([SHORT[v] + (" (control)" if "MLP" in v else ""), f4(ref[v].mean()), f"{wH[v].mean():.4f} [{lo:.4f}, {hi:.4f}]", sgn4(d.mean()), pfmt(wil(wH[v], ref[v]))])
TB = json.loads((DIAG / "tabular.json").read_text())["runs"]
tb = pd.DataFrame([dict(model=r["model"], fs=r["features"], seed=r["seed"], ap=r["test"]["ap"], auc=r["test"]["auc"],
                        tf1=r["test_tuned"]["f1"], vap=r["val"]["ap"]) for r in TB])
NAMES = {"random_forest": "Random forest", "hist_gradient_boosting": "Gradient-boosted trees", "logistic_regression": "Logistic regression"}
tab_rows = []
for m in ["hist_gradient_boosting", "random_forest", "logistic_regression"]:
    a = tb[(tb.model == m) & (tb.fs == "all_165")]; b = tb[(tb.model == m) & (tb.fs == "local_93")]
    tab_rows.append([NAMES[m], str(len(a)), f4(a.ap.mean()), f4(b.ap.mean()), f4(a.auc.mean()), f3(a.tf1.mean())])
best_nn = max(wD[v].mean() for v in GCN8); best_nn_name = SHORT[max(GCN8, key=lambda v: wD[v].mean())]
tab_rows.append([f"Best neural design, 165 features ({best_nn_name}, budget 2)", "20", f4(best_nn), "–",
                 f4(df[(df.exp == 'D') & (df.variant == max(GCN8, key=lambda v: wD[v].mean()))].auc.mean()),
                 f3(df[(df.exp == 'D') & (df.variant == max(GCN8, key=lambda v: wD[v].mean()))].tun_f1.mean())])
tab_rows.append(["FedAvg-MLP, 93 local features (protocol G)", "20", "–", f4(wH.FedAvg_MLP.mean()),
                 f4(df[(df.exp == 'G') & (df.variant == 'FedAvg_MLP')].auc.mean()), f3(df[(df.exp == 'G') & (df.variant == 'FedAvg_MLP')].tun_f1.mean())])
T["features"] = {"header": ["Design", "Budget 2, 165 features", "Budget 2, 93 local features [95% CI]", "Change", "p"], "rows": h_rows}
T["tabular"] = {"header": ["Model (centralized unless stated)", "Runs", "Test AP, 165 features", "Test AP, 93 local features", "Test ROC-AUC, 165 features", "F1 at tuned threshold, 165 features"],
                "rows": tab_rows}
N["H_cent_gcn"] = f4(wH.Centralized_GCN.mean()); N["H_cent_mlp"] = f4(wH.Centralized_MLP.mean())
N["H_fed_gcn"] = f4(wH.FedAvg_GCN.mean()); N["H_fed_mlp"] = f4(wH.FedAvg_MLP.mean())
for key, x, y in [("H_c", wH.Centralized_MLP, wH.Centralized_GCN), ("H_f", wH.FedAvg_MLP, wH.FedAvg_GCN)]:
    lo, hi = ci_str(x - y); N[f"{key}_d"] = sgn4((x - y).mean()); N[f"{key}_ci"] = f"[{lo:+.4f}, {hi:+.4f}]"; N[f"{key}_p"] = pfmt(wil(x, y)); N[f"{key}_w"] = str(int((x > y).sum()))
N["H_cgcn_drop"] = sgn4((wH.Centralized_GCN - wD.Centralized_GCN).mean()); N["H_fmlp_gain"] = sgn4((wH.FedAvg_MLP - wE.FedAvg_MLP).mean())
N["H_cmlp_gain"] = sgn4((wH.Centralized_MLP - wE.Centralized_MLP).mean()); N["H_fgcn_gain"] = sgn4((wH.FedAvg_GCN - wD.FedAvg_GCN).mean())
N["H_fmlp_vs_cmlp"] = sgn4((wH.FedAvg_MLP - wH.Centralized_MLP).mean()); N["H_fmlp_vs_cmlp_p"] = pfmt(wil(wH.FedAvg_MLP, wH.Centralized_MLP))
N["H_best_vs_best165"] = sgn4(wH.FedAvg_MLP.mean() - best_nn); N["best_nn"] = f4(best_nn); N["best_nn_name"] = best_nn_name
gb = tb[(tb.model == "hist_gradient_boosting") & (tb.fs == "all_165")]; rf = tb[(tb.model == "random_forest") & (tb.fs == "all_165")]
rf93 = tb[(tb.model == "random_forest") & (tb.fs == "local_93")]; gb93 = tb[(tb.model == "hist_gradient_boosting") & (tb.fs == "local_93")]
N["tab_gb"] = f4(gb.ap.mean()); N["tab_rf"] = f4(rf.ap.mean()); N["tab_rf_sd"] = f4(rf.ap.std(ddof=1)); N["tab_lr"] = f4(tb[(tb.model == "logistic_regression") & (tb.fs == "all_165")].ap.mean())
N["tab_rf93"] = f4(rf93.ap.mean()); N["tab_gb93"] = f4(gb93.ap.mean())
N["tab_gap_nn"] = f4(gb.ap.mean() - best_nn); N["tab_gap_rf"] = f4(rf.ap.mean() - best_nn)
N["tab_gap_H"] = f4(gb.ap.mean() - wH.FedAvg_MLP.mean())
N["tab_rf_vap"] = f4(rf.vap.mean()); N["tab_gb_vap"] = f4(gb.vap.mean())

# ---- protocol F (random partition, budget 2)
N["G_fed_gcn"] = f4(wG.FedAvg_GCN.mean()); N["G_fed_mlp"] = f4(wG.FedAvg_MLP.mean())
for key, c, f_ in [("G_gap_mlp", wE.Centralized_MLP, wG.FedAvg_MLP), ("G_gap_gcn", wD.Centralized_GCN, wG.FedAvg_GCN),
                   ("D_gap_mlp", wE.Centralized_MLP, wE.FedAvg_MLP), ("D_gap_gcn", wD.Centralized_GCN, wD.FedAvg_GCN)]:
    lo, hi = ci_str(c - f_); N[key] = f4((c - f_).mean()); N[f"{key}_ci"] = f"[{lo:.4f}, {hi:.4f}]"; N[f"{key}_p"] = pfmt(wil(c, f_)); N[f"{key}_w"] = str(int((c > f_).sum()))
d = wG.FedAvg_MLP - wE.FedAvg_MLP; lo, hi = ci_str(d); N["G_mlp_vs_D"] = sgn4(d.mean()); N["G_mlp_vs_D_ci"] = f"[{lo:+.4f}, {hi:+.4f}]"; N["G_mlp_vs_D_p"] = pfmt(wil(wG.FedAvg_MLP, wE.FedAvg_MLP))
N["skew_B_G"] = f4(df[(df.exp == "F") & df.label_skew.notna()].label_skew.mean())
lG = wide(df, "F", "loc_ap"); N["G_fed_gcn_local"] = f4(lG.FedAvg_GCN.mean())
gr = []
for v, lab, c in [("FedAvg_GCN", "FedAvg (GCN)", wD.Centralized_GCN), ("FedAvg_MLP", "FedAvg-MLP (control)", wE.Centralized_MLP)]:
    ref = wD if v in wD else wE
    gr.append([lab, f4(c.mean()), f4(ref[v].mean()), f4(wG[v].mean()), f4((c - ref[v]).mean()), f4((c - wG[v]).mean())])
T["gapG"] = {"header": ["Federated design", "Centralized counterpart", "Federated, Louvain (D/E)", "Federated, random (F)",
                        "Gap, Louvain", "Gap, random"], "rows": gr}

# ---- budget: ranges
FED7 = ['FedAvg_GCN', 'FedProx_GCN', 'FedProto_GCN', 'FedGNN_Hard_GCN', 'FedGNN_Soft_GCN', 'FedGNN_Soft_Virtual_GCN', 'FedGNN_Soft_Virtual_GAT']
fg = {v: wD[v].mean() - wA[v].mean() for v in FED7}
N["D_fed_gain_min"] = sgn4(min(fg.values())); N["D_fed_gain_max"] = sgn4(max(fg.values()))
N["D_fed_gain_min6"] = sgn4(min(fg[v] for v in FED7[:-1])); N["D_fed_gain_max6"] = sgn4(max(fg[v] for v in FED7[:-1])); N["D_gat_gain"] = sgn4(fg["FedGNN_Soft_Virtual_GAT"])
o, nm, pp = null_range(wD[FED7]); N.update(D_null_obs=f4(o), D_null_exp=f4(nm), D_null_p=f3(pp))
pwD7 = pairwise(wD[FED7]); N["D_pairs7_sig"] = str(int((pwD7.p_holm < .05).sum())); assert len(pwD7) == 21
rA = wA[FED7].mean().rank(ascending=False); rD = wD[FED7].mean().rank(ascending=False)
N["gat_rank_A"] = str(int(rA["FedGNN_Soft_Virtual_GAT"])); N["gat_rank_D"] = str(int(rD["FedGNN_Soft_Virtual_GAT"]))
N["D_cent_conv"] = str(int(df[(df.exp == 'D') & (df.variant == 'Centralized_GCN')].converged.sum())); N["A_cent_conv"] = str(int(df[(df.exp == 'A') & (df.variant == 'Centralized_GCN')].converged.sum()))

# ---- virtual nodes: ablation numbers under protocol A
N["A_vn_abl"] = sgn4((wA.FedGNN_Soft_Virtual_GCN - wA.FedGNN_Soft_GCN).mean()); N["A_vn_abl_p"] = pfmt(wil(wA.FedGNN_Soft_Virtual_GCN, wA.FedGNN_Soft_GCN))

# ---- temporal breakdown
TS = json.loads((DIAG / "timesteps.json").read_text())
prev = {int(t): v for t, v in TS["prevalence"].items()}
pv_val = sum(prev[t]["pos"] for t in range(35, 40)) / sum(prev[t]["n"] for t in range(35, 40))
pv_te = sum(prev[t]["pos"] for t in range(40, 50)) / sum(prev[t]["n"] for t in range(40, 50))
pv_post = sum(prev[t]["pos"] for t in range(43, 48)) / sum(prev[t]["n"] for t in range(43, 48))
pv_pre = sum(prev[t]["pos"] for t in range(40, 43)) / sum(prev[t]["n"] for t in range(40, 43))
N["prev_val"] = f"{100*pv_val:.1f}%"; N["prev_test"] = f"{100*pv_te:.1f}%"; N["prev_post"] = f"{100*pv_post:.1f}%"; N["prev_pre"] = f"{100*pv_pre:.1f}%"
N["prev_train"] = f"{100*sum(prev[t]['pos'] for t in range(1, 35))/sum(prev[t]['n'] for t in range(1, 35)):.1f}%"
def step_curve(job_prefix, variant, key="p_test_global"):
    rs = [r for r in TS["runs"] if r["job"].startswith(job_prefix) and r["run"].startswith(variant + "__")]
    steps = sorted(int(t) for t in rs[0][key]); return steps, np.array([[r[key][str(t)] if r[key][str(t)] is not None else np.nan for t in steps] for r in rs])
st, cg = step_curve("A", "Centralized_GCN"); _, fa = step_curve("A", "FedAvg_GCN"); _, so = step_curve("A", "FedGNN_Soft_GCN"); _, cm = step_curve("A", "Centralized_MLP")
post = [st.index(t) for t in range(43, 48)]; pre = [st.index(t) for t in range(40, 43)]
N["post_ap_cent"] = f3(np.nanmean(cg[:, post])); N["pre_ap_cent"] = f3(np.nanmean(cg[:, pre]))
N["post_ap_max"] = f3(max(np.nanmean(x[:, post]) for x in (cg, fa, so, cm)))
cA = df[(df.exp == "A") & (df.variant == "Centralized_GCN")]
N["lift_val"] = f"{cA.best_val_ap.mean()/pv_val:.1f}"; N["lift_test"] = f"{cA.ap.mean()/pv_te:.1f}"
cD = df[(df.exp == "D") & (df.variant == "Centralized_GCN")]
N["lift_val_D"] = f"{cD.best_val_ap.mean()/pv_val:.1f}"; N["lift_test_D"] = f"{cD.ap.mean()/pv_te:.1f}"
# validation AP is the maximum over checkpoints; AP of the selected checkpoint recomputed from saved scores
vsel = [r for r in TS["runs"] if r["job"].startswith("A") and r["run"].startswith("Centralized_GCN__")]

fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax2 = ax.twinx(); ax2.bar(st, [100 * prev[t]["pos"] / prev[t]["n"] for t in st], color="#e3e3e3", width=0.7, zorder=0)
ax2.set_ylabel("Positive rate (%)", color=GREY); ax2.tick_params(axis="y", colors=GREY); ax2.spines["right"].set_visible(True)
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
for arr_, lab, c, mk in [(cg, "Central-GCN", INK, "o"), (fa, "FedAvg", "#4f7ca8", "s"), (so, "Soft", ACC, "^"), (cm, "Central-MLP (control)", GREY, "D")]:
    ax.plot(st, np.nanmean(arr_, 0), marker=mk, ms=3.5, lw=1.2, color=c, label=lab)
ax.axvline(42.5, color=GREY, ls=":", lw=0.8); ax.text(42.6, 0.95, "dark-market\nshutdown", fontsize=7, color=GREY, va="top")
ax.set_xticks(st); ax.set_xlabel("Test time step"); ax.set_ylabel("Test AP (mean over 20 seeds)"); ax.set_ylim(0, 1.0)
ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.6, 0.98))
fig.tight_layout(); fig.savefig(FIG / "fig6_timesteps.png", dpi=300); fig.savefig(FIG / "fig6_timesteps.pdf"); plt.close(fig)

N["C_pos_sig_sa_w"] = WORDS[sig_pos]; N["C_neg_sig_sa_w"] = WORDS[sig_neg]
N["C_part_any_sig_w"] = WORDS[int(N["C_part_any_sig"])]; N["C_part_soft_sig_w"] = WORDS[int(N["C_part_soft_sig"])]
N["C_part_soft_pos_w"] = WORDS[int(N["C_part_soft_pos"])]; N["C_part_soft_neg_w"] = WORDS[int(N["C_part_soft_neg"])]
N["C_proto_sig_w"] = WORDS[int(N["C_proto_sig"])]; N["C_proto_neg_w"] = WORDS[int(N["C_proto_neg"])]; N["C_n_winners_w"] = WORDS[int(N["C_n_winners"])]
T["enc"] = {"header": ["Partition", "Soft − FedAvg (AP)", "Effective number of clients", "Most frequently top-weighted client (share of rounds)"], "rows": enc_rows}

# ---- Fig 5: null-range distribution of the six distinct convolutional designs (FedProx equals FedAvg here)
X = A8[CONV6].values; r = np.random.default_rng(0); null = []
for _ in range(20000):
    P = np.array([row[r.permutation(len(row))] for row in X]); mm_ = P.mean(0); null.append(mm_.max() - mm_.min())
null = np.array(null); o6 = float(N["A_null6_obs"]); p6 = float(N["A_null6_p"])
fig, ax = plt.subplots(figsize=(5.2, 2.8))
ax.hist(null, bins=60, color="#c9c9c9", edgecolor="white")
ax.axvline(o6, color=ACC, lw=1.6, label=f"observed spread {o6:.4f} (p = {p6:.2f})")
ax.set_xlabel("Spread of six design means under exchangeability (AP)"); ax.set_ylabel("Permutations")
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "fig5_null_range.png", dpi=300); fig.savefig(FIG / "fig5_null_range.pdf"); plt.close(fig)
N["B_abl_params"] = "32,260 against 15,300"
N["ap49_cent"] = f3(np.nanmean(cg[:, st.index(49)])); N["prev49"] = f"{100*prev[49]['pos']/prev[49]['n']:.1f}%"
N["ap42_cent"] = f3(np.nanmean(cg[:, st.index(42)])); N["prev42"] = f"{100*prev[42]['pos']/prev[42]['n']:.1f}%"

# ---- regime-dependent noise and gaps to the tabular references
N["H_fmlp_sd"] = f4(wH.FedAvg_MLP.std(ddof=1)); N["E_fmlp_sd"] = f4(wE.FedAvg_MLP.std(ddof=1))
N["H_cmlp_sd"] = f4(wH.Centralized_MLP.std(ddof=1)); N["E_cmlp_sd"] = f4(wE.Centralized_MLP.std(ddof=1))
N["H_fgcn_sd"] = f4(wH.FedAvg_GCN.std(ddof=1)); N["H_cgcn_sd"] = f4(wH.Centralized_GCN.std(ddof=1))
best_all = wH.FedAvg_MLP.mean()
N["tab_gb_gap_best"] = f4(gb.ap.mean() - best_all); N["tab_rf_gap_best"] = f4(rf.ap.mean() - best_all)
N["tab_rf93_gap_best"] = f4(rf93.ap.mean() - best_all); N["tab_gb93_gap_best"] = f4(gb93.ap.mean() - best_all)
d = wH.FedAvg_GCN - wH.Centralized_GCN; N["H_fgcn_vs_cgcn"] = sgn4(d.mean()); N["H_fgcn_vs_cgcn_p"] = pfmt(wil(wH.FedAvg_GCN, wH.Centralized_GCN))
N["f1_design_min"] = f3(mm.f1.min()); N["f1_design_max"] = f3(mm.f1.max())
N["ap_design_min"] = f4(mm.ap.min()); N["ap_design_max"] = f4(mm.ap.max())

# ------------------------------------------------------------- 25 fixed partitions
# (C0-C24, including the full 20 x 20 matrix of partition seed x training seed
# for seeds 42-61) and the temperature sweep
K25 = list(range(25)); PS = {k: (k if k < 5 else 37 + k) for k in K25}
W25 = {k: wide(df, f"C{k}") for k in K25}
fp25 = {k: vs_base(W25[k][["FedAvg_GCN", "FedProx_GCN"] + DES5]).set_index("variant") for k in K25}
N["n_part_fixed"] = "25"
N["n_runs_C_all"] = f"{int(df.exp.isin([f'C{k}' for k in K25]).sum()):,}"
N["n_runs_T"] = f"{int(df.exp.str.match(r'^T').sum()):,}"

# accounting table (the jobs named G and H are reported as protocols F and G)
acc = [("A", "Louvain, redrawn per seed", "1", "165", 10), ("B", "random node, redrawn per seed", "1", "165", 10),
       ("C0–C4", "Louvain, one fixed partition each (seeds 0–4)", "1", "165", 7),
       ("C5–C24", "Louvain, one fixed partition each (seeds 42–61)", "1", "165", 7),
       ("D", "Louvain, redrawn per seed", "2", "165", 8), ("E", "Louvain, redrawn per seed", "2", "165", 2),
       ("F", "random node, redrawn per seed", "2", "165", 2), ("G", "Louvain, redrawn per seed", "2", "93 (local only)", 4),
       ("T", "partitions of C0–C9; Soft with τ = 0.25, 1, 2", "1", "165", 1)]
def nruns(lab):
    if lab == "C0–C4": return int(df.exp.isin([f"C{k}" for k in range(5)]).sum()), 5
    if lab == "C5–C24": return int(df.exp.isin([f"C{k}" for k in range(5, 25)]).sum()), 20
    if lab == "T": return int(df.exp.str.match(r"^T").sum()), 30
    if lab == "V": return int(df.exp.str.match(r"^V\d").sum()), 10
    if lab == "Q": return int(df.exp.str.match(r"^Q\d").sum()), 10
    if lab == "P0–P24": return int(df.exp.str.match(r"^P\d").sum()), 25
    return int((df.exp == lab).sum()), 1
hrows = [[f"C{k}" if k < 25 else "", str(PS[k]), df[df.exp == f"C{k}"].config_hash.iloc[0]] for k in K25]
for lab, job in [("A", "A"), ("B", "B"), ("D", "D"), ("E", "E"), ("F", "F"), ("G", "G")]:
    hrows.append([lab, "–", df[df.exp == job].config_hash.iloc[0]])
T["hashes"] = {"header": ["Protocol", "Partition seed", "Configuration hash"], "rows": hrows}

# per-partition detail (supplement) and summary (main text)
det = []
for k in K25:
    r = [f"C{k}", str(PS[k])]
    for v in DES5:
        x = fp25[k].loc[v]; r.append(f"{x['diff']:+.4f}{'*' if x.p_holm < 0.05 else ''} ({int(x.wins)})")
    det.append(r)
T["fixed_detail"] = {"header": ["Partition", "Seed", "FedProto", "Hard", "Soft", "Soft-VN", "Soft-VN-GAT"], "rows": det}
summ = []; S25 = {}
for v in DES5:
    d = np.array([fp25[k].loc[v, "diff"] for k in K25]); ph = np.array([fp25[k].loc[v, "p_holm"] for k in K25])
    raw = np.array([fp25[k].loc[v, "p"] for k in K25]); g = holm(raw) < 0.05
    h = stats.t.ppf(0.975, 24) * d.std(ddof=1) / 5; t_ = d.mean() / (d.std(ddof=1) / 5); p_ = 2 * stats.t.sf(abs(t_), 24)
    Wd = np.array([(W25[k][v] - W25[k].FedAvg_GCN).values for k in K25]); inter = np.sqrt(max(d.var(ddof=1) - Wd.var(1, ddof=1).mean() / 20, 0))
    Wv = np.array([W25[k][v].values for k in K25]); bsd = np.sqrt(max(Wv.mean(1).var(ddof=1) - Wv.var(1, ddof=1).mean() / 20, 0))
    S25[v] = dict(mean=d.mean(), lo=d.mean() - h, hi=d.mean() + h, p=p_, pos=int(((d > 0) & (ph < .05)).sum()), neg=int(((d < 0) & (ph < .05)).sum()),
                  gpos=int(((d > 0) & g).sum()), gneg=int(((d < 0) & g).sum()), inter=inter, bsd=bsd, mn=d.min(), mx=d.max())
    q = S25[v]
    summ.append([SHORT[v], f"{q['mn']:+.4f} to {q['mx']:+.4f}", f"{q['pos']} / {q['neg']}", f"{q['gpos']} / {q['gneg']}",
                 f"{q['mean']:+.4f} [{q['lo']:+.4f}, {q['hi']:+.4f}]", pfmt(q["p"]), f4(q["inter"])])
T["fixedsum"] = {"header": ["Design vs FedAvg", "Range over partitions", "Significant better / worse (Holm within partition)",
                            "Significant better / worse (Holm over 25 partitions)", "Mean over partitions [95% CI]", "p", "Interaction SD"], "rows": summ}
for v, key in [("FedGNN_Soft_GCN", "soft"), ("FedGNN_Hard_GCN", "hard"), ("FedGNN_Soft_Virtual_GCN", "softvn"), ("FedGNN_Soft_Virtual_GAT", "gat"), ("FedProto_GCN", "proto")]:
    q = S25[v]
    N.update({f"p25_{key}": sgn4(q["mean"]), f"p25_{key}_ci": f"[{q['lo']:+.4f}, {q['hi']:+.4f}]", f"p25_{key}_p": pfmt(q["p"]),
              f"p25_{key}_pos": str(q["pos"]), f"p25_{key}_neg": str(q["neg"]), f"p25_{key}_gpos": str(q["gpos"]), f"p25_{key}_gneg": str(q["gneg"]),
              f"p25_{key}_inter": f4(q["inter"]), f"p25_{key}_bsd": f4(q["bsd"]), f"p25_{key}_min": sgn4(q["mn"]), f"p25_{key}_max": sgn4(q["mx"])})
    N[f"p25_{key}_sig_w"] = WORDS.get(q["pos"] + q["neg"], str(q["pos"] + q["neg"]))
N["p25_sa_pos"] = str(sum(S25[v]["pos"] for v in SA4)); N["p25_sa_neg"] = str(sum(S25[v]["neg"] for v in SA4))
N["p25_part_any"] = str(sum(int((fp25[k].loc[SA4].p_holm < .05).any()) for k in K25))
N["p25_soft_bsd_A"] = f4(wA.FedGNN_Soft_GCN.std(ddof=1))
N["p25_avg_bsd"] = f4(np.sqrt(max(np.array([W25[k].FedAvg_GCN.mean() for k in K25]).var(ddof=1) - np.array([W25[k].FedAvg_GCN.var(ddof=1) for k in K25]).mean() / 20, 0)))

# 20 x 20 matrix: protocol A is its diagonal
P20 = list(range(5, 25)); mat_rows = []; Mx = {}
for v in FED6:
    Mm = np.array([W25[k][v].values for k in P20]); Mx[v] = Mm
    assert np.allclose(np.diag(Mm), wA[v].values)
    off = Mm[~np.eye(20, dtype=bool)]
    within = Mm.var(1, ddof=1).mean(); between = max(Mm.mean(1).var(ddof=1) - within / 20, 0)
    mat_rows.append([SHORT[v], f4(np.diag(Mm).std(ddof=1)), f4(off.std(ddof=1)), f4(np.sqrt(within)), f4(np.sqrt(between)),
                     f4(np.diag(Mm).mean()), f4(off.mean())])
rng3 = np.random.default_rng(0); obsj = np.mean([np.log(np.diag(Mx[v]).std(ddof=1)) for v in FED6]); sims = np.empty(20000); per = {v: np.empty(20000) for v in FED6}
for i in range(20000):
    perm = rng3.permutation(20); vals = []
    for v in FED6:
        s_ = Mx[v][perm, np.arange(20)].std(ddof=1); per[v][i] = s_; vals.append(np.log(s_))
    sims[i] = np.mean(vals)
for r_, v in zip(mat_rows, FED6):
    r_.append(f3(float((per[v] <= np.diag(Mx[v]).std(ddof=1)).mean())))
T["matrix"] = {"header": ["Design", "SD on diagonal (protocol A)", "SD off diagonal", "Within-partition SD", "Between-partition SD",
                          "Mean on diagonal", "Mean off diagonal", "Quantile of diagonal SD"], "rows": mat_rows}
N["mat_joint_p"] = f3(float((sims <= obsj).mean()))
N["mat_soft_q"] = f3(float((per["FedGNN_Soft_GCN"] <= np.diag(Mx["FedGNN_Soft_GCN"]).std(ddof=1)).mean()))
N["mat_soft_off_sd"] = f4(Mx["FedGNN_Soft_GCN"][~np.eye(20, dtype=bool)].std(ddof=1)); N["mat_soft_med"] = f4(float(np.median(per["FedGNN_Soft_GCN"])))
N["mat_n_cells"] = str(20 * 20 * 7)

# composition vs Soft effect over 25 partitions
softd25 = [fp25[k].loc["FedGNN_Soft_GCN", "diff"] for k in K25]
rh = {f: stats.spearmanr([comp[PS[k]][f] for k in K25], softd25) for f in ["minvalpos", "mintrpos", "sdval", "ent"]}
N["comp25_rho_maxabs"] = f"{max(abs(r[0]) for r in rh.values()):.2f}"; N["comp25_rho_pmin"] = f"{min(r[1] for r in rh.values()):.2f}"
N["minvalpos_lo"] = str(min(comp[PS[k]]["minvalpos"] for k in K25)); N["minvalpos_hi"] = str(max(comp[PS[k]]["minvalpos"] for k in K25))

# temperature sweep on C0-C9
H2 = json.loads(gzip.open(DIAG / "history2.json.gz").read()); HH = dict(H); HH.update(H2)
def enc_T(run, part, temp):
    o = []
    for e in run["rounds"][: max(run["best_round_idx"] or 1, 1) + 1]:
        sc = e["score"]
        if not isinstance(sc, dict) or not sc:
            continue
        ks = sorted(sc, key=int); s = np.array([sc[q] for q in ks]) / temp; w = np.exp(s - s.max()); w /= w.sum()
        dw = np.array([comp[part]["ntrain"][q] for q in ks], float); w = w * dw / dw.sum(); w /= w.sum(); o.append(1 / np.sum(w ** 2))
    return float(np.mean(o))
temp_rows = []; TS_ = {}
for tag, temp in [("T025", 0.25), ("base", 0.5), ("T1", 1.0), ("T2", 2.0), ("inf", None)]:
    ds = []; Wt = []; sg = []; encs = []
    for k in range(10):
        b = W25[k]
        if tag == "base": x = b.FedGNN_Soft_GCN; pref = f"C{k}_"
        elif tag == "inf": x = b.FedProto_GCN; pref = None
        else: x = wide(df, f"{tag}C{k}").FedGNN_Soft_GCN; pref = f"{tag}C{k}/"
        d_ = x - b.FedAvg_GCN; ds.append(d_.mean()); Wt.append(d_.values); sg.append(wil(x, b.FedAvg_GCN))
        if pref:
            e_ = [enc_T(r_, PS[k], temp) for key, r_ in HH.items() if key.startswith(pref) and "/FedGNN_Soft_GCN__" in key]
            encs.append(np.mean(e_))
    ds = np.array(ds); Wt = np.array(Wt); sg = np.array(sg) < 0.05
    inter = np.sqrt(max(ds.var(ddof=1) - Wt.var(1, ddof=1).mean() / 20, 0))
    TS_[tag] = dict(enc=np.mean(encs) if encs else 10.0, inter=inter, mabs=np.abs(ds).mean(), pos=int(((ds > 0) & sg).sum()), neg=int(((ds < 0) & sg).sum()), mean=ds.mean(), ds=ds)
    lab = {"T025": "0.25", "base": "0.5 (Soft as specified)", "T1": "1", "T2": "2", "inf": "∞ (FedProto)"}[tag]
    q = TS_[tag]
    temp_rows.append([lab, "10 (data-size weights only)" if tag == "inf" else f"{q['enc']:.2f}", f4(q["inter"]), f4(q["mabs"]), f"{q['pos']} / {q['neg']}", sgn4(q["mean"])])
T["temp"] = {"header": ["Softmax temperature", "Effective number of clients", "Interaction SD (difference to FedAvg)",
                        "Mean |difference to FedAvg|", "Significant better / worse (of 10)", "Mean difference"], "rows": temp_rows}
for tag in TS_:
    N[f"t_{tag}_enc"] = f"{TS_[tag]['enc']:.1f}"; N[f"t_{tag}_inter"] = f4(TS_[tag]["inter"]); N[f"t_{tag}_mabs"] = f4(TS_[tag]["mabs"])
    N[f"t_{tag}_pos"] = str(TS_[tag]["pos"]); N[f"t_{tag}_neg"] = str(TS_[tag]["neg"]); N[f"t_{tag}_mean"] = sgn4(TS_[tag]["mean"])
N["t_mean_min"] = sgn4(min(TS_[t]["mean"] for t in TS_)); N["t_mean_max"] = sgn4(max(TS_[t]["mean"] for t in TS_))
sgn_keep = np.mean([np.sign(TS_["T025"]["ds"][k]) == np.sign(TS_["T2"]["ds"][k]) for k in range(10)])
rho_t = stats.spearmanr(TS_["T025"]["ds"], TS_["T2"]["ds"])[0]; N["t_rho_025_2"] = f"{rho_t:+.2f}"
N["t_c0"] = ", ".join(sgn4(TS_[t]["ds"][0]) for t in ["T025", "base", "T1", "T2"]); N["t_c1"] = ", ".join(sgn4(TS_[t]["ds"][1]) for t in ["T025", "base", "T1", "T2"])

# Fig 2 (fixed partitions, 25 rows) and Fig 7 (temperature dose-response)
fig, axes = plt.subplots(1, 5, figsize=(7.2, 6.4), sharey=True)
labels = [f"C{k}" for k in K25] + ["A", "mean"]
for j, v in enumerate(DES5):
    ax = axes[j]
    for i, k in enumerate(K25):
        d_ = W25[k][v] - W25[k].FedAvg_GCN; lo, hi = ci95(d_); sg_ = fp25[k].loc[v, "p_holm"] < 0.05; c = INK if k < 5 else "#4f7ca8"
        ax.errorbar(d_.mean(), i, xerr=[[d_.mean() - lo], [hi - d_.mean()]], fmt="o", ms=3.2, color=c, mfc=c if sg_ else "white", capsize=1.2, lw=0.8)
    d_ = wA[v] - wA.FedAvg_GCN; lo, hi = ci95(d_)
    ax.errorbar(d_.mean(), 25.4, xerr=[[d_.mean() - lo], [hi - d_.mean()]], fmt="s", ms=3.5, color="black", mfc="black" if pA[v] < 0.05 else "white", capsize=1.2, lw=0.9)
    q = S25[v]; ax.errorbar(q["mean"], 26.4, xerr=[[q["mean"] - q["lo"]], [q["hi"] - q["mean"]]], fmt="D", ms=3.8, color=ACC, mfc=ACC, capsize=1.2, lw=1.1)
    ax.axvline(0, color=GREY, lw=0.7); ax.set_title(SHORT[v], fontsize=8.5); ax.tick_params(axis="x", labelsize=6.5); ax.set_xlim(-0.095, 0.075)
axes[0].set_yticks(list(range(25)) + [25.4, 26.4]); axes[0].set_yticklabels(labels, fontsize=6.5); axes[0].invert_yaxis()
fig.text(0.55, 0.01, "AP difference to FedAvg (mean and 95% CI over 20 seeds; mean over 25 partitions with 95% CI over partitions)", ha="center", fontsize=7.5)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(FIG / "fig2_fixed_partitions.png", dpi=300); fig.savefig(FIG / "fig2_fixed_partitions.pdf"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
order_t = ["T025", "base", "T1", "T2", "inf"]; xt = [TS_[t]["enc"] for t in order_t]
axs[0].plot(xt, [TS_[t]["inter"] for t in order_t], "o-", color=INK)
for t, xx in zip(order_t, xt):
    axs[0].annotate({"T025": "τ=0.25", "base": "τ=0.5", "T1": "τ=1", "T2": "τ=2", "inf": "FedProto"}[t], (xx, TS_[t]["inter"]), textcoords="offset points", xytext=(4, 4), fontsize=7)
axs[0].set_xlabel("Effective number of clients"); axs[0].set_ylabel("Interaction SD of difference to FedAvg")
for k in range(10):
    axs[1].plot(range(5), [TS_[t]["ds"][k] for t in order_t], "-", color=INK if k < 5 else "#4f7ca8", lw=0.9, alpha=0.8)
axs[1].axhline(0, color=GREY, lw=0.7); axs[1].set_xticks(range(5)); axs[1].set_xticklabels(["0.25", "0.5", "1", "2", "∞"])
axs[1].set_xlabel("Softmax temperature τ"); axs[1].set_ylabel("Soft − FedAvg (AP)")
fig.tight_layout(); fig.savefig(FIG / "fig7_temperature.png", dpi=300); fig.savefig(FIG / "fig7_temperature.pdf"); plt.close(fig)

# per-partition partition-quality table for the supplement (all 25)
pq = []
for e, desc in [("A", "Louvain, redrawn per seed"), ("B", "random node, redrawn per seed")] + [(f"C{k}", f"Louvain, fixed (seed {PS[k]})") for k in K25]:
    s_ = df[(df.exp == e) & df.retention.notna()]
    pq.append([e, desc, f"{s_.retention.mean():.5f}", f"{1-s_.retention.mean():.2%}", f"{s_.label_skew.mean():.4f}",
               str(comp[PS[int(e[1:])]]["minvalpos"]) if e.startswith("C") else "–"])
T["partitions"] = {"header": ["Protocol", "Partition", "Edge retention", "Severed arcs", "Label skew", "Fewest validation positives in a client"], "rows": pq}
lab_gap = T["gapG"]["header"]; T["gapG"]["header"] = ["Federated design", "Centralized counterpart", "Federated, Louvain (D/E)", "Federated, random (F)", "Gap, Louvain", "Gap, random"]

# ------------------------------------------------------------- pairwise family
# without the FedProx duplicate, bootstrap intervals for the temperature sweep
DIST7 = [v for v in GCN8 if v != "FedProx_GCN"]
pw7 = pairwise(A8[DIST7]); N["A_pairs7_n"] = str(len(pw7)); N["A_pairs7_holm"] = str(int((pw7.p_holm < .05).sum()))
N["A_pairs7_holm_w"] = WORDS[int((pw7.p_holm < .05).sum())]
s7 = pw7[pw7.p_holm < .05]
N["A_sig_pairs7"] = "; ".join(f"{SHORT[r.a]} vs {SHORT[r.b]} (ΔAP {r['diff']:+.4f}, Holm p = {pfmt(r.p_holm)})" for _, r in s7.iterrows())
rngb = np.random.default_rng(1)
temp_rows = []
for tag in ["T025", "base", "T1", "T2", "inf"]:
    q = TS_[tag]
    # per-partition seed-level differences are needed for the bootstrap
    if tag == "base": X = np.array([(W25[k].FedGNN_Soft_GCN - W25[k].FedAvg_GCN).values for k in range(10)])
    elif tag == "inf": X = np.array([(W25[k].FedProto_GCN - W25[k].FedAvg_GCN).values for k in range(10)])
    else: X = np.array([(wide(df, f"{tag}C{k}").FedGNN_Soft_GCN - W25[k].FedAvg_GCN).values for k in range(10)])
    bs = []
    for _ in range(5000):
        rows_ = X[rngb.integers(0, 10, 10)]
        bs.append(np.sqrt(max(rows_.mean(1).var(ddof=1) - rows_.var(1, ddof=1).mean() / 20, 0)))
    lo, hi = np.quantile(bs, [0.025, 0.975])
    raw = np.array([wil(X[k] + 0, np.zeros(20)) if False else stats.wilcoxon(X[k], alternative="two-sided", method="exact").pvalue if np.any(X[k] != 0) else 1.0 for k in range(10)])
    order = np.argsort(raw); adj = np.empty(10); run = 0.0
    for r_, k_ in enumerate(order):
        run = max(run, min(1.0, (10 - r_) * raw[k_])); adj[k_] = run
    pos = int(((X.mean(1) > 0) & (adj < .05)).sum()); neg = int(((X.mean(1) < 0) & (adj < .05)).sum())
    q.update(lo=lo, hi=hi, upos=pos, uneg=neg)
    lab = {"T025": "0.25", "base": "0.5 (Soft as specified)", "T1": "1", "T2": "2", "inf": "∞ (FedProto)"}[tag]
    temp_rows.append([lab, "10 (data-size weights only)" if tag == "inf" else f"{q['enc']:.2f}", f"{q['inter']:.4f} [{lo:.4f}, {hi:.4f}]",
                      f4(q["mabs"]), f"{pos} / {neg}", sgn4(q["mean"])])
    N[f"t_{tag}_ci"] = f"[{lo:.4f}, {hi:.4f}]"
T["temp"] = {"header": ["Softmax temperature", "Effective number of clients", "Interaction SD [95% bootstrap interval over partitions]",
                        "Mean |difference to FedAvg|", "Significant better / worse (of 10, Holm over partitions)", "Mean difference"], "rows": temp_rows}
N["inter10_soft"] = f4(TS_["base"]["inter"]); N["fmlp_sd_A"] = f4(wA.FedAvg_MLP.std(ddof=1))
N["p25_soft_nonsig"] = str(25 - S25["FedGNN_Soft_GCN"]["pos"] - S25["FedGNN_Soft_GCN"]["neg"])
N["n_runs_tab"] = str(len(tb))
Xs = {}
for tag in ["T025", "T2"]:
    Xs[tag] = np.array([(wide(df, f"{tag}C{k}").FedGNN_Soft_GCN - W25[k].FedAvg_GCN).values for k in range(10)])
def isd(M): return np.sqrt(max(M.mean(1).var(ddof=1) - M.var(1, ddof=1).mean() / 20, 0))
rngp = np.random.default_rng(2); dd = []
for _ in range(5000):
    idx = rngp.integers(0, 10, 10); dd.append(isd(Xs["T025"][idx]) - isd(Xs["T2"][idx]))
lo, hi = np.quantile(dd, [0.025, 0.975]); N["t_diff_ci"] = f"[{lo:.4f}, {hi:.4f}]"; N["t_diff"] = f4(TS_["T025"]["inter"] - TS_["T2"]["inter"])

# ------------------------------------------------------------- seven distinct designs
# FedProx equals FedAvg under budget 1, so paired SDs, seed subsampling and the
# fragmentation cost use the seven distinct designs; Friedman test as a
# complement to the permutation test
sdp7 = {(a, b): (A8[a] - A8[b]).std(ddof=1) for a, b in itertools.combinations(DIST7, 2)}
sv = np.array(list(sdp7.values())); sdd = float(np.median(sv))
N.update(sigma_d=f4(sv.mean()), sigma_d_med=f4(sdd), sigma_d_min=f4(sv.min()), sigma_d_max=f4(sv.max()), n_pairs_sd=str(len(sv)),
         n_equiv_005=str(int(np.ceil((1.645 + 1.282) ** 2 * sdd ** 2 / 0.005 ** 2))),
         n_detect_003=str(int(np.ceil((1.96 + 0.8416) ** 2 * sdd ** 2 / 0.003 ** 2))),
         n_detect_003_lo=str(int(np.ceil((1.96 + 0.8416) ** 2 * sv.min() ** 2 / 0.003 ** 2))),
         n_detect_003_hi=str(int(np.ceil((1.96 + 0.8416) ** 2 * sv.max() ** 2 / 0.003 ** 2))))
M7 = A8[DIST7].values.T; full7 = stats.rankdata(-A8[DIST7].mean().values)
sub_rows = []
for k in (3, 5, 6):
    first = np.zeros(7); exact = 0; rhos = []; tot = 0
    for c in itertools.combinations(range(20), k):
        mu = M7[:, c].mean(1); first[np.argmax(mu)] += 1
        r = stats.rankdata(-mu); exact += int((r == full7).all()); rhos.append(np.corrcoef(r, full7)[0, 1]); tot += 1
    order = np.argsort(-first)
    sub_rows.append([f"{k} ({tot:,} subsets)",
                     ", ".join(f"{SHORT[DIST7[i]]} " + (f"{100*first[i]/tot:.1f}" if 100*first[i]/tot >= 0.05 else "<0.1") for i in order if first[i] > 0),
                     f"{np.mean(rhos):+.3f}", f"{100*exact/tot:.2f}%"])
    if k == 6:
        N["sub6_n_first"] = str(int((first > 0).sum())); N["sub6_n_first_w"] = WORDS[int((first > 0).sum())]
        N["sub6_exact"] = f"{100*exact/tot:.2f}"; N["sub6_rho"] = f"{np.mean(rhos):+.3f}"
T["subsample"]["rows"] = sub_rows
N["B_fed_cost_global"] = f4(np.mean([wA[v].mean() - wB[v].mean() for v in FED6]))
N["B_fed_cost_local"] = f4(np.mean([wA[v].mean() - lB[v].mean() for v in FED6]))
o7, n7, p7d = null_range(A8[DIST7]); N.update(A_spread7d=f4(o7), A_null7d_p=f3(p7d))
fr6 = stats.friedmanchisquare(*[A8[v].values for v in CONV6]); fr7 = stats.friedmanchisquare(*[A8[v].values for v in DIST7])
N.update(A_fried6_chi=f"{fr6.statistic:.2f}", A_fried6_p=f3(fr6.pvalue), A_fried7_chi=f"{fr7.statistic:.2f}", A_fried7_p=f3(fr7.pvalue))
rk = A8[DIST7].rank(axis=1, ascending=False); N["A_gat_first7"] = str(int((rk.FedGNN_Soft_Virtual_GAT == 1).sum())); N["A_gat_last7"] = str(int((rk.FedGNN_Soft_Virtual_GAT == 7).sum()))
for row in T["fixedsum"]["rows"]:
    v = [k for k in S25 if SHORT[k] == row[0]][0]
    row[5] = "<0.0001" if S25[v]["p"] < 1e-4 else f"{S25[v]['p']:.4f}"
T["runA"]["header"][-1] = "Early-stopped"
T["budget"]["header"][-1] = "Early-stopped, budget 2"

# ------------------------------------------------------------- budget 2 on fixed partitions
# P: the 25 fixed partitions at budget 2; V: separate validation halves for
# client weighting and checkpoint selection; N: protocol D without early stopping
H3 = json.loads(gzip.open(DIAG / "history3.json.gz").read())

def inter_sd(X):
    return np.sqrt(max(X.mean(1).var(ddof=1) - X.var(1, ddof=1).mean() / 20, 0))

def boot_inter(X, rng, B=5000):
    n = X.shape[0]; bs = [inter_sd(X[rng.integers(0, n, n)]) for _ in range(B)]
    return np.quantile(bs, [0.025, 0.975])

def tci(d):
    d = np.asarray(d); n = len(d); h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    p_ = 2 * stats.t.sf(abs(d.mean() / (d.std(ddof=1) / np.sqrt(n))), n - 1)
    return d.mean(), d.mean() - h, d.mean() + h, p_

# ---- P: budget 2 on the 25 fixed partitions
WP = {k: wide(df, f"P{k}") for k in K25}
rngb6 = np.random.default_rng(6)
prow = []; PB = {}
for v, key in [("FedGNN_Hard_GCN", "hard"), ("FedGNN_Soft_GCN", "soft")]:
    X1 = np.array([(W25[k][v] - W25[k].FedAvg_GCN).values for k in K25])
    X2 = np.array([(WP[k][v] - WP[k].FedAvg_GCN).values for k in K25])
    raw_w = {k: [wil(WP[k][x], WP[k].FedAvg_GCN) for x in ("FedGNN_Hard_GCN", "FedGNN_Soft_GCN")] for k in K25}
    ph = np.array([holm(raw_w[k])[0 if key == "hard" else 1] for k in K25])
    raw = np.array([wil(WP[k][v], WP[k].FedAvg_GCN) for k in K25]); g = holm(raw) < .05
    d2 = X2.mean(1); d1 = X1.mean(1)
    m, lo, hi, p_ = tci(d2); it = inter_sd(X2); blo, bhi = boot_inter(X2, rngb6)
    rho, prho = stats.spearmanr(d1, d2)
    PB[key] = dict(mean=m, lo=lo, hi=hi, p=p_, inter=it, blo=blo, bhi=bhi, pos=int(((d2 > 0) & (ph < .05)).sum()),
                   neg=int(((d2 < 0) & (ph < .05)).sum()), gpos=int(((d2 > 0) & g).sum()), gneg=int(((d2 < 0) & g).sum()),
                   mn=d2.min(), mx=d2.max(), rho=rho, prho=prho, d1=d1, d2=d2, inter1=inter_sd(X1))
    q = PB[key]
    prow.append([SHORT[v], f"{q['mn']:+.4f} to {q['mx']:+.4f}", f"{q['pos']} / {q['neg']}", f"{q['gpos']} / {q['gneg']}",
                 f"{q['mean']:+.4f} [{q['lo']:+.4f}, {q['hi']:+.4f}]", "<0.0001" if q["p"] < 1e-4 else f"{q['p']:.4f}",
                 f"{q['inter']:.4f} [{q['blo']:.4f}, {q['bhi']:.4f}]", f"{q['inter1']:.4f}", f"{q['rho']:+.2f}"])
    N.update({f"pb_{key}": sgn4(m), f"pb_{key}_ci": f"[{lo:+.4f}, {hi:+.4f}]", f"pb_{key}_p": pfmt(p_),
              f"pb_{key}_pos": str(q["pos"]), f"pb_{key}_neg": str(q["neg"]), f"pb_{key}_gpos": str(q["gpos"]), f"pb_{key}_gneg": str(q["gneg"]),
              f"pb_{key}_inter": f4(it), f"pb_{key}_inter_ci": f"[{blo:.4f}, {bhi:.4f}]", f"pb_{key}_rho": f"{rho:+.2f}",
              f"pb_{key}_prho": pfmt(prho), f"pb_{key}_min": sgn4(q["mn"]), f"pb_{key}_max": sgn4(q["mx"]),
              f"pb_{key}_c0": sgn4(d2[0]), f"pb_{key}_c1": sgn4(d2[1])})
    # paired bootstrap of the change in interaction SD between budgets
    dd_ = []
    for _ in range(5000):
        ii = rngb6.integers(0, 25, 25); dd_.append(inter_sd(X1[ii]) - inter_sd(X2[ii]))
    N[f"pb_{key}_dinter"] = f4(q["inter1"] - it); N[f"pb_{key}_dinter_ci"] = "[{:.4f}, {:.4f}]".format(*np.quantile(dd_, [0.025, 0.975]))
T["fixed_b2"] = {"header": ["Design vs FedAvg", "Range over partitions", "Significant better / worse (Holm within partition)",
                            "Significant better / worse (Holm over 25 partitions)", "Mean over partitions [95% CI]", "p",
                            "Interaction SD [95% bootstrap interval]", "Interaction SD, budget 1", "Rank correlation with budget 1"], "rows": prow}
fa1 = np.mean([W25[k].FedAvg_GCN.mean() for k in K25]); fa2 = np.mean([WP[k].FedAvg_GCN.mean() for k in K25])
N["pb_fedavg_b1"] = f4(fa1); N["pb_fedavg_b2"] = f4(fa2)
N["pb_es_frac"] = f"{100 * df[df.exp.str.match(r'^P\d')].converged.mean():.0f}"
N["pb_bestround"] = f"{df[df.exp.str.match(r'^P\d')].best_round.mean():.0f}"
# figure: partition-level differences, budget 1 against budget 2
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, key, lab in zip(axs, ["hard", "soft"], ["Hard − FedAvg", "Soft − FedAvg"]):
    q = PB[key]; lim = max(np.abs(q["d1"]).max(), np.abs(q["d2"]).max()) * 1.1
    ax.axhline(0, color="#999", lw=0.6); ax.axvline(0, color="#999", lw=0.6); ax.plot([-lim, lim], [-lim, lim], ls=":", color="#999", lw=0.8)
    ax.scatter(q["d1"][:5], q["d2"][:5], s=22, color="#1f3b57", label="C0–C4"); ax.scatter(q["d1"][5:], q["d2"][5:], s=22, color="#7fa3c4", label="C5–C24")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xlabel(f"{lab}, budget 1 (AP)"); ax.set_ylabel(f"{lab}, budget 2 (AP)")
    ax.set_title(f"{lab}: ρ = {q['rho']:+.2f}", fontsize=9)
axs[0].legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "figures" / "fig8_budget_partitions.png", dpi=300); plt.close(fig)

# ---- V: separate validation halves, C0-C9
WV = {k: wide(df, f"V{k}") for k in range(10)}
XV = np.array([(WV[k].FedGNN_Soft_GCN - WV[k].FedAvg_GCN).values for k in range(10)])
XB = np.array([(W25[k].FedGNN_Soft_GCN - W25[k].FedAvg_GCN).values for k in range(10)])
rawv = np.array([wil(WV[k].FedGNN_Soft_GCN, WV[k].FedAvg_GCN) for k in range(10)]); gv = holm(rawv) < .05
dV = XV.mean(1); dB = XB.mean(1)
iv = inter_sd(XV); ivlo, ivhi = boot_inter(XV, rngb6)
dd_ = []
for _ in range(5000):
    ii = rngb6.integers(0, 10, 10); dd_.append(inter_sd(XB[ii]) - inter_sd(XV[ii]))
encv = []
for k in range(10):
    e_ = [enc_T(r_, PS[k], 0.5) for key_, r_ in H3.items() if key_.startswith(f"V{k}_") and "/FedGNN_Soft_GCN__" in key_]
    encv.append(np.mean(e_))
rv, prv = stats.spearmanr(dB, dV)
m, lo, hi, p_ = tci(dV)
N.update(v_inter=f4(iv), v_inter_ci=f"[{ivlo:.4f}, {ivhi:.4f}]", v_base_inter=f4(inter_sd(XB)),
         v_dinter=f4(inter_sd(XB) - iv), v_dinter_ci="[{:.4f}, {:.4f}]".format(*np.quantile(dd_, [0.025, 0.975])),
         v_enc=f"{np.mean(encv):.2f}", v_pos=str(int(((dV > 0) & gv).sum())), v_neg=str(int(((dV < 0) & gv).sum())),
         v_mean=sgn4(m), v_ci=f"[{lo:+.4f}, {hi:+.4f}]", v_mabs=f4(np.abs(dV).mean()), v_rho=f"{rv:+.2f}", v_prho=pfmt(prv),
         v_c0=sgn4(dV[0]), v_c1=sgn4(dV[1]),
         v_fedavg_shift=f4(np.mean([np.abs(WV[k].FedAvg_GCN - W25[k].FedAvg_GCN).mean() for k in range(10)])))
base_q = TS_["base"]
T["valsplit"] = {"header": ["Validation data", "Effective number of clients", "Interaction SD [95% bootstrap interval]",
                            "Mean |difference to FedAvg|", "Significant better / worse (of 10, Holm over partitions)", "Mean difference"],
                 "rows": [["shared (as specified)", f"{base_q['enc']:.2f}", f"{base_q['inter']:.4f} {N['t_base_ci']}", f4(base_q["mabs"]),
                           f"{base_q['upos']} / {base_q['uneg']}", sgn4(base_q["mean"])],
                          ["separate halves for weighting and selection", N["v_enc"], f"{iv:.4f} [{ivlo:.4f}, {ivhi:.4f}]", N["v_mabs"],
                           f"{N['v_pos']} / {N['v_neg']}", N["v_mean"]]]}

# ---- N: no early stopping
wD, wE, wN = wide(df, "D"), wide(df, "E"), wide(df, "N")
for enc_, cen, fed, key in [("GCN", wD.Centralized_GCN, wN.FedAvg_GCN, "gcn"), ("MLP", wE.Centralized_MLP, wN.FedAvg_MLP, "mlp")]:
    d = cen - fed; m, lo, hi, _ = tci(d)
    es = wD.FedAvg_GCN if key == "gcn" else wE.FedAvg_MLP
    N.update({f"n_gap_{key}": f4(m), f"n_gap_{key}_ci": f"[{lo:+.4f}, {hi:+.4f}]", f"n_gap_{key}_p": pfmt(wil(cen, fed)),
              f"n_fed_{key}": f4(fed.mean()), f"n_fed_{key}_es": f4(es.mean()), f"n_fed_{key}_chg": sgn4((fed - es).mean()),
              f"n_fed_{key}_chg_p": pfmt(wil(fed, es))})
nb = df[df.exp == "N"]
N["n_best_gcn"] = f"{nb[nb.variant == 'FedAvg_GCN'].best_round.mean():.0f}"; N["n_best_mlp"] = f"{nb[nb.variant == 'FedAvg_MLP'].best_round.mean():.0f}"
dd0 = df[(df.exp == "D") & (df.variant == "FedAvg_GCN")]; ee0 = df[(df.exp == "E") & (df.variant == "FedAvg_MLP")]
N["n_best_gcn_es"] = f"{dd0.best_round.mean():.0f}"; N["n_best_mlp_es"] = f"{ee0.best_round.mean():.0f}"

# ---- accounting: protocols V, P and N
acc += [("V", "partitions of C0–C9; separate validation data for weighting and selection", "1", "165", 2),
        ("N", "Louvain, redrawn per seed; no early stopping", "2", "165", 2),
        ("P0–P24", "partitions of C0–C24", "2", "165", 3)]
N["n_runs_rev3"] = f"{int(VALSPLIT.sum()):,}"

# ------------------------------------------------------------- full-budget regime
# budget 2 without early stopping: designs (M + N), fixed partitions C0-C9 (Q),
# random partition (R), local features (L)
H4 = json.loads(gzip.open(DIAG / "history4.json.gz").read())
wM, wN, wR, wL = wide(df, "M"), wide(df, "N"), wide(df, "R"), wide(df, "L")
wD, wE, wF, wG = wide(df, "D"), wide(df, "E"), wide(df, "F"), wide(df, "G")
FULL = wM.join(wN[["FedAvg_GCN", "FedAvg_MLP"]]).join(wD[["Centralized_GCN"]]).join(wE[["Centralized_MLP"]])
FED7 = ["FedAvg_GCN", "FedProx_GCN", "FedProto_GCN", "FedGNN_Hard_GCN", "FedGNN_Soft_GCN", "FedGNN_Soft_Virtual_GCN", "FedGNN_Soft_Virtual_GAT"]
ES = wD.join(wE[["Centralized_MLP", "FedAvg_MLP"]])
pf = holm([wil(FULL[v], FULL.FedAvg_GCN) for v in ["Centralized_GCN"] + FED7[1:]])
pfd = dict(zip(["Centralized_GCN"] + FED7[1:], pf))
best_full = {v: df[(df.exp.isin(["M", "N"])) & (df.variant == v)].best_round.mean() for v in FED7 + ["FedAvg_MLP"]}
best_full["Centralized_GCN"] = df[(df.exp == "D") & (df.variant == "Centralized_GCN")].best_round.mean()
best_full["Centralized_MLP"] = df[(df.exp == "E") & (df.variant == "Centralized_MLP")].best_round.mean()
rows = []
for v in ["Centralized_GCN"] + FED7 + ["Centralized_MLP", "FedAvg_MLP"]:
    x = FULL[v]; lo, hi = ci95(x)
    dlt = "–" if v == "FedAvg_GCN" else (f"{(x - FULL.FedAvg_GCN).mean():+.4f} ({pfmt(pfd[v])})" if v in pfd else
                                         f"{(x - FULL.FedAvg_MLP).mean():+.4f} vs FedAvg-MLP" if v == "Centralized_MLP" else
                                         f"{(x - FULL.FedAvg_GCN).mean():+.4f} vs FedAvg")
    lab = SHORT[v] + (" (control)" if "MLP" in v else "")
    rows.append([lab, f4(wA[v].mean()), f4(ES[v].mean()), f"{x.mean():.4f} [{lo:.4f}, {hi:.4f}]", f4(x.std(ddof=1)),
                 f"{best_full[v]:.0f}", dlt])
T["full"] = {"header": ["Design", "Budget 1 (A)", "Budget 2, early stopping (D, E)", "Budget 2, full budget [95% CI]",
                        "SD, full budget", "Best round or step, full budget", "Difference to FedAvg, full budget (Holm p)"], "rows": rows}
o, nm, pp = null_range(FULL[FED7]); fr = stats.friedmanchisquare(*[FULL[v].values for v in FED7])
D8 = FED7 + ["Centralized_GCN"]
pw = [(a, b, wil(FULL[a], FULL[b])) for a, b in itertools.combinations(D8, 2)]; ph = holm([p for _, _, p in pw])
N.update(F_spread=f4(o), F_null_p="<0.0001" if pp < 1e-4 else f3(pp), F_fried_chi=f"{fr.statistic:.1f}", F_fried_p="<0.0001" if fr.pvalue < 1e-4 else f3(fr.pvalue),
         F_pairs_sig=str(int((np.array(ph) < .05).sum())), F_pairs_n=str(len(pw)),
         F_fedavg=f4(FULL.FedAvg_GCN.mean()), F_fedprox=f4(FULL.FedProx_GCN.mean()), F_proto=f4(FULL.FedProto_GCN.mean()),
         F_hard=f4(FULL.FedGNN_Hard_GCN.mean()), F_soft=f4(FULL.FedGNN_Soft_GCN.mean()), F_softvn=f4(FULL.FedGNN_Soft_Virtual_GCN.mean()),
         F_gat=f4(FULL.FedGNN_Soft_Virtual_GAT.mean()), F_cent=f4(FULL.Centralized_GCN.mean()),
         F_d_proto=sgn4((FULL.FedProto_GCN - FULL.FedAvg_GCN).mean()), F_d_hard=sgn4((FULL.FedGNN_Hard_GCN - FULL.FedAvg_GCN).mean()),
         F_d_soft=sgn4((FULL.FedGNN_Soft_GCN - FULL.FedAvg_GCN).mean()), F_d_softvn=sgn4((FULL.FedGNN_Soft_Virtual_GCN - FULL.FedAvg_GCN).mean()),
         F_d_gat=sgn4((FULL.FedGNN_Soft_Virtual_GAT - FULL.FedAvg_GCN).mean()), F_d_prox=sgn4((FULL.FedProx_GCN - FULL.FedAvg_GCN).mean()),
         F_p_prox=pfmt(pfd["FedProx_GCN"]), F_p_proto=pfmt(pfd["FedProto_GCN"]), F_pmax_sel=pfmt(max(pfd[v] for v in FED7[3:])),
         F_d_cent=sgn4((FULL.Centralized_GCN - FULL.FedAvg_GCN).mean()), F_p_cent=pfmt(pfd["Centralized_GCN"]),
         F_best_fedavg=f"{best_full['FedAvg_GCN']:.0f}", F_best_sel_lo=f"{min(best_full[v] for v in FED7[3:]):.0f}",
         F_best_sel_hi=f"{max(best_full[v] for v in FED7[3:]):.0f}", F_best_proto=f"{best_full['FedProto_GCN']:.0f}",
         F_sd_lo=f4(min(FULL[v].std(ddof=1) for v in FED7)), F_sd_hi=f4(max(FULL[v].std(ddof=1) for v in FED7)),
         F_sd_fedavg=f4(FULL.FedAvg_GCN.std(ddof=1)), F_sd_soft=f4(FULL.FedGNN_Soft_GCN.std(ddof=1)),
         ES_best_fedavg=f"{df[(df.exp == 'D') & (df.variant == 'FedAvg_GCN')].best_round.mean():.0f}",
         ES_stop_fedavg=f"{df[(df.exp == 'D') & (df.variant == 'FedAvg_GCN')].rounds.mean():.0f}",
         ES_fedavg=f4(wD.FedAvg_GCN.mean()), F_gain_fedavg=sgn4((FULL.FedAvg_GCN - wD.FedAvg_GCN).mean()))
rk = FULL[FED7].rank(axis=1, ascending=False); N["F_avg_first"] = str(int((rk.FedAvg_GCN <= 2).sum()))
# graph vs graph-free under full budget
dd = FULL.FedAvg_MLP - FULL.FedAvg_GCN; lo, hi = ci95(dd)
N.update(F_mlp_gcn=sgn4(dd.mean()), F_mlp_gcn_ci=f"[{lo:+.4f}, {hi:+.4f}]", F_mlp_gcn_p=pfmt(wil(FULL.FedAvg_MLP, FULL.FedAvg_GCN)),
         F_mlp_gcn_w=str(int((FULL.FedAvg_GCN > FULL.FedAvg_MLP).sum())), F_fedmlp=f4(FULL.FedAvg_MLP.mean()))

# fixed partitions under three regimes (C0-C9)
regs = [("Budget 1 (C)", "C"), ("Budget 2, early stopping (P)", "P"), ("Budget 2, full budget (Q)", "Q")]
rrows = []; RG = {}
for lab, pre in regs:
    Wk = [wide(df, f"{pre}{k}") for k in range(10)]
    X = np.array([(w.FedGNN_Soft_GCN - w.FedAvg_GCN).values for w in Wk]); d = X.mean(1)
    raw = np.array([wil(w.FedGNN_Soft_GCN, w.FedAvg_GCN) for w in Wk]); g = holm(raw) < .05
    m, lo, hi, _ = tci(d); it = inter_sd(X); blo, bhi = boot_inter(X, rngb6)
    RG[pre] = dict(d=d, X=X, fa=np.mean([w.FedAvg_GCN.mean() for w in Wk]))
    rho = stats.spearmanr(RG["C"]["d"], d)[0] if pre != "C" else 1.0
    rrows.append([lab, f4(RG[pre]["fa"]), f"{int(((d > 0) & g).sum())} / {int(((d < 0) & g).sum())}", f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]",
                  f"{it:.4f} [{blo:.4f}, {bhi:.4f}]", sgn4(d[0]), sgn4(d[1]), "–" if pre == "C" else f"{rho:+.2f}"])
    N.update({f"rg{pre}_fa": f4(RG[pre]["fa"]), f"rg{pre}_pos": str(int(((d > 0) & g).sum())), f"rg{pre}_neg": str(int(((d < 0) & g).sum())),
              f"rg{pre}_mean": sgn4(m), f"rg{pre}_ci": f"[{lo:+.4f}, {hi:+.4f}]", f"rg{pre}_inter": f4(it), f"rg{pre}_inter_ci": f"[{blo:.4f}, {bhi:.4f}]",
              f"rg{pre}_c0": sgn4(d[0]), f"rg{pre}_c1": sgn4(d[1]), f"rg{pre}_min": sgn4(d.min()), f"rg{pre}_max": sgn4(d.max()),
              f"rg{pre}_rho": f"{rho:+.2f}", f"rg{pre}_nneg": str(int((d < 0).sum()))})
T["regimes"] = {"header": ["Training", "FedAvg AP (mean over partitions)", "Soft better / worse (Holm, of 10)",
                           "Soft − FedAvg, mean [95% CI]", "Interaction SD [95% bootstrap interval]", "C0", "C1",
                           "Rank correlation with budget 1"], "rows": rrows}
encq = []
for k in range(10):
    e_ = [enc_T(r_, PS[k], 0.5) for key_, r_ in H4.items() if key_.startswith(f"Q{k}_") and "/FedGNN_Soft_GCN__" in key_]
    encq.append(np.mean(e_))
N["rgQ_enc"] = f"{np.mean(encq):.2f}"
N["rgQ_best_soft"] = f"{df[df.exp.str.match(r'^Q\d') & (df.variant == 'FedGNN_Soft_GCN')].best_round.mean():.0f}"
N["rgQ_best_avg"] = f"{df[df.exp.str.match(r'^Q\d') & (df.variant == 'FedAvg_GCN')].best_round.mean():.0f}"

# federated-centralized gap, early stopping vs full budget
grow = []
spec = [("Louvain, 165 features", "GCN", wD.Centralized_GCN, wD.FedAvg_GCN, wN.FedAvg_GCN, "lg"),
        ("Louvain, 165 features", "MLP (control)", wE.Centralized_MLP, wE.FedAvg_MLP, wN.FedAvg_MLP, "lm"),
        ("random, 165 features", "GCN", wD.Centralized_GCN, wF.FedAvg_GCN, wR.FedAvg_GCN, "rg"),
        ("random, 165 features", "MLP (control)", wE.Centralized_MLP, wF.FedAvg_MLP, wR.FedAvg_MLP, "rm"),
        ("Louvain, 93 local features", "GCN", wG.Centralized_GCN, wG.FedAvg_GCN, wL.FedAvg_GCN, "xg"),
        ("Louvain, 93 local features", "MLP (control)", wG.Centralized_MLP, wG.FedAvg_MLP, wL.FedAvg_MLP, "xm")]
for part, enc_, cen, es, fu, key in spec:
    g1 = cen - es; g2 = cen - fu; a1, b1 = ci95(g1); a2, b2 = ci95(g2)
    grow.append([part, enc_, f4(cen.mean()), f4(es.mean()), f"{g1.mean():+.4f} [{a1:+.4f}, {b1:+.4f}]", f4(fu.mean()),
                 f"{g2.mean():+.4f} [{a2:+.4f}, {b2:+.4f}]", pfmt(wil(cen, fu))])
    N.update({f"gap_{key}_cen": f4(cen.mean()), f"gap_{key}_es": sgn4(g1.mean()), f"gap_{key}_es_ci": f"[{a1:+.4f}, {b1:+.4f}]",
              f"gap_{key}_full": sgn4(g2.mean()), f"gap_{key}_full_ci": f"[{a2:+.4f}, {b2:+.4f}]", f"gap_{key}_full_p": pfmt(wil(cen, fu)),
              f"gap_{key}_fedfull": f4(fu.mean()), f"gap_{key}_fedes": f4(es.mean())})
T["gap_all"] = {"header": ["Partition and features", "Encoder", "Centralized AP", "FedAvg AP, early stopping", "Gap, early stopping [95% CI]",
                           "FedAvg AP, full budget", "Gap, full budget [95% CI]", "p, full budget"], "rows": grow}
dd = wR.FedAvg_MLP - wN.FedAvg_MLP; lo, hi = ci95(dd)
N.update(F_rand_vs_louv_mlp=sgn4(dd.mean()), F_rand_vs_louv_mlp_ci=f"[{lo:+.4f}, {hi:+.4f}]", F_rand_vs_louv_mlp_p=pfmt(wil(wR.FedAvg_MLP, wN.FedAvg_MLP)))
dd = wL.FedAvg_MLP - wL.FedAvg_GCN; lo, hi = ci95(dd)
N.update(L_mlp_gcn=sgn4(dd.mean()), L_mlp_gcn_ci=f"[{lo:+.4f}, {hi:+.4f}]")

# figure: three training regimes
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]; xs = np.arange(len(FED7))
for j, (lab, W_, col, mk) in enumerate([("budget 1", wA, "#9db4c8", "o"), ("budget 2, early stopping", wD, "#4f7393", "s"),
                                        ("budget 2, full budget", FULL, "#1f3b57", "D")]):
    mu = [W_[v].mean() for v in FED7]; ci = [ci95(W_[v]) for v in FED7]
    ax.errorbar(xs + (j - 1) * 0.22, mu, yerr=[[m - c[0] for m, c in zip(mu, ci)], [c[1] - m for m, c in zip(mu, ci)]],
                fmt=mk, color=col, ms=4, capsize=2, lw=1, label=lab)
ax.set_xticks(xs); ax.set_xticklabels([SHORT[v] for v in FED7], rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Test AP (mean, 95% CI)"); ax.legend(fontsize=7, frameon=False, loc="lower left"); ax.set_title("Federated designs, redrawn partitions", fontsize=9)
ax = axs[1]
for k in range(10):
    ys = [RG[p]["d"][k] for p in ("C", "P", "Q")]
    ax.plot([0, 1, 2], ys, color="#1f3b57" if k < 5 else "#7fa3c4", lw=1, marker="o", ms=2.5)
ax.axhline(0, color="#999", lw=0.6); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["budget 1", "budget 2,\nearly stop", "budget 2,\nfull budget"], fontsize=8)
ax.set_ylabel("Soft − FedAvg (AP)"); ax.set_title("Fixed partitions C0–C9", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "figures" / "fig9_regimes.png", dpi=300); fig.savefig(OUT / "figures" / "fig9_regimes.pdf"); plt.close(fig)

# accounting and hashes
acc += [("Q", "partitions of C0–C9; no early stopping", "2", "165", 2), ("R", "random node, redrawn per seed; no early stopping", "2", "165", 2),
        ("L", "Louvain, redrawn per seed; no early stopping", "2", "93 (local only)", 2), ("M", "Louvain, redrawn per seed; no early stopping", "2", "165", 6)]
for lab in ["N", "R", "L", "M"]:
    T["hashes"]["rows"].append([lab, "–", df[df.exp == lab].config_hash.iloc[0]])
for pre in ["V", "P", "Q"]:
    for k in (range(10) if pre != "P" else K25):
        T["hashes"]["rows"].append([f"{pre}{k}", str(PS[k]), df[df.exp == f"{pre}{k}"].config_hash.iloc[0]])
N["n_runs_old"] = f"{int(OLD.sum()):,}"
for k_ in ["F_d_proto", "F_d_hard", "F_d_soft", "F_d_softvn", "F_d_gat", "F_d_cent", "rgQ_max", "rgQ_min", "rgQ_c1", "gap_lg_es", "F_mlp_gcn"]:
    N[k_ + "_abs"] = f"{abs(float(N[k_])):.4f}"
# best neural design with all features (fully trained federated)
bv = max(FED7, key=lambda v: FULL[v].mean()); bsel = df[df.exp.isin(["M", "N"]) & (df.variant == bv)]
T["tabular"]["rows"][3] = [f"Best neural design, 165 features ({SHORT[bv]}, budget 2, full budget)", "20", f4(FULL[bv].mean()), "–",
                           f4(bsel.auc.mean()), f3(bsel.tun_f1.mean())]
gbap = tb[(tb.model == "hist_gradient_boosting") & (tb.fs == "all_165")].ap.mean(); rfap = tb[(tb.model == "random_forest") & (tb.fs == "all_165")].ap.mean()
N["tab_gap_nn"] = f4(gbap - FULL[bv].mean()); N["tab_gap_rf"] = f4(rfap - FULL[bv].mean()); N["tab_best165_name"] = SHORT[bv]; N["tab_best165"] = f4(FULL[bv].mean())

# ------------------------------------------------------------- 600-round training
# protocol W (600 rounds without early stopping), last column of Table 6, and
# the factor varied by each protocol in the accounting table
wW = wide(df, "W"); W4 = ["FedAvg_GCN", "FedProto_GCN", "FedGNN_Hard_GCN", "FedGNN_Soft_GCN"]
ph = dict(zip(W4[1:], holm([wil(wW[v], wW.FedAvg_GCN) for v in W4[1:]])))
wrows = []
for v in W4:
    x = wW[v]; lo, hi = ci95(x); ch = x - FULL[v]
    b3 = df[df.exp.isin(["M", "N"]) & (df.variant == v)].best_round.mean(); b6 = df[(df.exp == "W") & (df.variant == v)].best_round.mean()
    wrows.append([SHORT[v], f4(FULL[v].mean()), f"{x.mean():.4f} [{lo:.4f}, {hi:.4f}]", f"{ch.mean():+.4f} ({pfmt(wil(x, FULL[v]))})",
                  f"{b3:.0f}", f"{b6:.0f}", "–" if v == "FedAvg_GCN" else f"{(x - wW.FedAvg_GCN).mean():+.4f} ({pfmt(ph[v])})"])
    k = SHORT[v].lower().replace("-", "")
    N.update({f"W_{k}": f4(x.mean()), f"W_{k}_chg": sgn4(ch.mean()), f"W_{k}_chg_p": pfmt(wil(x, FULL[v])), f"W_{k}_best": f"{b6:.0f}"})
    if v != "FedAvg_GCN":
        N.update({f"W_{k}_d": sgn4((x - wW.FedAvg_GCN).mean()), f"W_{k}_d_abs": f"{abs((x - wW.FedAvg_GCN).mean()):.4f}", f"W_{k}_p": pfmt(ph[v])})
T["w600"] = {"header": ["Design", "AP, 300 rounds", "AP, 600 rounds [95% CI]", "Change (p)", "Best round, 300", "Best round, 600",
                        "Difference to FedAvg, 600 rounds (Holm p)"], "rows": wrows}
frW = stats.friedmanchisquare(*[wW[v].values for v in W4]); N["W_fried_p"] = f3(frW.pvalue)
N["W_gap_cent"] = sgn4((wD.Centralized_GCN - wW.FedAvg_GCN).mean()); N["W_gap_cent_p"] = pfmt(wil(wD.Centralized_GCN, wW.FedAvg_GCN))
# Table 6: one sign convention, difference to FedAvg (GCN) for every row
for row, v in zip(T["full"]["rows"], ["Centralized_GCN"] + FED7 + ["Centralized_MLP", "FedAvg_MLP"]):
    if v in ("Centralized_MLP", "FedAvg_MLP"):
        row[-1] = f"{(FULL[v] - FULL.FedAvg_GCN).mean():+.4f}"
T["full"]["header"][-1] = "Difference to FedAvg (GCN), full budget (Holm p)"
acc += [("W", "Louvain, redrawn per seed; 600 rounds, no early stopping", "2 (600 rounds)", "165", 4)]
FACT = {"A": "designs (reference)", "B": "random partition", "C0–C4": "fixed partition", "C5–C24": "fixed partition (20 × 20 matrix)",
        "D": "budget", "E": "budget (graph-free)", "F": "random partition, budget", "G": "local features", "T": "softmax temperature",
        "V": "separate validation data", "N": "no early stopping", "P0–P24": "fixed partition, budget", "Q": "fixed partition, no early stopping",
        "R": "random partition, no early stopping", "L": "local features, no early stopping", "M": "no early stopping", "W": "600 rounds"}
T["hashes"]["rows"].append(["W", "–", df[df.exp == "W"].config_hash.iloc[0]])

# ------------------------------------------------------------- patience of early stopping
# protocols S50 and S100, logistic-regression convergence check, final accounting
prow = []
for lab, src in [("20", None), ("50", "S50"), ("100", "S100"), ("none (full budget)", "N")]:
    for enc, cen, col, srcd in [("GCN", wD.Centralized_GCN, "FedAvg_GCN", "D"), ("MLP (control)", wE.Centralized_MLP, "FedAvg_MLP", "E")]:
        e_ = src or srcd; w = wide(df, e_)[col]; s = df[(df.exp == e_) & (df.variant == col)]
        g = cen - w; lo, hi = ci95(g)
        prow.append([lab, enc, f4(w.mean()), f"{g.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]", f"{s.best_round.mean():.0f}", f"{s.rounds.mean():.0f}"])
        k = {"20": "p20", "50": "p50", "100": "p100", "none (full budget)": "pinf"}[lab] + ("_g" if enc == "GCN" else "_m")
        N.update({f"{k}_fed": f4(w.mean()), f"{k}_gap": sgn4(g.mean()), f"{k}_gap_ci": f"[{lo:+.4f}, {hi:+.4f}]", f"{k}_p": pfmt(wil(cen, w)),
                  f"{k}_best": f"{s.best_round.mean():.0f}", f"{k}_stop": f"{s.rounds.mean():.0f}"})
T["patience"] = {"header": ["Patience (rounds)", "Encoder", "FedAvg AP", "Gap to centralized [95% CI]", "Best round", "Last round run"], "rows": prow}
LR = json.loads((DIAG / "logreg.json").read_text())["runs"]
base = [r for r in LR if r["features"] == "all_165" and r["C"] == 1.0 and r["class_weight"] == "none"][0]
N.update(lr_iter=str(base["n_iter"]), lr_iter_max=str(max(r["n_iter"] for r in LR)), lr_warn=str(sum(r["convergence_warning"] for r in LR)),
         lr_ap_lo=f"{min(r['test']['ap'] for r in LR if r['features'] == 'all_165'):.3f}", lr_ap_hi=f"{max(r['test']['ap'] for r in LR if r['features'] == 'all_165'):.3f}")
acc += [("S50", "Louvain, redrawn per seed; patience 50", "2", "165", 2), ("S100", "Louvain, redrawn per seed; patience 100", "2", "165", 2)]
FACT.update({"S50": "patience of early stopping", "S100": "patience of early stopping"})
# ------------------------------------------------------------- checkpoint selection on the aggregated model
# Z20, Z50, Z100, ZN, ZM repeat D/E, S50, S100, N and M with fed.select_on_aggregate;
# the training trajectory is unchanged, only the selected (and, with early stopping, the last) round
def runs_of(exp, v):
    return df[(df.exp == exp) & (df.variant == v)].set_index("seed").loc[SEEDS]

sel_rows, ch, chp = [], [], []
cen = {"FedAvg_GCN": wD.Centralized_GCN, "FedAvg_MLP": wE.Centralized_MLP}
for lab, z, srcs in [("20", "Z20", {"FedAvg_GCN": "D", "FedAvg_MLP": "E"}), ("50", "Z50", "S50"), ("100", "Z100", "S100"),
                     ("none (full budget)", "ZN", "N")]:
    for v, enc in [("FedAvg_GCN", "GCN"), ("FedAvg_MLP", "MLP (control)")]:
        s = srcs[v] if isinstance(srcs, dict) else srcs
        a, o = runs_of(z, v), runs_of(s, v)
        d = a.ap - o.ap; lo, hi = ci95(d); pz = wil(a.ap, o.ap); ch.append(d.mean()); chp.append(pz)
        g = cen[v] - a.ap; glo, ghi = ci95(g)
        sel_rows.append([lab, enc, f4(o.ap.mean()), f4(a.ap.mean()), f"{d.mean():+.4f}", pfmt(pz),
                         f"{o.best_round.mean():.0f} / {a.best_round.mean():.0f}", f"{o.rounds.mean():.0f} / {a.rounds.mean():.0f}",
                         f"{g.mean():+.4f} [{glo:+.4f}, {ghi:+.4f}]"])
        k = {"20": "z20", "50": "z50", "100": "z100", "none (full budget)": "zn"}[lab] + ("_g" if enc == "GCN" else "_m")
        N.update({f"{k}_ap": f4(a.ap.mean()), f"{k}_d": sgn4(d.mean()), f"{k}_p": pfmt(pz), f"{k}_gap": sgn4(g.mean()),
                  f"{k}_gap_ci": f"[{glo:+.4f}, {ghi:+.4f}]", f"{k}_gap_p": pfmt(wil(cen[v], a.ap)),
                  f"{k}_stop": f"{a.rounds.mean():.0f}", f"{k}_stop_local": f"{o.rounds.mean():.0f}"})
T["aggsel"] = {"header": ["Patience (rounds)", "Encoder", "FedAvg AP, local-model selection", "FedAvg AP, aggregate selection",
                          "Change", "p", "Best round, local / aggregate", "Last round, local / aggregate",
                          "Gap to centralized, aggregate selection [95% CI]"], "rows": sel_rows}
chh = holm(chp)
N.update(z_maxabs_change=f4(max(abs(x) for x in ch)), z_min_p_holm=pfmt(min(chh)), z_n_tests=str(len(ch)))

ZF = wide(df, "ZM").join(wide(df, "ZN")[["FedAvg_GCN"]]); LF = wide(df, "M").join(wide(df, "N")[["FedAvg_GCN"]])
FED7Z = ["FedAvg_GCN", "FedProx_GCN", "FedProto_GCN", "FedGNN_Hard_GCN", "FedGNN_Soft_GCN", "FedGNN_Soft_Virtual_GCN", "FedGNN_Soft_Virtual_GAT"]
others = FED7Z[1:]
phz = dict(zip(others, holm([wil(ZF[v], ZF.FedAvg_GCN) for v in others])))
phl = dict(zip(others, holm([wil(LF[v], LF.FedAvg_GCN) for v in others])))
same_sel = {v: int((runs_of("ZM" if v != "FedAvg_GCN" else "ZN", v).best_round == runs_of("M" if v != "FedAvg_GCN" else "N", v).best_round).sum())
            for v in FED7Z}
full_rows = []
for v in FED7Z:
    gain = ZF[v] - LF[v]
    full_rows.append([SHORT[v], f4(LF[v].mean()), f4(ZF[v].mean()), sgn4(gain.mean()),
                      "–" if v == "FedAvg_GCN" else f"{(LF[v] - LF.FedAvg_GCN).mean():+.4f} ({pfmt(phl[v])})",
                      "–" if v == "FedAvg_GCN" else f"{(ZF[v] - ZF.FedAvg_GCN).mean():+.4f} ({pfmt(phz[v])})", f"{same_sel[v]} / 20"])
T["aggsel_full"] = {"header": ["Design", "AP, local-model selection", "AP, aggregate selection", "Change",
                               "Difference to FedAvg, local (Holm p)", "Difference to FedAvg, aggregate (Holm p)",
                               "Same selected round"], "rows": full_rows}
SEL4 = ["FedGNN_Hard_GCN", "FedGNN_Soft_GCN", "FedGNN_Soft_Virtual_GCN", "FedGNN_Soft_Virtual_GAT"]
defz = [(ZF[v] - ZF.FedAvg_GCN).mean() for v in SEL4]; defl = [(LF[v] - LF.FedAvg_GCN).mean() for v in SEL4]
gains = [(ZF[v] - LF[v]).mean() for v in SEL4]
frz = stats.friedmanchisquare(*[ZF[v].values for v in FED7Z])
N.update(zf_sel_def_lo=f4(abs(max(defz))), zf_sel_def_hi=f4(abs(min(defz))),
         zf_sel_def_lo_local=f4(abs(max(defl))), zf_sel_def_hi_local=f4(abs(min(defl))),
         zf_sel_gain_lo=f4(min(gains)), zf_sel_gain_hi=f4(max(gains)),
         zf_sel_share=f"{100 * np.mean([g / abs(d) for g, d in zip(gains, defl)]):.0f}",
         zf_proto_d=f4(abs((ZF.FedProto_GCN - ZF.FedAvg_GCN).mean())), zf_proto_p=pfmt(phz["FedProto_GCN"]),
         zf_prox_d=sgn4((ZF.FedProx_GCN - ZF.FedAvg_GCN).mean()), zf_prox_p=pfmt(phz["FedProx_GCN"]),
         zf_pmax_sel=pfmt(max(phz[v] for v in SEL4)), zf_fried_p=pfmt(frz.pvalue),
         zf_rank_rho=f"{stats.spearmanr(ZF[FED7Z].mean(), LF[FED7Z].mean()).statistic:.2f}",
         zf_fedavg=f4(ZF.FedAvg_GCN.mean()), zf_fedavg_d=sgn4((ZF.FedAvg_GCN - LF.FedAvg_GCN).mean()),
         zf_same_min=str(min(same_sel.values())), zf_same_max=str(max(same_sel.values())),
         z_traj_n=str(int(df.exp.isin(["ZN", "ZM"]).sum())),
         z_traj_ok=str(int(sum((runs_of(z, v).best_round_local == runs_of(s, v).best_round).sum()
                               for z, s, vs in [("ZN", "N", ["FedAvg_GCN", "FedAvg_MLP"]), ("ZM", "M", others)] for v in vs))))
acc += [("Z20", "Louvain, redrawn per seed; patience 20; selection on the aggregated model", "2", "165", 2),
        ("Z50", "Louvain, redrawn per seed; patience 50; selection on the aggregated model", "2", "165", 2),
        ("Z100", "Louvain, redrawn per seed; patience 100; selection on the aggregated model", "2", "165", 2),
        ("ZN", "Louvain, redrawn per seed; no early stopping; selection on the aggregated model", "2", "165", 2),
        ("ZM", "Louvain, redrawn per seed; no early stopping; selection on the aggregated model", "2", "165", 6)]
FACT.update({z: "checkpoint-selection criterion" for z in ["Z20", "Z50", "Z100", "ZN", "ZM"]})
for lab in ["Z20", "Z50", "Z100", "ZN", "ZM"]:
    T["hashes"]["rows"].append([lab, "–", df[df.exp == lab].config_hash.iloc[0]])

# run accounting table (Table 3): every protocol, the factor it varies and whether it was planned
PLANNED = {"A", "B", "C0–C4", "D", "E"}
acc_rows = []
for lab, desc, bud, feat, nd in acc:
    n, nprot = nruns(lab); assert n == nd * 20 * nprot, (lab, n)
    acc_rows.append([lab, FACT[lab], "yes" if lab in PLANNED else "no", desc, bud, feat, str(nd), "20" if nprot == 1 else f"20 × {nprot}", f"{n:,}"])
acc_rows.append(["Total", "", "", "", "", "", "", "", f"{len(df):,}"])
assert sum(int(r[-1].replace(",", "")) for r in acc_rows[:-1]) == len(df)
T["accounting"] = {"header": ["Protocol", "Factor varied", "Planned", "Client partition and training", "Budget", "Features", "Designs", "Seeds", "Runs"],
                   "rows": acc_rows}
for lab in ["S50", "S100"]:
    T["hashes"]["rows"].append([lab, "–", df[df.exp == lab].config_hash.iloc[0]])
N["H_cgcn_drop_abs"] = f"{abs(float(N['H_cgcn_drop'])):.4f}"
for tag in ["T025", "base", "T1", "T2"]:
    N[f"t_{tag}_enc"] = f"{TS_[tag]['enc']:.2f}"
# one bootstrap for the budget-1 interaction on C0-C9 (Table 8 uses the interval of Table 9)
T["regimes"]["rows"][0][4] = f"{TS_['base']['inter']:.4f} {N['t_base_ci']}"; N["rgC_inter_ci"] = N["t_base_ci"]
sm = wide(df, "S100").FedAvg_MLP; nm = wide(df, "N").FedAvg_MLP
N["s100_mlp_ndiff"] = str(int((np.abs(sm - nm) > 1e-9).sum()))
dup = int(((df.exp.isin(["A", "B"] + [f"C{k}" for k in K25])) & (df.variant == "FedProx_GCN")).sum()) + 6 * 20
N["n_dup"] = f"{dup:,}"; N["n_distinct"] = f"{len(df) - dup:,}"
gbl = tb[(tb.model == "hist_gradient_boosting") & (tb.fs == "local_93")].ap.mean()
N["tab_gb_local"] = f4(gbl); N["tab_gb_local_gap"] = f4(gbl - wH.FedAvg_MLP.mean())

(OUT / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False), encoding="utf-8")
(OUT / "tables.json").write_text(json.dumps(T, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{len(N)} numbers and {len(T)} tables written to {OUT}")

import itertools
import numpy as np
import pandas as pd
from scipy import stats

SEEDS = list(range(42, 62))


def wide(df, exp, col="ap", variants=None):
    w = df[df.exp == exp].pivot(index="seed", columns="variant", values=col).loc[SEEDS]
    return w[variants] if variants else w


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(1, run)
    return adj


def wil(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.all(d == 0):
        return 1.0
    return stats.wilcoxon(a, b, alternative="two-sided", method="exact").pvalue


def pairwise(w):
    V = list(w.columns); pairs = list(itertools.combinations(V, 2))
    ps = [wil(w[a], w[b]) for a, b in pairs]
    adj = holm(ps)
    return pd.DataFrame([dict(a=a, b=b, diff=w[a].mean() - w[b].mean(), p=p, p_holm=q)
                         for (a, b), p, q in zip(pairs, ps, adj)])


def vs_base(w, base="FedAvg_GCN"):
    others = [v for v in w.columns if v != base]
    ps = [wil(w[v], w[base]) for v in others]
    adj = holm(ps)
    return pd.DataFrame([dict(variant=v, diff=w[v].mean() - w[base].mean(), p=p, p_holm=q,
                              wins=int((w[v] > w[base]).sum()))
                         for v, p, q in zip(others, ps, adj)])


def null_range(w, n_perm=20000, rng=0):
    """
    Is the observed spread of variant means larger than expected if all
    variants were exchangeable within each seed? Within-seed permutation of
    variant labels keeps the seed (partition, init) coupling intact.
    """
    X = w.values  # seeds x variants
    obs = X.mean(0).max() - X.mean(0).min()
    r = np.random.default_rng(rng)
    null = np.empty(n_perm)
    for i in range(n_perm):
        P = np.array([row[r.permutation(len(row))] for row in X])
        m = P.mean(0); null[i] = m.max() - m.min()
    return obs, float(null.mean()), float((null >= obs).mean())


def summary(w):
    t = pd.DataFrame({"mean": w.mean(), "sd": w.std(ddof=1), "min": w.min(), "max": w.max(),
                      "se": w.std(ddof=1) / np.sqrt(len(w))})
    return t.sort_values("mean", ascending=False)

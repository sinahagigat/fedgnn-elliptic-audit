"""
The v7 job table. One job = one experiment config x a block of seeds.
Each job writes its own artifacts file, so several Colab accounts can run
different jobs at the same time without touching each other's files.

    python tools/jobs.py            # print the table
"""

SEED_BLOCKS_4 = ["42-46", "47-51", "52-56", "57-61"]
SEED_BLOCKS_10 = ["42-43", "44-45", "46-47", "48-49", "50-51",
                  "52-53", "54-55", "56-57", "58-59", "60-61"]

JOBS = {}

for i, s in enumerate(SEED_BLOCKS_4, 1):
    JOBS[f"A{i}"] = {"experiment": "A", "config": "configs/v7_A_louvain.yaml", "seeds": s}
    JOBS[f"B{i}"] = {"experiment": "B", "config": "configs/v7_B_random.yaml", "seeds": s}
for k in range(5):
    for i, s in enumerate(SEED_BLOCKS_4, 1):
        JOBS[f"C{k}_{i}"] = {"experiment": f"C{k}", "config": f"configs/v7_C{k}_fixed.yaml", "seeds": s}
# the budget experiment is ~5-10x slower per run, so it gets smaller blocks
for i, s in enumerate(SEED_BLOCKS_10, 1):
    JOBS[f"D{i}"] = {"experiment": "D", "config": "configs/v7_D_budget.yaml", "seeds": s}

for i, s in enumerate(SEED_BLOCKS_4, 1):
    JOBS[f"E{i}"] = {"experiment": "E", "config": "configs/v7_E_budget_mlp.yaml", "seeds": s}

# revision experiments (configs from tools/make_revision_configs.py):
# C5..C9 fixed partitions with partition seeds 42..46, F random partition at
# budget 2 (FedAvg GCN and MLP), G protocol A at budget 2 with the 93 local
# features only. G must run with its own data folder: the graph cache is keyed
# by split mode only, so it must not share a cache with the 165-feature runs
# (tools/run_revision.py takes care of this).
for k in range(5, 10):
    for i, s in enumerate(SEED_BLOCKS_4, 1):
        JOBS[f"C{k}_{i}"] = {"experiment": f"C{k}", "config": f"configs/v7_C{k}_fixed.yaml", "seeds": s}
for i, s in enumerate(SEED_BLOCKS_4, 1):
    JOBS[f"F{i}"] = {"experiment": "F", "config": "configs/v7_F_random_budget.yaml", "seeds": s}
    JOBS[f"G{i}"] = {"experiment": "G", "config": "configs/v7_G_local_features_budget.yaml", "seeds": s}
REVISION_JOBS = ([f"C{k}_{i}" for k in range(5, 10) for i in range(1, 5)]
                 + [f"F{i}" for i in range(1, 5)] + [f"G{i}" for i in range(1, 5)])

# second revision: C10..C24 (partition seeds 47..61) complete the 20 x 20
# matrix of partition seed x training seed; T* jobs vary the softmax
# temperature of Soft on the ten fixed partitions (one job of 20 seeds each)
for k in range(10, 25):
    for i, s in enumerate(SEED_BLOCKS_4, 1):
        JOBS[f"C{k}_{i}"] = {"experiment": f"C{k}", "config": f"configs/v7_C{k}_fixed.yaml", "seeds": s}
TEMP_JOBS = []
for tag in ["T025", "T1", "T2"]:
    for k in range(10):
        JOBS[f"{tag}C{k}"] = {"experiment": f"{tag}C{k}", "config": f"configs/v7_{tag}_C{k}.yaml", "seeds": "42-61"}
        TEMP_JOBS.append(f"{tag}C{k}")
REVISION2_JOBS = [f"C{k}_{i}" for k in range(10, 25) for i in range(1, 5)] + TEMP_JOBS

# third revision (configs from tools/make_revision3_configs.py):
# V0..V9 separate validation data for client weighting and checkpoint
# selection on C0..C9; N protocol D without early stopping (FedAvg GCN and
# MLP); P0..P24 the 25 fixed partitions at budget 2 (FedAvg, Hard, Soft).
# The list order is the run order: V first (fastest), then N, then P.
REVISION3_JOBS = []
for k in range(10):
    for i, s in enumerate(SEED_BLOCKS_4, 1):
        JOBS[f"V{k}_{i}"] = {"experiment": f"V{k}", "config": f"configs/v7_V{k}_valsplit.yaml", "seeds": s}
        REVISION3_JOBS.append(f"V{k}_{i}")
for i, s in enumerate(SEED_BLOCKS_4, 1):
    JOBS[f"N{i}"] = {"experiment": "N", "config": "configs/v7_N_no_early_stop.yaml", "seeds": s}
    REVISION3_JOBS.append(f"N{i}")
for k in range(25):
    for i, s in enumerate(SEED_BLOCKS_4, 1):
        JOBS[f"P{k}_{i}"] = {"experiment": f"P{k}", "config": f"configs/v7_P{k}_fixed_budget.yaml", "seeds": s}
        REVISION3_JOBS.append(f"P{k}_{i}")

# fourth batch (configs from tools/make_revision4_configs.py): budget 2
# without early stopping. R and L first (short), then Q0..Q9. L uses the 93
# local features and must run with the local-features data folder.
REVISION4_JOBS = []
for i, s in enumerate(SEED_BLOCKS_4, 1):
    JOBS[f"R{i}"] = {"experiment": "R", "config": "configs/v7_R_random_budget_full.yaml", "seeds": s}
    JOBS[f"L{i}"] = {"experiment": "L", "config": "configs/v7_L_local_features_full.yaml", "seeds": s}
    REVISION4_JOBS += [f"R{i}", f"L{i}"]
for k in range(10):
    for i, s in enumerate(SEED_BLOCKS_4, 1):
        JOBS[f"Q{k}_{i}"] = {"experiment": f"Q{k}", "config": f"configs/v7_Q{k}_fixed_budget_full.yaml", "seeds": s}
        REVISION4_JOBS.append(f"Q{k}_{i}")
LOCAL_FEATURE_EXPERIMENTS = {"G", "L"}

# fifth batch (tools/make_revision5_configs.py): protocol D designs without
# early stopping, in blocks of two seeds
REVISION5_JOBS = []
for i, s in enumerate(SEED_BLOCKS_10, 1):
    JOBS[f"M{i}"] = {"experiment": "M", "config": "configs/v7_M_designs_full.yaml", "seeds": s}
    REVISION5_JOBS.append(f"M{i}")

# sixth batch (tools/make_revision6_configs.py): 600 rounds without early
# stopping for FedAvg, FedProto, Hard and Soft, in blocks of two seeds
REVISION6_JOBS = []
for i, s in enumerate(SEED_BLOCKS_10, 1):
    JOBS[f"W{i}"] = {"experiment": "W", "config": "configs/v7_W_designs_600rounds.yaml", "seeds": s}
    REVISION6_JOBS.append(f"W{i}")

# seventh batch (tools/make_revision7_configs.py): patience 50 and 100
REVISION7_JOBS = []
for pat in (50, 100):
    for i, s in enumerate(SEED_BLOCKS_4, 1):
        JOBS[f"S{pat}_{i}"] = {"experiment": f"S{pat}", "config": f"configs/v7_S{pat}_patience.yaml", "seeds": s}
        REVISION7_JOBS.append(f"S{pat}_{i}")

# checkpoint selection on the aggregated model (fed.select_on_aggregate)
AGGSEL_JOBS = []
for exp, conf in [("Z20", "v7_Z20_aggsel.yaml"), ("Z50", "v7_Z50_aggsel.yaml"), ("Z100", "v7_Z100_aggsel.yaml"),
                  ("ZN", "v7_ZN_aggsel_full.yaml"), ("ZM", "v7_ZM_aggsel_full.yaml")]:
    for i, s in enumerate(SEED_BLOCKS_4, 1):
        JOBS[f"{exp}_{i}"] = {"experiment": exp, "config": f"configs/{conf}", "seeds": s}
        AGGSEL_JOBS.append(f"{exp}_{i}")

# rough GPU hours per job on a Colab T4, from the v6a timings
HOURS = {"A": 0.9, "B": 1.0, "C": 0.8, "D": 2.5, "E": 0.5, "F": 1.0, "G": 1.5, "T": 0.6,
         "V": 0.4, "N": 1.5, "P": 1.2, "Q": 1.5, "R": 1.5, "L": 1.5, "M": 1.5, "W": 2.0, "S": 0.7, "Z": 1.0}

# suggested order: A and B first (they also reproduce the paper's runs),
# then C, then D
PRIORITY = ([f"A{i}" for i in range(1, 5)] + [f"B{i}" for i in range(1, 5)]
            + [f"C{k}_{i}" for k in range(5) for i in range(1, 5)]
            + [f"D{i}" for i in range(1, 11)]
            + [f"E{i}" for i in range(1, 5)]
            + REVISION_JOBS + REVISION2_JOBS + REVISION3_JOBS + REVISION4_JOBS + REVISION5_JOBS + REVISION6_JOBS + REVISION7_JOBS + AGGSEL_JOBS)

if __name__ == "__main__":
    total = 0.0
    for name in PRIORITY:
        j = JOBS[name]
        h = HOURS[j["experiment"][0]]
        total += h
        print(f"{name:6s} {j['experiment']:3s} seeds {j['seeds']:6s} ~{h:.1f} h  {j['config']}")
    print(f"\n{len(JOBS)} jobs, ~{total:.0f} GPU hours in total")
